import torch.nn as nn
import torch
import math
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils import checkpoint

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops

# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.conv(x)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.deconv(x).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.in_channels * self.kernel_size ** 2 / self.stride ** 2
        flops += H * W * self.in_channels * self.out_channels
        return flops


#########################################
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v

    def flops(self, H, W):
        flops = 0
        flops += self.to_q.flops(H, W)
        flops += self.to_k.flops(H, W)
        flops += self.to_v.flops(H, W)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 3
        return flops


class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = torch.cat((k_d, k_e), dim=2)
        v = torch.cat((v_d, v_e), dim=2)
        return q, k, v

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 5
        return flops

    #########################################


########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., se_layer=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H, W)
        # attn = (q @ k.transpose(-2, -1))
        if self.token_projection != 'linear_concat':
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        else:
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N * 2
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N * 2 * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.in_features * self.hidden_features
        # fc2
        flops += H * W * self.hidden_features * self.out_features
        print("MLP:{%.2f}" % (flops / 1e9))
        return flops


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        return flops


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection, se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, mask=None):
        # N C H W -> N H*W C
        x = x.flatten(2).transpose(1, 2)
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        #     attn_mask = attn_mask + self.shift_attn_mask if attn_mask is not None else self.shift_attn_mask

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask

        # N H*W C -> N C H W
        x = x.transpose(1, 2).reshape(B, -1, H, W).contiguous()
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        print("LeWin:{%.2f}" % (flops / 1e9))
        return flops


class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='leff', se_layer=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0 if (i % 2 == 0) else win_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                  se_layer=se_layer)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops