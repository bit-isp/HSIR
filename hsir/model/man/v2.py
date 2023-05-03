import torch
import torch.nn as nn
import torch.nn.functional as F


""" Utility block """


class UpsampleConv3d(torch.nn.Module):
    """ UpsampleConvLayer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None, group=1):
        super(UpsampleConv3d, self).__init__()
        self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True)
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=group)

    def forward(self, x):
        x = self.upsample_layer(x)
        x = self.conv3d(x)
        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP)
    """

    def __init__(self, channel, bias=True):
        super().__init__()
        self.w_1 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)
        self.w_2 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)

    def forward(self, x):
        return self.w_2(F.tanh(self.w_1(x)))


""" The proposed blocks
"""


class PSCA(nn.Module):
    """ Progressive Spectral Channel Attention (PSCA) 
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Conv3d(d_model, d_ff, 1, bias=False)
        self.w_2 = nn.Conv3d(d_ff, d_model, 1, bias=False)
        self.w_3 = nn.Conv3d(d_model, d_model, 1, bias=False)

        nn.init.zeros_(self.w_3.weight)

    def forward(self, x):
        x = self.w_3(x) * x + x
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        return x


class ASC(nn.Module):
    """ Attentive Skip Connection
    """

    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 1),
            nn.LeakyReLU(),
            nn.Conv3d(channel, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.weight(torch.cat([x, y], dim=1))
        out = (1 - w) * x + w * y
        return out


class MHRSA(nn.Module):
    """ Multi-Head Recurrent Spectral Attention
    """

    def __init__(self, channels, multi_head=True, ffn=True):
        super().__init__()
        self.channels = channels
        self.multi_head = multi_head
        self.ffn = ffn

        if ffn:
            self.ffn1 = MLP(channels)
            self.ffn2 = MLP(channels)

    def _conv_step(self, inputs):
        if self.ffn:
            Z = self.ffn1(inputs).tanh()
            F = self.ffn2(inputs).sigmoid()
        else:
            Z, F = inputs.split(split_size=self.channels, dim=1)
            Z, F = Z.tanh(), F.sigmoid()
        return Z, F

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        Z, F = self._conv_step(inputs)

        if self.multi_head:
            Z1, Z2 = Z.split(self.channels // 2, 1)
            Z2 = torch.flip(Z2, [2])
            Z = torch.cat([Z1, Z2], dim=1)

            F1, F2 = F.split(self.channels // 2, 1)
            F2 = torch.flip(F2, [2])
            F = torch.cat([F1, F2], dim=1)

        h = None
        h_time = []

        if not reverse:
            for _, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for _, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        y = torch.cat(h_time, dim=2)

        if self.multi_head:
            y1, y2 = y.split(self.channels // 2, 1)
            y2 = torch.flip(y2, [2])
            y = torch.cat([y1, y2], dim=1)

        return y


class MAB(nn.Module):
    def __init__(self, conv_layer, channels, multi_head=True, ffn=True):
        super().__init__()
        self.conv = conv_layer
        self.inter_sa = MHRSA(channels, multi_head=multi_head, ffn=ffn)
        self.intra_sa = PSCA(channels, channels * 2)

    def forward(self, x, reverse=False):
        x = self.conv(x)
        x = self.inter_sa(x, reverse=reverse)
        x = self.intra_sa(x)
        return x


""" Encoder-Decoder
"""


def PlainMAB(in_ch, out_ch, bias=False):
    return MAB(nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=bias), out_ch)


def DownMAB(in_ch, out_ch, bias=False):
    return MAB(nn.Conv3d(in_ch, out_ch, 3, (1, 2, 2), 1, bias=bias), out_ch)


def UpMAB(in_ch, out_ch, bias=False):
    return MAB(UpsampleConv3d(in_ch, out_ch, 3, 1, 1, bias=bias, upsample=(1, 2, 2)), out_ch)


class Encoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = PlainMAB(channels, channels)
            else:
                encoder_layer = DownMAB(channels, 2 * channels)
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs, reverse=False):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer - 1):
            x = self.layers[i](x, reverse)
            reverse = not reverse
            xs.append(x)
        x = self.layers[-1](x, reverse)
        reverse = not reverse
        return x


class Decoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, Fusion=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.enable_fusion = Fusion is not None

        if self.enable_fusion:
            self.fusions = nn.ModuleList()
            ch = channels
            for i in reversed(range(num_half_layer)):
                fusion_layer = Fusion(ch)
                if i in sample_idx:
                    ch //= 2
                self.fusions.append(fusion_layer)

        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = PlainMAB(channels, channels)
            else:
                decoder_layer = UpMAB(channels, channels // 2)
                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs, reverse=False):
        num_half_layer = len(self.layers)
        x = self.layers[0](x, reverse)
        reverse = not reverse
        for i in range(1, num_half_layer):
            if self.enable_fusion:
                x = self.fusions[i](x, xs.pop())
            else:
                x = x + xs.pop()
            x = self.layers[i](x, reverse)
            reverse = not reverse
        return x


class MAN(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, Fusion=None):
        super().__init__()
        self.head = PlainMAB(in_channels, channels)
        self.encoder = Encoder(channels, num_half_layer, sample_idx)
        self.decoder = Decoder(channels * (2**len(sample_idx)), num_half_layer, sample_idx, Fusion=Fusion)
        self.tail = nn.Conv3d(channels, in_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        xs = [x]
        out = self.head(xs[0])
        xs.append(out)
        reverse = True
        out = self.encoder(out, xs, reverse)
        out = self.decoder(out, xs, reverse)
        out = out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()
        return out


class MAN_T(MAN):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, Fusion=None):
        super().__init__(in_channels, channels, num_half_layer, sample_idx, Fusion)
        self.tail = PlainMAB(channels, in_channels, bias=True)


""" Models
"""


def man_s():
    net = MAN(1, 12, 5, [1, 3], Fusion=ASC)
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_m():
    net = MAN(1, 16, 5, [1, 3], Fusion=ASC)
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_l():
    net = MAN(1, 20, 5, [1, 3], Fusion=ASC)
    net.use_2dconv = False
    net.bandwise = False
    return net

