from .conv import *
from .attention import *
from .combinations import *


class BI_before(nn.Module):
    def __init__(self, in_channels, channels, med_channels):
        super(BI_before, self).__init__()
        self.layer_0 = BasicConv3d(in_channels, 1, k=1, s=1, p=0)
        self.layer_1 = BasicConv2d(med_channels, channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.layer_0(x)  # N C B H W -> N 1 B H W
        x = x.squeeze(1)  # N 1 B H W -> N B H W
        x = self.layer_1(x)  # N B H W -> N C H W
        return x


class BI_after(nn.Module):
    def __init__(self, in_channels, channels, med_channels):
        super(BI_after, self).__init__()
        self.layer_0 = BasicConv2d(channels, med_channels, k=1, s=1, p=0)
        self.layer_1 = BasicConv3d(1, in_channels, k=1, s=1, p=0)

    def forward(self, x):
        x = self.layer_0(x)  # N C H W -> N B H W
        x = x.unsqueeze(1)  # N B H W -> N 1 B H W
        x = self.layer_1(x)  # N 1 B H W -> N C B H W
        return x


class TRQ3DBasicLayer(nn.Module):
    def __init__(self, in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels,
                 conv_layer, tr_layer, sample_layer=None):
        super(TRQ3DBasicLayer, self).__init__()

        # couple layer
        self.bi_qrnn_tr = BI_before(in_channels, in_channels_tr, med_channels)
        self.bi_tr_qrnn = BI_after(hidden_channels_tr, hidden_channels, med_channels)
        self.proj = BasicConv3d(hidden_channels, hidden_channels, k=1, s=1, p=0)

        # quasi_conv_layer
        self.qrnn_layer = conv_layer

        # transformer layer
        self.tr_layer = tr_layer
        self.sample_layer = sample_layer

    def _qrnn(self, inputs, reverse):
        return self.qrnn_layer(inputs, reverse)

    def _tr(self, inputs):
        outputs = self.tr_layer(inputs)
        outputs = self.sample_layer(outputs) if self.sample_layer is not None else outputs
        return outputs

    def forward(self, inputs_qrnn, inputs_tr, reverse=False):
        output_qrnn = self._qrnn(inputs_qrnn, reverse)
        output_tr = self.bi_qrnn_tr(inputs_qrnn)
        output_tr = self._tr(inputs_tr + output_tr)
        output_qrnn += self.bi_tr_qrnn(output_tr)
        output_qrnn = self.proj(output_qrnn)

        return output_qrnn, output_tr


class TRQ3DLayer(TRQ3DBasicLayer):
    def __init__(self, in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels, k=3, s=1, p=1,
                 bn=True, act='tanh', input_resolution=(64, 64)):
        super(TRQ3DLayer, self).__init__(
            in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels,
            QRNNConv3D(in_channels, hidden_channels, k, s, p, bn, act),
            BasicUformerLayer(dim=in_channels_tr, output_dim=hidden_channels_tr,
                              input_resolution=input_resolution, depth=2, num_heads=4, win_size=8,
                              mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                              drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                              token_projection='linear', token_mlp='leff', se_layer=False),
            None
            # BasicConv3D(in_channels, hidden_channels, k, s, p, bn=bn),
            )

class TRQ3DDeLayer(TRQ3DBasicLayer):
    def __init__(self, in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels, k=3, s=1, p=1,
                 bn=True, act='tanh', input_resolution=(64, 64)):
        super(TRQ3DDeLayer, self).__init__(
            in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels,
            QRNNDeConv3D(in_channels, hidden_channels, k, s, p, bn, act),
            BasicUformerLayer(dim=in_channels_tr, output_dim=hidden_channels_tr,
                              input_resolution=input_resolution, depth=2, num_heads=4, win_size=8,
                              mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                              drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                              token_projection='linear', token_mlp='leff', se_layer=False),
            None
            # BasicConv3D(in_channels, hidden_channels, k, s, p, bn=bn),
            )

class TRQ3DDownsampleLayer(TRQ3DBasicLayer):
    def __init__(self, in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels, k=3, s=1, p=1,
                 bn=True, act='tanh', input_resolution=(64, 64)):
        super(TRQ3DDownsampleLayer, self).__init__(
            in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels,
            QRNNConv3D(in_channels, hidden_channels, k, s, p, bn=bn, act=act),
            BasicUformerLayer(dim=in_channels_tr, output_dim=hidden_channels_tr,
                              input_resolution=input_resolution, depth=2, num_heads=4, win_size=8,
                              mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                              drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                              token_projection='linear', token_mlp='leff', se_layer=False),
            Downsample(in_channels, hidden_channels),
            )

class TRQ3DUpsampleLayer(TRQ3DBasicLayer):
    def __init__(self, in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels, k=3, s=1, p=1,
                 bn=True, act='tanh', input_resolution=(64, 64)):
        super(TRQ3DUpsampleLayer, self).__init__(
            in_channels, in_channels_tr, hidden_channels, hidden_channels_tr, med_channels,
            QRNNUpsampleConv3D(in_channels, hidden_channels, k, s, p, bn=bn, act=act),
            BasicUformerLayer(dim=in_channels_tr, output_dim=hidden_channels_tr,
                              input_resolution=input_resolution, depth=2, num_heads=4, win_size=8,
                              mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                              drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                              token_projection='linear', token_mlp='leff', se_layer=False),
            Upsample(in_channels, hidden_channels),
            )

class TRQ3DEncoder(nn.Module):
    def __init__(self, channels, med_channels, channels_tr, num_half_layer, sample_idx,
                 has_ad=True, bn=True, act='tanh', input_resolution=(64, 64)):
        super(TRQ3DEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = TRQ3DLayer(channels, channels_tr, channels, channels_tr, med_channels, bn=bn, act=act,
                                           input_resolution=input_resolution)
            else:
                encoder_layer = TRQ3DDownsampleLayer(channels, channels_tr, 2 * channels, channels_tr * 2,
                                                     med_channels, k=3, s=(1, 2, 2), p=1, bn=bn, act=act,
                                                     input_resolution=input_resolution)
                input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
                channels *= 2
                channels_tr *= 2
            self.layers.append(encoder_layer)

    def forward(self, x_qrnn, xs_qrnn, x_tr, xs_tr, reverse=False):
        if not self.enable_ad:
            num_half_layer = len(self.layers)
            for i in range(num_half_layer - 1):
                x_qrnn, x_tr = self.layers[i](x_qrnn, xs_qrnn)
                xs_qrnn.append(x_qrnn)
                xs_tr.append(x_tr)
            x_qrnn, x_tr = self.layers[-1](x_qrnn, x_tr)
            return x_qrnn, x_tr
        else:
            num_half_layer = len(self.layers)
            for i in range(num_half_layer - 1):
                x_qrnn, x_tr = self.layers[i](x_qrnn, x_tr, reverse=reverse)
                reverse = not reverse
                xs_qrnn.append(x_qrnn)
                xs_tr.append(x_tr)
            x_qrnn, x_tr = self.layers[-1](x_qrnn, x_tr, reverse=reverse)
            reverse = not reverse
            return x_qrnn, x_tr, reverse


class TRQ3DDecoder(nn.Module):
    def __init__(self, channels, med_channels, channels_tr, num_half_layer, sample_idx,
                 has_ad=True, bn=True, act='tanh', input_resolution=(64, 64)):
        super(TRQ3DDecoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = TRQ3DDeLayer(channels, channels_tr, channels, channels_tr, med_channels, bn=bn, act=act,
                                             input_resolution=input_resolution)
            else:
                decoder_layer = TRQ3DUpsampleLayer(channels, channels_tr, channels // 2, channels_tr // 2,
                                                   med_channels, bn=bn, act=act,
                                                   input_resolution=input_resolution)
                input_resolution = (input_resolution[0] * 2, input_resolution[1] * 2)
                channels //= 2
                channels_tr //= 2
            self.layers.append(decoder_layer)

    def forward(self, x_qrnn, xs_qrnn, x_tr, xs_tr, reverse=False):
        if not self.enable_ad:
            num_half_layer = len(self.layers)
            x_qrnn, x_tr = self.layers[0](x_qrnn, x_tr)
            for i in range(1, num_half_layer):
                x_qrnn = x_qrnn + xs_qrnn.pop()
                x_tr = x_tr + xs_tr.pop()
                x_qrnn, x_tr = self.layers[i](x_qrnn, x_tr)
            return x_qrnn, x_tr
        else:
            num_half_layer = len(self.layers)
            x_qrnn, x_tr = self.layers[0](x_qrnn, x_tr, reverse=reverse)
            reverse = not reverse
            for i in range(1, num_half_layer):
                x_qrnn = x_qrnn + xs_qrnn.pop()
                x_tr = x_tr + xs_tr.pop()
                x_qrnn, x_tr = self.layers[i](x_qrnn, x_tr, reverse=reverse)
                reverse = not reverse
            return x_qrnn, x_tr


class TRQ3D(nn.Module):
    def __init__(self, in_channels, in_channels_tr, channels, channels_tr, med_channels, num_half_layer, sample_idx,
                 Encoder=TRQ3DEncoder, Decoder=TRQ3DDecoder,
                 has_ad=True, bn=True, act='tanh', input_resolution=(64, 64)):
        super(TRQ3D, self).__init__()

        # binding properties
        assert sample_idx is None or isinstance(sample_idx, list)
        self.enable_ad = has_ad
        if sample_idx is None: sample_idx = []

        # feature extract
        self.feature_extractor_qrnn = BiQRNNConv3D(in_channels, channels, bn=bn, act=act)
        self.feature_extractor_tr = InputProj(in_channel=in_channels_tr, out_channel=channels_tr, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm)
        # encoder and decoder
        self.encoder = Encoder(channels, med_channels, channels_tr, num_half_layer, sample_idx,
                                     has_ad=has_ad, bn=bn, act=act,
                                      input_resolution=input_resolution)
        self.decoder = Decoder(channels * (2 ** len(sample_idx)), med_channels,
                                     channels_tr * (2 ** len(sample_idx)), num_half_layer, sample_idx,
                                     has_ad=has_ad, bn=bn, act=act, input_resolution=(
            input_resolution[0] // (2 ** len(sample_idx)), input_resolution[1] // (2 ** len(sample_idx))))
        # feature reconstruct
        if act == 'relu':
            act = 'none'
        self.reconstructor_qrnn = BiQRNNDeConv3D(channels, in_channels, bias=True, bn=bn, act=act)
        self.reconstructor_tr = OutputProj(in_channel=channels_tr, out_channel=in_channels_tr, kernel_size=3, stride=1,
                                      norm_layer=BatchNorm2d)
        # final merge
        self.merge = nn.Conv2d(med_channels * 2, med_channels, 1, 1, 0)

    def forward(self, x):
        xs_qrnn = [x]
        xs_tr = [x]
        out_qrnn = self.feature_extractor_qrnn(x)
        out_tr = self.feature_extractor_tr(x.squeeze(1))
        xs_qrnn.append(out_qrnn)
        xs_tr.append(out_tr)
        if self.enable_ad:
            out_qrnn, out_tr, reverse = self.encoder(out_qrnn, xs_qrnn, out_tr, xs_tr, reverse=False)
            out_qrnn, out_tr = self.decoder(out_qrnn, xs_qrnn, out_tr, xs_tr, reverse=reverse)
        else:
            out_qrnn, out_tr = self.encoder(out_qrnn, xs_qrnn, out_tr, xs_tr)
            out_qrnn, out_tr = self.decoder(out_qrnn, xs_qrnn, out_tr, xs_tr)
        out_qrnn = out_qrnn + xs_qrnn.pop()
        out_tr = out_tr + xs_tr.pop()
        out_qrnn = self.reconstructor_qrnn(out_qrnn)
        out_tr = self.reconstructor_tr(out_tr)
        out_tr = out_tr.unsqueeze(1)
        out_qrnn = out_qrnn + xs_qrnn.pop()
        out_tr = out_tr + xs_tr.pop()
        outputs = self.merge(torch.cat((out_qrnn.squeeze(1), out_tr.squeeze(1)), dim=1))
        outputs = outputs.unsqueeze(1)
        return outputs


