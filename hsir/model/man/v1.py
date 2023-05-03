from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Skip Connection """


class AdditiveConnection(nn.Module):
    def __init__(self, channel):
        super().__init__()

    def forward(self, x, y):
        return x + y


class ConcatSkipConnection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 3, 1, 1),
        )

    def forward(self, x, y):
        return self.conv(torch.cat([x, y], dim=1))


class AdaptiveSkipConnection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_weight = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 1),
            nn.Tanh(),
            nn.Conv3d(channel, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.conv_weight(torch.cat([x, y], dim=1))
        return (1 - w) * x + w * y


""" Channel Attention """


class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sum = torch.sum(x, dim=(2, 3, 4), keepdim=True)
        return sum / (x.shape[2] * x.shape[3] * x.shape[4])


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool3d(1),
            GlobalAvgPool3d(),
            nn.Conv3d(ch, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sca(x) * x


class ChannelAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = nn.Sequential(
            GlobalAvgPool3d(),
            nn.Conv3d(ch, ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(ch, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ca(x) * x


""" Mixed Attention Block """


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, channel):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv3d(channel, channel, kernel_size=1)
        self.w_2 = nn.Conv3d(channel, channel, kernel_size=1)

    def forward(self, x):
        return self.w_2(torch.tanh(self.w_1(x)))


class MAB(nn.Module):
    def __init__(self, channels, enable_ca=True, reverse=False):
        super(MAB, self).__init__()
        self.channels = channels
        self.enable_ca = enable_ca
        self.reverse = reverse
        self.sca = SimplifiedChannelAttention(channels)
        self.ffn_f = PositionwiseFeedForward(channels)
        self.ffn_w = PositionwiseFeedForward(channels)

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs):
        h = None
        Z = self.ffn_f(inputs).tanh()
        F = self.ffn_w(inputs).sigmoid()
        h_time = []

        if not self.reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        out = torch.cat(h_time, dim=2)
        if self.enable_ca:
            out = self.sca(out)
        return out


class BiMAB(MAB):
    def __init__(self, channels, enable_ca=True):
        super().__init__(channels, enable_ca, None)
        self.ffn_w2 = PositionwiseFeedForward(channels)

    def forward(self, inputs):
        Z = self.ffn_f(inputs).tanh()
        F1 = self.ffn_w(inputs).sigmoid()
        F2 = self.ffn_w2(inputs).sigmoid()

        h = None
        hsl = []
        hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):
            h = self._rnn_step(z, f, h)
            hsl.append(h)

        h = None
        for time, (z, f) in enumerate((zip(reversed(zs), reversed(F2.split(1, 2))))):
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)

        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)

        out = hsl + hsr
        if self.enable_ca:
            out = self.sca(out)
        return out


""" Mixed Attention Network """


class Encoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, Attn=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(channels, channels, 3, 1, 1, bias=False)),
                    ('attn', Attn(channels, reverse=i % 2 == 1)) if Attn else None
                ]))
            else:
                encoder_layer = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(channels, channels * 2, 3, (1, 2, 2), 1, bias=False)),
                    ('attn', Attn(channels * 2, reverse=i % 2 == 1)) if Attn else None
                ]))
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer - 1):
            x = self.layers[i](x)
            xs.append(x)
        x = self.layers[-1](x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, Fusion=None, Attn=None):
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
                decoder_layer = nn.Sequential(OrderedDict([
                    ('conv', nn.ConvTranspose3d(channels, channels, 3, 1, 1, bias=False)),
                    ('attn', Attn(channels, reverse=i % 2 == 0)) if Attn else None
                ]))
            else:
                decoder_layer = nn.Sequential(OrderedDict([
                    ('up', nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)),
                    ('conv', nn.Conv3d(channels, channels // 2, 3, 1, 1, bias=False)),
                    ('attn', Attn(channels // 2, reverse=i % 2 == 0)) if Attn else None
                ]))
                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        x = self.layers[0](x)
        for i in range(1, num_half_layer):
            if self.enable_fusion:
                x = self.fusions[i](x, xs.pop())
            else:
                x = x + xs.pop()
            x = self.layers[i](x)
        return x


class MAN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels=16,
        num_half_layer=5,
        sample_idx=[1, 3],
        Attn=MAB,
        BiAttn=BiMAB,
        Fusion=AdaptiveSkipConnection,
    ):
        super(MAN, self).__init__()

        self.head = nn.Sequential(
            nn.Conv3d(in_channels, channels, 3, 1, 1, bias=False),
            BiAttn(channels) if BiAttn else None
        )

        self.encoder = Encoder(channels, num_half_layer, sample_idx, Attn)
        self.decoder = Decoder(channels * (2**len(sample_idx)), num_half_layer, sample_idx,
                               Fusion=Fusion, Attn=Attn)

        self.tail = nn.Sequential(
            nn.ConvTranspose3d(channels, in_channels, 3, 1, 1, bias=True),
            BiAttn(in_channels) if BiAttn else None
        )

    def forward(self, x):
        xs = [x]
        out = self.head(xs[0])
        xs.append(out)
        out = self.encoder(out, xs)
        out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()
        return out


""" Model Variants """


def man():
    net = MAN(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_m():
    net = MAN(1, 12, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_s():
    net = MAN(1, 8, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_b():
    net = MAN(1, 24, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_deep():
    net = MAN(1, 16, 7, [1, 3, 5])
    net.use_2dconv = False
    net.bandwise = False
    return net


""" Baseline """


def baseline():
    net = MAN(1, 16, 5, [1, 3], Attn=None, BiAttn=None, Fusion=None)
    net.use_2dconv = False
    net.bandwise = False
    return net
