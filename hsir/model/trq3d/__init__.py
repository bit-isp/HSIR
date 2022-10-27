# the original version, with FCU before and after
from functools import partial
from .trq3d import *


TRQ3D = partial(
    TRQ3D,
    Encoder=TRQ3DEncoder,
    Decoder=TRQ3DDecoder
)


def trq3d():
    net = TRQ3D(in_channels=1, in_channels_tr=31, channels=16, channels_tr=16, med_channels=31, num_half_layer=4, sample_idx=[1, 3], has_ad=True,
                input_resolution=(512, 512))
    net.use_2dconv = False
    net.bandwise = False
    return net
