from functools import partial
from .net import QRNN3DDecoder, QRNN3DEncoder, QRNNREDC3D
from .qrnn import QRNNConv3D, QRNNDeConv3D, QRNNUpsampleConv3d, BiQRNNConv3D, BiQRNNDeConv3D


QRNN3DEncoder = partial(
    QRNN3DEncoder,
    QRNNConv3D=QRNNConv3D)

QRNN3DDecoder = partial(
    QRNN3DDecoder,
    QRNNDeConv3D=QRNNDeConv3D,
    QRNNUpsampleConv3d=QRNNUpsampleConv3d)

QRNNREDC3D = partial(
    QRNNREDC3D,
    BiQRNNConv3D=BiQRNNConv3D,
    BiQRNNDeConv3D=BiQRNNDeConv3D,
    QRNN3DEncoder=QRNN3DEncoder,
    QRNN3DDecoder=QRNN3DDecoder
)


def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net


def qrnn3d_nobn():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True, bn=False)
    net.use_2dconv = False
    net.bandwise = False
    return net
