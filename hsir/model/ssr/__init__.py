def mirnet():
    from .MIRNet import MIRNet
    return MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1)

def mst_plus_plus():
    from .MST_Plus_Plus import MST_Plus_Plus
    return MST_Plus_Plus()

def mst():
    from .MST import MST
    return MST(dim=31, stage=2, num_blocks=[4, 7, 5])

def hinet():
    from .hinet import HINet
    return HINet(depth=4)

def mprnet():
    from .MPRNet import MPRNet
    return MPRNet(num_cab=4)

def edsr():
    from .edsr import EDSR
    return EDSR()

def hdnet():
    from .HDNet import HDNet
    return HDNet()

def hrnet():
    from .hrnet import SGN
    return SGN()

def hscnn_plus():
    from .HSCNN_Plus import HSCNN_Plus
    return HSCNN_Plus

def awan():
    from .AWAN import AWAN
    return AWAN()
