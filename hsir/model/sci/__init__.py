from .cst import CST
from .mst import MST


def mst_s():
    model = MST(dim=28, stage=2, num_blocks=[2, 2, 2])
    return model


def mst_m():
    model = MST(dim=28, stage=2, num_blocks=[2, 4, 4])
    return model


def mst_l():
    model = MST(dim=28, stage=2, num_blocks=[4, 7, 5])
    return model


def cst_s():
    return CST(num_blocks=[1, 1, 2], sparse=True)


def cst_m():
    return CST(num_blocks=[2, 2, 2], sparse=True)


def cst_l():
    return CST(num_blocks=[2, 4, 6], sparse=True)


def cst_l_plus():
    return CST(num_blocks=[2, 4, 6], sparse=False)


def gap_net():
    from .gap_net import GAP_net
    return GAP_net()


def admm_net():
    from .admm_net import ADMM_net
    return ADMM_net()


def tsa_net():
    from .tsa_net import TSA_Net
    return TSA_Net()


def hdnet():
    from .hdnet import HDNet
    return HDNet()


def fdl_loss():
    from .hdnet import FDL
    return FDL(
        loss_weight=0.7,
        alpha=2.0,
        patch_factor=4,
        ave_spectrum=True,
        log_matrix=True,
        batch_matrix=True,
    )


def dgsmp():
    from .dgsmp import HSI_CS
    return HSI_CS(Ch=28, stages=4)


def birnat():
    from .birnat import BIRNAT
    return BIRNAT()


def mst_plus_plus():
    from .mst_plus_plus import MST_Plus_Plus
    return MST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3)


def lambda_net():
    from .lambda_net import Lambda_Net
    return Lambda_Net(out_ch=28)
