import torch
from .mst import MST
from .gap_net import GAP_net
from .admm_net import ADMM_net
from .tsa_net import TSA_Net
from .hdnet import HDNet, FDL
from .dgsmp import HSI_CS
from .birnat import BIRNAT
from .mst_plus_plus import MST_Plus_Plus
from .lambda_net import Lambda_Net
from .cst import CST

def model_generator(method, pretrained_model_path=None):
    if method == 'mst_s':
        model = mst(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
    elif method == 'mst_m':
        model = mst(dim=28, stage=2, num_blocks=[2, 4, 4]).cuda()
    elif method == 'mst_l':
        model = mst(dim=28, stage=2, num_blocks=[4, 7, 5]).cuda()
    elif method == 'gap_net':
        model = GAP_net().cuda()
    elif method == 'admm_net':
        model = ADMM_net().cuda()
    elif method == 'tsa_net':
        model = tsa_net().cuda()
    elif method == 'hdnet':
        model = hdnet().cuda()
        fdl_loss = FDL(loss_weight=0.7,
             alpha=2.0,
             patch_factor=4,
             ave_spectrum=True,
             log_matrix=True,
             batch_matrix=True,
             ).cuda()
    elif method == 'dgsmp':
        model = HSI_CS(Ch=28, stages=4).cuda()
    elif method == 'birnat':
        model = birnat().cuda()
    elif method == 'mst_plus_plus':
        model = mst_plus_plus(in_channels=28, out_channels=28, n_feat=28, stage=3).cuda()
    elif method == 'lambda_net':
        model = lambda_net(out_ch=28).cuda()
    elif method == 'cst_s':
        model = cst(num_blocks=[1, 1, 2], sparse=True).cuda()
    elif method == 'cst_m':
        model = cst(num_blocks=[2, 2, 2], sparse=True).cuda()
    elif method == 'cst_l':
        model = cst(num_blocks=[2, 4, 6], sparse=True).cuda()
    elif method == 'cst_l_plus':
        model = cst(num_blocks=[2, 4, 6], sparse=False).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    if method == 'hdnet':
        return model,fdl_loss
    return model