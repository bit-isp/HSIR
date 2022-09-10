import torchlight as tl
import torch
import numpy as np


def visualize_gray(hsi, band=20):
    return hsi[:, band:band + 1, :, :]


def visualize_color(hsi):
    srf = tl.transforms.HSI2RGB().srf
    srf = torch.from_numpy(srf).float().to(hsi.device)
    hsi = hsi.permute(0, 2, 3, 1) @ srf.T
    return hsi.permute(0, 3, 1, 2)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
