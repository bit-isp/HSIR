import torchvision.transforms as T
import torch.utils.data as D
import hsir.data.transform.noise as N
import hsir.data.transform.general as G
from hsir.data import HSITestDataset, HSITrainDataset
from hsir.data.utils import worker_init_fn


def gaussian_loader_train_s1(root, use_cdhw=False):
    common_transform = G.Identity()
    input_transform = T.Compose([
        N.AddNoise(50),
        G.HSI2Tensor(use_cdhw=use_cdhw)
    ])
    target_transform = G.HSI2Tensor(use_cdhw=use_cdhw)
    dataset = HSITrainDataset(root, input_transform, target_transform, common_transform)
    loader = D.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8,
                          pin_memory=True, worker_init_fn=worker_init_fn)
    return loader


def gaussian_loader_train_s2(root, use_cdhw=False):
    common_transform = G.RandomCrop(32, 32)
    input_transform = T.Compose([
        N.AddNoiseBlind([10, 30, 50, 70]),
        G.HSI2Tensor(use_cdhw=use_cdhw)
    ])
    target_transform = G.HSI2Tensor(use_cdhw=use_cdhw)
    dataset = HSITrainDataset(root, input_transform, target_transform, common_transform)
    loader = D.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8,
                          pin_memory=True, worker_init_fn=worker_init_fn)
    return loader


def gaussian_loader_train_s2_16(root, use_cdhw=False):
    common_transform = G.Identity()
    input_transform = T.Compose([
        N.AddNoiseBlind([10, 30, 50, 70]),
        G.HSI2Tensor(use_cdhw=use_cdhw)
    ])
    target_transform = G.HSI2Tensor(use_cdhw=use_cdhw)
    dataset = HSITrainDataset(root, input_transform, target_transform, common_transform)
    loader = D.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8,
                          pin_memory=True, worker_init_fn=worker_init_fn)
    return loader


def gaussian_loader_val(root, use_cdhw=False):
    dataset = HSITestDataset(root, size=5, use_cdhw=use_cdhw)
    loader = D.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    return loader


def gaussian_loader_test(root, use_cdhw=False):
    dataset = HSITestDataset(root, return_name=True, use_cdhw=use_cdhw)
    loader = D.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    return loader
