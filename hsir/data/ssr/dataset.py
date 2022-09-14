import os
import random

import cv2
import numpy as np
from tqdm import tqdm
from hdf5storage import loadmat
from torch.utils.data import Dataset


class ValidDataset(Dataset):
    def __init__(self, root):
        self.hsis = []
        self.rgbs = []

        hsi_root = os.path.join(root, 'Valid_spectral')
        rgb_root = os.path.join(root, 'Valid_RGB')
        # with open(os.path.join(root, 'split_txt', 'valid_list.txt'), 'r') as f:
        #     names = [n.stripe() for n in f.readlines()]
        names = [n.split('.')[0] for n in os.listdir(hsi_root) if n.endswith('.mat')]

        for name in names:
            hsi_path = os.path.join(hsi_root, name + '.mat')
            rgb_path = os.path.join(rgb_root, name + '.jpg')

            hsi = np.float32(loadmat(hsi_path)['cube'])
            hsi = np.transpose(hsi, [2, 0, 1])

            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rgb = np.float32(rgb)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = np.transpose(rgb, [2, 0, 1])  # [3,482,512]

            self.hsis.append(hsi)
            self.rgbs.append(rgb)

        self.num_img = len(self.hsis)

    def __getitem__(self, idx):
        rgb = self.rgbs[idx]
        hsi = self.hsis[idx]
        return {
            'input': rgb,
            'target': hsi
        }

    def __len__(self):
        return self.num_img


class TrainDataset(Dataset):
    def __init__(self, root, crop_size=128, stride=8, augment=True, size=None):
        self.crop_size = crop_size
        self.hsis = []
        self.rgbs = []
        self.augment = augment

        h, w = 482, 512  # img shape
        self.stride = stride
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        hsi_root = os.path.join(root, 'Train_spectral')
        rgb_root = os.path.join(root, 'Train_RGB')
        # with open(os.path.join(root, 'split_txt', 'train_list.txt'), 'r') as f:
        #     names = [n.strip() for n in f.readlines()]
        names = [n.split('.')[0] for n in os.listdir(hsi_root) if n.endswith('.mat')]
        names.sort()
        if size is not None: names = names[:size]

        for name in tqdm(names, desc='Load data to memory'):
            hsi_path = os.path.join(hsi_root, name + '.mat')
            rgb_path = os.path.join(rgb_root, name + '.jpg')
            try:
                hsi = np.float32(loadmat(hsi_path)['cube'])
            except:
                print('fail to load', hsi_path)
                continue
            hsi = np.transpose(hsi, [2, 0, 1])
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rgb = np.float32(rgb)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = np.transpose(rgb, [2, 0, 1])  # [3,482,512]

            self.hsis.append(hsi)
            self.rgbs.append(rgb)

        self.num_img = len(self.hsis)
        self.length = self.patch_per_img * self.num_img

    def _augment(self, img, rotTimes, vFlip, hFlip):
        for _ in range(rotTimes):
            img = np.rot90(img, axes=(1, 2))
        for _ in range(vFlip):
            img = img[:, :, ::-1]
        for _ in range(hFlip):
            img = img[:, ::-1, :]
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line

        rgb = self.rgbs[img_idx]
        hsi = self.hsis[img_idx]
        rgb = rgb[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        hsi = hsi[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]

        if self.augment:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            rgb = self._augment(rgb, rotTimes, vFlip, hFlip)
            hsi = self._augment(hsi, rotTimes, vFlip, hFlip)

        return {
            'input': rgb.copy(),
            'target': hsi.copy()
        }

    def __len__(self):
        return self.patch_per_img * self.num_img
