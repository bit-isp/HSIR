import random
import os

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from tqdm import tqdm

def augment(hsi):
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)

    for _ in range(rotTimes):
        hsi = np.rot90(hsi)
    for _ in range(vFlip):
        hsi = hsi[:, ::-1, :]
    for _ in range(hFlip):
        hsi = hsi[::-1, :, :]
    return hsi


def random_crop(hsi, size):
    h, w = hsi.shape[:2]
    px = random.randint(0, h - size)
    py = random.randint(0, w - size)
    return hsi[px:px + size, py:py + size, :]


def compress(label, mask, size):
    temp = mask * label
    temp_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    temp_shift[:, 0:size, :] = temp
    for t in range(28):
        temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
    meas = np.sum(temp_shift, axis=2)
    input = meas / 28 * 2 * 1.2

    QE, bit = 0.4, 2048
    input = np.random.binomial((input * bit / QE).astype(int), QE)
    input = np.float32(input) / np.float32(bit)
    return input


def shift(mask_3d, size):
    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    return mask_3d_shift, mask_3d_shift_s


class SCIDataset(Dataset):
    def __init__(self, data, mask_path, crop_size, train=True):
        super(SCIDataset, self).__init__()
        self.train = train
        self.size = crop_size

        self.cache = data

        self.mask = sio.loadmat(mask_path)['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))

    def __getitem__(self, index):
        hsi = self.cache[index]

        label = random_crop(hsi, self.size)
        mask_3d = random_crop(self.mask_3d, self.size)

        if self.train:
            label = augment(label)

        input = compress(label, mask_3d, self.size)
        mask_3d_shift, mask_3d_shift_s = shift(mask_3d, self.size)

        return {
            'target': label.copy().transpose(2, 0, 1),
            'input': input.copy(),
            'mask_3d': mask_3d,
            'mask_3d_shift': mask_3d_shift.copy().transpose(2, 0, 1),
            'mask_3d_shift_s': mask_3d_shift_s.copy(),
        }

    def __len__(self):
        return len(self.cache)


def load_CAVE(root, file_num):
    HR_HSI = np.zeros((512, 512, 28, file_num))
    file_list = os.listdir(root)
    for idx in tqdm(range(file_num), desc='Load CAVE'):
        HR_code = file_list[idx]
        path1 = os.path.join(root) + HR_code
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = data['data_slice'] / 65535.0
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI


def load_KAIST(root, file_num):
    HR_HSI = np.zeros((2704, 3376, 28, file_num))
    file_list = os.listdir(root)
    for idx in tqdm(range(file_num), desc='Load KAIST'):
        HR_code = file_list[idx]
        path1 = os.path.join(root) + HR_code
        data = sio.loadmat(path1)
        HR_HSI[:, :, :, idx] = data['HSI']
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI
