import os
import pickle

import scipy.io
import torch
import lmdb
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from functools import partial

__all__ = [
    'HSITestDataset',
    'HSITrainDataset',
    'HSITransformTestDataset'
]


class HSITestDataset(Dataset):
    def __init__(self, root, size=None, use_cdhw=True, return_name=False):
        super().__init__()
        self.dataset = MatDataFromFolder(root, size=size)
        self.transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                       transform=partial(np.expand_dims, axis=0) if use_cdhw else None),
        ])
        self.return_name = return_name

    def __getitem__(self, index):
        mat, filename = self.dataset[index]
        inputs, targets = self.transform(mat)
        if self.return_name:
            return inputs, targets, filename
        return inputs, targets

    def __len__(self):
        return len(self.dataset)


class HSITransformTestDataset(Dataset):
    def __init__(self, root, transform, size=None):
        super().__init__()
        self.dataset = MatDataFromFolder(root, size=size)
        self.transform = transform

    def __getitem__(self, index):
        mat, filename = self.dataset[index]
        gt = mat['gt'].transpose(2, 0, 1).astype('float32')
        input = self.transform(gt)
        return (input, gt), filename

    def __len__(self):
        return len(self.dataset)


class HSITrainDataset(Dataset):
    def __init__(self,
                 root,
                 input_transform,
                 target_transform,
                 common_transform,
                 repeat=1,
                 ):
        super().__init__()
        self.dataset = LMDBDataset(root, repeat=repeat)
        self.common_transform = common_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.dataset[index]
        img = self.common_transform(img)
        target = img.copy()
        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.dataset)


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform

    def __call__(self, mat):
        if self.transform:
            input = self.transform(mat[self.input_key].transpose((2, 0, 1)))
            gt = self.transform(mat[self.gt_key].transpose((2, 0, 1)))
        else:
            input = mat[self.input_key].transpose((2, 0, 1))
            gt = mat[self.gt_key].transpose((2, 0, 1))

        input = torch.from_numpy(input).float()
        gt = torch.from_numpy(gt).float()

        return input, gt


class MatDataFromFolder(Dataset):
    """Wrap mat data from folder"""

    def __init__(self, data_dir, load=scipy.io.loadmat, suffix='mat', fns=None, size=None):
        super(MatDataFromFolder, self).__init__()
        if fns is not None:
            self.filenames = [
                os.path.join(data_dir, fn) for fn in fns
            ]
        else:
            self.filenames = [
                os.path.join(data_dir, fn)
                for fn in os.listdir(data_dir)
                if fn.endswith(suffix)
            ]

        self.load = load

        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

    def __getitem__(self, index):
        filename = self.filenames[index]
        mat = self.load(filename)
        return mat, filename

    def __len__(self):
        return len(self.filenames)


class LMDBDataset(Dataset):
    def __init__(self, db_path, repeat=1, backend='pickle'):
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        self.repeat = repeat
        self.backend = backend

    def __getitem__(self, index):
        index = index % self.length
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get('{:08}'.format(index).encode('ascii'))

        if self.backend == 'caffe':
            import caffe
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)

            flat_x = np.fromstring(datum.data, dtype=np.float32)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
        else:
            x = pickle.loads(raw_datum)

        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
