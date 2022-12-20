"""Create lmdb dataset"""
import os
import numpy as np
import lmdb
from itertools import product
from scipy.ndimage import zoom
import random
from scipy.io import loadmat
import pickle

def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    def PatNum(l, k, s): return (np.floor((l - k) / s) + 1)

    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])

    V = np.zeros([int(TotalPatNum)] + ksizes)  # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i] + key + 1 or None, strides[i]) for i, key in enumerate(s)])
        V[s1] = np.reshape(data[s2], (-1,))

    return V


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def crop_center(img, cropx, cropy):
    _, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx]


def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    def flipud(x): return x[:, ::-1, :]

    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    # we apply spectrum reversal for training 3D CNN, e.g. QRNN3D.
    # disable it when training 2D CNN, e.g. MemNet
    if random.random() < 0.5:
        image = image[::-1, :, :]

    return np.ascontiguousarray(image)


def create_lmdb_train(
        datadir, fns, name, matkey,
        crop_sizes, scales, ksizes, strides,
        load=loadmat, augment=True,
        seed=2022):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        new_data = []
        data = np.float32(data)
        # data = minmax_normalize(data.transpose((2, 0, 1)))  
        data = data.transpose(2, 0, 1) / 4096
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])

        for i in range(len(scales)):
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(os.path.join(datadir, fns[0]))
    gt = data['gt']
    input = data['input']
    data = preprocess(gt)
    
    N = data.shape[0]

    print(data.shape)
    map_size = data.nbytes * len(fns) * 2.4
    print('map size (GB):', map_size / 1024 / 1024 / 1024)

    if os.path.exists(name + '.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name + '.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                data = load(os.path.join(datadir, fn))
                gt = data['gt']
                input = data['input']
            except:
                print('loading', datadir + fn, 'fail')
                continue
            gt = preprocess(gt)
            input = preprocess(input)
            if augment:
                for t in range(gt.shape[0]):
                    mode = random.randint(0, 7)
                    gt[t, ...] = data_augmentation(gt[t, ...], mode)
                    input[t, ...] = data_augmentation(input[t, ...], mode)
                    
            N = gt.shape[0]
            for j in range(N):
                str_id = '{:08}'.format(k)
                k += 1
                data = {'gt': gt[j], 'input': input[j]}
                data = pickle.dumps(data)
                txn.put(str_id.encode('ascii'), data)
            print('load mat (%d/%d): %s' % (i, len(fns), fn))

        print('done')

def create_real():
    print('create real...')
    datadir = '/media/exthdd/datasets/hsi/real_hsi/real_dataset/mat/'
    with open('real_train.txt') as f:
        fns = [l.strip() for l in f.readlines()]

    create_lmdb_train(
        datadir, fns, 'Real64_31_2', 'gt',
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],        
        load=loadmat, augment=True,
    )


if __name__ == '__main__':
    create_real()
