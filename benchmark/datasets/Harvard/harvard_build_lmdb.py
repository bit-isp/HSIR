"""Create lmdb dataset"""
import os
import numpy as np
import lmdb
import caffe
from itertools import product
from scipy.ndimage import zoom
import random
from scipy.io import loadmat

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
        # data = minmax_normalize(data)
        # data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        data = minmax_normalize(data.transpose((2, 0, 1)))  # for Remote Sensing
        # Visualize3D(data)
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
        if augment:
            for i in range(new_data.shape[0]):
                new_data[i, ...] = data_augmentation(new_data[i, ...])

        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    data = preprocess(data)
    N = data.shape[0]

    print(data.shape)
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)

    # import ipdb; ipdb.set_trace()
    if os.path.exists(name + '.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name + '.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir + fn, 'fail')
                continue
            X = preprocess(X)
            N = X.shape[0]
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' % (i, len(fns), fn))

        print('done')

def create_cave():
    print('create harvard...')
    datadir = '/home/wzliu/projects/data/harvard/CZ_hsdb/'
    with open('harvard_train.txt') as f:
        fns = [l.strip() for l in f.readlines()]

    create_lmdb_train(
        datadir, fns, 'Harvard64_31', 'ref',
        crop_sizes=(1024,1024),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],        
        load=loadmat, augment=True,
    )


if __name__ == '__main__':
    create_cave()
