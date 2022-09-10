import torch
import random
import threading


class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """

    def __init__(self, use_cdhw=True):
        """ use_cdhw: True for (C, D, H, W) and False for (C, H, W) """
        self.use_cdhw = use_cdhw

    def __call__(self, hsi):
        img = torch.from_numpy(hsi)
        if self.use_cdhw:
            img = img.unsqueeze(0)
        return img.float()


class Identity(object):
    """
    Identity transformw
    """

    def __call__(self, data):
        return data


class RandomCrop(object):
    """
    Random crop transform
    """

    def __init__(self, croph, cropw):
        self.croph = croph
        self.cropw = cropw

    def __call__(self, img):
        _, h, w = img.shape
        croph, cropw = self.croph, self.cropw
        h1 = random.randint(0, h - croph)
        w1 = random.randint(0, w - cropw)
        return img[:, h1:h1 + croph, w1:w1 + cropw]


class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out
