import os
import numpy as np
import cv2
import scipy.io as sio

root = '/home/wzliu/projects/data/cave'
names = os.listdir(root)
names = [n for n in names if n.endswith('ms')]

save_dir = '/home/wzliu/projects/data/cave_mat'
os.makedirs(save_dir, exist_ok=True)

for n in names:
    data = []
    for b in range(31):
        path = os.path.join(root, n, n, '{}_{:0>2d}.png'.format(n, b + 1))
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        data.append(np.expand_dims(img, 2))
    data = np.concatenate(data, axis=2)

    print(n)
    sio.savemat(os.path.join(save_dir, n + '.mat'), {'gt': data})
