from skimage import io
import numpy as np
import scipy.io as sio
import os

def load_tif_img(filepath):
    img = io.imread(filepath)
    img = img.astype(np.float32)
    return img


root = '/media/exthdd/datasets/hsi/real_hsi/real_dataset'

names = os.listdir(os.path.join(root, 'gt'))
names = [n for n in names if n.endswith('tif')]

save_dir = '/media/exthdd/datasets/hsi/real_hsi/real_dataset/test34_50'
os.makedirs(save_dir, exist_ok=True)

names = os.listdir('data/real34')

for n in names:
    n = n[:-4]
    print(n)
    gt = load_tif_img(os.path.join(root, 'gt', n+'.tif')).transpose((1, 2, 0))
    input = load_tif_img(os.path.join(root, 'input50', n+'.tif')).transpose((1, 2, 0))
    gt = gt / gt.max()
    input = input / input.max()
    from skimage import exposure
    input = exposure.match_histograms(input, gt, multichannel=False)
    input = input[:688,:512,:]
    gt = gt[:688,:512,:]
    sio.savemat(os.path.join(save_dir, n + '.mat'), {'gt': gt, 'input': input})