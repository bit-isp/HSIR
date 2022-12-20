import os
import numpy as np
from scipy.io import loadmat, savemat


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)
    

def main(get_noise, sigma):
    root = '/home/wzliu/projects/data/cave_mat/'
    save_root = os.path.join('/home/wzliu/projects/data/cave_test/', 'cave_512_{}'.format(sigma))
    os.makedirs(save_root, exist_ok=True)

    with open('cave_test.txt') as f:
        fns = [l.strip() for l in f.readlines()]

    np.random.seed(2022)
    for fn in fns:
        print(fn)
        path = os.path.join(root, fn)
        data = loadmat(path)
        gt = data['gt']
        gt = minmax_normalize(gt)
        noise = get_noise(gt.shape)
        input = gt + noise
        data = {'input': input, 'gt': gt, 'sigma': sigma}
        savemat(os.path.join(save_root, fn+'.mat'), data)
 
 
def fixed(sigma):
    def get_noise(shape):
        return np.random.randn(*shape) * sigma / 255.    
    return get_noise

def blind(min, max):
    def get_noise(shape):
        return np.random.randn(*shape) * (min+np.random.rand(1)*(max-min)) / 255.    
    return get_noise

main(fixed(30),30)
main(fixed(50),50)
main(fixed(70),70)
main(blind(10,70),'blind')

