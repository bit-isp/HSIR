import argparse
from collections import OrderedDict
import os
from os.path import join, exists

import torch
import torch.utils.data
import imageio
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

import hsir.model
import hsir.data.utils
from hsir.data import HSITestDataset

import torchlight as tl

tl.metrics.set_data_format('chw')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bchw2hwc(x):
    return np.uint8(x.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255)


def eval(net, loader, name, logdir, clamp, bandwise):
    print('Evaluating {}'.format(name))
    os.makedirs(join(logdir, 'color'), exist_ok=True)
    os.makedirs(join(logdir, 'gray'), exist_ok=True)

    net.eval()
    tracker = tl.trainer.util.MetricTracker()
    detail_stat = {}

    with torch.no_grad():
        pbar = tqdm(total=len(loader), dynamic_ncols=True)
        pbar.set_description(name)
        for data in loader:
            filename = data['filename'][0]
            basename = tl.utils.filename(filename)
            inputs, targets = data['input'].to(device), data['target'].to(device)

            if clamp:
                inputs = torch.clamp(inputs, 0., 1.)
            tl.utils.timer.tic()
            outputs = net(inputs)
            torch.cuda.synchronize()
            run_time = tl.utils.timer.toc() / 1000

            inputs = inputs.squeeze(1)
            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)

            imageio.imwrite(
                join(logdir, 'color', basename + '.png'),
                bchw2hwc(hsir.data.utils.visualize_color(outputs))
            )
            imageio.imwrite(
                join(logdir, 'gray', basename + '.png'),
                bchw2hwc(hsir.data.utils.visualize_gray(outputs))
            )

            psnr = tl.metrics.mpsnr(targets, outputs).item()
            ssim = tl.metrics.mssim(targets, outputs).item()
            sam = tl.metrics.sam(targets, outputs).item()

            tracker.update('MPSNR', psnr)
            tracker.update('MSSIM', ssim)
            tracker.update('SAM', sam)
            tracker.update('Time', run_time)

            pbar.set_postfix({k: f'{v:.4f}' for k, v in tracker.result().items()})
            pbar.update()

            detail_stat[basename] = {'MPSNR': psnr, 'MSSIM': ssim, 'SAM': sam, 'Time': run_time}

        pbar.close()

    avg_speed = tl.utils.format_time(tracker['Time'])
    print(f'Average speed {avg_speed}')
    print(f'Average results {tracker.summary()}')

    # log structural results
    avg_stat = {k: v for k, v in tracker.result().items()}
    tl.utils.io.jsonwrite(join(logdir, 'log.json'), {'avg': avg_stat, 'detail': detail_stat})


def pretty_summary(logdir):
    stat = []
    print('')
    for folder in os.listdir(logdir):
        if os.path.isdir(join(logdir, folder)):
            path = join(logdir, folder, 'log.json')
            if exists(path):
                data = tl.utils.io.jsonload(path)
                s = OrderedDict()
                s['Name'] = folder
                s.update(data['avg'])
                stat.append(s)
    print(tabulate(stat, headers='keys', tablefmt='github'))
    print('')


def main(args, logdir):
    net = tl.utils.instantiate(args.arch)
    net = net.to(device)

    ckpt = tl.utils.dict_get(torch.load(args.resume), args.key_path)
    if ckpt is None: print(f'key_path {args.key_path} might be wrong')
    net.load_state_dict(ckpt)

    for testset in args.testset:
        testdir = join(args.basedir, testset)
        dataset = HSITestDataset(testdir, use_chw=args.use_conv2d, return_name=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        eval(net, loader, testset, join(logdir, testset), args.clamp, args.bandwise)

    pretty_summary(logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSIR Test script')
    parser.add_argument('-a', '--arch', required=True, help='architecture name')
    parser.add_argument('-n', '--name', default=None, help='save name')
    parser.add_argument('-r', '--resume', required=True, help='checkpoint')
    parser.add_argument('-t', '--testset', nargs='+', default=['icvl_512_50'], help='testset')
    parser.add_argument('--basedir', default='data', help='basedir')
    parser.add_argument('--logdir', default='results', help='logdir')
    parser.add_argument('--save_img', action='store_true', help='whether to save image')
    parser.add_argument('--clamp', action='store_true', help='whether clamp input into [0, 1]')
    parser.add_argument('-kp', '--key_path', default='net', help='key path to access network state_dict in ckpt')
    parser.add_argument('--bandwise', action='store_true')
    parser.add_argument('--use-conv2d', action='store_true')
    args = parser.parse_args()

    save_name = args.arch if args.name is None else args.name
    logdir = join(args.logdir, save_name)
    if exists(logdir):
        print(f'It seems that you have evaluated {args.arch} before.')
        pretty_summary(logdir)
        action = input('Are you sure you want to continue? (y) continue (n) exit\n')
        if action != 'y': exit()

    os.makedirs(logdir, exist_ok=True)
    with open(join(logdir, 'meta.txt'), 'w') as f:
        f.write(tl.utils.get_datetime() + '\n')
        f.write(tl.utils.get_cmd() + '\n')
        f.write(str(args) + '\n')

    main(args, logdir)
