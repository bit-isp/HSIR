import argparse
from os.path import join

import torch
import torch.utils.data
from tqdm import tqdm

import hsir.model
import hsir.data.utils
from hsir.data import HSITestDataset

import torchlight as tl
import torchlight.nn as tlnn

tl.metrics.set_data_format('chw')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(net, loader, name, logger, visualize, clamp):
    logger.info('Evaluating {}'.format(name))

    net.eval()
    tracker = tl.trainer.util.MetricTracker()
    visualize_fns = {
        'gray': hsir.data.utils.visualize_gray,
        'color': hsir.data.utils.visualize_color
    }

    with torch.no_grad():
        pbar = tqdm(total=len(loader), dynamic_ncols=True)
        pbar.set_description(f'Test {name}')
        for i, (data, filename) in enumerate(loader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            if clamp:
                inputs = torch.clamp(inputs, 0., 1.)
            tl.utils.timer.tic()
            outputs = net(inputs)
            torch.cuda.synchronize()
            run_time = tl.utils.timer.toc()

            inputs = inputs.squeeze(1)
            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)

            if visualize:
                logger.save_img(
                    join(name, tl.utils.filename(filename[0]) + '.png'),
                    visualize_fns[visualize](outputs)
                )

            psnr = tl.metrics.mpsnr(targets, outputs)
            ssim = tl.metrics.mssim(targets, outputs)
            sam = tl.metrics.sam(targets, outputs)

            tracker.update('psnr', psnr)
            tracker.update('ssim', ssim)
            tracker.update('sam', sam)
            tracker.update('time', run_time)

            summary = {
                'psnr': '{0:.4f}'.format(psnr),
                'ssim': '{0:.4f}'.format(ssim),
                'sam': '{0:.4f}'.format(sam),
                'run_time': '{0}'.format(tl.utils.format_time(run_time)),
            }
            logger.debug('{}: {}'.format(filename[0], summary))
            pbar.set_postfix({k: f'{v:.4f}' for k, v in tracker.result().items()})
            pbar.update()
        pbar.close()

    avg_speed = tl.utils.format_time(tracker['time'])
    logger.info(f'Average speed {avg_speed}')
    logger.info(f'Average results {tracker.summary()}')

    # log structural results
    results = {k: str(v) for k, v in tracker.result().items()}
    tl.utils.io.yamlwrite(join(logger.log_dir, 'results.yaml'), results)

def main(args, logger):
    net = tl.utils.instantiate_from(hsir.model, args.arch)
    net = net.to(device)

    ckpt = tl.utils.dict_get(torch.load(args.resume), args.key_path)
    if ckpt is None: logger.debug(f'key_path {args.key_path} might be wrong')
    net.load_state_dict(ckpt)
    logger.info('Model size: {:.5f}M'.format(tlnn.benchmark.model_size(net)))
    
    for testset in args.testset:
        testdir = join(args.basedir, testset)
        dataset = HSITestDataset(testdir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        eval(net, loader, testset, logger, args.vis if args.save_img else None, args.clamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSIR Test script')
    parser.add_argument('-a', '--arch', default='qrnn3d', help='architecture name')
    parser.add_argument('-t', '--testset', nargs='+', default=['icvl_512_50'], help='testset')
    parser.add_argument('-r', '--resume', required=True, help='checkpoint')
    parser.add_argument('--basedir', default='/share/dataset/hsi/icvl/test', help='basedir')
    parser.add_argument('--logdir', default='RESULTS/TEST', help='logdir')
    parser.add_argument('--save_img', action='store_true', help='whether to save image')
    parser.add_argument('--clamp', action='store_true', help='whether clamp input into [0, 1]')
    parser.add_argument('--vis', default='color', choices=['color', 'gray'], help='how to visualize hsi')
    parser.add_argument('-kp', '--key_path', default='net', help='key path to access network state_dict in ckpt')
    args = parser.parse_args()

    logdir = tl.utils.auto_rename(join(args.logdir, args.arch), ignore_ext=True)
    logger = tl.logging.Logger(logdir, name='test')

    logger.debug(tl.utils.get_cmd())
    logger.debug(f'arch: {args.arch}')
    logger.debug(f'ckpt: {args.resume}')
    logger.debug(f'basedir: {args.basedir}')
    logger.debug(f'testset: {args.testset}')
    logger.debug(f'save_img: {args.save_img}')

    main(args, logger)
