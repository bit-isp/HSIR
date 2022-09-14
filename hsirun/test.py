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


def eval(net, loader, name, logger, visualize, clamp, bandwise):
    logger.info('Evaluating {}'.format(name))

    net.eval()
    tracker = tl.trainer.util.MetricTracker()
    visualize_fns = {
        'gray': hsir.data.utils.visualize_gray,
        'color': hsir.data.utils.visualize_color
    }
    detail_stat = {}
    
    with torch.no_grad():
        pbar = tqdm(total=len(loader), dynamic_ncols=True)
        pbar.set_description(f'Test {name}')
        for data in loader:
            filename = data['filename'][0]
            inputs, targets = data['input'].to(device), data['target'].to(device)

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
                    join(name, tl.utils.filename(filename) + '.png'),
                    visualize_fns[visualize](outputs)
                )

            psnr = tl.metrics.mpsnr(targets, outputs).item()
            ssim = tl.metrics.mssim(targets, outputs).item()
            sam = tl.metrics.sam(targets, outputs).item()

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
            
            detail_stat[filename] = {'psnr': psnr, 'ssim': ssim, 'sam': sam}
        pbar.close()

    avg_speed = tl.utils.format_time(tracker['time'])
    logger.info(f'Average speed {avg_speed}')
    logger.info(f'Average results {tracker.summary()}')

    # log structural results
    avg_stat = {k: v for k, v in tracker.result().items()}
    tl.utils.io.yamlwrite(join(logger.log_dir, 'results.yaml'), {'avg': avg_stat, 'detail': detail_stat})

def main(args, logger):
    net = tl.utils.instantiate(args.arch)
    net = net.to(device)

    ckpt = tl.utils.dict_get(torch.load(args.resume), args.key_path)
    if ckpt is None: logger.debug(f'key_path {args.key_path} might be wrong')
    net.load_state_dict(ckpt)
    logger.info('Model size: {:.5f}M'.format(tlnn.benchmark.model_size(net)))
    
    for testset in args.testset:
        testdir = join(args.basedir, testset)
        dataset = HSITestDataset(testdir, use_chw=args.use_conv2d, return_name=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        logger = tl.logging.Logger(join(logdir, testset), name=testset)
        eval(net, loader, testset, logger, 
             args.vis if args.save_img else None, 
             args.clamp,
             args.bandwise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSIR Test script')
    parser.add_argument('-a', '--arch', required=True, help='architecture name')
    parser.add_argument('-r', '--resume', required=True, help='checkpoint')
    parser.add_argument('-t', '--testset', nargs='+', default=['icvl_512_50'], help='testset')
    parser.add_argument('--basedir', default='data', help='basedir')
    parser.add_argument('--logdir', default='results/test', help='logdir')
    parser.add_argument('--save_img', action='store_true', help='whether to save image')
    parser.add_argument('--clamp', action='store_true', help='whether clamp input into [0, 1]')
    parser.add_argument('--vis', default='color', choices=['color', 'gray'], help='how to visualize hsi')
    parser.add_argument('-kp', '--key_path', default='net', help='key path to access network state_dict in ckpt')
    parser.add_argument('--bandwise', action='store_true')
    parser.add_argument('--use-conv2d', action='store_true')
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
