import argparse
import numpy as np
from os.path import join

from torchlight.utils import instantiate, locate
from torchlight.nn.utils import adjust_learning_rate, get_learning_rate

import hsir.data.dataloader as loaders
from hsir.trainer import Trainer
from hsir.scheduler import MultiStepSetLR


def train_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', required=True)
    parser.add_argument('--noise', default='gaussian', choices=['gaussian', 'complex'])
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='name of the experiment, if not specified, arch will be used.')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('-s', '--schedule', type=str, default='hsir.schedule.denoise_default')
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--bandwise', action='store_true')
    parser.add_argument('--use-conv2d', action='store_true')
    parser.add_argument('--train-root', type=str, default='data/ICVL64_31_100.db')
    parser.add_argument('--test-root', type=str, default='data')
    parser.add_argument('--save-root', type=str, default='checkpoints')
    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    cfg = parser.parse_args()
    cfg.gpu_ids = [int(id) for id in cfg.gpu_ids.split(',')]
    cfg.name = cfg.arch if cfg.name is None else cfg.name
    return cfg


def main():
    cfg = train_cfg()
    net = instantiate(cfg.arch)
    schedule = locate(cfg.schedule)
    trainer = Trainer(
        net,
        lr=schedule.base_lr,
        save_dir=join(cfg.save_root, cfg.name),
        gpu_ids=cfg.gpu_ids,
        bandwise=cfg.bandwise,
    )
    trainer.logger.print(cfg)
    if cfg.resume: trainer.load(cfg.resume)

    # preare dataset
    if cfg.noise == 'gaussian':
        train_loader1 = loaders.gaussian_loader_train_s1(cfg.train_root, cfg.use_conv2d)
        train_loader2 = loaders.gaussian_loader_train_s2_16(cfg.train_root, cfg.use_conv2d)
        val_name = 'icvl_512_50'
        val_loader = loaders.gaussian_loader_val(join(cfg.test_root, val_name), cfg.use_conv2d)
    else:
        train_loader = loaders.complex_loader_train(cfg.train_root, cfg.use_conv2d)
        val_name = 'icvl_512_mixture'
        val_loader = loaders.complex_loader_val(join(cfg.test_root, val_name), cfg.use_conv2d)

    """Main loop"""
    if cfg.lr: adjust_learning_rate(trainer.optimizer, cfg.lr)  # override lr
    lr_scheduler = MultiStepSetLR(trainer.optimizer, schedule.lr_schedule, epoch=trainer.epoch)
    epoch_per_save = cfg.save_freq
    best_psnr = 0
    while trainer.epoch < schedule.max_epochs:
        np.random.seed()  # reset seed per epoch, otherwise the noise will be added with a specific pattern
        trainer.logger.print('Epoch [{}] Use lr={}'.format(trainer.epoch, get_learning_rate(trainer.optimizer)))

        # train
        if cfg.noise == 'gaussian':
            if trainer.epoch == 30: best_psnr = 0
            if trainer.epoch < 30: trainer.train(train_loader1)
            else: trainer.train(train_loader2, warm_up=trainer.epoch == 30)
        else:
            trainer.train(train_loader, warm_up=trainer.epoch == 80)

        # save ckpt
        metrics = trainer.validate(val_loader, val_name)
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            trainer.save_checkpoint('model_best.pth')
        if trainer.epoch % epoch_per_save == 0:
            trainer.save_checkpoint()
        trainer.save_checkpoint('model_latest.pth')

        lr_scheduler.step()


if __name__ == '__main__':
    main()
