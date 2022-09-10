import os
import argparse
import numpy as np

from torchlight.utils import instantiate_from
from torchlight.nn.utils import adjust_learning_rate, get_learning_rate

import hsir.model
import hsir.data.dataloader as loaders
from hsir.trainer import Trainer


def train_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', required=True)
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='name of the experiment, if not specified, arch will be used.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--bandwise', action='store_true')
    parser.add_argument('--use-conv2d', action='store_true')
    parser.add_argument('--resume-path', '-rp', type=str, default=None)
    parser.add_argument('--train-root', type=str, default='data/ICVL64_31_100.db')
    parser.add_argument('--test-root', type=str, default='data')
    parser.add_argument('--save-root', type=str, default='checkpoints')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    cfg = parser.parse_args()
    cfg.gpu_ids = [int(id) for id in cfg.gpu_ids.split(',')]
    cfg.name = cfg.arch if cfg.name is None else cfg.name
    return cfg


def main():
    cfg = train_cfg()
    net = instantiate_from(hsir.model, cfg.arch)
    trainer = Trainer(
        net,
        save_dir=os.path.join(cfg.save_root, cfg.name),
        gpu_ids=cfg.gpu_ids,
        bandwise=cfg.bandwise,
        use_conv2d=cfg.use_conv2d,
    )
    trainer.logger.print(cfg)
    if cfg.resume: trainer.load(cfg.resume_path)

    train_loader1 = loaders.gaussian_loader_train_s1(cfg.train_root, not cfg.use_conv2d)
    train_loader2 = loaders.gaussian_loader_train_s2_16(cfg.train_root, not cfg.use_conv2d)

    test_root = cfg.test_root
    testsets = ['icvl_512_50']
    val_loaders = [
        loaders.gaussian_loader_val(os.path.join(test_root, testset), not cfg.use_conv2d)
        for testset in testsets
    ]

    """Main loop"""
    base_lr = 1e-3
    adjust_learning_rate(trainer.optimizer, cfg.lr)

    epoch_per_save = 10
    best_psnr = 0
    while trainer.epoch < 80:
        np.random.seed()  # reset seed per epoch, otherwise the noise will be added with a specific pattern
        if trainer.epoch == 20:
            adjust_learning_rate(trainer.optimizer, base_lr * 0.1)
        if trainer.epoch == 30:
            adjust_learning_rate(trainer.optimizer, base_lr)
        if trainer.epoch == 45:
            adjust_learning_rate(trainer.optimizer, base_lr * 0.1)
        if trainer.epoch == 50:
            adjust_learning_rate(trainer.optimizer, base_lr * 0.1)
        if trainer.epoch == 55:
            adjust_learning_rate(trainer.optimizer, base_lr * 0.05)
        if trainer.epoch == 60:
            adjust_learning_rate(trainer.optimizer, base_lr * 0.01)
        if trainer.epoch == 65:
            adjust_learning_rate(trainer.optimizer, base_lr * 0.005)
        if trainer.epoch == 75:
            adjust_learning_rate(trainer.optimizer, base_lr * 0.001)
        trainer.logger.print('lr=', get_learning_rate(trainer.optimizer))

        if trainer.epoch < 30:
            trainer.train(train_loader1)
        else:
            trainer.train(train_loader2, warm_up=trainer.epoch == 30)

        if trainer.epoch == 30: best_psnr = 0

        metrics = trainer.validate(val_loaders[0], 'icvl-validate-50')
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            trainer.save_checkpoint('model_best.pth')

        trainer.save_checkpoint('model_latest.pth')
        if trainer.epoch % epoch_per_save == 0:
            trainer.save_checkpoint()


if __name__ == '__main__':
    main()
