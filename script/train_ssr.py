import os
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torchlight.utils import instantiate, locate
from torchlight.nn.utils import adjust_learning_rate, get_learning_rate

from hsir.data.ssr.dataset import TrainDataset, ValidDataset
from hsir.trainer import Trainer
from hsir.scheduler import MultiStepSetLR


def train_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', required=True)
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='name of the experiment, if not specified, arch will be used.')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--schedule', type=str, default='hsir.schedule.denoise_default')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--resume-path', '-rp', type=str, default=None)
    parser.add_argument('--data-root', type=str, default='data/rgb2hsi')
    parser.add_argument('--save-root', type=str, default='checkpoints/ssr')
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
        save_dir=os.path.join(cfg.save_root, cfg.name),
        gpu_ids=cfg.gpu_ids,
    )
    trainer.logger.print(cfg)
    if cfg.resume: trainer.load(cfg.resume_path)

    dataset = TrainDataset(cfg.data_root)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=8, pin_memory=True)

    dataset = ValidDataset(cfg.data_root)
    val_loader = DataLoader(dataset, batch_size=1)

    """Main loop"""
    if cfg.lr: adjust_learning_rate(trainer.optimizer, cfg.lr)  # override lr
    lr_scheduler = MultiStepSetLR(trainer.optimizer, schedule.lr_schedule, epoch=trainer.epoch)
    epoch_per_save = 10
    best_psnr = 0
    while trainer.epoch < schedule.max_epochs:
        trainer.logger.print('Epoch [{}] Use lr={}'.format(trainer.epoch, get_learning_rate(trainer.optimizer)))

        trainer.train(train_loader)

        # save ckpt
        trainer.save_checkpoint('model_latest.pth')
        metrics = trainer.validate(val_loader, 'NITRE')
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            trainer.save_checkpoint('model_best.pth')
        if trainer.epoch % epoch_per_save == 0:
            trainer.save_checkpoint()

        lr_scheduler.step()


if __name__ == '__main__':
    main()
