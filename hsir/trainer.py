import os
from functools import partial
from os.path import join
from datetime import datetime

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import torchlight as tl

from qqdm import format_str, qqdm
from tqdm import tqdm
from torchlight.trainer.util import MetricTracker
from torchlight.nn.benchmark import model_size
from skimage.metrics import peak_signal_noise_ratio
from sync_batchnorm import DataParallelWithCallback, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

tl.metrics.set_data_format('chw')

PBAR = {
    'qqdm': partial(qqdm),
    'tqdm': partial(tqdm, dynamic_ncols=True, ascii=' >=')
    
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def init_params(net, init_type='kn'):
    if init_type != 'edsr':
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                if init_type == 'kn':
                    init.kaiming_normal_(m.weight, mode='fan_out')
                if init_type == 'ku':
                    init.kaiming_uniform_(m.weight, mode='fan_out')
                if init_type == 'xn':
                    init.xavier_normal_(m.weight)
                if init_type == 'xu':
                    init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, 
                                SynchronizedBatchNorm2d, SynchronizedBatchNorm3d)):        
                init.constant_(m.weight, 1)
                if m.bias is not None: 
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
       
def MSIQA(X, Y):
    psnr = tl.metrics.mpsnr(X, Y)
    ssim = tl.metrics.mssim(X, Y)
    sam = tl.metrics.sam(X, Y)
    return psnr, ssim, sam
             
class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, content):
        content = str(content)
        with open(self.path, 'a') as f:
            dtstr = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            f.write(dtstr+' - ')
            f.write(content + '\n')
            
    def print(self, *value, **kwargs):
        print(*value, **kwargs)
        self.log(' '.join([str(v) for v in value]))

class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

mpsnr = Bandwise(partial(peak_signal_noise_ratio, data_range=1))
      
class Trainer:
    def __init__(
        self, 
        net, 
        lr=1e-3,
        clip=1e6,
        seed=2022,
        init_type='kn',
        save_dir='results',
        gpu_ids=[0],
        pbar='qqdm',
        bandwise=False,
    ):
        self.net = net
        self.clip = clip
        self.save_dir = ensure_dir(save_dir)
        self.logger = Logger(os.path.join(save_dir, 'train.txt'))
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda' if gpu_ids else 'cpu')
        self.pbar = PBAR[pbar]
        self.bandwise = bandwise
        self.epoch = 0
        self.iteration = 0
        
        torch.manual_seed(seed)
        if gpu_ids: torch.cuda.manual_seed(seed)

        init_params(self.net, init_type=init_type)  # disable for default initialization
        
        if len(self.gpu_ids) > 1:
            self.net = DataParallelWithCallback(self.net, device_ids=self.gpu_ids)

        self.criterion = nn.MSELoss()

        self.net = self.net.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.logger.print(self.net)
        self.logger.print("Number of parameter: %.2fM" % (model_size(self.net)))

    def _train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        total_loss = 0
        total_norm = None
        if self.bandwise:
            outputs = []
            for _, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                o = self.net(i)
                outputs.append(o)
                loss = self.criterion(o, t)
                loss.backward()
                total_loss += loss.item()
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            total_loss += loss.item()

        total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.optimizer.step()
        return outputs, total_loss, total_norm     
            
    def train(self, train_loader, warm_up=False):
        self.net.train()

        tracker = MetricTracker()

        if warm_up:
            from warmup_scheduler import GradualWarmupScheduler
            scheduler_warmup = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1,
                total_epoch=len(train_loader)
            )

        pbar = self.pbar(total=len(train_loader))
        pbar.set_description('Epoch: {}'.format(format_str('yellow', self.epoch)))

        for batch_idx, data in enumerate(train_loader):
            inputs, targets = data['input'].to(self.device), data['target'].to(self.device)
            outputs, loss_data, total_norm = self._train_step(inputs, targets)
            psnr = np.mean(mpsnr(outputs, targets))
            # psnr = tl.metrics.mpsnr(outputs.squeeze(1), targets.squeeze(1))
            
            tracker.update('loss', loss_data)
            tracker.update('psnr', psnr)

            stat = {
                'Loss': '{:.4e}'.format(tracker['loss']),
                'Norm': '{:.4e}'.format(total_norm),
                'PSNR': '{:.4f}'.format(tracker['psnr']),
            }
            pbar.set_postfix(stat)
            pbar.update()

            if batch_idx % 20 == 0:
                self.logger.log('Epoch %d - [%d|%d]: AvgLoss: %.4e | Loss: %.4e | Norm: %.4e | PSNR: %.4f'
                                % (self.epoch, batch_idx, len(train_loader),
                                   tracker['loss'], loss_data, total_norm, tracker['psnr']))

            if warm_up:
                scheduler_warmup.step()

            self.iteration += 1

        self.epoch += 1

    def _eval_step(self, inputs):
        if self.bandwise:
            outputs = []
            for i in inputs.split(1, 1):
                o = self.net(i)
                outputs.append(o)
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = self.net(inputs)
        return outputs
                
    def validate(self, valid_loader, name):
        self.net.eval()

        tracker = MetricTracker()
        pbar = self.pbar(total=len(valid_loader), dynamic_ncols=True)
        self.logger.log('[i] Eval dataset {}...'.format(name))

        with torch.no_grad():
            for data in valid_loader:
                inputs, targets = data['input'].to(self.device), data['target'].to(self.device)
                outputs = self._eval_step(inputs)
                if not self.use_2dconv:
                    outputs = outputs.squeeze(1)
                    targets = targets.squeeze(1)
                psnr, ssim, sam = MSIQA(outputs, targets)

                tracker.update('psnr', psnr)
                tracker.update('ssim', ssim)
                tracker.update('sam', sam)

                stat = {
                    'PSNR': '{:.4f}'.format(tracker['psnr']),
                    'SSIM': '{:.4f}'.format(tracker['ssim']),
                    'SAM': '{:.4f}'.format(tracker['sam']),
                }
                pbar.set_postfix(stat)
                pbar.update()
            
        self.logger.log('Epoch {} | {}'.format(self.epoch, tracker.summary()))
        return tracker.result()

    def load(self, resume_path=None, load_opt=True):
        if resume_path is None:
            resume_path = join(self.save_dir, 'model_latest.pth')

        self.logger.print('==> Resuming from checkpoint %s..' % resume_path)
        checkpoint = torch.load(resume_path)

        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.get_net().load_state_dict(checkpoint['net'])
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.print("LR: %f" % self.optimizer.param_groups[0]['lr'])

    def save_checkpoint(self, name=None, **kwargs):
        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        state.update(kwargs)

        if name is None:
            name = f'model_epoch_{self.epoch}_{self.iteration}.pth'
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = join(self.save_dir, name)
        torch.save(state, save_path)
        self.logger.print("Checkpoint saved to {}".format(save_path))

    def get_net(self):
        if len(self.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net

  