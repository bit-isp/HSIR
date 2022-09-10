import os
from .multilayer import MultilayerModel


def build_t3sc(num_band):
    from omegaconf import OmegaConf
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(CURRENT_DIR, 'configs', 't3sc.yaml')
    cfg = OmegaConf.load(cfg_path)
    cfg.params.channels = num_band
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net


def t3sc():
    return build_t3sc(31)
