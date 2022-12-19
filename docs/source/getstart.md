# Getting Started

```{toctree}
:maxdepth: 2
```


[![PyPI](https://img.shields.io/pypi/v/hsir)](https://pypi.org/project/hsir/)

Out-of-box Hyperspectral Image Restoration Toolbox


## Install

```shell
pip install hsir
```

## Usage

Here are some runable examples, please refer to the code for more options.

```shell
python hsirun/train.py -a qrnn3d.qrnn3d
python hsirun/test.py -a qrnn3d.qrnn3d -r qrnn3d.pth -t icvl_512_50
```