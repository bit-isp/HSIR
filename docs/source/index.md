---
hide-toc: true
---

# Welcome to HSIR


```{toctree}
:maxdepth: 3
:hidden: true

getstart
dataset
benchmark
```


```{toctree}
:caption: Useful Links
:hidden:
PyPI page <https://pypi.org/project/torchlights/>
GitHub Repository <https://github.com/Zeqiang-Lai/torchlight>
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


## Acknowledgement

- [QRNN3D](https://github.com/Vandermode/QRNN3D)
- [DPHSIR](https://github.com/Zeqiang-Lai/DPHSIR)
- [MST](https://github.com/caiyuanhao1998/MST)
- [TLC](https://github.com/megvii-research/TLC)

## Citation

If you find this repo helpful, please considering citing us.

```bibtex
@article{LAI2022281,
    title = {Deep plug-and-play prior for hyperspectral image restoration},
    journal = {Neurocomputing},
    volume = {481},
    pages = {281-293},
    year = {2022},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2022.01.057},
    author = {Zeqiang Lai and Kaixuan Wei and Ying Fu},
}
```

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fbit-isp%2FHSIR&countColor=%23263759&style=flat)