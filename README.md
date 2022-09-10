# HSIR

Out-of-box Hyperspectral Image Restoration Toolbox

<img src="https://github.com/Vandermode/QRNN3D/raw/master/imgs/PaviaU.gif" height="140px"/>  
<img src="https://github.com/Vandermode/QRNN3D/raw/master/imgs/Indian_pines.gif" height="140px"/>  
<img src="https://github.com/Vandermode/QRNN3D/raw/master/imgs/Urban.gif" height="140px"/> 

<cite><small>Denoising for remotely sensed images from [QRNN3D](https://github.com/Vandermode/QRNN3D)</small></cite>

## Install

```shell
pip install hsir
```

## Usage

Here are some runable examples, please refer to the code for more options.

```shell
python script/train.py -a qrnn3d.qrnn3d
python script/test.py -a qrnn3d.qrnn3d -t icvl_512_30 icvl_512_50 --save_img
```

## Benchmark

<details>
  <summary>Gaussian Denoising on ICVL</summary>
  <table>
<thead>
  <tr>
    <th rowspan="2"></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="3">Sigma=30</th>
    <th colspan="3">Sigma=50</th>
    <th colspan="3">Sigma=70</th>
    <th colspan="3">Sigma=Blind</th>
  </tr>
  <tr>
    <th>Params(M)</th>
    <th>Runtime(s)</th>
    <th>FLOPs</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>SAM</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>SAM</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>SAM</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>SAM</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Noisy</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>BM4D</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>
</details>

<details>
  <summary>Complex Denoising on ICVL</summary>
  
</details>

## Citation

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
