# HSIR

[![PyPI](https://img.shields.io/pypi/v/hsir)](https://pypi.org/project/hsir/) [![Documentation Status](https://readthedocs.org/projects/hsir/badge/?version=latest)](https://hsir.readthedocs.io/en/latest/?badge=latest)

Out-of-box Hyperspectral Image Restoration Toolbox

<img src="https://github.com/Vandermode/QRNN3D/raw/master/imgs/PaviaU.gif" height="140px"/>  <img src="https://github.com/Vandermode/QRNN3D/raw/master/imgs/Indian_pines.gif" height="140px"/>  <img src="https://github.com/Vandermode/QRNN3D/raw/master/imgs/Urban.gif" height="140px"/> 

<sub>*Denoising for remotely sensed images from [QRNN3D](https://github.com/Vandermode/QRNN3D)*</sub>

## Install

```shell
pip install hsir
```

## Usage

Here are some runable examples, please refer to the code for more options.

```shell
python hsirun.train -a qrnn3d.qrnn3d
python hsirun.test -a qrnn3d.qrnn3d -r qrnn3d.pth -t icvl_512_50
```

## Benchmark

[Pretrained Models](https://1drv.ms/u/s!AuS3o7sEiuJnf6F4THmqDMtDCwQ?e=JpfLP3) | [Training Log](https://pan.baidu.com/s/1OiuTArnqhjQmkCdehpmelQ) | [Datasets](https://pan.baidu.com/s/1BkNYhb9CBtXnKsQjNwYFyg) 
<br>
<sub>*Baidu Drive's Share Code=HSIR*</sub>


<details>
<summary>Supported Models</summary>

<br>

- [HSID-CNN](https://github.com/qzhang95/HSID-CNN) (TGRS'2018)
- [QRNN3D](https://github.com/Vandermode/QRNN3D) (TNNLS'2020)
- [TS3C](https://github.com/inria-thoth/T3SC) (NeurIPS'2021)
- [GRUNet](https://github.com/Zeqiang-Lai/DPHSIR) (Neurocomputing'2021)
- [TRQ3D](https://github.com/LiPang/TRQ3DNet) (Remote Sensing'2022)
- [SST](https://github.com/MyuLi/SST) (AAAI'2022)
- [SERT](https://github.com/MyuLi/SERT) (CVPR'2023)
- [MAN](https://github.com/Zeqiang-Lai/MAN) (arXiv'2023)
- [HSDT](https://github.com/Zeqiang-Lai/HSDT) (arXiv'2023)
</details>

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
    <td>18.59</td>
    <td>0.110</td>
    <td>.0807</td>
    <td>14.15</td>
    <td>0.046</td>
    <td>0.991</td>
    <td>11.23</td>
    <td>0.025</td>
    <td>1.105</td>
    <td>17.34</td>
    <td>0.114</td>
    <td>0.859</td>
  </tr>
  <tr>
    <td>BM4D</td>
    <td></td>
    <td>154</td>
    <td></td>
    <td>38.45</td>
    <td>0.934</td>
    <td>0.126</td>
    <td>35.60</td>
    <td>0.889</td>
    <td>0.169</td>
    <td>33.70</td>
    <td>0.845</td>
    <td>0.207</td>
    <td>37.66</td>
    <td>0.914</td>
    <td>0.143</td>
  </tr>
  <tr>
    <td>TDL</td>
    <td></td>
    <td>18</td>
    <td></td>
    <td>40.58</td>
    <td>0.957</td>
    <td>0.062</td>
    <td>38.01</td>
    <td>0.932</td>
    <td>0.085</td>
    <td>36.36</td>
    <td>0.909</td>
    <td>0.105</td>
    <td>39.91</td>
    <td>0.946</td>
    <td>0.072</td>
  </tr>
  <tr>
    <td>ITSReg</td>
    <td></td>
    <td>907</td>
    <td></td>
    <td>41.48</td>
    <td>0.961</td>
    <td>0.088</td>
    <td>38.88</td>
    <td>0.941</td>
    <td>0.098</td>
    <td>36.71</td>
    <td>0.923</td>
    <td>0.112</td>
    <td>40.62</td>
    <td>0.953</td>
    <td>0.087</td>
  </tr>
  <tr>
    <td>LLRT</td>
    <td></td>
    <td>627</td>
    <td></td>
    <td>41.99</td>
    <td>0.967</td>
    <td>0.056</td>
    <td>38.99</td>
    <td>0.945</td>
    <td>0.075</td>
    <td>37.36</td>
    <td>0.930</td>
    <td>0.087</td>
    <td>40.97</td>
    <td>0.956</td>
    <td>0.064</td>
  </tr>
  <tr>
    <td>KBR</td>
    <td></td>
    <td>1755</td>
    <td></td>
    <td>41.48</td>
    <td>0.984</td>
    <td>0.088</td>
    <td>39.16</td>
    <td>0.974</td>
    <td>0.100</td>
    <td>36.71</td>
    <td>0.961</td>
    <td>0.113</td>
    <td>40.68</td>
    <td>0.979</td>
    <td>0.080</td>
  </tr>
  <tr>
    <td>WLRTR</td>
    <td></td>
    <td>1600</td>
    <td></td>
    <td>42.62</td>
    <td>0.988</td>
    <td>0.056</td>
    <td>39.72</td>
    <td>0.978</td>
    <td>0.073</td>
    <td>37.52</td>
    <td>0.967</td>
    <td>0.095</td>
    <td>41.66</td>
    <td>0.983</td>
    <td>0.064</td>
  </tr>
  <tr>
    <td>NGmeet</td>
    <td></td>
    <td>166</td>
    <td></td>
    <td>42.99</td>
    <td>0.989</td>
    <td>0.050</td>
    <td>40.26</td>
    <td>0.980</td>
    <td>0.059</td>
    <td>38.66</td>
    <td>0.974</td>
    <td>0.067</td>
    <td>42.23</td>
    <td>0.985</td>
    <td>0.053</td>
  </tr>
  <tr>
    <td>HSID</td>
    <td>0.40</td>
    <td>3</td>
    <td></td>
    <td>38.70</td>
    <td>0.949</td>
    <td>0.103</td>
    <td>36.17</td>
    <td>0.919</td>
    <td>0.134</td>
    <td>34.31</td>
    <td>0.886</td>
    <td>0.161</td>
    <td>37.80</td>
    <td>0.935</td>
    <td>0.116</td>
  </tr>
  <tr>
    <td>QRNN3D</td>
    <td>0.86</td>
    <td>0.73</td>
    <td></td>
    <td>42.22</td>
    <td>0.988</td>
    <td>0.062</td>
    <td>40.15</td>
    <td>0.982</td>
    <td>0.074</td>
    <td>38.30</td>
    <td>0.974</td>
    <td>0.094</td>
    <td>41.37</td>
    <td>0.985</td>
    <td>0.068</td>
  </tr>
  <tr>
    <td>TS3C</td>
    <td>0.83</td>
    <td>0.95</td>
    <td></td>
    <td>42.36</td>
    <td>0.986</td>
    <td>0.079</td>
    <td>40.47</td>
    <td>0.980</td>
    <td>0.087</td>
    <td>39.05</td>
    <td>0.974</td>
    <td>0.096</td>
    <td>41.52</td>
    <td>0.983</td>
    <td>0.085</td>
  </tr>
  <tr>
    <td>GRUNet</td>
    <td>14.2</td>
    <td>0.87</td>
    <td></td>
    <td>42.84</td>
    <td>0.989</td>
    <td>0.052</td>
    <td>40.75</td>
    <td>0.983</td>
    <td>0.062</td>
    <td>39.02</td>
    <td>0.977</td>
    <td>0.080</td>
    <td>42.03</td>
    <td>0.987</td>
    <td>0.057</td>
  </tr>
</tbody>
</table>
</details>

<details>
  <summary>Complex Denoising on ICVL</summary>
<table>
<thead>
  <tr>
    <th rowspan="2"></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="3">non-iid</th>
    <th colspan="3">g+stripe</th>
    <th colspan="3">g+deadline</th>
    <th colspan="3">g+impulse</th>
    <th colspan="3">mixture</th>
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
    <td>18.25</td>
    <td>0.168</td>
    <td>0.898</td>
    <td>17.80</td>
    <td>0.159</td>
    <td>0.910</td>
    <td>17.61</td>
    <td>0.155</td>
    <td>0.917</td>
    <td>14.80</td>
    <td>0.114</td>
    <td>0.926</td>
    <td>14.08</td>
    <td>0.099</td>
    <td>0.944</td>
  </tr>
  <tr>
    <td>LRMR</td>
    <td></td>
    <td></td>
    <td></td>
    <td>32.80</td>
    <td>0.719</td>
    <td>0.185</td>
    <td>32.62</td>
    <td>0.717</td>
    <td>0.187</td>
    <td>31.83</td>
    <td>0.709</td>
    <td>0.227</td>
    <td>29.70</td>
    <td>0.623</td>
    <td>0.311</td>
    <td>28.68</td>
    <td>0.608</td>
    <td>0.353</td>
  </tr>
  <tr>
    <td>LRTV</td>
    <td></td>
    <td></td>
    <td></td>
    <td>33.62</td>
    <td>0.905</td>
    <td>0.077</td>
    <td>33.49</td>
    <td>0.905</td>
    <td>0.078</td>
    <td>32.37</td>
    <td>0.895</td>
    <td>0.115</td>
    <td>31.56</td>
    <td>0.871</td>
    <td>0.242</td>
    <td>30.47</td>
    <td>0.858</td>
    <td>0.287</td>
  </tr>
  <tr>
    <td>NMoG</td>
    <td></td>
    <td></td>
    <td></td>
    <td>34.51</td>
    <td>0.812</td>
    <td>0.187</td>
    <td>33.87</td>
    <td>0.799</td>
    <td>0.265</td>
    <td>32.87</td>
    <td>0.797</td>
    <td>0.276</td>
    <td>28.60</td>
    <td>0.652</td>
    <td>0.486</td>
    <td>27.31</td>
    <td>0.632</td>
    <td>0.513</td>
  </tr>
  <tr>
    <td>TDTV</td>
    <td></td>
    <td></td>
    <td></td>
    <td>38.14</td>
    <td>0.944</td>
    <td>0.075</td>
    <td>37.67</td>
    <td>0.940</td>
    <td>0.081</td>
    <td>36.15</td>
    <td>0.930</td>
    <td>0.099</td>
    <td>36.67</td>
    <td>0.935</td>
    <td>0.094</td>
    <td>34.77</td>
    <td>0.919</td>
    <td>0.113</td>
  </tr>
  <tr>
    <td>HSID</td>
    <td>0.40</td>
    <td>3</td>
    <td></td>
    <td>38.40</td>
    <td>0.947</td>
    <td>0.095</td>
    <td>37.77</td>
    <td>0.942</td>
    <td>0.104</td>
    <td>37.65</td>
    <td>0.940</td>
    <td>0.102</td>
    <td>35.00</td>
    <td>0.899</td>
    <td>0.174</td>
    <td>34.05</td>
    <td>0.888</td>
    <td>0.181</td>
  </tr>
  <tr>
    <td>TS3C</td>
    <td>0.83</td>
    <td>0.95</td>
    <td></td>
    <td>41.12</td>
    <td>0.986</td>
    <td>0.069</td>
    <td>40.66</td>
    <td>0.985</td>
    <td>0.077</td>
    <td>39.38</td>
    <td>0.982</td>
    <td>0.100</td>
    <td>35.92</td>
    <td>0.951</td>
    <td>0.205</td>
    <td>34.36</td>
    <td>0.945</td>
    <td>0.230</td>
  </tr>
  <tr>
    <td>QRNN3D</td>
    <td>0.86</td>
    <td>0.73</td>
    <td></td>
    <td>42.79</td>
    <td>0.978</td>
    <td>0.052</td>
    <td>42.35</td>
    <td>0.976</td>
    <td>0.055</td>
    <td>42.23</td>
    <td>0.976</td>
    <td>0.056</td>
    <td>39.23</td>
    <td>0.945</td>
    <td>0.109</td>
    <td>38.25</td>
    <td>0.938</td>
    <td>0.107</td>
  </tr>
  <tr>
    <td>GRUNet</td>
    <td>14.2</td>
    <td>0.87</td>
    <td></td>
    <td>42.89</td>
    <td>0.992</td>
    <td>0.047</td>
    <td>42.39</td>
    <td>0.991</td>
    <td>0.050</td>
    <td>42.11</td>
    <td>0.991</td>
    <td>0.050</td>
    <td>40.70</td>
    <td>0.985</td>
    <td>0.067</td>
    <td>38.51</td>
    <td>0.981</td>
    <td>0.081</td>
  </tr>
</tbody>
</table>
</details>


## Citation

If you find this repo helpful, please considering citing us.

```bibtex
@misc{hsir,
	author={Zeqiang Lai, Miaoyu Li, Ying Fu},
	title={HSIR: Out-of-box Hyperspectral Image Restoration Toolbox},
	year={2022},
	url={https://github.com/bit-isp/HSIR},
}
```

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fbit-isp%2FHSIR&countColor=%23263759&style=flat)
