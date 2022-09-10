# HSIR

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
    <td>42.24</td>
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
    <td>41.42</td>
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
