# GPU-accelerated real-time stixel computation

This is the implementation of [GPU-accelerated real-time stixel computation](https://arxiv.org/abs/1610.04124), [D. Hernandez-Juarez](http://www.cvc.uab.es/people/dhernandez/) et al.

Performance obtained measured in Frames Per Second (FPS):

|                 | 1280 x 240    |   640 x 480   |   1280 x 480    |
| -------------   |:-------------:|:-------------:|:---------------:|
| NVIDIA Tegra X1 | 86.8          |    45.7       |     22.3        |
| NVIDIA Titan X  | 1000          |     581       |     373         |

## How to compile and test

Simply use CMake and target the output directory as "build". In command line this would be (from the project root folder):

```
mkdir build
cd build
cmake ..
make
```

## How to use it

Type: `./stixels dir max_disparity`

The arguments `max_disparity` is the maximum disparity of the given disparity images, there are lots of parameters you can set in "main.cu".

`dir` is the name of the directory which needs this format:

```
dir
---- left (images taken from the left camera)
---- right (right camera)
---- disparities (disparity maps)
---- stixels (results will be here)
```

An example is provided, to run it type: `./stixels ./example 128`


## Related Publications

[Embedded real-time stereo estimation via Semi-Global Matching on the GPU](http://www.sciencedirect.com/science/article/pii/S1877050916306561)
[D. Hernandez-Juarez](http://www.cvc.uab.es/people/dhernandez/), A. Chacón, A. Espinosa, D. Vázquez, J. C. Moure, and A. M. López
ICCS2016 – International Conference on Computational Science 2016

## Requirements

- OpenCV
- CUDA
- CMake

## Limitations

- Maximum image height can not be greather than 1024

## What to cite

If you use this code for your research, please kindly cite:

```
@article{stixels_gpu,
  author    = {Daniel Hernandez-Juarez and
               Antonio Espinosa and
               David V{\'{a}}zquez and
               Antonio M. L{\'{o}}pez and
               Juan Carlos Moure},
  title     = {{GPU}-accelerated real-time stixel computation},
  journal   = {CoRR},
  volume    = {abs/1610.04124},
  year      = {2016},
  url       = {http://arxiv.org/abs/1610.04124},
}

```
