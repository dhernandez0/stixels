/**
    This file is part of stixels. (https://github.com/dhernandez0/stixels).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    stixels is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    stixels is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with stixels.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef ROADESTIMATIONKERNELS_H_
#define ROADESTIMATIONKERNELS_H_

#include <stdint.h>
#include "configuration.h"

__global__ void ComputeHistogram(pixel_t* __restrict__ d_disparity, int* __restrict__ d_vDisp,
		const int rows, const int cols, const int max_dis);
__global__ void ComputeMaximum(int* __restrict__ d_im, int *maximum, const int rows, const int cols);
__global__ void ComputeBinaryImage(int* __restrict__ d_im, uint8_t* __restrict__ d_out, int *maximum,
		float threshold, const int rows, const int cols);

#endif /* ROADESTIMATIONKERNELS_H_ */
