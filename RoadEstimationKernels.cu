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

#include "RoadEstimationKernels.h"

__global__ void ComputeHistogram(pixel_t* __restrict__ d_disparity, int* __restrict__ d_vDisp, const int rows,
		const int cols, const int max_dis) {
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int row = idx / cols;

	if(idx < cols*rows) {
		const pixel_t d = d_disparity[idx];
		if(d != 0) {
			int col = (int) d;

			atomicAdd(&d_vDisp[row*max_dis+col], 1);
		}

	}
}


__global__ void ComputeMaximum(int* __restrict__ d_im, int *maximum, const int rows, const int cols) {
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if(idx < cols*rows) {
		const int p = d_im[idx];

		atomicMax(maximum, p);
	}
}

__global__ void ComputeBinaryImage(int* __restrict__ d_im, uint8_t* __restrict__ d_out, int *maximum,
		float threshold, const int rows, const int cols) {
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if(idx < cols*rows) {
		const float p = (float) d_im[idx];
		d_out[idx] = (p > (*maximum)*threshold) ? 255 : 0;
	}
}
