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
