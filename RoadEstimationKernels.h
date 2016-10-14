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
