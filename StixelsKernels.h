#ifndef STIXELSKERNELS_H_
#define STIXELSKERNELS_H_

#include <stdint.h>
#include "Stixels.hpp"
#include "configuration.h"

__global__ void StixelsKernel(const pixel_t* __restrict__ d_disparity, const StixelParameters params,
		const float* __restrict__ d_ground_function, const float* __restrict__ d_normalization_ground,
		const float* __restrict__ d_inv_sigma2_ground, const float* __restrict__ d_object_disparity_range,
		const float* __restrict__ d_object_lut, Section* __restrict__ d_stixels);

__global__ void JoinColumns(pixel_t* __restrict__ d_disparity, pixel_t* __restrict__ d_out,
		const int step_size, const bool median, const int width_margin, const int rows,
		const int cols, const int real_cols);

__global__ void ComputeObjectLUT(const pixel_t* __restrict__ d_disparity,
		const float* __restrict__ d_obj_cost_lut, float* __restrict__ d_object_lut,
		const StixelParameters params, const int n_power2);

template<typename T>
__inline__ __device__ void ComputePrefixSum(T *arr, const int n) {
	int offset = 1;

    for (int d = n>>1; d > 0; d >>= 1) {
		__syncthreads();
		if (threadIdx.x < d) {
			int ai = offset*(2*threadIdx.x+1)-1;
			int bi = offset*(2*threadIdx.x+2)-1;
			arr[bi] += arr[ai];
		}
		offset *= 2;
	}

    if (threadIdx.x == 0) {
    	arr[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
    	offset >>= 1;
    	__syncthreads();
    	if (threadIdx.x < d) {
    		int ai = offset*(2*threadIdx.x+1)-1;
    		int bi = offset*(2*threadIdx.x+2)-1;

    		T tmp = arr[ai];
    		arr[ai] = arr[bi];
    		arr[bi] += tmp;
    	}
    }
}

#endif /* STIXELSKERNELS_H_ */
