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

#include "StixelsKernels.h"

__inline__ __device__ float FastLog(const float v) {
	return __logf(v);
}

__inline__ __device__ float NegFastLogDiv(const float v, const float v2) {
	//return -__logf(v/v2);
	return -__logf(v) + __logf(v2);
}

__inline__ __device__ float GetPriorCost(const int vB, const int rows) {
	return NegFastLogDiv(1.0f, (float)(rows-vB));
}

__inline__ __device__ pixel_t ComputeMean(const int vB, const int vT, const pixel_t *d_sum,
		const pixel_t *d_valid, const pixel_t *d_column) {
#if ALLOW_INVALID_DISPARITIES
	const pixel_t valid_dif = d_valid[vT+1]-d_valid[vB];
	pixel_t mean = (valid_dif == 0) ? 0 : (d_sum[vT+1]-d_sum[vB])/valid_dif;
#else
	const pixel_t mean = (d_sum[vT+1]-d_sum[vB])/(vT+1-vB);
#endif

	return mean;
}

__inline__ __device__ float GetPriorCostSkyFromObject(pixel_t previous_mean, const float epsilon,
		const float prior_cost) {
	float cost = logf(2.0f)+prior_cost;

	if(previous_mean < epsilon) {
		cost = MAX_LOGPROB;
	}
	return cost;
}

__inline__ __device__ float GetPriorCostSkyFromGround(const int vB, pixel_t previous_mean,
		const int vhor, const float epsilon, float *ground_function, const float prior_cost) {
	const int previous_vT = vB-1;

	// FIXME: No tengo claro que sea asi
	const float prev_gf = ground_function[previous_vT];
	const float cost = (prev_gf == 0.0f) ? prior_cost : MAX_LOGPROB;

	return cost;
}

__inline__ __device__ float ComputeObjectDisparityRange(const float previous_mean, const float baseline,
		const float focal, const float range_objects_z) {
	float range_disp = 0;
	if(previous_mean != 0) {
		const float pmean_plus_z = (baseline*focal/previous_mean) + range_objects_z;
		range_disp = previous_mean - (baseline*focal/pmean_plus_z);
	}
	return range_disp;
}

__inline__ __device__ float GetPriorCostObjectFromGround(const int vB, float fn, const pixel_t previous_mean,
		const float *prior_objfromground, const int max_dis, const float max_disf,
		const float *ground_function, const float prior_cost, const float epsilon, const float pgrav,
		const float pblg) {
	float cost = -logf(0.7f) + prior_cost;


	const int previous_vT = vB-1;
	const float fn_previous = ground_function[previous_vT];

	if(fn > (fn_previous+epsilon)) {
		// It should not be 0, fn_previous could be almost m_max_dis-1 but m_epsilon should be small
		cost += NegFastLogDiv(pgrav, max_disf-fn_previous-epsilon);
	} else if(fn < (fn_previous-epsilon)) {
		// fn >= 0 then previous_mean-dif_dis > 0
		const float pmean_sub = fn_previous-epsilon;
		cost += NegFastLogDiv(pblg, pmean_sub);
	} else {
		cost += NegFastLogDiv(1.0f-pgrav-pblg, 2.0f*epsilon);
	}
	return cost;
}

__inline__ __device__ float GetPriorCostObjectFromObject(const int vB, const float fn,
		const pixel_t previous_mean, float *prior_objfromobj, const float *object_disparity_range,
		const int vhor, const int max_dis, const float max_disf, const float pord, const float prior_cost) {
	const int previous_vT = vB-1;
	float cost = (previous_vT < vhor) ? -logf(0.7f) : logf(2.0f);
	cost += prior_cost;


	const float dif_dis = object_disparity_range[(int) previous_mean];

	if(fn > (previous_mean+dif_dis)) {
		// It should not be 0, previous_mean could be almost m_max_dis-1 but dif_dis should be small
		cost += NegFastLogDiv(pord, max_disf-previous_mean-dif_dis);
	} else if(fn < (previous_mean-dif_dis)) {
		// fn >= 0 then previous_mean-dif_dis > 0
		const float pmean_sub = previous_mean-dif_dis;
		cost += NegFastLogDiv(1.0f-pord, pmean_sub);
	} else {
		cost = MAX_LOGPROB;
	}
	return cost;
}

__inline__ __device__ float GetPriorCostObjectFromSky(const float fn, const float max_disf,
		const float prior_cost, const float epsilon) {
	float cost = MAX_LOGPROB;

	if(fn > epsilon) {
		cost = NegFastLogDiv(1.0f, max_disf-epsilon) + prior_cost;
	}

	return cost;
}

__inline__ __device__ float GetPriorCostGround(const float prior_cost) {
	return -logf(0.3f)+prior_cost;
}

__inline__ __device__ float GetPriorCostObjectFirst(const bool below_vhor_vT, const float rows_log,
		const float max_dis_log) {
	const float pvt = below_vhor_vT ? logf(2.0f) : 0.0f;
	return rows_log + pvt + max_dis_log;
}

__inline__ __device__ float GetPriorCostGroundFirst(const float rows_log) {
	// Only below horizon
	return logf(2.0f) + rows_log;
}

__inline__ __device__ float GetDataCostSky(const pixel_t d, const float pnexists_given_sky_log,
		const float normalization_sky, const float inv_sigma2_sky, const float puniform_sky,
		const float nopnexists_given_sky_log) {

	float data_cost = pnexists_given_sky_log;
	if(!ALLOW_INVALID_DISPARITIES || d != INVALID_DISPARITY) {
		const float pgaussian = normalization_sky + d*d*inv_sigma2_sky;

		const float p_data = fminf(puniform_sky, pgaussian);
		data_cost = p_data+nopnexists_given_sky_log;
	}
	return data_cost;
}

__inline__ __device__ float GetDataCostGround(const float fn, const int v, const pixel_t d,
		const float pnexists_given_ground_log, const float *normalization_ground,
		const float *inv_sigma2_ground, const float puniform, const float nopnexists_given_ground_log) {

	float data_cost = pnexists_given_ground_log;
	if(!ALLOW_INVALID_DISPARITIES || d != INVALID_DISPARITY) {
		const float model_diff = (d-fn);
		const float pgaussian = normalization_ground[v] + model_diff*model_diff*inv_sigma2_ground[v];

		const float p_data = fminf(puniform, pgaussian);
		data_cost = p_data + nopnexists_given_ground_log;
	}
	return data_cost;
}

__inline__ __device__ float warp_prefix_sum(const int i, const int fn, const pixel_t* __restrict__ d_disparity,
		const float* __restrict__ d_obj_cost_lut, const StixelParameters params, float *s_data, const float add) {
	const int lane = threadIdx.x % WARP_SIZE;
	const int col = blockIdx.x;

	int dis = 0;
	if(i+lane < params.rows) {
		dis = d_disparity[col*params.rows+i+lane];
	}
	float cost = d_obj_cost_lut[fn*params.max_dis+dis];
	if(lane == 0) {
		cost += add;
	}

	#pragma unroll
	for (int j = 1; j <= WARP_SIZE; j *= 2) {
		float n = __shfl_up(cost, j);

		if (lane >= j) cost += n;
	}

	s_data[i+lane+1] = cost;

	return __shfl(cost, WARP_SIZE-1);
}

__inline__ __device__ void ComputePrefixSumWarp2(const int fn, const pixel_t* __restrict__ d_disparity,
		const float* __restrict__ d_obj_cost_lut, const StixelParameters params, float *arr,
		const int n, const int n_power2) {
	float add = 0.0f;
	const int lane = threadIdx.x % WARP_SIZE;

	if(lane == 0) {
		arr[0] = 0.0f;
	}

	for(int i = 0; i < n_power2; i += WARP_SIZE) {
		add = warp_prefix_sum(i, fn, d_disparity, d_obj_cost_lut, params, arr, add);
	}
}

__global__ void StixelsKernel(const pixel_t* __restrict__ d_disparity, const StixelParameters params,
		const float* __restrict__ d_ground_function, const float* __restrict__ d_normalization_ground,
		const float* __restrict__ d_inv_sigma2_ground, const float* __restrict__ d_object_disparity_range,
		const float* __restrict__ d_object_lut, Section* __restrict__ d_stixels) {
	const int col = blockIdx.x;
	const int row = threadIdx.x;

	extern __shared__ int s[];
	float *sky_lut = (float*)&s;											// rows+1
	float *ground_lut = &sky_lut[params.rows_power2];						// rows+1
	float *ground_function = &ground_lut[params.rows_power2];				// rows
	float *object_disparity_range = &ground_function[params.rows_power2];	// max_dis
	float *cost_table = &object_disparity_range[params.max_dis];			// rows*3
	int16_t *index_table = (int16_t*)&cost_table[params.rows_power2*3];		// rows*3
	pixel_t *sum = (pixel_t*) &index_table[params.rows_power2*3];			// rows+1
	pixel_t *valid = NULL;
	pixel_t *column = NULL;

	float *prior_objfromground = NULL;
	float *prior_objfromobj = NULL;


	if(row < params.rows) {
		const pixel_t d = d_disparity[col*params.rows+row];

		cost_table[row] = MAX_LOGPROB;
		cost_table[params.rows+row] = MAX_LOGPROB;
		cost_table[2*params.rows+row] = MAX_LOGPROB;

		if(row < params.max_dis) {
			object_disparity_range[row] = d_object_disparity_range[row];
		}

#if ALLOW_INVALID_DISPARITIES
		const int va = d != INVALID_DISPARITY;
		valid[row] = (pixel_t) va;
		sum[row] = ((pixel_t)va)*d;
#else
		sum[row] = d;
#endif

		sky_lut[row] = (row < params.vhor) ? MAX_LOGPROB : GetDataCostSky(d, params.pnexists_given_sky_log,
				params.normalization_sky, params.inv_sigma2_sky, params.puniform_sky,
				params.nopnexists_given_sky_log);

		ground_function[row] = d_ground_function[row];
		const float gf = ground_function[row];
		ground_lut[row] = (row >= params.vhor) ? MAX_LOGPROB : GetDataCostGround(gf, row, d,
				params.pnexists_given_ground_log, d_normalization_ground, d_inv_sigma2_ground,
				params.puniform, params.nopnexists_given_ground_log);

		// Reason: Usage of "column" in the precomputation of Object LUT and
		//			need writes to luts before ComputePrefixSum
		__syncthreads();

#if ALLOW_INVALID_DISPARITIES
		ComputePrefixSum(valid, params.rows_power2);
#endif
		ComputePrefixSum(sum, params.rows_power2);
		ComputePrefixSum(ground_lut, params.rows_power2);
		ComputePrefixSum(sky_lut, params.rows_power2);

		const float max_disf = (float) params.max_dis;

		const int vT = row;
		const int obj_data_idx = col*params.rows_power2*params.max_dis;

		// First segment: Special case vB = 0
		{
			const int vB = 0;
			__syncthreads();

			// Compute data terms
			const pixel_t obj_fn = ComputeMean(vB, vT, sum, valid, column);
			const int obj_fni = (int) floorf(obj_fn);

			const float cost_ground_data = ground_lut[vT+1] - ground_lut[vB];
			const float cost_object_data = d_object_lut[obj_data_idx+obj_fni*params.rows_power2+vT+1] -
					d_object_lut[obj_data_idx+obj_fni*params.rows_power2+vB];

			// Compute priors costs
			const int index_pground = vT*3+GROUND;
			const int index_pobject = vT*3+OBJECT;
			const bool below_vhor_vT = vT <= params.vhor;

			if(below_vhor_vT) {
				const float cost_ground_prior = GetPriorCostGroundFirst(params.rows_log);
				// Ground
				const float curr_cost_ground = cost_table[index_pground];
				const float cost_ground = cost_ground_data + cost_ground_prior;
				if(cost_ground < curr_cost_ground) {
					cost_table[index_pground] = cost_ground;
					index_table[index_pground] = GROUND;
				}
			}

			// Object
			const float cost_object_prior = GetPriorCostObjectFirst(below_vhor_vT, params.rows_log,
					params.max_dis_log);
			const float curr_cost_object = cost_table[index_pobject];
			const float cost_object = cost_object_data + cost_object_prior;
			if(cost_object < curr_cost_object) {
				cost_table[index_pobject] = cost_object;
				index_table[index_pobject] = OBJECT;
			}
		}

		for(int vB = 1; vB < params.rows; vB++) {
			__syncthreads();

			if(vT >= vB) {
				const pixel_t obj_fn = ComputeMean(vB, vT, sum, valid, column);
				const int obj_fni = (int) floorf(obj_fn);
				const float cost_object_data = d_object_lut[obj_data_idx+obj_fni*params.rows_power2+vT+1] -
						d_object_lut[obj_data_idx+obj_fni*params.rows_power2+vB];
				const float prior_cost = GetPriorCost(vB, params.rows);

				const int previous_vT = vB-1;
				const bool below_vhor_vTprev = previous_vT < params.vhor;
				const int previous_object_vB = index_table[previous_vT*3+OBJECT] / 3;
				const pixel_t previous_mean = ComputeMean(previous_object_vB, previous_vT, sum, valid, column);

				if(below_vhor_vTprev) {
					// Ground
					const float cost_ground_data = ground_lut[vT+1] - ground_lut[vB];
					const int index_pground = vT*3+GROUND;

					const float prev_cost = GetPriorCostGround(prior_cost);
					const float cost_ground_prior1 = prev_cost + cost_table[previous_vT*3+GROUND];
					const float cost_ground_prior2 = prev_cost + cost_table[previous_vT*3+OBJECT];

					const float curr_cost_ground = cost_table[index_pground];
					const float cost_ground = cost_ground_data + fminf(cost_ground_prior1, cost_ground_prior2);
					if(cost_ground < curr_cost_ground) {
						cost_table[index_pground] = cost_ground;
						int min_prev = OBJECT;
						if(cost_ground_prior1 < cost_ground_prior2) {
							min_prev = GROUND;
						}
						index_table[index_pground] = vB*3+min_prev;
					}
				} else {
					// Sky
					const float cost_sky_data = sky_lut[vT+1] - sky_lut[vB];
					const int index_psky = vT*3+SKY;

					const float cost_sky_prior1 = GetPriorCostSkyFromGround(vB, previous_mean, params.vhor,
							params.epsilon, ground_function, prior_cost) + cost_table[previous_vT*3+GROUND];

					const float cost_sky_prior2 = GetPriorCostSkyFromObject(previous_mean, params.epsilon,
							prior_cost) + cost_table[previous_vT*3+OBJECT];

					const float curr_cost_sky = cost_table[index_psky];
					const float cost_sky = cost_sky_data + fminf(cost_sky_prior1, cost_sky_prior2);
					if(cost_sky < curr_cost_sky) {
						cost_table[index_psky] = cost_sky;
						int min_prev = OBJECT;
						if(cost_sky_prior1 < cost_sky_prior2) {
							min_prev = GROUND;
						}
						index_table[index_psky] = vB*3+min_prev;
					}
				}

				// Object
				const int index_pobject = vT*3+OBJECT;

				const float cost_object_prior1 = GetPriorCostObjectFromGround(vB, obj_fn, previous_mean,
						prior_objfromground, params.max_dis, max_disf, ground_function, prior_cost,
						params.epsilon, params.pgrav, params.pblg) + cost_table[previous_vT*3+GROUND];

				const float cost_object_prior2 = GetPriorCostObjectFromObject(vB, obj_fn, previous_mean,
						prior_objfromobj, object_disparity_range, params.vhor, params.max_dis, max_disf,
						params.pord, prior_cost) + cost_table[previous_vT*3+OBJECT];
				const float cost_object_prior3 = GetPriorCostObjectFromSky(obj_fn, max_disf, prior_cost,
						params.epsilon) + cost_table[previous_vT*3+SKY];

				const float curr_cost_object = cost_table[index_pobject];
				const float cost_object = cost_object_data + fminf(fminf(cost_object_prior1, cost_object_prior2),
						cost_object_prior3);

				if(cost_object < curr_cost_object) {
					cost_table[index_pobject] = cost_object;
					int min_prev = OBJECT;
					if(cost_object_prior1 < cost_object_prior2) {
						min_prev = GROUND;
					}
					if(cost_object_prior3 < fminf(cost_object_prior1, cost_object_prior2)) {
						min_prev = SKY;
					}
					index_table[index_pobject] = vB*3+min_prev;
				}
			}
		}
		__syncthreads();

		if(row == 0) {
			int vT = params.rows-1;
			const float last_ground = cost_table[vT*3+GROUND];
			const float last_object = cost_table[vT*3+OBJECT];
			const float last_sky = cost_table[vT*3+SKY];

			int type = GROUND;

			if(last_object < last_ground) {
				type = OBJECT;
			}
			if(last_sky < fminf(last_ground, last_object)) {
				type = SKY;
			}
			int min_idx = vT*3+type;

			int prev_vT;
			int i = 0;
			do {
				prev_vT = (index_table[min_idx] / 3)-1;
				Section sec;
				sec.vT = vT;
				sec.type = type;
				sec.vB = prev_vT+1;
				sec.disparity = (float) ComputeMean(sec.vB, sec.vT, sum, valid, column);

				d_stixels[col*params.max_sections+i] = sec;

				type = index_table[min_idx] % 3;
				vT = prev_vT;
				min_idx = prev_vT*3+type;
				i++;
			} while(prev_vT != -1);
			Section sec;
			sec.type = -1;
			d_stixels[col*params.max_sections+i] = sec;
		}
	}
}

__global__ void ComputeObjectLUT(const pixel_t* __restrict__ d_disparity,
		const float* __restrict__ d_obj_cost_lut, float* __restrict__ d_object_lut,
		const StixelParameters params, const int n_power2) {
	const int col = blockIdx.x;
	const int warp_id = threadIdx.x / WARP_SIZE;

	const int blck_step = blockDim.x / WARP_SIZE;
	for(int fn = warp_id; fn < params.max_dis; fn += blck_step) {
		ComputePrefixSumWarp2(fn, d_disparity, d_obj_cost_lut, params,
				&d_object_lut[col*params.rows_power2*params.max_dis+fn*params.rows_power2],
				params.rows, n_power2);
	}
}

__global__ void JoinColumns(pixel_t* __restrict__ d_disparity, pixel_t* __restrict__ d_out,
		const int step_size, const bool median, const int width_margin, const int rows,
		const int cols, const int real_cols) {
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int row = idx / real_cols;
	const int col = idx % real_cols;

	if(idx < real_cols*rows) {
		if(median) {
			pixel_t tmp_row[16];
			for(int i = 0; i < step_size; i++) {
				tmp_row[i] = d_disparity[row*cols+col*step_size+i+width_margin];
			}
			// Sort
			for(int i = 0; i < (step_size/2)+1; i++) {
				int min_idx = i;
				for(int j = i+1; j < step_size; j++) {
					if(tmp_row[j] < tmp_row[min_idx]) {
						min_idx = j;
					}
				}
				const pixel_t tmp = tmp_row[i];
				tmp_row[i] = tmp_row[min_idx];
				tmp_row[min_idx] = tmp;
			}
			pixel_t median = tmp_row[step_size/2];
			if(step_size % 2 == 0) {
				median = (median+tmp_row[(step_size/2)-1])/2.0f;
			}
			d_out[col*rows+rows-row-1] = median;
		} else {
			pixel_t mean = 0.0f;
			for(int i = 0; i < step_size; i++) {
				mean += d_disparity[row*cols+col*step_size+i+width_margin];
			}
			d_out[col*rows+rows-row-1] = mean / step_size;
		}
	}
}
