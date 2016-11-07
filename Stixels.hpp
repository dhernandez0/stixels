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

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include "configuration.h"
#include "StixelsKernels.h"
#include "util.h"

#ifndef STIXELS_HPP_
#define STIXELS_HPP_

#define PIFLOAT 3.1416f

#define INVALID_DISPARITY	128.0f
#define MAX_LOGPROB			10000.0f

#define GROUND	0
#define OBJECT	1
#define SKY		2

struct Section {
	int type;
	int vB, vT;
	float disparity;
};

struct StixelParameters {
	int vhor;
	int rows;
	int rows_power2;
	int cols;
	int max_dis;
	float rows_log;
	float pnexists_given_sky_log;
	float normalization_sky;
	float inv_sigma2_sky;
	float puniform_sky;
	float nopnexists_given_sky_log;
	float pnexists_given_ground_log;
	float puniform;
	float nopnexists_given_ground_log;
	float pnexists_given_object_log;
	float nopnexists_given_object_log;
	float baseline;
	float focal;
	float range_objects_z;
	float pord;
	float epsilon;
	float pgrav;
	float pblg;
	float max_dis_log;
	int max_sections;
	int width_margin;
};


class Stixels {
public:
    // Constructors and destructors
	Stixels();
	~Stixels();

    // Initialize and finalizes
	void Initialize();
	void Finish();

    // Methods
	float Compute();
	Section* GetStixels();
	int GetRealCols();
	int GetMaxSections();
	void SetDisparityImage(pixel_t *disp_im);
	void SetProbabilities(float pout, float pout_sky, float pground_given_nexist,
			float pobject_given_nexist, float psky_given_nexist, float pnexist_dis, float pground,
			float pobject, float psky, float pord, float pgrav, float pblg);
	void SetCameraParameters(int vhor, float focal, float baseline, float camera_tilt,
			float sigma_camera_tilt, float camera_height, float sigma_camera_height, float alpha_ground);
    void SetDisparityParameters(const int rows, const int cols, const int max_dis,
    		const float sigma_disparity_object, const float sigma_disparity_ground, float sigma_sky);
	void SetModelParameters(const int column_step, const bool median_step, float epsilon, float range_objects_z,
			int width_margin);
// ATRIBUTES
private:
	// Managers

	// GPU
	pixel_t *d_disparity;
	pixel_t *d_disparity_big;
	float *d_ground_function;
	float *d_normalization_ground;
	float *d_inv_sigma2_ground;
	float *d_normalization_object;
	float *d_inv_sigma2_object;
	float *d_object_lut;
	float *d_object_disparity_range;
	float *d_obj_cost_lut;
	cudaStream_t m_stream1, m_stream2;
	StixelParameters m_params;
	int m_max_sections;
	Section *d_stixels;

	// Variables
	pixel_t *m_disp_im;
	pixel_t *m_disp_im_modified;

	// Probabilities
	float m_pout;
	float m_pout_sky;
	float m_pnexists_given_ground;
	float m_pnexists_given_object;
	float m_pnexists_given_sky;
	float m_pord;
	float m_pgrav;
	float m_pblg;

	// Camera parameters
	float m_focal;
	float m_baseline;
	float m_camera_tilt;
	float m_sigma_camera_tilt;
	float m_camera_height;
	float m_sigma_camera_height;
	int m_vhor;

	// Disparity Parameters
	int m_max_dis;
	float m_max_disf;
	int m_rows, m_cols, m_realcols;
	float m_sigma_disparity_object;
	float m_sigma_disparity_ground;
	float m_sigma_sky;

	// Other model parameters
	int m_column_step;
	bool m_median_step;
	float m_alpha_ground;
	float m_range_objects_z;
	float m_epsilon;
	int m_width_margin;

	// Tables
	float *m_cost_table;
	int16_t *m_index_table;

	// LUTs
	pixel_t *m_sum;
	pixel_t *m_valid;
	float *m_log_lut;
	float *m_obj_cost_lut;

	// Current column data
	pixel_t *m_column;

	// Values of ground function
	float *m_ground_function;

	// Frequently used values
	float m_max_dis_log;
	float m_rows_log;
	float m_pnexists_given_sky_log;
	float m_nopnexists_given_sky_log;
	float m_pnexists_given_ground_log;
	float m_nopnexists_given_ground_log;
	float m_pnexists_given_object_log;
	float m_nopnexists_given_object_log;

	// Data Term precomputation
	float m_puniform;
	float m_puniform_sky;
	float m_normalization_sky;
	float m_inv_sigma2_sky;
	float *m_normalization_ground;
	float *m_inv_sigma2_ground;
	float *m_normalization_object;
	float *m_inv_sigma2_object;
	float *m_object_disparity_range;

	// Result
	Section *m_stixels;

	// Methods
	void PrecomputeSky();
	void PrecomputeGround();
	void PrecomputeObject();
	float GetDataCostObject(const float fn, const int dis, const float d);
	float ComputeObjectDisparityRange(const float previous_mean);
	pixel_t ComputeMean(const int vB, const int vT, const int u);
	float GroundFunction(const int v);
	float FastLog(float v);

	template<typename T>
	void PrintTable(T *table) {
		for(int i = m_rows-1; i >= 0; i--) {
			std::cout << i << "\t" << table[i*3] << "\t" << table[i*3+1]
			              << "\t" << table[i*3+2] << std::endl;
		}
	}
};



#endif /* STIXELS_HPP_ */
