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

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include "Stixels.hpp"
#include "RoadEstimation.h"
#include "configuration.h"

#define OVERWRITE	true

void SaveStixels(std::vector<Section> *stixels, const int real_cols, const char *fname) {
	std::ofstream fp;
	fp.open (fname, std::ofstream::out | std::ofstream::trunc);
	//fp << "Writing this to a file.\n";
	if(fp.is_open()) {
		for(size_t i = 0; i < real_cols; i++) {
			std::vector<Section> sections_vec = stixels[i];
			for(size_t j = 0; j < sections_vec.size(); j++) {
				Section section = sections_vec.at(j);
				fp << section.type << "," << section.vB << "," << section.vT << "," << section.disparity << ";";
			}
			// Column finished
			fp << std::endl;
		}
		fp.close();
	} else {
		std::cerr << "Counldn't write file: " << fname << std::endl;
	}
}

void SaveStixels(Section *stixels, const int real_cols, const int max_segments, const char *fname) {
	std::ofstream fp;
	fp.open (fname, std::ofstream::out | std::ofstream::trunc);
	//fp << "Writing this to a file.\n";
	if(fp.is_open()) {
		for(size_t i = 0; i < real_cols; i++) {
			for(size_t j = 0; j < max_segments; j++) {
				Section section = stixels[i*max_segments+j];
				if(section.type == -1) {
					break;
				}
				// If disparity is 0 it is sky
				if(section.type == OBJECT && section.disparity < 1.0f) {
					section.type = SKY;
				}
				fp << section.type << "," << section.vB << "," << section.vT << "," << section.disparity << ";";
			}
			// Column finished
			fp << std::endl;
		}
		fp.close();
	} else {
		std::cerr << "Counldn't write file: " << fname << std::endl;
	}
}

bool FileExists(const char *fname) {
	struct stat buffer;
	return (stat (fname, &buffer) == 0);
}

int main(int argc, char *argv[]) {
	if(argc < 3) {
		std::cerr << "Usage: stixels dir max_disparity" << std::endl;
		return -1;
	}
	//nvtxNameOsThread("Stixels");
	const char* directory = argv[1];
	const int max_dis = atoi(argv[2]);
	const char* disparity_dir = "disparities";
	const char* stixel_dir = "stixels";

	DIR *dp;
	struct dirent *ep;
	char abs_dis_dir[PATH_MAX];
	sprintf(abs_dis_dir, "%s/%s", directory, disparity_dir);
	dp = opendir(abs_dis_dir);
	if (dp == NULL) {
		std::cerr << "Invalid directory: " << abs_dis_dir << std::endl;
		exit(EXIT_FAILURE);
	}
	char dis_file[PATH_MAX];
	char stixel_file[PATH_MAX];

	/* Parameters
	 *
	 */

	/* Disparity Parameters */
	const float sigma_disparity_object = 1.0f;
	const float sigma_disparity_ground = 2.0f;
	const float sigma_sky = 0.1f; // Should be small compared to sigma_dis

	/* Probabilities */
	const float pout = 0.15f;
	const float pout_sky = 0.4f;
	const float pord = 0.2f;
	const float pgrav = 0.1f;
	const float pblg = 0.04f;

	//
	// Must add 1
	const float pground_given_nexist = 0.36f;
	const float pobject_given_nexist = 0.28f;
	const float psky_given_nexist = 0.36f;

	const float pnexist_dis = 0.0f;
	const float pground = 1.0f/3.0f;
	const float pobject = 1.0f/3.0f;
	const float psky = 1.0f/3.0f;

	/* Camera Paramters */
	int vhor;

	// Virtual parameters
	const float focal = 704.7082f;
	const float baseline = 0.8f;
	const float camera_center_y = 384.0f;
	const int column_step = 5;
	const int width_margin = 0;

	float camera_tilt;
	const float sigma_camera_tilt = 0.05f;
	float camera_height;
	const float sigma_camera_height = 0.05f;
	//const float camera_center_x = 651.216186523f;
	float alpha_ground;

	/* Model Parameters */
	const bool median_step = false;
	const float epsilon = 3.0f;
	const float range_objects_z = 10.20f; // in meters

	bool first_time = true;
	Stixels stixles;
	RoadEstimation road_estimation;
	std::vector<float> times;
	pixel_t *im;

	while ((ep = readdir(dp)) != NULL) {
		if (!strcmp (ep->d_name, "."))
			continue;
		if (!strcmp (ep->d_name, ".."))
			continue;
		sprintf(dis_file, "%s/%s/%s", directory, disparity_dir, ep->d_name);
		sprintf(stixel_file, "%s/%s/%s.%s", directory, stixel_dir, ep->d_name, "stixels");

		if(!FileExists(stixel_file) || OVERWRITE) {
			cv::Mat dis = cv::imread(dis_file, cv::IMREAD_UNCHANGED);
			if(!dis.data) {
				std::cerr << "Couldn't read the file " << dis_file << std::endl;
				return EXIT_FAILURE;
			}

			std::cout << ep->d_name << std::endl;

			const int rows = dis.rows;
			const int cols = dis.cols;

			if(first_time) {
				stixles.SetDisparityParameters(rows, cols, max_dis, sigma_disparity_object, sigma_disparity_ground, sigma_sky);
				stixles.SetProbabilities(pout, pout_sky, pground_given_nexist, pobject_given_nexist, psky_given_nexist, pnexist_dis, pground, pobject, psky, pord, pgrav, pblg);
				stixles.SetModelParameters(column_step, median_step, epsilon, range_objects_z, width_margin);
				stixles.SetCameraParameters(0.0f, focal, baseline, 0.0f, sigma_camera_tilt, 0.0f, sigma_camera_height, 0.0f);
				stixles.Initialize();
				road_estimation.Initialize(camera_center_y, baseline, focal, rows, cols, max_dis);

				CUDA_CHECK_RETURN(cudaMallocHost((void**)&im, rows*cols*sizeof(pixel_t)));
			}
			for(int i = 0; i < dis.rows; i++) {
				for(int j = 0; j < dis.cols; j++) {
					const pixel_t d = (float) dis.at<uint16_t>(i, j)/256.0f;
					im[i*dis.cols+j] = d;
				}
			}

			// Compute some camera parameters
			stixles.SetDisparityImage(im);

			const bool ok = road_estimation.Compute(im);
			if(!ok) {
				printf("Can't compute\n");
				first_time = false;
				continue;
			}

			// Get Camera Parameters
			camera_tilt = road_estimation.GetPitch();
			camera_height = road_estimation.GetCameraHeight();
			vhor = road_estimation.GetHorizonPoint();
			alpha_ground = road_estimation.GetSlope();

			std::cout << "Camera Parameters -> Tilt: " << camera_tilt << " Height: " << camera_height << " vHor: " << vhor << " alpha_ground: " << alpha_ground << std::endl;

			stixles.SetCameraParameters(vhor, focal, baseline, camera_tilt, sigma_camera_tilt, camera_height, sigma_camera_height, alpha_ground);

			const float elapsed_time_ms = stixles.Compute();
			times.push_back(elapsed_time_ms);

			Section *stx = stixles.GetStixels();

			SaveStixels(stx, stixles.GetRealCols(), stixles.GetMaxSections(), stixel_file);
			first_time = false;
		}
	}
	if(!first_time) {
		stixles.Finish();
		road_estimation.Finish();
	}

	float mean = 0.0f;
	for(int i = 0; i < times.size(); i++) {
		mean += times.at(i);
	}
	mean = mean / times.size();
	std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f/mean << " fps" << std::endl;
	CUDA_CHECK_RETURN(cudaFreeHost(im));

	return 0;
}
