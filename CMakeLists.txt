#    This file is part of stixels. (https://github.com/dhernandez0/stixels).
#
#    Copyright (c) 2016 Daniel Hernandez Juarez.

#    stixels is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    stixels is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with stixels.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.4)
project(stixels)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --use_fast_math
    -gencode=arch=compute_30,code=sm_30
    -gencode=arch=compute_35,code=sm_35
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_70,code=sm_70
    )

cuda_add_executable(
    stixels
    main.cu RoadEstimation.cu RoadEstimationKernels.cu Stixels.cu StixelsKernels.cu)

target_link_libraries( stixels ${OpenCV_LIBS} )
