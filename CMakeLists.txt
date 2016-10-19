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
    )

cuda_add_executable(
    stixels
    main.cu RoadEstimation.cu RoadEstimationKernels.cu Stixels.cu StixelsKernels.cu)

target_link_libraries( stixels ${OpenCV_LIBS} )
