cmake_minimum_required(VERSION 3.8)
find_package(CUDA QUIET REQUIRED)

project(matrix_reduction)

set(
    CUDA_NVCC_FLAGS
    $(CUDA_NVCC_FLAGS);
    -o3 -gencode arch=compute_22,code=sm_22
    CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp chunk_reduction_algorithm.cpp gpu_boundary_matrix.cpp gpu_common.h test.cpp)

include_directories(phat_lib mallocmc_lib /opt/cuda/include)

add_executable(main
    /home/chen/CLionProjects/matrix_reduction/main.cpp
    ${SOURCE_FILES})

target_link_libraries(
    main
    )
