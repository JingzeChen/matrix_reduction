cmake_minimum_required(VERSION 3.8)
project(matrix_reduction)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES chunk_reduction_algorithm.cpp gpu_boundary_matrix.cpp gpu_common.cpp gpu_spectral_sequence_reduction.cpp test.cpp)

include_directories(phat_lib mallocmc_lib /Developer/NVIDIA/CUDA-9.2/include)

add_executable(matrix_reduction ${SOURCE_FILES})