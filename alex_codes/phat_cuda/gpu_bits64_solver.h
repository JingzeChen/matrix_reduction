//
// Created by 唐艺峰 on 2018/11/23.
//

#ifndef PHAT_CUDA_BITS64_SOLVER_H
#define PHAT_CUDA_BITS64_SOLVER_H

#include "gpu_common.h"
#include "gpu_matrix.h"

#define BLOCK_BITS      16

//structure stores columns of boundary matrix.
struct bits64_cols {
    indx * pos; //the indx of non-zero 64-bits array in the current column.
    bits64 * value; //the value of non-zero 64-bits array represented by long integer.
    size_t data_length;  //the number of non-zero 64-bits arrays in the current column.
    size_t size;  //maximal number of non-zero 64-bits arrays in the current column.
};

__host__ bits64_cols * create_bits64_cols(indx column_num);

__global__ void transform_to(bits64_cols * aim_cols, gpu_matrix * matrix, int column_num,
        ScatterAllocator::AllocatorHandle allocator);

__global__ void transform_back(bits64_cols * aim_cols, gpu_matrix * matrix, int column_num,
        ScatterAllocator::AllocatorHandle allocator);

__device__ void bits64_add_two_columns(bits64_cols * cols, indx target, indx source,
        ScatterAllocator::AllocatorHandle * allocator);

__device__ void bits64_chunk_local_reduction(bits64_cols * cols, gpu_matrix * matrix, dimension max_dim,
        ScatterAllocator::AllocatorHandle *allocator);

__device__ void bits64_add_two_cols_locally(bits64_cols * cols, gpu_matrix * matrix, indx my_col_id, indx target_col,
        ScatterAllocator::AllocatorHandle *allocator);

__device__ void bits64_check_lowest_one_locally(bits64_cols * cols, gpu_matrix * matrix, indx my_col_id, indx block_id, indx chunk_start, indx row_begin,
        dimension cur_dim, indx *target_col, bool * ive_added);

__device__ void bits64_mark_and_clean_locally(bits64_cols * cols, gpu_matrix * matrix, indx my_col_id, indx row_begin, dimension cur_dim);

__device__ indx bits64_get_max_index(bits64_cols * cols, indx my_col_id);

__device__ void bits64_clear_column(bits64_cols * cols, indx col_id);

#endif //PHAT_CUDA_BITS64_SOLVER_H
