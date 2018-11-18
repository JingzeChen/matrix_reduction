#ifndef CHUNK_REDUCTION_ALGORITHM_H
#define CHUNK_REDUCTION_ALGORITHM_H

#include "gpu_common.h"
#include "gpu_boundary_matrix.h"

#define LOCAL_NEGATIVE -1
#define LOCAL_POSITIVE 1

__device__ void gpu_chunk_local_reduction(gpu_boundary_matrix * matrix, dimension max_dim, ScatterAllocator::AllocatorHandle * allocator);

__device__ void gpu_mark_active_column(gpu_boundary_matrix * matrix, dimension max_dim);

__device__ void gpu_check_lowest_one_locally(gpu_boundary_matrix * matrix, indx my_col_id, indx block_id, indx chunk_start, indx row_begin,
                                             dimension cur_dim, indx * target_col, bool * ive_added);

__device__ void gpu_mark_and_clean_locally(gpu_boundary_matrix * matrix, indx my_col_id, indx row_begin, dimension cur_dim);

__device__ void gpu_add_two_cols_locally(gpu_boundary_matrix * matrix, indx my_col_id, indx target_col, ScatterAllocator::AllocatorHandle * allocator);

struct gpu_stack {
    indx * data;
    size_t data_length;
    size_t size;
};

__device__ gpu_stack * initial_gpu_stack(size_t length);

__device__ void gpu_stack_push(gpu_stack * stack, indx elem);

__device__ void gpu_resize(gpu_stack * stack, size_t new_size);

__device__ void gpu_column_simplification(gpu_boundary_matrix *matrix, indx my_col_id, ScatterAllocator::AllocatorHandle *allocator);

#endif


