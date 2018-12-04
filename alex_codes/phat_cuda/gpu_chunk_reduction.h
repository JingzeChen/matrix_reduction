//
// Created by 唐艺峰 on 2018/8/7.
//

#ifndef PHAT_CUDA_GPU_CHUNK_REDUCTION_H
#define PHAT_CUDA_GPU_CHUNK_REDUCTION_H

#include "gpu_common.h"
#include "gpu_matrix.h"

//! \file

//! \brief Doing columns reduction locally.
//!
//! This function does the reduction by two steps. In the first step, each chunk
//! only searches itself for target column for reduction while in the second step
//! it also searches its left neighbor.
//!
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param max_dim is the max number of dimension of the matrix.
//! \param allocator is the instance of the mallocmc_lib who manages the gpu memory.
//!
//! \return void
//!
//! \note This function cannot be simply implemented by busy spinning because of GPU's
//! warps are accurately synchronized which doesn't allow any thread jump out of loop
//! singly.
//! \warning This function still might have unseen issue when working parallel.

__device__ void gpu_chunk_local_reduction(gpu_matrix * matrix, dimension max_dim, ScatterAllocator::AllocatorHandle * allocator);

//! \brief Marking active columns parallel.
//!
//! This function marks the active columns parallel by DFS.
//!
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param max_dim is the max number of dimension of the matrix.
//!
//! \return void
//!
//! \todo Hopefully, this function could be rewritten by better parallel algorithm
//! but I'm not able to figure out.

__device__ void gpu_mark_active_column(gpu_matrix * matrix, dimension max_dim);

//! \brief Check and find current target \ref column each column.
//!
//! This function searches column's leftmost neighbors in chunk for the current
//! source column who has the same lowest row.
//!
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param my_col_id is the ID of current working column this thread.
//! \param block_id is the ID of current working block.
//! \param chunk_start is the ID of the column starting this block.
//! \param row_begin is the ID of the row considered by this block.
//! \param cur_dim is the number of current dimension.
//! \param target_col is the ID of target column we found
//! \param ive_added is the flag avoiding extra atomic operations.
//!
//! \return void

__device__ void gpu_check_lowest_one_locally(gpu_matrix * matrix, indx my_col_id, indx block_id, indx chunk_start, indx row_begin,
                                                   dimension cur_dim, indx * target_col, bool * ive_added);

//! \brief Mark local positive or negative and clear itself.
//!
//! This function checks each turn to see whether current \ref column can be marked to
//! clear unnecessary columns to reduce more.
//!
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param my_col_id is the ID of current working column this thread.
//! \param row_begin is the ID of the row considered by this block.
//! \param cur_dim is the number of current dimension.
//!
//! \return void

__device__ void gpu_mark_and_clean_locally(gpu_matrix * matrix, indx my_col_id, indx row_begin, dimension cur_dim);

//! \brief Add target \ref column to source \ref column
//!
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param my_col_id is the ID of current working column this thread.
//! \param target_col is the ID of the target column
//! \param allocator is the instance of the mallocmc_lib who manages the gpu memory.
//!
//! \return void

__device__ void gpu_add_two_cols_locally(gpu_matrix * matrix, indx my_col_id, indx target_col, ScatterAllocator::AllocatorHandle * allocator);

//! \class gpu_stack
//! \brief This is designed to be used in global columns simplification.

struct gpu_stack {
    indx * data;
    size_t data_length;
    size_t size;
};

__device__ gpu_stack * initial_gpu_stack(size_t length);

__device__ void gpu_stack_push(gpu_stack * stack, indx elem);

__device__ void gpu_resize(gpu_stack * stack, size_t new_size);

//! \brief Simplify the global columns
//!
//! Simplify all global columns which still exists.
//! After then, we only need to pass all global columns to CPU to continue the last step.
//!
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param my_col_id is the ID of current working column this thread.
//! \param allocator is the instance of the mallocmc_lib who manages the gpu memory.
//! \todo this part is the hotspot of the whole program in most time, please try optimizing it.

__device__ void gpu_column_simplification(gpu_matrix *matrix, indx my_col_id, ScatterAllocator::AllocatorHandle *allocator);

#endif //PHAT_CUDA_GPU_CHUNK_REDUCTION_H
