#include "chunk_reduction_algorithm.h"

//__device__ void gpu_chunk_local_reduction(gpu_boundary_matrix *matrix, dimension max_dim, ScatterAllocator::AllocatorHandle *allocator) {
//    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
//    int block_id = blockIdx.x;
//
//    indx chunk_start = matrix->chunk_offset[block_id];       // ID of column who starts the chunk.
//    indx chunk_end = matrix->chunk_offset[block_id + 1];     // ID of column who ends the chunk.
//    indx chunk_size = chunk_end - chunk_start;                      // Size of the chunk.
//    indx target_col = -1;
//    bool ive_added;                                                 // To make sure each column will only increase the chunk_finish_col_num once
//    indx left_chunk_size = block_id != 0 ?
//                           matrix->chunk_offset[block_id] - matrix->chunk_offset[block_id - 1] : 0;
//
//    for (dimension cur_dim = max_dim; cur_dim >= 1; cur_dim--) {
//        ive_added = false;
//        do {
//            gpu_check_lowest_one_locally(matrix, thread_id, block_id, chunk_start, chunk_start, cur_dim, &target_col, &ive_added);
//            gpu_add_two_cols_locally(matrix, thread_id, target_col, allocator);
//            target_col = -1;
//            __syncthreads();    // In fact, we don't even need this barrier but it helps us to improve efficiency by cutting down the number of loops.
//        } while (matrix->chunk_columns_finished[block_id] < (block_id != 0 ? chunk_size * ((max_dim - cur_dim) * 2 + 1) : chunk_size * (max_dim - cur_dim + 1)));
//        // If all of columns have been done this term, we can jump out of the loop to do the next step.
//        // Chunk_finish_col_num will only be added once per loop.
//        gpu_mark_and_clean_locally(matrix, thread_id, chunk_start, cur_dim);
//
//        if (block_id != 0) {    // Chunk 0 has no left neighbor who won't run following codes.
//            while (matrix->chunk_columns_finished[block_id - 1] < (block_id != 1 ?
//                                                                 left_chunk_size * ((max_dim - cur_dim) * 2 + 1) :
//                                                                 left_chunk_size * (max_dim - cur_dim + 1))) {
//                __threadfence();    // Wait until the leftmost neighbor has done his work.
//            }
//
//            ive_added = false;
//            do {
//                gpu_check_lowest_one_locally(matrix, thread_id, block_id, chunk_start, chunk_start - 1, cur_dim, &target_col, &ive_added);
//                gpu_add_two_cols_locally(matrix, thread_id, target_col, allocator);
//                target_col = -1;
//                __syncthreads();
//            } while (matrix->chunk_columns_finished[block_id] < chunk_size * (max_dim - cur_dim + 1) * 2);
//            gpu_mark_and_clean_locally(matrix, thread_id, chunk_start - 1, cur_dim);
//        }
//    }
//    matrix->column_length[thread_id] = (indx) matrix->matrix[thread_id].data_length;
//}
//
//__device__ void gpu_add_two_cols_locally(gpu_boundary_matrix *matrix, indx my_col_id, indx target_col,
//                                         ScatterAllocator::AllocatorHandle *allocator) {
//    if (target_col == -1 || matrix->column_type[my_col_id] != GLOBAL || target_col == my_col_id) {
//        return;
//    }
//    mx_add_to(matrix, target_col, my_col_id, allocator);
//}
//
//__device__ void
//gpu_check_lowest_one_locally(gpu_boundary_matrix *matrix, indx my_col_id, indx block_id, indx chunk_start, indx row_begin,
//                             dimension cur_dim, indx *target_col, bool * ive_added) {
//    if (cur_dim != mx_get_dim(matrix, my_col_id) || matrix->column_type[my_col_id] != GLOBAL) {
//        if (!*ive_added) {
//            atomicAdd((unsigned long long *) &matrix->chunk_finish_col_num[block_id], (unsigned long long) 1);
//            *ive_added = true;
//        }
//        return;
//    }
//
//    indx my_lowest_one = mx_get_max_indx(matrix, my_col_id);
//    if (my_lowest_one >= row_begin) {
//        for (indx col_id = chunk_start; col_id < my_col_id; col_id++) {
//            indx this_lowest_one = mx_get_max_indx(matrix, col_id);
//            if (this_lowest_one == my_lowest_one) {
//                *target_col = col_id;
//                if (*ive_added) {
//                    atomicAdd((unsigned long long *) &matrix->chunk_finish_col_num[block_id], (unsigned long long) -1);
//                    // If a column has been set to unfinished again, it decrease the chunk_finish_col_num to let the loop go on
//                    *ive_added = false;
//                }
//                return;
//            }
//        }
//    }
//    if (!*ive_added) {
//        atomicAdd((unsigned long long *) &matrix->chunk_finish_col_num[block_id], (unsigned long long) 1);
//        *ive_added = true;
//    }
//}

__device__ gpu_stack * initial_gpu_stack(size_t length) {
    auto data_size = round_up_to_2s(length);
    auto stack = new gpu_stack;
    stack->data = new indx[data_size];
    stack->data_length = 0;
    stack->size = (size_t) data_size;
    return stack;
}

__device__ void gpu_stack_push(gpu_stack * stack, indx elem) {
    gpu_resize(stack, stack->data_length + 1);
    stack->data[stack->data_length] = elem;
    stack->data_length++;
}

__device__ void gpu_resize(gpu_stack * stack, size_t new_size) {
    if (new_size > stack->size) {
        auto new_data = new indx[stack->size * 2];
        memcpy(new_data, stack->data, sizeof(indx) * stack->size);
        delete[] stack->data;
        stack->data = new_data;
        stack->size = stack->size * 2;
    }
}