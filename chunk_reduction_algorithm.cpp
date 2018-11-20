#include "chunk_reduction_algorithm.h"

__device__ void gpu_chunk_local_reduction(gpu_boundary_matrix *matrix, dimension max_dim, ScatterAllocator::AllocatorHandle *allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;

    indx chunk_start = matrix->chunk_offset[block_id];       // ID of column who starts the chunk.
    indx chunk_end = matrix->chunk_offset[block_id + 1];     // ID of column who ends the chunk.
    indx chunk_size = chunk_end - chunk_start;                      // Size of the chunk.
    indx target_col = -1;
    bool ive_added;                                                 // To make sure each column will only increase the chunk_finish_col_num once
    indx left_chunk_size = block_id != 0 ?
                           matrix->chunk_offset[block_id] - matrix->chunk_offset[block_id - 1] : 0;

    for (dimension cur_dim = max_dim; cur_dim >= 1; cur_dim--) {
        ive_added = false;
        do {
            gpu_check_lowest_one_locally(matrix, thread_id, block_id, chunk_start, chunk_start, cur_dim, &target_col, &ive_added);
            gpu_add_two_cols_locally(matrix, thread_id, target_col, allocator);
            target_col = -1;
            __syncthreads();    // In fact, we don't even need this barrier but it helps us to improve efficiency by cutting down the number of loops.
        } while (matrix->chunk_columns_finished[block_id] < (block_id != 0 ? chunk_size * ((max_dim - cur_dim) * 2 + 1) : chunk_size * (max_dim - cur_dim + 1)));
        // If all of columns have been done this term, we can jump out of the loop to do the next step.
        // Chunk_finish_col_num will only be added once per loop.
        gpu_mark_and_clean_locally(matrix, thread_id, chunk_start, cur_dim);

        if (block_id != 0) {    // Chunk 0 has no left neighbor who won't run following codes.
            while (matrix->chunk_columns_finished[block_id - 1] < (block_id != 1 ?
                                                                 left_chunk_size * ((max_dim - cur_dim) * 2 + 1) :
                                                                 left_chunk_size * (max_dim - cur_dim + 1))) {
                __threadfence();    // Wait until the leftmost neighbor has done his work.
            }

            ive_added = false;
            do {
                gpu_check_lowest_one_locally(matrix, thread_id, block_id, chunk_start, chunk_start - 1, cur_dim, &target_col, &ive_added);
                gpu_add_two_cols_locally(matrix, thread_id, target_col, allocator);
                target_col = -1;
                __syncthreads();
            } while (matrix->chunk_columns_finished[block_id] < chunk_size * (max_dim - cur_dim + 1) * 2);
            gpu_mark_and_clean_locally(matrix, thread_id, chunk_start - 1, cur_dim);
        }
    }
    matrix->column_length[thread_id] = (indx) matrix->matrix[thread_id].data_length;
}

__device__ void gpu_add_two_cols_locally(gpu_boundary_matrix *matrix, indx my_col_id, indx target_col,
                                         ScatterAllocator::AllocatorHandle *allocator) {
    if (target_col == -1 || matrix->column_type[my_col_id] != GLOBAL || target_col == my_col_id) {
        return;
    }
    add_two_columns(matrix->matrix, target_col, my_col_id, allocator);
}

__device__ void
gpu_check_lowest_one_locally(gpu_boundary_matrix * matrix, indx my_col_id, indx block_id, indx chunk_start, indx row_begin,
                             dimension cur_dim, indx *target_col, bool * ive_added) {
    if (cur_dim != get_dim(matrix->dims, my_col_id) || matrix->column_type[my_col_id] != GLOBAL) {
        if (!*ive_added) {
            atomicAdd((unsigned long long *) &matrix->chunk_columns_finished[block_id], (unsigned long long) 1);
            *ive_added = true;
        }
        return;
    }

    indx my_lowest_one = get_max_index(matrix->matrix, my_col_id);
    if (my_lowest_one >= row_begin) {
        for (indx col_id = chunk_start; col_id < my_col_id; col_id++) {
            indx this_lowest_one = get_max_index(matrix->matrix, col_id);
            if (this_lowest_one == my_lowest_one) {
                *target_col = col_id;
                if (*ive_added) {
                    atomicAdd((unsigned long long *) &matrix->chunk_columns_finished[block_id], (unsigned long long) -1);
                    // If a column has been set to unfinished again, it decrease the chunk_finish_col_num to let the loop go on
                    *ive_added = false;
                }
                return;
            }
        }
    }
    if (!*ive_added) {
        atomicAdd((unsigned long long *) &matrix->chunk_columns_finished[block_id], (unsigned long long) 1);
        *ive_added = true;
    }
}

__device__ void gpu_mark_and_clean_locally(gpu_boundary_matrix * matrix, indx my_col_id, indx row_begin,
                                           dimension cur_dim) {
    if (cur_dim != get_dim(matrix->dims, my_col_id) || matrix->column_type[my_col_id] != GLOBAL) {
        return;
    }

    indx my_lowest_one = get_max_index(matrix->matrix, my_col_id);
    if (matrix->lowest_one_lookup[my_lowest_one] == -1 && my_lowest_one >= row_begin) {
        matrix->lowest_one_lookup[my_lowest_one] = my_col_id;
        matrix->column_type[my_col_id] = LOCAL_NEGATIVE;
        matrix->column_type[my_lowest_one] = LOCAL_POSITIVE;
        clear_column(matrix->matrix, my_lowest_one);
    }
}

__device__ void gpu_mark_active_column(gpu_boundary_matrix * matrix, dimension max_dim) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;

    indx chunk_start = matrix->chunk_offset[block_id];
    indx chunk_end = matrix->chunk_offset[block_id + 1];
    indx chunk_size = chunk_end - chunk_start;

    // Please see the paper to know what happens below
    // https://arxiv.org/pdf/1303.0477.pdf

    bool im_done = false;
    indx cur_row_idx = 0, col_data_length = 0;
    auto col = &matrix->matrix[thread_id];
    indx * col_src_data = new indx[col->data_length * 64];

    for (int i = 0; i < col->data_length; i++) {
        indx cur_block_pos = col->pos[i];
        indx cur_block_inside_pos = 0;
        unsigned long long cur_block_value = col->value[i];
        unsigned long long mask = 1lu << 63;
        while (mask != 0) {
            if (mask & cur_block_value != 0) {
                col_src_data[col_data_length] = cur_block_pos * 64 + cur_block_inside_pos;
                col_data_length++;
            }
            mask >>= 1;
            cur_block_inside_pos++;
        }
    }

    do {
        if (!im_done) {
            if (is_empty(matrix->matrix, thread_id) || cur_row_idx == col_data_length || matrix->column_type[thread_id] == GLOBAL) {
                im_done = true;
                matrix->is_active[thread_id] = matrix->column_type[thread_id] == GLOBAL;
                matrix->is_ready_for_mark[thread_id] = true;
                atomicAdd((unsigned long long *) &matrix->chunk_columns_finished[block_id], (unsigned long long) 1);
            } else {
                indx cur_row = col_src_data[cur_row_idx];
                if (matrix->column_type[cur_row] == GLOBAL) {
                    im_done = true;
                    matrix->is_active[thread_id] = true;
                    matrix->is_ready_for_mark[thread_id] = true;
                    atomicAdd((unsigned long long *) &matrix->chunk_columns_finished[block_id], (unsigned long long) 1);
                } else if (matrix->column_type[cur_row] == LOCAL_POSITIVE) {
                    indx cur_row_lowest_one = matrix->lowest_one_lookup[cur_row];
                    if (cur_row_lowest_one == thread_id || cur_row_lowest_one == -1) {
                        cur_row_idx++;
                    } else if (matrix->is_ready_for_mark[cur_row_lowest_one]) {
                        if (matrix->is_active[cur_row_lowest_one]) {
                            im_done = true;
                            matrix->is_active[thread_id] = true;
                            matrix->is_ready_for_mark[thread_id] = true;
                            atomicAdd((unsigned long long *) &matrix->chunk_columns_finished[block_id], (unsigned long long) 1);
                        } else {
                            cur_row_idx++;
                        }
                    }
                } else {
                    cur_row_idx++;
                }
            }
        }
        __syncthreads();
    } while (matrix->chunk_columns_finished[block_id] < ((block_id == 0) ? (max_dim + 1) * chunk_size : (2 * max_dim + 1) * chunk_size));

    delete col_src_data;
    // To exit when all columns have done their work
}

__device__ void gpu_column_simplification(gpu_boundary_matrix *matrix, indx my_col_id,
        ScatterAllocator::AllocatorHandle * allocator) {
    if (matrix->column_type[my_col_id] != GLOBAL) {
        matrix->is_ready_for_mark[my_col_id] = true;
        return;
    }

    auto col = &matrix->matrix[my_col_id];
    auto stack = initial_gpu_stack(col->data_length);

    while(!is_empty(matrix->matrix, my_col_id)) {
        indx cur_row = get_max_index(matrix->matrix, my_col_id);
        if (matrix->column_type[cur_row] == LOCAL_NEGATIVE) {
            remove_max_index(matrix->matrix, my_col_id);
        } else if (matrix->column_type[cur_row] == LOCAL_POSITIVE) {
            if (matrix->is_ready_for_mark[matrix->lowest_one_lookup[cur_row]]) {
                if (matrix->is_active[matrix->lowest_one_lookup[cur_row]]) {
                    add_two_columns(matrix->matrix, matrix->lowest_one_lookup[cur_row], my_col_id, allocator);
                } else {
                    remove_max_index(matrix->matrix, my_col_id);
                }
            }
        } else {
            gpu_stack_push(stack, cur_row);
            remove_max_index(matrix->matrix, my_col_id);
        }
    }

    auto src_length = stack->data_length;
    auto src_data = stack->data;
    col->data_length = 0;
    indx last_pos = -1;
    for (long i = src_length - 1; i >= 0; i--) {
        indx current_pos = src_data[i] / BLOCK_BITS;
        if (last_pos != current_pos) {
            col->data_length++;
            last_pos = current_pos;
        }
    }

    if (col->data_length > col->size) {
        auto new_size = round_up_to_2s(col->data_length);
        auto new_pos = (indx *) allocator->malloc(sizeof(indx) * new_size);
        auto new_value = (unsigned long long *) allocator->malloc(sizeof(unsigned long long) * new_size);
        allocator->free(col->pos);
        allocator->free(col->value);
        col->pos = new_pos;
        col->value = new_value;
        col->size = (size_t) new_size;
    }
    last_pos = -1;
    unsigned long long last_value = 0;
    int cur_block_id = 0;

    for (long i = src_length - 1; i >= 0; i--) {
        indx current_pos = src_data[i] / BLOCK_BITS;
        if (last_pos != current_pos) {
            if (last_pos != -1) {
                col->pos[cur_block_id] = last_pos;
                col->value[cur_block_id] = last_value;
                cur_block_id++;
            }
            last_pos = current_pos;
            last_value = 0;
        }
        unsigned long long mask = ((unsigned long long) 1) << (src_data[i] % BLOCK_BITS);
        last_value |= mask;
        if(i == (src_length-1)) {
            col->pos[cur_block_id] = last_pos;
            col->value[cur_block_id] = last_value;
        }
    }
    matrix->is_ready_for_mark[my_col_id] = true;
    __threadfence();
    delete[] stack->data;
    delete stack;
}

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