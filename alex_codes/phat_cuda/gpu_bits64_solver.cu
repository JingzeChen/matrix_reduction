//
// Created by 唐艺峰 on 2018/11/23.
//

#include "gpu_bits64_solver.h"
#include "gpu_matrix.h"
#include<math.h>
#define BLOCK_BITS 16
#define LOG2_BLOCK_BITS 4

__host__ bits64_cols * create_bits64_cols(indx column_num) {
    bits64_cols * new_bits64_cols;
    gpuErrchk(cudaMalloc((void **) &new_bits64_cols, sizeof(bits64_cols) * column_num));
    return new_bits64_cols;
}

__global__ void transform_to(bits64_cols * aim_cols, gpu_matrix * matrix, int column_num,
        ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    auto src_length = matrix->data_length[thread_id];
    auto src_data = matrix->data[thread_id];
    auto col = &aim_cols[thread_id];

    col->data_length = 0;
    indx last_pos = -1;
    for (size_t i = 0; i < src_length; i++) {
        indx current_pos = src_data.data[i] >> LOG2_BLOCK_BITS;
        if (last_pos != current_pos) {
            col->data_length++;
            last_pos = current_pos;
        }
    }

    col->size = (size_t) round_up_to_2s(col->data_length);
    col->size = (size_t) (col->size > INITIAL_SIZE ? col->size : INITIAL_SIZE);
    col->pos = (indx *) allocator.malloc(sizeof(indx) * col->size);
    col->value = (bits64 *) allocator.malloc(sizeof(bits64) * col->size);

    last_pos = -1;
    bits64 last_value = 0;
    int cur_block_id = 0;

    for (int i = 0; i < src_length; i++) {
        indx current_pos = src_data.data[i] >> LOG2_BLOCK_BITS;
        if (last_pos != current_pos) {
            if (last_pos != -1) {
                col->pos[cur_block_id] = last_pos;
                col->value[cur_block_id] = last_value;
                cur_block_id++;
            }
            last_pos = current_pos;
            last_value = 0;
        }
        bits64 mask = ((bits64) 1) << (BLOCK_BITS - 1 - src_data.data[i] & (BLOCK_BITS - 1));
        last_value |= mask;
        if (i == src_length - 1) {
            col->pos[cur_block_id] = last_pos;
            col->value[cur_block_id] = last_value;
        }
    }
}

__global__ void transform_back(bits64_cols * aim_cols, gpu_matrix * matrix, int column_num,
        ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    auto src_length = aim_cols[thread_id].data_length;
    auto src_pos = aim_cols[thread_id].pos;
    auto src_value = aim_cols[thread_id].value;
    auto col = &matrix->data[thread_id];

    matrix->data_length[thread_id] = 0;
    if (src_length == 0) {
        mx_clear(matrix, thread_id);
        return;
    }

    for (size_t i = 0; i < src_length; i++) {
        matrix->data_length[thread_id] += count_bit_sets(src_value[i]);
    }

    auto size_needed = (size_t) round_up_to_2s(matrix->data_length[thread_id]);
    size_needed = size_needed > ADDED_SIZE ? size_needed : ADDED_SIZE;
    auto is_need_new_mem = col->size < size_needed;

    col->data_length = (size_t) matrix->data_length[thread_id];
    col->size = (is_need_new_mem) ? size_needed : col->size;

    if (is_need_new_mem) {
        allocator.free(col->data);
        col->data = (indx *) allocator.malloc(sizeof(indx) * col->size);
    }

    size_t pos = 0;
    for (size_t i = 0; i < src_length; i++) {
        auto value_block = src_pos[i];
        while (src_value[i] != 0) {
            indx value_inside = most_significant_bit_pos(src_value[i]);
            bits64 mask = 1lu << (BLOCK_BITS - 1 - value_inside);
            src_value[i] ^= mask;
            col->data[pos] = value_block * BLOCK_BITS + value_inside;
            pos++;
        }
    }

    matrix->data_length[thread_id] = col->data_length;
}

__device__ void bits64_add_two_columns(bits64_cols * matrix, indx source, indx target, ScatterAllocator::AllocatorHandle * allocator) {
    auto tgt_col = &matrix[target];
    size_t tgt_id = 0, src_id = 0, temp_id = 0;
    indx max_size = round_up_to_2s(matrix[target].data_length + matrix[source].data_length);

    max_size = max_size > ADDED_SIZE ? max_size : ADDED_SIZE;
    bool need_new_mem = tgt_col->size < max_size;
    auto new_pos = (need_new_mem) ? (indx *) allocator->malloc(sizeof(indx) * max_size) : new indx[max_size];
    auto new_value = (need_new_mem) ? (bits64 *) allocator->malloc(sizeof(bits64) * max_size)
            : new bits64 [max_size];

    while (tgt_id < matrix[target].data_length && src_id < matrix[source].data_length) {
        if (matrix[target].pos[tgt_id] == matrix[source].pos[src_id]) {
            if ((matrix[target].value[tgt_id] ^ matrix[source].value[src_id]) != 0) {
                new_pos[temp_id] = matrix[target].pos[tgt_id];
                new_value[temp_id] = matrix[target].value[tgt_id] ^ matrix[source].value[src_id];
                temp_id++;
            }
            tgt_id++;
            src_id++;
        } else if (matrix[target].pos[tgt_id] < matrix[source].pos[src_id]) {
            new_value[temp_id] = matrix[target].value[tgt_id];
            new_pos[temp_id] = matrix[target].pos[tgt_id];
            tgt_id++;
            temp_id++;
        } else {
            new_value[temp_id] = matrix[source].value[src_id];
            new_pos[temp_id] = matrix[source].pos[src_id];
            src_id++;
            temp_id++;
        }
    }

    if (src_id < matrix[source].data_length) {
        memcpy(&new_pos[temp_id], &matrix[source].pos[src_id], sizeof(indx) * (matrix[source].data_length - src_id));
        memcpy(&new_value[temp_id], &matrix[source].value[src_id], sizeof(bits64) * (matrix[source].data_length - src_id));
        temp_id += matrix[source].data_length - src_id;
    } else if (tgt_id < matrix[target].data_length) {
        memcpy(&new_pos[temp_id], &matrix[target].pos[tgt_id], sizeof(indx) * (matrix[target].data_length - tgt_id));
        memcpy(&new_value[temp_id], &matrix[target].value[tgt_id], sizeof(bits64) * (matrix[target].data_length - tgt_id));
        temp_id += matrix[target].data_length - tgt_id;
    }

    if (need_new_mem) {
        allocator->free(tgt_col->pos);
        allocator->free(tgt_col->value);
        tgt_col->pos = new_pos;
        tgt_col->value = new_value;
        tgt_col->size = (size_t) max_size;
    } else {
        memcpy(tgt_col->pos, new_pos, sizeof(indx) * temp_id);
        memcpy(tgt_col->value, new_value, sizeof(bits64) * temp_id);
        delete new_pos;
        delete new_value;
    }

    tgt_col->data_length = temp_id;
}

__device__ void bits64_chunk_local_reduction(bits64_cols * cols, gpu_matrix * matrix, dimension max_dim, ScatterAllocator::AllocatorHandle *allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;

    indx chunk_start = matrix->chunks_start_offset[block_id];       // ID of column who starts the chunk.
    indx chunk_end = matrix->chunks_start_offset[block_id + 1];     // ID of column who ends the chunk.
    indx chunk_size = chunk_end - chunk_start;                      // Size of the chunk.
    indx target_col = -1;
    bool ive_added;                                                 // To make sure each column will only increase the chunk_finish_col_num once
    indx left_chunk_size = block_id != 0 ?
            matrix->chunks_start_offset[block_id] - matrix->chunks_start_offset[block_id - 1] : 0;

    unsigned long long count_loop = 0;
    for (dimension cur_dim = max_dim; cur_dim >= 1; cur_dim--) {
        ive_added = false;
        do {
            count_loop++;
            bits64_check_lowest_one_locally(cols, matrix, thread_id, block_id, chunk_start, chunk_start, cur_dim, &target_col, &ive_added);
            bits64_add_two_cols_locally(cols, matrix, thread_id, target_col, allocator);
            target_col = -1;
            __syncthreads();    // In fact, we don't even need this barrier but it helps us to improve efficiency by cutting down the number of loops.
        } while (matrix->chunk_finish_col_num[block_id] < (block_id != 0 ? chunk_size * ((max_dim - cur_dim) * 2 + 1) : chunk_size * (max_dim - cur_dim + 1)));
        // If all of columns have been done this term, we can jump out of the loop to do the next step.
        // Chunk_finish_col_num will only be added once per loop.
        bits64_mark_and_clean_locally(cols, matrix, thread_id, chunk_start, cur_dim);
        printf("the first loop:column %d and loop times %lu and cur_dim is %ld\n", thread_id, count_loop, cur_dim);
        count_loop = 0;
        if (block_id != 0) {    // Chunk 0 has no left neighbor who won't run following codes.
            while (matrix->chunk_finish_col_num[block_id - 1] < (block_id != 1 ?
                        left_chunk_size * ((max_dim - cur_dim) * 2 + 1) :
                        left_chunk_size * (max_dim - cur_dim + 1))) {
                __threadfence();    // Wait until the leftmost neighbor has done his work.
                //count_loop++;
            }
            printf("loop of while:%lu\n", count_loop);
            count_loop=0;
            ive_added = false;
            do {
                count_loop++;
                bits64_check_lowest_one_locally(cols, matrix, thread_id, block_id, chunk_start, chunk_start - 1, cur_dim, &target_col, &ive_added);
                bits64_add_two_cols_locally(cols, matrix, thread_id, target_col, allocator);
                target_col = -1;
                __syncthreads();
            } while (matrix->chunk_finish_col_num[block_id] < chunk_size * (max_dim - cur_dim + 1) * 2);
            bits64_mark_and_clean_locally(cols, matrix, thread_id, chunk_start - 1, cur_dim);
            printf("the second loop:column %d and loop times %lu and cur_dim is %ld\n", thread_id, count_loop, cur_dim);
        }
    }
}

__device__ void bits64_add_two_cols_locally(bits64_cols * cols, gpu_matrix * matrix, indx my_col_id, indx target_col,
                                         ScatterAllocator::AllocatorHandle *allocator) {
    if (target_col == -1 || matrix->column_type[my_col_id] != GLOBAL || target_col == my_col_id) {
        return;
    }
    bits64_add_two_columns(cols, target_col, my_col_id, allocator);
}

__device__ void bits64_check_lowest_one_locally(bits64_cols * cols, gpu_matrix * matrix, indx my_col_id, indx block_id, indx chunk_start, indx row_begin,
                                                dimension cur_dim, indx *target_col, bool * ive_added) {
    if (cur_dim != mx_get_dim(matrix, my_col_id) || matrix->column_type[my_col_id] != GLOBAL) {
        if (!*ive_added) {
            atomicAdd((unsigned long long *) &matrix->chunk_finish_col_num[block_id], (unsigned long long) 1);
            *ive_added = true;
        }
        return;
    }

    indx my_lowest_one = bits64_get_max_index(cols, my_col_id);
    if (my_lowest_one >= row_begin) {
        for (indx col_id = chunk_start; col_id < my_col_id; col_id++) {
            indx this_lowest_one = bits64_get_max_index(cols, col_id);
            if (this_lowest_one == my_lowest_one) {
                *target_col = col_id;
                if (*ive_added) {
                    atomicAdd((unsigned long long *) &matrix->chunk_finish_col_num[block_id], (unsigned long long) -1);
                    // If a column has been set to unfinished again, it decrease the chunk_finish_col_num to let the loop go on
                    *ive_added = false;
                }
                return;
            }
        }
    }
    if (!*ive_added) {
        atomicAdd((unsigned long long *) &matrix->chunk_finish_col_num[block_id], (unsigned long long) 1);
        *ive_added = true;
    }
}

__device__ void bits64_mark_and_clean_locally(bits64_cols * cols, gpu_matrix * matrix, indx my_col_id, indx row_begin, dimension cur_dim) {
    if (cur_dim != mx_get_dim(matrix, my_col_id) || matrix->column_type[my_col_id] != GLOBAL) {
        return;
    }

    indx my_lowest_one = bits64_get_max_index(cols, my_col_id);
    if (matrix->lowest_one_lookup[my_lowest_one] == -1 && my_lowest_one >= row_begin) {
        matrix->lowest_one_lookup[my_lowest_one] = my_col_id;
        matrix->column_type[my_col_id] = LOCAL_NEGATIVE;
        matrix->column_type[my_lowest_one] = LOCAL_POSITIVE;
        bits64_clear_column(cols, my_lowest_one);
    }
}

__device__ indx bits64_get_max_index(bits64_cols * cols, indx idx) {
    if (cols[idx].data_length == 0) {
        return -1;
    }
    auto value_length = cols[idx].data_length;
    auto result = least_significant_bit_pos(cols[idx].value[value_length - 1]) +
            BLOCK_BITS * cols[idx].pos[value_length - 1];
    return result;
}

__device__ void bits64_clear_column(bits64_cols * cols, indx col_id) {
    auto aim_col = &cols[col_id];
    aim_col->data_length = 0;
}
