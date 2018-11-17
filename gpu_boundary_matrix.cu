#include <iostream>
#include <cstdlib>
#include <device_launch_parameters.h>
#include "gpu_boundary_matrix.h"
#define BLOCK_BITS 4

typedef long indx;
typedef short dimension;

__global__ void test_length(size_t * column_length, int column_num) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;
}

__global__ void allocate_all_columns(indx ** tmp_gpu_columns, size_t * column_length, int column_num,
        ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    auto length = column_length[thread_id];
    tmp_gpu_columns[thread_id] = (indx *) allocator.malloc(sizeof(indx) * length);
}

__global__ void transform_all_columns(indx ** tmp_gpu_columns, size_t * column_length, column *matrix, int column_num, bool* is_reduced, indx* lowest_one_lookup,
        ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    auto src_length = column_length[thread_id];
    auto src_data = tmp_gpu_columns[thread_id];
    auto col = &matrix[thread_id];
    col->data_length = 0;
    indx last_pos = -1;
    for (size_t i = 0; i < src_length; i++) {
        indx current_pos = src_data[i] / BLOCK_BITS;
        if (last_pos != current_pos) {
            col->data_length++;
            last_pos = current_pos;
        }
    }

    col->pos = (indx *) allocator.malloc(sizeof(indx) * col->data_length);
    col->value = (unsigned long long *) allocator.malloc(sizeof(unsigned long long) * col->data_length);

    last_pos = -1;
    unsigned long long last_value = 0;
    int cur_block_id = 0;

    for (int i = 0; i < src_length; i++)
    {
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
	    if(i == (src_length-1))
	    {
		    col->pos[cur_block_id] = last_pos;
		    col->value[cur_block_id] = last_value;
	    }
    }
}

gpu_boundary_matrix::gpu_boundary_matrix(phat::boundary_matrix <phat::vector_vector> *src_matrix,
                                         indx chunks_num, ScatterAllocator::AllocatorHandle allocator) {
    auto cols_num = (size_t) src_matrix->get_num_cols();
    auto h_matrix = new column[cols_num];
    auto h_chunk_offset = new indx[cols_num];
    auto h_column_type = new short[cols_num];
    auto h_column_length = new size_t[cols_num];
    auto h_dims = new dimension[cols_num];
    auto h_lowest_one_lookup = new indx[cols_num];
    auto h_chunks_start_offset = new indx[chunks_num + 1];
    auto h_is_reduced = new bool[cols_num];

    auto chunk_size = (size_t) CUDA_THREADS_EACH_BLOCK(cols_num);

    for (phat::index i = 0, chunk_pos = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        h_column_length[i] = col.size();
        h_column_type[i] = GLOBAL;
        h_lowest_one_lookup[i] = -1;
        h_is_reduced[i] = false;
        h_dims[i] = src_matrix->get_dim(i);
        if (i % chunk_size == 0) {
            h_chunks_start_offset[chunk_pos] = i;
            chunk_pos++;
        }
    }

    h_chunk_offset[chunks_num] = (indx)(cols_num);

    gpuErrchk(cudaMalloc((void **) &matrix, sizeof(column) * cols_num));
    gpuErrchk(cudaMalloc((void **) &chunk_offset, sizeof(indx) * (chunks_num + 1)));
    gpuErrchk(cudaMalloc((void **) &chunk_columns_finished, sizeof(unsigned long long) * chunks_num));
    gpuErrchk(cudaMalloc((void **) &column_type, sizeof(short) * cols_num));
    gpuErrchk(cudaMalloc((void **) &dims, sizeof(dimension) * cols_num));
    gpuErrchk(cudaMalloc((void **) &column_length, sizeof(size_t) * cols_num));
    gpuErrchk(cudaMalloc((void **) &lowest_one_lookup, sizeof(indx) * cols_num));
    gpuErrchk(cudaMalloc((void **) &is_active, sizeof(bool) * cols_num));
    gpuErrchk(cudaMalloc((void **) &is_ready_for_mark, sizeof(bool) * cols_num));
    gpuErrchk(cudaMalloc((void **) &is_reduced, sizeof(bool) * cols_num));

    size_t * d_column_length;
    gpuErrchk(cudaMalloc((void **) &d_column_length, sizeof(size_t) * cols_num));
    gpuErrchk(cudaMemcpy(d_column_length, h_column_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dims, h_dims, sizeof(dimension) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(column_length, h_column_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(lowest_one_lookup, h_lowest_one_lookup, sizeof(indx) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk_offset, h_chunk_offset, sizeof(indx) * (chunks_num + 1), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(column_type, h_column_type, sizeof(short) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(is_reduced, h_is_reduced, sizeof(bool) * cols_num, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(matrix, h_matrix, sizeof(column) * cols_num, cudaMemcpyHostToDevice));

    indx ** tmp_gpu_columns, ** h_tmp_gpu_columns;
    h_tmp_gpu_columns = new indx * [cols_num];
    gpuErrchk(cudaMalloc((void **) &tmp_gpu_columns, sizeof(indx *) * cols_num));
    allocate_all_columns <<< CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >>> (tmp_gpu_columns,
            d_column_length, cols_num, allocator);
    cudaMemcpy(h_tmp_gpu_columns, tmp_gpu_columns, sizeof(indx *) * cols_num, cudaMemcpyDeviceToHost);

    for (phat::index i = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        auto col_data_ptr = &col[0];

        gpuErrchk(cudaMemcpy(h_tmp_gpu_columns[i], col_data_ptr, sizeof(indx) * col.size(),
                             cudaMemcpyHostToDevice));
    }

    transform_all_columns <<< CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >>> (tmp_gpu_columns,
            column_length, matrix, cols_num, is_reduced, lowest_one_lookup, allocator);

    gpuErrchk(cudaFree(tmp_gpu_columns));
    delete[] h_matrix;
    delete[] h_chunk_offset;
    delete[] h_column_type;
    delete[] h_column_length;
    delete[] h_dims;
    delete[] h_lowest_one_lookup;
    delete[] h_chunks_start_offset;
    delete[] h_tmp_gpu_columns;
}

__global__ void transform_unpacked_data(column *matrix, unpacked_matrix* u_matrix, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    //int block_id = blockDim.x;

    int count_length = 0;
    for(int i = 0; i<matrix[thread_id].data_length; i++)
    {
       // unsigned long long temp_value = matrix[thread_id].value[i];
        for(int j = 0; j < BLOCK_BITS; j++)
        {
            unsigned long long temp_value = 1 << j;
            if(matrix[thread_id].value[i] & temp_value)
            {
                count_length++;
            }
        }
    }

    u_matrix->column[thread_id].data = (indx*) allocator.malloc(sizeof(indx) * count_length);
    u_matrix->column[thread_id].data_length = count_length;
    count_length = 0;
    for(int i = 0; i < matrix[thread_id].data_length; i++)
    {
        for(int j = 0; j < BLOCK_BITS; j++)
        {
            unsigned long long temp_value = 1 << j;
            if(matrix[thread_id].value[i] & temp_value)
            {
                u_matrix->column[thread_id].data[count_length] = matrix[thread_id].pos[i] * BLOCK_BITS + j;
                count_length++;
            }
        }
    }
}

/*__host__ void free_cuda_memory(columns *matrix, dimension* dims, indx* lowest_one_lookup, indx* chunks_start_offset, indx* chunk_columns_finished, short* column_type, bool* is_active, bool* is_ready_for_mark) {
    gpuErrchk(cudaFree(dims));
    gpuErrchk(cudaFree(matrix));
    gpuErrchk(cudaFree(lowest_one_lookup));
    gpuErrchk(cudaFree(chunks_start_offset));
    gpuErrchk(cudaFree(chunk_columns_finished));
    gpuErrchk(cudaFree(column_type));
    gpuErrchk(cudaFree(is_active));
    gpuErrchk(cudaFree(is_ready_for_mark));
}
*/
__host__ void transfor_data_backto_cpu(phat::boundary_matrix<phat::vector_vector> *src_matrix,unpacked_matrix *d_matrix)
{
    indx cols_num = src_matrix->get_num_cols();
    auto h_all_columns = new unpacked_column[cols_num];

    gpuErrchk(cudaMemcpy(h_all_columns, d_matrix->column, sizeof(column) * cols_num, cudaMemcpyDeviceToHost));
    for (int i = 0; i < cols_num; i++) {
    auto h_single_column = h_all_columns[i];

    phat::column tmp_vector(h_single_column.data_length);
    gpuErrchk(cudaMemcpy(&tmp_vector[0], h_single_column.data,
                sizeof(indx) * h_single_column.data_length, cudaMemcpyDeviceToHost));
    src_matrix->set_col(i, tmp_vector);
    }

    delete [] h_all_columns;
}

__device__ dimension get_dim(dimension* dims, int col_id) {
    return dims[col_id];
}

__device__ bool is_empty(column* matrix, int col_id) {
    return matrix[col_id].data_length == 0 || matrix[col_id].value[0] == 0;
}

__device__ indx get_max_index(column* matrix, int col_id) {
    if (matrix[col_id].data_length == 0 || matrix[col_id].value[0] == 0)
        return -1;
    else {
        unsigned long long t = matrix[col_id].value[matrix[col_id].data_length - 1];
        int cnt = 0;
        while (t >> 1 != 0) {
            t = t >> 1;
            cnt++;
        }
        return (matrix[col_id].pos[matrix[col_id].data_length - 1] * BLOCK_BITS + cnt);
    }
}

__device__ void clear_column(column* matrix, int col_id) {
    matrix[col_id].data_length = 0;
}

__device__ void remove_max_index(column* matrix, int col) {
    if(matrix[col].data_length == 0)
        return;
    unsigned long long t = matrix[col].value[matrix[col].data_length - 1];
    int cnt = 1;
    while (t >> 1 != 0) {
        t = t >> 1;
        cnt++;
    }
    int tx = (1 << (cnt-1));
    matrix[col].value[matrix[col].data_length - 1]  ^= tx;
    if (matrix[col].value[matrix[col].data_length - 1] == 0)
        matrix[col].data_length--;
}

__device__ void check_lowest_one_locally(column* matrix, short* column_type, unsigned long long* chunk_columns_finished, dimension* dims,indx my_col_id, indx chunk_start, indx row_begin, indx num_cols, dimension cur_dim, indx *target_col, bool *ive_added) {
    if (cur_dim != get_dim(dims, my_col_id) || column_type[my_col_id] != GLOBAL) {
        if (!*ive_added) {
            atomicAdd((unsigned long long *) &chunk_columns_finished[0], (unsigned long long) 1);
            *ive_added = true;
        }
        return;
    }

    indx my_lowest_one = get_max_index(matrix, my_col_id);
    if (my_lowest_one >= row_begin) {
        for (indx col_id = chunk_start; col_id < num_cols; col_id++) {
            indx this_lowest_one = get_max_index(matrix, col_id);
            if (this_lowest_one == my_lowest_one) {
                *target_col = col_id;
                if (*ive_added) {
                    atomicAdd((unsigned long long *) &chunk_columns_finished[0], (unsigned long long) -1);
                }
                return;
            }
        }
        if (!*ive_added) {
            atomicAdd((unsigned long long *) &chunk_columns_finished[0], (unsigned long long) 1);
            *ive_added = true;
        }
    }
}

__device__ void add_two_columns(column* matrix, int target, int source, ScatterAllocator::AllocatorHandle allocator) {
    if(target == -1)
        return;
    size_t tgt_id = 0; size_t src_id = 0; size_t temp_id = 0;
    int msize = round_up_to_2s(matrix[target].data_length + matrix[source].data_length);
    auto new_pos = (indx *) allocator.malloc(sizeof(indx) * msize);
    auto new_value = (unsigned long long *) allocator.malloc(sizeof(unsigned long long) * msize);
    while (tgt_id < matrix[target].data_length && src_id < matrix[source].data_length) {
        if (matrix[target].pos[tgt_id] == matrix[source].pos[src_id]) {
            if (matrix[target].value[tgt_id] ^ matrix[source].value[src_id] != 0) {
                new_pos[temp_id] = matrix[target].pos[tgt_id];
                new_value[temp_id] = matrix[target].value[tgt_id] ^ matrix[source].value[src_id];
                temp_id++;
            }
            tgt_id++;
            src_id++;
        } else if (matrix[target].pos[tgt_id] < matrix[source].pos[src_id]) {
            if (matrix[target].pos[tgt_id] == matrix[source].pos[src_id + 1])
                tgt_id++;
            else {
                new_value[temp_id] = matrix[target].value[tgt_id];
                new_pos[temp_id] = matrix[target].pos[tgt_id];
                tgt_id++;
                temp_id++;
            }
        } else {
            if (matrix[target].pos[tgt_id + 1] == matrix[source].pos[src_id])
                src_id++;
            else {
                new_value[temp_id] = matrix[source].value[src_id];
                new_pos[temp_id] = matrix[source].pos[src_id];
                src_id++;
                temp_id++;
            }
        }
    }

    if (src_id < matrix[source].data_length) {
        memcpy(&new_value[temp_id], &matrix[source].value[src_id],
               sizeof(indx) * (matrix[source].data_length - src_id));
        memcpy(&new_pos[temp_id], &matrix[source].pos[temp_id],
               sizeof(indx) * (matrix[source].data_length - src_id));
        temp_id += matrix[source].data_length - src_id;
    } else if (tgt_id < matrix[target].data_length) {
        memcpy(&new_value[temp_id], &matrix[target].value[tgt_id],
               sizeof(indx) * (matrix[target].data_length - tgt_id));
        memcpy(&new_pos[temp_id], &matrix[target].pos[tgt_id],
               sizeof(indx) * (matrix[target].data_length - tgt_id));
        temp_id += matrix[target].data_length - tgt_id;
    }
    matrix[target].pos = new_pos;
    matrix[target].value = new_value;
    matrix[target].data_length = temp_id;
}

__device__ void mark_and_clean(column* matrix, indx* lowest_one_lookup, short* column_type, dimension* dims, indx my_col_id, indx row_begin, dimension cur_dim) {
    if (cur_dim != get_dim(dims, my_col_id) || column_type[my_col_id] != GLOBAL) {
        return;
    }

    indx my_lowest_one = get_max_index(matrix, my_col_id);
    if (lowest_one_lookup[my_lowest_one] == -1 && my_lowest_one >= row_begin) {
      	lowest_one_lookup[my_lowest_one] = my_col_id;
        column_type[my_col_id] = LOCAL_NEGATIVE;
        column_type[my_lowest_one] = LOCAL_POSITIVE;
        clear_column(matrix, my_lowest_one);
    }
}

