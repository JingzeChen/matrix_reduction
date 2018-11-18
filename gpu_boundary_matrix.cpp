#include <iostream>
#include <cstdlib>
#include <device_launch_parameters.h>
#include "gpu_boundary_matrix.h"
#define BLOCK_BITS 4

typedef long indx;
typedef short dimension;

__global__ void allocate_all_columns(indx ** tmp_gpu_columns, size_t * column_length, int column_num,
        ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    auto length = column_length[thread_id];
    tmp_gpu_columns[thread_id] = (indx *) allocator.malloc(sizeof(indx) * length);
}

__global__ void transform_all_columns(indx ** tmp_gpu_columns, size_t * column_length, column *matrix, int column_num,
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

gpu_boundary_matrix * gpu_boundary_matrix::create_gpu_boundary_matrix(phat::boundary_matrix<phat::vector_vector> *src_matrix,
        indx chunks_num, ScatterAllocator::AllocatorHandle allocator) {
    auto h_boundary_matrix = new gpu_boundary_matrix();
    gpu_boundary_matrix * d_boundary_matrix;

    auto cols_num = (size_t) src_matrix->get_num_cols();
    auto h_matrix = new column[cols_num];
    auto h_chunk_offset = new indx[cols_num];
    auto h_column_type = new short[cols_num];
    auto h_column_length = new size_t[cols_num];
    auto h_dims = new dimension[cols_num];
    auto h_lowest_one_lookup = new indx[cols_num];
    auto h_chunks_start_offset = new indx[chunks_num + 1];

    auto chunk_size = (size_t) CUDA_THREADS_EACH_BLOCK(cols_num);

    for (phat::index i = 0, chunk_pos = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        h_column_length[i] = col.size();
        h_column_type[i] = GLOBAL;
        h_lowest_one_lookup[i] = -1;
        h_dims[i] = src_matrix->get_dim(i);
        if (i % chunk_size == 0) {
            h_chunks_start_offset[chunk_pos] = i;
            chunk_pos++;
        }
    }

    h_chunk_offset[chunks_num] = (indx)(cols_num);

    gpuErrchk(cudaMalloc((void **) &d_boundary_matrix, sizeof(gpu_boundary_matrix)));

    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->matrix, sizeof(column) * cols_num));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->chunk_offset, sizeof(indx) * (chunks_num + 1)));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->chunk_columns_finished, sizeof(indx) * chunks_num));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->column_type, sizeof(short) * cols_num));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->dims, sizeof(dimension) * cols_num));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->column_length, sizeof(size_t) * cols_num));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->lowest_one_lookup, sizeof(indx) * cols_num));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->is_active, sizeof(bool) * cols_num));
    gpuErrchk(cudaMalloc((void **) &h_boundary_matrix->is_ready_for_mark, sizeof(bool) * cols_num));

    size_t * d_column_length;
    gpuErrchk(cudaMalloc((void **) &d_column_length, sizeof(size_t) * cols_num));
    gpuErrchk(cudaMemcpy(d_column_length, h_column_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(h_boundary_matrix->dims, h_dims, sizeof(dimension) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_boundary_matrix->column_length, h_column_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_boundary_matrix->lowest_one_lookup, h_lowest_one_lookup, sizeof(indx) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_boundary_matrix->chunk_offset, h_chunk_offset, sizeof(indx) * (chunks_num + 1), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_boundary_matrix->column_type, h_column_type, sizeof(short) * cols_num, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(h_boundary_matrix->matrix, h_matrix, sizeof(column) * cols_num, cudaMemcpyHostToDevice));

    indx ** tmp_gpu_columns, ** h_tmp_gpu_columns;
    h_tmp_gpu_columns = new indx * [cols_num];
    gpuErrchk(cudaMalloc((void **) &tmp_gpu_columns, sizeof(indx *) * cols_num));
    allocate_all_columns <<< CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >>> (tmp_gpu_columns,
            d_column_length, cols_num, allocator);
    gpuErrchk(cudaMemcpy(h_tmp_gpu_columns, tmp_gpu_columns, sizeof(indx *) * cols_num, cudaMemcpyDeviceToHost));

    for (phat::index i = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        auto col_data_ptr = &col[0];

        gpuErrchk(cudaMemcpy(h_tmp_gpu_columns[i], col_data_ptr, sizeof(indx) * col.size(),
                             cudaMemcpyHostToDevice));
    }

    transform_all_columns <<< CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >>> (tmp_gpu_columns,
            h_boundary_matrix->column_length, h_boundary_matrix->matrix, cols_num, allocator);

    gpuErrchk(cudaMemcpy(d_boundary_matrix, h_boundary_matrix, sizeof(gpu_boundary_matrix), cudaMemcpyHostToDevice));


    gpuErrchk(cudaFree(tmp_gpu_columns));
    delete h_matrix;
    delete h_chunk_offset;
    delete h_column_type;
    delete h_column_length;
    delete h_dims;
    delete h_lowest_one_lookup;
    delete[] h_chunks_start_offset;
    delete[] h_tmp_gpu_columns;

    return d_boundary_matrix;
}

/*__host__ __device__
gpu_boundary_matrix::~gpu_boundary_matrix(phat::boundary_matrix<phat::vector_vector> *src_matrix,
                                          column *d_matrix, , std::vector<indx> &lowest_one_lookup, std::vector<short> &column_type)
{
    indx cols_num = src_matrix->get_num_cols();
    auto h_all_columns = new column[cols_num];

    gpuErrchk(cudaMemcpy(h_matrix, d_matrix, sizeof(column), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_all_columns, h_matrix->data, sizeof(column) * cols_num, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&lowest_one_lookup[0], h_matrix->lowest_one_lookup, sizeof(indx) * cols_num,
                     cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&column_type[0], h_matrix->column_type, sizeof(short) * cols_num, cudaMemcpyDeviceToHost));

    for (int i = 0; i < cols_num; i++) {
        auto h_single_column = h_all_columns[i];
        if (h_single_column.data_length == 0 || column_type[i] != GLOBAL) {
            continue;
        }

    phat::column tmp_vector(h_single_column.data_length);
    gpuErrchk(cudaMemcpy(&tmp_vector[0], h_single_column.data, sizeof(indx) * h_single_column.data_length, cudaMemcpyDeviceToHost));
    src_matrix->set_col(i, tmp_vector);
    }
}*/

__device__ dimension get_dim(dimension* dims, int col_id) {
    return dims[col_id];
}

__device__ bool is_empty(column* matrix, int col_id) {
    return matrix[col_id].data_length == 0;
}

__device__ indx get_max_index(column* matrix, int col_id) {
    if (matrix[col_id].data_length == 0)
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

__device__ void add_two_columns(column * matrix, int target, int source, ScatterAllocator::AllocatorHandle * allocator) {
    auto src_col = &matrix[source];
    auto tgt_col = &matrix[target];
    size_t tgt_id = 0, src_id = 0, temp_id = 0;
    indx max_size = round_up_to_2s(matrix[target].data_length + matrix[source].data_length);
    max_size = max_size > ADDED_SIZE ? max_size : ADDED_SIZE;

    bool need_new_mem = tgt_col->data_length < max_size;
    auto new_pos = (need_new_mem) ? (indx *) allocator->malloc(sizeof(indx) * max_size) : new indx[max_size];
    auto new_value = (need_new_mem) ? (unsigned long long *) allocator->malloc(sizeof(unsigned long long) * max_size) : new unsigned long long [max_size];

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
        memcpy(&new_pos[temp_id], &matrix[source].pos[temp_id], sizeof(indx) * (matrix[source].data_length - src_id));
        memcpy(&new_value[temp_id], &matrix[source].value[src_id], sizeof(unsigned long long) * (matrix[source].data_length - src_id));
        temp_id += matrix[source].data_length - src_id;
    } else if (tgt_id < matrix[target].data_length) {
        memcpy(&new_pos[temp_id], &matrix[target].pos[tgt_id], sizeof(indx) * (matrix[target].data_length - tgt_id));
        memcpy(&new_value[temp_id], &matrix[target].value[tgt_id], sizeof(unsigned long long) * (matrix[target].data_length - tgt_id));
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
        memcpy(tgt_col->value, new_value, sizeof(unsigned long long) * temp_id);
        delete new_pos;
        delete new_value;
    }

    tgt_col->data_length = temp_id;
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
