#include<iostream>
#include<cstdlib>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/function.h>
#include<thrust/list.h>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "mallocMC.hpp"

#define GLOBAL 0
#define LOCAL_NEGATIVE -1
#define LOCAL_POSITIVE 1

typedef long index;
typedef short dimension;

__global__ void allocate_all_columns(gpu_matrix &matrix, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num)
        return;
    auto col = &matrix[thread_id];
    auto length = matrix->data_length[thread_id];
    auto size = round_up_to_2s(length);
    col->data_length = (size_t) length;
    col->size = (size_t) (size > INITIAL_SIZE ? size : INITIAL_SIZE);
    col->data = (indx *) allocator.malloc(sizeof(indx) * size);
}

gpu_boundary_matrix::gpu_boundary_matrix(phat::boundary_matrix<phat::vector_vector>* src_matrix,
            index chunk_num, ScatterAllocator::AllocatorHandle allocator)
    {
        auto cols_num = (size_t)src_matrix->get_num_col();
        auto matrix = new column[cols_num];
        auto h_column_length = new size_t[cols_num];
        auto h_column_type = new short[cols_num];
        auto h_chunk_offset = new index[cols_num];
        auto h_lowest_one_lookup = new index[cols_num];
        auto h_dims = new dimension[cols_num];

        auto chunk_size = (size_t) CUDA_THREADS_EACH_BLOCK(cols_num);

        ScatterAllocator allocator((size_t)8 * 1024 * 1024 * 1024);

        for (phat::index i = 0, chunk_pos = 0; i < cols_num; i++)
        {
            phat::column col;
            src_matrix->get_col(i, col);
            h_column_length[i] = col.size();
            h_dims[i] = src_matrix->get_dim(i);
            h_column_type[i] = GLOBAL;
            h_lowest_one_lookup[i] = -1;
            if (i % chunk_size == 0)
            {
                h_chunks_start_offset[chunk_pos] = i;
                chunk_pos++;
            }
        }

        h_chunk_offset[chunks_num] = (indx) (cols_num);

        gpuErrchk(cudaMalloc((void **) &dims, sizeof(dimension) * cols_num));
        gpuErrchk(cudaMalloc((void **) &column_length, sizeof(indx) * cols_num));
        gpuErrchk(cudaMalloc((void **) &lowest_one_lookup, sizeof(indx) * cols_num));
        gpuErrchk(cudaMalloc((void **) &chunk_offset, sizeof(indx) * (chunks_num + 1)));
        gpuErrchk(cudaMalloc((void **) &chunk_columns_finished, sizeof(indx) * chunks_num));
        gpuErrchk(cudaMalloc((void **) &column_type, sizeof(short) * cols_num));
        gpuErrchk(cudaMalloc((void **) &d_cols_ptr, sizeof(column) * cols_num));
        gpuErrchk(cudaMalloc((void **) &is_active, sizeof(bool) * cols_num));

        gpuErrchk(cudaMemcpy(dims, h_dims_ptr, sizeof(dimension) * cols_num, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(column_length, h_data_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(lowest_one_lookup, h_lowest_one, sizeof(indx) * cols_num, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(chunk_offset, h_chunk_offset, sizeof(indx) * (chunks_num + 1),cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(column_type, h_column_type, sizeof(short) * cols_num, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc())
        gpuErrchk(cudaMemcpy(d_matrix, h_matrix, sizeof(gpu_matrix), cudaMemcpyHostToDevice));

        mx_allocate_all_columns << < CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >> >
            (d_matrix, allocator);

        for (phat::index i = 0; i < cols_num; i++)
        {
            phat::column col;
            src_matrix->get_col(i, col);
            auto col_data_ptr = &col[0];
            column h_single_column;
            gpuErrchk(cudaMemcpy(&h_single_column, &d_cols_ptr[i], sizeof(column), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_single_column.data, col_data_ptr, sizeof(indx) * h_single_column.data_length,
                        cudaMemcpyHostToDevice));
        }

        delete h_matrix;
        delete[] h_data_length;
        delete[] h_chunks_start_offset;
        delete[] h_column_type;
        delete[] h_lowest_one;
        delete[] h_dims_ptr;
    }

__device__ dimension gpu_boundary_matrix::get_max_dim(int col)
{
    return dims[col];
}

__device__ bool gpu_boundary_matrix::is_empty(int col)
{
    return matrix[col].data_length == 0;
}

__device__ index gpu_boundary_matrix::get_max_index(int col)
{
    if(is_empty(col))
        return -1;
    else
    {
        unsigned long long t = matrix[col_id].value[matrix[col_id].data_length - 1];
        int cnt = 0;
        while(t % 2 == 0)
        {
            t = t >> 1;
            cnt++;
        }
        return (matrix[col_id].pos[matrix[col_id].data_length - 1] - 1) * 64 + (64 - cnt);
    }
}

__device__ void gpu_boundary_matrix::clear_column(int col)
{
    matrix[col].data_length = 0;
}

__device__ void gpu_boundary_matrix::remove_max_index(int col)
{
    unsigned long long t = matrix[col].value[matrix[col].data_length - 1];
    int cnt = 1;
    while(t % 2 ==0)
    {
        t =  t >> 1;
        cnt =  cnt << 1;
    }
    matrix[col].value[matrix[col].data_length - 1] = t - cnt;
    if(matrix[col].value[matrix[col].data_length - 1] == 0)
        matrix[col].data_length--;
}

__device__ void gpu_boundary_matrix::check_lowest_one_locally(index my_col_id, index block_id,
            index chunk_start, index row_begin,dimension cur_dim, index *target_col, bool * ive_added)
{
    if (cur_dim != this->get_max_dim( my_col_id) || this->column_type[my_col_id] != GLOBAL)
    {
        if (!*ive_added)
        {
            atomicAdd((unsigned long long *) &this->column_finished[block_id], (unsigned long long) 1);
            *ive_added = true;
        }
        return;
    }

    indx my_lowest_one = this->get_max_index(my_col_id);
    if (my_lowest_one >= row_begin)
    {
        for (indx col_id = chunk_start; col_id < my_col_id; col_id++)
        {
            indx this_lowest_one = this->get_max_index(col_id);
            if (this_lowest_one == my_lowest_one)
            {
                 *target_col = col_id;
                if (*ive_added)
                {
                atomicAdd((unsigned long long *) &this->column_finished[block_id], (unsigned long long) -1);
                }
                return;
            }
        }
        if (!*ive_added)
        {
            atomicAdd((unsigned long long *) &this->column_finished[block_id], (unsigned long long) 1);
            *ive_added = true;
        }
    }
}

__device__ void gpu_boundary_matrix::add_two_columns(int target, int source, ScatterAllocator::AllocatorHandle allocator)
{
    size_t tgt_id = 0, src_id = 0, temp_id = 0;
    int msize = round_up_to_2s(matrix[target].data_length + matrix[source].data_length);
    auto new_pos = (index *) allocator->malloc(sizeof(index) * msize) : new index[msize];
    auto new_value = (index *) allocator->malloc(sizeof(index) * msize) : new index[msize];
    while(tgt_id <= matrix[target].data_length && src_id <= matrix[source].data_length)
    {
        if(matrix[target].pos[tgt_id] == matrix[source].pos[src_id])
        {
            new_value[temp_id] = matrix[target].value[tgt_id] & matrix[source].value[src_id];
            new_pos[temp_id] = matrix[target].pos[tgt_id];
            tgt_id++;
            src_id++;
            temp_id++;
        }
        else if(matrix[target].pos[tgt_id] < matrix[source].pos[src_id])
        {
            new_value[temp_id] = matrix[target].value[tgt_id];
            new_pos[temp_id] = matrix[target].pos[tgt_id];
            tgt_id++;
            temp_id++;
        }
        else
        {
            new_value[temp_id] = matrix[source].value[src_id];
            new_pos[temp_id] = matrix[source].pos[src_id];
            src_id++;
            temp_id++;
        }
    }

    if(src_id < matrix[source].data_length)
    {
        memcpy(&new_value[temp_id], &matrix[source].value[src_id],
                sizeof(index) * (matrix[source].data_length - src_id));
        memcpy(&new_pos[temp_id], &matrix[source].pos[temp_id],
                sizeof(index) * (matrix[source].data_length - src_id));
        temp_id += matrix[source].data_length - src_id;
    }
    else if(tgt_id < matrix[target].data_length)
    {
        memcpy(&new_value[temp_id], &matrix[target].value[tgt_id],
                sizeof(index) * (matrix[target].data_length - tgt_id));
        memcpy(&new_pos[temp_id], &matrix[target].pos[tgt_id],
                sizeof(index) * (matrix[target].data_length - tgt_id));
        temp_id += matrix[target].data_length - tgt_id;
    }
    matrix[target].pos = new_pos;
    matrix[target].value = new_value;
    matrix[target].data_length = temp_id;
}

__device__ void gpu_boundary_matrix::mark_and_clean(index my_col_id, indx row_begin, dimension cur_dim)
{
    if (cur_dim != this->get_max_dim(my_col_id) || this->column_type[my_col_id] != GLOBAL)
    {
        return;
    }

    indx my_lowest_one = get_max_index(my_col_id);
    if (this->lowest_one_lookup[my_lowest_one] == -1 && my_lowest_one >= row_begin)
    {
        this->lowest_one_lookup[my_lowest_one] = my_col_id;
        this->column_type[my_col_id] = LOCAL_NEGATIVE;
        this->column_type[my_lowest_one] = LOCAL_POSITIVE;
        this->clear_column(my_lowest_one);
    }
}

