#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>

#include <persistence_pairs.h>
#include <algorithms/twist_reduction.h>
#include <algorithms/chunk_reduction.h>
#include <compute_persistence_pairs.h>

#include "gpu_boundary_matrix.h"
#include "gpu_common.h"

__global__ void count_num_dims(dimension* dims, dimension cur_dim, indx* cnt)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if(dims[thread_id] == cur_dim)
        cnt[cur_dim]++;
}

__global__ void gpu_spectral_sequence_reduction(column* matrix, dimension* dims, indx col_in_block, dimension cur_dim, indx block_size, indx phases, indx cur_phase, indx num_cols, indx* lowest_one_lookup, bool* is_reduced, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;

    if(dims[thread_id] != cur_dim || is_reduced[thread_id] == true)
        return;

    if(threadIdx.x != col_in_block)
        return;
    //indx phases = (num_cols % block_size == 0) ? (num_cols / block_size) : (num_cols / block_size + 1);
    //indx row_begin = (((thread_id/block_size - cur_phase) < 0) ? 0 : (thread_id/block_size - cur_phase)) * block_size;
    indx row_begin = (block_id - cur_phase) * block_size;
    indx row_end = ((row_begin + block_size) > num_cols) ? num_cols : (row_begin + block_size);
    indx cur_lowest_one = get_max_index(matrix, thread_id);
    if(cur_lowest_one == -1)
    {
        is_reduced[thread_id] = true;
        return;
    }

    if(!is_reduced[thread_id])
    {
        while(cur_lowest_one != -1 && cur_lowest_one >= row_begin && cur_lowest_one < row_end  && lowest_one_lookup[cur_lowest_one] != -1 )
        {
            add_two_columns(matrix, thread_id, lowest_one_lookup[cur_lowest_one], allocator);
            cur_lowest_one = get_max_index(matrix, thread_id);
        }
        if(cur_lowest_one != -1)
        {
            if(cur_lowest_one >= row_begin && cur_lowest_one < row_end )
            {
                lowest_one_lookup[cur_lowest_one] = thread_id;
                is_reduced[thread_id] = true;
                clear_column(matrix, cur_lowest_one);
                is_reduced[cur_lowest_one] = true;
                __syncthreads();
            }
            else
                {
                    is_reduced[thread_id] = false;
                }
            }
    }
    __syncthreads();
}

__global__ void show(column* matrix, indx* lowest_one_lookup, bool* is_reduced)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < matrix[thread_id].data_length; i++) {
        printf("the column %d : pos: %ld value: %lu\n", thread_id, matrix[thread_id].pos[i],
               matrix[thread_id].value[i]);
    }
    printf("the column %d length is %lu\n", thread_id, matrix[thread_id].data_length);
    if(lowest_one_lookup[thread_id] != -1){
        printf("the persistence pairs are %d and %ld\n", thread_id+1, lowest_one_lookup[thread_id]+1);
    }
    printf("whether column %d is reduced: %d\n", thread_id, is_reduced[thread_id]);
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        return 0;
    }
    return (double) time.tv_sec + (double) time.tv_usec * .000001;
}

int main()
{
    double total_time = 0, time_start = 0, time_end = 0;
    phat::boundary_matrix<phat::vector_vector> boundary_matrix;
    boundary_matrix.set_num_cols(12);
    boundary_matrix.set_dim( 0, 0  );
    boundary_matrix.set_dim( 1, 0  );
    boundary_matrix.set_dim( 2, 1  );
    boundary_matrix.set_dim( 3, 0  );
    boundary_matrix.set_dim( 4, 1  );
    boundary_matrix.set_dim( 5, 1  );
    boundary_matrix.set_dim( 6, 0  );
    boundary_matrix.set_dim( 7, 0  );
    boundary_matrix.set_dim( 8, 0  );
    boundary_matrix.set_dim( 9, 1  );
    boundary_matrix.set_dim( 10, 1  );
    boundary_matrix.set_dim( 11, 2  );
    std::vector< phat::index > temp_col;

    boundary_matrix.set_col( 0, temp_col  );

    boundary_matrix.set_col( 1, temp_col  );

    temp_col.push_back( 0  );
    temp_col.push_back( 1  );
    boundary_matrix.set_col( 2, temp_col  );
    temp_col.clear();

    boundary_matrix.set_col(3, temp_col);
    temp_col.push_back(0);
    temp_col.push_back(3);
    boundary_matrix.set_col( 4, temp_col  );
    temp_col.clear();

    temp_col.push_back( 1  );
    temp_col.push_back( 3  );
    boundary_matrix.set_col( 5, temp_col  );
    temp_col.clear();
    boundary_matrix.set_col(6, temp_col);
    boundary_matrix.set_col(7, temp_col);

    temp_col.push_back( 6  );
    temp_col.push_back( 7  );
    boundary_matrix.set_col( 8, temp_col  );
    temp_col.clear();

    temp_col.push_back( 3  );
    temp_col.push_back( 7  );
    boundary_matrix.set_col( 9, temp_col  );
    temp_col.clear();

    temp_col.push_back(1);
    temp_col.push_back(6);
    boundary_matrix.set_col(10, temp_col);
    temp_col.clear();

    temp_col.push_back(2);
    temp_col.push_back(4);
    temp_col.push_back(5);
    boundary_matrix.set_col(11, temp_col);
    temp_col.clear();
    //boundary_matrix.load_binary("/home/chen/benchmark_data/high_genus_extended.bin");
    ScatterAllocator allocator((size_t)8 * 1024 * 1024 * 1024);
    auto block_num = CUDA_BLOCKS_NUM(boundary_matrix.get_num_cols());
    auto threads_block = CUDA_THREADS_EACH_BLOCK(boundary_matrix.get_num_cols());
    dimension max_dim = boundary_matrix.get_max_dim();
    /*indx h_cnt[max_dim+1];
    for(int i = 0; i<= max_dim+1 ; i++)
        h_cnt[i] = 0;
    //memcpy(h_cnt, 0, sizeof(indx) * (max_dim+1));
    indx* d_cnt;
    gpuErrchk(cudaMalloc((void **) &d_cnt, sizeof(indx) * (max_dim+1)));
    gpuErrchk(cudaMemcpy(d_cnt, h_cnt, sizeof(indx) * (max_dim+1), cudaMemcpyHostToDevice));
    */
    indx num_cols = boundary_matrix.get_num_cols();
    gpu_boundary_matrix g_matrix(&boundary_matrix, block_num, allocator);
    //show<<<block_num, threads_block>>>(g_matrix.matrix, g_matrix.lowest_one_lookup, g_matrix.is_reduced);
    cudaDeviceSynchronize();
    printf("gpu part starts at %lf\n", time_start = get_wall_time());
    //indx block_size = 0;
    /*for(dimension cur_dim = max_dim; cur_dim >= 1; cur_dim--)
    {
        count_num_dims<<<block_num, threads_block>>>(g_matrix.dims, cur_dim, d_cnt);
    }
    gpuErrchk(cudaMemcpy(h_cnt, d_cnt, sizeof(indx) * max_dim, cudaMemcpyDeviceToHost));
    for(dimension cur_dim = max_dim; cur_dim >= 1; cur_dim--)
    {
        if(block_size < h_cnt[cur_dim])
            block_size = h_cnt[cur_dim];
    }
    block_size++;*/
    for(dimension cur_dim = max_dim; cur_dim>=1; cur_dim--)
    {
        for(indx cur_phase = 0; cur_phase < block_num; cur_phase++) {
            for (indx col_in_block = 0; col_in_block < threads_block; col_in_block++) {
                gpu_spectral_sequence_reduction << < block_num, threads_block >> >
                                                                (g_matrix.matrix, g_matrix.dims, col_in_block,
                                                                        cur_dim, threads_block, block_num, cur_phase, num_cols, g_matrix.lowest_one_lookup, g_matrix.is_reduced, allocator);
            }
        }
    }
    show<<<block_num, threads_block>>>(g_matrix.matrix, g_matrix.lowest_one_lookup, g_matrix.is_reduced);
    cudaDeviceSynchronize();
    time_end = get_wall_time();
    printf("gpu parts ends at %lf\n", time_end);
    total_time = time_end - time_start;
    printf("total_time is %lf\n", total_time);
    return 0;
}
