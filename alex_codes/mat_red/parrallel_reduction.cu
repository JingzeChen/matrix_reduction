#include<iostream>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<cstdio>
#include<cstdlib>
#include<persistence_pairs.h>
#include <algorithms/twist_reduction.h>
#include <algorithms/chunk_reduction.h>

#include "gpu_boundary_matrix.h"
#include "gpu_common.h"

#include<time.h>
#include<sys/time.h>

__global__ void simplified_reduction(column* matrix, short* column_type, dimension max_dim, dimension cur_dim, indx num_cols, dimension* dims, indx* lowest_one_lookup, unsigned long long* chunk_columns_finished, indx matrix_size, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;

    indx target_col = -1;
    bool ive_added = false;

    do {
        check_lowest_one_locally(matrix, column_type, chunk_columns_finished, dims, thread_id, 0, 0, num_cols, cur_dim, &target_col, &ive_added);
        add_two_columns(matrix, target_col, thread_id, allocator);
        target_col = -1;
        __syncthreads();
    } while (chunk_columns_finished[0] < matrix_size * (max_dim - cur_dim + 1));
    mark_and_clean(matrix, lowest_one_lookup, column_type, dims, thread_id, 0, cur_dim);
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


int main ()
{
    double total_time = 0, time_start = 0, time_end = 0;
    phat::boundary_matrix<phat::vector_vector> boundary_matrix;
    boundary_matrix.set_num_cols(12);
    boundary_matrix.set_dim( 0, 0 );
    boundary_matrix.set_dim( 1, 0 );
    boundary_matrix.set_dim( 2, 1 );
    boundary_matrix.set_dim( 3, 0 );
    boundary_matrix.set_dim( 4, 1 );
    boundary_matrix.set_dim( 5, 1 );
    boundary_matrix.set_dim( 6, 0 );
    boundary_matrix.set_dim( 7, 0 );
    boundary_matrix.set_dim( 8, 0 );
    boundary_matrix.set_dim( 9, 1 );
    boundary_matrix.set_dim( 10, 1 );
    boundary_matrix.set_dim( 11, 2 );
    std::vector< phat::index > temp_col;

    boundary_matrix.set_col( 0, temp_col );

    boundary_matrix.set_col( 1, temp_col );

    temp_col.push_back( 0 );
    temp_col.push_back( 1 );
    boundary_matrix.set_col( 2, temp_col );
    temp_col.clear();

    boundary_matrix.set_col(3, temp_col);
    temp_col.push_back(0);
    temp_col.push_back(3);
    boundary_matrix.set_col( 4, temp_col );
    temp_col.clear();

    temp_col.push_back( 1 );
    temp_col.push_back( 3 );
    boundary_matrix.set_col( 5, temp_col );
    temp_col.clear();
    boundary_matrix.set_col(6, temp_col);
    boundary_matrix.set_col(7, temp_col);

    temp_col.push_back( 6 );
    temp_col.push_back( 7 );
    boundary_matrix.set_col( 8, temp_col );
    temp_col.clear();

    temp_col.push_back( 3 );
    temp_col.push_back( 7 );
    boundary_matrix.set_col( 9, temp_col );
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
    //c_boundary_matrix.load_binary("/home/chen/benchmark_data/high_genus_extended.bin");
    ScatterAllocator allocator((size_t)8 * 1024 * 1024 * 1024);
    auto block_num = CUDA_BLOCKS_NUM(boundary_matrix.get_num_cols());
    auto threads_block = CUDA_THREADS_EACH_BLOCK(boundary_matrix.get_num_cols());
    dimension max_dim = boundary_matrix.get_max_dim();
    indx matrix_size = boundary_matrix.get_num_cols();
    unpacked_matrix* u_matrix;
    //u_matrix->column_num = c_boundary_matrix.get_num_cols();
    gpu_boundary_matrix g_matrix(&boundary_matrix, block_num, allocator);
    cudaDeviceSynchronize();
    printf("gpu part starts at %lf\n", time_start = get_wall_time());
    for(dimension cur_dim = max_dim; cur_dim>=1; cur_dim--)
    {
        simplified_reduction<<<block_num, threads_block>>>(g_matrix.matrix, g_matrix.column_type, max_dim, cur_dim, matrix_size,
                g_matrix.dims, g_matrix.lowest_one_lookup, g_matrix.chunk_columns_finished, matrix_size, allocator);
    }
    return 0;
    transform_unpacked_data<<<block_num, threads_block>>>(g_matrix.matrix, u_matrix, allocator);
    transfor_data_backto_cpu(&boundary_matrix, u_matrix);
    cudaDeviceSynchronize();
    printf("gpu part ends at %lf\n", time_end = get_wall_time());
    total_time = time_end - time_start;

    printf("total time of processing is %lf\n", total_time);

    return 0;
}
