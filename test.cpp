//
// Created by chen on 11/8/18.
//

#include <time.h>
#include <sys/time.h>

#include <algorithms/twist_reduction.h>
#include <algorithms/chunk_reduction.h>
#include <compute_persistence_pairs.h>
#include <persistence_pairs.h>

#include "gpu_common.h"
#include "gpu_boundary_matrix.h"

__global__ void test_add(gpu_boundary_matrix * matrix, int column_num, ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    if (thread_id == 2) {
        add_two_columns(matrix->matrix, thread_id, thread_id + 1, &allocator);
    }
    __syncthreads();
    printf("I'm thread %d Lengh is %lu\n", thread_id, matrix->matrix[thread_id].data_length);
}

int main()
{
    phat::boundary_matrix<phat::vector_vector> boundary_matrix;
    ScatterAllocator allocator((size_t)8U * 1024U * 1024U * 1024U);

    /*boundary_matrix.set_num_cols( 7 );

    // set the dimension of the cell that a column represents:
    boundary_matrix.set_dim( 0, 0 );
    boundary_matrix.set_dim( 1, 0 );
    boundary_matrix.set_dim( 2, 1 );
    boundary_matrix.set_dim( 3, 0 );
    boundary_matrix.set_dim( 4, 1 );
    boundary_matrix.set_dim( 5, 1 );
    boundary_matrix.set_dim( 6, 2 );

    // set the respective columns -- the columns entries have to be sorted
    std::vector< phat::index > temp_col;

    boundary_matrix.set_col( 0, temp_col );

    boundary_matrix.set_col( 1, temp_col );

    temp_col.push_back( 0 );
    temp_col.push_back( 1 );
    boundary_matrix.set_col( 2, temp_col );
    temp_col.clear();

    boundary_matrix.set_col( 3, temp_col );

    temp_col.push_back( 1 );
    temp_col.push_back( 3 );
    boundary_matrix.set_col( 4, temp_col );
    temp_col.clear();

    temp_col.push_back( 0 );
    temp_col.push_back( 3 );
    boundary_matrix.set_col( 5, temp_col );
    temp_col.clear();

    temp_col.push_back( 2 );
    temp_col.push_back( 4 );
    temp_col.push_back( 5 );
    boundary_matrix.set_col( 6, temp_col );
    temp_col.clear();*/

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

    temp_col.push_back( 0 );
    temp_col.push_back( 1 );
    boundary_matrix.set_col(3, temp_col);
    temp_col.clear();

//    temp_col.push_back(0);
//    temp_col.push_back(3);
    boundary_matrix.set_col( 4, temp_col );
    temp_col.clear();

//    temp_col.push_back( 1 );
//    temp_col.push_back( 3 );
    boundary_matrix.set_col( 5, temp_col );
    temp_col.clear();
    boundary_matrix.set_col(6, temp_col);
    boundary_matrix.set_col(7, temp_col);

//    temp_col.push_back( 6 );
//    temp_col.push_back( 7 );
    boundary_matrix.set_col( 8, temp_col );
    temp_col.clear();

//    temp_col.push_back( 3 );
//    temp_col.push_back( 7 );
    boundary_matrix.set_col( 9, temp_col );
    temp_col.clear();

//    temp_col.push_back(1);
//    temp_col.push_back(6);
    boundary_matrix.set_col(10, temp_col);
    temp_col.clear();

//    temp_col.push_back(2);
//    temp_col.push_back(4);
//    temp_col.push_back(5);
    boundary_matrix.set_col(11, temp_col);
    temp_col.clear();

    auto block_num = CUDA_BLOCKS_NUM(boundary_matrix.get_num_cols());
    auto threads_block = CUDA_THREADS_EACH_BLOCK(boundary_matrix.get_num_cols());

    auto g_matrix = gpu_boundary_matrix::create_gpu_boundary_matrix(&boundary_matrix, block_num, allocator);


    //test_standard_reduction_algorithm<<<1, boundary_matrix.get_num_cols()>>>(g_matrix.matrix, boundary_matrix.get_num_cols(),allocator);
    //unpacking<<<1, boundary_matrix.get_num_cols()>>>(g_matrix.matrix);
    //test_show_matrix(g_matrix.matrix, &boundary_matrix);
    test_add<<<block_num, threads_block>>>(g_matrix, boundary_matrix.get_num_cols(), allocator);
    //test_construction<<<block_num, threads_block>>>(g_matrix.matrix);
    //test_get_max_index<<<block_num, threads_block>>>(g_matrix.matrix);
    //test_dims<<<block_num, threads_block>>>(g_matrix.dims);
    //test_remove_max_index<<<block_num, threads_block>>>(g_matrix.matrix);

    return 0;
}
