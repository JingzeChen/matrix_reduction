//
// Created by chen on 11/8/18.
//

#include<iostream>
#include<cstdio>
#include <persistence_pairs.h>
#include <algorithms/twist_reduction.h>
#include <algorithms/chunk_reduction.h>
#include <compute_persistence_pairs.h>
#include "gpu_boundary_matrix.h"
#include <time.h>
#include <sys/time.h>

__global__ void test_construction(column* matrix)
{
	int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	int i;

	//if(thread_id == 4) {
        for (i = 0; i < matrix[thread_id].data_length; i++) {
            printf("the column %d : pos: %ld value: %lu\n", thread_id, matrix[thread_id].pos[i],
                   matrix[thread_id].value[i]);
        }
        printf("the column %d length is %lu\n", thread_id, matrix[thread_id].data_length);
    //}
}

__global__ void test_add(column * matrix, indx chunks_num, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    //int block_id = blockDim.x;

    add_two_columns(matrix, 10, 11, allocator);
    __syncthreads();

    if(thread_id == 10)
    {
	int i;
	for(i=0 ;i<matrix[thread_id].data_length; i++)
	{
		printf("after add column 11 to column 10, the result value %d is %lu\n", i, matrix[thread_id].value[i]);
	}
    }
    //printf("I'm thread %d Lengh is %lu\n", thread_id, matrix[thread_id].data_length);
}

__global__ void test_dims(dimension* dims) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    printf("Pre : column %d: dimension: %d\n", thread_id, dims[thread_id]);
    dimension
    max_dim = get_dim(dims, thread_id);
    __syncthreads();
    //if(thread_id == 7){
    printf(" Aft : column %d : dimension is %d\n", thread_id, max_dim);
    //}
}

__global__ void test_is_empty(column* matrix)
{
	int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	bool is_emp = is_empty(matrix, thread_id);
	__syncthreads();


	printf("column %d is ", thread_id);
	printf("%s\n", is_emp ? "true" : "false");
}

__global__ void test_get_max_index(column* matrix)
{
	int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	//printf("column %d length %ld\n", thread_id, matrix[thread_id].data_length);
	indx max_index = get_max_index(matrix, thread_id);
	__syncthreads();

	//if(thread_id == 6)
	    printf("column %d: lowest row index is %ld\n", thread_id, max_index);
}

__global__ void test_remove_max_index(column* matrix)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    remove_max_index(matrix, thread_id);
    __syncthreads();

    if(matrix[thread_id].data_length == 0)
    {
        printf("After removed, currrent column %d is empty\n", thread_id);
    }
    else
    {
        for(int i=0; i < matrix[thread_id].data_length; i++)
        {
            printf("After removed, current column %d : pos: %ld, value: %lu\n", thread_id, matrix[thread_id].pos[i], matrix[thread_id].value[i]);
        }
    }
}

/*__global__ void test_check_lowest_one(column* matrix, short* column_type, indx* chunk_finished_column, dimension* dims, indx my_col_id, indx chunk_start, indx row_begin, dimension cur_dim, indx* target_col, bool* ive_added)*/
/*__device__ void check_lowest(column* matrix, indx my_lowest, int cur_col_id, bool* ive_added)
{
    int i;
    unsigned long long my_rowest_bits = 1 << (my_lowest % BLOCK_BITS);
    indx my_lowest_pos = my_lowest / BLOCK_BITS;


    for(i = 0; i<matrix[cur_col_id].data_length; i++)
    {
        unsigned long long temp = matrix[cur_col_id].value[i];
        while(temp)
        {
            unsigne d
        }
    }
}*/

__global__ void test_standard_reduction_algorithm(column* matrix, int this_col, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(thread_id <= this_col)
        return;
    indx this_lowest_row = get_max_index(matrix, this_col);
    if(this_lowest_row == -1)
        return;
    unsigned long long my_lowest_bits = 1 << (this_lowest_row % BLOCK_BITS);
    int temp = (this_lowest_row == -1)? 0 : this_lowest_row/BLOCK_BITS;
    if(temp >= matrix[thread_id].data_length)
        return;
    unsigned long long t = my_lowest_bits & matrix[thread_id].value[temp];
    if(t != 0)
     {
        add_two_columns(matrix, thread_id, this_col, allocator);
     }
}

__global__ void show(column* matrix)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < matrix[thread_id].data_length; i++) {
        printf("the column %d : pos: %ld value: %lu\n", thread_id, matrix[thread_id].pos[i],
               matrix[thread_id].value[i]);
    }
    __syncthreads();
    printf("the column %d length is %lu\n", thread_id, matrix[thread_id].data_length);
}
/*__global__ void unpacking(column* matrix)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    indx num_cols = matrix[thread_id].data_length;
    indx cur_col = num_cols-1;
    while(!is_empty(matrix, thread_id))
    {
        printf("%d\n", is_empty(matrix, thread_id));
        matrix[thread_id].unpacked_data[cur_col] = get_max_index(matrix, thread_id);
        remove_max_index(matrix, thread_id);
        cur_col--;
    }
    __syncthreads();
}*/

/*__host__ void test_show_matrix(column* matrix, phat::boundary_matrix<phat::vector_vector> *src_matrix)
{
    indx num_cols = src_matrix->get_num_cols();
    for(int i=0;i<num_cols;i++)
    {
        auto h_columns = new indx [matrix[i].data_length];
        gpuErrchk(cudaMemcpy(h_columns, matrix[i].unpacked_data, sizeof(indx) * matrix[i].data_length,
                             cudaMemcpyDeviceToHost));
        phat::column tmp_vector(matrix[i].data_length);
        memcpy(&tmp_vector[0], h_columns, sizeof(indx) * matrix[i].data_length);
        src_matrix->set_col(i, tmp_vector);
        delete[] h_columns;
    }
}*/

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

    auto block_num = CUDA_BLOCKS_NUM(boundary_matrix.get_num_cols());
    auto threads_block = CUDA_THREADS_EACH_BLOCK(boundary_matrix.get_num_cols());

    gpu_boundary_matrix g_matrix(&boundary_matrix, block_num, allocator);

    for(int i=0; i < boundary_matrix.get_num_cols(); i++)
    {
        test_standard_reduction_algorithm << <block_num, threads_block>> >
                                                  (g_matrix.matrix, i, allocator);
    }
    show<<<block_num, threads_block>>>(g_matrix.matrix);
    //unpacking<<<1, boundary_matrix.get_num_cols()>>>(g_matrix.matrix);
    //test_show_matrix(g_matrix.matrix, &boundary_matrix);
    //test_add<<<block_num, threads_block>>>(g_matrix.matrix, block_num, allocator);
    //test_construction<<<block_num, threads_block>>>(g_matrix.matrix);
    //test_get_max_index<<<block_num, threads_block>>>(g_matrix.matrix);
    //test_dims<<<block_num, threads_block>>>(g_matrix.dims);
    //test_remove_max_index<<<block_num, threads_block>>>(g_matrix.matrix);

    return 0;
}
