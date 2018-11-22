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

__device__ bool check_done_work(bool* is_done, int thread_id_in_block, int block_begin)
{
    int id = block_begin + thread_id_in_block;
    return is_done[id];
}

__device__ void set_done_work(bool* is_done, int thread_id_in_block, int block_begin)
{
    int id = block_begin + thread_id_in_block;
    is_done[id] = true;
}

__device__ bool gpu_simplify_column(column* matrix, dimension* dims, bool* is_done, int thread_id, int thread_id_in_block, int block_begin, int block_id, dimension max_dim, dimension cur_dim,
        indx cur_phase, indx* lowest_one_lookup, bool* is_reduced, indx block_size, ScatterAllocator::AllocatorHandle allocator)
{
    if(thread_id_in_block == 0 || check_done_work(is_done, thread_id_in_block-1, block_begin))
    {
    indx cur_lowest_one = get_max_index(matrix, thread_id);
    if(dims[thread_id] == cur_dim && cur_lowest_one != -1 && is_reduced[thread_id] == false)
    {
        indx row_begin = (block_id - cur_phase) * block_size;
        indx row_end = row_begin + block_size;

        while (cur_lowest_one != -1 && cur_lowest_one >= row_begin && cur_lowest_one < row_end && lowest_one_lookup[cur_lowest_one] != -1 )
        {
            add_two_columns(matrix, thread_id, lowest_one_lookup[cur_lowest_one], allocator);
            cur_lowest_one = get_max_index(matrix, thread_id);
        }
        if (cur_lowest_one != -1)
        {
            if (cur_lowest_one >= row_begin && cur_lowest_one < row_end)
            {
                lowest_one_lookup[cur_lowest_one] = thread_id;
                is_reduced[thread_id] = true;
                clear_column(matrix, cur_lowest_one);
                is_reduced[cur_lowest_one] = true;
            }
            else{
                is_reduced[thread_id] = false;
            }
        }
        else{
            is_reduced[thread_id] = true;
        }
        if(cur_lowest_one == -1)
            is_reduced[thread_id] = true;
    }
    set_done_work(is_done, thread_id_in_block, block_begin);
    return true;
    }
    return false;
}


__global__ void gpu_spectral_sequence_reduction(column* matrix, unsigned long long *chunk_columns_finished, dimension* dims, bool* is_done, dimension max_dim, dimension cur_dim, indx block_size,
        indx num_cols, int block_num, indx* lowest_one_lookup, bool* is_reduced, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;
    int threadidx = threadIdx.x;
    int column_start = blockDim.x * blockIdx.x;
    //int column_end = column_start + block_size;
    __shared__ unsigned int count;
    bool im_done;

    if(thread_id >= num_cols)
        return;

    for (indx cur_phase = 0; cur_phase<block_num; cur_phase++)
    {
        count = 0;
        im_done = false;
        do{
            if(!im_done)
            {
                bool result = gpu_simplify_column(matrix, dims, is_done, thread_id, threadidx, column_start, block_id, max_dim, cur_dim,
                        cur_phase, lowest_one_lookup, is_reduced, block_size, allocator);
                if(result)
                {
                    atomicAdd(&count, (unsigned int)1);
                    im_done = true;
                }
            }
            __syncthreads();
            printf("column is %d, count is %d block id is %d cur_dim is %d cur_phase is %ld\n", thread_id, count, block_id, cur_dim, cur_phase);
        }while(count < block_size);
        __threadfence();
    }
}

__global__ void show(column* matrix, indx* lowest_one_lookup, bool* is_reduced, dimension * dims)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    /*if(thread_id == 2561538) {
        for (int i = 0; i < matrix[thread_id].data_length; i++) {
            printf("the column %d : pos: %ld value: %lu\n", thread_id, matrix[thread_id].pos[i],
                   matrix[thread_id].value[i]);
        }
        printf("the column %d length is %lu\n", thread_id, matrix[thread_id].data_length);
        if (lowest_one_lookup[thread_id] != -1) {
            printf("the persistence pairs are %d and %ld\n", thread_id + 1, lowest_one_lookup[thread_id] + 1);
        }
        printf("whether column %d is reduced: %d\n", thread_id, is_reduced[thread_id]);
    }*/
    printf("column %d dimension %d\n", thread_id, dims[thread_id]);
}

__host__ void save_ascii(std::vector<indx> &lowest_one_lookup, const char *filename) {
    FILE *file = fopen(filename, "w");
    size_t num = 0;
    for (auto idx : lowest_one_lookup) {
        if (idx != -1) {
            num++;
        }
    }
    fprintf(file, "%ld\n", num);
    for (size_t i = 0; i < lowest_one_lookup.size(); i++) {
        if (lowest_one_lookup[i] != -1) {
            fprintf(file, "%ld %ld\n", i, lowest_one_lookup[i]);
        }
    }
    fclose(file);
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

int main(int argc, char **argv)
{
    phat::boundary_matrix<phat::vector_vector> boundary_matrix;
    /*if (argc == 3) {
        if (strcmp(argv[1], "-b") == 0) {
            int read_successful = boundary_matrix.load_binary(std::string(argv[2]));
        } else if (strcmp(argv[1], "-a") == 0) {
            int read_successful = boundary_matrix.load_ascii(std::string(argv[2]));
        } else {
            printf("Usages: ./matReduct_debug <-b/a> <input_data>\n");
            exit(1);
        }
    } else {
        printf("Usages: ./matReduct_debug <-b/a> <input_data>\n");
        exit(1);
    }*/
    double total_time = 0, time_start = 0, time_end = 0;

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
    ScatterAllocator allocator((size_t)8 * 1024 * 1024 * 1024);
    auto block_num = CUDA_BLOCKS_NUM(boundary_matrix.get_num_cols());
    auto threads_block = CUDA_THREADS_EACH_BLOCK(boundary_matrix.get_num_cols());
    printf("BLOCK NUM:%d, BLOCK_SIZE:%d\n", block_num, threads_block);
    dimension max_dim = boundary_matrix.get_max_dim();
    indx num_cols = boundary_matrix.get_num_cols();
    gpu_boundary_matrix g_matrix(&boundary_matrix, block_num, allocator);
    cudaDeviceSynchronize();
    //show<<<block_num, threads_block>>>(g_matrix.matrix, g_matrix.lowest_one_lookup, g_matrix.is_reduced, g_matrix.dims);
    printf("gpu part starts at %lf\n", time_start = get_wall_time());

    for(dimension cur_dim = max_dim; cur_dim>=1; cur_dim--) {
        gpu_spectral_sequence_reduction << < block_num, threads_block >> >
                                                        (g_matrix.matrix, g_matrix.chunk_columns_finished, g_matrix.dims, g_matrix.is_done, max_dim, cur_dim, threads_block, num_cols,
                                                                block_num, g_matrix.lowest_one_lookup, g_matrix.is_reduced, allocator);
        cudaDeviceSynchronize();
    }

    std::vector<indx> lookup_table(boundary_matrix.get_num_cols());
    gpuErrchk(cudaMemcpy(&lookup_table[0], g_matrix.lowest_one_lookup, sizeof(indx) * num_cols, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    time_end = get_wall_time();
    printf("gpu parts ends at %lf\n", time_end);
    total_time = time_end - time_start;
    printf("total_time is %lf\n", total_time);
    save_ascii(lookup_table, "result.txt");
    return 0;
}
