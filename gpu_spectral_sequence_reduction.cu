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
#define TABLE_SIZE 512

__global__ void gpu_spectral_sequence_reduction(column* matrix, unsigned long long *chunk_columns_finished, dimension* dims, indx* leftmost_lookup_lowest_row, dimension max_dim,
        dimension cur_dim, indx block_size, indx num_cols, int block_num, indx* lowest_one_lookup, bool* is_reduced,
        indx cur_phase, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;
    int threadidx = threadIdx.x;
    indx column_start = ((blockDim.x * blockIdx.x - cur_phase * block_size) > 0) ? (blockDim.x * blockIdx.x - cur_phase * block_size) : 0;
    indx column_end = column_start + block_size * (cur_phase+1);
    indx row_begin = ((blockDim.x * blockIdx.x - cur_phase * block_size) > 0) ? (blockDim.x * blockIdx.x - cur_phase * block_size) : 0;
    indx row_end = row_begin + block_size;

    bool ive_added = false;
    int target_col = -1;
    do{
        check_lowest_one_locally(matrix, chunk_columns_finished, dims,is_reduced, leftmost_lookup_lowest_row, thread_id, block_id,
                column_start, column_end, row_begin, row_end, cur_dim, num_cols, &target_col, &ive_added);
        add_two_columns(matrix, thread_id, target_col, allocator);
        //__syncthreads();
        if(target_col != -1){
            update_lookup_lowest_table(matrix, leftmost_lookup_lowest_row, thread_id);
        }
        __threadfence();
        target_col = -1;
        __syncthreads();
    }while(chunk_columns_finished[block_id] < block_size);
    mark_and_clean(matrix, lowest_one_lookup, leftmost_lookup_lowest_row, is_reduced, dims, thread_id, row_begin, row_end, cur_dim);
    __syncthreads();
    if(threadidx == 0)
        chunk_columns_finished[block_id] = 0;
}

__global__ void show(column* matrix, indx* lowest_one_lookup, bool* is_reduced, dimension * dims)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    //if(thread_id == 2561538) {
        for (int i = 0; i < matrix[thread_id].data_length; i++) {
            printf("the column %d : pos: %ld value: %lu\n", thread_id, matrix[thread_id].pos[i],
                   matrix[thread_id].value[i]);
        }
        printf("the column %d length is %lu\n", thread_id, matrix[thread_id].data_length);
        if (lowest_one_lookup[thread_id] != -1) {
            printf("the persistence pairs are %d and %ld\n", thread_id + 1, lowest_one_lookup[thread_id] + 1);
        }
        printf("whether column %d is reduced: %d\n", thread_id, is_reduced[thread_id]);
    //}*/
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
    if (argc == 3) {
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
    }
    double total_time = 0, time_start = 0, time_end = 0;

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
        for (indx cur_phase = 0; cur_phase<block_num; cur_phase++) {
            gpu_spectral_sequence_reduction << < block_num, threads_block >> >
                                                            (g_matrix.matrix, g_matrix.chunk_columns_finished, g_matrix.dims,
                                                                    g_matrix.leftmost_lookup_lowest_row, max_dim, cur_dim, threads_block,
                                                                    num_cols,block_num, g_matrix.lowest_one_lookup, g_matrix.is_reduced, cur_phase, allocator);
            cudaDeviceSynchronize();
            update_table<<<block_num, threads_block>>>(g_matrix.matrix, g_matrix.leftmost_lookup_lowest_row, cur_phase, threads_block);
            cudaDeviceSynchronize();
        }
    }

    std::vector<indx> lookup_table(boundary_matrix.get_num_cols());
    gpuErrchk(cudaMemcpy(&lookup_table[0], g_matrix.lowest_one_lookup, sizeof(indx) * num_cols, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    time_end = get_wall_time();
    printf("gpu parts ends at %lf\n", time_end);
    total_time = time_end - time_start;
    printf("total_time is %lf\n", total_time);
    save_ascii(lookup_table, "result.txt");

    printf("<---------------------------------------------------------------->\n");

    phat::boundary_matrix<phat::vector_vector> new_matrix;
    if (strcmp(argv[1], "-b") == 0) {
        int read_successful = new_matrix.load_binary(std::string(argv[2]));
    } else if (strcmp(argv[1], "-a") == 0) {
        int read_successful = new_matrix.load_ascii(std::string(argv[2]));
    }

    phat::persistence_pairs pairs;
    printf("cpu starts at %lf\n", time_start = get_wall_time());
    phat::compute_persistence_pairs<phat::twist_reduction>(pairs, new_matrix);
    printf("cpu ends at %lf\n", time_end = get_wall_time());
    total_time = time_end - time_start;
    pairs.save_ascii("phat_result.txt");
    printf("cpu costs %lf time without counting IO\n", total_time);

    return 0;
}
