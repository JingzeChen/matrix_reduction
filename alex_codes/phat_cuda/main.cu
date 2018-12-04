#include <persistence_pairs.h>
#include <algorithms/twist_reduction.h>
#include <algorithms/chunk_reduction.h>
#include <compute_persistence_pairs.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "gpu_matrix.h"
#include "gpu_chunk_reduction.h"
#include "gpu_bits64_solver.h"

#include <time.h>
#include <sys/time.h>

double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double) time.tv_sec + (double) time.tv_usec * .000001;
}

__global__ void old_reduce_locals(gpu_matrix * matrix, dimension max_dim, ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num) {
        return;
    }
    gpu_chunk_local_reduction(matrix, max_dim, &allocator);
}

__global__ void reduce_locals(bits64_cols * cols, gpu_matrix * matrix, dimension max_dim, ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num) {
        return;
    }
    bits64_chunk_local_reduction(cols, matrix, max_dim, &allocator);
}
__global__ void mark_active(gpu_matrix *matrix, dimension max_dim, ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num) {
        return;
    }
    gpu_mark_active_column(matrix, max_dim);
}
__global__ void simplify_globals(gpu_matrix *matrix, dimension max_dim, ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num) {
        return;
    }
    gpu_column_simplification(matrix, thread_id, &allocator);
}

__global__ void show_chunk_columns(gpu_matrix* matrix)
{
    int thread_id = threadIdx.x + blockIdx.x + blockDim.x;
    int block_id = blockIdx.x;
    printf("matrix->chunk_columns[%d] is %ld\n", block_id, matrix->chunk_finish_col_num[block_id]);
}

__device__ void phase1_bits64_chunk_local_reduction(bits64_cols * cols, gpu_matrix* matrix, dimension cur_dim, dimension max_dim, ScatterAllocator::AllocatorHandle* allocator)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int block_id = blockIdx.x;
    indx chunk_start = matrix->chunks_start_offset[block_id];       // ID of column who starts the chunk.
    indx chunk_end = matrix->chunks_start_offset[block_id + 1];     // ID of column who ends the chunk.
    indx chunk_size = chunk_end - chunk_start;                      // Size of the chunk.
    indx target_col = -1;
    bool ive_added = false;                                                 // To make sure each column will only increase the chunk_finish_col_num once
    indx left_chunk_size = block_id != 0 ?
                       matrix->chunks_start_offset[block_id] - matrix->chunks_start_offset[block_id - 1] : 0;
    unsigned long long count_loop = 0;
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
    dimension d = cur_dim;
    //printf("phase 1:loop count is %lu, cur_dim is %d, column is %d\n", count_loop, d, thread_id);
}

__global__ void phase1_local_reduce(bits64_cols * cols, gpu_matrix * matrix, dimension cur_dim, dimension max_dim, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num) {
        return;
    }
    phase1_bits64_chunk_local_reduction(cols, matrix, cur_dim, max_dim, &allocator);
}

__device__ void phase2_bits64_chunk_local_reduction(bits64_cols * cols, gpu_matrix* matrix, dimension cur_dim, dimension max_dim,ScatterAllocator::AllocatorHandle* allocator)
{
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
    if (block_id != 0) {    // Chunk 0 has no left neighbor who won't run following codes.
//count_loop=0;
        ive_added = false;
        do {
            count_loop++;
            bits64_check_lowest_one_locally(cols, matrix, thread_id, block_id, chunk_start, chunk_start - 1, cur_dim, &target_col, &ive_added);
            bits64_add_two_cols_locally(cols, matrix, thread_id, target_col, allocator);
            target_col = -1;
            __syncthreads();
        } while (matrix->chunk_finish_col_num[block_id] < chunk_size * (max_dim - cur_dim + 1) * 2);
        bits64_mark_and_clean_locally(cols, matrix, thread_id, chunk_start - 1, cur_dim);
        //printf("phase 1:loop count is %lu, cur_dim is %d, column is %d\n", count_loop, cur_dim, thread_id);
    }
}

__global__ void phase2_local_reduce(bits64_cols * cols, gpu_matrix * matrix, dimension cur_dim, dimension max_dim, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num) {
        return;
    }
    phase2_bits64_chunk_local_reduction(cols, matrix, cur_dim, max_dim, &allocator);
}

int main(int argc, char **argv) {
    double total_time = 0, time_start = 0, time_end = 0;

    double local_reduction_time_start = 0, local_reduction_time_end = 0;
    double transform_time_start = 0, transform_time_end = 0;
    double mark_active_time_start = 0, mark_active_time_end = 0;
    double simplify_globals_time_start = 0, simplify_globals_time_end = 0;
    double IO_part_time_start = 0, IO_part_time_end = 0;

    phat::boundary_matrix<phat::vector_vector> boundary_matrix;
    if (argc == 3) {
        if (strcmp(argv[1], "-b") == 0) {
            int read_successful = boundary_matrix.load_binary(std::string(argv[2]));
        } else if (strcmp(argv[1], "-a") == 0) {
            int read_successful = boundary_matrix.load_ascii(std::string(argv[2]));
        } else {
            printf("Usages: ./phat_cuda <-b/a> <input_data>\n");
            exit(1);
        }
    } else {
        printf("Usages: ./phat_cuda <-b/a> <input_data>\n");
        exit(1);
    }

    ScatterAllocator allocator((size_t)8 * 1024 * 1024 * 1024);
    std::vector<indx> lowest_one_lookup((size_t) boundary_matrix.get_num_cols());
    std::vector<short> column_type((size_t) boundary_matrix.get_num_cols());
    std::vector<indx> global_columns;
    auto block_num = CUDA_BLOCKS_NUM(boundary_matrix.get_num_cols());
    auto threads_block = CUDA_THREADS_EACH_BLOCK(boundary_matrix.get_num_cols());
    printf("block num:%d, threads num:%d\n", block_num, threads_block);

    printf("transfering data from CPU to GPU starts at %lf\n", IO_part_time_start = get_wall_time());
    auto gpu_matrix = mx_init_stable_matrix(allocator, &boundary_matrix, CUDA_BLOCKS_NUM(boundary_matrix.get_num_cols()));
    printf("transfering data from CPU to GPU ends at %lf\n", IO_part_time_end = get_wall_time());
    printf("transfering data from CPU to GPU costs %lf\n", IO_part_time_end - IO_part_time_start);

    auto bits64_cols = create_bits64_cols(boundary_matrix.get_num_cols());

    printf("gpu part starts at %lf\n", time_start = get_wall_time());
    printf("transform_to starts at %lf\n", transform_time_start = get_wall_time());
    transform_to <<< block_num, threads_block >>> (bits64_cols, gpu_matrix, boundary_matrix.get_num_cols(), allocator);
    cudaDeviceSynchronize();
    printf("transform_to ends at %lf\n", transform_time_end = get_wall_time());
    printf("transform_to timing is %lf\n", transform_time_end - transform_time_start);

    printf("reduce local matrix starts at %lf\n", local_reduction_time_start = get_wall_time());
    //reduce_locals <<< block_num, threads_block >>> (bits64_cols, gpu_matrix, boundary_matrix.get_max_dim(), allocator);
    for(dimension cur_dim = boundary_matrix.get_max_dim(); cur_dim>=1 ; cur_dim--)
    {
        phase1_local_reduce<<<block_num, threads_block>>>(bits64_cols, gpu_matrix, cur_dim, boundary_matrix.get_max_dim(), allocator);
        //cudaDeviceSynchronize();
        phase2_local_reduce<<<block_num, threads_block>>>(bits64_cols, gpu_matrix, cur_dim, boundary_matrix.get_max_dim(), allocator);
        //cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    printf("reduce local matrix ends at %lf\n", local_reduction_time_end = get_wall_time());
    printf("reudce local matrix costs %lf\n", local_reduction_time_end - local_reduction_time_start);
    //show_chunk_columns<<<block_num, threads_block>>>(gpu_matrix);
    printf("transform_back starts at %lf\n", transform_time_start = get_wall_time());
    transform_back <<< block_num, threads_block >>> (bits64_cols, gpu_matrix, boundary_matrix.get_num_cols(), allocator);
    cudaDeviceSynchronize();
    printf("transform_back ends at %lf\n", transform_time_end = get_wall_time());
    printf("transform_back costs %lf\n", transform_time_end - transform_time_start);

//    old_reduce_locals <<< block_num, threads_block >>> (gpu_matrix, boundary_matrix.get_max_dim(), allocator);
    printf("mark_active starts at %lf\n", mark_active_time_start = get_wall_time());
    mark_active <<< block_num, threads_block >>> (gpu_matrix, boundary_matrix.get_max_dim(), allocator);
    cudaDeviceSynchronize();
    printf("mark_active ends at %lf\n", mark_active_time_end = get_wall_time());
    printf("mark_active costs %lf\n", mark_active_time_end - mark_active_time_start);

    printf("simplify_globals starts at %lf\n", simplify_globals_time_start = get_wall_time());
    simplify_globals <<< block_num, threads_block >>> (gpu_matrix, boundary_matrix.get_max_dim(), allocator);
    cudaDeviceSynchronize();
    printf("simplify_globals ends at %lf\n", simplify_globals_time_end = get_wall_time());
    printf("simplify_globals costs %lf\n", simplify_globals_time_end - simplify_globals_time_start);

    printf("transfering data from GPU to CPU starts at %lf\n", IO_part_time_start = get_wall_time());
    mx_copy_matrix_to_cpu_with_one_step(&boundary_matrix, gpu_matrix, lowest_one_lookup, column_type);
    printf("transfering data from GPU to CPU ends at %lf\n", IO_part_time_end = get_wall_time());
    printf("transfering data from GPU to CPU costs %lf\n", IO_part_time_end - IO_part_time_start);


    double CPU_part_start_time = 0, CPU_part_end_time = 0;
    printf("CPU parts starts at %lf\n", CPU_part_start_time = get_wall_time());
    int count = 0;
    for (indx cur_col_idx = 0; cur_col_idx < boundary_matrix.get_num_cols(); cur_col_idx++) {
        if (column_type[cur_col_idx] == GLOBAL) {
            count++;
            global_columns.push_back(cur_col_idx);
        }
    }
    printf("%d\n", count);

    for (dimension cur_dim = boundary_matrix.get_max_dim(); cur_dim >= 1; cur_dim--) {
        for (auto cur_col : global_columns) {
            if (boundary_matrix.get_dim(cur_col) == cur_dim && column_type[cur_col] == GLOBAL) {
                indx lowest_one = boundary_matrix.get_max_index(cur_col);
                while (lowest_one != -1 && lowest_one_lookup[lowest_one] != -1) {
                    boundary_matrix.add_to(lowest_one_lookup[lowest_one], cur_col);
                    lowest_one = boundary_matrix.get_max_index(cur_col);
                }
                if (lowest_one != -1) {
                    lowest_one_lookup[lowest_one] = cur_col;
                    boundary_matrix.clear(lowest_one);
                }
                boundary_matrix.finalize(cur_col);
            }
        }
    }

    boundary_matrix.sync();

    printf("CPU part ends at %lf\n", CPU_part_end_time = get_wall_time());
    printf("CPU part costs %lf\n", CPU_part_end_time - CPU_part_start_time);
    printf("gpu part ends at %lf\n", time_end = get_wall_time());
    total_time += time_end - time_start;
    printf("gpu costs %lf time without counting IO\n", total_time);
    mx_save_ascii(lowest_one_lookup, "result.txt");
    mx_free_cuda_memory(gpu_matrix);

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
}
