#include "gpu_boundary_matrix.h"

__global__ void local_chunk_reduction(gpu_boundary_matrix &gpu_matrix, dimension max_dim,
        dimension cur_dim, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x*blockIdx.x;
    int block_id = blockIdx.x;
    int chunk_start = gpu_matrix.chunk_offset[block_id];
    int chunk_end = gpu_matrix.chunk_offset[block_id+1];
    int chunk_size = chunk_end - chunk_start;
    int left_chunk_size = block_id == 0 ? 0 : chunk_start_offset[block_id - 1];

    bool is_added = false;
    int target = -1;
    do{
        gpu_matrix.check_lowest_one(matrix, thread_id, block_id, chunk_start, chunk_start, cur_dim, &target, &is_added);
        gpu_matrix.add_two_columns(matrix, target, thread_id, allocator);
        target = -1;
        __synthreads();
    }while(gpu_matrix.column_finished[block_id] < ((block_id == 0) ?
                (max_dim + 1) * chunk_size : (2 * max_dim + 1) * chunk_size));

    gpu_matrix.mark_and_clean(thread_id, chunk_start, cur_dim);

    if(block_id != 0)
    {
        while (gpu_matrix.column_finished[block_id - 1] < (block_id != 1 ?
                    left_chunk_size * ((max_dim - cur_dim) * 2 + 1) :left_chunk_size * (max_dim - cur_dim + 1)))
        {
            __threadfence();    // Wait until the leftmost neighbor has finised its own work
        }

        is_added = false;
        do{
            gpu_matrix.check_lowest_one(thread_id, block_id, chunk_start, chunk_start-1, cur_dim, &target, &is_added);
            gpu_matrix.add_two_columns(target, thread_id, allocator);
            __synthreads();
        }while(gpu_matrix.column_finished[block_id] < chunk_size * (max_dim - cur_dim + 1) * 2);
    }
}


