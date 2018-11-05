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
        gpu_matrix.check_lowest_one(thread_id, block_id, chunk_start, chunk_start, cur_dim, &target, &is_added);
        gpu_matrix.add_two_columns(target, thread_id, allocator);
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

__global__ void mark_active_column(gpu_boundary_matrix &gpu_matrix)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;

    index chunk_start = gpu_matrix.chunks_start_offset[block_id];
    index chunk_end = gpu_matrix.chunks_start_offset[block_id + 1];
    index chunk_size = chunk_end - chunk_start;

    bool im_done = false;
    index cur_row_idx = 0;
    auto col = &gpu_matrix.matrix[thread_id];
    do{
        if (!im_done)
        {
            if (gpu_matrix.is_empty(thread_id) || cur_row_idx == col->data_length
                    || gpu_matrix.column_type[thread_id] == GLOBAL)
            {
                im_done = true;
                gpu_matrix.is_active[thread_id] = gpu_matrix.column_type[thread_id] == GLOBAL;
                gpu_matrix.is_ready_for_mark[thread_id] = true;
                atomicAdd((unsigned long long *) &gpu_matrix.column_finished[block_id], (unsigned long long) 1);
            }
            else
            {
                index cur_row = gpu_matrix.current_row_index(thread_id, cur_row_idx);
                if(gpu_matrix.column_type[cur_row] == GLOBAL)
                {
                    im_done = true;
                    gpu_matrix.is_active[thread_id] = true;
                    gpu_matrix.is_ready_for_mark[thread_id] = true;
                    atomicAdd((unsigned long long *) &gpu_matrix.column_finished[block_id], (unsigned long long) 1);
                }
                else if (gpu_matrix.column_type[cur_row] == LOCAL_POSITIVE)
                {
                    index cur_row_lowest_one = gpu_matrix.lowest_one_lookup[cur_row];
                    if (cur_row_lowest_one == thread_id || cur_row_lowest_one == -1)
                    {
                        cur_row_idx++;
                    }
                    else if (gpu_matrix.is_ready_for_mark[cur_row_lowest_one])
                    {
                        if (gpu_matrix.is_active[cur_row_lowest_one])
                        {
                            im_done = true;
                            gpu_matrix.is_active[thread_id] = true;
                            gpu_matrix.is_ready_for_mark[thread_id] = true;
                            atomicAdd((unsigned long long *) &gpu_matrix.column_finished[block_id],
                                    (unsigned long long) 1);
                        }
                        else
                        {
                            cur_row_idx++;
                        }
                    }
                }
                else
                {
                    cur_row_idx++;
                }
            }
        }
        __syncthreads();
    } while (gpu_matrix.column_finished[block_id] < ((block_id == 0) ?
                (max_dim + 1) * chunk_size : (2 * max_dim + 1) * chunk_size));
}

