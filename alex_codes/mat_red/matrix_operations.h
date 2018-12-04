#ifndef _GPU_BOUNDARY_MATRIX_H_
#define _GPU_BOUNDARY_MATRIX_H_


__global__ void local_chunk_reduction(column* matrix, indx* chunk_offset, indx* chunk_columns_finished, dimension max_dim,
        dimension cur_dim, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x*blockIdx.x;
    int block_id = blockIdx.x;
    int chunk_start = chunk_offset[block_id];
    int chunk_end = chunk_offset[block_id+1];
    int chunk_size = chunk_end - chunk_start;
    int left_chunk_size = block_id == 0 ? 0 : chunk_offset[block_id - 1];

    bool is_added = false;
    int target = -1;
    do{
        check_lowest_one_locally(matrix, column_type, chunk_finished_column, dims, thread_id, block_id, chunk_start, chunk_start, cur_dim, &target, &is_added);
        add_two_columns(matrix, target, thread_id, allocator);
        target = -1;
        __synthreads();
    }while(column_finished[block_id] < ((block_id == 0) ?
                (max_dim + 1) * chunk_size : (2 * max_dim + 1) * chunk_size));

    mark_and_clean(matrix, lowest_one_lookup, column_type, dims, thread_id, chunk_start, cur_dim);

    if(block_id != 0)
    {
        while (column_finished[block_id - 1] < (block_id != 1 ?
                    left_chunk_size * ((max_dim - cur_dim) * 2 + 1) :left_chunk_size * (max_dim - cur_dim + 1)))
        {
            __threadfence();    // Wait until the leftmost neighbor has finised its own work
        }

        is_added = false;
        do{
            check_lowest_one_locally(matrix, column_type, chunk_finished_column, dims, thread_id, block_id, chunk_start, chunk_start-1, cur_dim, &target, &is_added);
            add_two_columns(matrix, target, thread_id, allocator);
            __synthreads();
        }while(column_finished[block_id] < chunk_size * (max_dim - cur_dim + 1) * 2);
    }
}

__global__ void mark_active_column(column* matrix, indx* chunk_offset, short* column_type, bool* is_active, bool* is_ready_for_mark, indx *chunk_columns_finished)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = blockIdx.x;

    index chunk_start = chunk_offset[block_id];
    index chunk_end = chunk_offset[block_id + 1];
    index chunk_size = chunk_end - chunk_start;

    bool im_done = false;
    index cur_row_idx = 0;
    auto col = &matrix[thread_id];
    do{
        if (!im_done)
        {
            if (is_empty(thread_id) || cur_row_idx == col->data_length
                    || column_type[thread_id] == GLOBAL)
            {
                im_done = true;
                is_active[thread_id] = column_type[thread_id] == GLOBAL;
                is_ready_for_mark[thread_id] = true;
                atomicAdd((unsigned long long *) &chunk_columns_finished[block_id], (unsigned long long) 1);
            }
            else
            {
                index cur_row = current_row_index(thread_id, cur_row_idx);
                if(gpu_matrix.column_type[cur_row] == GLOBAL)
                {
                    im_done = true;
                    is_active[thread_id] = true;
                    is_ready_for_mark[thread_id] = true;
                    atomicAdd((unsigned long long *) &chunk_column_finished[block_id], (unsigned long long) 1);
                }
                else if (column_type[cur_row] == LOCAL_POSITIVE)
                {
                    index cur_row_lowest_one = lowest_one_lookup[cur_row];
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

#endif