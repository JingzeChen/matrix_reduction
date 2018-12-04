#include <iostream>
#include <cstdlib>
#include <device_launch_parameters.h>
#include "gpu_boundary_matrix.h"
#define BLOCK_BITS 64
#define ADDED_SIZE 32
#define TABLE_SIZE 512

typedef long indx;
typedef short dimension;

__global__ void test_length(size_t * column_length, int column_num) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;
}

__global__ void allocate_all_columns(indx ** tmp_gpu_columns, size_t * column_length, int column_num,
        ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    auto length = column_length[thread_id];
    tmp_gpu_columns[thread_id] = (indx *) allocator.malloc(sizeof(indx) * length);
    //leftmost_lookup_lowest_row[thread_id] = (indx *)allocator.malloc(sizeof(indx) * TABLE_SIZE);
}

__global__ void transform_all_columns(indx ** tmp_gpu_columns, size_t * column_length, column *matrix, int column_num, ScatterAllocator::AllocatorHandle allocator, dimension* dims) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= column_num)
        return;

    auto src_length = column_length[thread_id];
    auto src_data = tmp_gpu_columns[thread_id];
    auto col = &matrix[thread_id];
    col->data_length = 0;
    indx last_pos = -1;
    for (size_t i = 0; i < src_length; i++) {
        indx current_pos = src_data[i] / BLOCK_BITS;
        if (last_pos != current_pos) {
            col->data_length++;
            last_pos = current_pos;
        }
    }

    col->pos = (indx *) allocator.malloc(sizeof(indx) * col->data_length);
    col->value = (unsigned long long *) allocator.malloc(sizeof(unsigned long long) * col->data_length);

    last_pos = -1;
    unsigned long long last_value = 0;
    int cur_block_id = 0;

    for (int i = 0; i < src_length; i++) {
        indx current_pos = src_data[i] / BLOCK_BITS;
        if (last_pos != current_pos) {
            if (last_pos != -1) {
                col->pos[cur_block_id] = last_pos;
                col->value[cur_block_id] = last_value;
                cur_block_id++;
            }
        last_pos = current_pos;
        last_value = 0;
        }
        unsigned long long mask = ((unsigned long long) 1) << (src_data[i] % BLOCK_BITS);
        last_value |= mask;
        if(i == (src_length-1)) {
            col->pos[cur_block_id] = last_pos;
            col->value[cur_block_id] = last_value;
        }
    }
}

gpu_boundary_matrix::gpu_boundary_matrix(phat::boundary_matrix <phat::vector_vector> *src_matrix,
                                         indx chunks_num, ScatterAllocator::AllocatorHandle allocator) {
    auto cols_num = (size_t) src_matrix->get_num_cols();
    auto h_matrix = new column[cols_num];
    auto h_column_length = new size_t[cols_num];
    auto h_dims = new dimension[cols_num];
    auto h_lowest_one_lookup = new indx[cols_num];
    auto h_is_reduced = new bool[cols_num];
    auto h_leftmost_lookup_lowest_row = new indx[cols_num];

    auto chunk_size = (size_t) CUDA_THREADS_EACH_BLOCK(cols_num);

    for(indx j = 0; j<cols_num; j++)
        h_leftmost_lookup_lowest_row[j] = -1;

    for (phat::index i = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        h_column_length[i] = col.size();
        h_lowest_one_lookup[i] = -1;
        h_is_reduced[i] = false;
        h_dims[i] = src_matrix->get_dim(i);
        indx max_index = src_matrix->get_max_index(i);
        if(max_index != -1) {
            if(h_leftmost_lookup_lowest_row[max_index] == -1)
               h_leftmost_lookup_lowest_row[max_index] = i;
        }
    }

    //for(indx j=0; j<cols_num;j++)
    //    printf("max_index %d col id is %ld\n", j, h_leftmost_lookup_lowest_row[j]);

    gpuErrchk(cudaMalloc((void **) &matrix, sizeof(column) * cols_num));
    gpuErrchk(cudaMalloc((void **) &chunk_columns_finished, sizeof(unsigned long long) * chunks_num));
    gpuErrchk(cudaMalloc((void **) &dims, sizeof(dimension) * cols_num));
    gpuErrchk(cudaMalloc((void **) &column_length, sizeof(size_t) * cols_num));
    gpuErrchk(cudaMalloc((void **) &lowest_one_lookup, sizeof(indx) * cols_num));
    gpuErrchk(cudaMalloc((void **) &is_reduced, sizeof(bool) * cols_num));
    gpuErrchk(cudaMalloc((void **) &leftmost_lookup_lowest_row, sizeof(indx) * cols_num));

    size_t * d_column_length;
    gpuErrchk(cudaMalloc((void **) &d_column_length, sizeof(size_t) * cols_num));
    gpuErrchk(cudaMemcpy(d_column_length, h_column_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dims, h_dims, sizeof(dimension) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(column_length, h_column_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(lowest_one_lookup, h_lowest_one_lookup, sizeof(indx) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(is_reduced, h_is_reduced, sizeof(bool) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(leftmost_lookup_lowest_row, h_leftmost_lookup_lowest_row, sizeof(indx) * cols_num, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(matrix, h_matrix, sizeof(column) * cols_num, cudaMemcpyHostToDevice));

    indx ** tmp_gpu_columns, ** h_tmp_gpu_columns;
    h_tmp_gpu_columns = new indx * [cols_num];
    gpuErrchk(cudaMalloc((void **) &tmp_gpu_columns, sizeof(indx *) * cols_num));
    //gpuErrchk(cudaMalloc((void **) &leftmost_lookup_lowest_row, sizeof(indx *) * cols_num));
    allocate_all_columns <<< CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >>> (tmp_gpu_columns, d_column_length, cols_num, allocator);
    cudaMemcpy(h_tmp_gpu_columns, tmp_gpu_columns, sizeof(indx *) * cols_num, cudaMemcpyDeviceToHost);

    for (phat::index i = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        auto col_data_ptr = &col[0];

        gpuErrchk(cudaMemcpy(h_tmp_gpu_columns[i], col_data_ptr, sizeof(indx) * col.size(),
                             cudaMemcpyHostToDevice));
    }

    transform_all_columns <<< CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >>> (tmp_gpu_columns,
            column_length, matrix, cols_num, allocator, dims);

    gpuErrchk(cudaFree(tmp_gpu_columns));
    delete[] h_matrix;
    delete[] h_column_length;
    delete[] h_dims;
    delete[] h_lowest_one_lookup;
    delete[] h_tmp_gpu_columns;
    delete[] h_leftmost_lookup_lowest_row;
}

__global__ void transform_unpacked_data(column *matrix, unpacked_matrix* u_matrix, ScatterAllocator::AllocatorHandle allocator)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    int count_length = 0;
    for(int i = 0; i<matrix[thread_id].data_length; i++)
    {
        for(int j = 0; j < BLOCK_BITS; j++)
        {
            unsigned long long temp_value = 1 << j;
            if(matrix[thread_id].value[i] & temp_value)
            {
                count_length++;
            }
        }
    }

    u_matrix->column[thread_id].data = (indx*) allocator.malloc(sizeof(indx) * count_length);
    u_matrix->column[thread_id].data_length = count_length;
    count_length = 0;
    for(int i = 0; i < matrix[thread_id].data_length; i++)
    {
        for(int j = 0; j < BLOCK_BITS; j++)
        {
            unsigned long long temp_value = 1 << j;
            if(matrix[thread_id].value[i] & temp_value)
            {
                u_matrix->column[thread_id].data[count_length] = matrix[thread_id].pos[i] * BLOCK_BITS + j;
                count_length++;
            }
        }
    }
}

/*__host__ void free_cuda_memory(columns *matrix, dimension* dims, indx* lowest_one_lookup, indx* chunks_start_offset, indx* chunk_columns_finished, short* column_type, bool* is_active, bool* is_ready_for_mark) {
    gpuErrchk(cudaFree(dims));
    gpuErrchk(cudaFree(matrix));
    gpuErrchk(cudaFree(lowest_one_lookup));
    gpuErrchk(cudaFree(chunks_start_offset));
    gpuErrchk(cudaFree(chunk_columns_finished));
    gpuErrchk(cudaFree(column_type));
    gpuErrchk(cudaFree(is_active));
    gpuErrchk(cudaFree(is_ready_for_mark));
}
*/
__host__ void transfor_data_backto_cpu(phat::boundary_matrix<phat::vector_vector> *src_matrix,unpacked_matrix *d_matrix)
{
    indx cols_num = src_matrix->get_num_cols();
    auto h_all_columns = new unpacked_column[cols_num];

    gpuErrchk(cudaMemcpy(h_all_columns, d_matrix->column, sizeof(column) * cols_num, cudaMemcpyDeviceToHost));
    for (int i = 0; i < cols_num; i++) {
    auto h_single_column = h_all_columns[i];

    phat::column tmp_vector(h_single_column.data_length);
    gpuErrchk(cudaMemcpy(&tmp_vector[0], h_single_column.data,
                sizeof(indx) * h_single_column.data_length, cudaMemcpyDeviceToHost));
    src_matrix->set_col(i, tmp_vector);
    }

    delete [] h_all_columns;
}

__device__ dimension get_dim(dimension* dims, int col_id) {
    return dims[col_id];
}

__device__ bool is_empty(column* matrix, int col_id) {
    return matrix[col_id].data_length == 0;
}

__device__ indx get_max_index(column* matrix, int col_id) {
    if (matrix[col_id].data_length == 0)
        return -1;
    else {
        unsigned long long t = matrix[col_id].value[matrix[col_id].data_length - 1];
        int cnt = 0;
        while ((t >> 1) > 0) {
            t = (t >> 1);
            cnt++;
        }
        return (matrix[col_id].pos[matrix[col_id].data_length - 1] * BLOCK_BITS + cnt);
    }
}

__device__ indx get_second_max_index(column* matrix, int col_id)
{
    if (matrix[col_id].data_length == 0)
        return -1;
    else
    {
        unsigned long long t = matrix[col_id].value[matrix[col_id].data_length - 1];
        int cnt = 0;
        while ((t >> 1) > 1) {
            t = (t >> 1);
            cnt++;
        }
        if(cnt == 0 && matrix[col_id].value[matrix[col_id].data_length - 1] == 0)
            return -1;
        return (matrix[col_id].pos[matrix[col_id].data_length - 1] * BLOCK_BITS + cnt);
    }
}

__device__ void clear_column(column* matrix, int col_id) {
    matrix[col_id].data_length = 0;
}

__device__ void remove_max_index(column* matrix, int col) {
    if(matrix[col].data_length == 0)
        return;
    unsigned long long t = matrix[col].value[matrix[col].data_length - 1];
    int cnt = 1;
    while ((t >> 1) != 0) {
        t = t >> 1;
        cnt++;
    }
    int tx = (1 << (cnt-1));
    matrix[col].value[matrix[col].data_length - 1]  ^= tx;
    if (matrix[col].value[matrix[col].data_length - 1] == 0)
        matrix[col].data_length--;
}

__device__ void check_lowest_one_locally(column* matrix, unsigned long long* chunk_columns_finished, dimension * dims,
        bool* is_reduced, indx* leftmost_lookup_lowest_row, int thread_id, int block_id, indx column_start, indx column_end, indx row_begin, indx row_end,
        dimension cur_dim, indx num_cols, int* target_col, bool* ive_added, indx* front_lowest_one) {
    indx my_lowest_one = get_max_index(matrix, thread_id);
    if (cur_dim != get_dim(dims, thread_id) || is_reduced[thread_id] == true || thread_id >= num_cols || my_lowest_one == -1)
    {
        if (!*ive_added) {
            atomicAdd((unsigned long long *) &chunk_columns_finished[block_id], (unsigned long long) 1);
            *ive_added = true;
        }
        return;
    }

    //bool flag = false;
    if (my_lowest_one >= row_begin && my_lowest_one < row_end) {
    //for(indx col_id = column_start; col_id < thread_id; col_id++)
    //{
        //indx this_lowest_one = get_max_index(matrix, col_id);
        //if(this_lowest_one == my_lowest_one){
        //    *target_col = col_id;
        //leftmost_lookup_lowest_row[my_lowest_one]>=column_start;
        if (leftmost_lookup_lowest_row[my_lowest_one]>=column_start && leftmost_lookup_lowest_row[my_lowest_one] < thread_id) {
            *target_col = leftmost_lookup_lowest_row[my_lowest_one];
            *front_lowest_one = my_lowest_one;
            //flag = true;
            if (*ive_added) {
                atomicAdd((unsigned long long *) &chunk_columns_finished[block_id], (unsigned long long) -1);
                *ive_added = false;
            }
            return;
        }
    }
    //}

    if (!*ive_added) {
       atomicAdd((unsigned long long *) &chunk_columns_finished[block_id], (unsigned long long) 1);
       *ive_added = true;
    }
}

__device__ void add_two_columns(column* matrix, indx target, indx source, ScatterAllocator::AllocatorHandle allocator)
{
    if(source == -1 || source == target)
        return;
    auto tgt_col = &matrix[target];
    size_t tgt_id = 0, src_id = 0, temp_id = 0;
    indx max_size = round_up_to_2s(matrix[target].data_length + matrix[source].data_length);
    max_size = max_size > ADDED_SIZE ? max_size : ADDED_SIZE;

    bool need_new_mem = tgt_col->data_length < max_size;
    auto new_pos = (need_new_mem) ? (indx *) allocator.malloc(sizeof(indx) * max_size) : new indx[max_size];
    auto new_value = (need_new_mem) ? (unsigned long long *) allocator.malloc(sizeof(unsigned long long) * max_size) : new unsigned long long [max_size];

    while ((tgt_id < matrix[target].data_length) && (src_id < matrix[source].data_length)) {
        if (matrix[target].pos[tgt_id] == matrix[source].pos[src_id]) {
            if ((matrix[target].value[tgt_id] ^ matrix[source].value[src_id]) != 0) {
                new_pos[temp_id] = matrix[target].pos[tgt_id];
                new_value[temp_id] = (matrix[target].value[tgt_id] ^ matrix[source].value[src_id]);
                temp_id++;
            }
                tgt_id++;
                src_id++;
        }
        else if (matrix[target].pos[tgt_id] < matrix[source].pos[src_id]) {
            new_value[temp_id] = matrix[target].value[tgt_id];
            new_pos[temp_id] = matrix[target].pos[tgt_id];
            tgt_id++;
            temp_id++;
        }
        else {
            new_value[temp_id] = matrix[source].value[src_id];
            new_pos[temp_id] = matrix[source].pos[src_id];
            src_id++;
            temp_id++;
        }
    }

    if (src_id < matrix[source].data_length) {
        memcpy(&new_pos[temp_id], &matrix[source].pos[src_id], sizeof(indx) * (matrix[source].data_length - src_id));
        memcpy(&new_value[temp_id], &matrix[source].value[src_id], sizeof(unsigned long long) * (matrix[source].data_length - src_id));
        temp_id += matrix[source].data_length - src_id;
    }
    if (tgt_id < matrix[target].data_length) {
        memcpy(&new_pos[temp_id], &matrix[target].pos[tgt_id], sizeof(indx) * (matrix[target].data_length - tgt_id));
        memcpy(&new_value[temp_id], &matrix[target].value[tgt_id], sizeof(unsigned long long) * (matrix[target].data_length - tgt_id));
        temp_id += matrix[target].data_length - tgt_id;
    }

    if (need_new_mem) {
        allocator.free(tgt_col->pos);
        allocator.free(tgt_col->value);
        tgt_col->pos = new_pos;
        tgt_col->value = new_value;
        tgt_col->size = (size_t) max_size;
    }
    else {
        memcpy(tgt_col->pos, new_pos, sizeof(indx) * temp_id);
        memcpy(tgt_col->value, new_value, sizeof(unsigned long long) * temp_id);
        delete new_pos;
        delete new_value;
    }

    tgt_col->data_length = temp_id;
}

__device__ void mark_and_clean(column* matrix, indx* lowest_one_lookup, indx* leftmost_lookup_lowest_row, bool* is_reduced, dimension* dims, indx my_col_id,
        indx row_begin, indx row_end, dimension cur_dim) {
    if (cur_dim != get_dim(dims, my_col_id) || is_reduced[my_col_id]) {
        return;
    }

    indx my_lowest_one = get_max_index(matrix, my_col_id);
    if(my_lowest_one == -1){
        is_reduced[my_col_id] = true;
        return;
    }
    if (lowest_one_lookup[my_lowest_one] == -1 && my_lowest_one >= row_begin && my_lowest_one < row_end) {
        lowest_one_lookup[my_lowest_one] = my_col_id;
        is_reduced[my_col_id] = true;
        clear_column(matrix, my_lowest_one);
        //update_lookup_lowest_table(matrix, leftmost_lookup_lowest_row, my_lowest_one);
        is_reduced[my_lowest_one] = true;
    }
}

__device__ void update_lookup_lowest_table(column* matrix, indx* leftmost_lookup_lowest_row, indx front_lowest_one, int thread_id)
{
    indx cur_lowest_one = get_max_index(matrix, thread_id);
    if(front_lowest_one != -1 && leftmost_lookup_lowest_row[front_lowest_one] == thread_id)
        atomicExch((int *)&leftmost_lookup_lowest_row[front_lowest_one], -1);
    if(cur_lowest_one == -1)
        return;
    if(leftmost_lookup_lowest_row[cur_lowest_one] == -1)
    {
        atomicExch((int *)&leftmost_lookup_lowest_row[cur_lowest_one], thread_id);
        __threadfence();
    }
    else
    {
        atomicMin((int *)&leftmost_lookup_lowest_row[cur_lowest_one], thread_id);
        __threadfence();
    }
}

__global__ void update_table(column* matrix, indx* leftmost_lookup_lowest_row, indx cur_phase, indx block_size)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    //int block_id = blockIdx.x;
    //int threadidx = threadIdx.x;
    indx column_start = ((blockDim.x * blockIdx.x - cur_phase * block_size) > 0) ? (blockDim.x * blockIdx.x - cur_phase * block_size) : 0;
    indx column_end = column_start + block_size * (cur_phase+1);
    //indx row_begin = ((blockDim.x * blockIdx.x - cur_phase * block_size) > 0) ? (blockDim.x * blockIdx.x - cur_phase * block_size) : 0;
    //indx row_end = row_begin + block_size;

    indx this_max_index = get_max_index(matrix, thread_id);
    __syncthreads();
    if(thread_id >= column_start && thread_id < column_end)// && this_max_index >= row_begin && this_max_index < row_end)
    {
        if(leftmost_lookup_lowest_row[this_max_index] == -1)
            atomicExch((int *)&leftmost_lookup_lowest_row[this_max_index], thread_id);
        else
            atomicMin((int *)&leftmost_lookup_lowest_row[this_max_index], thread_id);
    }
}

/*__global__ void init_table(column* matrix, lookup_table *lookup_lowest_row_table, indx num_cols, ScatterAllocator::AllocatorHandle allocator)
{
    //for(indx cur_lowest_one = 0; cur_lowest_one < num_cols; cur_lowest_one++)
    //{
    indx cur_lowest_one = threadIdx.x + blockDim.x * blockIdx.x;
    indx temp_id = 0;
    for(indx cur_id = 0; cur_id < num_cols; cur_id++)
    {
        indx this_lowest_one = get_max_index(matrix, cur_id);
        if(this_lowest_one == cur_lowest_one)
        {
            temp_id++;
        }
    }
    lookup_lowest_row_table[cur_lowest_one].data = (indx *) allocator.malloc(sizeof(indx) * temp_id);
    lookup_lowest_row_table[cur_lowest_one].data_length = temp_id;
//gpuErrchk(cudaMalloc((void **) &lookup_lowest_row_table[cur_lowest_one].data, sizeof(indx) * temp_id));
//gpuErrchk(cudaMemcpy(lookup_lowest_row_table[cur_lowest_one].data, temp_lowest_table, sizeof(indx) * temp_id, cudaMemcpyHostToDevice));
//gpuErrchk(cudaMalloc((void **) &lookup_lowest_row_table[cur_lowest_one].data_length, sizeof(indx)));
//gpuErrchk(cudaMemcpy(&lookup_lowest_row_table[cur_lowest_one].data_length, &temp_id, sizeof(indx), cudaMemcpyHostToDevice));
//}
    temp_id = 0;
    for(indx cur_id = 0; cur_id < num_cols; cur_id++)
    {
        indx this_lowest_one = get_max_index(matrix, cur_id);
        if(this_lowest_one == cur_lowest_one)
        {
            lookup_lowest_row_table[cur_lowest_one].data[temp_id] = cur_id;
            temp_id++;
        }
    }
}

__device__ indx lookup_col_index(lookup_table* lookup_lowest_row_table, indx cur_lowest_one, indx column_begin, indx cur_col)
{
    for(indx cur_id = 0; cur_id < lookup_lowest_row_table[cur_lowest_one].data_length; cur_id++)
    {
        indx temp_col = lookup_lowest_row_table[cur_lowest_one].data[cur_id];
        if(temp_col >= column_begin && temp_col < cur_col)
            return temp_col;
    }
}

__device__ indx check_col_in_table(lookup_table* lookup_lowest_row_table, indx cur_lowest_one, indx col_id)
{
    for(indx id = 0; id < lookup_lowest_row_table[cur_lowest_one].data_length; id++)
        if(id == col_id)
            return id;
    return -1;
}

__device__ void delete_col_in_table(lookup_table* lookup_lowest_row_table, indx cur_lowest_one, indx col_id)
{
    indx col_index = check_col_in_table(lookup_table* lookup_lowest_row_table, indx cur_lowest_one, indx col_id);
    if(col_index == -1)
        return;
    size_t len = lookup_lowest_row_table[cur_lowest_one].data_length;
    for(indx id = col_index; id < len; id++)
        lookup_lowest_row_table[cur_lowest_one].data[id] = lookup_lowest_row_table[cur_lowest_one].data[id+1];
    lookup_lowest_row_table[cur_lowest_one].data_length--;
}

__device__ void add_col_in_table(lookup_table* lookup_lowest_row_table, indx cur_lowest_one, indx col_id, ScatterAllocator::AllocatorHandle allocator)
{
    indx insert_id = -1;
    for(indx id=0; id<lookup_lowest_row_table[cur_lowest_one].data_length; id++)
    {
        if(lookup_lowest_row_table[cur_lowest_one].data[id] > col_id)
        {
            insert_id = id;
            break;
        }
    }
    if(insert_id == -1)
        return;
    auto new_data = (indx *)allocator.malloc(sizeof(indx) * (lookup_lowest_row_table[cur_lowest_one].data_length+1));
    for(indx id = 0; id < insert_id; id++)
        new_data[id] = lookup_lowest_row_table[cur_lowest_one].data[id];
    new_data[insert_id] = col_id;
    for(indx id=(insert_id+1); id<(lookup_lowest_row_table[cur_lowest_one].data_length+1); id++)
        new_data[id] = lookup_lowest_row_table[cur_lowest_one].data[id-1];

    lookup_lowest_row_table[cur_lowest_one].data = new_data;
    lookup_lowest_row_table[cur_lowest_one].data_length++;
}*/


