//
// Created by 唐艺峰 on 2018/7/27.
//

#include "gpu_matrix.h"

__device__ dimension mx_get_dim(gpu_matrix *matrix, indx idx) {
    return matrix->dims[idx];
}

__device__ bool mx_is_empty(gpu_matrix *matrix, indx idx) {
    return matrix->data[idx].data_length == 0;
}

__device__ indx mx_get_max_indx(gpu_matrix *matrix, indx idx) {
    return matrix->data[idx].data_length == 0 ? -1 :
           matrix->data[idx].data[matrix->data[idx].data_length - 1];
}

__device__ void mx_remove_max(gpu_matrix * matrix, indx idx) {
    if (matrix->data[idx].data_length != 0) {
        matrix->data[idx].data_length--;
    }
}

__device__ void mx_add_to(gpu_matrix *matrix, indx source, indx target, ScatterAllocator::AllocatorHandle *allocator) {
    auto src_col = &matrix->data[source];
    auto tgt_col = &matrix->data[target];
    auto tgt_data = tgt_col->data;
    indx max_size = round_up_to_2s(src_col->data_length + tgt_col->data_length);
    max_size = max_size > ADDED_SIZE ? max_size : ADDED_SIZE;       // To ensure a smallest size of each column

    auto new_data = (tgt_col->size < max_size) ? (indx *) allocator->malloc(sizeof(indx) * max_size) : new indx[max_size];
    // If the old column is able to contain the new data, we don't need to allocate a new block of memory for it which
    //  would take a lot of time because mallocmc is not too efficient.

    size_t src_pos = 0, tgt_pos = 0, new_pos = 0;
    while (src_pos < src_col->data_length && tgt_pos < tgt_col->data_length) {
        auto cur_src_data = src_col->data[src_pos];
        auto cur_tgt_data = tgt_col->data[tgt_pos];
        if (cur_src_data == cur_tgt_data) {
            src_pos++;
            tgt_pos++;
        } else if (cur_src_data > cur_tgt_data) {
            new_data[new_pos] = cur_tgt_data;
            new_pos++;
            tgt_pos++;
        } else {
            new_data[new_pos] = cur_src_data;
            new_pos++;
            src_pos++;
        }
    }

    if (src_pos < src_col->data_length) {
        memcpy(&new_data[new_pos], &src_col->data[src_pos], sizeof(indx) * (src_col->data_length - src_pos));
        new_pos += src_col->data_length - src_pos;
    } else if (tgt_pos < tgt_col->data_length) {
        memcpy(&new_data[new_pos], &tgt_col->data[tgt_pos], sizeof(indx) * (tgt_col->data_length - tgt_pos));
        new_pos += tgt_col->data_length - tgt_pos;
    }

    if (tgt_col->size < max_size) {
        allocator->free(tgt_data);
        tgt_col->data = new_data;
        tgt_col->size = (size_t) max_size;
    } else {
        memcpy(tgt_col->data, new_data, sizeof(indx) * new_pos);
        delete new_data;
    }
    tgt_col->data_length = new_pos;
}

__device__ void mx_clear(gpu_matrix *matrix, indx idx) {
    matrix->data[idx].data_length = 0;
}

__host__ gpu_matrix *mx_init_stable_matrix(ScatterAllocator::AllocatorHandle allocator,
                                           phat::boundary_matrix<phat::vector_vector> *src_matrix,
                                           indx chunks_num) {
    gpu_matrix *d_matrix;
    dimension *d_dims_ptr;
    column *d_cols_ptr;
    indx *d_data_length;
    indx *d_chunks_start_offset;
    indx *d_chunk_finish_col_num;
    short *d_column_type;
    indx *d_lowest_one;
    bool *d_is_active;
    bool *d_is_ready_for_marking;

    auto h_matrix = new gpu_matrix;
    auto cols_num = (size_t) src_matrix->get_num_cols();
    auto h_data_length = new size_t[cols_num];
    auto h_chunks_start_offset = new indx[chunks_num + 1];
    auto h_column_type = new short[cols_num];
    auto h_lowest_one = new indx[cols_num];
    auto h_dims_ptr = new dimension[cols_num];

    auto chunk_size = (size_t) CUDA_THREADS_EACH_BLOCK(cols_num);

    for (phat::index i = 0, chunk_pos = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        h_data_length[i] = col.size();
        h_dims_ptr[i] = src_matrix->get_dim(i);
        h_column_type[i] = GLOBAL;
        h_lowest_one[i] = -1;
        if (i % chunk_size == 0) {
            h_chunks_start_offset[chunk_pos] = i;
            chunk_pos++;
        }
    }
    h_chunks_start_offset[chunks_num] = (indx) (cols_num);

    gpuErrchk(cudaMalloc((void **) &d_dims_ptr, sizeof(dimension) * cols_num));
    gpuErrchk(cudaMalloc((void **) &d_data_length, sizeof(indx) * cols_num));
    gpuErrchk(cudaMalloc((void **) &d_lowest_one, sizeof(indx) * cols_num));
    gpuErrchk(cudaMalloc((void **) &d_chunks_start_offset, sizeof(indx) * (chunks_num + 1)));
    gpuErrchk(cudaMalloc((void **) &d_chunk_finish_col_num, sizeof(indx) * chunks_num));
    gpuErrchk(cudaMalloc((void **) &d_column_type, sizeof(short) * cols_num));
    gpuErrchk(cudaMalloc((void **) &d_cols_ptr, sizeof(column) * cols_num));
    gpuErrchk(cudaMalloc((void **) &d_is_active, sizeof(bool) * cols_num));
    gpuErrchk(cudaMalloc((void **) &d_is_ready_for_marking, sizeof(bool) * cols_num));

    gpuErrchk(cudaMemcpy(d_dims_ptr, h_dims_ptr, sizeof(dimension) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_data_length, h_data_length, sizeof(size_t) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lowest_one, h_lowest_one, sizeof(indx) * cols_num, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_chunks_start_offset, h_chunks_start_offset, sizeof(indx) * (chunks_num + 1),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_column_type, h_column_type, sizeof(short) * cols_num, cudaMemcpyHostToDevice));

    h_matrix->dims = d_dims_ptr;
    h_matrix->data = d_cols_ptr;
    h_matrix->data_length = d_data_length;
    h_matrix->column_num = cols_num;
    h_matrix->chunks_start_offset = d_chunks_start_offset;
    h_matrix->chunk_finish_col_num = d_chunk_finish_col_num;
    h_matrix->column_type = d_column_type;
    h_matrix->lowest_one_lookup = d_lowest_one;
    h_matrix->is_active = d_is_active;
    h_matrix->is_ready_for_mark = d_is_ready_for_marking;

    gpuErrchk(cudaMalloc(&d_matrix, sizeof(gpu_matrix)));
    gpuErrchk(cudaMemcpy(d_matrix, h_matrix, sizeof(gpu_matrix), cudaMemcpyHostToDevice));

    mx_allocate_all_columns << < CUDA_BLOCKS_NUM(cols_num), CUDA_THREADS_EACH_BLOCK(cols_num) >> >
                                                            (d_matrix, allocator);

    for (phat::index i = 0; i < cols_num; i++) {
        phat::column col;
        src_matrix->get_col(i, col);
        auto col_data_ptr = &col[0];

        column h_single_column;
        gpuErrchk(cudaMemcpy(&h_single_column, &d_cols_ptr[i], sizeof(column), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_single_column.data, col_data_ptr, sizeof(indx) * h_single_column.data_length,
                             cudaMemcpyHostToDevice));
    }

    delete h_matrix;
    delete[] h_data_length;
    delete[] h_chunks_start_offset;
    delete[] h_column_type;
    delete[] h_lowest_one;
    delete[] h_dims_ptr;
    return d_matrix;
}

__host__ void mx_free_cuda_memory(gpu_matrix *matrix) {
    auto h_matrix = new gpu_matrix;

    gpuErrchk(cudaMemcpy(h_matrix, matrix, sizeof(gpu_matrix), cudaMemcpyDeviceToHost)); // fixme: too slow ?

    gpuErrchk(cudaFree(h_matrix->dims));
    gpuErrchk(cudaFree(h_matrix->data_length));
    gpuErrchk(cudaFree(h_matrix->lowest_one_lookup));
    gpuErrchk(cudaFree(h_matrix->chunks_start_offset));
    gpuErrchk(cudaFree(h_matrix->chunk_finish_col_num));
    gpuErrchk(cudaFree(h_matrix->column_type));
    gpuErrchk(cudaFree(h_matrix->data));
    gpuErrchk(cudaFree(h_matrix->is_active));
    gpuErrchk(cudaFree(h_matrix->is_ready_for_mark));
    gpuErrchk(cudaFree(matrix));

    delete h_matrix;
}

__global__ void mx_allocate_all_columns(gpu_matrix *matrix, ScatterAllocator::AllocatorHandle allocator) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= matrix->column_num) {
        return;
    }
    auto col = &matrix->data[thread_id];
    auto length = matrix->data_length[thread_id];
    auto size = round_up_to_2s(length);
    col->data_length = (size_t) length;
    col->size = (size_t) (size > INITIAL_SIZE ? size : INITIAL_SIZE);
    col->data = (indx *) allocator.malloc(sizeof(indx) * size);
}

__host__ void mx_copy_matrix_to_cpu_with_one_step(phat::boundary_matrix<phat::vector_vector> *src_matrix,
                                                  gpu_matrix *d_matrix, std::vector<indx> &lowest_one_lookup, std::vector<short> &column_type) {
    auto h_matrix = new gpu_matrix;
    indx cols_num = src_matrix->get_num_cols();
    auto h_all_columns = new column[cols_num];

    gpuErrchk(cudaMemcpy(h_matrix, d_matrix, sizeof(gpu_matrix), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_all_columns, h_matrix->data, sizeof(column) * cols_num, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&lowest_one_lookup[0], h_matrix->lowest_one_lookup, sizeof(indx) * cols_num,
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&column_type[0], h_matrix->column_type, sizeof(short) * cols_num, cudaMemcpyDeviceToHost));

    for (int i = 0; i < cols_num; i++) {
        auto h_single_column = h_all_columns[i];
        if (h_single_column.data_length == 0 || column_type[i] != GLOBAL) {
            continue;
        }

        phat::column tmp_vector(h_single_column.data_length);
        gpuErrchk(cudaMemcpy(&tmp_vector[0], h_single_column.data, sizeof(indx) * h_single_column.data_length, cudaMemcpyDeviceToHost));
        src_matrix->set_col(i, tmp_vector);
    }

    delete h_matrix;
    delete[] h_all_columns;
}

__host__ void mx_save_ascii(std::vector<indx> &lowest_one_lookup, const char *filename) {
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