#ifndef _GPU_BOUNDARY_MATRIX_H_
#define _GPU_BOUNDARY_MATRIX_H_

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "gpu_common.h"

#define BLOCK_BITS 64
#define GLOBAL 0
#define LOCAL_NEGATIVE -1
#define LOCAL_POSITIVE 1

typedef long indx;
typedef short dimension;

//! \def INITIAL_SIZE
//! \brief Initial size is small because most of columns can be cleared soon
#define INITIAL_SIZE    8

//structure stores columns of boundary matrix.
struct column {
    indx *pos; //the indx of non-zero 64-bits array in the current column.
    unsigned long long *value; //the value of non-zero 64-bits array represented by long integer.
    size_t data_length;  //the number of non-zero 64-bits arrays in the current column.
    size_t size;  //maximal number of non-zero 64-bits arrays in the current column.
};

struct unpacked_column {
    indx *data;
    size_t data_length;
};

struct unpacked_matrix {
    unpacked_column *column;
    size_t column_num;
};

struct lookup_table {
    indx *data;
    size_t data_length;
};

class gpu_boundary_matrix {
public:
    column *matrix; //stores columns of boundary matrix.
    unsigned long long *chunk_columns_finished; //counts the number of operations that are finished.
    dimension *dims;     //stores dimension of each column of boundary matrix.
    size_t *column_length; //stores length of each column.
    //look-up table L[i], i represents the row indx, and corresponding L[i] represents the column indx
    //whose lowest row is i.
    indx *lowest_one_lookup;
    bool *is_reduced;
    indx *leftmost_lookup_lowest_row;
    //lookup_table* lookup_lowest_row_table;

public:
    //construct boundary matrix on gpu.
    gpu_boundary_matrix(phat::boundary_matrix<phat::vector_vector> *src_matrix,
                        indx chunks_num, ScatterAllocator::AllocatorHandle allocator);

    __device__ indx current_row_index(indx col_id, indx cur_row_idx);
};
//get the dimension of column indexed with col.
__device__ dimension get_dim(dimension *dims, int col);

//to know whether current column col is non-zero(false) or not(true).
__device__ bool is_empty(column *matrix, int col);

//to get the lowest row indx of column col.
__device__ indx get_max_index(column *matrix, int col);

//set column col as zero.
__device__ void clear_column(column *matrix, int col_id);

//search for column in the same chunk and the row indx of the column is equal to the lowest row indx of column my_col_id.
__device__ void
check_lowest_one_locally(column *matrix, unsigned long long *chunk_columns_finished, dimension *dims, bool *is_reduced,
                         indx *leftmost_lookup_lowest_row,
                         int thread_id, int block_id, indx column_start, indx column_end, indx row_begin, indx row_end,
                         dimension cur_dim, indx num_cols,
                         int *target_col, bool *ive_added);
//mark column as global, local positive or local negative and clean corresponding column as zero.
__device__ void
mark_and_clean(column *matrix, indx *lowest_one_lookup, indx *leftmost_lookup_lowest_row, bool *is_reduced,
               dimension *dims, indx my_col_id, indx row_begin, indx row_end, dimension cur_dim);

//add two columns locally.
__device__ void add_two_columns(column *matrix, indx target, indx source, ScatterAllocator::AllocatorHandle allocator);

//set lowest non-zero row of column col as zero.
__device__ void remove_max_index(column *matrix, int col);

__global__ void
transform_unpacked_data(column *matrix, unpacked_matrix *u_matrix, ScatterAllocator::AllocatorHandle allocator);

//__host__ void free_cuda_memory(columns *matrix, dimension* dims, indx* lowest_one_lookup, indx* chunks_start_offset, indx* chunk_columns_finished, short* column_type, bool* is_active, bool* is_ready_for_mark);

__host__ void
transfor_data_backto_cpu(phat::boundary_matrix<phat::vector_vector> *src_matrix, unpacked_matrix *d_matrix);

__device__ indx get_second_max_index(column *matrix, int col_id);

//__host__ void init_table(column* matrix, indx num_cols);
__device__ void update_lookup_lowest_table(column *matrix, indx *leftmost_lookup_lowest_row, int thread_id);

__global__ void update_table(column *matrix, indx *leftmost_lookup_lowest_row, indx cur_phase, indx block_size);

#endif
