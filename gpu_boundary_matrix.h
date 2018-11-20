#ifndef _GPU_BOUNDARY_MATRIX_H_
#define _GPU_BOUNDARY_MATRIX_H_

#include "gpu_common.h"

#define BLOCK_BITS      4
#define GLOBAL          0

#define ADDED_SIZE      32
#define INITIAL_SIZE    8

typedef long indx;
typedef short dimension;

//! \def INITIAL_SIZE
//! \brief Initial size is small because most of columns can be cleared soon
#define INITIAL_SIZE    8

//structure stores columns of boundary matrix.
struct column {
    indx * pos; //the indx of non-zero 64-bits array in the current column.
    unsigned long long * value; //the value of non-zero 64-bits array represented by long integer.
    size_t data_length;  //the number of non-zero 64-bits arrays in the current column.
    size_t size;  //maximal number of non-zero 64-bits arrays in the current column.
};

class gpu_boundary_matrix {
public:
    column * matrix; //stores columns of boundary matrix.
    indx * chunk_offset; //stores the offset of each chunk/block in gpu.
    indx * chunk_columns_finished; //counts the number of operations that are finished.
    short * column_type;  //denotes type of each column: global, local negative or local positive.
    dimension * dims;     //stores dimension of each column of boundary matrix.
    size_t * column_length; //stores length of each column.
                         //look-up table L[i], i represents the row indx, and corresponding L[i] represents the column indx
                         //whose lowest row is i.
    indx * lowest_one_lookup;
    bool * is_active; //denotes each column as active or inactive.
    bool * is_ready_for_mark;

public:
    //construct boundary matrix on gpu.
    static gpu_boundary_matrix * create_gpu_boundary_matrix(phat::boundary_matrix <phat::vector_vector> *src_matrix,
                        indx chunks_num, ScatterAllocator::AllocatorHandle allocator);

    __device__ indx current_row_index(indx col_id, indx cur_row_idx);
};
//get the dimension of column indexed with col.
__device__ dimension get_dim(dimension* dims, indx col);

//to know whether current column col is non-zero(false) or not(true).
__device__ bool is_empty(column* matrix, indx col);

//to get the lowest row indx of column col.
__device__ indx get_max_index(column* matrix, indx col);

//set column col as zero.
__device__ void clear_column(column* matrix, indx col);

//add two columns locally.
__device__ void add_two_columns(column* matrix, indx target, indx source, ScatterAllocator::AllocatorHandle * allocator);

//set lowest non-zero row of column col as zero.
__device__ void remove_max_index(column* matrix, indx col);
#endif
