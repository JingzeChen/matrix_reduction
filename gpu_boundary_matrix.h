#ifndef PHAT_CUDA_COMMON_H
#define PHAT_CUDA_COMMON_H

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cuda.h"

#include "mallocMC.hpp"

#define GLOBAL 0
#define LOCAL_NEGATIVE -1
#define LOCAL_POSITIVE 1

typedef long index;
typedef short dimension;

//structure stores columns of boundary matrix.
struct column{
    index* pos; //the index of non-zero 64-bits array in the current column.
    unsigned long long *value; //the value of non-zero 64-bits array represented by long integer.
    size_t data_length;  //the number of non-zero 64-bits arrays in the current column.
    size_t _size;  //maximal number of non-zero 64-bits arrays in the current column.
}

class gpu_boundary_matrix
{
public:
    column* matrix; //stores columns of boundary matrix.
    index* chunk_offset; //stores the offset of each chunk/block in gpu.
    index* chunk_columns_finished; //counts the number of operations that are finished.
    short* column_type;  //denotes type of each column: global, local negative or local positive.
    dimension* dims;   //stores dimension of each column of boundary matrix.
    index* column_length; //stores length of each column.
//look-up table L[i], i represents the row index, and corresponding L[i] represents the column index
// whose lowest row is i.
    index* lowest_one_lookup;
    bool* is_active; //denotes each column as active or inactive.
    bool* is_ready_for_mark;

public:
    //construct boundary matrix on gpu.
    gpu_boundary_matrix();

    //destory boundary matrix on gpu.
    ~gpu_boundary_matrix();

    //get the dimension of column indexed with col.
    __device__ dimension get_max_dim(int col);

    //to know whether current column col is non-zero(false) or not(true).
    __device__ bool is_empty(int col);

    //to get the lowest row index of column col.
    __device__ index get_max_index(int col);

    //set column col as zero.
    __device__ void clear_column(int col);

    //set lowest non-zero row of column col as zero.
    __device__ void remove_max_index(int col);

    //search for column in the same chunk and the row index of the column is equal to the lowest row index of column my_col_id.
    __device__ void check_lowest_one_locally(index my_col_id, index block_id,index chunk_start,
            index row_begin,dimension cur_dim, index *target_col, bool * ive_added);

    //add two columns locally.
    __device__ void add_two_columns(int target, int source, ScatterAllocator::AllocatorHandle allocator);

    //mark column as global, local positive or local negative and clean corresponding column as zero.
    __device__ void mark_and_clean(index my_col_id, indx row_begin, dimension cur_dim);

    //to get the row index of
    __device__ index current_row_index(col_id, cur_row_idx);
}
