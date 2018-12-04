//
// Created by 唐艺峰 on 2018/7/27.
//

#ifndef PHAT_CUDA_GPU_MATRIX_H
#define PHAT_CUDA_GPU_MATRIX_H

#include "gpu_common.h"

//! \file

#define GLOBAL          0       //!< Using in column type as global column
#define LOCAL_POSITIVE  1       //!< Using in column type as local positive column
#define LOCAL_NEGATIVE  -1      //!< Using in column type as local negative column

//! \def ADDED_SIZE
//! \brief Once a column has been added, it may be added many times so bigger size can avoid too many memory operations.
//! \bug Unexpectedly, ADDED_SIZE \b MUST be bigger than INITIAL_SIZE. There should be something wrong inside mallocmc_lib.
#define ADDED_SIZE      32

//! \def INITIAL_SIZE
//! \brief Initial size is small because most of columns can be cleared soon
#define INITIAL_SIZE    8

//! \class gpu_matrix
//! \brief The data structure for the whole matrix.
//! \note The whole gpu_matrix is stored at global memory in GPU.
//! \todo Can we store all or some data below on shared memory? I can see
//! some really good locality inside, but I have no time to implement.

struct gpu_matrix {
    indx * chunks_start_offset;                 //!< Start column ID of each chunk.
    indx * chunk_finish_col_num;                //!< The number of columns have been finished each chunk. One column is going to add 1 to it many times to finish.
    short * column_type;                        //!< Types of each column.
    dimension * dims;                           //!< Dimension of each column.
    indx * data_length;                         //!< Data length of each column.
    column * data;                              //!< The stored columns which are dynamic allocated by mallomc library.
    size_t column_num;                          //!< The number of columns.
    indx * lowest_one_lookup;                   //!< The column who occupies the lowest row.
    bool * is_active;                           //!< Whether current column is still unfinished.
    bool * is_ready_for_mark;                   //!< A flag using during marking. (Can be moved)
};

//! \brief Get the dimension of the column with ID idx.
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param idx is the ID of column
//! \return The dimension of this column.

__device__ dimension mx_get_dim(gpu_matrix * matrix, indx idx);

//! \brief Check whether column with ID idx is empty.
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param idx is the ID of column
//! \return Whether this column is empty.

__device__ bool mx_is_empty(gpu_matrix * matrix, indx idx);

//! \brief Get the max index (ID of lowest row) of column with ID idx.
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param idx is the ID of column
//! \return The max index gotten.

__device__ indx mx_get_max_indx(gpu_matrix * matrix, indx idx);

//! \brief Add source column to target column
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param source is the ID of source column
//! \param target is the ID of target column
//! \param allocator is the instance of the mallocmc_lib who manages the gpu memory.
//! \return void.
//! \note We do it the way like merge operation. High performance we got by avoiding
//! too many global memory operations useing mallocmc.
//! \todo Can we do it inplace when there is enough memory? I didn't find out any way
//! on Internet.

__device__ void mx_add_to(gpu_matrix * matrix, indx source, indx target, ScatterAllocator::AllocatorHandle * allocator);

//! \brief Clear the column with ID idx.
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param idx is the ID of column
//! \return void.
//! \note We use lazy delete here to avoid memory operations.
//! \warning Memory is easy to run out with lazy delete. Maybe we need to do it with better
//! idea when really big data appears.

__device__ void mx_clear(gpu_matrix * matrix, indx idx);

//! \brief Remove the lowest one of the column
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param idx is the ID of column
//! \return void.
//! \note We use lazy delete here to avoid memory operations by just decrease the length of data

__device__ void mx_remove_max(gpu_matrix * matrix, indx idx);

//! \brief Create GPU matrix and copy data from CPU.
//! \param allocator is the instance of the mallocmc_lib who manages the gpu memory.
//! \param src_matrix is the origin CPU matrix in PHAT library.
//! \param chunks_num is the number of chunks.
//! \return GPU matrix which is stored in GPU.
//! \note We use simple cudaMemcpy to copy one by one.
//! \warning This is extremely slow because of too many cudaMemcpy operations which may be avoided.
//! \todo Try using different ways to get faster speed. We provide some possible way below.
//!     1. Move all data into a large block memory in CPU, copy to GPU and then each thread gets the data it needs parallel.
//!     2. Using new feature provided by CUDA such as streams synchronization and so on.
//!     3. Try zero copy provided by CUDA.
//!     4. etc...

__host__ gpu_matrix * mx_init_stable_matrix(ScatterAllocator::AllocatorHandle allocator, phat::boundary_matrix<phat::vector_vector> * src_matrix,
                                            indx chunks_num);

//! \brief Free all memory in CUDA
//! \param matrix is the \ref gpu_matrix we are working on.

__host__ void mx_free_cuda_memory(gpu_matrix * matrix);

//! \brief Allocate memory for each column
//! \param matrix is the \ref gpu_matrix we are working on.
//! \param allocator is the instance of the mallocmc_lib who manages the gpu memory.

__global__ void mx_allocate_all_columns(gpu_matrix * matrix, ScatterAllocator::AllocatorHandle allocator);

//! \brief Copy memory back to CPU
//! \param src_matrix is the origin CPU matrix from PHAT library.
//! \param d_matrix is the \ref gpu_matrix we are working on.
//! \param lowest_one_lookup is the copy of lowest_one_lookup in CPU.
//! \param column_type is the copy of column_type in CPU.
//! \param is_active is the copy of is_active in CPU.
//! \return void.
//! \todo This is also a little bit slow but better. Please fix it later.

__host__ void mx_copy_matrix_to_cpu_with_one_step(phat::boundary_matrix<phat::vector_vector> * src_matrix, gpu_matrix * d_matrix,
                                                  std::vector<indx> &lowest_one_lookup, std::vector<short> &column_type);

//! \brief Write result to ascii files.
//! \param lowest_one_lookup is the copy of lowest_one_lookup in CPU.
//! \param filename is the name of result file.

__host__ void mx_save_ascii(std::vector<indx> &lowest_one_lookup, const char * filename);

#endif //PHAT_CUDA_GPU_MATRIX_H
