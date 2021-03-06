//
// Created by 唐艺峰 on 2018/7/25.
//

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

#include "boundary_matrix.h"
#define BLOCK_BITS 16
//! \file

//! \def CUDA_MAX_BLOCK_THREADS_NUM
//! \brief The max number of threads in one block of CUDA.
//! \note This number is different from that on different GPU device. 512 seems to be
//! a good number for GTX 1080ti.

#define CUDA_MAX_BLOCK_THREADS_NUM 512

//! \def CUDA_BLOCKS_NUM
//! \brief The number of blocks we need for the matrix.
//! \param cols is the number of columns of the whole matrix.
//! \return The number of blocks uesd.

#define CUDA_BLOCKS_NUM(cols)           ((indx)(cols / CUDA_THREADS_EACH_BLOCK(cols) + (cols % CUDA_THREADS_EACH_BLOCK(cols) == 0 ? 0 : 1)))

//! \def CUDA_THREADS_EACH_BLOCK
//! \brief The number of threads each thread.
//! \param cols is the number of columns of the whole matrix.
//! \return The number of threads.
//! \note We use square root to compute the possible result which can be tested to be a
//! good choice.
//! \warning For most of time, we reach the \ref CUDA_MAX_BLOCK_THREADS_NUM.

#define CUDA_THREADS_EACH_BLOCK(cols)   (((indx)(sqrt((double)cols))) > CUDA_MAX_BLOCK_THREADS_NUM ? \
                            CUDA_MAX_BLOCK_THREADS_NUM : ((indx)(sqrt((double)cols))))

typedef long indx;         //!< We use long to store all data or indices.
typedef short dimension;        //!< We use short to store dimensions which cannot be too large until now.
typedef unsigned short bits64;

//! \class column
//! \brief This is how we store the whole matrix by sparse matrix method.
//! \todo Can we have better idea like bit_pivot_tree in phat library to store columns?

struct column {
    indx * data;            //!< Data stores the number of row which is not empty.
    size_t data_length;     //!< Current length of data.
    size_t size;            //!< The size of memory allocated for column which can be extended or shrunk.
};

//! \brief Round a number up to the nearest power of 2
//! \author Sean Anderson
//! \param number is a number needing rounding up.
//! \return The the nearest power of 2.
//! \note See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2

__device__ indx round_up_to_2s(indx number);

__device__ int count_bit_sets(bits64 i);

__device__ indx most_significant_bit_pos(bits64 v);

__device__ indx least_significant_bit_pos(bits64 v);

//! \def gpuErrchk
//! \brief Check the return value by CUDA functions and print error code.
//! \param ans is a CUDA function.

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//! \brief Check whether the result is right.
//! \author talonmies
//! \param code is the return value by CUDA functions.
//! \param file is the file where printing the output.
//! \param line is the number of error line.
//! \param abort is whether abort the program here if something goes wrong
//! \note See https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// configurate the CreationPolicy "Scatter"
struct ScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
    typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new mMCator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator<
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>
> ScatterAllocator;

#endif //PHAT_CUDA_COMMON_H
