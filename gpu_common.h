#ifndef _GPU_COMMON_H_
#define _GPU_COMMON_H_

#include <mallocMC.hpp>
#include <representations/vector_vector.h>
#include <boundary_matrix.h>

#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cuda.h"

#define CUDA_MAX_BLOCK_THREADS_NUM      1024
#define CUDA_THREADS_EACH_BLOCK(cols)   (((indx)(sqrt((double)cols))) > CUDA_MAX_BLOCK_THREADS_NUM ? \
                            CUDA_MAX_BLOCK_THREADS_NUM : ((indx)(sqrt((double)cols))))
//#define CUDA_THREADS_EACH_BLOCK(cols) CUDA_MAX_BLOCK_THREADS_NUM
#define CUDA_BLOCKS_NUM(cols) ((indx)(cols / CUDA_THREADS_EACH_BLOCK(cols) + (cols % CUDA_THREADS_EACH_BLOCK(cols) == 0 ? 0 : 1)))

typedef long indx;
//! \def gpuErrchk
//! \brief Check the return value by CUDA functions and print error code.
//! \param ans is a CUDA function.

__device__ indx round_up_to_2s(indx number);

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

#endif //MATRIX_REDUCTION_GPU_COMMON_H
