//
// Created by 唐艺峰 on 2018/11/5.
//

#ifndef MATRIX_REDUCTION_GPU_COMMON_H
#define MATRIX_REDUCTION_GPU_COMMON_H

#include <mallocMC.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <representations/vector_vector.h>
#include <boundary_matrix.h>

#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cuda.h"

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

#endif //MATRIX_REDUCTION_GPU_COMMON_H
