#pragma once


#include <cuda_runtime.h>
#include <curand.h>

#include <misc/common.h>
#include <cuda/cuda_context.h>
#include <cuda/device/dense_matrix.h>


namespace Cuda {
namespace Ops {


class RandomGenerator {
    using DenseMatrix = Device::DenseMatrix;

public:
    // RandomGenerator
    DeviceCudaContext& ctx;

    RandomGenerator(DeviceCudaContext& ctx) : ctx(ctx) {}

    /////////////////////////////////////////////////////////////////
    // Uniform distribution.

    void GenerateUniformAsync(float* dev_ptr, size_t n) {
        ctx.SetDevice();
        CURAND_CALL(
            curandGenerateUniform(ctx.randGen, dev_ptr, n)
        );
    }

    void InitializeUniformAsync(DenseMatrix& mat) {
        GenerateUniformAsync(mat.data, mat.n_entries);
    }

};



} // Ops
} // Cuda

