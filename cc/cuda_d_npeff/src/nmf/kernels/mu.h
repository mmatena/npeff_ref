#pragma once
/** Multiplicative update kernel. */

#include <cuda_runtime.h>

#include <misc/macros.h>

#include <cuda/cuda_statuses.h>
#include <cuda/device/dense_matrix.h>


// #define 


namespace Nmf {

namespace Kernels {


// __global__
// void MultiplicativeUpdate_Kernel(long n, float* F, const float* numer, const float* denom, float eps) {
//     // F *= numer / (denom + eps)
//     // n is equal to the number of entries of F.
//     long index = blockIdx.x * blockDim.x + threadIdx.x;
//     long stride = blockDim.x * gridDim.x;

//     for (long i = index; i < n; i += stride) {
//         F[i] *= numer[i] / (denom[i] + eps);
//     }
// }

__global__
void MultiplicativeUpdate_Kernel(long n, float* F, const float* numer, const float* denom, float eps) {
    // F *= numer / (denom + eps)
    // n is equal to the number of entries of F.
    INDEX_STRIDE_1D(n, i) {
        F[i] *= numer[i] / (denom[i] + eps);
    }
}

} // Kernels


namespace Ops {


class MultiplicativeUpdate {
    using DenseMatrix = Cuda::Device::DenseMatrix;

protected:
    DeviceCudaContext& ctx;
    DenseMatrix& out;
    DenseMatrix& numer;
    DenseMatrix& denom;
    float eps;

    // TODO: Figure out how to set this.
    long block_size = 256;

public:
    MultiplicativeUpdate(
        DeviceCudaContext& ctx,
        DenseMatrix& out,
        DenseMatrix& numer,
        DenseMatrix& denom,
        float eps
    ) : 
        ctx(ctx), out(out), numer(numer), denom(denom), eps(eps)
    {
        // Validation.
        THROWSERT(numer.n_rows == out.n_rows);
        THROWSERT(denom.n_rows == out.n_rows);
        THROWSERT(numer.n_cols == out.n_cols);
        THROWSERT(denom.n_cols == out.n_cols);
    }

    void CallAsync() {
        ctx.SetDevice();
        long n = out.n_entries;
        long n_blocks = (n + block_size - 1) / block_size;

        Kernels::MultiplicativeUpdate_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            n, out.data, numer.data, denom.data, eps
        );
    }

};


} // Ops


} // Nmf