#pragma once
// The multiplicatively update parameters given the numerator and denominator.

#include <cstdint>

#include <cuda_runtime.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>


namespace npeff {
namespace gpu {
namespace ops {
namespace custom {


__global__
void MultiplicativeUpdate_Kernel(
    long n, float* F, const float* numer, const float* denom, float eps
) {
    // F *= numer / (denom + eps)
    INDEX_STRIDE_1D(n, i) {
        F[i] *= numer[i] / (denom[i] + eps);
    }
}

class MultiplicativeUpdate {
    DeviceContext& ctx;

    DenseMatrix& out;
    DenseMatrix& numer;
    DenseMatrix& denom;
    float eps;

    // TODO: Figure out how to set this.
    const int64_t block_size = 256;

public:
    MultiplicativeUpdate(
        DeviceContext& ctx,
        DenseMatrix& out,
        DenseMatrix& numer,
        DenseMatrix& denom,
        float eps
    ) : 
        ctx(ctx), out(out), numer(numer), denom(denom), eps(eps)
    {
        // Validation.
        THROW_IF_FALSE(numer.n_rows == out.n_rows);
        THROW_IF_FALSE(denom.n_rows == out.n_rows);
        THROW_IF_FALSE(numer.n_cols == out.n_cols);
        THROW_IF_FALSE(denom.n_cols == out.n_cols);
    }

    void call_async() {
        ctx.set_device();
        long n = out.n_entries;
        long n_blocks = (n + block_size - 1) / block_size;

        MultiplicativeUpdate_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            n, (float*) out.data, (float*) numer.data, (float*) denom.data, eps
        );
    }

};


}  // custom
}  // ops
}  // gpu
}  // npeff
