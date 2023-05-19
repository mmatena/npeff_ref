#pragma once
// Element-wise product of two matrices.


#include <cstdint>

#include <cuda_runtime.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>


namespace npeff {
namespace gpu {
namespace ops {
namespace custom {


__global__
void HadamardProduct_Kernel(int64_t n, float* left, float* right, float* out) {
    INDEX_STRIDE_1D(n, i) {
        out[i] = left[i] * right[i];
    }
}


class HadamardProduct {
    DeviceContext& ctx;

    DenseMatrix& left;
    DenseMatrix& right;
    DenseMatrix& out;

    const int64_t n_elements;

    // TODO: Figure out how to set this.
    const int64_t block_size = 256;

public:
    HadamardProduct(
        DeviceContext& ctx,
        DenseMatrix& left,
        DenseMatrix& right,
        DenseMatrix& out
    ) :
        ctx(ctx),
        left(left), right(right), out(out),
        n_elements(out.n_rows * out.n_cols)
    {
        THROW_IF_FALSE(left.n_rows == out.n_rows);
        THROW_IF_FALSE(right.n_rows == out.n_rows);
        THROW_IF_FALSE(left.n_cols == out.n_cols);
        THROW_IF_FALSE(right.n_cols == out.n_cols);

    }

    void call_async() {
        ctx.set_device();
        long n_blocks = (n_elements + block_size - 1) / block_size;

        HadamardProduct_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            n_elements, (float*) left.data, (float*) right.data, (float*) out.data
        );
    }

};


}  // custom
}  // ops
}  // gpu
}  // npeff
