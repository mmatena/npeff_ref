#pragma once
// Multiplies a square matrix by a constant and adds the identity matrix.


#include <cstdint>

#include <cuda_runtime.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>


namespace npeff {
namespace gpu {
namespace ops {
namespace custom {



// This assumes that the matrix is in column-major format.
__global__
void MultiplyAndAddIdentity_InPlace_Kernel(const int64_t r, float* data, const float multiply_factor) {
    int64_t n = r * r;
    INDEX_STRIDE_1D(n, i) {
        int64_t row = i % r;
        int64_t col = i / r;
        if (row == col) {
            data[i] = multiply_factor * data[i] + 1;
        } else {
            data[i] *= multiply_factor;
        }
    }
}

class MultiplyAndAddIdentity_InPlace {
    DeviceContext& ctx;
    DenseMatrix& mat;

    const float multiply_factor;

    // TODO: Figure out how to set this.
    const int64_t block_size = 256;

public:
    MultiplyAndAddIdentity_InPlace(
        DeviceContext& ctx,
        DenseMatrix& mat,
        float multiply_factor
    ) : 
        ctx(ctx), mat(mat), multiply_factor(multiply_factor)
    {
        THROW_IF_FALSE(mat.n_rows == mat.n_cols);
    }

    void call_async() {
        ctx.set_device();

        long n_blocks = (mat.n_entries + block_size - 1) / block_size;

        MultiplyAndAddIdentity_InPlace_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            mat.n_rows, (float*) mat.data, multiply_factor
        );
    }
};


}  // custom
}  // ops
}  // gpu
}  // npeff
