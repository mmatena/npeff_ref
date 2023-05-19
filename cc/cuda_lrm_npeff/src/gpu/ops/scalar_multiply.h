#pragma once
// Multiply a matrix by a scalar.


#include <cstdint>

#include <cuda_runtime.h>

#include <gpu/macros.h>
#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>

#include <util/macros.h>

namespace npeff {
namespace gpu {
namespace ops {


__global__
void ScalarMultiply_InPlace_Kernel(int64_t n, float* in_data, float scalar) {
    INDEX_STRIDE_1D(n, i) {
        in_data[i] *= scalar;
    }
}

class ScalarMultiply_InPlace {
    DeviceContext& ctx;
    DenseMatrix& mat;
    const float scalar;

    // TODO: Figure out how to set this.
    const int64_t block_size = 256;

public:
    ScalarMultiply_InPlace(
        DeviceContext& ctx, DenseMatrix& mat, float scalar
    ) : 
        ctx(ctx), mat(mat), scalar(scalar)
    {}

    void call_async() {
        ctx.set_device();
        long n_blocks = (mat.n_entries + block_size - 1) / block_size;
        ScalarMultiply_InPlace_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            mat.n_entries, (float*) mat.data, scalar
        );
    }
};

}  // ops
}  // gpu
}  // npeff

