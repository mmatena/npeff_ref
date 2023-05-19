#pragma once
// Elementwise-square of a matrix/array on the GPU.

#include <cstdint>

#include <cuda_runtime.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>


namespace npeff {
namespace gpu {
namespace ops {
namespace custom {


__global__
void ElwiseSquare_Kernel(int64_t n, float* in_data, float* out_data) {
    INDEX_STRIDE_1D(n, i) {
        out_data[i] = in_data[i] * in_data[i];
    }
}


class ElwiseSquare {
    DeviceContext& ctx;

    // Pointer to start of data on device.
    float const* in_data;
    float const* out_data;

    // Number of elements in the data.
    const int64_t n_elements;

    // TODO: Figure out how to set this.
    const int64_t block_size = 256;

public:
    ElwiseSquare(DeviceContext& ctx, float* in_data, float* out_data, int64_t n_elements) :
        ctx(ctx), in_data(in_data), out_data(out_data), n_elements(n_elements)
    {}

    ElwiseSquare(DeviceContext& ctx, DenseMatrix& in, DenseMatrix& out) :
        ctx(ctx), in_data(in.data), out_data(out.data), n_elements(in.n_entries)
    {
        THROW_IF_FALSE(in.n_entries == out.n_entries);
    }

    void call_async() {
        ctx.set_device();
        long n_blocks = (n_elements + block_size - 1) / block_size;

        ElwiseSquare_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            n_elements, (float*) in_data, (float*) out_data
        );
    }


};


}  // custom
}  // ops
}  // gpu
}  // npeff
