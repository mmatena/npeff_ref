#pragma once
// Gradient descent update step.

#include <cstdint>

#include <cuda_runtime.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>


namespace npeff {
namespace gpu {
namespace ops {
namespace custom {


__global__
void GradientDescentUpdate_Kernel(
    int64_t n, float* data, float* gradient, const float learning_rate, const float alpha = 0.0f
) {
    INDEX_STRIDE_1D(n, i) {
        // data[i] -= learning_rate * gradient[i];
        data[i] -= learning_rate * gradient[i] + learning_rate * alpha * data[i];
    }
}

class GradientDescentUpdate {
    DeviceContext& ctx;

    // Pointer to start of data on device.
    float const* data;
    float const* gradient;

    // Number of elements in the data.
    const int64_t n_elements;

    // Controls the size of the update step.
    const float learning_rate;

    // Subtracts learning * alpha * data[i] at each update step. Added to
    // efficiently support orthogonal regularization.
    const float alpha;

    // TODO: Figure out how to set this.
    const int64_t block_size = 256;

public:
    GradientDescentUpdate(
        DeviceContext& ctx, float* data, float* gradient, int64_t n_elements, float learning_rate, float alpha = 0.0f
    ) :
        ctx(ctx), data(data), gradient(gradient), n_elements(n_elements), learning_rate(learning_rate), alpha(alpha)
    {}

    GradientDescentUpdate(
        DeviceContext& ctx, DenseMatrix& params, DenseMatrix& gradient, float learning_rate, float alpha = 0.0f
    ) :
        ctx(ctx), data(params.data), gradient(gradient.data), n_elements(params.n_entries), learning_rate(learning_rate),
        alpha(alpha)
    {
        THROW_IF_FALSE(params.n_entries == gradient.n_entries);
    }

    void call_async() {
        ctx.set_device();
        long n_blocks = (n_elements + block_size - 1) / block_size;

        GradientDescentUpdate_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            n_elements, (float*) data, (float*) gradient, learning_rate, alpha
        );
    }


};


}  // custom
}  // ops
}  // gpu
}  // npeff
