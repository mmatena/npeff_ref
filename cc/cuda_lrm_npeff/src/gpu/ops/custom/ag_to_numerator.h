#pragma once
// Converts the AG matrix into the numerator of the W-update step.
// This is equivalent to squaring its entries and then summing
// along the "classes" dimension.

#include <cstdint>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>

#include <gpu/macros.h>
#include <util/macros.h>

namespace npeff {
namespace gpu {
namespace ops {
namespace custom {


class AgToNumerator {
    DeviceContext& ctx;

    DenseMatrix& AG;
    DenseMatrix& W_update_numerator;

    int64_t n_classes;
    int64_t n_examples;

public:
    AgToNumerator(
        DeviceContext& ctx,
        DenseMatrix& AG,
        DenseMatrix& W_update_numerator,
        int64_t n_classes
    ) :
        ctx(ctx), AG(AG), W_update_numerator(W_update_numerator),
        n_classes(n_classes), n_examples(AG.n_rows / n_classes)
    {
        if((AG.n_rows % n_classes) != 0) {
            THROW_MSG("The number of classes must divide the number of rows.");
        }
    }

    void call_async() {
        ctx.set_device();

        // Treat this as the batched dot-product of n_examples * rank column
        // vectors, each of size n_classes.
        int64_t rank = AG.n_cols;
        int64_t batch_count = n_examples * rank;
        
        CUBLAS_CALL(cublasSgemmStridedBatched(
            ctx.dense_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            1, 1, n_classes,
            ctx.dev_1f,
            (float *) AG.data, n_classes, n_classes,
            (float *) AG.data, n_classes, n_classes,
            ctx.dev_0f,
            (float *) W_update_numerator.data, 1, 1,
            batch_count
        ));
    }

};


///////////////////////////////////////////////////////////////////////////////


__global__
void AgToNumeratorLvrm_Kernel(
    int64_t n_examples, int64_t n_rows, int64_t n_cols,
    float* d_AG, int64_t* d_example_row_offsets,
    float* d_W_update_numerator
) {
    int64_t example_index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t col_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (example_index < n_examples && col_index < n_cols) {
        int64_t start_row = d_example_row_offsets[example_index];
        int64_t end_row = d_example_row_offsets[example_index + 1];

        float out_entry = 0.0f;

        for(int64_t i = start_row; i < end_row; i++) {
            float value = d_AG[col_index * n_rows + i];
            out_entry += value * value;
        }

        d_W_update_numerator[col_index * n_examples + example_index] = out_entry;
    }
}



class AgToNumeratorLvrm {
    DeviceContext& ctx;

    // AG.shape = [n_rows, rank]
    DenseMatrix& AG;

    // W_update_numerator.shape = [n_examples, rank]
    DenseMatrix& W_update_numerator;

    int64_t* d_example_row_offsets;

    const int64_t n_examples;
    const int64_t n_rows;
    const int64_t n_cols;

    // TODO: Figure out how to set this.
    const int64_t block_size = 16;

public:
    AgToNumeratorLvrm(
        DeviceContext& ctx,
        DenseMatrix& AG,
        DenseMatrix& W_update_numerator,
        int64_t* d_example_row_offsets
    ) :
        ctx(ctx), AG(AG), W_update_numerator(W_update_numerator),
        d_example_row_offsets(d_example_row_offsets),
        n_examples(W_update_numerator.n_rows), n_rows(AG.n_rows), n_cols(AG.n_cols)
    {
        // Matrix shape checks.
        THROW_IF_FALSE(AG.n_cols == n_cols);
        THROW_IF_FALSE(W_update_numerator.n_cols == n_cols);
    }

    void call_async() {
        ctx.set_device();

        int64_t n_blocks_x = (n_examples + block_size - 1) / block_size;
        int64_t n_blocks_y = (n_cols + block_size - 1) / block_size;

        dim3 n_blocks(n_blocks_x, n_blocks_y);
        dim3 block_sizes(block_size, block_size);

        AgToNumeratorLvrm_Kernel<<<n_blocks, block_sizes, 0, ctx.stream>>>(
            n_examples, n_rows, n_cols,
            (float*) AG.data, d_example_row_offsets, (float*) W_update_numerator.data
        );
    }

};


}  // custom
}  // ops
}  // gpu
}  // npeff
