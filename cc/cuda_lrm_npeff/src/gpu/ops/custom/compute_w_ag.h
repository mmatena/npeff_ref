#pragma once
// Part of the G-update step.


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


class Compute_W_AG {
    DeviceContext& ctx;

    DenseMatrix& W;
    DenseMatrix& AG;
    DenseMatrix& W_AG;

    int64_t n_classes;
    int64_t n_examples;

public:
    Compute_W_AG(
        DeviceContext& ctx,
        DenseMatrix& W,
        DenseMatrix& AG,
        DenseMatrix& W_AG,
        int64_t n_classes
    ) :
        ctx(ctx), W(W), AG(AG), W_AG(W_AG),
        n_classes(n_classes), n_examples(AG.n_rows / n_classes)
    {
        if((AG.n_rows % n_classes) != 0) {
            THROW_MSG("The number of classes must divide the number of rows.");
        }
    }

    void call_async() {
        ctx.set_device();

        // Treat this as the batch multiplication of [n_classes, 1] x [1, 1]
        // matrices, with n_examples * rank batches in total
        int64_t rank = AG.n_cols;
        int64_t batch_count = n_examples * rank;
        
        CUBLAS_CALL(cublasSgemmStridedBatched(
            ctx.dense_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n_classes, 1, 1,
            ctx.dev_1f,
            (float *) AG.data, n_classes, n_classes,
            (float *) W.data, 1, 1,
            ctx.dev_0f,
            (float *) W_AG.data, n_classes, n_classes,
            batch_count
        ));
    }

};


///////////////////////////////////////////////////////////////////////////////


__global__
void Compute_W_AG_Lvrm_Kernel(
    int64_t n_examples, int64_t n_rows, int64_t n_cols,
    float* d_W, float* d_AG,
    int64_t* d_example_row_offsets,
    float* d_W_AG
) {
    int64_t example_index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t col_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (example_index < n_examples && col_index < n_cols) {
        float w = d_W[col_index * n_examples + example_index];

        int64_t start_row = d_example_row_offsets[example_index];
        int64_t end_row = d_example_row_offsets[example_index + 1];

        for(int64_t i = start_row; i < end_row; i++) {
            d_W_AG[col_index * n_rows + i] = w * d_AG[col_index * n_rows + i];
        }
    }
}


class Compute_W_AG_Lvrm {
    DeviceContext& ctx;

    // W.shape = [n_examples, rank]
    DenseMatrix& W;

    // AG.shape = [n_rows, rank]
    DenseMatrix& AG;

    // W_AG.shape = [n_rows, rank]
    DenseMatrix& W_AG;

    int64_t* d_example_row_offsets;

    const int64_t n_examples;
    const int64_t n_rows;
    const int64_t n_cols;


    // TODO: Figure out how to set this.
    const int64_t block_size = 16;

public:
    Compute_W_AG_Lvrm(
        DeviceContext& ctx,
        DenseMatrix& W,
        DenseMatrix& AG,
        DenseMatrix& W_AG,
        int64_t* d_example_row_offsets
    ) :
        ctx(ctx), W(W), AG(AG), W_AG(W_AG), d_example_row_offsets(d_example_row_offsets),
        n_examples(W.n_rows), n_rows(AG.n_rows), n_cols(AG.n_cols)
    {
        // Matrix shape checks.
        THROW_IF_FALSE(W.n_cols == n_cols);
        THROW_IF_FALSE(AG.n_cols == n_cols);
        THROW_IF_FALSE(W_AG.n_cols == n_cols);

        THROW_IF_FALSE(AG.n_rows == n_rows);
        THROW_IF_FALSE(W_AG.n_rows == n_rows);
    }

    void call_async() {
        ctx.set_device();

        int64_t n_blocks_x = (n_examples + block_size - 1) / block_size;
        int64_t n_blocks_y = (n_cols + block_size - 1) / block_size;

        dim3 n_blocks(n_blocks_x, n_blocks_y);
        dim3 block_sizes(block_size, block_size);

        Compute_W_AG_Lvrm_Kernel<<<n_blocks, block_sizes, 0, ctx.stream>>>(
            n_examples, n_rows, n_cols,
            (float*) W.data, (float*) AG.data,
            d_example_row_offsets,
            (float*) W_AG.data
        );
    }

};

}  // custom
}  // ops
}  // gpu
}  // npeff
