#pragma once
// The overall context of the fitting process.

#include <cstdint>
#include <memory>

#include "./config.h"
#include "./host_context.h"

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>
#include <gpu/containers/sparse_matrix.h>
#include <gpu/containers/transfers.h>

#include <gpu/ops/dndn_matmul.h>
#include <gpu/ops/spdn_matmul.h>
#include <gpu/ops/custom/ag_to_numerator.h>
#include <gpu/ops/custom/elwise_square.h>
#include <gpu/ops/custom/multiplicative_update.h>


namespace npeff {
namespace coeff_fitting {


// Utility/helper stuff.
namespace internal_ {

struct DeviceAllocPtrs {

    ////////////////////////////////////////////////////////////
    // Holders of chunks of matrices used in the pre-iterative stages.

    // Pointer to hold column-wise chunk of G as it gets
    // read in and used.
    float* d_G_chunk = nullptr;

    // These hold column-wise and example-wise chunks of
    // the input pefs matrix as they get read in.
    float *d_A_value_chunk = nullptr;
    int32_t* d_A_row_offsets_chunk = nullptr;
    int32_t* d_A_col_indices_chunk = nullptr;

    ////////////////////////////////////////////////////////////
    // Memory that changes only after each examples chunk.

    // Memory used to store the coefficients for each examples chunk.
    float* d_W = nullptr;

    // Memory of the AG matrix that accumulates over column chunks in a given
    // examples chunk.
    float* d_AG = nullptr;

    // The numerator/denominator the mulitplicative update in for a given examples chunk.
    float* d_numerator = nullptr;
    float* d_denominator = nullptr;


    ////////////////////////////////////////////////////////////
    // Memory that should not change after the first pre-iterative stage.

    // Hold the matrix HH^T along with the accumlator for
    // GG^T before the in-place element-wise square turns
    // it into HH^T.
    float* d_HH = nullptr;


    // TODO: More
};

// struct DeviceMatrices {};

}  // internal_


class FittingContext {
    using DenseMatrixPtr = std::unique_ptr<gpu::DenseMatrix>;
    using CsrMatrixPtr = std::unique_ptr<gpu::CsrMatrix<int32_t>>;

    CoeffFittingConfig config;

    gpu::DeviceContext dctx;

    // Holds pointers to the device memory allocations. Simply
    // put here for convenience.
    internal_::DeviceAllocPtrs d_ptrs;

    DenseMatrixPtr HH;

public:
    std::unique_ptr<HostContext> host_ctx;
    
    FittingContext(
        std::unique_ptr<HostContext> host_ctx
    ) :
        config(host_ctx->config),
        host_ctx(std::move(host_ctx)),
        dctx(gpu::DeviceContext(0, config.rand_gen_seed))
    {}

    // Initializes contexts, allocates device memory, and computes HH^T.
    void set_up_work() {
        dctx.initialize();
        create_device_allocs();
        compute_HH_async();
        dctx.synchronize_stream();
    }

    // Must be called after set_up_work has been called.
    // 
    // The returned matrix is in row-major format.
    std::unique_ptr<npeff::DenseMatrix<float>> compute_W_row_major() {
        auto* row_major_W = new npeff::DenseMatrix<float>(config.n_examples, config.rank);

        for(int64_t i=0; i<config.n_example_chunks(); i++) {
            // std::cout << "Starting example chunk " << i << "\n";
            auto W_chunk = compute_W_for_chunk(i);
            float* write_location = row_major_W->data.get() + i * config.n_examples_per_chunk * config.rank;
            W_chunk->convert_to_row_major_onto_buffer(write_location);
        }

        return std::unique_ptr<npeff::DenseMatrix<float>>(row_major_W);
    }

protected:

    void create_device_allocs() {
        int64_t n_examples_per_chunk = config.n_examples_per_chunk;
        int64_t n_columns_per_chunk = config.n_columns_per_chunk;

        int64_t n_classes = config.n_classes;
        int64_t rank = config.rank;

        int64_t max_nnz_per_chunk = n_examples_per_chunk * config.max_nnz_per_example;
        int64_t n_rows_per_chunk = n_examples_per_chunk * n_classes;

        d_ptrs.d_G_chunk = dctx.dmalloc<float>(rank * n_columns_per_chunk);

        d_ptrs.d_A_value_chunk = dctx.dmalloc<float>(max_nnz_per_chunk);
        d_ptrs.d_A_row_offsets_chunk = dctx.dmalloc<int32_t>(n_rows_per_chunk + 1);
        d_ptrs.d_A_col_indices_chunk = dctx.dmalloc<int32_t>(max_nnz_per_chunk);

        d_ptrs.d_W = dctx.dmalloc<float>(n_examples_per_chunk * rank);

        d_ptrs.d_AG = dctx.dmalloc<float>(n_rows_per_chunk * rank);

        d_ptrs.d_numerator = dctx.dmalloc<float>(n_examples_per_chunk * rank);
        d_ptrs.d_denominator = dctx.dmalloc<float>(n_examples_per_chunk * rank);

        d_ptrs.d_HH = dctx.dmalloc<float>(rank * rank);

        HH = DenseMatrixPtr(new gpu::DenseMatrix(rank, rank, d_ptrs.d_HH));
    }

    ////////////////////////////////////////////////////////////
    // Common pre-iterative methods.

    // Returns the unique_ptr to the device matrix of the loaded chunk of G.
    DenseMatrixPtr load_G_chunk_onto_device_async(int64_t col_chunk_index) {
        int64_t n_rows = config.rank;
        int64_t n_cols = host_ctx->get_n_cols_in_chunk(col_chunk_index);
        dctx.copy_to_device_async(d_ptrs.d_G_chunk,
                                  host_ctx->get_G_chunk_start_ptr(col_chunk_index),
                                  n_rows * n_cols);
        return DenseMatrixPtr(new gpu::DenseMatrix(n_rows, n_cols, d_ptrs.d_G_chunk));
    }

    ////////////////////////////////////////////////////////////
    // Computing HH.

    void compute_HH_async() {
        // Compute GG^T chunk by chunk.
        for(int64_t i=0; i<config.n_column_chunks(); i++) {
            auto G_chunk = load_G_chunk_onto_device_async(i);
            gpu::ops::DnDnMatMul(
                dctx, *G_chunk, *G_chunk, *HH,
                false, true,
                dctx.dev_1f,
                i == 0 ? dctx.dev_0f : dctx.dev_1f
            ).call_async();
        }
        // In-place element-wise square of GG^T to get HH^T.
        gpu::ops::custom::ElwiseSquare(dctx, *HH, *HH).call_async();
    }

    ////////////////////////////////////////////////////////////
    // Computing AG/numerator.

    CsrMatrixPtr load_A_chunk_onto_device_async(HostPefsChunk& pefs_chunk, int64_t col_chunk_index) {
        auto& host_matrix_ptr = pefs_chunk.A_partitions[col_chunk_index];

        auto* device_matrix_ptr = new gpu::CsrMatrix<int32_t>(
            host_matrix_ptr->n_rows, host_matrix_ptr->n_cols, host_matrix_ptr->nnz,
            d_ptrs.d_A_value_chunk,
            d_ptrs.d_A_row_offsets_chunk,
            d_ptrs.d_A_col_indices_chunk
        );

        dctx.copy_to_device_async(*device_matrix_ptr, *host_matrix_ptr);

        return CsrMatrixPtr(device_matrix_ptr);
    }

    DenseMatrixPtr compute_numerator_for_chunk(int64_t example_chunk_index) {
        int64_t n_examples_in_chunk = host_ctx->get_n_examples_in_chunk(example_chunk_index);
        HostPefsChunk pefs_chunk = host_ctx->load_partitioned_A_chunk(example_chunk_index);

        // Compute AG.
        gpu::DenseMatrix AG(
            config.n_classes * n_examples_in_chunk,
            config.rank,
            d_ptrs.d_AG);
        for(int64_t i=0; i<config.n_column_chunks(); i++) {
            // std::cout << "Computing numerator for column chunk " << i << "\n";
            auto G_chunk = load_G_chunk_onto_device_async(i);
            auto A_chunk = load_A_chunk_onto_device_async(pefs_chunk, i);

            gpu::ops::SpDnMatMul<int32_t> matmul(
                dctx,
                *A_chunk, *G_chunk, AG,
                false, true,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dctx.dev_1f,
                i == 0 ? dctx.dev_0f : dctx.dev_1f
            );
            matmul.set_up_async();
            matmul.call_async();

            // Synchronize stream so that the buffer allocated by the matmul does
            // not get deallocated before the multiplication has completed.
            dctx.synchronize_stream();
        }

        // Compute the numerator.
        auto* numerator = new gpu::DenseMatrix(
            n_examples_in_chunk, config.rank, d_ptrs.d_numerator);
        gpu::ops::custom::AgToNumerator(
            dctx, AG, *numerator, config.n_classes)
            .call_async();

        // Synchronize the stream to make sure that all of the A chunks have been transferred
        // to the device before they get deallocated on the CPU.
        dctx.synchronize_stream();

        return DenseMatrixPtr(numerator);
    }

    ////////////////////////////////////////////////////////////
    // Multiplicative update steps for a given example chunk.

    void multiplicative_update_step_async(gpu::DenseMatrix& W, gpu::DenseMatrix& numerator) {
        // Compute the denominator.
        gpu::DenseMatrix denominator(numerator.n_rows, numerator.n_cols, d_ptrs.d_denominator);
        gpu::ops::DnDnMatMul(
            dctx, W, *HH, denominator,
            false, false
        ).call_async();

        // Multiplicative update to W.
        gpu::ops::custom::MultiplicativeUpdate(
            dctx, W, numerator, denominator, config.mu_eps)
            .call_async();
    }

    DenseMatrixPtr multiplicative_update_stage(gpu::DenseMatrix& numerator) {
        auto* W = new gpu::DenseMatrix(
            numerator.n_rows, numerator.n_cols, d_ptrs.d_W);

        // Initialize the coefficients matrix W.
        CURAND_CALL(
            curandGenerateUniform(dctx.rand_gen, (float*) W->data, W->n_rows * W->n_cols)
        );

        // Perform the multiplicative update step repeatedly.
        for(int64_t i=0; i<config.n_iters; i++) {
            multiplicative_update_step_async(*W, numerator);
        }

        dctx.synchronize_stream();

        return DenseMatrixPtr(W);
    }

    ////////////////////////////////////////////////////////////
    // Higher level steps/functions.

    // Note that the returned W will be in column-major format.
    std::unique_ptr<npeff::DenseMatrix<float>> compute_W_for_chunk(int64_t example_chunk_index) {
        // Compute W on the device.
        auto numerator = compute_numerator_for_chunk(example_chunk_index);
        auto d_W = multiplicative_update_stage(*numerator);
        dctx.synchronize_stream();

        // Copy W onto the host.
        auto W = std::unique_ptr<DenseMatrix<float>>(
            new DenseMatrix<float>(d_W->n_rows, d_W->n_cols));
        dctx.copy_to_host_async(W->data.get(), (float*) d_W->data, W->n_entries);
        dctx.synchronize_stream();

        return W;
    }

};



}  // coeff_fitting
}  // npeff
