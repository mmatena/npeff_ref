#pragma once

#include <cmath>
#include <memory>

#include <util/macros.h>
#include <containers/sparse_matrix.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>
#include <gpu/containers/sparse_matrix.h>
#include <gpu/containers/transfers.h>

#include <gpu/ops/dndn_matmul.h>
#include <gpu/ops/frobenius_product.h>
#include <gpu/ops/spdn_matmul.h>
#include <gpu/ops/custom/ag_to_numerator.h>
#include <gpu/ops/custom/compute_w_ag.h>
#include <gpu/ops/custom/elwise_square.h>
#include <gpu/ops/custom/gradient_descent.h>
#include <gpu/ops/custom/hadamard_product.h>
#include <gpu/ops/custom/multiplicative_update.h>

#include "./config.h"

namespace npeff {
namespace factorization {


// Utility/helper stuff.
namespace internal {

// Struct to just have a place to put all of the device memory
// allocation pointers in one place.
template<typename IndT>
struct DeviceAllocPtrs {
    // Scalars.
    float* minus_1 = nullptr;
    float* ortho_reg_str = nullptr;

    float* d_tr_WW_HH = nullptr;
    float* d_tr_WHX = nullptr;

    // The input matrix.
    float* d_A_values = nullptr;
    IndT* d_A_row_offsets = nullptr;
    IndT* d_A_col_indices = nullptr;

    // Pointers to parameters to learn.
    float* d_W = nullptr;
    float* d_G = nullptr;

    // Allocations for holding intermediates:
    // 
    // Information about their sizes are included in their names.
    //      n = number of examples
    //      r = rank of decomposition
    //      c = number of classes
    //      m = number of parameters within this partition

    // Size = n * c * r.
    float* d_ncr1 = nullptr;
    float* d_ncr2 = nullptr;

    // Size = n * r.
    float* d_nr = nullptr;

    // All have sizes of r * r.
    float* d_rr1 = nullptr;
    float* d_rr2 = nullptr;
    float* d_rr3 = nullptr;

    // Size = r * m
    float* d_rm = nullptr;
};


// If a matrix gets modified in-place without any changes in
// its shape, I might not create a separate matrix entry here.
// 
// Note that the actual device memory chunks associates to each
// matrix can overlap. Furthermore, it will be impossible to 
// use some combinations of these matrices at the same time due
// to this.
template<typename IndT>
struct DeviceMatrices {
    using DenseMatrixPtr = std::unique_ptr<gpu::DenseMatrix>;
    using CsrMatrixPtr = std::unique_ptr<gpu::CsrMatrix<IndT>>;

    CsrMatrixPtr A;

    DenseMatrixPtr W;
    DenseMatrixPtr G;

    // Simple/common intermediates.
    DenseMatrixPtr WW;

    DenseMatrixPtr GG;
    DenseMatrixPtr HH;

    DenseMatrixPtr AG;

    // W-step specific intermediates.
    DenseMatrixPtr W_update_numerator;
    DenseMatrixPtr W_update_denominator;

    // G-step specific intermediates.
    DenseMatrixPtr WW_GG;
    DenseMatrixPtr W_AG;
    DenseMatrixPtr G_gradient;

};


template<typename IndT>
struct StatefulFactorizationOps {
    using SpDnMatMulPtr = std::unique_ptr<gpu::ops::SpDnMatMul<IndT>>;
    using DnSpMatMulPtr = std::unique_ptr<gpu::ops::DnSpMatMul<IndT>>;

    SpDnMatMulPtr matmul_AG;
    DnSpMatMulPtr matmul_W_AG_A;
};



}  // internal


// Worker associated to a single GPU.
template<typename IndT>
class DeviceWorker {
    using DnDnMatMul = gpu::ops::DnDnMatMul;
    using FrobeniousInnerProduct = gpu::ops::FrobeniousInnerProduct;

    FactorizationConfig config;
    int64_t partition_index;
    int64_t n_partitions;

    gpu::DeviceContext dctx;

    // Holds pointers to the device memory allocations. Simply
    // put here for convenience.
    internal::DeviceAllocPtrs<IndT> d_ptrs;

    // Holds the device matrix objects.
    internal::DeviceMatrices<IndT> d_ms;

    // Holds operations that require some state to be created before
    // being used.
    internal::StatefulFactorizationOps<IndT> s_ops;

    // NOTE: This will become a null pointer and the
    // associated matrix deleted once its data has
    // been moved to the GPU.
    std::unique_ptr<npeff::CsrMatrix<IndT>> host_matrix_partition;

public:
    DeviceWorker(
        std::unique_ptr<npeff::CsrMatrix<IndT>> host_matrix_partition,
        FactorizationConfig config,
        int64_t partition_index,
        int64_t n_partitions,
        ncclComm_t comm
    ) : 
        host_matrix_partition(std::move(host_matrix_partition)),
        config(config),
        partition_index(partition_index),
        n_partitions(n_partitions),
        dctx(gpu::DeviceContext(partition_index, comm, config.rand_gen_seed))
    {}

    int64_t get_device() {
        return dctx.device;
    }

    float* get_W_data_ptr() {
        return (float*) d_ms.W->data;
    }

    void synchronize_stream() {
        dctx.synchronize_stream();
    }

    // Must be called once before doing anything else. This allocates memory
    // on the device and moves data to the GPU. Parameters will be randomly
    // initialized.
    // 
    // NOTE: The W matrices (and any other parameters shared across multiple
    // devices) should/must be made consistent AFTER this is called.
    void set_up_work() {
        dctx.initialize();

        write_scalars_to_gpu();

        allocate_and_create_device_matrices();
        initialize_device_matrices();

        construct_and_initialize_stateful_ops();
    }

    void nccl_broadcast_of_W(DeviceWorker<IndT>& src_worker) {
        NCCL_CALL(
            ncclBroadcast(
                src_worker.get_W_data_ptr(),
                get_W_data_ptr(),
                d_ms.W->n_entries,
                ncclFloat,
                src_worker.get_device(),
                dctx.comm,
                dctx.stream
            )
        );
    }

    /////////////////////////////////////////////////////////////////
    // General stuff.

    void compute_local_AG_GG_async() {
        s_ops.matmul_AG->call_async();
        DnDnMatMul(dctx, *d_ms.G, *d_ms.G, *d_ms.GG, false, true).call_async();
    }

    void nccl_all_reduce_AG_GG() {
        NCCL_CALL(
            ncclAllReduce(
                (float*) d_ms.AG->data,
                (float*) d_ms.AG->data,
                d_ms.AG->n_entries,
                ncclFloat,
                ncclSum,
                dctx.comm,
                dctx.stream)
        );
        NCCL_CALL(
            ncclAllReduce(
                (float*) d_ms.GG->data,
                (float*) d_ms.GG->data,
                d_ms.GG->n_entries,
                ncclFloat,
                ncclSum,
                dctx.comm,
                dctx.stream)
        );
    }

    /////////////////////////////////////////////////////////////////
    // W-update stuff.

    void update_local_W_after_all_reduces_async() {
        // Must called be after the AG and GG matrices have been all-reduced
        // across all devices (or slated to do so in the streams).

        // Compute the numerator.
        gpu::ops::custom::AgToNumerator(
            dctx, *d_ms.AG, *d_ms.W_update_numerator, config.n_classes)
            .call_async();

        // Square GG to get HH.
        gpu::ops::custom::ElwiseSquare(dctx, *d_ms.GG, *d_ms.HH).call_async();
        // Compute W(HH) to get the denominator.
        DnDnMatMul(dctx, *d_ms.W, *d_ms.HH, *d_ms.W_update_denominator, false, false)
            .call_async();

        // Update the local copy of W.
        gpu::ops::custom::MultiplicativeUpdate(
            dctx, *d_ms.W, *d_ms.W_update_numerator, *d_ms.W_update_denominator, config.mu_eps)
            .call_async();
    }

    /////////////////////////////////////////////////////////////////
    // G-update stuff.

    void update_local_G_after_all_reduces_async(float learning_rate_G) {
        // Compute the first term and write it to the buffer storing the gradient.
        DnDnMatMul(dctx, *d_ms.W, *d_ms.W, *d_ms.WW, true, false).call_async();
        gpu::ops::custom::HadamardProduct(
            dctx, *d_ms.WW, *d_ms.GG, *d_ms.WW_GG)
            .call_async();
        DnDnMatMul(dctx, *d_ms.WW_GG, *d_ms.G, *d_ms.G_gradient, false, false).call_async();

        // Compute the second term and accumulate it onto the gradient buffer.
        gpu::ops::custom::Compute_W_AG(
            dctx, *d_ms.W, *d_ms.AG, *d_ms.W_AG, config.n_classes)
            .call_async();
        s_ops.matmul_W_AG_A->call_async();


        // If we have orthogonal regularization, then add the contribution of
        // the GG^TG term to the gradient. The other term will be incorporated
        // into the gradient descent update step via the gd_alpha argument.
        float gd_alpha = 0.0f;
        if (config.has_orthogonal_regularization()) {
            double target_scale = config.compute_orthogonal_regularization_target_scale();
            gd_alpha = -config.ortho_reg_config.regularization_strength * target_scale;

            DnDnMatMul(
                dctx, *d_ms.GG, *d_ms.G, *d_ms.G_gradient,
                false, false,
                d_ptrs.ortho_reg_str, dctx.dev_1f
            ).call_async();
        }

        // Update the parameters G given the gradient.
        // 
        // The factor of 4 comes from the gradient being multiplied by that
        // but not accounted for in our computation of it.
        gpu::ops::custom::GradientDescentUpdate(
            dctx, *d_ms.G, *d_ms.G_gradient, 4.0f * learning_rate_G, gd_alpha)
            .call_async();
    }

    /////////////////////////////////////////////////////////////////
    // Loss computation stuff.
    // 
    // TODO: Right now, we are only computing the reconstruction loss. Things
    // like regularization losses are not included. Ideally, we'd compute and
    // saved each of them separately.

    void compute_loss_after_all_reduces_async() {
        // Compute tr_WW_HH.
        gpu::ops::custom::ElwiseSquare(dctx, *d_ms.GG, *d_ms.HH).call_async();
        DnDnMatMul(dctx, *d_ms.W, *d_ms.W, *d_ms.WW, true, false).call_async();
        FrobeniousInnerProduct(dctx, *d_ms.WW, *d_ms.HH, d_ptrs.d_tr_WW_HH)
            .call_async();

        // Compute tr_WHX.
        gpu::ops::custom::AgToNumerator(
            dctx, *d_ms.AG, *d_ms.W_update_numerator, config.n_classes)
            .call_async();
        FrobeniousInnerProduct(dctx, *d_ms.W, *d_ms.W_update_numerator, d_ptrs.d_tr_WHX)
            .call_async();
    }

    // NOTE: The actual loss will be a fixed constant plus what this
    // function returns.
    float read_loss_term_from_device() {
        dctx.set_device();
        float tr_WW_HH, tr_WHX; 
        dctx.copy_to_host_async<float>(&tr_WW_HH, d_ptrs.d_tr_WW_HH, 1);
        dctx.copy_to_host_async<float>(&tr_WHX, d_ptrs.d_tr_WHX, 1);
        dctx.synchronize_stream();
        return -2.0f * tr_WHX + tr_WW_HH;
    }

    /////////////////////////////////////////////////////////////////
    // Other stuff.

    // The host_write_location must be on the host.
    void read_W_from_gpu_async(float* host_write_location) {
        read_matrix_from_gpu_async(*d_ms.W, host_write_location);
    }

    // The host_write_location must be on the host.
    void read_G_from_gpu_async(float* host_write_location) {
        read_matrix_from_gpu_async(*d_ms.G, host_write_location);
    }


protected:

    void write_scalars_to_gpu() {
        // Allocate memory for the scalars representing intermediate quantities.
        d_ptrs.d_tr_WW_HH = dctx.dmalloc<float>(2);
        d_ptrs.d_tr_WHX = d_ptrs.d_tr_WW_HH + 1;

        // Allocate memory for and write the constant scalars to the device.
        const int64_t n_scalars = 2;
        float* d_scalars = dctx.dmalloc<float>(n_scalars);
        d_ptrs.minus_1 = d_scalars + 0;
        d_ptrs.ortho_reg_str = d_scalars + 1;

        float scalars[n_scalars] = {
            -1.0f,
            config.ortho_reg_config.regularization_strength
        };
        dctx.copy_to_device_async(d_scalars, scalars, n_scalars);
    }

    void allocate_and_create_device_matrices() {
        int64_t n_rows = host_matrix_partition->n_rows;
        int64_t n_cols = host_matrix_partition->n_cols;
        int64_t nnz = host_matrix_partition->nnz;

        int64_t n_classes = config.n_classes;
        int64_t rank = config.rank;

        int64_t n_examples = n_rows / n_classes;

        /////////////////////////////////////////
        // Allocate the memory on the device.

        d_ptrs.d_A_values = dctx.dmalloc<float>(nnz);
        d_ptrs.d_A_row_offsets = dctx.dmalloc<IndT>(n_rows + 1);
        d_ptrs.d_A_col_indices = dctx.dmalloc<IndT>(nnz);

        d_ptrs.d_W = dctx.dmalloc<float>(n_examples * rank);
        d_ptrs.d_G = dctx.dmalloc<float>(rank * n_cols);

        d_ptrs.d_ncr1 = dctx.dmalloc<float>(n_classes * n_examples * rank);
        d_ptrs.d_ncr2 = dctx.dmalloc<float>(n_classes * n_examples * rank);
        d_ptrs.d_nr = dctx.dmalloc<float>(n_examples * rank);
        d_ptrs.d_rr1 = dctx.dmalloc<float>(rank * rank);
        d_ptrs.d_rr2 = dctx.dmalloc<float>(rank * rank);
        d_ptrs.d_rr3 = dctx.dmalloc<float>(rank * rank);
        d_ptrs.d_rm = dctx.dmalloc<float>(rank * n_cols);

        /////////////////////////////////////////
        // Create the matrices.

        // Input/parameter matrices.
        d_ms.A = gpu::CsrMatrix<IndT>::make_unique_ptr(
            n_rows, n_cols, nnz,
            d_ptrs.d_A_values, d_ptrs.d_A_row_offsets, d_ptrs.d_A_col_indices);

        d_ms.W = gpu::DenseMatrix::make_unique_ptr(n_examples, rank, d_ptrs.d_W);
        d_ms.G = gpu::DenseMatrix::make_unique_ptr(rank, n_cols, d_ptrs.d_G);

        // Simple/common intermediate matrices.
        d_ms.WW = gpu::DenseMatrix::make_unique_ptr(rank, rank, d_ptrs.d_rr1);

        d_ms.GG = gpu::DenseMatrix::make_unique_ptr(rank, rank, d_ptrs.d_rr2);
        d_ms.HH = gpu::DenseMatrix::make_unique_ptr(rank, rank, d_ptrs.d_rr3);

        d_ms.AG = gpu::DenseMatrix::make_unique_ptr(n_classes * n_examples, rank, d_ptrs.d_ncr1);

        // W-step specific intermediate matrices.
        d_ms.W_update_numerator = gpu::DenseMatrix::make_unique_ptr(n_examples, rank, d_ptrs.d_nr);
        d_ms.W_update_denominator = gpu::DenseMatrix::make_unique_ptr(n_examples, rank, d_ptrs.d_ncr2);

        // G-step specific intermediate matrices.
        d_ms.WW_GG = gpu::DenseMatrix::make_unique_ptr(rank, rank, d_ptrs.d_rr1);
        d_ms.W_AG = gpu::DenseMatrix::make_unique_ptr(n_classes * n_examples, rank, d_ptrs.d_ncr2);
        d_ms.G_gradient = gpu::DenseMatrix::make_unique_ptr(rank, n_cols, d_ptrs.d_rm);
    }

    void initialize_device_matrices() {
        dctx.set_device();

        // Move A onto the GPU.
        dctx.copy_to_device_async(*d_ms.A, *host_matrix_partition);

        // If we are the first partition, initialize W with a uniform random
        // distribution.
        if(partition_index == 0) {
            auto& W = d_ms.W;
            CURAND_CALL(
                curandGenerateUniform(dctx.rand_gen, (float*) W->data, W->n_rows * W->n_cols)
            );
        }

        auto& G = d_ms.G;
        // double inv_g_factor = std::sqrt((double) (n_partitions * G->n_rows * G->n_cols) / 2.0);
        double inv_g_factor = config.compute_inv_g_initialization_scale_factor();
        CURAND_CALL(
            curandGenerateNormal(dctx.rand_gen, (float*) G->data, G->n_rows * G->n_cols, 0.0f, 1.0 / inv_g_factor)
        );
    
        // Synchronize the stream to assure that everything associated to A
        // has been copied onto the GPU. Then free the memory assoicated to
        // A on the host.
        dctx.synchronize_stream();
        host_matrix_partition.reset();
    }

    void construct_and_initialize_stateful_ops() {
        // Create the ops.
        s_ops.matmul_AG = std::unique_ptr<gpu::ops::SpDnMatMul<IndT>>(
            new gpu::ops::SpDnMatMul<IndT>(
                dctx,
                *d_ms.A, *d_ms.G, *d_ms.AG,
                false, true)
        );
        s_ops.matmul_W_AG_A = std::unique_ptr<gpu::ops::DnSpMatMul<IndT>>(
            new gpu::ops::DnSpMatMul<IndT>(
                dctx,
                *d_ms.W_AG, *d_ms.A, *d_ms.G_gradient,
                true, false,
                CUSPARSE_SPMM_ALG_DEFAULT,
                d_ptrs.minus_1, dctx.dev_1f)
        );

        // Set up the ops.
        s_ops.matmul_AG->set_up_async();
        s_ops.matmul_W_AG_A->set_up_async();
    }

    // The host_write_location must be on the host.
    void read_matrix_from_gpu_async(gpu::DenseMatrix& matrix, float* host_write_location) {
        dctx.set_device();
        dctx.copy_to_host_async(host_write_location, (float*) matrix.data, matrix.n_entries);
    }

};


}  // factorization
}  // npeff
