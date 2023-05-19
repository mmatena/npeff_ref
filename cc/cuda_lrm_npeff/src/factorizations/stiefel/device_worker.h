#pragma once

#include <cmath>
#include <memory>

#include <util/macros.h>
#include <containers/sparse_matrix.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>
#include <gpu/containers/sparse_matrix.h>
#include <gpu/containers/transfers.h>

#include <gpu/ops/copy_submatrices.h>
#include <gpu/ops/dndn_matmul.h>
#include <gpu/ops/frobenius_product.h>
#include <gpu/ops/orthonormalize.h>
#include <gpu/ops/scalar_multiply.h>
#include <gpu/ops/solve_linear_system.h>
#include <gpu/ops/spdn_matmul.h>
#include <gpu/ops/transpose.h>
#include <gpu/ops/custom/ag_to_numerator.h>
#include <gpu/ops/custom/compute_w_ag.h>
#include <gpu/ops/custom/elwise_square.h>
#include <gpu/ops/custom/gradient_descent.h>
#include <gpu/ops/custom/hadamard_product.h>
#include <gpu/ops/custom/multiplicative_update.h>
#include <gpu/ops/custom/multiply_and_add_identity.h>

#include "./config.h"


namespace npeff {
namespace factorizations {
namespace stiefel {



// Utility/helper stuff.
namespace internal_ {

// Struct to just have a place to put all of the device memory
// allocation pointers in one place.
template<typename IndT>
struct DeviceAllocPtrs {
    // Scalars.
    float* minus_1 = nullptr;

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

    // Size = r * m.
    float* d_rm1 = nullptr;
    float* d_rm2 = nullptr;

    // Buffers for Stiefel-specific intermediates.

    // Size = r * 2r.
    float* d_2rr = nullptr;

    // Size = 2r * 2r.
    float* d_4rr = nullptr;
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
    using SubDenseMatrixPtr = std::unique_ptr<gpu::SubDenseMatrix>;
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

    // Stiefel-specific intermediates.
    DenseMatrixPtr stiefel_G_gradient;

    DenseMatrixPtr VG;
    SubDenseMatrixPtr VG_GG;  // top
    SubDenseMatrixPtr VG_DG;  // bottom, is -DG

    DenseMatrixPtr middle_stiefel_factor;
    SubDenseMatrixPtr msf_GD;  // top left
    SubDenseMatrixPtr msf_DD;  // bottom left, is -DD
    SubDenseMatrixPtr msf_GG;  // top right
    SubDenseMatrixPtr msf_DG;  // bottom right, is -DG

};


template<typename IndT>
struct StatefulFactorizationOps {
    using SpDnMatMulPtr = std::unique_ptr<gpu::ops::SpDnMatMul<IndT>>;
    using DnSpMatMulPtr = std::unique_ptr<gpu::ops::DnSpMatMul<IndT>>;
    using LuSolve_InPlacePtr = std::unique_ptr<gpu::ops::LuSolve_InPlace>;

    SpDnMatMulPtr matmul_AG;
    DnSpMatMulPtr matmul_W_AG_A;

    LuSolve_InPlacePtr stiefel_lu_solve;
};



}  // internal_


// Worker associated to a single GPU.
template<typename IndT>
class DeviceWorker {
    using DnDnMatMul = gpu::ops::DnDnMatMul;
    using DnDnMatMul_SubMatrices = gpu::ops::DnDnMatMul_SubMatrices;
    using FrobeniousInnerProduct = gpu::ops::FrobeniousInnerProduct;

    FactorizationConfig config;
    int64_t partition_index;
    int64_t n_partitions;

    gpu::DeviceContext dctx;

    // Holds pointers to the device memory allocations. Simply
    // put here for convenience.
    internal_::DeviceAllocPtrs<IndT> d_ptrs;

    // Holds the device matrix objects.
    internal_::DeviceMatrices<IndT> d_ms;

    // Holds operations that require some state to be created before
    // being used.
    internal_::StatefulFactorizationOps<IndT> s_ops;

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

        // We allocate and initialize the G before anything else
        // because it temporarily allocates some extra device memory.
        allocate_and_initialize_G();

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

    void compute_local_G_update_intermediates_after_AG_GG_all_reduces_async() {
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

        // TODO: See if this is needed.
        gpu::ops::ScalarMultiply_InPlace(dctx, *d_ms.G_gradient, 4.0).call_async();

        // Compute the local contributions to the matrix multiplies involving the euclidiean gradient.
        DnDnMatMul_SubMatrices(
            dctx,
            *d_ms.G->get_as_submatrix_cast(), *d_ms.G_gradient->get_as_submatrix_cast(),
            *d_ms.msf_GD,
            false, true
        ).call_async();
        DnDnMatMul_SubMatrices(
            dctx,
            *d_ms.G_gradient->get_as_submatrix_cast(), *d_ms.G_gradient->get_as_submatrix_cast(),
            *d_ms.msf_DD,
            false, true,
            d_ptrs.minus_1
        ).call_async();
    }

    void nccl_all_reduce_GD_DD() {
        NCCL_CALL(
            ncclAllReduce(
                (float*) d_ms.middle_stiefel_factor->data,
                (float*) d_ms.middle_stiefel_factor->data,
                2 * config.rank * config.rank,
                ncclFloat,
                ncclSum,
                dctx.comm,
                dctx.stream)
        );
    }

    void update_local_G_after_GD_DD_all_reduces_async(float learning_rate_G) {
        // The matrices GG, msf_GD, msf_DD are up to date at the onset of this function.
        int64_t rank = config.rank;

        // Fill out the VG matrix.
        gpu::ops::DnCopy_SubMatrices(
            dctx, *d_ms.GG->get_as_submatrix_cast(), *d_ms.VG_GG).call_async();
        gpu::ops::DnTranspose_SubMatrices(dctx, *d_ms.msf_GD, *d_ms.VG_DG, d_ptrs.minus_1).call_async();

        // Fill out the VU matrix.
        gpu::ops::DnCopy_SubMatrices(
            dctx, *d_ms.GG->get_as_submatrix_cast(), *d_ms.msf_GG).call_async();
        gpu::ops::DnTranspose_SubMatrices(dctx, *d_ms.msf_GD, *d_ms.msf_DG, d_ptrs.minus_1).call_async();

        // Go from the VU to I + lr / 2 * VU.
        gpu::ops::custom::MultiplyAndAddIdentity_InPlace(
            dctx, *d_ms.middle_stiefel_factor, 0.5 * learning_rate_G).call_async();

        // Solve a system of linear equations to get (I + lr/2 VU)^-1 VG^T.
        // The result will be stored in the d_ms.VG matrix.
        s_ops.stiefel_lu_solve->call_sync();

        // Compute the local stiefel gradient.
        gpu::SubDenseMatrix M1(*d_ms.VG, 0, 0, rank, rank);
        gpu::SubDenseMatrix M2(*d_ms.VG, rank, 0, rank, rank);

        DnDnMatMul_SubMatrices(
            dctx,
            M1,
            *d_ms.G_gradient->get_as_submatrix_cast(),
            *d_ms.stiefel_G_gradient->get_as_submatrix_cast(),
            true, false
        ).call_async();
        DnDnMatMul_SubMatrices(
            dctx,
            M2,
            *d_ms.G->get_as_submatrix_cast(),
            *d_ms.stiefel_G_gradient->get_as_submatrix_cast(),
            true, false,
            dctx.dev_1f, dctx.dev_1f
        ).call_async();

        // Update the parameters G given the gradient.
        gpu::ops::custom::GradientDescentUpdate(
            dctx, *d_ms.G, *d_ms.stiefel_G_gradient, learning_rate_G)
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
        const int64_t n_scalars = 1;
        float* d_scalars = dctx.dmalloc<float>(n_scalars);
        d_ptrs.minus_1 = d_scalars + 0;

        float scalars[n_scalars] = {
            -1.0f,
        };
        dctx.copy_to_device_async(d_scalars, scalars, n_scalars);
    }

    void allocate_and_initialize_G() {
        int64_t n_cols = host_matrix_partition->n_cols;
        int64_t rank = config.rank;

        // Allocate device memory.
        d_ptrs.d_G = dctx.dmalloc<float>(rank * n_cols);
        d_ptrs.d_rm2 = dctx.dmalloc<float>(rank * n_cols);

        // Create the dense matrix wrapper.
        d_ms.G = gpu::DenseMatrix::make_unique_ptr(rank, n_cols, d_ptrs.d_G);

        // Create the transposed matrix. We need to do this as we can only
        // orthogonalize the columns of a matrix, not its rows.
        gpu::DenseMatrix GT(n_cols, rank, d_ptrs.d_rm2);

        // Randomly initialize.
        CURAND_CALL(
            curandGenerateNormal(dctx.rand_gen, (float*) GT.data, GT.n_rows * GT.n_cols, 0.0f, 1.0f)
        );

        // Orthonormalize G.
        gpu::ops::Orthonormalize_InPlace ortho_op(dctx, GT);
        ortho_op.set_up_sync();
        ortho_op.call_sync();

        // Tranpose from the GT buffer to the G buffer.
        auto& G = d_ms.G;
        gpu::ops::DnTranspose(dctx, GT, *G).call_async();

        // Since we orthonormalize each partition of G, the full G will have orthogonal
        // columns but each column will have L2 norm of sqrt(n_partitions). We hence
        // rescale each partition such that the norm of the entire column is 1.
        gpu::ops::ScalarMultiply_InPlace(dctx, *G, 1.0 / std::sqrt((double) n_partitions)).call_async();

        dctx.synchronize_stream();
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

        d_ptrs.d_ncr1 = dctx.dmalloc<float>(n_classes * n_examples * rank);
        d_ptrs.d_ncr2 = dctx.dmalloc<float>(n_classes * n_examples * rank);
        d_ptrs.d_nr = dctx.dmalloc<float>(n_examples * rank);
        d_ptrs.d_rr1 = dctx.dmalloc<float>(rank * rank);
        d_ptrs.d_rr2 = dctx.dmalloc<float>(rank * rank);
        d_ptrs.d_rr3 = dctx.dmalloc<float>(rank * rank);
        d_ptrs.d_rm1 = dctx.dmalloc<float>(rank * n_cols);
        // d_ptrs.d_rm2 is allocated in the allocate_and_initialize_G method.

        d_ptrs.d_2rr = dctx.dmalloc<float>(2 * rank * rank);
        d_ptrs.d_4rr = dctx.dmalloc<float>(4 * rank * rank);

        /////////////////////////////////////////
        // Create the matrices.

        // Input/parameter matrices.
        d_ms.A = gpu::CsrMatrix<IndT>::make_unique_ptr(
            n_rows, n_cols, nnz,
            d_ptrs.d_A_values, d_ptrs.d_A_row_offsets, d_ptrs.d_A_col_indices);

        d_ms.W = gpu::DenseMatrix::make_unique_ptr(n_examples, rank, d_ptrs.d_W);

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
        d_ms.G_gradient = gpu::DenseMatrix::make_unique_ptr(rank, n_cols, d_ptrs.d_rm1);

        // Stiefel specific intermediate matrices.
        d_ms.stiefel_G_gradient = gpu::DenseMatrix::make_unique_ptr(rank, n_cols, d_ptrs.d_rm2);

        d_ms.VG = gpu::DenseMatrix::make_unique_ptr(2 * rank, rank, d_ptrs.d_2rr);
        d_ms.VG_GG = d_ms.VG->create_submatrix(0, 0, rank, rank);
        d_ms.VG_DG = d_ms.VG->create_submatrix(rank, 0, rank, rank);

        d_ms.middle_stiefel_factor = gpu::DenseMatrix::make_unique_ptr(2 * rank, 2 * rank, d_ptrs.d_4rr);
        d_ms.msf_GD = d_ms.middle_stiefel_factor->create_submatrix(0, 0, rank, rank);
        d_ms.msf_DD = d_ms.middle_stiefel_factor->create_submatrix(rank, 0, rank, rank);
        d_ms.msf_GG = d_ms.middle_stiefel_factor->create_submatrix(0, rank, rank, rank);
        d_ms.msf_DG = d_ms.middle_stiefel_factor->create_submatrix(rank, rank, rank, rank);
    }

    // Does NOT intialize G as it is initialized previously from another method.
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

            double scale_factor = config.compute_w_initialization_scale_factor();
            gpu::ops::ScalarMultiply_InPlace(dctx, *W, scale_factor).call_async();
        }

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
        s_ops.stiefel_lu_solve = std::unique_ptr<gpu::ops::LuSolve_InPlace>(
            new gpu::ops::LuSolve_InPlace(
                dctx,
                *d_ms.middle_stiefel_factor,
                *d_ms.VG
            )
        );

        // Set up the ops.
        s_ops.matmul_AG->set_up_async();
        s_ops.matmul_W_AG_A->set_up_async();
        s_ops.stiefel_lu_solve->set_up_sync();
    }

    // The host_write_location must be on the host.
    void read_matrix_from_gpu_async(gpu::DenseMatrix& matrix, float* host_write_location) {
        dctx.set_device();
        dctx.copy_to_host_async(host_write_location, (float*) matrix.data, matrix.n_entries);
    }

};





}  // stiefel
}  // factorizations
}  // npeff
