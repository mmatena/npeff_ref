#pragma once

#include <memory>

#include <util/macros.h>
#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>
#include <gpu/containers/sparse_matrix.h>
#include <gpu/containers/transfers.h>

#include <gpu/ops/dndn_matmul.h>
#include <gpu/ops/frobenius_product.h>
#include <gpu/ops/spdn_matmul.h>
#include <gpu/ops/transpose.h>
#include <gpu/ops/custom/ag_to_numerator.h>
#include <gpu/ops/custom/compute_w_ag.h>
#include <gpu/ops/custom/elwise_square.h>
#include <gpu/ops/custom/gradient_descent.h>
#include <gpu/ops/custom/hadamard_product.h>
#include <gpu/ops/custom/multiplicative_update.h>


// // Debugging stuff.
// #include <gpu/ops/debugging/numerical_checks.h>



#include "./config.h"


namespace npeff {
namespace expansion {

// Utility/helper stuff.
namespace internal_ {

// TODO: Make better sharing of device memory amongst matrices used at different times.

// Struct to just have a place to put all of the device memory
// allocation pointers in one place.
template<typename IndT>
struct DeviceAllocPtrs {
    // Scalars.
    float* minus_1 = nullptr;

    float* d_tr_WW_HH_ee_ef = nullptr;
    float* d_tr_WW_HH_fe_ff = nullptr;
    float* d_tr_WHX = nullptr;

    // The input matrix.
    float* d_A_values = nullptr;
    IndT* d_A_row_offsets = nullptr;
    IndT* d_A_col_indices = nullptr;

    // Pointers to parameters to possibly learn.
    float* d_W = nullptr;

    float* d_G_e = nullptr;
    float* d_G_f = nullptr;

    // Allocations for intermediates.

    float* d_WW_ee_ef = nullptr;

    float* d_GG_ee_ef = nullptr;
    float* d_GG_fe_ff = nullptr;

    float* d_HH_ee_ef = nullptr;
    float* d_HH_fe_ff = nullptr;

    float* d_AG_e = nullptr;

    // Allocations for W-specific intermediates.
    float* d_W_update_numerator = nullptr;
    float* d_W_update_denominator = nullptr;

    // Allocations for G-specific intermediates.
    float* d_W_AG_e = nullptr;
    float* d_G_gradient_expanded = nullptr;

    // Loss computation-specific intermediates
    float* d_WW_fe_ff = nullptr;
};


// Note that the actual device memory chunks associates to each
// matrix can overlap. Furthermore, it will be impossible to 
// use some combinations of these matrices at the same time due
// to this.
template<typename IndT>
struct DeviceMatrices {
    using DenseMatrixPtr = std::unique_ptr<gpu::DenseMatrix>;
    using CsrMatrixPtr = std::unique_ptr<gpu::CsrMatrix<IndT>>;

    // The suffix _f denotes belonging to the frozen subset of components
    // while the suffix _e denotes belonging to the expansion set of
    // components.

    CsrMatrixPtr A;

    // The W includes both of the _f and _e matrices as submatrices;
    // their memory allocations overlap.
    DenseMatrixPtr W;
    DenseMatrixPtr W_e;
    DenseMatrixPtr W_f;

    DenseMatrixPtr G_e;
    DenseMatrixPtr G_f;

    // Simple/common intermediates.
    DenseMatrixPtr WW_ee_ef;

    // GG_ee_ef is concatenation of GG_ee and GG_ef by taking
    // the union of their columns. The latter two are submatrices
    // of the former on the device.
    DenseMatrixPtr GG_ee_ef;
    DenseMatrixPtr GG_ee;
    DenseMatrixPtr GG_ef;

    // GG_fe_ff is concatenation of GG_fe and GG_ff by taking
    // the union of their columns. The latter two are submatrices
    // of the former on the device.
    DenseMatrixPtr GG_fe_ff;
    DenseMatrixPtr GG_fe;
    DenseMatrixPtr GG_ff;

    // HH_ee_ef is concatenation of HH_ee and HH_ef by taking
    // the union of their columns. The latter two are submatrices
    // of the former on the device.
    DenseMatrixPtr HH_ee_ef;
    DenseMatrixPtr HH_ee;
    DenseMatrixPtr HH_ef;

    // HH_fe_ff is concatenation of HH_fe and HH_ff by taking
    // the union of their columns. The latter two are submatrices
    // of the former on the device.
    DenseMatrixPtr HH_fe_ff;
    DenseMatrixPtr HH_fe;
    DenseMatrixPtr HH_ff;

    DenseMatrixPtr AG_e;

    // W-step specific intermediates.

    // The W_update_numerator includes both of the _frozen and _expanded
    // matrices as submatrices; their memory allocations overlap.
    DenseMatrixPtr W_update_numerator;
    DenseMatrixPtr W_update_numerator_expanded;
    DenseMatrixPtr W_update_numerator_frozen;

    // The W_update_numerator includes the _expanded
    // matrix as a submatrix; their memory allocations overlap.
    DenseMatrixPtr W_update_denominator;
    DenseMatrixPtr W_update_denominator_expanded;

    // G-step specific intermediates.

    // The _ee and _ef are submatrices of WW_GG_ee_ef.
    DenseMatrixPtr WW_GG_ee_ef;
    DenseMatrixPtr WW_GG_ee;
    DenseMatrixPtr WW_GG_ef;

    DenseMatrixPtr W_AG_e;
    DenseMatrixPtr G_gradient_expanded;

    // Loss computation-specific intermediates
    // TODO: I can probably combine these with interedmiates used elsewhere.
    DenseMatrixPtr WW_fe_ff;
    DenseMatrixPtr WW_HH_fe_ff;
};

template<typename IndT>
struct StatefulFactorizationOps {
    using SpDnMatMulPtr = std::unique_ptr<gpu::ops::SpDnMatMul<IndT>>;
    using DnSpMatMulPtr = std::unique_ptr<gpu::ops::DnSpMatMul<IndT>>;

    SpDnMatMulPtr matmul_AG_e;
    DnSpMatMulPtr matmul_W_AG_e_A;
};


}  // internal_



// Worker associated to a single GPU.
template<typename IndT>
class DeviceWorker {
    using DnDnMatMul = gpu::ops::DnDnMatMul;
    using FrobeniousInnerProduct = gpu::ops::FrobeniousInnerProduct;

public:
    // This attribute must be declared near the top of this class so that
    // its destructor is called after the destructors of all of the objects
    // that allocated device memory using it.
    gpu::DeviceContext dctx;
protected:
    ExpansionConfig config;
    int64_t partition_index;
    int64_t n_partitions;


    // Holds pointers to the device memory allocations. Simply
    // put here for convenience.
    internal_::DeviceAllocPtrs<IndT> d_ptrs;

    // Holds the device matrix objects.
    internal_::DeviceMatrices<IndT> d_ms;

    // Holds operations that require some state to be created before
    // being used.
    internal_::StatefulFactorizationOps<IndT> s_ops;


    // NOTE: These will become a null pointer and the
    // associated matrices deleted once their data has
    // been moved to the GPU.
    std::unique_ptr<npeff::CsrMatrix<IndT>> host_matrix_partition;
    std::unique_ptr<npeff::DenseMatrix<float>> frozen_G_partition;

public:

    DeviceWorker(
        std::unique_ptr<npeff::CsrMatrix<IndT>> host_matrix_partition,
        std::unique_ptr<npeff::DenseMatrix<float>> frozen_G_partition,
        ExpansionConfig config,
        int64_t partition_index,
        int64_t n_partitions,
        ncclComm_t comm
    ) : 
        host_matrix_partition(std::move(host_matrix_partition)),
        frozen_G_partition(std::move(frozen_G_partition)),
        config(config),
        partition_index(partition_index),
        n_partitions(n_partitions),
        dctx(gpu::DeviceContext(partition_index, comm, config.rand_gen_seed))
    {
        // std::cout << "HOST MATRIX VALID: " << this->host_matrix_partition->validate_indices() << "\n";

        // for(int i=0;i<10;i++) {
        //     // std::cout << this->host_matrix_partition->values.get()[i] << "\n";
        //     // std::cout << this->host_matrix_partition->col_indices.get()[i] << "\n";
        //     std::cout << this->host_matrix_partition->row_offsets.get()[i] << "\n";
        // }

        // for(int i=0;i<10;i++) {
        //     // std::cout << this->host_matrix_partition->values.get()[this->host_matrix_partition->nnz - 1 - i] << "\n";
        //     // std::cout << this->host_matrix_partition->col_indices.get()[this->host_matrix_partition->nnz - 1 - i] << "\n";
        //     std::cout << this->host_matrix_partition->row_offsets.get()[this->host_matrix_partition->n_rows - i] << "\n";
        // }

        // std::cout << "AAAAAA: " << this->host_matrix_partition->col_indices.get()[this->host_matrix_partition->nnz - 1]
        //     << ", " << this->host_matrix_partition->n_cols << "\n";

        // std::cout << this->host_matrix_partition->row_offsets.get()[this->host_matrix_partition->n_rows] 
        //     << ", " << this->host_matrix_partition->row_offsets.get()[this->host_matrix_partition->n_rows - 1] 
        //     << ", " << this->host_matrix_partition->row_offsets.get()[this->host_matrix_partition->n_rows - 2] 
        //     << ", " << this->host_matrix_partition->row_offsets.get()[this->host_matrix_partition->n_rows - 3] 
        //     << "\n";
    }

    int64_t get_device() {
        return dctx.device;
    }

    float* get_W_data_ptr() {
        return (float*) d_ms.W->data;
    }

    void synchronize_stream() {
        dctx.synchronize_stream();
    }

    /////////////////////////////////////////////////////////////////
    // Initialization-related methods.

    // Must be called once before doing anything else. This allocates memory
    // on the device and moves data to the GPU. Parameters will be randomly
    // initialized.
    // 
    // NOTE: The W matrices (and any other parameters shared across multiple
    // devices) should/must be made consistent AFTER this is called. Furthermore,
    // some values must be precomputed once AFTER this is called but before the
    // rest of the factorization.
    void set_up_work(npeff::DenseMatrix<float>* initial_W_f) {
        // NOTE: initial_W_f can be a nullptr, which corresponds to randomly initializing
        // those coefficients.
        dctx.initialize();
        std::cout << "Starting to set up work.\n";
        write_scalars_to_gpu();
        std::cout << "Written scalars to GPU.\n";

        allocate_and_create_device_matrices();
        std::cout << "Device matrices allocated and created.\n";
        initialize_device_matrices(initial_W_f);
        std::cout << "Device matrices initialized.\n";

        construct_and_initialize_stateful_ops();
        std::cout << "Stateful ops contructed and intitialized initialized.\n";
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

    // The local constants to precompute are:
    //      GG_ff, HH_ff, W_update_numerator_frozen

    void precompute_local_constants_async() {
        // Compute the local portion of AG_f.
        gpu::DenseMatrix AG_f = get_AG_f_handle();



        // std::cout << "A.values: " <<  d_ms.A->values << "\n";
        // std::cout << "A.row_offsets: " <<  d_ms.A->row_offsets << "\n";
        // std::cout << "A.col_indices: " <<  d_ms.A->col_indices << "\n";
        // std::cout << "G_f: " <<  d_ms.G_f->data << "\n";
        // std::cout << "AG_f: " <<  AG_f.data << "\n";

        // std::cout << "A.shape = (" << d_ms.A->n_rows << ", " << d_ms.A->n_cols << ")\n";
        // std::cout << "G_f.shape = (" << d_ms.G_f->n_rows << ", " << d_ms.G_f->n_cols << ")\n";
        // std::cout << "AG_f.shape = (" << AG_f.n_rows << ", " << AG_f.n_cols << ")\n";


        gpu::ops::SpDnMatMul<IndT> matmul_AG_f(
                dctx,
                *d_ms.A, *d_ms.G_f, AG_f,
                false, true);

        matmul_AG_f.set_up_async();

        synchronize_stream();std::cout << "matmul_AG_f set_up\n";

        matmul_AG_f.call_async();

        synchronize_stream();std::cout << "matmul_AG_f call\n";

        // Compute the local version of GG_ff.
        DnDnMatMul(dctx, *d_ms.G_f, *d_ms.G_f, *d_ms.GG_ff, false, true).call_async();

        synchronize_stream();std::cout << "GG_ff compute call\n";
    }

    void nccl_all_reduce_precomputed_constants() {
        gpu::DenseMatrix AG_f = get_AG_f_handle();
        NCCL_CALL(
            ncclAllReduce(
                (float*) AG_f.data,
                (float*) AG_f.data,
                AG_f.n_entries,
                ncclFloat,
                ncclSum,
                dctx.comm,
                dctx.stream)
        );
        NCCL_CALL(
            ncclAllReduce(
                (float*) d_ms.GG_ff->data,
                (float*) d_ms.GG_ff->data,
                d_ms.GG_ff->n_entries,
                ncclFloat,
                ncclSum,
                dctx.comm,
                dctx.stream)
        );
    }

    void finish_precomputing_constants_after_all_reduce_async() {
        // Must called be after the AG_f and GG_ff matrices have been all-reduced
        // across all devices (or slated to do so in the streams).

        // Compute the numerator.
        gpu::DenseMatrix AG_f = get_AG_f_handle();
        gpu::ops::custom::AgToNumerator(
            dctx, AG_f, *d_ms.W_update_numerator_frozen, config.n_classes)
            .call_async();

        // Square GG to get HH.
        gpu::ops::custom::ElwiseSquare(dctx, *d_ms.GG_ff, *d_ms.HH_ff).call_async();




        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.W_update_numerator_frozen, 1).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.HH_ff, 2).call_sync();

        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.W_e, 1).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.W_f, 2).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.G_e, 2).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.G_f, 3).call_sync();
    }

    /////////////////////////////////////////////////////////////////
    // General stuff.

    void compute_local_AG_e_GG_ee_ef_async() {
        s_ops.matmul_AG_e->call_async();
        DnDnMatMul(dctx, *d_ms.G_e, *d_ms.G_e, *d_ms.GG_ee, false, true).call_async();
        DnDnMatMul(dctx, *d_ms.G_e, *d_ms.G_f, *d_ms.GG_ef, false, true).call_async();




        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.GG_ee, 1).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.GG_ef, 2).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.AG_e, 3).call_sync();
    }

    // NOTE: The G_fe will not be updated by. If needed, it will have to created
    // manually after this by tranposing G_ef.
    void nccl_all_reduce_AG_e_GG_ee_ef() {
        NCCL_CALL(
            ncclAllReduce(
                (float*) d_ms.AG_e->data,
                (float*) d_ms.AG_e->data,
                d_ms.AG_e->n_entries,
                ncclFloat,
                ncclSum,
                dctx.comm,
                dctx.stream)
        );
        NCCL_CALL(
            ncclAllReduce(
                (float*) d_ms.GG_ee_ef->data,
                (float*) d_ms.GG_ee_ef->data,
                d_ms.GG_ee_ef->n_entries,
                ncclFloat,
                ncclSum,
                dctx.comm,
                dctx.stream)
        );
    }

    /////////////////////////////////////////////////////////////////
    // W-update stuff.

    void update_local_W_after_all_reduces_async(bool update_only_expansion = false) {
        // Must called be after the AG and GG matrices have been all-reduced
        // across all devices (or slated to do so in the streams).

        // Set the GG_fe to be the transpose of GG_ef.
        gpu::ops::DnTranspose(dctx, *d_ms.GG_ef, *d_ms.GG_fe).call_async();

        // Compute the numerator.
        gpu::ops::custom::AgToNumerator(
            dctx, *d_ms.AG_e, *d_ms.W_update_numerator_expanded, config.n_classes)
            .call_async();

        // Square GG to get HH. The HH_ff should be already precomputed.
        gpu::ops::custom::ElwiseSquare(dctx, *d_ms.GG_ee_ef, *d_ms.HH_ee_ef).call_async();
        gpu::ops::custom::ElwiseSquare(dctx, *d_ms.GG_fe, *d_ms.HH_fe).call_async();

        // Compute W(HH^T) to get the denominator. We do it in two steps due to how we have
        // subdivided the matrices in device memory.
        DnDnMatMul(dctx, *d_ms.W_e, *d_ms.HH_ee_ef, *d_ms.W_update_denominator, false, false)
            .call_async();
        DnDnMatMul(dctx, *d_ms.W_f, *d_ms.HH_fe_ff, *d_ms.W_update_denominator, false, false, dctx.dev_1f, dctx.dev_1f)
            .call_async();

        // Update the local copy of W.
        if (update_only_expansion) {
            gpu::ops::custom::MultiplicativeUpdate(
                dctx, *d_ms.W_e, *d_ms.W_update_numerator_expanded, *d_ms.W_update_denominator_expanded, config.mu_eps)
                .call_async();
        } else {
            gpu::ops::custom::MultiplicativeUpdate(
                dctx, *d_ms.W, *d_ms.W_update_numerator, *d_ms.W_update_denominator, config.mu_eps)
                .call_async();
        }
    }


    /////////////////////////////////////////////////////////////////
    // G-update stuff.

    void update_local_G_after_all_reduces_async(float learning_rate_G) {

        
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.GG_ee, 1001).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.GG_ef, 1002).call_sync();
        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.AG_e, 1003).call_sync();




        // Compute the first term and write it to the buffer storing the gradient.
        DnDnMatMul(dctx, *d_ms.W_e, *d_ms.W, *d_ms.WW_ee_ef, true, false).call_async();
        gpu::ops::custom::HadamardProduct(
            dctx, *d_ms.WW_ee_ef, *d_ms.GG_ee_ef, *d_ms.WW_GG_ee_ef)
            .call_async();
        // Need to split into two steps due to how we subdivided the matrices in device memory.
        DnDnMatMul(dctx, *d_ms.WW_GG_ee, *d_ms.G_e, *d_ms.G_gradient_expanded, false, false)
            .call_async();
        DnDnMatMul(dctx, *d_ms.WW_GG_ef, *d_ms.G_f, *d_ms.G_gradient_expanded, false, false, dctx.dev_1f, dctx.dev_1f)
            .call_async();

        // Compute the second term and accumulate it onto the gradient buffer.
        gpu::ops::custom::Compute_W_AG(
            dctx, *d_ms.W_e, *d_ms.AG_e, *d_ms.W_AG_e, config.n_classes)
            .call_async();
        s_ops.matmul_W_AG_e_A->call_async();

        // Update the parameters G given the gradient.
        // 
        // The factor of 4 comes from the gradient being multiplied by that
        // but not accounted for in our computation of it.
        gpu::ops::custom::GradientDescentUpdate(
            dctx, *d_ms.G_e, *d_ms.G_gradient_expanded, 4.0f * learning_rate_G)
            .call_async();





        // gpu::ops::debugging::AssertFinite(dctx, *d_ms.G_e, 2).call_sync();
    }

    /////////////////////////////////////////////////////////////////
    // Loss computation stuff.
    // 
    // TODO: Right now, we are only computing the reconstruction loss. Things
    // like regularization losses are not included. Ideally, we'd compute and
    // saved each of them separately.

    void compute_loss_after_all_reduces_async() {
        //////////////////////////////////////////////
        // Compute tr_WW_HH.

        // Set the GG_fe to be the transpose of GG_ef.
        gpu::ops::DnTranspose(dctx, *d_ms.GG_ef, *d_ms.GG_fe).call_async();
        // Square GG to get HH. The HH_ff should be already precomputed.
        gpu::ops::custom::ElwiseSquare(dctx, *d_ms.GG_ee_ef, *d_ms.HH_ee_ef).call_async();
        gpu::ops::custom::ElwiseSquare(dctx, *d_ms.GG_fe, *d_ms.HH_fe).call_async();

        DnDnMatMul(dctx, *d_ms.W_e, *d_ms.W, *d_ms.WW_ee_ef, true, false).call_async();
        DnDnMatMul(dctx, *d_ms.W_f, *d_ms.W, *d_ms.WW_fe_ff, true, false).call_async();

        auto& WW_HH_ee_ef = d_ms.WW_GG_ee_ef;
        gpu::ops::custom::HadamardProduct(
            dctx, *d_ms.WW_ee_ef, *d_ms.GG_ee_ef, *WW_HH_ee_ef)
            .call_async();
        gpu::ops::custom::HadamardProduct(
            dctx, *d_ms.WW_fe_ff, *d_ms.GG_fe_ff, *d_ms.WW_HH_fe_ff)
            .call_async();
        FrobeniousInnerProduct(dctx, *d_ms.WW_ee_ef, *d_ms.HH_ee_ef, d_ptrs.d_tr_WW_HH_ee_ef)
            .call_async();
        FrobeniousInnerProduct(dctx, *d_ms.WW_fe_ff, *d_ms.HH_fe_ff, d_ptrs.d_tr_WW_HH_fe_ff)
            .call_async();

        //////////////////////////////////////////////
        // Compute tr_WHX.
        gpu::ops::custom::AgToNumerator(
            dctx, *d_ms.AG_e, *d_ms.W_update_numerator_expanded, config.n_classes)
            .call_async();
        FrobeniousInnerProduct(dctx, *d_ms.W, *d_ms.W_update_numerator, d_ptrs.d_tr_WHX)
            .call_async();
    }

    // NOTE: The actual loss will be a fixed constant plus what this
    // function returns.
    float read_loss_term_from_device() {
        dctx.set_device();
        float tr_WW_HH_ee_ef, tr_WW_HH_fe_ff, tr_WHX; 
        dctx.copy_to_host_async<float>(&tr_WW_HH_ee_ef, d_ptrs.d_tr_WW_HH_ee_ef, 1);
        dctx.copy_to_host_async<float>(&tr_WW_HH_fe_ff, d_ptrs.d_tr_WW_HH_fe_ff, 1);
        dctx.copy_to_host_async<float>(&tr_WHX, d_ptrs.d_tr_WHX, 1);
        dctx.synchronize_stream();
        return -2.0f * tr_WHX + tr_WW_HH_ee_ef + tr_WW_HH_fe_ff;
    }

    /////////////////////////////////////////////////////////////////
    // Other stuff.

    gpu::DenseMatrix* get_W_ptr() {
        return d_ms.W.get();
    }

    gpu::DenseMatrix* get_G_e_ptr() {
        return d_ms.G_e.get();
    }

    gpu::DenseMatrix* get_G_f_ptr() {
        return d_ms.G_f.get();
    }

protected:

    //////////////////////////////////////////////////////////////////////
    // Initialization-related functions.

    void write_scalars_to_gpu() {
        // Allocate memory for the scalars representing intermediate quantities.
        d_ptrs.d_tr_WW_HH_ee_ef = dctx.dmalloc<float>(3);
        d_ptrs.d_tr_WW_HH_fe_ff = d_ptrs.d_tr_WW_HH_ee_ef + 1;
        d_ptrs.d_tr_WHX = d_ptrs.d_tr_WW_HH_ee_ef + 2;

        // Allocate memory for and write the constant scalars to the device.
        const int64_t n_scalars = 1;
        float* d_scalars = dctx.dmalloc<float>(n_scalars);
        d_ptrs.minus_1 = d_scalars + 0;

        float scalars[n_scalars] = {
            -1.0f,
        };
        dctx.copy_to_device_async(d_scalars, scalars, n_scalars);
    }

    void allocate_and_create_device_matrices() {
        int64_t n_rows = host_matrix_partition->n_rows;
        int64_t n_cols = host_matrix_partition->n_cols;
        int64_t nnz = host_matrix_partition->nnz;

        int64_t n_classes = config.n_classes;
        int64_t rank_expansion = config.rank_expansion;
        int64_t rank_frozen = config.rank_frozen;
        int64_t total_rank = config.total_rank();

        int64_t n_examples = n_rows / n_classes;

        /////////////////////////////////////////
        // Allocate the memory on the device.

        d_ptrs.d_A_values = dctx.dmalloc<float>(nnz);
        d_ptrs.d_A_row_offsets = dctx.dmalloc<IndT>(n_rows + 1);
        d_ptrs.d_A_col_indices = dctx.dmalloc<IndT>(nnz);

        d_ptrs.d_W = dctx.dmalloc<float>(n_examples * total_rank);
        d_ptrs.d_G_e = dctx.dmalloc<float>(rank_expansion * n_cols);
        d_ptrs.d_G_f = dctx.dmalloc<float>(rank_frozen * n_cols);

        d_ptrs.d_WW_ee_ef = dctx.dmalloc<float>(rank_expansion * total_rank);

        d_ptrs.d_GG_ee_ef = dctx.dmalloc<float>(rank_expansion * total_rank);
        d_ptrs.d_GG_fe_ff = dctx.dmalloc<float>(rank_frozen * total_rank);

        d_ptrs.d_HH_ee_ef = dctx.dmalloc<float>(rank_expansion * total_rank);
        d_ptrs.d_HH_fe_ff = dctx.dmalloc<float>(rank_frozen * total_rank);

        d_ptrs.d_AG_e = dctx.dmalloc<float>(n_classes * n_examples * rank_expansion);

        d_ptrs.d_W_update_numerator = dctx.dmalloc<float>(n_examples * total_rank);
        d_ptrs.d_W_update_denominator = dctx.dmalloc<float>(n_examples * total_rank);

        d_ptrs.d_W_AG_e = dctx.dmalloc<float>(n_classes * n_examples * rank_expansion);
        d_ptrs.d_G_gradient_expanded = dctx.dmalloc<float>(rank_expansion * n_cols);

        d_ptrs.d_WW_fe_ff = dctx.dmalloc<float>(rank_frozen * total_rank);

        /////////////////////////////////////////
        // Create the matrices.

        // TODO: Make better sharing of device memory amongst matrices used at different times.

        // Input/parameter matrices.
        d_ms.A = gpu::CsrMatrix<IndT>::make_unique_ptr(
            n_rows, n_cols, nnz,
            d_ptrs.d_A_values, d_ptrs.d_A_row_offsets, d_ptrs.d_A_col_indices);

        d_ms.W = gpu::DenseMatrix::make_unique_ptr(n_examples, total_rank, d_ptrs.d_W);
        d_ms.W_e = gpu::DenseMatrix::make_unique_ptr(n_examples, rank_expansion, d_ptrs.d_W);
        d_ms.W_f = gpu::DenseMatrix::make_unique_ptr(n_examples, rank_frozen, d_ptrs.d_W + d_ms.W_e->n_entries);

        d_ms.G_e = gpu::DenseMatrix::make_unique_ptr(rank_expansion, n_cols, d_ptrs.d_G_e);
        d_ms.G_f = gpu::DenseMatrix::make_unique_ptr(rank_frozen, n_cols, d_ptrs.d_G_f);

        // Intermediate matrices.

        d_ms.WW_ee_ef = gpu::DenseMatrix::make_unique_ptr(rank_expansion, total_rank, d_ptrs.d_WW_ee_ef);

        d_ms.GG_ee_ef = gpu::DenseMatrix::make_unique_ptr(rank_expansion, total_rank, d_ptrs.d_GG_ee_ef);
        d_ms.GG_ee = gpu::DenseMatrix::make_unique_ptr(rank_expansion, rank_expansion, d_ptrs.d_GG_ee_ef);
        d_ms.GG_ef = gpu::DenseMatrix::make_unique_ptr(rank_expansion, rank_frozen, d_ptrs.d_GG_ee_ef + d_ms.GG_ee->n_entries);

        d_ms.GG_fe_ff = gpu::DenseMatrix::make_unique_ptr(rank_frozen, total_rank, d_ptrs.d_GG_fe_ff);
        d_ms.GG_fe = gpu::DenseMatrix::make_unique_ptr(rank_frozen, rank_expansion, d_ptrs.d_GG_fe_ff);
        d_ms.GG_ff = gpu::DenseMatrix::make_unique_ptr(rank_frozen, rank_frozen, d_ptrs.d_GG_fe_ff + d_ms.GG_fe->n_entries);

        d_ms.HH_ee_ef = gpu::DenseMatrix::make_unique_ptr(rank_expansion, total_rank, d_ptrs.d_HH_ee_ef);
        d_ms.HH_ee = gpu::DenseMatrix::make_unique_ptr(rank_expansion, rank_expansion, d_ptrs.d_HH_ee_ef);
        d_ms.HH_ef = gpu::DenseMatrix::make_unique_ptr(rank_expansion, rank_frozen, d_ptrs.d_HH_ee_ef + d_ms.HH_ee->n_entries);

        d_ms.HH_fe_ff = gpu::DenseMatrix::make_unique_ptr(rank_frozen, total_rank, d_ptrs.d_HH_fe_ff);
        d_ms.HH_fe = gpu::DenseMatrix::make_unique_ptr(rank_frozen, rank_expansion, d_ptrs.d_HH_fe_ff);
        d_ms.HH_ff = gpu::DenseMatrix::make_unique_ptr(rank_frozen, rank_frozen, d_ptrs.d_HH_fe_ff + d_ms.HH_fe->n_entries);

        d_ms.AG_e = gpu::DenseMatrix::make_unique_ptr(n_classes * n_examples, rank_expansion, d_ptrs.d_AG_e);

        // W-update step specific intermediates.
        d_ms.W_update_numerator = gpu::DenseMatrix::make_unique_ptr(n_examples, total_rank, d_ptrs.d_W_update_numerator);
        d_ms.W_update_numerator_expanded = gpu::DenseMatrix::make_unique_ptr(n_examples, rank_expansion, d_ptrs.d_W_update_numerator);
        d_ms.W_update_numerator_frozen = gpu::DenseMatrix::make_unique_ptr(n_examples, rank_frozen, d_ptrs.d_W_update_numerator + d_ms.W_update_numerator_expanded->n_entries);
        
        d_ms.W_update_denominator = gpu::DenseMatrix::make_unique_ptr(n_examples, total_rank, d_ptrs.d_W_update_denominator);
        d_ms.W_update_denominator_expanded = gpu::DenseMatrix::make_unique_ptr(n_examples, rank_expansion, d_ptrs.d_W_update_denominator);
        
        // G-update step specific intermediates.

        // NOTE: This is intended to re-use the memory of d_ms.WW_ee_ef.
        d_ms.WW_GG_ee_ef = gpu::DenseMatrix::make_unique_ptr(rank_expansion, total_rank, d_ptrs.d_WW_ee_ef);
        d_ms.WW_GG_ee = gpu::DenseMatrix::make_unique_ptr(rank_expansion, rank_expansion, d_ptrs.d_WW_ee_ef);
        d_ms.WW_GG_ef = gpu::DenseMatrix::make_unique_ptr(rank_expansion, rank_frozen, d_ptrs.d_WW_ee_ef + d_ms.WW_GG_ee->n_entries);
    
        d_ms.W_AG_e = gpu::DenseMatrix::make_unique_ptr(n_classes * n_examples, rank_expansion, d_ptrs.d_W_AG_e);
        d_ms.G_gradient_expanded = gpu::DenseMatrix::make_unique_ptr(rank_expansion, n_cols, d_ptrs.d_G_gradient_expanded);

        // Loss computation specific intermediates.
        d_ms.WW_fe_ff = gpu::DenseMatrix::make_unique_ptr(rank_frozen, total_rank, d_ptrs.d_WW_fe_ff);
        d_ms.WW_HH_fe_ff = gpu::DenseMatrix::make_unique_ptr(rank_frozen, total_rank, d_ptrs.d_WW_fe_ff);
    }

    void initialize_device_matrices(npeff::DenseMatrix<float>* initial_W_f) {

        // for(int64_t i=0; i<initial_W_f->n_entries; i++) {
        //     if (isnan(initial_W_f->data.get()[i]) || isinf(isnan(initial_W_f->data.get()[i]))) {
        //         THROW_MSG("ADASFDAFADFADF");
        //     }
        // }


        dctx.set_device();

        // Move A onto the GPU.
        dctx.copy_to_device_async(*d_ms.A, *host_matrix_partition);

        // Move G_f onto the GPU.
        dctx.copy_to_device_async(*d_ms.G_f, *frozen_G_partition);

        // If we are the first partition, initialize W.
        if(partition_index == 0) {
            gpu::DenseMatrix* W_rand_init = d_ms.W.get();

            if (initial_W_f != nullptr) {
                W_rand_init = d_ms.W_e.get();
                dctx.copy_to_device_async(*d_ms.W_f, *initial_W_f);
            }

            CURAND_CALL(
                curandGenerateUniform(dctx.rand_gen, (float*) W_rand_init->data, W_rand_init->n_rows * W_rand_init->n_cols)
            );
        }

        auto& G_e = d_ms.G_e;
        double inv_g_factor = config.compute_inv_g_initialization_scale_factor();
        CURAND_CALL(
            curandGenerateNormal(dctx.rand_gen, (float*) G_e->data, G_e->n_rows * G_e->n_cols, 0.0f, 1.0 / inv_g_factor)
        );

        // Synchronize the stream to assure that everything associated to A
        // has been copied onto the GPU. Then free the memory assoicated to
        // A and G on the host.
        dctx.synchronize_stream();
        host_matrix_partition.reset();
        frozen_G_partition.reset();
    }

    void construct_and_initialize_stateful_ops() {
        // Create the ops.
        s_ops.matmul_AG_e = std::unique_ptr<gpu::ops::SpDnMatMul<IndT>>(
            new gpu::ops::SpDnMatMul<IndT>(
                dctx,
                *d_ms.A, *d_ms.G_e, *d_ms.AG_e,
                false, true)
        );
        s_ops.matmul_W_AG_e_A = std::unique_ptr<gpu::ops::DnSpMatMul<IndT>>(
            new gpu::ops::DnSpMatMul<IndT>(
                dctx,
                *d_ms.W_AG_e, *d_ms.A, *d_ms.G_gradient_expanded,
                true, false,
                CUSPARSE_SPMM_ALG_DEFAULT,
                d_ptrs.minus_1, dctx.dev_1f)
        );

        // Set up the ops.
        s_ops.matmul_AG_e->set_up_async();
        s_ops.matmul_W_AG_e_A->set_up_async();
    }

    // Makes a temporary handle for the AG_f matrix, which is to be used only
    // during the precompute constants phase.
    gpu::DenseMatrix get_AG_f_handle() {
        gpu::DenseMatrix AG_f(d_ms.AG_e->n_rows, config.rank_frozen, d_ptrs.d_G_gradient_expanded);
        // We use the buffer associated to the G gradient update as we assume it is the largest.
        // Throw an exception if it is not big enough. This shouldn't happen in practice, but
        // it is good to check.
        THROW_IF_FALSE(AG_f.n_entries <= d_ms.G_gradient_expanded->n_entries);
        return AG_f;
    }

};

// Set up work (includes moving G and A parition to GPU)
// Precompute constants (precompute things that wont change during training, requires multi-gpu stuff)
// Broadcast W, includes initializing part of it from the decomposition if so desired.
// Then ready to run 



}  // expansion
}  // npeff
