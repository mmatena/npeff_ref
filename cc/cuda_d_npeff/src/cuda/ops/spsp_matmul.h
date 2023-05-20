#pragma once
/** Some specialized versions of sparse-sparse matrix multiply. */


#include <vector>
#include <tuple>
#include <memory>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <misc/common.h>

#include <cuda/cuda_context.h>
#include <cuda/descr.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>

#include <cuda/ops/matrix_conversion.h>
#include <cuda/ops/sum.h>


namespace Cuda {
namespace Ops {



template <typename IndT>
class Partitioned_SpSpMatmul_ToDense_SingleUse : public DeviceCudaContext::Freeable {
    using DenseMatrix = Device::DenseMatrix;
    using DnMatrix = Device::DnMatrix<COL_MAJOR>;
    using CsrMatrix = Device::CsrMatrix<IndT>;
    using Shape = std::tuple<long, long>;

protected:
    DeviceCudaContext& ctx;

    std::vector<CsrMatrix>& left;
    std::vector<CsrMatrix>& right;

    DenseMatrix& out;

    // Only these are currently supported by cusparse.
    const cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;
    const cusparseOperation_t left_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t right_op = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // Holds the memory for the sparse products.
    std::unique_ptr<CsrMatrix> sp_product_container;
    std::unique_ptr<DnMatrix> dn_product_container;


public:

    Partitioned_SpSpMatmul_ToDense_SingleUse(
        DeviceCudaContext& ctx,
        std::vector<CsrMatrix>& left,
        std::vector<CsrMatrix>& right,
        DenseMatrix& out
    ) :
        ctx(ctx), right(right), left(left), out(out)
    {
        ValidateInputs();
    }


    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        std::vector<void*> ret;
        auto sp = sp_product_container->GetDeviceAllocs();
        ret.insert(ret.end(), sp.begin(), sp.end());
        auto dn = dn_product_container->GetDeviceAllocs();
        ret.insert(ret.end(), dn.begin(), dn.end());
        return ret;
    }


    void SetUp() {
        ctx.SetDevice();

        // Make the fixed size buffer for the sparse product outputs.
        sp_product_container = std::unique_ptr<CsrMatrix>(
            new CsrMatrix(out.n_rows, out.n_cols, out.n_entries));
        sp_product_container->AllocMemory(ctx);

        // Make the fixed size buffer for the densification of the
        // the sparse product outputs.
        dn_product_container = std::unique_ptr<DnMatrix>(
            new DnMatrix(out.n_rows, out.n_cols));
        dn_product_container->AllocMemory(ctx);
    }

    void Call() {
        ctx.SetDevice();

        // Zero out the out matrix.
        CUDA_CALL(cudaMemset(out.data, 0, out.size_bytes));

        for(int i=0;i<left.size();i++) {
            // std::cout << "SpSpMatMul Step " << i << "\n";

            SingleSpMatmul matmul(this, left[i], right[i]);
            matmul.Call();
            ctx.SynchronizeStream();

            CsrMatrix sp_product = matmul.GetOutput();

            ctx.FreeDeviceAllocs(matmul);

            if (sp_product.nnz > 0) {
                CsrToDense<IndT, COL_MAJOR> to_dense_op(ctx, sp_product, *dn_product_container);
                
                to_dense_op.SetUpAsync();
                ctx.SynchronizeStream();

                to_dense_op.CallAsync();
                ctx.SynchronizeStream();

                DenseMatrix dense_summand = dn_product_container->AsDenseMatrix();
                AddDenseDense add_op(ctx, out, dense_summand, out);
                add_op.CallAsync();
                ctx.SynchronizeStream();

                ctx.FreeDeviceAllocs(to_dense_op);
            }

        }

    }


protected:

    void ValidateInputs() {
        THROWSERT(left.size() == right.size());
        for(long i=0;i<left.size();i++) {
            // Confirm that we can multiply the left and right factors.
            THROWSERT(left[i].n_cols == right[i].n_rows);
            // Confirm the left and right factors produce an output of the
            // correct shape.
            THROWSERT(left[i].n_rows == out.n_rows);
            THROWSERT(right[i].n_cols == out.n_cols);
        }
    }


    // Zero out the out matrix.
    // Create fixed size buffer for sparse output matrix (space for full-size).
    // Do the individual matmul followed by cusparseAxpby (need to set up some more descrs).

    //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
    // Helper classes.

    class SingleSpMatmul : public DeviceCudaContext::Freeable {
    protected:
        DeviceCudaContext& ctx;

        CsrMatrix& left;
        CsrMatrix& right;
        std::unique_ptr<CsrMatrix>& sp_product_container;

        Shape out_shape;

        const cusparseSpGEMMAlg_t alg;
        const cusparseOperation_t left_op;
        const cusparseOperation_t right_op;

        UniqueDescr<cusparseSpGEMMDescr_t> mm_descr;
        UniqueDescr<cusparseSpMatDescr_t> out_descr;
        
        size_t buffer_size1 = 0;
        void *buffer1 = nullptr;

        size_t buffer_size2 = 0;
        void *buffer2 = nullptr;

    public:
        SingleSpMatmul(
            Partitioned_SpSpMatmul_ToDense_SingleUse<IndT>* op,
            CsrMatrix& left,
            CsrMatrix& right
        ) :
            ctx(op->ctx),
            left(left),
            right(right),
            sp_product_container(op->sp_product_container),
            out_shape(op->out.n_rows, op->out.n_cols),
            alg(op->alg),
            left_op(op->left_op),
            right_op(op->right_op)
        {
            ctx.SetDevice();
            CreateOutDescr();
            CUSPARSE_CALL(cusparseSpGEMM_createDescr(&mm_descr.descr));
        }

        // DeviceCudaContext::Freeable
        virtual std::vector<void*> GetDeviceAllocs() {
            return {buffer1, buffer2};
        }

        void Call() {
            ctx.SetDevice();

            Call_WorkEstimation(nullptr);
            ctx.SynchronizeStream();
            buffer1 = ctx.dmalloc<void>(buffer_size1);
            Call_WorkEstimation(buffer1);

            Call_Compute(nullptr);
            ctx.SynchronizeStream();
            buffer2 = ctx.dmalloc<void>(buffer_size2);
            Call_Compute(buffer2);

            Call_Copy();
            ctx.SynchronizeStream();
        }

        CsrMatrix GetOutput() {
            int64_t out_n_rows, out_n_cols, out_nnz;
            CUSPARSE_CALL(
                cusparseSpMatGetSize(out_descr.descr, &out_n_rows, &out_n_cols, &out_nnz));

            CsrMatrix ret(out_n_rows, out_n_cols, out_nnz);
            ret.values = sp_product_container->values;
            ret.row_offsets = sp_product_container->row_offsets;
            ret.col_indices = sp_product_container->col_indices;

            // TODO: Do this without this hacky move.
            ret.descr.descr = out_descr.descr;
            out_descr.descr = nullptr;

            return ret;
        }

    protected:

        void Call_WorkEstimation(void* buffer1) {
            CUSPARSE_CALL(
                cusparseSpGEMM_workEstimation(ctx.sparseHandle, left_op, right_op,
                                              ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out_descr.descr,
                                              CUDA_R_32F, alg,
                                              mm_descr.descr, &buffer_size1, buffer1));
        }

        void Call_Compute(void* buffer2) {
            // MEMORY REQUIREMENT: the first invocation of cusparseSpGEMM_compute provides an upper bound of the memory required
            // for the computation that is generally several times larger of the actual memory used. The user can provide an
            // arbitrary buffer size bufferSize2 in the second invocation. If it is not sufficient, the routine will returns
            // CUSPARSE_STATUS_INSUFFICIENT_RESOURCES status.
            CUSPARSE_CALL(
                cusparseSpGEMM_compute(ctx.sparseHandle, left_op, right_op,
                                              ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out_descr.descr,
                                              CUDA_R_32F, alg,
                                              mm_descr.descr, &buffer_size2, buffer2));
        }

        void Call_Copy() {
            CUSPARSE_CALL(
                cusparseSpGEMM_copy(ctx.sparseHandle, left_op, right_op,
                                    ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out_descr.descr,
                                    CUDA_R_32F, alg, mm_descr.descr));
        }



        Shape GetOutputShape() {
            THROWSERT(left.n_cols == right.n_rows);
            return std::make_tuple(left.n_rows, right.n_cols);
        }

        void CreateOutDescr() {
            ctx.SetDevice();
            long n_rows, n_cols;
            std::tie(n_rows, n_cols) = out_shape;
            cusparseIndexType_t ind_type = ToCuSparseIndexType<IndT>::value;
            CUSPARSE_CALL(
                cusparseCreateCsr(&out_descr.descr, n_rows, n_cols, 0,
                                  sp_product_container->row_offsets,
                                  sp_product_container->col_indices,
                                  sp_product_container->values,
                                  ind_type, ind_type,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
            );
        }


    };
};




} // Ops
} // Cuda

