#pragma once

#include <tuple>
#include <memory>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <misc/common.h>

#include <cuda/cuda_context.h>
#include <cuda/descr.h>

#include <cuda/host/dense_matrix.h>
#include <cuda/host/sparse_matrix.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>


namespace Cuda {
namespace Ops {





class DenseDenseMatMul {
    using DenseMatrix = Device::DenseMatrix;
    using MNK = std::tuple<long, long, long>;

protected:
    DeviceCudaContext& ctx;

    DenseMatrix& left;
    DenseMatrix& right;
    DenseMatrix& out;

    const bool transpose_left;
    const bool transpose_right;

    const MNK mnk;

public:
    DenseDenseMatMul(
        DeviceCudaContext& ctx,
        DenseMatrix& left,
        DenseMatrix& right,
        DenseMatrix& out,
        bool transpose_left = false,
        bool transpose_right = false
    ) :
        ctx(ctx),
        left(left), right(right), out(out),
        transpose_left(transpose_left), transpose_right(transpose_right),
        mnk(GetMNK())
    {}

    void CallAsync() {
        ctx.SetDevice();

        long m, n, k;
        std::tie(m, n, k) = mnk;

        CUBLAS_CALL(cublasSgemm(
            ctx.denseHandle,
            transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            ctx.dev1f,
            (float*) left.GetData(), left.n_rows,
            (float*) right.GetData(), right.n_rows,
            ctx.dev0f,
            (float*) out.GetData(), m
        ));
    }

protected:

    MNK GetMNK() {
        long m, n, k;

        if (transpose_left) {
            m = left.n_cols;
            k = left.n_rows;
        } else {
            m = left.n_rows;
            k = left.n_cols;
        }

        if (transpose_right) {
            n = right.n_rows;
            if (k != right.n_cols) {
                THROW;
            }
        } else {
            n = right.n_cols;
            if (k != right.n_rows) {
                THROW;
            }
        }

        // TODO: Some more validation that the output is the correct shape?

        return std::make_tuple(m, n, k);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////


template <
    typename IndT,
    MatrixOrder order_right = MatrixOrder::COL_MAJOR,
    MatrixOrder order_out = order_right
>
class SparseDenseMatMul : public DeviceCudaContext::Freeable {
    using CsrMatrix = Device::CsrMatrix<IndT>;
    using InDnMatrix = Device::DnMatrix<order_right>;
    using OutDnMatrix = Device::DnMatrix<order_out>;

protected:
    DeviceCudaContext& ctx;

    CsrMatrix& left;
    InDnMatrix& right;
    OutDnMatrix& out;

    const bool transpose_left;
    const bool transpose_right;

    const cusparseSpMMAlg_t alg;


    const cusparseOperation_t left_op;
    const cusparseOperation_t right_op;


    void* buffer = nullptr;
    size_t buffer_size = 0;

public:
    SparseDenseMatMul(
        DeviceCudaContext& ctx,
        CsrMatrix& left,
        InDnMatrix& right,
        OutDnMatrix& out,
        bool transpose_left,
        bool transpose_right,
        cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT
    ) :
        ctx(ctx),
        left(left), right(right), out(out),
        transpose_left(transpose_left), transpose_right(transpose_right),
        alg(alg),
        left_op(transpose_left ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE),
        right_op(transpose_right ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        // TODO: Validate the shapes of left, right, and out.
    }

    void SetUpAsync() {
        // NOTE: The CreateBuffer step will have a synchronizing memory allocation.
        ctx.SetDevice();
        CreateBuffer();
        Preprocess();
    }

    void CallAsync() {
        ctx.SetDevice();
        CUSPARSE_CALL(
            cusparseSpMM(ctx.sparseHandle,
                         left_op, right_op,
                         ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out.descr.descr, CUDA_R_32F,
                         alg, buffer)
        );
    }

    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        return {buffer};
    }

protected:

    void CreateBuffer() {
        ctx.SetDevice();

        THROWSERT(buffer == nullptr);

        CUSPARSE_CALL(
            cusparseSpMM_bufferSize(ctx.sparseHandle,
                         left_op, right_op,
                         ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out.descr.descr, CUDA_R_32F,
                         alg, &buffer_size)
        );
        ctx.SynchronizeStream();
        buffer = ctx.dmalloc<void>(buffer_size);
    }

    void Preprocess() {
        ctx.SetDevice();
        CUSPARSE_CALL(
            cusparseSpMM_preprocess(ctx.sparseHandle,
                         left_op, right_op,
                         ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out.descr.descr, CUDA_R_32F,
                         alg, buffer)
        );
    }
};


/////////////////////////////////////////////////////////////////////////////////////////


// NOTE: The cusparse implementation does not support any transposing and only int32 indices.
// I can probably implemented some hacks to get around those retrictions.
template<typename IndT>
class SparseSparseMatMul_SingleUse : public DeviceCudaContext::Freeable {
    using CsrMatrix = Device::CsrMatrix<IndT>;
    using Shape = std::tuple<long, long>;

protected:
    DeviceCudaContext& ctx;

    CsrMatrix& left;
    CsrMatrix& right;


    UniqueDescr<cusparseSpGEMMDescr_t> mm_descr;

    const Shape out_shape;
    UniqueDescr<cusparseSpMatDescr_t> out_descr;
    std::unique_ptr<CsrMatrix> out;


    // Only these are currently supported by cusparse.
    const cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;
    const cusparseOperation_t left_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t right_op = CUSPARSE_OPERATION_NON_TRANSPOSE;

    int step = 0;

    size_t buffer_size1 = 0;
    void *buffer1 = nullptr;

    size_t buffer_size2 = 0;
    void *buffer2 = nullptr;

public:
    SparseSparseMatMul_SingleUse(
        DeviceCudaContext& ctx,
        CsrMatrix& left,
        CsrMatrix& right
    ) :
        ctx(ctx), right(right), left(left), out_shape(GetOutputShape())
    {
        ctx.SetDevice();
        CreateOutDescr();
        CUSPARSE_CALL(cusparseSpGEMM_createDescr(&mm_descr.descr));
    }

    // TODO: IDK if stuff could get f*cky-wucky with the mm_descr and out_descr attributes.
    // 
    // Use smart pointers [or more likely custom smart-pointer-like classes] for the descrs stuff.
    SparseSparseMatMul_SingleUse(SparseSparseMatMul_SingleUse&& o) = default;

    static std::vector<std::unique_ptr<CsrMatrix>> Call(std::vector<SparseSparseMatMul_SingleUse<IndT>>& ops) {
        while(true) {
            bool ands = true;
            bool ors = false;
            for(auto& op : ops) {
                bool res = op.Call_Step();
                ands &= res;
                ors |= res;
            }
            if(ands != ors) {
                std::cout << "Not all ops completed at the same step;\n";
                THROW;
            }
            if(!ands) { break; }
        }

        // TODO: See if this causes some weirdness with copies and whatnot.
        std::vector<std::unique_ptr<CsrMatrix>> ret;
        for (auto& op : ops) {
            ret.push_back(std::move(op.out));
        }
        return ret;
    }

    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        return {buffer1, buffer2};
    }

protected:

    // Runs a step consisting of the next largest block of contiguous async operations
    // or the next largest block of contiguous sync operations.
    // 
    // Returns true if there is another step remaining. Returns false if finished.
    bool Call_Step() {
        ctx.SetDevice();
        step++;
        switch(step - 1) {
            case 0: Step_0(); return true;
            case 1: Step_1(); return true;
            case 2: Step_2(); return true;
            case 3: Step_3(); return true;
            case 4: Step_4(); return true;
            case 5: Step_5(); return true;
            case 6: Step_6(); return true;
            case 7: Step_7(); return true;
            case 8: Step_8(); return false;
            default:
                std::cout << "Invalid state: step " << step - 1 << " not valid.\n";
                THROW;
        }
    }


    void Step_0() {
        std::cout << buffer_size1 << ", " << buffer1 << "  1\n";
        CUSPARSE_CALL(
            cusparseSpGEMM_workEstimation(ctx.sparseHandle, left_op, right_op,
                                          ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out_descr.descr,
                                          CUDA_R_32F, alg,
                                          mm_descr.descr, &buffer_size1, buffer1));
    }

    void Step_1() {
        std::cout << buffer_size1 << "  1\n";
        ctx.SynchronizeStream();
        buffer1 = ctx.dmalloc<void>(buffer_size1);
    }

    void Step_2() {
        // buffer1 should have been set to a non-null value.
        Step_0();
    }

    void Step_3() {
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

    void Step_4() {
        std::cout << buffer_size2 << "  2\n";
        // buffer_size2 /= 4;
        ctx.SynchronizeStream();
        buffer2 = ctx.dmalloc<void>(buffer_size2);
    }

    void Step_5() {
        // buffer2 should have been set to a non-null value.
        Step_3();
    }

    void Step_6() {
        int64_t out_n_rows, out_n_cols, out_nnz;
        CUSPARSE_CALL(
            cusparseSpMatGetSize(out_descr.descr, &out_n_rows, &out_n_cols, &out_nnz));

        // Note that out_descr is not updated to reflect the new pointers. We will
        // use out.descr for now on.
        // out = std::move(CsrMatrix(out_n_rows, out_n_cols, out_nnz));
        out = std::unique_ptr<CsrMatrix>(new CsrMatrix(out_n_rows, out_n_cols, out_nnz));
        out->AllocMemory(ctx);
    }

    void Step_7() {
        CUSPARSE_CALL(
            cusparseSpGEMM_copy(ctx.sparseHandle, left_op, right_op,
                                ctx.dev1f, left.descr.descr, right.descr.descr, ctx.dev0f, out->descr.descr,
                                CUDA_R_32F, alg, mm_descr.descr));
    }

    void Step_8() {
        ctx.dfree(buffer1);
        ctx.dfree(buffer2);
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
                              nullptr, nullptr, nullptr,
                              ind_type, ind_type,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        );
    }



};



} // Ops
} // Cuda

