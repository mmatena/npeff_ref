#pragma once
// Dense-dense matrix multiply.

#include <cstdint>
#include <tuple>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <gpu/macros.h>
#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>
#include <gpu/containers/sparse_matrix.h>

#include <util/macros.h>

namespace npeff {
namespace gpu {
namespace ops {


template <typename IndT>
class Base_SpDnMatMul {

    DeviceContext& ctx;

    const cusparseOperation_t left_op;
    const cusparseOperation_t right_op;

    const cusparseSpMMAlg_t alg;

    const cusparseSpMatDescr_t left_descr;
    const cusparseDnMatDescr_t right_descr;
    const cusparseDnMatDescr_t out_descr;

    float const* d_alpha;
    float const* d_beta;

    void* buffer = nullptr;
    size_t buffer_size = 0;

public:
    Base_SpDnMatMul(
        DeviceContext& ctx,
        // 
        cusparseSpMatDescr_t left_descr,
        cusparseDnMatDescr_t right_descr,
        cusparseDnMatDescr_t out_descr,
        // 
        cusparseOperation_t left_op,
        cusparseOperation_t right_op,
        //
        cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT,
        // 
        // The following, if set, must be pointers on the device.
        float* d_alpha = nullptr,
        float* d_beta = nullptr
    ) :
        ctx(ctx),
        left_descr(left_descr), right_descr(right_descr), out_descr(out_descr),
        left_op(left_op), right_op(right_op),
        alg(alg),
        d_alpha(d_alpha == nullptr ? ctx.dev_1f : d_alpha),
        d_beta(d_beta == nullptr ? ctx.dev_0f : d_beta)
    {}

    ~Base_SpDnMatMul() {
        if(buffer != nullptr) { ctx.dfree(buffer); }
    }

    void set_up_async() {
        ctx.set_device();
        create_buffer();
        preprocess();
    }

    void call_async() {
        ctx.set_device();

        CUSPARSE_CALL(
            cusparseSpMM(ctx.sparse_handle,
                         left_op, right_op,
                         d_alpha, left_descr, right_descr, d_beta, out_descr, CUDA_R_32F,
                         alg, buffer)
        );
    }

protected:

    void create_buffer() {
        ctx.set_device();

        THROW_IF_FALSE(buffer == nullptr);

        CUSPARSE_CALL(
            cusparseSpMM_bufferSize(ctx.sparse_handle,
                         left_op, right_op,
                         d_alpha, left_descr, right_descr, d_beta, out_descr, CUDA_R_32F,
                         alg, &buffer_size)
        );
        ctx.synchronize_stream();

        buffer = ctx.dmalloc<void>(buffer_size);
    }

    void preprocess() {
        ctx.set_device();
        CUSPARSE_CALL(
            cusparseSpMM_preprocess(ctx.sparse_handle,
                         left_op, right_op,
                         d_alpha, left_descr, right_descr, d_beta, out_descr, CUDA_R_32F,
                         alg, buffer)
        );
    }
};


// Sparse-Dense multiplication.
template <typename IndT>
class SpDnMatMul : public Base_SpDnMatMul<IndT> {
public:
    SpDnMatMul(
        DeviceContext& ctx,
        CsrMatrix<IndT>& left,
        DenseMatrix& right,
        DenseMatrix& out,
        bool transpose_left = false,
        bool transpose_right = false,
        cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT,
        float* d_alpha = nullptr,
        float* d_beta = nullptr
    ) : Base_SpDnMatMul<IndT>(
        ctx,
        left.descr,
        right.get_col_major_descr(),
        out.get_col_major_descr(),
        transpose_left ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        transpose_right ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        alg,
        d_alpha,
        d_beta
    ) {}
};


// Dense-Sparse multiplication.
template <typename IndT>
class DnSpMatMul : public Base_SpDnMatMul<IndT> {
public:
    DnSpMatMul(
        DeviceContext& ctx,
        DenseMatrix& left,
        CsrMatrix<IndT>& right,
        DenseMatrix& out,
        bool transpose_left = false,
        bool transpose_right = false,
        cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT,
        float* d_alpha = nullptr,
        float* d_beta = nullptr
    ) : Base_SpDnMatMul<IndT>(
        ctx,
        right.descr,
        left.get_transpose_row_major_descr(),
        out.get_transpose_row_major_descr(),
        // NOTE: The transpose of the sparse matrix is flipped on purpose.
        transpose_right ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
        transpose_left ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
        alg,
        d_alpha,
        d_beta
    ) {}
};


}  // ops
}  // gpu
}  // npeff
