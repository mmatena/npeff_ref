#pragma once
// Dense-dense matrix multiply.

#include <cstdint>
#include <tuple>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <gpu/macros.h>
#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>

#include <util/macros.h>

namespace npeff {
namespace gpu {
namespace ops {


namespace dndn_matmul_internal_ {

template<typename MatT>
std::tuple<int64_t, int64_t, int64_t> compute_mnk(
    MatT& left, MatT& right, bool transpose_left, bool transpose_right
) {
    int64_t m, n, k;

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

}  // dndn_matmul_internal_


class DnDnMatMul {
    using MNK = std::tuple<int64_t, int64_t, int64_t>;

    DeviceContext& ctx;

    DenseMatrix& left;
    DenseMatrix& right;
    DenseMatrix& out;

    const bool transpose_left;
    const bool transpose_right;

    float const* d_alpha;
    float const* d_beta;

    const MNK mnk;

public:
    DnDnMatMul(
        DeviceContext& ctx,
        DenseMatrix& left,
        DenseMatrix& right,
        DenseMatrix& out,
        bool transpose_left = false,
        bool transpose_right = false,
        // The following, if set, must be pointers on the device.
        float* d_alpha = nullptr,
        float* d_beta = nullptr
    ) :
        ctx(ctx),
        left(left), right(right), out(out),
        transpose_left(transpose_left), transpose_right(transpose_right),
        d_alpha(d_alpha == nullptr ? ctx.dev_1f : d_alpha),
        d_beta(d_beta == nullptr ? ctx.dev_0f : d_beta),
        mnk(dndn_matmul_internal_::compute_mnk(left, right, transpose_left, transpose_right))
    {}

    void call_async() {
        ctx.set_device();

        long m, n, k;
        std::tie(m, n, k) = mnk;

        CUBLAS_CALL(cublasSgemm(
            ctx.dense_handle,
            transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            d_alpha,
            (float*) left.data, left.n_rows,
            (float*) right.data, right.n_rows,
            d_beta,
            (float*) out.data, m
        ));
    }
};



class DnDnMatMul_SubMatrices {
    using MNK = std::tuple<int64_t, int64_t, int64_t>;

    DeviceContext& ctx;

    SubDenseMatrix& left;
    SubDenseMatrix& right;
    SubDenseMatrix& out;

    const bool transpose_left;
    const bool transpose_right;

    float const* d_alpha;
    float const* d_beta;

    const MNK mnk;

public:
    DnDnMatMul_SubMatrices(
        DeviceContext& ctx,
        SubDenseMatrix& left,
        SubDenseMatrix& right,
        SubDenseMatrix& out,
        bool transpose_left = false,
        bool transpose_right = false,
        // The following, if set, must be pointers on the device.
        float* d_alpha = nullptr,
        float* d_beta = nullptr
    ) :
        ctx(ctx),
        left(left), right(right), out(out),
        transpose_left(transpose_left), transpose_right(transpose_right),
        d_alpha(d_alpha == nullptr ? ctx.dev_1f : d_alpha),
        d_beta(d_beta == nullptr ? ctx.dev_0f : d_beta),
        mnk(dndn_matmul_internal_::compute_mnk(left, right, transpose_left, transpose_right))
    {}

    void call_async() {
        ctx.set_device();

        long m, n, k;
        std::tie(m, n, k) = mnk;

        CUBLAS_CALL(cublasSgemm(
            ctx.dense_handle,
            transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            d_alpha,
            left.get_data_ptr(), left.get_leading_dimension(),
            right.get_data_ptr(), right.get_leading_dimension(),
            d_beta,
            out.get_data_ptr(), out.get_leading_dimension()
        ));
    }
};




}  // ops
}  // gpu
}  // npeff
