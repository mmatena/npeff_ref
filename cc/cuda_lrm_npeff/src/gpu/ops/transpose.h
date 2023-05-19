#pragma once
// Matrix transpose.

#include <cstdint>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <gpu/macros.h>
#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>

#include <util/macros.h>

namespace npeff {
namespace gpu {
namespace ops {


// Transpose of a dense matrix.
class DnTranspose {
    DeviceContext& ctx;
    DenseMatrix& in;
    DenseMatrix& out;

public:
    DnTranspose(
        DeviceContext& ctx,
        DenseMatrix& in,
        DenseMatrix& out
    ) :
        ctx(ctx), in(in), out(out)
    {
        THROW_IF_FALSE(in.n_rows == out.n_cols);
        THROW_IF_FALSE(in.n_cols == out.n_rows);
    }


    void call_async() {
        ctx.set_device();
        CUBLAS_CALL(cublasSgeam(
            ctx.dense_handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            out.n_rows, out.n_cols,
            ctx.dev_1f,
            (float*) in.data, in.n_rows,
            ctx.dev_0f,
            (float*) in.data, in.n_rows,
            (float*) out.data, out.n_rows
        ));
    }
};


// Transpose of a dense matrix using submatrices.
class DnTranspose_SubMatrices {
    DeviceContext& ctx;
    SubDenseMatrix& in;
    SubDenseMatrix& out;

    float const* d_alpha;
    float const* d_beta;

public:
    DnTranspose_SubMatrices(
        DeviceContext& ctx,
        SubDenseMatrix& in,
        SubDenseMatrix& out,
        // The following, if set, must be pointers on the device.
        float* d_alpha = nullptr,
        float* d_beta = nullptr
    ) :
        ctx(ctx), in(in), out(out),
        d_alpha(d_alpha == nullptr ? ctx.dev_1f : d_alpha),
        d_beta(d_beta == nullptr ? ctx.dev_0f : d_beta)
    {
        THROW_IF_FALSE(in.n_rows == out.n_cols);
        THROW_IF_FALSE(in.n_cols == out.n_rows);
    }

    void call_async() {
        ctx.set_device();
        CUBLAS_CALL(cublasSgeam(
            ctx.dense_handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            out.n_rows, out.n_cols,
            d_alpha,
            in.get_data_ptr(), in.get_leading_dimension(),
            d_beta,
            in.get_data_ptr(), in.get_leading_dimension(),
            out.get_data_ptr(), out.get_leading_dimension()
        ));
    }
};


}  // ops
}  // gpu
}  // npeff

