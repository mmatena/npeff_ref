#pragma once

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


namespace Cuda {
namespace Ops {


class AddDenseDense {
    using DenseMatrix = Device::DenseMatrix;

protected:
    DeviceCudaContext& ctx;

    DenseMatrix& left;
    DenseMatrix& right;

    DenseMatrix& out;

    bool transpose_left;
    bool transpose_right;

public:
    AddDenseDense(
        DeviceCudaContext& ctx,
        DenseMatrix& left,
        DenseMatrix& right,
        DenseMatrix& out,
        bool transpose_left = false,
        bool transpose_right = false
    ) :
        ctx(ctx),
        left(left), right(right), out(out),
        transpose_left(transpose_left),
        transpose_right(transpose_right)
    {}

    void CallAsync() {
        ctx.SetDevice();
        CUBLAS_CALL(cublasSgeam(ctx.denseHandle,
            transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
            transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
            left.n_rows, left.n_cols,
            ctx.dev1f,
            left.data, left.n_rows,
            ctx.dev1f,
            right.data, right.n_rows,
            out.data, out.n_rows
        ));
    }

protected:
    void ValidateInputs() {
        THROWSERT(left.n_rows == out.n_rows);
        THROWSERT(right.n_rows == out.n_rows);
        THROWSERT(left.n_cols == out.n_cols);
        THROWSERT(right.n_cols == out.n_cols);
    }
};




} // Ops
} // Cuda

