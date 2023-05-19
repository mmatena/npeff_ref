#pragma once
// Frobenius inner product between two dense matrices.

#include <climits>
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



class FrobeniousInnerProduct {
    static const int64_t MAX_N_ENTRIES_ = INT_MAX;

    DeviceContext& ctx;
    DenseMatrix& left;
    DenseMatrix& right;

    // Pointer on device to where we write the values of the
    // Frobenius inner product.
    float const* d_result;

    const int64_t n_entries;

public:
    FrobeniousInnerProduct(
        DeviceContext& ctx,
        DenseMatrix& left,
        DenseMatrix& right,
        float* d_result
    ) : 
        ctx(ctx), left(left), right(right),
        d_result(d_result),
        n_entries(left.n_entries)
    {
        if(n_entries > MAX_N_ENTRIES_) {
            // TODO: This can be solved by "chunking" the arrays.
            THROW_MSG("Only supporting up to INT_MAX entries due to cublas limitation.");
        }
        THROW_IF_FALSE(left.n_rows == right.n_rows);
        THROW_IF_FALSE(left.n_cols == right.n_cols);
    }

    void call_async() {
        ctx.set_device();
        CUBLAS_CALL(cublasSdot(
            ctx.dense_handle, n_entries,
            (float*) left.data, 1,
            (float*) right.data, 1,
            (float*) d_result
        ));
    }
};


}  // ops
}  // gpu
}  // npeff
