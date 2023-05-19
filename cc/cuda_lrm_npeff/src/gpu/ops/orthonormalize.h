#pragma once
// Orthonormalizes a matrix.

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <gpu/macros.h>
#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>

#include <util/macros.h>

namespace npeff {
namespace gpu {
namespace ops {


// Orthonormalizes the columns of a matrix via a QR decomposition.
class Orthonormalize_InPlace {
    DeviceContext& ctx;
    DenseMatrix& mat;

    const int64_t m;
    const int64_t n;
    const int64_t lda;
    float const* d_A;

    int* d_info = nullptr;
    float* d_tau = nullptr;

    float* d_work = nullptr;
    int lwork = 0;

public:
    Orthonormalize_InPlace(DeviceContext& ctx, DenseMatrix& mat) :
        ctx(ctx), mat(mat), m(mat.n_rows), n(mat.n_cols), lda(mat.n_rows),
        d_A(mat.data)
    {
        THROW_IF_FALSE(m >= n);
    }

    ~Orthonormalize_InPlace() {
        if(d_tau != nullptr) { ctx.dfree(d_tau); }
        if(d_info != nullptr) { ctx.dfree(d_info); }
        if(d_work != nullptr) { ctx.dfree(d_work); }
    }

    void set_up_sync() {
        ctx.set_device();

        // Allocate the known size device buffers.
        d_info = ctx.dmalloc<int>(1);
        d_tau = ctx.dmalloc<float>(n);

        // Figure out the size of the workspace and allocate on the device.
        int lwork_geqrf = 0;
        CUSOLVER_CALL(cusolverDnSgeqrf_bufferSize(ctx.cusolver_handle, m, n, (float*) d_A, lda, &lwork_geqrf));
        int lwork_orgqr = 0;
        CUSOLVER_CALL(cusolverDnSorgqr_bufferSize(ctx.cusolver_handle, m, n, n, (float*) d_A, lda, d_tau, &lwork_orgqr));

        lwork = std::max(lwork_geqrf, lwork_orgqr);
        d_work = ctx.dmalloc<float>(lwork);
    }

    void call_sync() {
        ctx.set_device();
        int info = 0;

        // Compute QR factorization.
        CUSOLVER_CALL(cusolverDnSgeqrf(
            ctx.cusolver_handle, m, n, (float*) d_A, lda, d_tau, d_work, lwork, d_info));
        // Check if QR is successful or not.
        ctx.copy_to_host_async(&info, d_info, 1);
        ctx.synchronize_stream();
        if (info < 0) {
            std::printf("cusolverDnSgeqrf: %d-th parameter is wrong \n", -info);
            THROW;
        }

        // Compute Q.
        CUSOLVER_CALL(cusolverDnSorgqr(ctx.cusolver_handle, m, n, n, (float*) d_A, lda, d_tau, d_work, lwork, d_info));
        // Check if QR is good or not.
        ctx.copy_to_host_async(&info, d_info, 1);
        ctx.synchronize_stream();
        if (info < 0) {
            std::printf("cusolverDnDorgqr: %d-th parameter is wrong \n", -info);
            THROW;
        }
    }

};


}  // ops
}  // gpu
}  // npeff

