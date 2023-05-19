#pragma once

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


// // Solves a linear system of equations using LU decomposition.
// // Solves AX = B for X, where the matrix A must be square. The
// // solution matrix X will override the matrix B.
// class LuSolve_InPlace {

//     DeviceContext& ctx;

//     DenseMatrix& A;
//     DenseMatrix& B;

//     bool transpose_A;

//     const int64_t n;
//     const int64_t lda;
//     float const* d_A;

//     const int64_t nrhs;
//     const int64_t ldb;
//     float const* d_B;

//     int* d_info = nullptr;
//     int* d_piv = nullptr;

//     float* d_work = nullptr;
//     int lwork = 0;

// public:
//     LuSolve_InPlace(
//         DeviceContext& ctx,
//         DenseMatrix& A,
//         DenseMatrix& B,
//         bool transpose_A = false
//     ) :
//         ctx(ctx), A(A), B(B), transpose_A(transpose_A),
//         n(A.n_rows), lda(A.n_rows), d_A(A.data),
//         nrhs(B.n_cols), ldb(B.n_rows), d_B(B.data)
//     {
//         validate_matrix_shapes();
//     }

//     ~LuSolve_InPlace() {
//         if(d_info != nullptr) { ctx.dfree(d_info); }
//         if(d_piv != nullptr) { ctx.dfree(d_piv); }
//         if(d_work != nullptr) { ctx.dfree(d_work); }
//     }

//     void set_up_sync() {
//         ctx.set_device();

//         // Allocate the known size device buffers.
//         d_info = ctx.dmalloc<int>(1);
//         d_piv = ctx.dmalloc<int>(n);

//         // Figure out the size of the workspace and allocate on the device.
//         CUSOLVER_CALL(cusolverDnSgetrf_bufferSize(ctx.cusolver_handle, n, n, (float*) d_A, lda, &lwork));
//         d_work = ctx.dmalloc<float>(lwork);
//     }

//     void call_sync() {
//         ctx.set_device();
//         int info = 0;

//         // Compute the LU decomposition.
//         CUSOLVER_CALL(cusolverDnSgetrf(ctx.cusolver_handle, n, n, (float*) d_A, lda, d_work, d_piv, d_info));
//         // Check if LU is successful or not.
//         ctx.copy_to_host_async(&info, d_info, 1);
//         ctx.synchronize_stream();
//         if (info < 0) {
//             std::printf("cusolverDnSgetrf: %d-th parameter is wrong \n", -info);
//             THROW;
//         } 
//         else if(info > 0) {
//             THROW_MSG("Singular matrix.");
//         }

//         // Solve AX = B.
//         auto op = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
//         CUSOLVER_CALL(cusolverDnSgetrs(
//             ctx.cusolver_handle, op, n, nrhs, (float*) d_A, lda, d_piv, (float*) d_B, ldb, d_info));
//         // Check if the solve was good or not.
//         ctx.copy_to_host_async(&info, d_info, 1);
//         ctx.synchronize_stream();
//         if (info < 0) {
//             std::printf("cusolverDnSgetrs: %d-th parameter is wrong \n", -info);
//             THROW;
//         }
//     }

// protected:
//     void validate_matrix_shapes() {
//         THROW_IF_FALSE(A.n_rows == A.n_cols);
//         THROW_IF_FALSE(A.n_rows == B.n_rows);
//     }
// };



// Solves a linear system of equations using LU decomposition.
// Solves AX = B for X, where the matrix A must be square. The
// solution matrix X will override the matrix B.
class LuSolve_InPlace {

    DeviceContext& ctx;

    DenseMatrix& A;
    DenseMatrix& B;

    bool transpose_A;

    cusolverDnParams_t params;

    const int64_t n;
    const int64_t lda;
    float* d_A;

    const int64_t nrhs;
    const int64_t ldb;
    float* d_B;

    int* d_info = nullptr;
    int64_t* d_piv = nullptr;

    float* d_work = nullptr;
    size_t d_lwork = 0;

    float* h_work = nullptr;
    size_t h_lwork = 0;

public:
    LuSolve_InPlace(
        DeviceContext& ctx,
        DenseMatrix& A,
        DenseMatrix& B,
        bool transpose_A = false
    ) :
        ctx(ctx), A(A), B(B), transpose_A(transpose_A),
        n(A.n_rows), lda(A.n_rows), d_A((float*) A.data),
        nrhs(B.n_cols), ldb(B.n_rows), d_B((float*) B.data)
    {
        validate_matrix_shapes();
    }

    ~LuSolve_InPlace() {
        if(d_info != nullptr) { ctx.dfree(d_info); }
        if(d_piv != nullptr) { ctx.dfree(d_piv); }
        if(d_work != nullptr) { ctx.dfree(d_work); }
    }

    void set_up_sync() {
        ctx.set_device();

        // Create and set-up the params
        CUSOLVER_CALL(cusolverDnCreateParams(&params));
        CUSOLVER_CALL(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

        // Allocate the known size device buffers.
        d_info = ctx.dmalloc<int>(1);
        d_piv = ctx.dmalloc<int64_t>(n);

        // Figure out the size of the workspace and allocate on the device.
        CUSOLVER_CALL(cusolverDnXgetrf_bufferSize(
            ctx.cusolver_handle, params, n, n, CUDA_R_32F, d_A, lda, CUDA_R_32F, &d_lwork, &h_lwork));

        THROW_IF_FALSE(h_lwork == 0);
        d_work = ctx.dmalloc<float>(d_lwork);
    }

    void call_sync() {
        ctx.set_device();
        int info = 0;

        // Compute the LU decomposition.
        CUSOLVER_CALL(cusolverDnXgetrf(
            ctx.cusolver_handle, params, n, n,
            CUDA_R_32F, d_A, lda, d_piv,
            CUDA_R_32F, d_work, d_lwork, h_work, h_lwork, d_info));

        // Check if LU is successful or not.
        ctx.copy_to_host_async(&info, d_info, 1);
        ctx.synchronize_stream();
        if (info < 0) {
            std::printf("cusolverDnXgetrf: %d-th parameter is wrong \n", -info);
            THROW;
        } else if(info > 0) {
            THROW_MSG("Singular matrix.");
        }

        // Solve AX = B.
        auto op = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUSOLVER_CALL(cusolverDnXgetrs(
            ctx.cusolver_handle, params, op, n, nrhs,
            CUDA_R_32F, d_A, lda, d_piv,
            CUDA_R_32F, d_B, ldb, d_info));

        // Check if the solve was good or not.
        ctx.copy_to_host_async(&info, d_info, 1);
        ctx.synchronize_stream();
        if (info < 0) {
            std::printf("cusolverDnXgetrs: %d-th parameter is wrong \n", -info);
            THROW;
        }
    }

protected:
    void validate_matrix_shapes() {
        THROW_IF_FALSE(A.n_rows == A.n_cols);
        THROW_IF_FALSE(A.n_rows == B.n_rows);
    }
};


}  // ops
}  // gpu
}  // npeff
