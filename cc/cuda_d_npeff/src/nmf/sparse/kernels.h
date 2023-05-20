#ifndef Z3BBBC9F8_EFF5_476D_89F3_35879CBE1FFE_H
#define Z3BBBC9F8_EFF5_476D_89F3_35879CBE1FFE_H
#include <cmath>
#include <cuda_runtime.h>


// __global__
// void kernelNmfMuUpdate(long n, float* F, const float* numer, const float* denom, float eps, float maxOutputMag) {
//     // F *= numer / (denom + eps)
//     // n is equal to the number of entries of F.
//     long index = blockIdx.x * blockDim.x + threadIdx.x;
//     long stride = blockDim.x * gridDim.x;

//     for (long i = index; i < n; i += stride) {
//         F[i] *= numer[i] / (denom[i] + eps);
//         // F[i] = fminf(maxOutputMag, F[i] * numer[i] / (denom[i] + eps));
//         // F[i] = fmaxf(0.0f, fminf(maxOutputMag, F[i] * numer[i] / (denom[i] + eps)));

//         // F[i] *= fmaxf(0.5f, fminf(2.0f, numer[i] / (denom[i] + eps)));

//     }
// }

__global__
void kernelNmfMuUpdate(long n, float* F, const float* numer, const float* denom, float eps) {
    // F *= numer / (denom + eps)
    // n is equal to the number of entries of F.
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = index; i < n; i += stride) {
        F[i] *= numer[i] / (denom[i] + eps);
    }
}


__global__
void kernelVecSub(long n, float* x, float* y) {
    // x -> x - y
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = index; i < n; i += stride) {
        x[i] -= y[i];
    }
}

__global__
void kernelRescale(long n, float* x, float factor) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = index; i < n; i += stride) {
        x[i] *= factor;
    }
}


#endif
