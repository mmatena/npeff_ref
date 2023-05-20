#pragma once

#include <cmath>
#include <cuda_runtime.h>


// Have 1d (or effectively 1d) blocks, 2-d grid with ...
// 
// Or maybe just go for it. Memory reads will probably be cached.


// NOTE: These probably assume that we are sparsifying W.

__global__
void kernel_computeInitialS(long vecSize, long rank, float* F, float* sumPtr, float L1) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    long n = rank * vecSize;

    for (long i = index; i < n; i += stride) {
        float summedVec = sumPtr[i / vecSize];
        F[i] += (L1 - summedVec) / (float) vecSize;
    }
}


__global__
void kernel_setFloatValue(long n, float* F, float value) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    for (long i = index; i < n; i += stride) {
        F[i] = value;
    }
}


__global__
void kernel_computeAlpha(
    long rank,
    float* SS, float* SY, float* YY,
    float L1, float sq_L2,
    float* alpha,
    float eps = 1e-9
) {
    // TODO: Need some way of having "solved" mask.
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;


    // TODO: If this is correct, just pass L2 intead of sq_L2.
    float L2 = sqrtf(sq_L2);


    for (long i = index; i < rank; i += stride) {
        // TODO: Double check this.
        float beta = L1 / YY[i];

        float m2 = beta * beta * YY[i];
        float ms = beta * SY[i];

        // float m_s_2 = beta * beta * YY[i] + SS[i] - 2.0f * beta * SY[i];
        float m_s_2 = fmaxf(eps, beta * beta * YY[i] + SS[i] - 2.0f * beta * SY[i]);

        alpha[i] = (L2 * sqrtf(m_s_2) + m2 - ms) / m_s_2;
    }
}


__global__
void kernel_updateSAfterAlpha(
    long vecSize, long rank,
    float* S, float* Y, float* devActiveMaskBufferPtr,
    float* alpha, float* YY,
    float* activeMask,
    float L1
) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    long n = rank * vecSize;

    for (long i = index; i < n; i += stride) {
        if (activeMask[i / vecSize]) {
            float m = Y[i] * L1 / YY[i / vecSize];
            float s = m + alpha[i / vecSize] * (S[i] - m);
            S[i] = s;
            devActiveMaskBufferPtr[i] = (float) (s < 0.0f);
        }
    }
}


__global__
void kernel_updateYAndS(
    long vecSize, long rank,
    float* Y, float* S,
    float* activeMask
) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    long n = rank * vecSize;

    // TODO: I think this is effectively a no-op for columns that
    // are not active, but double check that.
    for (long i = index; i < n; i += stride) {
        if (activeMask[i / vecSize]) {
            if(Y[i] && S[i] < 0.0f) {
                Y[i] = 0.0f;
            }

            if(!Y[i]) {
                S[i] = 0.0f;
            }
        }
    }
}


__global__
void kernel_updateSLastStep(
    long vecSize, long rank,
    float* S, float* Y,
    float* summedS, float* summedY,
    float* activeMask,
    float L1
) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    long n = rank * vecSize;
     for (long i = index; i < n; i += stride) {
        if (activeMask[i / vecSize] && Y[i]) {
            float sum_s = summedS[i / vecSize];
            float sum_y = summedY[i / vecSize];
            float c = (sum_s - L1) / sum_y;
            S[i] -= c;
        }
     }
}

