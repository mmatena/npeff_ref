#pragma once

#include <iostream>
#include <curand.h>
#include "nccl.h"
#include <nvml.h>

#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <misc/macros.h>


#define CUDA_CALL(x)                                                           \
{                                                                              \
    cudaError_t status = (x);                                                  \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d in file %s with error: %s (%d)\n",             \
               __LINE__, __FILE__, cudaGetErrorString(status), status);                  \
        throw;                                                                 \
    }                                                                          \
}

#define CUSPARSE_CALL(x)                                                      \
{                                                                              \
    cusparseStatus_t status = (x);                                             \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d in file %s with error: %s (%d)\n",         \
               __LINE__, __FILE__, cusparseGetErrorString(status), status);              \
        throw;                                                                 \
    }                                                                          \
}

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { std::cout << x << "\n"; THROW; } } while(0)
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { std::cout << x << "\n"; THROW; } } while(0)
#define NCCL_CALL(x) do { if((x)!=ncclSuccess) { std::cout << x << "\n"; THROW; } } while(0)
#define NVML_CALL(x) do { if((x)!=NVML_SUCCESS) { std::cout << x << "\n"; THROW; } } while(0)


