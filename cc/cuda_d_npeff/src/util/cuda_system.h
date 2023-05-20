#ifndef Z52D38597_FB8D_435C_8530_768E603235C1
#define Z52D38597_FB8D_435C_8530_768E603235C1
/* Utilities for querying and printing properties of the CUDA system. */
#include <stdio.h>
#include <cuda_runtime.h>

namespace CudaSystem {

    void printDevicesInfo() {
        // Look at the following for more infor on the cudaDeviceProp struct:
        // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        int device;
        for (device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            printf("Device %d has compute capability %d.%d.\n",
                   device, deviceProp.major, deviceProp.minor);
        }
    }
}

#endif
