#pragma once
#include <cuda_runtime.h>


namespace Cuda {

    int GetDeviceCount() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        return device_count;
    }

} // Cuda

