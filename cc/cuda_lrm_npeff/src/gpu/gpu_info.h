#pragma once
// Gets information about the GPUs associated to the system.

#include <cstdint>

#include "./macros.h"

namespace npeff {
namespace gpu {

int64_t get_device_count() {
    int device_count;
    CUDA_CALL(cudaGetDeviceCount(&device_count));
    return device_count;
}

}  // gpu
}  // npeff
