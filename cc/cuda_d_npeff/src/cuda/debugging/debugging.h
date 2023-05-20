#pragma once

#include <string>

#include <cuda_runtime.h>
#include <nvml.h>

#include <cuda/cuda_statuses.h>


namespace Cuda {
namespace Debug {


std::string ToHumanReadable(size_t n) {
    std::string suffix = "";
    if (n < 1000L) {
        // Nothing here

    } else if (n < 1000L * 1000L) {
        suffix = "K";
        n /= 1000L;

    } else if (n < 1000L * 1000L * 1000L) {
        suffix = "M";
        n /= 1000L * 1000L;

    } else if (n < 1000L * 1000L * 1000L * 1000L) {
        suffix = "G";
        n /= 1000L * 1000L * 1000L;

    } else {
        suffix = "T";
        n /= 1000L * 1000L * 1000L * 1000L;
    }
    return std::to_string(n) + suffix;
}



nvmlMemory_t GetMemoryInfo(int device) {
    // TODO: See if any of this leaks memory.
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(device, &dev);
    nvmlMemory_t mem_info;
    NVML_CALL(nvmlDeviceGetMemoryInfo(dev, &mem_info));
    return mem_info;
}

void LogMemoryUsage(int device) {
    nvmlMemory_t info = GetMemoryInfo(device);
    size_t free_mem = info.total - info.used;
    std::cout << ToHumanReadable(info.used) << " / " << ToHumanReadable(info.total)  << " [" << ToHumanReadable(free_mem) << " free]\n";
}



} // Debug
} // Cuda

