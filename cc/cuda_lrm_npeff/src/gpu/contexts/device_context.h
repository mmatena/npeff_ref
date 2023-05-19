#pragma once
// Context for a single device.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusparse.h>
#include <cusolverDn.h>
#include "nccl.h"

#include <gpu/macros.h>
#include <util/macros.h>

namespace npeff {
namespace gpu {


class DeviceContext {
protected:
    int64_t rand_gen_seed;
    std::vector<void*> dmalloced;

public:
    int device;

    ncclComm_t comm;

    cudaStream_t stream = nullptr;
    cusparseHandle_t sparse_handle = nullptr;
    cublasHandle_t dense_handle = nullptr;
    cusolverDnHandle_t cusolver_handle = nullptr;
    curandGenerator_t rand_gen = nullptr;

    // Pointers to floats containing 0 and 1 stored on the device.
    float* dev_0f = nullptr;
    float* dev_1f = nullptr;

    DeviceContext(int device, ncclComm_t comm, int64_t rand_gen_seed) :
        device(device), comm(comm), rand_gen_seed(rand_gen_seed)
    {}

    DeviceContext(int device, int64_t rand_gen_seed) :
        device(device), comm(nullptr), rand_gen_seed(rand_gen_seed)
    {}

    ~DeviceContext();

    // Must be called before interacting with the instance.
    void initialize();

    // Synchronizes the stream.
    void synchronize_stream(); 


    // Alloc memory on the device. If the managed argument is true,
    // then this will store the returned pointer in the dmalloced
    // vector attribute of this.
    template<typename T>
    T* dmalloc(size_t n_elements, bool managed = true) {
        T* ret;
        set_device();
        CUDA_CALL(cudaMalloc(&ret, sizeof(T) * n_elements));
        if (managed && ret != nullptr) {
            dmalloced.push_back((void *) ret);
        }
        return ret;
    }

    // The ptr must be managed by this instance, i.e. stored within
    // the dmalloced vector.
    template<typename T>
    void dfree(T* ptr) {
        set_device();
        if (ptr == nullptr) { return; }
        for(long i=0;i<dmalloced.size();i++) {
            if ((void*) ptr == dmalloced[i]) {
                CUDA_CALL(cudaFree(ptr));
                dmalloced.erase(dmalloced.begin() + i);
                return;
            }
        }
        THROW_MSG("Pointer not found in the device allocated memory.");
    }

    // Copies memory from the host to the device asynchronously.
    template<typename T>
    void copy_to_device_async(T* device, T* host, size_t n_elements) {
        set_device();
        CUDA_CALL(
            cudaMemcpyAsync(device, host, sizeof(T) * n_elements, cudaMemcpyHostToDevice, stream)
        );
    }

    // Copies memory from the device to the host asynchronously.
    template<typename T>
    void copy_to_host_async(T* host, T* device, size_t n_elements) {
        set_device();
        CUDA_CALL(
            cudaMemcpyAsync(host, device, sizeof(T) * n_elements, cudaMemcpyDeviceToHost, stream)
        );
    }

    // Specialize this template to allow custom behavior for objects.
    template<typename DeviceT, typename HostT>
    void copy_to_device_async(DeviceT& device, HostT& host);

    // Specialize this template to allow custom behavior for objects.
    template<typename HostT, typename DeviceT>
    void copy_to_host_async(HostT& host, DeviceT& device);

    void set_device();

};


// Forward declaration of this template specialization. This prevents
// the compiler from outputting some annoying spurious warnings.
template<>
void* DeviceContext::dmalloc<void>(size_t n_elements, bool managed);


}  // gpu
}  // npeff
