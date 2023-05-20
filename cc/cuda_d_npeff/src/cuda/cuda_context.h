#pragma once
/**
 * Basically a utility class to hold some CUDA-related stuff.
 */
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusparse.h>
#include "nccl.h"

#include <cuda/cuda_statuses.h>



struct DeviceCudaContext {
    int device;

    cudaStream_t stream;

    cusparseHandle_t sparseHandle;
    cublasHandle_t denseHandle;

    curandGenerator_t randGen;
    long randGenSeed;

    ncclComm_t comm;

    float *dev0f, *dev1f;

    DeviceCudaContext(int device, ncclComm_t comm, long randGenSeed) :
            device(device), comm(comm), randGenSeed(randGenSeed) {
        CUDA_CALL(cudaSetDevice(device));

        CUDA_CALL(cudaStreamCreate(&stream));

        CUBLAS_CALL(cublasCreate(&denseHandle));
        CUBLAS_CALL(cublasSetPointerMode(denseHandle, CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_CALL(cublasSetStream(denseHandle, stream));

        CUSPARSE_CALL(cusparseCreate(&sparseHandle));
        CUSPARSE_CALL(cusparseSetPointerMode(sparseHandle, CUSPARSE_POINTER_MODE_DEVICE));
        CUSPARSE_CALL(cusparseSetStream(sparseHandle, stream));

        CURAND_CALL(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randGen, randGenSeed));
        CURAND_CALL(curandSetStream(randGen, stream));

        // NOTE: See if initializing these at the time of construction and doing them
        // async leads to any funny business.
        float f01[] = {0.0f, 1.0f};
        dev0f = dmalloc<float>(2);
        dev1f = dev0f + 1;
        CopyToDeviceAsync(dev0f, f01, 2);
    }

    ~DeviceCudaContext() {
        cudaSetDevice(device);

        for (auto ptr : dmalloced) {
            cudaFree(ptr);
        }

        curandDestroyGenerator(randGen);
        cusparseDestroy(sparseHandle);
        cublasDestroy(denseHandle);
        cudaStreamDestroy(stream);
    }

    template<typename T>
    T* dmalloc(size_t n_elements, bool managed = true) {
        T* ret;
        SetDevice();
        CUDA_CALL(cudaMalloc(&ret, sizeof(T) * n_elements));
        if (managed && ret != nullptr) {
            dmalloced.push_back((void *) ret);
        }
        return ret;
    }

    template<typename T>
    void dfree(T* ptr) {
        SetDevice();
        if (ptr == nullptr) { return; }
        for(long i=0;i<dmalloced.size();i++) {
            if ((void*) ptr == dmalloced[i]) {
                CUDA_CALL(cudaFree(ptr));
                dmalloced.erase(dmalloced.begin() + i);
                return;
            }
        }
        std::cout << "Pointer not found in the device allocated memory.\n";
        THROW;
    }

    void SetDevice() {
        CUDA_CALL(cudaSetDevice(device));
    }


    template<typename T>
    void CopyToDeviceAsync(T* device, T* host, size_t n_elements) {
        SetDevice();
        CUDA_CALL(
            cudaMemcpyAsync(device, host, sizeof(T) * n_elements, cudaMemcpyHostToDevice, stream)
        );
    }

    template<typename T>
    void CopyToHostAsync(T* host, T* device, size_t n_elements) {
        SetDevice();
        CUDA_CALL(
            cudaMemcpyAsync(host, device, sizeof(T) * n_elements, cudaMemcpyDeviceToHost, stream)
        );
    }


    template<typename T>
    void CopyOnDeviceAsync(T* dst, T* src, size_t n_elements) {
        SetDevice();
        CUDA_CALL(
            cudaMemcpyAsync(dst, src, sizeof(T) * n_elements, cudaMemcpyDeviceToDevice, stream)
        );
    }

    template<typename DeviceT, typename HostT>
    void CopyToDeviceAsync(DeviceT& device, HostT& host);

    template<typename HostT, typename DeviceT>
    void CopyToHostAsync(HostT& host, DeviceT& device);


    void SynchronizeStream() {
        SetDevice();
        CUDA_CALL(cudaStreamSynchronize(stream));
    }


    struct Freeable {
        virtual std::vector<void*> GetDeviceAllocs() = 0;
    };


    void FreeDeviceAllocs(Freeable& f) {
        for (void* ptr : f.GetDeviceAllocs()) { dfree(ptr); }
    }

protected:
    std::vector<void*> dmalloced;

};


template<>
void* DeviceCudaContext::dmalloc<void>(size_t n_elements, bool managed) {
    return (void *) dmalloc<char>(n_elements, managed);
}





struct HostCudaContext {
    int n_devices;

    long randGenSeed;
    ncclComm_t* comms = nullptr;
    std::vector<DeviceCudaContext> device_contexts;

    HostCudaContext(int n_devices, long randGenSeed) :
            n_devices(n_devices),
            randGenSeed(randGenSeed)
    {
        comms = new ncclComm_t[n_devices];
        NCCL_CALL(ncclCommInitAll(comms, n_devices, NULL));

        for (int i=0;i<n_devices;i++) {
            device_contexts.emplace_back(i, comms[i], randGenSeed + i);
        }
    }

    ~HostCudaContext() {
        for (int i=0; i<n_devices; i++) {
            ncclCommDestroy(comms[i]);
        }
        delete[] comms;
    }

    void SynchronizeStreams() {
        for(auto& dc : device_contexts) { dc.SynchronizeStream(); }
    }

};

