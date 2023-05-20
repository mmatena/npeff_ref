#ifndef CCF9923B_BB9C_4FF5_8937_93A635F807CF
#define CCF9923B_BB9C_4FF5_8937_93A635F807CF

#include <algorithm>
#include <iostream>

#include <curand.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nccl.h"

#include <util/cuda_statuses.h>
#include <util/matrices.h>


// NOTE: This is for private use within MuNMF.
class DeviceMatrix {
public:
    long n_rows;
    long n_cols;
    float* devPtr;

    DeviceMatrix() {}

    DeviceMatrix(long n_rows, long n_cols) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->devPtr = nullptr;
    }

    DeviceMatrix(long n_rows, long n_cols, float* devPtr) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->devPtr = devPtr;
    }

    long nEntries() {
        return n_rows * n_cols;
    }

    long sizeInBytes() {
        return sizeof(float) * nEntries();
    }

};



///////////////////////////////////////////////////////////////////////////////
// Kernels and stuff.


// TODO: This can maybe be improved through the use of shared memory.
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


///////////////////////////////////////////////////////////////////////////////




// NOTE: This is for private use within MuNMF.
class DeviceContext {
public:
    int nDevices;
    int device;
    long seed;

    long n_rows;
    long n_cols;
    long n_cols_slice;
    long rank;

    float eps;

    cudaStream_t stream;
    cublasHandle_t handle;
    curandGenerator_t randGen;

    // Owned by the coordinator, not this class.
    ncclComm_t* comm;

    DeviceMatrix devA;
    DeviceMatrix devWH;

    DeviceMatrix devW;
    DeviceMatrix devH;

    DeviceMatrix devFF;
    float* devFFGPtr;
    float* devAFPtr;

    float localLoss;
    float *devLossPtr;

    // TODO: See if there are better ways of setting scalar parameters for cuBLAS.
    float* dev1f;
    float* dev0f;

    DeviceContext(int nDevices, int device, long seed, long n_rows, long n_cols, long rank, float eps) {
        if (n_rows > n_cols) {
            // TODO: Handle this case.
            throw;
        } else if (n_cols % nDevices) {
            // TODO: Support cases where the number of columns is not a multiple
            // of the number of devices.
            throw;
        }

        this->nDevices = nDevices;
        this->device = device;
        this->seed = seed;

        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->n_cols_slice = n_cols / nDevices;
        this->rank = rank;

        this->eps = eps;

        this->devA = DeviceMatrix(n_rows, n_cols_slice);
        // TODO: Maybe use this buffer instead of having separate buffers for
        // the other intermediates to reduce VRAM usage.
        // TODO: Also provide an option to compute the loss in chunks to further
        // lower the memory usage. Maybe even see if it can be done streaming. 
        this->devWH = DeviceMatrix(n_rows, n_cols_slice);

        this->devW = DeviceMatrix(n_rows, rank);
        this->devH = DeviceMatrix(rank, n_cols_slice);

        this->devFF = DeviceMatrix(rank, rank);

        initializeCuda();
        allocDeviceMemory();
    }

    ~DeviceContext() {
        cudaSetDevice(device);

        // TODO: Maybe make sure this works OK if some of the stuff
        // isn't initialization when the destructor is called.
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        curandDestroyGenerator(randGen);

        cudaFree(devA.devPtr);
        cudaFree(devWH.devPtr);
        cudaFree(devW.devPtr);
        cudaFree(devH.devPtr);
        cudaFree(devFF.devPtr);
        cudaFree(devAFPtr);
        cudaFree(devFFGPtr);
        cudaFree(devLossPtr);
        cudaFree(dev1f);
        cudaFree(dev0f);
    }

    void setComm(ncclComm_t* comm) {
        this->comm = comm;
    }


    void moveAToDeviceAsync(MeMatrix* hostA) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(
            cudaMemcpyAsync(
                devA.devPtr,
                hostA->data + device * n_rows * n_cols_slice,
                devA.sizeInBytes(), cudaMemcpyHostToDevice, stream)
        );
    }

    // TODO: COMPUTE ONCE AND BROADCAST FOR W SINCE IT NEEDS TO BE CONSISTENT.
    void initializeFactor(DeviceMatrix& mat) {
        CUDA_CALL(cudaSetDevice(device));

        // TODO: Using uniform does not violate NMF constraints, but IDK if there
        // are better random distributions to initialize with.
        CURAND_CALL(
            curandGenerateUniform(randGen, mat.devPtr, mat.nEntries())
        );
    }

    void initializeScalars() {
        CUDA_CALL(cudaSetDevice(device));
        float one = 1.0f;
        CUDA_CALL(
            cudaMemcpyAsync(dev1f, &one,  sizeof(float), cudaMemcpyHostToDevice, stream)
        );

        float zero = 0.0f;
        CUDA_CALL(
            cudaMemcpyAsync(dev0f, &zero,  sizeof(float), cudaMemcpyHostToDevice, stream)
        );
    }

    void updateH() {
        CUDA_CALL(cudaSetDevice(device));

        // I think this can be done without any device-device communication.
        DeviceMatrix devWA(rank, n_cols_slice, devAFPtr);
        matMul(devW, devA, devWA.devPtr, true, false);
        matMul(devW, devW, devFF.devPtr, true, false);

        DeviceMatrix devWWH(rank, n_cols_slice, devFFGPtr);
        matMul(devFF, devH, devWWH.devPtr, false, false);

        multiplicativeUpdate(devH, devWA, devWWH);
    }

    //////////////////////////////////////////////////////////////////////
    // Stuff for updating W.

    void computeLocalAH() {
        CUDA_CALL(cudaSetDevice(device));
        // DeviceMatrix devAH(n_rows, rank, devAFPtr);
        matMul(devA, devH, devAFPtr, false, true);
    }

    void computeLocalHH() {
        CUDA_CALL(cudaSetDevice(device));
        matMul(devH, devH, devFF.devPtr, false, true);
    }

    void updateWAfterAllReduces() {
        CUDA_CALL(cudaSetDevice(device));

        DeviceMatrix devWHH(n_rows, rank, devFFGPtr);
        matMul(devW, devFF, devWHH.devPtr, false, false);

        DeviceMatrix devAH(n_rows, rank, devAFPtr);
        multiplicativeUpdate(devW, devAH, devWHH);
    }

    //////////////////////////////////////////////////////////////////////

    void computeLoss() {
        CUDA_CALL(cudaSetDevice(device));

        matMul(devW, devH, devWH.devPtr, false, false);

        long n = devA.nEntries();

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;

        kernelVecSub<<<numBlocks, blockSize, 0, stream>>>(n, devWH.devPtr, devA.devPtr);

        CUBLAS_CALL(cublasSnrm2(handle, n, devWH.devPtr, 1, devLossPtr));

        CUDA_CALL(
            cudaMemcpyAsync(&localLoss, devLossPtr, sizeof(float), cudaMemcpyDeviceToHost, stream)
        );
    }

private:

    void initializeCuda() {
        CUDA_CALL(cudaSetDevice(device));

        CUDA_CALL(cudaStreamCreate(&stream));

        CUBLAS_CALL(cublasCreate(&handle));
        CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_CALL(cublasSetStream(handle, stream));

        CURAND_CALL(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randGen, seed));
        CURAND_CALL(curandSetStream(randGen, stream));
    }

    void allocDeviceMemory() {
        CUDA_CALL(cudaSetDevice(device));

        CUDA_CALL(cudaMalloc(&devA.devPtr, devA.sizeInBytes()));
        CUDA_CALL(cudaMalloc(&devW.devPtr, devW.sizeInBytes()));
        CUDA_CALL(cudaMalloc(&devH.devPtr, devH.sizeInBytes()));

        CUDA_CALL(cudaMalloc(&devWH.devPtr, devWH.sizeInBytes()));

        CUDA_CALL(cudaMalloc(&devFF.devPtr, devFF.sizeInBytes()));

        // NOTE: If including alignments, these would need special consideration due to how they are used.
        CUDA_CALL(cudaMalloc(&devAFPtr, sizeof(float) * std::max(n_rows, n_cols_slice) * rank));
        CUDA_CALL(cudaMalloc(&devFFGPtr, sizeof(float) * std::max(n_rows, n_cols_slice) * rank));

        CUDA_CALL(cudaMalloc(&devLossPtr, sizeof(float)));
        CUDA_CALL(cudaMalloc(&dev1f, sizeof(float)));
        CUDA_CALL(cudaMalloc(&dev0f, sizeof(float)));
    }


    void matMul(DeviceMatrix& A, DeviceMatrix& B, float* devCPtr, bool transposeA, bool transposeB) {
        // This is local, on-device matrix multiplication.
        int m, n, k;

        if (transposeA) {
            m = A.n_cols;
            k = A.n_rows;
        } else {
            m = A.n_rows;
            k = A.n_cols;
        }

        if (transposeB) {
            n = B.n_rows;
        } else {
            n = B.n_cols;
        }

        CUDA_CALL(cudaSetDevice(device));
        CUBLAS_CALL(cublasSgemm(
            handle,
            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            m, n, k,
            dev1f,
            A.devPtr, A.n_rows,
            B.devPtr, B.n_rows,
            dev0f,
            devCPtr, m
        ));
    }

    void multiplicativeUpdate(DeviceMatrix& F, const DeviceMatrix& numer, const DeviceMatrix& denom) {
        CUDA_CALL(cudaSetDevice(device));
        long n = F.nEntries();

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;

        kernelNmfMuUpdate<<<numBlocks, blockSize, 0, stream>>>(n, F.devPtr, numer.devPtr, denom.devPtr, eps);
    }

};


#endif
