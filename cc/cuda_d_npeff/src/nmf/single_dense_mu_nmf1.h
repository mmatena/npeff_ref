#ifndef F82A8E0A_0BAB_46A1_BC2A_5A1DFFE7FC67_SINGLE_DENSE_MU_NMF1_H
#define F82A8E0A_0BAB_46A1_BC2A_5A1DFFE7FC67_SINGLE_DENSE_MU_NMF1_H
/* Basic dense matrix multiplicative update NMF. */
#include <algorithm>
#include <iostream>

#include <curand.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <util/cuda_statuses.h>
#include <util/matrices.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: The matrices should probably all be in column major format!!!!
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


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

class MuNmf {
public:

    MuNmf(MeMatrix* A, long rank, int max_iters, float eps, float tol, long seed) {
        if (A->n_rows > A->n_cols) {
            // TODO: Handle this case.
            throw;
        }
        this->max_iters = max_iters;
        this->eps = eps;
        this->tol = tol;
        this->seed = seed;

        this->rank = rank;

        this->n_rows = (long) A->n_rows;
        this->n_cols = (long) A->n_cols;

        this->hostA = A;

        this->devA = DeviceMatrix(n_rows, n_cols);
        this->devW = DeviceMatrix(n_rows, rank);
        this->devH = DeviceMatrix(rank, n_cols);

        this->devWH = DeviceMatrix(n_rows, n_cols);
        this->devFF = DeviceMatrix(rank, rank);

        this->initializeCuda();
        this->allocDeviceMemory();
    }

    ~MuNmf() {
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

    void run() {
        // Doesn't clean anything up afterwards. Probably should only be called once.
        moveAToDeviceAsync();
        initializeFactor(devW);
        initializeFactor(devH);
        initializeScalars();

        ///////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////
        // TODO: Need multiple streams for concurrency!!! A single stream is in order.
        ///////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////

        for (int step = 0; step < max_iters; step++) {
            // Remember stream synchronization.

            updateW();
            updateH();

            computeLoss();

            float loss = -1.0f;
            CUDA_CALL(
                cudaMemcpyAsync(&loss, devLossPtr, sizeof(float), cudaMemcpyDeviceToHost, stream)
            );
            CUDA_CALL(cudaStreamSynchronize(stream));

            // TODO: Remove this.
            std::cout << loss << '\n';

        }
    }

private:
    int max_iters;
    float eps;
    float tol;

    long seed;

    long rank;

    long n_rows;
    long n_cols;

    MeMatrix* hostA;

    cudaStream_t stream;
    cublasHandle_t handle;
    curandGenerator_t randGen;

    // All matrices below are assumed to be in column-major format.

    DeviceMatrix devA;
    DeviceMatrix devWH;

    DeviceMatrix devW;
    DeviceMatrix devH;

    DeviceMatrix devFF;
    float* devFFGPtr;
    float* devAFPtr;

    float *devLossPtr;

    // TODO: See if there are better ways of setting scalar parameters for cuBLAS.
    float* dev1f;
    float* dev0f;

    void initializeCuda() {
        cudaStreamCreate(&stream);

        cublasCreate(&handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSetStream(handle, stream);

        curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(randGen, seed);
        curandSetStream(randGen, stream);
    }

    void allocDeviceMemory() {
        // TODO: Wrap these in CUDA_CALL, probably.
        // 
        // TODO: Maybe allocate a single contiguous chunk and split up in my
        // code? Need to make sure alignment and stuff is ok though.
        CUDA_CALL(cudaMalloc(&devA.devPtr, devA.sizeInBytes()));
        CUDA_CALL(cudaMalloc(&devW.devPtr, devW.sizeInBytes()));
        CUDA_CALL(cudaMalloc(&devH.devPtr, devH.sizeInBytes()));

        CUDA_CALL(cudaMalloc(&devWH.devPtr, devWH.sizeInBytes()));
        CUDA_CALL(cudaMalloc(&devFF.devPtr, devFF.sizeInBytes()));

        // NOTE: If including alignments in the matrixSizeInBytes, these would
        // need special consideration due to how it is used.
        CUDA_CALL(cudaMalloc(&devAFPtr, matrixSizeInBytes(std::max(n_rows, n_cols), rank)));
        CUDA_CALL(cudaMalloc(&devFFGPtr, matrixSizeInBytes(std::max(n_rows, n_cols), rank)));

        CUDA_CALL(cudaMalloc(&devLossPtr, sizeof(float)));

        CUDA_CALL(cudaMalloc(&dev1f, sizeof(float)));
        CUDA_CALL(cudaMalloc(&dev0f, sizeof(float)));
    }

    void moveAToDeviceAsync() {
        CUDA_CALL(
            cudaMemcpyAsync(devA.devPtr, hostA->data, hostA->sizeInBytes(), cudaMemcpyHostToDevice, stream)
        );
    }

    void initializeFactor(DeviceMatrix& mat) {
        // TODO: Using uniform does not violate NMF constraints, but IDK if there
        // are better random distributions to initialize with.
        CURAND_CALL(
            curandGenerateUniform(randGen, mat.devPtr, mat.nEntries())
        );
    }

    void initializeScalars() {
        float one = 1.0f;
        CUDA_CALL(
            cudaMemcpyAsync(dev1f, &one,  sizeof(float), cudaMemcpyHostToDevice, stream)
        );

        float zero = 0.0f;
        CUDA_CALL(
            cudaMemcpyAsync(dev0f, &zero,  sizeof(float), cudaMemcpyHostToDevice, stream)
        );
    }


    void updateW() {
        DeviceMatrix devAH(n_rows, rank, devAFPtr);
        matMul(devA, devH, devAH.devPtr, false, true);
        matMul(devH, devH, devFF.devPtr, false, true);

        DeviceMatrix devWHH(n_rows, rank, devFFGPtr);
        matMul(devW, devFF, devWHH.devPtr, false, false);

        multiplicativeUpdate(devW, devAH, devWHH);
    }

    void updateH() {
        DeviceMatrix devWA(rank, n_cols, devAFPtr);
        matMul(devW, devA, devWA.devPtr, true, false);
        matMul(devW, devW, devFF.devPtr, true, false);

        DeviceMatrix devWWH(rank, n_cols, devFFGPtr);
        matMul(devFF, devH, devWWH.devPtr, false, false);

        multiplicativeUpdate(devH, devWA, devWWH);
    }


    ////////////////////////////////////////////////////////////
    // Some private utility functions.
    ////////////////////////////////////////////////////////////

    void multiplicativeUpdate(DeviceMatrix& F, const DeviceMatrix& numer, const DeviceMatrix& denom) {
        long n = F.nEntries();

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;

        kernelNmfMuUpdate<<<numBlocks, blockSize, 0, stream>>>(n, F.devPtr, numer.devPtr, denom.devPtr, eps);
    }

    void computeLoss() {
        matMul(devW, devH, devWH.devPtr, false, false);

        long n = devA.nEntries();

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;

        kernelVecSub<<<numBlocks, blockSize, 0, stream>>>(n, devWH.devPtr, devA.devPtr);

        CUBLAS_CALL(cublasSnrm2(handle, n, devWH.devPtr, 1, devLossPtr));
    }

    void matMul(DeviceMatrix& A, DeviceMatrix& B, float* devCPtr, bool transposeA, bool transposeB) {
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

    long matrixSizeInBytes(long n_rows, long n_cols) {
        return sizeof(float) * n_rows * n_cols;
    }
};


#endif
