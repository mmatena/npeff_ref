#pragma once
/**
 * Non-negative Matrix Factorization with Sparseness Constraints
 * https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf
 */
#include <math.h>
#include <ctime>
#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <util/cuda_system.h>
#include <util/cuda_statuses.h>
#include <util/matrices.h>
#include <util/misc_util.h>
#include "./kernels.h"
#include "./device_matrix_util.h"
#include "./sparsity_constraint_kernels.h"


// Forward declarations
template <typename IndT>
class DeviceContext;



// TODO: Make this private or whatever the equivalent is.
enum NmfFactorId { W, H };


struct FactorSparsenessConstraints {
    // Non-negative Matrix Factorization with Sparseness Constraints
    // https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

    float l1_norm = -1.0f;
    float l2_norm = 1.0f;

    bool isNull() {
       return l1_norm <= 0.0f;
    }


};


// TODO: Experiment with doing this for W on a single GPU vs doing this for different
// column subsets of W on different GPUs followed by a communicate.

template <typename IndT>
class DenseFactorSparsityContext {
    // Buffers needed:
    //   - sum(across non-rank rows/cols)
    //   - can do some stuff writing back into s.
    //   - Something for Z (or its complement?)
    //      - For W, array of bytes (or maybe even bits) is probably fine (at least as first pass).
    //        1 float = 4 bytes = 32 bits, so using bytes/bits should have arrays smaller than W/H.
    //   - Maybe mask for completed rows/cols (so they don't get updated)
    //   - Maybe ones vector to implement summing by matmul.
    // 
    //   - devAFPtr and devFFGPtr each provide a rank x max(n_rows, n_cols_slice) float allocation
    //     that I can use during the projection (can do H and W concurrently if I only use 1 each)

    /*
        Stuff to pass to this class:
            - devPtr to all ones array. Shared between W and H and set to largest individual size needed.
            - devPtr to Z buffer (either devAFPtr or devFFGPtr)
                - Let's pass both devAFPtr and devFFGPtr for now so I have a scratch space. I'll use both
                  when developing the first pass and re-evaluate whether both are needed later if there is
                  a reason to do so.
    */

public:
    DeviceContext<IndT>* deviceContext;
    DenseDeviceMatrix* devF;

    cudaStream_t& stream;
    cusparseHandle_t& sparseHandle;
    cublasHandle_t& denseHandle;
    curandGenerator_t& randGen;

    NmfFactorId factorId;

    int device;
    long rank;

    long vecSize;

    float l1_norm;
    float l2_norm;

    float* devSumSPtr;
    float* devSumYPtr;

    float* devSSPtr;
    float* devSYPtr;
    // float* devYYPtr;

    float* hostActiveMaskPtr = nullptr;
    float* devActiveMaskPtr;
    // Size [vecSize, rank], stores the s_ij < 0 boolean value.
    float* devActiveMaskBufferPtr;

    // Stuff that must be intialized from the DeviceContex.
    float* dev1f;
    float* dev0f;

    // TODO: Use smallest type compatible here instead of float.
    // Y is 1 - Z.
    float* devYPtr;
    float* devOnesVecPtr;

    DenseFactorSparsityContext(
        FactorSparsenessConstraints p,
        DeviceContext<IndT>* deviceContext,
        DenseDeviceMatrix* devF
    ) : stream(deviceContext->stream),
        sparseHandle(deviceContext->sparseHandle),
        denseHandle(deviceContext->denseHandle),
        randGen(deviceContext->randGen)
    {
        if(p.isNull()) {
            THROW;
        }

        if (devF == &deviceContext->devW) {
            factorId = NmfFactorId::W;
            vecSize = deviceContext->n_rows;

        } else if(devF == &deviceContext->devH) {
            factorId = NmfFactorId::H;

            std::cout << "TODO: Not supported regularization on the H factor yet.\n";
            THROW;

        } else {
            std::cout << "The factor matrix must either be W or H.\n";
            THROW;
        }

        this->devF = devF;
        this->deviceContext = deviceContext;

        this->device = deviceContext->device;
        this->rank = deviceContext->rank;

        this->l1_norm = p.l1_norm;
        this->l2_norm = p.l2_norm;

        // if (rank < some_value) {
        //     THROW;
        // }
    }

    ~DenseFactorSparsityContext() {
        // TODO
        delete[] hostActiveMaskPtr;
    }

    void setUpDeviceMemory() {
        // Must be called after everything else in the deviceContext's allocDeviceMemory function.
        CUDA_CALL(cudaSetDevice(device));

        // These are used the same way as in the device context. Basically just
        // constants that reside in the device memory.
        dev1f = deviceContext->dev1f;
        dev0f = deviceContext->dev0f;

        // These will be overwritten when used by this class.
        devYPtr = deviceContext->devFFGPtr;
        devActiveMaskBufferPtr = deviceContext->devAFPtr;

        float *ffBufferStart;
        switch(factorId) {
            case W:
                // Start at the begining of the devFF buffer for W.
                ffBufferStart = deviceContext->devFF.devPtr;
                break;
            case H:
                // Start halfway in the devFF buffer for H.
                ffBufferStart = deviceContext->devFF.devPtr + rank * (rank / 2);
                break;
        }

        // For storing the sum of each vector for the factor and reused later for Z.
        devSumSPtr = ffBufferStart + 0 * rank;
        devSumYPtr = ffBufferStart + 1 * rank;

        // For storing the dot products needed for solving for alpha.
        devSSPtr = ffBufferStart + 2 * rank;
        devSYPtr = ffBufferStart + 3 * rank;
        // devYYPtr = ffBufferStart + 3 * rank;

        devActiveMaskPtr = ffBufferStart + 4 * rank;

        // Actually need to allocate some GPU memory.
        CUDA_CALL(cudaMalloc(&devOnesVecPtr, sizeof(float) * vecSize));
        _setFloatValue(vecSize, devOnesVecPtr, 1.0f);

        hostActiveMaskPtr = new float[rank];
    }

    void sparsifyF() {
        CUDA_CALL(cudaSetDevice(device));

        // Set up.
        setInitialActiveMask();
        setInitialY();
        computeFSum();
        computeYSum();
        computeInitialS();

        // Then loop over iterations.
        // TODO: Maybe have some max iteration and return something if it
        // is reached without being finished.
        for(long i=0; true; i++) {
            // TODO: Need to ensure that columns of S do not change if their
            // corresponding activeMask value is zero.


            // TODO: Probably not needed.
            computeYSum();


            computeInnerProductsPreQuadratic();
            computeAlpha();
            updateSAfterAlpha();

            updateAndReadActiveMask();
            if (isFinishedSync()) break;

            updateYAndS();
            updateSLastStep();
        }
    }

private:

    void setInitialActiveMask() {
        CUDA_CALL(cudaSetDevice(device));
        _setFloatValue(rank, devActiveMaskPtr, 1.0f);
    }

    void setInitialY() {
        CUDA_CALL(cudaSetDevice(device));
        _setFloatValue(devF->nEntries(), devYPtr, 1.0f);
    }

    void computeFSum() {
        // Does the \sum_i x_i step in the algorithm.
        CUDA_CALL(cudaSetDevice(device));
        auto& devW = devF;
        _computeFactorShapedColumnSums(devW->devPtr, devSumSPtr);
    }

    void computeYSum() {
        // Does the size(Z) step in the algorithm.
        CUDA_CALL(cudaSetDevice(device));
        _computeFactorShapedColumnSums(devYPtr, devSumYPtr);
    }

    void computeInitialS() {
        // Computes the first s vectors. Done in place, overwriting devF.
        CUDA_CALL(cudaSetDevice(device));

        auto& devW = devF;

        long n = devW->nEntries();

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;

        kernel_computeInitialS<<<numBlocks, blockSize, 0, stream>>>(
            vecSize, rank, devW->devPtr, devSumSPtr, l1_norm);
    }


    void computeInnerProductsPreQuadratic() {
        CUDA_CALL(cudaSetDevice(device));
        auto& devW = devF;
        _dotProductForQuadratic(devW->devPtr, devW->devPtr, devSSPtr);
        _dotProductForQuadratic(devW->devPtr, devYPtr, devSYPtr);
    }

    void computeAlpha() {
        CUDA_CALL(cudaSetDevice(device));
        // _computeFactorShapedColumnSums(devYPtr, devSumYPtr);

        // This should contain the sum of entries of y. Since y is a
        // vector of zeros and ones, this equals its dot product with itself.
        float* devYYPtr = devSumYPtr;


        // TODO: Make sure this makes sense.
        float* devAlphaPtr = devSYPtr;

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (rank + blockSize - 1) / blockSize;

        kernel_computeAlpha<<<numBlocks, blockSize, 0, stream>>>(
            rank,
            devSSPtr, devSYPtr, devYYPtr,
            l1_norm, l2_norm * l2_norm,
            devAlphaPtr
        );
    }

    void updateSAfterAlpha() {
        CUDA_CALL(cudaSetDevice(device));
        auto& devW = devF;

        // TODO: Make sure this makes sense.
        float* devAlphaPtr = devSYPtr;
        float* devYYPtr = devSumYPtr;

        float* devSPtr = devW->devPtr;

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (vecSize * rank + blockSize - 1) / blockSize;

        kernel_updateSAfterAlpha<<<numBlocks, blockSize, 0, stream>>>(
            vecSize, rank,
            devSPtr, devYPtr, devActiveMaskBufferPtr,
            devAlphaPtr, devYYPtr,
            devActiveMaskPtr,
            l1_norm
        );
    }

    void updateAndReadActiveMask() {
        CUDA_CALL(cudaSetDevice(device));
        _computeFactorShapedColumnSums(devActiveMaskBufferPtr, devActiveMaskPtr);

        // TODO: See if it is faster to sum on GPU vs what I will do here.
        CUDA_CALL(
            cudaMemcpyAsync(hostActiveMaskPtr, devActiveMaskPtr, sizeof(float) * rank, cudaMemcpyDeviceToHost, stream)
        );
    }

    bool isFinishedSync() {
        // NOTE: Causes sync.
        CUDA_CALL(cudaSetDevice(device));
        // Wait until the hostActiveMaskPtr is finished updating.
        CUDA_CALL(cudaStreamSynchronize(stream));
        for (long i=0; i<rank; i++) {
            if (hostActiveMaskPtr[i] != 0.0f) {
                return false;
            }
        }
        return true;
    }

    void updateYAndS() {
        CUDA_CALL(cudaSetDevice(device));
        auto& devW = devF;

        float* devSPtr = devW->devPtr;

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (vecSize * rank + blockSize - 1) / blockSize;

        kernel_updateYAndS<<<numBlocks, blockSize, 0, stream>>>(
            vecSize, rank,
            devYPtr, devSPtr,
            devActiveMaskPtr
        );
    }

    void updateSLastStep() {
        CUDA_CALL(cudaSetDevice(device));
        auto& devW = devF;
        float* devSPtr = devW->devPtr;

        _computeFactorShapedColumnSums(devSPtr, devSumSPtr);
        _computeFactorShapedColumnSums(devYPtr, devSumYPtr);

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (vecSize * rank + blockSize - 1) / blockSize;

        kernel_updateSLastStep<<<numBlocks, blockSize, 0, stream>>>(
            vecSize, rank,
            devSPtr, devYPtr,
            devSumSPtr, devSumYPtr,
            devActiveMaskPtr,
            l1_norm
        );
    }


    // Also need to see if s is valid (i.e. non-negative after the updateSAfterAlpha step)


    ////////////////////////////////////////////////////////////////////////////////
    // Something

    void _dotProductForQuadratic(float* devPtr1, float* devPtr2, float* devOutPtr) {
        // Assumes the factor is W.
        CUDA_CALL(cudaSetDevice(device));

        // auto& devW = devF;

        CUBLAS_CALL(
            cublasSgemmStridedBatched(
                denseHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                1, 1, vecSize,
                dev1f,
                devPtr1, vecSize,
                vecSize,
                devPtr2, vecSize,
                vecSize,
                dev0f,
                devOutPtr, 1,
                1,
                rank
            )
        );
    }

    void _setFloatValue(long n, float* devPtr, float value) {
        CUDA_CALL(cudaSetDevice(device));
        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;
        kernel_setFloatValue<<<numBlocks, blockSize, 0, stream>>>(n, devPtr, value);
    }

    void _computeFactorShapedColumnSums(float* devMatrixPtr, float* devOutputPtr) {
        // Does the size(Z) step in the algorithm.
        CUDA_CALL(cudaSetDevice(device));

        auto& devW = devF;

        CUBLAS_CALL(
            cublasSgemv(
                denseHandle, CUBLAS_OP_T,
                devW->n_rows, devW->n_cols,
                dev1f,
                devMatrixPtr, devW->n_rows,
                devOnesVecPtr, 1,
                dev0f,
                devOutputPtr, 1
            )
        );
    }

};

