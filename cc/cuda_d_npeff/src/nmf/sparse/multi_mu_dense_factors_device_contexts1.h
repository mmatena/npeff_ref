#ifndef F6F5A02E_0ED9_429F_B57F_BFD0545889DF_H
#define F6F5A02E_0ED9_429F_B57F_BFD0545889DF_H

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
#include <util/matrix_util.h>
#include "./kernels.h"
#include "./device_matrix_util.h"
#include "./dense_factor_sparsity_constraints.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: Put these parameters into their own header file.



// Only contains information if we are initializing with a known dense/sparse matrix.
struct HInitializationInfo {
    MeMatrix* initialDenseH = nullptr;

    bool hasInitialization() {
        return initialDenseH != nullptr;
    }
};


struct MuNmfParams {
    int rank;
    int max_iters;
    float eps;
    long seed;

    FactorSparsenessConstraints sparsenessConstraints_W;
    // FactorSparsenessConstraints sparsenessConstraints_H;

    HInitializationInfo H_initInfo;
};


struct MuNmfDeviceContextParams {
    MuNmfParams nmfParams;
    int nDevices;
    int device;
    long n_rows;
    long n_cols;
};

///////////////////////////////////////////////////////////////////////////////////////////////////


// NOTE: This is for private use within MuNMF.
template <typename IndT>
class DeviceContext {
public:
    static const cusparseSpMMAlg_t AH_SPMM_ALG = CUSPARSE_SPMM_ALG_DEFAULT;
    static const cusparseSpMMAlg_t AW_SPMM_ALG = CUSPARSE_SPMM_ALG_DEFAULT;

    // 2^30.
    static const long LOSS_CHUNK_SIZE = 1073741824;

    // Not owned by this class. Owned by coordinator.
    ElCsrMatrix<IndT>* hostAPartition;

    int nDevices;
    int device;
    long seed;

    long n_rows;
    long n_cols;
    long n_cols_slice;
    long rank;

    float eps;

    cudaStream_t stream;
    cusparseHandle_t sparseHandle;
    cublasHandle_t denseHandle;
    curandGenerator_t randGen;

    // Owned by the coordinator, not this class.
    ncclComm_t* comm;

    CsrDeviceMatrix<IndT> devA;

    DenseDeviceMatrix devW;
    cusparseDnMatDescr_t devWTDescr;

    DenseDeviceMatrix devH;

    DenseDeviceMatrix devFF;

    float* devFFGPtr = nullptr;

    float* devAFPtr = nullptr;
    cusparseDnMatDescr_t devAHDescr;
    cusparseDnMatDescr_t devWADescr;

    SpDenseMatMul<IndT> spmmAH;
    SpDenseMatMul<IndT> spmmAW;

    float* dev1f = nullptr;
    float* dev0f = nullptr;

    int n_normAChunks;
    float *devNormAPtr;
    float local_sq_A_norm;

    int n_posLossChunks;
    float* devLossPosPtr;
    float* posLoss;

    int n_negLossChunks;
    float* devLossNegPtr;
    float* negLoss;

    DenseFactorSparsityContext<IndT>* sparseContextW = nullptr;

    HInitializationInfo H_initInfo;

    DeviceContext(ElCsrMatrix<IndT>* hostAPartition, MuNmfDeviceContextParams p) {
        if (p.n_rows > p.n_cols) {
            // TODO: Handle this case.
            std::cout << "Error: Not handling matrices with more rows than columns yet.\n";
            throw;
        }

        this->hostAPartition = hostAPartition;

        this->nDevices = p.nDevices;
        this->device = p.device;
        this->seed = p.nmfParams.seed;

        this->n_rows = p.n_rows;
        this->n_cols = p.n_cols;
        this->n_cols_slice = getSizeOfSplit(p.n_cols, p.nDevices, p.device);
        this->rank = p.nmfParams.rank;

        if (hostAPartition->n_cols != this->n_cols_slice || hostAPartition->n_rows != this->n_rows) {
            THROW;
        }

        this->eps = p.nmfParams.eps;

        this->H_initInfo = p.nmfParams.H_initInfo;

        this->devA = CsrDeviceMatrix<IndT>(n_rows, n_cols_slice, hostAPartition->nnz);

        this->devW = DenseDeviceMatrix(n_rows, rank);
        this->devH = DenseDeviceMatrix(rank, n_cols_slice);

        this->devFF = DenseDeviceMatrix(rank, rank);

        // Only the first device gets a sparsity context for now.
        if(device == 0 && !p.nmfParams.sparsenessConstraints_W.isNull()) {
            sparseContextW = new DenseFactorSparsityContext<IndT>(
                p.nmfParams.sparsenessConstraints_W, this, &devW);
        }

        initializeCuda();
        allocDeviceMemory();
    }

    ~DeviceContext() {
        // TODO:
        delete sparseContextW;
    }

    void setComm(ncclComm_t* comm) {
        this->comm = comm;
    }


    void moveAToDeviceAsync() {
        CUDA_CALL(cudaSetDevice(device));
        long nnz = devA.nnz;

        CUDA_CALL(
            cudaMemcpyAsync(devA.devCsrValA, hostAPartition->csrValA, sizeof(float) * nnz, cudaMemcpyHostToDevice, stream)
        );
        CUDA_CALL(
            cudaMemcpyAsync(devA.devCsrRowPtrA, hostAPartition->csrRowPtrA, sizeof(IndT) * (n_rows + 1), cudaMemcpyHostToDevice, stream)
        );
        CUDA_CALL(
            cudaMemcpyAsync(devA.devCsrColIndA, hostAPartition->csrColIndA, sizeof(IndT) * nnz, cudaMemcpyHostToDevice, stream)
        );
    }

    void initializeFactor(DenseDeviceMatrix& mat) {
        CUDA_CALL(cudaSetDevice(device));

        // TODO: Using uniform does not violate NMF constraints, but IDK if there
        // are better random distributions to initialize with.

        // TODO: Maybe see if there are any integer overflow issues here.
        CURAND_CALL(
            curandGenerateUniform(randGen, mat.devPtr, mat.nEntries())
        );
    }

    void initializeW(DenseDeviceMatrix& mat) {
        initializeFactor(mat);
    }

    void initializeH(DenseDeviceMatrix& mat) {
        if(H_initInfo.hasInitialization()) {
            MeMatrix* initialDenseH = H_initInfo.initialDenseH;
            if (initialDenseH == nullptr) {
                // Adding this in for once I support sparse Hs.
                THROW;
            }

            CUDA_CALL(cudaSetDevice(device));
            CUDA_CALL(
                cudaMemcpyAsync(
                    mat.devPtr,
                    getColumnWiseSplitStart(*initialDenseH, nDevices, device),
                    sizeof(float) * getSizeOfSplit(n_cols, nDevices, device),
                    cudaMemcpyHostToDevice, stream)
            );

        } else {
            initializeFactor(mat);
        }
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


    void preprocessSparseMatMuls() {
        // TODO: Exeriment with the algorithms used. See
        // https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm
        CUDA_CALL(cudaSetDevice(device));

        spmmAH = SpDenseMatMul<IndT>(
            sparseHandle,
            devA, devH, devAHDescr, false, true, AH_SPMM_ALG,
            dev1f, dev0f);

        DenseDeviceMatrix devWT(rank, n_rows, devW.devPtr);
        devWT.dnMatDescr = devWTDescr;
        spmmAW = SpDenseMatMul<IndT>(
            sparseHandle,
            devA, devWT, devWADescr, true, true, AW_SPMM_ALG,
            dev1f, dev0f);

        spmmAH.createBuffer();
        spmmAW.createBuffer();

        spmmAH.preprocess();
        spmmAW.preprocess();
    }

    void computeASqNorm() {
        CUDA_CALL(cudaSetDevice(device));

        if (devA.nnz > (long) INT32_MAX) {
            std::cout << "TODO: Split cublasSnrm2 call into chunks when devA.nnz > INT32_MAX\n";
            THROW;
        }

        // TODO: Handle cases where nnz > INT32_MAX.
        CUBLAS_CALL(cublasSnrm2(denseHandle, devA.nnz, devA.devCsrValA, 1, devNormAPtr));

        float A_norm;
        CUDA_CALL(
            cudaMemcpyAsync(&A_norm, devNormAPtr, sizeof(float), cudaMemcpyDeviceToHost, stream)
        );

        // TODO: Probably find better way of doing this, maybe callbacks or something?
        CUDA_CALL(cudaStreamSynchronize(stream));

        this->local_sq_A_norm = A_norm * A_norm;
    }

    void updateH() {
        CUDA_CALL(cudaSetDevice(device));

        // The values of devWADescr are stored starting at the devAFPtr.
        DenseDeviceMatrix devWA(rank, n_cols_slice, devAFPtr);
        spmmAW.callAsync();

        denseMatMul(devW, devW, devFF.devPtr, true, false);

        DenseDeviceMatrix devWWH(rank, n_cols_slice, devFFGPtr);
        denseMatMul(devFF, devH, devWWH.devPtr, false, false);

        multiplicativeUpdate(devH, devWA, devWWH);
    }


    //////////////////////////////////////////////////////////////////////
    // Stuff for updating W.

    void computeLocalAH() {
        CUDA_CALL(cudaSetDevice(device));
        spmmAH.callAsync();
    }

    void computeLocalHH() {
        CUDA_CALL(cudaSetDevice(device));
        denseMatMul(devH, devH, devFF.devPtr, false, true);
    }

    void updateWAfterAllReduces() {
        CUDA_CALL(cudaSetDevice(device));

        DenseDeviceMatrix devWHH(n_rows, rank, devFFGPtr);
        denseMatMul(devW, devFF, devWHH.devPtr, false, false);

        DenseDeviceMatrix devAH(n_rows, rank, devAFPtr);
        multiplicativeUpdate(devW, devAH, devWHH);
    }

    //////////////////////////////////////////////////////////////////////

    void computeUncachedLoss() {
        CUDA_CALL(cudaSetDevice(device));

        // Positive component of the loss.
        denseMatMul(devW, devW, devFF.devPtr, true, false);
        DenseDeviceMatrix devWWH(rank, n_cols_slice, devFFGPtr);
        denseMatMul(devFF, devH, devWWH.devPtr, false, false);
        frobeniusProductToHost(devH, devWWH, devLossPosPtr, posLoss);

        // Negative component of the loss.
 
        // This writes A^t(W^t)^t = A^t W to devAFPtr in row-major format.
        // Interpreting this in column major format gives (A^t W)^t = W^t A.
        spmmAW.callAsync();
        // The values of devWADescr are stored starting at the devAFPtr.
        DenseDeviceMatrix devWA(rank, n_cols_slice, devAFPtr);
        frobeniusProductToHost(devWA, devH, devLossNegPtr, negLoss);
    }

    //////////////////////////////////////////////////////////////////////

private:

    void initializeCuda() {
        CUDA_CALL(cudaSetDevice(device));

        CUDA_CALL(cudaStreamCreate(&stream));

        CUBLAS_CALL(cublasCreate(&denseHandle));
        CUBLAS_CALL(cublasSetPointerMode(denseHandle, CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_CALL(cublasSetStream(denseHandle, stream));

        CUSPARSE_CALL(cusparseCreate(&sparseHandle));
        CUSPARSE_CALL(cusparseSetPointerMode(sparseHandle, CUSPARSE_POINTER_MODE_DEVICE));
        CUSPARSE_CALL(cusparseSetStream(sparseHandle, stream));

        CURAND_CALL(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randGen, seed));
        CURAND_CALL(curandSetStream(randGen, stream));
    }

    int _compute_nLossChunks(long nEntries) {
        // We can include an additional up to LOSS_CHUNK_SIZE - 1 entries in
        // the last chunk, so this works.
        return std::max(1, (int) (nEntries / LOSS_CHUNK_SIZE));
    }

    void allocDeviceMemory() {
        CUDA_CALL(cudaSetDevice(device));

        devA.allocDeviceMemory();

        devW.allocDeviceMemory();
        devH.allocDeviceMemory();
        devFF.allocDeviceMemory();

        // NOTE: If including alignments in the matrixSizeInBytes, these would
        // need special consideration due to how it is used.
        CUDA_CALL(cudaMalloc(&devAFPtr, sizeof(float) * (long) std::max(n_rows, n_cols_slice) * (long) rank));
        CUDA_CALL(cudaMalloc(&devFFGPtr, sizeof(float) * (long) std::max(n_rows, n_cols_slice) * (long) rank));

        // TODO: Slightly more sophisticated chunk size computation.
        n_normAChunks = _compute_nLossChunks(devA.nnz);
        n_posLossChunks = _compute_nLossChunks(devH.nEntries());
        n_negLossChunks = _compute_nLossChunks(devH.nEntries());

        CUDA_CALL(cudaMalloc(&devNormAPtr, sizeof(float) * n_normAChunks));
        CUDA_CALL(cudaMalloc(&devLossPosPtr, sizeof(float) * n_posLossChunks));
        CUDA_CALL(cudaMalloc(&devLossNegPtr, sizeof(float) * n_negLossChunks));

        posLoss = new float[n_posLossChunks];
        negLoss = new float[n_negLossChunks];

        CUDA_CALL(cudaMalloc(&dev1f, sizeof(float)));
        CUDA_CALL(cudaMalloc(&dev0f, sizeof(float)));

        // Need W^T in row-major format.
        CUSPARSE_CALL(
            cusparseCreateDnMat(&devWTDescr, rank, n_rows, n_rows, devW.devPtr,
                                CUDA_R_32F, CUSPARSE_ORDER_ROW)
        );

        // These two point to the same memory. They are not used at the same time.
        CUSPARSE_CALL(
            cusparseCreateDnMat(&devAHDescr, n_rows, rank, n_rows, devAFPtr,
                                CUDA_R_32F, CUSPARSE_ORDER_COL)
        );
        // We compute A^TW in row-major form. The rest of the code treats the matrix in
        // column-major form, so we effectively compute W^TA.
        CUSPARSE_CALL(
            cusparseCreateDnMat(&devWADescr, n_cols_slice, rank, rank, devAFPtr,
                                CUDA_R_32F, CUSPARSE_ORDER_ROW)
        );

        if(sparseContextW) {
            sparseContextW->setUpDeviceMemory();
        }
    }

    void denseMatMul(DenseDeviceMatrix& A, DenseDeviceMatrix& B, float* devCPtr, bool transposeA, bool transposeB) {
        CUDA_CALL(cudaSetDevice(device));

        long m, n, k;

        if (transposeA) {
            m = A.n_cols;
            k = A.n_rows;
        } else {
            m = A.n_rows;
            k = A.n_cols;
        }

        if (transposeB) {
            n = B.n_rows;
            if (k != B.n_cols) {
                THROW;
            }
        } else {
            n = B.n_cols;
            if (k != B.n_rows) {
                THROW;
            }
        }

        CUBLAS_CALL(cublasSgemm(
            denseHandle,
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

    void multiplicativeUpdate(DenseDeviceMatrix& F, const DenseDeviceMatrix& numer, const DenseDeviceMatrix& denom) {
        CUDA_CALL(cudaSetDevice(device));
        long n = F.nEntries();

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;

        kernelNmfMuUpdate<<<numBlocks, blockSize, 0, stream>>>(n, F.devPtr, numer.devPtr, denom.devPtr, eps);
    }

    // void frobeniusProductToHost(DenseDeviceMatrix& A, DenseDeviceMatrix& B, float* devResultPtr, float* hostResultPtr) {
    //     CUDA_CALL(cudaSetDevice(device));

    //     if (A.n_cols != B.n_cols || A.n_rows != B.n_rows) {
    //         THROW;
    //     }
    //     std::cout << A.nEntries() << "\n";
    //     std::cout << (int) A.nEntries() << "\n";
    //     CUBLAS_CALL(cublasSdot(
    //         denseHandle, A.nEntries(),
    //         A.devPtr, 1,
    //         B.devPtr, 1,
    //         devResultPtr
    //     ));
    //     CUDA_CALL(
    //         cudaMemcpyAsync(hostResultPtr, devResultPtr, sizeof(float), cudaMemcpyDeviceToHost, stream)
    //     );
    // }
    void frobeniusProductToHost(DenseDeviceMatrix& A, DenseDeviceMatrix& B, float* devResultPtr, float* hostResultPtr) {
        CUDA_CALL(cudaSetDevice(device));

        if (A.n_cols != B.n_cols || A.n_rows != B.n_rows) {
            THROW;
        }
        // std::cout << A.nEntries() << "\n";
        // std::cout << (int) A.nEntries() << "\n";

        long n_entries = A.nEntries();
        int n_chunks = _compute_nLossChunks(A.nEntries());

        for(int i=0; i < n_chunks; i++) {
            long chunkSize;
            if (i == n_chunks - 1) {
                chunkSize = n_entries - i * LOSS_CHUNK_SIZE;
            } else {
                chunkSize = LOSS_CHUNK_SIZE;
            }
            if ((long) ((int) chunkSize) != chunkSize) {
                std::cout << "Integer overflow\n";
                THROW;
            }
            CUBLAS_CALL(cublasSdot(
                denseHandle, chunkSize,
                A.devPtr + i * LOSS_CHUNK_SIZE, 1,
                B.devPtr + i * LOSS_CHUNK_SIZE, 1,
                devResultPtr + i
            ));
            CUDA_CALL(
                cudaMemcpyAsync(hostResultPtr + i, devResultPtr + i, sizeof(float), cudaMemcpyDeviceToHost, stream)
            );

        }
    }

};


#endif
