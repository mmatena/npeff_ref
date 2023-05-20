#ifndef A467F36E_BB34_4653_BB19_C90F52D8EC42_2_H
#define A467F36E_BB34_4653_BB19_C90F52D8EC42_2_H

/* Single GPU sparse data with dense factors and intermediates.

CuSPARSE targets matrices with a number of (structural) zero elements
which represent > 95% of the total entries.

*/
#include <math.h>
#include <ctime>
#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <util/cuda_statuses.h>
#include <util/matrices.h>


#define _DEV__SYNC_FOR_TIMING() CUDA_CALL(cudaStreamSynchronize(stream))



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
class DenseDeviceMatrix {
public:
    long n_rows;
    long n_cols;
    float* devPtr;

    cusparseDnMatDescr_t dnMatDescr;

    DenseDeviceMatrix() {}

    DenseDeviceMatrix(long n_rows, long n_cols) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->devPtr = nullptr;
    }

    DenseDeviceMatrix(long n_rows, long n_cols, float* devPtr) {
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

    void allocDeviceMemory() {
        CUDA_CALL(cudaMalloc(&devPtr, sizeInBytes()));
        createMatDescr();
    }

private:
    void createMatDescr() {
        // allocDeviceMemory should have been called before calling this.
        CUSPARSE_CALL(
            cusparseCreateDnMat(&dnMatDescr, n_rows, n_cols, n_rows, devPtr,
                                CUDA_R_32F, CUSPARSE_ORDER_COL)
        );
    }
};


class CsrDeviceMatrix {
public:
    // TODO: Think about how to handle larger than int sizes, or if that is even needed.
    int n_rows;
    int n_cols;

    int nnz;
    float* devCsrValA;
    int* devCsrRowPtrA;
    int* devCsrColIndA;

    cusparseSpMatDescr_t spMatDescr;

    CsrDeviceMatrix() {}

    CsrDeviceMatrix(int n_rows, int n_cols, int nnz) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->nnz = nnz;
    }

    void allocDeviceMemory() {
        CUDA_CALL(cudaMalloc(&devCsrValA, sizeof(float) * nnz));
        CUDA_CALL(cudaMalloc(&devCsrRowPtrA, sizeof(int) * (n_rows + 1)));
        CUDA_CALL(cudaMalloc(&devCsrColIndA, sizeof(int) * nnz));

        createMatDescr();
    }

private:
    void createMatDescr() {
        // allocDeviceMemory should have been created already.
        CUSPARSE_CALL(
            cusparseCreateCsr(&spMatDescr, n_rows, n_cols, nnz,
                              devCsrRowPtrA, devCsrColIndA, devCsrValA,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        );
    }
};


///////////////////////////////////////////////////////////////////////////////

class SpDenseMatMul {
public:
    cusparseHandle_t sparseHandle;

    CsrDeviceMatrix A;
    DenseDeviceMatrix B;
    cusparseDnMatDescr_t C;

    bool transposeA;
    bool transposeB;

    cusparseSpMMAlg_t alg;

    float* dev1f;
    float* dev0f;

    size_t bufferSize = 0;
    void* buffer = nullptr;

    SpDenseMatMul() {}

    SpDenseMatMul(
        cusparseHandle_t sparseHandle,
        CsrDeviceMatrix A, DenseDeviceMatrix B, cusparseDnMatDescr_t C,
        bool transposeA, bool transposeB,
        cusparseSpMMAlg_t alg,
        float* dev1f, float* dev0f
    ) : sparseHandle(sparseHandle),
        A(A), B(B), C(C),
        transposeA(transposeA), transposeB(transposeB),
        alg(alg),
        dev1f(dev1f), dev0f(dev0f) {}

    void computeBufferSize() {
        CUSPARSE_CALL(
            cusparseSpMM_bufferSize(sparseHandle,
                         transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         dev1f, A.spMatDescr, B.dnMatDescr, dev0f, C, CUDA_R_32F,
                         alg, &bufferSize)
        );
    }

    void preprocess() {
        CUSPARSE_CALL(
            cusparseSpMM_preprocess(sparseHandle,
                         transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         dev1f, A.spMatDescr, B.dnMatDescr, dev0f, C, CUDA_R_32F,
                         alg, buffer)
        );
    }

    void callAsync() {
        if (buffer == nullptr) {
            throw;
        }
        CUSPARSE_CALL(
            cusparseSpMM(sparseHandle,
                         transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         dev1f, A.spMatDescr, B.dnMatDescr, dev0f, C, CUDA_R_32F,
                         alg, buffer)
        );
    }


};

///////////////////////////////////////////////////////////////////////////////


class MuNmf {
private:
    static const cusparseSpMMAlg_t AH_SPMM_ALG = CUSPARSE_SPMM_ALG_DEFAULT;
    static const cusparseSpMMAlg_t AW_SPMM_ALG = CUSPARSE_SPMM_ALG_DEFAULT;
    // static const cusparseSpMMAlg_t AH_SPMM_ALG = CUSPARSE_SPMM_CSR_ALG1;
    // static const cusparseSpMMAlg_t AW_SPMM_ALG = CUSPARSE_SPMM_CSR_ALG2;

    int max_iters;
    float eps;

    long seed;

    int rank;

    int n_rows;
    int n_cols;

    ElCsrMatrix* hostA;

    cudaStream_t stream;
    cusparseHandle_t sparseHandle;
    cublasHandle_t denseHandle;
    curandGenerator_t randGen;

    CsrDeviceMatrix devA;
    // DenseDeviceMatrix devWH;

    DenseDeviceMatrix devW;
    cusparseDnMatDescr_t devWTDescr;

    DenseDeviceMatrix devH;

    DenseDeviceMatrix devFF;

    float* devFFGPtr;

    float* devAFPtr;
    cusparseDnMatDescr_t devAHDescr;
    cusparseDnMatDescr_t devWADescr;

    void* spMatMulBuffer = nullptr;
    SpDenseMatMul spmmAH;
    SpDenseMatMul spmmAW;

    float *devNormAPtr;
    float *devLossPosPtr;
    float *devLossNegPtr;

    // TODO: See if there are better ways of setting scalar parameters for cuBLAS.
    float* dev1f;
    float* dev0f;


    float loss;
    float sq_A_norm;

public:
    MuNmf(ElCsrMatrix* A, int rank, int max_iters, float eps, long seed) {
        if (A->n_rows > A->n_cols) {
            // TODO: Handle this case, also see if sparsity patterns might influence this.
            throw;
        }

        this->max_iters = max_iters;
        this->eps = eps;
        this->seed = seed;

        this->rank = rank;

        this->n_rows = A->n_rows;
        this->n_cols = A->n_cols;

        this->hostA = A;

        this->devA = CsrDeviceMatrix(n_rows, n_cols, A->nnz);

        this->devW = DenseDeviceMatrix(n_rows, rank);
        this->devH = DenseDeviceMatrix(rank, n_cols);

        // this->devWH = DenseDeviceMatrix(n_rows, n_cols);
        this->devFF = DenseDeviceMatrix(rank, rank);

        initializeCuda();
        allocDeviceMemory();
    }

    ~MuNmf() {
        // TODO
    }

private:
    void initializeCuda() {
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


    void allocDeviceMemory() {
        // TODO: Maybe allocate a single contiguous chunk and split up in my
        // code? Need to make sure alignment and stuff is ok though.

        devA.allocDeviceMemory();

        devW.allocDeviceMemory();
        devH.allocDeviceMemory();
        devFF.allocDeviceMemory();

        // CUDA_CALL(cudaMalloc(&devWH.devPtr, devWH.sizeInBytes()));

        // NOTE: If including alignments in the matrixSizeInBytes, these would
        // need special consideration due to how it is used.
        CUDA_CALL(cudaMalloc(&devAFPtr, sizeof(float) * std::max(n_rows, n_cols) * rank));
        CUDA_CALL(cudaMalloc(&devFFGPtr, sizeof(float) * std::max(n_rows, n_cols) * rank));

        CUDA_CALL(cudaMalloc(&devNormAPtr, sizeof(float)));
        CUDA_CALL(cudaMalloc(&devLossPosPtr, sizeof(float)));
        CUDA_CALL(cudaMalloc(&devLossNegPtr, sizeof(float)));

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
            cusparseCreateDnMat(&devWADescr, n_cols, rank, rank, devAFPtr,
                                CUDA_R_32F, CUSPARSE_ORDER_ROW)
        );
    }

    void moveAToDeviceAsync() {
        int nnz = devA.nnz;

        CUDA_CALL(
            cudaMemcpyAsync(devA.devCsrValA, hostA->csrValA, sizeof(float) * nnz, cudaMemcpyHostToDevice, stream)
        );
        CUDA_CALL(
            cudaMemcpyAsync(devA.devCsrRowPtrA, hostA->csrRowPtrA, sizeof(int) * (n_rows + 1), cudaMemcpyHostToDevice, stream)
        );
        CUDA_CALL(
            cudaMemcpyAsync(devA.devCsrColIndA, hostA->csrColIndA, sizeof(int) * nnz, cudaMemcpyHostToDevice, stream)
        );
    }

    void initializeFactor(DenseDeviceMatrix& mat) {
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

    void preprocessSparseMatMuls() {
        // TODO: Exeriment with the algorithms used. See
        // https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm

        if (spMatMulBuffer != nullptr)  {
            // Guard to prevent this from being called twice.
            throw;
        }
        CUDA_CALL(cudaStreamSynchronize(stream));

        spmmAH = SpDenseMatMul(
            sparseHandle,
            devA, devH, devAHDescr, false, true, AH_SPMM_ALG,
            dev1f, dev0f);

        DenseDeviceMatrix devWT(rank, n_rows, devW.devPtr);
        devWT.dnMatDescr = devWTDescr;
        spmmAW = SpDenseMatMul(
            sparseHandle,
            devA, devWT, devWADescr, true, true, AW_SPMM_ALG,
            dev1f, dev0f);

        spmmAH.computeBufferSize();
        spmmAW.computeBufferSize();

        // NOTE: IDK if this is needed (the calls might not actually be async).
        CUDA_CALL(cudaStreamSynchronize(stream));

        std::cout << "AH^T required buffer size: " << spmmAH.bufferSize << "\n";
        std::cout << "W^TA required buffer size: " << spmmAW.bufferSize << "\n";

        size_t bufferSize = std::max(spmmAH.bufferSize, spmmAW.bufferSize);

        CUDA_CALL(cudaMalloc(&spMatMulBuffer, bufferSize));
        spmmAH.buffer = spMatMulBuffer;
        spmmAW.buffer = spMatMulBuffer;

        spmmAH.preprocess();
        spmmAW.preprocess();
    }

    void updateW() {
        // The values of devAHDescr are stored starting at the devAFPtr.
        DenseDeviceMatrix devAH(n_rows, rank, devAFPtr);
        // sparseDenseMatMul(devA, devH, devAHDescr, false, true, AH_SPMM_ALG);


        _DEV__SYNC_FOR_TIMING();
        auto t_start = std::chrono::high_resolution_clock::now();

        spmmAH.callAsync();

        _DEV__SYNC_FOR_TIMING();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "AH time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << "ms\n";



        denseMatMul(devH, devH, devFF.devPtr, false, true);

        DenseDeviceMatrix devWHH(n_rows, rank, devFFGPtr);
        denseMatMul(devW, devFF, devWHH.devPtr, false, false);

        multiplicativeUpdate(devW, devAH, devWHH);
    }

    void updateH(bool alsoComputeLoss) {
        // DenseDeviceMatrix devWT(rank, n_rows, devW.devPtr);
        // devWT.dnMatDescr = devWTDescr;

        // The values of devWADescr are stored starting at the devAFPtr.
        DenseDeviceMatrix devWA(rank, n_cols, devAFPtr);
        // We compute A^TW into a row major matrix. This leads to the
        // column-major devWA representing its tranpose W^TA.
        // sparseDenseMatMul(devA, devWT, devWADescr, true, true, AW_SPMM_ALG);



        _DEV__SYNC_FOR_TIMING();
        auto t_start = std::chrono::high_resolution_clock::now();

        spmmAW.callAsync();

        _DEV__SYNC_FOR_TIMING();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "AW time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << "ms\n";





        denseMatMul(devW, devW, devFF.devPtr, true, false);

        DenseDeviceMatrix devWWH(rank, n_cols, devFFGPtr);
        denseMatMul(devFF, devH, devWWH.devPtr, false, false);

        multiplicativeUpdate(devH, devWA, devWWH);

        if (alsoComputeLoss) {
            _computeLossAfterUpdateH();
        }
    }

    void _computeLossAfterUpdateH() {
        // This relies on devFF containing devWW, which happens in
        // the updateH step.
        DenseDeviceMatrix devWWH(rank, n_cols, devFFGPtr);
        // Need to recompute this since devH has since been updated.
        denseMatMul(devFF, devH, devWWH.devPtr, false, false);

        float pos;
        frobeniusProductToHost(devWWH, devH, devLossPosPtr, &pos);

        // Need to recompute this since the updateH step overwrote this.
        DenseDeviceMatrix devAH(n_rows, rank, devAFPtr);
        // sparseDenseMatMul(devA, devH, devAHDescr, false, true, AH_SPMM_ALG);
        spmmAH.callAsync();

        float neg;
        frobeniusProductToHost(devAH, devW, devLossNegPtr, &neg);

        // TODO: Maybe find better way of doing this, maybe callbacks or something?
        CUDA_CALL(cudaStreamSynchronize(stream));

        // std::cout << "(" << pos << ", " << neg << ")\n";

        this->loss = sqrt(sq_A_norm + pos - 2 * neg);
    }

public:
    void run() {
        // Doesn't clean anything up afterwards. Probably should only be called once.
        moveAToDeviceAsync();
        initializeFactor(devW);
        initializeFactor(devH);
        initializeScalars();

        computeASqNorm();

        // Allocates buffers and stuff for sparse-dense matrix multiplies.
        preprocessSparseMatMuls();

        for (int step = 0; step < max_iters; step++) {
            // Remember stream synchronization.

            updateW();
            updateH(/*alsoComputeLoss=*/ true);

            CUDA_CALL(cudaStreamSynchronize(stream));

            // TODO: Remove this.
            std::cout << loss << '\n';
        }
    }

private:

    void computeASqNorm() {
        // A must be on device.
        CUBLAS_CALL(cublasSnrm2(denseHandle, devA.nnz, devA.devCsrValA, 1, devNormAPtr));

        float A_norm;
        CUDA_CALL(
            cudaMemcpyAsync(&A_norm, devNormAPtr, sizeof(float), cudaMemcpyDeviceToHost, stream)
        );

        // TODO: Probably find better way of doing this, maybe callbacks or something?
        CUDA_CALL(cudaStreamSynchronize(stream));

        this->sq_A_norm = A_norm * A_norm;
    }

    void denseMatMul(DenseDeviceMatrix& A, DenseDeviceMatrix& B, float* devCPtr, bool transposeA, bool transposeB) {
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
        long n = F.nEntries();

        // TODO: Figure out how to set these.
        long blockSize = 256;
        long numBlocks = (n + blockSize - 1) / blockSize;

        kernelNmfMuUpdate<<<numBlocks, blockSize, 0, stream>>>(n, F.devPtr, numer.devPtr, denom.devPtr, eps);
    }

    void frobeniusProductToHost(DenseDeviceMatrix& A, DenseDeviceMatrix& B, float* devResultPtr, float* hostResultPtr) {
        if (A.n_cols != B.n_cols || A.n_rows != B.n_rows) {
            throw;
        }
        CUBLAS_CALL(cublasSdot(
            denseHandle, A.nEntries(),
            A.devPtr, 1,
            B.devPtr, 1,
            devResultPtr
        ));
        CUDA_CALL(
            cudaMemcpyAsync(hostResultPtr, devResultPtr, sizeof(float), cudaMemcpyDeviceToHost, stream)
        );
    }


};

#endif
