#ifndef E8207BF9_2AF2_4DE3_9503_92A7BFD0B3C4
#define E8207BF9_2AF2_4DE3_9503_92A7BFD0B3C4


#include <math.h>
#include <ctime>
#include <chrono>
#include <type_traits>

#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <util/cuda_system.h>
#include <util/cuda_statuses.h>


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


template <typename IndT>
class CsrDeviceMatrix {
public:
    int64_t n_rows;
    int64_t n_cols;

    int64_t nnz;
    float* devCsrValA;
    IndT* devCsrRowPtrA;
    IndT* devCsrColIndA;

    cusparseSpMatDescr_t spMatDescr;

    CsrDeviceMatrix() {}

    CsrDeviceMatrix(int64_t n_rows, int64_t n_cols, int64_t nnz) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->nnz = nnz;
    }

    void allocDeviceMemory() {
        CUDA_CALL(cudaMalloc(&devCsrValA, sizeof(float) * nnz));
        CUDA_CALL(cudaMalloc(&devCsrRowPtrA, sizeof(IndT) * (n_rows + 1)));
        CUDA_CALL(cudaMalloc(&devCsrColIndA, sizeof(IndT) * nnz));

        createMatDescr();
    }

private:
    void createMatDescr() {
        // allocDeviceMemory should have been created already.
        cusparseIndexType_t indType;

        if (std::is_same<IndT, int32_t>::value) {
            indType = CUSPARSE_INDEX_32I;

        } else if(std::is_same<IndT, int64_t>::value) {
            indType = CUSPARSE_INDEX_64I;

        } else {
            std::cout << "Invalid type.\n";
            THROW;
        }

        CUSPARSE_CALL(
            cusparseCreateCsr(&spMatDescr, n_rows, n_cols, nnz,
                              devCsrRowPtrA, devCsrColIndA, devCsrValA,
                              indType, indType,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        );
    }
};


///////////////////////////////////////////////////////////////////////////////

template <typename IndT>
class SpDenseMatMul {
public:
    cusparseHandle_t sparseHandle;

    CsrDeviceMatrix<IndT> A;
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
        CsrDeviceMatrix<IndT> A, DenseDeviceMatrix B, cusparseDnMatDescr_t C,
        bool transposeA, bool transposeB,
        cusparseSpMMAlg_t alg,
        float* dev1f, float* dev0f
    ) : sparseHandle(sparseHandle),
        A(A), B(B), C(C),
        transposeA(transposeA), transposeB(transposeB),
        alg(alg),
        dev1f(dev1f), dev0f(dev0f) {}

    void createBuffer() {
        if (buffer != nullptr) {
            THROW;
        }
        CUSPARSE_CALL(
            cusparseSpMM_bufferSize(sparseHandle,
                         transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         dev1f, A.spMatDescr, B.dnMatDescr, dev0f, C, CUDA_R_32F,
                         alg, &bufferSize)
        );
        CUDA_CALL(cudaMalloc(&buffer, bufferSize));
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
        CUSPARSE_CALL(
            cusparseSpMM(sparseHandle,
                         transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                         dev1f, A.spMatDescr, B.dnMatDescr, dev0f, C, CUDA_R_32F,
                         alg, buffer)
        );
    }


};


#endif
