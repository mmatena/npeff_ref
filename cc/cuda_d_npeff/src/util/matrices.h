#ifndef E2771048_5380_414A_8462_D5298FF40869_MATRICES_H
#define E2771048_5380_414A_8462_D5298FF40869_MATRICES_H
/* Code for matrices. */

#include <math.h>
#include <stdlib.h>
#include <random>
#include <cstring>
#include <type_traits>

#include <util/cuda_statuses.h>

#include <cuda/host/dense_matrix.h>
#include <cuda/host/sparse_matrix.h>


// Single precision, column-major matrix.
class MeMatrix {
    using DenseMatrix = Cuda::Host::DenseMatrix;

public:
    long n_rows;
    long n_cols;
    float *data;

    MeMatrix(long n_rows, long n_cols) {
        this->n_cols = n_cols;
        this->n_rows = n_rows;
        this->data = (float*) malloc(sizeof(float) * (long) n_rows * (long) n_cols);
    }

    ~MeMatrix() {
        free(this->data);
    }

    long nEntries() {
        return (long) n_rows * (long) n_cols;
    }

    long sizeInBytes() {
        return sizeof(float) * nEntries();
    }

    float* getEntryPtr(long i, long j) {
        // i = row index, j = column index
        return data + (j * n_rows + i);
    }

    void toRowMajor_inPlace() {
        // NOTE: Kind of a temporary measure. Maybe not even that efficient.
        // MeMatrix ret(n_rows, n_cols);
        long n_rows = (long) this->n_rows;
        long n_cols = (long) this->n_cols;
        float* newData = (float*) malloc(sizeof(float) * n_rows * n_cols);
        for (long i=0; i < n_rows; i++) {
            for (long j=0; j < n_cols; j++) {
                newData[i * n_cols + j] = data[j * n_rows + i];
            }
        }
        free(data);
        data = newData;
    }

    double frobeniusNorm() {
        double sqNorm = 0.0;
        for (long i = 0; i < nEntries(); i++) {
            sqNorm += data[i] * data[i];
        }
        return sqrt(sqNorm);
    }


    DenseMatrix move_to_dense_matrix() {
        DenseMatrix ret(n_rows, n_cols, data);
        data = nullptr;
        return ret;
    }

    static MeMatrix MoveFromDenseMatrix(DenseMatrix& mat) {
        MeMatrix ret(mat.n_rows, mat.n_cols, mat.data);
        mat.data = nullptr;
        return ret;
    }

    static MeMatrix multiply(MeMatrix& A, MeMatrix& B) {
        if (A.n_cols != B.n_rows) {
            THROW;
        }

        MeMatrix C(A.n_rows, B.n_cols);
        for (long i = 0; i < C.nEntries(); i++) {
            C.data[i] = 0.0f;
        }

        for (long i = 0; i < A.n_rows; i++) {
            for (long j = 0; j < A.n_cols; j++) {
                for (long k = 0; k < B.n_cols; k++) {
                    *C.getEntryPtr(i, k) += *A.getEntryPtr(i, j) * *B.getEntryPtr(j, k);
                }
            }
        }
        return C;
    }

    static MeMatrix subtract(MeMatrix& A, MeMatrix& B) {
        if (A.n_rows != B.n_rows || A.n_cols != B.n_cols) {
            THROW;
        }

        MeMatrix C(A.n_rows, A.n_cols);
        for (long i = 0; i < C.nEntries(); i++) {
            C.data[i] = A.data[i] - B.data[i];
        }

        return C;
    }

    static MeMatrix random_uniform_matrix(long n_rows, long n_cols, long seed = std::default_random_engine::default_seed) {
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<float> distribution(0.0,1.0);

        MeMatrix matrix(n_rows, n_cols);
        for(long i=0; i < n_rows * n_cols; i++) {
            matrix.data[i] = distribution(generator);
        }
        return matrix;
    }

protected:

    MeMatrix(long n_rows, long n_cols, float *data) {
        this->n_cols = n_cols;
        this->n_rows = n_rows;
        this->data = data;
    }
};


MeMatrix random_matrix(long n_rows, long n_cols) {
    MeMatrix matrix(n_rows, n_cols);
    for(long i=0; i < n_rows * n_cols; i++) {
        matrix.data[i] = (float) rand() / (float) RAND_MAX;
    }
    return matrix;
}


///////////////////////////////////////////////////////////////////////////////
// TODO: Move into some common utility file:

template <typename T>
bool areArraysEqual(T a, T b, long n) {
    for (long i=0; i<n; i++) {
        if (a[i] != b[i]) {
            // std::cout << i << ": " << a[i] << ", " << b[i] << "\n";
            return false;
        }
    }
    return true;
}


template<typename S, typename T>
void convertNumericArrays(S* a, T* b, long n) {
    for (long i=0; i<n; i++) {
        b[i] = static_cast<T>(a[i]);
    }
}


///////////////////////////////////////////////////////////////////////////////

template <typename IndT>
class ElCsrMatrix {
    using CsrMatrix = Cuda::Host::CsrMatrix<IndT>;

public:
    long n_rows;
    long n_cols;

    long nnz;
    float* csrValA;
    IndT* csrRowPtrA;
    IndT* csrColIndA;

    ElCsrMatrix(int n_rows, int n_cols, long nnz) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->nnz = nnz;

        this->csrValA = (float*) malloc(sizeof(float) * nnz);
        this->csrRowPtrA = (IndT*) malloc(sizeof(IndT) * (n_rows + 1));
        this->csrColIndA = (IndT*) malloc(sizeof(IndT) * nnz);
    }

    ~ElCsrMatrix() {
        free(csrValA);
        free(csrRowPtrA);
        free(csrColIndA);
    }

    double density() {
        return (double) nnz / (double) (n_rows * n_cols);
    }

    MeMatrix asDenseMatrix() {
        MeMatrix ret(n_rows, n_cols);
        for (long rowIndex = 0; rowIndex < n_rows; rowIndex++) {
            IndT start = csrRowPtrA[rowIndex];
            IndT end = csrRowPtrA[rowIndex + 1];

            for (long j = start; j < end; j++) {
                *ret.getEntryPtr(rowIndex, csrColIndA[j]) = csrValA[j];
            }
        }
        return ret;
    }

    bool canReindexWithInt32() {
        if (std::is_same<IndT, int32_t>::value) {
            return true;
        }
        return nnz > INT32_MAX;
    }

    ElCsrMatrix<int32_t> reindexWithInt32() {
        ElCsrMatrix<int32_t> ret(n_rows, n_cols, nnz);
        std::memcpy(ret.csrValA, csrValA, sizeof(float) * nnz);
        convertNumericArrays(ret.csrRowPtrA, csrRowPtrA, n_rows + 1);
        convertNumericArrays(ret.csrColIndA, csrColIndA, nnz);
        return ret;
    }

    ElCsrMatrix<IndT> clone() {
        // This is a deep clone.
        ElCsrMatrix ret(n_rows, n_cols, nnz);
        std::memcpy(ret.csrValA, csrValA, sizeof(float) * nnz);
        std::memcpy(ret.csrRowPtrA, csrRowPtrA, sizeof(IndT) * (n_rows + 1));
        std::memcpy(ret.csrColIndA, csrColIndA, sizeof(IndT) * nnz);
        return ret;
    }

    CsrMatrix move_to_csr_matrix() {
        CsrMatrix ret(n_rows, n_cols, nnz, csrValA, csrRowPtrA, csrColIndA);
        csrValA = nullptr;
        csrRowPtrA = csrColIndA = nullptr;
        return ret;
    }

    static ElCsrMatrix MoveFromSparseMatrix(CsrMatrix& mat) {
        ElCsrMatrix ret(mat.n_rows, mat.n_cols, mat.nnz, mat.values, mat.row_offsets, mat.col_indices);
        mat.values = nullptr;
        mat.row_offsets = nullptr;
        mat.col_indices = nullptr;
        return ret;
    }

    static bool areEqual(ElCsrMatrix<IndT>& A, ElCsrMatrix<IndT>& B) {
        if (A.n_rows != B.n_rows || A.n_cols != B.n_cols || A.nnz != B.nnz) {
            return false;
        }
        if (!areArraysEqual(A.csrRowPtrA, B.csrRowPtrA, A.n_rows + 1)) {
            return false;
        }
        if (!areArraysEqual(A.csrValA, B.csrValA, A.nnz)) {
            return false;
        }
        if (!areArraysEqual(A.csrColIndA, B.csrColIndA, A.nnz)) {
            return false;
        }
        return true;
    }

protected:

    ElCsrMatrix(int n_rows, int n_cols, long nnz, float* csrValA, IndT* csrRowPtrA, IndT* csrColIndA) :
        n_rows(n_rows),
        n_cols(n_cols),
        nnz(nnz),
        csrValA(csrValA),
        csrRowPtrA(csrRowPtrA),
        csrColIndA(csrColIndA)
    {}
};



template <typename IndT>
ElCsrMatrix<IndT> random_csr_matrix(long n_rows, long n_cols, float density, long seed = std::default_random_engine::default_seed) {
    // NOTE: THis is mostly for quick testing of my other code.
    // TODO: Make faster and perhaps add more options.
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(0.0,1.0);

    MeMatrix dense = random_matrix(n_rows, n_cols);

    int nnf = 0;
    for(long i=0; i<dense.nEntries(); i++) {
        if (distribution(generator) > density) {
            dense.data[i] = 0.0f;
        } else {
            nnf++;
        }
    }

    ElCsrMatrix<IndT> ret(n_rows, n_cols, nnf);
    ret.csrRowPtrA[0] = (IndT) 0;

    long k = 0;
    for (long i=0; i<n_rows; i++) {
        long nnzRow = 0;
        for (long j=0; j<n_cols; j++) {
            float entry = *dense.getEntryPtr(i, j);
            if (entry == 0.0f) continue;
            ret.csrValA[k] = entry;
            ret.csrColIndA[k] = (IndT) j;
            nnzRow++;
            k++;
        }
        ret.csrRowPtrA[i + 1] = (IndT) (ret.csrRowPtrA[i] + nnzRow);
    }

    return ret;
}

#endif
