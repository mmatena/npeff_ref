#pragma once
#include <iostream>

#include <util/cuda_system.h>
#include <util/cuda_statuses.h>
#include <util/matrices.h>
#include <util/misc_util.h>


long getSizeOfSplit(long totalSize, long n_splits, long splitIndex) {
    // The last split will be the one with potentially a few extra elements.
    long baselineSize = totalSize / n_splits;
    if (splitIndex >= n_splits) {
        THROW;
    } else if(splitIndex == n_splits - 1) {
        return baselineSize + (totalSize % n_splits);
    } else {
        return baselineSize;
    }
}


template <typename IndT>
ElCsrMatrix<IndT>* splitColumnWise(ElCsrMatrix<IndT>& A, int n_splits) {
    // NOTE: Not the most efficient algorithm.

    long n_rows = A.n_rows;
    long n_cols = A.n_cols;

    ElCsrMatrix<IndT> *splits = (ElCsrMatrix<IndT>*) malloc(sizeof(ElCsrMatrix<IndT>) * n_splits);

    long startCol = 0;
    for (long i=0; i<n_splits; i++) {
        long n_colsInSlice = getSizeOfSplit(n_cols, n_splits, i);

        long endCol = startCol + n_colsInSlice;

        long nnz = 0;
        for (long j=0; j < A.nnz; j++) {
            if (startCol <= A.csrColIndA[j] && A.csrColIndA[j] < endCol) {
                nnz++;
            }
        }


        ElCsrMatrix<IndT> *mat = new(splits + i) ElCsrMatrix<IndT>(n_rows, n_colsInSlice, nnz);
        // csrRowPtrA
        mat->csrRowPtrA[0] = 0;

        long k = 0;
        long rowIndex = 0;
        for (long j=0; j < A.nnz; j++) {
            // TODO: Double check that this is correct.
            while(rowIndex <= n_rows && A.csrRowPtrA[rowIndex] == j) {
                mat->csrRowPtrA[rowIndex++] = k;
            }
            if (startCol <= A.csrColIndA[j] && A.csrColIndA[j] < endCol) {
                mat->csrValA[k] = A.csrValA[j];
                mat->csrColIndA[k] = A.csrColIndA[j] - startCol;
                k++;
            }
        }
        mat->csrRowPtrA[n_rows] = nnz;

        startCol += n_colsInSlice;
    }

    return splits;
}


float* getColumnWiseSplitStart(MeMatrix& A, int n_splits, int split) {
    long colIndex = split * (A.n_cols / n_splits);
    return A.data + colIndex * A.n_rows;
}
