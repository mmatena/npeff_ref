#ifndef Z2E5450D0_21E0_4AA1_84D2_27F1BCBD646A_H
#define Z2E5450D0_21E0_4AA1_84D2_27F1BCBD646A_H

#include <vector>
#include <numeric>
#include <algorithm>

#include <util/matrices.h>

// namespace _TEMP {

//     template <typename T>
//     std::vector<int> argsort(const T* v, const int size) {

//       // initialize original index locations
//       std::vector<int> idx(size);
//       std::iota(idx.begin(), idx.end(), 0);

//       // sort indexes based on comparing values in v
//       // using std::stable_sort instead of std::sort
//       // to avoid unnecessary index re-orderings
//       // when v contains elements of equal values 
//       std::stable_sort(idx.begin(), idx.end(),
//            [&v](int i1, int i2) {return v[i1] < v[i2];});

//       return idx;
//     }

// }


template <typename IndT>
void removeAllZeroColumns_inPlace(ElCsrMatrix<IndT>* mat, IndT** newToOgIndex2) {
    // Modifies mat in place.
    // TODO: Add option to remove columns with fewer than k non-zero entries.

    long n_cols = mat->n_cols;
    long nnz = mat->nnz;
    IndT* csrColIndA = mat->csrColIndA;

    char* colMask = new char[n_cols] {0};
    for (long i=0; i<nnz; i++) {
        colMask[csrColIndA[i]] = 1;
    }

    IndT* ogToNewIndex = new IndT[n_cols];
    std::vector<IndT> newToOgIndex;

    long newIndex = 0;
    for (long i=0; i<n_cols; i++) {
        ogToNewIndex[i] = newIndex;
        if (colMask[i] != 0) {
            newToOgIndex.push_back(i);
            newIndex++;
        }
    }

    long n_newCols = newToOgIndex.size();

    // Modify the matrix.
    mat->n_cols = n_newCols;
    for (long i=0; i < nnz; i++) {
        csrColIndA[i] = ogToNewIndex[csrColIndA[i]];
    }

    delete[] colMask;
    delete[] ogToNewIndex;

    IndT* newToOgIndex3 = new IndT[n_newCols];
    for (long i=0; i<n_newCols; i++) {
        newToOgIndex3[i] = newToOgIndex[i];
    }

    *newToOgIndex2 = newToOgIndex3;
}


template <typename IndT>
void removeSmallestL0NormColumns_inPlace(
    ElCsrMatrix<IndT>* mat,
    IndT** newToOgIndex2,
    int minL0Norm = 1,
    const std::vector<IndT>& forceRetainColInds = {}
) {
    // Modifies mat in place. As of now, this does not free up any memory associated
    // with the matrix.

    long n_cols = mat->n_cols;
    long nnz = mat->nnz;
    float* csrValA = mat->csrValA;
    IndT* csrRowPtrA = mat->csrRowPtrA;
    IndT* csrColIndA = mat->csrColIndA;

    int* colMask = new int[n_cols] {0};
    for (long i=0; i<nnz; i++) {
        colMask[csrColIndA[i]] += 1;
    }
    for (IndT i : forceRetainColInds) {
        colMask[i] = minL0Norm;
    }


    // A value of negative 1 means that the column's L0 norm
    // is too small.
    IndT* ogToNewIndex = new IndT[n_cols];
    std::vector<IndT> newToOgIndex;

    long newIndex = 0;
    for (long i=0; i<n_cols; i++) {
        if (colMask[i] >= minL0Norm) {
            ogToNewIndex[i] = newIndex;
            newToOgIndex.push_back(i);
            newIndex++;
        } else {
            ogToNewIndex[i] = -1;
        }
    }

    long n_newCols = newToOgIndex.size();

    // // TODO: Can potentially reduce the size of the matrix.
    // THROW;

    // Modify the matrix.
    mat->n_cols = n_newCols;

    long i2 = 0;
    long rowIndex = 0;

    // TODO: Double check this.
    for (long i=0; i < nnz; i++) {
        IndT newColInd = ogToNewIndex[csrColIndA[i]];

        while (i == csrRowPtrA[rowIndex]) {
            csrRowPtrA[rowIndex++] = i2;
        }

        // TODO: Is this is wrong? Shouldn't the check be (ogToNewIndex[newColInd] != -1) ?
        if (newColInd != -1) {
            csrColIndA[i2] = newColInd;
            csrValA[i2] = csrValA[i];
            i2++;
        }
    }

    // i2 will now be the nnz of the new matrix.
    mat->nnz = i2;
    csrRowPtrA[rowIndex] = i2;

    delete[] colMask;
    delete[] ogToNewIndex;

    IndT* newToOgIndex3 = new IndT[n_newCols];
    for (long i=0; i<n_newCols; i++) {
        newToOgIndex3[i] = newToOgIndex[i];
    }

    *newToOgIndex2 = newToOgIndex3;
}


// void sortColumnIndices_inPlace(ElCsrMatrix* mat) {
//     const int n_rows = fishers->n_rows;
//     const int* csrRowPtrA = fishers->csrRowPtrA;

//     int* csrColIndA = fishers->csrColIndA;
//     float* csrValA = fishers->csrValA;

//     for (int rowIndex = 0; rowIndex < n_rows; rowIndex++) {
//         int start = csrRowPtrA[rowIndex];
//         int end = csrRowPtrA[rowIndex + 1];

//         std::vector<int> asdf =  _TEMP::argsort(csrColIndA + start, end - start);

//         for (int j = start; j < end; j++) {


//         }
//     }
// }


#endif
