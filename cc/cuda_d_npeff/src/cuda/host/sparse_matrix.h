#pragma once

#include <algorithm>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <misc/macros.h>
#include <misc/general.h>
#include <cuda/cuda_statuses.h>
#include <cuda/cuda_context.h>


// Forward declarations.
template<typename IndT>
class ElCsrMatrix;


namespace Cuda {
namespace Host {


template<typename IndT>
class CsrMatrix {
public:
    long n_rows;
    long n_cols;

    long nnz;

    CsrMatrix(long n_rows, long n_cols, long nnz) :
        n_rows(n_rows),
        n_cols(n_cols),
        nnz(nnz),
        values(new float[nnz]),
        row_offsets(new IndT[n_rows + 1]),
        col_indices(new IndT[nnz])
    {}
    
    ~CsrMatrix() {
        delete[] values;
        delete[] row_offsets;
        delete[] col_indices;
    }

    void ReIndexColumns_InPlace(IndT* indexToOgIndex, IndT fullDenseSize);

    // Modifies mat in place. As of now, this does not free up any memory associated
    // with the matrix.
    void RetainColumns_InPlace(const std::vector<IndT>& col_indices);

    std::vector<CsrMatrix<IndT>> SplitColumnWise(int n_splits);
    std::vector<CsrMatrix<IndT>> SplitRowWise(int n_splits);


    // No copying.
    CsrMatrix(CsrMatrix<IndT>& o) = delete;
    CsrMatrix(const CsrMatrix<IndT>& o) = delete;
    CsrMatrix<IndT>& operator=(const CsrMatrix<IndT>&) = delete;


    CsrMatrix(CsrMatrix<IndT>&& o) : 
        n_rows(o.n_rows),
        n_cols(o.n_cols),
        nnz(o.nnz),
        values(o.values),
        row_offsets(o.row_offsets),
        col_indices(o.col_indices)
    {
        o.values = nullptr;
        o.row_offsets = nullptr;
        o.col_indices = nullptr;
    }


    bool CanUseInt32Indices() { return nnz < INT32_MAX; }

    // Returns a copy and leaves this unchanged.
    // TODO: Maybe some special handling when IndT == int32_t.
    CsrMatrix<int32_t> ReIndexWithInt32() {
        THROWSERT(CanUseInt32Indices());
        CsrMatrix<int32_t> ret(n_rows, n_cols, nnz);
        std::memcpy(ret.values, this->values, sizeof(float) * nnz);
        Misc::ConvertNumericArrays(ret.row_offsets, this->row_offsets, n_rows + 1);
        Misc::ConvertNumericArrays(ret.col_indices, this->col_indices, nnz);
        return ret;
    }



protected:
    // These are pointers on the host memory. The memory IS owned
    // by this class.
    float * values;
    IndT * row_offsets;
    IndT * col_indices;


private:
    CsrMatrix(long n_rows, long n_cols, long nnz, float* values, IndT* row_offsets, IndT* col_indices) :
        n_rows(n_rows),
        n_cols(n_cols),
        nnz(nnz),
        values(values),
        row_offsets(row_offsets),
        col_indices(col_indices)
    {}

    static long GetSizeOfSplit(long totalSize, long n_splits, long splitIndex) {
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


    friend class ::DeviceCudaContext;
    friend class ::ElCsrMatrix<IndT>;

    template <typename IndT_>
    friend class CsrMatrix;
};


template<typename IndT>
void CsrMatrix<IndT>::ReIndexColumns_InPlace(IndT* col_index_to_new_index, IndT new_n_cols) {
    n_cols = new_n_cols;
    for(long i=0;i<nnz;i++) {
        col_indices[i] = col_index_to_new_index[col_indices[i]];
    }
}



template<typename IndT>
void CsrMatrix<IndT>::RetainColumns_InPlace(const std::vector<IndT>& retained_col_indices) {
    IndT* old_to_new_col_index = new IndT[n_cols];
    std::fill_n(old_to_new_col_index, n_cols, -1);

    for(long i=0; i<retained_col_indices.size(); i++) {
        old_to_new_col_index[retained_col_indices[i]] = i;
    }
    this->n_cols = retained_col_indices.size();

    long i2 = 0;
    long rowIndex = 0;
    for (long i=0; i < this->nnz; i++) {
        IndT newColInd = old_to_new_col_index[col_indices[i]];

        while (i == row_offsets[rowIndex]) {
            row_offsets[rowIndex++] = i2;
        }

        if (newColInd != -1) {
            col_indices[i2] = newColInd;
            values[i2] = values[i];
            i2++;
        }
    }
    // i2 will now be the nnz of the new matrix.
    this->nnz = i2;
    row_offsets[rowIndex] = i2;

    delete[] old_to_new_col_index;
}



template<typename IndT>
std::vector<CsrMatrix<IndT>> CsrMatrix<IndT>::SplitColumnWise(int n_splits) {
    std::vector<CsrMatrix<IndT>> splits;

    long startCol = 0;
    // TODO: Modify to use OpenMP for parallism across splits.
    for (long i=0; i<n_splits; i++) {
        long n_colsInSlice = GetSizeOfSplit(n_cols, n_splits, i);

        long endCol = startCol + n_colsInSlice;

        long nnz = 0;
        for (long j=0; j < this->nnz; j++) {
            if (startCol <= col_indices[j] && col_indices[j] < endCol) {
                nnz++;
            }
        }

        splits.emplace_back(n_rows, n_colsInSlice, nnz);
        auto& mat = splits[i];
        mat.row_offsets[0] = 0;

        long k = 0;
        long rowIndex = 0;
        for (long j=0; j < this->nnz; j++) {
            // TODO: Double check that this is correct.
            while(rowIndex <= n_rows && row_offsets[rowIndex] == j) {
                mat.row_offsets[rowIndex++] = k;
            }
            if (startCol <= col_indices[j] && col_indices[j] < endCol) {
                mat.values[k] = values[j];
                mat.col_indices[k] = col_indices[j] - startCol;
                k++;
            }
        }
        mat.row_offsets[n_rows] = nnz;

        startCol += n_colsInSlice;
    }

    return splits;
}


template<typename IndT>
std::vector<CsrMatrix<IndT>> CsrMatrix<IndT>::SplitRowWise(int n_splits) {
    std::vector<CsrMatrix<IndT>> splits;

    long start_row = 0;
    // TODO: Modify to use OpenMP for parallism across splits.
    for (long i=0; i<n_splits; i++) {
        long n_rows_in_slice = GetSizeOfSplit(n_rows, n_splits, i);
        long end_row = start_row + n_rows_in_slice;

        long start_offset = row_offsets[start_row];
        long split_nnz = row_offsets[end_row] - start_offset;

        splits.emplace_back(n_rows_in_slice, n_cols, split_nnz);
        auto& mat = splits[i];

        std::memcpy(mat.values, values + start_offset, sizeof(float) * split_nnz);
        std::memcpy(mat.col_indices, col_indices + start_offset, sizeof(IndT) * split_nnz);

        for(long j=0;j<=n_rows_in_slice;j++) {
            mat.row_offsets[j] = row_offsets[start_row + j] - start_offset;
        }

        start_row += n_rows_in_slice;
    }

    return splits;
}



} // Host
} // Cuda

