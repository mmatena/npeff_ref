#pragma once
// Converting between container types. Mostly for
// testing and debugging purposes.
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "./dense_matrix.h"
#include "./sparse_matrix.h"

namespace npeff {
namespace containers {


///////////////////////////////////////////////////////////////////////////////
// Constructing DenseMatrix

// Intended mostly for use in tests. Makes a copy of the values.
template<typename T>
DenseMatrix<T> dense_from_row_major_array(
    T* values,
    int64_t n_rows,
    int64_t n_cols
) {
    std::unique_ptr<T> values_transposed(new T[n_rows * n_cols]);

    // Transpose the values
    for(int64_t i=0; i<n_rows; i++) {
        for(int64_t j=0; j<n_cols; j++) {
            values_transposed.get()[j * n_rows + i] = values[i * n_cols + j];
        }
    }

    return DenseMatrix<T>(n_rows, n_cols, std::move(values_transposed));
}

// Intended mostly for use in tests. Makes a copy of the values.
template<typename T>
DenseMatrix<T> dense_from_col_major_array(
    T* values,
    int64_t n_rows,
    int64_t n_cols
) {
    std::unique_ptr<T> values_copy(new T[n_rows * n_cols]);
    std::copy(values, values + n_rows * n_cols, values_copy.get());
    return DenseMatrix<T>(n_rows, n_cols, std::move(values_copy));
}


///////////////////////////////////////////////////////////////////////////////
// Constructing CsrMatrix

// Intended mostly for use in tests.
template<typename IndT>
CsrMatrix<IndT> csr_from_dense(DenseMatrix<float>& dense, float threshold = 0.0f) {

    std::vector<float> v_values;
    std::vector<IndT> v_col_indices;
    IndT* row_offsets = new IndT[dense.n_rows + 1];
    row_offsets[0] = 0;

    for(int64_t i=0; i<dense.n_rows; i++) {
        int64_t row_nnz = 0;
        for(int64_t j=0; j<dense.n_cols; j++) {
            float entry = dense.get_entry(i, j);
            if (std::abs(entry) > threshold) {
                v_values.push_back(entry);
                v_col_indices.push_back(j);
                row_nnz++;
            }
        }
        row_offsets[i + 1] = row_offsets[i] + row_nnz;
    }

    int64_t nnz = row_offsets[dense.n_rows];
    float* values = new float[nnz];
    IndT* col_indices = new IndT[nnz];
    for(int64_t i=0; i<nnz; i++) {
        values[i] = v_values[i];
        col_indices[i] = v_col_indices[i];
    }

    return CsrMatrix<IndT>(dense.n_rows, dense.n_cols, nnz,
                           std::unique_ptr<float>(values),
                           std::unique_ptr<IndT>(row_offsets),
                           std::unique_ptr<IndT>(col_indices));
}


// Intended mostly for use in tests. Makes a copy of the values.
template<typename IndT>
CsrMatrix<IndT> csr_from_col_major_array(
    float* values,
    int64_t n_rows,
    int64_t n_cols,
    float threshold = 0.0f
) {
    std::unique_ptr<float> values_copy(new float[n_rows * n_cols]);
    std::copy(values, values + n_rows * n_cols, values_copy.get());
    DenseMatrix<float> dense(n_rows, n_cols, std::move(values_copy));
    return csr_from_dense<IndT>(dense, threshold);
}


// Intended mostly for use in tests. Makes a copy of the values.
template<typename IndT>
CsrMatrix<IndT> csr_from_row_major_array(
    float* values,
    int64_t n_rows,
    int64_t n_cols,
    float threshold = 0.0f
) {
    DenseMatrix<float> dense = dense_from_row_major_array(values, n_rows, n_cols);
    return csr_from_dense<IndT>(dense, threshold);
}


}  // containers
}  // npeff
