#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <util/macros.h>
#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

namespace npeff {
namespace preprocessing {


// "Private" methods.
namespace partition_by_columns_internal {

// The last partition will be the one with potentially a few extra elements.
int64_t compute_partition_size(int64_t total_size, int64_t n_partitions, int64_t partition_index) {
    int64_t base_size = total_size / n_partitions;
    if (partition_index >= n_partitions) {
        THROW;
    } else if(partition_index == n_partitions - 1) {
        return base_size + (total_size % n_partitions);
    } else {
        return base_size;
    }
}

int64_t compute_partition_start_index(int64_t total_size, int64_t n_partitions, int64_t partition_index) {
    // This works because only the last partition can potentially have extra elements.
    int64_t base_size = total_size / n_partitions;
    return partition_index * base_size;
}


// TODO: See if I can parallize any of this, possibly with OpenMP.
template<typename IndT>
std::unique_ptr<CsrMatrix<IndT>> create_columnwise_partition(
    CsrMatrix<IndT>& matrix,
    int64_t p_start_col,
    int64_t p_n_cols
) {
    int64_t n_cols = matrix.n_cols;

    float* values = matrix.values.get();
    IndT* col_indices = matrix.col_indices.get();
    IndT* row_offsets = matrix.row_offsets.get();

    int64_t p_end_col = p_start_col + p_n_cols;

    // Verify that the start and end columns are valid.
    THROW_IF_FALSE(p_start_col < n_cols);
    THROW_IF_FALSE(p_end_col <= n_cols);

    // Figure out the number of entries in the partition.
    int64_t p_nnz = 0;
    for (int64_t i=0; i < matrix.nnz; i++) {
        if (p_start_col <= col_indices[i] && col_indices[i] < p_end_col) {
            p_nnz++;
        }
    }
    // Create the partition's matrix, which allocates the memory to store it.
    CsrMatrix<IndT>* partition = new CsrMatrix<IndT>(matrix.n_rows, p_n_cols, p_nnz);

    float* p_values = partition->values.get();
    IndT* p_col_indices = partition->col_indices.get();
    IndT* p_row_offsets = partition->row_offsets.get();

    // Fill out the partition matrix.
    p_row_offsets[0] = 0;

    long row_index = 0;
    long p_i = 0;
    for (long i=0; i < matrix.nnz; i++) {
        // TODO: Double check that this is correct.
        while(row_index <= matrix.n_rows && row_offsets[row_index] == i) {
            p_row_offsets[row_index++] = p_i;
        }
        if (p_start_col <= col_indices[i] && col_indices[i] < p_end_col) {
            p_values[p_i] = values[i];
            p_col_indices[p_i] = col_indices[i] - p_start_col;
            p_i++;
        }
    }

    p_row_offsets[matrix.n_rows] = p_nnz;

    return std::unique_ptr<CsrMatrix<IndT>>(partition);
}


template<typename T>
std::unique_ptr<DenseMatrix<T>> create_columnwise_partition(
    DenseMatrix<T>& matrix,
    int64_t p_start_col,
    int64_t p_n_cols
) {
    int64_t n_cols = matrix.n_cols;
    int64_t n_rows = matrix.n_rows;

    int64_t p_end_col = p_start_col + p_n_cols;

    // Verify that the start and end columns are valid.
    THROW_IF_FALSE(p_start_col < n_cols);
    THROW_IF_FALSE(p_end_col <= n_cols);

    DenseMatrix<T>* partition = new DenseMatrix<T>(n_rows, p_n_cols);

    int64_t start_offset = p_start_col * n_rows;
    int64_t end_offset = p_end_col * n_rows;
    T* matrix_data = matrix.data.get();

    std::copy(matrix_data + start_offset, matrix_data + end_offset, partition->data.get());

    return std::unique_ptr<DenseMatrix<T>>(partition);
}


// TODO: See if I can parallize any of this, possibly with OpenMP.
template<typename MatT>
std::unique_ptr<MatT> create_columnwise_uniform_partition(
    MatT& matrix,
    int64_t n_partitions,
    int64_t partition_index
) {
    int64_t n_cols = matrix.n_cols;

    // Figure out the size and position of the partition.
    int64_t p_start_col = compute_partition_start_index(n_cols, n_partitions, partition_index);
    int64_t p_n_cols = compute_partition_size(n_cols, n_partitions, partition_index);
    
    return create_columnwise_partition(matrix, p_start_col, p_n_cols);
}
// template<typename IndT>
// std::unique_ptr<CsrMatrix<IndT>> create_columnwise_uniform_partition(
//     CsrMatrix<IndT>& matrix,
//     int64_t n_partitions,
//     int64_t partition_index
// ) {
//     int64_t n_cols = matrix.n_cols;

//     // Figure out the size and position of the partition.
//     int64_t p_start_col = compute_partition_start_index(n_cols, n_partitions, partition_index);
//     int64_t p_n_cols = compute_partition_size(n_cols, n_partitions, partition_index);
    
//     return create_columnwise_partition(matrix, p_start_col, p_n_cols);
// }



} // partition_by_columns_internal


// Partitions the matrix by contiguous chunks of columns. There will be
// n_partitions in total. Each partition should have the same number of
// columns. If n_partitions does not evenly divide the number of columns,
// then the last partition will have extra columns.
template<typename MatT>
std::vector<std::unique_ptr<MatT>> partition_by_columns_uniformly(
    MatT& matrix,
    int64_t n_partitions
) {
    std::vector<std::unique_ptr<MatT>> partitions;
    for (int64_t i=0; i<n_partitions; i++) {
        auto partition = partition_by_columns_internal::create_columnwise_uniform_partition(matrix, n_partitions, i);
        partitions.push_back(std::move(partition));
    }
    return partitions;
}
// template<typename IndT>
// std::vector<std::unique_ptr<CsrMatrix<IndT>>> partition_by_columns_uniformly(
//     CsrMatrix<IndT>& matrix,
//     int64_t n_partitions
// ) {
//     std::vector<std::unique_ptr<CsrMatrix<IndT>>> partitions;
//     for (int64_t i=0; i<n_partitions; i++) {
//         auto partition = partition_by_columns_internal::create_columnwise_uniform_partition(matrix, n_partitions, i);
//         partitions.push_back(std::move(partition));
//     }
//     return partitions;
// }


template<typename IndT>
std::vector<std::unique_ptr<CsrMatrix<IndT>>> partition_by_columns_given_start_indices(
    CsrMatrix<IndT>& matrix,
    std::vector<IndT>& partition_start_column_indices
) {
    THROW_IF_FALSE(partition_start_column_indices[0] == 0);

    int64_t n_cols = matrix.n_cols;
    int64_t n_partitions = partition_start_column_indices.size();

    std::vector<std::unique_ptr<CsrMatrix<IndT>>> partitions;
    for (int64_t i=0; i<n_partitions; i++) {
        int64_t p_start_col = partition_start_column_indices[i];
        int64_t p_end_col = (i == n_partitions - 1) ? n_cols : partition_start_column_indices[i + 1];
        int64_t p_n_cols = p_end_col - p_start_col;

        auto partition = partition_by_columns_internal::create_columnwise_partition(matrix, p_start_col, p_n_cols);
        partitions.push_back(std::move(partition));
    }

    return partitions;
}


}  // preprocessing
}  // npeff
