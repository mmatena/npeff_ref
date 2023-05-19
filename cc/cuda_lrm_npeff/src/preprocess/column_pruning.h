#pragma once

#include <memory>
#include <unordered_map>

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

namespace npeff {
namespace preprocessing {


// TODO: See if I can speed up some stuff using OpenMP.
template<typename IndT>
void prune_columns_in_place(
    CsrMatrix<IndT>* matrix,
    std::unique_ptr<DenseMatrix<IndT>>* new_to_old_col_indices,
    int64_t min_nnz_per_col
) {
    float* values = matrix->values.get();
    IndT* col_indices = matrix->col_indices.get();
    IndT* row_offsets = matrix->row_offsets.get();

    // Compute the number of non-zero entries per column.
    int32_t* col_counts = new int32_t[matrix->n_cols] {0};

    for(int64_t i=0; i<matrix->nnz; i++) {
        col_counts[col_indices[i]] += 1;
    }

    // Change from a map of old to new indices. Value of -1
    // means that the old index is not included in the new indices.
    IndT* old_to_new_col_indices = new IndT[matrix->n_cols];
    IndT new_index = 0;
    for(int64_t i=0; i<matrix->n_cols; i++) {
        if (col_counts[i] >= min_nnz_per_col) {
            old_to_new_col_indices[i] = new_index++;
        } else {
            old_to_new_col_indices[i] = -1;
        }
    }

    delete[] col_counts;
    int64_t new_n_cols = (int64_t) new_index;

    // Create the new_to_old_col_indices map.
    *new_to_old_col_indices = std::unique_ptr<DenseMatrix<IndT>>(new DenseMatrix<IndT>(1, new_n_cols));

    IndT* new_to_old_ptr = (*new_to_old_col_indices)->data.get();
    for(int64_t i=0, i2=0; i<matrix->n_cols; i++) {
        if (old_to_new_col_indices[i] != -1) {
            new_to_old_ptr[i2++] = i;
        }
    }

    // Re-index the matrix while removing pruned entries.
    int64_t row_index = 0;
    int64_t new_entry_index = 0;

    for(int64_t old_entry_index=0; old_entry_index<matrix->nnz; old_entry_index++) {
        IndT old_col_index = col_indices[old_entry_index];
        IndT new_col_index = old_to_new_col_indices[old_col_index];

        // Update the row offsets while correctly handling empty rows. Empty
        // rows do NOT get pruned by this method.
        while (row_offsets[row_index] == old_entry_index) {
            row_offsets[row_index++] = new_entry_index;
        }

        // If the column is included in the pruned matrix, then update
        // the values and col_indices arrays.
        if (new_col_index != -1) {
            col_indices[new_entry_index] = new_col_index;
            values[new_entry_index] = values[old_entry_index];
            new_entry_index++;
        }
    }

    // Update attributes of the new matrix.
    row_offsets[matrix->n_rows] = new_entry_index;
    matrix->nnz = new_entry_index;
    matrix->n_cols = new_n_cols;

    delete[] old_to_new_col_indices;
}


template<typename IndT>
void prune_columns_given_indices_in_place(
    CsrMatrix<IndT>* matrix,
    DenseMatrix<IndT>& new_to_old_col_indices
) {
    float* values = matrix->values.get();
    IndT* col_indices = matrix->col_indices.get();
    IndT* row_offsets = matrix->row_offsets.get();

    // TODO: See if there is a better/faster way to create this map.
    std::unordered_map<IndT, IndT> old_to_new_col_indices;
    for(int64_t i=0; i<new_to_old_col_indices.n_entries; i++) {
        old_to_new_col_indices[new_to_old_col_indices.data.get()[i]] = i;
    }

    int64_t row_index = 0;
    int64_t new_entry_index = 0;

    for(int64_t old_entry_index=0; old_entry_index<matrix->nnz; old_entry_index++) {
        IndT old_col_index = col_indices[old_entry_index];
        auto entry = old_to_new_col_indices.find(old_col_index);

        // Update the row offsets while correctly handling empty rows. Empty
        // rows do NOT get pruned by this method.
        while (row_offsets[row_index] == old_entry_index) {
            row_offsets[row_index++] = new_entry_index;
        }

        // If the column is included in the pruned matrix, then update
        // the values and col_indices arrays.
        if (entry != old_to_new_col_indices.end()) {
            IndT new_col_index = entry->second;

            col_indices[new_entry_index] = new_col_index;
            values[new_entry_index] = values[old_entry_index];
            new_entry_index++;
        }
    }

    // Update attributes of the new matrix.
    row_offsets[matrix->n_rows] = new_entry_index;
    matrix->nnz = new_entry_index;
    matrix->n_cols = new_to_old_col_indices.n_entries;
}


}  // preprocessing
}  // npeff
