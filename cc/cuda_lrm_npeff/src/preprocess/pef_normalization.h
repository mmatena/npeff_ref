#pragma once

#include <cmath>

#include <containers/sparse_matrix.h>
#include <inputs/lrm_pefs.h>

namespace npeff {
namespace preprocessing {

template<typename IndT>
void normalize_pefs_in_place(CsrMatrix<IndT>* matrix, inputs::LrmPefs& pefs, float eps = 1e-12) {
    int64_t n_classes = pefs.n_classes;
    int64_t n_examples = pefs.n_examples();

    float* values = matrix->values.get();
    IndT* row_offsets = matrix->row_offsets.get();
    float* pef_frobenius_norms = pefs.pef_frobenius_norms->data.get();

    for (int64_t row_index = 0; row_index < matrix->n_rows; row_index++) {
        // Recall that the rows of the matrix can be grouped into consecutive
        // chunks n_classes in size. These correspond to the factors of a single
        // example.
        int64_t example_index = row_index / n_classes;

        // We divide by the square root of the Frobenious norm of the PEF matrix.
        // This is because we are storing effectively a matrix A while the PEF matrix
        // is given by AA^T.
        float sqrt_norm = std::sqrt(pef_frobenius_norms[example_index]);
        sqrt_norm = std::max(sqrt_norm, eps);

        int64_t start = row_offsets[row_index];
        int64_t end = row_offsets[row_index + 1];

        for (int64_t j = start; j < end; j++) {
            values[j] /= sqrt_norm;
        }
    }
}

template<typename IndT>
void normalize_lvrm_pefs_in_place(
    CsrMatrix<IndT>* matrix,
    npeff::DenseMatrix<float>& pef_frobenius_norms_,
    npeff::DenseMatrix<int64_t>& example_row_offsets,
    float eps = 1e-6
) {
    int64_t n_examples = example_row_offsets.n_entries - 1;

    float* values = matrix->values.get();
    IndT* row_offsets = matrix->row_offsets.get();
    float* pef_frobenius_norms = pef_frobenius_norms_.data.get();

    for(int64_t example_index = 0; example_index < n_examples; example_index++) {
        // We divide by the square root of the Frobenious norm of the PEF matrix.
        // This is because we are storing effectively a matrix A while the PEF matrix
        // is given by AA^T.
        float sqrt_norm = std::sqrt(pef_frobenius_norms[example_index]);
        sqrt_norm = std::max(sqrt_norm, eps);

        int64_t ex_start_row_offset = example_row_offsets.data.get()[example_index];
        int64_t ex_end_row_offset = example_row_offsets.data.get()[example_index + 1];

        int64_t start = row_offsets[ex_start_row_offset];
        int64_t end = row_offsets[ex_end_row_offset];

        for (int64_t j = start; j < end; j++) {
            values[j] /= sqrt_norm;
        }
    }
}


}  // preprocessing
}  // npeff
