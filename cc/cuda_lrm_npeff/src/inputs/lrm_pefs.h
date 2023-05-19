#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <containers/dense_matrix.h>

namespace npeff {
namespace inputs {


// Pretty much just mirrors what gets saved to the hdf5 file
// by the python code that generates the LRM-PEFs. Only the
// fields relevant to the decomposition are included though.
struct LrmPefs {

    int64_t n_classes;
    int64_t n_parameters;

    // Note that the shapes below are transposes of the python version
    // since the DenseMatrix class used here are column-major.

    // values.shape = [nnz_pef_example, n_examples]
    std::unique_ptr<npeff::DenseMatrix<float>> values = nullptr;
    // col_offsets.shape = [n_classes + 1, n_examples]
    std::unique_ptr<npeff::DenseMatrix<int32_t>> col_offsets = nullptr;
    // row_indices.shape = [nnz_pef_example, n_examples]
    std::unique_ptr<npeff::DenseMatrix<int32_t>> row_indices = nullptr;

    // pef_frobenius_norms.shape = [1, n_examples]
    std::unique_ptr<npeff::DenseMatrix<float>> pef_frobenius_norms = nullptr;

    int64_t n_examples() {
        return values->n_cols;
    }

    int64_t n_values_per_example() {
        return values->n_rows;
    }

    static LrmPefs load(std::string& filepath, int64_t n_examples=-1, int64_t examples_offset=0);

};


}  // inputs
}  // npeff
