#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <util/macros.h>

#include <containers/dense_matrix.h>

namespace npeff {
namespace inputs {


std::unique_ptr<npeff::DenseMatrix<int64_t>> compute_lvrm_example_row_offsets(
    npeff::DenseMatrix<int32_t>& ranks
);


// Pretty much just mirrors what gets saved to the hdf5 file
// by the python code that generates the LVRM-PEFs. Only the
// fields relevant to the decomposition are included though.
struct LvrmPefs {
    int64_t n_parameters;
    
    // Note that the shapes below are transposes of the python version
    // since the DenseMatrix class used here are column-major.

    // values.shape = [nnz_pef_example, n_examples]
    std::unique_ptr<npeff::DenseMatrix<float>> values = nullptr;
    // row_indices.shape = [nnz_pef_example, n_examples]
    std::unique_ptr<npeff::DenseMatrix<int32_t>> row_indices = nullptr;

    // ranks.shape = [1, n_examples]
    std::unique_ptr<npeff::DenseMatrix<int32_t>> ranks = nullptr;
    // col_sizes.shape = [1, sum(ranks)]
    std::unique_ptr<npeff::DenseMatrix<int32_t>> col_sizes = nullptr;

    // pef_frobenius_norms.shape = [1, n_examples]
    std::unique_ptr<npeff::DenseMatrix<float>> pef_frobenius_norms = nullptr;

    int64_t n_examples() {
        return values->n_cols;
    }

    int64_t n_values_per_example() {
        return values->n_rows;
    }

    int64_t sum_of_ranks() {
        return col_sizes->n_entries;
    }

    std::unique_ptr<npeff::DenseMatrix<int64_t>> compute_example_row_offsets() {
        auto example_row_offsets = compute_lvrm_example_row_offsets(*ranks);

        int64_t* data = example_row_offsets->data.get();
        THROW_IF_FALSE(data[n_examples()] == sum_of_ranks());

        return std::move(example_row_offsets);
    }

    static LvrmPefs load(std::string& filepath, int64_t n_examples=-1, int64_t examples_offset=0);

};



}  // inputs
}  // npeff
