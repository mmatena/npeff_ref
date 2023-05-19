#pragma once
// Constructs the CSR-matrix that will, after further preprocessing, be
// passed for decomposition. Technically represents a partially flatten
// version of a ragged rank 3 tensor.

#include <containers/sparse_matrix.h>
#include <inputs/lvrm_pefs.h>
#include <util/array_util.h>

namespace npeff {
namespace preprocessing {


namespace internal_ {

template<typename IndT>
std::unique_ptr<IndT> make_csr_matrix_col_indices(inputs::LvrmPefs& pefs) {
    int64_t nnz = pefs.values->n_entries;
    std::unique_ptr<IndT> col_indices = std::move(std::unique_ptr<IndT>(new IndT[nnz]));
    npeff::util::convert_numeric_arrays<IndT, int32_t>(col_indices.get(), pefs.row_indices->data.get(), nnz);
    return col_indices;
}

template<>
std::unique_ptr<int32_t> make_csr_matrix_col_indices<int32_t>(inputs::LvrmPefs& pefs) {
    return std::move(pefs.row_indices->data);
}

}  // internal_


bool can_csr_matrix_use_int32_indices(inputs::LvrmPefs& pefs) {
    return pefs.values->n_entries < INT32_MAX;
}


// We move some of the unique_ptrs from the pefs to the returned
// CSR matrix. The values matrix and maybe the row_indices matrix datas
// will have been moved to the returned CsrMatrix.
template<typename IndT>
CsrMatrix<IndT> construct_csr_matrix(inputs::LvrmPefs& pefs) {
    int64_t n_examples = pefs.n_examples();
    int64_t n_values_per_example = pefs.n_values_per_example();

    int64_t n_rows = pefs.sum_of_ranks();
    int64_t n_cols = pefs.n_parameters;
    int64_t nnz = pefs.values->n_entries;

    // We use the array directly from the pefs if the types are correct.
    // Otherwise we have to create our own.
    std::unique_ptr<IndT> col_indices = internal_::make_csr_matrix_col_indices<IndT>(pefs);

    // We can use the values array directly from the pefs.
    CsrMatrix<IndT> ret(n_rows, n_cols, nnz,
                        std::move(pefs.values->data),
                        std::unique_ptr<IndT>(new IndT[n_rows + 1]),
                        std::move(col_indices));

    // Compute the row offsets.
    int32_t* col_sizes = pefs.col_sizes->data.get();
    IndT* row_offsets = ret.row_offsets.get();

    row_offsets[0] = 0;
    for(int64_t i = 0; i < n_rows; i++) {
        row_offsets[i + 1] = row_offsets[i] + col_sizes[i];
    }

    THROW_IF_FALSE(row_offsets[n_rows] == nnz);

    return ret;
}



}  // preprocessing
}  // npeff
