#pragma once
// Constructs the CSR-matrix that will, after further preprocessing, be
// passed for decomposition. Technically represents a rank 3 tensor.

#include <containers/sparse_matrix.h>
#include <inputs/lrm_pefs.h>
#include <util/array_util.h>

namespace npeff {
namespace preprocessing {


namespace internal_ {


template<typename IndT>
std::unique_ptr<IndT> make_csr_matrix_col_indices(inputs::LrmPefs& pefs) {
    int64_t nnz = pefs.values->n_entries;
    std::unique_ptr<IndT> col_indices = std::move(std::unique_ptr<IndT>(new IndT[nnz]));
    npeff::util::convert_numeric_arrays<IndT, int32_t>(col_indices.get(), pefs.row_indices->data.get(), nnz);
    return col_indices;
}

template<>
std::unique_ptr<int32_t> make_csr_matrix_col_indices<int32_t>(inputs::LrmPefs& pefs) {
    return std::move(pefs.row_indices->data);
}


}  // internal_


bool can_csr_matrix_use_int32_indices(inputs::LrmPefs& pefs) {
    return pefs.values->n_entries < INT32_MAX;
}

// We move some of the unique_ptrs from the pefs to the returned
// CSR matrix. The values matrix and maybe the row_indices matrix datas
// will have been moved to the returned CsrMatrix.
// 
// The col_offsets_non_cumulative parameter is here due to error in generating
// the PEFs file.
template<typename IndT>
CsrMatrix<IndT> construct_csr_matrix(inputs::LrmPefs& pefs, bool col_offsets_non_cumulative = false) {
    int64_t n_examples = pefs.n_examples();
    int64_t n_classes = pefs.n_classes;
    int64_t n_values_per_example = pefs.n_values_per_example();

    int64_t n_rows = n_examples * n_classes;
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
    for (int64_t i=0; i<n_examples; i++) {
        int64_t ret_base_offset = i * n_values_per_example;
        int32_t* ex_offsets_ptr = pefs.col_offsets->get_col_ptr(i);

        if (col_offsets_non_cumulative) {
            // NOTE: The top block of code might be wrong with a possible corrected version
            // below. I have no idea why the top version used to work.

            // NOTE: This was needed to due to error in generating pefs.
            int64_t running_offset = 0;
            for(int64_t j=0; j<n_classes; j++) {
                ret.row_offsets.get()[n_classes * i + j] = ret_base_offset + running_offset;
                running_offset += ex_offsets_ptr[j];
            }

            // // NOTE: This was needed to due to error in generating pefs.
            // int64_t running_offset = 0;
            // THROW_IF_FALSE(ex_offsets_ptr[0] == 0);
            // for(int64_t j=0; j<n_classes; j++) {
            //     ret.row_offsets.get()[n_classes * i + j] = ret_base_offset + running_offset;
            //     running_offset += ex_offsets_ptr[j + 1];
            // }
            // THROW_IF_FALSE(running_offset == n_values_per_example);


        } else {
            for(int64_t j=0; j<n_classes; j++) {
                ret.row_offsets.get()[n_classes * i + j] = ret_base_offset + ex_offsets_ptr[j];
            }
        }

    }

    ret.row_offsets.get()[n_rows] = nnz;

    return ret;
}


}  // preprocessing
}  // npeff
