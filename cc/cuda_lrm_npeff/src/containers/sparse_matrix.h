#pragma once
#include <cstddef>
#include <cstdint>
#include <climits>
#include <iostream>
#include <memory>
#include <unordered_set>

#include <util/array_util.h>
#include <util/macros.h>

namespace npeff {

template<typename IndT>
struct CsrMatrix {
    int64_t n_rows;
    int64_t n_cols;

    int64_t nnz;

    // values.length = nnz
    std::unique_ptr<float> values;
    // row_offsets.length = n_rows + 1
    std::unique_ptr<IndT> row_offsets;
    // col_indices.length = nnz
    std::unique_ptr<IndT> col_indices;

    CsrMatrix(int64_t n_rows, int64_t n_cols, int64_t nnz) :
        CsrMatrix(n_rows, n_cols, nnz,
                  std::unique_ptr<float>(new float[nnz]),
                  std::unique_ptr<IndT>(new IndT[n_rows + 1]),
                  std::unique_ptr<IndT>(new IndT[nnz]))
    {}
  
    CsrMatrix(int64_t n_rows, int64_t n_cols, int64_t nnz,
              std::unique_ptr<float> values,
              std::unique_ptr<IndT> row_offsets,
              std::unique_ptr<IndT> col_indices) :
        n_rows(n_rows),
        n_cols(n_cols),
        nnz(nnz),
        values(std::move(values)),
        row_offsets(std::move(row_offsets)),
        col_indices(std::move(col_indices))
    {}

    // NOTE: This only compares for an EXACT match. It is possible
    // to have different CSR-representations of the same matrix by
    // permuting the order of the col_indices and associated values
    // within a row. If the col_indices are sorted for each row, then
    // this should be an equality operator on the actual matrices.
    bool operator==(const CsrMatrix<IndT> &o) const {
        return n_rows == o.n_rows && n_cols == o.n_cols && nnz == o.nnz
            && util::arrays_are_equal<float>(values.get(), o.values.get(), nnz)
            && util::arrays_are_equal<IndT>(row_offsets.get(), o.row_offsets.get(), n_rows + 1)
            && util::arrays_are_equal<IndT>(col_indices.get(), o.col_indices.get(), nnz);
    }

    bool can_use_int32_indices() {
        return nnz < INT32_MAX && n_cols < INT32_MAX;
    }

    static std::unique_ptr<CsrMatrix<int32_t>> reindex_with_int32(std::unique_ptr<CsrMatrix<IndT>> mat) {
        THROW_IF_FALSE(mat->can_use_int32_indices());

        auto row_offsets = std::unique_ptr<int32_t>(new int32_t[mat->n_rows + 1]);
        util::convert_numeric_arrays(row_offsets.get(), mat->row_offsets.get(), mat->n_rows + 1);

        auto col_indices = std::unique_ptr<int32_t>(new int32_t[mat->nnz]);
        util::convert_numeric_arrays(col_indices.get(), mat->col_indices.get(), mat->nnz);

        return std::unique_ptr<CsrMatrix<int32_t>>(
            new CsrMatrix<int32_t>(mat->n_rows, mat->n_cols, mat->nnz,
                std::move(mat->values), std::move(row_offsets), std::move(col_indices)));
    }


    // Returns true if the indices are valid. Intended mostly for
    // debugging purposes.
    bool validate_indices() {
        // Validate row offsets.
        IndT* ros = row_offsets.get();
        if(ros[0] != 0 || ros[n_rows] != nnz) { return false; }
        for(int64_t i=0; i<n_rows; i++) {
            if(ros[i] > ros[i + 1]) { return false; }
        }

        // Validate col_indices
        IndT* cis = col_indices.get();
        for(int64_t i=0; i<nnz; i++) {
            if(cis[i] < 0 || cis[i] >= n_cols) { return false; }
        }

        // Check for duplicate col_indices in a row.
        for(int64_t i=0; i<n_rows; i++) {
            IndT row_start = ros[i];
            IndT row_end = ros[i+1];
            std::unordered_set<IndT> col_indices_set;
            for (int64_t j=row_start; j<row_end; j++) {
                col_indices_set.insert(cis[j]);
            }
            // std::cout << col_indices_set.size() << ", " << (row_end - row_start) << "\n";
            if (col_indices_set.size() != row_end - row_start) { return false; }
        }

        return true;
    }

    void print_representation() {
        std::cout << "values = {";
        for(int64_t i=0; i<nnz; i++) {
            std::cout << values.get()[i];
            if (i < nnz - 1)  std::cout << ", ";
        }
        std::cout << "}\n";

        std::cout << "row_offsets = {";
        for(int64_t i=0; i<n_rows + 1; i++) {
            std::cout << row_offsets.get()[i];
            if (i < n_rows)  std::cout << ", ";
        }
        std::cout << "}\n";

        std::cout << "col_indices = {";
        for(int64_t i=0; i<nnz; i++) {
            std::cout << col_indices.get()[i];
            if (i < nnz - 1)  std::cout << ", ";
        }
        std::cout << "}\n";

    }

};


}  // npeff
