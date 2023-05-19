#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <gpu/macros.h>
#include <gpu/types.h>


namespace npeff {
namespace gpu {


template<typename IndT>
struct CsrMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t nnz;

    // These are pointers on the device memory. The memory is NOT owned
    // by this class.
    float const* values;
    IndT const* row_offsets;
    IndT const* col_indices;
    
    cusparseSpMatDescr_t descr;

    CsrMatrix(int64_t n_rows, int64_t n_cols, int64_t nnz,
              float* values,
              IndT* row_offsets,
              IndT* col_indices) :
        n_rows(n_rows),
        n_cols(n_cols),
        nnz(nnz),
        values(values),
        row_offsets(row_offsets),
        col_indices(col_indices)
    {
        cusparseIndexType_t ind_type = ToCuSparseIndexType<IndT>::value;
        CUSPARSE_CALL(
            cusparseCreateCsr(&descr, n_rows, n_cols, nnz,
                              row_offsets, col_indices, values,
                              ind_type, ind_type,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        );
    }

    ~CsrMatrix() {
        cusparseDestroySpMat(descr);
    }


    static std::unique_ptr<CsrMatrix<IndT>> make_unique_ptr(
        int64_t n_rows, int64_t n_cols, int64_t nnz,
        float* values,
        IndT* row_offsets,
        IndT* col_indices)
    {
        return std::unique_ptr<CsrMatrix<IndT>>(new CsrMatrix<IndT>(
            n_rows, n_cols, nnz, values, row_offsets, col_indices));
    }
};



}  // gpu
}  // npeff
