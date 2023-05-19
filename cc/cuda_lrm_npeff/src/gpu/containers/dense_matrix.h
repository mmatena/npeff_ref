#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <gpu/macros.h>

namespace npeff {
namespace gpu {


// Fordward declarations.
struct DenseMatrix;
struct SubDenseMatrix;



// Assumed to be column major.
struct DenseMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t n_entries;

    // Pointer to memory on device. The memory is not owned
    // by this class.
    float const* data;

    DenseMatrix(int64_t n_rows, int64_t n_cols, float* data) :
        n_rows(n_rows), n_cols(n_cols), data(data), n_entries(n_rows * n_cols)
    {}

    ~DenseMatrix();

    // Returns an instance of SubDenseMatrix equivalent to the full matrix.
    SubDenseMatrix as_submatrix();

    SubDenseMatrix* get_as_submatrix_cast();

    std::unique_ptr<SubDenseMatrix> create_submatrix(
        int64_t start_row, int64_t start_col,
        int64_t n_rows, int64_t n_cols);

    static std::unique_ptr<DenseMatrix>
        make_unique_ptr(int64_t n_rows, int64_t n_cols, float* data)
    {
        return std::unique_ptr<DenseMatrix>(new DenseMatrix(n_rows, n_cols, data));
    }

    // Returns the cusparse's description of this matrix in
    // column major form, which is how you would regularly interpret
    // this matrix. The descr will be created if it has not already
    // been created.
    cusparseDnMatDescr_t get_col_major_descr() {
        if (col_major_descr_ == nullptr) {
            create_col_major_descr();
        }
        return col_major_descr_;
    }

    // Useful for doing dense-sparse matmuls using cusparse. The descr
    // will represent the transpose of this matrix, represented in row
    // major form.
    cusparseDnMatDescr_t get_transpose_row_major_descr() {
        if (transpose_row_major_descr_ == nullptr) {
            create_transpose_row_major_descr();
        }
        return transpose_row_major_descr_;
    }

protected:
    cusparseDnMatDescr_t col_major_descr_ = nullptr;
    cusparseDnMatDescr_t transpose_row_major_descr_ = nullptr;

    SubDenseMatrix* as_submatrix_cast_ = nullptr;

    void create_col_major_descr() {
        CUSPARSE_CALL(
            cusparseCreateDnMat(&col_major_descr_, n_rows, n_cols, n_rows, (float*) data,
                                CUDA_R_32F, CUSPARSE_ORDER_COL)
        );
    }

    void create_transpose_row_major_descr() {
        CUSPARSE_CALL(
            cusparseCreateDnMat(&transpose_row_major_descr_, n_cols, n_rows, n_rows, (float*) data,
                                CUDA_R_32F, CUSPARSE_ORDER_ROW)
        );
    }

};


// Submatrix of a DenseMatrix. The DenseMatrices are assumed to be
// column major.
struct SubDenseMatrix {
    DenseMatrix const& parent;

    const int64_t start_row;
    const int64_t start_col;

    const int64_t n_rows;
    const int64_t n_cols;

    const int64_t n_entries;

    SubDenseMatrix(
        DenseMatrix& parent,
        int64_t start_row, int64_t start_col,
        int64_t n_rows, int64_t n_cols
    ) :
        parent(parent),
        start_row(start_row), start_col(start_col),
        n_rows(n_rows), n_cols(n_cols),
        n_entries(n_rows * n_cols)
    {
        validate_construction();
    }

    float* get_data_ptr() const {
        return (float*) parent.data + start_col * parent.n_rows + start_row;
    }

    int64_t get_leading_dimension() const {
        return parent.n_rows;
    }

    static std::unique_ptr<SubDenseMatrix>
        make_unique_ptr(
            DenseMatrix& parent,
            int64_t start_row, int64_t start_col,
            int64_t n_rows, int64_t n_cols
        )
    {
        return std::unique_ptr<SubDenseMatrix>(new SubDenseMatrix(
            parent,
            start_row, start_col,
            n_rows, n_cols));
    }

protected:
    void validate_construction() {
        THROW_IF_FALSE(start_row >= 0 && start_col >= 0);
        THROW_IF_FALSE(start_row + n_rows <= parent.n_rows && start_col + n_cols <= parent.n_cols);
    }

};


}  // gpu
}  // npeff
