#include "./dense_matrix.h"

namespace npeff {
namespace gpu {


DenseMatrix::~DenseMatrix() {
    if (col_major_descr_ != nullptr) {
        cusparseDestroyDnMat(col_major_descr_);
    }
    if (transpose_row_major_descr_ != nullptr) {
        cusparseDestroyDnMat(transpose_row_major_descr_);
    }
    if(as_submatrix_cast_ != nullptr) {
        delete as_submatrix_cast_;
    }
}

SubDenseMatrix DenseMatrix::as_submatrix() {
    return SubDenseMatrix(*this, 0, 0, n_rows, n_cols);
}

SubDenseMatrix* DenseMatrix::get_as_submatrix_cast() {
    if(as_submatrix_cast_ == nullptr) {
        as_submatrix_cast_ = new SubDenseMatrix(*this, 0, 0, n_rows, n_cols);
    }
    return as_submatrix_cast_;
}

std::unique_ptr<SubDenseMatrix> DenseMatrix::create_submatrix(
    int64_t start_row, int64_t start_col,
    int64_t n_rows, int64_t n_cols)
{
    return SubDenseMatrix::make_unique_ptr(*this, start_row, start_col, n_rows, n_cols);
}


}  // gpu
}  // npeff
