#include <catch2/catch_test_macros.hpp>

#include <preprocess/construct_csr_matrix.h>

#include <containers/conversions.h>
#include <util/array_util.h>

using npeff::CsrMatrix;
using npeff::DenseMatrix;
using npeff::inputs::LrmPefs;

using npeff::containers::dense_from_col_major_array;
using npeff::containers::csr_from_row_major_array;


LrmPefs make_test_lrm_pefs() {
    const int64_t n_classes = 2;
    const int64_t nnz_pef_example = 3;
    const int64_t n_examples = 4;

    LrmPefs pefs;

    pefs.n_classes = n_classes;
    pefs.n_parameters = 10;

    DenseMatrix<float> values = dense_from_col_major_array<float>(new float[nnz_pef_example * n_examples] {
         1.0,  2.0,  3.0,
         4.0,  5.0,  6.0,
         7.0,  8.0,  9.0,
        -1.0, -2.0, -3.0,
    }, nnz_pef_example, n_examples);

    DenseMatrix<int32_t> col_offsets = dense_from_col_major_array<int32_t>(new int32_t[(n_classes + 1) * n_examples] {
        0, 0, 3,
        0, 1, 3,
        0, 2, 3,
        0, 3, 3,
    }, n_classes + 1, n_examples);


    DenseMatrix<int32_t> row_indices = dense_from_col_major_array<int32_t>(new int32_t[nnz_pef_example * n_examples] {
        0, 8, 9,
        4, 4, 7,
        6, 9, 0,
        1, 2, 3,
    }, nnz_pef_example, n_examples);

    pefs.values = std::unique_ptr<DenseMatrix<float>>(new DenseMatrix<float>(std::move(values)));
    pefs.col_offsets = std::unique_ptr<DenseMatrix<int32_t>>(new DenseMatrix<int32_t>(std::move(col_offsets)));
    pefs.row_indices = std::unique_ptr<DenseMatrix<int32_t>>(new DenseMatrix<int32_t>(std::move(row_indices)));
    return pefs;
}


template<typename IndT>
CsrMatrix<IndT> make_expected_A() {
    return CsrMatrix<IndT>(
        8, 10, 12,
        std::unique_ptr<float>(new float[12] {
             1.0,  2.0,  3.0,
             4.0,  5.0,  6.0,
             7.0,  8.0,  9.0,
            -1.0, -2.0, -3.0,
        }),
        std::unique_ptr<IndT>(new IndT[9] {
            0, 0, 3,
            4, 6,
            8, 9,
            12, 12
        }),
        std::unique_ptr<IndT>(new IndT[12] {
            0, 8, 9,
            4, 4, 7,
            6, 9, 0,
            1, 2, 3,
        })
    );
}


TEST_CASE("construct_csr_matrix with int32 indices", "[construct_csr_matrix]") {
    LrmPefs pefs = make_test_lrm_pefs();

    CsrMatrix<int32_t> A = npeff::preprocessing::construct_csr_matrix<int32_t>(pefs);
    CsrMatrix<int32_t> expected = make_expected_A<int32_t>();

    REQUIRE(A.validate_indices());
    REQUIRE(A == expected);
}

TEST_CASE("construct_csr_matrix with int64 indices", "[construct_csr_matrix]") {
    LrmPefs pefs = make_test_lrm_pefs();

    CsrMatrix<int64_t> A = npeff::preprocessing::construct_csr_matrix<int64_t>(pefs);
    CsrMatrix<int64_t> expected = make_expected_A<int64_t>();

    REQUIRE(A == expected);
}
