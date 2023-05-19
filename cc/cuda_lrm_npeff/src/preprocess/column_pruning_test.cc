#include <catch2/catch_test_macros.hpp>

#include <preprocess/column_pruning.h>

#include <containers/conversions.h>
#include <util/array_util.h>

using npeff::CsrMatrix;
using npeff::DenseMatrix;


CsrMatrix<int32_t> make_test_matrix() {
    float data[] = {
        1.0, 0.0, 0.0, -4.0, 0.0,
        0.0, 2.0, 0.0, -0.5, 0.0,
        0.0, 4.0, 0.0, -0.5, 0.0,
    };
    return npeff::containers::csr_from_row_major_array<int32_t>(
        data, 3, 5
    );
}


TEST_CASE("prune_columns_in_place prunes correctly", "[column_pruning]") {

    SECTION("min_nnz_per_col = 1") {
        CsrMatrix<int32_t> matrix = make_test_matrix();

        std::unique_ptr<DenseMatrix<int32_t>> new_to_old_col_indices;
        npeff::preprocessing::prune_columns_in_place(&matrix, &new_to_old_col_indices, 1);

        CsrMatrix<int32_t> expected_matrix = npeff::containers::csr_from_row_major_array<int32_t>(new float[9] {
            1.0, 0.0, -4.0,
            0.0, 2.0, -0.5,
            0.0, 4.0, -0.5,
        }, 3, 3);
        DenseMatrix<int32_t> expected_new_to_old_col_indices(
            1, 3, std::unique_ptr<int32_t>(new int32_t[3] {0, 1, 3})
        );
        REQUIRE(matrix == expected_matrix);
        REQUIRE(*new_to_old_col_indices == expected_new_to_old_col_indices);
    }

    SECTION("min_nnz_per_col = 2") {
        CsrMatrix<int32_t> matrix = make_test_matrix();

        std::unique_ptr<DenseMatrix<int32_t>> new_to_old_col_indices;
        npeff::preprocessing::prune_columns_in_place(&matrix, &new_to_old_col_indices, 2);

        CsrMatrix<int32_t> expected_matrix = npeff::containers::csr_from_row_major_array<int32_t>(new float[6] {
            0.0, -4.0,
            2.0, -0.5,
            4.0, -0.5,
        }, 3, 2);
        DenseMatrix<int32_t> expected_new_to_old_col_indices(
            1, 2, std::unique_ptr<int32_t>(new int32_t[2] {1, 3})
        );
        REQUIRE(matrix == expected_matrix);
        REQUIRE(*new_to_old_col_indices == expected_new_to_old_col_indices);
    }

    SECTION("min_nnz_per_col = 3") {
        CsrMatrix<int32_t> matrix = make_test_matrix();

        std::unique_ptr<DenseMatrix<int32_t>> new_to_old_col_indices;
        npeff::preprocessing::prune_columns_in_place(&matrix, &new_to_old_col_indices, 3);

        CsrMatrix<int32_t> expected_matrix = npeff::containers::csr_from_row_major_array<int32_t>(new float[3] {
            -4.0,
            -0.5,
            -0.5,
        }, 3, 1);
        DenseMatrix<int32_t> expected_new_to_old_col_indices(
            1, 1, std::unique_ptr<int32_t>(new int32_t[1] {3})
        );
        REQUIRE(matrix == expected_matrix);
        REQUIRE(*new_to_old_col_indices == expected_new_to_old_col_indices);
    }
}


TEST_CASE("prune_columns_in_place handles empty rows in output correctly", "[column_pruning]") {
    float data[] = {
        1.0, 0.0, 0.0, -4.0, 0.0,
        0.0, 0.0, 1.0,  0.0, 0.5,
        0.0, 4.0, 0.0, -0.5, 0.0,
    };
    CsrMatrix<int32_t> matrix = npeff::containers::csr_from_row_major_array<int32_t>(
        data, 3, 5
    );

    std::unique_ptr<DenseMatrix<int32_t>> new_to_old_col_indices;
    npeff::preprocessing::prune_columns_in_place(&matrix, &new_to_old_col_indices, 2);

    CsrMatrix<int32_t> expected_matrix = npeff::containers::csr_from_row_major_array<int32_t>(new float[9] {
        -4.0,
         0.0,
        -0.5,
    }, 3, 1);
    DenseMatrix<int32_t> expected_new_to_old_col_indices(
        1, 1, std::unique_ptr<int32_t>(new int32_t[1] {3})
    );
    REQUIRE(matrix == expected_matrix);
    REQUIRE(*new_to_old_col_indices == expected_new_to_old_col_indices);
}


// TEST_CASE("prune_columns_in_place handle empty rows in input correctly", "[column_pruning]") {
//     // TODO
// }

