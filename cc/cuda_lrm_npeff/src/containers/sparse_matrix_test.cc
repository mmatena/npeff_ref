#include <catch2/catch_test_macros.hpp>
#include <containers/sparse_matrix.h>

using npeff::CsrMatrix;


TEST_CASE("CsrMatrix's == operator", "[sparse_matrix]") {
    CsrMatrix<int32_t> matrix(
        2, 4, 6,
        std::unique_ptr<float>(new float[6] {1.5, -2.3, 1.0, 4.0, -4.0, 1.0}),
        std::unique_ptr<int32_t>(new int32_t[3] {0, 2, 6}),
        std::unique_ptr<int32_t>(new int32_t[6] {0, 3, 0, 1, 2, 3})
    );
    CsrMatrix<int32_t> different_matrix(
        2, 4, 6,
        std::unique_ptr<float>(new float[6] {-9999.0, -2.3, 1.0, 4.0, -4.0, 1.0}),
        std::unique_ptr<int32_t>(new int32_t[3] {0, 2, 6}),
        std::unique_ptr<int32_t>(new int32_t[6] {0, 3, 0, 1, 2, 3})
    );

    REQUIRE(matrix == matrix);
    REQUIRE_FALSE(matrix == different_matrix);
}


TEST_CASE("CsrMatrix's validate_indices", "[sparse_matrix]") {

    SECTION("valid matrix") {
        CsrMatrix<int32_t> matrix(
            2, 4, 6,
            std::unique_ptr<float>(new float[6] {1.5, -2.3, 1.0, 4.0, -4.0, 1.0}),
            std::unique_ptr<int32_t>(new int32_t[3] {0, 2, 6}),
            std::unique_ptr<int32_t>(new int32_t[6] {0, 3, 0, 1, 2, 3})
        );
        REQUIRE(matrix.validate_indices());
    }

    SECTION("invalid row_offsets") {
        SECTION("first entry is not 0") {
            CsrMatrix<int32_t> matrix(
                2, 4, 6,
                std::unique_ptr<float>(new float[6] {1.5, -2.3, 1.0, 4.0, -4.0, 1.0}),
                std::unique_ptr<int32_t>(new int32_t[3] {1, 2, 6}),
                std::unique_ptr<int32_t>(new int32_t[6] {0, 3, 0, 1, 2, 3})
            );
            REQUIRE_FALSE(matrix.validate_indices());
        }
        SECTION("last entry is not nnz") {
            CsrMatrix<int32_t> matrix(
                2, 4, 6,
                std::unique_ptr<float>(new float[6] {1.5, -2.3, 1.0, 4.0, -4.0, 1.0}),
                std::unique_ptr<int32_t>(new int32_t[3] {0, 2, 5}),
                std::unique_ptr<int32_t>(new int32_t[6] {0, 3, 0, 1, 2, 3})
            );
            REQUIRE_FALSE(matrix.validate_indices());
        }
        SECTION("not non-decreasing") {
            CsrMatrix<int32_t> matrix(
                4, 4, 6,
                std::unique_ptr<float>(new float[6] {1.5, -2.3, 1.0, 4.0, -4.0, 1.0}),
                std::unique_ptr<int32_t>(new int32_t[5] {0, 2, 4, 3, 6}),
                std::unique_ptr<int32_t>(new int32_t[6] {0, 3, 0, 1, 2, 3})
            );
            REQUIRE_FALSE(matrix.validate_indices());
        }
    }

    SECTION("invalid col_indices") {
        SECTION("entry is too low") {
            CsrMatrix<int32_t> matrix(
                2, 4, 6,
                std::unique_ptr<float>(new float[6] {1.5, -2.3, 1.0, 4.0, -4.0, 1.0}),
                std::unique_ptr<int32_t>(new int32_t[3] {0, 2, 6}),
                std::unique_ptr<int32_t>(new int32_t[6] {0, -3, 0, 1, 2, 3})
            );
            REQUIRE_FALSE(matrix.validate_indices());
        }
        SECTION("entry is too high") {
            CsrMatrix<int32_t> matrix(
                2, 4, 6,
                std::unique_ptr<float>(new float[6] {1.5, -2.3, 1.0, 4.0, -4.0, 1.0}),
                std::unique_ptr<int32_t>(new int32_t[3] {0, 2, 6}),
                std::unique_ptr<int32_t>(new int32_t[6] {0, 4, 0, 1, 2, 3})
            );
            REQUIRE_FALSE(matrix.validate_indices());
        }
    }
}
