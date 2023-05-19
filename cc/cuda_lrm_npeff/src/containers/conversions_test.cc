#include <catch2/catch_test_macros.hpp>
#include <containers/conversions.h>

using npeff::CsrMatrix;
using namespace npeff::containers;


TEST_CASE("csr_from_col_major_array", "[conversions]") {
    // Need to transpose as the DenseMatrix is column major.
    float transposed_data[] = {
        1.0, 0.0, -4.0,
        0.0, 2.0, -0.5
    };

    SECTION("default threshold of zero") {
        CsrMatrix<int32_t> actual = csr_from_col_major_array<int32_t>(
            transposed_data, 3, 2
        );
        CsrMatrix<int32_t> expected(
            3, 2, 4,
            std::unique_ptr<float>(new float[4] {1.0, 2.0, -4.0, -0.5}),
            std::unique_ptr<int32_t>(new int32_t[4] {0, 1, 2, 4}),
            std::unique_ptr<int32_t>(new int32_t[4] {0, 1, 0, 1})
        );
        REQUIRE(actual == expected);
    }

}

TEST_CASE("csr_from_row_major_array", "[conversions]") {
    float data[] = {
        1.0, 0.0, -4.0,
        0.0, 2.0, -0.5
    };

    SECTION("default threshold of zero") {
        CsrMatrix<int32_t> actual = csr_from_row_major_array<int32_t>(
            data, 2, 3
        );
        CsrMatrix<int32_t> expected(
            2, 3, 4,
            std::unique_ptr<float>(new float[4] {1.0, -4.0, 2.0, -0.5}),
            std::unique_ptr<int32_t>(new int32_t[3] {0, 2, 4}),
            std::unique_ptr<int32_t>(new int32_t[4] {0, 2, 1, 2})
        );
        REQUIRE(actual == expected);
    }

    SECTION("handles empty rows") {
        CsrMatrix<int32_t> actual = csr_from_row_major_array<int32_t>(
            data, 2, 3, 3.0
        );
        CsrMatrix<int32_t> expected(
            2, 3, 1,
            std::unique_ptr<float>(new float[1] {-4.0}),
            std::unique_ptr<int32_t>(new int32_t[3] {0, 1, 1}),
            std::unique_ptr<int32_t>(new int32_t[1] {2})
        );
        REQUIRE(actual == expected);
    }

}
