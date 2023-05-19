#include <catch2/catch_test_macros.hpp>
#include <containers/dense_matrix.h>

using npeff::DenseMatrix;


template<typename T>
void write_range_to_data(DenseMatrix<T>& matrix, T start = 0) {
    for(int64_t i=0; i<matrix.n_entries; i++) {
        matrix.data.get()[i] = (T) i + start;
    }
}


TEST_CASE("DenseMatrix works with floats.", "[dense_matrix]") {
    DenseMatrix<float> matrix(2, 4);
    REQUIRE(matrix.n_entries == 8);
    REQUIRE(matrix.size_bytes == 32);

    write_range_to_data(matrix);

    SECTION("get_col_ptr works properly") {
        REQUIRE(*matrix.get_col_ptr(0) == 0.0f);
        REQUIRE(*matrix.get_col_ptr(1) == 2.0f);
        REQUIRE(*matrix.get_col_ptr(2) == 4.0f);
    }
}

TEST_CASE("DenseMatrix works with int32s.", "[dense_matrix]") {
    DenseMatrix<int32_t> matrix(2, 4);
    REQUIRE(matrix.n_entries == 8);
    REQUIRE(matrix.size_bytes == 32);

    write_range_to_data(matrix);

    SECTION("get_col_ptr works properly") {
        REQUIRE(*matrix.get_col_ptr(0) == 0);
        REQUIRE(*matrix.get_col_ptr(1) == 2);
        REQUIRE(*matrix.get_col_ptr(2) == 4);
    }
}

TEST_CASE("DenseMatrix works with int64s.", "[dense_matrix]") {
    DenseMatrix<int64_t> matrix(2, 4);
    REQUIRE(matrix.n_entries == 8);
    REQUIRE(matrix.size_bytes == 64);

    write_range_to_data(matrix);

    SECTION("get_col_ptr works properly") {
        REQUIRE(*matrix.get_col_ptr(0) == 0);
        REQUIRE(*matrix.get_col_ptr(1) == 2);
        REQUIRE(*matrix.get_col_ptr(2) == 4);
    }
}


TEST_CASE("DenseMatrix equality operator works", "[dense_matrix]") {
    DenseMatrix<float> matrix(2, 4);
    DenseMatrix<float> matrix_same(2, 4);
    DenseMatrix<float> matrix_different_data(2, 4);
    DenseMatrix<float> matrix_same_data_different_shape(4, 2);

    write_range_to_data(matrix);
    write_range_to_data(matrix_same);
    write_range_to_data(matrix_different_data, 32.f);
    write_range_to_data(matrix_same_data_different_shape);

    REQUIRE(matrix == matrix);
    REQUIRE(matrix == matrix_same);

    REQUIRE_FALSE(matrix == matrix_different_data);
    REQUIRE_FALSE(matrix == matrix_same_data_different_shape);
}