#include <catch2/catch_test_macros.hpp>
#include <util/array_util.h>

using namespace npeff::util;

TEST_CASE("convert_numeric_arrays", "[array_util]") {

    SECTION("works for int32 to int64") {
        int64_t a[] = {0, 0, 0, 0};
        int32_t b[] = {1, 7, 33, -2};

        convert_numeric_arrays(a, b, 4);

        REQUIRE(a[0] == 1);
        REQUIRE(a[1] == 7);
        REQUIRE(a[2] == 33);
        REQUIRE(a[3] == -2);
    }

}


TEST_CASE("arrays_are_equal", "[array_util]") {
    int64_t a[] = {0, 0, 0, 0};
    int64_t b[] = {1, 7, 33, -2};
    int64_t c[] = {1, 7, 33, -2};
    size_t n = 4;


    SECTION("detects non-equal arrays") {
        REQUIRE_FALSE(arrays_are_equal<int64_t>(a, b, n));
    }
    SECTION("detects equal arrays") {
        REQUIRE(arrays_are_equal<int64_t>(b, c, n));
    }
    SECTION("handles nullptrs correctly") {
        REQUIRE_FALSE(arrays_are_equal<int64_t>(nullptr, a, n));
        REQUIRE_FALSE(arrays_are_equal<int64_t>(a, nullptr, n));
    }
}
