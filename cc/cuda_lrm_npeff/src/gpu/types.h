#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cusparse.h>

namespace npeff {
namespace gpu {


#define CT_DEFINE_CONVERTER_(name, input, output) \
    template<input T> struct name {static output value;}

#define CT_ADD_PAIR_(name, input, output, outputT) \
    template<> outputT name<input>::value {output}


CT_DEFINE_CONVERTER_(ToCuSparseIndexType, typename, cusparseIndexType_t);
CT_ADD_PAIR_(ToCuSparseIndexType, int32_t, CUSPARSE_INDEX_32I, cusparseIndexType_t);
CT_ADD_PAIR_(ToCuSparseIndexType, int64_t, CUSPARSE_INDEX_64I, cusparseIndexType_t);


// CT_DEFINE_CONVERTER_(ToCuSparseOrder, MatrixOrder, cusparseOrder_t);
// CT_ADD_PAIR_(ToCuSparseOrder, MatrixOrder::COL_MAJOR, CUSPARSE_ORDER_COL, cusparseOrder_t);
// CT_ADD_PAIR_(ToCuSparseOrder, MatrixOrder::ROW_MAJOR, CUSPARSE_ORDER_ROW, cusparseOrder_t);


}  // gpu
}  // npeff
