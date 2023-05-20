#pragma once
/**
 * Utilities related to dealing with CUDA types.
 */

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <misc/common.h>


#define CT_DEFINE_CONVERTER_(name, input, output) \
    template<input T> struct name {static output value;}

#define CT_ADD_PAIR_(name, input, output, outputT) \
    template<> outputT name<input>::value {output}



namespace Cuda {

CT_DEFINE_CONVERTER_(ToCuSparseIndexType, typename, cusparseIndexType_t);
CT_ADD_PAIR_(ToCuSparseIndexType, int32_t, CUSPARSE_INDEX_32I, cusparseIndexType_t);
CT_ADD_PAIR_(ToCuSparseIndexType, int64_t, CUSPARSE_INDEX_64I, cusparseIndexType_t);


CT_DEFINE_CONVERTER_(ToCuSparseOrder, MatrixOrder, cusparseOrder_t);
CT_ADD_PAIR_(ToCuSparseOrder, MatrixOrder::COL_MAJOR, CUSPARSE_ORDER_COL, cusparseOrder_t);
CT_ADD_PAIR_(ToCuSparseOrder, MatrixOrder::ROW_MAJOR, CUSPARSE_ORDER_ROW, cusparseOrder_t);


// template<typename T>
// struct ToCuSparseIndexType {
//     static cusparseIndexType_t value;
// };
// template<>
// cusparseIndexType_t ToCuSparseIndexType<int32_t>::value {CUSPARSE_INDEX_32I};
// template<>
// cusparseIndexType_t ToCuSparseIndexType<int64_t>::value {CUSPARSE_INDEX_64I};


// template<MatrixOrder T>
// struct ToCuSparseOrder {
//     static cusparseOrder_t value;
// };
// template<>
// cusparseOrder_t ToCuSparseOrder<MatrixOrder::COL_MAJOR>::value {CUSPARSE_ORDER_COL};
// template<>
// cusparseOrder_t ToCuSparseOrder<MatrixOrder::ROW_MAJOR>::value {CUSPARSE_ORDER_ROW};







}

#undef CT_DEFINE_CONVERTER_
#undef CT_ADD_PAIR_