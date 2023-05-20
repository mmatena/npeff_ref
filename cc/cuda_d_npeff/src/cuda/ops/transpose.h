#pragma once

#include <vector>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h> // For thrust::device
#include <thrust/scan.h>

#include <misc/common.h>

#include <cuda/cuda_context.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>



namespace Cuda {
namespace Ops {

//////////////////////////////////////////////////////////////////////////////////////////////
// Custom transpose implementation.
//
// Needed as cusparse does not have anything that supports non-int32 indices.

// Scipy's implementation.
// https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L376


namespace internal {
namespace Custom_CsrTranspose_ {

template<class T>
struct TypeToAtomicAddType {
    typedef T type;
};
// Needed since atomic add appears to not support unsigned longs.
// Assuming signed integers are represented using two's-complement, the
// bitwise representations for the long and unsigned long should be
// identical for non-negative longs. Since the indices are always non-negative
// and we assume that overflow is not an issue, this should be fine.
template<>
struct TypeToAtomicAddType<int64_t> {
    typedef unsigned long long type;
};


// The array nnz_per_row should be zeroed out before calling this function.
template <typename IndT>
__global__ void ComputeNnzPerCol(const IndT* in_col_indices, IndT* nnz_per_col, const long nnz) {
    using AAT = typename TypeToAtomicAddType<IndT>::type;
    INDEX_STRIDE_1D(nnz, i) {
        atomicAdd((AAT*) nnz_per_col + in_col_indices[i], (AAT) 1);
    }
}



template <typename IndT>
struct DeviceHelper {
    long nnz;

    long in_n_rows;
    long in_n_cols;

    float* in_values;
    IndT* in_row_offsets;
    IndT* in_col_indices;

    float* out_values;
    IndT* out_row_offsets;
    IndT* out_col_indices;
};



template <typename IndT>
__global__ void ComputeColIndicesAndValues_InnerKernel(
    DeviceHelper<IndT> h, long row_idx, IndT start_idx, IndT span_size
) {
    using AAT = typename TypeToAtomicAddType<IndT>::type;
    INDEX_STRIDE_1D(span_size, i) {
        IndT j = start_idx + i;
        IndT col_idx = h.in_col_indices[j];

        IndT dest = atomicAdd((AAT*) h.out_row_offsets + col_idx, (AAT) 1);
        h.out_col_indices[dest] = (IndT) row_idx;
        h.out_values[dest] = h.in_values[j];
    }
}

template <typename IndT>
__global__ void ComputeColIndicesAndValues(DeviceHelper<IndT> h, long block_size) {
    INDEX_STRIDE_1D(h.in_n_rows, row_idx) {
        IndT start_idx = h.in_row_offsets[row_idx];
        IndT end_idx = h.in_row_offsets[row_idx + 1];
        IndT span_size = end_idx - start_idx;

        if (span_size == 0)  { continue; }

        long n_blocks = (span_size + block_size - 1) / block_size;

        if (n_blocks == 1) {
            block_size = span_size;
        }

        ComputeColIndicesAndValues_InnerKernel<IndT><<<n_blocks, block_size>>>(
            h, row_idx, start_idx, span_size
        );
    }
}



// template <typename IndT>
// __global__ void ComputeColIndicesAndValues2(DeviceHelper<IndT> h) {
//     INDEX_STRIDE_1D(h.nnz, i) {
//         // IndT start_idx = h.in_row_offsets[row_idx];
//         // IndT end_idx = h.in_row_offsets[row_idx + 1];
//         // IndT span_size = end_idx - start_idx;

//         // long n_blocks = (span_size + block_size - 1) / block_size;

//         // ComputeColIndicesAndValues_InnerKernel<IndT><<<n_blocks, block_size>>>(
//         //     h, row_idx, start_idx, span_size
//         // );
//     }
// }



}  // Custom_CsrTranspose_
}  // internal


template<typename IndT>
class Custom_CsrTranspose {
    using CsrMatrix = Device::CsrMatrix<IndT>;
    using DeviceHelper = internal::Custom_CsrTranspose_::DeviceHelper<IndT>;

protected:
    DeviceCudaContext& ctx;
    // If in.shape = [m, n], then out.shape = [n, m].
    CsrMatrix& in;
    CsrMatrix& out;

public:
    Custom_CsrTranspose(
        DeviceCudaContext& ctx,
        CsrMatrix& in,
        CsrMatrix& out
    ) :
        ctx(ctx), in(in), out(out)
    {
        // Validation.
        THROWSERT(in.n_rows == out.n_cols);
        THROWSERT(in.n_cols == out.n_rows);
        THROWSERT(in.nnz == out.nnz);
    }

    void Call_SingleUse() {
        ctx.SetDevice();

        ComputeOutRowOffsets();
        ctx.SynchronizeStream();
        // std::cout << "aaaaa\n";
        ComputeColIndicesAndValues();
        ctx.SynchronizeStream();
        // std::cout << "2222222\n";

        // TODO: See what scipy does. We essentially need to shift the
        // row offsets back by 1. We recompute from scratch here since it was
        // simplest to implement. We could also save a copy of the first time
        // we computed this.
        ComputeOutRowOffsets();
        // std::cout << "A\n";
    }

protected:

    void ComputeOutRowOffsets() {
        ctx.SetDevice();

        // For now, we will use this memory to compute the number of non-zero entries
        // per column of `in`.
        CUDA_CALL(cudaMemset(out.row_offsets, 0, sizeof(IndT) * (out.n_rows + 1)));
        ctx.SynchronizeStream();
        // std::cout << "bbbbbb\n";

        ComputeNnzPerCol();

        // NOTE: This might or might not be synchronous depending on the thrust
        // version and/or the functions being called.
        auto exec_policy = thrust::cuda::par.on(ctx.stream);

        // Cumsum to get the row_offsets of the transpose.
        thrust::device_ptr<IndT> out_row_offsets_thr(out.row_offsets);
        thrust::exclusive_scan(
            exec_policy,
            out_row_offsets_thr,
            out_row_offsets_thr + out.n_rows,
            out_row_offsets_thr
        );
        ctx.CopyToDeviceAsync(out.row_offsets + out.n_rows, &out.nnz, 1);
        ctx.SynchronizeStream();
    }

    void ComputeNnzPerCol() {
        ctx.SetDevice();
        long nnz = out.nnz;
        const long block_size = 512;
        long n_blocks = (nnz + block_size - 1) / block_size;
        internal::Custom_CsrTranspose_::ComputeNnzPerCol<IndT><<<n_blocks, block_size, 0, ctx.stream>>>(
            in.col_indices, out.row_offsets, nnz
        );
    }

    void ComputeColIndicesAndValues() {
        ctx.SetDevice();
        DeviceHelper h = MakeDeviceHelper();

        // std::cout << ((unsigned long) h.in_row_offsets) % 8 << "\n";
        // std::cout << h.in_row_offsets << ", " << h.in_col_indices << ", " << h.out_row_offsets << ", " << h.out_col_indices << "\n";

        const long block_size = 512;
        long n_blocks = (h.in_n_rows + block_size - 1) / block_size;
        internal::Custom_CsrTranspose_::ComputeColIndicesAndValues<IndT><<<n_blocks, block_size, 0, ctx.stream>>>(
            h, block_size
        );
    }

    DeviceHelper MakeDeviceHelper() {
        DeviceHelper h;

        h.nnz = in.nnz;
        h.in_n_rows = in.n_rows;
        h.in_n_cols = in.n_cols;

        h.in_values = in.values;
        h.in_row_offsets = in.row_offsets;
        h.in_col_indices = in.col_indices;

        h.out_values = out.values;
        h.out_row_offsets = out.row_offsets;
        h.out_col_indices = out.col_indices;

        return h;
    }

};



// Specialization for int32_t.
template<>
class Custom_CsrTranspose<int32_t> {
    using IndT = int32_t;
    using CsrMatrix = Device::CsrMatrix<IndT>;

protected:
    DeviceCudaContext& ctx;
    // If in.shape = [m, n], then out.shape = [n, m].
    CsrMatrix& in;
    CsrMatrix& out;

    const cusparseCsr2CscAlg_t alg;

public:
    Custom_CsrTranspose(
        DeviceCudaContext& ctx,
        CsrMatrix& in,
        CsrMatrix& out,
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG2
    ) :
        ctx(ctx), in(in), out(out), alg(alg)
    {
        // Validation.
        THROWSERT(in.n_rows == out.n_cols);
        THROWSERT(in.n_cols == out.n_rows);
        THROWSERT(in.nnz == out.nnz);
    }

    void Call_SingleUse() {
        ctx.SetDevice();
        void* buffer = SetUp();
        CallAsync_(buffer);
        ctx.SynchronizeStream();
        ctx.dfree(buffer);
    }

protected:

    void* SetUp() {
        ctx.SetDevice();

        size_t buffer_size;
        CUSPARSE_CALL(
            cusparseCsr2cscEx2_bufferSize(
                ctx.sparseHandle,
                in.n_rows, in.n_cols, in.nnz,
                in.values, in.row_offsets, in.col_indices,
                out.values, out.row_offsets, out.col_indices,
                CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                alg, &buffer_size
            )
        );

        ctx.SynchronizeStream();
        return ctx.dmalloc<void>(buffer_size);
    }

    void CallAsync_(void* buffer) {
        ctx.SetDevice();
        CUSPARSE_CALL(
            cusparseCsr2cscEx2(
                ctx.sparseHandle,
                in.n_rows, in.n_cols, in.nnz,
                in.values, in.row_offsets, in.col_indices,
                out.values, out.row_offsets, out.col_indices,
                CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                alg, buffer
            )
        );
    }


};


//////////////////////////////////////////////////////////////////////////////////////////////



// TODO: This is only supported for int32_t indices right now
// due to CuSparse only having support for int* indices.
template<typename IndT>
class CsrTranspose : public DeviceCudaContext::Freeable {
    using CsrMatrix = Device::CsrMatrix<IndT>;

protected:
    DeviceCudaContext& ctx;
    // If in.shape = [m, n], then out.shape = [n, m].
    CsrMatrix& in;
    CsrMatrix& out;

    const cusparseCsr2CscAlg_t alg;


    size_t buffer_size;
    void* buffer = nullptr;


public:
    CsrTranspose(
        DeviceCudaContext& ctx,
        CsrMatrix& in,
        CsrMatrix& out,
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG2
    ) :
        ctx(ctx), in(in), out(out), alg(alg)
    {
        // Validation.
        THROWSERT(in.n_rows == out.n_cols);
        THROWSERT(in.n_cols == out.n_rows);
        THROWSERT(in.nnz == out.nnz);
    }

    void SetUpAsync() {
        // NOTE: Not really async.
        ctx.SetDevice();

        CUSPARSE_CALL(
            cusparseCsr2cscEx2_bufferSize(
                ctx.sparseHandle,
                in.n_rows, in.n_cols, in.nnz,
                in.values, in.row_offsets, in.col_indices,
                out.values, out.row_offsets, out.col_indices,
                CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                alg, &buffer_size
            )
        );

        ctx.SynchronizeStream();
        buffer = ctx.dmalloc<void>(buffer_size);
    }

    void CallAsync() {
        ctx.SetDevice();
        CUSPARSE_CALL(
            cusparseCsr2cscEx2(
                ctx.sparseHandle,
                in.n_rows, in.n_cols, in.nnz,
                in.values, in.row_offsets, in.col_indices,
                out.values, out.row_offsets, out.col_indices,
                CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                alg, buffer
            )
        );
    }

    static void Perform_SingleUse(std::vector<CsrTranspose<IndT>>& ops) {
        for(auto& op : ops) {
            op.SetUpAsync();
        }
        for(auto& op : ops) {
            op.CallAsync();
        }
        for(auto& op : ops) {
            op.ctx.SynchronizeStream();
            op.FreeBuffer();
        }
    }

    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        return {buffer};
    }

protected:

    void FreeBuffer() {
        if(buffer != nullptr) {
            ctx.dfree(buffer);
            buffer = nullptr;
        }
    }


};



} // Ops
} // Cuda

