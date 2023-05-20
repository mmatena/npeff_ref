#pragma once

#include <vector>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <misc/common.h>

#include <cuda/cuda_context.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>


namespace Cuda {
namespace Ops {


template<typename IndT, MatrixOrder order>
class CsrToDense : public DeviceCudaContext::Freeable {
    using CsrMatrix = Device::CsrMatrix<IndT>;
    using DnMatrix = Device::DnMatrix<order>;

protected:
    DeviceCudaContext& ctx;
    CsrMatrix& in;
    DnMatrix& out;

    size_t buffer_size;
    void* buffer = nullptr;

    const cusparseSparseToDenseAlg_t alg = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;

public:
    CsrToDense(
        DeviceCudaContext& ctx,
        CsrMatrix& in,
        DnMatrix& out
    ) :
        ctx(ctx), in(in), out(out)
    {
        // Validation.
        // Only validate the total size as sometimes I might use the ordering
        // of the output matrix and transposing to do stuff I need to do.
        THROWSERT(in.n_rows * in.n_cols == out.n_rows * out.n_cols);
    }


    void SetUpAsync() {
        // NOTE: Not really async.
        ctx.SetDevice();

        CUSPARSE_CALL(
            cusparseSparseToDense_bufferSize(
                ctx.sparseHandle,
                in.descr.descr, out.descr.descr,
                alg, &buffer_size
            )
        );

        ctx.SynchronizeStream();
        buffer = ctx.dmalloc<void>(buffer_size);
    }

    void CallAsync() {
        ctx.SetDevice();
        CUSPARSE_CALL(
            cusparseSparseToDense(
                ctx.sparseHandle,
                in.descr.descr, out.descr.descr,
                alg, buffer
            )
        );
    }

    static void Perform_SingleUse(std::vector<CsrToDense<IndT, order>> ops) {
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


//////////////////////////////////////////////////////////////////////////////////////////////


namespace internal {

template <typename D, typename S>
__global__ void CopyAndCast_Kernel(D* dst, const S* src, size_t n) {
    INDEX_STRIDE_1D(n, i) {
        dst[i] = (D) src[i];
    }
}

} // internal



template<typename IndT>
class ReIndexWithInt32 {
    using CsrMatrix = Device::CsrMatrix<IndT>;
    using CsrMatrix32 = Device::CsrMatrix<int32_t>;

protected:
    DeviceCudaContext& ctx;
    CsrMatrix& mat;

    bool free_memory = true;


public:
    CsrMatrix32 out;
    
    ReIndexWithInt32(DeviceCudaContext& ctx, CsrMatrix& mat, bool free_memory = true) :
        ctx(ctx), mat(mat), out(mat.n_rows, mat.n_cols, mat.nnz), free_memory(free_memory)
    {
        if(!mat.CanUseInt32Indices()) {
            std::cout << "Sparse matrix is too large to use int32 indices.\n";
            THROW;
        }
    }

    void Call() {
        out.values = mat.values;
        out.row_offsets = ctx.dmalloc<int32_t>(out.n_rows + 1);
        out.col_indices = ctx.dmalloc<int32_t>(out.nnz);

        CallKernel(out.row_offsets, mat.row_offsets, out.n_rows + 1);
        CallKernel(out.col_indices, mat.col_indices, out.nnz);
        ctx.SynchronizeStream();

        out.CreateDescr();

        // "Clear" the original matrix.
        mat.values = nullptr;
        if (free_memory) {
            ctx.dfree(mat.row_offsets);
            ctx.dfree(mat.col_indices);
        }
        mat.row_offsets = mat.col_indices = nullptr;

    }

protected:
    void CallKernel(int32_t* dst, IndT* src, size_t n) {
        long block_size = 512;
        long n_blocks = (n + block_size - 1) / block_size;
        internal::CopyAndCast_Kernel<int32_t, IndT><<<n_blocks, block_size, 0, ctx.stream>>>(
            dst, src, n);
    }



};

// Specialize to essentially be a no-op when the indices are already int32_t.
template<>
void ReIndexWithInt32<int32_t>::Call() {
    out = std::move(mat);
};


} // Ops
} // Cuda

