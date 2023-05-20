#pragma once
/** Frobenious inner product and Frobenious norm. */
#include <math.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>

#include <misc/common.h>

#include <cuda/cuda_context.h>
#include <cuda/descr.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>


namespace Cuda {
namespace Ops {


class FrobeniousInnerProduct : public DeviceCudaContext::Freeable {
    using DenseMatrix = Device::DenseMatrix;

protected:
    // Needed as some of the blas functionality uses only int32s. We
    // use 2^30 to give us freedom to make the last chunk up to (but
    // not including) twice as long to handle differences in size.
    static const long CHUNK_SIZE_ = 1 << 30;

    DeviceCudaContext& ctx;
    DenseMatrix& left;
    DenseMatrix& right;

    long n_entries;

    int n_chunks;
    float* device_result_chunks = nullptr;
    float* host_result_chunks = nullptr;

    float host_result = 0.0f;

public:
    FrobeniousInnerProduct(
        DeviceCudaContext& ctx,
        DenseMatrix& left,
        DenseMatrix& right
    ) : 
        ctx(ctx), left(left), right(right),
        n_entries(left.n_entries),
        n_chunks(ComputeChunksNeeded(n_entries))
    {
        THROWSERT(left.n_rows == right.n_rows);
        THROWSERT(left.n_cols == right.n_cols);
    }

    ~FrobeniousInnerProduct() {
        delete[] host_result_chunks;
    }

    void SetUpAsync() {
        host_result_chunks = new float[n_chunks];

        // NOTE: The dmalloc will have a synchronizing memory allocation.
        ctx.SetDevice();
        device_result_chunks = ctx.dmalloc<float>(n_chunks);
    }

    void CallAsync() {
        ctx.SetDevice();
        for (int i=0;i<n_chunks;i++) {
            CallAsync_Chunk(i);
        }
        ctx.CopyToHostAsync(host_result_chunks, device_result_chunks, n_chunks);
        CUDA_CALL(cudaLaunchHostFunc(ctx.stream, (cudaHostFn_t) SumHostResults_Cb, (void*) this));
    }

    // NOTE: Does no synchronization.
    float Result() {
        return host_result;
    }
    
    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        return {device_result_chunks};
    }

protected:

    static void CUDART_CB SumHostResults_Cb(void *data) {
        FrobeniousInnerProduct* fip = (FrobeniousInnerProduct*) data;
        fip->SumHostResults();
    }

    void SumHostResults() {
        host_result = 0.0f;
        for (int i=0;i<n_chunks;i++) {host_result += host_result_chunks[i]; }
    }

    void CallAsync_Chunk(int chunk) {
        ctx.SetDevice();

        long chunk_size = GetSizeOfChunk(chunk);
        long chunk_offset = chunk * CHUNK_SIZE_;

        CUBLAS_CALL(cublasSdot(
            ctx.denseHandle, chunk_size,
            left.data + chunk_offset, 1,
            right.data + chunk_offset, 1,
            device_result_chunks + chunk
        ));
    }

    long GetSizeOfChunk(int chunk) {
        long chunk_size;
        if (chunk == n_chunks - 1) {
            chunk_size = n_entries - chunk * CHUNK_SIZE_;
        } else {
            chunk_size = CHUNK_SIZE_;
        }

        // Sanity check.
        if ((long) ((int) chunk_size) != chunk_size) {
            std::cout << "Integer overflow\n";
            THROW;
        }

        return chunk_size;
    }

    int ComputeChunksNeeded(long n_entries) {
        // We can include an additional up to LOSS_CHUNK_SIZE - 1 entries in
        // the last chunk, so this works.
        return std::max(1, (int) (n_entries / CHUNK_SIZE_));
    }

};




// class FrobeniousSquaredNorm : public FrobeniousInnerProduct {
//     using DenseMatrix = Device::DenseMatrix;

//     template<typename IndT>
//     using CsrMatrix = Device::CsrMatrix<IndT>;

// public:
//     FrobeniousSquaredNorm(DeviceCudaContext& ctx, DenseMatrix& mat) : 
//         FrobeniousInnerProduct(ctx, mat, mat)
//     {}
// };


} // Ops
} // Cuda


