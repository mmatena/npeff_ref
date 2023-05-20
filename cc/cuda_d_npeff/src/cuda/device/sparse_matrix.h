#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <misc/macros.h>
#include <cuda/cuda_statuses.h>
#include <cuda/cuda_context.h>
#include <cuda/cuda_types.h>
#include <cuda/descr.h>

#include <cuda/device/dense_matrix.h>


// Forward declarations.
namespace Cuda {
namespace Ops {
    template<typename IndT>
    class SparseSparseMatMul_SingleUse;

    template<typename IndT>
    class Partitioned_SpSpMatmul_ToDense_SingleUse;

    template<typename IndT>
    class CsrTranspose;

    template<typename IndT>
    class SplitByRows_InPlace;

    template<typename IndT_>
    class ReIndexWithInt32;
}
}



namespace Cuda {
namespace Device {


template<typename IndT>
class CsrMatrix : public DeviceCudaContext::Freeable {
    using DenseMatrix = ::Cuda::Device::DenseMatrix;

public:
    long n_rows;
    long n_cols;

    long nnz;

    // These are pointers on the device memory. The memory is NOT owned
    // by this class.
    float* values = nullptr;
    IndT* row_offsets = nullptr;
    IndT* col_indices = nullptr;

    UniqueDescr<cusparseSpMatDescr_t> descr;

    CsrMatrix() : n_rows(0), n_cols(0), nnz(0) {}

    CsrMatrix(long n_rows, long n_cols, long nnz) :
        n_rows(n_rows),
        n_cols(n_cols),
        nnz(nnz)
    {}

    CsrMatrix(CsrMatrix<IndT>&& o) = default;
    CsrMatrix<IndT>& operator=(CsrMatrix<IndT>&& o) = default;

    void AllocMemory(DeviceCudaContext& ctx) {
        // We should not allocate memory if the instance already has memory
        // assocaited to it.
        THROWSERT(values == nullptr);
        THROWSERT(row_offsets == nullptr);
        THROWSERT(col_indices == nullptr);

        // To be clear, the memory is owned by the context, NOT this class.
        ctx.SetDevice();
        values = ctx.dmalloc<float>(nnz);
        row_offsets = ctx.dmalloc<IndT>(n_rows + 1);
        col_indices = ctx.dmalloc<IndT>(nnz);

        CreateDescr();
    }

    // Returns the sequence of floats represented by values as a DenseMatrix
    // with a single column. This does not incorporate any information about
    // the sparse structure of this matrix.
    // 
    // The returned DenseMatrix is a just a view on the data.
    DenseMatrix ViewValuesAsVector() {
        return DenseMatrix(nnz, 1, values);
    }

    bool CanUseInt32Indices() {
        if (std::is_same<IndT, int32_t>::value) {
            return true;
        }
        return nnz < INT32_MAX;
    }

    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        return {values, row_offsets, col_indices};
    }

protected:
    void CreateDescr() {
        // Memory must be allocated before calling this.
        THROWSERT(row_offsets != nullptr);
        if(nnz > 0) {
            THROWSERT(values != nullptr);
            THROWSERT(col_indices != nullptr);
        }

        cusparseIndexType_t ind_type = ToCuSparseIndexType<IndT>::value;
        CUSPARSE_CALL(
            cusparseCreateCsr(&descr.descr, n_rows, n_cols, nnz,
                              row_offsets, col_indices, values,
                              ind_type, ind_type,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        );
    }

    friend class ::DeviceCudaContext;

    template<typename IndT_>
    friend class ::Cuda::Ops::SparseSparseMatMul_SingleUse;

    template<typename IndT_>
    friend class ::Cuda::Ops::Partitioned_SpSpMatmul_ToDense_SingleUse;

    template<typename IndT_>
    friend class ::Cuda::Ops::CsrTranspose;

    template<typename IndT_>
    friend class ::Cuda::Ops::SplitByRows_InPlace;

    template<typename IndT_>
    friend class ::Cuda::Ops::ReIndexWithInt32;

};



}
}
