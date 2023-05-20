#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <misc/macros.h>
#include <misc/common.h>
#include <cuda/cuda_statuses.h>
#include <cuda/cuda_context.h>
#include <cuda/cuda_types.h>
#include <cuda/descr.h>


namespace Cuda {
namespace Device {


// Assumed to be column major.
class DenseMatrix: public DeviceCudaContext::Freeable {
public:
    long n_rows;
    long n_cols;

    long n_entries;
    size_t size_bytes;

    // This is a pointer on the device memory. The memory is NOT owned
    // by this class.
    float* data = nullptr;
    
    DenseMatrix(long n_rows, long n_cols) :
        DenseMatrix(n_rows, n_cols, nullptr)
    {}

    DenseMatrix(long n_rows, long n_cols, float* data) :
        n_rows(n_rows), n_cols(n_cols), data(data),
        n_entries(n_rows * n_cols),
        size_bytes(n_entries * sizeof(float))
    {}


    void AllocMemory(DeviceCudaContext& ctx) {
        // We should not allocate memory if the instance already has memory
        // assocaited to it.
        THROWSERT(data == nullptr);

        // To be clear, the memory is owned by the context, NOT this class.
        data = ctx.dmalloc<float>(n_entries);
        AfterAlloc();
    }

    float const * GetData() { return data; }


    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        return {data};
    }


protected:

    // Meant to be overwritten by subclasses if needed.
    virtual void AfterAlloc() {}


    friend class ::DeviceCudaContext;
};


// TODO: Maybe template the following based on col/row major order.

template<MatrixOrder order>
class DnMatrix : public DenseMatrix {
public:
    UniqueDescr<cusparseDnMatDescr_t> descr;

    DnMatrix(long n_rows, long n_cols) : DnMatrix(n_rows, n_cols, nullptr) {}

    // NOTE: You probably need to be careful to ensure that cudaSetDevice is
    // called before with the correct device if data is not null. Otherwise
    // the descr might not point to the correct place.
    DnMatrix(long n_rows, long n_cols, float* data) :
        DenseMatrix(n_rows, n_cols, data)
    {
        if(data != nullptr) { CreateDescr(); }
    }

    // Returns a copy.
    DenseMatrix AsDenseMatrix() {
        return DenseMatrix(n_rows, n_cols, data);
    }


    // DeviceCudaContext::Freeable
    virtual std::vector<void*> GetDeviceAllocs() {
        return {data};
    }

protected:
    void CreateDescr() {
        // Memory must be allocated before calling this.
        THROWSERT(data != nullptr);
        CUSPARSE_CALL(
            cusparseCreateDnMat(&descr.descr, n_rows, n_cols, n_rows, data,
                                CUDA_R_32F, ToCuSparseOrder<order>::value)
        );
    }

    virtual void AfterAlloc() { CreateDescr(); }
};


}
}
