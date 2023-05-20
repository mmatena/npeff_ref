#pragma once

#include <cuda/cuda_context.h>

#include <cuda/host/dense_matrix.h>
#include <cuda/host/sparse_matrix.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>


/////////////////////////////////////////////////////////////////////////////////////////

template<>
void DeviceCudaContext::CopyToDeviceAsync<Cuda::Device::DenseMatrix, Cuda::Host::DenseMatrix>(
    Cuda::Device::DenseMatrix& dev_mat,
    Cuda::Host::DenseMatrix& host_mat
) {
    THROWSERT(dev_mat.n_entries == host_mat.n_entries);
    SetDevice();
    CopyToDeviceAsync<float>(dev_mat.data, host_mat.data, host_mat.n_entries);
}


template<>
void DeviceCudaContext::CopyToHostAsync<Cuda::Host::DenseMatrix, Cuda::Device::DenseMatrix>(
    Cuda::Host::DenseMatrix& host_mat,
    Cuda::Device::DenseMatrix& dev_mat
) {
    THROWSERT(dev_mat.n_entries == host_mat.n_entries);
    SetDevice();
    CopyToHostAsync<float>(host_mat.data, dev_mat.data, host_mat.n_entries);
}


/////////////////////////////////////////////////////////////////////////////////////////

namespace internal_ {

template<typename A, typename B>
void AssertCsrMatricesCompatible(A& a, B& b) {
    THROWSERT(a.n_rows == b.n_rows);
    THROWSERT(a.n_cols == b.n_cols);
    THROWSERT(a.nnz == b.nnz);
}

}  // internal_


#define DO_CSR_COPY_(T, a, b) \
    CopyToDeviceAsync<float>(a.values, b.values, a.nnz); \
    CopyToDeviceAsync<T>(a.row_offsets, b.row_offsets, a.n_rows + 1); \
    CopyToDeviceAsync<T>(a.col_indices, b.col_indices, a.nnz)


#define SPECIALIZE_CSR_TO_DEVICE_ASYNC_(IndT) \
    template<> \
    void DeviceCudaContext::CopyToDeviceAsync<Cuda::Device::CsrMatrix<IndT>, Cuda::Host::CsrMatrix<IndT>>( \
        Cuda::Device::CsrMatrix<IndT>& dev_mat, \
        Cuda::Host::CsrMatrix<IndT>& host_mat \
    ) { \
        internal_::AssertCsrMatricesCompatible(dev_mat, host_mat); \
        SetDevice(); \
        DO_CSR_COPY_(IndT, dev_mat, host_mat); \
    }


SPECIALIZE_CSR_TO_DEVICE_ASYNC_(int32_t);
SPECIALIZE_CSR_TO_DEVICE_ASYNC_(int64_t);



#undef DO_CSR_COPY_
#undef WRITE_CSR_TO_DEVICE_ASYNC_
