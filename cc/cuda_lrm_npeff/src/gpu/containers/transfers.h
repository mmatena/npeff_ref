#pragma once
// QoL improvement code for moving data to and from the GPU.

#include <util/macros.h>
#include <gpu/contexts/device_context.h>

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include "./dense_matrix.h"
#include "./sparse_matrix.h"


namespace npeff {
namespace gpu {


template<>
void DeviceContext::copy_to_device_async<gpu::DenseMatrix, npeff::DenseMatrix<float>>(
    gpu::DenseMatrix& dev_mat,
    npeff::DenseMatrix<float>& host_mat
) {
    THROW_IF_FALSE(dev_mat.n_rows == host_mat.n_rows);
    THROW_IF_FALSE(dev_mat.n_cols == host_mat.n_cols);
    set_device();
    copy_to_device_async<float>((float*) dev_mat.data, host_mat.data.get(), host_mat.n_entries);
}


///////////////////////////////////////////////////////////
// Need to do this hacky stuff as we cannot do a partial specialization for this.

namespace internal_ {

template<typename A, typename B>
void assert_csr_matrices_compatible(A& a, B& b) {
    THROW_IF_FALSE(a.n_rows == b.n_rows);
    THROW_IF_FALSE(a.n_cols == b.n_cols);
    THROW_IF_FALSE(a.nnz == b.nnz);
}

}  // internal_

#define DO_CSR_COPY_(T, a, b) \
    copy_to_device_async<float>((float*) a.values, b.values.get(), a.nnz); \
    copy_to_device_async<T>((T*) a.row_offsets, b.row_offsets.get(), a.n_rows + 1); \
    copy_to_device_async<T>((T*) a.col_indices, b.col_indices.get(), a.nnz)


#define SPECIALIZE_CSR_TO_DEVICE_ASYNC_(IndT) \
    template<> \
    void DeviceContext::copy_to_device_async<gpu::CsrMatrix<IndT>, npeff::CsrMatrix<IndT>>( \
        gpu::CsrMatrix<IndT>& dev_mat, \
        npeff::CsrMatrix<IndT>& host_mat \
    ) { \
        internal_::assert_csr_matrices_compatible(dev_mat, host_mat); \
        set_device(); \
        DO_CSR_COPY_(IndT, dev_mat, host_mat); \
    }

SPECIALIZE_CSR_TO_DEVICE_ASYNC_(int32_t);
SPECIALIZE_CSR_TO_DEVICE_ASYNC_(int64_t);

#undef DO_CSR_COPY_
#undef WRITE_CSR_TO_DEVICE_ASYNC_

///////////////////////////////////////////////////////////


void copy_to_host_into_submatrix_async(
    DeviceContext& ctx, 
    gpu::DenseMatrix& dev_mat,
    npeff::DenseMatrix<float>& host_mat,
    int64_t row_offset,
    int64_t col_offset
) {
    ctx.set_device();

    // TODO: First verify that the dev_mat fits as submatrix.
    THROW_IF_FALSE(row_offset + dev_mat.n_rows <= host_mat.n_rows);
    THROW_IF_FALSE(col_offset + dev_mat.n_cols <= host_mat.n_cols);

    if(dev_mat.n_rows == host_mat.n_rows) {
        // If the number of rows matches between the two matrices, we only need to do
        // a single copy_to_host_async call.
        THROW_IF_FALSE(row_offset == 0);
        float* h_start = host_mat.data.get() + col_offset * host_mat.n_rows + row_offset;
        ctx.copy_to_host_async<float>(h_start, (float*) dev_mat.data, dev_mat.n_entries);

    } else {
        // Otherwise copy the columns one by one.
        for(int64_t i=0; i<dev_mat.n_cols; i++) {
            float* h_start = host_mat.data.get() + (col_offset + i) * host_mat.n_rows + row_offset;
            float* d_start = (float*) dev_mat.data + i * dev_mat.n_rows;
            ctx.copy_to_host_async<float>(h_start, d_start, dev_mat.n_rows);
        }
    }
}


}  // gpu
}  // npeff
