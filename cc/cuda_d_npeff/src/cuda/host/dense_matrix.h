#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <misc/macros.h>
#include <cuda/cuda_statuses.h>
#include <cuda/cuda_context.h>


// Forward declarations.
class MeMatrix;


namespace Cuda {
namespace Host {


class DenseMatrix {
public:
    long n_rows;
    long n_cols;

    long n_entries;
    size_t size_bytes;

    // This is a pointer on the host memory. The memory IS owned
    // by this class.
    float* data;

    DenseMatrix(long n_rows, long n_cols) :
        n_rows(n_rows),
        n_cols(n_cols),
        n_entries(n_rows * n_cols),
        size_bytes(n_entries * sizeof(float)),
        data(new float[n_entries])
    {}
    ~DenseMatrix() {
        delete[] data;
    }

    float* GetData() { return data; }

protected:

    DenseMatrix(long n_rows, long n_cols, float* data) :
        n_rows(n_rows),
        n_cols(n_cols),
        n_entries(n_rows * n_cols),
        size_bytes(n_entries * sizeof(float)),
        data(data)
    {}

    friend class ::DeviceCudaContext;
    friend class ::MeMatrix;

};


// class DenseMatrixView {
// };



}
}

