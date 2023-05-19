#pragma once
// Checking numerical values.

#include "stdio.h"

#include <assert.h>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <gpu/contexts/device_context.h>
#include <gpu/containers/dense_matrix.h>


namespace npeff {
namespace gpu {
namespace ops {
namespace debugging {

// // The + 1 accounts for the zero termination.
// const size_t ASSERT_FINITE_MAX_MSG_LENGTH = 80 + 1;


__global__
void AssertFinite_Kernel(int64_t n, float* data, int64_t marker) {
    INDEX_STRIDE_1D(n, i) {
        
        if (isnan(data[i]) || isinf(data[i])) {
            printf("\e[1;93mNaN or inf found at position %ld. MARKER: %ld\033[m\n ", i, marker);
            assert(false);
        }
    }
}

class AssertFinite {
    DeviceContext& ctx;
    DenseMatrix& mat;

    // Use an integer so that we can tag specific assert locations in code. Ideally,
    // we could use a string, but I was having trouble finding an easy way to pass
    // it to the kernel.
    int64_t marker;

    // Number of elements in the data.
    const int64_t n_elements;

    // TODO: Figure out how to set this.
    const int64_t block_size = 256;

public:
    AssertFinite(DeviceContext& ctx, DenseMatrix& mat, int64_t marker = -1) :
        ctx(ctx), mat(mat), marker(marker), n_elements(mat.n_entries)
    {}

    void call_async() {
        ctx.set_device();
        long n_blocks = (n_elements + block_size - 1) / block_size;

        AssertFinite_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
            n_elements, (float*) mat.data, marker
        );
    }

    void call_sync() {
        call_async();
        ctx.synchronize_stream();
    }

};


// __global__
// void AssertFinite_Kernel(int64_t n, float* data, char msg[ASSERT_FINITE_MAX_MSG_LENGTH]) {
//     INDEX_STRIDE_1D(n, i) {
//         // if (isnan(data[i]) || isinf(data[i])) {
//         //     printf("\e[1;93mNaN or inf found at position %ld. %s\033[m\n", i, msg);
//         //     assert(false);
//         // }
//         if ((i % 1000) == 0) {
//             printf("\e[1;93mNaN or inf found at position %ld. %s\033[m\n ", i, msg);
//         }
//     }
// }

// class AssertFinite {
//     DeviceContext& ctx;
//     DenseMatrix& mat;
//     // std::string msg;

//     char msg[ASSERT_FINITE_MAX_MSG_LENGTH] = {0};

//     // Number of elements in the data.
//     const int64_t n_elements;

//     // TODO: Figure out how to set this.
//     const int64_t block_size = 256;

// public:
//     AssertFinite(DeviceContext& ctx, DenseMatrix& mat, std::string msg = "") :
//         ctx(ctx), mat(mat), n_elements(mat.n_entries)
//     {
//         if (msg.size() + 1 > ASSERT_FINITE_MAX_MSG_LENGTH) {
//             std::cout << "AssertFinite message is too long: " << msg << "\n";
//             THROW;
//         }
//         strcpy(this->msg, msg.c_str());
//     }

//     void call_async() {
//         ctx.set_device();
//         long n_blocks = (n_elements + block_size - 1) / block_size;

//         AssertFinite_Kernel<<<n_blocks, block_size, 0, ctx.stream>>>(
//             n_elements, (float*) mat.data, msg
//         );
//     }

//     void call_sync() {
//         call_async();
//         ctx.synchronize_stream();
//     }

// };


}  // debugging
}  // ops
}  // gpu
}  // npeff


