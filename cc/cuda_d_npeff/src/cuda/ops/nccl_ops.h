#pragma once

#include <vector>

#include "nccl.h"
#include <cuda_runtime.h>

#include <misc/common.h>

#include <cuda/cuda_context.h>

#include <cuda/device/dense_matrix.h>
#include <cuda/device/sparse_matrix.h>


namespace Cuda {
namespace Ops {


template<typename DenseT>
class DenseAllReduce {

protected:
    HostCudaContext& ctx;
    std::vector<DenseT>& operands;
    const ncclRedOp_t op;

public:
    DenseAllReduce(
        HostCudaContext& ctx,
        std::vector<DenseT>& operands,
        ncclRedOp_t op = ncclSum
    ) :
        ctx(ctx), operands(operands), op(op)
    {
        THROWSERT(operands.size() == ctx.n_devices);
    }

    void CallAsync() {
        NCCL_CALL(ncclGroupStart());
        for(int i=0;i<ctx.n_devices;i++) {
            auto& ds = ctx.device_contexts[i];
            auto& mat = operands[i];
            NCCL_CALL(
                ncclAllReduce(
                    mat.data,
                    mat.data,
                    mat.n_entries,
                    ncclFloat,
                    op,
                    ds.comm,
                    ds.stream)
            );
        }
        NCCL_CALL(ncclGroupEnd());
    }

};


} // Ops
} // Cuda
