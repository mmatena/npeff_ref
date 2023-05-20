#pragma once
/**
 * Generic/AbstractBase class for learning the W given a fixed H on
 * multiple GPUs.
 */
#include <vector>

#include <cuda/cuda_context.h>
#include "./only_W_learner_params.h"
#include "./only_W_learner_device.h"


namespace Nmf {
namespace OnlyW {


template<
    typename IndT,
    typename MatrixH,
    typename MatrixA = ElCsrMatrix<IndT>,
    typename Params = ::Nmf::OnlyW::LearnerParams<IndT, MatrixH, MatrixA>,
    typename DcParams = ::Nmf::OnlyW::LearnerDeviceContextParams<IndT, MatrixH, MatrixA, Params>,
    typename LearnerDeviceContext = ::Nmf::OnlyW::LearnerDeviceContext<IndT, MatrixH, MatrixA, Params, DcParams>
>
class Learner {
protected:
    int n_devices;

    Params params;
    HostCudaContext cudaCtx;

    std::vector<LearnerDeviceContext> deviceCtxs;

public:
    Learner(Params p) : 
        params(p),
        n_devices(p.n_devices()),
        cudaCtx(HostCudaContext(p.n_devices(), p.seed))
    {
        for(int i=0;i<n_devices;i++) {
            deviceCtxs.emplace_back(DcParams(params, i), cudaCtx);
        }
    }

};



}
}
