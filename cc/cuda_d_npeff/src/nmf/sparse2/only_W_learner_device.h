#pragma once
/**
 * Generic/AbstractBase class for the device contexts of the OnlyWLearner.
 */

#include <vector>

#include <cuda/cuda_context.h>
#include "./only_W_learner_params.h"


namespace Nmf {
namespace OnlyW {


// // TODO: Put this in its own file.
// template<
//     typename IndT
// >
// class MultiplicativeUpdater {
// protected:

// public:
//     MultiplicativeUpdater() {}

// };



template<
    typename IndT,
    typename MatrixH,
    typename MatrixA = ElCsrMatrix<IndT>,
    typename Params = ::Nmf::OnlyW::LearnerParams<IndT, MatrixH, MatrixA>,
    typename DcParams = ::Nmf::OnlyW::LearnerDeviceContextParams<IndT, MatrixH, MatrixA, Params>
>
class LearnerDeviceContext {
protected:
    DcParams params;
    DeviceCudaContext cudaCtx;

public:
    LearnerDeviceContext(DcParams params, HostCudaContext& hostCudaCtx) :
        params(params),
        cudaCtx(hostCudaCtx.deviceContexts[params.device])
    {}

    virtual void computeAH() = 0;
    virtual void computeHH() = 0;


};



}
}
