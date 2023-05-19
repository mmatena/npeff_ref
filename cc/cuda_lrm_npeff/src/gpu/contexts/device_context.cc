
#include "./device_context.h"

namespace npeff {
namespace gpu {


DeviceContext::~DeviceContext() {
    cudaSetDevice(device);

    for (auto ptr : dmalloced) {
        cudaFree(ptr);
    }

    // NOTE: I'm not sure if the null checks are needed, but
    // I'm including them anyways because they don't hurt.
    if(rand_gen != nullptr) { curandDestroyGenerator(rand_gen); }
    if(sparse_handle != nullptr) { cusparseDestroy(sparse_handle); }
    if(dense_handle != nullptr) { cublasDestroy(dense_handle); }
    if(cusolver_handle != nullptr) { cusolverDnDestroy(cusolver_handle); }
    if(stream != nullptr) { cudaStreamDestroy(stream); }
}


void DeviceContext::initialize() {
    set_device();

    CUDA_CALL(cudaStreamCreate(&stream));

    CUBLAS_CALL(cublasCreate(&dense_handle));
    CUBLAS_CALL(cublasSetPointerMode(dense_handle, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CALL(cublasSetStream(dense_handle, stream));

    CUSPARSE_CALL(cusparseCreate(&sparse_handle));
    CUSPARSE_CALL(cusparseSetPointerMode(sparse_handle, CUSPARSE_POINTER_MODE_DEVICE));
    CUSPARSE_CALL(cusparseSetStream(sparse_handle, stream));

    CUSOLVER_CALL(cusolverDnCreate(&cusolver_handle));
    CUSOLVER_CALL(cusolverDnSetStream(cusolver_handle, stream));

    CURAND_CALL(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_gen, rand_gen_seed));
    CURAND_CALL(curandSetStream(rand_gen, stream));

    float f01[] = {0.0f, 1.0f};
    dev_0f = dmalloc<float>(2);
    dev_1f = dev_0f + 1;
    copy_to_device_async(dev_0f, f01, 2);
    synchronize_stream();
}


void DeviceContext::synchronize_stream() {
    set_device();
    CUDA_CALL(cudaStreamSynchronize(stream));
}


void DeviceContext::set_device() {
    CUDA_CALL(cudaSetDevice(device));
}


// Allow us allocate memory in bytes by using the void typename.
template<>
void* DeviceContext::dmalloc<void>(size_t n_elements, bool managed) {
    return (void *) dmalloc<char>(n_elements, managed);
}



}  // gpu
}  // npeff
