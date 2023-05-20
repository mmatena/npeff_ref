#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>




namespace Cuda {



template <typename T>
struct UniqueDescr {
    T descr;

    UniqueDescr() : descr(nullptr) {}
    UniqueDescr(T descr) : descr(descr) {}

    // No copying.
    UniqueDescr(UniqueDescr<T>& o) = delete;
    UniqueDescr(const UniqueDescr<T>& o) = delete;
    UniqueDescr<T>& operator=(const UniqueDescr<T>&) = delete;


    UniqueDescr(UniqueDescr<T>&& o): descr(std::move(o.descr)) {
        o.descr = nullptr;
    }
    UniqueDescr<T>& operator=(UniqueDescr<T>&& o) {
       // if (this == o) return *this;
       // this->descr = std::move(o.descr);
       // o.descr = nullptr;
       std::swap(this->descr, o.descr);
       return *this;
    }

    ~UniqueDescr() {
        if (descr != nullptr) { DestroyDescr(); }
    }

protected:
    // Needs to be defined by template specialization.
    void DestroyDescr();

};



// Specializations.

template<>
void UniqueDescr<cusparseSpGEMMDescr_t>::DestroyDescr() {
    cusparseSpGEMM_destroyDescr(descr);
}

template<>
void UniqueDescr<cusparseSpMatDescr_t>::DestroyDescr() {
    cusparseDestroySpMat(descr);
}

template<>
void UniqueDescr<cusparseDnMatDescr_t>::DestroyDescr() {
    cusparseDestroyDnMat(descr);
}



}  // Cuda
