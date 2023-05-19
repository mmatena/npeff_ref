#pragma once
#include <stddef.h>

namespace npeff {
namespace util {


template<typename S, typename T>
void convert_numeric_arrays(S* a, T* b, size_t n) {
    for (size_t i=0; i<n; i++) {
        a[i] = static_cast<T>(b[i]);
    }
}

// Intended for use mostly in test code.
template<typename T>
bool arrays_are_equal(T* a, T* b, size_t n) {
    if((a == nullptr) != (b == nullptr)) { return false; }
    // NOTE: Not sure what the best to handle the case when
    // both are null but n != 0.
    if(a == nullptr && b == nullptr) { return true; }
    for (size_t i=0; i<n; i++) {
        if(a[i] != b[i]) { return false; }
    }
    return true;
}



}  // util
}  // npeff
