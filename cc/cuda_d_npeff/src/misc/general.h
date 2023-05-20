/**
 * Miscellaneous general purpose stuff.
 */
#pragma once
#include <stddef.h>



namespace Misc {


template<typename S, typename T>
void ConvertNumericArrays(S* a, T* b, size_t n) {
    for (size_t i=0; i<n; i++) {
        a[i] = static_cast<T>(b[i]);
    }
}



} // Misc
