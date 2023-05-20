#pragma once
/**
 * Various general-purpose macros that I am using.
 */
#include <iostream>


#define THROW std::cout << "Error at line " << __LINE__ << " in file " << __FILE__ << ".\n"; throw

#define THROWSERT(x) if(!(x)) {std::cout << "Exception at line " << __LINE__ << " in file " << __FILE__ << "\n"; throw;}

#define THROW_MSG(msg) \
    std::cout << msg << "\n"; \
    std::cout << "Error at line " << __LINE__ << " in file " << __FILE__ << ".\n"; \
    throw


#define INDEX_STRIDE_1D(n, i) \
    long index = blockIdx.x * blockDim.x + threadIdx.x; \
    long stride = blockDim.x * gridDim.x; \
    for (long i = index; i < n; i += stride)

