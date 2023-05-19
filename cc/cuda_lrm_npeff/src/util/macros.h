#pragma once

#include <iostream>


#define THROW std::cout << "Error at line " << __LINE__ << " in file " << __FILE__ << ".\n"; throw

#define THROW_IF_FALSE(x) if(!(x)) {std::cout << "Exception at line " << __LINE__ << " in file " << __FILE__ << "\n"; throw;}

#define THROW_MSG(msg) \
    std::cout << msg << "\n"; \
    std::cout << "Error at line " << __LINE__ << " in file " << __FILE__ << ".\n"; \
    throw
