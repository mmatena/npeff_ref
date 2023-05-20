#pragma once
/**
 * Interfaces related to matrices.
 */

namespace Nmf {
namespace Interface {


class AbstractMatrix {
public:
    long n_rows;
    long n_cols;

    virtual ~AbstractMatrix() {}
};


// class TransferableMatrix : protected AbstractMatrix {
// public:
//     virtual 
    

// };


}
}
