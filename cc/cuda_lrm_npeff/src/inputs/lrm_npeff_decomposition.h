#pragma once
// Stuff for reading from LRM-NPEFF decompositions.

#include <cstdint>
#include <memory>
#include <string>
#include <util/macros.h>
#include <containers/dense_matrix.h>

namespace npeff {
namespace inputs {


class DenseLrmNpeffDecompositionFromFile {
    using FloatMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;
    using IndMatrixPtr = std::unique_ptr<npeff::DenseMatrix<int32_t>>;

    const std::string filepath;

public:
    // These fields will be read from the file in the constructor.
    int64_t n_parameters;
    int64_t n_classes;

    DenseLrmNpeffDecompositionFromFile(std::string filepath) : filepath(filepath) {
        read_scalar_fields_from_file();
    }

    IndMatrixPtr read_new_to_old_col_indices();

    FloatMatrixPtr read_W();
    FloatMatrixPtr read_G();

    int64_t read_n_examples();

protected:
    // Reads scalar fields from file. This should be fast and not take up much memory.
    void read_scalar_fields_from_file();
};


}  // inputs
}  // npeff
