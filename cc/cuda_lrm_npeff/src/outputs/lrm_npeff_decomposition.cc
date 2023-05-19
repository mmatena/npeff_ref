#include "./lrm_npeff_decomposition.h"

namespace npeff {
namespace outputs {

template<>
void DenseLrmNpeffDecomposition::set_new_to_old_col_indices(
    std::unique_ptr<npeff::DenseMatrix<int32_t>> new_to_old_col_indices) {
    this->new_to_old_col_indices = std::move(new_to_old_col_indices);
}


}  // outputs
}  // npeff
