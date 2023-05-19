#pragma once
// Options controlling the factorization.

#include <cstdint>
#include <cmath>


namespace npeff {
namespace factorizations {
namespace util {


// int64_t compute_n_examples(std::vector<CsrMatrixPtr>& column_partitions) {
//     int64_t n_rows = column_partitions[0]->n_rows;
//     THROW_IF_FALSE((n_rows % config.n_classes) == 0);
//     return n_rows / config.n_classes;
// }

// int64_t compute_n_cols_total(std::vector<CsrMatrixPtr>& column_partitions) {
//     int64_t n_cols = 0;
//     for(auto& mat : column_partitions) { n_cols += mat->n_cols; }
//     return n_cols;
// }

// std::vector<int64_t> compute_n_cols_per_partition(std::vector<CsrMatrixPtr>& column_partitions) {
//     std::vector<int64_t> ret;
//     for(auto& mat : column_partitions) {
//         ret.push_back(mat->n_cols);
//     }
//     return ret;
// }


}  // util
}  // factorizations
}  // npeff
