#pragma once
// Options controlling the factorization.

#include <cstdint>
#include <cmath>


namespace npeff {
namespace factorizations {
namespace lvrm_factorization {


struct FactorizationConfig {
    int64_t rank;

    // Total across all devices.
    int64_t n_cols_total;

    int64_t n_examples;

    int64_t n_iters_G_only;
    int64_t n_iters_joint;

    int64_t log_loss_frequency;

    int64_t rand_gen_seed;

    float learning_rate_G_joint;
    float learning_rate_G_G_only;
    float mu_eps;

    double tr_xx;

    
    double compute_inv_g_initialization_scale_factor() {
        // TODO: More theoretically principled computation of the scaling
        // initialization factor.  We would want each reconstructed PEF at
        // initialization to have roughly unit Frobenious norm.
        return std::sqrt((double) (rank * n_cols_total) / 2.0);
    }

};



}  // lvrm_factorization
}  // factorizations
}  // npeff
