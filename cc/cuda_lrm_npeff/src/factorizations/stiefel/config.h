#pragma once
// Options controlling the factorization.

#include <cstdint>
#include <cmath>


namespace npeff {
namespace factorizations {
namespace stiefel {


struct FactorizationConfig {
    int64_t rank;
    int64_t n_classes;

    // Total across all devices.
    int64_t n_cols_total;

    int64_t n_iters_G_only;
    int64_t n_iters_joint;

    int64_t log_loss_frequency;

    int64_t rand_gen_seed;

    float learning_rate_G_joint;
    float learning_rate_G_G_only;
    float mu_eps;

    double tr_xx;

    double compute_w_initialization_scale_factor() {
        return 2.0 / (double) rank;
    }
};



}  // stiefel
}  // factorizations
}  // npeff
