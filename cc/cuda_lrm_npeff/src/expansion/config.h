#pragma once

// Options controlling the expansion.

#include <cstdint>
#include <cmath>

namespace npeff {
namespace expansion {


struct ExpansionConfig {
    int64_t rank_expansion;
    int64_t rank_frozen;

    int64_t n_classes;

    // Total across all devices.
    int64_t n_cols_total;

    int64_t n_iters_G_only;
    int64_t n_iters_joint_expansion_only;
    int64_t n_iters_joint;

    int64_t log_loss_frequency;

    int64_t rand_gen_seed;

    float learning_rate_G_joint;
    float learning_rate_G_G_only;
    float mu_eps;

    double tr_xx;

    int64_t total_rank() const {
        return rank_frozen + rank_expansion;
    }


    double compute_inv_g_initialization_scale_factor() {
        // TODO: More theoretically principled computation of the scaling
        // initialization factor.  We would want each reconstructed PEF at
        // initialization to have roughly unit Frobenious norm.
        // 
        // TODO: Double check that this makes sense for initializing only the
        // expanded component's G.
        return std::sqrt((double) (total_rank() * n_cols_total) / 2.0);
    }

};


}  // expansion
}  // npeff
