#pragma once
// Options controlling the factorization.

#include <cstdint>
#include <cmath>


namespace npeff {
namespace factorization {


struct OrthogonalRegularizationConfig {
    float regularization_strength = 0.0f;
};


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

    OrthogonalRegularizationConfig ortho_reg_config;

    bool has_orthogonal_regularization() {
        return ortho_reg_config.regularization_strength > 0.0f;
    }

    double compute_inv_g_initialization_scale_factor() {
        // TODO: More theoretically principled computation of the scaling
        // initialization factor.  We would want each reconstructed PEF at
        // initialization to have roughly unit Frobenious norm.
        return std::sqrt((double) (rank * n_cols_total) / 2.0);
    }

    // Note that this function should not be called if we are not using orthogonal
    // regularization.
    double compute_orthogonal_regularization_target_scale() {
        // Intended for each g_i's squared norm at initialization to be roughly
        // equal to this.
        // 
        // Each g_i is initialized from a IID normal with zero mean and
        // standard derivation of g_initialization_scale_factor. If the standard
        // deviation a was 1, then the expected value of the Chi-squared distribution
        // could be used to compute the expected value of g_i^Tg_i, which would be
        // n_cols_total. Since we multiplied the samples of a standard normal
        // distribution by the constant g_initialization_scale_factor, linearity
        // of expectation gives us our result.
        double inv_g_factor = compute_inv_g_initialization_scale_factor();
        return (double) n_cols_total / (inv_g_factor * inv_g_factor);
    }
};


}  // factorization
}  // npeff
