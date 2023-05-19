
#include <iostream>
#include <string>

#include <gflags/gflags.h>

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include <inputs/lvrm_pefs.h>
#include <outputs/lrm_npeff_decomposition.h>

#include <gpu/gpu_info.h>

#include <factorizations/lvrm_factorization/config.h>
#include <factorizations/lvrm_factorization/io_util.h>


//////////////////////////////////////////////////////////////////////////////
// Flag definitions

DEFINE_string(pef_filepath, "", "Filepath of the HDF5 file containing the per-example Fishers.");
DEFINE_string(output_filepath, "", "Filepath where the HDF5 file containing the output will be written to.");

DEFINE_int64(n_components, -1, "The number of elements in the learned dictionary.");
DEFINE_int64(n_examples, -1, "The number of examples to use from pefs file. Set to -1 to use all.");

DEFINE_int64(n_iters_G_only, -1, "The number of iterations to learn an initial G given a fixed W.");
DEFINE_int64(n_iters_joint, -1, "The maximum number of iterations to run the factorization algorithm for, learning both W and G.");

DEFINE_double(learning_rate_G, 1e-3, "The learning rate to be used for the G-update steps.");
DEFINE_double(learning_rate_G_G_only, -1.0, "The learning rate to be used for the G-update steps in G only training. If not set, defaults to --learning_rate_G.");

DEFINE_double(mu_eps, 1e-9, "Epsilon for the multiplicative update on W.");

DEFINE_int64(min_nonzero_per_col, 1, "Prune columns with fewer than this many non-zero entries.");

DEFINE_int64(rand_gen_seed, 32497, "Seed to use for random number generation.");

DEFINE_int64(log_loss_frequency, 10, "Compute and log the loss every this number of steps.");

DEFINE_int64(n_preprocess_cpu_threads, 1, "The number of threads to use for preprocessing on the CPU.");


//////////////////////////////////////////////////////////////////////////////
using namespace npeff::factorizations::lvrm_factorization;
//////////////////////////////////////////////////////////////////////////////

// Reads the config from flags. Note that not all fields can be
// set directly from flags, so those will need to be written later.
FactorizationConfig read_partial_config_from_flags() {
    FactorizationConfig config;

    config.rank = FLAGS_n_components;

    config.n_iters_G_only = FLAGS_n_iters_G_only;
    config.n_iters_joint = FLAGS_n_iters_joint;

    config.rand_gen_seed = FLAGS_rand_gen_seed;
    config.log_loss_frequency = FLAGS_log_loss_frequency;
    config.mu_eps = FLAGS_mu_eps;

    config.learning_rate_G_joint = FLAGS_learning_rate_G;
    config.learning_rate_G_G_only = FLAGS_learning_rate_G_G_only;
    if (config.learning_rate_G_G_only <= 0.0) {
        config.learning_rate_G_G_only = config.learning_rate_G_joint;
    }

    // Config fields not set:
    //   - config.tr_xx
    //   - config.n_examples
    //   - config.n_cols_total

    // Validations.
    if(config.rank <= 0) {
        THROW_MSG("Must set the --n_components flag to a positive integer.");
    }
    if(config.n_iters_G_only < 0) {
        THROW_MSG("Must set the --n_iters_G_only flag to a non-negative integer.");
    }
    if(config.n_iters_joint <= 0) {
        THROW_MSG("Must set the --n_iters_joint flag to a positive integer.");
    }

    return config;
}

AdditionalRunContextConfig read_additional_run_context_config_from_flags() {
    AdditionalRunContextConfig ret;
    ret.output_filepath = FLAGS_output_filepath;
    ret.min_nonzero_per_col = FLAGS_min_nonzero_per_col;
    ret.n_preprocess_cpu_threads = FLAGS_n_preprocess_cpu_threads;
    return ret;
}



int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_output_filepath.empty()) {
        THROW_MSG("Please provide a valid --output_filepath flag value.");
    }

    FactorizationConfig partial_config = read_partial_config_from_flags();
    AdditionalRunContextConfig additional_config = read_additional_run_context_config_from_flags();

    auto pefs = npeff::inputs::LvrmPefs::load(FLAGS_pef_filepath, FLAGS_n_examples);
    std::cout << "LVRM-PEFS loaded from disk.\n";

    if(npeff::preprocessing::can_csr_matrix_use_int32_indices(pefs)) {
        std::cout << "Creating inputs using int32 indices.\n";
        auto ctx = create_run_context<int32_t>(pefs, partial_config, additional_config);
        ctx.run();
    } else {
        std::cout << "Creating inputs using int64 indices.\n";
        auto ctx = create_run_context<int64_t>(pefs, partial_config, additional_config);
        if(ctx.can_reindex_with_int32()) {
            std::cout << "Re-indexing with int32 indices.\n";
            auto ctx32 = ctx.reindex_with_int32();
            ctx32.run();
        } else {
            ctx.run();
        }
    }
}
