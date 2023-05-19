// Learns additional components given an existing decomposition. The
// Gs from the existing decomposition are frozen and some additional
// new Gs are introduced to be learned from scratch.

#include <iostream>
#include <string>

#include <gflags/gflags.h>

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include <inputs/lrm_pefs.h>
#include <inputs/lrm_npeff_decomposition.h>

#include <preprocess/construct_csr_matrix.h>
#include <preprocess/partition_by_columns.h>
#include <preprocess/column_pruning.h>
#include <preprocess/pef_normalization.h>
#include <preprocess/sort_by_col_indices.h>

#include <gpu/gpu_info.h>

// #include <factorization/compute_tr_xx.h>
#include <expansion/config.h>
#include <expansion/manager.h>
#include <expansion/io_util.h>



//////////////////////////////////////////////////////////////////////////////
// Flag definitions

DEFINE_string(pef_filepath, "", "Filepath of the HDF5 file containing the per-example Fishers.");
DEFINE_string(decomposition_filepath, "", "Filepath of the HDF5 file containing the decomposition whose G and maybe W will be used.");
DEFINE_string(output_filepath, "", "Filepath where the HDF5 file containing the output will be written to.");

DEFINE_bool(use_W_from_decomposition, false, "Whether to initialize a subset of the W from the decomposition. Only use if the decomposition is a result of fitting the PEFs to the G.");

DEFINE_int64(n_examples, -1, "The number of examples to use from pefs file. Set to -1 to use all.");
DEFINE_int64(n_additional_components, -1, "The number of components to learn in addition to those read from the decomposition.");

DEFINE_int64(n_iters_G_only, -1, "The number of iterations to learn an initial G given a fixed W.");
DEFINE_int64(n_iters_joint_expansion_only, 0, "The number of iterations run the factorization algorithm for only the expanded components.");
DEFINE_int64(n_iters_joint, -1, "The maximum number of iterations to run the factorization algorithm for, learning both W and G.");

DEFINE_double(learning_rate_G, 1e-3, "The learning rate to be used for the G-update steps.");
DEFINE_double(learning_rate_G_G_only, -1.0, "The learning rate to be used for the G-update steps in G only training. If not set, defaults to --learning_rate_G.");

DEFINE_double(mu_eps, 1e-9, "Epsilon for the multiplicative update on W.");

DEFINE_int64(rand_gen_seed, 32497, "Seed to use for random number generation.");

DEFINE_int64(log_loss_frequency, 10, "Compute and log the loss every this number of steps.");

DEFINE_int64(n_preprocess_cpu_threads, 1, "The number of threads to use for preprocessing on the CPU.");

// Maybe temporary flags here.
DEFINE_bool(pefs_col_offsets_non_cumulative, false, "Needed to account for bug in my first generation of the LRM-PEFS.");

//////////////////////////////////////////////////////////////////////////////
using namespace npeff::expansion;
//////////////////////////////////////////////////////////////////////////////

// Reads the config from flags. Note that not all fields can be
// set directly from flags, so those will need to be written later.
ExpansionConfig read_config_from_flags() {
    ExpansionConfig config;

    config.rank_expansion = FLAGS_n_additional_components;
    // config.rank_frozen
    // config.n_classes
    // config.n_cols_total
    config.n_iters_G_only = FLAGS_n_iters_G_only;
    config.n_iters_joint_expansion_only = FLAGS_n_iters_joint_expansion_only;
    config.n_iters_joint = FLAGS_n_iters_joint;
    config.log_loss_frequency = FLAGS_log_loss_frequency;
    config.rand_gen_seed = FLAGS_rand_gen_seed;
    config.mu_eps = FLAGS_mu_eps;
    // config.tr_xx

    config.learning_rate_G_joint = FLAGS_learning_rate_G;
    config.learning_rate_G_G_only = FLAGS_learning_rate_G_G_only;
    if (config.learning_rate_G_G_only <= 0.0) {
        config.learning_rate_G_G_only = config.learning_rate_G_joint;
    }

    // Validations.
    if(config.rank_expansion <= 0) {
        THROW_MSG("Must set the --n_additional_components flag to a positive integer.");
    }
    if(config.n_iters_G_only < 0) {
        THROW_MSG("Must set the --n_iters_G_only flag to a non-negative integer.");
    }
    if(config.n_iters_joint_expansion_only < 0) {
        THROW_MSG("Must set the --n_iters_joint_expansion_only flag to a non-negative integer.");
    }
    if(config.n_iters_joint < 0) {
        THROW_MSG("Must set the --n_iters_joint flag to a non-negative integer.");
    }

    return config;
}


//////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_output_filepath.empty()) {
        THROW_MSG("Please provide a valid --output_filepath flag value.");
    }
    if(FLAGS_decomposition_filepath.empty()) {
        THROW_MSG("Please provide a valid --decomposition_filepath flag value.");
    }

    // Note that this config will not have all fields set since some required
    // information not directly found in the flags.
    ExpansionConfig config = read_config_from_flags();

    auto lrm_pefs = npeff::inputs::LrmPefs::load(FLAGS_pef_filepath, FLAGS_n_examples);
    std::cout << "LRM-PEFS loaded from disk.\n";

    // This will only read the scalars attributes from the decomposition, so it
    // should be pretty fast.
    npeff::inputs::DenseLrmNpeffDecompositionFromFile frozen_decomposition(FLAGS_decomposition_filepath);

    if (!npeff::expansion::are_inputs_compatible(lrm_pefs, frozen_decomposition, FLAGS_use_W_from_decomposition)) {
        THROW_MSG("The decomposition is not compatible with the PEFs.");
    }

    CreateRunContextOptions run_ctx_opts;
    run_ctx_opts.n_partitions = npeff::gpu::get_device_count();
    run_ctx_opts.use_W_from_decomposition = FLAGS_use_W_from_decomposition;
    run_ctx_opts.pefs_col_offsets_non_cumulative = FLAGS_pefs_col_offsets_non_cumulative;
    run_ctx_opts.n_preprocess_cpu_threads = FLAGS_n_preprocess_cpu_threads;





    std::cout << "Creating inputs using int64 indices.\n";
    RunContext<int64_t> run_ctx = create_run_context<int64_t>(
        config, lrm_pefs, frozen_decomposition, run_ctx_opts);
    run_ctx.run(FLAGS_output_filepath);


    // if(npeff::preprocessing::can_csr_matrix_use_int32_indices(lrm_pefs)) {
    //     std::cout << "Creating inputs using int32 indices.\n";
    //     RunContext<int32_t> run_ctx = create_run_context<int32_t>(
    //         config, lrm_pefs, frozen_decomposition, run_ctx_opts);
    //     run_ctx.run(FLAGS_output_filepath);

    // } else {
    //     std::cout << "Creating inputs using int64 indices.\n";
    //     RunContext<int64_t> run_ctx = create_run_context<int64_t>(
    //         config, lrm_pefs, frozen_decomposition, run_ctx_opts);

    //     if(run_ctx.can_reindex_with_int32()) {
    //         std::cout << "Re-indexing with int32 indices.\n";
    //         RunContext<int32_t> run_ctx32 = run_ctx.reindex_with_int32();
    //         run_ctx32.run(FLAGS_output_filepath);

    //     } else{
    //         run_ctx.run(FLAGS_output_filepath);
    //     }
    // }
}
