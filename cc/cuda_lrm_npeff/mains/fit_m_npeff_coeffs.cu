// Fits the coefficients matrix W given the G from a decomposition
// and some LRM-PEFs from a file.
// 
// Currently runs only on a single GPU.
// 
// Assumes that the entire G can fit into the host memory.
#include <iostream>
#include <memory>
#include <string>

#include <gflags/gflags.h>

#include <outputs/lrm_npeff_decomposition.h>

#include <coeff_fitting/config.h>
#include <coeff_fitting/host_context.h>
#include <coeff_fitting/fitting_context.h>

//////////////////////////////////////////////////////////////////////////////
// Flag definitions

// TODO: Add option to only use the first n_examples from the PEFs.

DEFINE_string(pef_filepath, "", "Filepath of the HDF5 file containing the per-example Fishers.");
DEFINE_string(decomposition_filepath, "", "Filepath of the HDF5 file containing the decomposition whose G will be used.");
DEFINE_string(output_filepath, "", "Filepath where the HDF5 file containing the output will be written to.");

DEFINE_bool(write_G_to_output, true, "Whether to write a copy of the G into the output decomposition file.");

// TODO: Write descriptions for these.
// NOTE: Setting these to -1 means to use all of the columns/examples in a single chunk.
DEFINE_int64(n_examples_per_chunk, -1, "");
DEFINE_int64(n_columns_per_chunk, -1, "");

DEFINE_int64(n_iters, -1, "The number of iterations in the fitting process.");
DEFINE_double(mu_eps, 1e-9, "Epsilon for the multiplicative update on W.");

DEFINE_int64(rand_gen_seed, 32497, "Seed to use for random number generation.");

DEFINE_int64(n_preprocess_cpu_threads, 1, "The number of threads to use for preprocessing on the CPU.");

// Maybe temporary flags here.
DEFINE_bool(pefs_col_offsets_non_cumulative, false, "Needed to account for bug in my first generation of the LRM-PEFS.");
DEFINE_bool(truncate_when_G_index_map_mismatch, false, "Accounts for (now fixed) bug in LRM-NPEFF. Accidentally did not include the last n_cols % n_partitions columns when constructing G.");

//////////////////////////////////////////////////////////////////////////////

using namespace npeff::coeff_fitting;

//////////////////////////////////////////////////////////////////////////////

void validate_flags() {
    if(FLAGS_output_filepath.empty()) {
        THROW_MSG("Please provide a valid --output_filepath flag value.");
    }
    if(FLAGS_n_iters <= 0) {
        THROW_MSG("Must set the --n_iters flag to a positive integer.");
    }
}

// Reads the fields that mirror flags into the config struct.
CoeffFittingConfig read_config_from_flags() {
    CoeffFittingConfig config;

    config.n_examples_per_chunk = FLAGS_n_examples_per_chunk;
    config.n_columns_per_chunk = FLAGS_n_columns_per_chunk;

    config.n_iters = FLAGS_n_iters;
    config.mu_eps = FLAGS_mu_eps;

    config.rand_gen_seed = FLAGS_rand_gen_seed;

    config.n_preprocess_cpu_threads = FLAGS_n_preprocess_cpu_threads;

    config.pefs_col_offsets_non_cumulative = FLAGS_pefs_col_offsets_non_cumulative;
    config.truncate_when_G_index_map_mismatch = FLAGS_truncate_when_G_index_map_mismatch;

    return config;
}

//////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    validate_flags();

    auto config = read_config_from_flags();
    config.read_in_metadata(FLAGS_pef_filepath, FLAGS_decomposition_filepath);

    std::cout << "n_example_chunks: " << config.n_example_chunks() << "\n";
    std::cout << "n_column_chunks: " << config.n_column_chunks() << "\n";

    auto host_context = std::unique_ptr<HostContext>(new HostContext(config, FLAGS_pef_filepath));
    host_context->fill_out_fields_from_decomposition(FLAGS_decomposition_filepath);

    FittingContext fitting_ctx(std::move(host_context));
    fitting_ctx.set_up_work();
    auto row_major_W = fitting_ctx.compute_W_row_major();

    npeff::outputs::DenseLrmNpeffDecomposition output;
    output.set_W(std::move(row_major_W), false);
    if(FLAGS_write_G_to_output) {
        output.set_G(std::move(fitting_ctx.host_ctx->G));
    }
    output.set_new_to_old_col_indices(std::move(fitting_ctx.host_ctx->new_to_old_col_indices));
    output.set_n_parameters(config.n_parameters);
    output.set_n_classes(config.n_classes);

    // // TODO: If we end up computing losses, save them here.
    // output.set_log_loss_frequency(inputs.config.log_loss_frequency);
    // output.set_losses_G_only(manager.losses_G_only);

    output.save(FLAGS_output_filepath);
}


// Size of an example G: (256, 12_289_200)
