
#include <iostream>
#include <string>

#include <gflags/gflags.h>

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include <inputs/lrm_pefs.h>
#include <outputs/lrm_npeff_decomposition.h>

#include <preprocess/construct_csr_matrix.h>
#include <preprocess/partition_by_columns.h>
#include <preprocess/column_pruning.h>
#include <preprocess/pef_normalization.h>
#include <preprocess/sort_by_col_indices.h>

#include <gpu/gpu_info.h>

#include <factorization/compute_tr_xx.h>
#include <factorization/config.h>
#include <factorization/manager.h>

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

DEFINE_double(orthogonal_regularization_strength, 0.0, "Strength of orthonal regularization.");

DEFINE_int64(min_nonzero_per_col, 1, "Prune columns with fewer than this many non-zero entries.");

DEFINE_int64(rand_gen_seed, 32497, "Seed to use for random number generation.");

DEFINE_int64(log_loss_frequency, 10, "Compute and log the loss every this number of steps.");

DEFINE_int64(n_preprocess_cpu_threads, 1, "The number of threads to use for preprocessing on the CPU.");


// DEFINE_string(initialization_filepath, "", "Optional path to decomposition h5 file containing initialization for W and G.");
// DEFINE_bool(initialization_use_W, true, "Whether to use to the W from --initialization_filepath or randomly initialize. Has no effect if --initialization_filepath is not set.");
// DEFINE_string(initialization_column_pruning_policy, "", "");  // TODO: Is enum, document meaning and options.


// Maybe temporary flags here.
DEFINE_bool(pefs_col_offsets_non_cumulative, false, "Needed to account for bug in my first generation of the LRM-PEFS.");


//////////////////////////////////////////////////////////////////////////////

template<typename IndT>
npeff::factorization::FactorizationConfig create_config(
    npeff::inputs::LrmPefs& pefs, npeff::CsrMatrix<IndT>& A, double tr_xx
) {
    npeff::factorization::FactorizationConfig config;

    config.rank = FLAGS_n_components;
    config.n_classes = pefs.n_classes;

    config.n_cols_total = A.n_cols;

    config.n_iters_G_only = FLAGS_n_iters_G_only;
    config.n_iters_joint = FLAGS_n_iters_joint;

    config.rand_gen_seed = FLAGS_rand_gen_seed;
    
    config.log_loss_frequency = FLAGS_log_loss_frequency;

    config.learning_rate_G_joint = FLAGS_learning_rate_G;
    config.learning_rate_G_G_only = FLAGS_learning_rate_G_G_only;
    if (config.learning_rate_G_G_only <= 0.0) {
        config.learning_rate_G_G_only = config.learning_rate_G_joint;
    }

    config.mu_eps = FLAGS_mu_eps;

    if (FLAGS_orthogonal_regularization_strength > 0.0) {
        config.ortho_reg_config.regularization_strength = FLAGS_orthogonal_regularization_strength;
    }

    config.tr_xx = tr_xx;

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


template <typename IndT>
struct RunInputs {
    std::unique_ptr<npeff::DenseMatrix<IndT>> new_to_old_col_indices;

    npeff::factorization::FactorizationConfig config;
    std::vector<std::unique_ptr<npeff::CsrMatrix<IndT>>> A_partitions;

    bool can_reindex_with_int32() {
        for(auto& mat : A_partitions) {
            if(!mat->can_use_int32_indices()) { return false; }
        }
        return true;
    }

    RunInputs<int32_t> reindex_with_int32() {
        THROW_IF_FALSE(can_reindex_with_int32());
        RunInputs<int32_t> ret;
        ret.config = this->config;
        ret.new_to_old_col_indices = npeff::DenseMatrix<int32_t>::reindex<IndT>(std::move(this->new_to_old_col_indices));
        for(auto& partition : A_partitions) {
            ret.A_partitions.push_back(npeff::CsrMatrix<IndT>::reindex_with_int32(std::move(partition)));
        }
        return ret;
    }
};


template <typename IndT>
RunInputs<IndT> make_run_inputs(npeff::inputs::LrmPefs& lrm_pefs) {
    npeff::CsrMatrix<IndT> A = npeff::preprocessing::construct_csr_matrix<IndT>(
        lrm_pefs, FLAGS_pefs_col_offsets_non_cumulative);
    std::cout << "Constructed initial sparse PEFs matrix.\n";

    std::unique_ptr<npeff::DenseMatrix<IndT>> new_to_old_col_indices;
    npeff::preprocessing::prune_columns_in_place(&A, &new_to_old_col_indices, FLAGS_min_nonzero_per_col);
    npeff::preprocessing::normalize_pefs_in_place(&A, lrm_pefs);
    npeff::preprocessing::sort_by_col_indices(&A, FLAGS_n_preprocess_cpu_threads);
    std::cout << "Finished preprocessing PEFs matrix.\n";

    double tr_xx = npeff::factorization::compute_tr_xx(A, lrm_pefs.n_classes, FLAGS_n_preprocess_cpu_threads);
    std::cout << "Finished computing tr(XX^T).\n";

    auto config = create_config(lrm_pefs, A, tr_xx);
    
    int64_t n_partitions = npeff::gpu::get_device_count();

    // std::cout << "new_to_old_col_indices size = " << new_to_old_col_indices->n_entries << "\n";

    RunInputs<IndT> ret;
    ret.new_to_old_col_indices = std::move(new_to_old_col_indices);
    ret.config = config;
    ret.A_partitions = npeff::preprocessing::partition_by_columns_uniformly(A, n_partitions);
    return ret;
}


template <typename IndT>
void run_and_save(npeff::inputs::LrmPefs& lrm_pefs, RunInputs<IndT>& inputs) {
    std::cout << "Starting to run factorization.\n";
    npeff::factorization::MultiGpuManager<IndT> manager(inputs.A_partitions, inputs.config);
    manager.run();

    // NOTE: These will be made row major by the decomposition saving wrapper class.
    auto W = manager.read_W_from_gpu();
    auto G = manager.read_G_from_gpu();

    npeff::outputs::DenseLrmNpeffDecomposition output;
    output.set_W(std::move(W));
    output.set_G(std::move(G));
    output.set_new_to_old_col_indices(std::move(inputs.new_to_old_col_indices));
    output.set_n_parameters(lrm_pefs.n_parameters);
    output.set_n_classes(lrm_pefs.n_classes);

    output.set_log_loss_frequency(inputs.config.log_loss_frequency);
    output.set_losses_G_only(manager.losses_G_only);
    output.set_losses_joint(manager.losses_joint);

    output.save(FLAGS_output_filepath);
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_output_filepath.empty()) {
        THROW_MSG("Please provide a valid --output_filepath flag value.");
    }

    std::string pef_filepath = FLAGS_pef_filepath;

    auto lrm_pefs = npeff::inputs::LrmPefs::load(pef_filepath, FLAGS_n_examples);
    std::cout << "LRM-PEFS loaded from disk.\n";

    if(npeff::preprocessing::can_csr_matrix_use_int32_indices(lrm_pefs)) {
        std::cout << "Creating inputs using int32 indices.\n";
        auto inputs = make_run_inputs<int32_t>(lrm_pefs);
        run_and_save(lrm_pefs, inputs);
    } else {
        std::cout << "Creating inputs using int64 indices.\n";
        auto inputs = make_run_inputs<int64_t>(lrm_pefs);
        if(inputs.can_reindex_with_int32()) {
            std::cout << "Re-indexing with int32 indices.\n";
            auto inputs32 = inputs.reindex_with_int32();
            run_and_save(lrm_pefs, inputs32);
        } else {
            run_and_save(lrm_pefs, inputs);
        }
    }
    
}
