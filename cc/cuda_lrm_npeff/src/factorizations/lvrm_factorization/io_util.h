#pragma once

#include <cstdint>
#include <cmath>
#include <string>

#include <gpu/gpu_info.h>

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include <inputs/lvrm_pefs.h>
#include <outputs/lrm_npeff_decomposition.h>

#include <preprocess/construct_csr_matrix_from_lvrm_pefs.h>
#include <preprocess/partition_by_columns.h>
#include <preprocess/column_pruning.h>
#include <preprocess/pef_normalization.h>
#include <preprocess/sort_by_col_indices.h>

#include <factorizations/lvrm_factorization/config.h>
#include <factorizations/lvrm_factorization/compute_tr_xx.h>
#include <factorizations/lvrm_factorization/manager.h>

namespace npeff {
namespace factorizations {
namespace lvrm_factorization {

// Context for the run. Creating a specific struct for it to be cleaner.
template <typename IndT>
struct RunContext {
    using FloatMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;
    using IndMatrixPtr = std::unique_ptr<npeff::DenseMatrix<IndT>>;

    FactorizationConfig config;

    std::string output_filepath;

    int64_t n_parameters;

    // shape = [1, n_examples + 1]
    std::unique_ptr<npeff::DenseMatrix<int64_t>> example_row_offsets;

    IndMatrixPtr new_to_old_col_indices;
    std::vector<std::unique_ptr<npeff::CsrMatrix<IndT>>> A_partitions;

    void run() {
        std::cout << "Starting to run factorization.\n";

        MultiGpuManager<IndT> manager(A_partitions, config, std::move(example_row_offsets));
        manager.run();

        // NOTE: These will be made row major by the decomposition saving wrapper class.
        auto W = manager.read_W_from_gpu();
        auto G = manager.read_G_from_gpu();

        // NOTE: Since n_classes does not really make sense for lvrm pefs, we
        // just set it to -1.
        npeff::outputs::DenseLrmNpeffDecomposition output;
        output.set_W(std::move(W));
        output.set_G(std::move(G));
        output.set_new_to_old_col_indices(std::move(new_to_old_col_indices));
        output.set_n_parameters(n_parameters);
        output.set_n_classes(-1);

        output.set_log_loss_frequency(config.log_loss_frequency);
        output.set_losses_G_only(manager.losses_G_only);
        output.set_losses_joint(manager.losses_joint);

        output.save(output_filepath);

    }

    bool can_reindex_with_int32() {
        for(auto& mat : A_partitions) {
            if(!mat->can_use_int32_indices()) { return false; }
        }
        return true;
    }

    RunContext<int32_t> reindex_with_int32() {
        THROW_IF_FALSE(can_reindex_with_int32());

        RunContext<int32_t> ret;

        ret.config = config;
        ret.output_filepath = output_filepath;
        ret.n_parameters = this->n_parameters;

        ret.example_row_offsets = std::move(this->example_row_offsets);

        ret.new_to_old_col_indices = npeff::DenseMatrix<int32_t>::reindex<IndT>(std::move(this->new_to_old_col_indices));

        for(auto& partition : A_partitions) {
            ret.A_partitions.push_back(npeff::CsrMatrix<IndT>::reindex_with_int32(std::move(partition)));
        }

        return ret;
    }
};



// Additional configuration for the run context.
struct AdditionalRunContextConfig {
    std::string output_filepath;
    int64_t min_nonzero_per_col = 1;

    int64_t n_preprocess_cpu_threads = 1;
};


// Note that this function can modify its inputs.
template <typename IndT>
RunContext<IndT> create_run_context(
    npeff::inputs::LvrmPefs& pefs,
    FactorizationConfig& partial_config,
    AdditionalRunContextConfig& additional_config
) {
    std::unique_ptr<npeff::DenseMatrix<int64_t>> example_row_offsets = pefs.compute_example_row_offsets();

    npeff::CsrMatrix<IndT> A = npeff::preprocessing::construct_csr_matrix<IndT>(pefs);
    std::cout << "Constructed initial sparse PEFs matrix.\n";

    std::unique_ptr<npeff::DenseMatrix<IndT>> new_to_old_col_indices;
    npeff::preprocessing::prune_columns_in_place(&A, &new_to_old_col_indices, additional_config.min_nonzero_per_col);
    npeff::preprocessing::normalize_lvrm_pefs_in_place(&A, *pefs.pef_frobenius_norms, *example_row_offsets);
    npeff::preprocessing::sort_by_col_indices(&A, additional_config.n_preprocess_cpu_threads);
    std::cout << "Finished preprocessing PEFs matrix.\n";

    double tr_xx = npeff::factorizations::lvrm_factorization::compute_tr_xx(
        A, *example_row_offsets, additional_config.n_preprocess_cpu_threads);
    std::cout << "Finished computing tr(XX^T).\n";

    // Update the config.
    partial_config.n_examples = example_row_offsets->n_entries - 1;
    partial_config.n_cols_total = A.n_cols;


    // std::cout << "SETTING tr_xx to 0 for DEBUGGING.\n";
    partial_config.tr_xx = tr_xx;
    // partial_config.tr_xx = 0.0f;

    // std::cout << "n_cols_total: " << A.n_cols << "\n";
    std::cout << "tr_xx: " << tr_xx << "\n";

    int64_t n_partitions = npeff::gpu::get_device_count();

    RunContext<IndT> ret;
    ret.config = partial_config;
    ret.output_filepath = additional_config.output_filepath;
    ret.n_parameters = pefs.n_parameters;
    ret.example_row_offsets = std::move(example_row_offsets);
    ret.new_to_old_col_indices = std::move(new_to_old_col_indices);
    ret.A_partitions = npeff::preprocessing::partition_by_columns_uniformly(A, n_partitions);

    return ret;
}



}  // lvrm_factorization
}  // factorizations
}  // npeff
