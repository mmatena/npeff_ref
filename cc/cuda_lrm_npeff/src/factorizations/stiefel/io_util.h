#pragma once

#include <cstdint>
#include <cmath>
#include <string>

#include <gpu/gpu_info.h>

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include <inputs/lrm_pefs.h>
#include <outputs/lrm_npeff_decomposition.h>

#include <preprocess/construct_csr_matrix.h>
#include <preprocess/partition_by_columns.h>
#include <preprocess/column_pruning.h>
#include <preprocess/pef_normalization.h>
#include <preprocess/sort_by_col_indices.h>

#include <factorization/compute_tr_xx.h>

#include <factorizations/stiefel/manager.h>


namespace npeff {
namespace factorizations {
namespace stiefel {


// Context for the run. Creating a specific struct for it to be cleaner.
template <typename IndT>
struct RunContext {
    using FloatMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;
    using IndMatrixPtr = std::unique_ptr<npeff::DenseMatrix<IndT>>;

    FactorizationConfig config;

    std::string output_filepath;

    int64_t n_parameters;

    IndMatrixPtr new_to_old_col_indices;
    std::vector<std::unique_ptr<npeff::CsrMatrix<IndT>>> A_partitions;

    void run() {
        std::cout << "Starting to run factorization.\n";
        MultiGpuManager<IndT> manager(A_partitions, config);
        manager.run();

        // NOTE: These will be made row major by the decomposition saving wrapper class.
        auto W = manager.read_W_from_gpu();
        auto G = manager.read_G_from_gpu();

        npeff::outputs::DenseLrmNpeffDecomposition output;
        output.set_W(std::move(W));
        output.set_G(std::move(G));
        output.set_new_to_old_col_indices(std::move(new_to_old_col_indices));
        output.set_n_parameters(n_parameters);
        output.set_n_classes(config.n_classes);

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
    bool pefs_col_offsets_non_cumulative = true;
};


// Note that this function can modify its inputs.
template <typename IndT>
RunContext<IndT> create_run_context(
    npeff::inputs::LrmPefs& pefs,
    FactorizationConfig& partial_config,
    AdditionalRunContextConfig& additional_config
) {
    npeff::CsrMatrix<IndT> A = npeff::preprocessing::construct_csr_matrix<IndT>(
        pefs, additional_config.pefs_col_offsets_non_cumulative);
    std::cout << "Constructed initial sparse PEFs matrix.\n";

    std::unique_ptr<npeff::DenseMatrix<IndT>> new_to_old_col_indices;
    npeff::preprocessing::prune_columns_in_place(&A, &new_to_old_col_indices, additional_config.min_nonzero_per_col);
    npeff::preprocessing::normalize_pefs_in_place(&A, pefs);
    npeff::preprocessing::sort_by_col_indices(&A, additional_config.n_preprocess_cpu_threads);
    std::cout << "Finished preprocessing PEFs matrix.\n";

    double tr_xx = npeff::factorization::compute_tr_xx(A, pefs.n_classes, additional_config.n_preprocess_cpu_threads);
    std::cout << "Finished computing tr(XX^T).\n";

    // Update the config.
    partial_config.n_classes = pefs.n_classes;
    partial_config.n_cols_total = A.n_cols;
    partial_config.tr_xx = tr_xx;

    int64_t n_partitions = npeff::gpu::get_device_count();

    RunContext<IndT> ret;
    ret.config = partial_config;
    ret.output_filepath = additional_config.output_filepath;
    ret.n_parameters = pefs.n_parameters;
    ret.new_to_old_col_indices = std::move(new_to_old_col_indices);
    ret.A_partitions = npeff::preprocessing::partition_by_columns_uniformly(A, n_partitions);
    return ret;
}


}  // stiefel
}  // factorizations
}  // npeff
