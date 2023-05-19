#pragma once
// Expansion-specific utilities for reading inputs from files
// and writing outputs to files.

#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>

#include <inputs/lrm_pefs.h>
#include <inputs/lrm_npeff_decomposition.h>

#include <outputs/lrm_npeff_decomposition.h>

#include <preprocess/construct_csr_matrix.h>
#include <preprocess/partition_by_columns.h>
#include <preprocess/column_pruning.h>
#include <preprocess/pef_normalization.h>
#include <preprocess/sort_by_col_indices.h>

#include <gpu/gpu_info.h>

#include <factorization/compute_tr_xx.h>
#include "./config.h"
#include "./manager.h"


namespace npeff {
namespace expansion {



// Context for the run. Creating a specific struct for it to be cleaner.
template <typename IndT>
struct RunContext {
    using FloatMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;
    using IndMatrixPtr = std::unique_ptr<npeff::DenseMatrix<IndT>>;

    ExpansionConfig& config;

    int64_t n_parameters;
    int64_t n_classes;

    IndMatrixPtr new_to_old_col_indices;

    // Will be left empty if we do not use this to initialize W_f.
    FloatMatrixPtr initial_W_f;

    std::vector<std::unique_ptr<npeff::CsrMatrix<IndT>>> A_partitions;
    std::vector<FloatMatrixPtr> G_f_partitions;

protected:
    std::unique_ptr<MultiGpuManager<IndT>> manager;

public:
    RunContext(ExpansionConfig& config) : config(config) {}

    void run(std::string& output_filepath) {
        manager = std::unique_ptr<MultiGpuManager<IndT>>(new MultiGpuManager<IndT>(
            A_partitions, G_f_partitions, std::move(initial_W_f), config));
        manager->run();

        // NOTE: These will be made row major by the decomposition saving wrapper class.
        auto W = manager->read_W_from_gpu();
        auto G = manager->read_G_from_gpu();

        npeff::outputs::DenseLrmNpeffDecomposition output;
        output.set_W(std::move(W));
        output.set_G(std::move(G));
        output.set_new_to_old_col_indices(std::move(new_to_old_col_indices));
        output.set_n_parameters(n_parameters);
        output.set_n_classes(n_classes);

        output.set_log_loss_frequency(config.log_loss_frequency);
        output.set_losses_G_only(manager->losses_G_only);
        output.set_losses_joint(manager->losses_joint);

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

        RunContext<int32_t> ret(this->config);
        ret.n_parameters = this->n_parameters;
        ret.n_classes = this->n_classes;
        ret.initial_W_f = std::move(this->initial_W_f);

        ret.new_to_old_col_indices = npeff::DenseMatrix<int32_t>::reindex<IndT>(std::move(this->new_to_old_col_indices));

        for(auto& partition : A_partitions) {
            ret.A_partitions.push_back(npeff::CsrMatrix<IndT>::reindex_with_int32(std::move(partition)));
        }

        for(auto& partition : G_f_partitions) {
            ret.G_f_partitions.push_back(std::move(partition));
        }

        return ret;
    }

protected:

};


struct CreateRunContextOptions {
    int64_t n_partitions;

    bool use_W_from_decomposition;

    bool pefs_col_offsets_non_cumulative;
    int64_t n_preprocess_cpu_threads;
};


// Note that this function can modify its inputs.
template <typename IndT>
RunContext<IndT> create_run_context(
    ExpansionConfig& config,
    npeff::inputs::LrmPefs& lrm_pefs,
    npeff::inputs::DenseLrmNpeffDecompositionFromFile& frozen_decomposition,
    CreateRunContextOptions options
) {
    // TODO: Read G directly into partitions remove the RAM cost of having
    // to essentially make a copy for the partitions.
    auto G = frozen_decomposition.read_G();
    std::cout << "Finished reading in G_f.\n";

    std::unique_ptr<npeff::DenseMatrix<float>> W;
    if (options.use_W_from_decomposition) {
        W = frozen_decomposition.read_W();
    }

    auto new_to_old_col_indices_32 = frozen_decomposition.read_new_to_old_col_indices();
    auto new_to_old_col_indices = npeff::DenseMatrix<IndT>::reindex(std::move(new_to_old_col_indices_32));

    npeff::CsrMatrix<IndT> A = npeff::preprocessing::construct_csr_matrix<IndT>(
        lrm_pefs, options.pefs_col_offsets_non_cumulative);
    std::cout << "Constructed initial sparse PEFs matrix.\n";

    // std::cout << "MATRIX VALID 1: " << A.validate_indices() << "\n";

    npeff::preprocessing::prune_columns_given_indices_in_place(&A, *new_to_old_col_indices);
    // std::cout << "MATRIX VALID 2: " << A.validate_indices() << "\n";

    npeff::preprocessing::normalize_pefs_in_place(&A, lrm_pefs);
    // std::cout << "MATRIX VALID 3: " << A.validate_indices() << "\n";

    npeff::preprocessing::sort_by_col_indices(&A, options.n_preprocess_cpu_threads);
    // std::cout << "MATRIX VALID 4: " << A.validate_indices() << "\n";

    std::cout << "Finished preprocessing PEFs matrix.\n";

    config.rank_frozen = G->n_rows;
    config.n_classes = lrm_pefs.n_classes;
    config.n_cols_total = A.n_cols;
    config.tr_xx = npeff::factorization::compute_tr_xx(A, lrm_pefs.n_classes, options.n_preprocess_cpu_threads);

    RunContext<IndT> ret(config);
    ret.n_parameters = lrm_pefs.n_parameters;
    ret.n_classes = lrm_pefs.n_classes;

    ret.A_partitions = npeff::preprocessing::partition_by_columns_uniformly(A, options.n_partitions);
    ret.G_f_partitions = npeff::preprocessing::partition_by_columns_uniformly(*G, options.n_partitions);

    ret.new_to_old_col_indices = std::move(new_to_old_col_indices);
    ret.initial_W_f = std::move(W);

    return ret;
}


bool are_inputs_compatible(
    npeff::inputs::LrmPefs& lrm_pefs,
    npeff::inputs::DenseLrmNpeffDecompositionFromFile& frozen_decomposition,
    bool use_W_from_decomposition
) {
    if(lrm_pefs.n_parameters != frozen_decomposition.n_parameters) {
        return false;
    }
    if(lrm_pefs.n_classes != frozen_decomposition.n_classes) {
        return false;
    }

    if(use_W_from_decomposition) {
        if(lrm_pefs.n_examples() != frozen_decomposition.read_n_examples()) {
            return false;
        }
    }

    return true;
}



}  // expansion
}  // npeff
