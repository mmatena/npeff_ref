#include "./host_context.h"

#include <util/h5_util.h>

#include <inputs/lvrm_pefs.h>

#include <preprocess/construct_csr_matrix_from_lvrm_pefs.h>
#include <preprocess/partition_by_columns.h>
#include <preprocess/column_pruning.h>
#include <preprocess/pef_normalization.h>
#include <preprocess/sort_by_col_indices.h>

namespace npeff {
namespace factorizations {
namespace lvrm_coeff_fitting {


void HostContext::fill_out_fields_from_decomposition(std::string& decomposition_filepath) {
    hid_t file;

    // Fill in fields from the decomposition.

    file = H5Fopen(decomposition_filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    this->G = util::h5::read_dataset_to_matrix<float>(file, "data/G");
    // The G that is directly read in is the tranpose of the actual G. This is due to
    // column-major vs row-major annoyances.
    this->G->transpose_in_place();
    this->new_to_old_col_indices = util::h5::read_dataset_to_matrix<int32_t>(file, "data/new_to_old_col_indices");
    H5Fclose(file);

    if (G->n_cols != new_to_old_col_indices->n_cols) {
        // NOTE: This shouldn't be an issue anymore, but there exists a fix/work-around
        // in the non-LVRM version of this code.
        THROW_MSG("Unhandled mismatch in the number of columns indicated by G and new_to_old_col_indices.");
    }

    // Fill in fields from the PEFs file.

    file = H5Fopen(pef_filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::unique_ptr<npeff::DenseMatrix<int32_t>> ranks = util::h5::read_dataset_to_matrix<int32_t>(
        file, "data/ranks", config.n_examples);

    this->example_row_offsets = npeff::inputs::compute_lvrm_example_row_offsets(*ranks);

    int64_t ex_offset = 0;

    for(int64_t i = 0; i < config.n_example_chunks(); i++) {
        int64_t n_chunk_ex = get_n_examples_in_chunk(i);

        int64_t start_row = this->example_row_offsets->data.get()[ex_offset];
        int64_t end_row = this->example_row_offsets->data.get()[ex_offset + n_chunk_ex];

        this->n_rows_per_chunk.push_back(end_row - start_row);

        ex_offset += n_chunk_ex;
    }


    H5Fclose(file);
} 


////////////////////////////////////////////////////////////////////////////////////
// Methods for interacting with the host context.

int64_t HostContext::get_n_cols_in_chunk(int64_t col_chunk_index) {
    int64_t n_column_chunks = config.n_column_chunks();
    THROW_IF_FALSE(col_chunk_index < n_column_chunks);

    if (col_chunk_index == n_column_chunks - 1) {
        int64_t maybe_ret = config.n_cols % config.n_columns_per_chunk;
        if(maybe_ret > 0) {
            return maybe_ret;
        }
    }
    return config.n_columns_per_chunk;
}

int64_t HostContext::get_n_examples_in_chunk(int64_t example_chunk_index) {
    int64_t n_example_chunks = config.n_example_chunks();
    THROW_IF_FALSE(example_chunk_index < n_example_chunks);

    if (example_chunk_index == n_example_chunks - 1) {
        int64_t maybe_ret = config.n_cols % config.n_examples_per_chunk;
        if(maybe_ret > 0) {
            return maybe_ret;
        }
    }
    return config.n_examples_per_chunk;
}

int64_t HostContext::get_n_rows_in_chunk(int64_t example_chunk_index) {
    return n_rows_per_chunk[example_chunk_index];
}

int64_t HostContext::get_max_rows_in_chunks() {
    int64_t max_rows = 0;
    for(int64_t n_rows : n_rows_per_chunk) {
        if(n_rows > max_rows) {
            max_rows = n_rows;
        }
    }
    return max_rows;
}

std::vector<int32_t> create_partition_start_column_indices_(HostContext* hctx) {
    std::vector<int32_t> ret(hctx->config.n_column_chunks());
    int64_t col_offset = 0;
    for(int64_t i=0; i<hctx->config.n_column_chunks(); i++) {
        ret[i] = col_offset;
        col_offset += hctx->get_n_cols_in_chunk(i);
    }
    return ret;
}

HostPefsChunk HostContext::load_partitioned_A_chunk(int64_t example_chunk_index) {
    int64_t examples_offset = example_chunk_index * config.n_examples_per_chunk;
    int64_t n_examples = get_n_examples_in_chunk(example_chunk_index);

    // Read in the PEFs of the example chunk from disk.
    auto pefs = inputs::LvrmPefs::load(pef_filepath, n_examples, examples_offset);
    std::unique_ptr<npeff::DenseMatrix<int64_t>> example_row_offsets = pefs.compute_example_row_offsets();

    // Create a spare CSR-matrix given the data from the PEFs.
    npeff::CsrMatrix<int32_t> A = npeff::preprocessing::construct_csr_matrix<int32_t>(pefs);

    npeff::preprocessing::prune_columns_given_indices_in_place(&A, *new_to_old_col_indices);
    npeff::preprocessing::normalize_lvrm_pefs_in_place(&A, *pefs.pef_frobenius_norms, *example_row_offsets);
    npeff::preprocessing::sort_by_col_indices(&A, config.n_preprocess_cpu_threads);

    // std::cout << "A shape = (" << A.n_rows << ", " << A.n_cols << ")\n";

    // TODO [maybe]: Compute tr_xx?

    HostPefsChunk ret;

    auto start_indices = create_partition_start_column_indices_(this);
    ret.A_partitions = npeff::preprocessing::partition_by_columns_given_start_indices(
        A, start_indices);
    ret.example_row_offsets = std::move(example_row_offsets);

    return ret;
}


}  // lvrm_coeff_fitting
}  // factorizations
}  // npeff
