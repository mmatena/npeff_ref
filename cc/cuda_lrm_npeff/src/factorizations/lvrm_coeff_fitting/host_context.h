#pragma once
// Context containing and storing information about pieces of
// the coefficient fitting problem on the host (including the disk).
// 
// Intended to only be used for the pre-iterative steps.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "./config.h"

#include <util/macros.h>
#include <containers/dense_matrix.h>
#include <containers/sparse_matrix.h>


namespace npeff {
namespace factorizations {
namespace lvrm_coeff_fitting {


// Information related to an example-wise chunk of PEFs.
struct HostPefsChunk {
    std::vector<std::unique_ptr<CsrMatrix<int32_t>>> A_partitions;
    std::unique_ptr<DenseMatrix<int64_t>> example_row_offsets;
};


struct HostContext {
    using FloatMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;
    using IndMatrixPtr = std::unique_ptr<npeff::DenseMatrix<int32_t>>;

    HostContext(CoeffFittingConfig config, std::string pef_filepath) :
        config(config), pef_filepath(pef_filepath)
    {}

    // Must be fully filled out before being attached to the host context.
    CoeffFittingConfig config;

    // We read information abouts the PEFs as needed from the disk.
    std::string pef_filepath;

    // The information from the decomposition is stored on the host's RAM.
    FloatMatrixPtr G;
    IndMatrixPtr new_to_old_col_indices;

    // Information about the pefs. Unlike the same-named field in the HostPefsChunk
    // struct, this is for the entire set of PEFs being processed.
    std::unique_ptr<DenseMatrix<int64_t>> example_row_offsets;
    std::vector<int64_t> n_rows_per_chunk;


    // Fills out the G and new_to_old_col_indices attributes.
    void fill_out_fields_from_decomposition(std::string& decomposition_filepath);

    ////////////////////////////////////////////////////////////
    // Methods for interacting with the host context.

    int64_t get_n_cols_in_chunk(int64_t col_chunk_index);
    int64_t get_n_examples_in_chunk(int64_t example_chunk_index);
    int64_t get_n_rows_in_chunk(int64_t example_chunk_index);

    int64_t get_max_rows_in_chunks();

    float* get_G_chunk_start_ptr(int64_t col_chunk_index) {
        return G->data.get() + col_chunk_index * G->n_rows * config.n_columns_per_chunk;
    }

    HostPefsChunk load_partitioned_A_chunk(int64_t example_chunk_index);

};

}  // lvrm_coeff_fitting
}  // factorizations
}  // npeff
