#pragma once
// Options controlling the factorization.

#include <cstdint>
#include <cmath>
#include <string>


namespace npeff {
namespace factorizations {
namespace lvrm_coeff_fitting {


struct CoeffFittingConfig {
    ////////////////////////////////////////////////////////////
    // Fields that are set manually via flags.
    // 
    // All of these should be filled out before calling the function(s)
    // that automatically fill out the rest of the fields.

    // If the values of these passed as flags exceeds the number of
    // actual columns/examples, then these will get set to the actual
    // number of columns/examples in the read_in_metadata.
    //
    // The same thing happens if these are set to -1.
    int64_t n_examples_per_chunk;
    int64_t n_columns_per_chunk;

    int64_t n_iters;

    int64_t rand_gen_seed;

    float mu_eps;

    int64_t n_preprocess_cpu_threads;

    ////////////////////////////////////////////////////////////
    // Fields that get filled in based on the input files.

    int64_t rank;

    int64_t n_parameters;

    // These are totals.
    int64_t n_examples;
    int64_t n_cols;

    int64_t n_example_chunks();
    int64_t n_column_chunks();

    // Maximum number of non-zero values per example.
    int64_t max_nnz_per_example;

    // Fills out some fields based on information contained within these files.
    void read_in_metadata(std::string& pef_filepath, std::string& decomposition_filepath);

};


}  // lvrm_coeff_fitting
}  // factorizations
}  // npeff
