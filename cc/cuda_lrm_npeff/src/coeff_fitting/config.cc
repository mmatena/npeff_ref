#include "./config.h"

#include <climits>
#include <util/h5_util.h>

namespace npeff {
namespace coeff_fitting {


int64_t CoeffFittingConfig::n_example_chunks() {
    int64_t ret = n_examples / n_examples_per_chunk;
    if((n_examples % n_examples_per_chunk) != 0) {
        ret += 1;
    }
    return ret;
}


int64_t CoeffFittingConfig::n_column_chunks() {
    int64_t ret = n_cols / n_columns_per_chunk;
    if((n_cols % n_columns_per_chunk) != 0) {
        ret += 1;
    }
    return ret;
}


void CoeffFittingConfig::read_in_metadata(std::string& pef_filepath, std::string& decomposition_filepath) {
    hid_t file, dataG;

    ////////////////////////////////////////////////////////////
    // Read in stuff from the PEF file.

    file = H5Fopen(pef_filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dataG = H5Gopen(file, "data", H5P_DEFAULT);
    int64_t n_classes_pef;
    util::h5::read_attribute(dataG, "n_classes", &n_classes_pef);
    int64_t n_parameters_pef;
    util::h5::read_attribute(dataG, "n_parameters", &n_parameters_pef);
    H5Gclose(dataG);

    std::vector<int64_t> values_dims = util::h5::read_dataset_dims(file, "data/values");
    this->n_examples = values_dims[0];
    this->max_nnz_per_example = values_dims[1];

    H5Fclose(file);

    ////////////////////////////////////////////////////////////
    // Read in stuff from the decomposition file.

    file = H5Fopen(decomposition_filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dataG = H5Gopen(file, "data", H5P_DEFAULT);
    int64_t n_classes_decomposition;
    util::h5::read_attribute(dataG, "n_classes", &n_classes_decomposition);
    int64_t n_parameters_decomposition;
    util::h5::read_attribute(dataG, "n_parameters", &n_parameters_decomposition);
    H5Gclose(dataG);

    std::vector<int64_t> G_dims = util::h5::read_dataset_dims(file, "data/G");
    this->rank = G_dims[0];
    this->n_cols = G_dims[1];

    std::cout << "n_cols = " << n_cols << "\n";

    H5Fclose(file);

    ////////////////////////////////////////////////////////////
    // Some verification that some values are consistent across both files before
    // setting them.

    if(n_classes_pef != n_classes_decomposition) {
        THROW_MSG("Mismatch in stated number of classes from the PEFs and the decomposition.");
    }
    this->n_classes = n_classes_pef;

    if(n_parameters_pef != n_parameters_decomposition) {
        THROW_MSG("Mismatch in stated number of parameters from the PEFs and the decomposition.");
    }
    this->n_parameters = n_parameters_pef;

    ////////////////////////////////////////////////////////////
    // Modify some values based on what we have read in.

    if(this->n_examples_per_chunk < 0) {
        this->n_examples_per_chunk = this->n_examples;
    } else {
        this->n_examples_per_chunk = std::min(this->n_examples_per_chunk, this->n_examples);
    }
    
    if(this->n_columns_per_chunk < 0) {
        this->n_columns_per_chunk = this->n_cols;
    } else {
        this->n_columns_per_chunk = std::min(this->n_columns_per_chunk, this->n_cols);
    }

    ////////////////////////////////////////////////////////////
    // Ensure that we can use int32_t indices.

    if(this->n_examples_per_chunk * this->max_nnz_per_example >= (int64_t) INT32_MAX) {
        THROW_MSG("Int64 indices might be required. Please set n_examples_per_chunk to a lower value.");
    }

}


}  // coeff_fitting
}  // npeff
