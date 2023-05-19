#include <iostream>

#include <inputs/lrm_pefs.h>
#include <util/h5_util.h>


namespace npeff {
namespace inputs {


// Use a value of -1 for n_examples to load everything.
LrmPefs LrmPefs::load(std::string& filepath, int64_t n_examples, int64_t examples_offset) {
    LrmPefs p;

    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    hid_t dataG = H5Gopen(file, "data", H5P_DEFAULT);
    util::h5::read_attribute(dataG, "n_classes", &p.n_classes);
    util::h5::read_attribute(dataG, "n_parameters", &p.n_parameters);
    H5Gclose(dataG);

    p.values = util::h5::read_dataset_to_matrix<float>(file, "data/values", n_examples, examples_offset);
    p.col_offsets = util::h5::read_dataset_to_matrix<int32_t>(file, "data/col_offsets", n_examples, examples_offset);
    p.row_indices = util::h5::read_dataset_to_matrix<int32_t>(file, "data/row_indices", n_examples, examples_offset);

    p.pef_frobenius_norms = util::h5::read_dataset_to_matrix<float>(
        file, "data/pef_frobenius_norms", n_examples, examples_offset);

    H5Fclose(file);
    return p;
}


}  // inputs
}  // npeff
