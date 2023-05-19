#include <iostream>

#include <inputs/lvrm_pefs.h>
#include <util/h5_util.h>
#include <util/macros.h>


namespace npeff {
namespace inputs {


std::unique_ptr<npeff::DenseMatrix<int64_t>> compute_lvrm_example_row_offsets(
    npeff::DenseMatrix<int32_t>& ranks
) {
    auto example_row_offsets = std::unique_ptr<npeff::DenseMatrix<int64_t>>(
        new npeff::DenseMatrix<int64_t>(1, ranks.n_entries + 1));

    int64_t* data = example_row_offsets->data.get();
    data[0] = 0;
    for(int64_t i = 0; i < ranks.n_entries; i++) {
        data[i + 1] = data[i] + ranks.data.get()[i];
    }

    return example_row_offsets;
}


int64_t get_start_index_of_example_in_col_sizes(npeff::DenseMatrix<int32_t>& ranks, int64_t example_index) {
    int64_t n_examples = ranks.n_entries;
    int32_t* data = ranks.data.get();

    if (example_index > n_examples) {
        THROW_MSG("Example index greater than the number of examples.");
    }

    int64_t index = 0;
    for(int64_t i=0; i<example_index; i++) {
        index += data[i];
    }

    return index;
}

// Use a value of -1 for n_examples to load everything.
LvrmPefs LvrmPefs::load(std::string& filepath, int64_t n_examples, int64_t examples_offset) {
    LvrmPefs p;

    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    hid_t dataG = H5Gopen(file, "data", H5P_DEFAULT);
    util::h5::read_attribute(dataG, "n_parameters", &p.n_parameters);
    H5Gclose(dataG);

    p.values = util::h5::read_dataset_to_matrix<float>(file, "data/values", n_examples, examples_offset);
    p.row_indices = util::h5::read_dataset_to_matrix<int32_t>(file, "data/row_indices", n_examples, examples_offset);

    p.ranks = util::h5::read_dataset_to_matrix<int32_t>(file, "data/ranks", n_examples, examples_offset);
    
    int64_t col_sizes_offset = get_start_index_of_example_in_col_sizes(*p.ranks, examples_offset);
    if (n_examples >= 0) {
        int64_t col_sizes_end_offset = get_start_index_of_example_in_col_sizes(*p.ranks, examples_offset + n_examples);
        p.col_sizes = util::h5::read_dataset_to_matrix<int32_t>(
            file, "data/col_sizes", col_sizes_end_offset - col_sizes_offset, col_sizes_offset);
    } else {
        p.col_sizes = util::h5::read_dataset_to_matrix<int32_t>(file, "data/col_sizes", -1, col_sizes_offset);
    }

    p.pef_frobenius_norms = util::h5::read_dataset_to_matrix<float>(
        file, "data/pef_frobenius_norms", n_examples, examples_offset);

    H5Fclose(file);

    return p;
}


}  // inputs
}  // npeff
