#include "./lrm_npeff_decomposition.h"

#include <util/h5_util.h>


using FloatMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;
using IndMatrixPtr = std::unique_ptr<npeff::DenseMatrix<int32_t>>;


namespace npeff {
namespace inputs {


void DenseLrmNpeffDecompositionFromFile::read_scalar_fields_from_file() {
    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    hid_t dataG = H5Gopen(file, "data", H5P_DEFAULT);
    util::h5::read_attribute(dataG, "n_classes", &this->n_classes);
    util::h5::read_attribute(dataG, "n_parameters", &this->n_parameters);
    H5Gclose(dataG);

    H5Fclose(file);
}

IndMatrixPtr DenseLrmNpeffDecompositionFromFile::read_new_to_old_col_indices() {
    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    IndMatrixPtr ret = util::h5::read_dataset_to_matrix<int32_t>(
        file, "data/new_to_old_col_indices");
    H5Fclose(file);
    return std::move(ret);
}

FloatMatrixPtr DenseLrmNpeffDecompositionFromFile::read_W() {
    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    FloatMatrixPtr ret = util::h5::read_dataset_to_matrix<float>(
        file, "data/W");
    H5Fclose(file);
    ret->transpose_in_place();
    return std::move(ret);
}

FloatMatrixPtr DenseLrmNpeffDecompositionFromFile::read_G() {
    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    FloatMatrixPtr ret = util::h5::read_dataset_to_matrix<float>(
        file, "data/G");
    H5Fclose(file);
    ret->transpose_in_place();
    return std::move(ret);
}

int64_t DenseLrmNpeffDecompositionFromFile::read_n_examples() {
    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::vector<int64_t> dims = util::h5::read_dataset_dims(file, "data/W");
    H5Fclose(file);
    return dims[0];
}


}  // inputs
}  // npeff
