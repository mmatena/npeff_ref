#pragma once
// The output of an LRM-NPEFF run.


#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <util/h5_util.h>
#include <util/macros.h>
#include <containers/dense_matrix.h>

namespace npeff {
namespace outputs {


// data/W
// data/G
// data/new_to_old_col_indices
// data.attrs["n_parameters"]
// data.attrs["n_classes"]
// 
// losses/G_only
// losses/joint
// losses.attrs["log_loss_frequency"]
class DenseLrmNpeffDecomposition {
    using FloatMatrixPtr = std::unique_ptr<npeff::DenseMatrix<float>>;
    using IndMatrixPtr = std::unique_ptr<npeff::DenseMatrix<int32_t>>;

    FloatMatrixPtr W;
    FloatMatrixPtr G;

    IndMatrixPtr new_to_old_col_indices;

    int64_t n_parameters;
    int64_t n_classes;

    // Information about the losses during fitting.
    int64_t log_loss_frequency;
    std::vector<float> losses_G_only;
    std::vector<float> losses_joint;

public:

    void set_W(FloatMatrixPtr W, bool convert_to_row_major = true) {
        if(convert_to_row_major) {
            W->convert_to_row_major_in_place();
        }
        this->W = std::move(W);
    }

    void set_G(FloatMatrixPtr G, bool convert_to_row_major = true) {
        if(convert_to_row_major) {
            G->convert_to_row_major_in_place();
        }
        this->G = std::move(G);
    }

    // When IndT == int32_t, there is a template specialization that does a std::move instead of a copy.
    template<typename IndT>
    void set_new_to_old_col_indices(std::unique_ptr<npeff::DenseMatrix<IndT>> new_to_old_col_indices) {
        this->new_to_old_col_indices = IndMatrixPtr(
            new npeff::DenseMatrix<int32_t>(1, new_to_old_col_indices->n_entries));

        npeff::util::convert_numeric_arrays<int32_t, IndT>(
            this->new_to_old_col_indices->data.get(),
            new_to_old_col_indices->data.get(),
            new_to_old_col_indices->n_entries);

    }

    void set_n_parameters(int64_t n_parameters) {
        this->n_parameters = n_parameters;
    }

    void set_n_classes(int64_t n_classes) {
        this->n_classes = n_classes;
    }

    void set_log_loss_frequency(int64_t log_loss_frequency) {
        this->log_loss_frequency = log_loss_frequency;
    }

    // NOTE: We're purposefully making a copy here.
    void set_losses_G_only(std::vector<float> losses_G_only) {
        this->losses_G_only = losses_G_only;
    }

    // NOTE: We're purposefully making a copy here.
    void set_losses_joint(std::vector<float> losses_joint) {
        this->losses_joint = losses_joint;
    }

    void save(std::string& filepath) {
        hid_t file = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        // Write main data.
        hid_t dataG = H5Gcreate(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        util::h5::write_attribute_scalar(dataG, "n_parameters", n_parameters);
        util::h5::write_attribute_scalar(dataG, "n_classes", n_classes);

        util::h5::write_row_major_matrix_to_dataset(dataG, "W", *W);

        // The G matrix might not always be set if we are only fitting coefficients with
        // the --write_G_to_output set to false.
        if(G) {
            util::h5::write_row_major_matrix_to_dataset(dataG, "G", *G);
            // std::cout << "G shape = (" << G->n_rows << ", " << G->n_cols << ")\n";

        }

        util::h5::write_matrix_to_dataset_as_1d(dataG, "new_to_old_col_indices", *new_to_old_col_indices);
        // std::cout << "new_to_old_col_indices size = " << new_to_old_col_indices->n_entries << "\n";

        H5Gclose(dataG);

        // Write information about the losses.
        hid_t lossesG = H5Gcreate(file, "losses", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        util::h5::write_attribute_scalar(lossesG, "log_loss_frequency", log_loss_frequency);
        util::h5::write_vector_to_dataset(lossesG, "G_only", losses_G_only);
        util::h5::write_vector_to_dataset(lossesG, "joint", losses_joint);
        H5Gclose(lossesG);


        H5Fclose(file);
    }

};

template<>
void DenseLrmNpeffDecomposition::set_new_to_old_col_indices(
    std::unique_ptr<npeff::DenseMatrix<int32_t>> new_to_old_col_indices);


}  // outputs
}  // npeff
