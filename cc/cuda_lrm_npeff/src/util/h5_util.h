#pragma once

#include <memory>
#include <stddef.h>
#include <vector>

#include <hdf5.h>

#include <containers/dense_matrix.h>
#include <util/macros.h>

namespace npeff {
namespace util {
namespace h5 {


// Easy way to convert from C++ types to what hdf5 uses to indicate types.
// The specializations are included in the corresponding .cc file. For some reason,
// there appear to be issues with duplicate symbols if I include them here.
template<typename T>
struct TypeId {
    static hid_t id;
};


///////////////////////////////////////////////////////////////////////////////
// Helpers for reading from h5 files.


// This works with 1d or 2d datasets. 1d datasets will be returned as a matrix
// with the shape [1, ds_size].
// 
// The returned matrix will be TRANSPOSED since hdf5 stores in row-major
// order while the DenseMatrix is stored in column major order.
// 
// Set n_examples to -1 to read everything in.
template<typename T>
std::unique_ptr<DenseMatrix<T>> read_dataset_to_matrix(
    hid_t file, const std::string& ds_name, int64_t n_examples = -1, int64_t examples_offset = 0
) {
    hid_t valuesD = H5Dopen(file, ds_name.c_str(), H5P_DEFAULT);
    hid_t valuesS = H5Dget_space(valuesD);

    int rank = H5Sget_simple_extent_ndims(valuesS);

    hsize_t dims[rank];
    H5Sget_simple_extent_dims(valuesS, dims, NULL);

    THROW_IF_FALSE(examples_offset < dims[0]);

    if(n_examples >= 0) {
        THROW_IF_FALSE(examples_offset + n_examples <= dims[0]);
        dims[0] = n_examples;
    }

    hsize_t hs_start[rank];
    for(int i=0;i<rank;i++) { hs_start[i] = 0; }
    hs_start[0] = examples_offset;

    hid_t mem_space = H5Screate_simple(rank, dims, dims);

    int64_t n_rows, n_cols;
    if (rank == 1) {
        n_rows = 1;
        n_cols = dims[0];
    } else if(rank == 2) {
        n_rows = dims[1];
        n_cols = dims[0];
    } else {
        THROW_MSG("HDF5 data set must be rank 1 or 2.");
    }

    std::unique_ptr<DenseMatrix<T>> ret(new DenseMatrix<T>(n_rows, n_cols));

    H5Sselect_hyperslab(valuesS, H5S_SELECT_SET, hs_start, NULL, dims, NULL);
    H5Dread(valuesD, TypeId<T>::id, mem_space, valuesS, H5P_DEFAULT, ret->data.get());

    H5Sclose(mem_space);
    H5Sclose(valuesS);
    H5Dclose(valuesD);

    return ret;
}


template <typename T>
void read_attribute(hid_t id, const std::string& name, T* out_ptr) {
    hid_t attr = H5Aopen_name(id, name.c_str());
    H5Aread(attr, TypeId<T>::id, out_ptr);
    H5Aclose(attr);
}


std::vector<int64_t> read_dataset_dims(hid_t file, const std::string& ds_name);


///////////////////////////////////////////////////////////////////////////////
// Helpers for writing to h5 files.


template <typename T>
hid_t write_row_major_matrix_to_dataset(
    hid_t group_id,
    const std::string& name,
    DenseMatrix<T>& mat
) {
    hsize_t dims[2] = {(hsize_t) mat.n_rows, (hsize_t) mat.n_cols};
    hid_t space = H5Screate_simple(2, dims, dims);
    hid_t ds = H5Dcreate(group_id, name.c_str(), TypeId<T>::id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, TypeId<T>::id, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data.get());
    H5Sclose(space);
    H5Dclose(ds);
    return ds;
}


template <typename T>
hid_t write_matrix_to_dataset_as_1d(
    hid_t group_id,
    const std::string& name,
    DenseMatrix<T>& mat
) {
    hsize_t dims[1] = {(hsize_t) mat.n_entries};
    hid_t space = H5Screate_simple(1, dims, dims);
    hid_t ds = H5Dcreate(group_id, name.c_str(), TypeId<T>::id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, TypeId<T>::id, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data.get());
    H5Sclose(space);
    H5Dclose(ds);
    return ds;
}


template <typename T>
hid_t write_vector_to_dataset(
    hid_t group_id,
    const std::string& name,
    std::vector<T>& vec
) {
    hsize_t dims[1] = {(hsize_t) vec.size()};
    hid_t space = H5Screate_simple(1, dims, dims);
    hid_t ds = H5Dcreate(group_id, name.c_str(), TypeId<T>::id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, TypeId<T>::id, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
    H5Sclose(space);
    H5Dclose(ds);
    return ds;
}


template <typename T>
hid_t write_attribute_scalar(
    hid_t id,
    const std::string& name,
    T data
) {
    hsize_t Sdims[0] = {};
    hid_t SS = H5Screate_simple(0, Sdims, Sdims);
    hid_t SA = H5Acreate(id, name.c_str(), TypeId<T>::id, SS, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(SA, TypeId<T>::id, &data);
    H5Sclose(SS);
    H5Aclose(SA);
    return SA;
}




}  // h5
}  // util
}  // npeff
