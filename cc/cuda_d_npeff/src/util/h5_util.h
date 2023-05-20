#pragma once

#include <type_traits>

#include <hdf5.h>

#include <util/cuda_statuses.h>
#include <util/matrices.h>

#include <cuda/host/dense_matrix.h>


namespace H5Util {


template<typename T>
struct TypeToH5 {
    static hid_t type_id;
};

template<>
hid_t TypeToH5<int32_t>::type_id {H5T_NATIVE_INT32};
template<>
hid_t TypeToH5<int64_t>::type_id {H5T_NATIVE_INT64};
template<>
hid_t TypeToH5<float>::type_id {H5T_NATIVE_FLOAT};


/////////////////////////////////////////////////////////////////////////////////////////


MeMatrix* readDatasetToMatrix(hid_t file, const std::string& datasetName) {
    // NOTE: Due to column/row major stuff, I'll have to sometimes "tranpose" what this reads in.

    hid_t valuesD = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    hid_t valuesS = H5Dget_space(valuesD);

    int valuesRank = H5Sget_simple_extent_ndims(valuesS);
    THROWSERT(valuesRank == 2);


    hsize_t dims[2];
    H5Sget_simple_extent_dims(valuesS, dims, NULL);

    // TODO: Support values greater than MAX_INT.
    long n_rows = dims[0];
    long n_cols = dims[1];

    hsize_t hsStart[2] = {0, 0};
    hsize_t hsCount[2] = {(hsize_t) n_rows, (hsize_t) n_cols};
    hid_t memSpace = H5Screate_simple(2, hsCount, hsCount);

    MeMatrix* ret = new MeMatrix(n_rows, n_cols);

    H5Sselect_hyperslab(valuesS, H5S_SELECT_SET, hsStart, NULL, hsCount, NULL);
    H5Dread(valuesD, H5T_NATIVE_FLOAT, memSpace, valuesS, H5P_DEFAULT, ret->data);

    H5Sclose(memSpace);
    H5Sclose(valuesS);
    H5Dclose(valuesD);

    return ret;
}



template <typename T>
T* readDatasetToPtr(hid_t file, const std::string& datasetName) {
    hid_t valuesD = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    hid_t valuesS = H5Dget_space(valuesD);

    int valuesRank = H5Sget_simple_extent_ndims(valuesS);

    hsize_t dims[valuesRank];
    H5Sget_simple_extent_dims(valuesS, dims, NULL);

    hsize_t hsStart[valuesRank];
    for(int i=0;i<valuesRank;i++) {hsStart[i] = 0;}

    hid_t memSpace = H5Screate_simple(valuesRank, dims, dims);

    hsize_t n_els = 1;
    for(int i=0;i<valuesRank;i++) {n_els *= dims[i];}

    T* ret = new T[n_els];

    H5Sselect_hyperslab(valuesS, H5S_SELECT_SET, hsStart, NULL, dims, NULL);
    H5Dread(valuesD, TypeToH5<T>::type_id, memSpace, valuesS, H5P_DEFAULT, ret);

    H5Sclose(memSpace);
    H5Sclose(valuesS);
    H5Dclose(valuesD);

    return ret;
}


template <typename T>
void readAttribute(hid_t id, const std::string& name, T* writeLocation) {
    hid_t attr = H5Aopen_name(id, name.c_str());
    H5Aread(attr, H5Util::TypeToH5<T>::type_id, writeLocation);
    H5Aclose(attr);
}

/////////////////////////////////////////////////////////////////////////////////////////


hid_t WriteMatrixToDataset(
    hid_t group_id,
    const std::string& name,
    Cuda::Host::DenseMatrix& mat
) {
    hsize_t dims[2] = {(hsize_t) mat.n_rows, (hsize_t) mat.n_cols};
    hid_t space = H5Screate_simple(2, dims, dims);
    hid_t ds = H5Dcreate(group_id, name.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data);
    H5Sclose(space);
    H5Dclose(ds);
    return ds;
}

// TODO: Remove later.
hid_t WriteMatrixToDataset(
    hid_t group_id,
    const std::string& name,
    MeMatrix& mat
) {
    hsize_t dims[2] = {(hsize_t) mat.n_rows, (hsize_t) mat.n_cols};
    hid_t space = H5Screate_simple(2, dims, dims);
    hid_t ds = H5Dcreate(group_id, name.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data);
    H5Sclose(space);
    H5Dclose(ds);
    return ds;
}


template <typename T>
hid_t WriteToDataset1D(
    hid_t group_id,
    const std::string& name,
    T* data,
    size_t n_elements,
    bool close = true
) {
    const int n_dims = 1;
    hsize_t dims[n_dims] = {(hsize_t) n_elements};
    hid_t space = H5Screate_simple(n_dims, dims, dims);
    hid_t ds = H5Dcreate(group_id, name.c_str(), TypeToH5<T>::type_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, TypeToH5<T>::type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Sclose(space);
    if (close) { H5Dclose(ds); }
    return ds;
}


template <typename T>
hid_t WriteAttributeScalar(
    hid_t id,
    const std::string& name,
    T data
) {
    hsize_t Sdims[0] = {};
    hid_t SS = H5Screate_simple(0, Sdims, Sdims);
    hid_t SA = H5Acreate(id, name.c_str(), TypeToH5<T>::type_id, SS, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(SA, TypeToH5<T>::type_id, &data);
    H5Sclose(SS);
    H5Aclose(SA);
    return SA;
}


template <typename T>
hid_t WriteAttribute1D(
    hid_t id,
    const std::string& name,
    T* data,
    size_t n_elements
) {
    hsize_t Sdims[1] = {(hsize_t) n_elements};
    hid_t SS = H5Screate_simple(1, Sdims, Sdims);
    hid_t SA = H5Acreate(id, name.c_str(), TypeToH5<T>::type_id, SS, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(SA, TypeToH5<T>::type_id, data);
    H5Sclose(SS);
    H5Aclose(SA);
    return SA;
}

} // H5Util
