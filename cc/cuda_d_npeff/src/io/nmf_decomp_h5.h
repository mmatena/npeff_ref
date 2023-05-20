#ifndef Z_AJDJKASJDJLKASLJD_8635C1ED_DA53_4DE2_AC5B_746D754FC49E_H
#define Z_AJDJKASJDJLKASLJD_8635C1ED_DA53_4DE2_AC5B_746D754FC49E_H

#include <iostream>
#include <string>
#include <type_traits>

#include <hdf5.h>

#include <util/cuda_statuses.h>
#include <util/h5_util.h>
#include <util/matrices.h>



template <typename IndT>
class NmfDecomposition {
public:

    // W and H are assumed to be in row-major format.

    // shape = [n_examples, n_components]
    MeMatrix* W = nullptr;
    // shape = [n_components, n_features]
    MeMatrix* H = nullptr;

    // Size = H.n_cols = n_features.
    IndT* indexToOgIndex = nullptr;

    IndT fullDenseSize = -1;

    // NmfDecomposition() {}

    ~NmfDecomposition() {
        // delete[] indexToOgIndex;
    }

    void save(const std::string& filepath) {
        hid_t file = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        hid_t dataG = H5Gcreate(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        hsize_t Wdims[2] = {(hsize_t) W->n_rows, (hsize_t) W->n_cols};
        hid_t WS = H5Screate_simple(2, Wdims, Wdims);
        hid_t WD = H5Dcreate(dataG, "W", H5T_NATIVE_FLOAT, WS, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(WD, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, W->data);
        H5Sclose(WS);
        H5Dclose(WD);

        hsize_t Hdims[2] = {(hsize_t) H->n_rows, (hsize_t) H->n_cols};
        hid_t HS = H5Screate_simple(2, Hdims, Hdims);
        hid_t HD = H5Dcreate(dataG, "H", H5T_NATIVE_FLOAT, HS, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(HD, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, H->data);
        H5Sclose(HS);
        H5Dclose(HD);


        hid_t h5IndType;
        if (std::is_same<IndT, int32_t>::value) h5IndType = H5T_NATIVE_INT32;
        else if(std::is_same<IndT, int64_t>::value) h5IndType = H5T_NATIVE_INT64;
        else {
            std::cout << "Invalid type.\n";
            THROW;
        }


        hsize_t Idims[1] = {(hsize_t) H->n_cols};
        hid_t IS = H5Screate_simple(1, Idims, Idims);
        hid_t ID = H5Dcreate(dataG, "reduce_kept_indices", h5IndType, IS, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ID, h5IndType, H5S_ALL, H5S_ALL, H5P_DEFAULT, indexToOgIndex);

        hsize_t Sdims[1] = {1};
        hid_t SS = H5Screate_simple(1, Sdims, Sdims);
        hid_t SA = H5Acreate(ID, "full_dense_size", h5IndType, SS, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(SA, h5IndType, &fullDenseSize);

        H5Sclose(SS);
        H5Aclose(SA);
        H5Sclose(IS);
        H5Dclose(ID);


        // data/W
        // data/H
        // data/reduce_kept_indices
        // reduce_kept_indices_ds.attrs['full_dense_size']
        H5Gclose(dataG);
        H5Fclose(file);
    }
    
    // TODO: Add option to only read in parts of the decomp, like only H or only W.
    static NmfDecomposition<IndT> read(const std::string& filepath);

};


// TODO: Add something like pefRequiresInt64Indices for nmf.


template <typename IndT>
NmfDecomposition<IndT> NmfDecomposition<IndT>::read(
    const std::string& filename
) {
    NmfDecomposition<IndT> nmf;

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    nmf.W = H5Util::readDatasetToMatrix(file, "data/W");
    nmf.H = H5Util::readDatasetToMatrix(file, "data/H");

    // Needed since we store in row major format but use column major format in our code.
    // The function says toRowMajor, but calling it on a row-major matrix makes one in
    // column major format.
    nmf.W->toRowMajor_inPlace();
    nmf.H->toRowMajor_inPlace();

    nmf.indexToOgIndex = H5Util::readDatasetToPtr<IndT>(file, "data/reduce_kept_indices");

    hid_t dfsD = H5Dopen(file, "data/reduce_kept_indices", H5P_DEFAULT);
    H5Util::readAttribute(dfsD, "full_dense_size", &nmf.fullDenseSize);
    H5Dclose(dfsD);

    H5Fclose(file);

    return nmf;
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


template <typename IndT>
class SparseNmfDecomposition {
public:

    // W and H are assumed to be in row-major format.

    // shape = [n_examples, n_components]
    MeMatrix* W = nullptr;
    // shape = [n_components, n_features]
    ElCsrMatrix<IndT>* H = nullptr;

    // Size = H.n_cols = n_features.
    IndT* indexToOgIndex = nullptr;

    IndT fullDenseSize = -1;

    void save(const std::string& filepath) {
        hid_t file = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataG = H5Gcreate(file, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        H5Util::WriteMatrixToDataset(dataG, "W", *W);

        hid_t HG = H5Gcreate(dataG, "H", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Util::WriteToDataset1D(HG, "values", H->csrValA, H->nnz);
        H5Util::WriteToDataset1D(HG, "row_indices", H->csrRowPtrA, H->n_rows + 1);
        H5Util::WriteToDataset1D(HG, "column_indices", H->csrColIndA, H->nnz);
        long H_shape[2] = {H->n_rows, H->n_cols};
        H5Util::WriteAttribute1D(HG, "shape", H_shape, 2);
        H5Gclose(HG);

        hid_t dfsD = H5Util::WriteToDataset1D(dataG, "reduce_kept_indices", indexToOgIndex, H->n_cols, false);
        H5Util::WriteAttributeScalar(dfsD, "full_dense_size", fullDenseSize);
        H5Dclose(dfsD);

        H5Gclose(dataG);
        H5Fclose(file);
    }

    // TODO: Add option to only read in parts of the decomp, like only H or only W.
    static SparseNmfDecomposition<IndT> read(const std::string& filepath);
};



template <typename IndT>
ElCsrMatrix<IndT>* _readSparseHFromH5(hid_t file) {
    // Read in the group "H" "shape" attr.
    // Read in the size of H_values to get nnz.
    long shape[2];
    hsize_t nnz;

    hid_t hG = H5Gopen(file, "data/H", H5P_DEFAULT);
    H5Util::readAttribute(hG, "shape", shape);
    H5Gclose(hG);

    hid_t valuesD = H5Dopen(file, "data/H/values", H5P_DEFAULT);
    hid_t valuesS = H5Dget_space(valuesD);
    H5Sget_simple_extent_dims(valuesS, &nnz, NULL);
    H5Sclose(valuesS);
    H5Dclose(valuesD);

    float* values = H5Util::readDatasetToPtr<float>(file, "data/H/values");
    IndT* rowIndices = H5Util::readDatasetToPtr<IndT>(file, "data/H/row_indices");
    IndT* colIndices = H5Util::readDatasetToPtr<IndT>(file, "data/H/column_indices");

    auto ret = new ElCsrMatrix<IndT>(shape[0], shape[1], nnz);
    std::memcpy(ret->csrValA, values, sizeof(float) * nnz);
    std::memcpy(ret->csrRowPtrA, rowIndices, sizeof(IndT) * (ret->n_rows + 1));
    std::memcpy(ret->csrColIndA, colIndices, sizeof(IndT) * nnz);

    free(values);
    free(rowIndices);
    free(colIndices);

    return ret;
}


template <typename IndT>
SparseNmfDecomposition<IndT> SparseNmfDecomposition<IndT>::read(
    const std::string& filename
) {
    SparseNmfDecomposition<IndT> nmf;

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    nmf.W = H5Util::readDatasetToMatrix(file, "data/W");
    nmf.H = _readSparseHFromH5<IndT>(file);

    // Needed since we store in row major format but use column major format in our code.
    // The function says toRowMajor, but calling it on a row-major matrix makes one in
    // column major format.
    nmf.W->toRowMajor_inPlace();

    nmf.indexToOgIndex = H5Util::readDatasetToPtr<IndT>(file, "data/reduce_kept_indices");

    hid_t dfsD = H5Dopen(file, "data/reduce_kept_indices", H5P_DEFAULT);
    H5Util::readAttribute(dfsD, "full_dense_size", &nmf.fullDenseSize);
    H5Dclose(dfsD);

    H5Fclose(file);

    return nmf;
}



#endif
