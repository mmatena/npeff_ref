#ifndef Z8635C1ED_DA53_4DE2_AC5B_746D754FC49E_H
#define Z8635C1ED_DA53_4DE2_AC5B_746D754FC49E_H

#include <iostream>
#include <string>
#include <type_traits>

#include <hdf5.h>

#include <util/cuda_statuses.h>
#include <util/matrices.h>




template <typename IndT>
class PefSparseFishers {
    // Sparse per-example fishers information.
public:
    ElCsrMatrix<IndT>* fishers = nullptr;

    // Length = n_examples.
    float* denseFisherNorms = nullptr;

    long fisherDenseSize = -1;

    int n_examples = -1;
    int n_valuesPerExample = -1;

    ~PefSparseFishers() {
        delete fishers;
        delete[] denseFisherNorms;
    }

    static PefSparseFishers<IndT> read(const std::string& filename, int n_examples = -1, int n_fisherValues = -1);
    static void read(PefSparseFishers<IndT>* pef, const std::string& filename, int n_examples = -1, int n_fisherValues = -1);

    void normalizeFishers_inPlace(float eps = 1e-9) {
        const int n_rows = fishers->n_rows;
        const IndT* csrRowPtrA = fishers->csrRowPtrA;
        float* csrValA = fishers->csrValA;

        for (long rowIndex = 0; rowIndex < n_rows; rowIndex++) {
            IndT start = csrRowPtrA[rowIndex];
            IndT end = csrRowPtrA[rowIndex + 1];

            float exampleNorm = denseFisherNorms[rowIndex] + eps;
            for (long j = start; j < end; j++) {
                csrValA[j] /= exampleNorm;
            }
        }
    }

};

namespace Pef {
bool pefRequiresInt64Indices(const std::string& filename, int n_examples = -1, int n_fisherValues = -1) {
    // Negative values for hid_t means that the operation was invalid.
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    hid_t valuesD = H5Dopen(file, "data/fisher/values", H5P_DEFAULT);
    hid_t valuesS = H5Dget_space(valuesD);

    int valuesRank = H5Sget_simple_extent_ndims(valuesS);
    THROWSERT(valuesRank == 2);

    hsize_t dims[2];
    H5Sget_simple_extent_dims(valuesS, dims, NULL);

    H5Sclose(valuesS);
    H5Dclose(valuesD);
    H5Fclose(file);

    long n_rows = dims[0];
    long n_valuesPerRow = dims[1];

    if (n_examples > 0) {
        THROWSERT(n_examples <= n_rows);
        n_rows = n_examples;
    }

    if (n_fisherValues > 0) {
        THROWSERT(n_fisherValues <= n_valuesPerRow);
        n_valuesPerRow = n_fisherValues;
    }

    long nnz = (long) n_rows * (long) n_valuesPerRow;

    return (nnz > INT32_MAX);
}

}

template <typename IndT>
PefSparseFishers<IndT> PefSparseFishers<IndT>::read(
    const std::string& filename,
    int n_examples,
    int n_fisherValues
) {
    PefSparseFishers pef;
    PefSparseFishers<IndT>::read(&pef, filename, n_examples, n_fisherValues);
    return pef;
}


template <typename IndT>
void PefSparseFishers<IndT>::read(
    PefSparseFishers<IndT>* pef,
    const std::string& filename,
    int n_examples,
    int n_fisherValues
) {
    // Negative values for hid_t means that the operation was invalid.
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    hid_t valuesD = H5Dopen(file, "data/fisher/values", H5P_DEFAULT);
    hid_t valuesS = H5Dget_space(valuesD);

    int valuesRank = H5Sget_simple_extent_ndims(valuesS);
    THROWSERT(valuesRank == 2);

    hsize_t dims[2];
    H5Sget_simple_extent_dims(valuesS, dims, NULL);

    // TODO: Support values greater than MAX_INT.
    long n_rows = dims[0];
    long n_valuesPerRow = dims[1];

    if (n_examples > 0) {
        THROWSERT(n_examples <= n_rows);
        n_rows = n_examples;
    }

    if (n_fisherValues > 0) {        
        THROWSERT(n_fisherValues <= n_valuesPerRow);
        n_valuesPerRow = n_fisherValues;
    }

    long nnz = (long) n_rows * (long) n_valuesPerRow;

    // Read in the size of the dense per-example Fisher, which becomes
    // the number of columns in the sparse fishers matrix.

    hid_t fisherG = H5Gopen(file, "data/fisher", H5P_DEFAULT);
    hid_t dfsA = H5Aopen_name(fisherG, "dense_fisher_size");

    hid_t dfsT = H5Aget_type(dfsA);
    size_t dfsTypeSize = H5Tget_size(dfsT);

    // TODO: Can this be a long sometimes, if the value is large enough?
    THROWSERT(dfsTypeSize == 4);

    int n_cols;
    H5Aread(dfsA, H5T_NATIVE_INT, &n_cols);

    H5Tclose(dfsT);
    H5Aclose(dfsA);
    H5Gclose(fisherG);


    // Create the pef object and read in the data.

    // PefSparseFishers pef;
    pef->fisherDenseSize = n_cols;
    pef->n_examples = n_rows;
    pef->n_valuesPerExample = n_valuesPerRow;

    pef->denseFisherNorms = new float[n_rows];


    // NOTE: The column indices might not be sorted here. IDK if anything assumes that the
    // column indices in a row are sorted.
    pef->fishers = new ElCsrMatrix<IndT>(n_rows, n_cols, nnz);

    for (long i=0; i < n_rows + 1; i++) {
        pef->fishers->csrRowPtrA[i] = i * n_valuesPerRow;
    }

    hid_t memoryType;
    if (std::is_same<IndT, int32_t>::value) {
        memoryType = H5T_NATIVE_INT32;

    } else if(std::is_same<IndT, int64_t>::value) {
        memoryType = H5T_NATIVE_INT64;

    } else {
        std::cout << "Invalid type.\n";
        THROW;
    }


    hsize_t fishersHyperslabStart[2] = {(hsize_t) 0, (hsize_t) 0};
    hsize_t fishersHyperslabCount[2] = {(hsize_t) n_rows, (hsize_t) n_valuesPerRow};
    hid_t fishersMemSpace = H5Screate_simple(2, fishersHyperslabCount, fishersHyperslabCount);


    H5Sselect_hyperslab(valuesS, H5S_SELECT_SET, fishersHyperslabStart, NULL, fishersHyperslabCount, NULL);
    H5Dread(valuesD, H5T_NATIVE_FLOAT, fishersMemSpace, valuesS, H5P_DEFAULT, pef->fishers->csrValA);
    H5Sclose(valuesS);
    H5Dclose(valuesD);

    hid_t indicesD = H5Dopen(file, "data/fisher/indices", H5P_DEFAULT);
    hid_t indicesS = H5Dget_space(indicesD);
    H5Sselect_hyperslab(indicesS, H5S_SELECT_SET, fishersHyperslabStart, NULL, fishersHyperslabCount, NULL);
    // TODO: Can these be sometimes longs?
    H5Dread(indicesD, memoryType, fishersMemSpace, indicesS, H5P_DEFAULT, pef->fishers->csrColIndA);
    H5Sclose(indicesS);
    H5Sclose(fishersMemSpace);
    H5Dclose(indicesD);


    hsize_t normsHyperslabStart[1] = {0};
    hsize_t normsHyperslabCount[1] = {(hsize_t) n_rows};
    hid_t normsMemSpace = H5Screate_simple(1, normsHyperslabCount, normsHyperslabCount);

    hid_t normsD = H5Dopen(file, "data/dense_fisher_norms", H5P_DEFAULT);
    hid_t normsS = H5Dget_space(normsD);
    H5Sselect_hyperslab(normsS, H5S_SELECT_SET, normsHyperslabStart, NULL, normsHyperslabCount, NULL);
    H5Dread(normsD, H5T_NATIVE_FLOAT, normsMemSpace, normsS, H5P_DEFAULT, pef->denseFisherNorms);
    H5Sclose(normsMemSpace);
    H5Sclose(normsS);
    H5Dclose(normsD);


    H5Fclose(file);
}


#endif
