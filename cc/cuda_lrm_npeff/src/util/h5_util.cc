#include <util/h5_util.h>

namespace npeff {
namespace util {
namespace h5 {


// Add more specializations here if needed.
template<> hid_t TypeId<int32_t>::id {H5T_NATIVE_INT32};
template<> hid_t TypeId<int64_t>::id {H5T_NATIVE_INT64};
template<> hid_t TypeId<float>::id {H5T_NATIVE_FLOAT};


std::vector<int64_t> read_dataset_dims(hid_t file, const std::string& ds_name) {
    hid_t valuesD = H5Dopen(file, ds_name.c_str(), H5P_DEFAULT);
    hid_t valuesS = H5Dget_space(valuesD);

    int rank = H5Sget_simple_extent_ndims(valuesS);
    hsize_t dims[rank];
    H5Sget_simple_extent_dims(valuesS, dims, NULL);

    H5Sclose(valuesS);
    H5Dclose(valuesD);

    return std::vector<int64_t>(dims, dims + rank);
}


}  // h5
}  // util
}  // npeff
