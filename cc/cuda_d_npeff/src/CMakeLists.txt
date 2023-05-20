
add_library(
    d_npeff
    STATIC
)
target_include_directories(d_npeff PUBLIC ../src)

# Set up the hdf5 libraries.
target_include_directories(d_npeff PUBLIC ${HDF5_INCLUDE_DIR})
target_link_directories(d_npeff PUBLIC ${HDF5_LIB_DIR})
target_link_libraries(d_npeff PUBLIC hdf5)
target_link_libraries(d_npeff PUBLIC hdf5_cpp)

# Set up the CUDA-related libraries.
target_include_directories(d_npeff PUBLIC ${CUDA_INCLUDE_DIR})
target_link_directories(d_npeff PUBLIC ${CUDA_LIB_DIR})
target_link_libraries(d_npeff PUBLIC nccl)
target_link_libraries(d_npeff PUBLIC cublas)
target_link_libraries(d_npeff PUBLIC curand)
target_link_libraries(d_npeff PUBLIC cusparse)
target_link_libraries(d_npeff PUBLIC cusolver)
