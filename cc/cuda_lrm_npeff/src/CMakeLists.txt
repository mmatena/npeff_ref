
add_subdirectory(containers)
add_subdirectory(preprocess)
add_subdirectory(util)

add_library(
    m_npeff
    STATIC
    coeff_fitting/host_context.cc
    coeff_fitting/config.cc
    containers/dense_matrix.h
    containers/sparse_matrix.h
    factorizations/lvrm_coeff_fitting/config.cc
    factorizations/lvrm_coeff_fitting/host_context.cc
    gpu/containers/dense_matrix.cc
    gpu/contexts/device_context.cc
    gpu/ops/dndn_matmul.cc
    inputs/lrm_pefs.h
    inputs/lrm_pefs.cc
    inputs/lvrm_pefs.h
    inputs/lvrm_pefs.cc
    inputs/lrm_npeff_decomposition.h
    inputs/lrm_npeff_decomposition.cc
    outputs/lrm_npeff_decomposition.cc
    preprocess/construct_csr_matrix.h
    preprocess/construct_csr_matrix.cc
    preprocess/column_pruning.h
    util/array_util.h
    util/h5_util.h
    util/h5_util.cc
    #
    # Dummy files, just to get sublime text to work properly.
    expansion/dummy.cc
    factorization/dummy.cc
    factorizations/lvrm_factorization/dummy.cc
    factorizations/stiefel/dummy.cc
    factorizations/util/dummy.cc
    gpu/ops/custom/dummy.cc
    gpu/ops/debugging/dummy.cc
    outputs/dummy.cc
)
target_include_directories(m_npeff PUBLIC ../src)

# Set up the hdf5 libraries.
target_include_directories(m_npeff PUBLIC ${HDF5_INCLUDE_DIR})
target_link_directories(m_npeff PUBLIC ${HDF5_LIB_DIR})
target_link_libraries(m_npeff PUBLIC hdf5)
target_link_libraries(m_npeff PUBLIC hdf5_cpp)

# Set up the CUDA-related libraries.
target_include_directories(m_npeff PUBLIC ${CUDA_INCLUDE_DIR})
target_link_directories(m_npeff PUBLIC ${CUDA_LIB_DIR})
target_link_libraries(m_npeff PUBLIC nccl)
target_link_libraries(m_npeff PUBLIC cublas)
target_link_libraries(m_npeff PUBLIC curand)
target_link_libraries(m_npeff PUBLIC cusparse)
target_link_libraries(m_npeff PUBLIC cusolver)
