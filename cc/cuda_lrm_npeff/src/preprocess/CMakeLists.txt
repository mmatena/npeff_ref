
add_executable(
    preprocess_test
    column_pruning_test.cc
    construct_csr_matrix_test.cc
)
target_link_libraries(preprocess_test PRIVATE Catch2::Catch2WithMain)
target_link_libraries(preprocess_test PRIVATE m_npeff)

add_test(NAME preprocess_test COMMAND preprocess_test)

