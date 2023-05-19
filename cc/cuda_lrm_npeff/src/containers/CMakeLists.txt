
add_executable(
    containers_test
    conversions_test.cc
    dense_matrix_test.cc
    sparse_matrix_test.cc
)
target_link_libraries(containers_test PRIVATE Catch2::Catch2WithMain)
target_link_libraries(containers_test PRIVATE m_npeff)

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.
add_test(NAME containers_test COMMAND containers_test)

