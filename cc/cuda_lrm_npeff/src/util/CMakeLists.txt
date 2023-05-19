
add_executable(array_util_test array_util_test.cc)
target_link_libraries(array_util_test PRIVATE Catch2::Catch2WithMain)
target_link_libraries(array_util_test PRIVATE m_npeff)

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.
add_test(NAME array_util_test COMMAND array_util_test)

