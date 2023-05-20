
add_executable(run_nmf_on_pefs run_nmf_on_pefs.cu)
target_link_libraries(run_nmf_on_pefs PRIVATE m_npeff)
target_link_libraries(run_nmf_on_pefs PRIVATE gflags::gflags)
target_compile_options(run_nmf_on_pefs PRIVATE -O3)

add_executable(fit_coeffs_to_sparse_H fit_coeffs_to_sparse_H.cu)
target_link_libraries(fit_coeffs_to_sparse_H PRIVATE m_npeff)
target_link_libraries(fit_coeffs_to_sparse_H PRIVATE gflags::gflags)
target_compile_options(fit_coeffs_to_sparse_H PRIVATE -O3)
