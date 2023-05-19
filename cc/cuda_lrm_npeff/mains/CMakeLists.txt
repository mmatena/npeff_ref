
add_executable(run_m_npeff run_m_npeff.cu)
target_link_libraries(run_m_npeff PRIVATE m_npeff)
target_link_libraries(run_m_npeff PRIVATE gflags::gflags)
target_compile_options(run_m_npeff PRIVATE -O3)

add_executable(fit_m_npeff_coeffs fit_m_npeff_coeffs.cu)
target_link_libraries(fit_m_npeff_coeffs PRIVATE m_npeff)
target_link_libraries(fit_m_npeff_coeffs PRIVATE gflags::gflags)
target_compile_options(fit_m_npeff_coeffs PRIVATE -O3)

add_executable(run_m_npeff_expansion run_m_npeff_expansion.cu)
target_link_libraries(run_m_npeff_expansion PRIVATE m_npeff)
target_link_libraries(run_m_npeff_expansion PRIVATE gflags::gflags)
target_compile_options(run_m_npeff_expansion PRIVATE -O3)

add_executable(run_stiefel_m_npeff run_stiefel_m_npeff.cu)
target_link_libraries(run_stiefel_m_npeff PRIVATE m_npeff)
target_link_libraries(run_stiefel_m_npeff PRIVATE gflags::gflags)
target_compile_options(run_stiefel_m_npeff PRIVATE -O3)

add_executable(run_lvrm_npeff run_lvrm_npeff.cu)
target_link_libraries(run_lvrm_npeff PRIVATE m_npeff)
target_link_libraries(run_lvrm_npeff PRIVATE gflags::gflags)
target_compile_options(run_lvrm_npeff PRIVATE -O3)

add_executable(fit_lvrm_coeffs fit_lvrm_coeffs.cu)
target_link_libraries(fit_lvrm_coeffs PRIVATE m_npeff)
target_link_libraries(fit_lvrm_coeffs PRIVATE gflags::gflags)
target_compile_options(fit_lvrm_coeffs PRIVATE -O3)
