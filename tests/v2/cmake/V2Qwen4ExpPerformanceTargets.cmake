# QFN performance executables shared by Integration and production Release.
#
# These declarations deliberately live outside the model-free preflight guard:
# functional fixtures and their registrations are Integration-only, while timing
# must use optimized Release objects. This module creates no CTest registrations
# and adds no performance executable to a Unit or preflight dependency.
include_guard(DIRECTORY)
set(_qfn_test_sources "${CMAKE_CURRENT_LIST_DIR}/..")

add_executable(v2_perf_qwen_sparse_selection EXCLUDE_FROM_ALL
    "${_qfn_test_sources}/performance/kernels/cpu/attention/Perf__QwenSparseSelection.cpp")
target_link_libraries(v2_perf_qwen_sparse_selection PRIVATE llaminar2_core GTest::gtest_main)

if(HAVE_ROCM)
    set_source_files_properties(
        "${_qfn_test_sources}/performance/kernels/rocm/Perf__FloatingReduction.hip"
        PROPERTIES LANGUAGE HIP)
    add_executable(v2_perf_rocm_floating_reduction EXCLUDE_FROM_ALL
        "${_qfn_test_sources}/performance/kernels/rocm/Perf__FloatingReduction.hip")
    target_link_libraries(v2_perf_rocm_floating_reduction PRIVATE
        llaminar2_core GTest::gtest GTest::gtest_main hip::host)
    set_target_properties(v2_perf_rocm_floating_reduction PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
endif()

# The same component supports CPU-only builds; CPU selection performs no GPU setup.
add_executable(v2_perf_q8_residual_prepared_projection
    "${_qfn_test_sources}/performance/kernels/Perf__Q8ResidualPreparedProjection.cpp"
    "${_qfn_test_sources}/performance/kernels/recorded/RecordedHeadProjection.cpp")
target_include_directories(v2_perf_q8_residual_prepared_projection PRIVATE "${_qfn_test_sources}")
target_link_libraries(v2_perf_q8_residual_prepared_projection PRIVATE llaminar2_core GTest::gtest)
if(HAVE_CUDA OR HAVE_ROCM)
    v2_test_device_scope(v2_perf_q8_residual_prepared_projection FullInventory)

    add_executable(v2_perf_immutable_adaptive_lookup
        "${_qfn_test_sources}/performance/kernels/qwen4exp/Measure__ImmutableAdaptiveLookup.cpp")
    target_include_directories(v2_perf_immutable_adaptive_lookup PRIVATE "${_qfn_test_sources}")
    target_link_libraries(v2_perf_immutable_adaptive_lookup PRIVATE llaminar2_core)
endif()

foreach(_qfn_backend IN ITEMS CUDA ROCm)
    if((_qfn_backend STREQUAL "CUDA" AND HAVE_CUDA) OR
       (_qfn_backend STREQUAL "ROCm" AND HAVE_ROCM))
        string(TOLOWER "${_qfn_backend}" _qfn_backend_directory)
        add_executable(v2_perf_${_qfn_backend_directory}_qwen_sparse_selection EXCLUDE_FROM_ALL
            "${_qfn_test_sources}/integration/kernels/${_qfn_backend_directory}/Test__${_qfn_backend}QwenSparseBoundedSelection.cpp")
        target_link_libraries(v2_perf_${_qfn_backend_directory}_qwen_sparse_selection PRIVATE
            llaminar2_core GTest::gtest_main)
    endif()
endforeach()
