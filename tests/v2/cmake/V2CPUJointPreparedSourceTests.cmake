# @file V2CPUJointPreparedSourceTests.cmake
# @brief Register original-source CPU preparation contracts and loading lifecycles.
#
# The existing loader registry target owns tensor/view and prepared-engine
# fixtures. These device-free checks preserve original E/member/source identity;
# they do not qualify two-socket serving or native collective execution.
if(TARGET v2_unit_expert_gemm_registry)
    add_v2_test(V2_Integration_CPUJointPreparedSourceContract
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUJointPreparedSource.* --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;PreparedWeights;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointPreparedSourceContract)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointPreparedSourceContract)
    endif()
    add_v2_test(V2_Integration_CPUJointBoundedSourceLifecycle
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUJointPreparedSource.ActualLoaderRetainsOriginalMapAndSelectedNativeBytes:CPUJointPreparedSource.BoundedWindowsRetainSparseOriginalIdsAndNativeCuts --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;PreparedWeights;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointBoundedSourceLifecycle)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointBoundedSourceLifecycle)
    endif()
endif()

if(TARGET v2_test_model_loader_row_slice)
    add_v2_test(V2_Integration_CPUJointBoundedNativeFormatWindows
        COMMAND $<TARGET_FILE:v2_test_model_loader_row_slice>
            --gtest_filter=*ModelLoaderExpertSelectionTest.CPUJointBoundedNativeWindowsRetainOriginalEncodedBytes/* --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;PreparedWeights;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointBoundedNativeFormatWindows)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointBoundedNativeFormatWindows)
    endif()
endif()

if(TARGET v2_test_model_loader_row_slice)
    add_v2_test(V2_Integration_CPUJointBufferedSourceFailureLifecycle
        COMMAND $<TARGET_FILE:v2_test_model_loader_row_slice>
            --gtest_filter=CPUJointBufferedSource.* --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;PreparedWeights;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointBufferedSourceFailureLifecycle)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointBufferedSourceFailureLifecycle)
    endif()
endif()

if(TARGET v2_test_model_loader_row_slice)
    add_v2_test(V2_Integration_CPUJointBufferedSourceEconomy
        COMMAND $<TARGET_FILE:v2_test_model_loader_row_slice>
            --gtest_filter=CPUJointBufferedSource.RealisticStridedReadUsesBufferedSyscallsAndExactBytes --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;SourceLoading;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointBufferedSourceEconomy)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointBufferedSourceEconomy)
    endif()
endif()
