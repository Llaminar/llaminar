# @file V2PooledImmutableLocalHotTests.cmake
# @brief Explicit actual local-cache controller replay; physical/native qualification is separate.
if(TARGET v2_test_qwen4exp_primitives)
    target_sources(v2_test_qwen4exp_primitives PRIVATE
        unit/kernels/qwen4exp/Test__PooledImmutableLocalHot.cpp
        unit/kernels/qwen4exp/Test__PooledImmutableLocalHotRequestPublication.cpp)
    add_v2_test(V2_Integration_PooledImmutableLocalHotOpsReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives> --gtest_filter=PooledImmutableLocalHot.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Pooled;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableLocalHotOpsReplay)
    add_v2_test(V2_Integration_PooledImmutableLocalHotRequestPublication
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableLocalHotRequestPublication.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Pooled;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableLocalHotRequestPublication)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/V2LocalHotNativeForkTests.cmake")

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledImmutableLocalHotForwardControllerReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableLocalHot.OriginalForwardControllerAdmission --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Pooled;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableLocalHotForwardControllerReplay)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/V2PooledLocalHotNativeTests.cmake")

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledImmutableLocalHotTerminalReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableLocalHot.Terminal* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Pooled;Terminal;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableLocalHotTerminalReplay)
endif()

# Exact original planner admission/reuse; no accelerator or model payload is loaded.
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_PooledLocalHotTerminalHostAdmission
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupAdmission.LocalTerminalHostBOMIsOncePerConsumerAcrossResolvedReuse
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;Ngram;Planning;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledLocalHotTerminalHostAdmission)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledImmutableLocalHotReadAheadPolicyReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableLocalHot.FutureMissDiscardPreservesDemandPolicyAndNativeBytes:PooledImmutableLocalHot.FutureLateMissAdoptionCreditsActualDemandExactlyOnce:PooledImmutableLocalHot.RecordedConsumedScanPreservesHotDecodeRow
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Pooled;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableLocalHotReadAheadPolicyReplay)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledImmutableLocalHotActualDemandAdoptionReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableLocalHot.FutureActualDemand*:PooledImmutableLocalHot.FuturePredictionMissLeavesPendingLifetimeAndPolicyUncredited
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Pooled;Lifecycle;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableLocalHotActualDemandAdoptionReplay)
endif()
