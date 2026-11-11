# @file V2CPUJointStaticResidencyTests.cmake
# @brief Register Static joint-fragment source, ticket and publication lifecycles.
#
# The existing loader/registry fixture exercises original GGUF preparation,
# physical admission and immutable source-bound fragment lifetimes. These host
# lifecycle checks do not certify CPU2 serving, throughput or unfinished cohort migration.
if(TARGET v2_unit_expert_gemm_registry)
    add_v2_test(V2_Integration_OriginalPreparedSourceBindingLifecycle
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUPreparedSourceBindingContract.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;PreparedWeights;SourceBinding;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_OriginalPreparedSourceBindingLifecycle)

    add_v2_test(V2_Integration_CPUJointStaticResidencyLifecycle
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUJointStaticResidency.* --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;PreparedWeights;Residency;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointStaticResidencyLifecycle)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointStaticResidencyLifecycle)
    endif()
    add_v2_test(V2_Integration_CPUJointSourceAuthorityLifecycle
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUJointStaticResidency.ColdLoadGraphPreservesOriginalSourceAuthority:CPUJointStaticResidency.RetainedExecutorStartsWithOriginalSourceEngines --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;PreparedWeights;Residency;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointSourceAuthorityLifecycle)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointSourceAuthorityLifecycle)
    endif()
endif()

if(TARGET v2_unit_moe_overlay_participant_residency)
    # The real-source contract shares the canonical CPU admission/workspace fixtures.
    target_include_directories(v2_unit_moe_overlay_participant_residency PRIVATE "${CMAKE_CURRENT_LIST_DIR}/..")
    add_v2_test(V2_Integration_NonemptyPreparedSourceBankContract
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_participant_residency>
            --gtest_filter=PreparedSourceBindingContract.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;PreparedWeights;SourceBinding;Contract;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NonemptyPreparedSourceBankContract)
endif()

if(TARGET v2_integration_cross_tier_expert_arithmetic)
    # The actual native-stage contract borrows the existing test TP metadata
    # fixture; its mocks/ includes are rooted at tests/v2 for every owning build.
    target_include_directories(v2_integration_cross_tier_expert_arithmetic PRIVATE "${CMAKE_CURRENT_LIST_DIR}/..")
    foreach(backend CUDA ROCm)
        string(TOUPPER "${backend}" backend_upper)
        if(HAVE_${backend_upper})
            set(name "V2_Integration_${backend}_CPUJointPaddedPrefillContract")
            add_v2_test(${name}
                COMMAND $<TARGET_FILE:v2_integration_cross_tier_expert_arithmetic>
                    --gtest_filter=CPUJointPaddedPrefillContract.OriginalMappedTicket${backend} --gtest_brief=1
                LABELS "V2;Integration;ProductionTestPreflight;${backend};CPU;MoE;Ownership;Regression"
                MPI_PROCS 1 NO_MODELS TIMEOUT 60)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${name})
        endif()
    endforeach()
endif()

# Actual original residency/dispatch producers; recorder grants no completed native transport.
if(TARGET v2_unit_expert_gemm_registry)
add_v2_test(V2_Integration_CPUJointDispatchOwnershipContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUJointDispatchContract.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Source;Dispatch;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointDispatchOwnershipContract)
endif()

if(TARGET v2_integration_cross_tier_expert_arithmetic)
    foreach(backend CUDA ROCm)
        string(TOUPPER "${backend}" backend_upper)
        if(HAVE_${backend_upper})
            set(name "V2_Integration_${backend}_CPUJointSchedulingContextContract")
            add_v2_test(${name}
                COMMAND $<TARGET_FILE:v2_integration_cross_tier_expert_arithmetic>
                    --gtest_filter=CPUJointPaddedPrefillContract.OriginalMappedTicket${backend} --gtest_brief=1
                LABELS "V2;Integration;ProductionTestPreflight;${backend};CPU;MoE;Ownership;Regression"
                MPI_PROCS 1 NO_MODELS TIMEOUT 60)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${name})
        endif()
    endforeach()
endif()

if(TARGET v2_unit_expert_gemm_registry)
# Actual source-plan MTP completeness, before materialization/native readiness.
add_v2_test(V2_Integration_CPUJointMTPSourceContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUJointMTPSourceContract.*
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;MTP;Ownership;Contract"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointMTPSourceContract)

endif()

# Actual decodeStep exception unwinding must log the original failure before fatal MPI retirement.
if(TARGET v2_unit_prefill_decode_transition)
    add_v2_test(V2_Integration_MTPCommandExceptionDiagnostic
        COMMAND $<TARGET_FILE:v2_unit_prefill_decode_transition>
            --gtest_filter=Test__PrefillDecodeTransition.MTPCommandExceptionDiagnosticPrecedesFailureAbort
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MTP;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MTPCommandExceptionDiagnostic)
endif()

# Original source metadata remains valid after the bootstrap bank retires; packets own engine epochs.
if(TARGET v2_unit_expert_gemm_registry)
    add_v2_test(V2_Integration_CPUJointLateGraphSourceLifecycle
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUFragmentEpochBankContract.LateSourceOnlyGraphSurvivesBankRetirementAndExceptions
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;PreparedWeights;SourceBinding;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointLateGraphSourceLifecycle)
endif()

# Complete prepared weights establish the original cold source before either graph consumer runs.
if(TARGET v2_unit_expert_gemm_registry)
    add_v2_test(V2_Integration_CPUJointColdGraphAdmissionLifecycle
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUFragmentEpochBankContract.ColdGraphAdmissionBindsOriginalSourceBeforeDeclaration
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;PreparedWeights;SourceBinding;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointColdGraphAdmissionLifecycle)
endif()

# Actual layer sources remain distinct while mutually exclusive layers share checked physical route roots.
if(TARGET v2_unit_expert_gemm_registry)
    add_v2_test(V2_Integration_CPUJointSerialSourceGeometryLifecycle
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUFragmentEpochBankContract.DistinctLayerSourcesShareOriginalSerialGeometry
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;PreparedWeights;SourceBinding;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointSerialSourceGeometryLifecycle)
endif()
