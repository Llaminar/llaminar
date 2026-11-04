# @file V2CPUJointStaticResidencyTests.cmake
# @brief Register Static joint-fragment source, ticket and publication lifecycles.
#
# The existing loader/registry fixture exercises original GGUF preparation,
# physical admission and immutable source-bound fragment lifetimes. These host
# lifecycle checks do not certify CPU2 serving, throughput or unfinished cohort migration.
if(TARGET v2_unit_expert_gemm_registry)
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
            --gtest_filter=CPUJointStaticResidency.ColdLoadGraphPreservesOriginalSourceAuthority --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;PreparedWeights;Residency;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_CPUJointSourceAuthorityLifecycle)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointSourceAuthorityLifecycle)
    endif()
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
