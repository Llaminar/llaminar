#[=[
@file V2NativePayloadFusedTests.cmake
@brief Give each fused native-payload fixture case one execution owner per ISA.

Focused preflight registrations retain their worker and lifecycle contracts.
Only enabled registrations contribute exclusions to the residual suite, so
architecture-specific focused cases cannot create gaps on another backend.
The device-free CMake/CTest regression checks the emitted filter partition.
]=]
foreach(V2_FUSED_COMPLETION_ISA IN LISTS V2_CPU_NATIVE_ISA_CELLS)
    set(V2_FUSED_OWNED_FILTERS)
    v2_cpu_native_isa_runtime("${V2_FUSED_COMPLETION_ISA}" V2_FUSED_COMPLETION_RUNTIME_ISA)
    set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.ProjectionEvidenceUsesGeometryAndPolicyInsteadOfBundleOrdinal")
    add_v2_test(V2_Integration_CPU_NativeQuantisedFusedProjectionEvidenceIdentity_${V2_FUSED_COMPLETION_ISA}
        COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
            --gtest_filter=${V2_FUSED_CASE_FILTER}
        LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;PerfStats;Economy;Lifecycle;${V2_FUSED_COMPLETION_ISA};ByteExact;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=2")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedFusedProjectionEvidenceIdentity_${V2_FUSED_COMPLETION_ISA})
    list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.GPUAlignedOrderedNibbleAllFormatReplay")
    add_v2_test(V2_Integration_CPU_NativeQuantisedOrderedNibbleReplay_${V2_FUSED_COMPLETION_ISA}
        COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
            --gtest_filter=${V2_FUSED_CASE_FILTER}
        LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;MoE;Prefill;Decode;${V2_FUSED_COMPLETION_ISA};ByteExact;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=4" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedOrderedNibbleReplay_${V2_FUSED_COMPLETION_ISA})
    list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    # x86 fused-rows planner schedule; ARM executes its own NEON tiles.
    if(V2_CPU_TARGET_X86)
        set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.GPUAlignedOrderedThreeRowNibbleByteLifecycle")
        add_v2_test(V2_Integration_CPU_NativeQuantisedOrderedThreeRowNibble_${V2_FUSED_COMPLETION_ISA}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=${V2_FUSED_CASE_FILTER}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;MoE;MTP;Prefill;${V2_FUSED_COMPLETION_ISA};ByteExact;Lifecycle;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=4" "OMP_DYNAMIC=false")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedOrderedThreeRowNibble_${V2_FUSED_COMPLETION_ISA})
        list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    endif()
    # x86 fused-rows planner schedule; ARM executes its own NEON tiles.
    if(V2_CPU_TARGET_X86)
        set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.GPUAlignedPrefillOrderedThreeRowTile*")
        add_v2_test(V2_Integration_CPU_NativeQuantisedPrefillOrderedThreeTile_${V2_FUSED_COMPLETION_ISA}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=${V2_FUSED_CASE_FILTER}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;MoE;Prefill;${V2_FUSED_COMPLETION_ISA};ByteExact;Lifecycle;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=4" "OMP_DYNAMIC=false")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedPrefillOrderedThreeTile_${V2_FUSED_COMPLETION_ISA})
        list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    endif()
    set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.UnsignedCoefficientDomainContract:CPUNativeQuantisedFusedCompletion.UnsignedWeightDotAllFormatByteLifecycle")
    add_v2_test(V2_Integration_CPU_NativeQuantisedUnsignedWeightDot_${V2_FUSED_COMPLETION_ISA}
        COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
            --gtest_filter=${V2_FUSED_CASE_FILTER}
        LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;Lifecycle;${V2_FUSED_COMPLETION_ISA};ByteExact;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=4" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedUnsignedWeightDot_${V2_FUSED_COMPLETION_ISA})
    list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    # x86 fused-rows planner schedule; ARM executes its own NEON tiles.
    if(V2_CPU_TARGET_X86)
        set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.PairwiseContiguousOwnership*")
        add_v2_test(V2_Integration_CPU_NativeQuantisedPairwiseContiguousOwnership_${V2_FUSED_COMPLETION_ISA}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=${V2_FUSED_CASE_FILTER}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;MoE;Prefill;${V2_FUSED_COMPLETION_ISA};ByteExact;Lifecycle;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=7" "OMP_DYNAMIC=false")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedPairwiseContiguousOwnership_${V2_FUSED_COMPLETION_ISA})
        list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    endif()
    # x86 fused-rows planner schedule; ARM executes its own NEON tiles.
    if(V2_CPU_TARGET_X86)
        set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.PrefillContiguousOwnershipMixedPoliciesAllFormatLifecycle:CPUNativeQuantisedFusedCompletion.OutputWorkerStaticRangeContract:CPUNativeQuantisedFusedCompletion.PrefillPanelBlocksMixedPoliciesAllFormatLifecycle:CPUNativeQuantisedFusedCompletion.AutoPanelBlocksContract:CPUNativeQuantisedFusedCompletion.PrefillAutoPanelBlocksMixedPoliciesAllFormatLifecycle")
        add_v2_test(V2_Integration_CPU_NativeQuantisedPrefillContiguousMixedPolicies_${V2_FUSED_COMPLETION_ISA}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=${V2_FUSED_CASE_FILTER}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;MoE;Prefill;${V2_FUSED_COMPLETION_ISA};ByteExact;Lifecycle;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=7" "OMP_DYNAMIC=false")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedPrefillContiguousMixedPolicies_${V2_FUSED_COMPLETION_ISA})
        list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    endif()
    # x86 fused-rows planner schedule; ARM executes its own NEON tiles.
    if(V2_CPU_TARGET_X86)
        set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.GPUAlignedPrefillSamePanelPairRunReplay")
        add_v2_test(V2_Integration_CPU_NativeQuantisedSamePanelPairRunReplay_${V2_FUSED_COMPLETION_ISA}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=${V2_FUSED_CASE_FILTER}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;MoE;Prefill;${V2_FUSED_COMPLETION_ISA};ByteExact;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=7" "OMP_DYNAMIC=false")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedSamePanelPairRunReplay_${V2_FUSED_COMPLETION_ISA})
        list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    endif()
    # x86 fused-rows planner schedule; ARM executes its own NEON tiles.
    if(V2_CPU_TARGET_X86)
        set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.LivePlanFillValueAndPrefixReset")
        add_v2_test(V2_Integration_CPU_NativeQuantisedLivePlanFill_${V2_FUSED_COMPLETION_ISA}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=${V2_FUSED_CASE_FILTER}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;Lifecycle;${V2_FUSED_COMPLETION_ISA};ByteExact;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=7" "OMP_DYNAMIC=false")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedLivePlanFill_${V2_FUSED_COMPLETION_ISA})
        list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    endif()
    # x86 fused-rows planner schedule; ARM executes its own NEON tiles.
    if(V2_CPU_TARGET_X86)
        set(V2_FUSED_CASE_FILTER "CPUNativeQuantisedFusedCompletion.LivePrefixStorageLargeSmallReset")
        add_v2_test(V2_Integration_CPU_NativeQuantisedFusedLivePrefix_${V2_FUSED_COMPLETION_ISA}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=${V2_FUSED_CASE_FILTER}
            LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;AllFormats;Lifecycle;${V2_FUSED_COMPLETION_ISA};ByteExact;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}" "OMP_NUM_THREADS=7" "OMP_DYNAMIC=false")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedFusedLivePrefix_${V2_FUSED_COMPLETION_ISA})
        list(APPEND V2_FUSED_OWNED_FILTERS "${V2_FUSED_CASE_FILTER}")
    endif()
    list(JOIN V2_FUSED_OWNED_FILTERS ":" V2_FUSED_OWNED_FILTER)
    add_v2_test(V2_Integration_CPU_NativeQuantisedFusedCompletion_${V2_FUSED_COMPLETION_ISA}
        COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
            --gtest_filter=-${V2_FUSED_OWNED_FILTER}
            --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;CPU;NativeQuantised;MTP;AllFormats;${V2_FUSED_COMPLETION_ISA};ByteExact;Regression"
        MPI_PROCS 1
        NO_MODELS
        ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_FUSED_COMPLETION_RUNTIME_ISA}")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPU_NativeQuantisedFusedCompletion_${V2_FUSED_COMPLETION_ISA})
endforeach()
