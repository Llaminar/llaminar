# @file V2CPUJointGraphOwnershipTests.cmake
# @brief Verify original source-map reuse and external joint routing contracts.
#
# The existing complete Unit loader target includes these same cases. Explicit
# production-preflight entries preserve the two defects as focused regressions.
# Native CPU/NUMA admission and inference remain separate integration obligations.
add_executable(v2_unit_moe_overlay_joint_route_target
    unit/execution/moe/Test__MoEOverlayJointRouteTarget.cpp)
target_include_directories(v2_unit_moe_overlay_joint_route_target PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(v2_unit_moe_overlay_joint_route_target PRIVATE cxx_std_23)
target_link_libraries(v2_unit_moe_overlay_joint_route_target PRIVATE GTest::gtest_main)
add_v2_test(V2_Unit_MoEOverlayJointTargetMembership
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_joint_route_target>
    LABELS "V2;Unit;DeviceFree;MoE;Graph;Contract;Regression"
    NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_MoEOverlayJointTargetMembership
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_joint_route_target>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;Graph;Contract;Regression"
    NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30)
if(TEST V2_Integration_MoEOverlayJointTargetMembership)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayJointTargetMembership)
endif()
if(TARGET v2_unit_expert_gemm_registry)
    target_include_directories(v2_unit_expert_gemm_registry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    foreach(_cpu_joint_projection_case IN ITEMS
        ExternalJointProjectionPreservesOriginalMembership
        OrdinaryWholeOwnerProjectionIsUnchanged
        ActualQwenFamilyBorrowsOriginalPreparedMap
        WorkspaceFormatsRetainOriginalSourceGeometry)
        set(_cpu_joint_projection_test "V2_Integration_CPUJointGraph_${_cpu_joint_projection_case}")
        add_v2_test(${_cpu_joint_projection_test}
            COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
                --gtest_filter=CPUJointGraphOwnership.${_cpu_joint_projection_case} --gtest_brief=1
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Graph;Contract;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
        if(TEST ${_cpu_joint_projection_test})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_cpu_joint_projection_test})
        endif()
    endforeach()
    unset(_cpu_joint_projection_case)
    unset(_cpu_joint_projection_test)
endif()

# The actual portable row selector/writer guards payload economy independently of GPU availability.
add_executable(v2_unit_moe_overlay_shared_dispatch_selection
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/execution/moe/Test__MoEOverlaySharedDispatchSelection.cpp)
target_include_directories(v2_unit_moe_overlay_shared_dispatch_selection PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(v2_unit_moe_overlay_shared_dispatch_selection PRIVATE cxx_std_23)
target_link_libraries(v2_unit_moe_overlay_shared_dispatch_selection PRIVATE GTest::gtest_main)
add_v2_test(V2_Unit_SharedDispatchLiveUnion
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_shared_dispatch_selection>
    LABELS "V2;Unit;DeviceFree;MoE;LivePayload;Regression" NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30)
add_v2_test(V2_Integration_SharedDispatchLiveUnionContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_shared_dispatch_selection>
            --gtest_filter=MoEOverlaySharedDispatchSelection.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;LivePayload;Contract;Regression" NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30)

# Native capture must touch only the union of actual same-rank consumer rows.
if(TARGET v2_integration_mapped_activation_packet_cuda_rocm)
    foreach(vendor IN ITEMS CUDA ROCm)
        add_v2_test(V2_Integration_SharedDispatchLiveUnion_${vendor}
            COMMAND $<TARGET_FILE:v2_integration_mapped_activation_packet_cuda_rocm>
                    --gtest_filter=Test__MappedActivationPacketCUDAAndROCm.${vendor}SharedPhysicalUnionHasOneWriterAndOnlyLivePayload
                    --gtest_brief=1
            LABELS "V2;Integration;ProductionTestPreflight;${vendor};MoE;MappedHost;GraphCapture;LivePayload;Regression"
            MPI_PROCS 1 TIMEOUT 30 NO_MODELS)
    endforeach()
endif()

# Source-native fixture metadata must validate before any backend/device setup.
if(TARGET v2_integration_mapped_activation_packet_cuda_rocm)
    add_v2_test(V2_Integration_JointSourceFixtureColdPlacement
        COMMAND $<TARGET_FILE:v2_integration_mapped_activation_packet_cuda_rocm>
                --gtest_filter=Test__MappedActivationPacketCUDAAndROCm.JointSourceFixtureHasCompleteOriginalColdPlacement
                --gtest_repeat=20 --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;SourceOwnership;Contract;Regression"
        MPI_PROCS 1 TIMEOUT 30 NO_MODELS)
endif()

# Actual factory scalar projection, including authored CPU TP1 and joint TP2..8.
add_executable(v2_unit_moe_graph_compute_policy unit/execution/moe/Test__MoEGraphComputePolicy.cpp)
target_include_directories(v2_unit_moe_graph_compute_policy PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(v2_unit_moe_graph_compute_policy PRIVATE cxx_std_23)
target_link_libraries(v2_unit_moe_graph_compute_policy PRIVATE GTest::gtest_main llaminar2_core)
add_v2_test(V2_Unit_MoEGraphComputePolicy
    COMMAND $<TARGET_FILE:v2_unit_moe_graph_compute_policy>
    LABELS "V2;Unit;DeviceFree;CPU;MoE;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_MoEGraphComputePolicyContract
 COMMAND $<TARGET_FILE:v2_unit_moe_graph_compute_policy> --gtest_repeat=20
 LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Contract;Regression"
 MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
 ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
if(TEST V2_Integration_MoEGraphComputePolicyContract)
 list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEGraphComputePolicyContract)
endif()

# Fixed target identity is semantic channel data, not an unauthenticated reserved word.
add_v2_test(V2_Integration_LiveCohortChannelIdentityContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_activation_epoch_protocol>
        --gtest_filter=Test__MoEOverlayActivationEpochProtocol.FixedCohortTargetIsAuthenticatedAcrossReplay
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;Protocol;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_LiveCohortChannelIdentityContract)
if(TARGET v2_integration_mapped_activation_packet_cuda_rocm)
    foreach(vendor IN ITEMS CUDA ROCm)
        set(_zero_joint_test V2_Integration_JointZeroWeightCompactionReplay_${vendor})
        add_v2_test(${_zero_joint_test}
            COMMAND $<TARGET_FILE:v2_integration_mapped_activation_packet_cuda_rocm>
                --gtest_filter=Test__MappedActivationPacketCUDAAndROCm.${vendor}JointZeroWeightCompactionReplay
                --gtest_brief=1
            LABELS "V2;Integration;ProductionTestPreflight;${vendor};MoE;MappedHost;GraphCapture;LivePayload;Replay;Regression"
            MPI_PROCS 1 NO_MODELS TIMEOUT 30)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_zero_joint_test})
    endforeach()
    unset(_zero_joint_test)
endif()

add_v2_test(V2_Integration_LiveCohortPlacementBankContract
    COMMAND $<TARGET_FILE:v2_test_moe_runtime_table>
        --gtest_filter=Test__MoERuntimeTable.LiveCohortTargetsShareOriginalPlacementBankPublication
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;DeviceEpoch;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_LiveCohortPlacementBankContract)
