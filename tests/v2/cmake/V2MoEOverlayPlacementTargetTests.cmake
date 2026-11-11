# @file V2MoEOverlayPlacementTargetTests.cmake
# @brief Register actual whole/cohort migration placement and controller replay contracts.
# These tests prove metadata/slot obligations; native execution remains separately gated.
add_v2_routed_expert_policy_unit(
    v2_unit_moe_overlay_placement_targets
    V2_Unit_MoEOverlayPlacementTargets
    unit/execution/moe/Test__MoEOverlayPlacementTargets.cpp)
add_v2_test(V2_Integration_MoEOverlayPlacementCycleReplay
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_placement_targets>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Residency;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayPlacementCycleReplay)

# Exact original-DFS order and errors through the actual cycle builder.
add_v2_test(V2_Integration_MoEOrderedCycleAdjacencyEquivalence
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_placement_targets>
        --gtest_filter=MoEOverlayOrderedCycleEquivalence.*
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Residency;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOrderedCycleAdjacencyEquivalence)


# Actual original protocol/backend replay. Native captures remain a distinct gate.
add_v2_test(V2_Integration_MoEDeviceFinitePlacementProtocolReplay
    COMMAND $<TARGET_FILE:v2_test_moe_overlay_device_epoch_protocol>
        --gtest_filter=Test__MoEOverlayDeviceEpochProtocol.EpochArenaRequiresExactOriginalPMAAndReleasesPackedBytes:Test__MoEOverlayDeviceEpochProtocol.FiniteParentPinsUnreachedOwnerAndResumesPublicationReplay:Test__MoEOverlayDeviceEpochProtocol.FiniteParentKernelAdmissionAndRequestResetReplay:Test__MoEOverlayDeviceEpochProtocol.FiniteRetirementRequiresOriginalPerPeerTerminalProgressReplay
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Residency;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEDeviceFinitePlacementProtocolReplay)

# Actual native recorded inputs; original owner/model execution is a separate gate.
if(TARGET v2_integration_moe_overlay_device_controller_fabric_cuda_rocm)
    foreach(finite_backend IN ITEMS CUDA2 ROCm2 ROCm4)
        if(finite_backend STREQUAL "CUDA2")
            set(finite_label CUDA)
        else()
            set(finite_label ROCm)
        endif()
        set(finite_test V2_Integration_MoEFinitePlacement_${finite_backend}_CapturedProtocolReplay)
        add_v2_test(${finite_test}
            COMMAND $<TARGET_FILE:v2_integration_moe_overlay_device_controller_fabric_cuda_rocm>
                --gtest_filter=Test__MoEOverlayDeviceControllerFabricCUDAAndROCm.FinitePlacement${finite_backend}CapturedProtocolReplay
            LABELS "V2;Integration;ProductionTestPreflight;${finite_label};MoE;Residency;GraphCapture;Replay;Regression"
            MPI_PROCS 2 NO_MODELS TIMEOUT 120)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${finite_test})
    endforeach()
    unset(finite_backend)
    unset(finite_label)
    unset(finite_test)
endif()


# Missing original graph terminal proof cannot become a reusable worker result.
if(TARGET v2_unit_moe_overlay_inference_transaction)
    add_v2_test(V2_Integration_MoEGraphFailureTerminalLifecycle
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_inference_transaction>
            --gtest_filter=Test__MoEOverlayInferenceTransaction.MissingOriginalGraphFailureTerminalIsProcessFatal:Test__MoEOverlayInferenceTransaction.OriginalGraphFailureTerminalRemainsRecoverable
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Lifecycle;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEGraphFailureTerminalLifecycle)
endif()

# Device command projection must retain the same simple-cycle geometry as admission.
if(TARGET v2_unit_moe_overlay_device_physical_movement)
    add_v2_test(V2_Integration_MoEDevicePhysicalSimpleCycleReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_device_physical_movement>
            --gtest_filter=Test__MoEOverlayDevicePhysicalMovement.DurableCommandsRetainSimpleCyclesAcrossSharedTargets:Test__MoEOverlayDevicePhysicalMovement.AuthoredCycleIdentityRetainsOriginalLaneBound:Test__MoEOverlayDevicePhysicalMovement.MovementFlagsRejectInvalidIdentity
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Movement;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEDevicePhysicalSimpleCycleReplay)
endif()



# Preserve device-authored group identity through the original measured policy.
if(TARGET v2_unit_moe_overlay_device_placement_policy)
    add_v2_test(V2_Integration_MoEDeviceAuthoredCycleIdentityReplay
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_device_placement_policy>
            --gtest_filter=Test__MoEOverlayDevicePlacementPolicy.AuthoredCycleIdsSurviveTwoAndFiveCycleBudgets
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Movement;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEDeviceAuthoredCycleIdentityReplay)
endif()

# Both original native policy leaders must retain immutable authored groups.
if(TARGET v2_integration_moe_overlay_device_controller_fabric_cuda_rocm)
    add_v2_test(V2_Integration_MoEDeviceAuthoredCycleIdentity_CUDA_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_overlay_device_controller_fabric_cuda_rocm>
            --gtest_filter=Test__MoEOverlayDeviceControllerFabricCUDAAndROCm.DeviceAuthoredTwoCycleWaveAdvancesBothIndependentPlacementAxes:Test__MoEOverlayDeviceControllerFabricCUDAAndROCm.CUDAAuthoredPreparedRestoreHonorsLayerCapacityAcrossEpochs:Test__MoEOverlayDeviceControllerFabricCUDAAndROCm.ROCmAuthoredPreparedRestoreHonorsLayerCapacityAcrossEpochs
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;ROCm;MoE;Movement;Replay;Regression"
        MPI_PROCS 2 NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_PERF_STATS_SUMMARY=1" "LLAMINAR_PERF_STATS_FILTER=moe_overlay_controller")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEDeviceAuthoredCycleIdentity_CUDA_ROCm)
endif()

# Two real native cache streams serialize one original epoch request slot.
if(TARGET v2_integration_moe_overlay_device_controller_fabric_cuda_rocm)
    foreach(predecessor_backend IN ITEMS CUDA ROCm)
        set(predecessor_name V2_Integration_${predecessor_backend}_MoEFollowerPredecessorCapturedReplay)
        add_v2_test(${predecessor_name}
            COMMAND $<TARGET_FILE:v2_integration_moe_overlay_device_controller_fabric_cuda_rocm>
                --gtest_filter=MoEFollowerPredecessorColdSourceCUDAAndROCm.FollowerPredecessor${predecessor_backend}SharedSlotCapturedReplay
            LABELS "V2;Integration;ProductionTestPreflight;${predecessor_backend};MoE;CapturedReplay;Ownership;Regression"
            MPI_PROCS 2 NO_MODELS TIMEOUT 120)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${predecessor_name})
    endforeach()
    unset(predecessor_name)
    unset(predecessor_backend)
endif()

# Absorbing original abort must terminate a peer acquire before any successor admission.
foreach(peer_abort_backend IN ITEMS CUDA ROCm)
    if(peer_abort_backend STREQUAL "CUDA")
        set(peer_abort_target v2_integration_cuda_moe_overlay_epoch)
        set(peer_abort_suite CUDAMoEOverlayEpochTest)
    else()
        set(peer_abort_target v2_integration_rocm_moe_overlay_epoch)
        set(peer_abort_suite ROCmMoEOverlayEpochTest)
    endif()
    if(TARGET ${peer_abort_target})
        set(peer_abort_test V2_Integration_${peer_abort_backend}_MoEPeerEpochAbortCapturedReplay)
        add_v2_test(${peer_abort_test}
            COMMAND $<TARGET_FILE:${peer_abort_target}>
                --gtest_filter=${peer_abort_suite}.CapturedPeerAbortBeforeGenerationAndSuccessorReplay
            LABELS "V2;Integration;ProductionTestPreflight;${peer_abort_backend};MoE;Residency;GraphCapture;Replay;Regression"
            MPI_PROCS 1 NO_MODELS TIMEOUT 30)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${peer_abort_test})
    endif()
endforeach()
unset(peer_abort_backend)
unset(peer_abort_target)
unset(peer_abort_suite)
unset(peer_abort_test)

# Exact original follower cache/slot BOM; no native materialization is performed.
add_v2_test(V2_Integration_MoEOriginalNativeSlotBOMContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_capacity_resolver>
        --gtest_filter=MoEOverlayCapacityResolver.OriginalNativeSlotBOM*
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;MemoryBOM;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOriginalNativeSlotBOMContract)
