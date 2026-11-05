# @file V2ExpertTierRectangularWeightTests.cmake
# @brief Focused original-coordinate and compact-wire regressions for existing expert transports.
# Native replay uses each backend owning fixture; CPU contracts remain device-free.

# Append through the existing canonical test registration owner; no parallel runner.
if(TARGET v2_test_expert_tier_weight_stream)
    add_v2_test(V2_Integration_ExpertTierRectangularCPUWire
        COMMAND $<TARGET_FILE:v2_test_expert_tier_weight_stream>
            --gtest_filter=ExpertTierWeightStream.OriginalProjectionIntersectionsRetainIndependentCoordinates:ExpertTierWeightStream.RectangularEncodedColumnsAllFormatsPreserveNeighboursAndLiveBytes --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;ByteExact;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ExpertTierRectangularCPUWire)
endif()
if(TARGET v2_unit_moe_overlay_distributed_residency_protocol)
    add_v2_test(V2_Integration_ExpertTierRectangularRemoteProtocol
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_distributed_residency_protocol>
            --gtest_filter=Test__MoEOverlayDistributedResidencyProtocol.RectangularCpuGpuSourcesUseOriginalRootAndExistingSendStaging:Test__MoEOverlayDistributedResidencyProtocol.FloatingRectanglesMapOriginalRowsWithoutHostAssembly --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ExpertTierRectangularRemoteProtocol)
    add_v2_test(V2_Integration_ExpertTierGpuArrivalProjectionContract
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_distributed_residency_protocol>
            --gtest_filter=Test__MoEOverlayDistributedResidencyProtocol.GpuArrivalAuthenticatesOriginalDestinationAllNativeFormats
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ExpertTierGpuArrivalProjectionContract)
endif()
foreach(backend IN ITEMS CUDA ROCm)
    if(backend STREQUAL "CUDA")
        set(rectangular_target v2_integration_cuda_expert_tier_weight_kernels)
    else()
        set(rectangular_target v2_integration_rocm_expert_tier_weight_kernels)
    endif()
    if(TARGET ${rectangular_target})
        target_include_directories(${rectangular_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_v2_test(V2_Integration_${backend}_ExpertTierRectangularWeights
            COMMAND $<TARGET_FILE:${rectangular_target}>
                --gtest_filter=${backend}ExpertTierWeightKernelsTest.RectangularNativeMovementAllFormatsAndDirtyReplay:${backend}ExpertTierWeightKernelsTest.ContiguousFloatingLaneRoundTripsAllPrecisionsWithoutInferenceWaits
            LABELS "V2;Integration;ProductionTestPreflight;${backend};ExpertOverlay;ByteExact;ExplicitStream;Replay;Regression"
            MPI_PROCS 1 NO_MODELS TIMEOUT 30)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${backend}_ExpertTierRectangularWeights)
    endif()
endforeach()

# Append after native owning targets exist; root owns the current rectangle module.
foreach(_captured_rect_backend CUDA ROCm)
    string(TOLOWER "${_captured_rect_backend}" _captured_rect_lower)
    set(_captured_rect_target v2_integration_${_captured_rect_lower}_expert_tier_weight_kernels)
    if(TARGET ${_captured_rect_target})
        foreach(_captured_rect_edge Local Remote)
            add_v2_test(V2_Integration_${_captured_rect_backend}_Captured${_captured_rect_edge}RectangularWeightReplay
                COMMAND $<TARGET_FILE:${_captured_rect_target}>
                    --gtest_filter=${_captured_rect_backend}ExpertTierWeightKernelsTest.Captured${_captured_rect_edge}RectangularTransfersAllFormats
                LABELS "V2;Integration;ProductionTestPreflight;${_captured_rect_backend};GPU;WeightStreaming;Rectangle;GraphCapture;Replay;Economy;Regression"
                MPI_PROCS 1 NO_MODELS TIMEOUT 30)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_captured_rect_backend}_Captured${_captured_rect_edge}RectangularWeightReplay)
        endforeach()
        if(_captured_rect_backend STREQUAL "ROCm")
            add_v2_test(V2_Integration_ROCm_CapturedPackedTransferNativeProgress
                COMMAND $<TARGET_FILE:${_captured_rect_target}>
                    --gtest_filter=ROCmExpertTierWeightKernelsTest.CapturedLocalPackedTransfersProgressAllFormats:ROCmExpertTierWeightKernelsTest.CapturedRemotePackedTransfersProgressAllFormats
                LABELS "V2;Integration;ProductionTestPreflight;ROCm;GPU;WeightStreaming;GraphCapture;NativeAsync;Regression"
                MPI_PROCS 1 NO_MODELS TIMEOUT 30)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCm_CapturedPackedTransferNativeProgress)
        endif()
    endif()
endforeach()

# Compact CPU wire contracts use the existing binaries and canonical launchers.
if(TARGET v2_test_expert_tier_weight_stream)
    add_v2_test(V2_Integration_ExpertTierNativeCPUCompactWire
        COMMAND $<TARGET_FILE:v2_test_expert_tier_weight_stream>
            --gtest_filter=ExpertTierWeightStream.NativeCpuRectanglesUseOnlyLivePlanesAndIndependentRootStrides --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;ByteExact;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ExpertTierNativeCPUCompactWire)
endif()
if(TARGET v2_unit_moe_overlay_distributed_residency_protocol)
    add_v2_test(V2_Integration_ExpertTierNativeCPUCompactProtocol
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_distributed_residency_protocol>
            --gtest_filter=Test__MoEOverlayDistributedResidencyProtocol.NativeCpuRectanglesKeepStagingUntilExactAckAndScatterOnlyLiveColumns:Test__MoEOverlayDistributedResidencyProtocol.HostSourceRetainsEveryPayloadUntilItsExactMpiAcknowledgement:Test__MoEOverlayDistributedResidencyProtocol.RemoteCpuProjectionManifestAndChunksAreExactAndAuthenticated --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ExpertTierNativeCPUCompactProtocol)
endif()

if(TARGET v2_integration_moe_overlay_residency_consensus_mpi)
    add_v2_test(V2_Integration_MoEOverlayCPUCompactWire_MPI
        COMMAND $<TARGET_FILE:v2_integration_moe_overlay_residency_consensus_mpi>
            --gtest_filter=Test__MoEOverlayMPIResidencyConsensus.BidirectionalRemoteCpuProjectionLanesMoveExactFinalBytesWithoutBlocking:Test__MoEOverlayMPIResidencyConsensus.SelectedExecutionRemoteProjectionLanesRemainByteExactWithoutWorldTraffic:Test__MoEOverlayMPIResidencyConsensus.PreparedContextRestorationPurposeMovesExactRemoteBytes:Test__MoEOverlayMPIResidencyConsensus.DistributedPhysicalCpuFabricPublishesByteExactRCUBanks
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;Overlay;MPI;Async;ByteExact;Regression"
        MPI_PROCS 2 TIMEOUT 60 NO_MODELS
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayCPUCompactWire_MPI)
endif()
