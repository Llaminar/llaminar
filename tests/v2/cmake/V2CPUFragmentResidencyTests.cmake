# @file V2CPUFragmentResidencyTests.cmake
# @brief Register source-bound fragment, placement and RCU ownership regressions.
#
# These device-free component tests exercise the actual prepared-bank interfaces
# and complete-piece publication. They do not claim native shard migration or
# global placement commit, which require the corresponding physical path gates.
set(V2_CPU_FRAGMENT_TEST_ENV
    "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")

# This file is included after the canonical preflight list is initialized.
# Preserve the native row only when its actual physical-socket requirement was
# admitted by the earlier MPI registration; absent hardware remains unqualified.
if(TEST V2_Integration_MoEOverlayCPUCohortPhysicalStagingAbort)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_MoEOverlayCPUCohortPhysicalStagingAbort)
endif()
foreach(_floating_cohort_backend IN ITEMS CUDA ROCm)
    set(_floating_cohort_test V2_Integration_${_floating_cohort_backend}_CPUJointFloatingFabricRectangles)
    if(TEST ${_floating_cohort_test})
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_floating_cohort_test})
    endif()
endforeach()

add_v2_test(V2_Integration_PreparedProjectionIntersectionContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_participant_residency>
        --gtest_filter=PreparedProjectionIntersectionContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PreparedProjectionIntersectionContract)

add_v2_test(V2_Integration_CPUFragmentPayloadContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUFragmentPayloadContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;PreparedWeights;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUFragmentPayloadContract)

add_v2_test(V2_Integration_CPUJointPlacementCandidateContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointPlacementCandidateContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointPlacementCandidateContract)

add_v2_test(V2_Integration_CPUJointPlacementTransactionContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointPlacementTransactionContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointPlacementTransactionContract)

add_v2_test(V2_Integration_CPUFragmentEpochBankContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUFragmentEpochBankContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;PreparedWeights;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUFragmentEpochBankContract)

add_v2_test(V2_Integration_CPUJointInitialBankPrepassContract
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
        --gtest_filter=CPUJointInitialBankPrepassContract.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointInitialBankPrepassContract)

target_sources(v2_unit_moe_overlay_physical_residency_fabric PRIVATE
    unit/execution/moe/Test__MoEOverlayPhysicalProjectionPieces.cpp)
add_v2_test(V2_Integration_MoEOverlayPhysicalProjectionPieces
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_physical_residency_fabric>
        --gtest_filter=MoEOverlayPhysicalProjectionPieces.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayPhysicalProjectionPieces)

add_v2_test(V2_Integration_ExpertTierDirectCPUProjection
    COMMAND $<TARGET_FILE:v2_test_expert_tier_weight_stream>
        --gtest_filter=ExpertTierWeightStream.DirectCPUOriginalRootsAllFormatsPreserveIndependentStridesAndDirtyNeighbours:ExpertTierWeightStream.DirectCPUParallelPiecesPreserveSharedQ6WordsAcrossReset
        --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;Transfer;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ExpertTierDirectCPUProjection)

add_v2_test(V2_Integration_MoEOverlayPhysicalPieceDemand
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_capacity_resolver>
        --gtest_filter=MoEOverlayPhysicalPieceDemand.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;MemoryBOM;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayPhysicalPieceDemand)

add_v2_test(V2_Integration_MoEOverlayGlobalRemotePieceOrdinals
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_physical_residency_fabric>
        --gtest_filter=MoEOverlayPhysicalProjectionPieces.GlobalRemoteOrdinalsRemainUniqueThroughEightSkewedMembers
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;Transfer;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayGlobalRemotePieceOrdinals)

add_v2_test(V2_Integration_MoEOverlayShardPlacementGeometry
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointPlacementTransactionContract.GeometryAcceptsEveryFragmentWithoutCreatingScalarAliases
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayShardPlacementGeometry)

add_v2_test(V2_Integration_MoEOverlayLivePreparedWeightBytes
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_physical_residency_fabric>
        --gtest_filter=Test__MoEOverlayPhysicalResidencyFabric.SharedExactGeometryArenaRecyclesRetiredSlotsAcrossLayers:Test__MoEOverlayPhysicalResidencyFabric.SingleShadowSlotSustainsRepeatedByteExactCpuRebalancingAndReusableSealingForEveryWeightFormat:Test__MoEOverlayPhysicalResidencyFabric.PipelineStageTransfersAndSealsEveryPreparedFormat:Test__MoEOverlayPhysicalResidencyFabric.LiveNativeOracleDistinguishesDirtyPaddingAndEveryEncodedField
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;ByteExact;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayLivePreparedWeightBytes)

add_v2_test(V2_Integration_CPUJointRestorationSourceContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointPlacementTransactionContract.RestorationNoOpRetainsOriginalSourceAndSealedGeometry
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointRestorationSourceContract)
add_v2_test(V2_Integration_CPUJointHistogramCandidateSourceContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointPlacementTransactionContract.HistogramBodyBuildsFixedTargetCandidatesAndBoundedCycles
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointHistogramCandidateSourceContract)

# Append to late V2CPUFragmentResidencyTests.cmake, after its shared environment.
# Reconstructs unpublished source-bound proposals; does not admit dynamic execution.
add_v2_test(V2_Integration_CPUJointAuthoritativeTargetContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointAuthoritativeTargetContract.*
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Protocol;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointAuthoritativeTargetContract)

# Complete physical-piece evidence uses original geometry and bound transfer extents.
add_v2_test(V2_Integration_MoEOverlayPhysicalPieceMeasurement
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_migration_measurement_ledger>
        --gtest_filter=PhysicalPieceMeasurementContract.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
add_v2_test(V2_Integration_MoEOverlayPhysicalPieceMeasurementMapping
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_participant_residency>
        --gtest_filter=Test__MoEOverlayParticipantMigration.FlattenedOperationsRetainOriginalMeasurementCoordinates
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_MoEOverlayPhysicalPieceMeasurement V2_Integration_MoEOverlayPhysicalPieceMeasurementMapping)

# Dedicated executable: allocation replacements affect only this binary.
add_v2_routed_expert_policy_unit(
    v2_unit_moe_overlay_measurement_completion_allocations
    V2_Unit_MoEOverlayMeasurementCompletionAllocations
    unit/execution/moe/Test__MoEOverlayMeasurementCompletionAllocations.cpp)
add_v2_test(V2_Integration_MoEOverlayMeasurementCompletionAllocations
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_measurement_completion_allocations>
        --gtest_filter=MeasurementCompletionAllocations.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Ownership;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
add_v2_test(V2_Integration_MoEOverlayPhysicalMeasurementQuoteDrain
    COMMAND $<TARGET_FILE:v2_test_moe_overlay_tier_migration_transport>
        --gtest_filter=MoEOverlayTierMigrationTransport.BoundQuoteRejectsWrongSamplesBeforePrepareAndDrains
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
add_v2_test(V2_Integration_MoEOverlayPhysicalPieceFormatPermutation
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_migration_measurement_ledger>
        --gtest_filter=PhysicalPieceMeasurementContract.ActualProducerPreservesMixedFormatsUnderEveryRolePermutation
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Format;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayMeasurementCompletionAllocations
    V2_Integration_MoEOverlayPhysicalMeasurementQuoteDrain V2_Integration_MoEOverlayPhysicalPieceFormatPermutation)

# Native public source factory -> composite -> owned journal, one rank per physical CPU socket.
if(V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
    add_v2_test(V2_Integration_CPUJointPhysicalMeasurementAttribution
        COMMAND $<TARGET_FILE:v2_integration_moe_overlay_residency_consensus_mpi>
            --gtest_filter=MoEOverlayCPUCohortMeasurement.PublicCompositeOwnsTwoLogicalRowsForTwelvePhysicalPieces
            --gtest_repeat=20 --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;Residency;Transfer;Measurement;Lifecycle;Contract;Regression"
        MPI_PROCS 2 NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointPhysicalMeasurementAttribution)
else()
    message(STATUS "CPU cohort measurement attribution: unqualified without two physical sockets; no MPI launch")
endif()

# Context-bound, exact-live movement evidence and rejection lifetime contracts.
add_v2_test(V2_Integration_MoEOverlayLiveEconomyEvidenceCodec
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_migration_measurement_ledger>
        --gtest_filter=LiveEconomyEvidenceCodec.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Protocol;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
add_v2_test(V2_Integration_MoEOverlayMeasurementContextAbortDrain
    COMMAND $<TARGET_FILE:v2_test_moe_overlay_tier_migration_transport>
        --gtest_filter=MoEOverlayTierMigrationTransport.InvalidMeasurementContextRetainsActualAbortDrain
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
add_v2_test(V2_Integration_MoEOverlaySingletonLedgerCompletionAllocations
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_measurement_completion_allocations>
        --gtest_filter=MeasurementCompletionAllocations.DirectLedgerSingletonProjectionAllocatesNothingAndRejectsBeforeMutation
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Ownership;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_MoEOverlayLiveEconomyEvidenceCodec
    V2_Integration_MoEOverlayMeasurementContextAbortDrain
    V2_Integration_MoEOverlaySingletonLedgerCompletionAllocations)

# MPI entry: existing actual two-rank owning binary, no sockets/GPU qualification claim.
add_v2_test(V2_Integration_MoEOverlayLiveEconomyEvidence_MPI
    COMMAND $<TARGET_FILE:v2_integration_moe_overlay_residency_consensus_mpi>
        --gtest_filter=Test__MoEOverlayMPIResidencyConsensus.LiveEconomyEvidenceCountsPayloadAndEmptyReuse
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;ExpertOverlay;Measurement;MPI;Protocol;Regression"
    MPI_PROCS 2 NO_MODELS TIMEOUT 60
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayLiveEconomyEvidence_MPI)

add_v2_test(V2_Integration_MoEOverlayDistributedMeasurementContext
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_distributed_residency_transport>
        --gtest_filter=Test__MoEOverlayDistributedResidencyTransport.OriginalMeasurementContextSurvivesWrapperDrainAndReset
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Lifecycle;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayDistributedMeasurementContext)

add_v2_test(V2_Integration_MoEOverlayCalibrationCoordinateIdentity
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_economy_calibration_controller>
        --gtest_filter=MoEOverlayEconomyCalibrationController.RejectsSameSizedForeignCoordinatesBeforeTransfer
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayCalibrationCoordinateIdentity)

add_v2_test(V2_Integration_MoEOverlayCompleteTargetEvidenceIdentity
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_migration_measurement_ledger>
        --gtest_filter=LiveEconomyEvidenceCodec.CompleteCohortIdentityRejectsRepresentativeMemberAlias
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayCompleteTargetEvidenceIdentity)

# Append to late V2CPUFragmentResidencyTests.cmake before its environment unset.
# Source-bound metadata/unsupported-consumer contract; no physical readiness claim.
add_v2_test(V2_Integration_CPUJointCalibrationTargetContract
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_residency_authority>
        --gtest_filter=CPUJointCalibrationTargetContract.*
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Economy;Calibration;Contract;SourceBinding;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointCalibrationTargetContract)

# Genuine complete CPU-target shadow calibration; one MPI rank per physical socket.
if(V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
    add_v2_test(V2_Integration_CPUJointShadowCalibrationPhysicalFabric
        COMMAND $<TARGET_FILE:v2_integration_moe_overlay_residency_consensus_mpi>
            --gtest_filter=MoEOverlayCPUJointCalibration.RealTwoSocketShadowWaveNeverPublishesPlacement
            --gtest_repeat=20 --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;Residency;Transfer;Calibration;Measurement;Lifecycle;Contract"
        MPI_PROCS 2 NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointShadowCalibrationPhysicalFabric)
else()
    message(STATUS "CPU joint shadow calibration: unqualified without two physical sockets; no MPI launch")
endif()

# Append within the existing V2CPUFragmentResidencyTests scope, before its CPU environment is unset.
add_v2_test(V2_Integration_MoEOverlayCompleteTargetLedger
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_migration_measurement_ledger>
        --gtest_filter=TargetMigrationLedger.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Ownership;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
add_v2_test(V2_Integration_MoEOverlayCompleteTargetNormalization
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_economy_profile_composer>
        --gtest_filter=TargetMigrationNormalization.* --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Economy;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
add_v2_test(V2_Integration_MoEOverlayCompleteTargetCompletionAllocations
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_measurement_completion_allocations>
        --gtest_filter=MeasurementCompletionAllocations.TargetLedgerRecordsAndRejectsWithoutNestedAllocation
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Ownership;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_MoEOverlayCompleteTargetLedger
    V2_Integration_MoEOverlayCompleteTargetNormalization
    V2_Integration_MoEOverlayCompleteTargetCompletionAllocations)

add_v2_test(V2_Integration_MoEOverlayCompleteTargetCardinalityIdentity
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_economy_profile_composer>
        --gtest_filter=TargetMigrationNormalization.CompleteTargetIdentityRejectsSingletonAndMemberAliases
        --gtest_repeat=20 --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;ExpertOverlay;Measurement;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "${V2_CPU_FRAGMENT_TEST_ENV}")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MoEOverlayCompleteTargetCardinalityIdentity)

unset(V2_CPU_FRAGMENT_TEST_ENV)
