#[=[
@file V2DeclaredGPUCachePlacementTests.cmake
@brief Register exact declared cache ownership and source-role planning contracts.

Device-free tests exercise public requests and original resource identities,
including compute participants that also own storage and zero-capacity layouts.
Separate native and model admission gates retain their execution obligations.
]=]
if(TARGET v2_test_model_parity_definition)
    target_sources(v2_test_model_parity_definition PRIVATE
        unit/planning/Test__DeclaredGPUCachePlacement.cpp)
    add_v2_test(V2_Integration_DeclaredGPUCachePlacementReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=DeclaredGPUCachePlacement.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_DeclaredGPUCachePlacementReplay)
    add_v2_test(V2_Integration_DeclaredGPUCacheContinuationOwnerPlanningReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=DeclaredGPUCachePlacement.ContinuationOwnersRetainAuthoredCoordinatorAtPositiveAndZeroCapacity
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;Admission;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_DeclaredGPUCacheContinuationOwnerPlanningReplay)
endif()

add_v2_test(V2_Integration_DeclaredGPUCacheStorageRoleDiscoveryReplay
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_qfn_source_admission_ci.py
        QFNSourceAdmissionCI.test_declared_storage_requires_its_hardware_without_inventing_compute_domains
        QFNSourceAdmissionCI.test_declared_storage_rejects_unused_duplicate_foreign_and_missing_coordinator
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1" "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_DeclaredGPUCacheStorageRoleDiscoveryReplay)

# Model-free declaration/discovery proof; real model tests stay in canonical parity/source admission.
if(TARGET v2_test_model_parity_definition)
    target_sources(v2_test_model_parity_definition PRIVATE unit/parity/Test__Qwen4ExpOppositeVendorCells.cpp)
    add_v2_test(V2_Integration_Qwen4ExpOppositeVendorCellsReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition> --gtest_filter=Qwen4ExpOppositeVendorCells.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Admission;Parity;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpOppositeVendorCellsReplay)
    add_v2_test(V2_Integration_Qwen4ExpOppositeVendorActiveMTPCellsReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpOppositeVendorCells.ActivePoliciesRetainOriginalStorageAndSourceCapacity:Qwen4ExpOppositeVendorCells.ActiveRuntimeExportsNameOriginalOrdinaryControls
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;MTP;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpOppositeVendorActiveMTPCellsReplay)
endif()

add_v2_test(V2_Integration_Qwen4ExpOppositeVendorDiscoveryReplay
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_qfn_tp4_source_closure.py
        QFNTP4SourceClosureTests.test_opposite_vendor_ordinary_retained15_has_exact_sources_and_fresh_processes
        QFNTP4SourceClosureTests.test_opposite_vendor_missing_inactive_predictor_remains_fatal
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Admission;Parity;Discovery;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpOppositeVendorDiscoveryReplay)

# Movement intent stays independent of predictor depth and ngram storage-owner membership.
if(TARGET v2_test_model_parity_definition)
    foreach(_dynamic_case DynamicExpertFamiliesPreserveOriginalStaticSixteen DynamicExpertProductRetainsMovementAndOriginalStorageRoles DynamicExpertRuntimeExportKeepsMatchedMovementOffControls)
        set(_dynamic_gate "V2_Integration_Qwen4ExpOppositeVendor_${_dynamic_case}Replay")
        add_v2_test(${_dynamic_gate}
            COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
                --gtest_filter=Qwen4ExpOppositeVendorCells.${_dynamic_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;MTP;Dynamic;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_dynamic_gate})
    endforeach()
endif()

# Concurrent consumer-local channels are a distinct source contract, not shared GPU-owner pooling.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_ConsumerLocalAdaptiveLookupGeometryReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=DeclaredGPUCachePlacement.ConsumerLocalPluralLayoutProjectsOriginalRequesterWithoutSharedCoordinator:DeclaredGPUCachePlacement.ConsumerLocalServiceFramesAreDisjointAndPricedByCanonicalMappedAllocation:DeclaredGPUCachePlacement.ConsumerLocalTP2SourceSiblingsPreserveOriginalPoliciesAndSeparateQualification
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ConsumerLocalAdaptiveLookupGeometryReplay)
endif()
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_GPUCacheSharedDirectoryCapacity
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupAdmission.SharedDirectoryPressureBalancesEveryFeasibleOwner
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;Admission;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUCacheSharedDirectoryCapacity)
    add_v2_test(V2_Integration_ConsumerLocalTP2AdaptiveLookupAdmissionReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupAdmission.ConsumerLocalTP2PreservesAuthoredCoordinatesAndSharedSourceBOM:GPUAdaptiveImmutableLookupAdmission.ConsumerLocalTP2RankInputsKeepOriginalAssignments
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;TP;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ConsumerLocalTP2AdaptiveLookupAdmissionReplay)
endif()

# Generic value/receipt coverage is independent of initially qualified native topology selection.
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_ConsumerLocalGenericPopulationMetadataReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupAdmission.ConsumerLocalGenericMetadataPopulationBOMAndProvisionWireReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ConsumerLocalGenericPopulationMetadataReplay)
endif()

# One replicated query/source owns shared physical caches; foreign readers own result/ACK edges only.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_SharedCoordinatorAdaptiveLookupGeometryReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=DeclaredGPUCachePlacement.SharedCoordinatorReadersPreserveOneSourceAndExactResultACKBOM:DeclaredGPUCachePlacement.SharedCoordinatorTP2SourceSiblingsPreserveOriginalPolicies:DeclaredGPUCachePlacement.SharedCoordinatorCUDATP2SourceSiblingsPreserveOriginalPoliciesAndStorageRole:DeclaredGPUCachePlacement.SharedCoordinatorRAMSourceMTPKeepsDistinctOriginalProducts
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;TP;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_SharedCoordinatorAdaptiveLookupGeometryReplay)
endif()
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_SharedCoordinatorTP2LookupAdmissionReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupAdmission.SharedCoordinatorTP2OriginalPMAAndWireReplay:RankMemoryPlanInputs.SharedCoordinatorTP2KeepsOriginalComputeAndStorageRoles:RankMemoryPlanInputs.SharedCoordinatorCUDATP2KeepsOppositeVendorStorageAndNoPeerReader:GPUAdaptiveImmutableLookupAdmission.SharedCoordinatorCUDATP2OriginalPMAAndMappedWireReplay
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;TP;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_SharedCoordinatorTP2LookupAdmissionReplay)
endif()

# Actual cold shared-source method retains both original TP rendezvous phases and abort ownership.
if(TARGET v2_test_qwen4exp_metadata)
    add_v2_test(V2_Integration_SharedCoordinatorColdSourcePreparationReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_metadata>
            --gtest_filter=Qwen4ExpReplicatedPLECohort.SharedColdSourceUsesAuthoredCoordinatorAndBothOriginalPhases:Qwen4ExpReplicatedPLECohort.SharedColdSourceFailureAbortsAndUnblocksOriginalPeer
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;TP;Ownership;Lifecycle;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_SharedCoordinatorColdSourcePreparationReplay)
endif()

if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_ConsumerLocalTP4AndTP8RankAdmissionReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupAdmission.ConsumerLocalTP4AndTP8RankInputsKeepOriginalAssignments --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ConsumerLocalTP4AndTP8RankAdmissionReplay)
endif()
