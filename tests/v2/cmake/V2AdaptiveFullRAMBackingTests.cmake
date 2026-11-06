# @file V2AdaptiveFullRAMBackingTests.cmake
# @brief Explicit original source/PMA, schema and real host-channel backing replays; no accelerator startup.
foreach(_ram_case
    "v2_test_qwen4exp_primitives|HostImmutableRowProvider.FullRAMSubordinateSourceSharesExactOriginalOwnerAcrossConsumers:ImmutableLookupSubordinate.FullRAMSharedSourceUsesOneOwnerAcrossActualChannelReset|Owners"
    "v2_test_memory_planner|ImmutableRowTablePlacementPlanner.AdaptiveFullRAMDeclaresSourceOnceAndRejectsRedundantCache:ImmutableRowTableAdmissionWire.AdaptiveFullRAMBackingRoundTripsWithoutHostCacheAlias:ImmutableRowTableAdmissionWire.PlacementModeWireRejectsFullWidthAliases:GPUAdaptiveImmutableLookupAdmission.ConsumerLocalGenericMetadataPopulationBOMAndProvisionWireReplay|Admission"
    "v2_test_model_parity_definition|Qwen4ExpGPUAdaptiveParity.FullRAMBackingHasLosslessPublicIdentityAndRejectsRedundantCache:Qwen4ExpGPUAdaptiveLookupEvidence.FullRAMSourceRequiresExactModeAndZeroWindowWork:Qwen4ExpSharedCoordinatorTP2Execution.FullRAMMatchedCacheSiblingsPreserveOriginalDeclaredComparisons|PublicPolicy"
    "v2_test_orchestration_plan_config|OrchestrationConfigDocument.AdaptiveFullRAMBackingRejectsOldSchemaAndMissingField|ConfigSchema")
    string(REPLACE "|" ";" _ram_fields "${_ram_case}")
    list(GET _ram_fields 0 _ram_target)
    list(GET _ram_fields 1 _ram_filter)
    list(GET _ram_fields 2 _ram_name)
    if(TARGET ${_ram_target})
        add_v2_test(V2_Integration_AdaptiveFullRAM${_ram_name}
            COMMAND $<TARGET_FILE:${_ram_target}> --gtest_filter=${_ram_filter} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_AdaptiveFullRAM${_ram_name})
    endif()
endforeach()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_AdaptiveFullRAMTerminalPublication
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=ImmutableLookupProtocol.FullRAMTerminalCollectorPublishesCompleteSourceTuple:ImmutableLookupProtocol.FullRAMCPUCollectorRejectsRedundantCacheAndMappingWork:ImmutableLookupProtocol.PooledSubordinateTransportIsExactAndDoesNotEraseSourceMode:ImmutableLookupSubordinate.FullRAMSharedSourceUsesOneOwnerAcrossActualChannelReset --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_AdaptiveFullRAMTerminalPublication)
endif()

if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_AdaptiveFullRAMPhysicalSourceGeometry
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUAdaptiveImmutableLookupCohortGeometry.FullRAMSourceKeepsOriginalPhysicalOwnerAndCoordinatorGuards:GPUAdaptiveImmutableLookupAdmission.ConsumerLocalGenericMetadataPopulationBOMAndProvisionWireReplay --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_AdaptiveFullRAMPhysicalSourceGeometry)
endif()

# Original source/PMA fixture requires its existing fresh-process cohort owner.
if(TARGET v2_test_pooled_owner_pin_chain)
    add_v2_test(V2_Integration_ImmutableLookupAdmittedSetupEvidence
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py
            $<TARGET_FILE:v2_test_pooled_owner_pin_chain>
            --gtest_filter=PooledImmutableOwnerPinChain.ImmutableSetupCapacityUsesOriginalDisjointRootsAndRejectsPartialSources --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ImmutableLookupAdmittedSetupEvidence)
endif()

# Passive lookup comparisons use the same canonical diagnostic owner and fast Python schema replays.
foreach(_passive_case
        test_declared_combined_passive_binds_actual_local_ready_pairs_and_storage_windows
        test_declared_combined_passive_rejects_foreign_incomplete_or_impossible_native_work
        test_declared_passive_environment_overrides_caller_profiling_before_startup
        test_declared_passive_adaptive_setup_and_zero_activity_are_complete
        test_declared_passive_rejects_partial_wrong_grants_and_foreign_setup
        test_declared_passive_rejects_warmup_partial_failed_and_mapping_claims)
    add_test(NAME V2_Integration_DeclaredPassiveLookup_${_passive_case}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_production_pipeline.py
            BenchmarkPolicyTests.${_passive_case})
    set_tests_properties(V2_Integration_DeclaredPassiveLookup_${_passive_case} PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Benchmark;Ownership;TraceReplay;Regression"
        TIMEOUT 30)
    if(_passive_case MATCHES "^test_declared_combined_")
        set_tests_properties(V2_Integration_DeclaredPassiveLookup_${_passive_case} PROPERTIES
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Benchmark;Ownership;Configuration;Regression")
    endif()
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_DeclaredPassiveLookup_${_passive_case})
endforeach()

# Cold lineage metadata is a projection only; it creates no model/native authority.
if(TARGET v2_test_pooled_owner_pin_chain)
    add_v2_test(V2_Integration_ImmutableLookupLocalSetupLineage
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py
            $<TARGET_FILE:v2_test_pooled_owner_pin_chain>
            --gtest_filter=PooledImmutableOwnerPinChain.LocalSetupLineageKeepsExactConsumerAndLegacyTags --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;Configuration;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ImmutableLookupLocalSetupLineage)
endif()

add_test(NAME V2_Integration_DeclaredBenchmarkLosslessDocument
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_model_kld_corpus_execution.py
        DeclaredBenchmarkRepresentationTests)
set_tests_properties(V2_Integration_DeclaredBenchmarkLosslessDocument PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Benchmark;Ownership;Configuration;Regression"
    TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_DeclaredBenchmarkLosslessDocument)
