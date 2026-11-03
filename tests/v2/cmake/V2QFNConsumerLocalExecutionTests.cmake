# Canonical local-cache and shared-source TP2 execution use the same typed runtime export;
# this device-free gate grants no numerical/native/movement/performance result.
if(NOT V2_PERF_TESTS_ONLY AND TARGET v2_test_model_parity_definition)
    target_sources(v2_test_model_parity_definition PRIVATE
        unit/parity/Test__Qwen4ExpConsumerLocalTP2Execution.cpp
        unit/parity/Test__Qwen4ExpSharedCoordinatorTP2Execution.cpp)
    add_v2_test(V2_Integration_QFNConsumerLocalTP2ExecutionObligations
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpConsumerLocalTP2Execution.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;ExpertMovement;Economy;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNConsumerLocalTP2ExecutionObligations)
    add_v2_test(V2_Integration_QFNSharedCoordinatorTP2ExecutionObligations
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSharedCoordinatorTP2Execution.*:ParityHardwareRequirements.DeclaredNgramStorageRetainsExactIndependentGPURequirements --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;ExpertMovement;Economy;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNSharedCoordinatorTP2ExecutionObligations)
    add_v2_test(V2_Integration_QFNCanonicalSixHardwareMetadata
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSharedCoordinatorTP2Execution.CanonicalSixHardwareRowsAndTierPolicies --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Metadata;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNCanonicalSixHardwareMetadata)
    add_v2_test(V2_Integration_QFNProductionDefaultsResidencyMetadata
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSharedCoordinatorTP2Execution.SixHardwareProductionDefaultsUseOriginalPMAResidency --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Metadata;Economy;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNProductionDefaultsResidencyMetadata)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNMatchedCacheEconomyMetadata
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSharedCoordinatorTP2Execution.MatchedCacheEconomyKeepsExplicitResidencyAndPMAByteCeilings:Qwen4ExpSharedCoordinatorTP2Execution.MatchedCacheDefinitionsRetainFunctionalAxesAndRejectForeignOwners --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Metadata;Economy;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNMatchedCacheEconomyMetadata)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNConsumerLocalTP4ExecutionObligations
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition> --gtest_filter=Qwen4ExpConsumerLocalTP4Execution.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;ExpertMovement;Economy;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNConsumerLocalTP4ExecutionObligations)
endif()
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_ConsumerLocalTP4AndTP8OriginalService
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives> --gtest_filter=ImmutableLookupSubordinate.ConsumerLocalTP4AndTP8OriginalServiceRetiresEveryChannel
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ConsumerLocalTP4AndTP8OriginalService)
endif()

# Setup declarations cannot certify dynamic movement; this exact metadata replay
# guards the new callable workload while native/model execution remains separate.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNConsumerLocalTP4MovementObligations
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpConsumerLocalTP4Movement.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;ExpertMovement;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNConsumerLocalTP4MovementObligations)
endif()
