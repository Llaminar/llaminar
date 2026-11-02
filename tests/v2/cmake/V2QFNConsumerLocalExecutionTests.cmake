# Canonical TP2 execution declarations share source, topology and runtime export;
# this device-free gate grants no numerical/native/movement/performance result.
if(NOT V2_PERF_TESTS_ONLY AND TARGET v2_test_model_parity_definition)
    target_sources(v2_test_model_parity_definition PRIVATE
        unit/parity/Test__Qwen4ExpConsumerLocalTP2Execution.cpp)
    add_v2_test(V2_Integration_QFNConsumerLocalTP2ExecutionObligations
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpConsumerLocalTP2Execution.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;ExpertMovement;Economy;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNConsumerLocalTP2ExecutionObligations)
endif()
