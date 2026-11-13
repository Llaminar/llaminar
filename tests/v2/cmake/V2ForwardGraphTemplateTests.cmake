# Actual production-builder construction tests and an independent speedometer.
# Immutable source preparation has deterministic work-count/lifetime gates;
# elapsed construction time is reported without a shared-host timing threshold.
# Both targets use the canonical process owner and require no GPU or model file.

set(V2_PREPARED_GRAPH_TEMPLATE_PREFLIGHT_TESTS)
if(NOT V2_PERF_TESTS_ONLY)
    add_executable(v2_test_forward_graph_template
        unit/execution/local_execution/graph/Test__PreparedForwardGraphTemplate.cpp)
    target_include_directories(v2_test_forward_graph_template PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(v2_test_forward_graph_template PRIVATE
        llaminar2_core GTest::gtest GTest::gtest_main GTest::gmock)
    add_v2_test(V2_Unit_PreparedForwardGraphTemplate
        COMMAND $<TARGET_FILE:v2_test_forward_graph_template>
        LABELS "V2;Unit;CPU;DeviceFree;GraphConstruction;MoE;MTP;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    add_v2_test(V2_Integration_PreparedForwardGraphTemplateReplay
        COMMAND $<TARGET_FILE:v2_test_forward_graph_template>
            --gtest_filter=PreparedForwardGraphTemplate.*
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;GraphConstruction;MoE;MTP;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PREPARED_GRAPH_TEMPLATE_PREFLIGHT_TESTS
        V2_Integration_PreparedForwardGraphTemplateReplay)
endif()

# Keep the creation benchmark available in optimized Release test companions.
# It measures source preparation and graph specialization, never inference.
add_executable(v2_perf_forward_graph_template
    performance/Perf__ForwardGraphTemplateConstruction.cpp)
target_include_directories(v2_perf_forward_graph_template PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_perf_forward_graph_template PRIVATE
    llaminar2_core GTest::gtest GTest::gmock)
add_v2_test(V2_Perf_ForwardGraphTemplateConstruction
    COMMAND $<TARGET_FILE:v2_perf_forward_graph_template>
    LABELS "V2;Performance;Benchmark;CPU;DeviceFree;GraphConstruction;MoE;MTP;JSON"
    MPI_PROCS 1 NO_MPI NO_MODELS
    ENVIRONMENT "LLAMINAR_LOG_LEVEL=ERROR")
