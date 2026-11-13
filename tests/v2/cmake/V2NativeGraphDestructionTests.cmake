# Exact native cleanup bodies are replayed without linking any GPU runtime.
if(NOT V2_PERF_TESTS_ONLY)
add_executable(v2_test_native_graph_destruction
    unit/backends/Test__NativeGraphDestruction.cpp)
if(TARGET v2_test_native_graph_destruction)
    set(_graph_destruction_generated "${CMAKE_CURRENT_BINARY_DIR}/native-graph-destruction")
    add_custom_command(OUTPUT "${_graph_destruction_generated}/NativeGraphDestructionBodies.inc"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/generate_native_graph_destruction_replay.py"
            --source "${REPO_ROOT}" --output "${_graph_destruction_generated}"
        DEPENDS unit/scripts/generate_native_graph_destruction_replay.py
            "${REPO_ROOT}/src/v2/backends/cuda/CUDAGraphCapture.cu"
            "${REPO_ROOT}/src/v2/backends/rocm/HIPGraphCapture.cpp"
        VERBATIM)
    target_sources(v2_test_native_graph_destruction PRIVATE "${_graph_destruction_generated}/NativeGraphDestructionBodies.inc")
    target_include_directories(v2_test_native_graph_destruction PRIVATE "${_graph_destruction_generated}")
    target_compile_features(v2_test_native_graph_destruction PRIVATE cxx_std_23)
    target_link_libraries(v2_test_native_graph_destruction PRIVATE GTest::gtest_main Threads::Threads)
    add_v2_test(V2_Unit_NativeGraphDestruction
        COMMAND $<TARGET_FILE:v2_test_native_graph_destruction>
        LABELS "V2;Unit;DeviceFree;Ownership;GraphCapture;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_NativeGraphDestructionReplay
        COMMAND $<TARGET_FILE:v2_test_native_graph_destruction> --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;GraphCapture;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeGraphDestructionReplay)
endif()
endif()
