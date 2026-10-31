# Validate production capture records without linking an accelerator runtime.
# Native ACK cancellation replays independently authenticate the zero-register
# HIP kernel which exposed the former positive-register-only validation rule.
add_executable(v2_unit_gpu_graph_kernel_metadata
    unit/backends/Test__GPUGraphKernelMetadata.cpp)
target_include_directories(v2_unit_gpu_graph_kernel_metadata PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_gpu_graph_kernel_metadata PRIVATE cxx_std_20)
target_link_libraries(v2_unit_gpu_graph_kernel_metadata PRIVATE GTest::gtest_main)
add_v2_test(V2_Unit_GPUGraphKernelMetadata
    COMMAND $<TARGET_FILE:v2_unit_gpu_graph_kernel_metadata>
    LABELS "V2;Unit;DeviceFree;Capture;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUGraphKernelMetadataReplay
    COMMAND $<TARGET_FILE:v2_unit_gpu_graph_kernel_metadata>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Capture;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_GPUGraphKernelMetadataReplay)
