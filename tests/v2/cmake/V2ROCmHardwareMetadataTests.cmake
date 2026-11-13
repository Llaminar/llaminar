# Pure publication and launch-policy replay links no driver/backend library.
add_executable(v2_unit_rocm_hardware_metadata
    unit/backends/Test__ROCmHardwareMetadata.cpp)
target_include_directories(v2_unit_rocm_hardware_metadata PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_rocm_hardware_metadata PRIVATE cxx_std_20)
target_link_libraries(v2_unit_rocm_hardware_metadata PRIVATE
    GTest::gtest_main Threads::Threads)
add_v2_test(V2_Unit_ROCmHardwareMetadata
    COMMAND $<TARGET_FILE:v2_unit_rocm_hardware_metadata>
    LABELS "V2;Unit;DeviceFree;ROCm;HardwareMetadata;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_ROCmHardwareMetadataReplay
    COMMAND $<TARGET_FILE:v2_unit_rocm_hardware_metadata>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;HardwareMetadata;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_ROCmHardwareMetadataReplay)

add_v2_test(V2_Integration_ROCmServiceWallClockBindingReplay
    COMMAND $<TARGET_FILE:v2_unit_rocm_hardware_metadata>
        --gtest_filter=ROCmHardwareMetadata.ServiceWallClockBindingIsExactStableAndRequired
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;HardwareMetadata;MoE;Telemetry;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_ROCmServiceWallClockBindingReplay)

add_v2_test(V2_Integration_ROCmColumnAssemblyHardwareBindingReplay
    COMMAND $<TARGET_FILE:v2_unit_rocm_hardware_metadata>
        --gtest_filter=ROCmHardwareMetadata.ColumnAssemblyUsesExactPhysicalCountAndFreshFunctionOccupancy
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;HardwareMetadata;TensorParallel;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_ROCmColumnAssemblyHardwareBindingReplay)

include(${CMAKE_CURRENT_LIST_DIR}/V2HIPProjectionCaptureLifetimeTests.cmake)

add_v2_test(V2_Integration_ROCmStreamWaitValueCapabilityReplay
    COMMAND $<TARGET_FILE:v2_unit_rocm_hardware_metadata>
        --gtest_filter=ROCmHardwareMetadata.StreamWaitValueCapabilityPublishesOnceAndRejectsFailedObservation
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;HardwareMetadata;StreamTimeline;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmStreamWaitValueCapabilityReplay)

include(${CMAKE_CURRENT_LIST_DIR}/V2ROCmMoEWaveReductionTests.cmake)
