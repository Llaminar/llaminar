# Exercise the production process authority/pool with hardware-free worker factories.
add_executable(v2_unit_gpu_device_process_lifetime
    unit/backends/Test__GPUDeviceProcessLifetime.cpp)
target_include_directories(v2_unit_gpu_device_process_lifetime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_gpu_device_process_lifetime PRIVATE cxx_std_20)
target_link_libraries_with_groups(v2_unit_gpu_device_process_lifetime
    llaminar2_core GTest::gtest Threads::Threads)
add_v2_test(V2_Unit_GPUDeviceProcessLifetime
    COMMAND $<TARGET_FILE:v2_unit_gpu_device_process_lifetime>
    LABELS "V2;Unit;DeviceFree;Ownership;RuntimeRetirement;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUDeviceProcessLifetimeReplay
    COMMAND $<TARGET_FILE:v2_unit_gpu_device_process_lifetime>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;RuntimeRetirement;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUDeviceProcessLifetimeReplay)
