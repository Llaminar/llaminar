# Actual native-thread state binding and frozen original-body controls, with no runtime.
add_executable(v2_unit_native_device_binding
    unit/backends/Test__NativeDeviceBinding.cpp
    unit/backends/Test__NativeDeviceBindingAbsentROCm.cpp)
target_include_directories(v2_unit_native_device_binding PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_native_device_binding PRIVATE cxx_std_20)
target_link_libraries(v2_unit_native_device_binding PRIVATE GTest::gtest_main Threads::Threads)
add_v2_test(V2_Unit_NativeDeviceBinding
    COMMAND $<TARGET_FILE:v2_unit_native_device_binding>
    LABELS "V2;Unit;DeviceFree;CUDA;ROCm;Collective;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_NativeDeviceBindingReplay
    COMMAND $<TARGET_FILE:v2_unit_native_device_binding>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;ROCm;Collective;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeDeviceBindingReplay)

# Use the canonical backend-native target helper: shared core only, or the
# corresponding whole-backend static closure, with the canonical process entry.
add_v2_gpu_backend_test(v2_integration_native_device_binding
    integration/backends/Test__NativeDeviceBindingNative.cpp
    V2_Integration_NativeDeviceBindingNativeReplay
    "V2;Integration;ProductionTestPreflight;Collective;Ownership;Replay;Regression")
foreach(_binding_backend CUDA ROCm)
    if(TEST V2_Integration_NativeDeviceBindingNativeReplay_${_binding_backend})
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
            V2_Integration_NativeDeviceBindingNativeReplay_${_binding_backend})
    endif()
endforeach()
