include(${CMAKE_CURRENT_LIST_DIR}/V2NativeModuleOwnership.cmake)
# Native HIP membership proof complements the shared device-free owner replay.
if(HAVE_ROCM)
    add_executable(v2_integration_hip_projection_capture_lifetime
        integration/backends/Test__HIPProjectionCaptureLifetime.cpp)
    v2_link_native_test_core(v2_integration_hip_projection_capture_lifetime rocm_backend
        GTest::gtest_main hip::host)
    set_target_properties(v2_integration_hip_projection_capture_lifetime PROPERTIES
        V2_TEST_DEVICE_SCOPE FullInventory)
    target_compile_features(v2_integration_hip_projection_capture_lifetime PRIVATE cxx_std_20)
    add_v2_test(V2_Integration_HIPProjectionCaptureLifetimeNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_hip_projection_capture_lifetime>
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;GraphCapture;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_HIPProjectionCaptureLifetimeNativeReplay)
endif()

# Header-only contract compilation borrows selected SDK types, not runtime linkage.
if(HAVE_ROCM)
    add_executable(v2_unit_hip_graph_capture_owner_types
        unit/backends/Test__HIPGraphCaptureOwnerTypes.cpp)
    target_compile_features(v2_unit_hip_graph_capture_owner_types PRIVATE cxx_std_20)
    target_compile_definitions(v2_unit_hip_graph_capture_owner_types PRIVATE HAVE_ROCM=1
        "$<TARGET_PROPERTY:hip::host,INTERFACE_COMPILE_DEFINITIONS>"
        "$<TARGET_PROPERTY:hip::amdhip64,INTERFACE_COMPILE_DEFINITIONS>")
    target_include_directories(v2_unit_hip_graph_capture_owner_types PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
    target_include_directories(v2_unit_hip_graph_capture_owner_types SYSTEM PRIVATE
        "$<TARGET_PROPERTY:hip::host,INTERFACE_INCLUDE_DIRECTORIES>"
        "$<TARGET_PROPERTY:hip::amdhip64,INTERFACE_INCLUDE_DIRECTORIES>")
    target_link_libraries(v2_unit_hip_graph_capture_owner_types PRIVATE GTest::gtest_main)
    add_v2_test(V2_Unit_HIPGraphCaptureOwnerTypes
        COMMAND $<TARGET_FILE:v2_unit_hip_graph_capture_owner_types>
        LABELS "V2;Unit;DeviceFree;ROCm;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    add_v2_test(V2_Integration_HIPGraphCaptureOwnerTypesReplay
        COMMAND $<TARGET_FILE:v2_unit_hip_graph_capture_owner_types>
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_HIPGraphCaptureOwnerTypesReplay)
endif()
