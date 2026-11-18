#[=[
@file V2RetainedM1HCFFNTests.cmake
@brief Register the native retained-M1 replay, movement, and ABI owners.
]=]
include_guard(DIRECTORY)

add_v2_test(V2_Unit_RetainedClosedHCFFNSummary
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/kernels/rocm/castagna/test_retained_closed_summary.py
    LABELS "V2;Unit;DeviceFree;Benchmark;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15)

set(LLAMINAR_CASTAGNA_SOURCE_DIR "" CACHE PATH
    "Optional pinned Castagna checkout for the full-region native A/B diagnostic (no download)")
if(NOT HAVE_ROCM)
    if(NOT LLAMINAR_CASTAGNA_SOURCE_DIR STREQUAL "")
        message(FATAL_ERROR "Castagna kernel A/B was requested but the canonical ROCm compiler is unavailable")
    endif()
    return()
endif()

# The explicit external comparator request admits its canonical test owner in
# Release, where ordinary integration executables are omitted.
if(V2_PERF_TESTS_ONLY AND NOT LLAMINAR_CASTAGNA_SOURCE_DIR STREQUAL "")
    set(_retained_m1_release_owners "${LLAMINAR_RELEASE_NATIVE_TEST_TARGETS}")
    list(APPEND LLAMINAR_RELEASE_NATIVE_TEST_TARGETS v2_integration_moe_intermediate_ffn_finalization)
endif()

add_executable(v2_integration_moe_intermediate_ffn_finalization
    integration/loaders/Test__MoEIntermediateShardFFNFinalization.cpp
    integration/mpi_gtest_main.cpp
    integration/kernels/rocm/castagna/SelectedDescriptorObserver.hip)
if(TARGET v2_integration_moe_intermediate_ffn_finalization)
    if(DEFINED _retained_m1_release_owners)
        set_property(TARGET v2_integration_moe_intermediate_ffn_finalization PROPERTY EXCLUDE_FROM_ALL FALSE)
    endif()
    set_source_files_properties(integration/kernels/rocm/castagna/SelectedDescriptorObserver.hip PROPERTIES LANGUAGE HIP)
    target_compile_definitions(v2_integration_moe_intermediate_ffn_finalization PRIVATE LLAMINAR_CASTAGNA_FAST_EPOCH=1)
    set_target_properties(v2_integration_moe_intermediate_ffn_finalization PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
    target_link_libraries(v2_integration_moe_intermediate_ffn_finalization PRIVATE llaminar2_core GTest::gtest)
    target_include_directories(v2_integration_moe_intermediate_ffn_finalization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    v2_test_device_scope(v2_integration_moe_intermediate_ffn_finalization FullInventory)

    add_v2_test(V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmRetainedClosedReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Hyperconnection;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1;NativeISA-gfx906;DemandedOutputs"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
    if(TEST V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm)
    endif()

    if(V2_TEST_SOCKETS GREATER_EQUAL 2)
        add_v2_test(V2_Integration_FullNativeHCFFN_RetainedClosedMovement_ROCm
            COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                --gtest_filter=FullNativeHCFFN.ROCmRetainedFastGPUOriginalCohort2EpochSwitchReplay
            LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Hyperconnection;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;HardwareOptional;Devices6;NativeISA-gfx906"
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 180
            ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
        if(TEST V2_Integration_FullNativeHCFFN_RetainedClosedMovement_ROCm)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_RetainedClosedMovement_ROCm)
        endif()
    endif()
endif()

add_executable(v2_integration_retained_m1_descriptor_demand_host_abi
    integration/loaders/ROCmRetainedM1DescriptorDemandHostABI.cpp)
if(TARGET v2_integration_retained_m1_descriptor_demand_host_abi)
    target_link_libraries(v2_integration_retained_m1_descriptor_demand_host_abi PRIVATE llaminar2_core)
    add_v2_test(V2_Integration_ROCmRetainedM1DescriptorDemandHostABI
        COMMAND $<TARGET_FILE:v2_integration_retained_m1_descriptor_demand_host_abi>
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;HostABI;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    if(TEST V2_Integration_ROCmRetainedM1DescriptorDemandHostABI)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmRetainedM1DescriptorDemandHostABI)
    endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/V2CastagnaKernelABTests.cmake")

if(DEFINED _retained_m1_release_owners)
    set(LLAMINAR_RELEASE_NATIVE_TEST_TARGETS "${_retained_m1_release_owners}")
    unset(_retained_m1_release_owners)
endif()
