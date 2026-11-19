#[=[
@file V2ROCmRetainedM1NumericsTests.cmake
@brief Register retained MI50 arithmetic and grouped verifier wave-address replay in production preflight.
]=]
include_guard(DIRECTORY)
if(NOT HAVE_ROCM)
    return()
endif()

set_source_files_properties(integration/kernels/rocm/Test__ROCmRetainedM1Numerics.hip PROPERTIES LANGUAGE HIP)
add_executable(v2_integration_rocm_retained_m1_numerics
    integration/kernels/rocm/Test__ROCmRetainedM1Numerics.hip)
if(NOT TARGET v2_integration_rocm_retained_m1_numerics)
    return()
endif()
target_link_libraries(v2_integration_rocm_retained_m1_numerics PRIVATE
    llaminar2_core v2_core_gtest_main hip::host)
set_target_properties(v2_integration_rocm_retained_m1_numerics PROPERTIES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
v2_test_device_scope(v2_integration_rocm_retained_m1_numerics FullInventory)

set(qfn_wave_address_test ROCmRetainedM1Numerics.CapturedWaveAddressPreservesUnsignedHalves)
add_v2_test(V2_Integration_ROCmRetainedM1Numerics
    COMMAND $<TARGET_FILE:v2_integration_rocm_retained_m1_numerics>
        --gtest_filter=ROCmRetainedM1Numerics.*-${qfn_wave_address_test}
    LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;Regression;Devices1"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 90
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
if(TEST V2_Integration_ROCmRetainedM1Numerics)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmRetainedM1Numerics)
endif()


add_v2_test(V2_Integration_ROCmGroupedVerifierWaveAddressReplay
    COMMAND $<TARGET_FILE:v2_integration_rocm_retained_m1_numerics>
        --gtest_filter=${qfn_wave_address_test}
    LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;Regression;Devices1"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 90
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
if(TEST V2_Integration_ROCmGroupedVerifierWaveAddressReplay)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmGroupedVerifierWaveAddressReplay)
endif()
unset(qfn_wave_address_test)
