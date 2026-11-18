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

add_executable(v2_integration_rocm_routed_grid_layout
    integration/kernels/rocm/Test__ROCmRoutedGridLayout.cpp)
if(TARGET v2_integration_rocm_routed_grid_layout)
    target_link_libraries(v2_integration_rocm_routed_grid_layout PRIVATE llaminar2_core)
    add_v2_test(V2_Integration_ROCmRoutedGridLayout
        COMMAND $<TARGET_FILE:v2_integration_rocm_routed_grid_layout>
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    if(TEST V2_Integration_ROCmRoutedGridLayout)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmRoutedGridLayout)
    endif()
endif()

set(LLAMINAR_QFN_RETAINED_TP_GGUF "" CACHE FILEPATH
    "Optional local first QFN GGUF split for the canonical real-source TP1/2/4/8 replay")
if(NOT LLAMINAR_QFN_RETAINED_TP_GGUF STREQUAL "" AND NOT EXISTS "${LLAMINAR_QFN_RETAINED_TP_GGUF}")
    message(FATAL_ERROR "QFN retained TP replay requires an existing first GGUF split")
endif()

# Explicit real-source requests also admit the native owner in Release.
# The explicit external comparator request admits its canonical test owner in
# Release, where ordinary integration executables are omitted.
if(V2_PERF_TESTS_ONLY AND (NOT LLAMINAR_CASTAGNA_SOURCE_DIR STREQUAL "" OR
    NOT LLAMINAR_QFN_RETAINED_TP_GGUF STREQUAL ""))
    set(_retained_m1_release_owners "${LLAMINAR_RELEASE_NATIVE_TEST_TARGETS}")
    list(APPEND LLAMINAR_RELEASE_NATIVE_TEST_TARGETS v2_integration_moe_intermediate_ffn_finalization
        v2_integration_qfn_retained_tp_source)
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
            --gtest_filter=FullNativeHCFFN.ROCmRetainedClosedReplay:FullNativeHCFFN.ROCmRetainedBalancedTP2Replay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Hyperconnection;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1;NativeISA-gfx906;DemandedOutputs"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
    if(TEST V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_GroupedQ8DownReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmGroupedQ8DownByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1;NativeISA-gfx906"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
    if(TEST V2_Integration_FullNativeHCFFN_GroupedQ8DownReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_GroupedQ8DownReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_GroupedFrontierReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmGroupedFFNFrontierByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1;NativeISA-gfx906"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=0;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=0")
    if(TEST V2_Integration_FullNativeHCFFN_GroupedFrontierReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_GroupedFrontierReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_GroupedFFNDirectBankReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmGroupedFFNFrontierByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1;NativeISA-gfx906"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=0")
    if(TEST V2_Integration_FullNativeHCFFN_GroupedFFNDirectBankReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_GroupedFFNDirectBankReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmEqualF160GroupedFFNFrontierByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1;NativeISA-gfx906"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=1")
    if(TEST V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierReplay_ROCm)
    endif()

    # Rebuild original encodings and select the kernel from its prepared weights.
    # Both correctness modes run under the same rank allocation. Performance
    # uses an independent native event session with only consumer-visible outputs.
    foreach(group RANGE 1 15)
        set(group_id "${group}")
        if(group LESS 10)
            set(group_id "0${group}")
        endif()
        set(family "OriginalSources/FullNativeHCFFNFamily")
        set(case "G${group_id}_*")
        set(replay "V2_Integration_FullNativeHCFFN_G${group_id}_ROCm")
        add_v2_test(${replay}
            COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                --gtest_filter=${family}.Replay/${case}:${family}.DemandedReplay/${case}
            LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Hyperconnection;GraphCapture;Replay;Regression;ByteExact;GeneratedSource;Devices1;NativeISA-gfx906"
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
            ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
        if(TEST ${replay})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${replay})
        endif()
        add_v2_test(V2_Perf_FullNativeHCFFN_G${group_id}_ROCm
            COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                --gtest_also_run_disabled_tests --gtest_filter=${family}.DISABLED_Events/${case}
            LABELS "V2;Performance;ROCm;MoE;Hyperconnection;GraphCapture;ComponentTiming;GeneratedSource;Devices1;NativeISA-gfx906"
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120 RESOURCE_LOCK RealModel_Perf
            ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
    endforeach()

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
        foreach(gate IN ITEMS Q4K Q5K)
            set(replay "V2_Integration_FullNativeHCFFN_RetainedClosedMovement_${gate}Q8_ROCm")
            add_v2_test(${replay}
                COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                    --gtest_filter=FullNativeHCFFN.ROCmRetainedFastGPU${gate}Q8EpochSwitchReplay
                LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Hyperconnection;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;HardwareOptional;Devices6;NativeISA-gfx906"
                MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 180
                ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
            if(TEST ${replay})
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${replay})
            endif()
        endforeach()
    endif()
endif()

if(NOT LLAMINAR_QFN_RETAINED_TP_GGUF STREQUAL "")
    set(_qfn_tp_runner "${CMAKE_CURRENT_SOURCE_DIR}/integration/loaders/run_real_retained_tp.py")
    add_executable(v2_integration_qfn_retained_tp_source
        integration/loaders/RealRetainedTPSourceAudit.cpp)
    if(TARGET v2_integration_qfn_retained_tp_source)
        if(DEFINED _retained_m1_release_owners)
            set_property(TARGET v2_integration_qfn_retained_tp_source PROPERTY EXCLUDE_FROM_ALL FALSE)
        endif()
        target_link_libraries(v2_integration_qfn_retained_tp_source PRIVATE llaminar2_core)
        add_v2_test(V2_Integration_QFNRetainedTP_OriginalSource
            COMMAND ${Python3_EXECUTABLE} ${_qfn_tp_runner}
                "${LLAMINAR_QFN_RETAINED_TP_GGUF}" audit $<TARGET_FILE:v2_integration_qfn_retained_tp_source>
            LABELS "V2;Integration;RealModel;DeviceFree;MoE;ByteExact;TensorParallel"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 90)
    endif()
    if(TARGET v2_integration_moe_intermediate_ffn_finalization)
        foreach(_qfn_tp_width IN ITEMS 1 2 4 8)
            add_v2_test(V2_Integration_QFNRetainedTP${_qfn_tp_width}_LocalSourceWindow_ROCm
                COMMAND ${Python3_EXECUTABLE} ${_qfn_tp_runner}
                    "${LLAMINAR_QFN_RETAINED_TP_GGUF}" window
                    $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization> --width ${_qfn_tp_width}
                LABELS "V2;Integration;RealModel;ROCm;MoE;GraphCapture;Replay;ByteExact;Devices1;NativeISA-gfx906;LocalSourceWindow"
                MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 180
                ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
        endforeach()
        if(V2_TEST_SOCKETS GREATER_EQUAL 2)
            add_v2_test(V2_Integration_QFNRetainedTP4_OriginalSourceMovement_ROCm
                COMMAND ${Python3_EXECUTABLE} ${_qfn_tp_runner}
                    "${LLAMINAR_QFN_RETAINED_TP_GGUF}" movement
                    $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization> --width 4
                LABELS "V2;Integration;RealModel;ROCm;MoE;GraphCapture;Replay;ByteExact;HardwareOptional;Devices6;NativeISA-gfx906;TensorParallel;Movement"
                MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 180
                ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
        endif()
    endif()
    unset(_qfn_tp_runner)
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
