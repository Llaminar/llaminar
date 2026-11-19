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

add_v2_test(V2_Unit_VerifierRooflineMeasurement
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/qwen4exp/test_verifier_roofline.py
    LABELS "V2;Unit;DeviceFree;Benchmark;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15)

add_v2_test(V2_Unit_VerifierFFNArrivals
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/qwen4exp/test_ffn_arrivals.py
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

    # Required MI50 replays keep native selection in the fixture. NativeISA
    # labels are reserved for scheduler-optional hardware geometries.
    add_v2_test(V2_Integration_ROCmGroupedFFNFormatProgramDemand
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=ROCmGroupedFFNFormatProgramNative.RejectsForeignRuntimeAndIncorrectDemandBeforeRecording
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;NativeQuantised;GraphCapture;Replay;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_ALL_FORMATS=1;LLAMINAR_PERF_STATS_SUMMARY=1;LLAMINAR_PERF_STATS_FILTER=kernel.rocm_grouped_ffn_frontier_calls")
    if(TEST V2_Integration_ROCmGroupedFFNFormatProgramDemand)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmGroupedFFNFormatProgramDemand)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmRetainedClosedReplay:FullNativeHCFFN.ROCmRetainedBalancedTP2Replay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Hyperconnection;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1;DemandedOutputs"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
    if(TEST V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_RetainedClosed_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_GroupedQ8DownReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmGroupedQ8DownByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
    if(TEST V2_Integration_FullNativeHCFFN_GroupedQ8DownReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_GroupedQ8DownReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_GroupedFrontierReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmGroupedFFNFrontierByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=0;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=0")
    if(TEST V2_Integration_FullNativeHCFFN_GroupedFrontierReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_GroupedFrontierReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_GroupedFFNDirectBankReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmGroupedFFNFrontierByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=0")
    if(TEST V2_Integration_FullNativeHCFFN_GroupedFFNDirectBankReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_GroupedFFNDirectBankReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_GroupedFFNHeldEpochReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmGroupedFFNHeldEpochReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=0;LLAMINAR_QFN_ROCM_GROUPED_FFN_ALL_FORMATS=1;LLAMINAR_PERF_STATS_SUMMARY=1")
    if(TEST V2_Integration_FullNativeHCFFN_GroupedFFNHeldEpochReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_GroupedFFNHeldEpochReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmEqualF160GroupedFFNFrontierByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=1")
    if(TEST V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierDefaultReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmEqualF160GroupedFFNDefaultByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;Capture;MTP;MoE;ByteExact;Replay;Regression"
        ENVIRONMENT "LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0;LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120 RESOURCE_LOCK rocm_gpu_0)
    if(TEST V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierDefaultReplay_ROCm)
        set_property(TEST V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierDefaultReplay_ROCm
            APPEND PROPERTY ENVIRONMENT_MODIFICATION
                "LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=unset:"
                "LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=unset:"
                "LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=unset:"
                "LLAMINAR_QFN_ROCM_GROUPED_FFN_ALL_FORMATS=unset:"
                "LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS=unset:"
                "LLAMINAR_QFN_ROCM_GROUPED_FP16_ROUTER_FIXED_ROWS=unset:"
                "LLAMINAR_QFN_ROCM_GROUPED_TOPK_REGISTERS=unset:")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierDefaultReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_QFNFixedMTPConditionRowReplay
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmFixedConditionRowReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;Capture;MTP;MoE;ByteExact;Replay;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120 RESOURCE_LOCK rocm_gpu_0
        ENVIRONMENT "LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0;LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
    if(TEST V2_Integration_QFNFixedMTPConditionRowReplay)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNFixedMTPConditionRowReplay)
    endif()

    add_v2_test(V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierK32SumReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=FullNativeHCFFN.ROCmEqualF160GroupedFFNFrontierByteReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS=1")
    if(TEST V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierK32SumReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierK32SumReplay_ROCm)
    endif()

    # Assert both captured symbols and production counters in the focused replays.
    # The separate matrix runner times every original source/role combination.
    set(grouped_m4_all_formats_env
        LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16
        LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1
        LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1
        LLAMINAR_QFN_ROCM_GROUPED_FFN_ALL_FORMATS=1
        LLAMINAR_TEST_GROUPED_M4_PROGRAM=all_formats
        LLAMINAR_TEST_GROUPED_M4_REQUIRED_SYMBOL=groupedFFNFormatProgramGate
        LLAMINAR_TEST_GROUPED_M4_SYMBOL_PRESENT=1
        LLAMINAR_PERF_STATS_SUMMARY=1
        LLAMINAR_PERF_STATS_FILTER=kernel.rocm_grouped_ffn_frontier_calls
    )
    add_v2_test(V2_Integration_GroupedM4SourceCountedReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=GroupedM4SourceInventory.CoversOriginalCatalog:OriginalSources/GroupedM4Source.Static/Q4_K_routed_gate_up:OriginalSources/GroupedM4Source.Deferred/Q4_K_routed_gate_up:OriginalSources/GroupedM4Source.Static/Q8_1_source_boundary
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT ${grouped_m4_all_formats_env})
    if(TEST V2_Integration_GroupedM4SourceCountedReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GroupedM4SourceCountedReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_GroupedM6SourceCountedReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=GroupedM6Source.CountedReplay
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT ${grouped_m4_all_formats_env})
    if(TEST V2_Integration_GroupedM6SourceCountedReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GroupedM6SourceCountedReplay_ROCm)
    endif()

    set(grouped_m6_mixed_reader_env ${grouped_m4_all_formats_env})
    list(FILTER grouped_m6_mixed_reader_env EXCLUDE REGEX "^LLAMINAR_TEST_GROUPED_M4_REQUIRED_SYMBOL=")
    # Keep both precise route/format combinations in each bounded replay.
    foreach(replay IN ITEMS IndexedGateReuseReplay IndexedGateReuseG11G12Replay)
        if(replay STREQUAL "IndexedGateReuseReplay")
            set(reuse_filter "OriginalSources/GroupedM6Family.Deferred/G09_*:OriginalSources/GroupedM6Family.Static/G10_*")
        else()
            set(reuse_filter "OriginalSources/GroupedM6Family.Static/G11_*:OriginalSources/GroupedM6Family.Deferred/G12_*")
        endif()
        set(reuse_test "V2_Integration_GroupedM6${replay}_ROCm")
        add_v2_test(${reuse_test}
            COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                --gtest_filter=${reuse_filter}
            LABELS "V2;Integration;ProductionTestPreflight;ROCm;MTP;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
            ENVIRONMENT ${grouped_m6_mixed_reader_env}
                LLAMINAR_TEST_GROUPED_M4_REQUIRED_SYMBOL=groupedFFNFormatProgramGateILj15ELin1ELi0ELi1ELb0EE)
        if(TEST ${reuse_test})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${reuse_test})
        endif()
    endforeach()

    add_v2_test(V2_Integration_GroupedM6MixedReaderReplay_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=OriginalSources/GroupedM6Family.Static/G15_IQ3_XXS_DownQ6_K_SharedIQ2_S_DownQ8_0:OriginalSources/GroupedM6Family.Deferred/G15_IQ3_XXS_DownQ6_K_SharedIQ2_S_DownQ8_0:OriginalSources/GroupedM6Family.Deferred/G13_IQ3_XXS_DownQ5_1_SharedIQ2_S
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MTP;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT ${grouped_m6_mixed_reader_env}
            LLAMINAR_QFN_ROCM_GROUPED_FP16_ROUTER_FIXED_ROWS=1 LLAMINAR_QFN_ROCM_GROUPED_TOPK_REGISTERS=1
            LLAMINAR_TEST_GROUPED_M4_REQUIRED_SYMBOL=groupedFFNFormatProgramGateILj15ELi12ELi0ELi1ELb1EE)
    if(TEST V2_Integration_GroupedM6MixedReaderReplay_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GroupedM6MixedReaderReplay_ROCm)
    endif()

    add_v2_test(V2_Integration_GroupedM4BankPublicationRejects_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=OriginalSources/GroupedM4Source.Static/Q4_K_sparse_routes:OriginalSources/GroupedM4Source.Deferred/Q5_K_sparse_routes
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT ${grouped_m4_all_formats_env})
    if(TEST V2_Integration_GroupedM4BankPublicationRejects_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GroupedM4BankPublicationRejects_ROCm)
    endif()

    foreach(pitch IN ITEMS 128 256)
        set(replay "V2_Integration_GroupedM4Q51DownF${pitch}Replay_ROCm")
        set(replay_case "Q4_K_Q5_1_Down_F${pitch}")
        add_v2_test(${replay}
            COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                --gtest_filter=OriginalSources/GroupedM4Source.Static/${replay_case}:OriginalSources/GroupedM4Source.Deferred/${replay_case}
            LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
            ENVIRONMENT ${grouped_m4_all_formats_env} LLAMINAR_QFN_ROCM_GROUPED_FP16_ROUTER_FIXED_ROWS=1
                LLAMINAR_QFN_ROCM_GROUPED_TOPK_REGISTERS=1)
        if(TEST ${replay})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${replay})
        endif()
    endforeach()

    add_v2_test(V2_Integration_GroupedM4DeepGridPackets_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=OriginalSources/GroupedM4Source.Deferred/Q4_K_deep_grid:OriginalSources/GroupedM4Source.Static/Q4_K_underfilled_grid
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT ${grouped_m4_all_formats_env} LLAMINAR_QFN_ROCM_GROUPED_FP16_ROUTER_FIXED_ROWS=1
            LLAMINAR_QFN_ROCM_GROUPED_TOPK_REGISTERS=1)
    if(TEST V2_Integration_GroupedM4DeepGridPackets_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GroupedM4DeepGridPackets_ROCm)
    endif()

    foreach(pitch IN ITEMS 128 256)
        set(replay "V2_Integration_GroupedM6Q51DownF${pitch}Replay_ROCm")
        add_v2_test(${replay}
            COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                --gtest_filter=OriginalSources/GroupedM6Q51Down.Deferred/F${pitch}
            LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
            ENVIRONMENT ${grouped_m4_all_formats_env} LLAMINAR_QFN_ROCM_GROUPED_FP16_ROUTER_FIXED_ROWS=1
                LLAMINAR_QFN_ROCM_GROUPED_TOPK_REGISTERS=1)
        if(TEST ${replay})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${replay})
        endif()
    endforeach()

    add_v2_test(V2_Integration_GroupedM4WideDecoderTiles_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=OriginalSources/GroupedM4Source.Static/Q5_K_wide_direct_tiles:OriginalSources/GroupedM4Source.Deferred/G15_IQ3_XXS_wide_direct_tiles:OriginalSources/GroupedM4Source.Static/G13_IQ3_XXS_wide_direct_tiles
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT ${grouped_m4_all_formats_env} LLAMINAR_QFN_ROCM_GROUPED_FP16_ROUTER_FIXED_ROWS=1
            LLAMINAR_QFN_ROCM_GROUPED_TOPK_REGISTERS=1)
    if(TEST V2_Integration_GroupedM4WideDecoderTiles_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GroupedM4WideDecoderTiles_ROCm)
    endif()

    add_v2_test(V2_Integration_GroupedM4MeasuredSourceAlias_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_filter=OriginalSources/GroupedM4Source.Static/Q5_1_routed_gate_up
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;ByteExact;Replay;Regression;Devices1"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
        ENVIRONMENT LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16 LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1
            LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1 LLAMINAR_QFN_ROCM_GROUPED_FFN_EQUAL_FRONTIER=1
            LLAMINAR_QFN_ROCM_GROUPED_FFN_ALL_FORMATS=0 LLAMINAR_TEST_GROUPED_M4_PROGRAM=fixed
            LLAMINAR_TEST_GROUPED_M4_REQUIRED_SYMBOL=groupedFFNPairedGatePublish LLAMINAR_TEST_GROUPED_M4_SYMBOL_PRESENT=0
            LLAMINAR_PERF_STATS_SUMMARY=1 LLAMINAR_PERF_STATS_FILTER=kernel.rocm_grouped_ffn_frontier_calls)
    if(TEST V2_Integration_GroupedM4MeasuredSourceAlias_ROCm)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GroupedM4MeasuredSourceAlias_ROCm)
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
            LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;Hyperconnection;GraphCapture;Replay;Regression;ByteExact;GeneratedSource;Devices1"
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
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
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
                MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
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
