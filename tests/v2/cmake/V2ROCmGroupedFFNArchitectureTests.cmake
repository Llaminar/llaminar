# Native captured FFN parity of the shipped MI50 and portable gfx1100 schedules.
include_guard(DIRECTORY)
if(HAVE_ROCM AND TARGET v2_integration_moe_intermediate_ffn_finalization)
    foreach(_arch IN ITEMS Gfx906 Gfx1100)
        string(TOLOWER "${_arch}" _isa)
        set(_name "V2_Integration_ROCmGroupedFFNNativeArchitecture_${_isa}")
        add_v2_test("${_name}"
            COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
                "--gtest_filter=${_arch}/GroupedFFNNativeArchitecture.CountedReplay/*"
            LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;GraphCapture;Replay;ByteExact;GeneratedSource;HardwareOptional;Devices1;NativeISA-${_isa}"
            MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120
            ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16;LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_DIRECT_BANK=1;LLAMINAR_QFN_ROCM_GROUPED_FFN_ALL_FORMATS=1;LLAMINAR_PERF_STATS_SUMMARY=1;LLAMINAR_PERF_STATS_FILTER=kernel.rocm_grouped_ffn_frontier_calls")
        if(TEST "${_name}")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS "${_name}")
        endif()
    endforeach()
endif()
