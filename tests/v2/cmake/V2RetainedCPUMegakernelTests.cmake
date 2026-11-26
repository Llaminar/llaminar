#[[
@file V2RetainedCPUMegakernelTests.cmake
@brief Device-free Unit and ProductionTestPreflight replay tests for QFN CPU megakernels.
]]
add_executable(v2_test_retained_cpu_megakernel_replay
    unit/execution/qwen4exp/Test__RetainedCPUMegakernelReplay.cpp)
target_compile_features(v2_test_retained_cpu_megakernel_replay PRIVATE cxx_std_20)
target_include_directories(v2_test_retained_cpu_megakernel_replay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(v2_test_retained_cpu_megakernel_replay PRIVATE llaminar2_core v2_core_gtest_main GTest::gtest OpenMP::OpenMP_CXX)

add_v2_test(V2_Unit_RetainedCPUMegakernelReplay
    COMMAND $<TARGET_FILE:v2_test_retained_cpu_megakernel_replay>
    LABELS "V2;Unit;DeviceFree;CPU;Qwen4Exp;MoE;Replay"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
        "LLAMINAR_SKIP_ROCM_STARTUP=1")

set(cpu_mk_isa_cells scalar)
foreach(cpu_mk_cell IN LISTS V2_CPU_NATIVE_ISA_CELLS)
    string(TOLOWER "${cpu_mk_cell}" cpu_mk_cell_name)
    list(APPEND cpu_mk_isa_cells "${cpu_mk_cell_name}")
endforeach()
foreach(cpu_mk_isa IN LISTS cpu_mk_isa_cells)
    if(cpu_mk_isa STREQUAL "scalar")
        set(cpu_mk_runtime scalar)
    else()
        string(TOUPPER "${cpu_mk_isa}" cpu_mk_upper)
        v2_cpu_native_isa_runtime("${cpu_mk_upper}" cpu_mk_runtime)
    endif()
    set(cpu_mk_test_name "V2_Integration_RetainedCPUMegakernelReplay_${cpu_mk_isa}")
    add_v2_test(${cpu_mk_test_name}
        COMMAND $<TARGET_FILE:v2_test_retained_cpu_megakernel_replay>
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MoE;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "LLAMINAR_ISA_LEVEL=${cpu_mk_runtime}")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${cpu_mk_test_name})
endforeach()
