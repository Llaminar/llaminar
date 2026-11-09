#[=[
@file V2CPUExecutionExactPolicyTests.cmake
@brief Register exactly the execution-policy aliases supported by this CPU build.

Registration and explicit preflight membership share this one projection. An
AVX2 companion cannot register or require an AVX512 executable; an AVX512
companion retains both native runtime-dispatch controls.
]=]
include_guard(DIRECTORY)
set(V2_CPU_EXECUTION_EXACT_POLICY_PREFLIGHT_TESTS)
foreach(V2_EXECUTION_POLICY_ISA AVX2 AVX512)
    if(V2_EXECUTION_POLICY_ISA STREQUAL "AVX512" AND NOT LLAMINAR_CPU_ISA_UPPER STREQUAL "AVX512")
        continue()
    endif()
    string(TOLOWER "${V2_EXECUTION_POLICY_ISA}" V2_EXECUTION_POLICY_RUNTIME_ISA)
    set(V2_EXECUTION_POLICY_TEST
        "V2_Integration_CPU_NativeQuantisedExecutionExactPolicy_${V2_EXECUTION_POLICY_ISA}")
    add_v2_test(${V2_EXECUTION_POLICY_TEST}
        COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
            --gtest_filter=CPUNativeQuantisedExecutionExactPolicy.*
        LABELS "V2;Unit;Integration;ProductionTestPreflight;CPU;NativeQuantised;Dispatch;MTP;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_EXECUTION_POLICY_RUNTIME_ISA}" "OMP_NUM_THREADS=1")
    list(APPEND V2_CPU_EXECUTION_EXACT_POLICY_PREFLIGHT_TESTS ${V2_EXECUTION_POLICY_TEST})
endforeach()
