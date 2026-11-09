#[=[
@file V2CPUExecutionExactPolicyTests.cmake
@brief Register exactly the execution-policy aliases supported by this CPU build.

Registration and explicit preflight membership share this one projection. An
AVX2 companion cannot register or require an AVX512 executable; an AVX512
companion retains both native runtime-dispatch controls.
Unit and preflight use separate scoped registrations for the same native replay;
the installed-image inventory requires Unit names and Unit labels to agree.
]=]
include_guard(DIRECTORY)
set(V2_CPU_EXECUTION_EXACT_POLICY_PREFLIGHT_TESTS)
foreach(V2_EXECUTION_POLICY_ISA AVX2 AVX512)
    if(V2_EXECUTION_POLICY_ISA STREQUAL "AVX512" AND NOT LLAMINAR_CPU_ISA_UPPER STREQUAL "AVX512")
        continue()
    endif()
    string(TOLOWER "${V2_EXECUTION_POLICY_ISA}" V2_EXECUTION_POLICY_RUNTIME_ISA)
    foreach(V2_EXECUTION_POLICY_SCOPE Unit Integration)
        set(V2_EXECUTION_POLICY_PREFLIGHT_LABEL "")
        if(V2_EXECUTION_POLICY_SCOPE STREQUAL "Integration")
            set(V2_EXECUTION_POLICY_PREFLIGHT_LABEL ProductionTestPreflight)
        endif()
        set(V2_EXECUTION_POLICY_TEST
            "V2_${V2_EXECUTION_POLICY_SCOPE}_CPU_NativeQuantisedExecutionExactPolicy_${V2_EXECUTION_POLICY_ISA}")
        add_v2_test(${V2_EXECUTION_POLICY_TEST}
            COMMAND $<TARGET_FILE:v2_integration_cpu_native_quantised_fused_completion>
                --gtest_filter=CPUNativeQuantisedExecutionExactPolicy.*
            LABELS "V2;${V2_EXECUTION_POLICY_SCOPE};${V2_EXECUTION_POLICY_PREFLIGHT_LABEL};CPU;DeviceFree;NativeQuantised;Dispatch;MTP;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_ISA_LEVEL=${V2_EXECUTION_POLICY_RUNTIME_ISA}" "OMP_NUM_THREADS=1")
        if(V2_EXECUTION_POLICY_SCOPE STREQUAL "Integration")
            list(APPEND V2_CPU_EXECUTION_EXACT_POLICY_PREFLIGHT_TESTS ${V2_EXECUTION_POLICY_TEST})
        endif()
    endforeach()
endforeach()
