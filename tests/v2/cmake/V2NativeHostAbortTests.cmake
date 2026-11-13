#[=[
@file V2NativeHostAbortTests.cmake
@brief Replay actual LocalTP cancellation publication without native execution.

The existing LocalTP executable exercises original host waiters, immutable
recorded native identities, idempotent signal publication and deferred backend
reclamation. Dependency-builder native barrier replays and actual GPU capture
remain separate obligations; these control tests cannot certify native readiness.
]=]
include_guard(DIRECTORY)

if(TARGET v2_test_local_tp_context)
    foreach(_native_abort_suite IN ITEMS Unit Integration)
        set(_native_abort_test "V2_${_native_abort_suite}_LocalTPOriginalNativeAbortReplay")
        add_v2_test(${_native_abort_test}
            COMMAND $<TARGET_FILE:v2_test_local_tp_context>
                "--gtest_filter=Test__LocalTPContext.ReadyNativeAbortSignalsAfterHostWakeAndDefersDestruction:Test__LocalTPContext.NativeAbortIgnoresUnreadyOrForeignBackend:Test__LocalTPContext.AbortRequestDefersBackendAbortUntilContextDestruction"
            LABELS "V2;${_native_abort_suite};DeviceFree;LocalTP;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10)
        if(_native_abort_suite STREQUAL "Integration")
            set_property(TEST ${_native_abort_test} APPEND PROPERTY LABELS ProductionTestPreflight)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_native_abort_test})
        endif()
    endforeach()
endif()
