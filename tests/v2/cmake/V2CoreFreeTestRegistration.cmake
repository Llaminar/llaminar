# @file V2CoreFreeTestRegistration.cmake
# @brief Share core-free test declarations between the complete V2 and native port builds.
#
# The complete build retains its ordinary process owner, gate dependencies and
# labels through add_v2_test. The explicit test-only portability project uses
# plain CTest over the identical production sources; it does not certify the
# full engine or replace the complete Unit/preflight inventory. Undefined-
# behavior sanitizer diagnostics are fatal: a recovering runtime must never
# turn a GTest/CTest success into a false sanitizer-clean receipt. Named shared
# test resources follow that same atomic registration/omission boundary.
include_guard(GLOBAL)

# @brief Register a device-free test with the available directory's test authority.
# @param name Stable test name shared by both build contexts.
# @param ARGN COMMAND and LABELS lists, parsed without changing argument boundaries.
function(v2_add_core_free_test name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "COMMAND;LABELS;RESOURCE_LOCK")
    if(COMMAND add_v2_test)
        # The directory authority owns registration and intentional omission
        # atomically (notably perf-only builds); never decorate an absent test.
        add_v2_test(${name} COMMAND ${ARG_COMMAND} LABELS ${ARG_LABELS}
            NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30
            ENVIRONMENT "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
            RESOURCE_LOCK ${ARG_RESOURCE_LOCK})
    else()
        add_test(NAME ${name} COMMAND ${ARG_COMMAND})
        # Failure reporting only: no floating-point/kernel arithmetic changes.
        set_tests_properties(${name} PROPERTIES LABELS "${ARG_LABELS}" TIMEOUT 30
            ENVIRONMENT "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
            RESOURCE_LOCK "${ARG_RESOURCE_LOCK}")
    endif()
endfunction()
