# @file V2CoreFreeTestRegistration.cmake
# @brief Share core-free test declarations between the complete V2 and native port builds.
#
# The complete build retains its ordinary process owner, gate dependencies and
# labels through add_v2_test. The explicit test-only portability project uses
# plain CTest over the identical production sources; it does not certify the
# full engine or replace the complete Unit/preflight inventory.
include_guard(GLOBAL)

# @brief Register a device-free test with the available directory's test authority.
# @param name Stable test name shared by both build contexts.
# @param ARGN COMMAND and LABELS lists, parsed without changing argument boundaries.
function(v2_add_core_free_test name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "COMMAND;LABELS")
    if(COMMAND add_v2_test)
        add_v2_test(${name} COMMAND ${ARG_COMMAND} LABELS ${ARG_LABELS}
            NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30)
    else()
        add_test(NAME ${name} COMMAND ${ARG_COMMAND})
        set_tests_properties(${name} PROPERTIES LABELS "${ARG_LABELS}" TIMEOUT 30)
    endif()
endfunction()
