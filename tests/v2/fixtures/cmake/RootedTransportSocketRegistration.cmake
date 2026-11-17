#[[
@file RootedTransportSocketRegistration.cmake
@brief Exercise production registration of the CPU collectives that require two physical sockets.
This device-free fixture records the original MPI launch declarations for both
host shapes. The regular two-socket preflight executes their native bodies.
]]
cmake_minimum_required(VERSION 3.27)

# Capture the two launch contracts at the same add_v2_test boundary used in production.
function(add_v2_test name)
    cmake_parse_arguments(REPLAY "NO_MODELS" "MPI_PROCS;TIMEOUT;RESOURCE_LOCK" "COMMAND;LABELS;ENVIRONMENT" ${ARGN})
    if(NOT REPLAY_MPI_PROCS EQUAL 2 OR NOT "ProductionTestPreflight" IN_LIST REPLAY_LABELS)
        message(FATAL_ERROR "Missing two-rank preflight ownership for ${name}")
    endif()
    list(APPEND registered "${name}")
    set(registered "${registered}" PARENT_SCOPE)
endfunction()

foreach(V2_TEST_SOCKETS IN ITEMS 1 2)
    set(registered "")
    set(V2_ROOTED_TRANSPORT_SOCKET_REPLAY_TESTS "")
    set(expected "")
    if(V2_TEST_SOCKETS EQUAL 2)
        set(expected V2_Integration_CPURootedGatherParallelCopyReplay V2_Integration_ShmemSpinAbsorbingAbortReplay)
    endif()
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/V2RootedTransportSocketReplayTests.cmake")
    if(NOT "${registered}" STREQUAL "${expected}" OR NOT "${V2_ROOTED_TRANSPORT_SOCKET_REPLAY_TESTS}" STREQUAL "${expected}")
        message(FATAL_ERROR "Unexpected rooted preflight registration for ${V2_TEST_SOCKETS} sockets: ${registered}")
    endif()
endforeach()
