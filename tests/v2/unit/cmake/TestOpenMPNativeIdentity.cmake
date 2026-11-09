# @file TestOpenMPNativeIdentity.cmake
# @brief Exercise ELF/Mach-O runtime identity extraction and fail-closed admission.
#
# Mock native readers make both formats testable on either Unix host. They
# validate the reader arguments and exercise the production CMake function;
# these are parser/selection tests, not a substitute for native loader preflight.
cmake_minimum_required(VERSION 3.18)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/OpenMPRuntimeBinding.cmake")
if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "Set WORK_DIR to an isolated binary-tree test directory")
endif()

if(DEFINED IDENTITY_CHILD)
    file(MAKE_DIRECTORY "${WORK_DIR}")
    set(reader "${WORK_DIR}/reader")
    if(IDENTITY_CHILD MATCHES "^elf")
        set(APPLE FALSE)
        set(CMAKE_READELF "${reader}")
        set(OpenMP_CXX_LIBRARIES "${WORK_DIR}/libomp.so.5" "${WORK_DIR}/libpthread.so.0")
        file(WRITE "${reader}" [=[#!/bin/sh
[ "$1" = '-d' ] || exit 2
printf '%s\n' ' 0x000000000000000e (SONAME) Library soname: [libomp.so.5]'
]=])
        set(expected "libomp.so.5")
    else()
        set(APPLE TRUE)
        set(CMAKE_OTOOL "${reader}")
        set(OpenMP_CXX_LIBRARIES "${WORK_DIR}/libomp.dylib" "${WORK_DIR}/libpthread.dylib")
        file(WRITE "${reader}" [=[#!/bin/sh
[ "$1" = '-D' ] || exit 2
printf '%s:\n@rpath/%s\n' "$2" "${2##*/}"
]=])
        set(expected "@rpath/libomp.dylib")
    endif()
    if(IDENTITY_CHILD STREQUAL "duplicate")
        list(APPEND OpenMP_CXX_LIBRARIES "${WORK_DIR}/libomp.dylib")
    elseif(IDENTITY_CHILD STREQUAL "ambiguous")
        list(APPEND OpenMP_CXX_LIBRARIES "${WORK_DIR}/libiomp5.dylib")
    elseif(IDENTITY_CHILD STREQUAL "missing")
        set(OpenMP_CXX_LIBRARIES "${WORK_DIR}/libpthread.dylib")
    elseif(IDENTITY_CHILD MATCHES "reader_failure$")
        file(WRITE "${reader}" "#!/bin/sh\necho 'fixture reader failure' >&2\nexit 1\n")
    elseif(IDENTITY_CHILD MATCHES "malformed$")
        file(WRITE "${reader}" "#!/bin/sh\nprintf '%s\\n' 'not a runtime identity'\n")
    endif()
    execute_process(COMMAND chmod +x "${reader}" RESULT_VARIABLE chmod_result)
    if(NOT chmod_result EQUAL 0)
        message(FATAL_ERROR "Cannot prepare identity-reader fixture")
    endif()
    llaminar_openmp_runtime_soname(identity)
    if(NOT identity STREQUAL expected)
        message(FATAL_ERROR "Incorrect runtime identity: '${identity}', expected '${expected}'")
    endif()
    message(STATUS "PASS: ${IDENTITY_CHILD}")
    return()
endif()

# @brief Run one isolated fixture and require either an exact success or diagnostic.
# @param name Fixture selector for the production-reader invocation above.
# @param diagnostic Empty for success, otherwise the required failure text.
function(check_identity_case name diagnostic)
    execute_process(COMMAND "${CMAKE_COMMAND}" "-DIDENTITY_CHILD=${name}"
        "-DWORK_DIR=${WORK_DIR}/${name}" -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(diagnostic STREQUAL "")
        if(NOT result EQUAL 0 OR NOT output MATCHES "PASS: ${name}")
            message(FATAL_ERROR "Expected identity success: ${output}${error}")
        endif()
    elseif(result EQUAL 0 OR NOT "${output}${error}" MATCHES "${diagnostic}")
        message(FATAL_ERROR "Expected identity rejection '${diagnostic}': ${output}${error}")
    endif()
endfunction()
check_identity_case(macho "")
check_identity_case(duplicate "")
check_identity_case(elf "")
check_identity_case(missing "Select exactly one dynamic CPU OpenMP runtime")
check_identity_case(ambiguous "Select exactly one dynamic CPU OpenMP runtime")
foreach(name IN ITEMS reader_failure malformed elf_reader_failure elf_malformed)
    check_identity_case("${name}" "Cannot authenticate selected OpenMP runtime")
endforeach()
message(STATUS "PASS: ELF and Mach-O OpenMP identity selection, ambiguity and failed readers")
