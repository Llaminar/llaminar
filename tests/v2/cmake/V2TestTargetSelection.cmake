# Test executable selection for the production Release build.
#
# The test declarations are shared by Integration and Release, but Release owns
# only v2_perf_* executables by default. Explicit executable/proof owners retain their
# original sources and core ABI while remaining outside the default all target.
# Keep target creation and its property setters in the same scope: a skipped Unit executable must not break configuration when
# its declaration supplies language features instead of inheriting them from
# llaminar2_core. Existing targets still use CMake's ordinary validation.
include_guard(DIRECTORY)

set(LLAMINAR_RELEASE_NATIVE_TEST_TARGETS "" CACHE STRING
    "Explicit native test executable and marked proof targets to admit in Release (semicolon-separated; empty keeps performance-only selection; selected owners are excluded from default all)")

# This option admits declared test owners, not alternative build types or
# compile definitions. Alias selection reuses the exact target identity carried
# by each original COMMAND, including executables passed to script drivers.
function(v2_test_target_selected target result)
    if(NOT V2_PERF_TESTS_ONLY OR "${target}" MATCHES "^v2_perf_" OR
       "${target}" IN_LIST LLAMINAR_RELEASE_NATIVE_TEST_TARGETS)
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(v2_test_registration_selected test_name result)
    if(NOT V2_PERF_TESTS_ONLY OR "${test_name}" MATCHES "^V2_Perf_")
        set(${result} TRUE PARENT_SCOPE)
        return()
    endif()
    set(selected FALSE)
    set(missing "")
    foreach(argument IN LISTS ARGN)
        set(owner "")
        if(argument MATCHES "\\$<TARGET_FILE:([^>]+)>")
            set(owner "${CMAKE_MATCH_1}")
        elseif(TARGET "${argument}")
            set(owner "${argument}")
        endif()
        if(NOT owner STREQUAL "")
            if("${owner}" IN_LIST LLAMINAR_RELEASE_NATIVE_TEST_TARGETS)
                set(selected TRUE)
            elseif(NOT TARGET "${owner}")
                list(APPEND missing "${owner}")
            endif()
        endif()
    endforeach()
    if(selected AND missing)
        message(FATAL_ERROR
            "Selected Release native test ${test_name} also requires unselected test target(s): ${missing}")
    endif()
    set(${result} ${selected} PARENT_SCOPE)
endfunction()

function(v2_validate_release_native_test_targets)
    set(missing_owners "")
    foreach(requested IN LISTS LLAMINAR_RELEASE_NATIVE_TEST_TARGETS)
        if(NOT TARGET "${requested}")
            list(APPEND missing_owners "${requested}")
        endif()
    endforeach()
    if(missing_owners)
        list(JOIN missing_owners ", " missing_owners_text)
        message(FATAL_ERROR "Requested Release native test target was not declared: ${missing_owners_text}")
    endif()
    foreach(requested IN LISTS LLAMINAR_RELEASE_NATIVE_TEST_TARGETS)
        get_target_property(kind "${requested}" TYPE)
        get_target_property(imported "${requested}" IMPORTED)
        get_target_property(alias "${requested}" ALIASED_TARGET)
        get_target_property(declared "${requested}" V2_RELEASE_NATIVE_TEST_OWNER)
        get_target_property(proof_owner "${requested}" V2_RELEASE_NATIVE_PROOF_OWNER)
        set(original_owner FALSE)
        if(kind STREQUAL "EXECUTABLE" AND declared)
            set(original_owner TRUE)
        elseif(kind STREQUAL "UTILITY" AND proof_owner)
            # Original source/DSO proof commands are opt-in owners too. An
            # arbitrary dependency/custom target has no such declaration.
            set(original_owner TRUE)
        endif()
        if(NOT original_owner OR imported OR alias)
            message(FATAL_ERROR
                "Requested Release native test target is not an original declared test executable or proof owner: ${requested}")
        endif()
    endforeach()
endfunction()
if(V2_PERF_TESTS_ONLY)
    # Declarations may occur after their registrations. Validate only after the
    # entire original test directory is known, before generation can succeed.
    cmake_language(DEFER CALL v2_validate_release_native_test_targets)
endif()

# Keep startup device ownership at the same boundary as target selection.
# Release excludes unselected non-performance executables. Other missing targets and
# unknown scopes remain errors, never permission to lose device ownership.
function(v2_test_device_scope target scope)
    if(NOT scope STREQUAL "FullInventory" AND NOT scope STREQUAL "DeclaredBackends")
        message(FATAL_ERROR "Unknown V2 test device scope '${scope}' for ${target}")
    endif()
    v2_test_target_selected("${target}" selected)
    if(NOT selected)
        return()
    endif()
    set_property(TARGET ${target} PROPERTY V2_TEST_DEVICE_SCOPE "${scope}")
endfunction()

if(V2_PERF_TESTS_ONLY)
    message(STATUS "V2 Tests: Release mode — performance tests plus explicit native owners: ${LLAMINAR_RELEASE_NATIVE_TEST_TARGETS}")

    macro(add_executable target)
        v2_test_target_selected("${target}" selected)
        if(selected)
            _add_executable(${target} ${ARGN})
            if("${target}" IN_LIST LLAMINAR_RELEASE_NATIVE_TEST_TARGETS)
                get_target_property(imported "${target}" IMPORTED)
                get_target_property(alias "${target}" ALIASED_TARGET)
                if(NOT imported AND NOT alias)
                    set_property(TARGET ${target} PROPERTY V2_RELEASE_NATIVE_TEST_OWNER TRUE)
                    if(NOT "${target}" MATCHES "^v2_perf_")
                        set_property(TARGET ${target} PROPERTY EXCLUDE_FROM_ALL TRUE)
                    endif()
                endif()
            endif()
        endif()
    endmacro()

    macro(target_link_libraries target)
        if(TARGET ${target})
            _target_link_libraries(${target} ${ARGN})
        endif()
    endmacro()

    macro(target_include_directories target)
        if(TARGET ${target})
            _target_include_directories(${target} ${ARGN})
        endif()
    endmacro()

    macro(target_compile_definitions target)
        if(TARGET ${target})
            _target_compile_definitions(${target} ${ARGN})
        endif()
    endmacro()

    macro(target_compile_options target)
        if(TARGET ${target})
            _target_compile_options(${target} ${ARGN})
        endif()
    endmacro()

    macro(target_compile_features target)
        if(TARGET ${target})
            _target_compile_features(${target} ${ARGN})
        endif()
    endmacro()
endif()
