# Test executable selection for the production Release build.
#
# The test declarations are shared by Integration and Release, but Release owns
# only v2_perf_* executables. Keep target creation and its property setters in
# the same scope: a skipped Unit executable must not break configuration when
# its declaration supplies language features instead of inheriting them from
# llaminar2_core. Existing targets still use CMake's ordinary validation.
include_guard(DIRECTORY)

if(V2_PERF_TESTS_ONLY)
    message(STATUS "V2 Tests: Release mode — building performance tests only")

    macro(add_executable target)
        if("${target}" MATCHES "^v2_perf_")
            _add_executable(${target} ${ARGN})
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
