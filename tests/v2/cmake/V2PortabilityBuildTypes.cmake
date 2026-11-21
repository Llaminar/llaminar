# @file V2PortabilityBuildTypes.cmake
# @brief Seed genuine optimized Integration defaults before CMake initializes compiler caches.
#
# project()/enable_language() initializes a custom active configuration's cache,
# even when its initial flags are empty. A later set(... CACHE STRING ...) does
# not repair that value. Seed _INIT before language enablement; never FORCE an
# authored compiler envelope or disguise an existing unoptimized build tree.
include_guard(GLOBAL)
if(CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED)
    message(FATAL_ERROR "Focused Integration flag initialization must precede project()/enable_language()")
endif()
foreach(_language IN ITEMS C CXX)
    if(NOT DEFINED CMAKE_${_language}_FLAGS_INTEGRATION_INIT)
        set(CMAKE_${_language}_FLAGS_INTEGRATION_INIT "-O3 -g -DNDEBUG")
    endif()
endforeach()

# @brief Reject a stale empty Integration cache after language initialization.
# Explicit nonempty overrides remain owned by the caller; actual compile-command
# and codegen evidence must still authenticate their complete effective envelope.
function(v2_validate_portability_integration_flags)
    if(CMAKE_BUILD_TYPE STREQUAL "Integration")
        foreach(_language IN ITEMS C CXX)
            if("${CMAKE_${_language}_FLAGS_INTEGRATION}" STREQUAL "")
                message(FATAL_ERROR
                    "Focused Integration has an empty ${_language} compiler-flag cache. "
                    "Configure a fresh build directory or explicitly supply the required optimized flags; "
                    "an unoptimized Integration label is not qualifying evidence.")
            endif()
        endforeach()
    endif()
endfunction()
