# @file OpenMPRuntimeBinding.cmake
# @brief Seal the selected CPU OpenMP runtime's native ELF identity for preflight.
#
# The compiler's OpenMP target owns runtime selection. A GPU SDK search path
# must not replace that runtime with a same-filename compatibility alias. Record
# its SONAME, not a builder-only absolute path, so installed test runners can
# verify the actual loader closure without a compiler SDK or matching symlinks.
include_guard(GLOBAL)

# @brief Read the one dynamic runtime selected by FindOpenMP for C++.
# @param output_variable Caller variable receiving the runtime's ELF SONAME.
# @throws Fatal configuration error for missing or ambiguous runtime identity.
function(llaminar_openmp_runtime_soname output_variable)
    if(NOT CMAKE_READELF)
        message(FATAL_ERROR "OpenMP runtime binding requires the configured ELF reader")
    endif()
    set(runtime_sonames)
    foreach(library IN LISTS OpenMP_CXX_LIBRARIES)
        # FindOpenMP also includes pthread. Only shared libraries can own the
        # dynamic OpenMP ABI whose resolution the installed gate must prove.
        if(NOT library MATCHES "lib(gomp|omp|iomp5)\\.so")
            continue()
        endif()
        execute_process(COMMAND "${CMAKE_READELF}" -d "${library}"
            RESULT_VARIABLE result OUTPUT_VARIABLE dynamic_section
            ERROR_VARIABLE diagnostic)
        if(NOT result EQUAL 0 OR
           NOT dynamic_section MATCHES "\\(SONAME\\)[^\n]*\\[([^]]+)\\]")
            message(FATAL_ERROR
                "Cannot authenticate selected OpenMP runtime '${library}': ${diagnostic}")
        endif()
        list(APPEND runtime_sonames "${CMAKE_MATCH_1}")
    endforeach()
    list(REMOVE_DUPLICATES runtime_sonames)
    list(LENGTH runtime_sonames runtime_count)
    if(NOT runtime_count EQUAL 1)
        message(FATAL_ERROR
            "Select exactly one dynamic CPU OpenMP runtime; found '${runtime_sonames}'")
    endif()
    list(GET runtime_sonames 0 runtime_soname)
    set(${output_variable} "${runtime_soname}" PARENT_SCOPE)
endfunction()
