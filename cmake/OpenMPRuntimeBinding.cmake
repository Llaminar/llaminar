# @file OpenMPRuntimeBinding.cmake
# @brief Seal the selected CPU OpenMP runtime's ELF or Mach-O identity.
#
# The compiler's OpenMP target owns runtime selection. A GPU SDK search path
# must not replace that runtime with a same-filename compatibility alias. Record
# its ELF SONAME or Mach-O install name, so the configured dependency identity
# comes from the library itself rather than a guessed filename. Installed ELF
# test runners can verify that closure without a compiler SDK or matching symlinks.
include_guard(GLOBAL)

# @brief Read the one dynamic runtime selected by FindOpenMP for C++.
# @param output_variable Caller variable receiving the ELF SONAME or Mach-O install name.
# @throws Fatal configuration error for missing or ambiguous runtime identity.
function(llaminar_openmp_runtime_soname output_variable)
    if(APPLE)
        if(CMAKE_OTOOL)
            set(runtime_reader "${CMAKE_OTOOL}")
        else()
            find_program(LLAMINAR_OTOOL_EXECUTABLE NAMES otool REQUIRED)
            set(runtime_reader "${LLAMINAR_OTOOL_EXECUTABLE}")
        endif()
    elseif(NOT CMAKE_READELF)
        message(FATAL_ERROR "OpenMP runtime binding requires the configured ELF reader")
    endif()
    set(runtime_sonames)
    foreach(library IN LISTS OpenMP_CXX_LIBRARIES)
        # FindOpenMP also includes pthread. Only shared libraries can own the
        # dynamic OpenMP ABI whose resolution the installed gate must prove.
        if(APPLE)
            if(NOT library MATCHES "lib(gomp|omp|iomp5)(\\.[0-9]+)*\\.dylib$")
                continue()
            endif()
            execute_process(COMMAND "${runtime_reader}" -D "${library}"
                RESULT_VARIABLE result OUTPUT_VARIABLE dynamic_section
                ERROR_VARIABLE diagnostic OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(identity_pattern "^[^\n]+:\n([^\n]+)$")
        else()
            if(NOT library MATCHES "lib(gomp|omp|iomp5)\\.so")
                continue()
            endif()
            execute_process(COMMAND "${CMAKE_READELF}" -d "${library}"
                RESULT_VARIABLE result OUTPUT_VARIABLE dynamic_section
                ERROR_VARIABLE diagnostic)
            set(identity_pattern "\\(SONAME\\)[^\n]*\\[([^]]+)\\]")
        endif()
        if(NOT result EQUAL 0 OR NOT dynamic_section MATCHES "${identity_pattern}")
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
