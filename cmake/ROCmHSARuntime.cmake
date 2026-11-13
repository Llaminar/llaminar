# @file ROCmHSARuntime.cmake
# @brief Resolve HSA ABI1 only from the selected SDK, never the system loader cache.
# Real paths bind the physical DSO despite competing versioned/unversioned aliases.
include_guard(GLOBAL)

# @brief Resolve one exact SDK-owned HSA runtime without retaining a search cache.
# @param output_var Parent variable receiving the real selected ABI1 DSO path.
# Missing or escaping libraries are configuration errors; no other SDK is tried.
function(llaminar_find_rocm_hsa_runtime output_var)
    if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
        message(FATAL_ERROR "HSA runtime selection requires the selected ROCM_PATH")
    endif()
    unset(${output_var} CACHE)
    unset(_llaminar_hsa_candidate CACHE)
    set(_llaminar_hsa_candidate "_llaminar_hsa_candidate-NOTFOUND")
    find_file(_llaminar_hsa_candidate NAMES libhsa-runtime64.so.1
        PATHS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
    file(REAL_PATH "${_llaminar_hsa_candidate}" selected_hsa)
    unset(_llaminar_hsa_candidate CACHE)
    file(REAL_PATH "${ROCM_PATH}" selected_sdk)
    file(RELATIVE_PATH relative_hsa "${selected_sdk}" "${selected_hsa}")
    if(IS_ABSOLUTE "${relative_hsa}" OR relative_hsa MATCHES "^[.][.](/|$)" OR IS_DIRECTORY "${selected_hsa}")
        message(FATAL_ERROR "Selected HSA runtime escapes ROCM_PATH: ${selected_hsa}")
    endif()
    set(${output_var} "${selected_hsa}" PARENT_SCOPE)
endfunction()
