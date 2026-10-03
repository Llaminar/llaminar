# @file ROCmSDKHeaders.cmake
# @brief Bind optional ROCm headers to the one selected SDK, never host defaults.
#
# Finding a compiler in ROCM_PATH does not make /opt/rocm's cached component
# headers compatible. Each lookup is recomputed from that same authority, so
# changing SDKs or reusing a build tree cannot resurrect another installation.

include_guard(GLOBAL)

# @brief Resolve a relative component header inside the configured ROCm SDK.
# @param output_var Caller variable receiving the include root or NOTFOUND.
# @param relative_header Header name beneath the SDK's public include root.
# No directory cache, system search, or alternative SDK participates.
function(llaminar_find_rocm_sdk_header output_var relative_header)
    if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
        message(FATAL_ERROR "ROCm header discovery requires the selected ROCM_PATH")
    endif()
    if(IS_ABSOLUTE "${relative_header}" OR relative_header MATCHES "(^|/)\\.\\.(/|$)")
        message(FATAL_ERROR "ROCm component headers must remain within the selected SDK")
    endif()
    # Retire old find_path cache entries instead of leaving a misleading second
    # selection authority in the reused tree. ROCM_PATH alone owns this choice.
    unset(${output_var} CACHE)
    set(include_root "${ROCM_PATH}/include")
    if(EXISTS "${include_root}/${relative_header}")
        set(${output_var} "${include_root}" PARENT_SCOPE)
    else()
        set(${output_var} "${output_var}-NOTFOUND" PARENT_SCOPE)
    endif()
endfunction()
