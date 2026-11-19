# Explicit diagnostics with the repository's HIP architecture and spill policy.
# The host runner admits one MI50; no default build, test or runtime depends on these.
set(_roofline_probes ReadBandwidth VectorIssue QfnRoutedPacketRead)
set(_roofline_targets)
foreach(_roofline_probe IN LISTS _roofline_probes)
    set(_roofline_target v2_perf_rocm_mi50_${_roofline_probe})
    set_source_files_properties(${_roofline_probe}.hip PROPERTIES LANGUAGE HIP)
    add_executable(${_roofline_target} EXCLUDE_FROM_ALL ${_roofline_probe}.hip)
    set_target_properties(${_roofline_target} PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON OUTPUT_NAME ${_roofline_probe})
    target_link_libraries(${_roofline_target} PRIVATE hip::host)
    list(APPEND _roofline_targets ${_roofline_target})
endforeach()
add_custom_target(v2_perf_rocm_mi50_roofline DEPENDS ${_roofline_targets})
