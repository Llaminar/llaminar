# Bind shipping HIP targets before language discovery; never infer one local GPU.
# The shared installer declaration is the sole default for application and DSOs.
include_guard(GLOBAL)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../scripts/docker/rocm-release.env"
    _llaminar_rocm_targets REGEX "^LLAMINAR_ROCM_GPU_TARGETS=")
if(NOT _llaminar_rocm_targets MATCHES "^LLAMINAR_ROCM_GPU_TARGETS='([^']+)'$")
    message(FATAL_ERROR "Missing canonical ROCm device-code target declaration")
endif()
string(REPLACE "\\;" ";" LLAMINAR_ROCM_DEFAULT_ARCHITECTURES "${CMAKE_MATCH_1}")
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    if(DEFINED ENV{HIPARCHS} AND NOT "$ENV{HIPARCHS}" STREQUAL "")
        set(CMAKE_HIP_ARCHITECTURES "$ENV{HIPARCHS}" CACHE STRING "HIP device-code targets")
    else()
        set(CMAKE_HIP_ARCHITECTURES "${LLAMINAR_ROCM_DEFAULT_ARCHITECTURES}" CACHE STRING "HIP device-code targets")
    endif()
endif()
if(NOT CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR "HIP device-code targets must be explicit and nonempty")
endif()
# RCCL must cover the same selected devices, independent of the configure host.
if(NOT DEFINED RCCL_GPU_TARGETS)
    set(RCCL_GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "RCCL device-code targets")
endif()
# Existing production reduction trees require fixed 64-lane arithmetic order
# on both CDNA and RDNA. Apply that existing-kernel contract to current HIP
# targets, including standalone tests; RDNA wave32 compilation would silently
# change their reductions. This is not a permanent backend restriction: a future
# separately registered MMA/wave32 target needs its own explicit target wave
# contract and correctness/economy gates before this default can be overridden.
# Vendor RCCL and rocBLAS builds retain their independent upstream wave policy.
if("${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_INIT}" MATCHES "(^| )-mwavefrontsize32( |$)")
    message(FATAL_ERROR "Llaminar HIP kernels require wave64; remove -mwavefrontsize32")
endif()
add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize64>")
