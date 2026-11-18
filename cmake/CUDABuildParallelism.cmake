# @file CUDABuildParallelism.cmake
# @brief Share Ninja capacity between compilers and linkers on memory-constrained CUDA build hosts.
include_guard(GLOBAL)

# @brief Configure the pools inherited by subsequent targets and nvcc's per-TU architecture concurrency.
# @param memory_mib Physical memory on the build host or its smaller container limit, in MiB.
# @param processors Number of CPUs available to Ninja.
function(llaminar_configure_cuda_ninja_pools memory_mib processors)
    # The optimized C++ test compiles can each exceed 3 GiB RSS. Reserve 4 GiB
    # per native job and leave 4 GiB for BuildKit, the runner and host services.
    math(EXPR usable_mib "${memory_mib} - 4096")
    math(EXPR memory_slots "${usable_mib} / 4096")
    math(EXPR ninja_slots "${processors} + 2")
    if(memory_slots LESS ninja_slots)
        if(memory_slots LESS 1)
            set(memory_slots 1)
        endif()
        # Prevent mixed waves of compiles and links from spending the same RAM.
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS "native_build=${memory_slots}")
        set(CMAKE_JOB_POOL_COMPILE native_build PARENT_SCOPE)
        set(CMAKE_JOB_POOL_LINK native_build PARENT_SCOPE)
        set(LLAMINAR_CUDA_DEVICE_LINK_POOL native_build PARENT_SCOPE)
        # On machines below 2 GiB per default Ninja job, parallel architecture
        # passes also caused a distinct CUDA frontend peak during cold builds.
        math(EXPR cuda_parallel_mib "2048 * ${ninja_slots}")
        if(usable_mib LESS cuda_parallel_mib)
            set(nvcc_threads 1)
        else()
            set(nvcc_threads 0)
        endif()
        set(LLAMINAR_NVCC_THREADS "${nvcc_threads}" PARENT_SCOPE)
        message(STATUS "V2: native build pool=${memory_slots}, nvcc threads=${nvcc_threads} (${memory_mib} MiB, ${processors} CPUs)")
    else()
        # On larger hosts Ninja owns ordinary compile scheduling. Host linker
        # fanout and the per-library CUDA device-link step keep their own pools.
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS nvlink_pool=8 link_pool=24)
        set(CMAKE_JOB_POOL_LINK link_pool PARENT_SCOPE)
        set(LLAMINAR_CUDA_DEVICE_LINK_POOL nvlink_pool PARENT_SCOPE)
        set(LLAMINAR_NVCC_THREADS 0 PARENT_SCOPE)
    endif()
endfunction()
