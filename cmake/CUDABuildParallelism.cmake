# @file CUDABuildParallelism.cmake
# @brief Share Ninja capacity between compilers and linkers on memory-constrained CUDA build hosts.
include_guard(GLOBAL)

# @brief Configure the pools inherited by subsequent targets and nvcc's per-TU architecture concurrency.
# @param memory_mib Physical memory on the build host or its smaller container limit, in MiB.
# @param processors Number of CPUs available to Ninja.
function(llaminar_configure_cuda_ninja_pools memory_mib processors)
    # Ninja normally starts CPUs + 2 commands. In a four-SM build, nvcc -t 0
    # multiplied 32 nvcc invocations into 102 live cicc processes on a 31 GiB
    # host. Leave 4 GiB for the OS, runner and concurrent BuildKit stages.
    # A compiler or ld.lld job can itself use 2 GiB.
    math(EXPR memory_slots "(${memory_mib} - 4096) / 2048")
    math(EXPR ninja_slots "${processors} + 2")
    if(memory_slots LESS ninja_slots)
        if(memory_slots LESS 1)
            set(memory_slots 1)
        endif()
        # Compiles, host links and CUDA device links draw from the same slots.
        # Compile all requested SMs sequentially inside each admitted nvcc job.
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS "native_build=${memory_slots}")
        set(CMAKE_JOB_POOL_COMPILE native_build PARENT_SCOPE)
        set(CMAKE_JOB_POOL_LINK native_build PARENT_SCOPE)
        set(LLAMINAR_CUDA_DEVICE_LINK_POOL native_build PARENT_SCOPE)
        set(LLAMINAR_NVCC_THREADS 1 PARENT_SCOPE)
        message(STATUS "V2: native build pool=${memory_slots}, nvcc threads=1 (${memory_mib} MiB, ${processors} CPUs)")
    else()
        # On larger hosts Ninja owns ordinary compile scheduling. Host linker
        # fanout and the per-library CUDA device-link step keep their own pools.
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS nvlink_pool=8 link_pool=24)
        set(CMAKE_JOB_POOL_LINK link_pool PARENT_SCOPE)
        set(LLAMINAR_CUDA_DEVICE_LINK_POOL nvlink_pool PARENT_SCOPE)
        set(LLAMINAR_NVCC_THREADS 0 PARENT_SCOPE)
    endif()
endfunction()
