# @file TestCUDABuildParallelism.cmake
# @brief Replay build-host profiles through the production Ninja and CUDA scheduling policy.
cmake_minimum_required(VERSION 3.18)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/CUDABuildParallelism.cmake")

# @brief Check the inherited compile/link pools and the number of architecture passes per nvcc.
# @param memory_mib Available host or container memory, in MiB.
# @param processors CPUs available to Ninja.
# @param shared_slots Expected shared capacity, or zero for ordinary Ninja scheduling.
function(check_build_policy memory_mib processors shared_slots)
    set_property(GLOBAL PROPERTY JOB_POOLS "")
    llaminar_configure_cuda_ninja_pools("${memory_mib}" "${processors}")
    get_property(pools GLOBAL PROPERTY JOB_POOLS)
    if(shared_slots GREATER 0)
        if(NOT pools STREQUAL "native_build=${shared_slots}" OR
           NOT CMAKE_JOB_POOL_COMPILE STREQUAL "native_build" OR
           NOT CMAKE_JOB_POOL_LINK STREQUAL "native_build" OR
           NOT LLAMINAR_CUDA_DEVICE_LINK_POOL STREQUAL "native_build" OR
           NOT LLAMINAR_NVCC_THREADS STREQUAL "1")
            message(FATAL_ERROR "Expected all compiles and links to share ${shared_slots} slots and one pass per nvcc")
        endif()
    elseif(NOT pools STREQUAL "nvlink_pool=8;link_pool=24" OR
           DEFINED CMAKE_JOB_POOL_COMPILE OR
           NOT CMAKE_JOB_POOL_LINK STREQUAL "link_pool" OR
           NOT LLAMINAR_CUDA_DEVICE_LINK_POOL STREQUAL "nvlink_pool" OR
           NOT LLAMINAR_NVCC_THREADS STREQUAL "0")
        message(FATAL_ERROR "Expected unrestricted compiles and separate host and CUDA link pools")
    endif()
endfunction()

check_build_policy(31277 32 13)
check_build_policy(2048 32 1)
check_build_policy(64000 16 0)
