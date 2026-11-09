# @file CMakeLists.txt
# @brief Focused native CPU port checks without the not-yet-portable inference runtime.
#
# This test-only project compiles the real production primitives with the same
# target/ISA policy as V2. It cannot build or certify llaminar2, replace the full
# Unit/preflight inventory, or select a scalar implementation for ARM. Actual
# timeout-owner JSON fields require the same CMake/CTest4 interface as full tests.
# Retain the established policy envelope: a newer tool prerequisite must not
# silently opt these non-module sources into a different compiler pipeline.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_VERSION VERSION_LESS 4.0)
    message(FATAL_ERROR "Focused test builds require CMake/CTest >=4.0 for timeout-owner JSON metadata")
endif()
project(llaminar_cpu_portability LANGUAGES C CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release|Integration)$")
    message(FATAL_ERROR "Choose Debug, Release or Integration")
endif()
set(CMAKE_CXX_FLAGS_INTEGRATION "-O3 -g -DNDEBUG" CACHE STRING "Optimized port checks with symbols")
set(CMAKE_C_FLAGS_INTEGRATION "-O3 -g -DNDEBUG" CACHE STRING "Optimized port checks with symbols")
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/CPUInstructionSet.cmake")
llaminar_configure_cpu_instruction_set()
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/OpenMPRuntimeBinding.cmake")
llaminar_openmp_runtime_soname(LLAMINAR_OPENMP_RUNTIME_SONAME)
enable_testing()
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/V2CPUArchitectureTests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/V2CPUExplicitRoundingTests.cmake")
