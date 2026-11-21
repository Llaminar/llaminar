# Compile-only replay of hidden device headers through the production launcher.
cmake_minimum_required(VERSION 3.25)
project(HipDeviceInputIdentity LANGUAGES HIP)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(PROBE_CCACHE ccache REQUIRED)
set(CMAKE_HIP_COMPILER_LAUNCHER "${PROBE_CCACHE}")
add_compile_options("--hip-path=${ROCM_PATH}" "--rocm-path=${ROCM_PATH}")
include("${LLAMINAR_SOURCE_ROOT}/cmake/GpuSpillGuard.cmake")
add_library(probe OBJECT "${PROBE_SOURCE_ROOT}/probe.hip")
# System-classified device-only headers also belong in Ninja's rebuild closure.
target_include_directories(probe SYSTEM PRIVATE "${PROBE_SOURCE_ROOT}")
