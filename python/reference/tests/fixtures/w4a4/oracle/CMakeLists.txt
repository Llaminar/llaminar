# Build the exact upstream host implementation used to record arithmetic traces.
# This explicit fixture-authoring build is not a runtime dependency or test fallback.
cmake_minimum_required(VERSION 3.22)
project(W4A4RecordedOracle LANGUAGES C CXX)
if(NOT DEFINED ORACLE_SOURCE)
    message(FATAL_ERROR "ORACLE_SOURCE must name the pinned mx-llama.cpp checkout")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${ORACLE_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE oracle_revision OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
if(NOT oracle_revision STREQUAL "bfc4c5dc3c96df652f42924370ef49b728d585ad")
    message(FATAL_ERROR "W4A4 oracle source revision does not match the recorded arithmetic policy")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${ORACLE_SOURCE}" status --porcelain -- ggml
    OUTPUT_VARIABLE oracle_dirty OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
if(NOT oracle_dirty STREQUAL "")
    message(FATAL_ERROR "W4A4 oracle GGML source must be unmodified")
endif()
if(GGML_CUDA OR GGML_HIP)
    message(FATAL_ERROR "W4A4 recorded arithmetic oracle requires the actual CPU implementation")
endif()
set(GGML_BUILD_TESTS OFF CACHE BOOL "No upstream model tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "No upstream examples" FORCE)
add_subdirectory("${ORACLE_SOURCE}/ggml" upstream)
