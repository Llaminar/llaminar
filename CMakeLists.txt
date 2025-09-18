cmake_minimum_required(VERSION 3.18)

# Project configuration
project(Llaminar 
    VERSION 1.0.0
    DESCRIPTION "An LLM inferencing engine using COSMA / Open MPI for hyper-scalability"
    LANGUAGES CXX C
)

# Set C++ and C standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)

# NUMA support (optional)
pkg_check_modules(NUMA numa)
if(NUMA_FOUND)
    message(STATUS "NUMA support found")
    add_compile_definitions(HAVE_NUMA=1)
else()
    message(STATUS "NUMA support not found - continuing without it")
endif()

# COSMA configuration
message(STATUS "Configuring COSMA submodule...")
set(COSMA_BLAS "auto" CACHE STRING "BLAS backend for COSMA")
set(COSMA_SCALAPACK "CUSTOM" CACHE STRING "ScaLAPACK backend for COSMA") 
set(COSMA_WITH_PROFILING OFF CACHE BOOL "Disable COSMA profiling")
set(COSMA_WITH_TESTS OFF CACHE BOOL "Disable COSMA tests")
set(COSMA_WITH_INSTALL OFF CACHE BOOL "Disable COSMA install")

# Add COSMA submodule
add_subdirectory(COSMA)

# llama.cpp configuration
message(STATUS "Configuring llama.cpp submodule...")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries")
set(GGML_STATIC ON CACHE BOOL "Build ggml as static library")
set(GGML_OPENMP ON CACHE BOOL "Enable OpenMP in ggml")
set(GGML_MPI ON CACHE BOOL "Enable MPI in ggml")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama.cpp examples")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama.cpp server")

# Add llama.cpp submodule
add_subdirectory(llama.cpp)

# Include directories
include_directories(${MPI_INCLUDE_PATH})
include_directories(src)
include_directories(COSMA/src)
include_directories(llama.cpp)
include_directories(llama.cpp/ggml/include)
include_directories(llama.cpp/include)

# Source files
set(SOURCES
    src/main.cpp
    # Future source files:
    # src/qwen_inference_engine.cpp
    # src/numa_deployment.cpp  
    # src/custom_kernel_scheduler.cpp
    # src/cosma_backend.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        ${MPI_LIBRARIES}
        OpenMP::OpenMP_CXX
        cosma
        cosma_prefixed_pxgemm
        ggml
        llama
)# Add NUMA if available
if(NUMA_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${NUMA_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${NUMA_INCLUDE_DIRS})
endif()

# Compile definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE
        COSMA_HAVE_PROFILING=1
        $<$<BOOL:${NUMA_FOUND}>:NUMA_AWARE_ALLOCATION=1>
)

# MPI compile flags
if(MPI_COMPILE_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

# MPI link flags
if(MPI_LINK_FLAGS)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

# Enable compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# Create src directory and example files if they don't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# Print configuration summary
message(STATUS "=== Llaminar Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C standard: ${CMAKE_C_STANDARD}")
message(STATUS "MPI found: ${MPI_FOUND}")
message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI libraries: ${MPI_LIBRARIES}")
message(STATUS "OpenMP found: ${OpenMP_FOUND}")
message(STATUS "NUMA found: ${NUMA_FOUND}")
message(STATUS "COSMA BLAS backend: ${COSMA_BLAS}")
message(STATUS "COSMA ScaLAPACK backend: ${COSMA_SCALAPACK}")
message(STATUS "GGML static: ${GGML_STATIC}")
message(STATUS "GGML OpenMP: ${GGML_OPENMP}")
message(STATUS "GGML MPI: ${GGML_MPI}")
message(STATUS "===============================")