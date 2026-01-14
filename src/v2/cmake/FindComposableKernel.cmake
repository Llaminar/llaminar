# FindComposableKernel.cmake
# Find AMD ComposableKernel (CK) library for INT8 GEMM kernels
#
# This module defines:
#   ComposableKernel_FOUND - True if CK headers were found
#   ComposableKernel_INCLUDE_DIRS - CK include directories
#   ComposableKernel_VERSION - CK version (if available)
#
# CK is a header-only template library, so no library linking is needed.
# The main header is: ck/tensor_operation/gpu/device/device_gemm.hpp
#
# Search paths:
#   1. ROCM_PATH environment variable
#   2. /opt/rocm (default ROCm installation)
#   3. CMAKE_PREFIX_PATH
#
# Usage in CMakeLists.txt:
#   find_package(ComposableKernel)
#   if(ComposableKernel_FOUND)
#       target_include_directories(mytarget PRIVATE ${ComposableKernel_INCLUDE_DIRS})
#   endif()

# Get ROCM_PATH from environment or use default
if(DEFINED ENV{ROCM_PATH})
    set(_ROCM_PATH "$ENV{ROCM_PATH}")
else()
    set(_ROCM_PATH "/opt/rocm")
endif()

# Get local CK path (external/composable_kernel in workspace root)
get_filename_component(_WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(_LOCAL_CK_PATH "${_WORKSPACE_ROOT}/external/composable_kernel")

# Search for CK include directory
# PRIORITY: Local patched CK FIRST, then system ROCm
find_path(ComposableKernel_INCLUDE_DIR
    NAMES ck/ck.hpp
    HINTS
        ${_LOCAL_CK_PATH}/include          # Local patched CK (highest priority)
        ${_ROCM_PATH}/include
        ${_ROCM_PATH}/include/composable_kernel
        /opt/rocm/include
        /opt/rocm/include/composable_kernel
    PATH_SUFFIXES
        composable_kernel
        ck
    NO_DEFAULT_PATH  # Don't search system paths automatically, use our order
)

# Fallback if NO_DEFAULT_PATH didn't find it
if(NOT ComposableKernel_INCLUDE_DIR)
    find_path(ComposableKernel_INCLUDE_DIR
        NAMES ck/ck.hpp
        HINTS
            ${_LOCAL_CK_PATH}/include
            ${_ROCM_PATH}/include
            ${_ROCM_PATH}/include/composable_kernel
            /opt/rocm/include
            /opt/rocm/include/composable_kernel
        PATH_SUFFIXES
            composable_kernel
            ck
    )
endif()

# Also look for the tensor operation headers specifically
# Use same priority: local patched CK first
find_path(ComposableKernel_TENSOR_OP_DIR
    NAMES ck/tensor_operation/gpu/device/device_gemm.hpp
    HINTS
        ${_LOCAL_CK_PATH}/include          # Local patched CK (highest priority)
        ${ComposableKernel_INCLUDE_DIR}
        ${_ROCM_PATH}/include
        ${_ROCM_PATH}/include/composable_kernel
        /opt/rocm/include
        /opt/rocm/include/composable_kernel
    NO_DEFAULT_PATH
)

# Try to find version from header
if(ComposableKernel_INCLUDE_DIR)
    if(EXISTS "${ComposableKernel_INCLUDE_DIR}/ck/ck.hpp")
        file(STRINGS "${ComposableKernel_INCLUDE_DIR}/ck/ck.hpp" _CK_VERSION_LINE
             REGEX "#define CK_VERSION")
        if(_CK_VERSION_LINE)
            string(REGEX REPLACE ".*#define CK_VERSION \"([0-9.]+)\".*" "\\1" 
                   ComposableKernel_VERSION "${_CK_VERSION_LINE}")
        endif()
    endif()
endif()

# Handle REQUIRED and QUIET arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ComposableKernel
    REQUIRED_VARS 
        ComposableKernel_INCLUDE_DIR
    VERSION_VAR 
        ComposableKernel_VERSION
)

# Set output variables
if(ComposableKernel_FOUND)
    set(ComposableKernel_INCLUDE_DIRS 
        ${ComposableKernel_INCLUDE_DIR}
        ${ComposableKernel_TENSOR_OP_DIR}
    )
    # Remove duplicates
    list(REMOVE_DUPLICATES ComposableKernel_INCLUDE_DIRS)
    
    # Create imported target for modern CMake usage
    if(NOT TARGET ComposableKernel::ComposableKernel)
        add_library(ComposableKernel::ComposableKernel INTERFACE IMPORTED)
        set_target_properties(ComposableKernel::ComposableKernel PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ComposableKernel_INCLUDE_DIRS}"
        )
    endif()
    
    message(STATUS "Found ComposableKernel: ${ComposableKernel_INCLUDE_DIR}")
    if(ComposableKernel_VERSION)
        message(STATUS "  Version: ${ComposableKernel_VERSION}")
    endif()
endif()

mark_as_advanced(
    ComposableKernel_INCLUDE_DIR
    ComposableKernel_TENSOR_OP_DIR
)
