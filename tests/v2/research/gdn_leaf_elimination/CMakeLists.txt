# Standalone research bridge against an explicitly selected production core.
# Use that build's matching headers; never substitute a backend or precision
# if its native control fails. This is not an installed serving path or gate.
cmake_minimum_required(VERSION 3.18)

project(gdn_leaf_elimination_research LANGUAGES CXX)
enable_testing()
find_program(CCACHE_PROGRAM ccache REQUIRED)
set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "Configure the research bridge with CMAKE_BUILD_TYPE=Release")
endif()
if(NOT LLAMINAR_CORE_BUILD)
    message(FATAL_ERROR "Set LLAMINAR_CORE_BUILD to an existing shared-core build")
endif()
file(STRINGS "${LLAMINAR_CORE_BUILD}/CMakeCache.txt" HOME_LINE
    REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" NATIVE_SOURCE "${HOME_LINE}")
set(LLAMINAR_CORE_SOURCE "${NATIVE_SOURCE}" CACHE PATH "Exact source/header snapshot matching the selected core")
set(LLAMINAR_CORE_LIBRARY "${LLAMINAR_CORE_BUILD}/libllaminar2_core.so" CACHE FILEPATH "Exact production core DSO or its immutable copy")
set(NATIVE_SOURCE "${LLAMINAR_CORE_SOURCE}")
set(NATIVE_CORE "${LLAMINAR_CORE_LIBRARY}")
file(STRINGS "${LLAMINAR_CORE_BUILD}/CMakeCache.txt" BUILD_TYPE_LINE
    REGEX "^CMAKE_BUILD_TYPE:STRING=")
string(REPLACE "CMAKE_BUILD_TYPE:STRING=" "" NATIVE_BUILD_TYPE "${BUILD_TYPE_LINE}")
if(NOT EXISTS "${NATIVE_CORE}" OR NOT EXISTS "${NATIVE_SOURCE}/kernels/cpu/gdn/CPUGatedDeltaNet.h")
    message(FATAL_ERROR "Selected core library or matching headers are missing")
endif()
if(EXISTS "${NATIVE_SOURCE}/kernels/CPUGDNStateLayout.h")
    message(FATAL_ERROR "This historical dense-state bridge requires its frozen pre-epoch core and matching headers. Current CPU state includes checkpoint history; use v2_test_cpu_fixed_epoch_gdn and the production GDN performance targets for the installed implementation.")
endif()
add_library(gdn_research_selected_core SHARED IMPORTED)
set_target_properties(gdn_research_selected_core PROPERTIES IMPORTED_LOCATION "${NATIVE_CORE}")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
option(GDN_RESEARCH_SHARED_READOUT "Test the explicitly changed production CPU readout" OFF)
if(GDN_RESEARCH_SHARED_READOUT)
    add_library(v2_research_gdn_leaf_native SHARED SharedReadoutCPU.cpp SharedReadoutLifecycle.cpp ExactLeafCPU.cpp ExactLeafLifecycle.cpp)
    add_library(v2_research_gdn_backward_native SHARED BackwardReadoutCPU.cpp)
    target_compile_features(v2_research_gdn_backward_native PRIVATE cxx_std_20)
    target_include_directories(v2_research_gdn_backward_native PRIVATE "${NATIVE_SOURCE}")
    target_link_libraries(v2_research_gdn_backward_native PRIVATE v2_research_gdn_leaf_native)
    target_compile_options(v2_research_gdn_backward_native PRIVATE -mavx2 -mfma -fno-fast-math -ffp-contract=off)
    add_library(v2_research_gdn_epoch_native SHARED EpochReadoutCPU.cpp)
    target_compile_features(v2_research_gdn_epoch_native PRIVATE cxx_std_20)
    target_include_directories(v2_research_gdn_epoch_native PRIVATE "${NATIVE_SOURCE}")
    target_compile_options(v2_research_gdn_epoch_native PRIVATE -mavx2 -mfma -fno-fast-math -ffp-contract=off)
else()
    add_library(v2_research_gdn_leaf_native SHARED NativeCPU.cpp ExactLeafCPU.cpp ExactLeafLifecycle.cpp)
endif()
target_compile_features(v2_research_gdn_leaf_native PRIVATE cxx_std_20)
target_include_directories(v2_research_gdn_leaf_native PRIVATE "${NATIVE_SOURCE}")
target_link_libraries(v2_research_gdn_leaf_native PRIVATE
    gdn_research_selected_core MPI::MPI_CXX OpenMP::OpenMP_CXX)
# The explicit eight-value tile uses AVX2. The linked core independently
# selects its own runtime ISA; the bridge never changes its compiler flags.
target_compile_options(v2_research_gdn_leaf_native PRIVATE -mavx2 -mfma -fno-fast-math -ffp-contract=off)
file(SHA256 "${NATIVE_SOURCE}/kernels/cpu/gdn/CPUGatedDeltaNet.cpp" NATIVE_SOURCE_SHA)
file(SHA256 "${NATIVE_SOURCE}/kernels/cpu/gdn/CPUGatedDeltaNet.h" NATIVE_HEADER_SHA)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/native-build.json" CONTENT
    "{\n  \"core_library\": \"${NATIVE_CORE}\",\n  \"core_build_type\": \"${NATIVE_BUILD_TYPE}\",\n  \"source_directory\": \"${NATIVE_SOURCE}\",\n  \"gdn_source_sha256\": \"${NATIVE_SOURCE_SHA}\",\n  \"gdn_header_sha256\": \"${NATIVE_HEADER_SHA}\",\n  \"compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\"\n}\n")

add_test(NAME GDNResearch_Reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_harness.py)
set_tests_properties(GDNResearch_Reference PROPERTIES
    LABELS "Research;DeviceFree" TIMEOUT 30)
if(GDN_RESEARCH_SHARED_READOUT)
add_test(NAME GDNResearch_EpochCommonArithmetic
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/epoch.py
        --library $<TARGET_FILE:v2_research_gdn_epoch_native>
        --output ${CMAKE_CURRENT_BINARY_DIR}/epoch-common-arithmetic.json)
add_test(NAME GDNResearch_EpochStateLifecycle
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_epoch.py
        --library $<TARGET_FILE:v2_research_gdn_epoch_native>)
set_tests_properties(GDNResearch_EpochCommonArithmetic GDNResearch_EpochStateLifecycle PROPERTIES
    LABELS "Research;CPU" ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 120)
add_test(NAME GDNResearch_BackwardReference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_backward.py)
set_tests_properties(GDNResearch_BackwardReference PROPERTIES
    LABELS "Research;DeviceFree" TIMEOUT 30)
add_test(NAME GDNResearch_BackwardCPU
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/backward.py
        --library $<TARGET_FILE:v2_research_gdn_backward_native>)
set_tests_properties(GDNResearch_BackwardCPU PROPERTIES
    LABELS "Research;CPU" ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 120)
add_test(NAME GDNResearch_SharedReadoutCPU
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shared.py
        --library $<TARGET_FILE:v2_research_gdn_leaf_native>)
set_tests_properties(GDNResearch_SharedReadoutCPU PROPERTIES
    LABELS "Research;CPU" ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 120)
else()
add_test(NAME GDNResearch_AdjustedQueryCPU
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native.py
        --library $<TARGET_FILE:v2_research_gdn_leaf_native>)
add_test(NAME GDNResearch_ExactCompactCPU
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/exact.py
        --library $<TARGET_FILE:v2_research_gdn_leaf_native>)
set_tests_properties(GDNResearch_AdjustedQueryCPU PROPERTIES
    LABELS "Research;CPU"
    ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30)
set_tests_properties(GDNResearch_ExactCompactCPU PROPERTIES
    LABELS "Research;CPU"
    ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 120)
endif()
