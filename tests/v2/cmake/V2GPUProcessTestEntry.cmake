# @file V2GPUProcessTestEntry.cmake
# @brief Install the actual native lifetime main after all test targets exist.
include(${CMAKE_CURRENT_LIST_DIR}/V2GPUProcessTestEntryPolicy.cmake)
add_library(v2_core_gtest_main STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../integration/core_gtest_main.cpp)
target_link_libraries(v2_core_gtest_main PUBLIC llaminar2_core GTest::gtest)
target_include_directories(v2_core_gtest_main PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../src/v2)
target_compile_features(v2_core_gtest_main PRIVATE cxx_std_20)
v2_bind_gpu_process_test_directory("${CMAKE_CURRENT_SOURCE_DIR}" v2_core_gtest_main)

add_v2_test(V2_Unit_GPUProcessTestEntry
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_gpu_process_test_entry.py
    LABELS "V2;Unit;DeviceFree;Ownership;BuildSystem;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUProcessTestEntryReplay
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_gpu_process_test_entry.py
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;BuildSystem;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUProcessTestEntryReplay)
