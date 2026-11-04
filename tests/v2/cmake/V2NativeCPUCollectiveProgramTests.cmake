# @file V2NativeCPUCollectiveProgramTests.cmake
# @brief Replay actual PMA/session/shutdown ownership without creating native communicators.
#
# Compiling the actual producers in one CPU-only executable prevents stale owner
# Count ABI reuse. Only MPI/cache backend boundaries are doubled; the admission,
# borrow, lock, finalization order and retirement checks are production bodies.
# Actual GlobalTPContext/ShmemSpin/MPI native qualification remains separate.
set(_cpu_program_root "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
add_executable(v2_test_native_cpu_collective_program
    unit/collective/Test__NativeCPUCollectiveProgram.cpp
    ${_cpu_program_root}/collective/NativeCPUCollectiveProgram.cpp
    ${_cpu_program_root}/planning/PhysicalMemoryAuthority.cpp
    ${_cpu_program_root}/app/MPIProcessSession.cpp
    ${_cpu_program_root}/app/MPIShutdown.cpp
    ${_cpu_program_root}/backends/GlobalDeviceAddress.cpp
    ${_cpu_program_root}/utils/MPISessionOwner.cpp)
target_compile_features(v2_test_native_cpu_collective_program PRIVATE cxx_std_23)
target_include_directories(v2_test_native_cpu_collective_program PRIVATE
    $<TARGET_PROPERTY:llaminar2_core,INCLUDE_DIRECTORIES>)
target_compile_definitions(v2_test_native_cpu_collective_program PRIVATE
    $<TARGET_PROPERTY:llaminar2_core,COMPILE_DEFINITIONS>)
target_compile_options(v2_test_native_cpu_collective_program PRIVATE
    -ffunction-sections -fdata-sections)
target_link_options(v2_test_native_cpu_collective_program PRIVATE -Wl,--gc-sections)
target_link_libraries(v2_test_native_cpu_collective_program PRIVATE
    GTest::gtest_main MPI::MPI_CXX OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
add_v2_test(V2_Unit_NativeCPUCollectiveProgram
    COMMAND $<TARGET_FILE:v2_test_native_cpu_collective_program>
    LABELS "V2;Unit;CPU;Collective;Memory;Lifetime;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_NativeCPUCollectiveProgramReplay
    COMMAND $<TARGET_FILE:v2_test_native_cpu_collective_program>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Collective;Memory;Lifetime;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
if(TEST V2_Integration_NativeCPUCollectiveProgramReplay)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeCPUCollectiveProgramReplay)
endif()
unset(_cpu_program_root)
