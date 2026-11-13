# Actual production rollback authority, compiled without a vendor SDK/runtime.
add_executable(v2_unit_gpu_worker_initialization
    unit/backends/Test__GPUWorkerInitialization.cpp
    unit/backends/Test__GPUWorkerReadiness.cpp)
target_include_directories(v2_unit_gpu_worker_initialization PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_gpu_worker_initialization PRIVATE cxx_std_20)
target_link_libraries(v2_unit_gpu_worker_initialization PRIVATE GTest::gtest_main Threads::Threads)
add_v2_test(V2_Unit_GPUWorkerInitialization
    COMMAND $<TARGET_FILE:v2_unit_gpu_worker_initialization>
    LABELS "V2;Unit;DeviceFree;Ownership;RuntimeRetirement;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUWorkerInitializationReplay
    COMMAND $<TARGET_FILE:v2_unit_gpu_worker_initialization>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;RuntimeRetirement;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUWorkerInitializationReplay)

# Every constructor readiness defect retains an explicit actual-helper preflight replay.
add_v2_test(V2_Integration_GPUWorkerReadinessReplay
    COMMAND $<TARGET_FILE:v2_unit_gpu_worker_initialization>
        --gtest_filter=GPUWorkerReadiness.*
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Synchronization;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUWorkerReadinessReplay)
