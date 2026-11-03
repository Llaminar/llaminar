# These declarations own only functional targets and registrations. Release
# intentionally excludes them through V2TestTargetSelection/add_v2_test; return
# before attaching target properties or labels to omitted owners.
if(V2_PERF_TESTS_ONLY)
    return()
endif()

# Complete original Read/output/Retirement cohort records its own native programs.
# Actual bodies are builder-compiled; installed execution requires no SDK/compiler.
add_executable(v2_test_pooled_complete_cohort
    fixtures/qwen4exp/Test__PooledImmutableCompleteCohort.cpp
    unit/backends/Test__GPUExistingContextBorrow.cpp)
target_include_directories(v2_test_pooled_complete_cohort PRIVATE "${REPO_ROOT}/src/v2" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(v2_test_pooled_complete_cohort PRIVATE llaminar2_core GTest::gtest_main)
target_compile_features(v2_test_pooled_complete_cohort PRIVATE cxx_std_23)
target_link_options(v2_test_pooled_complete_cohort PRIVATE "LINKER:--export-dynamic")
v2_test_device_scope(v2_test_pooled_complete_cohort DeclaredBackends)
# Each original source owns ProcessRuntime tokens until its process exits.
# Keep exact Unit cases separate for CI margin; preflight populations below retain
# their original replay counts and standard 30-second guardian deadline.
set(_cohort_original_unit_cases
    PooledImmutableCompleteCohort.OriginalCohortP1RemotePositive
    PooledImmutableCompleteCohort.OriginalCohortP1RemoteEmpty
    PooledImmutableCompleteCohort.OriginalCohortP2RemotePositive
    PooledImmutableCompleteCohort.OriginalCohortP2RemoteEmpty
    PooledImmutableCompleteCohort.OriginalCohortP2LocalPositive
    PooledImmutableCompleteCohort.OriginalCohortP2LocalEmpty
    PooledImmutableCompleteCohort.OriginalCohortP4RemotePositive
    PooledImmutableCompleteCohort.OriginalCohortP4RemoteEmpty
    PooledImmutableCompleteCohort.OriginalCohortP4LocalPositive
    PooledImmutableCompleteCohort.OriginalCohortP4LocalEmpty
    PooledImmutableCompleteCohort.OriginalCohortP8RemotePositive
    PooledImmutableCompleteCohort.OriginalCohortP8RemoteEmpty
    PooledImmutableCompleteCohort.OriginalCohortP8LocalPositive
    PooledImmutableCompleteCohort.OriginalCohortP8LocalEmpty
    PooledImmutableCompleteCohort.MissingWrongOrStaleOriginalParticipantsRejectBeforeArm
    PooledImmutableCompleteCohort.ExternalLastBindingCannotReleaseAnActiveOriginalCapture
    PooledImmutableCompleteCohort.TerminalMetadataDriftStillJoinsTheOriginalWorker
    PooledImmutableCompleteCohort.MissingRequiredTerminalCohortNeverClassifiesSafeRetirement
    PooledImmutableCompleteCohort.PartialRemoteSetupRetiresEverySubmittedFrontierBeforeCancel
    PooledImmutableCompleteCohort.TypedResetUsesOnlyTheOriginalCompleteSourceAndRetainedWorker
    PooledImmutableCompleteCohort.ConstructionFailureDestroysCapturesBeforeOriginalRootsAndContexts
    PooledImmutableCompleteCohort.ActualRecorderVisitsEveryDeclaredStageAndRejectsReplacementBeforeNativeWork
    PooledImmutableCompleteCohort.PartialColdConstructionAndRecordingRetireOriginalFrontiersWithoutArm
    PooledImmutableCompleteCohort.ReplacedPublicStageRejectsBeforeAnyCachedLeafInspection
    PooledImmutableCompleteCohort.ConcurrentColdBindingPublishesOneOriginalOwnerBeforeAnyAdmission
    PooledImmutableCompleteCohort.CallerRuleAndOppositeSourceBindingsUseRealQueuedWorkerIdentity
    PooledImmutableCompleteCohort.ExpiredWeakOwnerCannotReplaceOriginalRetirementInProgress
    PooledImmutableCompleteCohort.FinalOwnerCannotBlockItsOriginalParticipatingWorker
    PooledImmutableCompleteCohort.QueuedOppositeSourceAdmissionsAndFailureNeverWaitForPeers
    GPUExistingContextBorrow.AbsentWrongFamilyOrdinalRetiringAndShutdownNeverCreate
    GPUExistingContextBorrow.ConcurrentOriginalBorrowExcludesRetirementUntilRelease
)
foreach(_case IN LISTS _cohort_original_unit_cases)
    string(REPLACE "." "_" _case_id "${_case}")
    add_v2_test("V2_Unit_PooledOriginal_${_case_id}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> "--gtest_filter=${_case}"
        LABELS "V2;Unit;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
endforeach()
add_v2_test(V2_Integration_PooledImmutableCohortP1RemotePositiveShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP1RemotePositiveShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP1RemotePositiveShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP1RemotePositiveShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP1RemoteEmptyShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP1RemoteEmptyShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP1RemoteEmptyShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP1RemoteEmptyShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP1RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemotePositiveShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemotePositiveShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemotePositiveShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemotePositiveShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemoteEmptyShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemoteEmptyShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemoteEmptyShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2RemoteEmptyShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalPositiveShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalPositiveShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalPositiveShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalPositiveShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalEmptyShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalEmptyShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalEmptyShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP2LocalEmptyShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP2LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemotePositiveShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemotePositiveShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemotePositiveShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemotePositiveShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemoteEmptyShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemoteEmptyShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemoteEmptyShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4RemoteEmptyShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalPositiveShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalPositiveShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalPositiveShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalPositiveShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalEmptyShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalEmptyShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalEmptyShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP4LocalEmptyShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP4LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemotePositiveShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemotePositiveShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemotePositiveShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemotePositiveShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemotePositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemoteEmptyShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemoteEmptyShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemoteEmptyShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8RemoteEmptyShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8RemoteEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalPositiveShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalPositiveShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalPositiveShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalPositiveShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalPositive --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalEmptyShard0Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalEmptyShard1Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalEmptyShard2Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortP8LocalEmptyShard3Replay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.OriginalCohortP8LocalEmpty --gtest_repeat=5
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortRejectionReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.MissingWrongOrStaleOriginalParticipantsRejectBeforeArm --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortActiveLifetimeReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.ExternalLastBindingCannotReleaseAnActiveOriginalCapture --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortTerminalJoinReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.TerminalMetadataDriftStillJoinsTheOriginalWorker --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortPartialSetupReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.PartialRemoteSetupRetiresEverySubmittedFrontierBeforeCancel --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortTypedResetReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.TypedResetUsesOnlyTheOriginalCompleteSourceAndRetainedWorker --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortConstructionLifetimeReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.ConstructionFailureDestroysCapturesBeforeOriginalRootsAndContexts --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortActualRecorderReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.ActualRecorderVisitsEveryDeclaredStageAndRejectsReplacementBeforeNativeWork --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortColdFailureReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.PartialColdConstructionAndRecordingRetireOriginalFrontiersWithoutArm --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortReplacedStageReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.ReplacedPublicStageRejectsBeforeAnyCachedLeafInspection --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortConcurrentBindingReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.ConcurrentColdBindingPublishesOneOriginalOwnerBeforeAnyAdmission --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortOppositeSourcesReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.CallerRuleAndOppositeSourceBindingsUseRealQueuedWorkerIdentity --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortRetirementClaimReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.ExpiredWeakOwnerCannotReplaceOriginalRetirementInProgress --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortWorkerDestructionReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.FinalOwnerCannotBlockItsOriginalParticipatingWorker --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableCohortOppositeAdmissionReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=PooledImmutableCompleteCohort.QueuedOppositeSourceAdmissionsAndFailureNeverWaitForPeers --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_PooledImmutableExistingWorkerBorrowReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=GPUExistingContextBorrow.* --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_PooledImmutableCohortP1RemotePositiveShard0Replay
    V2_Integration_PooledImmutableCohortP1RemotePositiveShard1Replay
    V2_Integration_PooledImmutableCohortP1RemotePositiveShard2Replay
    V2_Integration_PooledImmutableCohortP1RemotePositiveShard3Replay
    V2_Integration_PooledImmutableCohortP1RemoteEmptyShard0Replay
    V2_Integration_PooledImmutableCohortP1RemoteEmptyShard1Replay
    V2_Integration_PooledImmutableCohortP1RemoteEmptyShard2Replay
    V2_Integration_PooledImmutableCohortP1RemoteEmptyShard3Replay
    V2_Integration_PooledImmutableCohortP2RemotePositiveShard0Replay
    V2_Integration_PooledImmutableCohortP2RemotePositiveShard1Replay
    V2_Integration_PooledImmutableCohortP2RemotePositiveShard2Replay
    V2_Integration_PooledImmutableCohortP2RemotePositiveShard3Replay
    V2_Integration_PooledImmutableCohortP2RemoteEmptyShard0Replay
    V2_Integration_PooledImmutableCohortP2RemoteEmptyShard1Replay
    V2_Integration_PooledImmutableCohortP2RemoteEmptyShard2Replay
    V2_Integration_PooledImmutableCohortP2RemoteEmptyShard3Replay
    V2_Integration_PooledImmutableCohortP2LocalPositiveShard0Replay
    V2_Integration_PooledImmutableCohortP2LocalPositiveShard1Replay
    V2_Integration_PooledImmutableCohortP2LocalPositiveShard2Replay
    V2_Integration_PooledImmutableCohortP2LocalPositiveShard3Replay
    V2_Integration_PooledImmutableCohortP2LocalEmptyShard0Replay
    V2_Integration_PooledImmutableCohortP2LocalEmptyShard1Replay
    V2_Integration_PooledImmutableCohortP2LocalEmptyShard2Replay
    V2_Integration_PooledImmutableCohortP2LocalEmptyShard3Replay
    V2_Integration_PooledImmutableCohortP4RemotePositiveShard0Replay
    V2_Integration_PooledImmutableCohortP4RemotePositiveShard1Replay
    V2_Integration_PooledImmutableCohortP4RemotePositiveShard2Replay
    V2_Integration_PooledImmutableCohortP4RemotePositiveShard3Replay
    V2_Integration_PooledImmutableCohortP4RemoteEmptyShard0Replay
    V2_Integration_PooledImmutableCohortP4RemoteEmptyShard1Replay
    V2_Integration_PooledImmutableCohortP4RemoteEmptyShard2Replay
    V2_Integration_PooledImmutableCohortP4RemoteEmptyShard3Replay
    V2_Integration_PooledImmutableCohortP4LocalPositiveShard0Replay
    V2_Integration_PooledImmutableCohortP4LocalPositiveShard1Replay
    V2_Integration_PooledImmutableCohortP4LocalPositiveShard2Replay
    V2_Integration_PooledImmutableCohortP4LocalPositiveShard3Replay
    V2_Integration_PooledImmutableCohortP4LocalEmptyShard0Replay
    V2_Integration_PooledImmutableCohortP4LocalEmptyShard1Replay
    V2_Integration_PooledImmutableCohortP4LocalEmptyShard2Replay
    V2_Integration_PooledImmutableCohortP4LocalEmptyShard3Replay
    V2_Integration_PooledImmutableCohortP8RemotePositiveShard0Replay
    V2_Integration_PooledImmutableCohortP8RemotePositiveShard1Replay
    V2_Integration_PooledImmutableCohortP8RemotePositiveShard2Replay
    V2_Integration_PooledImmutableCohortP8RemotePositiveShard3Replay
    V2_Integration_PooledImmutableCohortP8RemoteEmptyShard0Replay
    V2_Integration_PooledImmutableCohortP8RemoteEmptyShard1Replay
    V2_Integration_PooledImmutableCohortP8RemoteEmptyShard2Replay
    V2_Integration_PooledImmutableCohortP8RemoteEmptyShard3Replay
    V2_Integration_PooledImmutableCohortP8LocalPositiveShard0Replay
    V2_Integration_PooledImmutableCohortP8LocalPositiveShard1Replay
    V2_Integration_PooledImmutableCohortP8LocalPositiveShard2Replay
    V2_Integration_PooledImmutableCohortP8LocalPositiveShard3Replay
    V2_Integration_PooledImmutableCohortP8LocalEmptyShard0Replay
    V2_Integration_PooledImmutableCohortP8LocalEmptyShard1Replay
    V2_Integration_PooledImmutableCohortP8LocalEmptyShard2Replay
    V2_Integration_PooledImmutableCohortP8LocalEmptyShard3Replay
    V2_Integration_PooledImmutableCohortRejectionReplay
    V2_Integration_PooledImmutableCohortActiveLifetimeReplay
    V2_Integration_PooledImmutableCohortTerminalJoinReplay
    V2_Integration_PooledImmutableCohortPartialSetupReplay
    V2_Integration_PooledImmutableCohortTypedResetReplay
    V2_Integration_PooledImmutableCohortConstructionLifetimeReplay
    V2_Integration_PooledImmutableCohortActualRecorderReplay
    V2_Integration_PooledImmutableCohortColdFailureReplay
    V2_Integration_PooledImmutableCohortReplacedStageReplay
    V2_Integration_PooledImmutableCohortConcurrentBindingReplay
    V2_Integration_PooledImmutableCohortOppositeSourcesReplay
    V2_Integration_PooledImmutableCohortRetirementClaimReplay
    V2_Integration_PooledImmutableCohortWorkerDestructionReplay
    V2_Integration_PooledImmutableCohortOppositeAdmissionReplay
    V2_Integration_PooledImmutableExistingWorkerBorrowReplay
)


# Exact original owner/author hardening successors; preserve all prior cases.
foreach(_case IN ITEMS
    PooledImmutableCompleteCohort.DirectedOwnerProgramSealsActualRolesAndEdgesBeforeCapture
    PooledImmutableCompleteCohort.DistinctBoundariesRaceOneFreshOriginalRemoteOwner
    PooledImmutableCompleteCohort.RetainedFacadeRejectsMutationAfterCohortRetirement)
    string(REPLACE "." "_" _name "${_case}")
    add_v2_test(V2_Unit_${_name}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=${_case}
        LABELS "V2;Unit;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    add_v2_test(V2_Integration_${_name}Replay
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py" $<TARGET_FILE:v2_test_pooled_complete_cohort> --gtest_filter=${_case} --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_name}Replay)
endforeach()

add_v2_test(V2_Integration_PooledRequiredTerminalCohortProofReplay
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cohort_original_cases.py"
        $<TARGET_FILE:v2_test_pooled_complete_cohort>
        --gtest_filter=PooledImmutableCompleteCohort.MissingRequiredTerminalCohortNeverClassifiesSafeRetirement
        --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
if(TEST V2_Integration_PooledRequiredTerminalCohortProofReplay)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledRequiredTerminalCohortProofReplay)
endif()
