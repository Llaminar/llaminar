## @file V2ReadAheadTests.cmake
## @brief Native read-ahead candidates and unconditional device-free ordering replay.
#
# These header-only tests link GTest without the core target, so they declare
# the C++20 contract required by production span and defaulted-equality APIs.

# Exact native stream/event ordering proof stays device-free; actual overlap is
# qualified separately with complete captured CUDA/ROCm parents.
add_executable(v2_test_read_ahead_capture_dag
    unit/execution/graph/Test__ReadAheadCaptureDAG.cpp)
if(TARGET v2_test_read_ahead_capture_dag)
target_compile_features(v2_test_read_ahead_capture_dag PRIVATE cxx_std_20)
target_include_directories(v2_test_read_ahead_capture_dag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2)
target_link_libraries(v2_test_read_ahead_capture_dag PRIVATE GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ReadAheadCaptureDAG
    COMMAND $<TARGET_FILE:v2_test_read_ahead_capture_dag>
    LABELS "V2;Unit;DeviceFree;GraphCapture;ReadAhead;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_ReadAheadCaptureDAGReplay
    COMMAND $<TARGET_FILE:v2_test_read_ahead_capture_dag>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;GraphCapture;ReadAhead;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ReadAheadCaptureDAGReplay)
endif()

# Setup-only diagnostic selection is tested without a model or GPU runtime.
add_executable(v2_test_read_ahead_diagnostic_selection
    unit/execution/graph/Test__ReadAheadDiagnosticSelection.cpp)
if(TARGET v2_test_read_ahead_diagnostic_selection)
target_compile_features(v2_test_read_ahead_diagnostic_selection PRIVATE cxx_std_20)
target_include_directories(v2_test_read_ahead_diagnostic_selection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2)
target_link_libraries(v2_test_read_ahead_diagnostic_selection PRIVATE GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ReadAheadDiagnosticSelection
    COMMAND $<TARGET_FILE:v2_test_read_ahead_diagnostic_selection>
    LABELS "V2;Unit;DeviceFree;ReadAhead;Policy;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_ReadAheadDiagnosticSelectionReplay
    COMMAND $<TARGET_FILE:v2_test_read_ahead_diagnostic_selection>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ReadAhead;Policy;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ReadAheadDiagnosticSelectionReplay)
endif()

# Native parents prove admitted source ownership, causal ordering and reset.
foreach(_read_ahead_backend IN ITEMS CUDA ROCm)
    if((_read_ahead_backend STREQUAL "CUDA" AND HAVE_CUDA) OR
       (_read_ahead_backend STREQUAL "ROCm" AND HAVE_ROCM))
        string(TOLOWER "${_read_ahead_backend}" _read_ahead_suffix)
        set(_read_ahead_target v2_perf_read_ahead_captured_overlap_${_read_ahead_suffix})
        add_executable(${_read_ahead_target} integration/execution/Test__ReadAheadCapturedOverlap.cpp)
        string(TOUPPER "${_read_ahead_backend}" _read_ahead_definition)
        target_compile_definitions(${_read_ahead_target} PRIVATE GPU_CONTEXT_TEST_BACKEND_${_read_ahead_definition}=1)
        if(LLAMINAR_SHARED_CORE)
            target_link_libraries(${_read_ahead_target} PRIVATE llaminar2_core GTest::gtest GTest::gtest_main)
        else()
            target_link_libraries(${_read_ahead_target} PRIVATE
                -Wl,--start-group llaminar2_core
                -Wl,--whole-archive ${_read_ahead_suffix}_backend -Wl,--no-whole-archive
                -Wl,--end-group GTest::gtest GTest::gtest_main)
        endif()
        # Native source preparation needs the same lexical GPU process owner
        # as production entrypoints; plain GoogleTest main owns no runtime.
        v2_bind_gpu_process_test_entry(${_read_ahead_target} v2_core_gtest_main)
        # Immutable startup inventory discovery may touch either vendor. Declare
        # this before registration so physical ownership and locks are complete.
        v2_test_device_scope(${_read_ahead_target} FullInventory)
        add_v2_test(V2_Integration_ReadAheadCapturedOverlap_${_read_ahead_backend}
            COMMAND $<TARGET_FILE:${_read_ahead_target}>
                --gtest_filter=*CausalForkJoinAndMatchedReplay
            LABELS "V2;Integration;ProductionTestPreflight;GraphCapture;ReadAhead;NativeQualification;Replay;Regression;${_read_ahead_backend}"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ReadAheadCapturedOverlap_${_read_ahead_backend})
        add_v2_test(V2_Perf_ReadAheadMatchedReplay_${_read_ahead_backend}
            COMMAND $<TARGET_FILE:${_read_ahead_target}>
                --gtest_filter=*CausalForkJoinAndMatchedReplay
            LABELS "V2;Performance;ReadAhead;NativeQualification;Unprofiled;${_read_ahead_backend}"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120)
    endif()
endforeach()

# The actual QFN schema and both executor entrypoints prove declaration safety.
if(TARGET v2_test_qwen4exp_metadata)
add_v2_test(V2_Integration_ReadAheadWorkspaceDeclarationReplay
    COMMAND $<TARGET_FILE:v2_test_qwen4exp_metadata>
        --gtest_filter=Qwen4ExpGraphStateBindings.ReadAheadWorkspaceDeclarationRetainsOriginalBufferBOMAndRejectsExecution
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ReadAhead;Workspace;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ReadAheadWorkspaceDeclarationReplay)
endif()

# Immutable event-closed producer snapshots share the original admission path.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/V2ReadAheadSnapshotTests.cmake)
