## @file V2ReadAheadSnapshotTests.cmake
## @brief Immutable side-stream snapshot ordering replay and explicit native qualification.
#
# These header-only tests link GTest without the core target, so they declare
# the C++20 contract required by production span and defaulted-equality APIs.
add_executable(v2_test_read_ahead_snapshot_ordering
    unit/execution/graph/Test__ReadAheadSnapshotOrdering.cpp)
if(TARGET v2_test_read_ahead_snapshot_ordering)
    target_compile_features(v2_test_read_ahead_snapshot_ordering PRIVATE cxx_std_20)
    target_include_directories(v2_test_read_ahead_snapshot_ordering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2)
    target_link_libraries(v2_test_read_ahead_snapshot_ordering PRIVATE GTest::gtest GTest::gtest_main)
    foreach(_snapshot_lane IN ITEMS Unit Integration)
        if(_snapshot_lane STREQUAL "Integration")
            set(_snapshot_name V2_Integration_ReadAheadSnapshotOrderingReplay)
            set(_snapshot_labels "V2;Integration;ProductionTestPreflight;DeviceFree;ReadAhead;Snapshot;Replay;Regression")
        else()
            set(_snapshot_name V2_Unit_ReadAheadSnapshotOrdering)
            set(_snapshot_labels "V2;Unit;DeviceFree;ReadAhead;Snapshot;Replay;Regression")
        endif()
        add_v2_test(${_snapshot_name}
            COMMAND $<TARGET_FILE:v2_test_read_ahead_snapshot_ordering>
            LABELS "${_snapshot_labels}"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    endforeach()
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ReadAheadSnapshotOrderingReplay)
endif()
foreach(_snapshot_backend IN ITEMS CUDA ROCm)
    string(TOLOWER "${_snapshot_backend}" _snapshot_suffix)
    set(_snapshot_target v2_perf_read_ahead_captured_overlap_${_snapshot_suffix})
    if(TARGET ${_snapshot_target})
        add_v2_test(V2_Integration_ReadAheadCapturedSnapshot_${_snapshot_backend}
            COMMAND $<TARGET_FILE:${_snapshot_target}>
                --gtest_filter=ReadAheadCapturedOverlap.${_snapshot_backend}CapturedSnapshotPublication
            LABELS "V2;Integration;GraphCapture;ReadAhead;Snapshot;NativeQualification;UnqualifiedCandidate;Replay;Regression;${_snapshot_backend}"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120)
    endif()
endforeach()
