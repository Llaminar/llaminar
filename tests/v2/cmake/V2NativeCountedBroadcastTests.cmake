#[=[
@file V2NativeCountedBroadcastTests.cmake
@brief Register actual-loader byte ABI and original retained native-pair replays.

The device-free entries use the existing loader/DSO fixtures. Native entries
reuse one original prepared source/PMA/communicator for row-valid ACK and
arbitrary-byte wire replay. No model/lookup placement qualification is implied.
]=]
include_guard(DIRECTORY)

# Release selects the original ABI/loader/native-transfer owners independently.
# Their existing registrations retain add_v2_test's exact owner filtering.
if(NOT V2_PERF_TESTS_ONLY OR
   TARGET v2_unit_nccl_rows_abi OR TARGET v2_unit_rccl_rows_abi OR
   TARGET v2_unit_nccl_path_configured OR TARGET v2_unit_rccl_path_configured OR
   TARGET v2_integration_captured_transfer_channel)
    foreach(_bytes_backend IN ITEMS NCCL RCCL)
        string(TOLOWER "${_bytes_backend}" _bytes_suffix)
        set(_bytes_target "v2_unit_${_bytes_suffix}_rows_abi")
        if(TARGET ${_bytes_target})
            add_v2_test(V2_Integration_${_bytes_backend}NativeByteBroadcastABI
                COMMAND $<TARGET_FILE:${_bytes_target}> --gtest_filter=NativeRowsABI.NativeBytes*
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Collective;CPU;Dependency;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_bytes_backend}NativeByteBroadcastABI)
        endif()
        set(_bytes_path_target "v2_unit_${_bytes_suffix}_path_configured")
        if(TARGET ${_bytes_path_target})
            add_v2_test(V2_Integration_${_bytes_backend}NativeByteMissingABINoFallback
                COMMAND $<TARGET_FILE:${_bytes_path_target}> --gtest_filter=${_bytes_backend}ConfiguredLibrary.MissingNativeByteABIHasNoFixedFallback
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Collective;CPU;Dependency;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_bytes_backend}NativeByteMissingABINoFallback)
            add_v2_test(V2_Integration_${_bytes_backend}CompletedSemanticProtocolReplay
                COMMAND $<TARGET_FILE:${_bytes_path_target}>
                    --gtest_filter=${_bytes_backend}ConfiguredLibrary.CompletedProtocolRejectsMissingOldAndFutureWithoutCallbacks:${_bytes_backend}ConfiguredLibrary.EarlierRequiredSymbolWinsOverCompletedProtocolFailure
                    --gtest_repeat=20
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Collective;CPU;Dependency;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_bytes_backend}CompletedSemanticProtocolReplay)
        endif()
    endforeach()
    if(TARGET v2_integration_captured_transfer_channel)
        target_include_directories(v2_integration_captured_transfer_channel PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        foreach(_bytes_backend IN ITEMS CUDA ROCm)
            if((_bytes_backend STREQUAL "CUDA" AND HAVE_CUDA AND HAVE_NCCL) OR
               (_bytes_backend STREQUAL "ROCm" AND HAVE_ROCM AND HAVE_RCCL))
                foreach(_bytes_shape IN ITEMS WireReplay WideWireReplay ReaderAckReplay SourceWaitReaderAckReplay)
                    add_v2_test(V2_Integration_${_bytes_backend}NativeCompletedCollective_${_bytes_shape}
                        COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                            --gtest_filter=NativeCompletedCollective.${_bytes_backend}_${_bytes_shape}
                        LABELS "V2;Integration;ProductionTestPreflight;Collective;GraphCapture;LiveExtent;ByteExact;${_bytes_backend};Devices2;Regression"
                        MPI_PROCS 2 NO_MODELS TIMEOUT 30)
                    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
                        V2_Integration_${_bytes_backend}NativeCompletedCollective_${_bytes_shape})
                endforeach()
            endif()
        endforeach()
    endif()
endif()

# Addressed original-communicator cases declare every actual participant. DevicesN
# labels name their exact required geometry for selection and qualification.
# A selected case fails missing participants; it never aliases, skips or shrinks.
if(NOT V2_PERF_TESTS_ONLY AND TARGET v2_integration_captured_transfer_channel)
    foreach(_addressed_backend IN ITEMS CUDA ROCm)
        if((_addressed_backend STREQUAL "CUDA" AND HAVE_CUDA AND HAVE_NCCL) OR
           (_addressed_backend STREQUAL "ROCm" AND HAVE_ROCM AND HAVE_RCCL))
            foreach(_addressed_degree IN ITEMS 2 4 8)
                math(EXPR _addressed_last "${_addressed_degree} - 1")
                foreach(_addressed_source IN ITEMS 0 ${_addressed_last})
                    set(_addressed_case "${_addressed_backend}_Addressed${_addressed_degree}Source${_addressed_source}Replay")
                    set(_addressed_optional "")
                    # These authored wide geometries can be absent locally. Required
                    # two-device and ROCm4 gates stay strict even if a board is lost.
                    if((_addressed_backend STREQUAL "CUDA" AND _addressed_degree GREATER 2) OR
                       (_addressed_backend STREQUAL "ROCm" AND _addressed_degree EQUAL 8))
                        set(_addressed_optional ";HardwareOptional")
                    endif()
                    add_v2_test(V2_Integration_NativeCompleted_${_addressed_case}
                        COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                            --gtest_filter=NativeCompletedCollective.${_addressed_case}
                        LABELS "V2;Integration;ProductionTestPreflight;Collective;GraphCapture;LiveExtent;ByteExact;${_addressed_backend};Devices${_addressed_degree};TraceReplay;Regression${_addressed_optional}"
                        MPI_PROCS 2 NO_MODELS TIMEOUT 30)
                    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeCompleted_${_addressed_case})
                endforeach()
            endforeach()
        endif()
    endforeach()
endif()

# Original connector observations prove empty/tiny step economy at retained capacity.
if(NOT V2_PERF_TESTS_ONLY AND TARGET v2_integration_captured_transfer_channel)
    foreach(_progress_backend IN ITEMS CUDA ROCm)
        if((_progress_backend STREQUAL "CUDA" AND HAVE_CUDA AND HAVE_NCCL) OR
           (_progress_backend STREQUAL "ROCm" AND HAVE_ROCM AND HAVE_RCCL))
            add_v2_test(V2_Integration_NativeCompleted_${_progress_backend}_Addressed2LiveProgressReplay
                COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                    --gtest_filter=NativeCompletedCollective.${_progress_backend}_Addressed2LiveProgressReplay
                LABELS "V2;Integration;ProductionTestPreflight;Collective;GraphCapture;LiveExtent;Economy;${_progress_backend};Devices2;TraceReplay;Regression"
                MPI_PROCS 2 NO_MODELS TIMEOUT 30)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeCompleted_${_progress_backend}_Addressed2LiveProgressReplay)
        endif()
    endforeach()
endif()

# The original Python owner proves actual scheduling, typed receipts and absent-card
# registration without a GPU probe, compiler or alternate native test implementation.
if(NOT V2_PERF_TESTS_ONLY)
    add_v2_test(V2_Integration_NativeCompletedOptionalHardwareEligibility
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_production_parity_campaigns.py
            ProductionParityCampaignTest.test_optional_native_geometry_uses_actual_inventory_and_keeps_required_cells_strict
            ProductionParityCampaignTest.test_optional_native_geometry_rejects_bad_hardware_and_dropped_receipt_evidence
            ProductionParityCampaignTest.test_optional_rank_transfer_geometry_requires_distinct_known_numa_owners
            ProductionParityCampaignTest.test_optional_native_geometry_reuse_requires_same_current_hardware
            ProductionParityCampaignTest.test_optional_native_exporter_uses_original_owner_and_preserves_native_failure
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Collective;Inventory;Regression"
        ENVIRONMENT "LLAMINAR_TEST_MPI_SESSION_HELPER=$<TARGET_FILE:llaminar_mpi_session>"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_NativeCompletedOptionalHardwareEligibility)
endif()
