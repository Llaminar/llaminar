#[[
@file V2NativePeerVisibilityTests.cmake
@brief Actual original native CCL visibility control for the archived causal byte failure.

The retired custom peer test's red remains external evidence. These controls
exercise actual NCCL/RCCL with its original local physical cohort, exact257-byte
unaligned geometry, alternating immutable sources and byte/guard oracle.
]]
if(TARGET v2_integration_captured_transfer_channel)
    foreach(_visibility_backend CUDA ROCm)
        if((_visibility_backend STREQUAL "CUDA" AND HAVE_CUDA AND HAVE_NCCL) OR
           (_visibility_backend STREQUAL "ROCm" AND HAVE_ROCM AND HAVE_RCCL))
            add_v2_test(V2_Integration_CapturedNativePeerVisibility_${_visibility_backend}2_257Offset1
                COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                    --gtest_filter=CapturedTransferNativePeerVisibility.${_visibility_backend}OriginalInventory257Offset1
                LABELS "V2;Integration;ProductionTestPreflight;${_visibility_backend};GPU;NativeCollective;Transfer;GraphCapture;ByteExact;Regression"
                TIMEOUT 30 MPI_PROCS 2 NO_MODELS)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
                V2_Integration_CapturedNativePeerVisibility_${_visibility_backend}2_257Offset1)
        endif()
    endforeach()
endif()
