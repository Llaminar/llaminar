#[[
@file V2CapturedPeerPayloadTests.cmake
@brief Register explicit native peer-payload ownership, extents and cancellation replays.

These use the existing channel executable and its original MPI/device scope.
Native cases require actual cached bidirectional peer access and fail if absent;
no mapped route substitutes for an unavailable peer endpoint. Native qualification
is separate from the fast device-free ownership preflight and timing thresholds.
]]
if(TARGET v2_integration_captured_transfer_channel)
    foreach(_peer_backend CUDA ROCm)
        if((_peer_backend STREQUAL "CUDA" AND HAVE_CUDA) OR
           (_peer_backend STREQUAL "ROCm" AND HAVE_ROCM))
            # This admission gate is useful on both actual Complete and
            # unsupported None/Partial hardware; its properties say whether
            # the peer payload actually ran, without a skip or mapped substitute.
            add_v2_test(V2_Integration_CapturedPeerPayload_${_peer_backend}2_OriginalInventory
                COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                    --gtest_filter=CapturedTransferPeerAdmission.${_peer_backend}OriginalInventory
                LABELS "V2;Integration;ProductionTestPreflight;${_peer_backend};GPU;NativePeerAdmission;Transfer;GraphCapture;Regression"
                TIMEOUT 30 MPI_PROCS 1 NO_MODELS)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
                V2_Integration_CapturedPeerPayload_${_peer_backend}2_OriginalInventory)
            add_v2_test(V2_Integration_CapturedPeerPayload_${_peer_backend}2_RoundTrip
                COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                    --gtest_filter=CapturedTransferChannelDevices.${_peer_backend}_${_peer_backend}_PeerPayloadRetainedRoundTrip
                LABELS "V2;Integration;${_peer_backend};GPU;NativePeerPayload;Transfer;GraphCapture;Regression"
                TIMEOUT 30 MPI_PROCS 1 NO_MODELS)
            add_v2_test(V2_Integration_CapturedPeerPayload_${_peer_backend}2_CountsAndReset
                COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                    --gtest_filter=DeviceCountedTransferChannel.${_peer_backend}_${_peer_backend}_PeerPayload
                LABELS "V2;Integration;${_peer_backend};GPU;NativePeerPayload;Transfer;GraphCapture;TraceReplay;Regression"
                TIMEOUT 30 MPI_PROCS 1 NO_MODELS)
            foreach(_peer_phase BeforeAcquire PublishedPeerAbsent PublishedPeerLate)
                add_v2_test(V2_Integration_CapturedPeerPayload_${_peer_backend}2_${_peer_phase}
                    COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                        --gtest_filter=CapturedTransferCancellation.${_peer_backend}Peer${_peer_phase}
                    LABELS "V2;Integration;${_peer_backend};GPU;NativePeerPayload;Transfer;Cancellation;GraphCapture;Regression"
                    TIMEOUT 30 MPI_PROCS 1 NO_MODELS)
            endforeach()
        endif()
    endforeach()
endif()
if(TARGET v2_test_captured_transfer_channel_binding)
    add_v2_test(V2_Integration_CapturedPeerPayloadOwnershipReplay
        COMMAND $<TARGET_FILE:v2_test_captured_transfer_channel_binding>
            --gtest_filter=CapturedTransferChannelBindingTest.PeerPayloadOriginalOwnerGeometryAndFailureReplay
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;CPU;DeviceFree;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1"
            "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CapturedPeerPayloadOwnershipReplay)
endif()
