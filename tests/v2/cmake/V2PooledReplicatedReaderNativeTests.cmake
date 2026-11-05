#[[
@file V2PooledReplicatedReaderNativeTests.cmake
@brief Bind actual replicated-reader metadata kernels and payload receipts to native preflight.

The existing captured-channel executable owns original endpoint discovery,
streams and native teardown. Each backend executes its actual cached route;
coverage properties distinguish peer VRAM from genuinely unavailable-peer RAM.
These are correctness replay gates, never timing or full-model qualification.
]]
if(TARGET v2_integration_captured_transfer_channel)
    foreach(_reader_backend CUDA ROCm)
        if((_reader_backend STREQUAL "CUDA" AND HAVE_CUDA) OR
           (_reader_backend STREQUAL "ROCm" AND HAVE_ROCM))
            add_v2_test(V2_Integration_PooledReplicatedReader_${_reader_backend}2_CapturedMetadataReplay
                COMMAND $<TARGET_FILE:v2_integration_captured_transfer_channel>
                    --gtest_filter=PooledReplicatedReaderNative.${_reader_backend}OriginalCapturedMetadataReplay
                LABELS "V2;Integration;ProductionTestPreflight;${_reader_backend};GPU;Ngram;Transfer;Capture;TraceReplay;Regression"
                NO_MODELS MPI_PROCS 1 TIMEOUT 30)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
                V2_Integration_PooledReplicatedReader_${_reader_backend}2_CapturedMetadataReplay)
        endif()
    endforeach()
endif()

# Device-free original-frame declaration proof complements native reader capture.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledReplicatedReaderSourceFrameDependencyReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.SourceReaderFrameDeclarativeDependencyReplay
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ownership;Ngram;TraceReplay;Regression"
        NO_MODELS NO_MPI MPI_PROCS 1 TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledReplicatedReaderSourceFrameDependencyReplay)
endif()

# A remote-only original source retains coordinator/source roots without a local cache role.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledRemoteOnlySourceRootRegistration
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.RemoteOnlySourceRootRegistration
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Graph;Ownership;Ngram;Contract;Regression"
        NO_MODELS NO_MPI MPI_PROCS 1 TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledRemoteOnlySourceRootRegistration)
endif()
