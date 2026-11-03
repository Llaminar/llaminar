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
