#[[
@file V2PooledReplicatedReaderNativeTests.cmake
@brief Original source-frame dependency proof complementing actual native reader replay.

NativeCompletedCollective owns actual native payload/metadata/ACK/cancellation
capture. This file retains the independent device-free original-frame contract.
]]
# NativeCompletedCollective.{CUDA,ROCm}_ReaderAckReplay owns the actual native
# reader metadata/ACK/reset/fault/abort regression. Its explicit preflight entries
# live in V2NativeCountedBroadcastTests.cmake; do not preserve a second captured
# channel alias that selects an obsolete homogeneous transport.

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
