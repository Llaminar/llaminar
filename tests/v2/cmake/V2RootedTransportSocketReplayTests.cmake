#[[
@file V2RootedTransportSocketReplayTests.cmake
@brief Register the rooted transport socket replays only on admitted distinct sockets.
The rooted publication transport replays drive the cross-socket UPI collective,
whose production program admits only distinct physical CPU owners. The
canonical discovered-socket authority (the immutable sysfs topology receipt
behind V2_TEST_SOCKETS) owns launch eligibility; a single-socket host records
an explicit unqualified configure status instead of failing an ineligible
geometry. Pure fixture callers admit an authenticated topology document
through v2_bind_test_host_topology and exercise this same registration module;
the genuine two-physical-owner requirement itself never changes.
]]
# The replay executable and the shared registration helper come from the
# including project; this module owns only the socket eligibility decision.
if(V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
    add_v2_test(V2_Integration_CPURootedGatherParallelCopyReplay
        COMMAND $<TARGET_FILE:v2_perf_cpu_rooted_publication_transport>
            --gtest_filter=Perf__CPURootedPublicationTransport.OriginalNUMAVariableRecordGatherReplay
        LABELS "V2;Integration;ProductionTestPreflight;CPU;NUMA;Collective;ShmemSpin;Replay;Regression"
        MPI_PROCS 2 NO_MODELS TIMEOUT 120
        RESOURCE_LOCK "ShmemSpin"
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "OMP_NUM_THREADS=3" "OMP_DYNAMIC=FALSE"
    )
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPURootedGatherParallelCopyReplay)
    list(APPEND V2_ROOTED_TRANSPORT_SOCKET_REPLAY_TESTS V2_Integration_CPURootedGatherParallelCopyReplay)

    # Terminal epoch replay runs the original admitted owner without payload timing.
    add_v2_test(V2_Integration_ShmemSpinAbsorbingAbortReplay
        COMMAND $<TARGET_FILE:v2_perf_cpu_rooted_publication_transport>
            --gtest_filter=Perf__CPURootedPublicationTransport.NativeEpochAbortWitnessReplay
        LABELS "V2;Integration;ProductionTestPreflight;CPU;Collective;ShmemSpin;Replay;Regression"
        MPI_PROCS 2 NO_MODELS TIMEOUT 60
        RESOURCE_LOCK "ShmemSpin"
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "OMP_NUM_THREADS=3" "OMP_DYNAMIC=FALSE")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ShmemSpinAbsorbingAbortReplay)
    list(APPEND V2_ROOTED_TRANSPORT_SOCKET_REPLAY_TESTS V2_Integration_ShmemSpinAbsorbingAbortReplay)
else()
    message(STATUS "CPU rooted transport socket replays: unqualified (requires two discovered physical sockets; found ${V2_TEST_SOCKETS}); no MPI launch")
endif()
