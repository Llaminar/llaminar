# Execute the actual custom MPI entry; the body must not manufacture an owner.
if(TARGET v2_integration_mpi_topology)
    add_v2_test(V2_Integration_MPITopologyProcessLifetime
        COMMAND $<TARGET_FILE:v2_integration_mpi_topology>
            --gtest_filter=MPITopologyProcessLifetime.MainOwnsNativeAdmissionBeforeDiscovery
        LABELS "V2;Integration;ProductionTestPreflight;MPI;Topology;Ownership;RuntimeRetirement;Replay;Regression"
        MPI_PROCS 2 NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MPITopologyProcessLifetime)
endif()
