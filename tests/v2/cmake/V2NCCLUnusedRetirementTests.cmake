# Native immediate-retirement coverage for the actual CUDA readiness race.
# The deterministic source replay remains device-free and independently registered.
if(HAVE_CUDA)
    add_executable(v2_integration_nccl_unused_communicator_retirement
        integration/collective/Test__NCCLUnusedCommunicatorRetirement.cpp)
    target_link_libraries(v2_integration_nccl_unused_communicator_retirement PRIVATE
        llaminar2_core GTest::gtest GTest::gtest_main CUDA::cudart)
    add_v2_test(V2_Integration_NCCLUnusedRetirement_DirectOwnerRetiresTwentyUnusedCliques_2Devices
        COMMAND $<TARGET_FILE:v2_integration_nccl_unused_communicator_retirement>
            --gtest_filter=NCCLUnusedCommunicatorRetirement.DirectOwnerRetiresTwentyUnusedCliques
        LABELS "V2;Integration;ProductionTestPreflight;NCCL;CUDA;Admission;Teardown;Devices2;Replay;Regression"
        MPI_PROCS 1 NO_MODELS TIMEOUT 120)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_NCCLUnusedRetirement_DirectOwnerRetiresTwentyUnusedCliques_2Devices)
endif()
