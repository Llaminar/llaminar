# The complete shared-gate program complements the actual wave-helper original-tree control.
# Include after v2_integration_shared_expert_column_gate has been declared.
if(HAVE_ROCM)
    add_v2_test(V2_Integration_ROCmSharedGateCrossDeviceCapturedReplay
        COMMAND $<TARGET_FILE:v2_integration_shared_expert_column_gate>
            --gtest_filter=SharedExpertColumnGate.ROCmCrossDeviceCapturedTraceV1
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MoE;SharedExpert;GraphCapture;ByteExact;Replay;Regression"
        MPI_PROCS 1 NO_MODELS TIMEOUT 60)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_ROCmSharedGateCrossDeviceCapturedReplay)
endif()
