# @file V2MoEOverlaySharedHiddenTrafficTests.cmake
# @brief Explicit terminal-only shared-row byte ownership and reset regressions.
#
# The existing full Unit protocol suite runs these cases too. Native union-copy
# tests separately prove that the recorded bytes were actually written on GPU.
if(TARGET v2_unit_moe_overlay_activation_epoch_protocol)
    foreach(_shared_hidden_case IN ITEMS
        SharedHiddenTrafficIsTerminalOwnedAndResetWithOriginalEpoch
        SharedHiddenTrafficRejectsForeignPublisher)
        set(_shared_hidden_test "V2_Integration_MoEOverlay_${_shared_hidden_case}")
        add_v2_test(${_shared_hidden_test}
            COMMAND $<TARGET_FILE:v2_unit_moe_overlay_activation_epoch_protocol>
                --gtest_filter=Test__MoEOverlayActivationEpochProtocol.${_shared_hidden_case}
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;MoE;Traffic;Replay;Regression"
            NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30)
        if(TEST ${_shared_hidden_test})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_shared_hidden_test})
        endif()
    endforeach()
    unset(_shared_hidden_case)
    unset(_shared_hidden_test)
endif()
