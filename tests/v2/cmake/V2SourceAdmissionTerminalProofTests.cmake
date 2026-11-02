# @file V2SourceAdmissionTerminalProofTests.cmake
# @brief Runner cleanup cannot substitute for original process-native lifetime retirement.
foreach(_terminal_case test_canonical_matrix_custom_main_owns_terminal_publication_before_mpi_finalize test_native_terminal_invocation_rejects_stale_directory_before_launch test_old_runner_only_report_schema_cannot_certify_final_native_retirement test_runner_receipt_before_abort_is_not_final_native_retirement
        test_terminal_retirement_rejects_missing_malformed_and_foreign_rank_or_invocation
        test_long_source_identity_retirement_and_nested_neighbor_cleanup
        test_source_artifact_cleanup_rejects_nested_symlinks_and_malformed_identity)
    set(_terminal_gate "V2_Integration_QFNSourceTerminal_${_terminal_case}")
    add_v2_test(${_terminal_gate}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_qfn_source_admission_ci.py QFNSourceAdmissionCI.${_terminal_case}
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Admission;Lifecycle;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_terminal_gate})
endforeach()
if(TARGET v2_test_model_parity_definition)
    foreach(_terminal_case ProcessCompletionFollowsLifetimeForPassingAndFailedTests FailedProcessLifetimeCannotPublishTerminalProof)
        set(_terminal_gate "V2_Integration_QFNSourceTerminal_${_terminal_case}")
        add_v2_test(${_terminal_gate} COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionRetirement.${_terminal_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Admission;Lifecycle;Replay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_terminal_gate})
    endforeach()
endif()
