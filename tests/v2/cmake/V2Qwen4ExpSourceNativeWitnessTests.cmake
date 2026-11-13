# Actual source-consumption predicate replay, never native/model qualification.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpConsumedNativeSourceReplay
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.ConsumedNativeSourceRejectsAbsentOriginalPreparation:Qwen4ExpSourceAdmissionCases.ConsumedNativeSourceRejectsPreboundUnusedOriginalAuthority
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpConsumedNativeSourceReplay)
endif()

# Source-only world2 metadata; real model startup is separately discovered as FullSourceAdmission.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpCPUJointWorld2SourceMetadata
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.CPUJointWorld2StaticRAMOffUsesExactSocketRoles:Qwen4ExpSourceAdmissionCI.CPUJointWorld2ExportsAllThreeOriginalPhysicalRoles --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Admission;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpCPUJointWorld2SourceMetadata)
endif()

# Exact physical-socket contract for the source-only CPU joint requirement.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpCPUJointPhysicalSocketMetadata
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCI.CPUJointSocketRequirementIsOriginalAndConditional:Qwen4ExpSourceAdmissionCI.CPUJointBoundRanksUseOriginalNodeAndPhysicalSocketIdentity:Qwen4ExpSourceAdmissionCI.CPUJointNativeSocketProjectionPreservesOriginalFacts:Qwen4ExpSourceAdmissionCI.CPUJointSourceObservationCountUsesOriginalLocalMembership --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Admission;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpCPUJointPhysicalSocketMetadata)
endif()
if(Python3_Interpreter_FOUND)
    add_v2_test(V2_Integration_Qwen4ExpCPUJointPhysicalSocketSelection
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_qfn_source_admission_ci.py
            QFNSourceAdmissionCI.test_cpu_joint_socket_selection_uses_original_physical_packages
            QFNSourceAdmissionCI.test_cpu_joint_socket_selection_rejects_incomplete_stale_and_duplicate_facts
            QFNSourceAdmissionCI.test_ordinary_source_selection_does_not_infer_socket_requirements
            QFNSourceAdmissionCI.test_cpu_joint_completed_source_requires_original_bound_rank_receipts
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Admission;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpCPUJointPhysicalSocketSelection)
endif()

# Canonical moving-placement MTP cells preserve original physical metadata.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_Qwen4ExpCPUJointMovingMTPMetadata
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=Qwen4ExpSourceAdmissionCases.CPUJointMovingMTPExactRetainsOriginalRequests
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;MTP;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_Qwen4ExpCPUJointMovingMTPMetadata)
endif()
