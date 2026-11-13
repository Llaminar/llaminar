# @file V2PlanningHostServiceEvidenceTests.cmake
# @brief Actual admission replay for bounded CPU service evidence and shared pipeline ownership.
if(TARGET v2_test_memory_planner)
    target_include_directories(v2_test_memory_planner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_sources(v2_test_memory_planner PRIVATE unit/planning/Test__PlanningHostServiceEvidence.cpp)
    add_v2_test(V2_Integration_PlanningHostServiceEvidenceReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=PlanningHostServiceEvidence.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Planning;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PlanningHostServicePreQuotaAdmission
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=OrchestrationCandidateNgramAdmission.HostedServiceBOMPrecedesOptionalPlacement:OrchestrationCandidateNgramAdmission.FixedVRAMRequiresNoHostedServiceClass
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Planning;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PipelineHostServiceDeclarationPublication
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=MoEPipelineNgramPublication.PublishesExactPlacementWithTheAdmittedRows
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Qwen4Exp;Planning;Memory;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_PlanningHostServiceEvidenceReplay
        V2_Integration_PlanningHostServicePreQuotaAdmission
        V2_Integration_PipelineHostServiceDeclarationPublication)
endif()
