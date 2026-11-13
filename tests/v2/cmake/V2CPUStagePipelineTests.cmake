#[[
@file V2CPUStagePipelineTests.cmake
@brief Actual original CPU-stage serving and its device-free authority/arithmetic regressions.
Only physical two-socket original MPI membership admits the serving jobs. Larger
source-authored populations are metadata/arithmetic evidence, never fabricated
native hardware qualification. CPU execution remains host-owned throughout.
]]
v2_test_target_selected(v2_integration_cpu_stage_pipeline cpu_stage_owner_selected)
if(cpu_stage_owner_selected)
    add_executable(v2_integration_cpu_stage_pipeline
        integration/execution/runner/Test__CPUStagePipelineGeneration.cpp
        integration/mpi_gtest_main.cpp)
    target_compile_features(v2_integration_cpu_stage_pipeline PRIVATE cxx_std_20)
    target_link_libraries(v2_integration_cpu_stage_pipeline PRIVATE llaminar2_core GTest::gtest)
    target_include_directories(v2_integration_cpu_stage_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
        foreach(cpu_stage_case IN ITEMS UPIApportioned UPITensorSharded MPIApportioned MPITensorSharded)
            set(cpu_stage_name "V2_Integration_CPUStagePipeline_${cpu_stage_case}OrdinaryReplay")
            add_v2_test(${cpu_stage_name}
                COMMAND $<TARGET_FILE:v2_integration_cpu_stage_pipeline>
                    "--gtest_filter=OriginalCPUStage/CPUStagePipelineGeneration.OriginalSocketsOrdinaryResetAndLiveRows/${cpu_stage_case}"
                LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;Pipeline;Lifecycle;Replay"
                MPI_PROCS 2 NO_MODELS TIMEOUT 120
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
                    "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${cpu_stage_name})
        endforeach()
    else()
        message(STATUS "CPU stage pipeline: unqualified without two distinct physical sockets; no MPI launch")
    endif()
endif()
if(TARGET v2_test_execution_plan_builder)
    add_v2_test(V2_Integration_CPUStageOriginalAuthorityReplay
        COMMAND $<TARGET_FILE:v2_test_execution_plan_builder>
            --gtest_filter=ResolvedRankOrchestration.CPUPipelineStage* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Pipeline;Ownership;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageOriginalAuthorityReplay)
    add_v2_test(V2_Integration_CPUStageOriginalTPValidationReplay
        COMMAND $<TARGET_FILE:v2_test_execution_plan_builder>
            --gtest_filter=ResolvedRankOrchestration.CPUPipelineStageOriginalTPValidationUsesDeclaredRanksAndModelGeometry
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Pipeline;Ownership;TP;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageOriginalTPValidationReplay)
endif()

if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_CPUStageOriginalMemoryScopeReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=MoEGlobalPipelineMemory.OriginalCPUStageMemoryInputsRequireExactProgramScopeAndRank
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Pipeline;MemoryBOM;Ownership;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageOriginalMemoryScopeReplay)
endif()

if(TARGET v2_test_global_orchestrator)
    add_v2_test(V2_Integration_CPUStageColdConstructionHostBoundaryReplay
        COMMAND $<TARGET_FILE:v2_test_global_orchestrator>
            --gtest_filter=Test__GlobalOrchestrator.ColdCPUConstructionWithoutNativeOwnersPreservesOriginalError --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Pipeline;Lifecycle;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageColdConstructionHostBoundaryReplay)
endif()
if(TARGET v2_integration_cpu_stage_pipeline AND V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
    add_v2_test(V2_Integration_CPUStageOriginalColdAbortDiagnostic
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/integration/scripts/test_cpu_pipeline_cold_abort.py
            --helper $<TARGET_FILE:llaminar_mpi_session> --probe $<TARGET_FILE:v2_integration_cpu_stage_pipeline>
            --mpirun ${MPIEXEC_EXECUTABLE}
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MPI;Pipeline;Lifecycle;Failure;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageOriginalColdAbortDiagnostic)
endif()

if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_CPUStageOriginalCapacityExchangeReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=MoEGlobalPipelineMemory.OriginalCPUStageCapacityExchange*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Pipeline;MemoryBOM;Ownership;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageOriginalCapacityExchangeReplay)
endif()
if(TARGET v2_integration_cpu_stage_pipeline AND V2_TEST_SOCKETS AND V2_TEST_SOCKETS GREATER_EQUAL 2)
    add_v2_test(V2_Integration_CPUStageOriginalAsymmetricCapacityReplay
        COMMAND $<TARGET_FILE:v2_integration_cpu_stage_pipeline>
            --gtest_filter=OriginalCPUStage/CPUStagePipelineGeneration.OriginalSocketsOrdinaryResetAndLiveRows/UPITensorShardedAsymmetricCapacity
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MoE;MPI;Pipeline;MemoryBOM;Failure;Replay"
        MPI_PROCS 2 NO_MODELS TIMEOUT 60
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageOriginalAsymmetricCapacityReplay)
    add_v2_test(V2_Integration_CPUStageOriginalCapacityFailureKindReplay
        COMMAND $<TARGET_FILE:v2_integration_cpu_stage_pipeline>
            --gtest_filter=OriginalCPUStageCapacity.StructuralFailureNeverSelectsFewerRows
        LABELS "V2;Integration;ProductionTestPreflight;CPU;MPI;Pipeline;MemoryBOM;Failure;Replay"
        MPI_PROCS 2 NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1"
            "LLAMINAR_SKIP_ROCM_STARTUP=1" "OMP_NUM_THREADS=1" "OMP_DYNAMIC=false")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUStageOriginalCapacityFailureKindReplay)
endif()
