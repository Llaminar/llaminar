# @file V2QFNActiveTensorShardedTests.cmake
# @brief Real-source grouped inference coverage for Dynamic QFN tensor shards.
# These cases own real model weights and remain separate from model-free preflight.
if(HAVE_ROCM)
add_v2_test(V2_Integration_QFNTP4DynamicTensorShardedStationaryReplay
    COMMAND $<TARGET_FILE:v2_integration_qwen4exp_mtp_admission>
        --gtest_filter=Qwen4ExpTensorShardedROCmTP4.DynamicResidencyOneCohortRetainsEpochAcrossGroupedReplay
    LABELS "V2;Integration;RealModel;FullModel;Qwen4Exp;MTP;ROCm;TP;ExpertOverlay;HardwareOptional;Devices4;NativeISA-gfx906;Regression"
    MPI_PROCS 1 TIMEOUT 900 REQUIRED_FILES ${_qfn_mtp_admission_files}
    RESOURCE_LOCK "rocm_gpu_0;rocm_gpu_1;rocm_gpu_2;rocm_gpu_3"
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0"
        "LLAMINAR_PERF_STATS_SUMMARY=1"
        "LLAMINAR_PERF_STATS_FILTER=mtp.grouped_decode_equivalent_stochastic_verifier_runs,mtp.grouped_decode_equivalent_greedy_verifier_runs,moe_overlay_residency.maintenance_not_started")

add_v2_test(V2_Integration_QFNROCmTP2CPUTP2DynamicTensorShardedMovementReplay
    COMMAND $<TARGET_FILE:v2_integration_qwen4exp_mtp_admission>
        --gtest_filter=Qwen4ExpMixedTensorShardedTP2.DynamicResidencyPublishesCompleteCohortSwapBeforeGroupedReplay
    LABELS "V2;Integration;RealModel;FullModel;Qwen4Exp;MTP;ROCm;TP;ExpertOverlay;HardwareOptional;Devices2;DistinctNUMARanks;NativeISA-gfx906;Regression"
    MPI_PROCS 2 TIMEOUT 900 REQUIRED_FILES ${_qfn_mtp_admission_files}
    RESOURCE_LOCK "rocm_gpu_0;rocm_gpu_1"
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0"
        "LLAMINAR_PERF_STATS_SUMMARY=1"
        "LLAMINAR_PERF_STATS_FILTER=mtp.grouped_decode_equivalent_stochastic_verifier_runs,mtp.grouped_decode_equivalent_greedy_verifier_runs,moe_overlay_residency.maintenance_not_started")
endif()

if(HAVE_CUDA)
add_v2_test(V2_Integration_QFNCUDATP2CPUTP2MovementPreservesGeneratedTokens
    COMMAND $<TARGET_FILE:v2_integration_qwen4exp_mtp_admission>
        --gtest_filter=Qwen4ExpMixedTensorShardedTP2.CUDACPUCohortMovementPreservesGeneratedTokens
    LABELS "V2;Integration;RealModel;FullModel;Qwen4Exp;MTP;CUDA;TP;ExpertOverlay;HardwareOptional;Devices2;DistinctNUMARanks;ByteExact;Replay;Regression"
    MPI_PROCS 2 TIMEOUT 900 REQUIRED_FILES ${_qfn_mtp_admission_files}
    RESOURCE_LOCK "cuda_gpu_0;cuda_gpu_1"
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1"
        "LLAMINAR_PERF_STATS_SUMMARY=1"
        "LLAMINAR_PERF_STATS_FILTER=mtp.grouped_decode_equivalent_stochastic_verifier_runs,mtp.grouped_decode_equivalent_greedy_verifier_runs,moe_overlay_residency.maintenance_not_started,moe.router_selected_descriptors_enqueued")
endif()
