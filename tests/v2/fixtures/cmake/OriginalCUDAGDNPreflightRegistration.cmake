# @file OriginalCUDAGDNPreflightRegistration.cmake
# @brief Preserve the unconditional native registration from 9ce2f7b36 as a negative control.
# This source-owned control must expose the CUDA build obligation when CUDA is disabled.

# Exact stage-level regression for external event prejoins and scoped capture retirement.
add_v2_test(V2_Integration_CUDA_GDNTensorCaptureOwnership
    COMMAND $<TARGET_FILE:v2_integration_cuda_gemm_parity>
        --gtest_filter=Test__CUDAGemmParity.GDNProjectionStageFusesCUDAQuantizedQKVAndZSmallM:Test__CUDAGemmParity.GDNProjectionStageCapturedMixedBF16AndFP32UsesPhysicalFormatSubgroups
        --gtest_brief=1
    LABELS "V2;Integration;ProductionTestPreflight;CUDA;GEMM;GDN;GraphCapture;TransferEngine;Regression"
    MPI_PROCS 1 NO_MODELS TIMEOUT 120)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CUDA_GDNTensorCaptureOwnership)
