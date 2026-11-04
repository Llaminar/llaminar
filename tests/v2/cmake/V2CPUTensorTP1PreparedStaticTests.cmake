# @file V2CPUTensorTP1PreparedStaticTests.cmake
# @brief Register original-loader/Static regression for explicit CPU TensorSharded degree one.
#
# The public compute policy stays unchanged while one member owns complete
# matrices. Source preparation and Static registry must not manufacture a
# multi-member cohort or a fragment authority. Native model qualification is
# separate from this device-free actual-producer replay.
if(TARGET v2_unit_expert_gemm_registry)
    add_v2_test(V2_Integration_CPUTensorTP1PreparedStaticReplay
        COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
            --gtest_filter=CPUTensorTP1PreparedStatic.* --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Source;Residency;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
    if(TEST V2_Integration_CPUTensorTP1PreparedStaticReplay)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUTensorTP1PreparedStaticReplay)
    endif()
endif()
