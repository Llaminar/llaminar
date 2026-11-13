# The source-admission replay exercises the same pure helper and actual CPU root
# as the real attention fixture, without loading a model or entering a driver.
add_executable(v2_unit_rocm_attention_fixture_admission
    unit/planning/Test__ROCmAttentionFixtureAdmission.cpp)
target_include_directories(v2_unit_rocm_attention_fixture_admission PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_link_libraries(v2_unit_rocm_attention_fixture_admission PRIVATE
    llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_ROCmAttentionFixtureAdmission
    COMMAND $<TARGET_FILE:v2_unit_rocm_attention_fixture_admission>
    LABELS "V2;Unit;DeviceFree;Attention;PhysicalMemory;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_ROCmAttentionSourceAdmissionReplay
    COMMAND $<TARGET_FILE:v2_unit_rocm_attention_fixture_admission>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Attention;PhysicalMemory;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_ROCmAttentionSourceAdmissionReplay)

# Real GGUF source extraction remains separate from model-free preflight.
if(TARGET v2_integration_rocm_flash_attention_parity)
    add_v2_test(V2_Integration_ROCmFlashAttentionRealQwen2SourceAdmission
        COMMAND $<TARGET_FILE:v2_integration_rocm_flash_attention_parity>
            --gtest_filter=Test__ROCmFlashAttentionParity.FlashAttn2_RealQwen2Layer3ContextParity:Test__ROCmFlashAttentionParity.FlashAttn2_RealQwen2Layer3InputSensitivity
        LABELS "V2;Integration;ROCm;CPU;RealModel;GGUF;Attention;PhysicalMemory;Regression"
        MPI_PROCS 1 RESOURCE_LOCK rocm_gpu_0 TIMEOUT 120)
endif()
