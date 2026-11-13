include(${CMAKE_CURRENT_LIST_DIR}/V2NativeModuleOwnership.cmake)
# Native production-ABI replay; no eager attention invocation initializes tables.
if(HAVE_CUDA)
    add_executable(v2_integration_cuda_attention_centroid_first_capture
        integration/kernels/cuda/attention/Test__CUDAAttentionCentroidFirstCapture.cpp)
    target_include_directories(v2_integration_cuda_attention_centroid_first_capture PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
    target_compile_features(v2_integration_cuda_attention_centroid_first_capture PRIVATE cxx_std_20)
    set_target_properties(v2_integration_cuda_attention_centroid_first_capture PROPERTIES
        V2_TEST_DEVICE_SCOPE FullInventory)
    v2_link_native_test_core(v2_integration_cuda_attention_centroid_first_capture cuda_backend
        GTest::gtest_main CUDA::cudart)
    add_v2_test(V2_Integration_CUDAAttentionCentroidFirstCapture
        COMMAND $<TARGET_FILE:v2_integration_cuda_attention_centroid_first_capture>
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;Attention;TurboQuant;AllCentroids;GraphCapture;RuntimeRetirement;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120
        RESOURCE_LOCK "cuda_gpu_0;cuda_gpu_1"
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_CUDAAttentionCentroidFirstCapture)
endif()
