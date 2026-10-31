include(${CMAKE_CURRENT_LIST_DIR}/V2NativeModuleOwnership.cmake)
# Pure physical-publication and actual attention-policy replay: no CUDA runtime.
add_executable(v2_unit_cuda_hardware_metadata
    unit/backends/Test__CUDAHardwareMetadata.cpp)
target_include_directories(v2_unit_cuda_hardware_metadata PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_cuda_hardware_metadata PRIVATE cxx_std_20)
target_link_libraries(v2_unit_cuda_hardware_metadata PRIVATE
    GTest::gtest_main Threads::Threads)
add_v2_test(V2_Unit_CUDAHardwareMetadata
    COMMAND $<TARGET_FILE:v2_unit_cuda_hardware_metadata>
    LABELS "V2;Unit;DeviceFree;CUDA;HardwareMetadata;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_CUDAHardwareMetadataReplay
    COMMAND $<TARGET_FILE:v2_unit_cuda_hardware_metadata>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;HardwareMetadata;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_CUDAHardwareMetadataReplay)

# Native authentication is deliberately separate from the device-free Unit proof.
# The test visits every visible device and explicitly leaves absent hardware unqualified.
if(HAVE_CUDA)
    add_executable(v2_integration_cuda_hardware_metadata_native
        integration/backends/Test__CUDAHardwareMetadataNative.cu
        integration/backends/CUDAHardwareQueryObserver.cpp)
    target_compile_features(v2_integration_cuda_hardware_metadata_native PRIVATE cxx_std_20)
    target_include_directories(v2_integration_cuda_hardware_metadata_native PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
    set_target_properties(v2_integration_cuda_hardware_metadata_native PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_RUNTIME_LIBRARY Shared
        V2_TEST_DEVICE_SCOPE FullInventory)
    v2_link_native_test_core(v2_integration_cuda_hardware_metadata_native cuda_backend
        GTest::gtest_main CUDA::cudart ${CMAKE_DL_LIBS})
    # Export observers so calls in the shared backend use the same counters as
    # static linkage; RTLD_NEXT always invokes the selected native runtime.
    foreach(_cuda_query cudaGetDeviceCount cudaGetDeviceProperties cudaDeviceGetPCIBusId)
        target_link_options(v2_integration_cuda_hardware_metadata_native PRIVATE
            "LINKER:--export-dynamic-symbol=${_cuda_query}")
    endforeach()
    add_v2_test(V2_Integration_CUDAHardwareMetadataNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_cuda_hardware_metadata_native>
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;HardwareMetadata;GraphCapture;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60
        RESOURCE_LOCK "cuda_gpu_0;cuda_gpu_1"
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_CUDAHardwareMetadataNativeReplay)
endif()

add_v2_test(V2_Integration_CUDARouterQ8HardwareBindingReplay
    COMMAND $<TARGET_FILE:v2_unit_cuda_hardware_metadata>
        --gtest_filter=CUDAHardwareMetadata.RouterBindingUsesAdmittedLimitsAndRejectsIncompleteCaptureIdentity
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;HardwareMetadata;MoE;Router;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_CUDARouterQ8HardwareBindingReplay)

add_v2_test(V2_Integration_CUDAColumnAssemblyHardwareBindingReplay
    COMMAND $<TARGET_FILE:v2_unit_cuda_hardware_metadata>
        --gtest_filter=CUDAHardwareMetadata.ColumnAssemblyUsesExactPhysicalCountAndFreshFunctionOccupancy
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;HardwareMetadata;TensorParallel;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_CUDAColumnAssemblyHardwareBindingReplay)

# Source ownership complements typed bindings and native no-rediscovery proof.
add_v2_test(V2_Unit_GPUHardwareQueryOwnership
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_gpu_hardware_query_ownership.py
    LABELS "V2;Unit;DeviceFree;HardwareMetadata;Architecture;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUHardwareQueryOwnership
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_gpu_hardware_query_ownership.py
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;HardwareMetadata;Architecture;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_GPUHardwareQueryOwnership)

# Shared context-owned execution-resource replay is independent of CUDA eligibility.

if(HAVE_CUDA)
    add_v2_test(V2_Integration_CUDAProjectionWorkspaceHardwareFactsReplay
        COMMAND $<TARGET_FILE:v2_unit_cuda_quantised_gemm_workspace>
            --gtest_filter=Test__CUDAQuantisedGemmKernel_Workspace.Qwen36LargeMDeclaresBoundedGroupedParticipantForMTPTotality:Test__CUDAQuantisedGemmKernel_Workspace.ExplicitPlanningHardwareKeepsUnobservedPoliciesPureAndSMCacheIdentityExact
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;Planning;Workspace;HardwareMetadata;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_CUDAProjectionWorkspaceHardwareFactsReplay)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/V2NativeDeviceBindingTests.cmake)
