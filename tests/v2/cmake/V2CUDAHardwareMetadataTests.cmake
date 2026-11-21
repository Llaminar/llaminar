## @file V2CUDAHardwareMetadataTests.cmake
#  @brief Register device-free loader and native CUDA hardware metadata tests.
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

# Typed optional-driver admission contract, driven through the shared native
# loader probe for genuine absence, breakage and loadability; no CUDA SDK,
# driver or device required.
#
# Build-owned native loader fixtures: the proof drives the real loader
# against dependency and import contracts this tree owns, on every
# supported host architecture, instead of inspecting arbitrary system
# libraries. The stub advertises a SONAME different from its on-disk filename.
# Linking records the missing SONAME while keeping the stub's build output
# intact for incremental builds. Test runner packaging removes CMakeFiles, so
# the link stub stays in the build tree and the runnable fixtures stay sealed.
add_library(llaminar_native_loader_stub SHARED
    fixtures/native-loader/stub.c)
set_target_properties(llaminar_native_loader_stub PROPERTIES
    PREFIX "lib" OUTPUT_NAME "native-loader-link-stub" NO_SONAME TRUE
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/native-loader-link-stub")
target_link_options(llaminar_native_loader_stub PRIVATE
    "LINKER:-soname,libllaminar-absent-dependency.so.9")

add_library(llaminar_native_loader_missing_dependency SHARED
    fixtures/native-loader/missing_dependency_fixture.c)
set_target_properties(llaminar_native_loader_missing_dependency PROPERTIES
    PREFIX "lib" OUTPUT_NAME "llaminar-native-loader-missing-dependency"
    SKIP_BUILD_RPATH TRUE)
target_link_libraries(llaminar_native_loader_missing_dependency PRIVATE
    llaminar_native_loader_stub)
# The fixture has no undefined imports of its own, so the default
# --as-needed link would drop the dependency record entirely; the explicit
# no-as-needed link keeps the owned NEEDED entry that the runtime loader
# must fail to resolve.
target_link_options(llaminar_native_loader_missing_dependency PRIVATE
    -Wl,--no-as-needed)

add_library(llaminar_native_loader_undefined_symbol SHARED
    fixtures/native-loader/undefined_symbol_fixture.c)
set_target_properties(llaminar_native_loader_undefined_symbol PROPERTIES
    PREFIX "lib" OUTPUT_NAME "llaminar-native-loader-undefined-symbol")

add_library(llaminar_native_loader_loadable SHARED
    fixtures/native-loader/loadable_fixture.c)
set_target_properties(llaminar_native_loader_loadable PROPERTIES
    PREFIX "lib" OUTPUT_NAME "llaminar-native-loader-loadable")

add_executable(v2_unit_cuda_discovery_admission
    unit/backends/Test__CUDADiscoveryAdmission.cpp
    ../../src/v2/backends/cuda/CUDADiscoveryAdmission.cpp)
target_include_directories(v2_unit_cuda_discovery_admission PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_cuda_discovery_admission PRIVATE cxx_std_20)
target_compile_definitions(v2_unit_cuda_discovery_admission PRIVATE
    LLAMINAR_NATIVE_LOADER_MISSING_DEPENDENCY_FIXTURE="$<TARGET_FILE:llaminar_native_loader_missing_dependency>"
    LLAMINAR_NATIVE_LOADER_UNDEFINED_SYMBOL_FIXTURE="$<TARGET_FILE:llaminar_native_loader_undefined_symbol>"
    LLAMINAR_NATIVE_LOADER_LOADABLE_FIXTURE="$<TARGET_FILE:llaminar_native_loader_loadable>")
target_link_libraries(v2_unit_cuda_discovery_admission PRIVATE
    GTest::gtest_main ${CMAKE_DL_LIBS})
# The fixtures are addressed by absolute path, not linked, so the proof
# target needs explicit build dependencies for them to exist before it runs.
add_dependencies(v2_unit_cuda_discovery_admission
    llaminar_native_loader_missing_dependency
    llaminar_native_loader_undefined_symbol
    llaminar_native_loader_loadable)
add_v2_test(V2_Unit_CUDADiscoveryAdmission
    COMMAND $<TARGET_FILE:v2_unit_cuda_discovery_admission>
    LABELS "V2;Unit;DeviceFree;CUDA;HardwareMetadata;Discovery;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_CUDADiscoveryAdmissionContract
    COMMAND $<TARGET_FILE:v2_unit_cuda_discovery_admission> --gtest_repeat=20
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;HardwareMetadata;Discovery;Contract;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 60)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_CUDADiscoveryAdmissionContract)

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

add_v2_test(V2_Integration_CUDAVerifierDispatchTableBindingReplay
    COMMAND $<TARGET_FILE:v2_unit_cuda_hardware_metadata>
        --gtest_filter=CUDAHardwareMetadata.VerifierDispatchUsesAdmittedNativeArchitecture
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;HardwareMetadata;GroupedVerifierRows;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CUDAVerifierDispatchTableBindingReplay)
