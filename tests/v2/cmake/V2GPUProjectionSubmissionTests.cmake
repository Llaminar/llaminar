# Replay the shared production owner without linking a GPU runtime.
add_executable(v2_unit_gpu_projection_submission unit/backends/Test__GPUProjectionSubmission.cpp)
target_include_directories(v2_unit_gpu_projection_submission PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_gpu_projection_submission PRIVATE cxx_std_20)
target_link_libraries(v2_unit_gpu_projection_submission PRIVATE GTest::gtest_main Threads::Threads)
add_v2_test(V2_Unit_GPUProjectionSubmission
    COMMAND $<TARGET_FILE:v2_unit_gpu_projection_submission>
    LABELS "V2;Unit;DeviceFree;Ownership;Replay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUProjectionSubmissionReplay
    COMMAND $<TARGET_FILE:v2_unit_gpu_projection_submission>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Replay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUProjectionSubmissionReplay)
if(TARGET v2_integration_cuda_graph_capture)
    add_v2_test(V2_Integration_CUDAProjectionSubmissionNativeReplay
        COMMAND $<TARGET_FILE:v2_integration_cuda_graph_capture>
            --gtest_filter=Test__CUDAGraphCapture.ProjectionResourcesRetainWholeNativeCapture
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;Ownership;GraphCapture;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK cuda_gpu_0 TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CUDAProjectionSubmissionNativeReplay)
endif()

# Exact pool retention replay uses a registered hardware-free worker.
add_executable(v2_unit_gpu_projection_context_borrow unit/backends/Test__GPUProjectionContextBorrow.cpp)
target_include_directories(v2_unit_gpu_projection_context_borrow PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_gpu_projection_context_borrow PRIVATE cxx_std_20)
target_link_libraries_with_groups(v2_unit_gpu_projection_context_borrow
    llaminar2_core GTest::gtest_main Threads::Threads)
add_v2_test(V2_Unit_GPUProjectionContextBorrow
    COMMAND $<TARGET_FILE:v2_unit_gpu_projection_context_borrow>
    LABELS "V2;Unit;DeviceFree;Ownership;Replay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_GPUProjectionContextBorrowReplay
    COMMAND $<TARGET_FILE:v2_unit_gpu_projection_context_borrow>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUProjectionContextBorrowReplay)

# Compile actual CUDA header types using the selected SDK without a runtime link.
if(HAVE_CUDA)
    add_executable(v2_unit_cuda_graph_capture_owner_types
        unit/backends/Test__CUDAGraphCaptureOwnerTypes.cpp)
    target_compile_features(v2_unit_cuda_graph_capture_owner_types PRIVATE cxx_std_20)
    target_compile_definitions(v2_unit_cuda_graph_capture_owner_types PRIVATE HAVE_CUDA=1
        "$<TARGET_PROPERTY:CUDA::cudart,INTERFACE_COMPILE_DEFINITIONS>")
    target_include_directories(v2_unit_cuda_graph_capture_owner_types PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
    target_include_directories(v2_unit_cuda_graph_capture_owner_types SYSTEM PRIVATE
        "$<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>")
    target_link_libraries(v2_unit_cuda_graph_capture_owner_types PRIVATE GTest::gtest_main)
    add_v2_test(V2_Unit_CUDAGraphCaptureOwnerTypes
        COMMAND $<TARGET_FILE:v2_unit_cuda_graph_capture_owner_types>
        LABELS "V2;Unit;DeviceFree;CUDA;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    add_v2_test(V2_Integration_CUDAGraphCaptureOwnerTypesReplay
        COMMAND $<TARGET_FILE:v2_unit_cuda_graph_capture_owner_types>
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CUDA;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CUDAGraphCaptureOwnerTypesReplay)
endif()
