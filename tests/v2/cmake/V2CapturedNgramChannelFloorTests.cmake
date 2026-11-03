#[[
@file V2CapturedNgramChannelFloorTests.cmake
@brief Matched captured native and mapped ngram transfer performance floors.

The Release performance role reuses the complete integration fixture, including
its counted-channel replay dependency. Compile that closure in either build
mode so the shared fixture cannot acquire an unresolved helper silently. Timing
does not enter Unit/preflight or replace variable-length lookup qualification.
]]
include_guard(DIRECTORY)
if(NOT HAVE_CUDA OR NOT HAVE_ROCM)
    return()
endif()
add_executable(v2_perf_captured_ngram_channel_floor
    integration/transfer/Test__CapturedTransferChannel.cpp
    integration/transfer/Test__DeviceCountedTransferChannel.cpp
    integration/transfer/CountedChannelTestKernels.cu
    integration/transfer/CountedChannelTestKernels.hip
    integration/mpi_gtest_main.cpp)
set_source_files_properties(integration/transfer/CountedChannelTestKernels.hip PROPERTIES LANGUAGE HIP)
set_target_properties(v2_perf_captured_ngram_channel_floor PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
target_link_libraries(v2_perf_captured_ngram_channel_floor PRIVATE llaminar2_core GTest::gtest)
# The shared original-source fixture uses the same nested include root as its integration target.
target_include_directories(v2_perf_captured_ngram_channel_floor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
v2_test_device_scope(v2_perf_captured_ngram_channel_floor FullInventory)
foreach(_ngram_floor_direction CUDA_ROCm ROCm_CUDA)
    foreach(_ngram_floor_payload Decode1440 M512737280)
        add_v2_test(V2_Perf_CapturedNgramChannelFloor_${_ngram_floor_direction}_${_ngram_floor_payload}
            COMMAND $<TARGET_FILE:v2_perf_captured_ngram_channel_floor>
                --gtest_filter=CapturedTransferChannelFloor.${_ngram_floor_direction}_${_ngram_floor_payload}
            LABELS "V2;Performance;CUDA;ROCm;CrossVendor;TransferEngine;GraphCapture;Ngram"
            MPI_PROCS 1 NO_MODELS TIMEOUT 30 RESOURCE_LOCK "cuda_gpu_0;rocm_gpu_0")
    endforeach()
endforeach()

# Native homogeneous fixed-size floors use original library transports even
# when direct peer access is absent. They grant no live-count ngram support.
foreach(_native_floor_backend CUDA ROCm)
    if(_native_floor_backend STREQUAL "CUDA")
        set(_native_floor_lock "cuda_gpu_0;cuda_gpu_1")
    else()
        set(_native_floor_lock "rocm_gpu_0;rocm_gpu_1")
    endif()
    foreach(_native_floor_direction Forward Reverse)
        foreach(_native_floor_payload Header64 Decode1440 M512737280)
            add_v2_test(V2_Perf_CapturedNgramChannelFloor_${_native_floor_backend}_Native${_native_floor_direction}_${_native_floor_payload}
                COMMAND $<TARGET_FILE:v2_perf_captured_ngram_channel_floor>
                    --gtest_filter=CapturedTransferChannelFloor.${_native_floor_backend}_Native${_native_floor_direction}_${_native_floor_payload}
                LABELS "V2;Performance;${_native_floor_backend};NativeCollective;FixedPayload;GraphCapture;Ngram"
                MPI_PROCS 1 NO_MODELS TIMEOUT 30
                RESOURCE_LOCK "${_native_floor_lock}")
        endforeach()
    endforeach()
endforeach()

# Diagnostic alternatives compare the exact SAME endpoint pair and payload with
# native12. They do not select an inference transport or require direct peer access.
foreach(_mapped_floor_backend CUDA ROCm)
    if(_mapped_floor_backend STREQUAL "CUDA")
        set(_mapped_floor_lock "cuda_gpu_0;cuda_gpu_1")
    else()
        set(_mapped_floor_lock "rocm_gpu_0;rocm_gpu_1")
    endif()
    foreach(_mapped_floor_direction Forward Reverse)
        foreach(_mapped_floor_payload Header64 Decode1440 M512737280)
            add_v2_test(V2_Perf_CapturedNgramChannelFloor_${_mapped_floor_backend}_Mapped${_mapped_floor_direction}_${_mapped_floor_payload}
                COMMAND $<TARGET_FILE:v2_perf_captured_ngram_channel_floor>
                    --gtest_filter=CapturedTransferChannelFloor.${_mapped_floor_backend}_Mapped${_mapped_floor_direction}_${_mapped_floor_payload}
                LABELS "V2;Performance;${_mapped_floor_backend};DiagnosticAlternative;MappedChannel;FixedPayload;GraphCapture;Ngram"
                MPI_PROCS 1 NO_MODELS TIMEOUT 30 RESOURCE_LOCK "${_mapped_floor_lock}")
        endforeach()
    endforeach()
endforeach()
