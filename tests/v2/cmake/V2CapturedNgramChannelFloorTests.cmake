# Exact production-channel round-trip floors, using the original integration
# fixture source/main in a Release performance role. No automatic Unit/preflight
# timing or alternate HostBackend/SDK path is introduced.
include_guard(DIRECTORY)
if(NOT HAVE_CUDA OR NOT HAVE_ROCM)
    return()
endif()
add_executable(v2_perf_captured_ngram_channel_floor
    integration/transfer/Test__CapturedTransferChannel.cpp
    integration/mpi_gtest_main.cpp)
target_link_libraries(v2_perf_captured_ngram_channel_floor PRIVATE llaminar2_core GTest::gtest)
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
