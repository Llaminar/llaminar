# Original-owner peer-transfer regressions. Each native case keeps one admitted
# source/PMA for its entire lifetime; the other actual MPI rank participates in
# setup and retirement. A rank-local two-GPU cohort is required, not a pair of
# GPUs borrowed across CPU sockets. The existing preparation/error controls
# remain in their separate original registrations.
include_guard(GLOBAL)

set(_v2_original_peer_sources integration/mpi_gtest_main.cpp)
set(_v2_original_peer_backends "")
foreach(_backend CUDA ROCm)
    if((_backend STREQUAL "CUDA" AND HAVE_CUDA) OR
       (_backend STREQUAL "ROCm" AND HAVE_ROCM))
        list(APPEND _v2_original_peer_sources unit/moe/Test__GPUExpertTransfer_${_backend}.cpp)
        list(APPEND _v2_original_peer_backends ${_backend})
    endif()
endforeach()

add_executable(v2_integration_original_expert_peer_transfer ${_v2_original_peer_sources})
target_include_directories(v2_integration_original_expert_peer_transfer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(v2_integration_original_expert_peer_transfer PRIVATE llaminar2_core GTest::gtest)
v2_test_device_scope(v2_integration_original_expert_peer_transfer FullInventory)

# The aggregate loses only cases that receive an exact process-owned replacement.
# Construct both registrations and exclusions from this single finite roster.
set(_v2_original_peer_exclusions "FiniteBodySiblingNativeIntegration.*")
foreach(_backend IN LISTS _v2_original_peer_backends)
    foreach(_case EventPolledTransferIsByteExact FP16ContiguousTransferIsByteExact
                  BF16ContiguousTransferIsByteExact FP32ContiguousTransferIsByteExact)
        set(_filter "Test__ExpertTierGpuPeerTransfer${_backend}.${_case}")
        set(_name "V2_Integration_OriginalExpertPeerTransfer_${_backend}_${_case}")
        list(APPEND _v2_original_peer_exclusions "${_filter}")
        add_v2_test(${_name}
            COMMAND $<TARGET_FILE:v2_integration_original_expert_peer_transfer>
                --gtest_filter=${_filter}
            LABELS "V2;Integration;ProductionTestPreflight;${_backend};HardwareOptional;Devices2;DirectedPeerEdge;MoE;ExpertOverlay;GPUTransfer;Async;PerfStats;ByteExact;NoBlocking;Ownership;Lifecycle;Regression"
            MPI_PROCS 2 NO_MODELS TIMEOUT 30)
        if(TEST ${_name})
            list(APPEND V2_ORIGINAL_EXPERT_PEER_PREFLIGHT_TESTS ${_name})
        endif()
    endforeach()
endforeach()
list(JOIN _v2_original_peer_exclusions ":" V2_GPU_EXPERT_TRANSFER_AGGREGATE_EXCLUSIONS)
