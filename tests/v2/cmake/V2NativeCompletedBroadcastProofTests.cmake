#[=[
@file V2NativeCompletedBroadcastProofTests.cmake
@brief Authenticate builder-owned native completed Broadcast ABI/protocol replays.

Installed runners verify exact source/DSO/receipt observations without a compiler,
vendor checkout or GPU. These device-free entries complement the actual retained
CUDA/ROCm wire and reader-ACK integration cases; they do not certify transport or
model correctness. The original dependency build remains the receipt authority.
]=]
include_guard(DIRECTORY)

set(V2_NATIVE_COMPLETED_SCRIPT "${REPO_ROOT}/scripts/docker/native-completed-broadcast/proof.py")
file(GLOB V2_NATIVE_COMPLETED_INPUTS CONFIGURE_DEPENDS
    "${REPO_ROOT}/scripts/docker/native-completed-broadcast/*.py"
    "${REPO_ROOT}/scripts/docker/native-completed-broadcast/*.h"
    "${REPO_ROOT}/scripts/docker/native-completed-broadcast/*.cpp")

foreach(V2_NATIVE_COMPLETED_BACKEND IN ITEMS nccl rccl)
    if((V2_NATIVE_COMPLETED_BACKEND STREQUAL "nccl" AND HAVE_CUDA AND HAVE_NCCL) OR
       (V2_NATIVE_COMPLETED_BACKEND STREQUAL "rccl" AND HAVE_ROCM AND HAVE_RCCL))
        string(TOUPPER "${V2_NATIVE_COMPLETED_BACKEND}" V2_NATIVE_COMPLETED_LABEL)
        set(V2_NATIVE_COMPLETED_REPORT
            "${CMAKE_CURRENT_BINARY_DIR}/v2_${V2_NATIVE_COMPLETED_BACKEND}_completed_broadcast.json")
        if(V2_NATIVE_COMPLETED_BACKEND STREQUAL "nccl")
            get_filename_component(V2_NATIVE_COMPLETED_LIBDIR "${NCCL_LIBRARY}" DIRECTORY)
            get_filename_component(V2_NATIVE_COMPLETED_PREFIX "${V2_NATIVE_COMPLETED_LIBDIR}" DIRECTORY)
            set(V2_NATIVE_COMPLETED_LIBRARY "${NCCL_LIBRARY}")
            set(V2_NATIVE_COMPLETED_RECEIPT "${V2_NATIVE_COMPLETED_LIBDIR}/llaminar-nccl-build.txt")
            set(V2_NATIVE_COMPLETED_INSTALLED
                "${V2_NATIVE_COMPLETED_PREFIX}/share/llaminar-nccl/completed-broadcast.json")
            add_custom_command(OUTPUT "${V2_NATIVE_COMPLETED_REPORT}"
                COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_COMPLETED_SCRIPT}" verify --backend nccl
                    --report "${V2_NATIVE_COMPLETED_INSTALLED}" --library "${NCCL_LIBRARY}"
                    --receipt "${V2_NATIVE_COMPLETED_RECEIPT}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${V2_NATIVE_COMPLETED_INSTALLED}" "${V2_NATIVE_COMPLETED_REPORT}"
                DEPENDS ${V2_NATIVE_COMPLETED_INPUTS}
                    "${REPO_ROOT}/scripts/docker/patches/nccl-device-completed-broadcast.patch"
                    "${REPO_ROOT}/scripts/docker/nccl-gin-selection/proof.py"
                    "${V2_NATIVE_COMPLETED_INSTALLED}" "${V2_NATIVE_COMPLETED_RECEIPT}" "${NCCL_LIBRARY}"
                VERBATIM)
        else()
            set(V2_NATIVE_COMPLETED_LIBRARY "${RCCL_LIBRARY}")
            set(V2_NATIVE_COMPLETED_RECEIPT "${RCCL_PROTOCOL_BUILD_DIR}/.llaminar-rccl-commit")
            file(STRINGS "${RCCL_PROTOCOL_BUILD_DIR}/CMakeCache.txt" V2_NATIVE_COMPLETED_SOURCE_ENTRY
                REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
            string(REGEX REPLACE "^CMAKE_HOME_DIRECTORY:INTERNAL=" ""
                V2_NATIVE_COMPLETED_SOURCE "${V2_NATIVE_COMPLETED_SOURCE_ENTRY}")
            if(NOT IS_DIRECTORY "${V2_NATIVE_COMPLETED_SOURCE}/src")
                message(FATAL_ERROR "Completed RCCL proof requires its actual source-build authority")
            endif()
            list(JOIN CMAKE_HIP_ARCHITECTURES "," V2_NATIVE_COMPLETED_ARCHITECTURE)
            add_custom_command(OUTPUT "${V2_NATIVE_COMPLETED_REPORT}"
                COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_COMPLETED_SCRIPT}" compile --backend rccl
                    --source "${V2_NATIVE_COMPLETED_SOURCE}" --build-dir "${RCCL_PROTOCOL_BUILD_DIR}"
                    --library "${RCCL_LIBRARY}" --receipt "${V2_NATIVE_COMPLETED_RECEIPT}"
                    --compiler "${CMAKE_CXX_COMPILER}" --sdk "${ROCM_PATH}" --architecture "${V2_NATIVE_COMPLETED_ARCHITECTURE}"
                    --report "${V2_NATIVE_COMPLETED_REPORT}"
                DEPENDS ${V2_NATIVE_COMPLETED_INPUTS}
                    "${REPO_ROOT}/scripts/docker/patches/rccl-device-completed-broadcast.patch"
                    "${RCCL_PROTOCOL_BUILD_DIR}/CMakeCache.txt" "${V2_NATIVE_COMPLETED_RECEIPT}"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/collectives.cc"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/enqueue.cc"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/init.cc"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/proxy.cc"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/group.cc"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/common.cu"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/common.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/broadcast.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/common.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/llaminar_device_payload.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/llaminar_device_completion.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/llaminar_device_p2p.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/sendrecv.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/primitives.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/prims_simple.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/device/generate.py"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/include/llaminar_p2p_launch.h"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/include/llaminar_p2p_entry.h"
                    "${REPO_ROOT}/scripts/docker/llvm-lto-linker/artifact.py"
                    "${REPO_ROOT}/scripts/build/gpu_spill_guard.py"
                    "${RCCL_PROTOCOL_BUILD_DIR}/llaminar-final-allocation/configuration.json"
                    "${V2_NATIVE_COMPLETED_SOURCE}/src/include/llaminar_device_extent.h"
                    "${RCCL_LIBRARY}" "${CMAKE_CXX_COMPILER}"
                VERBATIM)
        endif()
        set(V2_NATIVE_COMPLETED_TARGET "v2_${V2_NATIVE_COMPLETED_BACKEND}_completed_broadcast_proof")
        add_custom_target(${V2_NATIVE_COMPLETED_TARGET} DEPENDS "${V2_NATIVE_COMPLETED_REPORT}")
        set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS ${V2_NATIVE_COMPLETED_TARGET})
        foreach(V2_NATIVE_COMPLETED_SUITE Unit Integration)
            set(V2_NATIVE_COMPLETED_TEST
                "V2_${V2_NATIVE_COMPLETED_SUITE}_${V2_NATIVE_COMPLETED_LABEL}CompletedBroadcastContract")
            add_test(NAME ${V2_NATIVE_COMPLETED_TEST}
                COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_COMPLETED_SCRIPT}" verify
                    --backend "${V2_NATIVE_COMPLETED_BACKEND}" --report "${V2_NATIVE_COMPLETED_REPORT}"
                    --library "${V2_NATIVE_COMPLETED_LIBRARY}" --receipt "${V2_NATIVE_COMPLETED_RECEIPT}")
            set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NATIVE_COMPLETED_TEST}
                ${V2_NATIVE_COMPLETED_TARGET})
            set_tests_properties(${V2_NATIVE_COMPLETED_TEST} PROPERTIES
                LABELS "V2;${V2_NATIVE_COMPLETED_SUITE};${V2_NATIVE_COMPLETED_LABEL};DeviceFree;Replay;Regression" TIMEOUT 30)
            if(V2_NATIVE_COMPLETED_SUITE STREQUAL "Integration")
                set_property(TEST ${V2_NATIVE_COMPLETED_TEST} APPEND PROPERTY LABELS ProductionTestPreflight)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${V2_NATIVE_COMPLETED_TEST})
                set(V2_NATIVE_P2P_ENTRY_TEST "V2_Integration_${V2_NATIVE_COMPLETED_LABEL}AddressedP2pOwnedEntryContract")
                add_test(NAME ${V2_NATIVE_P2P_ENTRY_TEST}
                    COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_COMPLETED_SCRIPT}" verify
                        --backend "${V2_NATIVE_COMPLETED_BACKEND}" --report "${V2_NATIVE_COMPLETED_REPORT}"
                        --library "${V2_NATIVE_COMPLETED_LIBRARY}" --receipt "${V2_NATIVE_COMPLETED_RECEIPT}")
                set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NATIVE_P2P_ENTRY_TEST}
                    ${V2_NATIVE_COMPLETED_TARGET})
                set_tests_properties(${V2_NATIVE_P2P_ENTRY_TEST} PROPERTIES
                    LABELS "V2;Integration;ProductionTestPreflight;${V2_NATIVE_COMPLETED_LABEL};DeviceFree;Replay;Dependency;KernelResource;Regression" TIMEOUT 30)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${V2_NATIVE_P2P_ENTRY_TEST})
                set(V2_NATIVE_HOST_BARRIER_TEST "V2_Integration_${V2_NATIVE_COMPLETED_LABEL}CompletedHostBarrierContract")
                add_test(NAME ${V2_NATIVE_HOST_BARRIER_TEST}
                    COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_COMPLETED_SCRIPT}" verify
                        --backend "${V2_NATIVE_COMPLETED_BACKEND}" --report "${V2_NATIVE_COMPLETED_REPORT}"
                        --library "${V2_NATIVE_COMPLETED_LIBRARY}" --receipt "${V2_NATIVE_COMPLETED_RECEIPT}")
                set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NATIVE_HOST_BARRIER_TEST}
                    ${V2_NATIVE_COMPLETED_TARGET})
                set_tests_properties(${V2_NATIVE_HOST_BARRIER_TEST} PROPERTIES
                    LABELS "V2;Integration;ProductionTestPreflight;${V2_NATIVE_COMPLETED_LABEL};DeviceFree;Replay;Lifecycle;Regression" TIMEOUT 30)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${V2_NATIVE_HOST_BARRIER_TEST})
                set(V2_NATIVE_RETIREMENT_TEST "V2_Integration_${V2_NATIVE_COMPLETED_LABEL}CompletedAbortRetirementContract")
                add_test(NAME ${V2_NATIVE_RETIREMENT_TEST}
                    COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_COMPLETED_SCRIPT}" verify
                        --backend "${V2_NATIVE_COMPLETED_BACKEND}" --report "${V2_NATIVE_COMPLETED_REPORT}"
                        --library "${V2_NATIVE_COMPLETED_LIBRARY}" --receipt "${V2_NATIVE_COMPLETED_RECEIPT}")
                set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NATIVE_RETIREMENT_TEST}
                    ${V2_NATIVE_COMPLETED_TARGET})
                set_tests_properties(${V2_NATIVE_RETIREMENT_TEST} PROPERTIES
                    LABELS "V2;Integration;ProductionTestPreflight;${V2_NATIVE_COMPLETED_LABEL};DeviceFree;Replay;Lifecycle;Regression" TIMEOUT 30)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${V2_NATIVE_RETIREMENT_TEST})
                set(V2_NATIVE_COUNT_ABORT_TEST "V2_Integration_${V2_NATIVE_COMPLETED_LABEL}CompletedCountAbortReplay")
                add_test(NAME ${V2_NATIVE_COUNT_ABORT_TEST}
                    COMMAND ${Python3_EXECUTABLE} -B "${V2_NATIVE_COMPLETED_SCRIPT}" verify
                        --backend "${V2_NATIVE_COMPLETED_BACKEND}" --report "${V2_NATIVE_COMPLETED_REPORT}"
                        --library "${V2_NATIVE_COMPLETED_LIBRARY}" --receipt "${V2_NATIVE_COMPLETED_RECEIPT}")
                set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NATIVE_COUNT_ABORT_TEST}
                    ${V2_NATIVE_COMPLETED_TARGET})
                set_tests_properties(${V2_NATIVE_COUNT_ABORT_TEST} PROPERTIES
                    LABELS "V2;Integration;ProductionTestPreflight;${V2_NATIVE_COMPLETED_LABEL};DeviceFree;Replay;Lifecycle;Regression" TIMEOUT 30)
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${V2_NATIVE_COUNT_ABORT_TEST})
            endif()
        endforeach()
    endif()
endforeach()

foreach(V2_NATIVE_COMPLETED_SUITE Unit Integration)
    set(V2_NATIVE_COMPLETED_POLICY_TEST
        "V2_${V2_NATIVE_COMPLETED_SUITE}_NativeCompletedBroadcastProofPackaging")
    add_test(NAME ${V2_NATIVE_COMPLETED_POLICY_TEST}
        COMMAND ${Python3_EXECUTABLE} -B
            "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_completed_broadcast_proof.py")
    set_tests_properties(${V2_NATIVE_COMPLETED_POLICY_TEST} PROPERTIES
        LABELS "V2;${V2_NATIVE_COMPLETED_SUITE};DeviceFree;BuildSystem;Replay;Regression" TIMEOUT 30)
    if(V2_NATIVE_COMPLETED_SUITE STREQUAL "Integration")
        set_property(TEST ${V2_NATIVE_COMPLETED_POLICY_TEST} APPEND PROPERTY LABELS ProductionTestPreflight)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${V2_NATIVE_COMPLETED_POLICY_TEST})
    endif()
endforeach()

# The actual builder proves original GROUP failure and protocol3; installed policy rejects omissions.
add_test(NAME V2_Integration_NativeCompletedHostBarrierProofPackaging
    COMMAND ${Python3_EXECUTABLE} -B
        "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_completed_broadcast_proof.py"
        NativeCompletedProofTests.test_host_barrier_and_protocol_cannot_be_omitted)
set_tests_properties(V2_Integration_NativeCompletedHostBarrierProofPackaging PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;BuildSystem;Replay;Lifecycle;Regression" TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeCompletedHostBarrierProofPackaging)


# Resource policy regression is explicit in preflight in addition to the shared
# source/DSO-bound dependency report that includes actual compiler evidence.
add_test(NAME V2_Integration_RCCLAddressedP2pResourceProofPackaging
    COMMAND ${Python3_EXECUTABLE} -B
        "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_completed_broadcast_proof.py"
        NativeCompletedProofTests.test_rccl_owned_resources_require_complete_call_free_entry_coverage)
set_tests_properties(V2_Integration_RCCLAddressedP2pResourceProofPackaging PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;BuildSystem;Replay;KernelResource;Regression" TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RCCLAddressedP2pResourceProofPackaging)

# Live extent, required empty-channel metadata and passive original progress are
# an explicit focused regression within the same dependency proof authority.
add_test(NAME V2_Integration_NativeCompletedLiveStepProofPackaging
    COMMAND ${Python3_EXECUTABLE} -B
        "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_completed_broadcast_proof.py"
        NativeCompletedProofTests.test_live_step_and_original_progress_observations_are_mandatory)
set_tests_properties(V2_Integration_NativeCompletedLiveStepProofPackaging PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;BuildSystem;Replay;Economy;Regression" TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeCompletedLiveStepProofPackaging)
