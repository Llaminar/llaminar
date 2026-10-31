# Builder-owned replay of actual native worker creation, join and ignored-result
# callers. Installed Unit/preflight validates the source/compiler/DSO-bound
# proof without loading a GPU runtime or retaining the compiler/vendor source.
include_guard(DIRECTORY)

# This module emits a depfile from the tests subdirectory. Keep Ninja's path
# transformation explicit on CMake versions that provide this policy.
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()

if(HAVE_ROCM AND HAVE_RCCL AND NOT V2_PERF_TESTS_ONLY)
    set(RCCL_PROTOCOL_BUILD_DIR "${RCCL_BUILD_DIR}" CACHE PATH
        "Completed canonical RCCL native source/proof build")
    set(V2_RCCL_THREAD_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/test_rccl_thread_lifetime.py")
    set(V2_RCCL_THREAD_PROOF
        "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_thread_lifetime.json")
    set(V2_RCCL_THREAD_DEPFILE
        "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_thread_lifetime.d")
    list(GET CMAKE_HIP_ARCHITECTURES 0 V2_RCCL_THREAD_ARCHITECTURE)
    add_custom_command(OUTPUT "${V2_RCCL_THREAD_PROOF}"
        COMMAND ${Python3_EXECUTABLE} -B "${V2_RCCL_THREAD_SCRIPT}" compile
            --report "${V2_RCCL_THREAD_PROOF}" --library "${RCCL_LIBRARY}"
            --rccl-build "${RCCL_PROTOCOL_BUILD_DIR}" --compiler "${CMAKE_HIP_COMPILER}"
            --architecture "${V2_RCCL_THREAD_ARCHITECTURE}" --sdk "${ROCM_PATH}"
            --depfile "${V2_RCCL_THREAD_DEPFILE}"
        DEPENDS "${V2_RCCL_THREAD_SCRIPT}"
            integration/build/rccl_thread_lifetime_source.py
            integration/build/rccl_thread_lifetime_probe.cpp
            integration/build/rccl_thread_lifetime_original_probe.cpp
            integration/build/rccl_thread_lifetime_caller_probe.cpp
            integration/build/test_rccl_selector_taxonomy.py
            integration/build/test_rccl_small_allgather_policy.py
            unit/scripts/check_cpu_prepared_allocation_policy.py
            unit/scripts/check_gpu_blocking_sync_policy.py
            "${REPO_ROOT}/scripts/docker/apply-rccl-capture-patch.sh"
            "${REPO_ROOT}/scripts/docker/apply-rccl-patch-chain.py"
            "${REPO_ROOT}/scripts/docker/patches/rccl-created-thread-lifetime.patch"
            "${REPO_ROOT}/scripts/docker/patches/rccl-hip-capture-event-wait.patch"
            "${REPO_ROOT}/scripts/docker/patches/rccl-device-live-rows.patch"
            "${REPO_ROOT}/scripts/docker/patches/rccl-native-host-storage.patch"
            "${REPO_ROOT}/scripts/docker/patches/rccl-communicator-protocol-defaults.patch"
            "${REPO_ROOT}/scripts/docker/patches/rccl-collective-selector-taxonomy.patch"
            "${REPO_ROOT}/scripts/docker/patches/rccl-small-allgather-ll-policy.patch"
            "${RCCL_PROTOCOL_BUILD_DIR}/compile_commands.json"
            "${RCCL_PROTOCOL_BUILD_DIR}/.llaminar-rccl-commit"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/group.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/init.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/group.h"
            "${RCCL_LIBRARY}" "${CMAKE_HIP_COMPILER}"
        DEPFILE "${V2_RCCL_THREAD_DEPFILE}"
        VERBATIM)
    add_custom_target(v2_rccl_thread_lifetime_proof DEPENDS "${V2_RCCL_THREAD_PROOF}")
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_rccl_thread_lifetime_proof)
    foreach(V2_RCCL_THREAD_SUITE Unit Integration)
        set(V2_RCCL_THREAD_TEST "V2_${V2_RCCL_THREAD_SUITE}_RCCLThreadLifetimeReplay")
        add_test(NAME ${V2_RCCL_THREAD_TEST}
            COMMAND ${Python3_EXECUTABLE} -B "${V2_RCCL_THREAD_SCRIPT}" verify
                --report "${V2_RCCL_THREAD_PROOF}" --library "${RCCL_LIBRARY}")
        set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_RCCL_THREAD_TEST}
            v2_rccl_thread_lifetime_proof)
        set_tests_properties(${V2_RCCL_THREAD_TEST} PROPERTIES
            LABELS "V2;${V2_RCCL_THREAD_SUITE};RCCL;DeviceFree;Replay;Regression" TIMEOUT 30)
    endforeach()
    set_property(TEST V2_Integration_RCCLThreadLifetimeReplay APPEND PROPERTY LABELS ProductionTestPreflight)
endif()
