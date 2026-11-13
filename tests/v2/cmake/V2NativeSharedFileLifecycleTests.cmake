# Complete native host-file replay belongs to the dependency builder. Installed
# Unit/preflight passively authenticates its exact source/compiler/DSO outcomes.
# This certificate covers partial file creation, never full GPU transport setup.
include_guard(DIRECTORY)
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()

set(V2_SHARED_FILE_SCRIPT "${REPO_ROOT}/scripts/docker/shared-file-lifecycle/proof.py")
set(V2_SHARED_FILE_PROBE "${REPO_ROOT}/scripts/docker/shared-file-lifecycle/probe.cpp")
foreach(V2_SHARED_FILE_SUITE Unit Integration)
    set(V2_SHARED_FILE_TEST "V2_${V2_SHARED_FILE_SUITE}_NativeSharedFileProofPackaging")
    add_test(NAME ${V2_SHARED_FILE_TEST}
        COMMAND ${Python3_EXECUTABLE} -B
            "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_shared_file_lifecycle.py")
    set_tests_properties(${V2_SHARED_FILE_TEST} PROPERTIES
        LABELS "V2;${V2_SHARED_FILE_SUITE};DeviceFree;BuildSystem;Regression" TIMEOUT 30)
endforeach()
set_property(TEST V2_Integration_NativeSharedFileProofPackaging APPEND PROPERTY LABELS ProductionTestPreflight)

if(HAVE_CUDA AND HAVE_NCCL AND NOT V2_PERF_TESTS_ONLY)
    get_filename_component(V2_SHARED_FILE_NCCL_LIBDIR "${NCCL_LIBRARY}" DIRECTORY)
    get_filename_component(V2_SHARED_FILE_NCCL_PREFIX "${V2_SHARED_FILE_NCCL_LIBDIR}" DIRECTORY)
    set(V2_SHARED_FILE_NCCL_INSTALLED "${V2_SHARED_FILE_NCCL_PREFIX}/share/llaminar-nccl/shared-file-lifecycle.json")
    set(V2_SHARED_FILE_NCCL_RECEIPT "${V2_SHARED_FILE_NCCL_LIBDIR}/llaminar-nccl-build.txt")
    set(V2_SHARED_FILE_NCCL_REPORT "${CMAKE_CURRENT_BINARY_DIR}/v2_nccl_shared_file_lifecycle.json")
    add_custom_command(OUTPUT "${V2_SHARED_FILE_NCCL_REPORT}"
        COMMAND ${Python3_EXECUTABLE} -B "${V2_SHARED_FILE_SCRIPT}" verify --backend nccl
            --report "${V2_SHARED_FILE_NCCL_INSTALLED}" --library "${NCCL_LIBRARY}"
            --receipt "${V2_SHARED_FILE_NCCL_RECEIPT}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${V2_SHARED_FILE_NCCL_INSTALLED}" "${V2_SHARED_FILE_NCCL_REPORT}"
        DEPENDS "${V2_SHARED_FILE_SCRIPT}" "${V2_SHARED_FILE_PROBE}"
            "${REPO_ROOT}/scripts/docker/patches/nccl-partial-file-cleanup.patch"
            "${REPO_ROOT}/scripts/docker/nccl-gin-selection/proof.py"
            "${REPO_ROOT}/scripts/docker/nccl-gin-selection/probe.cpp"
            "${REPO_ROOT}/scripts/docker/nccl-thread-lifetime/proof.py"
            "${REPO_ROOT}/scripts/docker/nccl-thread-lifetime/probe.cpp"
            "${REPO_ROOT}/scripts/docker/patches/nccl-capture-reentry.patch"
            "${REPO_ROOT}/scripts/docker/patches/nccl-device-live-rows.patch"
            "${REPO_ROOT}/scripts/docker/patches/nccl-gin-auto-selection.patch"
            "${REPO_ROOT}/scripts/docker/patches/nccl-created-thread-ownership.patch"
            "${V2_SHARED_FILE_NCCL_INSTALLED}" "${V2_SHARED_FILE_NCCL_RECEIPT}" "${NCCL_LIBRARY}"
        VERBATIM)
    add_custom_target(v2_nccl_shared_file_lifecycle_proof DEPENDS "${V2_SHARED_FILE_NCCL_REPORT}")
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_nccl_shared_file_lifecycle_proof)
    foreach(V2_SHARED_FILE_SUITE Unit Integration)
        set(V2_SHARED_FILE_TEST "V2_${V2_SHARED_FILE_SUITE}_NCCLSharedFileLifecycle")
        add_test(NAME ${V2_SHARED_FILE_TEST}
            COMMAND ${Python3_EXECUTABLE} -B "${V2_SHARED_FILE_SCRIPT}" verify --backend nccl
                --report "${V2_SHARED_FILE_NCCL_REPORT}" --library "${NCCL_LIBRARY}"
                --receipt "${V2_SHARED_FILE_NCCL_RECEIPT}")
        set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_SHARED_FILE_TEST} v2_nccl_shared_file_lifecycle_proof)
        set_tests_properties(${V2_SHARED_FILE_TEST} PROPERTIES
            LABELS "V2;${V2_SHARED_FILE_SUITE};NCCL;DeviceFree;Replay;Regression" TIMEOUT 30)
    endforeach()
    set_property(TEST V2_Integration_NCCLSharedFileLifecycle APPEND PROPERTY LABELS ProductionTestPreflight)
endif()

if(HAVE_ROCM AND HAVE_RCCL AND NOT V2_PERF_TESTS_ONLY)
    set(V2_SHARED_FILE_RCCL_REPORT "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_shared_file_lifecycle.json")
    set(V2_SHARED_FILE_RCCL_DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_shared_file_lifecycle.d")
    list(GET CMAKE_HIP_ARCHITECTURES 0 V2_SHARED_FILE_RCCL_ARCH)
    add_custom_command(OUTPUT "${V2_SHARED_FILE_RCCL_REPORT}"
        COMMAND ${Python3_EXECUTABLE} -B "${V2_SHARED_FILE_SCRIPT}" compile --backend rccl
            --report "${V2_SHARED_FILE_RCCL_REPORT}" --library "${RCCL_LIBRARY}"
            --rccl-build "${RCCL_PROTOCOL_BUILD_DIR}" --compiler "${CMAKE_HIP_COMPILER}"
            --architecture "${V2_SHARED_FILE_RCCL_ARCH}" --sdk "${ROCM_PATH}"
            --depfile "${V2_SHARED_FILE_RCCL_DEPFILE}"
        DEPENDS "${V2_SHARED_FILE_SCRIPT}" "${V2_SHARED_FILE_PROBE}"
            "${REPO_ROOT}/scripts/docker/patches/rccl-partial-file-cleanup.patch"
            "${REPO_ROOT}/scripts/docker/apply-rccl-capture-patch.sh"
            "${REPO_ROOT}/scripts/docker/apply-rccl-patch-chain.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/test_rccl_thread_lifetime.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/test_rccl_selector_taxonomy.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/test_rccl_small_allgather_policy.py"
            "${RCCL_PROTOCOL_BUILD_DIR}/compile_commands.json"
            "${RCCL_PROTOCOL_BUILD_DIR}/.llaminar-rccl-commit"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/misc/shmutils.cc"
            "${RCCL_LIBRARY}" "${CMAKE_HIP_COMPILER}"
        DEPFILE "${V2_SHARED_FILE_RCCL_DEPFILE}"
        VERBATIM)
    add_custom_target(v2_rccl_shared_file_lifecycle_proof DEPENDS "${V2_SHARED_FILE_RCCL_REPORT}")
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_rccl_shared_file_lifecycle_proof)
    foreach(V2_SHARED_FILE_SUITE Unit Integration)
        set(V2_SHARED_FILE_TEST "V2_${V2_SHARED_FILE_SUITE}_RCCLSharedFileLifecycle")
        add_test(NAME ${V2_SHARED_FILE_TEST}
            COMMAND ${Python3_EXECUTABLE} -B "${V2_SHARED_FILE_SCRIPT}" verify --backend rccl
                --report "${V2_SHARED_FILE_RCCL_REPORT}" --library "${RCCL_LIBRARY}")
        set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_SHARED_FILE_TEST} v2_rccl_shared_file_lifecycle_proof)
        set_tests_properties(${V2_SHARED_FILE_TEST} PROPERTIES
            LABELS "V2;${V2_SHARED_FILE_SUITE};RCCL;DeviceFree;Replay;Regression" TIMEOUT 30)
    endforeach()
    set_property(TEST V2_Integration_RCCLSharedFileLifecycle APPEND PROPERTY LABELS ProductionTestPreflight)
endif()
