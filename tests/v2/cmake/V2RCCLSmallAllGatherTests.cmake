# Source/compiler/DSO-bound proof of original native small-AllGather policy.
# Compilation/execution is builder-owned and host-only; installed Unit/preflight
# authenticates complete native/opposite old-enqueue outcomes without an SDK/GPU.
include_guard(DIRECTORY)
if(HAVE_ROCM AND HAVE_RCCL AND NOT V2_PERF_TESTS_ONLY)
    set(RCCL_PROTOCOL_BUILD_DIR "${RCCL_BUILD_DIR}" CACHE PATH
        "Completed canonical RCCL native source/proof build")
    set(V2_RCCL_SMALL_GATHER_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/test_rccl_small_allgather_policy.py")
    set(V2_RCCL_SMALL_GATHER_PROOF
        "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_small_allgather_policy.json")
    list(GET CMAKE_HIP_ARCHITECTURES 0 V2_RCCL_SMALL_GATHER_ARCHITECTURE)
    add_custom_command(OUTPUT "${V2_RCCL_SMALL_GATHER_PROOF}"
        COMMAND ${Python3_EXECUTABLE} -B "${V2_RCCL_SMALL_GATHER_SCRIPT}" compile
            --report "${V2_RCCL_SMALL_GATHER_PROOF}" --library "${RCCL_LIBRARY}"
            --rccl-build "${RCCL_PROTOCOL_BUILD_DIR}" --compiler "${CMAKE_HIP_COMPILER}"
            --architecture "${V2_RCCL_SMALL_GATHER_ARCHITECTURE}" --sdk "${ROCM_PATH}"
        DEPENDS "${V2_RCCL_SMALL_GATHER_SCRIPT}"
            integration/build/rccl_small_allgather_probe.cpp
            integration/build/test_rccl_selector_taxonomy.py
            "${REPO_ROOT}/scripts/docker/patches/rccl-small-allgather-ll-policy.patch"
            "${RCCL_PROTOCOL_BUILD_DIR}/compile_commands.json"
            "${RCCL_PROTOCOL_BUILD_DIR}/.llaminar-rccl-commit"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/enqueue.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/init.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/SmallAllGatherPolicy.cpp"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/SmallAllGatherPolicy.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/graph/tuning.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/rccl_wrap.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/misc/archinfo.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/graph/connect.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/misc/param.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/register/register.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/transport/nvls.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/collectives.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/comm.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/llaminar_small_allgather_scope.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/graph/topo.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/transport.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/archinfo.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/enqueue.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/nccl_common.h"
            "${RCCL_LIBRARY}" "${CMAKE_HIP_COMPILER}"
        VERBATIM)
    add_custom_target(v2_rccl_small_allgather_policy DEPENDS "${V2_RCCL_SMALL_GATHER_PROOF}")
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_rccl_small_allgather_policy)
    foreach(V2_RCCL_SMALL_GATHER_SUITE Unit Integration)
        set(V2_RCCL_SMALL_GATHER_TEST "V2_${V2_RCCL_SMALL_GATHER_SUITE}_RCCLSmallAllGatherPolicy")
        add_test(NAME ${V2_RCCL_SMALL_GATHER_TEST}
            COMMAND ${Python3_EXECUTABLE} -B "${V2_RCCL_SMALL_GATHER_SCRIPT}" verify
                --report "${V2_RCCL_SMALL_GATHER_PROOF}" --library "${RCCL_LIBRARY}")
        set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_RCCL_SMALL_GATHER_TEST}
            v2_rccl_small_allgather_policy)
        set_tests_properties(${V2_RCCL_SMALL_GATHER_TEST} PROPERTIES
            LABELS "V2;${V2_RCCL_SMALL_GATHER_SUITE};RCCL;DeviceFree;Regression" TIMEOUT 30)
    endforeach()
    set_property(TEST V2_Integration_RCCLSmallAllGatherPolicy APPEND PROPERTY LABELS ProductionTestPreflight)
endif()
