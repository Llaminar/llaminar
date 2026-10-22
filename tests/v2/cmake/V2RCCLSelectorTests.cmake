# Build-bound device-free proof of actual RCCL enum-keyed collective selectors.
#
# Vendor source and SDK work belongs to the builder. The installed runner keeps
# only this source/DSO-bound outcome record, including an executable negative
# control of the original misindexed selector table. Native captured collective regressions separately
# prove the selected runtime on real two/four-device topologies.
include_guard(DIRECTORY)

if(HAVE_ROCM AND HAVE_RCCL AND NOT V2_PERF_TESTS_ONLY)
    set(RCCL_PROTOCOL_BUILD_DIR "${RCCL_BUILD_DIR}" CACHE PATH
        "Completed canonical RCCL build providing the selector regression's native sources")
    set(V2_RCCL_SELECTOR_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/test_rccl_selector_taxonomy.py")
    set(V2_RCCL_SELECTOR_PROOF
        "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_selector_taxonomy.json")
    list(GET CMAKE_HIP_ARCHITECTURES 0 V2_RCCL_SELECTOR_ARCHITECTURE)
    add_custom_command(OUTPUT "${V2_RCCL_SELECTOR_PROOF}"
        COMMAND ${Python3_EXECUTABLE} "${V2_RCCL_SELECTOR_SCRIPT}" compile
            --report "${V2_RCCL_SELECTOR_PROOF}" --library "${RCCL_LIBRARY}"
            --rccl-build "${RCCL_PROTOCOL_BUILD_DIR}"
            --compiler "${CMAKE_HIP_COMPILER}"
            --architecture "${V2_RCCL_SELECTOR_ARCHITECTURE}" --sdk "${ROCM_PATH}"
        DEPENDS "${V2_RCCL_SELECTOR_SCRIPT}"
            integration/build/rccl_selector_probe.cpp
            "${REPO_ROOT}/scripts/docker/patches/rccl-collective-selector-taxonomy.patch"
            "${RCCL_PROTOCOL_BUILD_DIR}/compile_commands.json"
            "${RCCL_PROTOCOL_BUILD_DIR}/.llaminar-rccl-commit"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/init.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/collectives.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/device.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/include/nccl_common.h"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/graph/tuning.cc"
            "${RCCL_LIBRARY}" "${CMAKE_HIP_COMPILER}"
        VERBATIM)
    add_custom_target(v2_rccl_selector_taxonomy DEPENDS "${V2_RCCL_SELECTOR_PROOF}")
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_rccl_selector_taxonomy)
    foreach(V2_RCCL_SELECTOR_SUITE Unit Integration)
        set(V2_RCCL_SELECTOR_TEST "V2_${V2_RCCL_SELECTOR_SUITE}_RCCLCollectiveSelectors")
        add_test(NAME ${V2_RCCL_SELECTOR_TEST}
            COMMAND ${Python3_EXECUTABLE} "${V2_RCCL_SELECTOR_SCRIPT}" verify
                --report "${V2_RCCL_SELECTOR_PROOF}" --library "${RCCL_LIBRARY}")
        set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_RCCL_SELECTOR_TEST}
            v2_rccl_selector_taxonomy)
        set_tests_properties(${V2_RCCL_SELECTOR_TEST} PROPERTIES
            LABELS "V2;${V2_RCCL_SELECTOR_SUITE};RCCL;DeviceFree;Regression" TIMEOUT 30)
    endforeach()
    set_property(TEST V2_Integration_RCCLCollectiveSelectors APPEND PROPERTY LABELS
        ProductionTestPreflight)
endif()
