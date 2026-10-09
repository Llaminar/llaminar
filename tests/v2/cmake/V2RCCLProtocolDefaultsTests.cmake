# Build-bound, device-free proof of RCCL's actual protocol geometry consumers.
#
# Vendor source and SDK work belongs to the builder. The installed runner keeps
# only this source/DSO-bound outcome record, including an executable negative
# control of the original race. Native captured collective regressions separately
# prove the selected runtime on real two/four-device topologies.
include_guard(DIRECTORY)

if(HAVE_ROCM AND HAVE_RCCL AND NOT V2_PERF_TESTS_ONLY)
    set(RCCL_PROTOCOL_BUILD_DIR "${RCCL_BUILD_DIR}" CACHE PATH
        "Completed canonical RCCL build providing the protocol regression's native sources")
    set(V2_RCCL_PROTOCOL_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/test_rccl_protocol_geometry.py")
    set(V2_RCCL_PROTOCOL_PROOF
        "${CMAKE_CURRENT_BINARY_DIR}/v2_rccl_protocol_geometry.json")
    list(GET CMAKE_HIP_ARCHITECTURES 0 V2_RCCL_PROTOCOL_ARCHITECTURE)
    add_custom_command(OUTPUT "${V2_RCCL_PROTOCOL_PROOF}"
        COMMAND ${Python3_EXECUTABLE} "${V2_RCCL_PROTOCOL_SCRIPT}" compile
            --report "${V2_RCCL_PROTOCOL_PROOF}" --library "${RCCL_LIBRARY}"
            --rccl-build "${RCCL_PROTOCOL_BUILD_DIR}"
            --compiler "${CMAKE_HIP_COMPILER}"
            --architecture "${V2_RCCL_PROTOCOL_ARCHITECTURE}" --sdk "${ROCM_PATH}"
        DEPENDS "${V2_RCCL_PROTOCOL_SCRIPT}"
            integration/build/rccl_protocol_geometry_probe.cpp
            "${REPO_ROOT}/scripts/docker/patches/rccl-communicator-protocol-defaults.patch"
            "${RCCL_PROTOCOL_BUILD_DIR}/compile_commands.json"
            "${RCCL_PROTOCOL_BUILD_DIR}/.llaminar-rccl-commit"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/rccl_wrap.cc"
            "${RCCL_PROTOCOL_BUILD_DIR}/hipify/src/misc/archinfo.cc"
            "${RCCL_LIBRARY}" "${CMAKE_HIP_COMPILER}"
        VERBATIM)
    add_custom_target(v2_rccl_protocol_geometry DEPENDS "${V2_RCCL_PROTOCOL_PROOF}")
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_rccl_protocol_geometry)
    foreach(V2_RCCL_PROTOCOL_SUITE Unit Integration)
        set(V2_RCCL_PROTOCOL_TEST "V2_${V2_RCCL_PROTOCOL_SUITE}_RCCLProtocolDefaults")
        add_test(NAME ${V2_RCCL_PROTOCOL_TEST}
            COMMAND ${Python3_EXECUTABLE} "${V2_RCCL_PROTOCOL_SCRIPT}" verify
                --report "${V2_RCCL_PROTOCOL_PROOF}" --library "${RCCL_LIBRARY}")
        set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_RCCL_PROTOCOL_TEST}
            v2_rccl_protocol_geometry)
        set_tests_properties(${V2_RCCL_PROTOCOL_TEST} PROPERTIES
            LABELS "V2;${V2_RCCL_PROTOCOL_SUITE};RCCL;DeviceFree;Regression" TIMEOUT 30)
    endforeach()
    set_property(TEST V2_Integration_RCCLProtocolDefaults APPEND PROPERTY LABELS
        ProductionTestPreflight)
endif()
