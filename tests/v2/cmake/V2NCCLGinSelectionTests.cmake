# Authenticate the dependency builder's real GIN selector and original-code
# negative. No installed runner needs a CUDA SDK, vendor source or GPU access.
# The selected DSO, source patches and exact replay source must match the proof.
include_guard(DIRECTORY)

if(HAVE_CUDA AND HAVE_NCCL AND NOT V2_PERF_TESTS_ONLY)
    get_filename_component(V2_NCCL_LIBRARY_DIR "${NCCL_LIBRARY}" DIRECTORY)
    get_filename_component(V2_NCCL_PREFIX "${V2_NCCL_LIBRARY_DIR}" DIRECTORY)
    set(V2_NCCL_GIN_INSTALLED_PROOF "${V2_NCCL_PREFIX}/share/llaminar-nccl/gin-selection.json")
    set(V2_NCCL_GIN_INSTALLED_RECEIPT "${V2_NCCL_LIBRARY_DIR}/llaminar-nccl-build.txt")
    set(V2_NCCL_GIN_SCRIPT "${REPO_ROOT}/scripts/docker/nccl-gin-selection/proof.py")
    set(V2_NCCL_GIN_PROOF "${CMAKE_CURRENT_BINARY_DIR}/v2_nccl_gin_selection.json")
    set(V2_NCCL_GIN_RECEIPT "${CMAKE_CURRENT_BINARY_DIR}/v2_nccl_gin_selection.txt")
    add_custom_command(OUTPUT "${V2_NCCL_GIN_PROOF}" "${V2_NCCL_GIN_RECEIPT}"
        COMMAND ${Python3_EXECUTABLE} "${V2_NCCL_GIN_SCRIPT}" verify
            --report "${V2_NCCL_GIN_INSTALLED_PROOF}" --library "${NCCL_LIBRARY}"
            --receipt "${V2_NCCL_GIN_INSTALLED_RECEIPT}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${V2_NCCL_GIN_INSTALLED_PROOF}" "${V2_NCCL_GIN_PROOF}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${V2_NCCL_GIN_INSTALLED_RECEIPT}" "${V2_NCCL_GIN_RECEIPT}"
        DEPENDS "${V2_NCCL_GIN_SCRIPT}"
            "${REPO_ROOT}/scripts/docker/nccl-gin-selection/probe.cpp"
            "${REPO_ROOT}/scripts/docker/patches/nccl-capture-reentry.patch"
            "${REPO_ROOT}/scripts/docker/patches/nccl-device-live-rows.patch"
            "${REPO_ROOT}/scripts/docker/patches/nccl-gin-auto-selection.patch"
            "${V2_NCCL_GIN_INSTALLED_PROOF}" "${V2_NCCL_GIN_INSTALLED_RECEIPT}" "${NCCL_LIBRARY}"
        VERBATIM)
    add_custom_target(v2_nccl_gin_selection_proof
        DEPENDS "${V2_NCCL_GIN_PROOF}" "${V2_NCCL_GIN_RECEIPT}")
    set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_nccl_gin_selection_proof)
    foreach(V2_NCCL_GIN_SUITE Unit Integration)
        set(V2_NCCL_GIN_TEST "V2_${V2_NCCL_GIN_SUITE}_NCCLGinSelection")
        add_test(NAME ${V2_NCCL_GIN_TEST}
            COMMAND ${Python3_EXECUTABLE} "${V2_NCCL_GIN_SCRIPT}" verify
                --report "${V2_NCCL_GIN_PROOF}" --library "${NCCL_LIBRARY}"
                --receipt "${V2_NCCL_GIN_RECEIPT}")
        set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NCCL_GIN_TEST}
            v2_nccl_gin_selection_proof)
        set_tests_properties(${V2_NCCL_GIN_TEST} PROPERTIES
            LABELS "V2;${V2_NCCL_GIN_SUITE};NCCL;DeviceFree;Regression" TIMEOUT 30)
    endforeach()
    set_property(TEST V2_Integration_NCCLGinSelection APPEND PROPERTY LABELS ProductionTestPreflight)
endif()
