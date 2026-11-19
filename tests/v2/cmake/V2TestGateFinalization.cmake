# @file V2TestGateFinalization.cmake
# @brief Complete directory-owned gate inventories and physical-core normalization.
# Defer exactly once until every registration, including late included modules,
# exists. CTest labels and add_v2_test target metadata remain the only authorities;
# late targets cannot silently escape CI builds or CPU scheduling budgets.
include_guard(DIRECTORY)

# @brief Derive the ARM64 PR build and CTest scope from the same registrations.
# @param tests Final configure-time test inventory for this directory.
function(v2_register_arm64_pr_gate tests)
    set(targets llaminar2)
    foreach(test IN LISTS tests)
        get_property(labels TEST ${test} PROPERTY LABELS)
        if(NOT labels MATCHES "(^|;)(ARM|NEON|NEON_DOTPROD|NEON_I8MM)(;|$)" OR
           NOT ("Unit" IN_LIST labels OR "ProductionTestPreflight" IN_LIST labels))
            continue()
        endif()
        get_property(fixtures TEST ${test} PROPERTY FIXTURES_REQUIRED)
        get_property(files TEST ${test} PROPERTY REQUIRED_FILES)
        if(fixtures OR files OR "CUDA" IN_LIST labels OR "ROCm" IN_LIST labels OR
           "FullDeviceInventory" IN_LIST labels OR "HardwareOptional" IN_LIST labels)
            message(FATAL_ERROR "ARM64 PR registration must be model-free and host-only: ${test}")
        endif()
        set_property(TEST ${test} APPEND PROPERTY LABELS ARM64PR)
        # A native test's GTest skip must fail even when its process exits zero.
        set_property(TEST ${test} APPEND PROPERTY FAIL_REGULAR_EXPRESSION
            "\\[ *SKIPPED *\\]")
        get_property(build_targets GLOBAL PROPERTY "V2_TEST_BUILD_TARGETS_${test}")
        list(APPEND targets ${build_targets})
    endforeach()
    list(REMOVE_DUPLICATES targets)
    add_custom_target(v2_arm64_pr_gate DEPENDS ${targets})
endfunction()

# Finalize this directory only after all of its test declarations have executed.
function(v2_finalize_test_gates)
    # CTest's unrestricted parallel level is expressed in logical CPUs, whereas
    # unit-test worker budgets above are deliberately expressed in physical cores.
    # Normalize every configured Unit test in one final pass, including direct
    # add_test() registrations and explicit heavyweight overrides. This keeps the
    # wrapper, raw Python tests, and nested-worker trainers under one accounting
    # authority instead of relying on per-test timeout luck on SMT hosts.
    get_property(V2_CONFIGURED_TESTS DIRECTORY PROPERTY TESTS)
    foreach(V2_CONFIGURED_TEST IN LISTS V2_CONFIGURED_TESTS)
        get_property(V2_CONFIGURED_TEST_LABELS
            TEST ${V2_CONFIGURED_TEST} PROPERTY LABELS)
        if("${V2_CONFIGURED_TEST_LABELS}" MATCHES "Unit")
            get_property(V2_TEST_HAS_PROCESSORS
                TEST ${V2_CONFIGURED_TEST} PROPERTY PROCESSORS SET)
            if(V2_TEST_HAS_PROCESSORS)
                get_property(V2_TEST_PHYSICAL_PROCESSORS
                    TEST ${V2_CONFIGURED_TEST} PROPERTY PROCESSORS)
            else()
                set(V2_TEST_PHYSICAL_PROCESSORS 1)
            endif()
            math(EXPR V2_TEST_LOGICAL_PROCESSORS
                 "${V2_TEST_PHYSICAL_PROCESSORS} * ${V2_TEST_THREADS_PER_CORE}")
            set_tests_properties(${V2_CONFIGURED_TEST} PROPERTIES
                PROCESSORS ${V2_TEST_LOGICAL_PROCESSORS})
        endif()
    endforeach()

    # Build authority for the complete registered unit suite.  Script-only tests
    # correctly add no dependency and execute in the subsequent CTest phase.
    # Dependencies are collected by add_v2_test(), where CMake still has the exact
    # target behind each registration; this avoids a hand-maintained mirror of the
    # 500+ executable inventory.
    if(NOT V2_PERF_TESTS_ONLY)
        get_property(V2_UNIT_GATE_TARGETS GLOBAL PROPERTY V2_UNIT_GATE_TARGETS)
        list(REMOVE_DUPLICATES V2_UNIT_GATE_TARGETS)
        if(NOT V2_UNIT_GATE_TARGETS)
            message(FATAL_ERROR "The V2 unit gate has no executable targets")
        endif()
        add_custom_target(v2_unit_gate DEPENDS ${V2_UNIT_GATE_TARGETS})

        # Use the final CTest label inventory, not a second hook-owned executable
        # list. Shared fixtures are built once; only the labeled tests are run.
        # POST_BUILD-discovered cases do not exist in configure-time TESTS.
        # Their same registration declares original build owners explicitly.
        get_property(V2_PREFLIGHT_GATE_TARGETS GLOBAL PROPERTY
            V2_DISCOVERED_PREFLIGHT_GATE_TARGETS)
        get_property(V2_PREFLIGHT_HARDWARE_EXPORT_REQUIRED GLOBAL PROPERTY
            V2_DISCOVERED_PREFLIGHT_HARDWARE_EXPORT_REQUIRED)
        foreach(V2_CONFIGURED_TEST IN LISTS V2_CONFIGURED_TESTS)
            get_property(V2_CONFIGURED_TEST_LABELS
                TEST ${V2_CONFIGURED_TEST} PROPERTY LABELS)
            if("ProductionTestPreflight" IN_LIST V2_CONFIGURED_TEST_LABELS)
                get_property(V2_TEST_BUILD_TARGETS GLOBAL PROPERTY
                    "V2_TEST_BUILD_TARGETS_${V2_CONFIGURED_TEST}")
                list(APPEND V2_PREFLIGHT_GATE_TARGETS ${V2_TEST_BUILD_TARGETS})
                if("HardwareOptional" IN_LIST V2_CONFIGURED_TEST_LABELS)
                    set(V2_PREFLIGHT_HARDWARE_EXPORT_REQUIRED TRUE)
                endif()
            endif()
        endforeach()
        if(V2_PREFLIGHT_HARDWARE_EXPORT_REQUIRED)
            # Reuse the original model-free native exporter and its complete
            # MPI/backend lifetime; no independent GPU discovery tool is added.
            if(NOT TARGET v2_integration_parity_qwen4exp_production_matrix)
                message(FATAL_ERROR "Optional native preflight geometry requires its original hardware exporter")
            endif()
            list(APPEND V2_PREFLIGHT_GATE_TARGETS v2_integration_parity_qwen4exp_production_matrix)
        endif()
        list(REMOVE_DUPLICATES V2_PREFLIGHT_GATE_TARGETS)
        if(NOT V2_PREFLIGHT_GATE_TARGETS)
            message(FATAL_ERROR "The production preflight gate has no executable targets")
        endif()
        add_custom_target(v2_production_test_preflight_gate
            DEPENDS ${V2_PREFLIGHT_GATE_TARGETS})
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
           LLAMINAR_CPU_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64|AARCH64)$" AND
           NOT (HAVE_CUDA OR HAVE_ROCM OR HAVE_VULKAN))
            v2_register_arm64_pr_gate("${V2_CONFIGURED_TESTS}")
        endif()
    endif()
endfunction()

cmake_language(DEFER CALL v2_finalize_test_gates)
