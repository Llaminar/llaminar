# Complete directory-owned gate inventory and physical-core normalization.
# Defer exactly once until every registration, including late included modules,
# exists. CTest labels and add_v2_test target metadata remain the only authorities;
# late targets cannot silently escape CI builds or CPU scheduling budgets.
include_guard(DIRECTORY)

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
        set(V2_PREFLIGHT_GATE_TARGETS "")
        foreach(V2_CONFIGURED_TEST IN LISTS V2_CONFIGURED_TESTS)
            get_property(V2_CONFIGURED_TEST_LABELS
                TEST ${V2_CONFIGURED_TEST} PROPERTY LABELS)
            if("ProductionTestPreflight" IN_LIST V2_CONFIGURED_TEST_LABELS)
                get_property(V2_TEST_BUILD_TARGETS GLOBAL PROPERTY
                    "V2_TEST_BUILD_TARGETS_${V2_CONFIGURED_TEST}")
                list(APPEND V2_PREFLIGHT_GATE_TARGETS ${V2_TEST_BUILD_TARGETS})
            endif()
        endforeach()
        list(REMOVE_DUPLICATES V2_PREFLIGHT_GATE_TARGETS)
        if(NOT V2_PREFLIGHT_GATE_TARGETS)
            message(FATAL_ERROR "The production preflight gate has no executable targets")
        endif()
        add_custom_target(v2_production_test_preflight_gate
            DEPENDS ${V2_PREFLIGHT_GATE_TARGETS})
    endif()
endfunction()

cmake_language(DEFER CALL v2_finalize_test_gates)
