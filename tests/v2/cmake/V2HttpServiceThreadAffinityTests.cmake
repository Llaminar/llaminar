# Register native OpenMP inheritance regressions for independent host services.
#
# These processes deliberately enter through cmake -E env without an MPI
# launcher: bound and unbound startup must reach the production listener with
# exactly the requested OpenMP policy. Remove KMP_AFFINITY because it can override
# OMP_PROC_BIND in LLVM runtimes. Keep registration with target selection
# so Release, which excludes v2_test_server_mode, has no dangling TARGET_FILE
# expression. The configure-only regression includes this same module.
include_guard(DIRECTORY)

if(NOT V2_PERF_TESTS_ONLY)
    add_test(NAME V2_Integration_HTTPServiceThreadAffinity
        COMMAND ${CMAKE_COMMAND} -E env --unset=KMP_AFFINITY OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=2
            $<TARGET_FILE:v2_test_server_mode> --gtest_filter=HttpServiceThreadAffinity.*)
    set_tests_properties(V2_Integration_HTTPServiceThreadAffinity PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;HTTP;Observability;Threading;Regression;DeviceFree"
        TIMEOUT 30)
    add_test(NAME V2_Integration_HTTPServiceThreadAffinityUnbound
        COMMAND ${CMAKE_COMMAND} -E env --unset=KMP_AFFINITY OMP_PROC_BIND=false OMP_NUM_THREADS=2
            $<TARGET_FILE:v2_test_server_mode> --gtest_filter=HttpServiceThreadAffinity.*)
    set_tests_properties(V2_Integration_HTTPServiceThreadAffinityUnbound PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;HTTP;Observability;Threading;Regression;DeviceFree"
        TIMEOUT 30)
    add_test(NAME V2_Integration_PrefixArchiveWorkerAffinity
        COMMAND ${CMAKE_COMMAND} -E env --unset=KMP_AFFINITY OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=2
            $<TARGET_FILE:v2_test_server_mode> --gtest_filter=PrefixArchiveWorkerAffinity.*)
    set_tests_properties(V2_Integration_PrefixArchiveWorkerAffinity PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;PrefixCache;Threading;Lifecycle;Regression;DeviceFree"
        TIMEOUT 30)
    add_test(NAME V2_Integration_PrefixArchiveWorkerAffinityUnbound
        COMMAND ${CMAKE_COMMAND} -E env --unset=KMP_AFFINITY OMP_PROC_BIND=false OMP_NUM_THREADS=2
            $<TARGET_FILE:v2_test_server_mode> --gtest_filter=PrefixArchiveWorkerAffinity.*)
    set_tests_properties(V2_Integration_PrefixArchiveWorkerAffinityUnbound PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;PrefixCache;Threading;Lifecycle;Regression;DeviceFree"
        TIMEOUT 30)
endif()
