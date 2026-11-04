# @file V2CPUJointSerialGeometryTests.cmake
# @brief Verify original source partitions, NUMA roots and CPU invocation/BOM contracts.
#
# Synthetic one-to-eight-participant geometry remains device-free. This gate exercises the
# same production source/cohort/storage/workspace producers used by admission;
# actual socket placement, MPI/context retirement and model qualification stay
# in the explicit native CPU expert suites.
if(TARGET v2_unit_moe_overlay_capacity_resolver)
    add_v2_test(V2_Integration_CPUJointSerialGeometryContract
        COMMAND $<TARGET_FILE:v2_unit_moe_overlay_capacity_resolver>
            --gtest_filter=CPUJointSerialGeometry.* --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MoE;Source;MemoryBOM;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
    if(TEST V2_Integration_CPUJointSerialGeometryContract)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUJointSerialGeometryContract)
    endif()
endif()
