#[[
@file V2PreparedGPUExpertServicePlanningTests.cmake
@brief Device-free coverage of original prepared-expert setup memory admission.

The existing capacity target exercises the source-format workspace factory and
actual rank BOM, including exact CPU0 host scratch ownership. No GPU discovery,
model payload, or second planner is needed for this metadata contract. Native
capture and dynamic movement remain separate explicit backend gates.
]]
if(TARGET v2_unit_moe_overlay_capacity_resolver)
# Original source metadata declares one serial prepared-service probe envelope.
add_v2_test(V2_Integration_PreparedGPUExpertServicePlanning
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_capacity_resolver>
        --gtest_filter=MoEOverlayLocalCapacityPlanner.PreparedGPUServiceOriginalGeometryAndSerialEnvelope:MoEOverlayLocalCapacityPlanner.GroupsContinuationAndEndpointFixedBytesByBoundPhysicalResource
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;ExpertOverlay;MemoryBOM;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PreparedGPUExpertServicePlanning)
endif()
