# @file V2ObservedNativePipelineTests.cmake
# @brief Original native pipeline geometries with complete control and pairwise asymmetric policies.
# Original observed MI50 pipeline preflight geometry and native registrations.
# The case name already encodes the exact ordered stage widths consumed by the
# native fixture. Derive hardware labels from it, never a second size table.
function(v2_mi50_pipeline_optional_hardware_labels topology result)
    if(NOT topology MATCHES "^MI50_EP[1-9][0-9]*(_MI50_EP[1-9][0-9]*)+$")
        message(FATAL_ERROR "Malformed original MI50 pipeline geometry: ${topology}")
    endif()
    string(REPLACE "MI50_EP" "" _widths "${topology}")
    string(REPLACE "_" ";" _widths "${_widths}")
    set(_degree 0)
    foreach(_width IN LISTS _widths)
        math(EXPR _degree "${_degree} + ${_width}")
    endforeach()
    string(JOIN "x" _stage_sizes ${_widths})
    set(${result} "HardwareOptional;Devices${_degree};NativeISA-gfx906;StageSizes${_stage_sizes}" PARENT_SCOPE)
endfunction()

if(HAVE_ROCM AND TARGET v2_integration_moe_pipeline_weight_preparation)
    # Preserve the complete balanced EP4+EP4 control. Uneven and three-stage
    # geometries retain every order/movement, order/MTP and movement/MTP pair;
    # preparation and full reset/prefix lifecycle remain separate native bodies.
    set(_ep_representatives
        Ordinal_Static_DisabledMTP Random_Dynamic_DisabledMTP
        Ordinal_Dynamic_Fixed1MTP Random_Static_Fixed1MTP
        Ordinal_Static_Fixed3MTP Random_Dynamic_Fixed3MTP
        Ordinal_Dynamic_DynamicMTP Random_Static_DynamicMTP)
    foreach(_ep_topology MI50_EP4_MI50_EP4 MI50_EP1_MI50_EP2 MI50_EP2_MI50_EP1 MI50_EP1_MI50_EP2_MI50_EP4)
        v2_mi50_pipeline_optional_hardware_labels("${_ep_topology}" _ep_hardware_labels)
        foreach(_ep_order Ordinal Random)
            foreach(_ep_movement Static Dynamic)
                # Each exact body owns one original source and must run in its
                # own canonical MPI2 process lifetime. A GPU process root does
                # not adopt a later case's independent PMA after runner reset.
                foreach(_ep_lane DisabledMTP Fixed1MTP Fixed3MTP DynamicMTP)
                    set(_ep_policy "${_ep_order}_${_ep_movement}_${_ep_lane}")
                    if(NOT _ep_topology STREQUAL "MI50_EP4_MI50_EP4" AND
                       NOT _ep_policy IN_LIST _ep_representatives)
                        continue()
                    endif()
                    set(_ep_case "${_ep_topology}_${_ep_order}_${_ep_movement}_${_ep_lane}")
                    set(_ep_preparation "V2_Integration_MoEPipelinePreparation_${_ep_case}")
                    set(_ep_lifecycle "V2_Integration_MoEPipelineLifecycle_${_ep_case}")
                    add_v2_test(${_ep_preparation}
                        COMMAND $<TARGET_FILE:v2_integration_moe_pipeline_weight_preparation>
                            --gtest_filter=Topologies/MoEPipelineWeightPreparation.NativePreparationContract/${_ep_case}
                        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MI50;${_ep_hardware_labels};ExpertParallel;Pipeline;MemoryBudget;GraphCapture;Ownership;Replay;Regression"
                        MPI_PROCS 2 NO_MODELS TIMEOUT 120)
                    add_v2_test(${_ep_lifecycle}
                        COMMAND $<TARGET_FILE:v2_integration_moe_pipeline_weight_preparation>
                            --gtest_filter=Topologies/MoEPipelineRequestLifecycle.FirstDecodeResetAndPrefixReuse/${_ep_case}
                        LABELS "V2;Integration;ProductionTestPreflight;ROCm;MI50;${_ep_hardware_labels};ExpertParallel;Pipeline;MTP;PrefixCache;RequestReset;GraphCapture;Ownership;Replay;Regression"
                        MPI_PROCS 2 NO_MODELS TIMEOUT 120)
                    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_ep_preparation} ${_ep_lifecycle})
                endforeach()
            endforeach()
        endforeach()

    endforeach()
endif()
