# @file V2AutomaticMoETP2PlanTests.cmake
# @brief Exercise public automatic layout and predictor selection on provisioned GGUFs.
function(add_v2_automatic_moe_tp2_plan test_name topology predictor)
    set(sources ${ARGN})
    list(GET sources 0 model)
    add_v2_test(${test_name}
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/integration/parity/check_automatic_moe_tp2_plan.py"
            --binary $<TARGET_FILE:llaminar2>
            --model "${model}" --topology "${topology}" --predictor "${predictor}"
        LABELS "V2;Integration;RealModel;CUDA;Planning;TP;MTP;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 120 CUDA_DEVICES 2
        REQUIRED_FILES ${sources}
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
    if(TEST ${test_name})
        set_tests_properties(${test_name} PROPERTIES RUN_SERIAL TRUE)
    endif()
endfunction()

# Resolve a split set as a unit so REQUIRED_FILES checks one original model root.
function(v2_provisioned_plan_sources output)
    set(sources ${ARGN})
    list(GET sources 0 first)
    if(NOT EXISTS "${first}")
        set(sources)
        foreach(source IN LISTS ARGN)
            get_filename_component(filename "${source}" NAME)
            list(APPEND sources "/opt/llaminar-models/${filename}")
        endforeach()
    endif()
    set(${output} "${sources}" PARENT_SCOPE)
endfunction()

if(HAVE_CUDA AND V2_TEST_SOCKETS GREATER_EQUAL 2)
    if(DEFINED ENV{LLAMINAR_QWEN36_MOE_MODEL} AND NOT "$ENV{LLAMINAR_QWEN36_MOE_MODEL}" STREQUAL "")
        set(_qwen36_plan_source "$ENV{LLAMINAR_QWEN36_MOE_MODEL}")
    else()
        v2_provisioned_plan_sources(_qwen36_plan_source "${V2_PARITY_MODEL_QWEN36_MOE}")
    endif()
    v2_provisioned_plan_sources(_qwen35_35b_plan_source "${V2_PARITY_MODEL_QWEN35_MOE_Q4}")
    v2_provisioned_plan_sources(_qwen35_122b_plan_sources ${V2_PARITY_MODEL_QWEN35_MOE_122B_Q8})
    add_v2_automatic_moe_tp2_plan(V2_Integration_Qwen36EmbeddedNextNAutomaticCUDATP2Plan
        cuda embedded "${_qwen36_plan_source}")
    add_v2_automatic_moe_tp2_plan(V2_Integration_Qwen35MoENoPredictorAutomaticCUDATP2Plan
        cuda absent "${_qwen35_35b_plan_source}")
    add_v2_automatic_moe_tp2_plan(V2_Integration_Qwen35MoE122BEmbeddedNextNAutomaticCUDACPUPlan
        mixed embedded ${_qwen35_122b_plan_sources})

    # The provisioned split model and benchmark campaign are optional real-model
    # evidence. The complete test owns one two-rank native MPI session.
    if(TARGET llaminar_mpi_session AND MPIEXEC_EXECUTABLE)
        list(GET _qwen35_122b_plan_sources 0 _qwen35_122b_campaign_model)
        add_v2_test(V2_Integration_Qwen35MoE122BFixedPlacementDynamicMTPTokenParity
            COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/integration/parity/check_qwen35_122b_fixed_placement_mtp.py"
                --binary $<TARGET_FILE:llaminar2>
                --model "${_qwen35_122b_campaign_model}"
                --mpi-session $<TARGET_FILE:llaminar_mpi_session>
                --mpiexec "${MPIEXEC_EXECUTABLE}"
                --evidence-dir "${CMAKE_CURRENT_BINARY_DIR}/real-model-evidence"
            LABELS "V2;Integration;RealModel;CUDA;CPU;ExpertOverlay;TP;MTP;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 900 CUDA_DEVICES 2
            REQUIRED_FILES ${_qwen35_122b_plan_sources}
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
        if(TEST V2_Integration_Qwen35MoE122BFixedPlacementDynamicMTPTokenParity)
            set_tests_properties(V2_Integration_Qwen35MoE122BFixedPlacementDynamicMTPTokenParity
                PROPERTIES RUN_SERIAL TRUE)
        endif()
    endif()
endif()
