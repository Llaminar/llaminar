# @file V2MTPShiftedTargetPositionTests.cmake
# @brief Register retained serial shifted-position replay in Release and Integration.
if(TARGET v2_test_qwen4exp_device_primitives)
    foreach(shifted_position_backend IN ITEMS CUDA ROCm)
        if(shifted_position_backend STREQUAL "CUDA" AND HAVE_CUDA)
            set(shifted_position_env "LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
        elseif(shifted_position_backend STREQUAL "ROCm" AND HAVE_ROCM)
            set(shifted_position_env "LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
        else()
            continue()
        endif()
        set(shifted_position_test "V2_Integration_${shifted_position_backend}_MTPShiftedTargetCachePosition")
        add_v2_test(${shifted_position_test}
            COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                --gtest_filter=MTPShiftedTargetCachePositionNative.${shifted_position_backend}
            LABELS "V2;Integration;ProductionTestPreflight;${shifted_position_backend};MTP;Capture;PrefixCache;Regression"
            ENVIRONMENT ${shifted_position_env} MPI_PROCS 1 NO_MODELS TIMEOUT 60)
        if(TEST ${shifted_position_test})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${shifted_position_test})
        endif()
    endforeach()
endif()
