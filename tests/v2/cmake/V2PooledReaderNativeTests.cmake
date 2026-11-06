# @file V2PooledReaderNativeTests.cmake
# @brief Native mapped owner reads retain explicit transport, capacity and lifetime coverage.
# Empty and positive cache geometries use independent registered cases so every
# format retains the standard deadline and its original publication/reset oracle.
# Actual supported cross-vendor source/reader components; native peer guards remain.
if(HAVE_CUDA AND HAVE_ROCM)
    foreach(_pooled_read_direction CUDAConsumerROCmStorage ROCmConsumerCUDAStorage)
        foreach(_pooled_read_capacity C0 Positive)
            set(_pooled_read_test "V2_Integration_CrossVendor_PooledImmutable_${_pooled_read_direction}_${_pooled_read_capacity}_ReadAllFormatsReplay")
            add_v2_test(${_pooled_read_test}
                COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                    --gtest_filter=PooledImmutableProviderSourceDevice.CrossVendorCaptured${_pooled_read_direction}${_pooled_read_capacity}ReadAllFormats
                LABELS "V2;Integration;ProductionTestPreflight;CUDA;ROCm;CrossVendor;Ngram;Source;Ownership;TraceReplay;Capture;Component;AllFormats;ByteExact;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 RESOURCE_LOCK "cuda_gpu_0;rocm_gpu_0"
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=0")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_pooled_read_test})
        endforeach()
    endforeach()
endif()

# Focused native first-error lifecycle uses the same original real-provider cohort.
if(HAVE_CUDA AND HAVE_ROCM AND TARGET v2_test_qwen4exp_device_primitives)
    foreach(_cold_direction CUDAConsumerROCmStorage ROCmConsumerCUDAStorage)
        set(_cold_gate "V2_Integration_CrossVendor_PooledImmutable_${_cold_direction}_ColdPrefillFirstErrorReplay")
        add_v2_test(${_cold_gate}
            COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                --gtest_filter=PooledImmutableProviderSourceDevice.CrossVendor${_cold_direction}ColdPrefillFirstError
            LABELS "V2;Integration;ProductionTestPreflight;CUDA;ROCm;CrossVendor;Ngram;Source;Ownership;TraceReplay;Capture;Component;ByteExact;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 RESOURCE_LOCK "cuda_gpu_0;rocm_gpu_0"
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=0")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_cold_gate})
    endforeach()
endif()


# Each actual topology/capacity trajectory has its own standard timeout. Missing
# endpoints fail in the selected executable; they never become skipped passes.
if(HAVE_CUDA AND HAVE_ROCM)
    foreach(_pooled_topology CUDAConsumer2ROCmOwners CUDAConsumer4ROCmOwners ROCmConsumer2CUDAOwners)
        if(_pooled_topology STREQUAL "CUDAConsumer2ROCmOwners")
            set(_pooled_multi_locks "cuda_gpu_0;rocm_gpu_0;rocm_gpu_1")
        elseif(_pooled_topology STREQUAL "CUDAConsumer4ROCmOwners")
            set(_pooled_multi_locks "cuda_gpu_0;rocm_gpu_0;rocm_gpu_1;rocm_gpu_2;rocm_gpu_3")
        else()
            set(_pooled_multi_locks "rocm_gpu_0;cuda_gpu_0;cuda_gpu_1")
        endif()
        foreach(_pooled_capacity Positive MixedC0 AllC0)
            set(_pooled_multi_test "V2_Integration_CrossVendor_PooledImmutable_${_pooled_topology}_${_pooled_capacity}_Replay")
            add_v2_test(${_pooled_multi_test}
                COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                    --gtest_filter=PooledImmutableProviderSourceDevice.CrossVendor${_pooled_topology}${_pooled_capacity}Replay
                LABELS "V2;Integration;ProductionTestPreflight;CUDA;ROCm;CrossVendor;Ngram;Source;Ownership;TraceReplay;Capture;Component;MultiOwner;ByteExact;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
                RESOURCE_LOCK "${_pooled_multi_locks}"
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=0")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_pooled_multi_test})
        endforeach()
    endforeach()
endif()

# Explicit same-vendor NoPeer candidates: eligibility and native economy remain unqualified until run.
if(TARGET v2_test_qwen4exp_device_primitives)
    foreach(_nopeer_backend CUDA ROCm)
        if((_nopeer_backend STREQUAL "CUDA" AND HAVE_CUDA) OR (_nopeer_backend STREQUAL "ROCm" AND HAVE_ROCM))
            if(_nopeer_backend STREQUAL "CUDA")
                set(_nopeer_skip_cuda 0)
                set(_nopeer_skip_rocm 1)
                set(_nopeer_locks "cuda_gpu_0;cuda_gpu_1")
            else()
                set(_nopeer_skip_cuda 1)
                set(_nopeer_skip_rocm 0)
                set(_nopeer_locks "rocm_gpu_0;rocm_gpu_1")
            endif()
            foreach(_nopeer_capacity C0 Positive)
                add_v2_test(V2_Integration_NoPeer_PooledImmutable_${_nopeer_backend}2_${_nopeer_capacity}_ReadAllFormatsReplay
                    COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                        --gtest_filter=PooledImmutableProviderSourceDevice.NoPeerCaptured${_nopeer_backend}Consumer${_nopeer_backend}Storage${_nopeer_capacity}ReadAllFormats
                    LABELS "V2;Integration;${_nopeer_backend};NoPeerCandidate;Ngram;Ownership;Capture;Component;AllFormats;ByteExact;Regression"
                    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
                    RESOURCE_LOCK "${_nopeer_locks}"
                    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=${_nopeer_skip_cuda};LLAMINAR_SKIP_ROCM_STARTUP=${_nopeer_skip_rocm}")
            endforeach()
        endif()
    endforeach()
endif()

# Ordinary production PLE early demand: useful trunk retires before the SAME delayed source worker.
if(TARGET v2_test_qwen4exp_device_primitives)
    foreach(_causal_backend CUDA ROCm)
        if((_causal_backend STREQUAL "CUDA" AND HAVE_CUDA) OR (_causal_backend STREQUAL "ROCm" AND HAVE_ROCM))
            string(TOLOWER "${_causal_backend}" _causal_lock)
            if(_causal_backend STREQUAL "CUDA")
                set(_causal_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
            else()
                set(_causal_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
            endif()
            foreach(_causal_backing SSD RAM)
                set(_causal_name "V2_Integration_PooledEarlyCurrentCausal_${_causal_backend}_${_causal_backing}")
                add_v2_test(${_causal_name}
                    COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                        --gtest_filter=PooledImmutableProviderSourceDevice.ProductionEarlyCurrent${_causal_backend}${_causal_backing}Causal
                    LABELS "V2;Integration;ProductionTestPreflight;${_causal_backend};Ngram;ReadAhead;Capture;Component;ByteExact;Lifecycle;Regression"
                    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 90 RESOURCE_LOCK "${_causal_lock}_gpu_0"
                    ENVIRONMENT "${_causal_env}")
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_causal_name})
            endforeach()
        endif()
    endforeach()
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_ImmutableLookupClosedWorkerGateLifecycle
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=ImmutableLookupSubordinate.ClosedWorkerGatePreservesOriginalSourceTaskAndRetirement:ImmutableLookupSubordinate.ClosedOriginalGroupWorkerGatePreservesAdmissionAndWithdraws:ImmutableLookupSubordinate.ClosedOriginalWorkerDispatchFailureWithdrawsBorrow:ImmutableLookupSubordinate.ClosedOriginalGroupRequestFrontierRejectsStaleReadyEvidence
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Ngram;ReadAhead;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ImmutableLookupClosedWorkerGateLifecycle)
endif()
