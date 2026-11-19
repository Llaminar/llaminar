# @file V2PooledExactIdentityNativeTests.cmake
# @brief Original native identity/immutable-field/ordered-commit replay gates and independent scalar controls.
# Actual integer identity dispatch neighbors, each a short complete native replay
# rather than another all-format sweep. Missing selected hardware is a failure.
foreach(_identity_backend CUDA ROCm)
    if((_identity_backend STREQUAL "CUDA" AND HAVE_CUDA) OR
       (_identity_backend STREQUAL "ROCm" AND HAVE_ROCM))
        if(_identity_backend STREQUAL "CUDA")
            set(_identity_lock cuda_gpu_0)
            set(_identity_startup "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
        else()
            set(_identity_lock rocm_gpu_0)
            set(_identity_startup "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
        endif()
        # Actual original phases reject malformed native inputs before any ordered mutation.
        # Each case reuses its captures 20 times across two generations, then the
        # unchanged full graph proves real partial/empty success and original close.
        # Pairwise rows retain every role/count, role/width and count/width
        # interaction. Identity dispatch neighbors are covered independently below;
        # row width does not change the integer identity rejection predicate.
        foreach(_identity_case Pin_65_32 Pin_257_64 Free_65_64 Free_257_32 Victim_65_32 Victim_257_64)
            string(REPLACE "_" ";" _identity_axes "${_identity_case}")
            list(GET _identity_axes 0 _identity_role)
            list(GET _identity_axes 1 _identity_live)
            list(GET _identity_axes 2 _identity_width)
            set(_identity_reject "V2_Integration_${_identity_backend}_PooledImmutable_CooperativeRejection_${_identity_role}_${_identity_live}_Row${_identity_width}")
            add_v2_test(${_identity_reject}
                COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                    --gtest_filter=PooledImmutableProviderSourceDevice.CooperativeRejection${_identity_backend}${_identity_role}${_identity_live}Row${_identity_width}
                LABELS "V2;Integration;ProductionTestPreflight;${_identity_backend};Ngram;Ownership;TraceReplay;Capture;ByteExact;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 RESOURCE_LOCK "${_identity_lock}"
                ENVIRONMENT "${_identity_startup}")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_identity_reject})
        endforeach()
        # Pairwise field/position/width coverage preserves early and late failure
        # publication for every field and both native row widths across the matrix.
        foreach(_record_case PinSlotBounds_Early_32 PinSlotBounds_Late_64 PinOrdinal_Early_64 PinOrdinal_Late_32 PinCommitFlags_Early_32 PinCommitFlags_Late_64)
            string(REPLACE "_" ";" _record_axes "${_record_case}")
            list(GET _record_axes 0 _record_fault)
            list(GET _record_axes 1 _record_position)
            list(GET _record_axes 2 _record_width)
            set(_record_test "V2_Integration_${_identity_backend}_PooledImmutable_Record_${_record_fault}_${_record_position}_Row${_record_width}")
            add_v2_test(${_record_test}
                COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                    --gtest_filter=PooledImmutableProviderSourceDevice.CooperativeFields${_identity_backend}${_record_fault}${_record_position}Row${_record_width}
                LABELS "V2;Integration;ProductionTestPreflight;${_identity_backend};Ngram;Ownership;TraceReplay;Capture;ByteExact;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 RESOURCE_LOCK "${_identity_lock}"
                ENVIRONMENT "${_identity_startup}")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_record_test})
        endforeach()
        # Both state paths retain every below/at/above dispatch boundary.
        foreach(_identity_trace Pin RetainedPinsAndAllocation)
            foreach(_identity_count 63 64 65 127 128 129 255 256 257 513)
                set(_identity_test "V2_Integration_${_identity_backend}_PooledImmutable_ExactIdentity_${_identity_trace}_${_identity_count}_Replay")
                add_v2_test(${_identity_test}
                    COMMAND $<TARGET_FILE:v2_test_qwen4exp_device_primitives>
                        --gtest_filter=PooledImmutableProviderSourceDevice.ExactIdentity${_identity_backend}${_identity_trace}${_identity_count}
                    LABELS "V2;Integration;ProductionTestPreflight;${_identity_backend};Ngram;Ownership;TraceReplay;Capture;ByteExact;Regression"
                    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 RESOURCE_LOCK "${_identity_lock}"
                    ENVIRONMENT "${_identity_startup}")
                list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_identity_test})
            endforeach()
        endforeach()
    endif()
endforeach()

# Actual scalar owner bodies retain every identity check around the GPU-only cooperative implementation.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledRetainedIdentityEmptyWavesReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableNative.RetainedIdentityEmptyAllocationWavesReplay:PooledImmutableNative.EmptyAllocationRejectsCorruptRetainedIdentityBeforeCommit
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledRetainedIdentityEmptyWavesReplay)
endif()

if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledRecordedSnapshotAlignmentReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableNative.RecordedSnapshotAlignmentAndActualOwnerReplay --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledRecordedSnapshotAlignmentReplay)
endif()

# Actual GPU-used immutable predicates versus the independent scalar state/commit oracle.
if(TARGET v2_test_qwen4exp_primitives)
    add_v2_test(V2_Integration_PooledCooperativeRecordPredicatesReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableNative.CooperativePinRecordFieldsMatchIndependentScalarReplay:PooledImmutableNative.CooperativeRetainedRecordFieldsMatchZeroIncomingScalarReplay
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCooperativeRecordPredicatesReplay)
endif()
