#[[
@file ModelKLDSourceIdentityTests.cmake
@brief Explicit frozen-source identity and coordinated original-owner observer replays.
]]
# Persistent reference file identity survives Linux device renumbering while
# current-run pins still reject mount/file changes. These are model-free replays.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_ModelKLDSourceDeviceRenumbering
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=ModelKLDQFNContract.OriginalStat* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    if(TEST V2_Integration_ModelKLDSourceDeviceRenumbering)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ModelKLDSourceDeviceRenumbering)
    endif()
endif()
if(Python3_Interpreter_FOUND)
    add_v2_test(V2_Integration_ModelKLDSourceDeviceRenumberingPython
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_model_kld_corpus_execution.py
            CorpusExecutionTests.test_reboot_device_renumbering_keeps_all_current_runtime_pins
            CorpusExecutionTests.test_durable_file_fields_and_device_types_remain_strict
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Numerics;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    add_v2_test(V2_Integration_ModelKLDSourceDeviceRenumberingReference
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_qfn_accuracy_reference.py
            QFNAccuracyReferenceTests.test_frozen_reference_survives_device_renumbering_before_admission
            QFNAccuracyReferenceTests.test_frozen_reference_rejects_device_change_during_actual_run
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Numerics;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    foreach(case IN ITEMS ModelKLDSourceDeviceRenumberingPython ModelKLDSourceDeviceRenumberingReference)
        if(TEST V2_Integration_${case})
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${case})
        endif()
    endforeach()
endif()

# Exact distributed observation owns only one frozen-history controller.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNDistributedObserverIntent
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition> --gtest_filter=ModelKLDQFNContract.DistributedCPUJoint* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNDistributedObserverIntent)
    add_v2_test(V2_Integration_QFNDistributedDynamicObserverIntent
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=FrozenParityHistoryReplay.CPUJointDynamicFrozenWorkersPreserveOriginalPolicy:ModelKLDQFNContract.DistributedCPUJointDynamicUsesOriginalHostAuthority:ModelKLDQFNContract.DistributedCPUJointRealCLITraversesCompleteRequestContract:ModelKLDQFNContract.DistributedCPUJointRejectsForeignPartialAndMigratingIntent
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNDistributedDynamicObserverIntent)
endif()
if(TARGET v2_test_native_source_final_retirement)
    foreach(case IN ITEMS AuthorityStopsBeforeOriginalRetirement AuthorityFailureAbortsBeforeOriginalRetirement FollowerOnlyRunsOriginalWorkerLoop)
        set(name V2_Integration_QFNDistributedObserver${case})
        add_v2_test(${name}
            COMMAND $<TARGET_FILE:v2_test_native_source_final_retirement> --gtest_filter=NativeSourceFinalRetirement.QFNDistributed${case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Replay"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${name})
    endforeach()
endif()
if(Python3_Interpreter_FOUND)
    add_v2_test(V2_Integration_QFNDistributedObserverRankReports
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_model_kld_corpus_v2.py QFNCorpusV2Test.test_distributed_rank_source_path_and_retirement_reports
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNDistributedObserverRankReports)
endif()

# Original launch/config extension reuses the same monitored canary process owner.
if(Python3_Interpreter_FOUND)
    add_v2_test(V2_Integration_QFNObserverExplicitRankfileLaunch
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_model_kld_corpus_execution.py
            CorpusExecutionTests.test_one_existing_lifecycle_retires_before_failed_native_artifact_cleanup
            CorpusExecutionTests.test_rankfile_launch_keeps_original_process_retirement_and_rank_evidence
            CorpusExecutionTests.test_rankfile_original_process_tree_tracks_and_retires_real_host_children
            CorpusExecutionTests.test_rankfile_command_is_explicit_pinned_and_old_single_rank_is_unchanged
            CorpusExecutionTests.test_rankfile_missing_scope_and_launcher_fail_before_native_execution
            CorpusExecutionTests.test_explicit_manifest_and_rankfile_cli_keep_v2_scope
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    add_v2_test(V2_Integration_QFNObserverExplicitMetadataManifest
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_model_kld_corpus_v2.py
            QFNCorpusV2Test.test_explicit_metadata_manifest_preserves_original_reference_root
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Replay"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNObserverExplicitRankfileLaunch V2_Integration_QFNObserverExplicitMetadataManifest)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNObserverOriginalRankHostContract
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=ModelKLDQFNContract.OriginalRankAndCanonicalHostOwnership:ModelKLDQFNContract.CanonicalHostTPPPAndPreparedDomains --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNObserverOriginalRankHostContract)
endif()
if(TARGET v2_test_native_source_final_retirement)
    add_v2_test(V2_Integration_QFNObserverFollowerFailureLifecycle
        COMMAND $<TARGET_FILE:v2_test_native_source_final_retirement>
            --gtest_filter=NativeSourceFinalRetirement.QFNDistributedFollowerRecordFailureRetiresWithoutRootAbort:NativeSourceFinalRetirement.QFNDistributedFollowerLoopFailureAbortsBeforeRetirement:NativeSourceFinalRetirement.QFNDistributedFollowerMissingOwnerAbortsBeforeLoop --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNObserverFollowerFailureLifecycle)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNObserverSourceGeometryContract
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=ModelKLDQFNContract.SourceGeometrySeparatesFollowerLogitCapability --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;OMP_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNObserverSourceGeometryContract)
endif()

# Optional row exceptions are count-bounded; all aggregate accuracy gates remain active.
if(Python3_Interpreter_FOUND)
    add_v2_test(V2_Integration_QFNBoundedRowOutlierPolicy
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_model_kld_corpus_v2.py
            QFNBoundedRowOutlierTest
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNBoundedRowOutlierPolicy)
endif()

# Failure attribution retains every original canonical checkpoint and metadata check.
if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_FrozenHistorySnapshotInventoryDiagnostics
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=FrozenParityHistoryReplay.CanonicalAll48InventoryRemainsMandatory:FrozenParityHistoryReplay.SixStepObservationKeepsLimitsAndRequiresOffsetFive
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Numerics;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FrozenHistorySnapshotInventoryDiagnostics)
endif()

# Late reset operands are original observations, never extra numerical panel rows.
if(Python3_Interpreter_FOUND)
    add_v2_test(V2_Integration_QFNDynamicSamePlacementResetObservation
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/../unit/scripts/test_model_kld_corpus_v2.py
            QFNCorpusV2Test.test_dynamic_reset_uses_actual_late_same_placement_pair
            QFNCorpusV2Test.test_static_reset_rejects_late_pair_substitution
            QFNCorpusV2Test.test_homogeneous_gpu_dynamic_keeps_original_exact_reset_oracle
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Reset;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNDynamicSamePlacementResetObservation)
endif()

if(TARGET v2_test_model_parity_definition)
    add_v2_test(V2_Integration_QFNResetComparisonPolicy
        COMMAND $<TARGET_FILE:v2_test_model_parity_definition>
            --gtest_filter=ModelKLDQFNContract.ResetComparisonPolicyScopesOnlyOriginalDynamicCPUJoint
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Numerics;Ownership;Reset;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_QFNResetComparisonPolicy)
endif()
