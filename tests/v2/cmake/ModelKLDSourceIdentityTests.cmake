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
