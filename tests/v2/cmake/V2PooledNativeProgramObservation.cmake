# @file V2PooledNativeProgramObservation.cmake
# @brief Explicit unqualified actual-program observers, separate from canonical timing.
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_perf_pooled_ngram_lookup_diagnostic EXCLUDE_FROM_ALL
        performance/kernels/qwen4exp/Measure__PooledNgramLookup.cpp
        integration/backends/PooledNativeProgramObservation.cpp)
    target_include_directories(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/integration/kernels/qwen4exp")
    target_link_libraries(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE llaminar2_core GTest::gtest ${CMAKE_DL_LIBS})
    target_compile_definitions(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE
        LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC=1
        $<$<CONFIG:Release>:LLAMINAR_NGRAM_PERF_RELEASE=1>)
    # These executable-local symbols observe the real linked SDK via RTLD_NEXT.
    # Canonical qualified/unprofiled targets do not link any observer object.
    target_link_options(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE -Wl,--export-dynamic)
    if(HAVE_CUDA)
        target_sources(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE
            integration/backends/PooledNativeProgramCUDAInterpose.cpp
            integration/kernels/qwen4exp/PooledACKReadyCancellationCUDA.cu)
        target_link_libraries(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE CUDA::cudart)
    endif()
    if(HAVE_ROCM)
        target_sources(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE
            integration/backends/PooledNativeProgramROCmInterpose.cpp
            integration/kernels/qwen4exp/PooledACKReadyCancellationROCm.hip)
        set_target_properties(v2_perf_pooled_ngram_lookup_diagnostic PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
        target_link_libraries(v2_perf_pooled_ngram_lookup_diagnostic PRIVATE hip::host)
    endif()
endif()
if(NOT V2_PERF_TESTS_ONLY)
    add_executable(v2_test_pooled_native_graph_inventory
        unit/backends/Test__PooledNativeGraphInventory.cpp integration/core_gtest_main.cpp)
    target_include_directories(v2_test_pooled_native_graph_inventory PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(v2_test_pooled_native_graph_inventory PRIVATE llaminar2_core GTest::gtest)
    target_compile_features(v2_test_pooled_native_graph_inventory PRIVATE cxx_std_23)
    foreach(_inventory_gate IN ITEMS V2_Unit_PooledNativeGraphInventory V2_Integration_PooledNativeGraphInventoryReplay)
        if(_inventory_gate MATCHES "_Unit_")
            set(_inventory_labels Unit)
        else()
            set(_inventory_labels Integration ProductionTestPreflight)
        endif()
        add_v2_test(${_inventory_gate}
            COMMAND $<TARGET_FILE:v2_test_pooled_native_graph_inventory> --gtest_repeat=20
            LABELS "V2;${_inventory_labels};DeviceFree;Ngram;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    endforeach()
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeGraphInventoryReplay)
    add_v2_test(V2_Unit_PooledNativeProgramObservation
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_pooled_native_program_observation.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/diagnostics/inspect_pooled_native_program_observation.py
        LABELS "V2;Unit;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    add_v2_test(V2_Integration_PooledNativeProgramObservationReplay
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_pooled_native_program_observation.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/diagnostics/inspect_pooled_native_program_observation.py
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledNativeProgramObservationReplay)
endif()

if(TARGET v2_test_pooled_native_program_diagnostic)
    target_sources(v2_test_pooled_native_program_diagnostic PRIVATE unit/backends/Test__PooledNativeProgramTerminalOrder.cpp)
    add_v2_test(V2_Unit_PooledNativeProgramTerminalOrder
        COMMAND $<TARGET_FILE:v2_test_pooled_native_program_diagnostic> --gtest_filter=PooledNativeProgramTerminalOrder.*
        LABELS "V2;Unit;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    # The executable's registration checker covers every actual compiled case,
    # including these terminal-order regressions added after the owner suite.
    foreach(_terminal_order_case IN ITEMS
        OriginalJoinCompletesNewMarkerAcrossResetAndRepeat
        FormerPostJoinPublicationLeavesTheNewMarkerPending
        OriginalStorageFailureStillRetiresAlreadyJoinedGpuReaders)
        set(_terminal_order_gate "V2_Integration_PooledNativeProgram_${_terminal_order_case}Replay")
        add_v2_test(${_terminal_order_gate}
            COMMAND $<TARGET_FILE:v2_test_pooled_native_program_diagnostic>
                --gtest_filter=PooledNativeProgramTerminalOrder.${_terminal_order_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1" "GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_terminal_order_gate})
    endforeach()
endif()

# Exact source/remote instantiate rejection selector; actual observer body, no SDK calls.
if(NOT V2_PERF_TESTS_ONLY)
    add_executable(v2_test_pooled_diagnostic_instantiation_selector
        unit/backends/Test__PooledDiagnosticInstantiationSelector.cpp integration/backends/PooledNativeProgramObservation.cpp)
    if(TARGET v2_test_pooled_diagnostic_instantiation_selector)
        target_include_directories(v2_test_pooled_diagnostic_instantiation_selector PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(v2_test_pooled_diagnostic_instantiation_selector PRIVATE GTest::gtest_main)
        target_compile_features(v2_test_pooled_diagnostic_instantiation_selector PRIVATE cxx_std_23)
        add_v2_test(V2_Unit_PooledDiagnosticInstantiationSelectorReplay
            COMMAND $<TARGET_FILE:v2_test_pooled_diagnostic_instantiation_selector>
            LABELS "V2;Unit;DeviceFree;Ngram;Ownership;TraceReplay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "GTEST_DEATH_TEST_STYLE=threadsafe")
        foreach(_selector_case OriginalUnqualifiedOrdinalAndAbsentControlRemainUnchanged SourceBackendFirstSucceedsBeforeExactRemoteBackendFirstRejects)
            set(_selector_gate "V2_Integration_PooledDiagnosticInstantiationSelector_${_selector_case}Replay")
            add_v2_test(${_selector_gate} COMMAND $<TARGET_FILE:v2_test_pooled_diagnostic_instantiation_selector>
                --gtest_filter=PooledDiagnosticInstantiationSelector.${_selector_case} --gtest_repeat=20
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "GTEST_DEATH_TEST_STYLE=threadsafe")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_selector_gate})
        endforeach()
        # Preserve every malformed selector/death control 20 times with comfortable
        # standard 30s CI headroom rather than reexecuting12 children ×20 in one gate.
        foreach(_selector_partition RANGE 1 4)
            set(_selector_gate "V2_Integration_PooledDiagnosticInstantiationSelector_MalformedReplay${_selector_partition}")
            add_v2_test(${_selector_gate} COMMAND $<TARGET_FILE:v2_test_pooled_diagnostic_instantiation_selector>
                --gtest_filter=PooledDiagnosticInstantiationSelector.MalformedOrAmbiguousQualifiedControlFailsBeforeAnyNativeCall --gtest_repeat=5
                LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Ownership;TraceReplay;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30 ENVIRONMENT "GTEST_DEATH_TEST_STYLE=threadsafe")
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS ${_selector_gate})
        endforeach()
    endif()
endif()
