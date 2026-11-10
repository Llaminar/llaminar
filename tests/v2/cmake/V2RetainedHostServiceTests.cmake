# @file V2RetainedHostServiceTests.cmake
# @brief Explicit device-free owner replay and separate unqualified native libc/TLS observation.
if(NOT COMMAND llaminar_native_source_fingerprint)
    include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/NativeSourceFingerprint.cmake)
endif()

v2_test_target_selected(v2_test_retained_host_service V2_RETAINED_HOST_SERVICE_OWNER_SELECTED)
if(NOT V2_PERF_TESTS_ONLY OR V2_RETAINED_HOST_SERVICE_OWNER_SELECTED)
    add_library(v2_host_service_tls_control MODULE
        unit/execution/local_execution/graph/support/HostServiceTLSControl.cpp)
    add_library(v2_host_service_non_tls_control MODULE
        unit/execution/local_execution/graph/support/HostServiceNonTLSControl.cpp)
    add_executable(v2_test_retained_host_service
        unit/execution/local_execution/graph/Test__RetainedHostServiceOwner.cpp)
    target_include_directories(v2_test_retained_host_service PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(v2_test_retained_host_service PRIVATE llaminar2_core GTest::gtest_main Threads::Threads ${CMAKE_DL_LIBS})
    target_compile_definitions(v2_test_retained_host_service PRIVATE
        LLAMINAR_HOST_SERVICE_TLS_CONTROL="$<TARGET_FILE:v2_host_service_tls_control>"
        LLAMINAR_HOST_SERVICE_NON_TLS_CONTROL="$<TARGET_FILE:v2_host_service_non_tls_control>")
    add_dependencies(v2_test_retained_host_service v2_host_service_tls_control v2_host_service_non_tls_control)
    v2_test_device_scope(v2_test_retained_host_service DeclaredBackends)
    v2_bind_gpu_process_test_entry(v2_test_retained_host_service v2_core_gtest_main)
    add_v2_test(V2_Unit_RetainedHostServiceOwner
        COMMAND $<TARGET_FILE:v2_test_retained_host_service>
        LABELS "V2;Unit;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_RetainedHostServiceOwnerReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service>
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceOwnerReplay)
    add_v2_test(V2_Integration_RetainedHostServiceOpenMPBudget
        COMMAND $<TARGET_FILE:v2_test_retained_host_service>
            --gtest_filter=RetainedHostServiceOwner.OriginalCallerOpenMPBudgetSurvivesForeignServiceEntry
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;OpenMP;Ownership;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceOpenMPBudget)
endif()
# Real loaded TLS code is unchanged by hardlink/permission metadata; replacement still rejects.
if(TARGET v2_test_retained_host_service)
    add_v2_test(V2_Integration_RetainedHostServiceFileMetadataReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service>
            --gtest_filter=RetainedHostServiceFamily.ColdEvidenceAcceptsTLSFileMetadataOnlyChanges:RetainedHostServiceFamily.ColdEvidenceRejectsTLSFileModification:RetainedHostServiceFamily.ColdEvidenceRejectsTLSFileReplacement
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceFileMetadataReplay)
endif()

# Explicit diagnostic source closure is independent of production creator identity and installed DATA.
set(_host_service_diagnostic_sources
    integration/execution/Measure__RetainedHostService.cpp
    integration/execution/NativeHostServiceTLSProgram.h
    integration/execution/NativeHostServicePopulationDiagnostic.h
    integration/execution/NativeHostServicePopulationRelease.h
    fixtures/execution/RetainedHostServiceCalibrationAccess.h)
set(_host_service_diagnostic_identity "original-host-service-diagnostic-v2")
foreach(_host_service_diagnostic_file IN LISTS _host_service_diagnostic_sources)
    llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/${_host_service_diagnostic_file}" _host_service_diagnostic_hash)
    string(APPEND _host_service_diagnostic_identity ";${_host_service_diagnostic_file}:${_host_service_diagnostic_hash}")
endforeach()
string(SHA256 _host_service_diagnostic_sha "${_host_service_diagnostic_identity}")
# This executable does not install opaque limits. It is not a timing or admission gate.
add_executable(v2_perf_native_host_service_calibration EXCLUDE_FROM_ALL
    integration/execution/Measure__RetainedHostService.cpp)
target_include_directories(v2_perf_native_host_service_calibration PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor")
target_compile_definitions(v2_perf_native_host_service_calibration PRIVATE LLAMINAR_HOST_SERVICE_CALIBRATION=1 HOST_SERVICE_DIAGNOSTIC_SOURCE_SHA256="${_host_service_diagnostic_sha}")
target_link_libraries(v2_perf_native_host_service_calibration PRIVATE llaminar2_core Threads::Threads ${CMAKE_DL_LIBS})
if(NOT LLAMINAR_SHARED_CORE)
    target_link_libraries(v2_perf_native_host_service_calibration PRIVATE -Wl,--start-group)
    if(HAVE_CUDA)
        target_link_libraries(v2_perf_native_host_service_calibration PRIVATE -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
    endif()
    if(HAVE_ROCM)
        target_link_libraries(v2_perf_native_host_service_calibration PRIVATE -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
    endif()
    target_link_libraries(v2_perf_native_host_service_calibration PRIVATE -Wl,--end-group)
endif()
# Receipt controls remain hardware-free and cannot promote an observed delta.
add_v2_test(V2_Unit_RetainedHostServiceCalibrationProtocol
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tests/test_host_service_calibration.py"
    LABELS "V2;Unit;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15)
add_v2_test(V2_Integration_RetainedHostServiceCalibrationProtocol
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/tests/test_host_service_calibration.py"
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 15)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceCalibrationProtocol)

# Confirmed duplicate-binding defect has its own exact model-free preflight replay.
if(TARGET v2_test_retained_host_service)
    target_sources(v2_test_retained_host_service PRIVATE
        unit/execution/local_execution/graph/Test__RetainedHostServiceInstances.cpp)
    add_v2_test(V2_Integration_RetainedHostServiceInstanceReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service>
            --gtest_filter=RetainedHostServiceInstances.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceInstanceReplay)
endif()

# The real resolver provider previously escaped the passive PT_TLS census.
if(TARGET v2_test_retained_host_service)
    target_sources(v2_test_retained_host_service PRIVATE
        unit/execution/local_execution/graph/Test__RetainedHostServiceTLSProvider.cpp)
    add_v2_test(V2_Integration_RetainedHostServiceTLSProviderReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service> --gtest_filter=RetainedHostServiceTLSProvider.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceTLSProviderReplay)
endif()

# Allocating observers cannot strand live diagnostic workers during unwind.
if(TARGET v2_test_retained_host_service)
    target_sources(v2_test_retained_host_service PRIVATE
        unit/execution/local_execution/graph/Test__HostServicePopulationRelease.cpp)
    add_v2_test(V2_Integration_HostServicePopulationReleaseReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service> --gtest_filter=HostServicePopulationRelease.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_HostServicePopulationReleaseReplay)
endif()

# The portable creator family excludes passive application payload while retaining live mutation rejection.
if(TARGET v2_test_retained_host_service)
    target_sources(v2_test_retained_host_service PRIVATE
        unit/execution/local_execution/graph/Test__RetainedHostServiceFamily.cpp)
    add_v2_test(V2_Integration_RetainedHostServiceFamilyReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service> --gtest_filter=RetainedHostServiceFamily.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceFamilyReplay)
endif()

# Creator scope is combined across roles and independently bound declaration sessions.
if(TARGET v2_test_retained_host_service)
    target_sources(v2_test_retained_host_service PRIVATE
        unit/execution/local_execution/graph/Test__RetainedHostServiceCombinedCap.cpp)
    add_v2_test(V2_Integration_RetainedHostServiceCombinedCapReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service> --gtest_filter=RetainedHostServiceCombinedCap.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceCombinedCapReplay)
endif()

# Explicit original-class consumer qualification: host-specific rows never gate ordinary device-free Unit.
add_executable(v2_perf_native_host_service_consumers EXCLUDE_FROM_ALL
    unit/execution/qwen4exp/Test__ImmutableLookupSubordinate.cpp
    unit/execution/local_execution/engine/Test__ForwardGraphTypes.cpp
    integration/core_gtest_main.cpp)
# Release retains only v2_perf_* targets, including its explicitly selected
# native diagnostics. This target checks correctness; its name grants no timing
# qualification. The canonical lexical process entry is a direct source so both
# Release and Integration own it without relying on an upstream GTest main.
target_compile_definitions(v2_perf_native_host_service_consumers PRIVATE LLAMINAR_HOST_SERVICE_ORIGINAL_CONSUMERS=1)
target_include_directories(v2_perf_native_host_service_consumers PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(v2_perf_native_host_service_consumers PRIVATE llaminar2_core GTest::gtest Threads::Threads ${CMAKE_DL_LIBS})
v2_test_device_scope(v2_perf_native_host_service_consumers DeclaredBackends)
add_v2_test(V2_Integration_RetainedHostServiceOriginalConsumers
    COMMAND $<TARGET_FILE:v2_perf_native_host_service_consumers>
        --gtest_filter=ImmutableLookupSubordinate.OriginalInstalledWorkerLargeSmallReset:Test__GraphSegmentCache.OriginalInstalledRetainedParentServicesManualTicket
    LABELS "V2;Integration;CPU;Memory;Ownership;TraceReplay;OriginalSource;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")

# The owned registration helper requires an exact tool path; a bare command
# is interpreted as a test target beneath the build directory.
find_program(V2_HOST_SERVICE_BASH_EXECUTABLE NAMES bash REQUIRED)
# The actual environment authority scanner also guards this discovered CPU owner defect.
add_v2_test(V2_Integration_HostServiceDebugEnvAuthority
    COMMAND "${V2_HOST_SERVICE_BASH_EXECUTABLE}" "${PROJECT_ROOT}/scripts/check_no_direct_getenv_outside_debugenv.sh" "${PROJECT_ROOT}"
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;StaticAnalysis;DebugEnv;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_HostServiceDebugEnvAuthority)

# Diagnose the actual cold metadata rejection without weakening loader/TLS policy.
if(TARGET v2_test_retained_host_service)
    add_v2_test(V2_Integration_RetainedHostServiceEvidenceDiagnostics
        COMMAND $<TARGET_FILE:v2_test_retained_host_service> --gtest_filter=RetainedHostServiceFamily.ColdEvidence* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceEvidenceDiagnostics)
endif()

# Current immutable TLS/provider/source facts, not incidental non-TLS loader history.
if(TARGET v2_test_retained_host_service)
    add_v2_test(V2_Integration_RetainedHostServiceCurrentResourceReplay
        COMMAND $<TARGET_FILE:v2_test_retained_host_service>
            --gtest_filter=RetainedHostServiceFamily.ColdEvidenceAcceptsNonTLSLoaderChurn:RetainedHostServiceFamily.ColdEvidenceAcceptsUnlinkedLoadedNonTLSImage:RetainedHostServiceFamily.ColdEvidenceRequiresCurrentTLSIdentityAcrossSameFileReload:RetainedHostServiceFamily.ColdEvidenceIdentifiesTLSGeometry:RetainedHostServiceFamily.ColdEvidenceRedactsRuntimePolicyValues --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceCurrentResourceReplay)
endif()

# Reborrow of one original admitted service is an ownership lifecycle contract.
if(TARGET v2_test_retained_host_service)
    add_v2_test(V2_Integration_RetainedHostServiceModelLifetimeContract
        COMMAND $<TARGET_FILE:v2_test_retained_host_service>
            --gtest_filter=RetainedHostServiceInstances.ModelOwnedServicesRetainOriginalWorkerAcrossRunnerBorrows
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Memory;Ownership;Contract;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RetainedHostServiceModelLifetimeContract)
endif()
