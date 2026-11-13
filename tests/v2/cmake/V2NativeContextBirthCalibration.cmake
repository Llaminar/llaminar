# Explicit raw original-worker diagnostic, never a production certificate.
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/NativeSourceFingerprint.cmake")
set(_native_birth_source_inputs "")
foreach(_native_birth_file
    backends/NativePhysicalLifetime.h
    backends/NativeContextOpaqueContract.h backends/NativeContextOpaqueContract.cpp
    backends/NativeContextQualifiedLimits.h
    backends/NativeContextPreparation.h backends/NativeContextPreparation.cpp
    backends/PreparedGPUWorkerThread.h backends/PreparedGPUWorkerThread.cpp
    collective/NativeStoragePreparationPmaBinding.h collective/NativeStoragePmaBinding.h
    planning/PhysicalMemoryBOM.h planning/PhysicalMemoryAuthority.h planning/PhysicalMemoryAuthority.cpp
    backends/NativeContextRetirement.h backends/NativeContextRetirement.cpp
    backends/GPUDeviceContextPool.h backends/GPUDeviceContextPool.cpp
    backends/GPUDeviceProcessLifetime.h backends/GPUDeviceProcessLifetime.cpp
    execution/local_execution/graph/RetainedHostServiceProgram.h execution/local_execution/graph/RetainedHostServiceProgram.cpp
    backends/GPUWorkerInitialization.h backends/IWorkerGPUContext.h
    backends/cuda/NvidiaDeviceContext.h backends/cuda/NvidiaDeviceContext.cu
    backends/cuda/NvidiaContextFactory.cu
    backends/BackendManager.h backends/BackendManager.cpp backends/IBackend.h
    backends/cuda/CUDABackend.h backends/cuda/CUDABackend.cu
    backends/rocm/ROCmBackend.h backends/rocm/ROCmBackend.cpp
    backends/cuda/CUDAHardwareMetadata.h backends/cuda/CUDAHardwareMetadata.cu
    backends/rocm/AMDDeviceContext.h backends/rocm/AMDDeviceContext.cpp
    backends/rocm/AMDContextFactory.cpp
    backends/rocm/ROCmHardwareMetadata.h backends/rocm/ROCmHardwareMetadata.cpp
    transfer/TransferEngine.h transfer/TransferEngine.cpp)
    set(_native_birth_path "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/${_native_birth_file}")
    llaminar_native_source_fingerprint("${_native_birth_path}" _native_birth_hash)
    string(APPEND _native_birth_source_inputs "${_native_birth_file}:${_native_birth_hash};")
endforeach()
string(SHA256 _native_birth_source_sha "${_native_birth_source_inputs}")
set(_native_birth_diagnostic_inputs "")
foreach(_native_birth_file
    NativeContextBirthCalibration.cpp NativeContextBirthDiagnosticHooks.h NativeContextBirthDiagnosticHooks.cpp
    NativeContextBirthCUDAInterpose.cpp NativeContextBirthROCmInterpose.cpp
    ReadAheadCalibrationVersions.h ReadAheadCalibrationCUDAVersions.cpp ReadAheadCalibrationROCmVersions.cpp)
    set(_native_birth_path "${CMAKE_CURRENT_SOURCE_DIR}/integration/backends/${_native_birth_file}")
    llaminar_native_source_fingerprint("${_native_birth_path}" _native_birth_hash)
    string(APPEND _native_birth_diagnostic_inputs "${_native_birth_file}:${_native_birth_hash};")
endforeach()
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/backends/NativeOrderingEventProgram.h" _native_lookup_program_hash)
string(APPEND _native_birth_diagnostic_inputs "NativeOrderingEventProgram.h:${_native_lookup_program_hash};")
string(SHA256 _native_birth_diagnostic_sha "${_native_birth_diagnostic_inputs}")
add_executable(v2_perf_native_context_birth_calibration EXCLUDE_FROM_ALL
    integration/backends/NativeContextBirthCalibration.cpp
    integration/backends/NativeContextBirthDiagnosticHooks.cpp)
if(TARGET v2_perf_native_context_birth_calibration)
    target_include_directories(v2_perf_native_context_birth_calibration PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor")
    target_compile_definitions(v2_perf_native_context_birth_calibration PRIVATE
        NATIVE_CONTEXT_BIRTH_SOURCE_SHA256="${_native_birth_source_sha}"
        NATIVE_CONTEXT_BIRTH_DIAGNOSTIC_SHA256="${_native_birth_diagnostic_sha}")
    # Only this diagnostic exports vendor creator interposers. Original calls
    # resolve RTLD_NEXT; production executables retain their original symbols.
    target_link_options(v2_perf_native_context_birth_calibration PRIVATE -Wl,--export-dynamic)
    target_link_libraries(v2_perf_native_context_birth_calibration PRIVATE llaminar2_core ${CMAKE_DL_LIBS})
    if(HAVE_CUDA)
        target_sources(v2_perf_native_context_birth_calibration PRIVATE
            integration/backends/NativeContextBirthCUDAInterpose.cpp
            integration/backends/ReadAheadCalibrationCUDAVersions.cpp)
    endif()
    if(HAVE_ROCM)
        target_sources(v2_perf_native_context_birth_calibration PRIVATE
            integration/backends/NativeContextBirthROCmInterpose.cpp
            integration/backends/ReadAheadCalibrationROCmVersions.cpp)
    endif()
    if(NOT LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_perf_native_context_birth_calibration PRIVATE -Wl,--start-group)
        if(HAVE_CUDA)
            target_link_libraries(v2_perf_native_context_birth_calibration PRIVATE -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
        endif()
        if(HAVE_ROCM)
            target_link_libraries(v2_perf_native_context_birth_calibration PRIVATE -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
        endif()
        target_link_libraries(v2_perf_native_context_birth_calibration PRIVATE -Wl,--end-group)
    endif()
endif()

# Device-free receipt/protocol negatives are unconditional; no SDK/GPU execution.
add_v2_test(V2_Unit_NativeContextBirthCalibrationDriver
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_context_birth_calibration.py
    LABELS "V2;Unit;DeviceFree;Memory;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
add_v2_test(V2_Integration_NativeContextBirthCalibrationDriverReplay
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_context_birth_calibration.py
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
    ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeContextBirthCalibrationDriverReplay)

# This target never links SDK interposers or exports replacement creator symbols.
set(_native_source_probe_files NativeContextSourceProbe.cpp NativeContextSourceProbeSupport.h NativeContextSourceProbePma.h
    NativeContextSourceProbeCUDA.cpp NativeContextSourceProbeROCm.cpp)
set(_native_source_probe_inputs "")
foreach(_native_source_probe_file IN LISTS _native_source_probe_files)
    llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/integration/backends/${_native_source_probe_file}" _native_source_probe_hash)
    string(APPEND _native_source_probe_inputs "${_native_source_probe_file}:${_native_source_probe_hash};")
endforeach()
string(SHA256 _native_source_probe_sha "${_native_source_probe_inputs}")
add_executable(v2_perf_native_context_source_probe EXCLUDE_FROM_ALL integration/backends/NativeContextSourceProbe.cpp)
if(TARGET v2_perf_native_context_source_probe)
    target_include_directories(v2_perf_native_context_source_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor")
    target_compile_definitions(v2_perf_native_context_source_probe PRIVATE
        NATIVE_CONTEXT_BIRTH_SOURCE_SHA256="${_native_birth_source_sha}"
        NATIVE_CONTEXT_SOURCE_PROBE_SHA256="${_native_source_probe_sha}")
    target_link_libraries(v2_perf_native_context_source_probe PRIVATE llaminar2_core ${CMAKE_DL_LIBS})
    if(NOT LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_perf_native_context_source_probe PRIVATE -Wl,--start-group)
        if(HAVE_CUDA)
            target_link_libraries(v2_perf_native_context_source_probe PRIVATE -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
        endif()
        if(HAVE_ROCM)
            target_link_libraries(v2_perf_native_context_source_probe PRIVATE -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
        endif()
        target_link_libraries(v2_perf_native_context_source_probe PRIVATE -Wl,--end-group)
    endif()
    if(HAVE_CUDA)
        target_sources(v2_perf_native_context_source_probe PRIVATE integration/backends/NativeContextSourceProbeCUDA.cpp)
        target_compile_definitions(v2_perf_native_context_source_probe PRIVATE
            NATIVE_SOURCE_EXPECTED_CUDA_BLAS="$<TARGET_FILE:CUDA::cublas>"
            NATIVE_SOURCE_EXPECTED_CUDA_LT="$<TARGET_FILE:CUDA::cublasLt>")
    endif()
    if(HAVE_ROCM)
        target_sources(v2_perf_native_context_source_probe PRIVATE integration/backends/NativeContextSourceProbeROCm.cpp)
        # The observer directly references these providers. The core's private
        # dependency closure does not satisfy the executable's own ELF imports.
        target_link_libraries(v2_perf_native_context_source_probe PRIVATE
            roc::hipblas "${ROCM_PATH}/lib/libhipblaslt.so")
        target_compile_definitions(v2_perf_native_context_source_probe PRIVATE
            NATIVE_SOURCE_EXPECTED_ROCM_BLAS="$<TARGET_FILE:roc::hipblas>"
            NATIVE_SOURCE_EXPECTED_ROCM_LT="${ROCM_PATH}/lib/libhipblaslt.so")
    endif()
endif()
foreach(_native_source_backend IN ITEMS CUDA ROCm)
    if((_native_source_backend STREQUAL "CUDA" AND HAVE_CUDA) OR (_native_source_backend STREQUAL "ROCm" AND HAVE_ROCM))
        if(_native_source_backend STREQUAL "CUDA")
            set(_native_source_device cuda:0)
            set(_native_source_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
            set(_native_source_lock cuda_gpu_0)
        else()
            set(_native_source_device rocm:0)
            set(_native_source_env "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=0")
            set(_native_source_lock rocm_gpu_0)
        endif()
        add_v2_test(V2_Integration_${_native_source_backend}_OriginalPreparedSourceLifecycle
            COMMAND $<TARGET_FILE:v2_perf_native_context_source_probe> --device ${_native_source_device} --source-lifecycle --replays 2
            LABELS "V2;Integration;ProductionTestPreflight;${_native_source_backend};Memory;Ownership;NativeContext;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS RESOURCE_LOCK ${_native_source_lock} TIMEOUT 30 ENVIRONMENT ${_native_source_env})
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_native_source_backend}_OriginalPreparedSourceLifecycle)
    endif()
endforeach()

foreach(_native_source_probe_gate IN ITEMS Unit Integration)
    set(_native_source_probe_labels "V2;${_native_source_probe_gate};DeviceFree;Memory;Ownership;TraceReplay;Regression")
    if(_native_source_probe_gate STREQUAL "Integration")
        list(APPEND _native_source_probe_labels ProductionTestPreflight)
    endif()
    add_v2_test(V2_${_native_source_probe_gate}_NativeContextSourceProbeReplay
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_context_source_probe.py
        LABELS "${_native_source_probe_labels}" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
endforeach()
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeContextSourceProbeReplay)

# Closed lookup-event measurement protocol is independently named in preflight.
foreach(_lookup_event_protocol_gate IN ITEMS Unit Integration)
    set(_lookup_event_protocol_labels "V2;${_lookup_event_protocol_gate};DeviceFree;Memory;Ownership;TraceReplay;Regression")
    if(_lookup_event_protocol_gate STREQUAL "Integration")
        list(APPEND _lookup_event_protocol_labels ProductionTestPreflight)
    endif()
    add_v2_test(V2_${_lookup_event_protocol_gate}_NativeLookupOrderingMeasurementReplay
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_context_birth_calibration.py LookupOrderingReceipt
        LABELS "${_lookup_event_protocol_labels}" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
endforeach()
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeLookupOrderingMeasurementReplay)
