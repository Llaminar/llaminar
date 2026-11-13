# Explicit diagnostic infrastructure, excluded from ordinary builds and test
# gates. Measurements cannot mint a production resource certificate.
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/NativeSourceFingerprint.cmake")
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/backends/ReadAheadNativeResourceFamily.cpp" _read_ahead_family_cpp_sha)
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/backends/ReadAheadNativeResourceFamily.h" _read_ahead_family_h_sha)
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/planning/PhysicalMemoryBOM.h" _read_ahead_family_bom_sha)
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/backends/ReadAheadNativeParentTerminal.h" _read_ahead_parent_h_sha)
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/backends/ReadAheadNativeParentTerminal.cpp" _read_ahead_parent_cpp_sha)
string(SHA256 _read_ahead_family_source_sha "${_read_ahead_family_cpp_sha}:${_read_ahead_family_h_sha}:${_read_ahead_family_bom_sha}:${_read_ahead_parent_h_sha}:${_read_ahead_parent_cpp_sha}")
# Bind native creation, capture, metadata and original context retirement code,
# not only the thin family wrapper. Binary/ELF identities remain independently
# recorded because source hashes alone do not authenticate a linked runtime.
set(_read_ahead_native_source_inputs "")
foreach(_read_ahead_native_file
    backends/IWorkerGPUContext.h
    backends/IGPUGraphCapture.h
    backends/GPUDeviceWorker.h
    backends/GPUDeviceWorker.cpp
    backends/GPUDeviceContextPool.h
    backends/GPUDeviceContextPool.cpp
    backends/NativeContextRetirement.h
    backends/NativeContextRetirement.cpp
    backends/ReadAheadNativeResourceObligation.h
    backends/GPUDeviceProcessLifetime.h
    backends/GPUDeviceProcessLifetime.cpp
    backends/GPUGraphMemoryContract.h
    backends/cuda/NvidiaDeviceContext.h
    backends/cuda/NvidiaDeviceContext.cu
    backends/rocm/AMDDeviceContext.h
    backends/rocm/AMDDeviceContext.cpp
    backends/NativeAllocationCaptureLease.h
    backends/IBackend.h
    backends/BackendManager.h
    backends/BackendManager.cpp
    backends/cuda/CUDAWorkerImpl.cu
    backends/cuda/CUDAGraphCapture.h
    backends/cuda/CUDAGraphCapture.cu
    backends/cuda/CUDABackend.h
    backends/cuda/CUDABackend.cu
    backends/cuda/CUDAHardwareMetadata.h
    backends/cuda/CUDAHardwareMetadata.cu
    backends/rocm/ROCmWorkerImpl.cpp
    backends/rocm/HIPGraphCapture.h
    backends/rocm/HIPGraphCapture.cpp
    backends/rocm/ROCmBackend.h
    backends/rocm/ROCmBackend.cpp
    backends/rocm/ROCmHardwareMetadata.h
    backends/rocm/ROCmHardwareMetadata.cpp
)
    llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2/${_read_ahead_native_file}" _read_ahead_native_sha)
    string(APPEND _read_ahead_native_source_inputs "${_read_ahead_native_file}:${_read_ahead_native_sha};")
endforeach()
string(SHA256 _read_ahead_family_source_sha "${_read_ahead_family_source_sha}:${_read_ahead_native_source_inputs}")
set(_read_ahead_calibration_source_inputs "")
foreach(_read_ahead_calibration_file
    ReadAheadNativeResourceCalibration.cpp ReadAheadCalibrationVersions.h
    ReadAheadCalibrationCUDAVersions.cpp ReadAheadCalibrationROCmVersions.cpp)
    llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/integration/backends/${_read_ahead_calibration_file}" _read_ahead_calibration_sha)
    string(APPEND _read_ahead_calibration_source_inputs "${_read_ahead_calibration_file}:${_read_ahead_calibration_sha};")
endforeach()
string(SHA256 _read_ahead_calibration_source_sha "${_read_ahead_calibration_source_inputs}")
add_executable(v2_perf_read_ahead_resource_calibration EXCLUDE_FROM_ALL
    integration/backends/ReadAheadNativeResourceCalibration.cpp)
target_link_libraries(v2_perf_read_ahead_resource_calibration PRIVATE llaminar2_core)
target_include_directories(v2_perf_read_ahead_resource_calibration PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor")
target_compile_definitions(v2_perf_read_ahead_resource_calibration PRIVATE
    READ_AHEAD_RESOURCE_SOURCE_SHA256="${_read_ahead_family_source_sha}"
    READ_AHEAD_CALIBRATION_SOURCE_SHA256="${_read_ahead_calibration_source_sha}")

if(HAVE_CUDA)
    target_sources(v2_perf_read_ahead_resource_calibration PRIVATE
        integration/backends/ReadAheadCalibrationCUDAVersions.cpp)
endif()
if(HAVE_ROCM)
    target_sources(v2_perf_read_ahead_resource_calibration PRIVATE
        integration/backends/ReadAheadCalibrationROCmVersions.cpp)
endif()

# Parser/protocol rejection replays are device-free and do not mint native limits.
add_v2_test(V2_Unit_ReadAheadResourceCalibrationDriver
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_read_ahead_resource_calibration.py
    LABELS "V2;Unit;DeviceFree;Memory;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
add_v2_test(V2_Integration_ReadAheadResourceCalibrationDriverReplay
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_read_ahead_resource_calibration.py
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ReadAheadResourceCalibrationDriverReplay)
