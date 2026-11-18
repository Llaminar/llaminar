#[=[
@file V2CastagnaKernelABTests.cmake
@brief Explicit external-source full-region comparison on the canonical FFN owner.

An empty path disables this optional diagnostic completely. A requested checkout
is authenticated by the source exporter; there are no downloads, production
execution flags, SDK substitutions, architecture reductions or spill exceptions.
The generated original device section and thin launch wrapper inherit the same
shipping HIP compiler, target set and allocation guard as the existing owner.
]=]
include_guard(DIRECTORY)
if(LLAMINAR_CASTAGNA_SOURCE_DIR STREQUAL "")
    return()
endif()
if(NOT HAVE_ROCM OR NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR "Castagna kernel A/B was requested but the canonical ROCm compiler is unavailable")
endif()
if(NOT TARGET v2_integration_moe_intermediate_ffn_finalization)
    message(FATAL_ERROR "Castagna kernel A/B requires the original intermediate FFN test owner")
endif()
if(NOT V2_TEST_SOCKETS OR V2_TEST_SOCKETS LESS 1)
    message(FATAL_ERROR "Castagna kernel A/B requires canonical rank/NUMA inventory")
endif()
get_filename_component(_castagna_source "${LLAMINAR_CASTAGNA_SOURCE_DIR}" REALPATH)
set(_castagna_original "${_castagna_source}/ggml/src/ggml-cuda/hc-persist.cu")
set(_castagna_license "${_castagna_source}/LICENSE")
set(_castagna_adapter "${CMAKE_CURRENT_SOURCE_DIR}/integration/kernels/rocm/castagna")
set(_castagna_exporter "${_castagna_adapter}/extract_original.py")
set(_castagna_native_exporter "${_castagna_adapter}/adapt_source.py")
foreach(_required IN ITEMS "${_castagna_original}" "${_castagna_license}"
        "${_castagna_exporter}" "${_castagna_adapter}/CastagnaHcPersistExport.h"
        "${_castagna_adapter}/CastagnaHcPersistExport.hip"
        "${_castagna_native_exporter}" "${_castagna_adapter}/CastagnaNativeWeightExport.h"
        "${_castagna_adapter}/CastagnaNativeWeightExport.hip"
        "${_castagna_adapter}/NativeWeightReaders.inc")
    if(NOT EXISTS "${_required}")
        message(FATAL_ERROR "Requested Castagna kernel A/B input is missing: ${_required}")
    endif()
endforeach()
set(_castagna_generated "${CMAKE_CURRENT_BINARY_DIR}/castagna-original")
# Fail during configuration for a wrong/dirty pin, then repeat authentication
# whenever a tracked source dependency changes before HIP compilation.
execute_process(COMMAND "${Python3_EXECUTABLE}" "${_castagna_exporter}"
    --source-root "${_castagna_source}" --output-dir "${_castagna_generated}"
    RESULT_VARIABLE _castagna_status OUTPUT_VARIABLE _castagna_stdout ERROR_VARIABLE _castagna_stderr)
if(NOT _castagna_status EQUAL 0)
    message(FATAL_ERROR "Castagna original-source authentication failed: ${_castagna_stdout}${_castagna_stderr}")
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" "${_castagna_native_exporter}"
    --source-root "${_castagna_source}" --output-dir "${_castagna_generated}"
    RESULT_VARIABLE _castagna_native_status OUTPUT_VARIABLE _castagna_native_stdout ERROR_VARIABLE _castagna_native_stderr)
if(NOT _castagna_native_status EQUAL 0)
    message(FATAL_ERROR "Castagna native-weight source adaptation failed: ${_castagna_native_stdout}${_castagna_native_stderr}")
endif()
file(GLOB_RECURSE _castagna_headers CONFIGURE_DEPENDS
    "${_castagna_source}/ggml/include/*.h"
    "${_castagna_source}/ggml/src/*.h"
    "${_castagna_source}/ggml/src/*.cuh"
    "${_castagna_source}/ggml/src/*.hpp"
    "${_castagna_source}/ggml/src/*.inc")
add_custom_command(
    OUTPUT "${_castagna_generated}/CastagnaOriginalDeviceSection.inc"
        "${_castagna_generated}/castagna-source-manifest.json"
        "${_castagna_generated}/CASTAGNA-LICENSE"
    COMMAND "${Python3_EXECUTABLE}" "${_castagna_exporter}"
        --source-root "${_castagna_source}" --output-dir "${_castagna_generated}"
    DEPENDS "${_castagna_exporter}" "${_castagna_original}" "${_castagna_license}" ${_castagna_headers}
    COMMENT "Authenticate and export the pinned original Castagna device section"
    VERBATIM)
add_custom_command(
    OUTPUT "${_castagna_generated}/CastagnaNativeDeviceSection.inc"
        "${_castagna_generated}/adapted-source-manifest.json"
    COMMAND "${Python3_EXECUTABLE}" "${_castagna_native_exporter}"
        --source-root "${_castagna_source}" --output-dir "${_castagna_generated}"
    DEPENDS "${_castagna_native_exporter}" "${_castagna_original}" "${_castagna_license}"
        "${_castagna_adapter}/NativeWeightReaders.inc" ${_castagna_headers}
    COMMENT "Adapt only the pinned Castagna weight readers to canonical prepared matrices"
    VERBATIM)
set(_castagna_wrapper "${_castagna_adapter}/CastagnaHcPersistExport.hip")
set(_castagna_native_wrapper "${_castagna_adapter}/CastagnaNativeWeightExport.hip")
set_source_files_properties("${_castagna_wrapper}" PROPERTIES LANGUAGE HIP
    OBJECT_DEPENDS "${_castagna_generated}/CastagnaOriginalDeviceSection.inc;${_castagna_generated}/castagna-source-manifest.json"
    COMPILE_DEFINITIONS "GGML_USE_HIP")
set_source_files_properties("${_castagna_native_wrapper}" PROPERTIES LANGUAGE HIP
    OBJECT_DEPENDS "${_castagna_generated}/CastagnaNativeDeviceSection.inc;${_castagna_generated}/adapted-source-manifest.json;${_castagna_adapter}/NativeWeightReaders.inc"
    COMPILE_DEFINITIONS "GGML_USE_HIP")
target_sources(v2_integration_moe_intermediate_ffn_finalization PRIVATE
    "${_castagna_wrapper}" "${_castagna_generated}/CastagnaOriginalDeviceSection.inc"
    "${_castagna_native_wrapper}" "${_castagna_generated}/CastagnaNativeDeviceSection.inc")
target_include_directories(v2_integration_moe_intermediate_ffn_finalization PRIVATE
    "${_castagna_adapter}" "${_castagna_generated}"
    "${_castagna_source}/ggml/src/ggml-cuda" "${_castagna_source}/ggml/src" "${_castagna_source}/ggml/include")
target_compile_definitions(v2_integration_moe_intermediate_ffn_finalization PRIVATE LLAMINAR_CASTAGNA_KERNEL_AB=1)
set_target_properties(v2_integration_moe_intermediate_ffn_finalization PROPERTIES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
target_link_libraries(v2_integration_moe_intermediate_ffn_finalization PRIVATE hip::host)

# Release-only real-weight comparison for the opt-in retained closed owner.
# The operator supplies the original GGUF and may select one exact rank-local PCI. NO_MODELS skips the unrelated
# managed V2_Models download; this test requires the operator-provided GGUF.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_v2_test(V2_Perf_RetainedClosedHCFFN_RealFinal_ROCm
        COMMAND $<TARGET_FILE:v2_integration_moe_intermediate_ffn_finalization>
            --gtest_also_run_disabled_tests --gtest_filter=CastagnaFullRegionAB.DISABLED_ROCmRetainedClosedRealFinalEvents
        LABELS "V2;Performance;ROCm;MoE;Hyperconnection;GraphCapture;ComponentTiming;RealModel;ExternalComparator;HardwareOptional;Devices1;NativeISA-gfx906"
        MPI_PROCS ${V2_TEST_SOCKETS} NO_MODELS TIMEOUT 120 RESOURCE_LOCK RealModel_Perf
        ENVIRONMENT "LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16")
endif()
