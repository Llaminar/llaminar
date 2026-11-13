#[=[
@file V2ROCmHSAConfiguredLibraryTests.cmake
@brief Replay configured HSA opens through all original callers without an SDK.

The production source property is checked separately from each inert replay's
sentinel path. A stale/bare production definition cannot pass by compiling only
correct test macros. The original control retains the actual former five opens.
]=]
include_guard(DIRECTORY)

if(HAVE_ROCM)
    get_source_file_property(hsa_definitions
        "${REPO_ROOT}/src/v2/backends/rocm/ROCmBackend.cpp"
        DIRECTORY "${REPO_ROOT}/src/v2" COMPILE_DEFINITIONS)
    list(FILTER hsa_definitions INCLUDE REGEX "^LLAMINAR_HSA_RUNTIME_LIBRARY_PATH=")
    list(LENGTH hsa_definitions hsa_binding_count)
    if(NOT hsa_binding_count EQUAL 1 OR NOT hsa_definitions STREQUAL
        "LLAMINAR_HSA_RUNTIME_LIBRARY_PATH=\"${ROCM_HSA_RUNTIME_LIBRARY}\"")
        message(FATAL_ERROR "ROCmBackend HSA opens must bind the exact CMake-selected SDK library")
    endif()
endif()

set(hsa_source "${REPO_ROOT}/src/v2/backends/rocm/ROCmBackend.cpp")
set(hsa_generator "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/make_rocm_hsa_loader_replay.py")
foreach(hsa_mode configured original)
    set(hsa_target "v2_unit_rocm_hsa_configured_library_${hsa_mode}")
    set(hsa_replay "${CMAKE_CURRENT_BINARY_DIR}/hsa-${hsa_mode}/actual.inc")
    set(hsa_original_argument "")
    if(hsa_mode STREQUAL "original")
        set(hsa_original_argument --original)
    endif()
    add_custom_command(OUTPUT "${hsa_replay}"
        COMMAND ${Python3_EXECUTABLE} -B "${hsa_generator}"
            --source "${hsa_source}" --output "${hsa_replay}" ${hsa_original_argument}
        DEPENDS "${hsa_source}" "${hsa_generator}"
            unit/scripts/check_cpu_prepared_allocation_policy.py
        VERBATIM)
    add_executable(${hsa_target} unit/backends/Test__ROCmHSAConfiguredLibrary.cpp "${hsa_replay}")
    if(TARGET ${hsa_target})
        target_compile_features(${hsa_target} PRIVATE cxx_std_20)
        target_compile_definitions(${hsa_target} PRIVATE "LLAMINAR_ROCM_HSA_REPLAY_CPP=\"${hsa_replay}\"")
        if(hsa_mode STREQUAL "original")
            target_compile_definitions(${hsa_target} PRIVATE LLAMINAR_ROCM_HSA_ORIGINAL=1)
        endif()
        target_link_libraries(${hsa_target} PRIVATE GTest::gtest_main)
        foreach(hsa_suite Unit Integration)
            add_v2_test(V2_${hsa_suite}_ROCmHSAConfiguredLibrary_${hsa_mode}
                COMMAND $<TARGET_FILE:${hsa_target}>
                LABELS "V2;${hsa_suite};DeviceFree;ROCm;HSA;Dependency;Regression"
                MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 10
                ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1")
        endforeach()
        if(TEST V2_Integration_ROCmHSAConfiguredLibrary_${hsa_mode})
            set_property(TEST V2_Integration_ROCmHSAConfiguredLibrary_${hsa_mode}
                APPEND PROPERTY LABELS ProductionTestPreflight)
            list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmHSAConfiguredLibrary_${hsa_mode})
        endif()
    endif()
endforeach()
# Like the existing SDK/header binding checks, this language-free script has
# no native executable owner. Keep it registered in performance-only Release
# too; selected-native-owner filtering must not erase a script-only contract.
add_test(NAME V2_Integration_ROCmHSASDKBinding
    COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_rocm_hsa_sdk_binding.py")
set_tests_properties(V2_Integration_ROCmHSASDKBinding PROPERTIES
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;HSA;SDK;Regression"
    TIMEOUT 30 ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1")
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ROCmHSASDKBinding)
