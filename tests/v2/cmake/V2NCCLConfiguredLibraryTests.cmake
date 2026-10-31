# Exact NCCL/RCCL configured-path ownership through actual production loader bodies.
# These wrappers intercept dlopen/dlsym and never load a DSO or query a runtime.
# Original mode requires exact observed legacy faults with ordinary successful assertions.
include_guard(DIRECTORY)

# Assert the real production source property, not the unit spy's sentinel macro.
# DIRECTORY matters: source properties are owned by src/v2, while this module
# executes in tests/v2. Removal, wrong scope or a competing definition is fatal.
foreach(V2_PATH_BACKEND NCCL RCCL)
    if((V2_PATH_BACKEND STREQUAL "NCCL" AND HAVE_CUDA AND HAVE_NCCL) OR
       (V2_PATH_BACKEND STREQUAL "RCCL" AND HAVE_ROCM AND HAVE_RCCL))
        get_source_file_property(V2_NCCL_PATH_PRODUCTION_DEFINITIONS
            "${REPO_ROOT}/src/v2/collective/backends/${V2_PATH_BACKEND}DynamicLoader.cpp"
            DIRECTORY "${REPO_ROOT}/src/v2" COMPILE_DEFINITIONS)
        set(V2_NCCL_PATH_PRODUCTION_BINDINGS ${V2_NCCL_PATH_PRODUCTION_DEFINITIONS})
        list(FILTER V2_NCCL_PATH_PRODUCTION_BINDINGS INCLUDE
            REGEX "^LLAMINAR_${V2_PATH_BACKEND}_LIBRARY_PATH=")
        list(LENGTH V2_NCCL_PATH_PRODUCTION_BINDINGS V2_NCCL_PATH_BINDING_COUNT)
        set(V2_NCCL_PATH_SELECTED "${${V2_PATH_BACKEND}_LIBRARY}")
        set(V2_NCCL_PATH_EXPECTED "LLAMINAR_${V2_PATH_BACKEND}_LIBRARY_PATH=\"${V2_NCCL_PATH_SELECTED}\"")
        if(NOT V2_NCCL_PATH_SELECTED OR NOT V2_NCCL_PATH_BINDING_COUNT EQUAL 1 OR
           NOT "${V2_NCCL_PATH_PRODUCTION_BINDINGS}" STREQUAL "${V2_NCCL_PATH_EXPECTED}")
            message(FATAL_ERROR
                "${V2_PATH_BACKEND} loader source must bind exactly the selected ${V2_PATH_BACKEND}_LIBRARY via LLAMINAR_${V2_PATH_BACKEND}_LIBRARY_PATH")
        endif()
    endif()
endforeach()

if(NOT V2_PERF_TESTS_ONLY)
    foreach(V2_PATH_BACKEND NCCL RCCL)
        string(TOLOWER "${V2_PATH_BACKEND}" V2_PATH_BACKEND_LOWER)
        set(V2_NCCL_PATH_SOURCE "${REPO_ROOT}/src/v2/collective/backends/${V2_PATH_BACKEND}DynamicLoader.cpp")
        set(V2_NCCL_PATH_FIXTURE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/collective/${V2_PATH_BACKEND}LoaderOriginalLoad.inc")
        set(V2_NCCL_PATH_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/integration/build/make_nccl_loader_path_original.py")
        set(V2_NCCL_PATH_ORIGINAL "${CMAKE_CURRENT_BINARY_DIR}/${V2_PATH_BACKEND_LOWER}-path-original/${V2_PATH_BACKEND}DynamicLoader.cpp")
        add_custom_command(OUTPUT "${V2_NCCL_PATH_ORIGINAL}"
            COMMAND ${Python3_EXECUTABLE} "${V2_NCCL_PATH_GENERATOR}"
                --backend ${V2_PATH_BACKEND} --source "${V2_NCCL_PATH_SOURCE}" --original-load "${V2_NCCL_PATH_FIXTURE}"
                --output "${V2_NCCL_PATH_ORIGINAL}"
            DEPENDS "${V2_NCCL_PATH_SOURCE}" "${V2_NCCL_PATH_FIXTURE}" "${V2_NCCL_PATH_GENERATOR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/check_cpu_prepared_allocation_policy.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/check_gpu_blocking_sync_policy.py"
            VERBATIM)
        add_custom_target(v2_${V2_PATH_BACKEND_LOWER}_path_original_source DEPENDS "${V2_NCCL_PATH_ORIGINAL}")
        foreach(V2_NCCL_PATH_MODE configured unconfigured empty original)
            set(V2_NCCL_PATH_TARGET "v2_unit_${V2_PATH_BACKEND_LOWER}_path_${V2_NCCL_PATH_MODE}")
            add_executable(${V2_NCCL_PATH_TARGET}
                "${CMAKE_CURRENT_SOURCE_DIR}/unit/collective/backends/Test__${V2_PATH_BACKEND}ConfiguredLibrary.cpp")
            target_include_directories(${V2_NCCL_PATH_TARGET} PRIVATE
                "${REPO_ROOT}/src/v2" "${REPO_ROOT}/src/v2/collective/backends")
            target_compile_features(${V2_NCCL_PATH_TARGET} PRIVATE cxx_std_20)
            target_link_libraries(${V2_NCCL_PATH_TARGET} PRIVATE GTest::gtest_main)
            if(V2_NCCL_PATH_MODE STREQUAL "unconfigured")
                target_compile_definitions(${V2_NCCL_PATH_TARGET} PRIVATE LLAMINAR_NCCL_TEST_UNCONFIGURED=1)
            elseif(V2_NCCL_PATH_MODE STREQUAL "empty")
                target_compile_definitions(${V2_NCCL_PATH_TARGET} PRIVATE LLAMINAR_NCCL_TEST_EMPTY_CONFIGURATION=1)
            elseif(V2_NCCL_PATH_MODE STREQUAL "original")
                add_dependencies(${V2_NCCL_PATH_TARGET} v2_${V2_PATH_BACKEND_LOWER}_path_original_source)
                target_compile_definitions(${V2_NCCL_PATH_TARGET} PRIVATE
                    LLAMINAR_NCCL_TEST_ORIGINAL=1
                    "LLAMINAR_NCCL_LOADER_CPP=\"${V2_NCCL_PATH_ORIGINAL}\"")
            endif()
            set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS ${V2_NCCL_PATH_TARGET})
            foreach(V2_NCCL_PATH_SUITE Unit Integration)
                set(V2_NCCL_PATH_TEST "V2_${V2_NCCL_PATH_SUITE}_${V2_PATH_BACKEND}ConfiguredLibrary_${V2_NCCL_PATH_MODE}")
                add_test(NAME ${V2_NCCL_PATH_TEST} COMMAND $<TARGET_FILE:${V2_NCCL_PATH_TARGET}>
                    --gtest_filter=${V2_PATH_BACKEND}ConfiguredLibrary.*)
                set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NCCL_PATH_TEST} ${V2_NCCL_PATH_TARGET})
                set_tests_properties(${V2_NCCL_PATH_TEST} PROPERTIES
                    LABELS "V2;${V2_NCCL_PATH_SUITE};Collective;${V2_PATH_BACKEND};DeviceFree;Dependency;Regression" TIMEOUT 10)
                if(V2_NCCL_PATH_SUITE STREQUAL "Integration")
                    set_property(TEST ${V2_NCCL_PATH_TEST} APPEND PROPERTY LABELS ProductionTestPreflight)
                endif()
            endforeach()
        endforeach()
        foreach(V2_NCCL_PARTIAL_VARIANT configured original)
            if(V2_NCCL_PARTIAL_VARIANT STREQUAL "original")
                set(V2_NCCL_PARTIAL_TEST V2_Integration_${V2_PATH_BACKEND}PartialAdmissionOriginal)
            else()
                set(V2_NCCL_PARTIAL_TEST V2_Integration_${V2_PATH_BACKEND}PartialAdmission)
            endif()
            add_test(NAME ${V2_NCCL_PARTIAL_TEST}
                COMMAND $<TARGET_FILE:v2_unit_${V2_PATH_BACKEND_LOWER}_path_${V2_NCCL_PARTIAL_VARIANT}>
                    --gtest_filter=${V2_PATH_BACKEND}ConfiguredLibrary.PartialAdmissionWithdrawsCallbacksBeforeCloseAndReload)
            set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_NCCL_PARTIAL_TEST}
                v2_unit_${V2_PATH_BACKEND_LOWER}_path_${V2_NCCL_PARTIAL_VARIANT})
            set_tests_properties(${V2_NCCL_PARTIAL_TEST} PROPERTIES
                LABELS "V2;Integration;ProductionTestPreflight;${V2_PATH_BACKEND};DeviceFree;Dependency;Lifecycle;Regression" TIMEOUT 10)
        endforeach()
        if(V2_PATH_BACKEND STREQUAL "RCCL")
            foreach(V2_RCCL_IDENTITY_VARIANT configured original)
                if(V2_RCCL_IDENTITY_VARIANT STREQUAL "original")
                    set(V2_RCCL_IDENTITY_TEST V2_Integration_RCCLRetainedLibraryIdentityOriginal)
                else()
                    set(V2_RCCL_IDENTITY_TEST V2_Integration_RCCLRetainedLibraryIdentity)
                endif()
                add_test(NAME ${V2_RCCL_IDENTITY_TEST}
                    COMMAND $<TARGET_FILE:v2_unit_rccl_path_${V2_RCCL_IDENTITY_VARIANT}>
                        --gtest_filter=RCCLConfiguredLibrary.LiveHandleRejectsDifferentSelectionUntilUnload)
                set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${V2_RCCL_IDENTITY_TEST}
                    v2_unit_rccl_path_${V2_RCCL_IDENTITY_VARIANT})
                set_tests_properties(${V2_RCCL_IDENTITY_TEST} PROPERTIES
                    LABELS "V2;Integration;ProductionTestPreflight;RCCL;DeviceFree;Dependency;Lifecycle;Regression"
                    TIMEOUT 10)
            endforeach()
        endif()
    endforeach()
    foreach(V2_NCCL_PATH_SUITE Unit Integration)
        set(V2_NCCL_PATH_CMAKE_TEST "V2_${V2_NCCL_PATH_SUITE}_NCCLConfiguredLibraryCMake")
        add_test(NAME ${V2_NCCL_PATH_CMAKE_TEST}
            COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_nccl_configured_library_cmake.py")
        set_tests_properties(${V2_NCCL_PATH_CMAKE_TEST} PROPERTIES
            LABELS "V2;${V2_NCCL_PATH_SUITE};NCCL;RCCL;DeviceFree;CMake;Dependency;Regression" TIMEOUT 30)
        if(V2_NCCL_PATH_SUITE STREQUAL "Integration")
            set_property(TEST ${V2_NCCL_PATH_CMAKE_TEST} APPEND PROPERTY LABELS ProductionTestPreflight)
        endif()
    endforeach()

endif()
