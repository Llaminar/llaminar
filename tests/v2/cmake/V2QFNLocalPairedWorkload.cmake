# Explicit local paired-document inputs for the existing QFN matrix only.
# The empty default reads nothing and preserves portable ordinary cases. The
# generated header freezes complete small metadata/source identity into the
# original executable; files cannot silently alter discovery after compilation.
include_guard(DIRECTORY)
set(V2_QFN_LOCAL_PAIRED_BINDINGS "" CACHE FILEPATH
    "Explicit local QFN paired binding JSON; empty preserves ordinary cases")

function(v2_qfn_local_paired_target target predictor_output)
    set(${predictor_output} "" PARENT_SCOPE)
    if(NOT V2_QFN_LOCAL_PAIRED_BINDINGS)
        return()
    endif()
    if(V2_PERF_TESTS_ONLY OR NOT TARGET ${target})
        message(FATAL_ERROR "Local QFN paired bindings require the actual Integration matrix target")
    endif()
    get_filename_component(_source_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../.." ABSOLUTE)
    set(_freezer "${_source_root}/scripts/ci/freeze_qfn_paired_profiles.py")
    set(_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/qfn-local-paired")
    set(_header "${_directory}/QFNLocalPairedBindings.generated.h")
    set(_metadata "${_directory}/build-input.json")
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${_freezer}"
        --source-root "${_source_root}" --bindings "${V2_QFN_LOCAL_PAIRED_BINDINGS}"
        --header "${_header}" --metadata "${_metadata}"
        RESULT_VARIABLE _status OUTPUT_VARIABLE _input ERROR_VARIABLE _error)
    if(NOT _status EQUAL 0)
        message(FATAL_ERROR "Local QFN paired inputs failed before registration: ${_error}")
    endif()
    string(JSON _predictor GET "${_input}" predictor)
    if(NOT EXISTS "${_predictor}")
        message(FATAL_ERROR "Local QFN paired predictor dependency is unavailable: ${_predictor}")
    endif()
    string(JSON _count LENGTH "${_input}" dependencies)
    math(EXPR _last "${_count}-1")
    set(_dependencies "${_freezer}")
    foreach(_index RANGE 0 ${_last})
        string(JSON _dependency GET "${_input}" dependencies ${_index})
        list(APPEND _dependencies "${_dependency}")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_dependencies})
    set_source_files_properties("${_header}" PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(${target} PRIVATE "${_header}")
    target_include_directories(${target} PRIVATE "${_directory}")
    target_compile_definitions(${target} PRIVATE LLAMINAR_QFN_LOCAL_PAIRED_BINDINGS=1)
    # A changed source/dependency document must regenerate actual CTest metadata,
    # including model manifests, through the existing post-link discovery owner.
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${_header}" "${_metadata}")
    set(${predictor_output} "${_predictor}" PARENT_SCOPE)
endfunction()
