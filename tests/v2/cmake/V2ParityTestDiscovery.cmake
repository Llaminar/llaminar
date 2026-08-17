# =============================================================================
# V2ParityTestDiscovery.cmake
#
# POST_BUILD helper script: runs a GTest binary with --gtest_list_tests
# (no mpirun), parses the output, and generates a CTest include file with
# isolated focused tests plus backend-grouped ProductionParity campaigns.
#
# Inputs (via -D):
#   TEST_EXECUTABLE  - Full path to the GTest binary
#   CTEST_FILE       - Output .cmake file to generate
#   TEST_PREFIX      - CTest name prefix (e.g. V2_Integration_Parity_Qwen2_SingleDevice)
#   LABELS           - Semicolon-separated CTest labels
#   MPI_PROCS        - Number of MPI processes for test execution
#   NUM_SOCKETS      - Detected CPU socket count
#   CORES_PER_SOCKET - Cores per socket
#   WORKING_DIR      - Working directory for tests
#   TIMEOUT          - Optional per-test timeout in seconds
#   PRODUCTION_PERF_STATS_FILTER - Optional campaign PerfStats domains
#   PRODUCTION_ISOLATED_TESTS_SERIALIZED - Pipe-separated literal GTest-name
#                         fragments whose matching cells require a fresh MPI
#                         application lifetime
#   MODEL_FILES_SERIALIZED - Pipe-separated GGUF inputs owned by this binary
#   <BACKEND>_MODEL_FILES_SERIALIZED - Additional inputs for that backend
# =============================================================================

# The one-hour economy requirement is a reporting target, never an execution
# timeout. This separate ceiling exists only to terminate a stuck standalone
# campaign; the aggregate runner applies the same safety horizon to the matrix.
set(_production_campaign_completion_timeout_seconds 21600)

# --- Run the binary to discover tests (no mpirun needed) --------------------
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "HWLOC_COMPONENTS=-gl,-opencl"
        "OMPI_MCA_btl_vader_single_copy_mechanism=none"
        "${TEST_EXECUTABLE}" --gtest_list_tests
    OUTPUT_VARIABLE _test_list_output
    ERROR_QUIET
    RESULT_VARIABLE _result
    TIMEOUT 30
)

if(NOT _result EQUAL 0)
    message(STATUS "V2ParityTestDiscovery: Failed to list tests from ${TEST_EXECUTABLE} (exit ${_result})")
    file(WRITE "${CTEST_FILE}" "# Discovery failed for ${TEST_EXECUTABLE}\n")
    return()
endif()

# --- Parse gtest_list_tests output -------------------------------------------
# Format:
#   SuiteName.
#     TestMethod/ConfigName  # GetParam() = ...
#     TestMethod/AnotherConfig
#   AnotherSuite.
#     ...
#
# Full test name = <suite_line><case_line_trimmed>  (suite ends with '.')

set(_current_suite "")
set(_all_tests "")

string(REPLACE "\n" ";" _lines "${_test_list_output}")
foreach(_line IN LISTS _lines)
    # Skip empty lines
    if("${_line}" STREQUAL "")
        continue()
    endif()

    # Check if this is a suite line (does not start with whitespace)
    string(REGEX MATCH "^[^ ]" _is_suite "${_line}")
    if(_is_suite)
        # Suite line: strip trailing whitespace, keep the trailing dot
        string(STRIP "${_line}" _current_suite)
    else()
        # Test case line: strip leading whitespace and trailing comments
        string(STRIP "${_line}" _case_name)
        # Remove gtest parameter comment:  "  # GetParam() = ..."
        string(REGEX REPLACE "  # .*$" "" _case_name "${_case_name}")
        string(STRIP "${_case_name}" _case_name)

        if(NOT "${_case_name}" STREQUAL "" AND NOT "${_current_suite}" STREQUAL "")
            # Full GTest name: SuiteName.TestCase  (suite already has trailing dot)
            set(_full_name "${_current_suite}${_case_name}")
            list(APPEND _all_tests "${_full_name}")
        endif()
    endif()
endforeach()

list(LENGTH _all_tests _test_count)
if(_test_count EQUAL 0)
    message(STATUS "V2ParityTestDiscovery: No tests discovered from ${TEST_EXECUTABLE}")
    file(WRITE "${CTEST_FILE}" "# No tests discovered from ${TEST_EXECUTABLE}\n")
    return()
endif()

# --- Build the mpirun command prefix ----------------------------------------
if(NOT MPI_PROCS)
    set(MPI_PROCS 1)
endif()

set(_mpi_cmd
    "mpirun"
    "-np" "${MPI_PROCS}"
    "--bind-to" "socket"
    "--map-by" "socket"
    "--mca" "mpi_leave_pinned" "1"
    "--mca" "btl_vader_single_copy_mechanism" "none"
    "--mca" "orte_allowed_exit_without_sync" "1"
)

# Add --oversubscribe if we have fewer sockets than MPI ranks
if(NUM_SOCKETS AND MPI_PROCS GREATER NUM_SOCKETS)
    list(APPEND _mpi_cmd "--oversubscribe")
endif()

# --- Build environment variables list ----------------------------------------
if(NOT CORES_PER_SOCKET OR CORES_PER_SOCKET EQUAL "" OR CORES_PER_SOCKET EQUAL 0)
    set(CORES_PER_SOCKET 1)
endif()

if(NOT PRODUCTION_PERF_STATS_FILTER)
    set(PRODUCTION_PERF_STATS_FILTER "forward_graph")
endif()

# Normalize the binary-owned model manifest once. Production campaigns expose
# the same list through REQUIRED_FILES for machine-readable aggregate discovery
# and through an environment contract for runtime path-remap validation.
foreach(_model_scope IN ITEMS COMMON CPU CUDA ROCM)
    if(_model_scope STREQUAL "COMMON")
        set(_serialized_scope_models "${MODEL_FILES_SERIALIZED}")
    else()
        set(_serialized_scope_models
            "${${_model_scope}_MODEL_FILES_SERIALIZED}")
    endif()
    set(_production_model_files_${_model_scope} "")
    if(_serialized_scope_models)
        string(REPLACE "|" ";" _declared_model_files
            "${_serialized_scope_models}")
    else()
        set(_declared_model_files "")
    endif()
    foreach(_model_file IN LISTS _declared_model_files)
        if(IS_ABSOLUTE "${_model_file}")
            get_filename_component(_model_file_absolute "${_model_file}" ABSOLUTE)
        else()
            get_filename_component(
                _model_file_absolute "${_model_file}" ABSOLUTE
                BASE_DIR "${WORKING_DIR}")
        endif()
        list(APPEND _production_model_files_${_model_scope}
            "${_model_file_absolute}")
    endforeach()
    list(REMOVE_DUPLICATES _production_model_files_${_model_scope})
endforeach()

set(_all_production_model_files
    ${_production_model_files_COMMON}
    ${_production_model_files_CPU}
    ${_production_model_files_CUDA}
    ${_production_model_files_ROCM})
list(REMOVE_DUPLICATES _all_production_model_files)

set(_env_vars
    "LLAMINAR_LOG_LEVEL=DEBUG"
    "LLAMINAR_TP_COLLECT_TIMEOUT_MS=30000"
    "HWLOC_COMPONENTS=-gl,-opencl"
    "OMP_NUM_THREADS=${CORES_PER_SOCKET}"
    "OMP_PLACES=sockets"
    "OMP_PROC_BIND=close"
    "OMP_NESTED=false"
    "OMP_DYNAMIC=false"
    "KMP_AFFINITY=granularity=fine,compact,1,0"
    "KMP_BLOCKTIME=0"
    "MKL_NUM_THREADS=${CORES_PER_SOCKET}"
    "MKL_DYNAMIC=false"
    "OMPI_MCA_mpi_leave_pinned=1"
    "OMPI_MCA_btl_vader_single_copy_mechanism=none"
    "OMPI_MCA_btl_openib_allow_ib=1"
)

# Add ROCm env if labels mention ROCm
if("${LABELS}" MATCHES "ROCm")
    list(APPEND _env_vars "HSA_OVERRIDE_GFX_VERSION=9.0.6")
endif()

# Graph-native parity is only meaningful when the captured production graph
# path is admitted.  Keep the full KV context capacity independent from the
# resident activation bucket: runtime planning chooses the largest fitting
# bucket and long prompts are scheduled through those bounded captures.
set(_graph_native_capture_env "")
if("${LABELS}" MATCHES "GraphNative")
    set(_graph_native_capture_env
        "LLAMINAR_GPU_GRAPHS=1"
        "LLAMINAR_PREFILL_GRAPH_BUCKETS=1"
        "LLAMINAR_PREFILL_GRAPH_REQUIRED=1")
    list(APPEND _env_vars ${_graph_native_capture_env})
endif()

# --- Generate CTest include file --------------------------------------------
set(_output "# Auto-generated by V2ParityTestDiscovery.cmake\n")
string(APPEND _output "# Source: ${TEST_EXECUTABLE}\n")
string(APPEND _output "# Tests discovered: ${_test_count}\n\n")
set(_production_campaign_keys "")
set(_production_isolated_test_patterns "")
if(PRODUCTION_ISOLATED_TESTS_SERIALIZED)
    string(REPLACE "|" ";" _production_isolated_test_patterns
        "${PRODUCTION_ISOLATED_TESTS_SERIALIZED}")
endif()

foreach(_full_name IN LISTS _all_tests)
    if("${_full_name}" MATCHES "ProductionParity")
        # One CTest campaign owns every precision cell for a backend signature
        # in this executable.  GTest still reports each parameterized cell, but
        # keeping them in one process lets the fixture retain the immutable
        # model/PreparedWeightStore authority while rebuilding exact runner,
        # arena, graph, and KV policy state for every cell.
        set(_backend_signature "")
        foreach(_backend IN ITEMS CPU CUDA ROCm)
            if("${_full_name}" MATCHES "${_backend}")
                if(_backend_signature)
                    string(APPEND _backend_signature "_${_backend}")
                else()
                    set(_backend_signature "${_backend}")
                endif()
            endif()
        endforeach()
        if(NOT _backend_signature)
            message(FATAL_ERROR
                "ProductionParity case has no backend identity: ${_full_name}")
        endif()

        # A process campaign has one exact MPI world size.  Parameterized
        # binaries can contain both 2xMPI and 4xMPI cells; grouping those under
        # the target-level default silently launches at least one cell with the
        # wrong topology.  An explicit `<N>xMPI` config suffix is authoritative,
        # while cases without one retain the target's declared MPI_PROCS.
        set(_campaign_mpi_procs "${MPI_PROCS}")
        if("${_full_name}" MATCHES "([0-9]+)xMPI")
            set(_campaign_mpi_procs "${CMAKE_MATCH_1}")
        endif()
        # A few production cells exercise a complete MPI application lifecycle
        # that cannot validly follow another runner teardown in the same world.
        # Give those explicitly declared cells a unique campaign key. The hash
        # is derived from the exact GTest name, so one fragment matching several
        # cases still creates one fresh MPI world per case.
        set(_campaign_variant "")
        foreach(_isolation_pattern IN LISTS _production_isolated_test_patterns)
            if(_isolation_pattern STREQUAL "")
                continue()
            endif()
            string(FIND "${_full_name}" "${_isolation_pattern}"
                _isolation_match)
            if(NOT _isolation_match EQUAL -1)
                string(MAKE_C_IDENTIFIER "${_isolation_pattern}"
                    _isolation_label)
                string(SHA256 _isolation_hash "${_full_name}")
                string(SUBSTRING "${_isolation_hash}" 0 12
                    _isolation_hash_short)
                set(_campaign_variant
                    "Isolated_${_isolation_label}_${_isolation_hash_short}")
                break()
            endif()
        endforeach()
        set(_campaign_key "${_backend_signature}__MPI_${_campaign_mpi_procs}")
        if(_campaign_variant)
            string(APPEND _campaign_key "__${_campaign_variant}")
        endif()
        string(MAKE_C_IDENTIFIER "${_campaign_key}" _signature_id)
        set(_campaign_tests_var "_production_campaign_tests_${_signature_id}")
        set(_campaign_backend_var "_production_campaign_backend_${_signature_id}")
        set(_campaign_mpi_var "_production_campaign_mpi_${_signature_id}")
        set(_campaign_variant_var "_production_campaign_variant_${_signature_id}")
        set(${_campaign_backend_var} "${_backend_signature}")
        set(${_campaign_mpi_var} "${_campaign_mpi_procs}")
        set(${_campaign_variant_var} "${_campaign_variant}")
        list(APPEND ${_campaign_tests_var} "${_full_name}")
        list(FIND _production_campaign_keys
            "${_campaign_key}" _signature_index)
        if(_signature_index EQUAL -1)
            list(APPEND _production_campaign_keys "${_campaign_key}")
        endif()
        continue()
    endif()

    # Sanitize the full gtest name into a valid CTest name suffix
    # Replace . / with _
    string(REPLACE "." "_" _sanitized "${_full_name}")
    string(REPLACE "/" "_" _sanitized "${_sanitized}")
    set(_ctest_name "${TEST_PREFIX}_${_sanitized}")

    set(_test_mpi_cmd ${_mpi_cmd})
    foreach(_export IN LISTS _graph_native_capture_env)
        # mpirun must explicitly forward graph admission settings to every
        # participant; CTest ENVIRONMENT alone is not reliable for OpenMPI.
        list(APPEND _test_mpi_cmd "-x" "${_export}")
    endforeach()
    if("${_full_name}" MATCHES "Profiler")
        # OpenMPI does not always forward CTest-set environment variables to
        # launched ranks unless they are explicitly exported. These legacy
        # profiler-row parity tests intentionally request the hand-instrumented
        # table instead of the graph-safe PerfStats compatibility alias.
        list(APPEND _test_mpi_cmd "-x" "LLAMINAR_PROFILE_KERNELS=1")
    endif()

    # Build the full command: mpirun ... <binary> --gtest_filter=<full_name>
    set(_cmd ${_test_mpi_cmd} "${TEST_EXECUTABLE}" "--gtest_filter=${_full_name}")

    # Convert command list to space-separated string for add_test
    # We need to properly quote each argument
    set(_cmd_str "")
    foreach(_arg IN LISTS _cmd)
        if(_cmd_str)
            string(APPEND _cmd_str " \"${_arg}\"")
        else()
            set(_cmd_str "\"${_arg}\"")
        endif()
    endforeach()

    string(APPEND _output "add_test(\"${_ctest_name}\" ${_cmd_str})\n")

    set(_test_labels ${LABELS})
    if("${_full_name}" MATCHES "Profiler")
        list(FIND _test_labels "Profiler" _profiler_label_index)
        if(_profiler_label_index EQUAL -1)
            list(APPEND _test_labels "Profiler")
        endif()
    endif()
    list(REMOVE_DUPLICATES _test_labels)
    string(JOIN ";" _labels_joined ${_test_labels})

    set(_test_env_vars ${_env_vars})
    if("${_full_name}" MATCHES "Profiler")
        list(APPEND _test_env_vars "LLAMINAR_PROFILE_KERNELS=1")
    endif()
    string(JOIN ";" _env_joined ${_test_env_vars})

    string(APPEND _output "set_tests_properties(\"${_ctest_name}\" PROPERTIES\n")
    string(APPEND _output "    FIXTURES_REQUIRED \"V2_Models\"\n")
    string(APPEND _output "    LABELS \"${_labels_joined}\"\n")
    string(APPEND _output "    ENVIRONMENT \"${_env_joined}\"\n")
    string(APPEND _output "    WORKING_DIRECTORY \"${WORKING_DIR}\"\n")
    string(APPEND _output "    PROCESSORS 4\n")
    if(DEFINED TIMEOUT AND NOT "${TIMEOUT}" STREQUAL "")
        string(APPEND _output "    TIMEOUT \"${TIMEOUT}\"\n")
    endif()
    # Serialize with ALL integration tests (same lock as add_v2_mpi_test).
    # Parity tests load large models and consume significant memory/GPU resources.
    # Using the same lock as regular integration tests prevents any concurrent execution.
    string(APPEND _output "    RESOURCE_LOCK \"Integration_Serial\"\n")
    string(APPEND _output "    RUN_SERIAL TRUE\n")
    string(APPEND _output ")\n\n")
endforeach()

if(_production_campaign_keys AND NOT _all_production_model_files)
    message(FATAL_ERROR
        "ProductionParity cases in ${TEST_EXECUTABLE} have no declared MODEL_FILES")
endif()

# Register one process-resident production campaign per backend signature.
# Each campaign is an atomic scheduling/resource unit, not an independent SLA
# unit. scripts/ci/run_production_parity_campaigns.py admits every registered
# campaign while measuring one shared 3,600-second target for the complete
# matrix. Crossing the target never prevents a campaign from running.
foreach(_campaign_key IN LISTS _production_campaign_keys)
    string(MAKE_C_IDENTIFIER "${_campaign_key}" _signature_id)
    set(_campaign_tests_var "_production_campaign_tests_${_signature_id}")
    set(_campaign_backend_var "_production_campaign_backend_${_signature_id}")
    set(_campaign_mpi_var "_production_campaign_mpi_${_signature_id}")
    set(_campaign_variant_var "_production_campaign_variant_${_signature_id}")
    set(_backend_signature "${${_campaign_backend_var}}")
    set(_campaign_mpi_procs "${${_campaign_mpi_var}}")
    set(_campaign_variant "${${_campaign_variant_var}}")
    set(_campaign_model_files ${_production_model_files_COMMON})
    foreach(_backend IN ITEMS CPU CUDA ROCm)
        if("${_backend_signature}" MATCHES "${_backend}")
            string(TOUPPER "${_backend}" _backend_upper)
            list(APPEND _campaign_model_files
                ${_production_model_files_${_backend_upper}})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _campaign_model_files)
    if(NOT _campaign_model_files)
        message(FATAL_ERROR
            "Production campaign ${_campaign_key} has no declared MODEL_FILES")
    endif()
    string(JOIN "|" _campaign_model_manifest ${_campaign_model_files})
    string(REPLACE ";" ":" _gtest_filter "${${_campaign_tests_var}}")
    set(_campaign_test_prefix "${TEST_PREFIX}")
    if(_campaign_variant)
        string(APPEND _campaign_test_prefix "_${_campaign_variant}")
    endif()
    if(_campaign_mpi_procs EQUAL MPI_PROCS)
        set(_ctest_name
            "${_campaign_test_prefix}_ProductionCampaign_${_backend_signature}_ALL_PRECISIONS")
    else()
        set(_ctest_name
            "${_campaign_test_prefix}_ProductionCampaign_${_backend_signature}_MPI_${_campaign_mpi_procs}_ALL_PRECISIONS")
    endif()

    # A campaign's backend signature is also its physical device claim.  Do
    # not let a CPU-only process create CUDA/HIP driver contexts, or let one
    # homogeneous GPU campaign enumerate the other vendor while the aggregate
    # scheduler is running that backend concurrently.  Hybrid signatures keep
    # every backend they name, so topology discovery remains production-real.
    set(_campaign_runtime_env
        "LLAMINAR_LOG_LEVEL=INFO"
        "LLAMINAR_PRODUCTION_PARITY=1"
        "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN=1"
        "LLAMINAR_PRODUCTION_PARITY_TARGET_SECONDS=3600"
        "LLAMINAR_GPU_GRAPHS=1"
        "LLAMINAR_PREFILL_GRAPH_BUCKETS=1"
        "LLAMINAR_PREFILL_GRAPH_REQUIRED=1"
        "LLAMINAR_PERF_STATS_SUMMARY=1"
        "LLAMINAR_PERF_STATS_FILTER=${PRODUCTION_PERF_STATS_FILTER}"
        "LLAMINAR_PRODUCTION_PARITY_DECLARED_MODELS=${_campaign_model_manifest}")
    if(_backend_signature STREQUAL "CPU")
        list(APPEND _campaign_runtime_env
            "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1")
    else()
        if(NOT "${_backend_signature}" MATCHES "CUDA")
            list(APPEND _campaign_runtime_env
                "LLAMINAR_SKIP_CUDA_STARTUP=1")
        endif()
        if(NOT "${_backend_signature}" MATCHES "ROCm")
            list(APPEND _campaign_runtime_env
                "LLAMINAR_SKIP_ROCM_STARTUP=1")
        endif()
    endif()

    set(_campaign_mpi_cmd
        "mpirun"
        "-np" "${_campaign_mpi_procs}"
        "--bind-to" "socket"
        "--map-by" "socket"
        "--mca" "mpi_leave_pinned" "1"
        "--mca" "btl_vader_single_copy_mechanism" "none"
        "--mca" "orte_allowed_exit_without_sync" "1")
    if(NUM_SOCKETS AND _campaign_mpi_procs GREATER NUM_SOCKETS)
        list(APPEND _campaign_mpi_cmd "--oversubscribe")
    endif()
    foreach(_export IN LISTS _campaign_runtime_env)
        list(APPEND _campaign_mpi_cmd "-x" "${_export}")
    endforeach()
    set(_cmd
        ${_campaign_mpi_cmd}
        "${TEST_EXECUTABLE}"
        "--gtest_filter=${_gtest_filter}")

    set(_cmd_str "")
    foreach(_arg IN LISTS _cmd)
        if(_cmd_str)
            string(APPEND _cmd_str " \"${_arg}\"")
        else()
            set(_cmd_str "\"${_arg}\"")
        endif()
    endforeach()
    string(APPEND _output "add_test(\"${_ctest_name}\" ${_cmd_str})\n")

    set(_campaign_labels ${LABELS})
    list(REMOVE_ITEM _campaign_labels "CPU" "CUDA" "ROCm")
    foreach(_backend IN ITEMS CPU CUDA ROCm)
        if("${_backend_signature}" MATCHES "${_backend}")
            list(APPEND _campaign_labels "${_backend}")
        endif()
    endforeach()
    list(APPEND _campaign_labels
        "ProductionPath"
        "FullModel"
        "Campaign"
        "AllPrecisions"
        "WholeMatrixOneHourTarget"
        "MPI${_campaign_mpi_procs}")
    if(_campaign_variant)
        list(APPEND _campaign_labels "FreshMPIWorld")
    endif()
    list(REMOVE_DUPLICATES _campaign_labels)
    string(JOIN ";" _labels_joined ${_campaign_labels})

    set(_campaign_env_vars ${_env_vars})
    # Integration tests default to DEBUG, but a complete real-weight campaign
    # emits canonical CSV/PerfStats evidence and must not benchmark synchronous
    # debug-log traffic.  A focused reproduction can still run at DEBUG.
    list(FILTER _campaign_env_vars EXCLUDE REGEX "^LLAMINAR_LOG_LEVEL=")
    list(APPEND _campaign_env_vars ${_campaign_runtime_env})
    string(JOIN ";" _env_joined ${_campaign_env_vars})

    string(APPEND _output "set_tests_properties(\"${_ctest_name}\" PROPERTIES\n")
    string(APPEND _output "    FIXTURES_REQUIRED \"V2_Models\"\n")
    string(APPEND _output "    LABELS \"${_labels_joined}\"\n")
    string(APPEND _output "    ENVIRONMENT \"${_env_joined}\"\n")
    string(APPEND _output "    REQUIRED_FILES \"${_campaign_model_files}\"\n")
    string(APPEND _output "    WORKING_DIRECTORY \"${WORKING_DIR}\"\n")
    string(APPEND _output "    PROCESSORS 4\n")
    string(APPEND _output
        "    TIMEOUT \"${_production_campaign_completion_timeout_seconds}\"\n")
    string(APPEND _output "    RESOURCE_LOCK \"Integration_Serial\"\n")
    string(APPEND _output "    RUN_SERIAL TRUE\n")
    string(APPEND _output ")\n\n")
endforeach()

file(WRITE "${CTEST_FILE}" "${_output}")
message(STATUS "V2ParityTestDiscovery: ${_test_count} tests from ${TEST_PREFIX}")
