# @file V2TestHostTopology.cmake
# @brief Admit a checked native test-host topology, never lscpu presentation fields.
#
# One immutable affinity/sysfs record owns socket eligibility and CPU scheduling.
# The current shared MPI/OpenMP environment and scalar CTest conversion can
# express uniform accessible package/core widths only. Preserve original vectors
# and reject an unrepresentable projection rather than silently substituting a
# minimum, maximum, average, first row or invented one-core geometry.
include_guard(GLOBAL)

# @brief Read one strictly positive JSON integer before CMake arithmetic.
# @param json Original immutable snapshot; @param output Destination scalar.
# @param ARGN Structured JSON member path.
function(_v2_topology_positive_integer json output)
    string(JSON _type ERROR_VARIABLE _error TYPE "${json}" ${ARGN})
    if(NOT _error STREQUAL "NOTFOUND" OR NOT _type STREQUAL "NUMBER")
        message(FATAL_ERROR "CPU test-host topology has a missing/non-numeric member: ${ARGN}")
    endif()
    string(JSON _value GET "${json}" ${ARGN})
    if(NOT _value MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "CPU test-host topology requires a positive integer: ${ARGN}")
    endif()
    set(${output} "${_value}" PARENT_SCOPE)
endfunction()

# @brief Bind a checked uniform projection while retaining original per-package evidence.
# @param json Complete canonical observation, produced by CPUHostTopology.py.
# @details No host observation occurs here. Pure fixture callers exercise the
# same boundary without a fake compiler/SDK or substituted discovery command.
function(v2_bind_test_host_topology json)
    _v2_topology_positive_integer("${json}" _schema schema_version)
    if(NOT _schema EQUAL 1)
        message(FATAL_ERROR "Unsupported CPU test-host topology schema ${_schema}")
    endif()
    _v2_topology_positive_integer("${json}" _packages package_count)
    _v2_topology_positive_integer("${json}" _physical physical_core_count)
    _v2_topology_positive_integer("${json}" _logical logical_cpu_count)
    string(JSON _package_type ERROR_VARIABLE _error TYPE "${json}" packages)
    if(NOT _error STREQUAL "NOTFOUND" OR NOT _package_type STREQUAL "ARRAY")
        message(FATAL_ERROR "CPU test-host topology is missing its original package array")
    endif()
    string(JSON _package_count LENGTH "${json}" packages)
    if(NOT _package_count EQUAL _packages)
        message(FATAL_ERROR "CPU test-host topology package evidence/count disagree")
    endif()
    set(_package_cores "")
    set(_physical_sum 0)
    set(_logical_sum 0)
    math(EXPR _last "${_package_count}-1")
    foreach(_index RANGE 0 ${_last})
        _v2_topology_positive_integer("${json}" _cores packages ${_index} physical_cores)
        _v2_topology_positive_integer("${json}" _threads packages ${_index} logical_cpus)
        list(APPEND _package_cores "${_cores}")
        math(EXPR _physical_sum "${_physical_sum}+${_cores}")
        math(EXPR _logical_sum "${_logical_sum}+${_threads}")
    endforeach()
    if(NOT _physical_sum EQUAL _physical OR NOT _logical_sum EQUAL _logical)
        message(FATAL_ERROR "CPU test-host topology package evidence/counts disagree")
    endif()
    string(JSON _width_type ERROR_VARIABLE _error TYPE "${json}" core_thread_widths)
    if(NOT _error STREQUAL "NOTFOUND" OR NOT _width_type STREQUAL "ARRAY")
        message(FATAL_ERROR "CPU test-host topology is missing original core thread widths")
    endif()
    string(JSON _core_count LENGTH "${json}" core_thread_widths)
    if(NOT _core_count EQUAL _physical)
        message(FATAL_ERROR "CPU test-host topology physical-core width evidence/count disagree")
    endif()
    set(_widths "")
    set(_width_sum 0)
    math(EXPR _last "${_core_count}-1")
    foreach(_index RANGE 0 ${_last})
        _v2_topology_positive_integer("${json}" _width core_thread_widths ${_index})
        list(APPEND _widths "${_width}")
        math(EXPR _width_sum "${_width_sum}+${_width}")
    endforeach()
    if(NOT _width_sum EQUAL _logical)
        message(FATAL_ERROR "CPU test-host topology SMT evidence/count disagree")
    endif()
    set(_distinct_cores "${_package_cores}")
    set(_distinct_widths "${_widths}")
    list(REMOVE_DUPLICATES _distinct_cores)
    list(REMOVE_DUPLICATES _distinct_widths)
    list(LENGTH _distinct_cores _core_shapes)
    list(LENGTH _distinct_widths _thread_shapes)
    if(NOT _core_shapes EQUAL 1 OR NOT _thread_shapes EQUAL 1)
        message(FATAL_ERROR
            "CPU test-host topology cannot be represented by the current shared MPI/OpenMP budget: "
            "accessible cores/package=[${_package_cores}], available threads/core=[${_widths}]. "
            "Implement rank-local budgets and physical-core reservations before admitting this geometry.")
    endif()
    list(GET _distinct_cores 0 _cores_per_socket)
    list(GET _distinct_widths 0 _threads_per_core)
    set(V2_TEST_SOCKETS "${_packages}" PARENT_SCOPE)
    set(V2_TEST_CORES_PER_SOCKET "${_cores_per_socket}" PARENT_SCOPE)
    set(V2_TEST_THREADS_PER_CORE "${_threads_per_core}" PARENT_SCOPE)
    set(V2_TEST_HOST_PHYSICAL_CORES "${_physical}" PARENT_SCOPE)
    set(V2_TEST_HOST_LOGICAL_CPUS "${_logical}" PARENT_SCOPE)
endfunction()

# @brief Withdraw old evidence before any new admission or producer can fail.
# @param output Build-owned receipt path for this attempt.
# @details Removal does not follow a file symlink. Unexpected directories or an
# unretirable path are fatal; no successful receipt may survive a rejected scope.
function(_v2_retire_test_host_topology output)
    set(_receipt "${CMAKE_CURRENT_BINARY_DIR}/v2-test-host-topology.json")
    foreach(_owned IN ITEMS "${_receipt}" "${_receipt}.tmp")
        file(REMOVE "${_owned}")
        if(EXISTS "${_owned}" OR IS_SYMLINK "${_owned}")
            message(FATAL_ERROR "Cannot retire build-owned CPU test-host topology evidence: ${_owned}")
        endif()
    endforeach()
    set(${output} "${_receipt}" PARENT_SCOPE)
endfunction()

# @brief Admit and atomically publish one complete immutable topology observation.
# @param json Original observed record; pure metadata fixtures use this same owner.
# @details Retirement precedes projection. Temporary bytes are never a published
# success; only a complete checked record becomes the receipt via same-directory
# rename. This is cold evidence publication, not a second live inventory.
function(v2_admit_test_host_topology json)
    _v2_retire_test_host_topology(_receipt)
    v2_bind_test_host_topology("${json}")
    file(WRITE "${_receipt}.tmp" "${json}\n")
    file(RENAME "${_receipt}.tmp" "${_receipt}" RESULT _published)
    if(NOT _published STREQUAL "0")
        message(FATAL_ERROR "Cannot publish admitted CPU test-host topology: ${_published}")
    endif()
    foreach(_name IN ITEMS V2_TEST_SOCKETS V2_TEST_CORES_PER_SOCKET V2_TEST_THREADS_PER_CORE
            V2_TEST_HOST_PHYSICAL_CORES V2_TEST_HOST_LOGICAL_CPUS)
        set(${_name} "${${_name}}" PARENT_SCOPE)
    endforeach()
    set(V2_TEST_TOPOLOGY_RECEIPT "${_receipt}" PARENT_SCOPE)
endfunction()

# @brief Observe the actual configure host and publish one immutable topology receipt.
# @details Only native Linux affinity/sysfs observation is admitted. No user flag
# selects another host, fake SDK, presentation-parser rescue or default geometry.
function(v2_observe_test_host_topology)
    _v2_retire_test_host_topology(_receipt)
    if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "Native test-host CPU topology requires Linux affinity/sysfs observation")
    endif()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CPUHostTopology.py"
        RESULT_VARIABLE _status OUTPUT_VARIABLE _json ERROR_VARIABLE _diagnostic
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _status STREQUAL "0")
        message(FATAL_ERROR "Native CPU test-host topology admission failed (${_status}): ${_diagnostic}")
    endif()
    v2_admit_test_host_topology("${_json}")
    foreach(_name IN ITEMS V2_TEST_SOCKETS V2_TEST_CORES_PER_SOCKET V2_TEST_THREADS_PER_CORE
            V2_TEST_HOST_PHYSICAL_CORES V2_TEST_HOST_LOGICAL_CPUS V2_TEST_TOPOLOGY_RECEIPT)
        set(${_name} "${${_name}}" PARENT_SCOPE)
    endforeach()
endfunction()
