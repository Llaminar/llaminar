# @file V2ParityEnvironment.cmake
# @brief Literal before-startup environment declarations for the existing parity discovery owner.
# Entries are NAME=value strings. Pipe, semicolon and line breaks are rejected
# because they delimit the existing CMake registration transport; shell syntax
# otherwise remains literal and is never executed by this module.

# @brief Validate and serialize one complete ordered environment declaration list.
# @param result Parent-scope variable receiving the existing pipe-delimited representation.
# @param ARGN Explicit NAME=value entries; duplicate names reject rather than choose an override.
function(v2_parity_environment_serialize result)
    set(_names "")
    foreach(_declaration IN LISTS ARGN)
        if(NOT "${_declaration}" MATCHES "^([A-Za-z_][A-Za-z0-9_]*)=" OR
           "${_declaration}" MATCHES "[|;\r\n]")
            message(FATAL_ERROR "Parity ENVIRONMENT requires literal NAME=value without pipe, semicolon or line breaks")
        endif()
        string(REGEX MATCH "^[A-Za-z_][A-Za-z0-9_]*" _name "${_declaration}")
        list(FIND _names "${_name}" _duplicate)
        if(NOT _duplicate EQUAL -1)
            message(FATAL_ERROR "Parity ENVIRONMENT repeats ${_name}")
        endif()
        list(APPEND _names "${_name}")
    endforeach()
    string(JOIN "|" _serialized ${ARGN})
    set(${result} "${_serialized}" PARENT_SCOPE)
endfunction()

# @brief Recover and authenticate the same declarations in the POST_BUILD discovery process.
# @param result Parent-scope declaration list.
# @param serialized Configure-owned representation; empty preserves original behavior.
function(v2_parity_environment_deserialize result serialized)
    string(REPLACE "|" ";" _environment "${serialized}")
    if(NOT "${serialized}" STREQUAL "")
        v2_parity_environment_serialize(_checked ${_environment})
    endif()
    set(${result} "${_environment}" PARENT_SCOPE)
endfunction()

# @brief Quote one literal value in a generated CTest CMake string, never a shell command.
# @param result Parent-scope escaped value.
# @param value Exact value; list delimiters are owned separately by the caller.
function(v2_parity_escape_literal result value)
    string(REPLACE "\\" "\\\\" _quoted "${value}")
    string(REPLACE "\"" "\\\"" _quoted "${_quoted}")
    string(REPLACE "$" "\\$" _quoted "${_quoted}")
    set(${result} "${_quoted}" PARENT_SCOPE)
endfunction()
