# @file V2CPUSecondaryEnvelope.cmake
# @brief Declare test-only CPU code-generation envelopes below the canonical build envelope.
#
# One build tree owns one canonical CPU envelope (CPUInstructionSet.cmake). A
# reduced-ISA numerical test target (for example the scalar or AVX2 copy of a
# proof inside an AVX512 build) must not inherit that envelope and then lower it
# with "-mno-*" options: its declared LLAMINAR_COMPILED_WITH_* values would
# contradict its compiler features, which CPUFeatures.h rejects at compile time.
#
# Instead, such targets live in a dedicated secondary-envelope directory. That
# directory removes the exact canonical contribution from its own scope and each
# target declares one complete, coherent envelope: compiler flags plus every
# LLAMINAR_COMPILED_WITH_* value. The x86 AVX2/AVX512 envelopes come from the
# canonical resolver; SCALAR is the explicit baseline x86-64 test envelope.
#
# Every declared target is recorded for cpu_host_compile_contract.py, which
# verifies its exact object rows against that target's own envelope while all
# other compile-database rows stay bound to the canonical envelope. Declaration
# after the contract test has consumed the inventory is a configure error.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/CPUInstructionSet.cmake")

# @brief Resolve one secondary test envelope for the target processor.
# @param processor Target processor (LLAMINAR_CPU_TARGET_PROCESSOR).
# @param envelope SCALAR (baseline x86-64 only) or any ISA the canonical resolver accepts.
# @param flags_output Caller variable receiving space-separated compiler flags.
# @param definitions_output Caller variable receiving the complete LLAMINAR_COMPILED_WITH_* list.
function(v2_cpu_secondary_envelope processor envelope flags_output definitions_output)
    string(TOUPPER "${envelope}" isa)
    if(isa STREQUAL "SCALAR")
        llaminar_default_cpu_isa("${processor}" default_isa)
        if(default_isa STREQUAL "NEON")
            message(FATAL_ERROR "SCALAR secondary envelope is x86-64 only; AArch64 always carries NEON")
        endif()
        # Baseline x86-64 (SSE2) code generation, independent of the compiler's
        # configured default -march, with every optional envelope feature absent.
        set(flags "-march=x86-64")
        set(definitions LLAMINAR_COMPILED_WITH_AVX2=0 LLAMINAR_COMPILED_WITH_AVX512=0
            LLAMINAR_COMPILED_WITH_NEON=0 LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0
            LLAMINAR_COMPILED_WITH_NEON_I8MM=0 LLAMINAR_COMPILED_WITH_FP16FML=0)
    else()
        llaminar_cpu_instruction_set("${processor}" "${isa}" flags definitions)
    endif()
    set(${flags_output} "${flags}" PARENT_SCOPE)
    set(${definitions_output} "${definitions}" PARENT_SCOPE)
endfunction()

# @brief Return whether one compiler option selects CPU ISA features.
# @param option Compiler option token.
# @param output Caller variable receiving TRUE or FALSE.
#
# Mirrors cpu_host_compile_contract.py: address width and code model do not
# select optional ISA features.
function(v2_cpu_isa_option option output)
    if(option MATCHES "^-m" AND NOT option STREQUAL "-m64" AND NOT option MATCHES "^-mcmodel=")
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

# @brief Remove the canonical envelope from the calling directory scope.
#
# Call once at the top of a dedicated secondary-envelope directory. The
# canonical definitions and flags were contributed exactly once by
# llaminar_configure_cpu_instruction_set(); removal is exact, and any residual
# envelope definition or ISA option inherited from elsewhere is fatal rather
# than silently combined with a secondary envelope.
macro(v2_enter_cpu_secondary_envelope_scope)
    if(NOT LLAMINAR_CPU_COMPILE_FLAGS OR NOT LLAMINAR_CPU_COMPILE_DEFINITIONS)
        message(FATAL_ERROR "Secondary CPU envelopes require the configured canonical envelope")
    endif()
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        message(FATAL_ERROR "A secondary CPU envelope scope must be a dedicated subdirectory")
    endif()
    get_directory_property(_v2_secondary_definitions COMPILE_DEFINITIONS)
    foreach(_v2_canonical_definition IN LISTS LLAMINAR_CPU_COMPILE_DEFINITIONS)
        list(FIND _v2_secondary_definitions "${_v2_canonical_definition}" _v2_definition_index)
        if(_v2_definition_index LESS 0)
            message(FATAL_ERROR "Canonical CPU definition was not inherited: ${_v2_canonical_definition}")
        endif()
        list(REMOVE_ITEM _v2_secondary_definitions "${_v2_canonical_definition}")
    endforeach()
    foreach(_v2_residual_definition IN LISTS _v2_secondary_definitions)
        if(_v2_residual_definition MATCHES "^LLAMINAR_COMPILED_WITH_")
            message(FATAL_ERROR "Residual CPU envelope definition in secondary scope: ${_v2_residual_definition}")
        endif()
    endforeach()
    set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${_v2_secondary_definitions}")
    foreach(_v2_language C CXX)
        string(FIND "${CMAKE_${_v2_language}_FLAGS}" " ${LLAMINAR_CPU_COMPILE_FLAGS}" _v2_flags_first)
        string(FIND "${CMAKE_${_v2_language}_FLAGS}" " ${LLAMINAR_CPU_COMPILE_FLAGS}" _v2_flags_last REVERSE)
        if(_v2_flags_first LESS 0 OR NOT _v2_flags_first EQUAL _v2_flags_last)
            message(FATAL_ERROR "CMAKE_${_v2_language}_FLAGS does not carry the canonical CPU flags exactly once")
        endif()
        string(REPLACE " ${LLAMINAR_CPU_COMPILE_FLAGS}" "" CMAKE_${_v2_language}_FLAGS
            "${CMAKE_${_v2_language}_FLAGS}")
    endforeach()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _v2_secondary_config)
    get_directory_property(_v2_secondary_options COMPILE_OPTIONS)
    separate_arguments(_v2_secondary_flag_tokens NATIVE_COMMAND
        "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_v2_secondary_config}}")
    foreach(_v2_secondary_token IN LISTS _v2_secondary_flag_tokens _v2_secondary_options)
        v2_cpu_isa_option("${_v2_secondary_token}" _v2_secondary_isa)
        if(_v2_secondary_isa)
            message(FATAL_ERROR "Residual ISA option in secondary CPU envelope scope: ${_v2_secondary_token}")
        endif()
    endforeach()
    set_property(DIRECTORY PROPERTY V2_CPU_SECONDARY_ENVELOPE_SCOPE TRUE)
endmacro()

# @brief Bind an existing target in the secondary scope to one declared envelope.
# @param target Executable target created in the calling secondary-envelope directory.
# @param envelope Envelope name accepted by v2_cpu_secondary_envelope().
#
# A target omitted by the build's test selection is not declared: there is no
# compile-database row to bind.
function(v2_declare_cpu_secondary_envelope_target target envelope)
    get_property(scope DIRECTORY PROPERTY V2_CPU_SECONDARY_ENVELOPE_SCOPE)
    if(NOT scope)
        message(FATAL_ERROR "${target}: secondary CPU envelopes require v2_enter_cpu_secondary_envelope_scope()")
    endif()
    get_property(sealed GLOBAL PROPERTY V2_CPU_SECONDARY_ENVELOPES_SEALED)
    if(sealed)
        message(FATAL_ERROR "${target}: secondary CPU envelope declared after the host compile contract consumed the inventory")
    endif()
    if(NOT TARGET ${target})
        return()
    endif()
    get_target_property(target_directory ${target} SOURCE_DIR)
    if(NOT target_directory STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        message(FATAL_ERROR "${target}: a secondary CPU envelope target must be created in its secondary scope")
    endif()
    v2_cpu_secondary_envelope("${LLAMINAR_CPU_TARGET_PROCESSOR}" "${envelope}" flags definitions)
    separate_arguments(options NATIVE_COMMAND "${flags}")
    target_compile_options(${target} PRIVATE ${options})
    target_compile_definitions(${target} PRIVATE ${definitions})
    set_target_properties(${target} PROPERTIES
        V2_CPU_SECONDARY_ENVELOPE_FLAGS "${flags}"
        V2_CPU_SECONDARY_ENVELOPE_DEFINITIONS "${definitions}")
    set_property(GLOBAL APPEND PROPERTY V2_CPU_SECONDARY_ENVELOPE_TARGETS ${target})
endfunction()

# @brief Produce typed host compile contract arguments for every declared secondary envelope.
# @param output Caller variable receiving the argument list.
#
# Each target contributes its name, flags, complete definitions and exact object
# paths. Consuming the inventory seals it against later declarations.
function(v2_cpu_secondary_envelope_contract_arguments output)
    get_property(targets GLOBAL PROPERTY V2_CPU_SECONDARY_ENVELOPE_TARGETS)
    set(arguments)
    foreach(target IN LISTS targets)
        get_target_property(flags ${target} V2_CPU_SECONDARY_ENVELOPE_FLAGS)
        get_target_property(definitions ${target} V2_CPU_SECONDARY_ENVELOPE_DEFINITIONS)
        list(APPEND arguments "--secondary-envelope=${target}"
            "--secondary-flags=${target}=${flags}")
        foreach(definition IN LISTS definitions)
            list(APPEND arguments "--secondary-definition=${target}=${definition}")
        endforeach()
        list(APPEND arguments "--secondary-objects=${target}=$<JOIN:$<TARGET_OBJECTS:${target}>,|>")
    endforeach()
    set_property(GLOBAL PROPERTY V2_CPU_SECONDARY_ENVELOPES_SEALED TRUE)
    set(${output} "${arguments}" PARENT_SCOPE)
endfunction()
