# @file V2CPUContractionProof.cmake
# @brief Register the bounded x86 contribution-helper contraction witness against production codegen.
#
# The witness compiles the real x86 single-scale contribution helpers with the
# actual ordered compile command of one production core translation unit,
# borrowed from CMake's compilation database. Only the named ISA envelope is
# substituted (each x86 envelope comes from llaminar_cpu_instruction_set), and a
# separately recorded stricter setting appends -ffp-contract=fast. The standard,
# extensions, PIC, OpenMP, configuration optimisation, target options and
# definitions are therefore exactly the core's; nothing is reconstructed here.
#
# Configure records the compiler identity (resolved path, SHA256 and --version)
# in a plan beside the receipt. The builder refuses a compiler that changed
# after configuration, and installed verification compares the receipt with
# that recorded identity without executing any tool.
#
# Qualification is per configuration. The unguarded negative control can fuse
# only when the configuration optimises, so only single-config trees whose
# build type is listed in V2_CPU_CONTRACTION_PROOF_BUILD_TYPES register the
# preflight proof. Any other single configuration, and every multi-config
# generator (one receipt binds one configuration's command), is reported as
# explicitly UNQUALIFIED and registers no proof; configure never fails for it.
#
# The generated-command fixtures (fixture projects configured with this module,
# the canonical ISA authority and the configured compiler) are build work in
# every configuration: a v2_unit_gate dependency writes a source- and
# compiler-bound receipt, and the Unit test verifies it without any tool.
#
# Inputs owned by the including directory: LLAMINAR_CPU_* from the canonical
# ISA authority, V2_CPU_TARGET_X86, Python3_EXECUTABLE and CMAKE_OBJDUMP.
# V2_CPU_CONTRACTION_CODEGEN_TARGET/SOURCE default to the production core and
# its routed-expert kernel TU; the focused configure fixture overrides them.
include_guard(GLOBAL)

get_filename_component(V2_CPU_CONTRACTION_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# Optimising configurations whose actual flags can make the unguarded negative
# control fuse. Debug compiles at -O0: no helper is inlined, nothing fuses, and
# a pass would be fabricated. It stays explicitly unqualified, never skip-passed.
set(V2_CPU_CONTRACTION_PROOF_BUILD_TYPES Release Integration)
# Every x86 envelope the canonical resolver admits; the fixture source declares
# its AVX512 witnesses only under __AVX512F__.
set(V2_CPU_CONTRACTION_ENVELOPES AVX2 AVX512)
if(NOT DEFINED V2_CPU_CONTRACTION_CODEGEN_TARGET)
    set(V2_CPU_CONTRACTION_CODEGEN_TARGET llaminar2_core)
endif()
if(NOT DEFINED V2_CPU_CONTRACTION_CODEGEN_SOURCE)
    set(V2_CPU_CONTRACTION_CODEGEN_SOURCE "${V2_CPU_CONTRACTION_ROOT}/src/v2/kernels/cpu/moe/CPUMoEKernel.cpp")
endif()
set(V2_CPU_CONTRACTION_SCRIPTS "${V2_CPU_CONTRACTION_ROOT}/tests/v2/scripts")

# @brief Quote arbitrary text as one JSON string literal.
# @param output Caller variable receiving the quoted literal.
# @param text Text such as a compiler --version banner.
function(v2_cpu_contraction_json_string output text)
    string(REPLACE "\\" "\\\\" text "${text}")
    string(REPLACE "\"" "\\\"" text "${text}")
    string(REPLACE "\n" "\\n" text "${text}")
    string(REPLACE "\r" "\\r" text "${text}")
    string(REPLACE "\t" "\\t" text "${text}")
    set(${output} "\"${text}\"" PARENT_SCOPE)
endfunction()

# @brief Quote a CMake list as one JSON array of strings.
# @param output Caller variable receiving the array literal.
# @param ARGN List items.
function(v2_cpu_contraction_json_array output)
    set(items)
    foreach(item IN LISTS ARGN)
        v2_cpu_contraction_json_string(quoted "${item}")
        list(APPEND items "${quoted}")
    endforeach()
    list(JOIN items "," joined)
    set(${output} "[${joined}]" PARENT_SCOPE)
endfunction()

# @brief Write a configure artifact only when its content changes, so an
# unchanged reconfigure does not rebuild the receipts that depend on it.
# @param path Output file.
# @param text Complete content.
function(v2_cpu_contraction_write_if_changed path text)
    set(previous "")
    if(EXISTS "${path}")
        file(READ "${path}" previous)
    endif()
    if(NOT previous STREQUAL text)
        file(WRITE "${path}" "${text}")
    endif()
endfunction()

set(V2_CPU_CONTRACTION_DIR "${CMAKE_CURRENT_BINARY_DIR}/cpu_contraction")
set(V2_CPU_CONTRACTION_IDENTITY "${V2_CPU_CONTRACTION_DIR}/compiler.json")
set(V2_CPU_CONTRACTION_GENERATED "${V2_CPU_CONTRACTION_DIR}/generated_commands.json")

# Configure-recorded compiler identity. A same-path upgrade changes the SHA256
# or banner; the builders then refuse it and retained receipts become stale.
get_filename_component(V2_CPU_CONTRACTION_COMPILER_REAL "${CMAKE_CXX_COMPILER}" REALPATH)
file(SHA256 "${V2_CPU_CONTRACTION_COMPILER_REAL}" V2_CPU_CONTRACTION_COMPILER_SHA256)
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --version
    RESULT_VARIABLE V2_CPU_CONTRACTION_VERSION_RESULT
    OUTPUT_VARIABLE V2_CPU_CONTRACTION_COMPILER_VERSION
    ERROR_VARIABLE V2_CPU_CONTRACTION_VERSION_ERROR)
if(NOT V2_CPU_CONTRACTION_VERSION_RESULT EQUAL 0)
    message(FATAL_ERROR "x86 contraction proof cannot record ${CMAKE_CXX_COMPILER} --version: ${V2_CPU_CONTRACTION_VERSION_ERROR}")
endif()
v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_COMPILER "${CMAKE_CXX_COMPILER}")
v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_REAL "${V2_CPU_CONTRACTION_COMPILER_REAL}")
v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_VERSION "${V2_CPU_CONTRACTION_COMPILER_VERSION}")
set(V2_CPU_CONTRACTION_J_IDENTITY "{\"path\":${V2_CPU_CONTRACTION_J_COMPILER},\
\"realpath\":${V2_CPU_CONTRACTION_J_REAL},\"sha256\":\"${V2_CPU_CONTRACTION_COMPILER_SHA256}\",\
\"version\":${V2_CPU_CONTRACTION_J_VERSION}}")
v2_cpu_contraction_write_if_changed("${V2_CPU_CONTRACTION_IDENTITY}" "${V2_CPU_CONTRACTION_J_IDENTITY}\n")

get_property(V2_CPU_CONTRACTION_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(V2_CPU_CONTRACTION_QUALIFIED OFF)
if(NOT V2_CPU_TARGET_X86)
    # The witness compiles x86 helpers; other targets have their own proofs.
elseif(V2_CPU_CONTRACTION_MULTI_CONFIG)
    message(STATUS "V2: x86 contribution contraction proof UNQUALIFIED for multi-config generator "
        "'${CMAKE_GENERATOR}' (the proof binds one configuration's command); registered only for "
        "single-config ${V2_CPU_CONTRACTION_PROOF_BUILD_TYPES}")
elseif(CMAKE_BUILD_TYPE IN_LIST V2_CPU_CONTRACTION_PROOF_BUILD_TYPES)
    set(V2_CPU_CONTRACTION_QUALIFIED ON)
else()
    set(V2_CPU_CONTRACTION_CONFIGURATION "${CMAKE_BUILD_TYPE}")
    if(NOT V2_CPU_CONTRACTION_CONFIGURATION)
        set(V2_CPU_CONTRACTION_CONFIGURATION "<none>")
    endif()
    message(STATUS "V2: x86 contribution contraction proof UNQUALIFIED for build type "
        "'${V2_CPU_CONTRACTION_CONFIGURATION}' (non-optimising); registered only for "
        "${V2_CPU_CONTRACTION_PROOF_BUILD_TYPES}")
endif()

if(V2_CPU_CONTRACTION_QUALIFIED)
    if(NOT TARGET ${V2_CPU_CONTRACTION_CODEGEN_TARGET})
        message(FATAL_ERROR "x86 contraction proof requires its codegen target ${V2_CPU_CONTRACTION_CODEGEN_TARGET}")
    endif()
    if(NOT CMAKE_OBJDUMP)
        message(FATAL_ERROR "x86 contraction proof requires CMAKE_OBJDUMP (GNU binutils or LLVM)")
    endif()
    # The witness borrows this target's command from the compilation database;
    # exporting it does not change any compile flag.
    set_property(TARGET ${V2_CPU_CONTRACTION_CODEGEN_TARGET} PROPERTY EXPORT_COMPILE_COMMANDS ON)
    set(V2_CPU_CONTRACTION_PLAN "${V2_CPU_CONTRACTION_DIR}/plan.json")
    set(V2_CPU_CONTRACTION_PROOF "${V2_CPU_CONTRACTION_DIR}/proof.json")
    set(V2_CPU_CONTRACTION_DATABASE "${CMAKE_BINARY_DIR}/compile_commands.json")

    v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_ISA "${LLAMINAR_CPU_ISA_UPPER}")
    v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_FLAGS "${LLAMINAR_CPU_COMPILE_FLAGS}")
    v2_cpu_contraction_json_array(V2_CPU_CONTRACTION_J_DEFINITIONS ${LLAMINAR_CPU_COMPILE_DEFINITIONS})
    v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_BUILD "${CMAKE_BUILD_TYPE}")
    v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_SOURCE "${V2_CPU_CONTRACTION_CODEGEN_SOURCE}")
    v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_DATABASE "${V2_CPU_CONTRACTION_DATABASE}")
    v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_OBJDUMP "${CMAKE_OBJDUMP}")
    set(V2_CPU_CONTRACTION_J_ENVELOPES "{}")
    foreach(V2_CPU_CONTRACTION_ENVELOPE IN LISTS V2_CPU_CONTRACTION_ENVELOPES)
        llaminar_cpu_instruction_set("${LLAMINAR_CPU_TARGET_PROCESSOR}" "${V2_CPU_CONTRACTION_ENVELOPE}"
            V2_CPU_CONTRACTION_ENVELOPE_FLAGS V2_CPU_CONTRACTION_ENVELOPE_DEFINITIONS)
        v2_cpu_contraction_json_string(V2_CPU_CONTRACTION_J_ENVELOPE_FLAGS "${V2_CPU_CONTRACTION_ENVELOPE_FLAGS}")
        v2_cpu_contraction_json_array(V2_CPU_CONTRACTION_J_ENVELOPE_DEFINITIONS
            ${V2_CPU_CONTRACTION_ENVELOPE_DEFINITIONS})
        string(JSON V2_CPU_CONTRACTION_J_ENVELOPES SET "${V2_CPU_CONTRACTION_J_ENVELOPES}"
            "${V2_CPU_CONTRACTION_ENVELOPE}"
            "{\"flags\":${V2_CPU_CONTRACTION_J_ENVELOPE_FLAGS},\"definitions\":${V2_CPU_CONTRACTION_J_ENVELOPE_DEFINITIONS}}")
    endforeach()
    set(V2_CPU_CONTRACTION_PLAN_TEXT "{\"schema\":1,\"build_type\":${V2_CPU_CONTRACTION_J_BUILD},\
\"compiler\":${V2_CPU_CONTRACTION_J_IDENTITY},\
\"configured\":{\"isa\":${V2_CPU_CONTRACTION_J_ISA},\"flags\":${V2_CPU_CONTRACTION_J_FLAGS},\
\"definitions\":${V2_CPU_CONTRACTION_J_DEFINITIONS}},\"envelopes\":${V2_CPU_CONTRACTION_J_ENVELOPES},\
\"codegen_source\":${V2_CPU_CONTRACTION_J_SOURCE},\"compile_database\":${V2_CPU_CONTRACTION_J_DATABASE},\
\"objdump\":${V2_CPU_CONTRACTION_J_OBJDUMP},\"stricter_setting\":\"-ffp-contract=fast\"}\n")
    string(JSON V2_CPU_CONTRACTION_PLAN_SCHEMA ERROR_VARIABLE V2_CPU_CONTRACTION_PLAN_ERROR
        GET "${V2_CPU_CONTRACTION_PLAN_TEXT}" schema)
    if(V2_CPU_CONTRACTION_PLAN_ERROR)
        message(FATAL_ERROR "x86 contraction proof generated an invalid plan: ${V2_CPU_CONTRACTION_PLAN_ERROR}")
    endif()
    v2_cpu_contraction_write_if_changed("${V2_CPU_CONTRACTION_PLAN}" "${V2_CPU_CONTRACTION_PLAN_TEXT}")

    file(GLOB_RECURSE V2_CPU_CONTRACTION_HEADERS CONFIGURE_DEPENDS
        "${V2_CPU_CONTRACTION_ROOT}/src/v2/*.h")
    add_custom_command(OUTPUT "${V2_CPU_CONTRACTION_PROOF}"
        COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_CONTRACTION_SCRIPTS}/cpu_contraction_proof.py" compile
            --plan "${V2_CPU_CONTRACTION_PLAN}" --report "${V2_CPU_CONTRACTION_PROOF}"
        DEPENDS ${V2_CPU_CONTRACTION_HEADERS} "${V2_CPU_CONTRACTION_PLAN}" "${V2_CPU_CONTRACTION_DATABASE}"
            "${V2_CPU_CONTRACTION_SCRIPTS}/cpu_contraction_proof.py"
            "${V2_CPU_CONTRACTION_SCRIPTS}/check_cpu_contraction_witness.py"
            "${V2_CPU_CONTRACTION_ROOT}/tests/v2/cmake/CPUCodegenEnvelope.py"
            "${V2_CPU_CONTRACTION_ROOT}/tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp"
            "${CMAKE_CXX_COMPILER}" "${CMAKE_OBJDUMP}"
        VERBATIM)
    add_custom_target(v2_cpu_contraction_proof DEPENDS "${V2_CPU_CONTRACTION_PROOF}")
    add_test(NAME V2_Integration_CPUContributionContraction
        COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_CONTRACTION_SCRIPTS}/cpu_contraction_proof.py" verify
            --plan "${V2_CPU_CONTRACTION_PLAN}" --report "${V2_CPU_CONTRACTION_PROOF}")
    set_property(GLOBAL PROPERTY
        V2_TEST_BUILD_TARGETS_V2_Integration_CPUContributionContraction v2_cpu_contraction_proof)
    set_tests_properties(V2_Integration_CPUContributionContraction PROPERTIES
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Build;Regression"
        TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CPUContributionContraction)
endif()

# Receipt, identity, command-derivation and matcher controls are tool-free and
# device-free and run in every configuration, including the unqualified ones.
add_test(NAME V2_Unit_CPUContractionProofAdmission
    COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_CONTRACTION_SCRIPTS}/test_cpu_contraction_proof.py"
        ContractionProofAdmission GeneratedCommandsReceipt)
set_tests_properties(V2_Unit_CPUContractionProofAdmission PROPERTIES
    LABELS "V2;Unit;DeviceFree;CPU;Build;Regression" TIMEOUT 30)
# Configure (never build) fixture projects with this module, the canonical ISA
# authority and the configured compiler at BUILD time; the Unit test verifies
# the source- and compiler-bound receipt without invoking any tool.
set(V2_CPU_CONTRACTION_GENERATED_SCRIPT "${V2_CPU_CONTRACTION_SCRIPTS}/cpu_contraction_generated_commands.py")
add_custom_command(OUTPUT "${V2_CPU_CONTRACTION_GENERATED}"
    COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_CONTRACTION_GENERATED_SCRIPT}" build
        --identity "${V2_CPU_CONTRACTION_IDENTITY}" --report "${V2_CPU_CONTRACTION_GENERATED}"
        --cmake "${CMAKE_COMMAND}" --ctest "${CMAKE_CTEST_COMMAND}" --ninja "${CMAKE_MAKE_PROGRAM}"
    DEPENDS "${V2_CPU_CONTRACTION_GENERATED_SCRIPT}" "${V2_CPU_CONTRACTION_IDENTITY}"
        "${V2_CPU_CONTRACTION_ROOT}/tests/v2/cmake/V2CPUContractionProof.cmake"
        "${V2_CPU_CONTRACTION_ROOT}/cmake/CPUInstructionSet.cmake"
        "${V2_CPU_CONTRACTION_SCRIPTS}/check_cpu_contraction_witness.py"
        "${V2_CPU_CONTRACTION_ROOT}/tests/v2/cmake/CPUCodegenEnvelope.py"
        "${CMAKE_CXX_COMPILER}"
    VERBATIM)
add_custom_target(v2_cpu_contraction_generated_commands DEPENDS "${V2_CPU_CONTRACTION_GENERATED}")
set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS v2_cpu_contraction_generated_commands)
add_test(NAME V2_Unit_CPUContractionProofGeneratedCommands
    COMMAND "${Python3_EXECUTABLE}" "${V2_CPU_CONTRACTION_GENERATED_SCRIPT}" verify
        --identity "${V2_CPU_CONTRACTION_IDENTITY}" --report "${V2_CPU_CONTRACTION_GENERATED}")
set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_V2_Unit_CPUContractionProofGeneratedCommands
    v2_cpu_contraction_generated_commands)
set_tests_properties(V2_Unit_CPUContractionProofGeneratedCommands PROPERTIES
    LABELS "V2;Unit;DeviceFree;CPU;Build;CMake;Regression" TIMEOUT 30)
