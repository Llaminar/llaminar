# @file CMakeLists.txt
# @brief Reduced-ISA Q8 finite-scale contract executables under their own CPU envelopes.
#
# V2CPUExplicitRoundingTests.cmake adds this directory with
# V2_Q8_FINITE_SCALE_SECONDARY_ISAS naming every x86 copy whose ISA differs
# from the canonical build envelope. This scope drops the canonical envelope;
# each executable then compiles under exactly one declared envelope (flags and
# LLAMINAR_COMPILED_WITH_* values agree), recorded for the host compile
# contract. Test registration stays with the including directory, which owns
# the CTest inventory and its preflight labels.
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/V2CPUSecondaryEnvelope.cmake")
v2_enter_cpu_secondary_envelope_scope()

foreach(V2_Q8_SCALE_ISA IN LISTS V2_Q8_FINITE_SCALE_SECONDARY_ISAS)
    string(TOLOWER "${V2_Q8_SCALE_ISA}" V2_Q8_SCALE_TARGET_ISA)
    set(V2_Q8_SCALE_TARGET v2_test_q8_finite_scale_contract_${V2_Q8_SCALE_TARGET_ISA})
    v2_add_cpu_rounding_target(${V2_Q8_SCALE_TARGET})
    target_compile_definitions(${V2_Q8_SCALE_TARGET} PRIVATE
        LLAMINAR_Q8_SCALE_TEST_${V2_Q8_SCALE_ISA}=1)
    v2_declare_cpu_secondary_envelope_target(${V2_Q8_SCALE_TARGET} ${V2_Q8_SCALE_ISA})
endforeach()
