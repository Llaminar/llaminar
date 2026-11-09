# @file TestCPUInstructionSet.cmake
# @brief Exercise the production target/ISA resolver without CPU emulation or dependencies.
#
# Target strings intentionally differ from the host so cross-toolchain policy
# can be checked on every machine. Invalid combinations must fail before any
# dependency probe can hide the real architecture mismatch.
cmake_minimum_required(VERSION 3.18)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/CPUInstructionSet.cmake")

if(DEFINED INVALID_PROCESSOR)
    llaminar_cpu_instruction_set("${INVALID_PROCESSOR}" "${INVALID_ISA}" flags definitions)
    return()
endif()

foreach(processor IN ITEMS arm64 aarch64 ARM64 AARCH64)
    llaminar_default_cpu_isa("${processor}" default)
    llaminar_cpu_instruction_set("${processor}" nEoN flags definitions)
    if(NOT default STREQUAL "NEON" OR NOT flags STREQUAL "-mcpu=generic -march=armv8-a+simd" OR
       NOT "LLAMINAR_COMPILED_WITH_NEON=1" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_AVX2=0" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_AVX512=0" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_NEON_I8MM=0" IN_LIST definitions)
        message(FATAL_ERROR "AArch64 policy is not native NEON: '${flags}' '${definitions}'")
    endif()
    llaminar_cpu_instruction_set("${processor}" neon_dotprod flags definitions)
    if(NOT flags STREQUAL "-mcpu=generic -march=armv8-a+simd+dotprod" OR
       NOT "LLAMINAR_COMPILED_WITH_NEON=1" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_NEON_DOTPROD=1" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_NEON_I8MM=0" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_AVX2=0" IN_LIST definitions)
        message(FATAL_ERROR "DotProd policy has the wrong native compiler envelope")
    endif()
    # I8MM is the DotProd envelope plus exactly one extension on the ARMv8.0
    # base; an ARMv8.2+ base would implicitly add unadmitted LSE/RDMA/CRC.
    llaminar_cpu_instruction_set("${processor}" Neon_I8mm flags definitions)
    if(NOT flags STREQUAL "-mcpu=generic -march=armv8-a+simd+dotprod+i8mm" OR
       NOT "LLAMINAR_COMPILED_WITH_NEON=1" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_NEON_DOTPROD=1" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_NEON_I8MM=1" IN_LIST definitions OR
       NOT "LLAMINAR_COMPILED_WITH_AVX2=0" IN_LIST definitions)
        message(FATAL_ERROR "I8MM policy has the wrong native compiler envelope: '${flags}' '${definitions}'")
    endif()
endforeach()

foreach(processor IN ITEMS x86_64 AMD64 amd64 x64)
    llaminar_default_cpu_isa("${processor}" default)
    if(NOT default STREQUAL "AVX512")
        message(FATAL_ERROR "Existing x86 default changed")
    endif()
    foreach(isa IN ITEMS AVX2 avx512)
        llaminar_cpu_instruction_set("${processor}" "${isa}" flags definitions)
        if(NOT flags MATCHES "-mavx2" OR flags MATCHES "-march=" OR
           NOT "LLAMINAR_COMPILED_WITH_AVX2=1" IN_LIST definitions OR
           NOT "LLAMINAR_COMPILED_WITH_NEON=0" IN_LIST definitions OR
           NOT "LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0" IN_LIST definitions OR
           NOT "LLAMINAR_COMPILED_WITH_NEON_I8MM=0" IN_LIST definitions)
            message(FATAL_ERROR "x86 compiler envelope changed: '${flags}' '${definitions}'")
        endif()
        if(isa STREQUAL "AVX2")
            if(flags MATCHES "avx512" OR NOT "LLAMINAR_COMPILED_WITH_AVX512=0" IN_LIST definitions)
                message(FATAL_ERROR "AVX2-only binary advertises AVX512")
            endif()
        elseif(NOT flags MATCHES "-mavx512vnni" OR NOT "LLAMINAR_COMPILED_WITH_AVX512=1" IN_LIST definitions)
            message(FATAL_ERROR "AVX512 compiler envelope lost VNNI")
        endif()
    endforeach()
endforeach()

# @brief Require one rejected pair and its precise diagnostic in a fresh CMake process.
function(expect_rejection processor isa diagnostic)
    execute_process(COMMAND "${CMAKE_COMMAND}"
        "-DINVALID_PROCESSOR=${processor}" "-DINVALID_ISA=${isa}"
        -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "${diagnostic}")
        message(FATAL_ERROR "Expected rejection of '${processor}/${isa}': ${output}${error}")
    endif()
endfunction()
expect_rejection(arm64 AVX512 "incompatible with AArch64")
expect_rejection(aarch64 AVX2 "incompatible with AArch64")
expect_rejection(x86_64 NEON "incompatible with x86-64")
expect_rejection(x86_64 NEON_DOTPROD "incompatible with x86-64")
expect_rejection(x86_64 NEON_I8MM "incompatible with x86-64")
expect_rejection(arm64 NEON_I8MM_SVE "incompatible with AArch64")
expect_rejection(arm64 scalar "incompatible with AArch64")
expect_rejection(x86_64 native "incompatible with x86-64")
expect_rejection(riscv64 NEON "Unsupported CPU target")
expect_rejection(armv7l NEON "Unsupported CPU target")
message(STATUS "PASS: CPU target/ISA defaults, compiler definitions and incompatible selections")
