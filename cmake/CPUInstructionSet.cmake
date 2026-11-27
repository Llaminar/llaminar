# @file CPUInstructionSet.cmake
# @brief Target-architecture CPU code-generation contracts shared by V2 and focused tests.
#
# Resolve the target, never the build host: a cross-compiled ARM binary must not
# inherit x86 flags or identify itself as an AVX2 artifact. NEON is a compiler
# envelope, not a claim that the complete inference backend has been ported.
include_guard(GLOBAL)

# @brief Choose the existing x86 default or the mandatory AArch64 SIMD baseline.
# @param processor CMAKE_SYSTEM_PROCESSOR of the target toolchain.
# @param output Caller variable receiving AVX512 or NEON.
function(llaminar_default_cpu_isa processor output)
    string(TOLOWER "${processor}" architecture)
    if(architecture MATCHES "^(aarch64|arm64)$")
        set(isa NEON)
    elseif(architecture MATCHES "^(x86_64|amd64|x64)$")
        set(isa AVX512)
    else()
        message(FATAL_ERROR "Unsupported CPU target '${processor}'; expected x86-64 or AArch64")
    endif()
    set(${output} "${isa}" PARENT_SCOPE)
endfunction()

# @brief Resolve an explicit ISA and reject incompatible target/ISA pairs.
# @param processor Target processor, independent of the machine running CMake.
# @param requested Case-insensitive ISA name (AVX512, AVX2, AVX512BF16, AMXBF16,
#        NEON, NEON_DOTPROD or NEON_I8MM).
# @param flags_output Caller variable receiving space-separated compiler flags.
# @param definitions_output Caller variable receiving compiled-ISA definitions.
function(llaminar_cpu_instruction_set processor requested flags_output definitions_output)
    llaminar_default_cpu_isa("${processor}" default_isa)
    string(TOUPPER "${requested}" isa)
    if(default_isa STREQUAL "NEON")
        # -march alone does not clear Clang's implicit target CPU on Darwin.
        # AppleClang can otherwise auto-vectorize with apple-m1-only SDOT even
        # while its preprocessor advertises an ARMv8.0 envelope. Reset the CPU
        # model explicitly; optional instructions then come only from -march.
        # NEON_I8MM adds exactly the Int8 matrix-multiply extension to the
        # DotProd envelope. Keep the ARMv8.0 base: an ARMv8.2+ base would also
        # enable LSE, RDMA and CRC, which no startup admission authenticates.
        if(isa STREQUAL "NEON")
            set(flags "-mcpu=generic -march=armv8-a+simd")
            set(dotprod 0)
            set(i8mm 0)
            set(llaminar_fp16fml 0)
        elseif(isa STREQUAL "NEON_DOTPROD")
            set(flags "-mcpu=generic -march=armv8-a+simd+dotprod")
            set(dotprod 1)
            set(i8mm 0)
            set(llaminar_fp16fml 0)
        elseif(isa STREQUAL "NEON_I8MM")
            # The I8MM envelope explicitly declares the FP16 FML long
            # multiply-add extension: the half-operand native leaf is part of
            # this envelope's contract, and no other envelope carries it. The
            # assemblability of the declared extension is verified where a
            # compiler exists (llaminar_configure_cpu_instruction_set); this
            # pure resolver only composes the declared contract.
            set(flags "-mcpu=generic -march=armv8-a+simd+dotprod+i8mm+fp16fml")
            set(dotprod 1)
            set(i8mm 1)
            set(llaminar_fp16fml 1)
        else()
            message(FATAL_ERROR "CPU ISA '${requested}' is incompatible with AArch64 target '${processor}'; use NEON, NEON_DOTPROD or NEON_I8MM")
        endif()
        # The FMLAL probe is scoped to the NEON_I8MM envelope above; the
        # baseline NEON and DotProd envelopes keep their declared -march
        # tokens unchanged so no binary silently requires FEAT_FHM.
        set(definitions LLAMINAR_COMPILED_WITH_AVX2=0 LLAMINAR_COMPILED_WITH_AVX512=0
            LLAMINAR_COMPILED_WITH_NEON=1 LLAMINAR_COMPILED_WITH_NEON_DOTPROD=${dotprod}
            LLAMINAR_COMPILED_WITH_NEON_I8MM=${i8mm}
            LLAMINAR_COMPILED_WITH_FP16FML=${llaminar_fp16fml}
            LLAMINAR_COMPILED_WITH_AVX512_BF16=0
            LLAMINAR_COMPILED_WITH_AMX_BF16=0)
    else()
        set(flags "-msse4.1 -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mpopcnt")
        set(bf16 0)
        set(amx 0)
        if(isa STREQUAL "AVX512")
            string(APPEND flags " -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni")
            set(avx512 1)
        elseif(isa STREQUAL "AVX2")
            set(avx512 0)
        elseif(isa STREQUAL "AVX512BF16")
            # The BF16 envelope is the full AVX512-VNNI envelope plus the
            # Cooper Lake+ BF16 dot-product extension. It contains no AMX
            # instructions, so any host that admits AVX512-VNNI plus BF16 runs
            # it; the machine admission lives in CPUFeatures.h.
            string(APPEND flags " -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512bf16")
            set(avx512 1)
            set(bf16 1)
        elseif(isa STREQUAL "AMXBF16")
            # The AMX envelope is the BF16 envelope plus the Sapphire Rapids+
            # tile matrix engine. AMX tile state requires XCR0 bits 17/18; the
            # machine admission in CPUFeatures.h authenticates both the CPUID
            # feature and the OS state before any tile instruction executes.
            string(APPEND flags " -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512bf16 -mamx-tile -mamx-bf16")
            set(avx512 1)
            set(bf16 1)
            set(amx 1)
        else()
            message(FATAL_ERROR "CPU ISA '${requested}' is incompatible with x86-64 target '${processor}'; use AVX512, AVX2, AVX512BF16 or AMXBF16")
        endif()
        set(definitions LLAMINAR_COMPILED_WITH_AVX2=1
            LLAMINAR_COMPILED_WITH_AVX512=${avx512} LLAMINAR_COMPILED_WITH_NEON=0
            LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0 LLAMINAR_COMPILED_WITH_NEON_I8MM=0
            LLAMINAR_COMPILED_WITH_FP16FML=0
            LLAMINAR_COMPILED_WITH_AVX512_BF16=${bf16}
            LLAMINAR_COMPILED_WITH_AMX_BF16=${amx})
    endif()
    set(${flags_output} "${flags}" PARENT_SCOPE)
    set(${definitions_output} "${definitions}" PARENT_SCOPE)
endfunction()

# @brief Forward the resolved CPU envelope only to GPU compiler host passes.
# @param flags Space-separated options from the canonical ISA resolver.
# @param cuda_output Caller variable receiving nvcc host options.
# @param hip_output Caller variable receiving Clang host-only options.
function(llaminar_cpu_host_compiler_options flags cuda_output hip_output)
    separate_arguments(options NATIVE_COMMAND "${flags}")
    set(cuda "")
    set(hip "")
    foreach(option IN LISTS options)
        if(option MATCHES ",")
            message(FATAL_ERROR "CPU host option contains an nvcc separator: ${option}")
        endif()
        string(APPEND cuda " -Xcompiler=${option}")
        string(APPEND hip " -Xarch_host ${option}")
    endforeach()
    set(${cuda_output} "${cuda}" PARENT_SCOPE)
    set(${hip_output} "${hip}" PARENT_SCOPE)
endfunction()

# @brief Configure the project-wide CPU envelope without inheriting native host flags.
macro(llaminar_configure_cpu_instruction_set)
    set(LLAMINAR_CPU_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
    # Darwin's -arch selection can differ from uname even without a toolchain
    # file. One build tree owns one SIMD envelope; universal binaries need
    # separate architecture builds rather than flags from whichever host ran CMake.
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
        list(LENGTH CMAKE_OSX_ARCHITECTURES LLAMINAR_OSX_ARCHITECTURE_COUNT)
        if(NOT LLAMINAR_OSX_ARCHITECTURE_COUNT EQUAL 1)
            message(FATAL_ERROR "Build each macOS CPU architecture in a separate build tree; universal SIMD envelopes are unsupported")
        endif()
        set(LLAMINAR_CPU_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
    endif()
    llaminar_default_cpu_isa("${LLAMINAR_CPU_TARGET_PROCESSOR}" LLAMINAR_CPU_DEFAULT_ISA)
    set(LLAMINAR_CPU_ISA "${LLAMINAR_CPU_DEFAULT_ISA}" CACHE STRING
        "CPU code-generation envelope: AVX512/AVX2/AVX512BF16/AMXBF16 on x86-64, NEON/NEON_DOTPROD/NEON_I8MM on AArch64")
    set_property(CACHE LLAMINAR_CPU_ISA PROPERTY STRINGS
        AVX512 AVX2 AVX512BF16 AMXBF16 NEON NEON_DOTPROD NEON_I8MM)
    string(TOUPPER "${LLAMINAR_CPU_ISA}" LLAMINAR_CPU_ISA_UPPER)
    llaminar_cpu_instruction_set("${LLAMINAR_CPU_TARGET_PROCESSOR}" "${LLAMINAR_CPU_ISA}"
        LLAMINAR_CPU_COMPILE_FLAGS LLAMINAR_CPU_COMPILE_DEFINITIONS)
    # The I8MM envelope declares the FP16 FML extension; verify the real
    # toolchain assembles the declared contract where a compiler exists. A
    # toolchain that cannot assemble it is a configure error for this
    # envelope, never a silent downgrade, and no other envelope is probed.
    if(LLAMINAR_CPU_ISA_UPPER STREQUAL "NEON_I8MM")
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_QUIET ON)
        set(CMAKE_REQUIRED_FLAGS "${LLAMINAR_CPU_COMPILE_FLAGS}")
        check_cxx_source_compiles([=[
#include <arm_neon.h>
#ifndef __ARM_FEATURE_FP16_FML
#error "the declared envelope does not define __ARM_FEATURE_FP16_FML"
#endif
int main()
{
    float32x4_t acc = vmovq_n_f32(0.0f);
    float16x8_t a = vreinterpretq_f16_u16(vmovq_n_u16(1u));
    float16x8_t b = vreinterpretq_f16_u16(vmovq_n_u16(2u));
    acc = vfmlalq_low_f16(acc, a, b);
    acc = vfmlalq_high_f16(acc, a, b);
    return vgetq_lane_f32(acc, 0) == 0.0f ? 0 : 1;
}
]=]
            llaminar_fp16fml_compiles)
        unset(CMAKE_REQUIRED_FLAGS)
        if(NOT llaminar_fp16fml_compiles)
            message(FATAL_ERROR "CPU ISA 'NEON_I8MM' requires the FP16FML extension (+fp16fml); the toolchain cannot assemble the declared envelope")
        endif()
    endif()
    # The BF16/AMX envelopes declare hardware dot-product instructions on top of
    # the AVX512-VNNI envelope. Verify the real toolchain assembles the same
    # portable shim the production kernel uses: memcpy-based BF16 chunk loads
    # (GCC and Clang vector ABIs) and the canonical <immintrin.h> AMX macros
    # (_tile_*), the API the GNU x86 toolchain exposes. A toolchain that cannot
    # assemble the declared contract is a configure error for that envelope,
    # never a silent downgrade.
    if(LLAMINAR_CPU_ISA_UPPER STREQUAL "AVX512BF16" OR LLAMINAR_CPU_ISA_UPPER STREQUAL "AMXBF16")
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_QUIET ON)
        set(CMAKE_REQUIRED_FLAGS "${LLAMINAR_CPU_COMPILE_FLAGS}")
        check_cxx_source_compiles([=[
#include <immintrin.h>
#include <cstdint>
#include <cstring>
int main()
{
    alignas(64) std::uint16_t av[32] = {};
    alignas(64) std::uint16_t bv[32] = {};
    __m512 acc = _mm512_setzero_ps();
    __m512bh a;
    __m512bh b;
    std::memcpy(&a, av, sizeof(a));
    std::memcpy(&b, bv, sizeof(b));
    acc = _mm512_dpbf16_ps(acc, a, b);
    return _mm512_cvtss_f32(acc) == 0.0f ? 0 : 1;
}
]=]
            llaminar_avx512bf16_compiles)
        if(NOT llaminar_avx512bf16_compiles)
            message(FATAL_ERROR "CPU ISA 'AVX512BF16' requires the AVX512-BF16 extension (-mavx512bf16); the toolchain cannot assemble the declared envelope")
        endif()
    endif()
    if(LLAMINAR_CPU_ISA_UPPER STREQUAL "AMXBF16")
        check_cxx_source_compiles([=[
#include <immintrin.h>
#include <cstdint>
int main()
{
    alignas(64) std::uint8_t cfg[64] = {};
    alignas(64) std::uint16_t a_data[16 * 16] = {};
    alignas(64) std::uint16_t b_data[16 * 16] = {};
    alignas(64) float c_data[16 * 16] = {};
    _tile_loadconfig(cfg);
    _tile_zero(2);
    _tile_loadd(0, a_data, 32);
    _tile_loadd(1, b_data, 32);
    _tile_dpbf16ps(2, 0, 1);
    _tile_stored(2, c_data, 64);
    _tile_release();
    return c_data[0] == 0.0f ? 0 : 1;
}
]=]
            llaminar_amxbf16_compiles)
        unset(CMAKE_REQUIRED_FLAGS)
        if(NOT llaminar_amxbf16_compiles)
            message(FATAL_ERROR "CPU ISA 'AMXBF16' requires AMX tile + BF16 (-mamx-tile -mamx-bf16); the toolchain cannot assemble the declared envelope")
        endif()
    else()
        unset(CMAKE_REQUIRED_FLAGS)
    endif()
    add_compile_definitions(${LLAMINAR_CPU_COMPILE_DEFINITIONS})
    string(APPEND CMAKE_CXX_FLAGS " ${LLAMINAR_CPU_COMPILE_FLAGS}")
    string(APPEND CMAKE_C_FLAGS " ${LLAMINAR_CPU_COMPILE_FLAGS}")
    llaminar_cpu_host_compiler_options("${LLAMINAR_CPU_COMPILE_FLAGS}"
        LLAMINAR_CUDA_HOST_OPTIONS LLAMINAR_HIP_HOST_OPTIONS)
    string(APPEND CMAKE_CUDA_FLAGS "${LLAMINAR_CUDA_HOST_OPTIONS}")
    string(APPEND CMAKE_HIP_FLAGS "${LLAMINAR_HIP_HOST_OPTIONS}")
    message(STATUS "V2: CPU ISA target: ${LLAMINAR_CPU_ISA_UPPER} (${LLAMINAR_CPU_COMPILE_FLAGS})")
endmacro()
