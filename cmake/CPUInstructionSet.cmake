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
# @param requested Case-insensitive ISA name (AVX512, AVX2, NEON or NEON_DOTPROD).
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
        if(isa STREQUAL "NEON")
            set(flags "-mcpu=generic -march=armv8-a+simd")
            set(dotprod 0)
        elseif(isa STREQUAL "NEON_DOTPROD")
            set(flags "-mcpu=generic -march=armv8-a+simd+dotprod")
            set(dotprod 1)
        else()
            message(FATAL_ERROR "CPU ISA '${requested}' is incompatible with AArch64 target '${processor}'; use NEON or NEON_DOTPROD")
        endif()
        set(definitions LLAMINAR_COMPILED_WITH_AVX2=0 LLAMINAR_COMPILED_WITH_AVX512=0
            LLAMINAR_COMPILED_WITH_NEON=1 LLAMINAR_COMPILED_WITH_NEON_DOTPROD=${dotprod})
    else()
        set(flags "-msse4.1 -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mpopcnt")
        if(isa STREQUAL "AVX512")
            string(APPEND flags " -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni")
            set(avx512 1)
        elseif(isa STREQUAL "AVX2")
            set(avx512 0)
        else()
            message(FATAL_ERROR "CPU ISA '${requested}' is incompatible with x86-64 target '${processor}'; use AVX512 or AVX2")
        endif()
        set(definitions LLAMINAR_COMPILED_WITH_AVX2=1
            LLAMINAR_COMPILED_WITH_AVX512=${avx512} LLAMINAR_COMPILED_WITH_NEON=0
            LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0)
    endif()
    set(${flags_output} "${flags}" PARENT_SCOPE)
    set(${definitions_output} "${definitions}" PARENT_SCOPE)
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
        "CPU code-generation envelope: AVX512/AVX2 on x86-64, NEON/NEON_DOTPROD on AArch64")
    set_property(CACHE LLAMINAR_CPU_ISA PROPERTY STRINGS AVX512 AVX2 NEON NEON_DOTPROD)
    string(TOUPPER "${LLAMINAR_CPU_ISA}" LLAMINAR_CPU_ISA_UPPER)
    llaminar_cpu_instruction_set("${LLAMINAR_CPU_TARGET_PROCESSOR}" "${LLAMINAR_CPU_ISA}"
        LLAMINAR_CPU_COMPILE_FLAGS LLAMINAR_CPU_COMPILE_DEFINITIONS)
    add_compile_definitions(${LLAMINAR_CPU_COMPILE_DEFINITIONS})
    string(APPEND CMAKE_CXX_FLAGS " ${LLAMINAR_CPU_COMPILE_FLAGS}")
    string(APPEND CMAKE_C_FLAGS " ${LLAMINAR_CPU_COMPILE_FLAGS}")
    message(STATUS "V2: CPU ISA target: ${LLAMINAR_CPU_ISA_UPPER} (${LLAMINAR_CPU_COMPILE_FLAGS})")
endmacro()
