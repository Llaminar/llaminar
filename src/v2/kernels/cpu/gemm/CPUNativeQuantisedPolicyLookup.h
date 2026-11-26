/**
 * @file CPUNativeQuantisedPolicyLookup.h
 * @brief Lightweight declarations for out-of-line generated CPU NativeQuantised policy tables.
 *
 * Isolates the 250,000+ lines of generated decode and verifier-row policy tables into a
 * single compiled translation unit (`CPUNativeQuantisedPolicyLookup.cpp`) so callers
 * including `CPUNativeQuantisedGemv.h` do not re-parse generated switch/table data.
 */
#pragma once

#include <cstdint>

#define LLAMINAR_CPU_NATIVE_QUANTISED_DECODE_POLICY_ABI 1
#define LLAMINAR_CPU_NATIVE_QUANTISED_DECODE_POLICY_CERTIFIED 1
#define LLAMINAR_CPU_NATIVE_QUANTISED_VERIFIER_POLICY_ABI 3

namespace llaminar2::cpu::native_quantised::generated
{

/** @brief Byte-exact CPU NativeQuantised M=1 decode schedule policy. */
enum class CPUNativeQuantisedDecodePolicy : uint8_t
{
    Nbc1 = 0,
    Nbc2 = 1,
    Nbc4 = 2,
    Nbc8 = 3,
    Nbc16 = 4,
};

/** @brief Compiled CPU baseline ISA dimension for decode schedule policy lookup. */
enum class CPUNativeQuantisedDecodeBuildISA : uint8_t
{
    AVX2 = 0,
    AVX512 = 1,
};

/** @brief Observed CPU runtime ISA dimension for decode schedule policy lookup. */
enum class CPUNativeQuantisedDecodeRuntimeISA : uint8_t
{
    AVX2 = 0,
    AVX512 = 1,
};

/** @brief Byte-exact CPU NativeQuantised grouped verifier-row schedule policy. */
enum class CPUNativeQuantisedVerifierRowsPolicy : uint8_t
{
    Pairwise = 0,
    WideRows = 1,
    FullKRowChunkGrid = 2,
    FullKTwoRowNbc1 = 3,
    FullKTwoRowNbc2 = 4,
    FullKTwoRowPairGridNbc1 = 5,
    FullKTwoRowPairGridNbc2 = 6,
    FullKTwoRowPairGridNbc4 = 7,
    FullKTwoRowPairGridNbc8 = 8,
};

/** @brief Compiled CPU baseline ISA dimension for verifier-row schedule lookup. */
enum class CPUNativeQuantisedBuildISA : uint8_t
{
    AVX2 = 0,
    AVX512 = 1,
};

/** @brief Observed CPU runtime ISA dimension for verifier-row schedule lookup. */
enum class CPUNativeQuantisedRuntimeISA : uint8_t
{
    AVX2 = 0,
    AVX512 = 1,
};

/** @brief Pack decode policy lookup key from codebook, serial-kpart flag, N, and K. */
inline constexpr uint64_t packCPUNativeQuantisedDecodePolicyKey(
    uint8_t codebook, bool serial_kpart, int n, int k)
{
    return (static_cast<uint64_t>(codebook) << 56) |
           (static_cast<uint64_t>(serial_kpart ? 1 : 0) << 55) |
           (static_cast<uint64_t>(k & 0x7FFFFFFF) << 24) |
           static_cast<uint64_t>(n & 0xFFFFFF);
}

/** @brief Pack grouped verifier-row policy lookup key from codebook, M, N, and K. */
inline constexpr uint64_t packCPUNativeQuantisedVerifierRowsPolicyKey(
    uint8_t codebook, int m, int n, int k)
{
    return (static_cast<uint64_t>(codebook) << 56) |
           (static_cast<uint64_t>(m & 0xFF) << 48) |
           (static_cast<uint64_t>(k & 0xFFFFFF) << 24) |
           static_cast<uint64_t>(n & 0xFFFFFF);
}

/** @brief Map every positive grouped runtime M onto measured policy support. */
inline constexpr int cpuNativeQuantisedVerifierPolicyM(int m)
{
    if (m < 2)
        return 0;
    return m <= 16 ? m : 31;
}

/**
 * @brief Resolve the generated CPU NativeQuantised M=1 decode schedule policy.
 * @param build_isa Compiled CPU ISA baseline.
 * @param runtime_isa Observed runtime CPU ISA.
 * @param threads Active worker thread budget.
 * @param codebook Physical NativeQuantised codebook ID.
 * @param n Output columns.
 * @param k Reduction width.
 * @param serial_kpart Whether serial K-partition reduction applies.
 * @param k_tiles Number of K tiles.
 * @param policy Receives selected decode policy when true is returned.
 * @return True when an exact or certified generic rule matched.
 */
bool selectCPUNativeQuantisedDecodeGeneratedPolicy(
    CPUNativeQuantisedDecodeBuildISA build_isa,
    CPUNativeQuantisedDecodeRuntimeISA runtime_isa, int threads,
    uint8_t codebook, int n, int k, bool serial_kpart, int k_tiles,
    CPUNativeQuantisedDecodePolicy &policy);

/**
 * @brief Resolve the generated CPU NativeQuantised grouped verifier-row schedule policy.
 * @param build_isa Compiled CPU ISA baseline.
 * @param runtime_isa Observed runtime CPU ISA.
 * @param threads Active worker thread budget.
 * @param codebook Physical NativeQuantised codebook ID.
 * @param m Grouped row count.
 * @param n Output columns.
 * @param k Reduction width.
 * @param k_tiles Number of K tiles.
 * @param policy Receives selected verifier-row policy when true is returned.
 * @param measured_speedup Optional speedup output when requested.
 * @return True when an exact or certified generic rule matched.
 */
bool selectCPUNativeQuantisedVerifierRowsGeneratedPolicy(
    CPUNativeQuantisedBuildISA build_isa,
    CPUNativeQuantisedRuntimeISA runtime_isa, int threads,
    uint8_t codebook, int m, int n, int k, int k_tiles,
    CPUNativeQuantisedVerifierRowsPolicy &policy, float *measured_speedup = nullptr);

} // namespace llaminar2::cpu::native_quantised::generated
