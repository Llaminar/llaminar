/**
 * @file Microbench__IQDecodeLab.hip
 * @brief Isolated IQ decode microbenchmark for experimenting with decode strategies
 *
 * PURPOSE: Measure pure decode throughput for IQ grid-index quantization formats
 * (IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M) in isolation from
 * the full GEMV pipeline.  This lets us compare alternative decode strategies
 * head-to-head without affecting the production kernel.
 *
 * ARCHITECTURE: Each "strategy" is a separate __device__ function that decodes
 * one 32-element block from native-VNNI payload → 8 × int32_t packed groups.
 * The benchmark kernel calls the decode function in a tight loop across N rows,
 * writing results to a sink buffer to prevent dead-code elimination.
 *
 * STRATEGIES TO EXPLORE:
 *   A. Baseline:       Current production decode (grid LUT + iq_apply_signs_4)
 *   B. vsub4 signs:    llama.cpp-style __vsub4(grid ^ signs, signs) sign application
 *   C. LDS grid cache: Wavefront-cooperative LDS preload of grid entries
 *   D. Warp shuffle:   Share grid lookups across lanes via __shfl
 *   E. ksigns64:       Pre-expanded sign table (for IQ2_XS/IQ3_XXS with 7-bit signs)
 *
 * USAGE:
 *   cmake --build build_v2_release --parallel --target v2_perf_iq_decode_lab
 *   ./build_v2_release/tests/v2/v2_perf_iq_decode_lab
 *
 * @note Standalone HIP file — does not link against the production kernel.
 *       Grid tables are copied to __constant__ memory independently.
 * @note Requires ROCm device (MI50/MI60 gfx906 target).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <functional>
#include <random>

// =============================================================================
// Include the canonical IQ grid tables from the main codebase
// =============================================================================
#include "tensors/IQQuantTables.h"

// =============================================================================
// Device constant memory for grid LUTs (independent copy for this TU)
// =============================================================================
__device__ __constant__ uint32_t dc_iq3s_grid[512];     // IQ3_S:   512 × 4B = 2KB
__device__ __constant__ uint32_t dc_iq3xxs_grid[256];   // IQ3_XXS: 256 × 4B = 1KB
__device__ __constant__ uint64_t dc_iq2s_grid[1024];    // IQ2_S:   1024 × 8B = 8KB
__device__ __constant__ uint64_t dc_iq2xs_grid[512];    // IQ2_XS:  512 × 8B = 4KB
__device__ __constant__ uint64_t dc_iq2xxs_grid[256];   // IQ2_XXS: 256 × 8B = 2KB
__device__ __constant__ uint64_t dc_iq1s_grid[2048];    // IQ1_S/M: 2048 × 8B = 16KB

// ksigns64 pre-expanded sign table (from llama.cpp): 128 entries × 8B = 1KB
// Maps 7-bit sign index → 8 bytes of 0x00/0xFF masks
__device__ __constant__ uint64_t dc_ksigns64[128];

// =============================================================================
// Compact IQ1 grid: 2-bit ternary encoding
//   Each {-1, 0, +1} value stored as 2 bits: 00=0, 01=+1, 11=-1
//   8 values per entry → 16 bits per entry → 2048 × 2B = 4KB (vs 16KB full)
//   Goal: 4KB fits comfortably in 16KB L1 cache, eliminating thrashing
// =============================================================================
__device__ __constant__ uint16_t dc_iq1_compact[2048];  // 2048 × 2B = 4KB

// Expansion LUT: maps 8 bits (4 × 2-bit packed values) → 4 packed int8 bytes
// 256 × 4B = 1KB — tiny, highly cacheable, shared across all lookups
__device__ __constant__ uint32_t dc_iq1_expand_lut[256]; // 256 × 4B = 1KB

// Split-table nibble LUT: maps 4-bit nibble (2 × 2-bit values) → 2 packed int8
// 16 × 2B = 32 bytes — fits in HALF a 64B cache line.
// The idea: decompose the 256-entry (1KB) expand LUT into a 16-entry (32B)
// nibble-indexed table. Each byte of compact data splits into 2 nibbles,
// each nibble indexes this tiny table for 2 values. 4 lookups per compact
// entry (vs 2 for full LUT), but 32× smaller = zero cache pressure.
__device__ __constant__ uint16_t dc_iq1_nibble_lut[16]; // 16 × 2B = 32B

// =============================================================================
// Host-side ksigns64 generation
// =============================================================================
static void generate_ksigns64(uint64_t out[128])
{
    // Each entry i (7 bits) → 8 bytes where byte j = (i & (1<<j)) ? 0xFF : 0x00
    // Only 7 bits used (indices 0..127), lower 7 bits of sign byte
    for (int i = 0; i < 128; ++i)
    {
        uint8_t bytes[8];
        for (int j = 0; j < 8; ++j)
            bytes[j] = (i & (1 << j)) ? 0xFF : 0x00;
        memcpy(&out[i], bytes, 8);
    }
}

// =============================================================================
// Compact IQ1 grid generation
// =============================================================================

// Compress full iq1s_grid[2048] × uint64_t → compact[2048] × uint16_t
// Encoding: for each of 8 signed int8 values {-1, 0, +1}:
//   00 = 0, 01 = +1, 11 = -1  (encoding 10 never occurs)
static void generate_iq1_compact_grid(uint16_t out[2048], const uint64_t* full_grid)
{
    for (int i = 0; i < 2048; i++)
    {
        uint16_t compact = 0;
        uint64_t entry = full_grid[i];
        for (int j = 0; j < 8; j++)
        {
            auto val = static_cast<int8_t>((entry >> (j * 8)) & 0xFF);
            uint16_t bits = (val == 0) ? 0u : (val == 1 ? 1u : 3u);
            compact |= static_cast<uint16_t>(bits << (j * 2));
        }
        out[i] = compact;
    }
}

// Generate expansion LUT: maps 8 bits (4 × 2-bit packed) → 4 packed int8
// Same encoding as above: 00→0x00, 01→0x01, 11→0xFF
static void generate_iq1_expand_lut(uint32_t out[256])
{
    for (int i = 0; i < 256; i++)
    {
        uint32_t expanded = 0;
        for (int j = 0; j < 4; j++)
        {
            int bits = (i >> (j * 2)) & 3;
            uint8_t val;
            if (bits == 0) val = 0;
            else if (bits == 1) val = 1;
            else val = static_cast<uint8_t>(0xFF); // -1 in int8
            expanded |= (static_cast<uint32_t>(val) << (j * 8));
        }
        out[i] = expanded;
    }
}

// Generate nibble split-table LUT: maps 4-bit nibble (2 × 2-bit packed) → 2 packed int8
// Same encoding as compact grid: 00→0x00, 01→0x01, 11→0xFF (-1)
static void generate_iq1_nibble_lut(uint16_t out[16])
{
    for (int nib = 0; nib < 16; nib++)
    {
        uint8_t bytes[2];
        for (int j = 0; j < 2; j++)
        {
            int bits = (nib >> (j * 2)) & 3;
            if (bits == 0) bytes[j] = 0;
            else if (bits == 1) bytes[j] = 1;
            else bytes[j] = static_cast<uint8_t>(0xFF); // -1 as int8
        }
        memcpy(&out[nib], bytes, 2);
    }
}

// Verify nibble LUT produces identical results to full expand LUT
static bool verify_iq1_nibble_lut(const uint16_t nibble_lut[16],
                                   const uint32_t expand_lut[256])
{
    for (int byte_val = 0; byte_val < 256; byte_val++)
    {
        uint32_t from_expand = expand_lut[byte_val];
        uint16_t lo_nib = nibble_lut[byte_val & 0xF];
        uint16_t hi_nib = nibble_lut[byte_val >> 4];
        uint32_t from_nibble = static_cast<uint32_t>(lo_nib)
                             | (static_cast<uint32_t>(hi_nib) << 16);
        if (from_expand != from_nibble)
        {
            fprintf(stderr, "Nibble LUT verification FAILED at byte %d\n", byte_val);
            fprintf(stderr, "  expand_lut: 0x%08x\n", from_expand);
            fprintf(stderr, "  nibble_lut: 0x%08x\n", from_nibble);
            return false;
        }
    }
    return true;
}

// Verify compact encoding roundtrips correctly against full grid
static bool verify_iq1_compact_grid(const uint16_t compact[2048],
                                    const uint32_t expand_lut[256],
                                    const uint64_t full_grid[2048])
{
    for (int i = 0; i < 2048; i++)
    {
        uint16_t c = compact[i];
        uint32_t lo = expand_lut[c & 0xFF];
        uint32_t hi = expand_lut[c >> 8];
        uint64_t reconstructed = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
        if (reconstructed != full_grid[i])
        {
            fprintf(stderr, "Compact grid verification FAILED at index %d\n", i);
            fprintf(stderr, "  full:   0x%016lx\n", full_grid[i]);
            fprintf(stderr, "  recon:  0x%016lx\n", reconstructed);
            return false;
        }
    }
    return true;
}

// =============================================================================
// Format descriptors
// =============================================================================
enum IQFormat : int {
    FMT_IQ3_S   = 0,
    FMT_IQ3_XXS = 1,
    FMT_IQ2_S   = 2,
    FMT_IQ2_XS  = 3,
    FMT_IQ2_XXS = 4,
    FMT_IQ1_S   = 5,
    FMT_IQ1_M   = 6,
    FMT_COUNT   = 7
};

struct IQFormatInfo {
    const char* name;
    int payload_bytes;      // bytes per 32-element block payload
    double bpw;             // bits per weight
    int grid_entries;       // grid table size (for LDS estimation)
    int grid_entry_bytes;   // 4 for IQ3, 8 for IQ2/IQ1
    int lookups_per_block;  // grid lookups needed per 32-element block
};

static const IQFormatInfo IQ_FORMATS[FMT_COUNT] = {
    {"IQ3_S",   13, 3.4, 512,  4, 8},  // 8 × 4-elem lookups, 9-bit index
    {"IQ3_XXS", 12, 3.1, 256,  4, 8},  // 8 × 4-elem lookups, 8-bit index
    {"IQ2_S",    9, 2.5, 1024, 8, 4},  // 4 × 8-elem lookups, 10-bit index
    {"IQ2_XS",   9, 2.3, 512,  8, 4},  // 4 × 8-elem lookups, 9-bit index
    {"IQ2_XXS",  8, 2.1, 256,  8, 4},  // 4 × 8-elem lookups, 8-bit index
    {"IQ1_S",    6, 1.6, 2048, 8, 4},  // 4 × 8-elem lookups, 11-bit index
    {"IQ1_M",   10, 1.9, 2048, 8, 4},  // 4 × 8-elem lookups, 11-bit index
};


// =============================================================================
// Strategy A: BASELINE — current production decode
// =============================================================================

// Sign application: expand 4 sign bits → byte mask → XOR + carry
__device__ __forceinline__ uint32_t iq_apply_signs_4_baseline(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t mask =
        (static_cast<uint32_t>(-(sign_lo4 & 1)) & 0xFFu) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 1) & 1)) & 0xFFu) << 8) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 2) & 1)) & 0xFFu) << 16) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 3) & 1)) & 0xFFu) << 24);
    uint32_t carry = mask & 0x01010101u;
    return (grid4 ^ mask) + carry;
}

// O4: SWAR multiply-based sign application (current production version)
// 2× v_mul_lo + 1× v_and + 1× v_xor + 1× v_add ≈ 5 VALU (vs ~15 for BFE baseline)
// Math: spread = (sign * 0x08040201) & 0x01010101 scatters bit[i] to byte[i]
//       mask = spread * 255 expands 0x01→0xFF per byte
//       (grid ^ mask) + spread is two's complement conditional negate
__device__ __forceinline__ uint32_t iq_apply_signs_4_swar(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t spread = (static_cast<uint32_t>(sign_lo4 & 0xFu) * 0x08040201u) & 0x01010101u;
    uint32_t mask = spread * 255u;
    return (grid4 ^ mask) + spread;
}

// --- IQ3 baseline ---
__device__ __forceinline__ void decode_iq3s_baseline(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh_byte = payload[8];  // IQ3_S: qh at offset 8
    #pragma unroll
    for (int g = 0; g < 8; ++g)
    {
        int idx = payload[g];
        idx |= (static_cast<int>((qh_byte >> g) & 1) << 8);  // 9-bit index
        const uint32_t grid4 = dc_iq3s_grid[idx];
        const uint8_t sb = payload[9 + g / 2];  // signs at offset 9
        const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
        out[g] = static_cast<int32_t>(iq_apply_signs_4_baseline(grid4, sn));
    }
}

__device__ __forceinline__ void decode_iq3xxs_baseline(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    #pragma unroll
    for (int g = 0; g < 8; ++g)
    {
        int idx = payload[g];  // 8-bit index, no qh
        const uint32_t grid4 = dc_iq3xxs_grid[idx];
        const uint8_t sb = payload[8 + g / 2];  // signs at offset 8
        const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
        out[g] = static_cast<int32_t>(iq_apply_signs_4_baseline(grid4, sn));
    }
}

// --- IQ2 baseline ---
__device__ __forceinline__ void decode_iq2s_baseline(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh_byte = payload[4];  // IQ2_S: qh at offset 4
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];
        idx |= (static_cast<int>((qh_byte >> (2 * l)) & 3) << 8);  // 10-bit index
        const uint64_t grid8 = dc_iq2s_grid[idx];
        const uint32_t g_lo = static_cast<uint32_t>(grid8);
        const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
        const uint8_t signs = payload[5 + l];  // signs at offset 5
        out[l * 2]     = static_cast<int32_t>(iq_apply_signs_4_baseline(g_lo, signs & 0xF));
        out[l * 2 + 1] = static_cast<int32_t>(iq_apply_signs_4_baseline(g_hi, signs >> 4));
    }
}

__device__ __forceinline__ void decode_iq2xs_baseline(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh_byte = payload[4];  // IQ2_XS: qh at offset 4
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];
        idx |= (static_cast<int>((qh_byte >> l) & 1) << 8);  // 9-bit index
        const uint64_t grid8 = dc_iq2xs_grid[idx];
        const uint32_t g_lo = static_cast<uint32_t>(grid8);
        const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
        const uint8_t signs = payload[5 + l];  // signs at offset 5
        out[l * 2]     = static_cast<int32_t>(iq_apply_signs_4_baseline(g_lo, signs & 0xF));
        out[l * 2 + 1] = static_cast<int32_t>(iq_apply_signs_4_baseline(g_hi, signs >> 4));
    }
}

__device__ __forceinline__ void decode_iq2xxs_baseline(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];  // 8-bit index, no qh
        const uint64_t grid8 = dc_iq2xxs_grid[idx];
        const uint32_t g_lo = static_cast<uint32_t>(grid8);
        const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
        const uint8_t signs = payload[4 + l];  // signs at offset 4
        out[l * 2]     = static_cast<int32_t>(iq_apply_signs_4_baseline(g_lo, signs & 0xF));
        out[l * 2 + 1] = static_cast<int32_t>(iq_apply_signs_4_baseline(g_hi, signs >> 4));
    }
}

// --- IQ1 baseline ---
__device__ __forceinline__ void decode_iq1s_baseline(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint64_t grid8 = dc_iq1s_grid[idx];
        out[l * 2]     = static_cast<int32_t>(static_cast<uint32_t>(grid8));
        out[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
    }
}

__device__ __forceinline__ void decode_iq1m_baseline(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    uint64_t g;
    g = dc_iq1s_grid[idx0];
    out[0] = static_cast<int32_t>(static_cast<uint32_t>(g));
    out[1] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
    g = dc_iq1s_grid[idx1];
    out[2] = static_cast<int32_t>(static_cast<uint32_t>(g));
    out[3] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
    g = dc_iq1s_grid[idx2];
    out[4] = static_cast<int32_t>(static_cast<uint32_t>(g));
    out[5] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
    g = dc_iq1s_grid[idx3];
    out[6] = static_cast<int32_t>(static_cast<uint32_t>(g));
    out[7] = static_cast<int32_t>(static_cast<uint32_t>(g >> 32));
}


// =============================================================================
// IQ1 Strategy: Compact 2-bit grid + expansion LUT (constant memory)
//   Grid: 2048 × 2B = 4KB in constant memory (vs 16KB full grid)
//   LUT:  256 × 4B = 1KB in constant memory
//   Total: 5KB — fits in 16KB L1 cache with 11KB headroom
//
//   Key insight: IQ1 values are ternary {-1, 0, +1}, storable as 2 bits.
//   16KB → 4KB eliminates L1 cache thrashing on gfx906 (16KB TCP per CU).
//   LUT is only 1KB with high reuse → near-zero cache overhead.
//
//   Tradeoff: extra lookup indirection (grid→LUT) adds ~5 cycles latency
//   but eliminates ~100+ cycle cache misses.
// =============================================================================

__device__ __forceinline__ void decode_iq1s_compact_lut(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint16_t compact = dc_iq1_compact[idx];  // 2B load (vs 8B)
        out[l * 2]     = static_cast<int32_t>(dc_iq1_expand_lut[compact & 0xFF]);
        out[l * 2 + 1] = static_cast<int32_t>(dc_iq1_expand_lut[compact >> 8]);
    }
}

__device__ __forceinline__ void decode_iq1m_compact_lut(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    uint16_t c;
    c = dc_iq1_compact[idx0];
    out[0] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[1] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
    c = dc_iq1_compact[idx1];
    out[2] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[3] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
    c = dc_iq1_compact[idx2];
    out[4] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[5] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
    c = dc_iq1_compact[idx3];
    out[6] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[7] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
}


// =============================================================================
// IQ1 Strategy: Compact 2-bit grid + SWAR expansion (no LUT, pure ALU)
//   Same 4KB compact grid, but expand 2-bit→byte using SWAR bit tricks.
//   Eliminates the expand LUT entirely — zero extra memory.
//
//   SWAR expansion for 4 × 2-bit ternary values → 4 packed int8:
//     1. Separate value bits (even positions) and sign bits (odd positions)
//     2. Scatter each to byte positions via multiply: * 0x00041041 & 0x01010101
//     3. Combine: value_bytes | (sign_bytes * 255)
//
//   Cost: ~9 VALU per 4-value group × 2 groups = ~18 VALU per entry.
//   Win if memory savings outweigh the ALU overhead.
// =============================================================================

// SWAR helper: expand 4 packed 2-bit ternary values → 4 packed int8 bytes
__device__ __forceinline__ uint32_t iq1_expand_swar(uint32_t packed8)
{
    // packed8 bits: [b3 a3 b2 a2 b1 a1 b0 a0]
    // where bi=sign, ai=value. Encoding: 00=0, 01=+1, 11=-1
    uint32_t a_raw = packed8 & 0x55u;                    // value bits at even positions
    uint32_t b_raw = (packed8 >> 1) & 0x55u;             // sign bits at even positions
    // Scatter bits 0,2,4,6 → bytes 0,1,2,3 via multiply trick:
    // bit_i at position 2i → byte i position 0
    uint32_t a = (a_raw * 0x00041041u) & 0x01010101u;    // value in each byte
    uint32_t b = (b_raw * 0x00041041u) & 0x01010101u;    // sign in each byte
    // Combine: value | sign_mask. Sign mask: 0→0x00, 1→0xFF.
    // When b=0: a|0 = a. When b=1,a=1: 1|0xFF = 0xFF (-1). b=1,a=0: never occurs.
    return a | (b * 255u);
}

__device__ __forceinline__ void decode_iq1s_compact_swar(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint16_t compact = dc_iq1_compact[idx];  // 2B load
        out[l * 2]     = static_cast<int32_t>(iq1_expand_swar(compact & 0xFFu));
        out[l * 2 + 1] = static_cast<int32_t>(iq1_expand_swar(compact >> 8));
    }
}

__device__ __forceinline__ void decode_iq1m_compact_swar(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    uint16_t c;
    c = dc_iq1_compact[idx0];
    out[0] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[1] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
    c = dc_iq1_compact[idx1];
    out[2] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[3] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
    c = dc_iq1_compact[idx2];
    out[4] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[5] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
    c = dc_iq1_compact[idx3];
    out[6] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[7] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
}


// =============================================================================
// IQ1 Strategy: Compact grid in LDS (4KB LDS + 1KB constant LUT)
//   Same compact encoding, but grid preloaded to LDS.
//   Only 4KB LDS (vs 16KB for full grid LDS).
//   On gfx906: 64KB LDS / 4KB = up to 16 WGs limited by LDS alone
//   (vs 64KB/16KB = 4 WGs for full grid in LDS).
//   Expansion LUT stays in constant memory (1KB, highly cacheable).
// =============================================================================

__device__ __forceinline__ void decode_iq1s_compact_lds(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint16_t* __restrict__ lds_compact)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint16_t compact = lds_compact[idx];  // LDS read (2B)
        out[l * 2]     = static_cast<int32_t>(dc_iq1_expand_lut[compact & 0xFF]);
        out[l * 2 + 1] = static_cast<int32_t>(dc_iq1_expand_lut[compact >> 8]);
    }
}

__device__ __forceinline__ void decode_iq1m_compact_lds(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint16_t* __restrict__ lds_compact)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    uint16_t c;
    c = lds_compact[idx0];
    out[0] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[1] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
    c = lds_compact[idx1];
    out[2] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[3] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
    c = lds_compact[idx2];
    out[4] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[5] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
    c = lds_compact[idx3];
    out[6] = static_cast<int32_t>(dc_iq1_expand_lut[c & 0xFF]);
    out[7] = static_cast<int32_t>(dc_iq1_expand_lut[c >> 8]);
}


// =============================================================================
// IQ1 Strategy: Explicit prefetch — hoist all index computation, then load
//   Separates index computation from grid loads, giving the compiler full
//   freedom to issue all 4 loads together.  On gfx906, buffer loads have
//   ~100+ cycle latency on cache miss — overlapping 4 loads saves ~300 cycles.
//   (The #pragma unroll baseline MAY already achieve this, but explicit
//   separation makes it guaranteed.)
// =============================================================================

__device__ __forceinline__ void decode_iq1s_prefetch(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);

    // Phase 1: compute ALL indices (pure ALU, no memory)
    const int idx0 = payload[0] | (((qh_word)       & 7) << 8);
    const int idx1 = payload[1] | (((qh_word >> 3)  & 7) << 8);
    const int idx2 = payload[2] | (((qh_word >> 6)  & 7) << 8);
    const int idx3 = payload[3] | (((qh_word >> 9)  & 7) << 8);

    // Phase 2: issue ALL loads together (compiler can pipeline)
    const uint64_t g0 = dc_iq1s_grid[idx0];
    const uint64_t g1 = dc_iq1s_grid[idx1];
    const uint64_t g2 = dc_iq1s_grid[idx2];
    const uint64_t g3 = dc_iq1s_grid[idx3];

    // Phase 3: unpack results
    out[0] = static_cast<int32_t>(static_cast<uint32_t>(g0));
    out[1] = static_cast<int32_t>(static_cast<uint32_t>(g0 >> 32));
    out[2] = static_cast<int32_t>(static_cast<uint32_t>(g1));
    out[3] = static_cast<int32_t>(static_cast<uint32_t>(g1 >> 32));
    out[4] = static_cast<int32_t>(static_cast<uint32_t>(g2));
    out[5] = static_cast<int32_t>(static_cast<uint32_t>(g2 >> 32));
    out[6] = static_cast<int32_t>(static_cast<uint32_t>(g3));
    out[7] = static_cast<int32_t>(static_cast<uint32_t>(g3 >> 32));
}

__device__ __forceinline__ void decode_iq1m_prefetch(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];

    // Phase 1: ALL indices
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    // Phase 2: ALL loads
    const uint64_t g0 = dc_iq1s_grid[idx0];
    const uint64_t g1 = dc_iq1s_grid[idx1];
    const uint64_t g2 = dc_iq1s_grid[idx2];
    const uint64_t g3 = dc_iq1s_grid[idx3];

    // Phase 3: unpack
    out[0] = static_cast<int32_t>(static_cast<uint32_t>(g0));
    out[1] = static_cast<int32_t>(static_cast<uint32_t>(g0 >> 32));
    out[2] = static_cast<int32_t>(static_cast<uint32_t>(g1));
    out[3] = static_cast<int32_t>(static_cast<uint32_t>(g1 >> 32));
    out[4] = static_cast<int32_t>(static_cast<uint32_t>(g2));
    out[5] = static_cast<int32_t>(static_cast<uint32_t>(g2 >> 32));
    out[6] = static_cast<int32_t>(static_cast<uint32_t>(g3));
    out[7] = static_cast<int32_t>(static_cast<uint32_t>(g3 >> 32));
}


// =============================================================================
// IQ1 Strategy: Compact grid in LDS + SWAR expansion (pure ALU, no LUT)
//   Combines 4KB LDS (compact grid) with SWAR expansion (no constant memory).
//   This avoids the compact_lds regression caused by constant-memory LUT reads.
//   Goal: LDS speed for grid reads + ALU-only expansion = zero constant pressure.
// =============================================================================

__device__ __forceinline__ void decode_iq1s_compact_lds_swar(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint16_t* __restrict__ lds_compact)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint16_t compact = lds_compact[idx];  // LDS read (2B, ~1 cycle)
        out[l * 2]     = static_cast<int32_t>(iq1_expand_swar(compact & 0xFFu));
        out[l * 2 + 1] = static_cast<int32_t>(iq1_expand_swar(compact >> 8));
    }
}

__device__ __forceinline__ void decode_iq1m_compact_lds_swar(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint16_t* __restrict__ lds_compact)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    uint16_t c;
    c = lds_compact[idx0];
    out[0] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[1] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
    c = lds_compact[idx1];
    out[2] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[3] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
    c = lds_compact[idx2];
    out[4] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[5] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
    c = lds_compact[idx3];
    out[6] = static_cast<int32_t>(iq1_expand_swar(c & 0xFFu));
    out[7] = static_cast<int32_t>(iq1_expand_swar(c >> 8));
}


// =============================================================================
// IQ1 Strategy: Split-table — nibble-indexed LUT (32B) replaces 1KB expand LUT
//   Key insight: decompose each 8-bit byte into two 4-bit nibbles.
//   A 16-entry × 2-byte LUT (32B total) maps each nibble → 2 packed int8.
//   4 nibble lookups per compact entry (vs 2 expand_lut lookups), but the
//   table is 32× smaller — fits in HALF a cache line, zero eviction ever.
//
//   Variants:
//     split_const: compact grid (4KB constant) + nibble LUT (32B constant)
//     split_lds:   compact grid (4KB LDS) + nibble LUT (32B LDS) = ~4KB total
//
//   Cost per entry: 4 masks + 4 shifts + 4 LUT reads + 2 OR + 2 shifts ≈ 16 ops
//   vs SWAR: ~19 VALU ops (pure ALU)
//   vs full LUT: 2 reads from 1KB table (cache-hostile on gfx906)
// =============================================================================

// Split-table expansion helper: 16-bit compact → 2 × int32 via nibble LUT
// Works with either constant-memory or LDS-resident LUT pointer
__device__ __forceinline__ void iq1_expand_split(
    uint16_t compact, int32_t& out_lo, int32_t& out_hi,
    const uint16_t* __restrict__ nib_lut)
{
    const uint32_t b0 = compact & 0xFFu;
    const uint32_t b1 = compact >> 8;
    out_lo = static_cast<int32_t>(
        static_cast<uint32_t>(nib_lut[b0 & 0xF]) |
        (static_cast<uint32_t>(nib_lut[b0 >> 4]) << 16));
    out_hi = static_cast<int32_t>(
        static_cast<uint32_t>(nib_lut[b1 & 0xF]) |
        (static_cast<uint32_t>(nib_lut[b1 >> 4]) << 16));
}

// IQ1_S: compact grid (constant) + nibble LUT (constant, 32B)
__device__ __forceinline__ void decode_iq1s_split_const(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint16_t compact = dc_iq1_compact[idx];
        iq1_expand_split(compact, out[l * 2], out[l * 2 + 1], dc_iq1_nibble_lut);
    }
}

// IQ1_M: compact grid (constant) + nibble LUT (constant, 32B)
__device__ __forceinline__ void decode_iq1m_split_const(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    uint16_t c;
    c = dc_iq1_compact[idx0];
    iq1_expand_split(c, out[0], out[1], dc_iq1_nibble_lut);
    c = dc_iq1_compact[idx1];
    iq1_expand_split(c, out[2], out[3], dc_iq1_nibble_lut);
    c = dc_iq1_compact[idx2];
    iq1_expand_split(c, out[4], out[5], dc_iq1_nibble_lut);
    c = dc_iq1_compact[idx3];
    iq1_expand_split(c, out[6], out[7], dc_iq1_nibble_lut);
}

// IQ1_S: compact grid (LDS) + nibble LUT (LDS)
__device__ __forceinline__ void decode_iq1s_split_lds(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint16_t* __restrict__ lds_compact,
    const uint16_t* __restrict__ lds_nibble)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint16_t compact = lds_compact[idx];
        iq1_expand_split(compact, out[l * 2], out[l * 2 + 1], lds_nibble);
    }
}

// IQ1_M: compact grid (LDS) + nibble LUT (LDS)
__device__ __forceinline__ void decode_iq1m_split_lds(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint16_t* __restrict__ lds_compact,
    const uint16_t* __restrict__ lds_nibble)
{
    const uint8_t qh0 = payload[4], qh1 = payload[5];
    const int idx0 = payload[0] | ((qh0 & 0x07) << 8);
    const int idx1 = payload[1] | (((qh0 >> 4) & 0x07) << 8);
    const int idx2 = payload[2] | ((qh1 & 0x07) << 8);
    const int idx3 = payload[3] | (((qh1 >> 4) & 0x07) << 8);

    uint16_t c;
    c = lds_compact[idx0];
    iq1_expand_split(c, out[0], out[1], lds_nibble);
    c = lds_compact[idx1];
    iq1_expand_split(c, out[2], out[3], lds_nibble);
    c = lds_compact[idx2];
    iq1_expand_split(c, out[4], out[5], lds_nibble);
    c = lds_compact[idx3];
    iq1_expand_split(c, out[6], out[7], lds_nibble);
}


// =============================================================================
// Strategy B: VSUB4 sign application (llama.cpp pattern)
//   __vsub4(grid ^ signs, signs)  — 2 ops instead of mask expand + xor + add
//   On gfx906: __vsub4 emits v_sub_u32 + v_cndmask (not a native packed op),
//   but the reduced instruction count may still help.
// =============================================================================

// Packed byte-wise subtract: a[i] - b[i] for 4 bytes packed in uint32_t
// On gfx906 this will use SWAR emulation (no native v_pk_sub_i8)
__device__ __forceinline__ uint32_t vsub4_emulated(uint32_t a, uint32_t b)
{
    // SWAR: exploit that borrow doesn't escape bytes because b ∈ {0x00, 0xFF}
    // and a ∈ [1..63] for IQ grid values (always > 0)
    // (a ^ b) - b:  if b_byte=0: a^0 - 0 = a.  if b_byte=0xFF: a^0xFF - 0xFF = ~a - 0xFF = -a (mod 256)
    // But we need saturated or wrapping subtract.  Let's use the direct approach:
    return ((a | 0x80808080u) - b) ^ 0x80808080u;  // XOR-SUB-XOR trick for sign extension
}

// llama.cpp style: expand 8 sign bits to 8 byte masks, apply via vsub4
__device__ __forceinline__ void apply_signs_vsub4_8(
    uint32_t g_lo, uint32_t g_hi, uint8_t signs8,
    int32_t& out_lo, int32_t& out_hi)
{
    // Expand lower 4 sign bits → byte mask
    uint32_t mask_lo =
        (static_cast<uint32_t>(-(signs8 & 1)) & 0xFFu) |
        ((static_cast<uint32_t>(-((signs8 >> 1) & 1)) & 0xFFu) << 8) |
        ((static_cast<uint32_t>(-((signs8 >> 2) & 1)) & 0xFFu) << 16) |
        ((static_cast<uint32_t>(-((signs8 >> 3) & 1)) & 0xFFu) << 24);
    // Upper 4 sign bits → byte mask
    uint32_t mask_hi =
        (static_cast<uint32_t>(-((signs8 >> 4) & 1)) & 0xFFu) |
        ((static_cast<uint32_t>(-((signs8 >> 5) & 1)) & 0xFFu) << 8) |
        ((static_cast<uint32_t>(-((signs8 >> 6) & 1)) & 0xFFu) << 16) |
        ((static_cast<uint32_t>(-((signs8 >> 7) & 1)) & 0xFFu) << 24);

    // __vsub4(grid ^ mask, mask) = conditional negate
    out_lo = static_cast<int32_t>(vsub4_emulated(g_lo ^ mask_lo, mask_lo));
    out_hi = static_cast<int32_t>(vsub4_emulated(g_hi ^ mask_hi, mask_hi));
}

__device__ __forceinline__ uint32_t apply_signs_vsub4_4(uint32_t grid4, uint8_t sign_lo4)
{
    uint32_t mask =
        (static_cast<uint32_t>(-(sign_lo4 & 1)) & 0xFFu) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 1) & 1)) & 0xFFu) << 8) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 2) & 1)) & 0xFFu) << 16) |
        ((static_cast<uint32_t>(-((sign_lo4 >> 3) & 1)) & 0xFFu) << 24);
    return vsub4_emulated(grid4 ^ mask, mask);
}

// IQ2_S with vsub4 signs
__device__ __forceinline__ void decode_iq2s_vsub4(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh_byte = payload[4];
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];
        idx |= (static_cast<int>((qh_byte >> (2 * l)) & 3) << 8);
        const uint64_t grid8 = dc_iq2s_grid[idx];
        const uint8_t signs = payload[5 + l];
        apply_signs_vsub4_8(
            static_cast<uint32_t>(grid8),
            static_cast<uint32_t>(grid8 >> 32),
            signs, out[l * 2], out[l * 2 + 1]);
    }
}

// IQ3_S with vsub4 signs
__device__ __forceinline__ void decode_iq3s_vsub4(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh_byte = payload[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g)
    {
        int idx = payload[g];
        idx |= (static_cast<int>((qh_byte >> g) & 1) << 8);
        const uint32_t grid4 = dc_iq3s_grid[idx];
        const uint8_t sb = payload[9 + g / 2];
        const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
        out[g] = static_cast<int32_t>(apply_signs_vsub4_4(grid4, sn));
    }
}


// =============================================================================
// Strategy C: LDS grid cache — wavefront-cooperative preload
//   Each wavefront (64 threads) preloads a stripe of grid entries into LDS,
//   then all threads in the wavefront read from LDS instead of constant memory.
//   Goal: reduce constant cache pressure by exploiting LDS bandwidth.
// =============================================================================

// LDS tile sizes: one wavefront loads a tile of grid entries
// For IQ2_S (1024 entries × 8B = 8KB), we can fit the whole table in 8KB LDS
// For IQ1_S (2048 entries × 8B = 16KB), we load in 2 passes or use partial cache

// IQ2_S with LDS cache (full table fits in 8KB < 64KB LDS)
__device__ __forceinline__ void decode_iq2s_lds(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint64_t* __restrict__ lds_grid)
{
    const uint8_t qh_byte = payload[4];
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];
        idx |= (static_cast<int>((qh_byte >> (2 * l)) & 3) << 8);
        const uint64_t grid8 = lds_grid[idx];  // Read from LDS instead of constant
        const uint32_t g_lo = static_cast<uint32_t>(grid8);
        const uint32_t g_hi = static_cast<uint32_t>(grid8 >> 32);
        const uint8_t signs = payload[5 + l];
        out[l * 2]     = static_cast<int32_t>(iq_apply_signs_4_baseline(g_lo, signs & 0xF));
        out[l * 2 + 1] = static_cast<int32_t>(iq_apply_signs_4_baseline(g_hi, signs >> 4));
    }
}

// IQ3_S with LDS cache (512 entries × 4B = 2KB, trivially fits)
__device__ __forceinline__ void decode_iq3s_lds(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint32_t* __restrict__ lds_grid)
{
    const uint8_t qh_byte = payload[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g)
    {
        int idx = payload[g];
        idx |= (static_cast<int>((qh_byte >> g) & 1) << 8);
        const uint32_t grid4 = lds_grid[idx];  // Read from LDS
        const uint8_t sb = payload[9 + g / 2];
        const uint8_t sn = (g & 1) ? (sb >> 4) : (sb & 0xF);
        out[g] = static_cast<int32_t>(iq_apply_signs_4_baseline(grid4, sn));
    }
}

// IQ1_S with LDS cache (2048 entries × 8B = 16KB)
__device__ __forceinline__ void decode_iq1s_lds(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out,
    const uint64_t* __restrict__ lds_grid)
{
    const uint16_t qh_word = static_cast<uint16_t>(payload[4])
                           | (static_cast<uint16_t>(payload[5]) << 8);
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int idx = payload[l] | (((qh_word >> (3 * l)) & 7) << 8);
        const uint64_t grid8 = lds_grid[idx];  // Read from LDS
        out[l * 2]     = static_cast<int32_t>(static_cast<uint32_t>(grid8));
        out[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
    }
}


// =============================================================================
// Strategy D: Warp shuffle — share grid lookups across lanes
//   When multiple threads in a wavefront decode consecutive rows, they may
//   hit the same grid indices.  Use __shfl to broadcast one thread's lookup
//   result to others that need it.
//
//   This strategy is speculative — divergent indices are the norm, but
//   for small grid tables (IQ2_XXS: 256 entries), collision probability
//   is higher: P(collision in 64 lanes) ≈ 1 - (255/256)^63 ≈ 22%.
// =============================================================================

// For now, placeholder — the warp shuffle strategy needs careful design
// around which lanes share which indices.  We'll implement a simple
// version: lane 0 broadcasts its grid lookup, others use it if their
// index matches (otherwise fall back to constant memory).
// This is a proof-of-concept; real benefit requires index clustering analysis.


// =============================================================================
// Strategy E: ksigns64 pre-expanded sign table
//   For formats where signs are packed into a single byte (8 sign bits),
//   use a 128-entry × 8-byte LUT that maps sign_byte → 8-byte mask directly.
//   Eliminates the 4-iteration mask expansion loop in iq_apply_signs_4.
//   Only useful for IQ2 formats (8 elements per lookup → 8 sign bits).
// =============================================================================

__device__ __forceinline__ void apply_signs_ksigns64(
    uint32_t g_lo, uint32_t g_hi, uint8_t signs8,
    int32_t& out_lo, int32_t& out_hi)
{
    // Look up pre-expanded mask (8 bytes of 0x00 or 0xFF)
    const uint64_t mask64 = dc_ksigns64[signs8 & 0x7F];
    const uint32_t mask_lo = static_cast<uint32_t>(mask64);
    const uint32_t mask_hi = static_cast<uint32_t>(mask64 >> 32);

    // Apply: (grid ^ mask) + (mask & 0x01010101)
    const uint32_t carry_lo = mask_lo & 0x01010101u;
    const uint32_t carry_hi = mask_hi & 0x01010101u;
    out_lo = static_cast<int32_t>((g_lo ^ mask_lo) + carry_lo);
    out_hi = static_cast<int32_t>((g_hi ^ mask_hi) + carry_hi);
}

// IQ2_S with ksigns64
__device__ __forceinline__ void decode_iq2s_ksigns64(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh_byte = payload[4];
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];
        idx |= (static_cast<int>((qh_byte >> (2 * l)) & 3) << 8);
        const uint64_t grid8 = dc_iq2s_grid[idx];
        const uint8_t signs = payload[5 + l];
        apply_signs_ksigns64(
            static_cast<uint32_t>(grid8),
            static_cast<uint32_t>(grid8 >> 32),
            signs, out[l * 2], out[l * 2 + 1]);
    }
}

// IQ2_XS with ksigns64
__device__ __forceinline__ void decode_iq2xs_ksigns64(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    const uint8_t qh_byte = payload[4];
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];
        idx |= (static_cast<int>((qh_byte >> l) & 1) << 8);
        const uint64_t grid8 = dc_iq2xs_grid[idx];
        const uint8_t signs = payload[5 + l];
        apply_signs_ksigns64(
            static_cast<uint32_t>(grid8),
            static_cast<uint32_t>(grid8 >> 32),
            signs, out[l * 2], out[l * 2 + 1]);
    }
}

// IQ2_XXS with ksigns64
__device__ __forceinline__ void decode_iq2xxs_ksigns64(
    const uint8_t* __restrict__ payload, int32_t* __restrict__ out)
{
    #pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        int idx = payload[l];
        const uint64_t grid8 = dc_iq2xxs_grid[idx];
        const uint8_t signs = payload[4 + l];
        apply_signs_ksigns64(
            static_cast<uint32_t>(grid8),
            static_cast<uint32_t>(grid8 >> 32),
            signs, out[l * 2], out[l * 2 + 1]);
    }
}


// =============================================================================
// Benchmark kernels
// =============================================================================

// Strategy tag for kernel dispatch
enum Strategy : int {
    STRAT_BASELINE     = 0,
    STRAT_VSUB4        = 1,
    STRAT_LDS_CACHE    = 2,
    STRAT_KSIGNS64     = 3,
    STRAT_COMPACT_LUT  = 4,  // IQ1 only: compact 2-bit grid + expansion LUT
    STRAT_COMPACT_SWAR = 5,  // IQ1 only: compact 2-bit grid + SWAR expansion
    STRAT_PREFETCH     = 6,  // IQ1 only: explicit index hoisting + load separation
    STRAT_SWAR_SIGNS   = 7,  // IQ2/IQ3: O4 production SWAR sign application
    STRAT_SPLIT_CONST  = 8,  // IQ1 only: compact grid (const) + nibble LUT (const, 32B)
    STRAT_SPLIT_LDS    = 9,  // IQ1 only: compact grid (LDS) + nibble LUT (LDS, 32B)
    STRAT_COUNT        = 10,
};

static const char* STRATEGY_NAMES[STRAT_COUNT] = {
    "Baseline",
    "vsub4_signs",
    "LDS_cache",
    "ksigns64",
    "compact_lut",
    "compact_swar",
    "prefetch",
    "swar_signs",
    "split_const",
    "split_lds",
};

// Generic decode-only benchmark kernel:
//   - N rows, blocks_per_row 32-element blocks per row
//   - Each thread decodes one block at a time across all N rows
//   - Results are accumulated into a sink to prevent dead-code elimination
//   - No sdot4 / accumulation — pure decode timing
template <IQFormat FMT, Strategy STRAT>
__global__ void bench_decode_kernel(
    const uint8_t* __restrict__ d_payload,
    int32_t*       __restrict__ d_sink,
    int N, int blocks_per_row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    constexpr int payload_bytes = (FMT == FMT_IQ3_S) ? 13 :
                                  (FMT == FMT_IQ3_XXS) ? 12 :
                                  (FMT == FMT_IQ2_S) ? 9 :
                                  (FMT == FMT_IQ2_XS) ? 9 :
                                  (FMT == FMT_IQ2_XXS) ? 8 :
                                  (FMT == FMT_IQ1_S) ? 6 : 10;

    int32_t acc = 0;  // Accumulator sink

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + tid;
        const uint8_t* payload = d_payload + linear * payload_bytes;

        int32_t packed_groups[8];

        // Dispatch to the appropriate decode function
        if constexpr (FMT == FMT_IQ3_S)
        {
            if constexpr (STRAT == STRAT_BASELINE)
                decode_iq3s_baseline(payload, packed_groups);
            else if constexpr (STRAT == STRAT_VSUB4)
                decode_iq3s_vsub4(payload, packed_groups);
            else
                decode_iq3s_baseline(payload, packed_groups);  // fallback
        }
        else if constexpr (FMT == FMT_IQ3_XXS)
        {
            if constexpr (STRAT == STRAT_BASELINE)
                decode_iq3xxs_baseline(payload, packed_groups);
            else
                decode_iq3xxs_baseline(payload, packed_groups);  // only baseline for now
        }
        else if constexpr (FMT == FMT_IQ2_S)
        {
            if constexpr (STRAT == STRAT_BASELINE)
                decode_iq2s_baseline(payload, packed_groups);
            else if constexpr (STRAT == STRAT_VSUB4)
                decode_iq2s_vsub4(payload, packed_groups);
            else if constexpr (STRAT == STRAT_KSIGNS64)
                decode_iq2s_ksigns64(payload, packed_groups);
            else
                decode_iq2s_baseline(payload, packed_groups);
        }
        else if constexpr (FMT == FMT_IQ2_XS)
        {
            if constexpr (STRAT == STRAT_BASELINE)
                decode_iq2xs_baseline(payload, packed_groups);
            else if constexpr (STRAT == STRAT_KSIGNS64)
                decode_iq2xs_ksigns64(payload, packed_groups);
            else
                decode_iq2xs_baseline(payload, packed_groups);
        }
        else if constexpr (FMT == FMT_IQ2_XXS)
        {
            if constexpr (STRAT == STRAT_BASELINE)
                decode_iq2xxs_baseline(payload, packed_groups);
            else if constexpr (STRAT == STRAT_KSIGNS64)
                decode_iq2xxs_ksigns64(payload, packed_groups);
            else
                decode_iq2xxs_baseline(payload, packed_groups);
        }
        else if constexpr (FMT == FMT_IQ1_S)
        {
            if constexpr (STRAT == STRAT_COMPACT_LUT)
                decode_iq1s_compact_lut(payload, packed_groups);
            else if constexpr (STRAT == STRAT_COMPACT_SWAR)
                decode_iq1s_compact_swar(payload, packed_groups);
            else if constexpr (STRAT == STRAT_PREFETCH)
                decode_iq1s_prefetch(payload, packed_groups);
            else if constexpr (STRAT == STRAT_SPLIT_CONST)
                decode_iq1s_split_const(payload, packed_groups);
            else
                decode_iq1s_baseline(payload, packed_groups);
        }
        else if constexpr (FMT == FMT_IQ1_M)
        {
            if constexpr (STRAT == STRAT_COMPACT_LUT)
                decode_iq1m_compact_lut(payload, packed_groups);
            else if constexpr (STRAT == STRAT_COMPACT_SWAR)
                decode_iq1m_compact_swar(payload, packed_groups);
            else if constexpr (STRAT == STRAT_PREFETCH)
                decode_iq1m_prefetch(payload, packed_groups);
            else if constexpr (STRAT == STRAT_SPLIT_CONST)
                decode_iq1m_split_const(payload, packed_groups);
            else
                decode_iq1m_baseline(payload, packed_groups);
        }

        // Accumulate into sink (prevents DCE)
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            acc += packed_groups[g];
    }

    d_sink[tid] = acc;
}

// LDS-preloaded version: separate kernel because LDS declaration is per-kernel
template <IQFormat FMT>
__global__ void bench_decode_lds_kernel(
    const uint8_t* __restrict__ d_payload,
    int32_t*       __restrict__ d_sink,
    int N, int blocks_per_row)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine LDS grid size based on format
    constexpr int grid_entries =
        (FMT == FMT_IQ3_S) ? 512 :
        (FMT == FMT_IQ3_XXS) ? 256 :
        (FMT == FMT_IQ2_S) ? 1024 :
        (FMT == FMT_IQ2_XS) ? 512 :
        (FMT == FMT_IQ2_XXS) ? 256 :
        2048;  // IQ1_S/IQ1_M

    constexpr bool is_iq3 = (FMT == FMT_IQ3_S || FMT == FMT_IQ3_XXS);
    constexpr int entry_bytes = is_iq3 ? 4 : 8;
    constexpr int lds_bytes = grid_entries * entry_bytes;

    // Declare LDS — shared across the workgroup
    __shared__ uint8_t lds_raw[lds_bytes];

    // Cast to appropriate types
    [[maybe_unused]] uint32_t* lds_grid32 = reinterpret_cast<uint32_t*>(lds_raw);
    [[maybe_unused]] uint64_t* lds_grid64 = reinterpret_cast<uint64_t*>(lds_raw);

    // Cooperative preload: each thread loads ceil(grid_entries / blockDim.x) entries
    if constexpr (is_iq3)
    {
        const uint32_t* src = (FMT == FMT_IQ3_S) ? dc_iq3s_grid : dc_iq3xxs_grid;
        for (int i = threadIdx.x; i < grid_entries; i += blockDim.x)
            lds_grid32[i] = src[i];
    }
    else
    {
        const uint64_t* src =
            (FMT == FMT_IQ2_S)   ? dc_iq2s_grid :
            (FMT == FMT_IQ2_XS)  ? dc_iq2xs_grid :
            (FMT == FMT_IQ2_XXS) ? dc_iq2xxs_grid :
            dc_iq1s_grid;
        for (int i = threadIdx.x; i < grid_entries; i += blockDim.x)
            lds_grid64[i] = src[i];
    }
    __syncthreads();

    if (tid >= N) return;

    constexpr int payload_bytes =
        (FMT == FMT_IQ3_S) ? 13 :
        (FMT == FMT_IQ3_XXS) ? 12 :
        (FMT == FMT_IQ2_S) ? 9 :
        (FMT == FMT_IQ2_XS) ? 9 :
        (FMT == FMT_IQ2_XXS) ? 8 :
        (FMT == FMT_IQ1_S) ? 6 : 10;

    int32_t acc = 0;

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + tid;
        const uint8_t* payload = d_payload + linear * payload_bytes;

        int32_t packed_groups[8];

        if constexpr (FMT == FMT_IQ3_S)
            decode_iq3s_lds(payload, packed_groups, lds_grid32);
        else if constexpr (FMT == FMT_IQ3_XXS)
            decode_iq3s_lds(payload, packed_groups, lds_grid32);  // same decode, different grid
        else if constexpr (FMT == FMT_IQ2_S)
            decode_iq2s_lds(payload, packed_groups, lds_grid64);
        else if constexpr (FMT == FMT_IQ1_S || FMT == FMT_IQ1_M)
            decode_iq1s_lds(payload, packed_groups, lds_grid64);
        else  // IQ2_XS, IQ2_XXS — use same decode but different grid already loaded
            decode_iq2s_lds(payload, packed_groups, lds_grid64);

        #pragma unroll
        for (int g = 0; g < 8; ++g)
            acc += packed_groups[g];
    }

    d_sink[tid] = acc;
}


// =============================================================================
// IQ1 Compact LDS kernel: only 4KB LDS needed (vs 16KB for full grid)
//   Preloads compact 2-bit grid to LDS, uses constant-memory LUT for expansion.
//   4KB LDS → up to 16 WGs/CU limited by LDS alone (vs 4 for full grid).
// =============================================================================

template <IQFormat FMT>
__global__ void bench_decode_compact_lds_kernel(
    const uint8_t* __restrict__ d_payload,
    int32_t*       __restrict__ d_sink,
    int N, int blocks_per_row)
{
    static_assert(FMT == FMT_IQ1_S || FMT == FMT_IQ1_M,
                  "compact LDS only for IQ1 formats");

    // Only 4KB LDS (vs 16KB for full grid)
    __shared__ uint16_t lds_compact[2048];  // 2048 × 2B = 4KB

    // Cooperative preload from constant memory
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        lds_compact[i] = dc_iq1_compact[i];
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    constexpr int payload_bytes = (FMT == FMT_IQ1_S) ? 6 : 10;
    int32_t acc = 0;

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + tid;
        const uint8_t* payload = d_payload + linear * payload_bytes;

        int32_t packed_groups[8];

        if constexpr (FMT == FMT_IQ1_S)
            decode_iq1s_compact_lds(payload, packed_groups, lds_compact);
        else
            decode_iq1m_compact_lds(payload, packed_groups, lds_compact);

        #pragma unroll
        for (int g = 0; g < 8; ++g)
            acc += packed_groups[g];
    }

    d_sink[tid] = acc;
}


// =============================================================================
// IQ1 Compact LDS + SWAR kernel: 4KB LDS + pure ALU expansion
//   Avoids the constant-memory LUT that made compact_lds regress.
//   Grid reads from LDS (~1 cycle), expansion via SWAR (~9 VALU per group).
//   4KB LDS = 4× less LDS than full grid → better occupancy potential.
// =============================================================================

template <IQFormat FMT>
__global__ void bench_decode_compact_lds_swar_kernel(
    const uint8_t* __restrict__ d_payload,
    int32_t*       __restrict__ d_sink,
    int N, int blocks_per_row)
{
    static_assert(FMT == FMT_IQ1_S || FMT == FMT_IQ1_M,
                  "compact LDS SWAR only for IQ1 formats");

    __shared__ uint16_t lds_compact[2048];  // 4KB

    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        lds_compact[i] = dc_iq1_compact[i];
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    constexpr int payload_bytes = (FMT == FMT_IQ1_S) ? 6 : 10;
    int32_t acc = 0;

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + tid;
        const uint8_t* payload = d_payload + linear * payload_bytes;

        int32_t packed_groups[8];

        if constexpr (FMT == FMT_IQ1_S)
            decode_iq1s_compact_lds_swar(payload, packed_groups, lds_compact);
        else
            decode_iq1m_compact_lds_swar(payload, packed_groups, lds_compact);

        #pragma unroll
        for (int g = 0; g < 8; ++g)
            acc += packed_groups[g];
    }

    d_sink[tid] = acc;
}


// =============================================================================
// IQ1 Split-table LDS kernel: compact grid (4KB) + nibble LUT (32B) in LDS
//   Total LDS: 4128 bytes — essentially same as compact_lds_swar.
//   Replaces SWAR ALU expansion with 4 tiny LUT reads from LDS.
//   The 32B nibble LUT spans only 8 LDS banks (16 entries × 2B).
// =============================================================================

template <IQFormat FMT>
__global__ void bench_decode_split_lds_kernel(
    const uint8_t* __restrict__ d_payload,
    int32_t*       __restrict__ d_sink,
    int N, int blocks_per_row)
{
    static_assert(FMT == FMT_IQ1_S || FMT == FMT_IQ1_M,
                  "split LDS only for IQ1 formats");

    // 4KB compact grid + 32B nibble LUT = 4128B total LDS
    __shared__ uint16_t lds_compact[2048];  // 4096 bytes
    __shared__ uint16_t lds_nibble[16];     // 32 bytes

    // Cooperative preload of compact grid
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        lds_compact[i] = dc_iq1_compact[i];
    // Preload nibble LUT (only first 16 threads needed)
    if (threadIdx.x < 16)
        lds_nibble[threadIdx.x] = dc_iq1_nibble_lut[threadIdx.x];
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    constexpr int payload_bytes = (FMT == FMT_IQ1_S) ? 6 : 10;
    int32_t acc = 0;

    for (int b = 0; b < blocks_per_row; ++b)
    {
        const size_t linear = static_cast<size_t>(b) * N + tid;
        const uint8_t* payload = d_payload + linear * payload_bytes;

        int32_t packed_groups[8];

        if constexpr (FMT == FMT_IQ1_S)
            decode_iq1s_split_lds(payload, packed_groups, lds_compact, lds_nibble);
        else
            decode_iq1m_split_lds(payload, packed_groups, lds_compact, lds_nibble);

        #pragma unroll
        for (int g = 0; g < 8; ++g)
            acc += packed_groups[g];
    }

    d_sink[tid] = acc;
}


// =============================================================================
// Host-side benchmark infrastructure
// =============================================================================

struct BenchResult {
    const char* format_name;
    const char* strategy_name;
    double min_us;
    double mean_us;
    double blocks_per_sec;  // decode throughput (32-elem blocks / second)
    double elements_per_sec; // elements decoded per second
};

static constexpr int WARMUP_RUNS = 5;
static constexpr int BENCH_RUNS  = 20;
static constexpr int THREADS_PER_BLOCK = 256;

// MI60 specs for reference
static constexpr double HBM2_PEAK_GBPS = 1000.0;

static bool initGridTables()
{
    hipError_t e;
    e = hipMemcpyToSymbol(dc_iq3s_grid,   llaminar2::iq3s_grid,   sizeof(dc_iq3s_grid));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq3s: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(dc_iq3xxs_grid,  llaminar2::iq3xxs_grid, sizeof(dc_iq3xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq3xxs: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(dc_iq2s_grid,    llaminar2::iq2s_grid,   sizeof(dc_iq2s_grid));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq2s: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(dc_iq2xs_grid,   llaminar2::iq2xs_grid,  sizeof(dc_iq2xs_grid));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq2xs: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(dc_iq2xxs_grid,  llaminar2::iq2xxs_grid, sizeof(dc_iq2xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq2xxs: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(dc_iq1s_grid,    llaminar2::iq1s_grid,   sizeof(dc_iq1s_grid));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq1s: %s\n", hipGetErrorString(e)); return false; }

    // Generate and upload ksigns64 table
    uint64_t h_ksigns64[128];
    generate_ksigns64(h_ksigns64);
    e = hipMemcpyToSymbol(dc_ksigns64, h_ksigns64, sizeof(dc_ksigns64));
    if (e != hipSuccess) { fprintf(stderr, "Failed ksigns64: %s\n", hipGetErrorString(e)); return false; }

    // Generate compact IQ1 grid (2-bit ternary encoding, 4KB)
    uint16_t h_iq1_compact[2048];
    generate_iq1_compact_grid(h_iq1_compact, llaminar2::iq1s_grid);
    e = hipMemcpyToSymbol(dc_iq1_compact, h_iq1_compact, sizeof(dc_iq1_compact));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq1_compact: %s\n", hipGetErrorString(e)); return false; }

    // Generate expansion LUT (maps 4×2-bit → 4×int8, 1KB)
    uint32_t h_iq1_expand_lut[256];
    generate_iq1_expand_lut(h_iq1_expand_lut);
    e = hipMemcpyToSymbol(dc_iq1_expand_lut, h_iq1_expand_lut, sizeof(dc_iq1_expand_lut));
    if (e != hipSuccess) { fprintf(stderr, "Failed iq1_expand_lut: %s\n", hipGetErrorString(e)); return false; }

    // Verify compact encoding matches full grid
    if (!verify_iq1_compact_grid(h_iq1_compact, h_iq1_expand_lut, llaminar2::iq1s_grid))
    {
        fprintf(stderr, "FATAL: Compact IQ1 grid verification failed\n");
        return false;
    }

    // Generate and upload nibble split-table LUT (32 bytes)
    uint16_t h_iq1_nibble_lut[16];
    generate_iq1_nibble_lut(h_iq1_nibble_lut);
    e = hipMemcpyToSymbol(dc_iq1_nibble_lut, h_iq1_nibble_lut, sizeof(dc_iq1_nibble_lut));
    if (e != hipSuccess) { fprintf(stderr, "Failed nibble_lut: %s\n", hipGetErrorString(e)); return false; }

    // Verify nibble LUT matches full expand LUT
    if (!verify_iq1_nibble_lut(h_iq1_nibble_lut, h_iq1_expand_lut))
    {
        fprintf(stderr, "FATAL: Nibble LUT verification failed\n");
        return false;
    }

    return true;
}

// Generate random payload data for a given format
static uint8_t* createRandomPayload(IQFormat fmt, int N, int blocks_per_row)
{
    const int payload_bytes = IQ_FORMATS[fmt].payload_bytes;
    const size_t total = static_cast<size_t>(blocks_per_row) * N * payload_bytes;

    uint8_t* h_data = new uint8_t[total];
    std::mt19937 rng(42 + fmt);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < total; ++i)
        h_data[i] = static_cast<uint8_t>(dist(rng));

    // Clamp grid indices to valid range for each format
    const int max_idx_bits =
        (fmt == FMT_IQ3_S) ? 8 :    // 8 in qs + 1 from qh
        (fmt == FMT_IQ3_XXS) ? 8 :  // 8 bits, no qh
        (fmt == FMT_IQ2_S) ? 8 :    // 8 in qs + 2 from qh
        (fmt == FMT_IQ2_XS) ? 8 :   // 8 in qs + 1 from qh
        (fmt == FMT_IQ2_XXS) ? 8 :  // 8 bits, no qh
        8;                            // IQ1: 8 in qs + 3 from qh

    // qs bytes are already 0-255 which is fine for the base index
    // qh bits are random which is also fine (will be masked in decode)

    // For IQ2_S: clamp qh byte bits to valid 2-bit range (already fine, mask in decode)
    // For IQ1_S: ensure qh 3-bit fields stay within 0..7 (already fine, mask in decode)

    return h_data;
}

// Type-erased kernel launcher
using LaunchFn = std::function<void(const uint8_t*, int32_t*, int, int, hipStream_t)>;

template <IQFormat FMT, Strategy STRAT>
void launchDecodeKernel(const uint8_t* d_payload, int32_t* d_sink,
                        int N, int blocks_per_row, hipStream_t stream)
{
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(bench_decode_kernel<FMT, STRAT>),
        dim3(grid), dim3(THREADS_PER_BLOCK), 0, stream,
        d_payload, d_sink, N, blocks_per_row);
}

template <IQFormat FMT>
void launchDecodeLDSKernel(const uint8_t* d_payload, int32_t* d_sink,
                           int N, int blocks_per_row, hipStream_t stream)
{
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(bench_decode_lds_kernel<FMT>),
        dim3(grid), dim3(THREADS_PER_BLOCK), 0, stream,
        d_payload, d_sink, N, blocks_per_row);
}

// IQ1-specific: compact grid in LDS (4KB instead of 16KB)
template <IQFormat FMT>
void launchCompactLDSKernel(const uint8_t* d_payload, int32_t* d_sink,
                            int N, int blocks_per_row, hipStream_t stream)
{
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(bench_decode_compact_lds_kernel<FMT>),
        dim3(grid), dim3(THREADS_PER_BLOCK), 0, stream,
        d_payload, d_sink, N, blocks_per_row);
}

// IQ1-specific: compact grid in LDS + SWAR expansion (no constant LUT)
template <IQFormat FMT>
void launchCompactLDSSWARKernel(const uint8_t* d_payload, int32_t* d_sink,
                                int N, int blocks_per_row, hipStream_t stream)
{
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(bench_decode_compact_lds_swar_kernel<FMT>),
        dim3(grid), dim3(THREADS_PER_BLOCK), 0, stream,
        d_payload, d_sink, N, blocks_per_row);
}

// IQ1-specific: compact grid in LDS + nibble LUT in LDS (split-table)
template <IQFormat FMT>
void launchSplitLDSKernel(const uint8_t* d_payload, int32_t* d_sink,
                          int N, int blocks_per_row, hipStream_t stream)
{
    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(bench_decode_split_lds_kernel<FMT>),
        dim3(grid), dim3(THREADS_PER_BLOCK), 0, stream,
        d_payload, d_sink, N, blocks_per_row);
}

static BenchResult runBenchmark(
    const char* fmt_name, const char* strat_name,
    LaunchFn launch,
    const uint8_t* d_payload, int32_t* d_sink,
    int N, int blocks_per_row, hipStream_t stream)
{
    BenchResult result{};
    result.format_name = fmt_name;
    result.strategy_name = strat_name;

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; ++i)
        launch(d_payload, d_sink, N, blocks_per_row, stream);
    (void)hipDeviceSynchronize();

    // Timed runs
    hipEvent_t ev_start, ev_stop;
    (void)hipEventCreate(&ev_start);
    (void)hipEventCreate(&ev_stop);

    std::vector<double> times_us;
    times_us.reserve(BENCH_RUNS);

    for (int i = 0; i < BENCH_RUNS; ++i)
    {
        (void)hipDeviceSynchronize();
        (void)hipEventRecord(ev_start, stream);
        launch(d_payload, d_sink, N, blocks_per_row, stream);
        (void)hipEventRecord(ev_stop, stream);
        (void)hipEventSynchronize(ev_stop);

        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
        times_us.push_back(static_cast<double>(ms) * 1000.0);
    }

    (void)hipEventDestroy(ev_start);
    (void)hipEventDestroy(ev_stop);

    // Stats
    double sum = 0.0;
    double min_val = times_us[0];
    for (double t : times_us) {
        sum += t;
        if (t < min_val) min_val = t;
    }
    result.min_us = min_val;
    result.mean_us = sum / times_us.size();

    const double total_blocks = static_cast<double>(N) * blocks_per_row;
    result.blocks_per_sec = total_blocks / (result.min_us * 1e-6);
    result.elements_per_sec = result.blocks_per_sec * 32.0;

    return result;
}

// =============================================================================
// Strategies available for each format
// =============================================================================
struct StrategyEntry {
    const char* name;
    LaunchFn launch;
};

template <IQFormat FMT>
std::vector<StrategyEntry> getStrategies()
{
    std::vector<StrategyEntry> strats;

    // Baseline always available
    strats.push_back({"Baseline", launchDecodeKernel<FMT, STRAT_BASELINE>});

    // LDS always available (for all formats)
    strats.push_back({"LDS_cache", launchDecodeLDSKernel<FMT>});

    // vsub4 signs — only for formats with sign application
    if constexpr (FMT == FMT_IQ3_S || FMT == FMT_IQ2_S)
        strats.push_back({"vsub4_signs", launchDecodeKernel<FMT, STRAT_VSUB4>});

    // ksigns64 — only for IQ2 formats (8 sign bits per lookup)
    if constexpr (FMT == FMT_IQ2_S || FMT == FMT_IQ2_XS || FMT == FMT_IQ2_XXS)
        strats.push_back({"ksigns64", launchDecodeKernel<FMT, STRAT_KSIGNS64>});

    // IQ1-specific strategies — the main experiment focus
    if constexpr (FMT == FMT_IQ1_S || FMT == FMT_IQ1_M)
    {
        // Compact 2-bit grid + expansion LUT (constant memory, 5KB total)
        strats.push_back({"compact_lut", launchDecodeKernel<FMT, STRAT_COMPACT_LUT>});
        // Compact 2-bit grid + SWAR bit expansion (no LUT, pure ALU)
        strats.push_back({"compact_swar", launchDecodeKernel<FMT, STRAT_COMPACT_SWAR>});
        // Compact grid in LDS (4KB) + LUT in constant memory
        strats.push_back({"compact_lds", launchCompactLDSKernel<FMT>});
        // Compact grid in LDS (4KB) + SWAR expansion (no constant memory)
        strats.push_back({"cmpct_lds_sw", launchCompactLDSSWARKernel<FMT>});
        // Split-table: compact grid (constant) + 32B nibble LUT (constant)
        strats.push_back({"split_const", launchDecodeKernel<FMT, STRAT_SPLIT_CONST>});
        // Split-table: compact grid (LDS) + 32B nibble LUT (LDS)
        strats.push_back({"split_lds", launchSplitLDSKernel<FMT>});
        // Explicit index hoisting + load separation (full grid)
        strats.push_back({"prefetch", launchDecodeKernel<FMT, STRAT_PREFETCH>});
    }

    return strats;
}


// =============================================================================
// Main benchmark runner
// =============================================================================
int main(int argc, char** argv)
{
    // Check for ROCm device
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0)
    {
        fprintf(stderr, "No ROCm device available, skipping benchmark\n");
        return 0;
    }
    (void)hipSetDevice(0);

    hipDeviceProp_t props;
    (void)hipGetDeviceProperties(&props, 0);
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║          IQ DECODE MICROBENCHMARK — Strategy Lab                  ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Device: %-56s ║\n", props.name);
    fprintf(stderr, "║ Arch:   %-56s ║\n", props.gcnArchName);
    fprintf(stderr, "║ CUs:    %-56d ║\n", props.multiProcessorCount);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n\n");

    // Initialize grid tables
    if (!initGridTables())
    {
        fprintf(stderr, "FATAL: Failed to initialize IQ grid tables\n");
        return 1;
    }
    fprintf(stderr, "[OK] Grid tables + ksigns64 + compact IQ1 grid (4KB) + expand LUT (1KB) + nibble LUT (32B) uploaded\n");
    fprintf(stderr, "     Compact IQ1 verification: PASSED\n");
    fprintf(stderr, "     Nibble LUT verification:  PASSED\n\n");

    // Benchmark shape: 7B FFN_Up (18944×3584) — representative of production workload
    // blocks_per_row = K / 32 = 112
    const int N = 18944;
    const int K = 3584;
    const int blocks_per_row = K / 32;

    fprintf(stderr, "Shape: N=%d × K=%d (%d blocks/row)\n", N, K, blocks_per_row);
    fprintf(stderr, "Timing: %d warmup + %d bench runs, reporting min/mean μs\n\n", WARMUP_RUNS, BENCH_RUNS);

    // Allocate sink buffer on GPU
    int32_t* d_sink = nullptr;
    (void)hipMalloc(&d_sink, N * sizeof(int32_t));

    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    // Collect all results
    std::vector<BenchResult> all_results;

    // ---- Benchmark each format ----
    auto run_format = [&](IQFormat fmt, const std::vector<StrategyEntry>& strats)
    {
        const auto& info = IQ_FORMATS[fmt];
        fprintf(stderr, "━━━ %s (%.1f bpw, %d lookups/block, grid=%d×%dB) ━━━\n",
                info.name, info.bpw, info.lookups_per_block,
                info.grid_entries, info.grid_entry_bytes);

        // Create and upload payload
        uint8_t* h_payload = createRandomPayload(fmt, N, blocks_per_row);
        const size_t payload_total = static_cast<size_t>(blocks_per_row) * N * info.payload_bytes;

        uint8_t* d_payload = nullptr;
        (void)hipMalloc(&d_payload, payload_total);
        (void)hipMemcpy(d_payload, h_payload, payload_total, hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();

        double baseline_min = 0.0;

        for (const auto& strat : strats)
        {
            auto result = runBenchmark(
                info.name, strat.name, strat.launch,
                d_payload, d_sink, N, blocks_per_row, stream);

            if (baseline_min == 0.0)
                baseline_min = result.min_us;

            double speedup = baseline_min / result.min_us;
            fprintf(stderr, "  %-15s  min=%8.1f μs  mean=%8.1f μs  %6.2f Gelem/s  %+.1f%%\n",
                    strat.name,
                    result.min_us, result.mean_us,
                    result.elements_per_sec / 1e9,
                    (speedup - 1.0) * 100.0);

            all_results.push_back(result);
        }
        fprintf(stderr, "\n");

        (void)hipFree(d_payload);
        delete[] h_payload;
    };

    // Run all IQ formats with their applicable strategies
    run_format(FMT_IQ3_S,   getStrategies<FMT_IQ3_S>());
    run_format(FMT_IQ3_XXS, getStrategies<FMT_IQ3_XXS>());
    run_format(FMT_IQ2_S,   getStrategies<FMT_IQ2_S>());
    run_format(FMT_IQ2_XS,  getStrategies<FMT_IQ2_XS>());
    run_format(FMT_IQ2_XXS, getStrategies<FMT_IQ2_XXS>());
    run_format(FMT_IQ1_S,   getStrategies<FMT_IQ1_S>());
    run_format(FMT_IQ1_M,   getStrategies<FMT_IQ1_M>());

    // ---- Summary table ----
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║                          SUMMARY: Best Strategy Per Format                        ║\n");
    fprintf(stderr, "╠═══════════╦═════════════════╦═══════════╦════════════════╦════════════════════════╣\n");
    fprintf(stderr, "║  Format   ║    Strategy     ║  Min (μs) ║  Gelem/s       ║  vs Baseline           ║\n");
    fprintf(stderr, "╠═══════════╬═════════════════╬═══════════╬════════════════╬════════════════════════╣\n");

    // Find best strategy per format
    for (int f = 0; f < FMT_COUNT; ++f)
    {
        const char* fmt_name = IQ_FORMATS[f].name;
        double best_min = 1e30;
        double baseline_min = 1e30;
        const char* best_strat = "?";
        double best_gelem = 0;

        for (const auto& r : all_results)
        {
            if (strcmp(r.format_name, fmt_name) != 0) continue;
            if (r.min_us < best_min) {
                best_min = r.min_us;
                best_strat = r.strategy_name;
                best_gelem = r.elements_per_sec / 1e9;
            }
            if (strcmp(r.strategy_name, "Baseline") == 0)
                baseline_min = r.min_us;
        }

        double speedup = baseline_min / best_min;
        fprintf(stderr, "║ %-9s ║ %-15s ║ %9.1f ║ %12.2f   ║ %+20.1f%%  ║\n",
                fmt_name, best_strat, best_min, best_gelem, (speedup - 1.0) * 100.0);
    }
    fprintf(stderr, "╚═══════════╩═════════════════╩═══════════╩════════════════╩════════════════════════╝\n\n");

    // Cleanup
    (void)hipFree(d_sink);
    (void)hipStreamDestroy(stream);

    fprintf(stderr, "[DONE] IQ decode microbenchmark complete\n");
    return 0;
}
