/**
 * @file Q16RegisterAllocation.h
 * @brief Q16-specific register allocation for JIT attention kernels
 * @author David Sanftenberg
 * @date December 2025
 *
 * Defines the register zone layout and typed aliases for Q16 Integer Attention
 * JIT kernels. This builds on the generic RegisterAllocation.h framework but
 * provides Q16-specific constants and zone purposes.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * Q16 ZMM REGISTER ZONE LAYOUT
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   zmm0-7   │ ACCUMULATOR ZONE   │ INT64 O accumulators (P×V output)
 *            │ [VEX-safe LOW]     │ 8 regs × 64B = 512B (supports head_dim=128)
 *   ─────────┼────────────────────┼───────────────────────────────────────────
 *   zmm8-15  │ INPUT ZONE         │ Q vectors (decode) / Q tile row (prefill)
 *            │ [VEX-safe LOW]     │ Also K/V streaming during dot product
 *   ─────────┼────────────────────┼───────────────────────────────────────────
 *   zmm16-19 │ STATE ZONE         │ Online softmax state (m, l, weight, corr)
 *            │ [EVEX-only HIGH]   │ Persistent across KV iterations
 *   ─────────┼────────────────────┼───────────────────────────────────────────
 *   zmm20-25 │ SCRATCH ZONE       │ Temporaries, intermediate results
 *            │ [EVEX-only HIGH]   │ NOTE: zmm20-23 alias xmm20-23 (ScoreZone)
 *   ─────────┼────────────────────┼───────────────────────────────────────────
 *   zmm26-31 │ RESERVED ZONE      │ Preloaded constants (never modified)
 *            │ [EVEX-only HIGH]   │ scale, LUT base, thresholds, etc.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * XMM OVERLAY (ScoreZone) - FA2 TILING ONLY
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   xmm20-23 │ SCORE ZONE         │ FA2 scalar scores (aliases Scratch0-3)
 *            │ [EVEX-only HIGH]   │ 4 scores for 4×4 micro-tile
 *
 *   CRITICAL: Using ScoreZone invalidates Scratch0-3 until scores consumed!
 *             Safe scratch during scoring: Scratch4 (zmm24), Scratch5 (zmm25)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * Q16 RESERVED CONSTANTS (zmm26-31)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   zmm26 │ Const128   │ 128.0f broadcast    │ Q16 dequant: val × d / 128
 *   zmm27 │ ConstScale │ 1/√head_dim         │ Attention scale factor
 *   zmm28 │ ConstLog2E │ 1.4427f broadcast   │ Exp2FixedSoftmax: log₂(e)
 *   zmm29 │ Const256   │ 256.0f broadcast    │ Exp2FixedSoftmax: LUT index
 *   zmm30 │ ConstOne   │ 1.0f broadcast      │ Various normalization
 *   zmm31 │ ConstZero  │ 0.0f broadcast      │ Initialization, masking
 */

#pragma once

#include "../../../jit/RegisterAllocation.h"
#include "../../../jit/RegisterGuard.h"

namespace llaminar2::kernels::q16_1::jit
{

    // ============================================================================
    // Q16-Specific Zone Configuration
    // ============================================================================

    /**
     * @brief Q16 register zone configuration constants
     */
    struct Q16ZoneConfig
    {
        // Accumulator zone: INT64 O accumulators (zmm0-7)
        static constexpr int ACCUM_BASE = 0;
        static constexpr int ACCUM_COUNT = 8;

        // Input zone: Q/K/V data (zmm8-15)
        static constexpr int INPUT_BASE = 8;
        static constexpr int INPUT_COUNT = 8;
        // Sub-allocation within Input zone:
        // - Input0-3 (zmm8-11): Q vector (persistent during decode)
        // - Input4-7 (zmm12-15): K/V streaming (cyclic reuse)

        // State zone: Online softmax state (zmm16-19)
        static constexpr int STATE_BASE = 16;
        static constexpr int STATE_COUNT = 4;
        // Named state registers:
        static constexpr int STATE_MAX_IDX = 0;    // zmm16: running max score
        static constexpr int STATE_SUM_IDX = 1;    // zmm17: running sum of weights
        static constexpr int STATE_WEIGHT_IDX = 2; // zmm18: current weight (broadcast)
        static constexpr int STATE_CORR_IDX = 3;   // zmm19: correction factor

        // Scratch zone: Temporaries (zmm20-25)
        static constexpr int SCRATCH_BASE = 20;
        static constexpr int SCRATCH_COUNT = 6;
        // CRITICAL: Scratch0-3 (zmm20-23) alias ScoreZone (xmm20-23) for FA2
        // Safe scratch during FA2 scoring: Scratch4 (zmm24), Scratch5 (zmm25)

        // Reserved zone: Constants (zmm26-31)
        static constexpr int RESERVED_BASE = 26;
        static constexpr int RESERVED_COUNT = 6;

        // Q16-specific constant assignments within Reserved zone:
        static constexpr int CONST_128_IDX = 0;   // zmm26: 128.0f (Q16 dequant)
        static constexpr int CONST_SCALE_IDX = 1; // zmm27: 1/sqrt(head_dim)
        static constexpr int CONST_LOG2E_IDX = 2; // zmm28: log2(e) = 1.4427...
        static constexpr int CONST_256_IDX = 3;   // zmm29: 256.0f (LUT index)
        static constexpr int CONST_ONE_IDX = 4;   // zmm30: 1.0f
        static constexpr int CONST_ZERO_IDX = 5;  // zmm31: 0.0f
    };

    // ============================================================================
    // Q16-Specific Register Type Aliases
    // ============================================================================

    // Accumulator zone (zmm0-7): INT64 O accumulators
    using Q16Accum0 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 0>; // zmm0
    using Q16Accum1 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 1>; // zmm1
    using Q16Accum2 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 2>; // zmm2
    using Q16Accum3 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 3>; // zmm3
    using Q16Accum4 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 4>; // zmm4
    using Q16Accum5 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 5>; // zmm5
    using Q16Accum6 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 6>; // zmm6
    using Q16Accum7 = llaminar2::jit::TypedZmm<llaminar2::jit::AccumulatorZone, 7>; // zmm7

    // Input zone (zmm8-15): Q/K/V data
    // Q vector sub-zone (persistent during decode): Input0-3
    using Q16Input0 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 0>; // zmm8  (Q block 0)
    using Q16Input1 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 1>; // zmm9  (Q block 1)
    using Q16Input2 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 2>; // zmm10 (Q block 2)
    using Q16Input3 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 3>; // zmm11 (Q block 3)
    // K/V streaming sub-zone (cyclic reuse): Input4-7
    using Q16Input4 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 4>; // zmm12 (K/V stream 0)
    using Q16Input5 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 5>; // zmm13 (K/V stream 1)
    using Q16Input6 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 6>; // zmm14 (K/V stream 2)
    using Q16Input7 = llaminar2::jit::TypedZmm<llaminar2::jit::QVectorZone, 7>; // zmm15 (K/V stream 3)

    // State zone (zmm16-19): Online softmax state
    using Q16StateMax = llaminar2::jit::TypedZmm<llaminar2::jit::StateZone, 0>;    // zmm16
    using Q16StateSum = llaminar2::jit::TypedZmm<llaminar2::jit::StateZone, 1>;    // zmm17
    using Q16StateWeight = llaminar2::jit::TypedZmm<llaminar2::jit::StateZone, 2>; // zmm18
    using Q16StateCorr = llaminar2::jit::TypedZmm<llaminar2::jit::StateZone, 3>;   // zmm19

    // Scratch zone (zmm20-25): Temporaries
    // NOTE: Scratch0-3 alias ScoreZone for FA2 - do not use simultaneously!
    using Q16Scratch0 = llaminar2::jit::TypedZmm<llaminar2::jit::ScratchZone, 0>; // zmm20
    using Q16Scratch1 = llaminar2::jit::TypedZmm<llaminar2::jit::ScratchZone, 1>; // zmm21
    using Q16Scratch2 = llaminar2::jit::TypedZmm<llaminar2::jit::ScratchZone, 2>; // zmm22
    using Q16Scratch3 = llaminar2::jit::TypedZmm<llaminar2::jit::ScratchZone, 3>; // zmm23
    // Safe scratch (do not alias ScoreZone):
    using Q16Scratch4 = llaminar2::jit::TypedZmm<llaminar2::jit::ScratchZone, 4>; // zmm24
    using Q16Scratch5 = llaminar2::jit::TypedZmm<llaminar2::jit::ScratchZone, 5>; // zmm25

    // Reserved zone (zmm26-31): Q16 constants
    using Q16Const128 = llaminar2::jit::TypedZmm<llaminar2::jit::ReservedZone, 0>;   // zmm26
    using Q16ConstScale = llaminar2::jit::TypedZmm<llaminar2::jit::ReservedZone, 1>; // zmm27
    using Q16ConstLog2E = llaminar2::jit::TypedZmm<llaminar2::jit::ReservedZone, 2>; // zmm28
    using Q16Const256 = llaminar2::jit::TypedZmm<llaminar2::jit::ReservedZone, 3>;   // zmm29
    using Q16ConstOne = llaminar2::jit::TypedZmm<llaminar2::jit::ReservedZone, 4>;   // zmm30
    using Q16ConstZero = llaminar2::jit::TypedZmm<llaminar2::jit::ReservedZone, 5>;  // zmm31

    // ============================================================================
    // ScoreZone XMM Aliases (FA2 tiling only)
    // ============================================================================

    // Score zone overlays low 128-bits of Scratch0-3
    // Used for 4×4 micro-tile scores in FA2 prefill mode
    using Q16Score0 = llaminar2::jit::TypedXmm<llaminar2::jit::ScratchZone, 0>; // xmm20
    using Q16Score1 = llaminar2::jit::TypedXmm<llaminar2::jit::ScratchZone, 1>; // xmm21
    using Q16Score2 = llaminar2::jit::TypedXmm<llaminar2::jit::ScratchZone, 2>; // xmm22
    using Q16Score3 = llaminar2::jit::TypedXmm<llaminar2::jit::ScratchZone, 3>; // xmm23

    // ============================================================================
    // Q16 Tiling Configuration
    // ============================================================================

    /**
     * @brief Q16 tile configuration for FA2-style attention
     *
     * Computed dynamically based on cache hierarchy, but default values
     * are provided for typical server CPUs (Intel Xeon / AMD EPYC).
     */
    struct Q16TileConfig
    {
        int Br = 16;           ///< Query tile size (FA2 prefill)
        int Bc = 64;           ///< KV tile size (FA2 prefill)
        int kv_micro_tile = 4; ///< KV positions per micro-iteration (decode/prefill)
        int head_dim = 128;    ///< Head dimension (determines blocks_per_head)

        // Derived values
        int blocks_per_head() const { return (head_dim + 31) / 32; } // Q16_1 blocks

        /**
         * @brief Compute optimal tile sizes based on cache hierarchy
         */
        static Q16TileConfig compute(int head_dim, size_t l2_size = 256 * 1024)
        {
            Q16TileConfig cfg;
            cfg.head_dim = head_dim;

            // Target: 50% of L2 for working set
            size_t l2_budget = l2_size / 2;

            // Per-query overhead: head_dim × 2 (Q) + head_dim × 8 (O_acc) = 10 × head_dim
            size_t per_query = head_dim * 10;

            // Per-KV overhead: head_dim × 4 (K + V as Q16_1 = 72B blocks)
            //                  + Br × 4 (S) + Br × 2 (P)
            // For Br=16: head_dim × 4 + 96 per KV position
            size_t per_kv = head_dim * 4 + 96;

            // Start with Br=16 (good register blocking)
            cfg.Br = 16;
            size_t q_budget = cfg.Br * per_query;

            // Remaining budget for K/V tiles
            size_t kv_budget = (l2_budget > q_budget) ? (l2_budget - q_budget) : 4096;
            cfg.Bc = static_cast<int>(kv_budget / per_kv);

            // Round down to multiple of 4 (VNNI blocking)
            cfg.Bc = (cfg.Bc / 4) * 4;

            // Clamp to reasonable range
            cfg.Bc = std::max(16, std::min(256, cfg.Bc));

            return cfg;
        }
    };

    // ============================================================================
    // Prefetch Configuration
    // ============================================================================

    /**
     * @brief Prefetch strategy for Q16 attention
     */
    struct Q16PrefetchConfig
    {
        int l1_distance = 4;  ///< KV positions ahead for PREFETCHT0
        int l2_distance = 16; ///< KV positions ahead for PREFETCHT1
        int l3_distance = 64; ///< KV positions ahead for PREFETCHT2

        /**
         * @brief Compute prefetch distances based on head_dim and cache latencies
         */
        static Q16PrefetchConfig compute(int head_dim)
        {
            Q16PrefetchConfig cfg;

            // Bytes per KV position (K only)
            int blocks_per_head = (head_dim + 31) / 32;
            size_t bytes_per_kv = blocks_per_head * 72; // Q16_1 block = 72 bytes

            // L1 prefetch: hide ~4 cycle latency, ~4 cache lines
            cfg.l1_distance = std::max(2, static_cast<int>((4 * 64) / bytes_per_kv));

            // L2 prefetch: hide ~12 cycle latency, ~16 cache lines
            cfg.l2_distance = std::max(8, static_cast<int>((16 * 64) / bytes_per_kv));

            // L3/DRAM prefetch: hide ~100+ cycle latency, ~64 cache lines
            cfg.l3_distance = std::max(32, static_cast<int>((64 * 64) / bytes_per_kv));

            return cfg;
        }
    };

    // ============================================================================
    // Q16 Microkernel Register Contracts
    // ============================================================================

    /**
     * @brief Register contract for Q16DotProduct microkernel
     *
     * INPUTS (read-only):
     *   Input0-3 (zmm8-11): Q vector (4 blocks for head_dim=128)
     *
     * STREAMING (cyclic reuse):
     *   Input4-7 (zmm12-15): K vectors for 4 KV positions
     *
     * OUTPUTS:
     *   Scratch0-3 (zmm20-23): INT32 scores for 4 KV positions
     *
     * SCRATCH:
     *   Scratch4-5 (zmm24-25): Intermediate dot products
     *
     * CONSTANTS:
     *   Const128 (zmm26): Q16 dequant scale
     */
    struct Q16DotProductContract
    {
        // Input zone usage
        static constexpr int Q_REGS_START = 0; // Input0-3
        static constexpr int Q_REGS_COUNT = 4;
        static constexpr int K_REGS_START = 4; // Input4-7
        static constexpr int K_REGS_COUNT = 4;

        // Output zone
        static constexpr int SCORE_REGS_START = 0; // Scratch0-3
        static constexpr int SCORE_REGS_COUNT = 4;

        // Scratch usage (safe during FA2)
        static constexpr int TEMP_REGS_START = 4; // Scratch4-5
        static constexpr int TEMP_REGS_COUNT = 2;
    };

    /**
     * @brief Register contract for Exp2FixedSoftmax microkernel
     *
     * INPUTS (consumed, then overwritten):
     *   Scratch0-3 (zmm20-23): INT32 scores from Q16DotProduct
     *
     * OUTPUTS (overwrite inputs):
     *   Scratch0-3 (zmm20-23): INT16 weights [0, 32767]
     *
     * STATE (persistent across KV iterations):
     *   StateMax (zmm16): Running max score
     *   StateSum (zmm17): Running sum of weights
     *   StateWeight (zmm18): Current max weight
     *   StateCorr (zmm19): Correction factor
     *
     * SCRATCH:
     *   Scratch4-5 (zmm24-25): Delta computation, LUT indexing
     *   Input4-7 (zmm12-15): LUT entries (loaded from memory)
     *
     * CONSTANTS:
     *   ConstLog2E (zmm28): log2(e) for domain conversion
     *   Const256 (zmm29): 256.0f for fractional index
     */
    struct Exp2FixedSoftmaxContract
    {
        // Input/Output (scores → weights)
        static constexpr int SCORE_WEIGHT_REGS_START = 0; // Scratch0-3
        static constexpr int SCORE_WEIGHT_REGS_COUNT = 4;

        // State zone (all 4 registers used)
        static constexpr bool USES_STATE_ZONE = true;

        // Scratch usage
        static constexpr int TEMP_REGS_START = 4; // Scratch4-5
        static constexpr int TEMP_REGS_COUNT = 2;

        // LUT loaded into Input4-7 (cyclically)
        static constexpr int LUT_REGS_START = 4; // Input4-7
        static constexpr int LUT_REGS_COUNT = 4;
    };

    /**
     * @brief Register contract for PVAccumulate microkernel
     *
     * INPUTS:
     *   Scratch0-3 (zmm20-23): INT16 weights from Exp2FixedSoftmax
     *
     * STREAMING:
     *   Input4-7 (zmm12-15): V vectors for 4 KV positions
     *
     * OUTPUTS (accumulate):
     *   Accum0-7 (zmm0-7): INT64 accumulators for head_dim
     *
     * SCRATCH:
     *   Scratch4-5 (zmm24-25): Broadcast weight, intermediate products
     *   Input0-3 (zmm8-11): Widened V values (INT16 → INT32)
     */
    struct PVAccumulateContract
    {
        // Weight inputs
        static constexpr int WEIGHT_REGS_START = 0; // Scratch0-3
        static constexpr int WEIGHT_REGS_COUNT = 4;

        // V streaming
        static constexpr int V_REGS_START = 4; // Input4-7
        static constexpr int V_REGS_COUNT = 4;

        // Accumulator outputs
        static constexpr int ACCUM_REGS_START = 0; // Accum0-7
        static constexpr int ACCUM_REGS_COUNT = 8;

        // Scratch usage
        static constexpr int TEMP_REGS_START = 4; // Scratch4-5
        static constexpr int TEMP_REGS_COUNT = 2;

        // Widened V temporary
        static constexpr int WIDE_V_REGS_START = 0; // Input0-3
        static constexpr int WIDE_V_REGS_COUNT = 4;
    };

    /**
     * @brief Register contract for WoProjectionVNNI microkernel
     *
     * INPUTS:
     *   Input0-7 (zmm8-15): O_int32 context (requantized to INT16)
     *
     * STREAMING:
     *   Scratch0-3 (zmm20-23): Wo weight rows (Q8_0 → INT16 sign-extended)
     *
     * OUTPUTS:
     *   (to memory): Q16_1 projection blocks
     *
     * SCRATCH:
     *   Scratch4-5 (zmm24-25): Dequant temps, FMA intermediates
     *   Accum4-7 (zmm4-7): Additional accumulators for unrolling
     *
     * CONSTANTS:
     *   Const128 (zmm26): Q8_0 dequant scale
     */
    struct WoProjectionContract
    {
        // Context inputs
        static constexpr int CONTEXT_REGS_START = 0; // Input0-7
        static constexpr int CONTEXT_REGS_COUNT = 8;

        // Weight streaming
        static constexpr int WEIGHT_REGS_START = 0; // Scratch0-3
        static constexpr int WEIGHT_REGS_COUNT = 4;

        // Scratch usage
        static constexpr int TEMP_REGS_START = 4; // Scratch4-5
        static constexpr int TEMP_REGS_COUNT = 2;

        // Additional accumulators for unrolling
        static constexpr int EXTRA_ACCUM_START = 4; // Accum4-7
        static constexpr int EXTRA_ACCUM_COUNT = 4;
    };

} // namespace llaminar2::kernels::q16_1::jit
