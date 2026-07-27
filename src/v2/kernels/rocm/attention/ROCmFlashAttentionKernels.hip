/**
 * @file ROCmFlashAttentionKernels.hip
 * @brief HIP device kernels for Flash Attention 2 and Flash Decoding on MI50
 *
 * This file contains the HIP device kernels and extern "C" wrapper functions
 * called from the C++ implementation file.
 *
 * Target Architecture: AMD MI50 (gfx906 / Vega 20)
 *
 * Algorithms implemented:
 * - Flash Attention 2: Tiled attention with online softmax
 *   - Uses v_dot2_f32_f16 for native FP16 dot products with FP32 accumulation
 *   - LDS double-buffering for overlapped load/compute
 *   - Explicit wavefront reductions via DPP
 *   - NO WMMA/Matrix Cores - pure VALU implementation
 * - Flash Decoding: Split-K parallelism for single-token decode
 *
 * MI50 Optimization Strategy:
 *   1. Maximize LDS utilization (64KB per CU)
 *   2. Use FP16 packed math where possible for 2x ALU throughput
 *   3. Wavefront-wide reductions using __shfl_xor (DPP under the hood)
 *   4. Balance VGPR usage for occupancy (256 VGPRs per wavefront limit)
 *   5. Explicit prefetching via buffer load instructions
 *
 * @author David Sanftenberg
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include "../../attention/AttentionDeviceParams.h"
#include "../../../tensors/BlockStructures.h"
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../utils/DebugEnv.h"

namespace
{
    // =========================================================================
    // Constants - MI50 Specific
    // =========================================================================

    // Wavefront size on GCN/CDNA architectures
    constexpr int WAVEFRONT_SIZE = 64;

    // LDS (Local Data Share) size per CU on MI50
    constexpr int LDS_SIZE_BYTES = 65536; // 64KB

    // Maximum VGPRs per wavefront (for occupancy tuning)
    constexpr int MAX_VGPRS_PER_WAVEFRONT = 256;

    /// Maximum compact verifier rows represented by DEVICE_PARAMS.
    constexpr int MAX_DYNAMIC_ATTENTION_PARAM_ROWS =
        llaminar2::attention::kMaxGroupedVerifierAttentionRows;

    // =========================================================================
    // KV Cache Type Tags for Template Dispatch
    // =========================================================================
    //
    // The decode kernel is templated on KV_TYPE so it can read K/V
    // directly in their native storage format.
    //   FP32: K/V are float*    (4 bytes/element) — baseline
    //   FP16: K/V are __half*   (2 bytes/element) — 2× BW savings + fdot2
    //   BF16: K/V are hip_bfloat16* (2 bytes/element) — native FP32 conversion
    //   Q8_1: K/V are Q8_1Block* (36 bytes/32 elements) — ~3.5× BW savings
    enum class KVTypeTag { FP32, FP16, BF16, Q8_1 };

    /**
     * @brief Derive graph-captured attention params from live KV cache count.
     *
     * This tiny helper runs in graph order immediately before attention, after
     * KV append has advanced the device-owned cache count. The resulting
     * `AttentionDeviceParams` block remains device-owned through MTP replay.
     */
    __global__ void hip_derive_attention_params_from_cached_tokens_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        const int *__restrict__ post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride)
    {
        const int row = static_cast<int>(threadIdx.x);
        if (!out || !post_append_cached_tokens || row >= query_rows)
            return;

        const int raw_kv_len = *post_append_cached_tokens;
        const int kv_len = raw_kv_len > 1 ? raw_kv_len : 1;
        const int logical_seq_len = seq_len > 1 ? seq_len : 1;
        const int base_position_raw = kv_len - logical_seq_len;
        const int base_position = base_position_raw > 0 ? base_position_raw : 0;
        const int row_kv_len_raw = kv_len - (query_rows - 1 - row);
        const int row_kv_len = row_kv_len_raw > 1 ? row_kv_len_raw : 1;
        out[row].kv_len = row_kv_len;
        out[row].kv_stride = kv_stride > kv_len ? kv_stride : kv_len;
        out[row].position_offset = base_position + row;
        out[row].mask_stride = kv_len;
    }

    /**
     * @brief Materialize explicit attention geometry directly in device memory.
     *
     * This is the stream-owned counterpart to the device-count path above for
     * eager tests and callers that already possess fixed launch geometry.  A
     * tiny HIP kernel writes the canonical parameter records, so capture never
     * depends on a pinned host allocation or an H2D parameter transfer.
     *
     * Earlier verifier rows expose successively shorter KV prefixes, matching
     * independent serial decode calls byte for byte.  Absolute positions still
     * advance from @p position_offset and all rows share the full KV mask
     * stride supplied by @p kv_len.
     */
    __global__ void hip_write_attention_params_from_geometry_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows)
    {
        const int row = static_cast<int>(threadIdx.x);
        if (!out || row >= query_rows)
            return;

        const int row_kv_len_raw = kv_len - (query_rows - 1 - row);
        const int row_kv_len = row_kv_len_raw > 1 ? row_kv_len_raw : 1;
        out[row].kv_len = row_kv_len;
        out[row].kv_stride = kv_stride > kv_len ? kv_stride : kv_len;
        out[row].position_offset = position_offset + row;
        out[row].mask_stride = kv_len;
    }

    /**
     * @brief Derive request-major attention metadata from canonical device counts.
     *
     * The resident request cache exposes one post-append token count per
     * request.  A request may contribute several compact verifier rows, so an
     * earlier row must see a shorter prefix than the final row.  Flattening the
     * request/query coordinates here keeps that visibility rule entirely on
     * the execution stream and makes the operation legal inside HIP graph
     * capture and replay.
     *
     * @param out Device workspace containing one parameter record per flat row.
     * @param post_append_cached_tokens Contiguous cache-owned request counts.
     * @param request_count Number of independent request cache banks.
     * @param query_rows Number of consecutive decode rows owned by each request.
     */
    __global__ void hip_derive_attention_params_from_request_counts_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        const int *__restrict__ post_append_cached_tokens,
        int request_count,
        int query_rows,
        int kv_stride)
    {
        const int flat_row = static_cast<int>(threadIdx.x);
        const int total_rows = request_count * query_rows;
        if (!out || !post_append_cached_tokens || flat_row >= total_rows)
            return;

        const int request = flat_row / query_rows;
        const int query_row = flat_row - request * query_rows;
        const int raw_post_append_kv_len = post_append_cached_tokens[request];
        const int post_append_kv_len =
            raw_post_append_kv_len > 1 ? raw_post_append_kv_len : 1;
        const int base_position_raw = post_append_kv_len - query_rows;
        const int base_position = base_position_raw > 0 ? base_position_raw : 0;
        const int row_kv_len_raw =
            post_append_kv_len - (query_rows - 1 - query_row);
        const int row_kv_len = row_kv_len_raw > 1 ? row_kv_len_raw : 1;

        out[flat_row].kv_len = row_kv_len;
        out[flat_row].kv_stride =
            kv_stride > post_append_kv_len ? kv_stride : post_append_kv_len;
        out[flat_row].position_offset = base_position + query_row;
        out[flat_row].mask_stride = post_append_kv_len;
    }

    // =========================================================================
    // Flash Attention 2 Configuration for MI50
    // =========================================================================
    //
    // Key design decisions for MI50 (no Matrix Cores):
    //   1. Tile sizes chosen to balance LDS usage and occupancy
    //   2. Use vectorized FP16 for Q@K^T computation
    //   3. FP32 accumulation for numerical stability
    //   4. Double-buffered K/V loading to hide memory latency
    //
    // LDS Layout:
    //   Q_tile:    [TILE_Q, head_dim] in FP16
    //   K_buffer:  [2, TILE_KV, head_dim] in FP16 (double-buffered)
    //   V_buffer:  [2, TILE_KV, head_dim] in FP16 (double-buffered)
    //   Scores:    [TILE_Q, TILE_KV] in FP32
    //
    // =========================================================================

    // Tile sizes for prefill kernel
    // Chosen to fit comfortably in 64KB LDS with good occupancy
    constexpr int FA2_TILE_Q = 64;   // Q tile rows
    constexpr int FA2_TILE_KV = 64;  // KV tile rows
    constexpr int FA2_NUM_STAGES = 2; // Double buffering

    // LDS padding to avoid bank conflicts (32 banks, 4-byte granularity)
    // The critical access pattern: threads in a wavefront access DIFFERENT ROWS
    // at the SAME column (e.g., different K rows in the Q@K^T dot product).
    // Bank index = (row × stride_dwords + col) mod 32.
    // To avoid conflicts, stride_dwords must be coprime with 32 (i.e., odd).
    //
    // For FP16: stride_dwords = (head_dim + pad) / 2
    //   head_dim=128: (128+2)/2 = 65 → 65 mod 32 = 1 → gcd(1,32) = 1 ✓
    //   head_dim=64:  (64+2)/2  = 33 → 33 mod 32 = 1 → gcd(1,32) = 1 ✓
    // For FP32: stride_dwords = tile_kv + pad
    //   tile_kv=32: 32+1 = 33 → coprime with 32 ✓
    //   tile_kv=64: 64+1 = 65 → coprime with 32 ✓
    //
    // Previous values (8/4) gave stride_dwords ≡ 4 (mod 32) → gcd(4,32)=4
    // → 8-way bank conflicts on K/V reads in the dot product inner loop!
    constexpr int FA2_LDS_PAD_FP16 = 2;
    constexpr int FA2_LDS_PAD_FP32 = 1;

    // Block configuration
    // We use 256 threads = 4 wavefronts per block for good occupancy
    constexpr int FA2_THREADS_PER_BLOCK = 256;
    constexpr int FA2_WAVEFRONTS_PER_BLOCK = FA2_THREADS_PER_BLOCK / WAVEFRONT_SIZE; // 4

    // =========================================================================
    // Flash Decoding Configuration for MI50
    // =========================================================================

    // Number of splits for Flash Decoding
    // More splits = more parallelism but more reduction overhead
    constexpr int FD_DEFAULT_NUM_SPLITS = 8;

    // Threads per block for decode kernel
    constexpr int FD_THREADS_PER_BLOCK = 256;

    // Max splits supported by reduction kernel shared-memory staging
    constexpr int FD_MAX_SPLITS_SUPPORTED = 32;

    constexpr int FD_TARGET_ELEMS_PER_SPLIT = 16384;
    constexpr int FD_MIN_SPLITS = 1;
    constexpr int FD_AUTOTUNE_MAX_TRIALS = 6;

    inline int clampInt(int v, int lo, int hi)
    {
        return (v < lo) ? lo : ((v > hi) ? hi : v);
    }

    inline int roundUpPow2Clamped(int value, int max_allowed)
    {
        int p = 1;
        while (p < value && p < max_allowed)
        {
            p <<= 1;
        }
        return clampInt(p, 1, max_allowed);
    }

    inline int chooseDecodeThreadsPerBlock(int head_dim, int kv_len, int effective_num_splits)
    {
        // Generic heuristic using work-per-split and split aspect.
        const int safe_head_dim = std::max(head_dim, 1);
        const int safe_splits = std::max(effective_num_splits, 1);
        const int rows_per_split = std::max(1, kv_len / safe_splits);
        const int64_t elems_per_split =
            (static_cast<int64_t>(kv_len) * safe_head_dim) / safe_splits;

        // Small per-split work or very short split rows favor lower TPB.
        if ((rows_per_split <= 64 && elems_per_split <= 16384) ||
            (rows_per_split <= 48 && elems_per_split <= 32768))
        {
            return 128;
        }

        return 256;
    }

    inline int chooseDecodeSplitsAuto(int kv_len, int head_dim, int max_allowed,
                                      int n_heads = 0, int num_cus = 0)
    {
        const int safe_head_dim = std::max(head_dim, 1);
        const int64_t total_elements = static_cast<int64_t>(kv_len) * safe_head_dim;

        // Target work-per-split (elements), architecture-agnostic default.
        const int target_elements_per_split = clampInt(
            FD_TARGET_ELEMS_PER_SPLIT,
            1024,
            1 << 20);

        int base_splits = static_cast<int>((total_elements + target_elements_per_split - 1) /
                                           target_elements_per_split);
        base_splits = std::max(base_splits, 1);

        // Aspect-ratio adjustment: deeper KV relative to head_dim benefits from more splits.
        // Uses ratio kv_len/head_dim (not hard-coded model dimensions).
        if (kv_len >= safe_head_dim * 8)
        {
            base_splits = (base_splits * 3 + 1) / 2; // ~1.5x
        }
        else if (kv_len <= safe_head_dim * 2)
        {
            base_splits = std::max(1, (base_splits + 1) / 2);
        }

        // Don't over-split beyond useful row granularity.
        base_splits = std::min(base_splits, std::max(kv_len, 1));

        // Prefer power-of-two split counts for predictable reduction behavior.
        int auto_splits = roundUpPow2Clamped(base_splits, max_allowed);

        // CU-aware occupancy floor: when n_heads * splits is too low to fill
        // the GPU, increase splits to target ~75% CU occupancy.  This is
        // critical for TP configurations where per-rank head count drops.
        if (n_heads > 0 && num_cus > 0)
        {
            const int target_blocks = (num_cus * 3) / 4; // 75% occupancy
            if (n_heads * auto_splits < target_blocks)
            {
                int occ_splits = (target_blocks + n_heads - 1) / n_heads;
                occ_splits = roundUpPow2Clamped(occ_splits, max_allowed);
                auto_splits = std::max(auto_splits, occ_splits);
            }
        }

        const int min_splits = clampInt(FD_MIN_SPLITS, 1, max_allowed);
        auto_splits = clampInt(auto_splits, min_splits, max_allowed);
        return auto_splits;
    }

    inline int chooseEffectiveDecodeSplits(int kv_len, int head_dim, int requested_num_splits,
                                           int n_heads = 0, int num_cus = 0)
    {
        const int workspace_limit = (requested_num_splits > 0)
            ? requested_num_splits
            : FD_MAX_SPLITS_SUPPORTED;
        const int max_allowed = std::max(1, std::min(FD_MAX_SPLITS_SUPPORTED, workspace_limit));

        const int auto_splits = chooseDecodeSplitsAuto(kv_len, head_dim, max_allowed, n_heads, num_cus);
        return std::max(1, std::min(auto_splits, max_allowed));
    }

    struct DecodeLaunchConfig
    {
        int splits = FD_DEFAULT_NUM_SPLITS;
        int tpb = FD_THREADS_PER_BLOCK;
    };

    struct DecodeAutotuneSelection
    {
        DecodeLaunchConfig config{};
        bool should_time = false;
        uint32_t key = 0;
        int candidate_index = -1;
    };

    struct DecodeAutotuneState
    {
        std::vector<DecodeLaunchConfig> candidates;
        int max_trials = 0;
        int next_trial = 0;
        bool has_best = false;
        DecodeLaunchConfig best{};
        float best_ms = std::numeric_limits<float>::infinity();
    };

    inline int ilog2Floor(int value)
    {
        int v = std::max(value, 1);
        int lg = 0;
        while (v > 1)
        {
            v >>= 1;
            lg++;
        }
        return lg;
    }

    inline int bucketAspectRatio(int kv_len, int head_dim)
    {
        const int safe_hd = std::max(head_dim, 1);
        const float ratio = static_cast<float>(kv_len) / static_cast<float>(safe_hd);
        if (ratio <= 1.0f)
            return 0;
        if (ratio <= 2.0f)
            return 1;
        if (ratio <= 4.0f)
            return 2;
        if (ratio <= 8.0f)
            return 3;
        if (ratio <= 16.0f)
            return 4;
        if (ratio <= 32.0f)
            return 5;
        return 6;
    }

    inline uint32_t makeDecodeAutotuneKey(int kv_len, int head_dim, int max_allowed_splits,
                                          int n_heads = 0)
    {
        const int64_t total_elements = static_cast<int64_t>(std::max(kv_len, 1)) * std::max(head_dim, 1);
        const int elem_bucket = clampInt(ilog2Floor(static_cast<int>(std::min<int64_t>(total_elements, INT32_MAX))), 0, 31);
        const int aspect_bucket = bucketAspectRatio(kv_len, head_dim); // 0..6
        const int hd_bucket = clampInt(ilog2Floor(std::max(head_dim, 1)), 0, 15);
        const int split_cap_bucket = clampInt(ilog2Floor(std::max(max_allowed_splits, 1)), 0, 15);
        const int heads_bucket = clampInt(ilog2Floor(std::max(n_heads, 1)), 0, 15);

        return static_cast<uint32_t>(
            (elem_bucket & 0x1F) |
            ((aspect_bucket & 0x07) << 5) |
            ((hd_bucket & 0x0F) << 8) |
            ((split_cap_bucket & 0x0F) << 12) |
            ((heads_bucket & 0x0F) << 16));
    }

    inline void addSplitCandidate(std::vector<int> &splits, int value, int max_allowed)
    {
        int clamped = clampInt(value, 1, max_allowed);
        clamped = roundUpPow2Clamped(clamped, max_allowed);
        if (std::find(splits.begin(), splits.end(), clamped) == splits.end())
        {
            splits.push_back(clamped);
        }
    }

    inline std::vector<DecodeLaunchConfig> buildDecodeAutotuneCandidates(int kv_len, int head_dim, int max_allowed,
                                                                         int n_heads = 0, int num_cus = 0)
    {
        const int min_splits = clampInt(FD_MIN_SPLITS, 1, max_allowed);

        const int base_splits = chooseDecodeSplitsAuto(kv_len, head_dim, max_allowed, n_heads, num_cus);

        std::vector<int> split_candidates;
        split_candidates.reserve(12);
        addSplitCandidate(split_candidates, min_splits, max_allowed);
        addSplitCandidate(split_candidates, base_splits / 2, max_allowed);
        addSplitCandidate(split_candidates, base_splits, max_allowed);
        addSplitCandidate(split_candidates, base_splits * 2, max_allowed);
        addSplitCandidate(split_candidates, 1, max_allowed);
        addSplitCandidate(split_candidates, max_allowed, max_allowed);

        // Occupancy-aware candidates: when the base heuristic has already been
        // bumped by the CU floor, also explore splits that hit 50% and 100%
        // occupancy so the autotuner can bracket the sweet spot.
        if (n_heads > 0 && num_cus > 0)
        {
            for (int pct : {50, 100})
            {
                int target = (num_cus * pct) / 100;
                int s = (target + n_heads - 1) / n_heads;
                addSplitCandidate(split_candidates, s, max_allowed);
            }
        }

        std::sort(split_candidates.begin(), split_candidates.end());

        const int preferred_tpb = chooseDecodeThreadsPerBlock(head_dim, kv_len, base_splits);
        int alt_tpb = (preferred_tpb == 128) ? 256 : 128;

        std::vector<DecodeLaunchConfig> candidates;
        candidates.reserve(split_candidates.size() * 2 + 2);

        auto addCandidate = [&](int splits, int tpb)
        {
            DecodeLaunchConfig cfg{splits, tpb};
            auto it = std::find_if(candidates.begin(), candidates.end(), [&](const DecodeLaunchConfig &existing)
                                   { return existing.splits == cfg.splits && existing.tpb == cfg.tpb; });
            if (it == candidates.end())
            {
                candidates.push_back(cfg);
            }
        };

        addCandidate(base_splits, preferred_tpb);
        addCandidate(base_splits, alt_tpb);

        for (int s : split_candidates)
        {
            addCandidate(s, preferred_tpb);
            addCandidate(s, alt_tpb);
        }

        return candidates;
    }

    inline std::unordered_map<uint32_t, DecodeAutotuneState> &decodeAutotuneMap()
    {
        static std::unordered_map<uint32_t, DecodeAutotuneState> states;
        return states;
    }

    inline std::mutex &decodeAutotuneMutex()
    {
        static std::mutex lock;
        return lock;
    }

    inline DecodeAutotuneSelection chooseDecodeLaunchConfig(int kv_len, int head_dim, int requested_num_splits,
                                                             int n_heads = 0, int num_cus = 0)
    {
        DecodeAutotuneSelection selection;

        const int heuristic_splits = chooseEffectiveDecodeSplits(kv_len, head_dim, requested_num_splits, n_heads, num_cus);
        const int heuristic_tpb = chooseDecodeThreadsPerBlock(head_dim, kv_len, heuristic_splits);
        selection.config = {heuristic_splits, heuristic_tpb};

        const int workspace_limit = (requested_num_splits > 0)
            ? requested_num_splits
            : FD_MAX_SPLITS_SUPPORTED;
        const int max_allowed = std::max(1, std::min(FD_MAX_SPLITS_SUPPORTED, workspace_limit));

        const auto &rocm_env = llaminar2::debugEnv().rocm;
        if (rocm_env.fa_decode_num_splits_present)
        {
            const int forced_splits = rocm_env.fa_decode_num_splits
                ? clampInt(*rocm_env.fa_decode_num_splits, 1, max_allowed)
                : heuristic_splits;
            const int forced_tpb = rocm_env.fa_decode_tpb
                ? clampInt(*rocm_env.fa_decode_tpb, 64, 256)
                : chooseDecodeThreadsPerBlock(head_dim, kv_len, forced_splits);
            selection.config = {forced_splits, forced_tpb};
            return selection;
        }

        if (!rocm_env.fa_decode_autotune || llaminar2::debugEnv().gemm.deterministic)
        {
            // MTP verifier decode requires the grouped row path and the serial
            // M=1 row path to use the same reduction shape. Online autotune
            // trial rotation is useful during deliberate profiling, but it is
            // process-local performance state rather than model state. If it is
            // active in production, two identical decode requests can execute
            // different split/TPB candidates and become only numerically close
            // instead of byte-for-byte identical.
            return selection;
        }

        const uint32_t key = makeDecodeAutotuneKey(kv_len, head_dim, max_allowed, n_heads);

        std::lock_guard<std::mutex> guard(decodeAutotuneMutex());
        auto &states = decodeAutotuneMap();
        auto &state = states[key];

        if (state.candidates.empty())
        {
            state.candidates = buildDecodeAutotuneCandidates(kv_len, head_dim, max_allowed, n_heads, num_cus);
            const int max_trials_env = clampInt(FD_AUTOTUNE_MAX_TRIALS, 1, 24);
            state.max_trials = std::min<int>(max_trials_env, static_cast<int>(state.candidates.size()));
            state.next_trial = 0;
            state.has_best = false;
            state.best_ms = std::numeric_limits<float>::infinity();
            state.best = selection.config;
        }

        selection.key = key;

        if (state.next_trial < state.max_trials)
        {
            selection.config = state.candidates[state.next_trial];
            selection.should_time = true;
            selection.candidate_index = state.next_trial;
            state.next_trial++;
            return selection;
        }

        if (state.has_best)
        {
            selection.config = state.best;
        }
        else
        {
            selection.config = state.best;
        }

        return selection;
    }

    inline void recordDecodeAutotuneResult(uint32_t key, int candidate_index, float elapsed_ms)
    {
        if (!(elapsed_ms > 0.0f))
        {
            return;
        }

        std::lock_guard<std::mutex> guard(decodeAutotuneMutex());
        auto &states = decodeAutotuneMap();
        auto it = states.find(key);
        if (it == states.end())
        {
            return;
        }

        auto &state = it->second;
        if (candidate_index < 0 || candidate_index >= static_cast<int>(state.candidates.size()))
        {
            return;
        }

        if (!state.has_best || elapsed_ms < state.best_ms)
        {
            state.has_best = true;
            state.best_ms = elapsed_ms;
            state.best = state.candidates[candidate_index];
        }
    }

    // =========================================================================
    // Device Property Caching
    // =========================================================================

    struct MI50DeviceConfig
    {
        int gcn_arch = 0;      // e.g., 906 for gfx906
        int max_lds = 0;       // Max LDS per block
        int max_vgprs = 0;     // Max VGPRs per wavefront
        int num_cus = 60;      // Number of compute units
        bool initialized = false;
    };

    // Per-device cached config
    static MI50DeviceConfig g_device_config[8]; // Support up to 8 GPUs

    /**
     * @brief Get cached device configuration (lazy init)
     */
    inline MI50DeviceConfig &getDeviceConfig(int device)
    {
        if (device < 0 || device >= 8)
            device = 0;
        MI50DeviceConfig &cfg = g_device_config[device];

        if (!cfg.initialized)
        {
            hipDeviceProp_t prop;
            (void)hipGetDeviceProperties(&prop, device);

            // Parse gcnArchName string (e.g., "gfx906") to extract arch number
            // gcnArchName format: "gfx" + arch_number
            cfg.gcn_arch = 0;
            if (strncmp(prop.gcnArchName, "gfx", 3) == 0) {
                cfg.gcn_arch = atoi(prop.gcnArchName + 3);
            }
            cfg.max_lds = static_cast<int>(prop.sharedMemPerBlock);
            cfg.max_vgprs = MAX_VGPRS_PER_WAVEFRONT; // Could query via rocm-smi
            cfg.num_cus = prop.multiProcessorCount;
            cfg.initialized = true;
        }
        return cfg;
    }

    // =========================================================================
    // HIP-specific Intrinsics and Helpers
    // =========================================================================

    /**
     * @brief Wavefront-wide reduction using DPP (Data Parallel Primitives)
     *
     * On AMD GPUs, __shfl_xor maps to DPP instructions for fast cross-lane ops.
     * Wavefront size is 64, so we need 6 steps for full reduction.
     */
    __device__ __forceinline__ float wavefrontReduceSum(float val)
    {
        // Butterfly reduction pattern for wavefront size 64
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        {
            val += __shfl_xor(val, offset);
        }
        return val;
    }

    __device__ __forceinline__ float wavefrontReduceMax(float val)
    {
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        {
            val = fmaxf(val, __shfl_xor(val, offset));
        }
        return val;
    }

    // exp2f(x * LOG2E) maps to hardware v_exp_f32 on GCN.
    // Forward-declared here so both prefill and decode kernels can use it.
    __device__ __forceinline__ float fast_expf(float x);

    /**
     * @brief Warp-level reduction of softmax state (m, l, O) for Flash Decoding
     *
     * Reduces across all 64 lanes in a wavefront.
     */
    __device__ void wavefrontReduceSoftmaxState(
        float &m, float &l, float *O, int head_dim)
    {
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        {
            float other_m = __shfl_xor(m, offset);
            float other_l = __shfl_xor(l, offset);

            float m_new = fmaxf(m, other_m);
            float scale_self = expf(m - m_new);
            float scale_other = expf(other_m - m_new);

            l = scale_self * l + scale_other * other_l;

            for (int d = 0; d < head_dim; d++)
            {
                float other_O = __shfl_xor(O[d], offset);
                O[d] = scale_self * O[d] + scale_other * other_O;
            }

            m = m_new;
        }
    }

    /**
     * @brief Convert FP32 to FP16 using HIP intrinsics
     */
    __device__ __forceinline__ __half float_to_half(float x)
    {
        return __float2half(x);
    }

    /**
     * @brief Convert FP16 to FP32 using HIP intrinsics
     */
    __device__ __forceinline__ float half_to_float(__half x)
    {
        return __half2float(x);
    }

    // =========================================================================
    // Flash Attention 2 - Prefill Kernel for MI50
    // =========================================================================
    //
    // This implementation uses vectorized FP16 operations since MI50 lacks
    // Matrix Cores. Key optimizations:
    //   1. Vectorized FP16 dot products (2 elements at a time)
    //   2. LDS double-buffering for K/V tiles
    //   3. Online softmax to avoid materializing full attention matrix
    //   4. FP32 accumulators for numerical stability
    //
    // Grid: (n_heads, num_q_tiles, batch_size)
    // Block: (256,) threads = 4 wavefronts
    //
    // =========================================================================

    /**
     * @brief Compute LDS layout sizes for given head_dim
     */
    struct FA2LDSLayout
    {
        int q_tile_size;      // Q tile in FP16 with padding
        int kv_tile_size;     // K or V tile in FP16 with padding
        int scores_size;      // Scores tile in FP32 with padding
        int total_size;       // Total LDS bytes
        int q_stride;         // Q leading dimension
        int kv_stride;        // K/V leading dimension
        int scores_stride;    // Scores leading dimension

        __host__ __device__ void compute(int head_dim, int tile_q, int tile_kv)
        {
            // Add padding to avoid bank conflicts
            q_stride = head_dim + FA2_LDS_PAD_FP16;
            kv_stride = head_dim + FA2_LDS_PAD_FP16;
            scores_stride = tile_kv + FA2_LDS_PAD_FP32;

            q_tile_size = tile_q * q_stride * sizeof(__half);
            kv_tile_size = tile_kv * kv_stride * sizeof(__half);
            scores_size = tile_q * scores_stride * sizeof(float);

            // Total: Q + 2*(K+V) for double buffer + scores
            total_size = q_tile_size + FA2_NUM_STAGES * 2 * kv_tile_size + scores_size;
        }
    };

    /**
     * @brief Load one logical KV element and convert it to the FP16 LDS format.
     *
     * The prefill kernel keeps K and V in their native cache representation
     * while they reside in HBM.  Conversion happens only as the live tile is
     * staged into LDS, preserving the bandwidth advantage of FP16 and Q8_1
     * caches without allocating a graph-unstable full-cache conversion buffer.
     *
     * @tparam KV_TYPE Physical cache representation selected at kernel launch.
     * @param source Beginning of the request-major native cache allocation.
     * @param logical_element Dense FP32-equivalent element index.  Q8_1 rows
     *        are block aligned because every supported head dimension is a
     *        multiple of the 32-element Q8_1 block width.
     * @return The requested value rounded to the FP16 arithmetic format used
     *         by the MI50 dot-product pipeline.
     */
    template <KVTypeTag KV_TYPE>
    __device__ __forceinline__ __half load_fa2_kv_element_as_half(
        const void *__restrict__ source,
        size_t logical_element)
    {
        if constexpr (KV_TYPE == KVTypeTag::FP32)
        {
            return float_to_half(static_cast<const float *>(source)[logical_element]);
        }
        else if constexpr (KV_TYPE == KVTypeTag::FP16)
        {
            return static_cast<const __half *>(source)[logical_element];
        }
        else if constexpr (KV_TYPE == KVTypeTag::BF16)
        {
            return float_to_half(
                static_cast<float>(
                    static_cast<const hip_bfloat16 *>(source)[logical_element]));
        }
        else
        {
            constexpr size_t ELEMENTS_PER_BLOCK = llaminar2::Q8_1Block::BLOCK_SIZE;
            const auto *blocks = static_cast<const llaminar2::Q8_1Block *>(source);
            const auto &block = blocks[logical_element / ELEMENTS_PER_BLOCK];
            const int element_in_block =
                static_cast<int>(logical_element % ELEMENTS_PER_BLOCK);
            const float scale = __half2float(
                *reinterpret_cast<const __half *>(&block.d));
            return float_to_half(scale * static_cast<float>(block.qs[element_in_block]));
        }
    }

    /**
     * @brief Load two adjacent KV values in the packed format consumed by fdot2.
     *
     * FP32 and FP16 caches retain vectorized global-memory reads.  Q8_1 uses
     * two scalar block lookups because its scale metadata is shared by 32
     * values; the resulting pair is still packed before entering LDS.
     */
    template <KVTypeTag KV_TYPE>
    __device__ __forceinline__ __half2 load_fa2_kv_pair_as_half2(
        const void *__restrict__ source,
        size_t logical_element)
    {
        if constexpr (KV_TYPE == KVTypeTag::FP32)
        {
            const float2 values = *reinterpret_cast<const float2 *>(
                static_cast<const float *>(source) + logical_element);
            return __float22half2_rn(values);
        }
        else if constexpr (KV_TYPE == KVTypeTag::FP16)
        {
            return *reinterpret_cast<const __half2 *>(
                static_cast<const __half *>(source) + logical_element);
        }
        else if constexpr (KV_TYPE == KVTypeTag::BF16)
        {
            return __halves2half2(
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element),
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element + 1));
        }
        else
        {
            return __halves2half2(
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element),
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element + 1));
        }
    }

    /**
     * @brief Flash Attention 2 prefill kernel for MI50
     *
     * Uses vectorized FP16 VALU operations for Q@K^T computation.
     * No WMMA/Matrix Cores - pure ALU implementation optimized for Vega.
     */
    template <KVTypeTag KV_TYPE, int THREADS_PER_ROW>
    __global__ __launch_bounds__(256, 2) void flash_attention_2_mi50_kernel(
        const float *__restrict__ Q,
        const void *__restrict__ K,
        const void *__restrict__ V,
        float *__restrict__ O,
        int batch_size,
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float softmax_scale,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        const float *__restrict__ mask,
        int tile_q,
        int tile_kv,
        int head_start,
        int gqa_n_rep)
    {
        int kv_stride = kv_len;
        int kv_len_runtime = kv_len;
        int position_offset_runtime = position_offset;
        int mask_stride = kv_len;

        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
            kv_stride = device_params->kv_stride;
            position_offset_runtime = device_params->position_offset;
            mask_stride = device_params->mask_stride;
        }

        // Block/thread indexing
        const int batch_idx = blockIdx.z;
        const int head_idx = blockIdx.x;
        const int q_tile_idx = blockIdx.y;
        const int tid = threadIdx.x;
        const int threads_per_block = blockDim.x;
        const int wavefront_id = tid / WAVEFRONT_SIZE;
        const int lane_id = tid % WAVEFRONT_SIZE;

        // GQA mapping: multiple Q heads can share one KV head
        // When gqa_n_rep > 0, KV heads are REPLICATED (each device has all KV heads),
        // so use the global head position (head_start + head_idx). When gqa_n_rep == 0,
        // KV heads are SHARDED (each device has only local_n_kv_heads), so use local indexing.
        const int effective_gqa_ratio = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
        const int kv_head_idx = (gqa_n_rep > 0)
            ? (head_start + head_idx) / effective_gqa_ratio
            : head_idx / effective_gqa_ratio;

        // Q row range for this block
        const int q_block_start = q_tile_idx * tile_q;
        if (q_block_start >= seq_len)
            return;
        const int q_tile_rows = min(tile_q, seq_len - q_block_start);

        // Compute LDS layout
        FA2LDSLayout lds_layout;
        lds_layout.compute(head_dim, tile_q, tile_kv);

        // LDS allocation
        extern __shared__ char smem[];

        __half *Q_tile = reinterpret_cast<__half *>(smem);
        __half *KV_buffers = Q_tile + tile_q * lds_layout.q_stride;
        float *scores = reinterpret_cast<float *>(KV_buffers + FA2_NUM_STAGES * 2 * tile_kv * lds_layout.kv_stride);

        // Helper lambdas for double-buffered K/V access
        auto get_K_tile = [&](int stage) -> __half *
        {
            return KV_buffers + stage * 2 * tile_kv * lds_layout.kv_stride;
        };
        auto get_V_tile = [&](int stage) -> __half *
        {
            return KV_buffers + (stage * 2 + 1) * tile_kv * lds_layout.kv_stride;
        };

        // =====================================================================
        // Thread-local accumulators for output
        // =====================================================================
        // Each thread is responsible for a subset of Q rows
        // With 256 threads and tile_q=64, each thread handles 64/256 = 0.25 rows
        // So we use a strided pattern where thread t handles rows t, t+256, ...

        // Use 4 threads per Q row so all 256 threads participate when tile_q=64.
        // Each thread owns a disjoint subset of output dims for that row.
        const int local_q_row = tid / THREADS_PER_ROW;
        const int row_lane = tid % THREADS_PER_ROW;
        const bool thread_has_row = (local_q_row < q_tile_rows);

        // Both launch variants cap ownership at 32 output dimensions per
        // thread.  Four lanes cooperate through head_dim 128; eight lanes
        // cooperate at head_dim 256.  Keeping this bound compile-time visible
        // prevents the accumulator array from growing with the context length.
        float O_acc[32];
        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        if (thread_has_row)
        {
            int d_local = 0;
            for (int d = row_lane; d < head_dim; d += THREADS_PER_ROW)
            {
                O_acc[d_local++] = 0.0f;
            }
        }

        // Batch/head pointers
        const float *Q_batch = Q + batch_idx * seq_len * n_heads * head_dim;
        const size_t request_element_offset =
            static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_stride) *
            static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

        // =====================================================================
        // Load Q tile into LDS (done once per block)
        // =====================================================================
        if ((head_dim & 1) == 0)
        {
            const int head_dim_vec2 = head_dim / 2;
            for (int i = tid; i < q_tile_rows * head_dim_vec2; i += threads_per_block)
            {
                int local_row = i / head_dim_vec2;
                int d2 = i % head_dim_vec2;
                int d = d2 * 2;
                int global_row = q_block_start + local_row;

                const float2 qv = *reinterpret_cast<const float2 *>(
                    Q_batch + global_row * n_heads * head_dim + head_idx * head_dim + d);
                reinterpret_cast<__half2 *>(Q_tile + local_row * lds_layout.q_stride + d)[0] = __float22half2_rn(qv);
            }
        }
        else
        {
            for (int i = tid; i < q_tile_rows * head_dim; i += threads_per_block)
            {
                int local_row = i / head_dim;
                int d = i % head_dim;
                int global_row = q_block_start + local_row;
                float val = Q_batch[global_row * n_heads * head_dim + head_idx * head_dim + d];
                Q_tile[local_row * lds_layout.q_stride + d] = float_to_half(val);
            }
        }
        __syncthreads();

        const int num_kv_tiles = (kv_len_runtime + tile_kv - 1) / tile_kv;

        // =====================================================================
        // Prologue: Load first KV tile
        // =====================================================================
        {
            const int kv_start = 0;
            const int kv_end = min(tile_kv, kv_len_runtime);
            const int actual_len = kv_end - kv_start;

            __half *K_dst = get_K_tile(0);
            __half *V_dst = get_V_tile(0);

            if ((head_dim & 1) == 0)
            {
                const int head_dim_vec2 = head_dim / 2;
                for (int i = tid; i < actual_len * head_dim_vec2; i += threads_per_block)
                {
                    int local_row = i / head_dim_vec2;
                    int d2 = i % head_dim_vec2;
                    int d = d2 * 2;
                    int global_row = kv_start + local_row;
                    const size_t element =
                        request_element_offset +
                        (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(d);

                    reinterpret_cast<__half2 *>(
                        K_dst + local_row * lds_layout.kv_stride + d)[0] =
                        load_fa2_kv_pair_as_half2<KV_TYPE>(K, element);
                    reinterpret_cast<__half2 *>(
                        V_dst + local_row * lds_layout.kv_stride + d)[0] =
                        load_fa2_kv_pair_as_half2<KV_TYPE>(V, element);
                }
            }
            else
            {
                for (int i = tid; i < actual_len * head_dim; i += threads_per_block)
                {
                    int local_row = i / head_dim;
                    int d = i % head_dim;
                    int global_row = kv_start + local_row;
                    const size_t element =
                        request_element_offset +
                        (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                            static_cast<size_t>(head_dim) +
                        static_cast<size_t>(d);

                    K_dst[local_row * lds_layout.kv_stride + d] =
                        load_fa2_kv_element_as_half<KV_TYPE>(K, element);
                    V_dst[local_row * lds_layout.kv_stride + d] =
                        load_fa2_kv_element_as_half<KV_TYPE>(V, element);
                }
            }
        }
        __syncthreads();

        // =====================================================================
        // Main loop over KV tiles
        // =====================================================================
        for (int kv_tile_iter = 0; kv_tile_iter < num_kv_tiles; kv_tile_iter++)
        {
            const int current_stage = kv_tile_iter % FA2_NUM_STAGES;
            const int next_stage = (kv_tile_iter + 1) % FA2_NUM_STAGES;

            const int kv_start = kv_tile_iter * tile_kv;
            const int kv_end = min(kv_start + tile_kv, kv_len_runtime);
            const int actual_kv_len = kv_end - kv_start;

            // Early exit for causal (all positions masked)
            if (causal && kv_start > (q_block_start + tile_q - 1) + position_offset_runtime)
            {
                continue;
            }

            __half *K_tile = get_K_tile(current_stage);
            __half *V_tile = get_V_tile(current_stage);

            // =================================================================
            // PHASE 1: Start loading next KV tile (if any)
            // =================================================================
            if (kv_tile_iter + 1 < num_kv_tiles)
            {
                const int next_kv_start = (kv_tile_iter + 1) * tile_kv;
                const int next_kv_end = min(next_kv_start + tile_kv, kv_len_runtime);
                const int next_len = next_kv_end - next_kv_start;

                __half *K_next = get_K_tile(next_stage);
                __half *V_next = get_V_tile(next_stage);

                // Cooperative load by all threads
                if ((head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = head_dim / 2;
                    for (int i = tid; i < next_len * head_dim_vec2; i += threads_per_block)
                    {
                        int local_row = i / head_dim_vec2;
                        int d2 = i % head_dim_vec2;
                        int d = d2 * 2;
                        int global_row = next_kv_start + local_row;
                        const size_t element =
                            request_element_offset +
                            (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                                static_cast<size_t>(head_dim) +
                            static_cast<size_t>(d);

                        reinterpret_cast<__half2 *>(
                            K_next + local_row * lds_layout.kv_stride + d)[0] =
                            load_fa2_kv_pair_as_half2<KV_TYPE>(K, element);
                        reinterpret_cast<__half2 *>(
                            V_next + local_row * lds_layout.kv_stride + d)[0] =
                            load_fa2_kv_pair_as_half2<KV_TYPE>(V, element);
                    }
                }
                else
                {
                    for (int i = tid; i < next_len * head_dim; i += threads_per_block)
                    {
                        int local_row = i / head_dim;
                        int d = i % head_dim;
                        int global_row = next_kv_start + local_row;
                        const size_t element =
                            request_element_offset +
                            (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                                static_cast<size_t>(head_dim) +
                            static_cast<size_t>(d);

                        K_next[local_row * lds_layout.kv_stride + d] =
                            load_fa2_kv_element_as_half<KV_TYPE>(K, element);
                        V_next[local_row * lds_layout.kv_stride + d] =
                            load_fa2_kv_element_as_half<KV_TYPE>(V, element);
                    }
                }
            }

            // =================================================================
            // PHASE 2: Compute Q @ K^T (v_dot2_f32_f16 with FP32 accumulator)
            // =================================================================
            // Each thread computes a subset of score matrix elements

            // Score matrix is [q_tile_rows, actual_kv_len]
            // We distribute computation: thread t computes elements at stride
            const int total_score_elements = q_tile_rows * actual_kv_len;

            for (int s_idx = tid; s_idx < total_score_elements; s_idx += threads_per_block)
            {
                int q_row = s_idx / actual_kv_len;
                int kv_col = s_idx % actual_kv_len;

                // Compute dot product Q[q_row] @ K[kv_col]
                float score = 0.0f;

                const __half *Q_ptr = Q_tile + q_row * lds_layout.q_stride;
                const __half *K_ptr = K_tile + kv_col * lds_layout.kv_stride;

                // Vectorized dot product using v_dot2_f32_f16 (MI50 gfx906+)
                // Each instruction: d.f32 = a.f16x2 · b.f16x2 + c.f32
                // Replaces 4× fp16→fp32 convert + 2× FP32 FMA per half2 pair
                // with a single hardware dot-product instruction.
                if ((head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = head_dim / 2;
                    const __half2 *Q_ptr2 = reinterpret_cast<const __half2 *>(Q_ptr);
                    const __half2 *K_ptr2 = reinterpret_cast<const __half2 *>(K_ptr);

                    // Use 4 independent accumulators to break dependency chain.
                    // v_dot2_f32_f16 has ~4 cycle latency; with 4 accumulators
                    // the pipeline stays full (ILP hiding).
                    float score0 = 0.0f, score1 = 0.0f, score2 = 0.0f, score3 = 0.0f;
                    int d2 = 0;
                    const int head_dim_vec2_aligned = head_dim_vec2 & ~3; // round down to multiple of 4
                    for (; d2 < head_dim_vec2_aligned; d2 += 4)
                    {
                        score0 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 0], K_ptr2[d2 + 0], score0, false);
                        score1 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 1], K_ptr2[d2 + 1], score1, false);
                        score2 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 2], K_ptr2[d2 + 2], score2, false);
                        score3 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 3], K_ptr2[d2 + 3], score3, false);
                    }
                    // Handle remaining pairs (head_dim not multiple of 8)
                    for (; d2 < head_dim_vec2; d2++)
                    {
                        score0 = __builtin_amdgcn_fdot2(Q_ptr2[d2], K_ptr2[d2], score0, false);
                    }
                    score = score0 + score1 + score2 + score3;
                }
                else
                {
                    for (int d = 0; d < head_dim; d++)
                    {
                        score += half_to_float(Q_ptr[d]) * half_to_float(K_ptr[d]);
                    }
                }

                scores[q_row * lds_layout.scores_stride + kv_col] = score * softmax_scale;
            }
            __syncthreads();

            // =================================================================
            // PHASE 3: Apply softmax and accumulate O = softmax(QK^T) @ V
            // =================================================================
            // Each thread processes its assigned Q rows
            //
            // Optimization: check at the tile level whether ALL scores in this
            // tile are unmasked (fully valid). For causal attention, ~90% of
            // tiles are fully valid since only the diagonal tiles need masking.
            // This eliminates 3-5 conditional branches per score element.

            if (thread_has_row)
            {
                int global_q_row = q_block_start + local_q_row;
                float *my_scores = scores + local_q_row * lds_layout.scores_stride;

                // Determine if this tile needs per-element masking
                // Tile is fully unmasked when:
                //   - No external mask
                //   - No sliding window (or window covers entire tile)
                //   - For causal: last KV position <= first Q position in block
                const bool tile_needs_masking =
                    mask != nullptr ||
                    (window_size > 0) ||
                    (causal && (kv_start + actual_kv_len - 1) > (global_q_row + position_offset_runtime));

                // Find max for this row (with masking)
                float m_ij = -FLT_MAX;

                if (!tile_needs_masking)
                {
                    // FAST PATH: no masking needed — scores already pre-scaled
                    for (int j = 0; j < actual_kv_len; j++)
                    {
                        m_ij = fmaxf(m_ij, my_scores[j]);
                    }
                }
                else
                {
                    // SLOW PATH: per-element masking
                    for (int j = 0; j < actual_kv_len; j++)
                    {
                        int kv_pos = kv_start + j;
                        bool masked = false;

                        // Causal masking
                        if (causal && kv_pos > global_q_row + position_offset_runtime)
                            masked = true;

                        // Sliding window
                        if (window_size > 0)
                        {
                            int q_pos = global_q_row + position_offset_runtime;
                            if (kv_pos < q_pos - window_size || kv_pos > q_pos + window_size)
                                masked = true;
                        }

                        if (mask)
                        {
                            float mask_val = mask[(batch_idx * seq_len + global_q_row) * mask_stride + kv_pos];
                            if (mask_val <= -1.0e20f)
                            {
                                masked = true;
                            }
                            else
                            {
                                my_scores[j] += mask_val;
                            }
                        }

                        if (masked)
                        {
                            my_scores[j] = -FLT_MAX;
                        }
                        else
                        {
                            m_ij = fmaxf(m_ij, my_scores[j]);
                        }
                    }
                }

                // Online softmax update
                float m_i_new = fmaxf(m_i, m_ij);
                float scale_old = fast_expf(m_i - m_i_new);

                // Rescale previous accumulators
                int d_local = 0;
                for (int d = row_lane; d < head_dim; d += THREADS_PER_ROW)
                {
                    O_acc[d_local++] *= scale_old;
                }
                l_i *= scale_old;

                // Accumulate P @ V for this tile
                float l_ij = 0.0f;
                for (int j = 0; j < actual_kv_len; j++)
                {
                    float s = my_scores[j];
                    if (s > -FLT_MAX / 2)
                    {
                        float p = fast_expf(s - m_i_new);
                        l_ij += p;

                        const __half *V_row = V_tile + j * lds_layout.kv_stride;
                        int d_local = 0;
                        for (int d = row_lane; d < head_dim; d += THREADS_PER_ROW)
                        {
                            O_acc[d_local++] += p * half_to_float(V_row[d]);
                        }
                    }
                }

                l_i += l_ij;
                m_i = m_i_new;
            }

            __syncthreads();
        }

        // =====================================================================
        // Write final output
        // =====================================================================
        float *O_batch = O + batch_idx * seq_len * n_heads * head_dim;

        if (thread_has_row)
        {
            int global_q_row = q_block_start + local_q_row;
            float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

            float *O_row = O_batch + global_q_row * n_heads * head_dim + head_idx * head_dim;
            int d_local = 0;
            for (int d = row_lane; d < head_dim; d += THREADS_PER_ROW)
            {
                O_row[d] = O_acc[d_local++] * inv_l;
            }
        }
    }

    // =========================================================================
    // Flash Decoding - Split-K Kernel for MI50 (v3: wavefront-cooperative)
    // =========================================================================
    //
    // For single-token decode (seq_len=1), we parallelize over the KV cache
    // using split-K. Each block handles a portion of the KV cache and produces
    // partial (m, l, O) that are merged in a reduction kernel.
    //
    // Grid: (n_heads, num_splits, batch_size)
    // Block: (256,) threads = 4 wavefronts
    //
    // v3 architecture (2026-02-09):
    //   LANES OWN OUTPUT DIMS, NOT KV POSITIONS.
    //
    //   Each wavefront cooperates on one KV position at a time:
    //     1. All 64 lanes load contiguous K elements (coalesced!)
    //     2. Partial dot product → wavefrontReduceSum → full score
    //     3. Online softmax update (score is uniform across lanes)
    //     4. All 64 lanes load contiguous V elements (coalesced!)
    //     5. Each lane updates only its 2 output dims of O
    //
    //   Benefits vs v2:
    //     - Coalesced K/V memory access (adjacent lanes → adjacent addresses)
    //     - 2-4 VGPRs for O per lane instead of 128 → 4x occupancy potential
    //     - Zero cross-lane O reduction (no shuffles for O at all)
    //     - Only 6 shuffles per KV position (dot product reduce)
    //     - v2 had 128 × wavefrontReduceSum = 768 shuffles TOTAL for O
    //
    //   v3+ optimization: exp2f hardware intrinsic
    //     - All expf() replaced with exp2f(x * LOG2E) for single-cycle v_exp_f32
    //     - GCN ISA: v_exp_f32 is a transcendental ALU op (~4 cycles vs ~50+ for expf)
    //
    // =========================================================================

    // exp2f(x * LOG2E) maps to hardware v_exp_f32 on GCN (single-cycle transcendental)
    __device__ __forceinline__ float fast_expf(float x)
    {
        return exp2f(x * 1.4426950408889634f); // log2(e) = 1/ln(2)
    }

    /**
     * @brief Physical cache ownership represented by one decode grid row.
     *
     * The arithmetic body is shared across all policies.  Only parameter-row
     * selection and the K/V bank base differ, which is essential for proving
     * grouped execution byte-identical to the ordinary scalar decode kernel.
     */
    enum class FlashDecodeRowMode
    {
        OrdinaryBatch,      ///< One physical K/V bank per ordinary batch row.
        SharedKVVerifier,   ///< Every verifier row reads one shared cache bank.
        IndependentRequest ///< Flat rows map to fixed-stride request cache banks.
    };

    /**
     * @brief Scalar-equivalent launch policy for one grouped decode row.
     */
    struct FlashDecodeRowConfig
    {
        int splits;
        int threads_per_block;
    };

    /**
     * @brief Clamp an integer without relying on host-only standard-library code.
     */
    __device__ __forceinline__ int deviceClampInt(int value, int lo, int hi)
    {
        return value < lo ? lo : (value > hi ? hi : value);
    }

    /**
     * @brief Round upward to a power of two, bounded by a row's split cap.
     */
    __device__ __forceinline__ int deviceRoundUpPow2Clamped(
        int value,
        int max_allowed)
    {
        int result = 1;
        while (result < value && result < max_allowed)
            result <<= 1;
        return deviceClampInt(result, 1, max_allowed);
    }

    /**
     * @brief Reproduce the production scalar ROCm decode policy on device.
     *
     * Grouped rows can own unequal live KV lengths even though graph capture
     * requires one fixed grid and block shape.  This helper mirrors
     * `selectFlashDecodeNumSplits()`, `chooseDecodeSplitsAuto()`, and
     * `chooseDecodeThreadsPerBlock()` for each row.  The phase kernel launches
     * the maximum physical geometry, then only the scalar-selected split and
     * wavefront prefix participates.  Consequently each row retains exactly
     * the same KV partition, wavefront merge order, and reduction order as an
     * independent M=1 call.
     */
    __device__ __forceinline__ FlashDecodeRowConfig selectGroupedDecodeRowConfig(
        int kv_len,
        int head_dim,
        int n_heads,
        int num_cus,
        int max_launch_splits,
        bool deterministic,
        bool forced_splits_present,
        int forced_splits,
        bool forced_tpb_present,
        int forced_tpb)
    {
        int requested_cap = FD_DEFAULT_NUM_SPLITS;
        if (deterministic || kv_len <= 64)
            requested_cap = 1;
        else if (kv_len < 128)
            requested_cap = 2;
        else if (kv_len < 256)
            requested_cap = 4;

        const int max_allowed = deviceClampInt(
            requested_cap < max_launch_splits ? requested_cap : max_launch_splits,
            1,
            FD_MAX_SPLITS_SUPPORTED);
        const int safe_head_dim = head_dim > 0 ? head_dim : 1;
        const int64_t total_elements =
            static_cast<int64_t>(kv_len) * static_cast<int64_t>(safe_head_dim);
        int base_splits = static_cast<int>(
            (total_elements + FD_TARGET_ELEMS_PER_SPLIT - 1) /
            FD_TARGET_ELEMS_PER_SPLIT);
        if (base_splits < 1)
            base_splits = 1;

        if (kv_len >= safe_head_dim * 8)
            base_splits = (base_splits * 3 + 1) / 2;
        else if (kv_len <= safe_head_dim * 2)
            base_splits = (base_splits + 1) / 2 > 1
                              ? (base_splits + 1) / 2
                              : 1;

        if (base_splits > kv_len)
            base_splits = kv_len > 0 ? kv_len : 1;
        int row_splits =
            deviceRoundUpPow2Clamped(base_splits, max_allowed);

        if (n_heads > 0 && num_cus > 0)
        {
            const int target_blocks = (num_cus * 3) / 4;
            if (n_heads * row_splits < target_blocks)
            {
                const int occupancy_splits =
                    (target_blocks + n_heads - 1) / n_heads;
                const int rounded_occupancy_splits =
                    deviceRoundUpPow2Clamped(occupancy_splits, max_allowed);
                if (rounded_occupancy_splits > row_splits)
                    row_splits = rounded_occupancy_splits;
            }
        }
        row_splits = deviceClampInt(row_splits, 1, max_allowed);
        if (forced_splits_present)
            row_splits = deviceClampInt(forced_splits, 1, max_allowed);

        const int rows_per_split = kv_len / row_splits > 1
                                       ? kv_len / row_splits
                                       : 1;
        const int64_t elements_per_split =
            total_elements / row_splits;
        int row_threads =
            ((rows_per_split <= 64 && elements_per_split <= 16384) ||
             (rows_per_split <= 48 && elements_per_split <= 32768))
                ? 128
                : 256;
        if (forced_tpb_present)
            row_threads = deviceClampInt(forced_tpb, 64, 256);

        return FlashDecodeRowConfig{row_splits, row_threads};
    }

    /**
     * @brief Flash Decoding kernel for single-query decode on MI50
     *        v3: wavefront-cooperative KV processing + exp2f intrinsic
     */
    template <KVTypeTag KV_TYPE = KVTypeTag::FP32,
              FlashDecodeRowMode ROW_MODE = FlashDecodeRowMode::OrdinaryBatch>
    __global__ __launch_bounds__(256, 4)
    void flash_decoding_mi50_kernel(
        const float *__restrict__ Q,
        const void *__restrict__ K_cache,
        const void *__restrict__ V_cache,
        float *__restrict__ O_partial,
        float *__restrict__ m_partial,
        float *__restrict__ l_partial,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int num_splits,
        float softmax_scale,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        int query_rows_per_request,
        int num_cus,
        bool deterministic,
        bool forced_splits_present,
        int forced_splits,
        bool forced_tpb_present,
        int forced_tpb,
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        constexpr bool ROW_LOCAL_PARAMS =
            ROW_MODE != FlashDecodeRowMode::OrdinaryBatch;
        constexpr bool SHARED_KV =
            ROW_MODE == FlashDecodeRowMode::SharedKVVerifier;

        int kv_len_runtime = kv_len;
        int kv_batch_stride = kv_len;
        if (device_params)
        {
            const int param_row = ROW_LOCAL_PARAMS ? blockIdx.z : 0;
            kv_len_runtime = device_params[param_row].kv_len;
            kv_batch_stride = device_params[param_row].kv_stride;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        FlashDecodeRowConfig row_config{
            num_splits,
            static_cast<int>(blockDim.x)};
        if constexpr (ROW_LOCAL_PARAMS)
        {
            row_config = selectGroupedDecodeRowConfig(
                kv_len_runtime,
                head_dim,
                n_heads,
                num_cus,
                num_splits,
                deterministic,
                forced_splits_present,
                forced_splits,
                forced_tpb_present,
                forced_tpb);
        }
        const int row_num_splits = row_config.splits;
        const int logical_threads = row_config.threads_per_block;

        /*
         * Every thread in the physical 256-thread block still reaches both
         * barriers.  Only the scalar-selected prefix performs arithmetic.  In
         * particular, a row whose ordinary decode uses 128 threads executes
         * with exactly two active wavefronts and merges those two wavefronts in
         * the same order even when another row in this grid needs four.
         */
        const int tid = threadIdx.x;
        const bool active_thread =
            !ROW_LOCAL_PARAMS || tid < logical_threads;
        const int wavefront_id = tid / WAVEFRONT_SIZE;
        const int lane_id = tid % WAVEFRONT_SIZE;
        const int num_wavefronts =
            logical_threads / WAVEFRONT_SIZE > 0
                ? logical_threads / WAVEFRONT_SIZE
                : 1;

        const int effective_gqa_ratio = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
        const int kv_head_idx = (gqa_n_rep > 0)
            ? (head_start + head_idx) / effective_gqa_ratio
            : head_idx / effective_gqa_ratio;

        /* Surplus split planes are outside this row's scalar launch shape. */
        if (split_idx >= row_num_splits)
            return;

        // Split range: identical integer partitioning to scalar M=1 decode.
        const int split_size =
            (kv_len_runtime + row_num_splits - 1) / row_num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len_runtime);

        const int partial_idx = (batch_idx * n_heads + head_idx) * num_splits + split_idx;

        // Handle empty splits
        if (kv_start >= kv_len_runtime)
        {
            if (tid == 0)
            {
                m_partial[partial_idx] = -FLT_MAX;
                l_partial[partial_idx] = 0.0f;
            }
            for (int d = tid * 2;
                 active_thread && d + 1 < head_dim;
                 d += logical_threads * 2)
            {
                const int base = (batch_idx * n_heads + head_idx) * head_dim;
                const int idx0 = (base + d + 0) * num_splits + split_idx;
                const int idx1 = (base + d + 1) * num_splits + split_idx;
                O_partial[idx0] = 0.0f;
                O_partial[idx1] = 0.0f;
            }
            for (int d = (head_dim & ~1) + tid;
                 active_thread && d < head_dim;
                 d += logical_threads)
            {
                const int base = (batch_idx * n_heads + head_idx) * head_dim;
                const int idx = (base + d) * num_splits + split_idx;
                O_partial[idx] = 0.0f;
            }
            return;
        }

        // Load Q into LDS (all threads cooperate)
        // FP16 KV path: store Q as half in LDS (convert once, use many times)
        // FP32/BF16/Q8_1 paths store Q as float in LDS so BF16 and Q8_1
        // dequantization retains the scalar FP32 decode arithmetic contract.
        extern __shared__ char smem[];
        [[maybe_unused]] float *Q_shared = nullptr;
        [[maybe_unused]] __half *Q_shared_h = nullptr;

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;

        if constexpr (KV_TYPE == KVTypeTag::FP16)
        {
            // Store Q as FP16 in LDS — eliminates per-KV-position FP32→FP16 conversion
            // Halves LDS bandwidth for Q reads (4B → 2B per element per lane)
            Q_shared_h = reinterpret_cast<__half *>(smem);
            for (int d = tid;
                 active_thread && d < head_dim;
                 d += logical_threads)
            {
                Q_shared_h[d] = __float2half(Q_ptr[d]);
            }
        }
        else
        {
            Q_shared = reinterpret_cast<float *>(smem);
            for (int d = tid * 2;
                 active_thread && d + 1 < head_dim;
                 d += logical_threads * 2)
            {
                const float2 qv = reinterpret_cast<const float2 *>(Q_ptr + d)[0];
                reinterpret_cast<float2 *>(Q_shared + d)[0] = qv;
            }
            for (int d = (head_dim & ~1) + tid;
                 active_thread && d < head_dim;
                 d += logical_threads)
            {
                Q_shared[d] = Q_ptr[d];
            }
        }
        __syncthreads();

        // =====================================================================
        // Thread-local O accumulators — only this lane's output dimensions
        //
        // head_dim / WAVEFRONT_SIZE dims per lane (strided by WAVEFRONT_SIZE).
        // head_dim=128 => 2 dims/lane, head_dim=256 => 4 dims/lane.
        // Lane i owns dims: i, i+64, i+128, i+192 (up to head_dim).
        //
        // This reduces VGPR usage to ceil(head_dim/64) floats per lane.
        // =====================================================================
        float O_lane[4] = {0, 0, 0, 0}; // Supports head_dim up to 256 (256/64=4)
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        // Byte-level base pointers for type-dispatched K/V access (KV_TYPE)
        // Batch + head offsets hoisted out of the main KV loop.
        const int kv_head_offset_elems = kv_head_idx * head_dim;
        const int kv_row_stride_elems = n_kv_heads * head_dim;

        // Q8_1 block layout (compiled out for non-Q8_1 via if constexpr)
        [[maybe_unused]] constexpr int Q8_BS = 32;
        [[maybe_unused]] const int q8_bph = head_dim / Q8_BS;    // blocks per head
        [[maybe_unused]] const int q8_bpr = n_kv_heads * q8_bph; // blocks per KV row

        int cache_batch_idx = batch_idx;
        if constexpr (SHARED_KV)
        {
            cache_batch_idx = 0;
        }
        else if constexpr (ROW_MODE == FlashDecodeRowMode::IndependentRequest)
        {
            cache_batch_idx =
                batch_idx / (query_rows_per_request > 0
                                 ? query_rows_per_request
                                 : 1);
        }

        size_t row_byte_stride;
        size_t base_byte_off;
        if constexpr (KV_TYPE == KVTypeTag::Q8_1) {
            row_byte_stride = static_cast<size_t>(q8_bpr) * sizeof(llaminar2::Q8_1Block);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_idx) * q8_bph * sizeof(llaminar2::Q8_1Block);
        } else if constexpr (KV_TYPE == KVTypeTag::FP16) {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(__half);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(__half);
        } else if constexpr (KV_TYPE == KVTypeTag::BF16) {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(hip_bfloat16);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(hip_bfloat16);
        } else {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(float);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(float);
        }
        const char *K_base = reinterpret_cast<const char*>(K_cache) + base_byte_off;
        const char *V_base = reinterpret_cast<const char*>(V_cache) + base_byte_off;

        // =====================================================================
        // Main loop: wavefront-cooperative KV processing
        //
        // Each wavefront processes a strided subset of KV positions.
        // Within each position, all 64 lanes cooperate on loads and dot product.
        // K/V loads are fully coalesced (adjacent lanes → adjacent addresses).
        // =====================================================================
        for (int kv_pos = kv_start + wavefront_id;
             active_thread && kv_pos < kv_end;
             kv_pos += num_wavefronts)
        {
            const char *K_pos = K_base + kv_pos * row_byte_stride;

            // Type-dispatched K load + dot product
            // FP16: half bandwidth, fdot2 for vectorized path (Q→FP16 on the fly)
            // Q8_1: inline dequant, ~3.5× BW savings
            // FP32: original vectorized path (float2)
            float partial_dot = 0.0f;
            if constexpr (KV_TYPE == KVTypeTag::FP16)
            {
                const __half *Kh = reinterpret_cast<const __half*>(K_pos);
                if ((head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = head_dim / 2;
                    const __half2 *Q_ptr2_h = reinterpret_cast<const __half2 *>(Q_shared_h);
                    const __half2 *K_ptr2 = reinterpret_cast<const __half2 *>(Kh);
                    for (int d2 = lane_id; d2 < head_dim_vec2; d2 += WAVEFRONT_SIZE)
                    {
                        const __half2 qh2 = Q_ptr2_h[d2]; // Pre-converted FP16 from LDS
                        const __half2 kh2 = K_ptr2[d2];
                        partial_dot = __builtin_amdgcn_fdot2(qh2, kh2, partial_dot, false);
                    }
                }
                else
                {
                    for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                    {
                        partial_dot += __half2float(Q_shared_h[d]) * __half2float(Kh[d]);
                    }
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::Q8_1)
            {
                const llaminar2::Q8_1Block *Kq = reinterpret_cast<const llaminar2::Q8_1Block*>(K_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                {
                    const int bi = d / Q8_BS;
                    const int bo = d % Q8_BS;
                    const float ks = __half2float(*reinterpret_cast<const __half*>(&Kq[bi].d));
                    partial_dot += Q_shared[d] * (static_cast<float>(Kq[bi].qs[bo]) * ks);
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::BF16)
            {
                const auto *K_bf16 =
                    reinterpret_cast<const hip_bfloat16 *>(K_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                {
                    partial_dot +=
                        Q_shared[d] * static_cast<float>(K_bf16[d]);
                }
            }
            else
            {
                const float *K_ptr = reinterpret_cast<const float*>(K_pos);
                if ((head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = head_dim / 2;
                    const float2 *Q_ptr2 = reinterpret_cast<const float2 *>(Q_shared);
                    const float2 *K_ptr2 = reinterpret_cast<const float2 *>(K_ptr);
                    for (int d2 = lane_id; d2 < head_dim_vec2; d2 += WAVEFRONT_SIZE)
                    {
                        const float2 qv = Q_ptr2[d2];
                        const float2 kv = K_ptr2[d2];
                        partial_dot += qv.x * kv.x + qv.y * kv.y;
                    }
                }
                else
                {
                    for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                    {
                        partial_dot += Q_shared[d] * K_ptr[d];
                    }
                }
            }

            // Reduce across wavefront → full dot product (6 shuffles)
            // After this, ALL lanes have the same score value
            float score = wavefrontReduceSum(partial_dot) * softmax_scale;

            // Online softmax — score is uniform across all lanes,
            // so m_local and l_local stay wavefront-uniform too
            // Uses fast_expf (exp2f hardware intrinsic) for ~10x faster exp
            float m_new = fmaxf(m_local, score);
            float scale_old = fast_expf(m_local - m_new);
            float p = fast_expf(score - m_new);

            l_local = l_local * scale_old + p;

            // Type-dispatched V load + O accumulation
            // Each lane updates only its own output dims (head_dim/64 per lane)
            const char *V_pos = V_base + kv_pos * row_byte_stride;
            int o_idx = 0;
            if constexpr (KV_TYPE == KVTypeTag::FP16)
            {
                const __half *Vh = reinterpret_cast<const __half*>(V_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
                {
                    O_lane[o_idx] = O_lane[o_idx] * scale_old + p * __half2float(Vh[d]);
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::Q8_1)
            {
                const llaminar2::Q8_1Block *Vq = reinterpret_cast<const llaminar2::Q8_1Block*>(V_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
                {
                    const int bi = d / Q8_BS;
                    const int bo = d % Q8_BS;
                    const float vs = __half2float(*reinterpret_cast<const __half*>(&Vq[bi].d));
                    O_lane[o_idx] = O_lane[o_idx] * scale_old + p * (static_cast<float>(Vq[bi].qs[bo]) * vs);
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::BF16)
            {
                const auto *V_bf16 =
                    reinterpret_cast<const hip_bfloat16 *>(V_pos);
                for (int d = lane_id;
                     d < head_dim;
                     d += WAVEFRONT_SIZE, ++o_idx)
                {
                    O_lane[o_idx] =
                        O_lane[o_idx] * scale_old +
                        p * static_cast<float>(V_bf16[d]);
                }
            }
            else
            {
                const float *V_ptr = reinterpret_cast<const float*>(V_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
                {
                    O_lane[o_idx] = O_lane[o_idx] * scale_old + p * V_ptr[d];
                }
            }

            m_local = m_new;
        }

        // =====================================================================
        // Inter-wavefront reduction using LDS
        //
        // Each wavefront has:
        //   - (m_local, l_local) — uniform across its lanes
        //   - O distributed across lanes (each lane owns head_dim/64 dims)
        //
        // We merge 4 wavefronts' results, then write the final partial output.
        // =====================================================================
        __shared__ float block_m[4];
        __shared__ float block_l[4];
        __shared__ float block_O[4 * 256]; // 4 wavefronts × max head_dim

        // Write this wavefront's results to LDS
        if (active_thread && lane_id == 0)
        {
            block_m[wavefront_id] = m_local;
            block_l[wavefront_id] = l_local;
        }
        if (active_thread && head_dim == 128)
        {
            const int base = wavefront_id * 128;
            block_O[base + lane_id] = O_lane[0];
            block_O[base + lane_id + WAVEFRONT_SIZE] = O_lane[1];
        }
        else if (active_thread)
        {
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
            {
                block_O[wavefront_id * head_dim + d] = O_lane[o_idx];
            }
        }
        __syncthreads();

        // Thread 0 computes per-wavefront rescaling factors
        __shared__ float wf_scales[4];
        if (tid == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            wf_scales[0] = 1.0f;

            for (int w = 1; w < num_wavefronts; w++)
            {
                float other_m = block_m[w];
                float other_l = block_l[w];

                float m_new = fmaxf(final_m, other_m);
                float scale_self = fast_expf(final_m - m_new);
                float scale_other = fast_expf(other_m - m_new);

                for (int prev = 0; prev < w; prev++)
                    wf_scales[prev] *= scale_self;
                wf_scales[w] = scale_other;

                final_l = scale_self * final_l + scale_other * other_l;
                final_m = m_new;
            }

            m_partial[partial_idx] = final_m;
            l_partial[partial_idx] = final_l;
        }
        __syncthreads(); // All threads hit this — wf_scales now visible

        // Parallel output write — wavefront 0 lanes each handle their output dims
        if (active_thread && wavefront_id == 0)
        {
            const int base = (batch_idx * n_heads + head_idx) * head_dim;

            if (head_dim == 128)
            {
                float sum0 = 0.0f;
                float sum1 = 0.0f;

                if (num_wavefronts == 4)
                {
                    sum0 = wf_scales[0] * block_O[lane_id] +
                           wf_scales[1] * block_O[128 + lane_id] +
                           wf_scales[2] * block_O[256 + lane_id] +
                           wf_scales[3] * block_O[384 + lane_id];

                    const int lane1 = lane_id + WAVEFRONT_SIZE;
                    sum1 = wf_scales[0] * block_O[lane1] +
                           wf_scales[1] * block_O[128 + lane1] +
                           wf_scales[2] * block_O[256 + lane1] +
                           wf_scales[3] * block_O[384 + lane1];
                }
                else
                {
                    for (int w = 0; w < num_wavefronts; w++)
                    {
                        const int wbase = w * head_dim;
                        sum0 += wf_scales[w] * block_O[wbase + lane_id];
                        sum1 += wf_scales[w] * block_O[wbase + lane_id + WAVEFRONT_SIZE];
                    }
                }

                const int idx0 = (base + lane_id) * num_splits + split_idx;
                const int idx1 = (base + lane_id + WAVEFRONT_SIZE) * num_splits + split_idx;
                O_partial[idx0] = sum0;
                O_partial[idx1] = sum1;
            }
            else
            {
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                {
                    float sum = 0.0f;
                    for (int w = 0; w < num_wavefronts; w++)
                    {
                        sum += wf_scales[w] * block_O[w * head_dim + d];
                    }
                    const int idx = (base + d) * num_splits + split_idx;
                    O_partial[idx] = sum;
                }
            }
        }
    }

    /**
     * @brief Flash Decoding reduction kernel
     *
     * Combines partial outputs from all splits using stable softmax merge.
     */
    template <FlashDecodeRowMode ROW_MODE = FlashDecodeRowMode::OrdinaryBatch>
    __global__ __launch_bounds__(256, 4)
    void flash_decoding_reduce_mi50_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ m_partial,
        const float *__restrict__ l_partial,
        float *__restrict__ O,
        int n_heads,
        int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        int num_cus,
        bool deterministic,
        bool forced_splits_present,
        int forced_splits,
        bool forced_tpb_present,
        int forced_tpb)
    {
        constexpr bool ROW_LOCAL_PARAMS =
            ROW_MODE != FlashDecodeRowMode::OrdinaryBatch;
        const int head_idx = blockIdx.x;
        const int batch_idx = blockIdx.y;

        const int tid = threadIdx.x;
        int row_num_splits = num_splits;
        if constexpr (ROW_LOCAL_PARAMS)
        {
            row_num_splits = selectGroupedDecodeRowConfig(
                                 device_params[batch_idx].kv_len,
                                 head_dim,
                                 n_heads,
                                 num_cus,
                                 num_splits,
                                 deterministic,
                                 forced_splits_present,
                                 forced_splits,
                                 forced_tpb_present,
                                 forced_tpb)
                                 .splits;
        }
        const int base_idx = (batch_idx * n_heads + head_idx) * num_splits;

        // Find global max and compute scales
        __shared__ float global_m;
        __shared__ float global_l;
        __shared__ float split_scales[32]; // Support up to 32 splits

        if (tid == 0)
        {
            float m_max = -FLT_MAX;
            for (int s = 0; s < row_num_splits; s++)
            {
                m_max = fmaxf(m_max, m_partial[base_idx + s]);
            }
            global_m = m_max;

            float l_sum = 0.0f;
            for (int s = 0; s < row_num_splits; s++)
            {
                float scale = expf(m_partial[base_idx + s] - m_max);
                split_scales[s] = scale;
                l_sum += scale * l_partial[base_idx + s];
            }
            global_l = l_sum;
        }
        __syncthreads();

        float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
        float *O_out = O + (batch_idx * n_heads + head_idx) * head_dim;
        const int head_base = (batch_idx * n_heads + head_idx) * head_dim;

        // Combine partial outputs
        for (int d = tid * 2; d + 1 < head_dim; d += blockDim.x * 2)
        {
            float O_sum0 = 0.0f;
            float O_sum1 = 0.0f;
            const int split_base0 = (head_base + d + 0) * num_splits;
            const int split_base1 = (head_base + d + 1) * num_splits;
            for (int s = 0; s < row_num_splits; s++)
            {
                const float scale = split_scales[s];
                O_sum0 += scale * O_partial[split_base0 + s];
                O_sum1 += scale * O_partial[split_base1 + s];
            }
            reinterpret_cast<float2 *>(O_out + d)[0] = make_float2(O_sum0 * inv_l, O_sum1 * inv_l);
        }

        for (int d = (head_dim & ~1) + tid; d < head_dim; d += blockDim.x)
        {
            float O_sum = 0.0f;
            const int split_base = (head_base + d) * num_splits;
            for (int s = 0; s < row_num_splits; s++)
            {
                O_sum += split_scales[s] * O_partial[split_base + s];
            }
            O_out[d] = O_sum * inv_l;
        }
    }

} // anonymous namespace

// =============================================================================
// Device-resident prefill launch template
// =============================================================================

/**
 * @brief Launch the native-format ROCm prefill kernel with stable graph shape.
 *
 * Tile selection depends only on the immutable head dimension and MI50 LDS
 * capacity.  In particular, neither the live KV length nor the current cache
 * fill level changes the grid or shared-memory allocation.  A captured graph
 * can therefore replay after later cache appends: the kernel reads the new
 * logical length and fixed physical request stride from @p device_params.
 *
 * The cache remains compressed in HBM for FP16 and Q8_1.  Only each live K/V
 * tile is converted to the half-precision LDS representation used by gfx906
 * `v_dot2_f32_f16`, avoiding full-cache conversion buffers and host-sized GEMM
 * launches.
 *
 * @tparam KV_TYPE Physical representation of the resident K/V cache.
 * @return Zero after a valid launch, or -1 when geometry is unsupported or
 *         HIP reports a launch error.
 */
template <KVTypeTag KV_TYPE>
static int launch_device_resident_prefill(
    const float *Q,
    const void *K,
    const void *V,
    float *O,
    int batch_size,
    int seq_len,
    int kv_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    bool causal,
    int window_size,
    int position_offset,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    const float *mask,
    hipStream_t hip_stream,
    int head_start,
    int gqa_n_rep)
{
    if (!Q || !K || !V || !O || batch_size <= 0 || seq_len <= 0 ||
        kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || head_dim > 256)
    {
        printf("[launch_device_resident_prefill] Invalid pointers or geometry\n");
        return -1;
    }
    if (n_heads % n_kv_heads != 0)
    {
        printf("[launch_device_resident_prefill] n_heads must be divisible by n_kv_heads\n");
        return -1;
    }
    if constexpr (KV_TYPE == KVTypeTag::Q8_1)
    {
        if (head_dim % static_cast<int>(llaminar2::Q8_1Block::BLOCK_SIZE) != 0)
        {
            printf("[launch_device_resident_prefill] Q8_1 head_dim must be a multiple of 32\n");
            return -1;
        }
    }

    struct TileCandidate
    {
        int q;
        int kv;
    };

    // Larger tile area wins; ties prefer more query rows so a K/V tile is
    // reused by more work.  The 32x16 and 16x16 candidates keep head_dim 256
    // comfortably below MI50's 64 KiB LDS limit.
    constexpr TileCandidate candidates[] = {
        {64, 64},
        {64, 32},
        {32, 64},
        {32, 32},
        {32, 16},
        {16, 32},
        {16, 16},
        {8, 16},
        {8, 8},
    };

    constexpr int THREADS_PER_BLOCK = 256;
    const int threads_per_row = head_dim <= 128 ? 4 : 8;
    const int max_tile_q = THREADS_PER_BLOCK / threads_per_row;
    int tile_q = -1;
    int tile_kv = -1;
    int best_area = -1;
    FA2LDSLayout lds_layout{};

    for (const auto &candidate : candidates)
    {
        if (candidate.q > max_tile_q)
            continue;

        FA2LDSLayout candidate_layout;
        candidate_layout.compute(head_dim, candidate.q, candidate.kv);
        if (candidate_layout.total_size > LDS_SIZE_BYTES)
            continue;

        const int area = candidate.q * candidate.kv;
        if (area > best_area || (area == best_area && candidate.q > tile_q))
        {
            tile_q = candidate.q;
            tile_kv = candidate.kv;
            best_area = area;
            lds_layout = candidate_layout;
        }
    }

    if (best_area < 0)
    {
        printf("[launch_device_resident_prefill] No LDS tile fits head_dim=%d\n", head_dim);
        return -1;
    }

    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
    const dim3 grid(n_heads, num_q_tiles, batch_size);
    const dim3 block(THREADS_PER_BLOCK);

    // Discard an error left by an earlier asynchronous operation before the
    // launch, then report only the status produced by this kernel.
    (void)hipGetLastError();
    if (threads_per_row == 4)
    {
        hipLaunchKernelGGL(
            (flash_attention_2_mi50_kernel<KV_TYPE, 4>),
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep);
    }
    else
    {
        hipLaunchKernelGGL(
            (flash_attention_2_mi50_kernel<KV_TYPE, 8>),
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep);
    }

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        printf("[launch_device_resident_prefill] Kernel launch failed: %s\n",
               hipGetErrorString(error));
        return -1;
    }
    return 0;
}

// =============================================================================
// Flash Decoding launch template (outside extern "C" — templates need C++ linkage)
// KV_TYPE selects native K/V format:
//   FP16: half bandwidth, fdot2 dot product on gfx906
//   Q8_1: inline dequant, ~3.5× BW savings
//   FP32: original path (backward-compatible)
// =============================================================================
template <KVTypeTag KV_TYPE>
static int launch_decode(
    const float *Q, const void *K_cache, const void *V_cache, float *O,
    float *O_partial, float *m_partial, float *l_partial,
    int batch_size, int kv_len,
    int n_heads, int n_kv_heads, int head_dim,
    int num_splits,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    hipStream_t hip_stream,
    int head_start = 0, int gqa_n_rep = 0)
{
    if (head_dim <= 0 || head_dim > 256)
    {
        printf("[launch_decode] Error: head_dim=%d must be in [1, 256]\n", head_dim);
        return -1;
    }

    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    int current_device = 0;
    (void)hipGetDevice(&current_device);
    const int num_cus = getDeviceConfig(current_device).num_cus;

    const DecodeAutotuneSelection tuned = chooseDecodeLaunchConfig(kv_len, head_dim, num_splits, n_heads, num_cus);
    const int effective_num_splits = tuned.config.splits;
    const int decode_threads_per_block = tuned.config.tpb;

    // Autotuning MUST be disabled during HIP stream capture.
    // hipEventCreate/Synchronize/Destroy are forbidden during capture and
    // will invalidate the capture graph.
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    if (hip_stream)
        (void)hipStreamIsCapturing(hip_stream, &capture_status);
    const bool is_capturing = (capture_status == hipStreamCaptureStatusActive);

    hipEvent_t tune_start = nullptr;
    hipEvent_t tune_stop = nullptr;
    const bool measure_tuning = tuned.should_time && !is_capturing;
    if (measure_tuning)
    {
        if (hipEventCreate(&tune_start) == hipSuccess && hipEventCreate(&tune_stop) == hipSuccess)
        {
            (void)hipEventRecord(tune_start, hip_stream);
        }
        else
        {
            if (tune_start)
                (void)hipEventDestroy(tune_start);
            if (tune_stop)
                (void)hipEventDestroy(tune_stop);
            tune_start = nullptr;
            tune_stop = nullptr;
        }
    }

    // Split-K kernel (type-dispatched via template)
    // FP16 path stores Q as half in shared memory (halves LDS usage + bandwidth)
    size_t smem_size;
    if constexpr (KV_TYPE == KVTypeTag::FP16) {
        smem_size = head_dim * sizeof(__half);
    } else {
        smem_size = head_dim * sizeof(float);
    }
    dim3 grid_split(n_heads, effective_num_splits, batch_size);
    dim3 block_split(decode_threads_per_block);

    hipLaunchKernelGGL(
        (flash_decoding_mi50_kernel<KV_TYPE,
                                    FlashDecodeRowMode::OrdinaryBatch>),
        grid_split, block_split, smem_size, hip_stream,
        Q, K_cache, V_cache,
        O_partial, m_partial, l_partial,
        kv_len, n_heads, n_kv_heads, head_dim,
        effective_num_splits, softmax_scale, device_params,
        /*query_rows_per_request=*/1,
        num_cus,
        /*deterministic=*/false,
        /*forced_splits_present=*/false,
        /*forced_splits=*/0,
        /*forced_tpb_present=*/false,
        /*forced_tpb=*/0,
        head_start, gqa_n_rep);

    // Reduction kernel (unchanged — operates on FP32 partials only)
    dim3 grid_reduce(n_heads, batch_size);
    dim3 block_reduce(256);

    hipLaunchKernelGGL(
        (flash_decoding_reduce_mi50_kernel<
            FlashDecodeRowMode::OrdinaryBatch>),
        grid_reduce, block_reduce, 0, hip_stream,
        O_partial, m_partial, l_partial, O,
        n_heads, head_dim, effective_num_splits,
        device_params,
        num_cus,
        /*deterministic=*/false,
        /*forced_splits_present=*/false,
        /*forced_splits=*/0,
        /*forced_tpb_present=*/false,
        /*forced_tpb=*/0);

    if (tune_start && tune_stop)
    {
        (void)hipEventRecord(tune_stop, hip_stream);
        if (hipEventSynchronize(tune_stop) == hipSuccess)
        {
            float elapsed_ms = 0.0f;
            if (hipEventElapsedTime(&elapsed_ms, tune_start, tune_stop) == hipSuccess)
            {
                recordDecodeAutotuneResult(tuned.key, tuned.candidate_index, elapsed_ms);
            }
        }
        (void)hipEventDestroy(tune_start);
        (void)hipEventDestroy(tune_stop);
    }

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("[launch_decode] Kernel launch failed: %s\n", hipGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * @brief Launch one economical decode phase and reduction for grouped rows.
 *
 * Unlike an ordinary batch, grouped verifier rows may share one K/V bank and
 * grouped request rows may map several query rows to each independent bank.
 * Device-resident metadata supplies every row's logical KV length.  The phase
 * grid uses the longest row's split cap and a 256-thread physical block, while
 * the kernel activates only the split and wavefront prefix selected by the
 * ordinary scalar ROCm policy for that row.  This preserves byte identity
 * without host count reads or one launch pair per row.
 *
 * Online autotune trial rotation is intentionally rejected.  It is an explicit
 * profiling mode whose per-call candidate mutation cannot define a stable
 * scalar oracle for multiple rows inside one graph replay.  The normal
 * production heuristic and explicit fixed split/TPB overrides are supported.
 */
template <KVTypeTag KV_TYPE, FlashDecodeRowMode ROW_MODE>
static int launch_grouped_decode(
    const float *Q,
    const void *K_cache,
    const void *V_cache,
    float *O,
    float *O_partial,
    float *m_partial,
    float *l_partial,
    int row_count,
    int query_rows_per_request,
    int max_kv_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_num_splits,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    hipStream_t hip_stream,
    int head_start,
    int gqa_n_rep)
{
    static_assert(
        ROW_MODE != FlashDecodeRowMode::OrdinaryBatch,
        "Grouped launcher requires a row-local cache ownership policy");

    if (!Q || !K_cache || !V_cache || !O || !O_partial || !m_partial ||
        !l_partial || !device_params || !hip_stream || row_count < 2 ||
        row_count >
            llaminar2::attention::kMaxGroupedVerifierAttentionRows ||
        query_rows_per_request <= 0 || max_kv_len <= 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 ||
        max_num_splits <= 0 || max_num_splits > FD_MAX_SPLITS_SUPPORTED)
    {
        printf("[launch_grouped_decode] Invalid grouped decode contract: "
               "rows=%d query_rows=%d max_kv_len=%d heads=%d kv_heads=%d "
               "head_dim=%d max_splits=%d stream=%p\n",
               row_count,
               query_rows_per_request,
               max_kv_len,
               n_heads,
               n_kv_heads,
               head_dim,
               max_num_splits,
               static_cast<void *>(hip_stream));
        return -1;
    }

    const auto &rocm_env = llaminar2::debugEnv().rocm;
    const bool deterministic = llaminar2::debugEnv().gemm.deterministic;
    if (rocm_env.fa_decode_autotune && !deterministic &&
        !rocm_env.fa_decode_num_splits_present)
    {
        printf("[launch_grouped_decode] Online flash-decode autotune is a "
               "profiling-only mode and cannot define grouped row parity\n");
        return -1;
    }

    const bool forced_splits_present =
        rocm_env.fa_decode_num_splits_present &&
        rocm_env.fa_decode_num_splits.has_value();
    const int forced_splits =
        forced_splits_present ? *rocm_env.fa_decode_num_splits : 0;
    const bool forced_tpb_present =
        rocm_env.fa_decode_num_splits_present &&
        rocm_env.fa_decode_tpb.has_value();
    const int forced_tpb =
        forced_tpb_present ? *rocm_env.fa_decode_tpb : 0;

    int current_device = 0;
    (void)hipGetDevice(&current_device);
    const int num_cus = getDeviceConfig(current_device).num_cus;
    const float softmax_scale =
        1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t smem_size =
        KV_TYPE == KVTypeTag::FP16
            ? static_cast<size_t>(head_dim) * sizeof(__half)
            : static_cast<size_t>(head_dim) * sizeof(float);

    /* Clear a pre-existing asynchronous launch error before attributing one. */
    (void)hipGetLastError();

    const dim3 split_grid(n_heads, max_num_splits, row_count);
    constexpr int physical_threads = 256;
    hipLaunchKernelGGL(
        (flash_decoding_mi50_kernel<KV_TYPE, ROW_MODE>),
        split_grid,
        dim3(physical_threads),
        smem_size,
        hip_stream,
        Q,
        K_cache,
        V_cache,
        O_partial,
        m_partial,
        l_partial,
        max_kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        max_num_splits,
        softmax_scale,
        device_params,
        query_rows_per_request,
        num_cus,
        deterministic,
        forced_splits_present,
        forced_splits,
        forced_tpb_present,
        forced_tpb,
        head_start,
        gqa_n_rep);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        printf("[launch_grouped_decode] Decode phase launch failed: %s\n",
               hipGetErrorString(error));
        return -1;
    }

    const dim3 reduce_grid(n_heads, row_count);
    hipLaunchKernelGGL(
        (flash_decoding_reduce_mi50_kernel<ROW_MODE>),
        reduce_grid,
        dim3(256),
        0,
        hip_stream,
        O_partial,
        m_partial,
        l_partial,
        O,
        n_heads,
        head_dim,
        max_num_splits,
        device_params,
        num_cus,
        deterministic,
        forced_splits_present,
        forced_splits,
        forced_tpb_present,
        forced_tpb);
    error = hipGetLastError();
    if (error != hipSuccess)
    {
        printf("[launch_grouped_decode] Reduction launch failed: %s\n",
               hipGetErrorString(error));
        return -1;
    }

    return 0;
}

// =============================================================================
// Extern "C" Wrapper Functions
// =============================================================================

extern "C"
{
    /**
     * @brief Launch Flash Attention 2 prefill kernel for MI50 (FP32 KV)
     *
     * @return 0 on success, -1 on invalid input, -2 on unsupported arch
     */
    int hipFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::FP32>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Attention 2 prefill kernel with FP16 KV cache
     *
     * Reads K/V directly as FP16 from device memory, eliminating the
     * FP32 conversion kernel. For head_dim >= 64 (DIMS_PER_LANE=2), uses
     * v_dot2_f32_f16 for 2× ALU throughput on the Q@K^T dot product.
     * Halves HBM bandwidth for K/V reads vs FP32.
     *
     * @return 0 on success, -1 on error
     */
    int hipFlashAttn_prefill_fa2_fp16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::FP16>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch device-resident prefill over a native BF16 KV cache.
     *
     * BF16 values remain two-byte resident data through HBM and are converted
     * only while the current tile enters LDS.  This avoids the packed FP32
     * shadow whose request stride could not follow a growing captured cache.
     */
    int hipFlashAttn_prefill_fa2_bf16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::BF16>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Attention 2 prefill kernel with Q8_1 KV cache
     *
     * Reads K/V directly as Q8_1Block from device memory, performing inline
     * dequantization (scale × int8 → float) per element. Eliminates the
     * FP32 conversion kernel and reduces HBM bandwidth ~3.5× vs FP32.
     *
     * @return 0 on success, -1 on error
     */
    int hipFlashAttn_prefill_fa2_q8_1(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::Q8_1>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Decoding kernel for MI50 (FP32 KV cache)
     */
    int hipFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_decode<KVTypeTag::FP32>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Decoding kernel for MI50 (FP16 KV cache — native half reads)
     *
     * K_cache/V_cache point to __half data in the same logical layout as FP32.
     * Halves KV read bandwidth; uses fdot2 (v_dot2_f32_f16) for K·Q dot product.
     */
    int hipFlashAttn_decode_fp16(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_decode<KVTypeTag::FP16>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch ordinary split-K decode with native BF16 K/V reads.
     */
    int hipFlashAttn_decode_bf16(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_decode<KVTypeTag::BF16>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Decoding kernel for MI50 (Q8_1 KV cache — inline dequant)
     *
     * K_cache/V_cache point to Q8_1Block data. Each block covers 32 elements.
     * ~3.5× bandwidth savings vs FP32. head_dim must be multiple of 32.
     */
    int hipFlashAttn_decode_q8_1(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        if (head_dim % 32 != 0)
        {
            printf("[hipFlashAttn_decode_q8_1] Error: head_dim must be multiple of 32 for Q8_1 (got %d)\n", head_dim);
            return -1;
        }
        return launch_decode<KVTypeTag::Q8_1>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows through one grouped FP32 decode.
     */
    int hipFlashAttn_decode_fp32_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP32,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows through one grouped FP16 decode.
     */
    int hipFlashAttn_decode_fp16_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP16,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows with native BF16 K/V reads.
     */
    int hipFlashAttn_decode_bf16_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::BF16,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows through one grouped Q8_1 decode.
     */
    int hipFlashAttn_decode_q8_1_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        if (head_dim % 32 != 0)
            return -1;
        return launch_grouped_decode<
            KVTypeTag::Q8_1,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent FP32 request banks through one grouped decode.
     */
    int hipFlashAttn_decode_fp32_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP32,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent FP16 request banks through one grouped decode.
     */
    int hipFlashAttn_decode_fp16_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP16,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent request banks with native BF16 K/V reads.
     */
    int hipFlashAttn_decode_bf16_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::BF16,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent Q8_1 request banks through one grouped decode.
     */
    int hipFlashAttn_decode_q8_1_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        if (head_dim % 32 != 0)
            return -1;
        return launch_grouped_decode<
            KVTypeTag::Q8_1,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Derive AttentionDeviceParams from device-owned KV cache count.
     */
    int hipFlashAttn_prepare_device_params_from_count(
        void *device_params,
        const int *post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride,
        void *stream)
    {
        if (!device_params || !post_append_cached_tokens || seq_len <= 0 ||
            query_rows <= 0 || kv_stride <= 0 ||
            query_rows > MAX_DYNAMIC_ATTENTION_PARAM_ROWS || !stream)
        {
            return -1;
        }

        hipLaunchKernelGGL(hip_derive_attention_params_from_cached_tokens_kernel,
                           dim3(1), dim3(query_rows), 0, static_cast<hipStream_t>(stream),
                           static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
                           post_append_cached_tokens,
                           seq_len,
                           query_rows,
                           kv_stride);
        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            printf("[hipFlashAttn_prepare_device_params_from_count] Kernel launch failed: %s\n",
                   hipGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Write explicit decode geometry to the device-owned param block.
     */
    int hipFlashAttn_prepare_device_params_from_geometry(
        void *device_params,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows,
        void *stream)
    {
        if (!device_params || kv_len <= 0 || kv_stride < kv_len ||
            position_offset < 0 ||
            query_rows <= 0 ||
            query_rows > MAX_DYNAMIC_ATTENTION_PARAM_ROWS || !stream)
        {
            return -1;
        }

        hipLaunchKernelGGL(
            hip_write_attention_params_from_geometry_kernel,
            dim3(1),
            dim3(query_rows),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
            kv_len,
            kv_stride,
            position_offset,
            query_rows);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess)
        {
            printf("[hipFlashAttn_prepare_device_params_from_geometry] "
                   "Kernel launch failed: %s\n",
                   hipGetErrorString(error));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Derive grouped request-row params from contiguous device counts.
     */
    int hipFlashAttn_prepare_device_params_from_request_counts(
        void *device_params,
        const int *post_append_cached_tokens,
        int request_count,
        int query_rows,
        int kv_stride,
        void *stream)
    {
        const int total_rows = request_count * query_rows;
        if (!device_params || !post_append_cached_tokens || request_count < 2 ||
            query_rows <= 0 ||
            total_rows >
                llaminar2::attention::kMaxGroupedVerifierAttentionRows ||
            kv_stride <= 0 || !stream)
        {
            return -1;
        }

        hipLaunchKernelGGL(
            hip_derive_attention_params_from_request_counts_kernel,
            dim3(1),
            dim3(total_rows),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
            post_append_cached_tokens,
            request_count,
            query_rows,
            kv_stride);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess)
        {
            printf("[hipFlashAttn_prepare_device_params_from_request_counts] "
                   "Kernel launch failed: %s\n",
                   hipGetErrorString(error));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Set active HIP device
     */
    int hipFlashAttn_setDevice(int device_idx)
    {
        return llaminar2::HipDeviceGuard::setDevice(device_idx);
    }

    // hipFlashAttn_synchronize() removed - caller manages coherence via events
}
