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
#include <hipblas/hipblas.h>
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

    // =========================================================================
    // KV Cache Type Tags for Template Dispatch
    // =========================================================================
    //
    // The decode kernel is templated on KV_TYPE so it can read K/V
    // directly in their native storage format.
    //   FP32: K/V are float*    (4 bytes/element) — baseline
    //   FP16: K/V are __half*   (2 bytes/element) — 2× BW savings + fdot2
    //   Q8_1: K/V are Q8_1Block* (36 bytes/32 elements) — ~3.5× BW savings
    enum class KVTypeTag { FP32, FP16, Q8_1 };

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
            hipGetDeviceProperties(&prop, device);

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
     * @brief Flash Attention 2 prefill kernel for MI50
     *
     * Uses vectorized FP16 VALU operations for Q@K^T computation.
     * No WMMA/Matrix Cores - pure ALU implementation optimized for Vega.
     */
    __global__ __launch_bounds__(256, 4) void flash_attention_2_mi50_kernel(
        const float *__restrict__ Q,
        const float *__restrict__ K,
        const float *__restrict__ V,
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
        const int effective_gqa_ratio = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
        const int kv_head_idx = (head_start + head_idx) / effective_gqa_ratio;

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
        constexpr int THREADS_PER_ROW = 4;
        const int local_q_row = tid / THREADS_PER_ROW;
        const int row_lane = tid % THREADS_PER_ROW;
        const bool thread_has_row = (local_q_row < q_tile_rows);

        // This tiled kernel runs only for head_dim < 64 (rocBLAS handles >= 64).
        // THREADS_PER_ROW=4 => max ceil(head_dim/4) = 16 dims per thread.
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
        const float *K_batch = K + batch_idx * kv_stride * n_kv_heads * head_dim;
        const float *V_batch = V + batch_idx * kv_stride * n_kv_heads * head_dim;

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

                    const float2 kv = *reinterpret_cast<const float2 *>(
                        K_batch + global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d);
                    const float2 vv = *reinterpret_cast<const float2 *>(
                        V_batch + global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d);

                    reinterpret_cast<__half2 *>(K_dst + local_row * lds_layout.kv_stride + d)[0] = __float22half2_rn(kv);
                    reinterpret_cast<__half2 *>(V_dst + local_row * lds_layout.kv_stride + d)[0] = __float22half2_rn(vv);
                }
            }
            else
            {
                for (int i = tid; i < actual_len * head_dim; i += threads_per_block)
                {
                    int local_row = i / head_dim;
                    int d = i % head_dim;
                    int global_row = kv_start + local_row;

                    float k_val = K_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                    float v_val = V_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];

                    K_dst[local_row * lds_layout.kv_stride + d] = float_to_half(k_val);
                    V_dst[local_row * lds_layout.kv_stride + d] = float_to_half(v_val);
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

                        const float2 kv = *reinterpret_cast<const float2 *>(
                            K_batch + global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d);
                        const float2 vv = *reinterpret_cast<const float2 *>(
                            V_batch + global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d);

                        reinterpret_cast<__half2 *>(K_next + local_row * lds_layout.kv_stride + d)[0] = __float22half2_rn(kv);
                        reinterpret_cast<__half2 *>(V_next + local_row * lds_layout.kv_stride + d)[0] = __float22half2_rn(vv);
                    }
                }
                else
                {
                    for (int i = tid; i < next_len * head_dim; i += threads_per_block)
                    {
                        int local_row = i / head_dim;
                        int d = i % head_dim;
                        int global_row = next_kv_start + local_row;

                        float k_val = K_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                        float v_val = V_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];

                        K_next[local_row * lds_layout.kv_stride + d] = float_to_half(k_val);
                        V_next[local_row * lds_layout.kv_stride + d] = float_to_half(v_val);
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
     * @brief Flash Decoding kernel for single-query decode on MI50
     *        v3: wavefront-cooperative KV processing + exp2f intrinsic
     */
    template <KVTypeTag KV_TYPE = KVTypeTag::FP32>
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
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        int kv_len_runtime = kv_len;
        int kv_batch_stride = kv_len;
        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa_ratio = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
        const int kv_head_idx = (head_start + head_idx) / effective_gqa_ratio;

        // Split range
        const int split_size = (kv_len_runtime + num_splits - 1) / num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len_runtime);

        const int partial_idx = (batch_idx * n_heads + head_idx) * num_splits + split_idx;

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        const int wavefront_id = tid / WAVEFRONT_SIZE;
        const int lane_id = tid % WAVEFRONT_SIZE;
        const int num_wavefronts = num_threads / WAVEFRONT_SIZE;

        // Handle empty splits
        if (kv_start >= kv_len_runtime)
        {
            if (tid == 0)
            {
                m_partial[partial_idx] = -FLT_MAX;
                l_partial[partial_idx] = 0.0f;
            }
            for (int d = tid * 2; d + 1 < head_dim; d += num_threads * 2)
            {
                const int base = (batch_idx * n_heads + head_idx) * head_dim;
                const int idx0 = (base + d + 0) * num_splits + split_idx;
                const int idx1 = (base + d + 1) * num_splits + split_idx;
                O_partial[idx0] = 0.0f;
                O_partial[idx1] = 0.0f;
            }
            for (int d = (head_dim & ~1) + tid; d < head_dim; d += num_threads)
            {
                const int base = (batch_idx * n_heads + head_idx) * head_dim;
                const int idx = (base + d) * num_splits + split_idx;
                O_partial[idx] = 0.0f;
            }
            return;
        }

        // Load Q into LDS (all threads cooperate)
        // FP16 KV path: store Q as half in LDS (convert once, use many times)
        // FP32/Q8_1 paths: store Q as float in LDS (matching their dot product type)
        extern __shared__ char smem[];
        [[maybe_unused]] float *Q_shared = nullptr;
        [[maybe_unused]] __half *Q_shared_h = nullptr;

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;

        if constexpr (KV_TYPE == KVTypeTag::FP16)
        {
            // Store Q as FP16 in LDS — eliminates per-KV-position FP32→FP16 conversion
            // Halves LDS bandwidth for Q reads (4B → 2B per element per lane)
            Q_shared_h = reinterpret_cast<__half *>(smem);
            for (int d = tid; d < head_dim; d += num_threads)
            {
                Q_shared_h[d] = __float2half(Q_ptr[d]);
            }
        }
        else
        {
            Q_shared = reinterpret_cast<float *>(smem);
            for (int d = tid * 2; d + 1 < head_dim; d += num_threads * 2)
            {
                const float2 qv = reinterpret_cast<const float2 *>(Q_ptr + d)[0];
                reinterpret_cast<float2 *>(Q_shared + d)[0] = qv;
            }
            for (int d = (head_dim & ~1) + tid; d < head_dim; d += num_threads)
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

        size_t row_byte_stride;
        size_t base_byte_off;
        if constexpr (KV_TYPE == KVTypeTag::Q8_1) {
            row_byte_stride = static_cast<size_t>(q8_bpr) * sizeof(llaminar2::Q8_1Block);
            base_byte_off = static_cast<size_t>(batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_idx) * q8_bph * sizeof(llaminar2::Q8_1Block);
        } else if constexpr (KV_TYPE == KVTypeTag::FP16) {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(__half);
            base_byte_off = static_cast<size_t>(batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(__half);
        } else {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(float);
            base_byte_off = static_cast<size_t>(batch_idx) * kv_batch_stride * row_byte_stride
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
        for (int kv_pos = kv_start + wavefront_id; kv_pos < kv_end; kv_pos += num_wavefronts)
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
        if (lane_id == 0)
        {
            block_m[wavefront_id] = m_local;
            block_l[wavefront_id] = l_local;
        }
        if (head_dim == 128)
        {
            const int base = wavefront_id * 128;
            block_O[base + lane_id] = O_lane[0];
            block_O[base + lane_id + WAVEFRONT_SIZE] = O_lane[1];
        }
        else
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
        if (wavefront_id == 0)
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
    __global__ __launch_bounds__(256, 4)
    void flash_decoding_reduce_mi50_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ m_partial,
        const float *__restrict__ l_partial,
        float *__restrict__ O,
        int n_heads,
        int head_dim,
        int num_splits)
    {
        const int head_idx = blockIdx.x;
        const int batch_idx = blockIdx.y;

        const int tid = threadIdx.x;
        const int base_idx = (batch_idx * n_heads + head_idx) * num_splits;

        // Find global max and compute scales
        __shared__ float global_m;
        __shared__ float global_l;
        __shared__ float split_scales[32]; // Support up to 32 splits

        if (tid == 0)
        {
            float m_max = -FLT_MAX;
            for (int s = 0; s < num_splits; s++)
            {
                m_max = fmaxf(m_max, m_partial[base_idx + s]);
            }
            global_m = m_max;

            float l_sum = 0.0f;
            for (int s = 0; s < num_splits; s++)
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
            for (int s = 0; s < num_splits; s++)
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
            for (int s = 0; s < num_splits; s++)
            {
                O_sum += split_scales[s] * O_partial[split_base + s];
            }
            O_out[d] = O_sum * inv_l;
        }
    }

} // anonymous namespace

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
    hipGetDevice(&current_device);
    const int num_cus = getDeviceConfig(current_device).num_cus;

    const DecodeAutotuneSelection tuned = chooseDecodeLaunchConfig(kv_len, head_dim, num_splits, n_heads, num_cus);
    const int effective_num_splits = tuned.config.splits;
    const int decode_threads_per_block = tuned.config.tpb;

    // Autotuning MUST be disabled during HIP stream capture.
    // hipEventCreate/Synchronize/Destroy are forbidden during capture and
    // will invalidate the capture graph.
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    if (hip_stream)
        hipStreamIsCapturing(hip_stream, &capture_status);
    const bool is_capturing = (capture_status == hipStreamCaptureStatusActive);

    hipEvent_t tune_start = nullptr;
    hipEvent_t tune_stop = nullptr;
    const bool measure_tuning = tuned.should_time && !is_capturing;
    if (measure_tuning)
    {
        if (hipEventCreate(&tune_start) == hipSuccess && hipEventCreate(&tune_stop) == hipSuccess)
        {
            hipEventRecord(tune_start, hip_stream);
        }
        else
        {
            if (tune_start)
                hipEventDestroy(tune_start);
            if (tune_stop)
                hipEventDestroy(tune_stop);
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
        (flash_decoding_mi50_kernel<KV_TYPE>),
        grid_split, block_split, smem_size, hip_stream,
        Q, K_cache, V_cache,
        O_partial, m_partial, l_partial,
        kv_len, n_heads, n_kv_heads, head_dim,
        effective_num_splits, softmax_scale, device_params,
        head_start, gqa_n_rep);

    // Reduction kernel (unchanged — operates on FP32 partials only)
    dim3 grid_reduce(n_heads, batch_size);
    dim3 block_reduce(256);

    hipLaunchKernelGGL(
        flash_decoding_reduce_mi50_kernel,
        grid_reduce, block_reduce, 0, hip_stream,
        O_partial, m_partial, l_partial, O,
        n_heads, head_dim, effective_num_splits);

    if (tune_start && tune_stop)
    {
        hipEventRecord(tune_stop, hip_stream);
        if (hipEventSynchronize(tune_stop) == hipSuccess)
        {
            float elapsed_ms = 0.0f;
            if (hipEventElapsedTime(&elapsed_ms, tune_start, tune_stop) == hipSuccess)
            {
                recordDecodeAutotuneResult(tuned.key, tuned.candidate_index, elapsed_ms);
            }
        }
        hipEventDestroy(tune_start);
        hipEventDestroy(tune_stop);
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

// =============================================================================
// rocBLAS-based Prefill Attention (Two-pass GEMM + Causal Softmax)
// =============================================================================
// Strategy: use rocBLAS (Tensile) tiled GEMMs for QK^T and PV, connected by a
// custom causal softmax kernel. This gives 24-32× better data reuse per tile
// compared to streaming K/V from HBM per query position.
//
// Per head:
//   1) S = (1/√d) * Q_h @ K_{kh}^T    [seq × kv_len]  via hipBLAS SGEMM
//   2) P = causal_softmax(S)            [seq × kv_len]  custom HIP kernel
//   3) O_h = P @ V_{kh}                [seq × hdim]    via hipBLAS SGEMM
//
// Memory layout (all row-major, interleaved heads):
//   Q[batch, seq, n_heads * head_dim]   — Q_h at offset h*head_dim, stride n_heads*head_dim
//   K[batch, kv,  n_kv_heads * head_dim]
//   V[batch, kv,  n_kv_heads * head_dim]
//   O[batch, seq, n_heads * head_dim]
// =============================================================================

/**
 * @brief Row-wise causal softmax (2-pass): P[i,j] = softmax(S[i, 0..i+pos_off])
 *
 * Optimizations vs the original 3-pass kernel:
 *   - softmax_scale is folded into the QK^T GEMM alpha, so scores arrive pre-scaled
 *   - Pass 1 is read-only (find max), eliminating seq×kv_len global writes
 *   - Pass 2 fuses exp+sum and normalize: each thread holds exp values in registers,
 *     reduces the sum, then writes final normalized values — 1 read + 1 write total
 *
 * Memory traffic: 2 reads + 1 write per element (was 3 reads + 3 writes).
 *
 * Each block processes one row. 256 threads (4 wavefronts) cooperate.
 * Register budget: MAX_ELEMS_PER_THREAD × 4 bytes. For kv_len ≤ 8192 with 256
 * threads, that's ≤ 32 registers per thread — well within gfx906's 256 VGPRs.
 */
__global__ __launch_bounds__(256, 4)
void causal_softmax_inplace_kernel(
    float* __restrict__ S,
    int total_rows,
    int seq_len,
    int kv_len,
    int position_offset)
{
    const int row = blockIdx.x;
    if (row >= total_rows) return;

    float* row_data = S + static_cast<size_t>(row) * kv_len;
    const int seq_pos = row % seq_len;
    const int last_valid = seq_pos + position_offset;
    constexpr int WARP_SIZE = 64;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Pass 1: Find row max (read-only — scores are pre-scaled by GEMM alpha)
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        if (j <= last_valid) {
            local_max = fmaxf(local_max, row_data[j]);
        }
    }

    // Wavefront-level max reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));

    __shared__ float shared_max[4];
    if (lane_id == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    float row_max;
    if (threadIdx.x < 4) {
        row_max = shared_max[threadIdx.x];
        for (int offset = 2; offset > 0; offset >>= 1)
            row_max = fmaxf(row_max, __shfl_xor(row_max, offset));
        shared_max[0] = row_max;
    }
    __syncthreads();
    row_max = shared_max[0];

    // Pass 2: Fused exp + sum + normalize
    // Each thread computes exp values into a register array, reduces the sum,
    // then writes normalized values — one read + one write per element.
    constexpr int MAX_ELEMS_PER_THREAD = 32;  // Supports kv_len ≤ 8192 @ 256 threads
    constexpr float LOG2E = 1.4426950408889634f;
    float local_exp[MAX_ELEMS_PER_THREAD];
    float local_sum = 0.0f;

    const int elems_per_thread = (kv_len + blockDim.x - 1) / blockDim.x;

    // Sub-pass 2a: compute exp values into registers, accumulate sum
    for (int i = 0; i < elems_per_thread; i++) {
        int j = threadIdx.x + i * blockDim.x;
        float val = 0.0f;
        if (j < kv_len && j <= last_valid) {
            val = exp2f((row_data[j] - row_max) * LOG2E);
        }
        local_exp[i] = val;
        local_sum += val;
    }

    // Wavefront-level sum reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_xor(local_sum, offset);

    __shared__ float shared_sum[4];
    if (lane_id == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    float row_sum;
    if (threadIdx.x < 4) {
        row_sum = shared_sum[threadIdx.x];
        for (int offset = 2; offset > 0; offset >>= 1)
            row_sum += __shfl_xor(row_sum, offset);
        shared_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = shared_sum[0];

    // Sub-pass 2b: normalize and write
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int i = 0; i < elems_per_thread; i++) {
        int j = threadIdx.x + i * blockDim.x;
        if (j < kv_len) {
            row_data[j] = local_exp[i] * inv_sum;
        }
    }
}

namespace {

// Cached hipBLAS handles (one per device, created on first use)
static hipblasHandle_t get_cached_rocblas_handle(int device_id) {
    static std::mutex mtx;
    static std::unordered_map<int, hipblasHandle_t> handles;

    std::lock_guard<std::mutex> lock(mtx);
    auto it = handles.find(device_id);
    if (it != handles.end()) return it->second;

    hipblasHandle_t h = nullptr;
    hipblasStatus_t status = hipblasCreate(&h);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("[rocblas_prefill] Failed to create hipBLAS handle for device %d\n", device_id);
        return nullptr;
    }
    // Disable atomic reductions inside rocBLAS/Tensile GEMM kernels.
    // Atomic accumulation produces nondeterministic floating-point reduction
    // order across runs, which causes parity tests to drift ~14x in KL
    // divergence and occasionally fail tight thresholds. The deterministic
    // (non-atomic) Tensile paths are selected instead; perf impact is minor
    // on gfx906 since SGEMM there is compute-bound, not reduction-bound.
    hipblasSetAtomicsMode(h, HIPBLAS_ATOMICS_NOT_ALLOWED);
    handles[device_id] = h;
    return h;
}

// ─── Multi-stream infrastructure for GQA group pipelining ───
// Each stream gets its own hipBLAS handle and scores buffer so that
// consecutive GQA groups can overlap QK^T / softmax / PV across streams.
constexpr int NUM_PREFILL_STREAMS = 2;

struct PrefillStreamCtx {
    hipStream_t stream = nullptr;
    hipblasHandle_t handle = nullptr;
    float* scores_buf = nullptr;
    size_t scores_cap = 0;
    hipEvent_t completion_event = nullptr;  // Signals when this stream's work is done
};

// Per-device input readiness event: recorded on the caller's stream so pool
// streams can wait for Q/K/V data to be fully written before reading.
struct PrefillPoolState {
    std::array<PrefillStreamCtx, NUM_PREFILL_STREAMS> streams;
    hipEvent_t input_ready = nullptr;  // Recorded on caller's stream before dispatch
};

static std::mutex s_stream_mtx;
static std::unordered_map<int, PrefillPoolState> s_stream_pools;

static PrefillPoolState& get_stream_pool(int device_id) {
    std::lock_guard<std::mutex> lock(s_stream_mtx);
    auto it = s_stream_pools.find(device_id);
    if (it != s_stream_pools.end()) return it->second;

    auto& state = s_stream_pools[device_id];
    for (int i = 0; i < NUM_PREFILL_STREAMS; i++) {
        hipStreamCreateWithFlags(&state.streams[i].stream, hipStreamNonBlocking);
        hipblasCreate(&state.streams[i].handle);
        hipblasSetStream(state.streams[i].handle, state.streams[i].stream);
        // Disable atomic reductions inside rocBLAS/Tensile GEMM kernels for
        // deterministic output (see get_cached_rocblas_handle for rationale).
        hipblasSetAtomicsMode(state.streams[i].handle, HIPBLAS_ATOMICS_NOT_ALLOWED);
        hipEventCreateWithFlags(&state.streams[i].completion_event, hipEventDisableTiming);
    }
    hipEventCreateWithFlags(&state.input_ready, hipEventDisableTiming);
    return state;
}

static float* ensure_scores_buffer(PrefillStreamCtx& ctx, size_t required_bytes) {
    if (required_bytes > ctx.scores_cap) {
        if (ctx.scores_buf) hipFree(ctx.scores_buf);
        hipError_t err = hipMalloc(&ctx.scores_buf, required_bytes);
        if (err != hipSuccess) {
            ctx.scores_buf = nullptr;
            ctx.scores_cap = 0;
            return nullptr;
        }
        ctx.scores_cap = required_bytes;
    }
    return ctx.scores_buf;
}

/**
 * @brief rocBLAS two-pass prefill attention (FP32 KV)
 *
 * Optimizations over the initial implementation:
 *   - softmax_scale folded into QK^T GEMM alpha (eliminates per-element multiply)
 *   - Multi-stream pipelining: 2 HIP streams overlap consecutive GQA groups
 *     (group N's PV overlaps with group N+1's QK^T)
 *   - Per-stream scores buffers (no contention, independent lifetimes)
 *   - PV GEMM uses OP_T for V to hit Tensile's optimized (OP_T, OP_N) path
 *   - 2-pass softmax: fused exp+sum+normalize eliminates one full R/W pass
 *
 * Uses hipblasSgemmStridedBatched to process all Q heads within each GQA group
 * in a single GEMM call (one QK^T batch + one PV batch per KV head).
 * For Qwen 7B (28h, 4kv): 8 GEMMs + 4 softmax = 12 launches across 2 streams.
 */
static int rocblas_prefill_impl(
    const float *Q, const float *K, const float *V, float *O,
    int batch_size, int seq_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim,
    float softmax_scale, bool causal, int position_offset,
    hipStream_t hip_stream,
    int head_start = 0, int gqa_n_rep = 0)
{
    int device_id = 0;
    hipGetDevice(&device_id);

    auto& pool_state = get_stream_pool(device_id);
    auto& pool = pool_state.streams;

    // ─── Input fence: pool streams must wait for caller's stream ───
    // Q/K/V data is written by upstream stages (QKV projection, RoPE, KV append)
    // on hip_stream. Pool streams use hipStreamNonBlocking, which does NOT
    // implicitly sync with the caller's stream. Without this fence, pool
    // streams can read stale Q/K/V data from prior pipeline iterations.
    hipEventRecord(pool_state.input_ready, hip_stream);
    for (int s = 0; s < NUM_PREFILL_STREAMS; s++) {
        hipStreamWaitEvent(pool[s].stream, pool_state.input_ready, 0);
    }

    const int gqa_ratio = n_heads / n_kv_heads;
    // Effective GQA repetition: use global ratio when provided (TP replicated KV)
    const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : gqa_ratio;

    // Per-stream scores buffer: sized for max local Q heads in any single KV group
    const size_t scores_per_head = static_cast<size_t>(seq_len) * kv_len;
    // When using global GQA mapping, a single KV head may serve up to n_heads local Q heads
    const int max_group_size = (gqa_n_rep > 0) ? n_heads : gqa_ratio;
    const size_t scores_per_group = static_cast<size_t>(max_group_size) * scores_per_head;
    const size_t scores_bytes = scores_per_group * sizeof(float);

    // Ensure all stream buffers are large enough
    for (int s = 0; s < NUM_PREFILL_STREAMS; s++) {
        if (!ensure_scores_buffer(pool[s], scores_bytes)) return -1;
    }

    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;

    // Leading dimensions (row strides for interleaved head layout)
    const int lda_q = n_heads * head_dim;
    const int lda_k = n_kv_heads * head_dim;
    const int lda_v = n_kv_heads * head_dim;
    const int lda_o = n_heads * head_dim;

    // QK^T strides (within a GQA group)
    const long long strideB_qkt = head_dim;             // Q: consecutive heads
    const long long strideA_qkt = 0;                    // K: shared across group
    const long long strideC_qkt = static_cast<long long>(scores_per_head);

    // PV strides — O^T = V^T @ P^T with OP_T on V
    // V is row-major with stride lda_v; transposing gives column-major access
    const long long strideA_pv = 0;                     // V: shared across group
    const long long strideB_pv = static_cast<long long>(scores_per_head);
    const long long strideC_pv = head_dim;              // O: interleaved heads

    int pos_off = causal ? position_offset : (kv_len - 1);

    for (int b = 0; b < batch_size; b++) {
        const float* Q_batch = Q + static_cast<size_t>(b) * seq_len * lda_q;
        const float* K_batch = K + static_cast<size_t>(b) * kv_len * lda_k;
        const float* V_batch = V + static_cast<size_t>(b) * kv_len * lda_v;
        float* O_batch = O + static_cast<size_t>(b) * seq_len * lda_o;

        for (int kh = 0; kh < n_kv_heads; kh++) {
            // TP-aware GQA mapping: compute which LOCAL Q heads map to this KV head
            // Global Q heads [kh * effective_gqa, (kh+1) * effective_gqa) use KV head kh
            // Local Q heads are [0, n_heads) corresponding to global [head_start, head_start + n_heads)
            int local_first, local_last;
            if (gqa_n_rep > 0) {
                int global_q_first = kh * effective_gqa;
                int global_q_last = (kh + 1) * effective_gqa - 1;
                local_first = max(0, global_q_first - head_start);
                local_last = min(n_heads - 1, global_q_last - head_start);
            } else {
                // Standard (non-TP) path: uniform gqa_ratio heads per KV head
                local_first = kh * gqa_ratio;
                local_last = local_first + gqa_ratio - 1;
            }
            if (local_first > local_last) continue;  // No local Q heads for this KV head
            const int local_group_size = local_last - local_first + 1;

            const int stream_idx = kh % NUM_PREFILL_STREAMS;
            auto& ctx = pool[stream_idx];

            const float* Q_group = Q_batch + local_first * head_dim;
            const float* K_kh = K_batch + kh * head_dim;
            const float* V_kh = V_batch + kh * head_dim;
            float* O_group = O_batch + local_first * head_dim;

            // ─── Pass 1: S = softmax_scale * Q @ K^T  (scale baked into alpha) ───
            hipblasStatus_t status = hipblasSgemmStridedBatched(
                ctx.handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                kv_len, seq_len, head_dim,
                &softmax_scale,                              // α = 1/√d (#4)
                K_kh, lda_k, strideA_qkt,
                Q_group, lda_q, strideB_qkt,
                &beta_zero,
                ctx.scores_buf, kv_len, strideC_qkt,
                local_group_size);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                printf("[rocblas_prefill] QK^T GEMM failed: %d (kv_head %d)\n", (int)status, kh);
                return -1;
            }

            // ─── Pass 2: causal softmax (no scale — already applied) ───
            dim3 grid_sm(local_group_size * seq_len);
            dim3 block_sm(256);
            hipLaunchKernelGGL(
                causal_softmax_inplace_kernel,
                grid_sm, block_sm, 0, ctx.stream,
                ctx.scores_buf, local_group_size * seq_len, seq_len, kv_len, pos_off);

            // ─── Pass 3: O = P @ V ───
            status = hipblasSgemmStridedBatched(
                ctx.handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                head_dim, seq_len, kv_len,
                &alpha_one,
                V_kh, lda_v, strideA_pv,
                ctx.scores_buf, kv_len, strideB_pv,
                &beta_zero,
                O_group, lda_o, strideC_pv,
                local_group_size);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                printf("[rocblas_prefill] PV GEMM failed: %d (kv_head %d)\n", (int)status, kh);
                return -1;
            }
        }

        // ─── Output fence: rejoin pool streams to caller's stream ───
        // Event fence: makes hip_stream aware of pool completion so
        // subsequent GPU work on hip_stream waits for attention output.
        for (int s = 0; s < NUM_PREFILL_STREAMS; s++) {
            hipEventRecord(pool[s].completion_event, pool[s].stream);
            hipStreamWaitEvent(hip_stream, pool[s].completion_event, 0);
        }
        // CPU-side sync: ensures pool work is fully complete before the CPU
        // returns. This is needed because hipStreamWaitEvent only creates a
        // GPU-side dependency — CPU-side operations (e.g., hipMemcpy for
        // coherence transfers) need the data to be globally visible.
        for (int s = 0; s < NUM_PREFILL_STREAMS; s++) {
            hipStreamSynchronize(pool[s].stream);
        }
    }

    return 0;
}

// =============================================================================
// FP16/Q8_1 KV Dequantization Kernels + rocBLAS Wrapper
// =============================================================================
// Strategy: dequantize K/V to a temporary FP32 buffer, then reuse the FP32
// rocblas_prefill_impl. This avoids per-position HBM streaming and leverages
// rocBLAS Tensile GEMM tiles for optimal throughput on gfx906.
//
// The dequant + SGEMM path trades a small dequant kernel launch for dramatically
// better GEMM data reuse. On MI50 (no matrix cores), FP32 SGEMM is already the
// optimal compute path, so pre-dequanting to FP32 has zero compute penalty.
// =============================================================================

/**
 * @brief Dequantize interleaved FP16 KV to FP32
 *
 * Layout: [kv_len, n_kv_heads * head_dim] — one element per thread.
 * Each FP16 value is converted in-place to FP32 in the output buffer,
 * preserving the interleaved head layout for rocBLAS.
 */
__global__ void dequant_fp16_to_fp32_kernel(
    const half* __restrict__ src,
    float* __restrict__ dst,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = __half2float(src[idx]);
    }
}

/**
 * @brief Dequantize interleaved Q8_1 KV to FP32
 *
 * Layout: K/V are stored as Q8_1Blocks in interleaved head layout:
 *   [kv_len, n_kv_heads, blocks_per_head] where blocks_per_head = head_dim / 32
 *
 * Each thread processes one Q8_1Block (32 elements): scale × int8 → float.
 * Output is written to the matching position in FP32 interleaved layout.
 */
__global__ void dequant_q8_1_to_fp32_kernel(
    const llaminar2::Q8_1Block* __restrict__ src,
    float* __restrict__ dst,
    int total_blocks,
    int n_kv_heads,
    int head_dim)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    constexpr int BLOCK_SIZE = 32;  // Q8_1Block::BLOCK_SIZE
    const int blocks_per_head = head_dim / BLOCK_SIZE;

    // Decode block position in [kv_pos, kv_head, block_within_head]
    const int blocks_per_row = n_kv_heads * blocks_per_head;
    const int kv_pos = block_idx / blocks_per_row;
    const int rem = block_idx % blocks_per_row;
    const int kv_head = rem / blocks_per_head;
    const int blk = rem % blocks_per_head;

    const auto& block = src[block_idx];
    // FP16 scale → FP32
    float scale = __half2float(*reinterpret_cast<const half*>(&block.d));

    // Output offset in FP32 interleaved layout: [kv_pos, kv_head * head_dim + blk * 32]
    int dst_base = kv_pos * (n_kv_heads * head_dim) + kv_head * head_dim + blk * BLOCK_SIZE;

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        dst[dst_base + i] = scale * static_cast<float>(block.qs[i]);
    }
}

// Cached dequant buffer (grow-only, per-device, for FP16/Q8_1 → FP32 KV)
struct DequantBuf {
    float* ptr = nullptr;
    size_t cap = 0;
};
static std::mutex s_dequant_mtx;
static std::unordered_map<int, DequantBuf> s_dequant_bufs;

static float* get_dequant_buffer(int device_id, size_t required_bytes) {
    std::lock_guard<std::mutex> lock(s_dequant_mtx);
    auto& buf = s_dequant_bufs[device_id];
    if (required_bytes > buf.cap) {
        if (buf.ptr) hipFree(buf.ptr);
        hipError_t err = hipMalloc(&buf.ptr, required_bytes);
        if (err != hipSuccess) {
            printf("[rocblas_prefill] hipMalloc failed for dequant buffer (%zu bytes)\n", required_bytes);
            buf.ptr = nullptr;
            buf.cap = 0;
            return nullptr;
        }
        buf.cap = required_bytes;
    }
    return buf.ptr;
}

/**
 * @brief rocBLAS prefill with FP16 KV cache
 *
 * Dequantizes K/V from FP16 to FP32 in a shared buffer, then delegates
 * to rocblas_prefill_impl. Single dequant buffer holds both K and V
 * sequentially: [K_fp32 | V_fp32].
 */
static int rocblas_prefill_fp16_impl(
    const float *Q, const void *K_fp16, const void *V_fp16, float *O,
    int batch_size, int seq_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim,
    float softmax_scale, bool causal, int position_offset,
    hipStream_t hip_stream,
    int head_start = 0, int gqa_n_rep = 0)
{
    int device_id = 0;
    hipGetDevice(&device_id);

    // Dequant buffer: space for both K and V as FP32
    const size_t kv_elements = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t dequant_bytes = 2 * kv_elements * sizeof(float);  // K + V
    float* dequant_buf = get_dequant_buffer(device_id, dequant_bytes);
    if (!dequant_buf) return -1;

    float* K_fp32 = dequant_buf;
    float* V_fp32 = dequant_buf + kv_elements;

    // Launch dequant kernels (can overlap on same stream since K and V are independent outputs)
    constexpr int DEQUANT_BLOCK = 256;
    int dequant_grid = (kv_elements + DEQUANT_BLOCK - 1) / DEQUANT_BLOCK;

    hipLaunchKernelGGL(dequant_fp16_to_fp32_kernel,
        dim3(dequant_grid), dim3(DEQUANT_BLOCK), 0, hip_stream,
        static_cast<const half*>(K_fp16), K_fp32, static_cast<int>(kv_elements));

    hipLaunchKernelGGL(dequant_fp16_to_fp32_kernel,
        dim3(dequant_grid), dim3(DEQUANT_BLOCK), 0, hip_stream,
        static_cast<const half*>(V_fp16), V_fp32, static_cast<int>(kv_elements));

    // Delegate to FP32 rocBLAS path
    return rocblas_prefill_impl(
        Q, K_fp32, V_fp32, O,
        batch_size, seq_len, kv_len,
        n_heads, n_kv_heads, head_dim,
        softmax_scale, causal, position_offset,
        hip_stream,
        head_start, gqa_n_rep);
}

/**
 * @brief rocBLAS prefill with Q8_1 KV cache
 *
 * Dequantizes K/V from Q8_1Block to FP32, then delegates to rocblas_prefill_impl.
 */
static int rocblas_prefill_q8_1_impl(
    const float *Q, const void *K_q8, const void *V_q8, float *O,
    int batch_size, int seq_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim,
    float softmax_scale, bool causal, int position_offset,
    hipStream_t hip_stream,
    int head_start = 0, int gqa_n_rep = 0)
{
    int device_id = 0;
    hipGetDevice(&device_id);

    const size_t kv_elements = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t dequant_bytes = 2 * kv_elements * sizeof(float);
    float* dequant_buf = get_dequant_buffer(device_id, dequant_bytes);
    if (!dequant_buf) return -1;

    float* K_fp32 = dequant_buf;
    float* V_fp32 = dequant_buf + kv_elements;

    // Total Q8_1 blocks for one K or V tensor
    const int blocks_per_head = head_dim / 32;
    const int total_blocks = kv_len * n_kv_heads * blocks_per_head;

    constexpr int DEQUANT_BLOCK = 256;
    int dequant_grid = (total_blocks + DEQUANT_BLOCK - 1) / DEQUANT_BLOCK;

    hipLaunchKernelGGL(dequant_q8_1_to_fp32_kernel,
        dim3(dequant_grid), dim3(DEQUANT_BLOCK), 0, hip_stream,
        static_cast<const llaminar2::Q8_1Block*>(K_q8), K_fp32,
        total_blocks, n_kv_heads, head_dim);

    hipLaunchKernelGGL(dequant_q8_1_to_fp32_kernel,
        dim3(dequant_grid), dim3(DEQUANT_BLOCK), 0, hip_stream,
        static_cast<const llaminar2::Q8_1Block*>(V_q8), V_fp32,
        total_blocks, n_kv_heads, head_dim);

    return rocblas_prefill_impl(
        Q, K_fp32, V_fp32, O,
        batch_size, seq_len, kv_len,
        n_heads, n_kv_heads, head_dim,
        softmax_scale, causal, position_offset,
        hip_stream,
        head_start, gqa_n_rep);
}

} // anonymous namespace

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
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        // Input validation
        if (head_dim <= 0 || head_dim > 256)
        {
            printf("[hipFlashAttn_prefill_fa2] Error: head_dim=%d must be in [1, 256]\n", head_dim);
            return -1;
        }
        if (n_heads % n_kv_heads != 0)
        {
            printf("[hipFlashAttn_prefill_fa2] Error: n_heads must be divisible by n_kv_heads\n");
            return -1;
        }

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // rocBLAS two-pass prefill: uses Tensile-optimized GEMM tiles for QK^T and PV.
        if (head_dim >= 64) {
            return rocblas_prefill_impl(
                Q, K, V, O,
                batch_size, seq_len, kv_len,
                n_heads, n_kv_heads, head_dim,
                softmax_scale, causal, position_offset,
                hip_stream, head_start, gqa_n_rep);
        }

        // Fall through to existing tiled kernel for head_dim < 64

        // Adaptively select tile sizes based on LDS constraints.
        // Prefer larger tile area, then larger tile_q (better Q reuse / fewer blocks).
        int tile_q = FA2_TILE_Q;
        int tile_kv = FA2_TILE_KV;
        FA2LDSLayout lds_layout;

        struct TileCandidate
        {
            int q;
            int kv;
        };

        // Keep candidates conservative and power-of-two for predictable behavior.
        // NOTE: 64x32 is intentionally included; for head_dim=128 it often fits
        // where 64x64 does not, and is usually better than dropping to 32x32.
        constexpr TileCandidate candidates[] = {
            {64, 64},
            {64, 32},
            {32, 64},
            {32, 32},
            {16, 32},
        };

        int best_q = -1;
        int best_kv = -1;
        int best_area = -1;
        FA2LDSLayout best_layout{};

        for (const auto &c : candidates)
        {
            FA2LDSLayout cand_layout;
            cand_layout.compute(head_dim, c.q, c.kv);
            if (cand_layout.total_size > LDS_SIZE_BYTES)
                continue;

            const int area = c.q * c.kv;
            const bool better =
                (area > best_area) ||
                (area == best_area && c.q > best_q);

            if (better)
            {
                best_q = c.q;
                best_kv = c.kv;
                best_area = area;
                best_layout = cand_layout;
            }
        }

        if (best_area < 0)
        {
            printf("[hipFlashAttn_prefill_fa2] Error: Cannot fit head_dim=%d in LDS with supported tiles\n",
                   head_dim);
            return -1;
        }

        tile_q = best_q;
        tile_kv = best_kv;
        lds_layout = best_layout;

        // Grid configuration
        const int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
        dim3 grid(n_heads, num_q_tiles, batch_size);
        dim3 block(FA2_THREADS_PER_BLOCK);

        // Launch kernel
        hipLaunchKernelGGL(
            flash_attention_2_mi50_kernel,
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep);

        (void)hipGetLastError();  // Clear stale errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            printf("[hipFlashAttn_prefill_fa2] Kernel launch failed: %s\n", hipGetErrorString(err));
            return -1;
        }

        return 0;
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
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        if (head_dim <= 0 || head_dim > 256)
        {
            printf("[hipFlashAttn_prefill_fa2_fp16] Error: head_dim=%d must be in [1, 256]\n", head_dim);
            return -1;
        }
        if (n_heads % n_kv_heads != 0)
        {
            printf("[hipFlashAttn_prefill_fa2_fp16] Error: n_heads must be divisible by n_kv_heads\n");
            return -1;
        }
        if (head_dim < 64)
        {
            printf("[hipFlashAttn_prefill_fa2_fp16] Error: FP16 KV requires head_dim >= 64 (got %d)\n", head_dim);
            return -1;
        }

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // rocBLAS path: dequant FP16→FP32 then Tensile SGEMM
        return rocblas_prefill_fp16_impl(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, position_offset,
            hip_stream,
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
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        if (head_dim <= 0 || head_dim > 256)
        {
            printf("[hipFlashAttn_prefill_fa2_q8_1] Error: head_dim=%d must be in [1, 256]\n", head_dim);
            return -1;
        }
        if (n_heads % n_kv_heads != 0)
        {
            printf("[hipFlashAttn_prefill_fa2_q8_1] Error: n_heads must be divisible by n_kv_heads\n");
            return -1;
        }
        if (head_dim < 64)
        {
            printf("[hipFlashAttn_prefill_fa2_q8_1] Error: Q8_1 KV requires head_dim >= 64 (got %d)\n", head_dim);
            return -1;
        }
        if (head_dim % 32 != 0)
        {
            printf("[hipFlashAttn_prefill_fa2_q8_1] Error: head_dim must be multiple of 32 for Q8_1 (got %d)\n", head_dim);
            return -1;
        }

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // rocBLAS path: dequant Q8_1→FP32 then Tensile SGEMM
        return rocblas_prefill_q8_1_impl(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, position_offset,
            hip_stream,
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
     * @brief Allocate workspace for Flash Decoding
     */
    int hipFlashAttn_allocWorkspace(
        void **partial_output, void **partial_m, void **partial_l,
        int batch_size, int n_heads, int head_dim, int num_splits)
    {
        size_t num_partials = static_cast<size_t>(batch_size) * n_heads * num_splits;

        size_t output_size = num_partials * head_dim * sizeof(float);
        size_t m_size = num_partials * sizeof(float);
        size_t l_size = num_partials * sizeof(float);

        if (hipMalloc(partial_output, output_size) != hipSuccess)
            return -1;
        if (hipMalloc(partial_m, m_size) != hipSuccess)
        {
            hipFree(*partial_output);
            return -1;
        }
        if (hipMalloc(partial_l, l_size) != hipSuccess)
        {
            hipFree(*partial_output);
            hipFree(*partial_m);
            return -1;
        }

        return 0;
    }

    /**
     * @brief Free workspace for Flash Decoding
     */
    void hipFlashAttn_freeWorkspace(void *partial_output, void *partial_m, void *partial_l)
    {
        if (partial_output)
            hipFree(partial_output);
        if (partial_m)
            hipFree(partial_m);
        if (partial_l)
            hipFree(partial_l);
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
