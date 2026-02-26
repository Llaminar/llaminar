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
    // The cooperative prefill kernel is templated on KV_TYPE so it can read K/V
    // directly in their native storage format, eliminating the FP32 conversion
    // kernel and halving HBM bandwidth for FP16.
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

    inline int chooseDecodeSplitsAuto(int kv_len, int head_dim, int max_allowed)
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

        const int min_splits = clampInt(FD_MIN_SPLITS, 1, max_allowed);
        auto_splits = clampInt(auto_splits, min_splits, max_allowed);
        return auto_splits;
    }

    inline int chooseEffectiveDecodeSplits(int kv_len, int head_dim, int requested_num_splits)
    {
        const int workspace_limit = (requested_num_splits > 0)
            ? requested_num_splits
            : FD_MAX_SPLITS_SUPPORTED;
        const int max_allowed = std::max(1, std::min(FD_MAX_SPLITS_SUPPORTED, workspace_limit));

        const int auto_splits = chooseDecodeSplitsAuto(kv_len, head_dim, max_allowed);
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

    inline uint32_t makeDecodeAutotuneKey(int kv_len, int head_dim, int max_allowed_splits)
    {
        const int64_t total_elements = static_cast<int64_t>(std::max(kv_len, 1)) * std::max(head_dim, 1);
        const int elem_bucket = clampInt(ilog2Floor(static_cast<int>(std::min<int64_t>(total_elements, INT32_MAX))), 0, 31);
        const int aspect_bucket = bucketAspectRatio(kv_len, head_dim); // 0..6
        const int hd_bucket = clampInt(ilog2Floor(std::max(head_dim, 1)), 0, 15);
        const int split_cap_bucket = clampInt(ilog2Floor(std::max(max_allowed_splits, 1)), 0, 15);

        return static_cast<uint32_t>(
            (elem_bucket & 0x1F) |
            ((aspect_bucket & 0x07) << 5) |
            ((hd_bucket & 0x0F) << 8) |
            ((split_cap_bucket & 0x0F) << 12));
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

    inline std::vector<DecodeLaunchConfig> buildDecodeAutotuneCandidates(int kv_len, int head_dim, int max_allowed)
    {
        const int min_splits = clampInt(FD_MIN_SPLITS, 1, max_allowed);

        const int base_splits = chooseDecodeSplitsAuto(kv_len, head_dim, max_allowed);

        std::vector<int> split_candidates;
        split_candidates.reserve(6);
        addSplitCandidate(split_candidates, min_splits, max_allowed);
        addSplitCandidate(split_candidates, base_splits / 2, max_allowed);
        addSplitCandidate(split_candidates, base_splits, max_allowed);
        addSplitCandidate(split_candidates, base_splits * 2, max_allowed);
        addSplitCandidate(split_candidates, 1, max_allowed);
        addSplitCandidate(split_candidates, max_allowed, max_allowed);

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

    inline DecodeAutotuneSelection chooseDecodeLaunchConfig(int kv_len, int head_dim, int requested_num_splits)
    {
        DecodeAutotuneSelection selection;

        const int heuristic_splits = chooseEffectiveDecodeSplits(kv_len, head_dim, requested_num_splits);
        const int heuristic_tpb = chooseDecodeThreadsPerBlock(head_dim, kv_len, heuristic_splits);
        selection.config = {heuristic_splits, heuristic_tpb};

        const int workspace_limit = (requested_num_splits > 0)
            ? requested_num_splits
            : FD_MAX_SPLITS_SUPPORTED;
        const int max_allowed = std::max(1, std::min(FD_MAX_SPLITS_SUPPORTED, workspace_limit));

        const uint32_t key = makeDecodeAutotuneKey(kv_len, head_dim, max_allowed);

        std::lock_guard<std::mutex> guard(decodeAutotuneMutex());
        auto &states = decodeAutotuneMap();
        auto &state = states[key];

        if (state.candidates.empty())
        {
            state.candidates = buildDecodeAutotuneCandidates(kv_len, head_dim, max_allowed);
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
        int tile_kv)
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
        const int gqa_ratio = n_heads / n_kv_heads;
        const int kv_head_idx = head_idx / gqa_ratio;

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

        // head_dim <= 128 and THREADS_PER_ROW=4 => max 32 dims per thread.
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
    // Flash Attention 2 - Cooperative Prefill Kernel for MI50
    // =========================================================================
    //
    // Wavefront-cooperative prefill kernel adapted from the decode kernel's
    // architecture (lanes own output dims, not score elements) to handle
    // multiple Q rows.
    //
    // Key differences from the tiled prefill kernel above:
    //   - No K/V tiles in LDS — K and V loaded per-position from HBM
    //   - No score matrix in LDS — scores computed inline and consumed
    //   - Only Q tile in LDS — ~10KB vs 58KB, dramatically reduced footprint
    //   - Each wavefront independently handles a subset of Q rows
    //   - 5x higher occupancy (20 waves/CU vs 4)
    //
    // CRITICAL: Template parameters ensure all per-Q-row arrays are
    // register-promoted (not spilled to scratch). Without compile-time
    // loop bounds, the compiler places O_lane/m_i/l_i in private memory
    // (432 bytes scratch → 2.3× slower than the tiled kernel).
    //
    // Grid: (n_heads, num_q_tiles, batch_size)
    // Block: (256,) threads = 4 wavefronts
    //
    // =========================================================================

    template <int Q_PER_WAVE, int DIMS_PER_LANE, KVTypeTag KV_TYPE = KVTypeTag::FP32>
    __global__ __launch_bounds__(256, 3)
    void flash_attention_2_cooperative_mi50_kernel(
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
        int tile_q)
    {
        // =================================================================
        // 1. Indexing and Setup
        // =================================================================
        const int batch_idx = blockIdx.z;
        const int head_idx = blockIdx.x;
        const int q_tile_idx = blockIdx.y;
        const int tid = threadIdx.x;
        constexpr int threads_per_block = 256;
        const int wavefront_id = tid / WAVEFRONT_SIZE;  // 0-3
        const int lane_id = tid % WAVEFRONT_SIZE;  // 0-63
        constexpr int num_wavefronts = threads_per_block / WAVEFRONT_SIZE;  // 4

        const int gqa_ratio = n_heads / n_kv_heads;
        const int kv_head_idx = head_idx / gqa_ratio;

        const int q_block_start = q_tile_idx * tile_q;
        if (q_block_start >= seq_len) return;
        const int q_tile_rows = min(tile_q, seq_len - q_block_start);

        // =================================================================
        // 2. LDS Layout (Q tile only)
        // =================================================================
        const int q_stride = head_dim + FA2_LDS_PAD_FP16;  // 130 for head_dim=128

        extern __shared__ char smem[];
        __half *Q_tile = reinterpret_cast<__half *>(smem);

        // =================================================================
        // 3. Q Row Assignment
        // =================================================================
        const int q_wave_start = wavefront_id * Q_PER_WAVE;
        const int my_q_count = min(Q_PER_WAVE, max(0, q_tile_rows - q_wave_start));

        // =================================================================
        // 4. Thread-Local Accumulators (register-promoted via template params)
        // =================================================================
        float O_lane[Q_PER_WAVE][DIMS_PER_LANE];
        float m_i[Q_PER_WAVE];
        float l_i[Q_PER_WAVE];

        #pragma unroll
        for (int qi = 0; qi < Q_PER_WAVE; qi++) {
            m_i[qi] = -FLT_MAX;
            l_i[qi] = 0.0f;
            #pragma unroll
            for (int di = 0; di < DIMS_PER_LANE; di++) {
                O_lane[qi][di] = 0.0f;
            }
        }

        // =================================================================
        // 5. Load Q into LDS (cooperative, all threads)
        // =================================================================
        const float *Q_batch = Q + batch_idx * seq_len * n_heads * head_dim;

        if ((head_dim & 1) == 0) {
            const int head_dim_vec2 = head_dim / 2;
            for (int i = tid; i < q_tile_rows * head_dim_vec2; i += threads_per_block) {
                int local_row = i / head_dim_vec2;
                int d2 = i % head_dim_vec2;
                int d = d2 * 2;
                int global_row = q_block_start + local_row;

                const float2 qv = *reinterpret_cast<const float2 *>(
                    Q_batch + global_row * n_heads * head_dim + head_idx * head_dim + d);
                reinterpret_cast<__half2 *>(Q_tile + local_row * q_stride + d)[0] = __float22half2_rn(qv);
            }
        } else {
            for (int i = tid; i < q_tile_rows * head_dim; i += threads_per_block) {
                int local_row = i / head_dim;
                int d = i % head_dim;
                int global_row = q_block_start + local_row;
                float val = Q_batch[global_row * n_heads * head_dim + head_idx * head_dim + d];
                Q_tile[local_row * q_stride + d] = float_to_half(val);
            }
        }
        __syncthreads();

        // =================================================================
        // 6. Runtime Parameters (device_params)
        // =================================================================
        int kv_stride_global = kv_len;
        int kv_len_runtime = kv_len;
        int position_offset_runtime = position_offset;
        int mask_stride = kv_len;

        if (device_params) {
            kv_len_runtime = device_params->kv_len;
            position_offset_runtime = device_params->position_offset;
            mask_stride = device_params->mask_stride;
        }

        // =================================================================
        // 7. Batch/Head Pointers for K/V (type-aware)
        // =================================================================
        // Compute byte-level base pointers so the batch+head offset is hoisted
        // out of the main KV loop. KV_TYPE determines element size and layout.
        const int kv_head_offset_elems = kv_head_idx * head_dim;
        const int kv_row_stride_elems = n_kv_heads * head_dim;

        // Q8_1 block layout (compiled out for non-Q8_1 via if constexpr)
        [[maybe_unused]] constexpr int Q8_BS = 32;
        [[maybe_unused]] const int q8_bph = head_dim / Q8_BS;    // blocks per head
        [[maybe_unused]] const int q8_bpr = n_kv_heads * q8_bph; // blocks per KV row

        // Byte-level base pointers: batch + head absorbed, only kv_pos varies in loop
        size_t row_byte_stride;
        size_t base_byte_off;
        if constexpr (KV_TYPE == KVTypeTag::Q8_1) {
            row_byte_stride = static_cast<size_t>(q8_bpr) * sizeof(llaminar2::Q8_1Block);
            base_byte_off = static_cast<size_t>(batch_idx) * kv_stride_global * row_byte_stride
                          + static_cast<size_t>(kv_head_idx) * q8_bph * sizeof(llaminar2::Q8_1Block);
        } else if constexpr (KV_TYPE == KVTypeTag::FP16) {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(__half);
            base_byte_off = static_cast<size_t>(batch_idx) * kv_stride_global * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(__half);
        } else {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(float);
            base_byte_off = static_cast<size_t>(batch_idx) * kv_stride_global * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(float);
        }
        const char *K_base = reinterpret_cast<const char*>(K) + base_byte_off;
        const char *V_base = reinterpret_cast<const char*>(V) + base_byte_off;

        // =================================================================
        // 8. Main Loop: Wavefront-Cooperative KV Processing
        // =================================================================
        // Each wavefront iterates over ALL KV positions. For each position,
        // all 64 lanes cooperate on dot products and V accumulation.
        // KV_TYPE determines load format: FP32/FP16 element loads or Q8_1 dequant.
        // All inner loops use #pragma unroll with compile-time template bounds
        // to ensure register promotion of O_lane/m_i/l_i arrays.

        for (int kv_pos = 0; kv_pos < kv_len_runtime; kv_pos++) {
            // Early termination for causal: if ALL Q rows in this wavefront
            // have kv_pos beyond their causal boundary, skip.
            if (causal) {
                int max_q_pos = q_block_start + q_wave_start + my_q_count - 1;
                if (kv_pos > max_q_pos + position_offset_runtime) break;
            }

            // Type-dispatched K/V load. Only the active KV_TYPE path survives
            // compilation (if constexpr eliminates dead branches → zero VGPR cost
            // for inactive paths).
            float k_vals[DIMS_PER_LANE] = {};   // FP32/Q8_1/FP16+DPL1
            float v_vals[DIMS_PER_LANE] = {};   // Always float for O accumulation
            [[maybe_unused]] __half2 k_h2;      // FP16+DPL>=2 fdot2 path only

            const char *K_pos = K_base + kv_pos * row_byte_stride;
            const char *V_pos = V_base + kv_pos * row_byte_stride;

            if constexpr (KV_TYPE == KVTypeTag::FP16) {
                // ---- FP16 KV: half bandwidth, fdot2 for dot product ----
                const __half *Kh = reinterpret_cast<const __half*>(K_pos);
                const __half *Vh = reinterpret_cast<const __half*>(V_pos);
                if constexpr (DIMS_PER_LANE >= 2) {
                    // Pack K into half2 for fdot2 (shared across all Q rows)
                    __half k0 = (lane_id < head_dim) ? Kh[lane_id] : __float2half(0.0f);
                    __half k1 = (lane_id + WAVEFRONT_SIZE < head_dim)
                                    ? Kh[lane_id + WAVEFRONT_SIZE] : __float2half(0.0f);
                    k_h2 = __halves2half2(k0, k1);
                } else {
                    // DPL=1: scalar convert, no fdot2
                    k_vals[0] = (lane_id < head_dim) ? __half2float(Kh[lane_id]) : 0.0f;
                }
                #pragma unroll
                for (int di = 0; di < DIMS_PER_LANE; di++) {
                    int d = lane_id + di * WAVEFRONT_SIZE;
                    v_vals[di] = (d < head_dim) ? __half2float(Vh[d]) : 0.0f;
                }
            } else if constexpr (KV_TYPE == KVTypeTag::Q8_1) {
                // ---- Q8_1 KV: inline dequant, ~3.5× BW savings ----
                const llaminar2::Q8_1Block *Kq = reinterpret_cast<const llaminar2::Q8_1Block*>(K_pos);
                const llaminar2::Q8_1Block *Vq = reinterpret_cast<const llaminar2::Q8_1Block*>(V_pos);
                #pragma unroll
                for (int di = 0; di < DIMS_PER_LANE; di++) {
                    int d = lane_id + di * WAVEFRONT_SIZE;
                    if (d < head_dim) {
                        int bi = d / Q8_BS;
                        int bo = d % Q8_BS;
                        float ks = __half2float(*reinterpret_cast<const __half*>(&Kq[bi].d));
                        k_vals[di] = static_cast<float>(Kq[bi].qs[bo]) * ks;
                        float vs = __half2float(*reinterpret_cast<const __half*>(&Vq[bi].d));
                        v_vals[di] = static_cast<float>(Vq[bi].qs[bo]) * vs;
                    }
                }
            } else {
                // ---- FP32 KV: original path ----
                const float *Kf = reinterpret_cast<const float*>(K_pos);
                const float *Vf = reinterpret_cast<const float*>(V_pos);
                #pragma unroll
                for (int di = 0; di < DIMS_PER_LANE; di++) {
                    int d = lane_id + di * WAVEFRONT_SIZE;
                    k_vals[di] = (d < head_dim) ? Kf[d] : 0.0f;
                    v_vals[di] = (d < head_dim) ? Vf[d] : 0.0f;
                }
            }

            // Process each Q row owned by this wavefront.
            // Fully unrolled: compiler maps O_lane[qi][di] → VGPRs.
            // Partial tiles: qi >= my_q_count → softmax stays at -FLT_MAX,
            // output write is masked, so extra work is harmless.
            #pragma unroll
            for (int qi = 0; qi < Q_PER_WAVE; qi++) {
                if (qi >= my_q_count) continue;

                int local_q = q_wave_start + qi;
                int global_q = q_block_start + local_q;

                // Cooperative dot product: type-dispatched
                // FP16+DPL>=2: single v_dot2_f32_f16 instruction (2 FP16 muls + FP32 acc)
                // All other paths: scalar FP32 multiply-add
                float partial_dot = 0.0f;
                if constexpr (KV_TYPE == KVTypeTag::FP16 && DIMS_PER_LANE >= 2) {
                    // fdot2: pack Q from LDS (already FP16), dot with pre-packed K half2
                    __half q0 = Q_tile[local_q * q_stride + lane_id];
                    __half q1 = (lane_id + WAVEFRONT_SIZE < head_dim)
                                    ? Q_tile[local_q * q_stride + lane_id + WAVEFRONT_SIZE]
                                    : __float2half(0.0f);
                    __half2 q_h2 = __halves2half2(q0, q1);
                    partial_dot = __builtin_amdgcn_fdot2(q_h2, k_h2, 0.0f, false);
                } else {
                    // Scalar path: FP32, Q8_1 (dequantized), or FP16+DPL=1
                    #pragma unroll
                    for (int di = 0; di < DIMS_PER_LANE; di++) {
                        int d = lane_id + di * WAVEFRONT_SIZE;
                        if (d < head_dim) {
                            float q_val = half_to_float(Q_tile[local_q * q_stride + d]);
                            partial_dot += q_val * k_vals[di];
                        }
                    }
                }

                // Wavefront-wide reduction (6 shuffles) -> score uniform across all lanes
                float score = wavefrontReduceSum(partial_dot) * softmax_scale;

                // Masking
                bool masked = false;
                if (causal && kv_pos > global_q + position_offset_runtime)
                    masked = true;
                if (window_size > 0) {
                    int q_pos = global_q + position_offset_runtime;
                    if (kv_pos < q_pos - window_size || kv_pos > q_pos + window_size)
                        masked = true;
                }
                // External mask
                float mask_addend = 0.0f;
                if (mask && !masked) {
                    float mask_val = mask[(batch_idx * seq_len + global_q) * mask_stride + kv_pos];
                    if (mask_val <= -1.0e20f) {
                        masked = true;
                    } else {
                        mask_addend = mask_val;
                    }
                }

                if (!masked) {
                    score += mask_addend;

                    // Online softmax update
                    float m_new = fmaxf(m_i[qi], score);
                    float scale_old = fast_expf(m_i[qi] - m_new);
                    float p = fast_expf(score - m_new);

                    // Update O accumulators and softmax state
                    l_i[qi] = l_i[qi] * scale_old + p;
                    #pragma unroll
                    for (int di = 0; di < DIMS_PER_LANE; di++) {
                        O_lane[qi][di] = O_lane[qi][di] * scale_old + p * v_vals[di];
                    }
                    m_i[qi] = m_new;
                }
            }
        }

        // =================================================================
        // 9. Write Final Output
        // =================================================================
        // Each wavefront writes its Q rows independently. Each lane writes
        // its owned output dims. No inter-wavefront coordination needed.

        float *O_batch = O + batch_idx * seq_len * n_heads * head_dim;

        #pragma unroll
        for (int qi = 0; qi < Q_PER_WAVE; qi++) {
            if (qi >= my_q_count) continue;

            int local_q = q_wave_start + qi;
            int global_q_row = q_block_start + local_q;
            float inv_l = (l_i[qi] > 0.0f) ? (1.0f / l_i[qi]) : 0.0f;

            float *O_row = O_batch + global_q_row * n_heads * head_dim + head_idx * head_dim;
            #pragma unroll
            for (int di = 0; di < DIMS_PER_LANE; di++) {
                int d = lane_id + di * WAVEFRONT_SIZE;
                if (d < head_dim) {
                    O_row[d] = O_lane[qi][di] * inv_l;
                }
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
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params)
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

        const int gqa_ratio = n_heads / n_kv_heads;
        const int kv_head_idx = head_idx / gqa_ratio;

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
        // For head_dim=128, WAVEFRONT_SIZE=64: each lane owns 2 dims
        // Lane i owns dims: i, i+64 (i.e., strided by WAVEFRONT_SIZE)
        //
        // This reduces VGPR usage from 128 to 2-4 floats per lane.
        // =====================================================================
        float O_lane[4] = {0, 0, 0, 0}; // Supports head_dim up to 256
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
            // Each lane updates only its own output dims (2 for head_dim=128)
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
        //   - O distributed across lanes (each lane owns 2 dims)
        //
        // We merge 4 wavefronts' results, then write the final partial output.
        // =====================================================================
        __shared__ float block_m[4];
        __shared__ float block_l[4];
        __shared__ float block_O[4 * 128]; // 4 wavefronts × max head_dim

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

    // =========================================================================
    // Cooperative Prefill Launch Dispatcher (KV-type templated)
    // =========================================================================
    //
    // Factored out of extern "C" to allow template instantiation per KV_TYPE.
    // Each instantiation generates {4,8,16} × {1,2} kernel variants (6 total).
    // With 3 KV types → 18 kernel instantiations. Binary size ~3× but no
    // runtime cost (only one variant executes per call).
    //
    template <KVTypeTag KV_TYPE>
    int launch_cooperative_prefill(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        float softmax_scale, bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        hipStream_t hip_stream)
    {
        int tile_q = 32;  // Optimal for head_dim=128

        // Allow override via environment
        const char *tq_env = getenv("LLAMINAR_FA2_COOPERATIVE_TILE_Q");
        if (tq_env) {
            int tq_override = atoi(tq_env);
            if (tq_override == 16 || tq_override == 32 || tq_override == 64)
                tile_q = tq_override;
        }

        const int q_stride = head_dim + FA2_LDS_PAD_FP16;
        const int lds_size = tile_q * q_stride * static_cast<int>(sizeof(__half));

        const int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
        dim3 grid(n_heads, num_q_tiles, batch_size);
        dim3 block(FA2_THREADS_PER_BLOCK);

        constexpr int WAVES_PER_BLOCK = 4;
        const int q_per_wave = tile_q / WAVES_PER_BLOCK;
        const int dims_per_lane = (head_dim + WAVEFRONT_SIZE - 1) / WAVEFRONT_SIZE;

        // Template dispatch — covers all valid (q_per_wave, dims_per_lane) combos
        #define LAUNCH_COOP_KV(QPW, DPL) \
            hipLaunchKernelGGL( \
                (flash_attention_2_cooperative_mi50_kernel<QPW, DPL, KV_TYPE>), \
                grid, block, lds_size, hip_stream, \
                Q, K, V, O, \
                batch_size, seq_len, kv_len, \
                n_heads, n_kv_heads, head_dim, \
                softmax_scale, causal, window_size, position_offset, \
                device_params, mask, \
                tile_q)

        bool dispatched = false;
        if (dims_per_lane == 2) {
            // head_dim 65-128 (most common: 128)
            if (q_per_wave == 8) { LAUNCH_COOP_KV(8, 2); dispatched = true; }
            else if (q_per_wave == 4) { LAUNCH_COOP_KV(4, 2); dispatched = true; }
            else if (q_per_wave == 16) { LAUNCH_COOP_KV(16, 2); dispatched = true; }
        } else if (dims_per_lane == 1) {
            // head_dim 1-64
            if (q_per_wave == 8) { LAUNCH_COOP_KV(8, 1); dispatched = true; }
            else if (q_per_wave == 4) { LAUNCH_COOP_KV(4, 1); dispatched = true; }
            else if (q_per_wave == 16) { LAUNCH_COOP_KV(16, 1); dispatched = true; }
        }
        #undef LAUNCH_COOP_KV

        if (!dispatched) {
            printf("[hipFlashAttn_prefill] Unsupported config: q_per_wave=%d, dims_per_lane=%d\n",
                   q_per_wave, dims_per_lane);
            return -1;
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            printf("[hipFlashAttn_prefill] Cooperative kernel launch failed: %s\n", hipGetErrorString(err));
            return -1;
        }
        return 0;
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
    hipStream_t hip_stream)
{
    if (head_dim <= 0 || head_dim > 128)
    {
        printf("[launch_decode] Error: head_dim=%d must be in [1, 128]\n", head_dim);
        return -1;
    }

    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    const DecodeAutotuneSelection tuned = chooseDecodeLaunchConfig(kv_len, head_dim, num_splits);
    const int effective_num_splits = tuned.config.splits;
    const int decode_threads_per_block = tuned.config.tpb;

    hipEvent_t tune_start = nullptr;
    hipEvent_t tune_stop = nullptr;
    const bool measure_tuning = tuned.should_time;
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
        effective_num_splits, softmax_scale, device_params);

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
        void *stream)
    {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        // Input validation
        if (head_dim <= 0 || head_dim > 128)
        {
            printf("[hipFlashAttn_prefill_fa2] Error: head_dim=%d must be in [1, 128]\n", head_dim);
            return -1;
        }
        if (n_heads % n_kv_heads != 0)
        {
            printf("[hipFlashAttn_prefill_fa2] Error: n_heads must be divisible by n_kv_heads\n");
            return -1;
        }

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        if (head_dim >= 64) {
            return launch_cooperative_prefill<KVTypeTag::FP32>(
                Q, K, V, O,
                batch_size, seq_len, kv_len,
                n_heads, n_kv_heads, head_dim,
                softmax_scale, causal, window_size, position_offset,
                device_params, mask, hip_stream);
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
            tile_q, tile_kv);

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
        void *stream)
    {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        if (head_dim <= 0 || head_dim > 128)
        {
            printf("[hipFlashAttn_prefill_fa2_fp16] Error: head_dim=%d must be in [1, 128]\n", head_dim);
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
        return launch_cooperative_prefill<KVTypeTag::FP16>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask, hip_stream);
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
        void *stream)
    {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        if (head_dim <= 0 || head_dim > 128)
        {
            printf("[hipFlashAttn_prefill_fa2_q8_1] Error: head_dim=%d must be in [1, 128]\n", head_dim);
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
        return launch_cooperative_prefill<KVTypeTag::Q8_1>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask, hip_stream);
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
        void *stream)
    {
        return launch_decode<KVTypeTag::FP32>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream));
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
        void *stream)
    {
        return launch_decode<KVTypeTag::FP16>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream));
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
        void *stream)
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
            num_splits, device_params, static_cast<hipStream_t>(stream));
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
