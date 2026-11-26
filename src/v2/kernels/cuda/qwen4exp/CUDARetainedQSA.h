/**
 * @file CUDARetainedQSA.h
 * @brief Resident CUDA cooperative QSA Pre-Attention and Post-Attention megakernel ABI.
 *
 * Replaces all 35 participant-local CUDA QSA attention kernels per layer with 4:
 *   1. CUDARetainedQSAPreAttention (1 cooperative kernel):
 *      - Replay & row_parameters publication (from pre-append ring metadata)
 *      - HC ResidualWrite / PublishedTrunk + BranchRMSNorm + HC Down (+ 4-CTA
 *        overlapped hc_injection) + HC Up + Read -> mixed
 *      - Blockwise Q8 staging + coalesced 256B/warp Wq (direct Q/fa_gate split),
 *        Wk, Wv, and indexer qsa_key (+ optional ranking-guarded qsa_query)
 *        supporting ALL 18 NativeQuantised codebooks plus BF16/FP16/FP32 indexer weights
 *      - Per-head Q/K RMSNorm + 64-dim partial RoPE + direct FP16 KV ring append
 *        + canonical device ring_head / cached_tokens advance
 *   2. Standalone QSA Indexer (2 tiny kernels in short ctx, bounded selector in long ctx)
 *   3. Standalone selectedAttentionFP16Tiles (FP16 tiled QK/PV -> gpu_partials)
 *   4. CUDARetainedQSAPostAttention (1 cooperative kernel):
 *      - Exact foldAttentionPrefix (Parts=33 or 8) + sigmoid fa_gate + attn_output
 *        write + inline blockwise Q8 staging (overlapped with commitPartial)
 *      - Coalesced 256B/warp 16-column Wo projection -> attn_proj for wo_allreduce
 *
 * General across arbitrary CUDA GPU counts, TP widths, head counts, MoE placements,
 * and arbitrary GGUF quantization mixes with zero register spills and exact
 * serial-decode bit equivalence.
 */
#pragma once

#include "kernels/attention/AttentionDeviceParams.h"
#include "kernels/common/NativeCooperativeKernel.h"
#include "kernels/cuda/moe/CUDARetainedHCPrefix.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "tensors/TensorKernels.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2::cuda::qwen4exp
{
    /** @brief Maximum participant-local QSA query heads supported in static SMEM/scratch. */
    inline constexpr unsigned kMaxCUDARetainedQSAQHeads = 24u;
    /** @brief Maximum participant-local QSA KV heads supported in static SMEM/scratch. */
    inline constexpr unsigned kMaxCUDARetainedQSAKVHeads = 2u;

    /** @brief Prepared NativeQuantised projection and its canonical M1 K-partition schedule. */
    struct CUDARetainedQSAProjection
    {
        DeviceNativeQuantisedMatrixDesc weights; ///< Prepared native weight descriptor (any of 18 codebooks).
        float *output = nullptr;                 ///< Public FP32 destination [1, N] (unused for split Wq).
        int k_partitions = 1;                    ///< Exact serial M1 K-partition count.
        int blocks_per_partition = 1;            ///< Ceiling(blocks_per_row / k_partitions).
        bool ordered_kpar = false;               ///< True when canonical M1 uses ordered KPAR fold.
    };

    /** @brief Weight encoding of an indexer key/query projection absorbed into Pre-Attention. */
    enum class CUDARetainedQSAIndexerFormat : std::uint8_t
    {
        Absent = 0,
        NativeQuantised,
        FP32,
        FP16,
        BF16
    };

    /** @brief Optional indexer projection (qsa_key [128, 2560] or qsa_query [512, 2560]). */
    struct CUDARetainedQSAIndexerProj
    {
        CUDARetainedQSAIndexerFormat format = CUDARetainedQSAIndexerFormat::Absent;
        CUDARetainedQSAProjection quantised{};
        const void *floating_weights = nullptr; ///< Contiguous row-major [n, 2560] when FP32/FP16/BF16.
        float *floating_output = nullptr;       ///< Public FP32 output [1, n].
        int n = 0;                              ///< Output columns (128 or 512).
        bool ranking_only_guard = false;        ///< Skip projection when prefix_tokens < QwenSparseFirstRankingLogicalRow.
    };

    /** @brief Complete by-value Pre-Attention invocation sealed before CUDA graph capture. */
    struct CUDARetainedQSAPreInvocation
    {
        moe::RetainedHCPrefix hc;            ///< Original POST/HC prefix operands and fold.
        CUDARetainedQSAProjection wq;        ///< Wq [local_q_heads * 512, 2560], split directly into Q & fa_gate.
        CUDARetainedQSAProjection wk;        ///< Wk [local_kv_heads * 256, 2560] -> K.
        CUDARetainedQSAProjection wv;        ///< Wv [local_kv_heads * 256, 2560] -> V.
        CUDARetainedQSAIndexerProj qsa_key;  ///< Indexer raw key projection [128, 2560].
        CUDARetainedQSAIndexerProj qsa_query;///< Optional indexer query projection [512, 2560].

        float *Q = nullptr;                  ///< Public post-norm/RoPE query output [local_q_heads * 256].
        float *fa_gate = nullptr;            ///< Public query output gate [local_q_heads * 256].
        const float *q_norm_gamma = nullptr; ///< Optional Q per-head RMSNorm weight [256].
        const float *k_norm_gamma = nullptr; ///< Optional K per-head RMSNorm weight [256].
        float qk_norm_eps = 1e-6f;           ///< Q/K RMSNorm epsilon.

        int local_q_heads = 0;               ///< Participant-local Q heads in [1, 24].
        int local_kv_heads = 0;              ///< Participant-local KV heads in [1, 2].
        int rotary_dim = 64;                 ///< Partial split-half rotary width (even, <= 256).
        float rope_neg_log_theta = 0.0f;     ///< Setup-derived -std::log(rope_theta).
        bool skip_k_rope = false;            ///< True under RoPE-on-read (pre-RoPE KV storage).

        const std::int32_t *position_ids_device = nullptr; ///< Device-resident absolute position [1].
        int *d_ring_head = nullptr;                        ///< Canonical mutable KV ring cursor.
        int *d_cached_tokens = nullptr;                    ///< Canonical mutable KV cached count.
        int kv_token_capacity = 0;                         ///< Admitted ring max_seq_len.

        void *kv_keys_data = nullptr;                      ///< Native FP16 ring key bank.
        void *kv_values_data = nullptr;                    ///< Native FP16 ring value bank.
        std::size_t kv_token_stride_bytes = 0;             ///< Native byte stride between token rows.
        std::size_t kv_head_stride_bytes = 0;              ///< Native byte stride between KV heads.

        llaminar2::qwen4exp::QwenSparseReplay *replay_out = nullptr; ///< Pre-append replay snapshot.
        llaminar2::attention::AttentionDeviceParams *row_params_out = nullptr; ///< Canonical row[0] params.

        void *scratch = nullptr;             ///< Workspace scratch (HC down completion counters).
        std::size_t scratch_bytes = 0;       ///< Exact validated byte size of scratch.
    };
    static_assert(std::is_trivially_copyable_v<CUDARetainedQSAPreInvocation>);

    /** @brief Complete by-value Post-Attention invocation sealed before CUDA graph capture. */
    struct CUDARetainedQSAPostInvocation
    {
        llaminar2::qwen4exp::QwenSparseAttention attention; ///< Sealed FP16 tiled attention params.
        llaminar2::qwen4exp::QwenSparsePartialCommit commit;///< Optional raw-tail partial commit.
        bool has_commit = false;                            ///< True when commitPartial is fused.
        const float *fa_gate = nullptr;                     ///< Sigmoid output gate [local_q_heads * 256].
        CUDARetainedQSAProjection wo;                       ///< Output projection [2560, local_q_heads * 256] -> attn_proj.
        void *scratch = nullptr;                            ///< Workspace scratch for Q8 attn_output.
        std::size_t scratch_bytes = 0;                      ///< Exact validated byte size of scratch.
    };
    static_assert(std::is_trivially_copyable_v<CUDARetainedQSAPostInvocation>);

    /** @return Exact scratch byte size required in workspace bank for CUDARetainedQSAPre/Post. */
    [[nodiscard]] std::size_t retainedCUDAQSAScratchBytes() noexcept;

    /** @brief Admitted cooperative CUDA QSA Pre-Attention megakernel ready for graph capture. */
    struct CUDARetainedQSAPreAdmission
    {
        CUDARetainedQSAPreInvocation invocation;
        NativeCooperativeKernelAdmission native;
        unsigned grid = 0;
    };

    /** @brief Admitted cooperative CUDA QSA Post-Attention megakernel ready for graph capture. */
    struct CUDARetainedQSAPostAdmission
    {
        CUDARetainedQSAPostInvocation invocation;
        NativeCooperativeKernelAdmission native;
        unsigned grid = 0;
    };

    /** @brief Validate operands and admit the spill-free cooperative CUDA QSA Pre-Attention kernel. */
    [[nodiscard]] CUDARetainedQSAPreAdmission prepareCUDARetainedQSAPreAttention(
        DeviceId device,
        const CUDARetainedQSAPreInvocation &invocation,
        std::uint64_t workspace_generation);

    /** @brief Enqueue the admitted CUDA QSA Pre-Attention megakernel on an exact non-null stream. */
    void launchCUDARetainedQSAPreAttention(
        const CUDARetainedQSAPreAdmission &admission,
        void *stream,
        std::uint64_t workspace_generation);

    /** @brief Validate operands and admit the spill-free cooperative CUDA QSA Post-Attention kernel. */
    [[nodiscard]] CUDARetainedQSAPostAdmission prepareCUDARetainedQSAPostAttention(
        DeviceId device,
        const CUDARetainedQSAPostInvocation &invocation,
        std::uint64_t workspace_generation);

    /** @brief Enqueue the admitted CUDA QSA Post-Attention megakernel on an exact non-null stream. */
    void launchCUDARetainedQSAPostAttention(
        const CUDARetainedQSAPostAdmission &admission,
        void *stream,
        std::uint64_t workspace_generation);
} // namespace llaminar2::cuda::qwen4exp
