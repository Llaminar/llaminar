/**
 * @file ROCmRetainedQSA.h
 * @brief Resident ROCm cooperative QSA Pre-Attention and Post-Attention megakernel ABI.
 *
 * Mirrors CUDARetainedQSA.h for full backend symmetry across all ROCm topologies
 * (4xROCm TP4, 2xROCm+2xCPU TP2, etc.) and all 18 NativeQuantised codebooks plus
 * BF16/FP16/FP32 indexer weights with zero VGPR/SGPR spills.
 */
#pragma once

#include "kernels/attention/AttentionDeviceParams.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace llaminar2::rocm::retained_qsa
{
    inline constexpr unsigned kMaxROCmRetainedQSAQHeads = 24u;
    inline constexpr unsigned kMaxROCmRetainedQSAKVHeads = 2u;

    /** @brief Prepared NativeQuantised projection and its canonical ROCm M1 K-partition count. */
    struct ROCmRetainedQSAProjection
    {
        DeviceNativeQuantisedMatrixDesc weights; ///< Prepared native weight descriptor (any of 18 codebooks).
        float *output = nullptr;                 ///< Public FP32 destination [1, N] (unused for split Wq).
        int k_partitions = 16;                   ///< Exact ROCm serial M1 K-partition count (1..32).
    };

    /** @brief Weight encoding of an indexer key/query projection absorbed into Pre-Attention. */
    enum class ROCmRetainedQSAIndexerFormat : std::uint8_t
    {
        Absent = 0,
        NativeQuantised,
        FP32,
        FP16,
        BF16
    };

    /** @brief Optional indexer projection (qsa_key [128, 2560] or qsa_query [512, 2560]). */
    struct ROCmRetainedQSAIndexerProj
    {
        ROCmRetainedQSAIndexerFormat format = ROCmRetainedQSAIndexerFormat::Absent;
        ROCmRetainedQSAProjection quantised{};
        const void *floating_weights = nullptr;
        float *floating_output = nullptr;
        int n = 0;
        bool ranking_only_guard = false;
    };

    /** @brief Complete Pre-Attention invocation sealed before HIP graph capture. */
    struct ROCmRetainedQSAPreInvocation
    {
        FullNativeHCFFNPrefixArgs prefix;
        ROCmRetainedQSAProjection wq;
        ROCmRetainedQSAProjection wk;
        ROCmRetainedQSAProjection wv;
        ROCmRetainedQSAIndexerProj qsa_key;
        ROCmRetainedQSAIndexerProj qsa_query;

        float *Q = nullptr;
        float *fa_gate = nullptr;
        const float *q_norm_gamma = nullptr;
        const float *k_norm_gamma = nullptr;
        float qk_norm_eps = 1e-6f;

        int local_q_heads = 0;
        int local_kv_heads = 0;
        int rotary_dim = 64;
        float rope_neg_log_theta = 0.0f;
        bool skip_k_rope = false;

        const std::int32_t *position_ids_device = nullptr;
        int *d_ring_head = nullptr;
        int *d_cached_tokens = nullptr;
        int kv_token_capacity = 0;

        void *kv_keys_data = nullptr;
        void *kv_values_data = nullptr;
        std::size_t kv_token_stride_bytes = 0;
        std::size_t kv_head_stride_bytes = 0;

        llaminar2::qwen4exp::QwenSparseReplay *replay_out = nullptr;
        llaminar2::attention::AttentionDeviceParams *row_params_out = nullptr;

        void *scratch = nullptr;
        std::size_t scratch_bytes = 0;
        bool has_previous_write = true;
    };

    /** @brief Complete Post-Attention invocation sealed before HIP graph capture. */
    struct ROCmRetainedQSAPostInvocation
    {
        llaminar2::qwen4exp::QwenSparseAttention attention;
        llaminar2::qwen4exp::QwenSparsePartialCommit commit;
        bool has_commit = false;
        const float *fa_gate = nullptr;
        ROCmRetainedQSAProjection wo;
        void *scratch = nullptr;
        std::size_t scratch_bytes = 0;
    };

    /** @brief Admitted ROCm QSA Pre-Attention megakernel ready for HIP graph capture. */
    struct ROCmRetainedQSAPreAdmission
    {
        ROCmRetainedQSAPreInvocation invocation;
        const void *function = nullptr;
        int device = -1;
        unsigned grid = 0;
        unsigned dynamic_shared_bytes = 0;
        NativeEmbeddingIQTables iq_tables{};
    };

    /** @brief Admitted ROCm QSA Post-Attention megakernel ready for HIP graph capture. */
    struct ROCmRetainedQSAPostAdmission
    {
        ROCmRetainedQSAPostInvocation invocation;
        const void *function = nullptr;
        int device = -1;
        unsigned grid = 0;
        unsigned dynamic_shared_bytes = 0;
        NativeEmbeddingIQTables iq_tables{};
    };

    /** @return Exact scratch byte size required in workspace bank for ROCm QSA megakernels. */
    [[nodiscard]] std::size_t retainedROCmQSAScratchBytes() noexcept;

    /** @brief Validate and admit the spill-free ROCm QSA Pre-Attention kernel. */
    hipError_t prepareROCmRetainedQSAPreAttention(
        const ROCmRetainedQSAPreInvocation &invocation,
        int device,
        ROCmRetainedQSAPreAdmission &admission);

    /** @brief Enqueue the admitted ROCm QSA Pre-Attention megakernel on an exact non-null stream. */
    hipError_t launchROCmRetainedQSAPreAttention(
        const ROCmRetainedQSAPreAdmission &admission,
        hipStream_t stream) noexcept;

    /** @brief Validate and admit the spill-free ROCm QSA Post-Attention kernel. */
    hipError_t prepareROCmRetainedQSAPostAttention(
        const ROCmRetainedQSAPostInvocation &invocation,
        int device,
        ROCmRetainedQSAPostAdmission &admission);

    /** @brief Enqueue the admitted ROCm QSA Post-Attention megakernel on an exact non-null stream. */
    hipError_t launchROCmRetainedQSAPostAttention(
        const ROCmRetainedQSAPostAdmission &admission,
        hipStream_t stream) noexcept;
} // namespace llaminar2::rocm::retained_qsa
