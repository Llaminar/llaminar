/**
 * @file CUDARetainedGDN.h
 * @brief Resident CUDA cooperative Full GDN Attention megakernel ABI.
 *
 * Fuses the entire participant-local QFN GDN attention block into one single
 * spill-free cooperative CUDA kernel bounded by the layer's TP collectives:
 *   1. HC ResidualWrite (or PublishedTrunk on Layer 0) + BranchRMSNorm +
 *      HC Down (+ overlapped 4-CTA FP32 hc_injection) + HC Up + Read -> mixed
 *   2. Blockwise Q8 & FP16RNE staging + coalesced 512B/warp QKV (+ inline
 *      ShortConv1d + SiLU) & Z projections + tail-shadow FP16 Alpha/Beta
 *   3. Exact row-split GDN delta-rule recurrence (KeyMajor or ValueMajor) +
 *      last-chunk inline GatedRMSNorm + blockwise Q8 staging
 *   4. Coalesced 512B/warp 16-column ssm_out projection -> attn_proj
 *
 * Works across arbitrary CUDA GPU counts, TP widths, head counts, and MoE
 * placement policies with zero hot-path allocation, synchronization, or host
 * readback. Every arithmetic tree retains exact serial-decode bit identity.
 */
#pragma once

#include "kernels/common/GatedNormActivation.h"
#include "kernels/common/NativeCooperativeKernel.h"
#include "kernels/cuda/moe/CUDARetainedHCPrefix.h"
#include "tensors/TensorKernels.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2::cuda::qwen4exp
{
    /** @brief Maximum participant-local GDN value heads supported in static SMEM/scratch. */
    inline constexpr unsigned kMaxCUDARetainedGDNVHeads = 48u;

    /** @brief Prepared Q8_0 projection and its exact canonical M1 K-partition schedule. */
    struct CUDARetainedGDNProjection
    {
        DeviceNativeQuantisedMatrixDesc weights; ///< Prepared native Q8_0 weight descriptor.
        float *output = nullptr;                 ///< Original public FP32 output row [1, N].
        int k_partitions = 1;                    ///< Exact serial M1 K-partition count.
        int blocks_per_partition = 1;            ///< Ceiling(blocks_per_row / k_partitions).
        bool ordered_kpar = false;               ///< True when canonical M1 uses ordered KPAR fold.
    };

    /** @brief Prepared contiguous FP16 alpha/beta control weights and FP32 outputs. */
    struct CUDARetainedGDNHalfControls
    {
        const std::uint16_t *alpha_weights = nullptr; ///< Contiguous FP16 [heads, 2560].
        const std::uint16_t *beta_weights = nullptr;  ///< Contiguous FP16 [heads, 2560].
        float *alpha_output = nullptr;                ///< Original FP32 [1, heads].
        float *beta_output = nullptr;                 ///< Original FP32 [1, heads].
        int heads = 0;                                ///< Participant-local value head count.
    };

    /** @brief Complete by-value invocation sealed before CUDA graph capture. */
    struct CUDARetainedGDNInvocation
    {
        moe::RetainedHCPrefix hc;         ///< Original POST/HC prefix operands and fold.
        CUDARetainedGDNProjection qkv;    ///< QKV projection [n_qkv, 2560].
        CUDARetainedGDNProjection z;      ///< Output-gate Z projection [n_z, 2560].
        CUDARetainedGDNHalfControls half; ///< FP16RNE alpha/beta control projections.

        const float *conv_weight = nullptr;      ///< ShortConv1d depthwise weight [n_qkv, 4].
        const float *conv_bias = nullptr;        ///< Optional ShortConv1d bias [n_qkv].
        float *conv_state = nullptr;             ///< Persistent live conv history [n_qkv, 3].
        float *gdn_recurrence_in = nullptr;      ///< Original public post-conv output [1, n_qkv].

        const float *A_log = nullptr;            ///< Recurrence log-decay parameter [n_v_heads].
        const float *dt_bias = nullptr;          ///< Recurrence softplus bias [n_v_heads].
        float *recurrence_state = nullptr;       ///< Persistent live recurrence state [n_v_heads, 128, 128].
        float *attn_output = nullptr;            ///< Original public recurrence/norm output [1, n_z].

        const float *ssm_norm_gamma = nullptr;   ///< Gated RMSNorm gamma [128].
        float ssm_norm_eps = 1e-6f;              ///< Gated RMSNorm epsilon.
        bool ssm_norm_subtract_one = false;      ///< True when gamma stores (weight - 1).
        GatedNormActivation ssm_norm_activation = GatedNormActivation::SiLU;

        int n_k_heads = 0;                       ///< Participant-local key/query head count.
        int n_v_heads = 0;                       ///< Participant-local value head count.
        int global_v_offset = 0;                 ///< Global value-head offset for key-head modulo.
        bool use_qk_l2norm = true;               ///< True when Q/K L2 normalization is enabled.
        GDNRecurrenceStateLayout state_layout = GDNRecurrenceStateLayout::KeyMajorValueContiguous;

        CUDARetainedGDNProjection ssm_out;       ///< Output projection [2560, n_z] -> attn_proj.

        void *scratch = nullptr;                 ///< Workspace scratch for Q8 out + head_done.
        std::size_t scratch_bytes = 0;           ///< Exact validated byte size of scratch.
    };

    static_assert(std::is_trivially_copyable_v<CUDARetainedGDNInvocation>);

    /** @return Exact scratch byte size required in workspace bank for CUDARetainedGDN. */
    [[nodiscard]] std::size_t retainedCUDAGDNScratchBytes() noexcept;

    /** @brief Admitted cooperative CUDA Full GDN megakernel ready for graph capture. */
    struct CUDARetainedGDNAdmission
    {
        CUDARetainedGDNInvocation invocation;
        NativeCooperativeKernelAdmission native;
        unsigned grid = 0;
    };

    /**
     * @brief Validate immutable operands and admit the cooperative CUDA Full GDN kernel.
     * @param device Owning CUDA participant.
     * @param invocation Sealed operands and workspace pointers.
     * @param workspace_generation Current workspace generation for capture identity.
     * @return Complete cooperative admission ready for exact-stream launch/capture.
     * @throws std::invalid_argument On invalid geometry, unaligned pointers, or register spills.
     */
    [[nodiscard]] CUDARetainedGDNAdmission prepareCUDARetainedGDN(
        DeviceId device,
        const CUDARetainedGDNInvocation &invocation,
        std::uint64_t workspace_generation);

    /**
     * @brief Enqueue the admitted cooperative CUDA Full GDN megakernel on an exact non-null stream.
     * @param admission Sealed admission produced by prepareCUDARetainedGDN().
     * @param stream Exact non-null CUDA stream (capture or direct).
     * @param workspace_generation Current workspace generation; mismatch throws fatally.
     */
    void launchCUDARetainedGDN(
        const CUDARetainedGDNAdmission &admission,
        void *stream,
        std::uint64_t workspace_generation);
} // namespace llaminar2::cuda::qwen4exp
