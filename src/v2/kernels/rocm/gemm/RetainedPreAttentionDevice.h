/**
 * @file RetainedPreAttentionDevice.h
 * @brief QFN gfx906 pre-attention POST/HC and prepared Q8 projection ABI.
 *
 * The caller retains every original buffer and pre-resolved numerical source.
 * The alternate operation borrows them on a single exact native stream.
 */
#pragma once
#include "kernels/common/GatedNormActivation.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"
#include <hip/hip_runtime_api.h>

namespace llaminar2::rocm::pre_attention_probe {
/** @brief Explicit observed source role program for the local early Q8 cohort. */
enum class EarlyQ8Geometry : std::uint32_t { GDN };
/** @brief A previously published trunk or a POST write owned by this retained operation. */
enum class TrunkSource : std::uint32_t { PreviousWrite, PublishedTrunk };
/** @brief Retain the original floating subgroup unless the prepared half leaf is selected explicitly. */
enum class HalfGDNPolicy : std::uint32_t { SeparateOriginalSubgroup, RetainedPreparedFP16RNE };

/** @brief Borrow original, fully prepared row-major FP16 weights and their raw FP32 publications. */
struct PreparedHalfMatrix
{
    const std::uint16_t* words = nullptr;
    std::size_t bytes = 0;
};

/** @brief Exact native FP16 GDN alpha/beta projections and their raw FP32 destinations. */
struct PreparedHalfGDN
{
    PreparedHalfMatrix alpha;
    PreparedHalfMatrix beta;
    float* raw_alpha = nullptr;
    float* raw_beta = nullptr;
    int heads = 0;
};

/** @brief Immutable borrowed prepared matrix and original public FP32 destination. */
struct Projection {
    DeviceNativeQuantisedMatrixDesc weights;
    float* output;
};

/**
 * @brief Complete single-row GDN megakernel suffix operands (conv1d + recurrence + gated norm + ssm_out).
 *
 * Every pointer borrows an immutable cache, weight, or arena owner bound outside
 * capture. When enabled, a single resident megakernel executes the entire local
 * GDN attention block from previous FFN HC write directly to `attn_proj`.
 */
struct FullGDNChain {
    bool enabled = false;
    const float* conv_weight = nullptr; ///< [qkv_dim, 4] FP32 depthwise causal conv weights.
    const float* conv_bias = nullptr;   ///< Optional [qkv_dim] FP32 conv bias.
    float* conv_state = nullptr;        ///< Cache-owned [qkv_dim, 3] FP32 conv state (updated in place).
    float* gdn_recurrence_in = nullptr; ///< Arena [qkv_dim] FP32 post-conv SiLU output.
    const float* A_log = nullptr;       ///< [n_v_heads] FP32 log-decay weights.
    const float* dt_bias = nullptr;     ///< [n_v_heads] FP32 gate bias weights.
    float* recurrence_state = nullptr;  ///< Cache-owned [n_v_heads, 128, 128] ValueMajor state (updated in place).
    float* attn_output = nullptr;       ///< Arena [n_v_heads * 128] FP32 recurrence + gated-norm output.
    const float* ssm_norm_gamma = nullptr; ///< [128] FP32 per-head RMSNorm gain.
    float ssm_norm_eps = 1e-6f;
    bool ssm_norm_subtract_one = false;
    GatedNormActivation ssm_norm_activation = GatedNormActivation::Sigmoid;
    int n_k_heads = 0;
    int n_v_heads = 0;
    int global_v_offset = 0;
    bool use_qk_l2norm = true;
    Projection ssm_out{};               ///< Prepared Q8_0 [d_model, n_v_heads*128] output projection -> attn_proj.
};

/** @brief POST input, exact original HC prepared layout, and typed early projection tuple. */
struct Invocation {
    FullNativeHCFFNPrefixArgs prefix;
    Projection projections[3];
    EarlyQ8Geometry geometry;
    void* scratch;
    std::size_t scratch_bytes;
    TrunkSource trunk_source = TrunkSource::PreviousWrite;
    HalfGDNPolicy half_policy = HalfGDNPolicy::SeparateOriginalSubgroup;
    PreparedHalfGDN half{};
    FullGDNChain full_gdn{};
};

/** @brief Original captured operation's lowered instance identity and observed physical resources. */
struct Admission {
    Invocation invocation;
    const void* function;
    hipFuncAttributes resources;
    int device;
    unsigned grid = 0;
    unsigned dynamic_shared_bytes = 0;
    bool general_codebooks = false;
    NativeEmbeddingIQTables iq_tables{};
};

/** @return The exact compact original HC state and weighted join byte extent. */
std::size_t scratchBytes() noexcept;
/** @brief Resolve the physical gfx906 kernel and validate only original declared projections. */
hipError_t prepare(const Invocation&,int device,Admission&);
/** @brief Submit this cold-bound invocation on its exact nondefault stream. */
hipError_t launch(const Admission&,hipStream_t) noexcept;
}
