/**
 * @file ROCmFlashAttentionQK.h
 * @brief Recording policy for byte-equivalent native attention dot engines.
 *
 * The worker selects a concrete kernel while recording. Retained graphs embed
 * that function and never consult host policy during replay. Four-plane WMMA
 * preserves the ordered half-dot chains; softmax and FP32 P/V remain shared.
 * The adaptive candidate reads the published live prefix on device and keeps
 * both arithmetic branches inside one retained query kernel. Recording never
 * downloads a live length or changes topology as the context grows or resets.
 * Automatic recording resolves the measured immutable domain once; its live
 * prefix threshold remains entirely device-owned inside the retained kernel.
 */
#pragma once

#include "ROCmFlashAttentionLaunchPolicy.h"
#include <optional>
#include <string_view>

namespace llaminar2::rocm
{
/** @brief Explicit attention Q/K recording choices for native qualification. */
enum class FlashAttentionQKEngine { Automatic, OrderedDot2, WmmaFourPlane, ContextAdaptive };

/** @brief Compile-time ownership of a kernel's block-uniform Q/K decision. */
enum class FlashAttentionQKBranchPolicy { Unconditional, ContextAdaptive };

/** @brief Authority supplying the live geometry consumed by a retained kernel. */
enum class FlashAttentionQKParameterSource { StaticArguments, DevicePublication };

/** @brief Resolve a recording request from actual immutable launch inputs.
 * @param requested Automatic production selection or explicit diagnostic engine.
 * @param native_isa Authenticated physical ISA; no compatibility/spoofed target.
 * @param geometry Complete original capture geometry, including KV head mapping.
 * @param tile Resolved hardware-admitted tile and score ownership.
 * @param parameters Whether a device parameter producer owns replay geometry.
 * @param direct_partition_limit Nonzero for a guarded leg of a context transaction.
 * @return Concrete engine, or no value for an invalid/unsupported explicit request.
 * Live counts never enter this host policy. The chosen function is embedded in
 * the native graph; existing immutable graph ownership and workspace generations
 * keep that capture identity valid across resets and changing live prefixes.
 */
[[nodiscard]] inline constexpr std::optional<FlashAttentionQKEngine> resolveFlashAttentionQKEngine(
    FlashAttentionQKEngine requested, std::string_view native_isa,
    const attention::AttentionPrefillCaptureGeometry &geometry,
    const fa2_policy::ROCmFA2TilePlan &tile,
    FlashAttentionQKParameterSource parameters, int direct_partition_limit) noexcept
{
    using Engine = FlashAttentionQKEngine;
    if (!geometry.valid() || geometry.visible_kv_heads <= 0 || !tile.valid || direct_partition_limit < 0)
        return std::nullopt;
    const bool native_format = geometry.native_kv_format == TensorType::FP16 ||
        geometry.native_kv_format == TensorType::FP32 || geometry.native_kv_format == TensorType::BF16 ||
        geometry.native_kv_format == TensorType::Q8_1;
    const bool native_tile = (native_isa == "gfx1100" || native_isa.starts_with("gfx1100:")) &&
        native_format && geometry.head_dim == 256 && tile.query_rows == 16 &&
        tile.kv_rows == 16 && tile.threads_per_row == 16;
    const bool adaptive = native_tile && geometry.batch_size == 1 && geometry.query_rows == 512 &&
        geometry.kv_capacity == 65536 && geometry.native_kv_format == TensorType::FP16 &&
        ((geometry.local_query_heads == 12 && geometry.visible_kv_heads == 2) ||
         (geometry.local_query_heads == 8 && geometry.visible_kv_heads == 1)) &&
        geometry.head_start == 0 && geometry.replicated_gqa_n_rep == 0 &&
        geometry.visibility == attention::AttentionVisibilityPolicy::CausalFullContext &&
        tile.schedule == fa2_policy::ROCmFA2PrefillPhysicalMode::QuerySequence &&
        tile.score_storage == fa2_policy::ROCmFA2ScoreStorage::RegisterOwnedXorTile &&
        parameters == FlashAttentionQKParameterSource::DevicePublication && direct_partition_limit == 0;
    switch (requested)
    {
    case Engine::Automatic: return adaptive ? Engine::ContextAdaptive : Engine::OrderedDot2;
    case Engine::OrderedDot2: return Engine::OrderedDot2;
    case Engine::WmmaFourPlane: return native_tile ? std::optional{Engine::WmmaFourPlane} : std::nullopt;
    case Engine::ContextAdaptive: return adaptive ? std::optional{Engine::ContextAdaptive} : std::nullopt;
    }
    return std::nullopt;
}

/** @param engine Resolved or requested engine. @return Stable counter/diagnostic name. */
[[nodiscard]] inline constexpr const char *flashAttentionQKEngineName(FlashAttentionQKEngine engine) noexcept
{
    switch (engine)
    {
    case FlashAttentionQKEngine::Automatic: return "automatic";
    case FlashAttentionQKEngine::OrderedDot2: return "ordered_dot2";
    case FlashAttentionQKEngine::WmmaFourPlane: return "wmma_four_plane";
    case FlashAttentionQKEngine::ContextAdaptive: return "context_adaptive";
    }
    return "invalid";
}

/** @brief Observe a successfully submitted native attention kernel on the host.
 * @param engine Actual resolved engine embedded in the graph.
 * @param geometry Complete immutable launch geometry, including head ownership.
 * @param tile Actual physical tile and schedule.
 * @param native_isa Authenticated physical target.
 * @param device Visible owning device.
 * This counts submissions, not device executions of the matrix branch. Retained
 * replay never calls this function or reads live counts back to the host.
 */
void recordFlashAttentionQKSubmission(FlashAttentionQKEngine engine,
    const attention::AttentionPrefillCaptureGeometry &geometry,
    const fa2_policy::ROCmFA2TilePlan &tile, std::string_view native_isa, int device);

/** @brief Select the measured long-prefix arithmetic from published device values.
 * @param live_kv Complete live key extent, including the current query bucket.
 * @param kv_stride Retained physical row capacity of each batch's KV bank.
 * @param position_offset Absolute position of the first query in its live prefix.
 * @param ring_origin Physical first row; only linear storage is qualified here.
 * @param ring_capacity Zero for linear storage; positive values name a ring.
 * @return Whether the exact measured prefix domain selects native matrix Q/K.
 * Short prefixes and other live storage layouts retain ordered Dot2 inside the
 * same candidate. Immutable geometry is admitted separately before recording.
 */
#if defined(__HIPCC__)
__host__ __device__
#endif
constexpr bool useContextAdaptiveAttentionWmma(int live_kv, int kv_stride,
    int position_offset, int ring_origin, int ring_capacity) noexcept
{
    return live_kv >= 1024 && live_kv <= 65536 && kv_stride == 65536 &&
        position_offset == live_kv - 512 && ring_origin == 0 && ring_capacity == 0;
}

/** @brief Nestable recording scope; existing graphs retain their original engine. */
class ScopedFlashAttentionQKEngine final
{
public:
    /** @brief Select the next recording's dot engine on this worker thread.
     * @param engine Installed policy or explicit native candidate.
     * @throws std::invalid_argument If engine is not a declared enum value. */
    explicit ScopedFlashAttentionQKEngine(FlashAttentionQKEngine engine);
    /** @brief Restore the enclosing recording policy. */
    ~ScopedFlashAttentionQKEngine();
    ScopedFlashAttentionQKEngine(const ScopedFlashAttentionQKEngine &) = delete;
    ScopedFlashAttentionQKEngine &operator=(const ScopedFlashAttentionQKEngine &) = delete;
private:
    FlashAttentionQKEngine previous_;
};
}
