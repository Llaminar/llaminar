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
 */
#pragma once

namespace llaminar2::rocm
{
/** @brief Explicit attention Q/K recording choices for native qualification. */
enum class FlashAttentionQKEngine { Automatic, OrderedDot2, WmmaFourPlane, ContextAdaptive };

/** @brief Compile-time ownership of a kernel's block-uniform Q/K decision. */
enum class FlashAttentionQKBranchPolicy { Unconditional, ContextAdaptive };

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
