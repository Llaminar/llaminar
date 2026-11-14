/**
 * @file ROCmFlashAttentionQK.h
 * @brief Recording policy for byte-equivalent native attention dot engines.
 *
 * The worker selects a concrete kernel while recording. Retained graphs embed
 * that function and never consult host policy during replay. Four-plane WMMA
 * preserves the ordered half-dot chains; softmax and FP32 P/V remain shared.
 */
#pragma once

namespace llaminar2::rocm
{
/** @brief Explicit attention Q/K recording choices for native qualification. */
enum class FlashAttentionQKEngine { Automatic, OrderedDot2, WmmaFourPlane };

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
