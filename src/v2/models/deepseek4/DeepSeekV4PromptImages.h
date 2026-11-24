/**
 * @file DeepSeekV4PromptImages.h
 * @brief DeepSeek-V4-Flash-Vision-Exp request image pipeline (IPromptImageEncoder).
 *
 * Each ``<｜deepseek_image｜>`` placeholder expands to a compressor-aligned
 * sentinel block (vLLM ``mm_preprocess.py``). The replacement bank holds one
 * learned row per sentinel type at its type index (the IMAGE slot repeats the
 * PAD vector and is never named) followed by every image's aligner rows in
 * block order; the position map names a bank row for each sentinel position.
 * The text graph overwrites only sentinel rows (DeepSeekV4PrimitiveStage
 * ImageMerge), routes them with ``exp_probs_b_vl`` and widens their raw window
 * to the whole image span.
 */
#pragma once

#include "DeepSeekV4VisionEncoder.h"
#include "../../multimodal/PromptImageEncoder.h"

#include <memory>

namespace llaminar2::deepseek4
{
/** @brief Host vision tower plus DeepSeek-V4 prompt expansion. */
class DeepSeekV4PromptImageEncoder final : public multimodal::IPromptImageEncoder
{
public:
    /** @param encoder Loaded vision tower (shared, immutable). */
    explicit DeepSeekV4PromptImageEncoder(std::shared_ptr<const DeepSeekV4VisionEncoder> encoder);

    const std::string &placeholderText() const noexcept override;
    std::int32_t placeholderToken() const noexcept override { return kImagePlaceholderToken; }
    int maxImagesPerRequest() const noexcept override { return kMaxImagesPerRequest; }
    multimodal::ExpandedPrompt expand(const std::vector<std::int32_t> &tokens,
                                      const std::vector<std::vector<std::uint8_t>> &images) const override;

    /**
     * @brief Expansion from already prepared images and their encoded rows (tests, diagnostics).
     * @param encoded Per image, [perm.size(), width] aligner rows in block order.
     * @param sentinels Learned rows for START, PAD, IMAGE (unused), NEWLINE, END.
     */
    static multimodal::ExpandedPrompt expandPrepared(const std::vector<std::int32_t> &tokens,
                                                     const std::vector<PreparedImage> &images,
                                                     const std::vector<std::vector<float>> &encoded,
                                                     const std::array<const std::vector<float> *, kImageSentinelCount> &sentinels,
                                                     int width);

private:
    std::shared_ptr<const DeepSeekV4VisionEncoder> encoder_;
};
} // namespace llaminar2::deepseek4
