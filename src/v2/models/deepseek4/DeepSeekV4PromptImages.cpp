/**
 * @file DeepSeekV4PromptImages.cpp
 * @brief Decode, preprocess and encode request images, then expand the prompt and build its bank.
 */
#include "DeepSeekV4PromptImages.h"

#include "../../multimodal/ImageIO.h"
#include "../../utils/Logger.h"

#include <stdexcept>
#include <string>

namespace llaminar2::deepseek4
{
namespace
{
/** Decoded-image pixel ceiling: far above any admitted geometry, bounds decoder memory. */
constexpr std::uint64_t kMaxDecodedPixels = 64ull << 20;
const std::string kPlaceholderText = "<\xEF\xBD\x9C" "deepseek_image" "\xEF\xBD\x9C>";
} // namespace

DeepSeekV4PromptImageEncoder::DeepSeekV4PromptImageEncoder(std::shared_ptr<const DeepSeekV4VisionEncoder> encoder)
    : encoder_(std::move(encoder))
{
    if (!encoder_) throw std::invalid_argument("DeepSeek-V4 prompt images require a vision encoder");
}

const std::string &DeepSeekV4PromptImageEncoder::placeholderText() const noexcept { return kPlaceholderText; }

multimodal::ExpandedPrompt DeepSeekV4PromptImageEncoder::expand(
    const std::vector<std::int32_t> &tokens, const std::vector<std::vector<std::uint8_t>> &images) const
{
    if (static_cast<int>(images.size()) > kMaxImagesPerRequest)
        throw std::invalid_argument("a request may carry at most " + std::to_string(kMaxImagesPerRequest) + " images");
    std::vector<PreparedImage> prepared;
    std::vector<std::vector<float>> encoded;
    prepared.reserve(images.size());
    encoded.reserve(images.size());
    for (std::size_t i = 0; i < images.size(); ++i)
    {
        multimodal::RgbImage rgb;
        try
        {
            rgb = multimodal::decodeImage(images[i], kMaxDecodedPixels);
        }
        catch (const std::exception &error)
        {
            throw std::invalid_argument("image " + std::to_string(i) + " could not be decoded: " + error.what());
        }
        prepared.push_back(prepareImage(rgb, encoder_->preprocessConfig()));
        LOG_DEBUG("[Vision] image " << i << " " << rgb.width << "x" << rgb.height << " -> ViT " << prepared.back().vit_h
                  << "x" << prepared.back().vit_w << ", aligner " << prepared.back().llm_h << "x" << prepared.back().llm_w
                  << ", block " << prepared.back().block.size() << " tokens");
        encoded.push_back(encoder_->encode(prepared.back()));
    }
    std::array<const std::vector<float> *, kImageSentinelCount> sentinels{};
    for (int t = 0; t < kImageSentinelCount; ++t)
    {
        const auto role = static_cast<ImageSentinel>(t);
        sentinels[static_cast<std::size_t>(t)] =
            &encoder_->sentinel(role == ImageSentinel::Image ? ImageSentinel::Pad : role);
    }
    return expandPrepared(tokens, prepared, encoded, sentinels, encoder_->textDim());
}

multimodal::ExpandedPrompt DeepSeekV4PromptImageEncoder::expandPrepared(
    const std::vector<std::int32_t> &tokens, const std::vector<PreparedImage> &images,
    const std::vector<std::vector<float>> &encoded,
    const std::array<const std::vector<float> *, kImageSentinelCount> &sentinels, int width)
{
    if (encoded.size() != images.size() || width <= 0)
        throw std::invalid_argument("DeepSeek-V4 prompt expansion needs one encoding per image");
    multimodal::ExpandedPrompt out;
    out.tokens = expandImagePlaceholders(tokens, images);
    auto rows = std::make_shared<PromptEmbeddingRows>();
    rows->width = width;
    const auto w = static_cast<std::size_t>(width);
    for (const auto *sentinel : sentinels)
    {
        if (!sentinel || sentinel->size() != w)
            throw std::invalid_argument("DeepSeek-V4 sentinel rows must match the text width");
        rows->rows.insert(rows->rows.end(), sentinel->begin(), sentinel->end());
    }
    // Walk the expanded prompt: sentinel positions name their type row, IMAGE
    // slots the next encoded row of the current image (block order).
    rows->position_rows.assign(out.tokens.size(), 0);
    std::size_t image = 0, slot = 0, image_base = 0;
    bool in_block = false;
    for (std::size_t p = 0; p < out.tokens.size(); ++p)
    {
        const auto token = out.tokens[p];
        if (!isImageSentinel(token)) continue;
        const auto type = static_cast<ImageSentinel>(token - kImageSentinelBase);
        if (type == ImageSentinel::Start)
        {
            if (image >= images.size()) throw std::invalid_argument("prompt text contains an image START sentinel");
            const auto &e = encoded[image];
            if (e.size() != images[image].perm.size() * w)
                throw std::invalid_argument("image encoding does not match its IMAGE slot count");
            image_base = rows->rowCount();
            rows->rows.insert(rows->rows.end(), e.begin(), e.end());
            slot = 0;
            in_block = true;
            rows->atomic_spans.emplace_back(static_cast<int>(p), static_cast<int>(p));
        }
        if (type == ImageSentinel::Image)
        {
            if (!in_block || slot >= images[image].perm.size())
                throw std::invalid_argument("prompt text contains an IMAGE sentinel outside an image");
            rows->position_rows[p] = static_cast<std::int32_t>(image_base + slot + 1);
            ++slot;
            continue;
        }
        rows->position_rows[p] = static_cast<std::int32_t>(token - kImageSentinelBase) + 1;
        if (type == ImageSentinel::End)
        {
            if (!in_block || slot != images[image].perm.size())
                throw std::invalid_argument("prompt text contains an image END sentinel");
            in_block = false;
            rows->atomic_spans.back().second = static_cast<int>(p) + 1;
            ++image;
        }
    }
    if (image != images.size()) throw std::invalid_argument("prompt text breaks an image block");
    out.rows = std::move(rows);
    return out;
}
} // namespace llaminar2::deepseek4
