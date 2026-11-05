/**
 * @file PromptImageEncoder.h
 * @brief Model-agnostic request-side image input: placeholder text, encoding and prompt expansion.
 *
 * The request layer (HTTP chat completion, single-shot chat) renders each image
 * as the model's placeholder text, tokenizes the conversation, then hands the
 * token ids and the raw image bytes (in placeholder order) to the model's
 * encoder. The encoder decodes and preprocesses the images, runs the vision
 * tower on the host and returns the final prompt ids together with the
 * immutable PromptEmbeddingRows the participant admits with the request.
 *
 * Errors in client data (undecodable images, too many images, a placeholder
 * count mismatch) throw std::invalid_argument so the request layer can answer
 * 400; nothing is silently dropped or rendered as text.
 */
#pragma once

#include "../execution/PromptEmbeddingRows.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2::multimodal
{
/** @brief The final prompt of a multimodal request. */
struct ExpandedPrompt
{
    std::vector<std::int32_t> tokens;                ///< Prompt ids with every placeholder expanded.
    std::shared_ptr<const PromptEmbeddingRows> rows; ///< Replacement rows indexed by prompt position.
};

/** @brief A model's host image pipeline; immutable and safe for concurrent use. */
class IPromptImageEncoder
{
public:
    virtual ~IPromptImageEncoder() = default;

    /** @return Text rendered at each image's position before tokenization. */
    virtual const std::string &placeholderText() const noexcept = 0;

    /** @return Token id the placeholder text tokenizes to. */
    virtual std::int32_t placeholderToken() const noexcept = 0;

    /** @return Most images one request may carry (the admitted bank capacity). */
    virtual int maxImagesPerRequest() const noexcept = 0;

    /**
     * @brief Expand @p tokens (one placeholder token per image) into the final prompt.
     * @param images Encoded image files (PNG/JPEG/...) in placeholder order.
     * @throws std::invalid_argument For client errors (decode failure, limits, count mismatch).
     */
    virtual ExpandedPrompt expand(const std::vector<std::int32_t> &tokens,
                                  const std::vector<std::vector<std::uint8_t>> &images) const = 0;
};
} // namespace llaminar2::multimodal
