/**
 * @file DeepSeekV4ImagePreprocess.h
 * @brief DeepSeek-V4-Flash-Vision-Exp image admission: patches, sentinel block, prompt expansion.
 *
 * A host-side port of the official ``inference/image_processor.py`` with the
 * production vLLM prompt layout (``vllm/models/deepseek_v4/common/mm_preprocess.py``):
 *  - the image is converted to RGB, its aspect clamped to 8:1, upscaled in
 *    bookkeeping to at least ``min_pixels``, and fitted to a 14-pixel grid
 *    whose sentinel block stays within 384 - 3 tokens (``safe_resize``);
 *  - images at least 8:1 wide are resized, all others ImageOps.pad-ded with
 *    gray 127 (Pillow-exact, multimodal/ImageIO.h);
 *  - patches are (x / 255 - 0.5) / 0.5 rounded to BF16, laid out
 *    [row-major patch][channel][ky][kx] (the ViT's linear patch embedding);
 *  - the block is [START] + aligner rows interleaved in pairs ("N-layout",
 *    every row ends with NEWLINE, an odd row count adds a PAD row) + PAD pairs
 *    + [END]; IMAGE slots take aligner outputs in ``perm`` order;
 *  - prompt expansion replaces each ``<｜deepseek_image｜>`` token with
 *    ``3 - start % 4`` IMAGE_PAD sentinels and the block, as in-vocab ids
 *    ``kSentinelBase + type`` (the official out-of-vocab ``vocab + type``
 *    scheme under vLLM's spelling).
 * Golden values: tests/v2/unit/models/deepseek4/fixtures (generated from
 * python/reference/deepseek4_vision.py, i.e. PIL).
 */
#pragma once

#include "../../multimodal/ImageIO.h"

#include <cstdint>
#include <vector>

namespace llaminar2::deepseek4
{
    /** @brief Sentinel roles; ids are ``kSentinelBase + type``. */
    enum class ImageSentinel : std::int32_t
    {
        Start = 0,
        Pad = 1,
        Image = 2,
        NewLine = 3,
        End = 4,
    };

    inline constexpr std::int32_t kImageSentinelBase = 129257;   ///< <|place_holder_mm_span_0431|>
    inline constexpr std::int32_t kImageSentinelCount = 5;
    inline constexpr std::int32_t kImagePlaceholderToken = 129264; ///< <｜deepseek_image｜>
    inline constexpr int kCompressPadTo = 4;

    /** @return Whether @p token is one of the five image sentinels. */
    constexpr bool isImageSentinel(std::int32_t token) noexcept
    {
        return token >= kImageSentinelBase && token < kImageSentinelBase + kImageSentinelCount;
    }

    /** @brief Vision geometry from the mmproj (``clip.vision.*``) and the fixed official limits. */
    struct VisionPreprocessConfig
    {
        int patch = 14;              ///< clip.vision.patch_size
        int merge = 3;               ///< clip.vision.projector.scale_factor (aligner downsample)
        int min_pixels = 147456;     ///< clip.vision.image_min_pixels
        int max_tokens = 384;        ///< vision_max_n_token (fixed by the official config)
        int max_wh_ratio = 8;        ///< vision_max_wh_ratio (fixed by the official config)
    };

    /** @brief One admitted image: ViT input, grid and sentinel block. */
    struct PreparedImage
    {
        int vit_h = 0, vit_w = 0;         ///< Patch grid.
        int llm_h = 0, llm_w = 0;         ///< Aligner grid (ceil(vit / merge)).
        std::vector<float> patches;       ///< [vit_h * vit_w, 3 * patch * patch], BF16-exact values.
        std::vector<std::int32_t> block;  ///< Pad-free sentinel types (START .. END).
        std::vector<std::int32_t> perm;   ///< Aligner output index of each IMAGE slot, block order.
        multimodal::RgbImage pixels;      ///< Resized/padded RGB8 (diagnostics and tests).
    };

    /**
     * @brief ``load_image`` + ``build_image_block`` for one decoded image.
     * @throws std::invalid_argument For degenerate geometry.
     */
    PreparedImage prepareImage(const multimodal::RgbImage &image, const VisionPreprocessConfig &config);

    /** @brief Pad-free block types and IMAGE-slot order for an aligner grid. */
    void buildImageBlock(int llm_h, int llm_w, std::vector<std::int32_t> &block, std::vector<std::int32_t> &perm);

    /** @brief Largest ViT patch count any admitted image can produce (capacity of the encoder). */
    int maxPatchesPerImage(const VisionPreprocessConfig &config);

    /**
     * @brief Expand every placeholder token into its compressor-aligned sentinel block.
     * @param tokens Prompt token ids from the chat encoder.
     * @param images Images in placeholder order.
     * @return Final prompt ids; throws std::invalid_argument on a count mismatch.
     */
    std::vector<std::int32_t> expandImagePlaceholders(const std::vector<std::int32_t> &tokens,
                                                      const std::vector<PreparedImage> &images);
} // namespace llaminar2::deepseek4
