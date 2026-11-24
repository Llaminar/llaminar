/**
 * @file DeepSeekV4ImagePreprocess.cpp
 * @brief Line-by-line port of image_processor.py (grid_tokens .. build_image_block).
 *
 * Floating-point steps use doubles in the same operation order as Python
 * (IEEE double, correctly rounded sqrt), so every grid decision matches the
 * reference; ``math.ceil(a / b)`` of positive integers is evaluated exactly.
 */
#include "DeepSeekV4ImagePreprocess.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llaminar2::deepseek4
{
    namespace
    {
        struct Grid
        {
            int llm_h, llm_w, tokens;
        };

        Grid gridTokens(int best_height, int best_width, int p, int r)
        {
            const int llm_h = static_cast<int>(std::ceil(static_cast<double>(best_height / p) / r));
            const int llm_w = static_cast<int>(std::ceil(static_cast<double>(best_width / p) / r));
            int tokens = llm_h * (llm_w + 1) + 2;
            if (llm_h % 2 == 1) tokens += llm_w + 1;
            tokens += (llm_h + 1) / 2 * (llm_w + 1) % 2 * 2;
            return {llm_h, llm_w, tokens};
        }

        struct Fit
        {
            Grid grid;
            int best_height, best_width;
        };

        Fit solveResizeRatio(int height, int width, int p, int r, int max_n_token)
        {
            const double ratio = static_cast<double>(height) / width;
            const double max_w_float = std::sqrt((max_n_token - 2) / ratio + 0.25) - 0.5;
            const double max_h_float = max_w_float * ratio;
            int best_width = 0, best_height = 0;
            if (max_w_float < 1.0)
            {
                const int max_w = 1;
                int max_h = (max_n_token - 2) / (max_w + 1);
                if (max_h % 2 == 1) --max_h;
                best_width = max_w * p * r;
                best_height = max_h * p * r;
            }
            else if (max_h_float < 2.0)
            {
                const int max_h = 2;
                const int max_w = ((max_n_token - 2) / max_h) - 1;
                if (max_w <= 1) throw std::invalid_argument("DeepSeek-V4 image grid has no admissible width");
                best_width = max_w * p * r;
                best_height = max_h * p * r;
            }
            else
            {
                const int max_w = static_cast<int>(std::floor(max_w_float));
                int max_h = static_cast<int>(std::floor(max_h_float));
                if (max_h % 2 == 1) --max_h;
                const double beta = std::min(static_cast<double>(max_w * p * r) / width,
                                             static_cast<double>(max_h * p * r) / height);
                best_width = static_cast<int>(std::floor(width * beta / p)) * p;
                best_height = static_cast<int>(std::floor(height * beta / p)) * p;
            }
            return {gridTokens(best_height, best_width, p, r), best_height, best_width};
        }

        float bf16Round(float value)
        {
            std::uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            const std::uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u); // round to nearest even
            bits = (bits + rounding) & 0xFFFF0000u;
            float out;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }
    } // namespace

    void buildImageBlock(int llm_h, int llm_w, std::vector<std::int32_t> &block, std::vector<std::int32_t> &perm)
    {
        const int pad_h = llm_h % 2;
        const int rows = llm_h + pad_h;
        const int row_len = llm_w + 1;
        const int pad_last = rows / 2 * row_len % 2 * 2;
        // Row-major types and aligner indices before the pairwise interleave.
        std::vector<std::int32_t> types(static_cast<std::size_t>(rows) * row_len);
        std::vector<std::int32_t> image_idx(types.size(), -1);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < row_len; ++x)
            {
                const std::size_t i = static_cast<std::size_t>(y) * row_len + x;
                if (y >= llm_h) types[i] = static_cast<std::int32_t>(ImageSentinel::Pad);
                else if (x == llm_w) types[i] = static_cast<std::int32_t>(ImageSentinel::NewLine);
                else
                {
                    types[i] = static_cast<std::int32_t>(ImageSentinel::Image);
                    image_idx[i] = y * llm_w + x;
                }
            }
        block.clear();
        perm.clear();
        block.push_back(static_cast<std::int32_t>(ImageSentinel::Start));
        // order = arange(rows*row_len).view(rows/2, 2, row_len).transpose(1, 2): for each row
        // pair, alternate the two rows column by column.
        for (int pair = 0; pair < rows / 2; ++pair)
            for (int x = 0; x < row_len; ++x)
                for (int sub = 0; sub < 2; ++sub)
                {
                    const std::size_t i = static_cast<std::size_t>(pair * 2 + sub) * row_len + x;
                    block.push_back(types[i]);
                    if (image_idx[i] >= 0) perm.push_back(image_idx[i]);
                }
        for (int i = 0; i < pad_last; ++i) block.push_back(static_cast<std::int32_t>(ImageSentinel::Pad));
        block.push_back(static_cast<std::int32_t>(ImageSentinel::End));
    }

    PreparedImage prepareImage(const multimodal::RgbImage &image, const VisionPreprocessConfig &c)
    {
        if (image.width <= 0 || image.height <= 0)
            throw std::invalid_argument("DeepSeek-V4 image has no pixels");
        const int p = c.patch, r = c.merge;
        long long width = image.width, height = image.height;
        if (width > height * c.max_wh_ratio) width = height * c.max_wh_ratio;
        if (0 < width * height && width * height < c.min_pixels)
        {
            const double ratio = std::pow(static_cast<double>(c.min_pixels) / static_cast<double>(width * height), 0.5);
            width = static_cast<long long>(width * ratio);
            height = static_cast<long long>(height * ratio);
        }
        int best_width = static_cast<int>((width + p - 1) / p * p);
        int best_height = static_cast<int>((height + p - 1) / p * p);

        // safe_resize
        const int budget_limit = c.max_tokens - (kCompressPadTo - 1);
        Grid grid = gridTokens(best_height, best_width, p, r);
        int budget = budget_limit;
        while (grid.tokens > budget_limit)
        {
            const Fit fit = solveResizeRatio(static_cast<int>(height), static_cast<int>(width), p, r, budget);
            grid = fit.grid;
            best_height = fit.best_height;
            best_width = fit.best_width;
            --budget;
            if (budget < 3) throw std::invalid_argument("DeepSeek-V4 image grid cannot fit the token budget");
        }

        PreparedImage out;
        out.vit_h = best_height / p;
        out.vit_w = best_width / p;
        out.llm_h = grid.llm_h;
        out.llm_w = grid.llm_w;
        if (out.vit_h <= 0 || out.vit_w <= 0)
            throw std::invalid_argument("DeepSeek-V4 image grid is empty");
        static constexpr std::uint8_t kGray[3] = {127, 127, 127};
        out.pixels = static_cast<long long>(image.width) >= static_cast<long long>(c.max_wh_ratio) * image.height
                         ? multimodal::resizeBicubic(image, best_width, best_height)
                         : multimodal::pad(image, best_width, best_height, kGray);

        const int pp = p * p;
        out.patches.assign(static_cast<std::size_t>(out.vit_h) * out.vit_w * 3 * pp, 0.0f);
        for (int py = 0; py < out.vit_h; ++py)
            for (int px = 0; px < out.vit_w; ++px)
            {
                float *patch = &out.patches[(static_cast<std::size_t>(py) * out.vit_w + px) * 3 * pp];
                for (int ch = 0; ch < 3; ++ch)
                    for (int ky = 0; ky < p; ++ky)
                        for (int kx = 0; kx < p; ++kx)
                        {
                            const float x = static_cast<float>(out.pixels.rgb[out.pixels.offset(px * p + kx, py * p + ky) + ch]) / 255.0f;
                            patch[ch * pp + ky * p + kx] = bf16Round((x - 0.5f) / 0.5f);
                        }
            }
        buildImageBlock(out.llm_h, out.llm_w, out.block, out.perm);
        return out;
    }

    int maxPatchesPerImage(const VisionPreprocessConfig &c)
    {
        // Every admissible aligner grid fits the token budget; the ViT grid is
        // at most merge times each aligner dimension.
        const int budget = c.max_tokens - (kCompressPadTo - 1);
        const int step = c.patch * c.merge;
        int best = 0;
        for (int h = 1; h < budget; ++h)
        {
            if (gridTokens(h * step, step, c.patch, c.merge).tokens > budget) break;
            for (int w = 1; w < budget; ++w)
            {
                if (gridTokens(h * step, w * step, c.patch, c.merge).tokens > budget) break;
                best = std::max(best, h * w * c.merge * c.merge);
            }
        }
        return best;
    }

    int maxImageSlotsPerImage(int max_tokens)
    {
        const int budget = max_tokens - (kCompressPadTo - 1);
        int best = 0;
        for (int h = 1; h < budget; ++h)
        {
            if (gridTokens(h * 42, 42, 14, 3).tokens > budget) break;
            for (int w = 1; w < budget; ++w)
            {
                if (gridTokens(h * 42, w * 42, 14, 3).tokens > budget) break;
                best = std::max(best, h * w);
            }
        }
        return best;
    }

    std::vector<std::int32_t> expandImagePlaceholders(const std::vector<std::int32_t> &tokens,
                                                      const std::vector<PreparedImage> &images)
    {
        std::vector<std::int32_t> out;
        out.reserve(tokens.size());
        std::size_t next = 0;
        for (const auto token : tokens)
        {
            if (token != kImagePlaceholderToken)
            {
                out.push_back(token);
                continue;
            }
            if (next >= images.size())
                throw std::invalid_argument("prompt has more image placeholders than images");
            const int pad = kCompressPadTo - 1 - static_cast<int>(out.size() % kCompressPadTo);
            for (int i = 0; i < pad; ++i)
                out.push_back(kImageSentinelBase + static_cast<std::int32_t>(ImageSentinel::Pad));
            for (const auto type : images[next].block) out.push_back(kImageSentinelBase + type);
            ++next;
        }
        if (next != images.size())
            throw std::invalid_argument("prompt has fewer image placeholders than images");
        return out;
    }
} // namespace llaminar2::deepseek4
