/**
 * @file Test__DeepSeekV4ImagePreprocess.cpp
 * @brief DeepSeek-V4 image admission equals the Pillow reference byte for byte.
 *
 * Golden values come from scripts/deepseek4/make_image_preprocess_golden.py
 * (python/reference/deepseek4_vision.py, i.e. the official image_processor.py
 * on Pillow 13): resize/pad pixels, BF16 patches, grids, sentinel blocks,
 * IMAGE-slot order, PNG decoding of every Pillow mode, prompt expansion.
 */
#include "models/deepseek4/DeepSeekV4ImagePreprocess.h"
#include "multimodal/ImageIO.h"
#include "utils/Sha256.h"

#include "DeepSeekV4ImageGolden.generated.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace llaminar2;

namespace
{
/** @brief Same formula as the golden generator's ``synthetic``. */
multimodal::RgbImage synthetic(int w, int h)
{
    multimodal::RgbImage img{w, h, std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h * 3)};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                img.rgb[img.offset(x, y) + c] =
                    static_cast<std::uint8_t>((static_cast<long long>(x) * 7 + y * 13 + c * 101 + (static_cast<long long>(x) * y) % 37) & 255);
    return img;
}

std::string sha(const void *data, std::size_t bytes)
{
    return *sha256BytesHex(std::string_view(static_cast<const char *>(data), bytes));
}
} // namespace

/** @test Every resize/pad/upscale/aspect case reproduces Pillow's pixels, patches, grid and block. */
TEST(DeepSeekV4ImagePreprocess, MatchesPillowReference)
{
    const auto golden = nlohmann::json::parse(test::dsv4::kImagePreprocessGolden);
    ASSERT_GE(golden["cases"].size(), 10u);
    for (const auto &c : golden["cases"])
    {
        SCOPED_TRACE(std::to_string(c["width"].get<int>()) + "x" + std::to_string(c["height"].get<int>()));
        const auto prepared = deepseek4::prepareImage(synthetic(c["width"], c["height"]), deepseek4::VisionPreprocessConfig{});
        EXPECT_EQ(prepared.vit_h, c["vit_h"].get<int>());
        EXPECT_EQ(prepared.vit_w, c["vit_w"].get<int>());
        EXPECT_EQ(prepared.llm_h, c["llm_h"].get<int>());
        EXPECT_EQ(prepared.llm_w, c["llm_w"].get<int>());
        EXPECT_EQ(prepared.pixels.width, c["out_width"].get<int>());
        EXPECT_EQ(prepared.pixels.height, c["out_height"].get<int>());
        EXPECT_EQ(sha(prepared.pixels.rgb.data(), prepared.pixels.rgb.size()), c["pixels_sha256"].get<std::string>());
        EXPECT_EQ(sha(prepared.patches.data(), prepared.patches.size() * sizeof(float)), c["patches_sha256"].get<std::string>());
        EXPECT_EQ(prepared.block, c["block"].get<std::vector<std::int32_t>>());
        EXPECT_EQ(sha(prepared.perm.data(), prepared.perm.size() * sizeof(std::int32_t)), c["perm_sha256"].get<std::string>());
        EXPECT_EQ(prepared.perm.size(), static_cast<std::size_t>(prepared.llm_h * prepared.llm_w));
        EXPECT_LE(prepared.vit_h * prepared.vit_w, deepseek4::maxPatchesPerImage(deepseek4::VisionPreprocessConfig{}));
    }
}

/** @test PNG decoding converts every Pillow mode to RGB like convert("RGB"). */
TEST(DeepSeekV4ImagePreprocess, PngDecodingMatchesPillowConvertRgb)
{
    const auto golden = nlohmann::json::parse(test::dsv4::kImagePreprocessGolden);
    for (const auto &c : golden["decode"])
    {
        SCOPED_TRACE(c["mode"].get<std::string>());
        const auto bytes = multimodal::decodeBase64(c["png_base64"].get<std::string>());
        const auto img = multimodal::decodeImage(bytes, 1u << 24);
        EXPECT_EQ(img.width, c["width"].get<int>());
        EXPECT_EQ(img.height, c["height"].get<int>());
        EXPECT_EQ(sha(img.rgb.data(), img.rgb.size()), c["rgb_sha256"].get<std::string>());
    }
    EXPECT_THROW(multimodal::decodeImage(std::vector<std::uint8_t>{1, 2, 3}, 1u << 24), std::invalid_argument);
    const auto png = multimodal::decodeBase64(golden["decode"][0]["png_base64"].get<std::string>());
    EXPECT_THROW(multimodal::decodeImage(png, 10), std::invalid_argument); // pixel budget
}

/** @test Placeholders expand to compressor-aligned in-vocab sentinel blocks (vLLM layout). */
TEST(DeepSeekV4ImagePreprocess, PromptExpansionMatchesReference)
{
    const auto golden = nlohmann::json::parse(test::dsv4::kImagePreprocessGolden);
    const auto &e = golden["expansion"];
    std::vector<deepseek4::PreparedImage> images(2);
    images[0].block = e["blocks"][0].get<std::vector<std::int32_t>>();
    images[1].block = e["blocks"][1].get<std::vector<std::int32_t>>();
    EXPECT_EQ(deepseek4::expandImagePlaceholders(e["tokens"].get<std::vector<std::int32_t>>(), images),
              e["expanded"].get<std::vector<std::int32_t>>());
    images.pop_back();
    EXPECT_THROW(deepseek4::expandImagePlaceholders(e["tokens"].get<std::vector<std::int32_t>>(), images),
                 std::invalid_argument);
    // Every block starts at a position = 3 (mod 4) after its pad.
    images = std::vector<deepseek4::PreparedImage>(2);
    images[0].block = e["blocks"][0].get<std::vector<std::int32_t>>();
    images[1].block = e["blocks"][1].get<std::vector<std::int32_t>>();
    const auto expanded = deepseek4::expandImagePlaceholders(e["tokens"].get<std::vector<std::int32_t>>(), images);
    for (std::size_t i = 0; i < expanded.size(); ++i)
        if (expanded[i] == deepseek4::kImageSentinelBase)
            EXPECT_EQ((i + 1) % 4, 0u) << i;
}

/** @test Data URLs and base64 framing. */
TEST(DeepSeekV4ImagePreprocess, DataUrlsAndBase64)
{
    EXPECT_EQ(multimodal::decodeBase64("TWFu"), (std::vector<std::uint8_t>{'M', 'a', 'n'}));
    EXPECT_EQ(multimodal::decodeBase64("TWE="), (std::vector<std::uint8_t>{'M', 'a'}));
    EXPECT_EQ(multimodal::decodeDataUrl("data:image/png;base64,TQ=="), (std::vector<std::uint8_t>{'M'}));
    EXPECT_THROW(multimodal::decodeDataUrl("https://example.com/a.png"), std::invalid_argument);
    EXPECT_THROW(multimodal::decodeDataUrl("data:image/png,raw"), std::invalid_argument);
    EXPECT_THROW(multimodal::decodeBase64("T!=="), std::invalid_argument);
    EXPECT_EQ(multimodal::pythonRound(2.5), 2);
    EXPECT_EQ(multimodal::pythonRound(3.5), 4);
}
