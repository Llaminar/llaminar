/**
 * @file Test__DeepSeekV4VisionEncoder.cpp
 * @brief Host vision tower equals the reference (official vision.py math) on synthetic weights.
 *
 * scripts/deepseek4/make_vision_encoder_golden.py builds a small tower
 * (ViT width 16, 2 heads, 2 blocks, text width 32) from the integer hash
 * below and records python/reference/deepseek4_vision.py outputs; this test
 * builds identical tensors and compares the block-ordered rows. Grids cover
 * aligner padding in both directions and the 1x1 image.
 */
#include "models/deepseek4/DeepSeekV4VisionEncoder.h"
#include "mocks/MockMPIContext.h"

#include "DeepSeekV4VisionEncoderGolden.generated.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::deepseek4;

namespace
{
std::vector<float> synth(std::int64_t seed, std::size_t n, float scale)
{
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::int64_t v = (static_cast<std::int64_t>(i) * 2654435761LL + seed * 97 + 12345) % 2001;
        float x = static_cast<float>(v - 1000);
        x = x / 1000.0f;
        out[i] = x * scale;
    }
    return out;
}

float bf16(float v)
{
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    b = (b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u;
    std::memcpy(&v, &b, 4);
    return v;
}

std::map<std::string, VisionHostTensor> tensors(const nlohmann::json &golden)
{
    // Same scales as the generator; seeds are the sorted-name ordinals.
    std::map<std::string, float> scales;
    for (const auto &[name, _] : golden["tensors"].items())
    {
        float s = 0.2f;
        if (name == "v.patch_embd.weight") s = 0.05f;
        else if (name.ends_with(".bias")) s = 0.1f;
        else if (name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "v.post_ln.weight") s = 0.5f;
        else if (name == "mm.1.weight") s = 0.08f;
        else if (name == "mm.2.weight") s = 0.15f;
        else if (name.starts_with("v.token_embd") || name == "v.image_newline") s = 1.0f;
        scales[name] = s;
    }
    std::map<std::string, VisionHostTensor> out;
    std::int64_t seed = 0;
    for (const auto &[name, scale] : scales) // std::map: sorted like Python's sorted()
    {
        VisionHostTensor t;
        t.shape = golden["tensors"][name].get<std::vector<std::size_t>>();
        std::size_t n = 1;
        for (auto d : t.shape) n *= d;
        t.data = synth(seed++, n, scale);
        if (name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "v.post_ln.weight")
            for (auto &v : t.data) v = v + 1.0f;
        out.emplace(name, std::move(t));
    }
    return out;
}
} // namespace

/** @test Rows and sentinel vectors match the reference within FP32 BLAS reassociation. */
TEST(DeepSeekV4VisionEncoder, MatchesReferenceOnSyntheticTower)
{
    const auto golden = nlohmann::json::parse(test::dsv4::kVisionEncoderGolden);
    DeepSeekV4VisionEncoder encoder(tensors(golden), VisionPreprocessConfig{}, golden["layers"], golden["heads"]);
    ASSERT_EQ(encoder.textDim(), golden["text_dim"].get<int>());
    for (const auto &c : golden["cases"])
    {
        PreparedImage img;
        img.vit_h = c["vit_h"];
        img.vit_w = c["vit_w"];
        img.llm_h = (img.vit_h + 2) / 3;
        img.llm_w = (img.vit_w + 2) / 3;
        SCOPED_TRACE(std::to_string(img.vit_h) + "x" + std::to_string(img.vit_w));
        img.patches = synth(1000 + img.vit_h * 10 + img.vit_w, static_cast<std::size_t>(img.vit_h) * img.vit_w * 588, 1.0f);
        for (auto &v : img.patches) v = bf16(v);
        buildImageBlock(img.llm_h, img.llm_w, img.block, img.perm);
        const auto rows = encoder.encode(img);
        const auto expected = c["rows"].get<std::vector<float>>();
        ASSERT_EQ(rows.size(), expected.size());
        double err = 0.0, ref = 0.0;
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            err += (rows[i] - expected[i]) * static_cast<double>(rows[i] - expected[i]);
            ref += expected[i] * static_cast<double>(expected[i]);
        }
        EXPECT_LT(std::sqrt(err / ref), 1e-5);
    }
    const auto sentinels = golden["sentinels"];
    for (const auto role : {ImageSentinel::Start, ImageSentinel::Pad, ImageSentinel::NewLine, ImageSentinel::End})
        EXPECT_EQ(encoder.sentinel(role), sentinels[static_cast<int>(role)].get<std::vector<float>>());
    EXPECT_THROW(encoder.sentinel(ImageSentinel::Image), std::invalid_argument);
}

/** @test Missing or mis-shaped tensors are rejected at construction. */
TEST(DeepSeekV4VisionEncoder, RejectsInconsistentProjector)
{
    const auto golden = nlohmann::json::parse(test::dsv4::kVisionEncoderGolden);
    auto t = tensors(golden);
    t.erase("v.blk.1.attn_out.bias");
    EXPECT_THROW(DeepSeekV4VisionEncoder(t, VisionPreprocessConfig{}, 2, 2), std::invalid_argument);
    t = tensors(golden);
    t["mm.2.bias"].shape = {31};
    EXPECT_THROW(DeepSeekV4VisionEncoder(t, VisionPreprocessConfig{}, 2, 2), std::invalid_argument);
}

namespace
{
/** @brief Minimal GGUF v3 writer: string/u32 metadata and F32 tensors with ggml (ne0-first) dims. */
void writeMmproj(const std::string &path, const std::map<std::string, VisionHostTensor> &tensors, int layers, int heads)
{
    std::ofstream out(path, std::ios::binary);
    const auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<const char *>(&v), 4); };
    const auto u64 = [&](std::uint64_t v) { out.write(reinterpret_cast<const char *>(&v), 8); };
    const auto str = [&](const std::string &v) { u64(v.size()); out.write(v.data(), static_cast<std::streamsize>(v.size())); };
    const std::vector<std::pair<std::string, std::string>> strings{
        {"general.architecture", "clip"}, {"clip.projector_type", "deepseek4v"}};
    const std::vector<std::pair<std::string, std::uint32_t>> ints{
        {"clip.vision.patch_size", 14}, {"clip.vision.projector.scale_factor", 3},
        {"clip.vision.image_min_pixels", 147456}, {"clip.vision.block_count", std::uint32_t(layers)},
        {"clip.vision.attention.head_count", std::uint32_t(heads)}};
    out.write("GGUF", 4);
    u32(3);
    u64(tensors.size());
    u64(strings.size() + ints.size());
    for (const auto &[k, v] : strings) { str(k); u32(8); str(v); }  // GGUF_TYPE_STRING
    for (const auto &[k, v] : ints) { str(k); u32(4); u32(v); }     // GGUF_TYPE_UINT32
    std::uint64_t offset = 0;
    const auto aligned = [](std::uint64_t v) { return (v + 31) / 32 * 32; };
    for (const auto &[name, t] : tensors)
    {
        str(name);
        u32(static_cast<std::uint32_t>(t.shape.size()));
        for (const auto d : t.shape) u64(d);
        u32(0); // F32
        u64(offset);
        offset = aligned(offset + t.data.size() * sizeof(float));
    }
    out.seekp(static_cast<std::streamoff>(aligned(static_cast<std::uint64_t>(out.tellp()))));
    const auto base = static_cast<std::uint64_t>(out.tellp());
    for (const auto &[name, t] : tensors)
    {
        out.write(reinterpret_cast<const char *>(t.data.data()), static_cast<std::streamsize>(t.data.size() * 4));
        const auto end = static_cast<std::uint64_t>(out.tellp()) - base;
        const std::vector<char> pad(aligned(end) - end, 0);
        out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    }
}
} // namespace

/**
 * @test A projector file loads to the same tower as the in-memory tensors.
 * @details Regression: the GGUF loader swaps only 2D dimensions, so the 4D
 * patch embedding must keep its ggml order (a full reversal broke --mmproj).
 */
TEST(DeepSeekV4VisionEncoder, MmprojFileLoadsTheSameTower)
{
    const auto golden = nlohmann::json::parse(test::dsv4::kVisionEncoderGolden);
    const auto t = tensors(golden);
    char pattern[] = "/tmp/dsv4-mmproj-XXXXXX";
    const int fd = ::mkstemp(pattern);
    ASSERT_GE(fd, 0);
    ::close(fd);
    const std::string path = pattern;
    writeMmproj(path, t, golden["layers"], golden["heads"]);
    test::MockMPIContext mpi;
    std::unique_ptr<DeepSeekV4VisionEncoder> loaded;
    EXPECT_NO_THROW(loaded = DeepSeekV4VisionEncoder::load(path, mpi, golden["text_dim"].get<int>()));
    EXPECT_THROW(DeepSeekV4VisionEncoder::load(path, mpi, golden["text_dim"].get<int>() + 1), std::invalid_argument);
    ::unlink(path.c_str());
    ASSERT_TRUE(loaded);
    DeepSeekV4VisionEncoder direct(t, VisionPreprocessConfig{}, golden["layers"], golden["heads"]);
    PreparedImage img;
    img.vit_h = 4;
    img.vit_w = 5;
    img.llm_h = 2;
    img.llm_w = 2;
    img.patches = synth(77, static_cast<std::size_t>(img.vit_h) * img.vit_w * 588, 1.0f);
    buildImageBlock(img.llm_h, img.llm_w, img.block, img.perm);
    EXPECT_EQ(loaded->encode(img), direct.encode(img));
    EXPECT_EQ(loaded->sentinel(ImageSentinel::End), direct.sentinel(ImageSentinel::End));
}
