/**
 * @file Test__DeepSeekV4CPUKernels.cpp
 * @brief Device-free regressions for the DeepSeek-V4 CPU primitive bundle.
 *
 * The central invariant of the DeepSeek-V4 state machines is that the value at
 * position p depends only on tokens [0, p]: a whole-prompt prefill, any chunk
 * split and token-by-token decode must publish identical compressed pools,
 * indexer selections and attention outputs. Each test compares the ring-based
 * kernels against an independent whole-sequence formulation of DeepSeek's
 * reference (inference/model.py) and against differently chunked execution.
 */
#include "kernels/deepseek4/DeepSeekV4Kernels.h"
#include "kernels/deepseek4/DeepSeekV4Math.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace llaminar2::deepseek4;

namespace
{
std::vector<float> randomVector(std::size_t n, unsigned seed, float scale = 1.0f)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

std::unique_ptr<IDeepSeekV4Kernels> kernels() { return createCPUDeepSeekV4Kernels(); }

/** @brief Whole-sequence compressor (DeepSeek Compressor.forward at start_pos 0). */
std::vector<float> referenceCompress(const std::vector<float> &kv, const std::vector<float> &score,
                                     const std::vector<float> &ape, const std::vector<float> &gain,
                                     int seq, int ratio, int d, const RopePolicy &rope)
{
    const bool overlap = ratio == 4;
    const int width = overlap ? 2 * d : d;
    const int blocks = seq / ratio;
    std::vector<float> out(static_cast<std::size_t>(blocks) * d);
    for (int b = 0; b < blocks; ++b)
    {
        for (int c = 0; c < d; ++c)
        {
            std::vector<std::pair<float, float>> entries; // (score, kv)
            if (overlap)
                for (int e = 0; e < ratio; ++e)
                {
                    const int p = (b - 1) * ratio + e;
                    if (p >= 0) entries.push_back({score[p * width + c] + ape[e * width + c], kv[p * width + c]});
                }
            for (int e = 0; e < ratio; ++e)
            {
                const int p = b * ratio + e;
                const int ch = overlap ? d + c : c;
                entries.push_back({score[p * width + ch] + ape[e * width + ch], kv[p * width + ch]});
            }
            float m = -INFINITY;
            for (auto &e : entries) m = std::max(m, e.first);
            double sum = 0, acc = 0;
            for (auto &e : entries)
            {
                const double w = std::exp(e.first - m);
                sum += w;
                acc += w * e.second;
            }
            out[b * d + c] = static_cast<float>(acc / sum);
        }
        double ss = 0;
        for (int c = 0; c < d; ++c) ss += out[b * d + c] * out[b * d + c];
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / d) + 1e-6f);
        for (int c = 0; c < d; ++c) out[b * d + c] *= inv * gain[c];
        math::rotateTail(&out[b * d], d, rope, b * ratio, false);
    }
    return out;
}

/** @brief Run store + pool over [0, seq) split into chunks of @p chunk rows. */
std::vector<float> chunkedCompress(const std::vector<float> &kv, const std::vector<float> &score,
                                   const std::vector<float> &ape, const std::vector<float> &gain,
                                   int seq, int ratio, int d, const RopePolicy &rope, int chunk)
{
    auto k = kernels();
    CompressorGeometry g{ratio, d, chunk + 2 * ratio, seq / ratio + 1};
    const int width = g.projectionWidth();
    std::vector<float> ringKV(static_cast<std::size_t>(g.ringRows) * width);
    std::vector<float> ringScore(ringKV.size());
    std::vector<float> pool(static_cast<std::size_t>(g.poolRows) * d, 0.0f);
    for (int start = 0; start < seq; start += chunk)
    {
        const int rows = std::min(chunk, seq - start);
        std::vector<std::int32_t> pos(rows);
        for (int r = 0; r < rows; ++r) pos[r] = start + r;
        CompressorStoreArgs s{{rows, nullptr}, g, pos.data(), &kv[start * width], &score[start * width],
                              ape.data(), ringKV.data(), ringScore.data()};
        k->compressorStore(s, nullptr);
        CompressorPoolArgs p{{rows, nullptr}, g, pos.data(), ringKV.data(), ringScore.data(), gain.data(),
                             1e-6f, rope, pool.data()};
        k->compressorPool(p, nullptr);
    }
    pool.resize(static_cast<std::size_t>(seq / ratio) * d);
    return pool;
}
} // namespace

TEST(DeepSeekV4CPUKernels, SinkhornIsDoublyStochastic)
{
    auto comb = randomVector(16, 1, 2.0f);
    math::sinkhorn(comb.data(), 4, 20, 1e-6f);
    for (int j = 0; j < 4; ++j)
    {
        float row = 0, col = 0;
        for (int k = 0; k < 4; ++k) { row += comb[j * 4 + k]; col += comb[k * 4 + j]; }
        EXPECT_NEAR(row, 1.0f, 1e-3f);
        EXPECT_NEAR(col, 1.0f, 1e-5f);
    }
}

TEST(DeepSeekV4CPUKernels, HyperConnectionReadWriteMatchesDefinition)
{
    const int rows = 3, hc = 4, H = 8, mix = (2 + hc) * hc;
    auto trunk = randomVector(rows * hc * H, 2);
    auto mixes = randomVector(rows * mix, 3);
    std::vector<float> scale{0.5f, 0.7f, 1.1f};
    auto base = randomVector(mix, 4, 0.3f);
    std::vector<float> out(rows * H), post(rows * hc), comb(rows * hc * hc);
    auto k = kernels();
    HCReadArgs read{{rows, nullptr}, hc, H, false, 1e-6f, 1e-6f, 20, trunk.data(), mixes.data(), scale.data(),
                    base.data(), nullptr, 1e-6f, nullptr, out.data(), post.data(), comb.data()};
    k->hcRead(read, nullptr);
    for (int r = 0; r < rows; ++r)
    {
        double ss = 0;
        for (int i = 0; i < hc * H; ++i) ss += trunk[r * hc * H + i] * trunk[r * hc * H + i];
        const float rs = 1.0f / std::sqrt(static_cast<float>(ss / (hc * H)) + 1e-6f);
        for (int d = 0; d < H; ++d)
        {
            float y = 0;
            for (int j = 0; j < hc; ++j)
                y += (math::sigmoid(mixes[r * mix + j] * rs * scale[0] + base[j]) + 1e-6f) * trunk[(r * hc + j) * H + d];
            EXPECT_NEAR(out[r * H + d], y, 1e-5f);
        }
        for (int j = 0; j < hc; ++j)
            EXPECT_NEAR(post[r * hc + j], 2 * math::sigmoid(mixes[r * mix + hc + j] * rs * scale[1] + base[hc + j]), 1e-6f);
    }
    auto block = randomVector(rows * H, 5);
    std::vector<float> written(trunk);
    HCWriteArgs write{{rows, nullptr}, hc, H, block.data(), post.data(), comb.data(), written.data(), written.data()};
    k->hcWrite(write, nullptr); // in place
    for (int r = 0; r < rows; ++r)
        for (int kk = 0; kk < hc; ++kk)
            for (int d = 0; d < H; ++d)
            {
                float y = post[r * hc + kk] * block[r * H + d];
                for (int j = 0; j < hc; ++j) y += comb[(r * hc + j) * hc + kk] * trunk[(r * hc + j) * H + d];
                EXPECT_NEAR(written[(r * hc + kk) * H + d], y, 1e-5f);
            }
}

TEST(DeepSeekV4CPUKernels, CompressorChunkingMatchesWholeSequence)
{
    const int d = 64;
    const RopePolicy rope = RopePolicy::make(16, 160000.0f, 65536, 16.0f, 32.0f, 1.0f);
    for (int ratio : {4, 128})
    {
        const int seq = ratio == 4 ? 37 : 300;
        const int width = ratio == 4 ? 2 * d : d;
        auto kv = randomVector(seq * width, 10 + ratio);
        auto score = randomVector(seq * width, 20 + ratio);
        auto ape = randomVector(ratio * width, 30 + ratio, 0.5f);
        auto gain = randomVector(d, 40 + ratio, 0.1f);
        for (auto &g : gain) g += 1.0f;
        const auto expected = referenceCompress(kv, score, ape, gain, seq, ratio, d, rope);
        for (int chunk : {seq, 7, 1})
        {
            const auto got = chunkedCompress(kv, score, ape, gain, seq, ratio, d, rope, chunk);
            ASSERT_EQ(got.size(), expected.size());
            for (std::size_t i = 0; i < got.size(); ++i)
                ASSERT_NEAR(got[i], expected[i], 2e-5f) << "ratio " << ratio << " chunk " << chunk << " at " << i;
        }
    }
}

TEST(DeepSeekV4CPUKernels, IndexerSelectsBestVisibleBlocksWithStableTies)
{
    const int heads = 2, dim = 4, poolRows = 6, topK = 3;
    std::vector<float> pool(poolRows * dim, 0.0f);
    for (int b = 0; b < poolRows; ++b) pool[b * dim] = static_cast<float>(b % 3); // ties: b, b+3
    std::vector<float> query(heads * dim, 0.0f);
    query[0] = 1.0f;                                  // head 0 sees channel 0
    std::vector<float> weights{1.0f, 0.5f};
    std::vector<std::int32_t> pos{23};                // (23+1)/4 = 6 visible
    std::vector<std::int32_t> selected(topK);
    auto k = kernels();
    IndexerTopKArgs a{{1, nullptr}, heads, dim, 4, topK, poolRows, pos.data(), query.data(), weights.data(),
                      1.0f, pool.data(), nullptr, selected.data()};
    k->indexerTopK(a, nullptr);
    EXPECT_EQ(selected, (std::vector<std::int32_t>{1, 2, 5}));
    pos[0] = 6; // one visible block
    k->indexerTopK(a, nullptr);
    EXPECT_EQ(selected, (std::vector<std::int32_t>{0, -1, -1}));
}

TEST(DeepSeekV4CPUKernels, SparseAttentionChunkingMatchesDecode)
{
    const int heads = 2, dim = 32, window = 5, seq = 13, ratio = 4;
    const RopePolicy rope = RopePolicy::make(8, 10000.0f, 0, 1.0f, 32.0f, 1.0f);
    auto q = randomVector(seq * heads * dim, 50);
    auto kv = randomVector(seq * dim, 51);
    auto pool = randomVector((seq / ratio) * dim, 52);
    std::vector<float> sinks{0.3f, -0.4f};
    auto k = kernels();
    auto run = [&](int chunk) {
        std::vector<float> ring(static_cast<std::size_t>(window + chunk) * dim);
        std::vector<float> out(q.size());
        for (int start = 0; start < seq; start += chunk)
        {
            const int rows = std::min(chunk, seq - start);
            std::vector<std::int32_t> pos(rows);
            for (int r = 0; r < rows; ++r) pos[r] = start + r;
            RawStoreArgs s{{rows, nullptr}, dim, window + chunk, pos.data(), &kv[start * dim], ring.data()};
            k->rawStore(s, nullptr);
            SparseAttentionArgs a{{rows, nullptr}, heads, dim, window, window + chunk, ratio, 0,
                                  CompressedAccess::All, 1.0f / std::sqrt(float(dim)), rope, pos.data(),
                                  &q[start * heads * dim], ring.data(), pool.data(), nullptr, sinks.data(),
                                  &out[start * heads * dim]};
            k->sparseAttention(a, nullptr);
        }
        return out;
    };
    const auto whole = run(seq);
    const auto decode = run(1);
    for (std::size_t i = 0; i < whole.size(); ++i) ASSERT_NEAR(whole[i], decode[i], 1e-6f);
}

TEST(DeepSeekV4CPUKernels, GroupedProjectionQ8MatchesDequantized)
{
    const int rows = 2, groups = 2, rank = 3, width = 64;
    auto x = randomVector(rows * groups * width, 60);
    std::vector<std::uint8_t> q8(static_cast<std::size_t>(groups * rank) * (width / 32) * 34);
    std::vector<float> dequant(static_cast<std::size_t>(groups * rank) * width);
    std::mt19937 rng(61);
    for (int o = 0; o < groups * rank; ++o)
        for (int b = 0; b < width / 32; ++b)
        {
            std::uint8_t *blk = &q8[(static_cast<std::size_t>(o) * (width / 32) + b) * 34];
            const std::uint16_t half = 0x2C00; // 0.0625
            std::memcpy(blk, &half, 2);
            for (int i = 0; i < 32; ++i)
            {
                const auto v = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127);
                blk[2 + i] = static_cast<std::uint8_t>(v);
                dequant[static_cast<std::size_t>(o) * width + b * 32 + i] = 0.0625f * v;
            }
        }
    std::vector<float> a(rows * groups * rank), b(a.size());
    auto k = kernels();
    k->groupedProjection({{rows, nullptr}, groups, rank, width, WeightFormat::Q8_0, q8.data(), x.data(), a.data()}, nullptr);
    k->groupedProjection({{rows, nullptr}, groups, rank, width, WeightFormat::FP32, dequant.data(), x.data(), b.data()}, nullptr);
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 1e-4f);
}

TEST(DeepSeekV4CPUKernels, RouteScoreAndHash)
{
    const int experts = 5, topK = 2;
    std::vector<float> logits{0.1f, 2.0f, -1.0f, 1.5f, 0.0f};
    std::vector<float> bias{0.0f, 0.0f, 10.0f, 0.0f, 0.0f};
    std::vector<std::int32_t> ids(topK);
    std::vector<float> w(topK);
    auto k = kernels();
    k->route({{1, nullptr}, experts, topK, 1.5f, logits.data(), bias.data(), nullptr, 0, nullptr, ids.data(), w.data()}, nullptr);
    EXPECT_EQ(ids, (std::vector<std::int32_t>{2, 1}));
    const float s2 = math::sqrtSoftplus(-1.0f), s1 = math::sqrtSoftplus(2.0f);
    EXPECT_NEAR(w[0], s2 / (s1 + s2) * 1.5f, 1e-6f);
    std::vector<std::int32_t> table{4, 0, 3, 1}, tokens{1};
    k->route({{1, nullptr}, experts, topK, 1.0f, logits.data(), nullptr, table.data(), 2, tokens.data(), ids.data(), w.data()}, nullptr);
    EXPECT_EQ(ids, (std::vector<std::int32_t>{3, 1}));
    EXPECT_NEAR(w[0] + w[1], 1.0f, 1e-6f);
}
