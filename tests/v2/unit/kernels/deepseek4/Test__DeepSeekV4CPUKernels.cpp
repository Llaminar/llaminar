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
 * The SwiGLU-clamp regressions prove that every CPU clamp entry point equals
 * the unchanged SwiGLU program applied to reference-clamped operands.
 */
#include "kernels/deepseek4/DeepSeekV4Kernels.h"
#include "kernels/deepseek4/DeepSeekV4Math.h"
#include "kernels/deepseek4/DeepSeekV4KVFormat.h"
#include "models/deepseek4/DeepSeekV4BufferLayout.h"
#include "models/deepseek4/DeepSeekV4Config.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/DeviceSwiGLUNumericalContract.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "tensors/TensorClasses.h"

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

/** @brief Decode @p blocks packed records of @p entryBytes into FP32 rows. */
std::vector<float> decodePool(const std::vector<std::uint8_t> &pool, CompressedStream stream, int d, int ropeDim,
                              int blocks)
{
    std::vector<float> out(static_cast<std::size_t>(blocks) * d);
    const CompressorGeometry g{4, d, blocks, stream, ropeDim};
    for (int b = 0; b < blocks; ++b)
        for (int c = 0; c < d; ++c)
        {
            const std::uint8_t *entry = &pool[static_cast<std::size_t>(b) * g.entryBytes()];
            out[static_cast<std::size_t>(b) * d + c] = stream == CompressedStream::Attention
                ? qat::decodeKVChannel({d, ropeDim}, entry, c) : qat::decodeIndexerChannel({d}, entry, c);
        }
    return out;
}

/**
 * @brief Run pool + store over [0, seq) split into chunks of @p chunk rows.
 * @details Mirrors the graph order: the pool reads the chunk scratch and the
 * carried state, then the store carries the chunk's last 2*ratio rows.
 * @return Packed pool bytes for every completed block.
 */
std::vector<std::uint8_t> chunkedCompress(const std::vector<float> &kv, const std::vector<float> &score,
                                          const std::vector<float> &ape, const std::vector<float> &gain, int seq,
                                          int ratio, int d, const RopePolicy &rope, int chunk,
                                          CompressedStream stream = CompressedStream::Attention)
{
    auto k = kernels();
    CompressorGeometry g{ratio, d, seq / ratio + 1, stream, stream == CompressedStream::Attention ? rope.ropeDim : 0};
    const int width = g.projectionWidth();
    std::vector<float> stateKV(static_cast<std::size_t>(g.stateRows()) * width, NAN);
    std::vector<float> stateScore(stateKV.size(), NAN);
    std::vector<std::uint8_t> pool(static_cast<std::size_t>(g.poolRows) * g.entryBytes(), 0xEE);
    for (int start = 0; start < seq; start += chunk)
    {
        const int rows = std::min(chunk, seq - start);
        std::vector<std::int32_t> pos(rows);
        for (int r = 0; r < rows; ++r) pos[r] = start + r;
        CompressorPoolArgs p{{rows, nullptr}, g, pos.data(), &kv[start * width], &score[start * width], ape.data(),
                             stateKV.data(), stateScore.data(), gain.data(), 1e-6f, rope, pool.data()};
        k->compressorPool(p, nullptr);
        CompressorStoreArgs s{{rows, nullptr}, g, pos.data(), &kv[start * width], &score[start * width],
                              ape.data(), stateKV.data(), stateScore.data()};
        k->compressorStore(s, nullptr);
    }
    pool.resize(static_cast<std::size_t>(seq / ratio) * g.entryBytes());
    return pool;
}

/** @brief Largest magnitude of a row (QAT error scale). */
float rowMax(const float *row, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(row[i]));
    return m;
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
    // Blocks straddle chunk boundaries for every chunk size below, including
    // sizes that are not multiples of 4 or 128; the carried 2*ratio state must
    // reproduce the whole-prompt pool byte-for-byte.
    const int d = 128;
    const RopePolicy rope = RopePolicy::make(64, 160000.0f, 65536, 16.0f, 32.0f, 1.0f);
    for (int ratio : {4, 128})
        for (auto stream : {CompressedStream::Attention, CompressedStream::Indexer})
        {
            if (ratio == 128 && stream == CompressedStream::Indexer) continue;
            const int seq = ratio == 4 ? 37 : 300;
            const int width = ratio == 4 ? 2 * d : d;
            auto kv = randomVector(seq * width, 10 + ratio);
            auto score = randomVector(seq * width, 20 + ratio);
            auto ape = randomVector(ratio * width, 30 + ratio, 0.5f);
            auto gain = randomVector(d, 40 + ratio, 0.1f);
            for (auto &g : gain) g += 1.0f;
            const auto whole = chunkedCompress(kv, score, ape, gain, seq, ratio, d, rope, seq, stream);
            const std::vector<int> chunks = ratio == 4 ? std::vector<int>{1, 3, 5, 6, 7, 9}
                                                       : std::vector<int>{1, 7, 100, 129, 200};
            for (int chunk : chunks)
                ASSERT_EQ(chunkedCompress(kv, score, ape, gain, seq, ratio, d, rope, chunk, stream), whole)
                    << "ratio " << ratio << " chunk " << chunk;
            // Against the independent whole-sequence formulation: exactly its QAT
            // rounding up to one quantization step where the FP32 pooled value
            // sits on a rounding boundary.
            auto expected = referenceCompress(kv, score, ape, gain, seq, ratio, d, rope);
            const int blocks = seq / ratio;
            for (int b = 0; b < blocks; ++b)
            {
                float *row = &expected[static_cast<std::size_t>(b) * d];
                if (stream == CompressedStream::Indexer) qat::hadamard(row, d);
                qat::apply(stream == CompressedStream::Attention ? ActivationQuant::FP8NopeBF16Rope
                                                                 : ActivationQuant::None, row, d, rope.ropeDim);
            }
            const auto got = decodePool(whole, stream, d, rope.ropeDim, blocks);
            for (int b = 0; b < blocks; ++b)
            {
                const float *want = &expected[static_cast<std::size_t>(b) * d];
                const float scale = rowMax(want, d);
                const float relative = stream == CompressedStream::Attention ? 0.0625f : 0.34f;
                const float absolute = stream == CompressedStream::Attention ? scale / 224.0f : scale / 6.0f;
                for (int c = 0; c < d; ++c)
                    ASSERT_NEAR(got[static_cast<std::size_t>(b) * d + c], want[c],
                                relative * std::fabs(want[c]) + absolute)
                        << "ratio " << ratio << " block " << b << " channel " << c;
            }
        }
}

TEST(DeepSeekV4CPUKernels, IndexerSelectsBestVisibleBlocksWithStableTies)
{
    const int heads = 2, dim = 32, poolRows = 6, topK = 3;
    const PackedIndexerKeyFormat format{dim};
    std::vector<std::uint8_t> pool(static_cast<std::size_t>(poolRows) * format.entryBytes());
    for (int b = 0; b < poolRows; ++b)
    {
        std::vector<float> key(dim, 0.0f);
        key[0] = static_cast<float>(b % 3); // ties: b, b+3 (exact in E2M1)
        qat::packIndexerKey(format, key.data(), &pool[static_cast<std::size_t>(b) * format.entryBytes()]);
    }
    std::vector<float> query(heads * dim, 0.0f);
    query[0] = 1.0f;                                  // head 0 sees channel 0
    std::vector<float> weights{1.0f, 0.5f};
    std::vector<std::int32_t> pos{23};                // (23+1)/4 = 6 visible
    std::vector<std::int32_t> selected(topK);
    auto k = kernels();
    IndexerTopKArgs a{{1, nullptr}, heads, format, 4, topK, poolRows, 1, pos.data(), query.data(), weights.data(),
                      1.0f, pool.data(), nullptr, selected.data()};
    k->indexerTopK(a, nullptr);
    EXPECT_EQ(selected, (std::vector<std::int32_t>{1, 2, 5}));
    pos[0] = 6; // one visible block
    k->indexerTopK(a, nullptr);
    EXPECT_EQ(selected, (std::vector<std::int32_t>{0, -1, -1}));
}

TEST(DeepSeekV4CPUKernels, IndexerScoreScratchIsBoundedAndTotal)
{
    // 1M context: 262144 CSA pool rows of FP32 scores are 1 MiB per row, so a
    // 4096-row chunk scores in passes of 64 rows (64 MiB) instead of 4 GiB.
    EXPECT_EQ(llaminar2::deepSeekV4IndexerScoreRows(4096, 1 << 20), 64);
    EXPECT_EQ(llaminar2::deepSeekV4IndexerScoreRows(4096, 32768), 2048);
    EXPECT_EQ(llaminar2::deepSeekV4IndexerScoreRows(16, 32768), 16);
    EXPECT_EQ(llaminar2::deepSeekV4IndexerScoreRows(4096, 1 << 30), 1); // never zero rows
    for (int rows : {1, 7, 64, 4096})
        for (int context : {16, 32768, 1 << 20, 1 << 24})
        {
            const int passRows = llaminar2::deepSeekV4IndexerScoreRows(rows, context);
            const std::size_t poolRows = (static_cast<std::size_t>(context) + 3) / 4;
            ASSERT_GE(passRows, 1);
            ASSERT_LE(passRows, rows);
            ASSERT_TRUE(passRows == 1 ||
                        passRows * poolRows * sizeof(float) <= llaminar2::kDeepSeekV4IndexerScoreBudgetBytes);
        }
}

TEST(DeepSeekV4CPUKernels, SparseAttentionChunkingMatchesDecode)
{
    // Window keys come from the chunk's QAT kv when in the chunk, else from the
    // packed window ring; packing is lossless for QAT values, so any chunking
    // reproduces whole-prompt attention bit-for-bit.
    const int heads = 2, dim = 128, rope_dim = 64, window = 5, seq = 13, ratio = 4;
    const PackedKVEntryFormat format{dim, rope_dim};
    const RopePolicy rope = RopePolicy::make(rope_dim, 10000.0f, 0, 1.0f, 32.0f, 1.0f);
    auto q = randomVector(seq * heads * dim, 50);
    auto kv = randomVector(seq * dim, 51);
    for (int p = 0; p < seq; ++p) qat::apply(ActivationQuant::FP8NopeBF16Rope, &kv[p * dim], dim, rope_dim);
    std::vector<std::uint8_t> pool(static_cast<std::size_t>(seq / ratio) * format.entryBytes());
    auto poolValues = randomVector((seq / ratio) * dim, 52);
    for (int b = 0; b < seq / ratio; ++b)
        qat::packKVEntry(format, &poolValues[b * dim], &pool[static_cast<std::size_t>(b) * format.entryBytes()]);
    std::vector<float> sinks{0.3f, -0.4f};
    auto k = kernels();
    auto run = [&](int chunk) {
        std::vector<std::uint8_t> ring(static_cast<std::size_t>(window) * format.entryBytes(), 0xEE);
        std::vector<float> out(q.size());
        for (int start = 0; start < seq; start += chunk)
        {
            const int rows = std::min(chunk, seq - start);
            std::vector<std::int32_t> pos(rows);
            for (int r = 0; r < rows; ++r) pos[r] = start + r;
            SparseAttentionArgs a{{rows, nullptr}, heads, format, window, ratio, 0, CompressedAccess::All,
                                  1.0f / std::sqrt(float(dim)), rope, pos.data(), &q[start * heads * dim],
                                  &kv[start * dim], ring.data(), pool.data(), nullptr, sinks.data(),
                                  &out[start * heads * dim]};
            k->sparseAttention(a, nullptr);
            RawStoreArgs s{{rows, nullptr}, format, window, pos.data(), &kv[start * dim], ring.data()};
            k->rawStore(s, nullptr);
        }
        return out;
    };
    const auto whole = run(seq);
    for (int chunk : {1, 2, 3, 6})
    {
        const auto chunked = run(chunk);
        for (std::size_t i = 0; i < whole.size(); ++i) ASSERT_EQ(whole[i], chunked[i]) << "chunk " << chunk;
    }
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

// ---------------------------------------------------------------------------
// SwiGLU operand clamp (DeepSeek-V4 `swiglu_clamp_exp` / `swiglu_clamp_shexp`)
// ---------------------------------------------------------------------------

namespace
{
/** @brief DeepSeek inference/model.py Expert.forward: clamp(up,±L), clamp(gate,max=L). */
void referenceClamp(float &gate, float &up, float limit)
{
    up = std::min(std::max(up, -limit), limit);
    gate = std::min(gate, limit);
}

std::uint32_t bits(float value)
{
    std::uint32_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}
} // namespace

TEST(DeepSeekV4SwiGLUClamp, ContractClampsOperandsAndKeepsUnclampedBytes)
{
    namespace contract = llaminar2::device_swiglu_contract;
    const auto gate = randomVector(4096, 71, 12.0f);
    const auto up = randomVector(4096, 72, 12.0f);
    int bound = 0;
    for (std::size_t i = 0; i < gate.size(); ++i)
    {
        // limit <= 0 is byte-identical to the ordinary contract.
        EXPECT_EQ(bits(contract::swigluValue(gate[i], up[i], 0.0f)), bits(contract::swigluValue(gate[i], up[i])));
        float g = gate[i], u = up[i];
        referenceClamp(g, u, 10.0f);
        bound += (g != gate[i]) || (u != up[i]);
        EXPECT_EQ(bits(contract::swigluValue(gate[i], up[i], 10.0f)), bits(contract::swigluValue(g, u)));
    }
    EXPECT_GT(bound, 100) << "fixture must exercise the clamp";
    // NaN propagates instead of being clamped to a finite value.
    EXPECT_TRUE(std::isnan(contract::swigluValue(std::nanf(""), 1.0f, 10.0f)));
}

TEST(DeepSeekV4SwiGLUClamp, CPUMoEKernelClampAndActivation)
{
    constexpr int n = 777; // ragged: not a vector multiple
    const auto gate = randomVector(n, 81, 12.0f);
    const auto up = randomVector(n, 82, 12.0f);
    auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(llaminar2::DeviceId::cpu());
    ASSERT_NE(kernel, nullptr);

    llaminar2::FP32Tensor gate_tensor({static_cast<std::size_t>(n)});
    llaminar2::FP32Tensor up_tensor({static_cast<std::size_t>(n)});
    std::memcpy(gate_tensor.mutable_data(), gate.data(), n * sizeof(float));
    std::memcpy(up_tensor.mutable_data(), up.data(), n * sizeof(float));
    ASSERT_TRUE(kernel->clampSwiGLUOperands(&gate_tensor, &up_tensor, n, 10.0f));
    for (int i = 0; i < n; ++i)
    {
        float g = gate[i], u = up[i];
        referenceClamp(g, u, 10.0f);
        EXPECT_EQ(bits(gate_tensor.data()[i]), bits(g));
        EXPECT_EQ(bits(up_tensor.data()[i]), bits(u));
    }
    // A non-positive limit never touches the operands.
    ASSERT_TRUE(kernel->clampSwiGLUOperands(&gate_tensor, &up_tensor, n, 0.0f));

    std::vector<float> activated = gate;
    kernel->setSwiGLULimit(10.0f);
    kernel->swiGLU(activated.data(), up.data(), n);
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(bits(activated[i]),
                  bits(llaminar2::device_swiglu_contract::swigluValue(gate[i], up[i], 10.0f)));
}

TEST(DeepSeekV4SwiGLUClamp, GroupedQ8PublicationEqualsPreClampedOperands)
{
    constexpr int rows = 3, k = 128, blocks = k / 32;
    const auto gate = randomVector(rows * k, 91, 12.0f);
    const auto up = randomVector(rows * k, 92, 12.0f);
    std::vector<float> gate_clamped = gate, up_clamped = up;
    for (int i = 0; i < rows * k; ++i)
        referenceClamp(gate_clamped[i], up_clamped[i], 10.0f);

    for (const auto policy : {llaminar2::CPUProjectionNumericalPolicy::BackendNative,
                              llaminar2::CPUProjectionNumericalPolicy::GPUAlignedExpert})
    {
        std::vector<llaminar2::Q8_1Block> fused(rows * blocks), expected(rows * blocks);
        llaminar2::cpu::native_vnni::swiglu_quantize_activations_to_q8_1(gate.data(), up.data(), fused.data(),
                                                       rows, k, blocks, policy, 10.0f);
        llaminar2::cpu::native_vnni::swiglu_quantize_activations_to_q8_1(gate_clamped.data(), up_clamped.data(), expected.data(),
                                                       rows, k, blocks, policy, 0.0f);
        EXPECT_EQ(std::memcmp(fused.data(), expected.data(), fused.size() * sizeof(llaminar2::Q8_1Block)), 0);
    }
}

// ============================================================================
// Official QAT rounding and packed persistent records (DeepSeekV4KVFormat.h)
// ============================================================================
namespace
{
/** @brief (block amax, probe value, expected QAT value) from kernel.py semantics. */
struct QATCase
{
    float amax, value, expected;
};

/**
 * Expected values were produced by python/reference/deepseek4.py fp8_qat /
 * fp4_qat (torch float8_e4m3fn cast, FP32 scale arithmetic): scale ties,
 * RNE mantissa ties, subnormals, the 1e-4 / 6*2^-126 amax floors, and the
 * FP32 rounding of 1/448 and 1/6 (6 * fl(1/6) rounds to exactly 1).
 */
constexpr QATCase kFP8Cases[] = {
    {0x1p+0f, 0x1.333334p-2f, 0x1.4p-2f},          // 0.3 -> 0.3125
    {0x1p+0f, -0x1.333334p-2f, -0x1.4p-2f},
    {0x1p+0f, 0x1.1p-4f, 0x1p-4f},                 // scaled 17: tie -> even mantissa 16
    {0x1p+0f, 0x1.3p-4f, 0x1.4p-4f},               // scaled 19: tie -> 20
    {0x1p+0f, 0x1.8p-17f, 0x1p-16f},               // subnormal 1.5 * 2^-9: tie -> 2 * 2^-9
    {0x1p+0f, 0x1p-18f, 0.0f},                     // subnormal 0.5 * 2^-9: tie -> 0
    {0x1p+0f, 0x1.2cp+0f, 0x1.2p+0f},              // 1.171875 -> 1.125
    {0x1p+0f, 0x1p+0f, 0x1p+0f},
    {0x1.cp+0f, 0x1p+0f, 0x1p+0f},
    {0x1.cp+0f, 0x1.f7cedap-4f, 0x1p-3f},          // mantissa carry into the next binade
    {0x1.cp+8f, 0x1.a66666p+1f, 0x1.ap+1f},        // amax 448
    {0x1.cp+8f, 0x1.cp+8f, 0x1.cp+8f},
    {0.0f, 0.0f, 0.0f},
    {0x1.a36e2ep-15f, 0x1.f75104p-16f, 0x1p-15f},  // amax below the 1e-4 floor
    {0x1.93e594p+99f, 0x1.9d971ep+89f, 0x1.ap+89f},
    {0x1.0624dep-10f, -0x1.6f0068p-11f, -0x1.6p-11f},
};

constexpr QATCase kFP4Cases[] = {
    {0x1.8p+2f, 0x1p+0f, 0x1p+0f},
    {0x1.8p+2f, 0x1.4p+2f, 0x1p+2f},               // 5 -> tie -> 4
    {0x1.8p+2f, 0x1p-1f, 0x1p-1f},                 // scale is exactly 1, not 2
    {0x1.8p+2f, 0x1.8p+2f, 0x1.8p+2f},
    {0x1.8p+1f, 0x1.4p+1f, 0x1p+1f},               // 2.5 -> 2
    {0x1.8p+1f, -0x1.4p+0f, -0x1p+0f},             // -1.25 -> -1
    {0x1.8p+1f, 0x1.8p-1f, 0x1.8p-1f},
    {0x1.8p+1f, 0x1.cp+0f, 0x1p+1f},               // 1.75 -> 2
    {0x1.0624dep-10f, 0x1.a36e2ep-12f, 0x1.8p-12f},
    {0.0f, 0.0f, 0.0f},
    {0x1.8p+3f, -0x1.cp+2f, -0x1p+3f},
};
} // namespace

TEST(DeepSeekV4QAT, FP8BlockRoundingMatchesOfficialHandValues)
{
    for (const auto &c : kFP8Cases)
    {
        std::vector<float> block(kFP8BlockSize, 0.0f);
        block[0] = c.amax;
        block[1] = c.value;
        qat::fakeQuantFP8Block(block.data(), kFP8BlockSize);
        EXPECT_EQ(block[1], c.expected) << "amax " << c.amax << " value " << c.value;
    }
}

TEST(DeepSeekV4QAT, FP4BlockRoundingMatchesOfficialHandValues)
{
    for (const auto &c : kFP4Cases)
    {
        std::vector<float> block(kFP4BlockSize, 0.0f);
        block[0] = c.amax;
        block[1] = c.value;
        qat::fakeQuantFP4Block(block.data(), kFP4BlockSize);
        EXPECT_EQ(block[1], c.expected) << "amax " << c.amax << " value " << c.value;
    }
}

TEST(DeepSeekV4QAT, CodecsAreExhaustivelyConsistent)
{
    // Every finite E4M3 code decodes to a value that re-encodes to itself, and
    // decoding is strictly monotonic over the positive codes.
    for (int code = 0; code < 256; ++code)
    {
        if ((code & 0x7F) == 0x7F) continue; // NaN encodings
        const float v = qat::decodeE4M3(static_cast<std::uint8_t>(code));
        if (code == 0x80) { EXPECT_TRUE(std::signbit(v) && v == 0.0f); continue; }
        EXPECT_EQ(qat::encodeE4M3(v), code) << code;
        if (code > 0 && code < 0x7F) EXPECT_GT(v, qat::decodeE4M3(static_cast<std::uint8_t>(code - 1)));
    }
    EXPECT_EQ(qat::decodeE4M3(0x7E), 448.0f);
    EXPECT_EQ(qat::decodeE4M3(0x01), 0x1p-9f);
    EXPECT_EQ(qat::encodeE4M3(1000.0f), 0x7E); // satfinite
    const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int code = 0; code < 16; ++code)
    {
        const float v = qat::decodeE2M1(static_cast<std::uint8_t>(code));
        EXPECT_EQ(std::fabs(v), e2m1[code & 7]);
        if (code != 8) EXPECT_EQ(qat::encodeE2M1(v), code);
    }
    EXPECT_EQ(qat::decodeBF16(qat::encodeBF16(1.00390625f)), 1.0f);     // tie -> even
    EXPECT_EQ(qat::decodeBF16(qat::encodeBF16(1.01171875f)), 1.015625f); // tie -> even (up)
}

TEST(DeepSeekV4QAT, HadamardIsSylvesterOrderedAndOrthonormal)
{
    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    qat::hadamard(v, 4);
    EXPECT_EQ(v[0], 5.0f);
    EXPECT_EQ(v[1], -1.0f);
    EXPECT_EQ(v[2], -2.0f);
    EXPECT_EQ(v[3], 0.0f);
    auto x = randomVector(128, 70);
    auto y = x;
    qat::hadamard(y.data(), 128);
    qat::hadamard(y.data(), 128);
    for (int i = 0; i < 128; ++i) EXPECT_NEAR(y[i], x[i], 1e-5f);
}

TEST(DeepSeekV4QAT, PackedRecordsUseTheReleaseLayoutAndRoundTripExactly)
{
    const PackedKVEntryFormat kv{512, 64};
    const PackedIndexerKeyFormat idx{128};
    ASSERT_EQ(kv.entryBytes(), 584);
    ASSERT_EQ(kv.ropeOffset(), 448);
    ASSERT_EQ(kv.scaleOffset(), 576);
    ASSERT_EQ(kv.scaleCount(), 7);
    ASSERT_EQ(idx.entryBytes(), 68);
    ASSERT_EQ(idx.scaleOffset(), 64);
    ASSERT_EQ((PackedKVEntryFormat{256, 64}.entryBytes()), 328); // tiny model geometry

    for (unsigned seed = 0; seed < 8; ++seed)
    {
        // Mixed magnitudes per block exercise subnormal codes and scale ties.
        auto raw = randomVector(512, 80 + seed, seed % 2 ? 3.0f : 1e-3f);
        raw[seed] = 448.0f * (seed + 1);
        std::vector<std::uint8_t> entry(kv.entryBytes(), 0xEE);
        qat::packKVEntry(kv, raw.data(), entry.data());
        EXPECT_EQ(entry[583], 0); // pad
        auto qatValues = raw;
        qat::apply(ActivationQuant::FP8NopeBF16Rope, qatValues.data(), 512, 64);
        for (int c = 0; c < 512; ++c) ASSERT_EQ(qat::decodeKVChannel(kv, entry.data(), c), qatValues[c]) << c;
        // Packing a QAT vector is lossless (idempotent values).
        std::vector<std::uint8_t> again(kv.entryBytes());
        qat::packKVEntry(kv, qatValues.data(), again.data());
        for (int c = 0; c < 512; ++c) ASSERT_EQ(qat::decodeKVChannel(kv, again.data(), c), qatValues[c]) << c;

        auto key = randomVector(128, 90 + seed, seed % 2 ? 0.7f : 1e-30f);
        qat::hadamard(key.data(), 128);
        std::vector<std::uint8_t> packed(idx.entryBytes(), 0xEE);
        qat::packIndexerKey(idx, key.data(), packed.data());
        auto keyQat = key;
        for (int b = 0; b < 128; b += kFP4BlockSize) qat::fakeQuantFP4Block(&keyQat[b], kFP4BlockSize);
        for (int c = 0; c < 128; ++c) ASSERT_EQ(qat::decodeIndexerChannel(idx, packed.data(), c), keyQat[c]) << c;
        qat::packIndexerKey(idx, keyQat.data(), packed.data());
        for (int c = 0; c < 128; ++c) ASSERT_EQ(qat::decodeIndexerChannel(idx, packed.data(), c), keyQat[c]) << c;

        // The exact-block indexer dot equals the dot of decoded values.
        auto q = randomVector(128, 100 + seed);
        qat::apply(ActivationQuant::HadamardFP4, q.data(), 128, 0);
        double reference = 0.0;
        for (int c = 0; c < 128; ++c) reference += static_cast<double>(q[c]) * keyQat[c];
        EXPECT_NEAR(qat::indexerDot(idx, q.data(), packed.data()), reference, 1e-6 * (1.0 + std::fabs(reference)));
    }
}

TEST(DeepSeekV4QAT, ScaleBytesFollowTheRecordLayout)
{
    const PackedKVEntryFormat kv{512, 64};
    std::vector<float> v(512, 0.0f);
    v[0] = 1.0f;        // block 0: scale 2^-8 -> UE8M0 119
    v[64] = 448.0f;     // block 1: scale 2^0  -> 127
    v[448] = 1.5f;      // first rope channel, BF16 0x3FC0
    std::vector<std::uint8_t> entry(kv.entryBytes());
    qat::packKVEntry(kv, v.data(), entry.data());
    EXPECT_EQ(entry[576], 119);
    EXPECT_EQ(entry[577], 127);
    EXPECT_EQ(entry[0], 0x78); // 256 = 2^8: exponent 15, mantissa 0
    EXPECT_EQ(entry[64], 0x7E);
    EXPECT_EQ(entry[448], 0xC0);
    EXPECT_EQ(entry[449], 0x3F);
}

// ============================================================================
// Admission: one inventory prices exactly what the arena allocates
// ============================================================================
TEST(DeepSeekV4BufferLayout, ExtensionBytesEqualResolvedDeclarations)
{
    using namespace llaminar2;
    DeepSeekV4MemoryGeometry g{43, 4096, 4, 64, 512, 1024, 8, 1024, 128, 64, 128, 512, 256, 6, 21, 20, 64};
    for (int rows : {1, 512, 4096})
        for (int context : {32768, 1 << 20})
        {
            GraphResolverConfig resolver;
            resolver.seq_len = rows;
            resolver.d_model = static_cast<int>(g.hidden);
            applyDeepSeekV4BufferGeometry(resolver, g, rows, context);
            std::size_t resolved = 0;
            for (const auto &entry : deepSeekV4BufferDeclarations())
            {
                std::size_t elements = 1;
                for (const auto &axis : entry.spec.shape)
                    elements *= BufferAllocator::evaluateFormula(axis, resolver);
                resolved += elements * 4;
            }
            const std::size_t trunk = 2ull * rows * (g.hc - 1) * g.hidden * sizeof(float);
            EXPECT_EQ(deepSeekV4ArenaExtensionBytes(g, rows, context), resolved + trunk)
                << "rows " << rows << " context " << context;
        }
    // Context-proportional state: 21 * (584/4 + 68/4) + 20 * 584/128 bytes per token.
    const double perToken = static_cast<double>(deepSeekV4PersistentPoolBytes(g, 1 << 20)) / (1 << 20);
    EXPECT_NEAR(perToken, 21.0 * (584.0 / 4 + 68.0 / 4) + 20.0 * 584.0 / 128, 0.5);
    // At 1M the scratch bound holds (64 rows x 262144 x 4 B) instead of 4 GiB.
    GraphResolverConfig resolver;
    applyDeepSeekV4BufferGeometry(resolver, g, 4096, 1 << 20);
    EXPECT_EQ(resolver.custom_formulas.at("dsv4_idx_score_rows") * resolver.custom_formulas.at("dsv4_csa_pool_rows") * 4,
              kDeepSeekV4IndexerScoreBudgetBytes);
}
