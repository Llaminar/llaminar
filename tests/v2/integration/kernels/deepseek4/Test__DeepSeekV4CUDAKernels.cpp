/**
 * @file Test__DeepSeekV4CUDAKernels.cpp
 * @brief CUDA replay of the DeepSeek-V4 compact-KV kernels against the CPU bundle.
 *
 * Recorded inputs are replayed through the production CUDA kernels on an
 * explicit non-default stream and compared with the CPU implementation:
 *  - QAT post-operations (FP8 nope + BF16 rope, Hadamard + FP4) and the raw
 *    window packer are byte-exact across backends;
 *  - the compressor's packed pool is identical for every prefill chunking
 *    (blocks straddling chunks read the carried 2*ratio state);
 *  - indexer scores are bit-identical to the CPU bundle (exact FP4 block dots,
 *    fused head accumulation), so selections match exactly, and the bounded
 *    score scratch (passes of scoreRows rows) selects exactly what one
 *    unbounded pass selects;
 *  - sparse attention over chunk kv + packed ring/pool matches the CPU result
 *    and is chunking-invariant;
 *  - the frontier-only (model_state) CUDA cache publishes no K/V conversion
 *    scratch and advances its device frontier exactly once per forward, at
 *    the terminal layer, with device-owned real row counts.
 * Reset and repeated replay are covered by rerunning the same buffers.
 */
#include "kernels/deepseek4/DeepSeekV4Kernels.h"
#include "kernels/deepseek4/DeepSeekV4KVFormat.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IKVCache.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/KernelFactory.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace llaminar2::deepseek4;

namespace
{
#define CUDA_OK(expr) ASSERT_EQ((expr), cudaSuccess) << #expr

std::vector<float> randomVector(std::size_t n, unsigned seed, float scale = 1.0f)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

/** @brief RAII device buffer. */
template <typename T> struct DeviceBuffer
{
    T *data = nullptr;
    std::size_t count = 0;
    explicit DeviceBuffer(std::size_t n) : count(n)
    {
        if (cudaMalloc(reinterpret_cast<void **>(&data), std::max<std::size_t>(1, n) * sizeof(T)) != cudaSuccess)
            data = nullptr;
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

class DeepSeekV4CUDAKernels : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_FAIL() << "CUDA integration suite requires a CUDA device";
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
        cpu_ = createCPUDeepSeekV4Kernels();
        gpu_ = createCUDADeepSeekV4Kernels();
    }
    void TearDown() override
    {
        if (stream_) cudaStreamDestroy(stream_);
    }

    template <typename T> T *upload(DeviceBuffer<T> &buffer, const std::vector<T> &host)
    {
        EXPECT_EQ(cudaMemcpyAsync(buffer.data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return buffer.data;
    }
    template <typename T> std::vector<T> download(const DeviceBuffer<T> &buffer)
    {
        std::vector<T> host(buffer.count);
        EXPECT_EQ(cudaMemcpyAsync(host.data(), buffer.data, host.size() * sizeof(T), cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        return host;
    }

    cudaStream_t stream_ = nullptr;
    std::unique_ptr<IDeepSeekV4Kernels> cpu_, gpu_;
};
} // namespace

TEST_F(DeepSeekV4CUDAKernels, QATPostOperationsAreByteExact)
{
    // Positions 0 make the rotation the identity on both backends, isolating QAT.
    const int rows = 5;
    const RopePolicy rope = RopePolicy::make(64, 10000.0f, 0, 1.0f, 32.0f, 1.0f);
    std::vector<std::int32_t> pos(rows, 0);
    DeviceBuffer<std::int32_t> dPos(rows);
    upload(dPos, pos);
    for (auto quant : {ActivationQuant::FP8NopeBF16Rope, ActivationQuant::HadamardFP4})
    {
        const int heads = quant == ActivationQuant::HadamardFP4 ? 64 : 1;
        const int dim = quant == ActivationQuant::HadamardFP4 ? 128 : 512;
        auto data = randomVector(static_cast<std::size_t>(rows) * heads * dim, 7, 2.0f);
        data[3] = 1e-30f;
        auto cpu = data;
        HeadNormRopeArgs a{{rows, nullptr}, heads, dim, false, 1e-6f, nullptr, rope, false, quant, pos.data(), cpu.data()};
        cpu_->headNormRope(a, nullptr);
        DeviceBuffer<float> dData(data.size());
        a.positions = dPos.data;
        a.data = upload(dData, data);
        gpu_->headNormRope(a, stream_);
        const auto gpu = download(dData);
        for (std::size_t i = 0; i < gpu.size(); ++i) ASSERT_EQ(gpu[i], cpu[i]) << static_cast<int>(quant) << " at " << i;
    }
}

TEST_F(DeepSeekV4CUDAKernels, RawStorePacksByteExactAndKeepsTheLastWindow)
{
    const int rows = 9, window = 4;
    const PackedKVEntryFormat format{512, 64};
    auto kv = randomVector(static_cast<std::size_t>(rows) * 512, 11, 3.0f);
    std::vector<std::int32_t> pos(rows);
    for (int r = 0; r < rows; ++r) pos[r] = 17 + r;
    std::vector<std::uint8_t> cpuRing(static_cast<std::size_t>(window) * format.entryBytes(), 0xEE);
    RawStoreArgs a{{rows, nullptr}, format, window, pos.data(), kv.data(), cpuRing.data()};
    cpu_->rawStore(a, nullptr);
    DeviceBuffer<std::int32_t> dPos(rows);
    DeviceBuffer<float> dKV(kv.size());
    DeviceBuffer<std::uint8_t> dRing(cpuRing.size());
    upload(dRing, std::vector<std::uint8_t>(cpuRing.size(), 0xEE));
    a.positions = upload(dPos, pos);
    a.kv = upload(dKV, kv);
    a.ring = dRing.data;
    for (int replay = 0; replay < 2; ++replay) gpu_->rawStore(a, stream_);
    EXPECT_EQ(download(dRing), cpuRing);
}

TEST_F(DeepSeekV4CUDAKernels, CompressorPoolIsChunkingInvariantOnDevice)
{
    const int d = 512, seq = 300;
    const RopePolicy rope = RopePolicy::make(64, 160000.0f, 65536, 16.0f, 32.0f, 1.0f);
    for (int ratio : {4, 128})
        for (auto stream : {CompressedStream::Attention, CompressedStream::Indexer})
        {
            if (ratio == 128 && stream == CompressedStream::Indexer) continue;
            const int dim = stream == CompressedStream::Indexer ? 128 : d;
            const CompressorGeometry g{ratio, dim, seq / ratio + 1, stream,
                                       stream == CompressedStream::Attention ? 64 : 0};
            const int width = g.projectionWidth();
            auto kv = randomVector(static_cast<std::size_t>(seq) * width, 21 + ratio);
            auto score = randomVector(kv.size(), 22 + ratio);
            auto ape = randomVector(static_cast<std::size_t>(ratio) * width, 23 + ratio, 0.5f);
            auto gain = randomVector(dim, 24 + ratio, 0.1f);
            for (auto &x : gain) x += 1.0f;
            DeviceBuffer<float> dKV(kv.size()), dScore(kv.size()), dApe(ape.size()), dGain(gain.size());
            DeviceBuffer<float> dStateKV(static_cast<std::size_t>(g.stateRows()) * width);
            DeviceBuffer<float> dStateScore(dStateKV.count);
            DeviceBuffer<std::uint8_t> dPool(static_cast<std::size_t>(g.poolRows) * g.entryBytes());
            DeviceBuffer<std::int32_t> dPos(seq);
            upload(dKV, kv);
            upload(dScore, score);
            upload(dApe, ape);
            upload(dGain, gain);
            const auto run = [&](int chunk) {
                upload(dPool, std::vector<std::uint8_t>(dPool.count, 0));
                for (int start = 0; start < seq; start += chunk)
                {
                    const int rows = std::min(chunk, seq - start);
                    std::vector<std::int32_t> pos(rows);
                    for (int r = 0; r < rows; ++r) pos[r] = start + r;
                    upload(dPos, pos);
                    const std::size_t at = static_cast<std::size_t>(start) * width;
                    gpu_->compressorPool({{rows, nullptr}, g, dPos.data, dKV.data + at, dScore.data + at, dApe.data,
                                          dStateKV.data, dStateScore.data, dGain.data, 1e-6f, rope, dPool.data},
                                         stream_);
                    gpu_->compressorStore({{rows, nullptr}, g, dPos.data, dKV.data + at, dScore.data + at, dApe.data,
                                           dStateKV.data, dStateScore.data}, stream_);
                }
                return download(dPool);
            };
            const auto whole = run(seq);
            for (int chunk : {1, 5, 7, 129}) EXPECT_EQ(run(chunk), whole) << "ratio " << ratio << " chunk " << chunk;
        }
}

TEST_F(DeepSeekV4CUDAKernels, IndexerScoresAreBitIdenticalAndBoundedPassesSelectExactly)
{
    const int heads = 64, dim = 128, topK = 16, rows = 37, poolRows = 300;
    const PackedIndexerKeyFormat format{dim};
    auto query = randomVector(static_cast<std::size_t>(rows) * heads * dim, 31);
    for (int t = 0; t < rows * heads; ++t) qat::apply(ActivationQuant::HadamardFP4, &query[t * dim], dim, 0);
    auto weights = randomVector(static_cast<std::size_t>(rows) * heads, 32);
    std::vector<std::uint8_t> pool(static_cast<std::size_t>(poolRows) * format.entryBytes());
    for (int b = 0; b < poolRows; ++b)
    {
        // Every fifth key duplicates its predecessor: exact score ties must
        // resolve to the lower block id on both backends.
        auto key = randomVector(dim, 1000 + (b % 5 == 4 ? b - 1 : b));
        qat::hadamard(key.data(), dim);
        qat::packIndexerKey(format, key.data(), &pool[static_cast<std::size_t>(b) * format.entryBytes()]);
    }
    std::vector<std::int32_t> pos(rows);
    for (int r = 0; r < rows; ++r) pos[r] = 1100 + r; // up to 284 visible blocks
    const float weightScale = 1.0f / std::sqrt(static_cast<float>(dim * heads));
    std::vector<std::int32_t> cpuSelected(static_cast<std::size_t>(rows) * topK);
    std::vector<float> cpuScores(static_cast<std::size_t>(rows) * poolRows, NAN);
    IndexerTopKArgs a{{rows, nullptr}, heads, format, 4, topK, poolRows, rows, pos.data(), query.data(),
                      weights.data(), weightScale, pool.data(), cpuScores.data(), cpuSelected.data()};
    cpu_->indexerTopK(a, nullptr);

    DeviceBuffer<std::int32_t> dPos(rows), dSelected(cpuSelected.size());
    DeviceBuffer<float> dQuery(query.size()), dWeights(weights.size());
    DeviceBuffer<std::uint8_t> dPool(pool.size());
    a.positions = upload(dPos, pos);
    a.query = upload(dQuery, query);
    a.weights = upload(dWeights, weights);
    a.pool = upload(dPool, pool);
    a.selected = dSelected.data;
    for (int scoreRows : {rows, 1, 3, 16})
    {
        DeviceBuffer<float> dScores(static_cast<std::size_t>(scoreRows) * poolRows);
        a.scoreRows = scoreRows;
        a.scores = dScores.data;
        upload(dSelected, std::vector<std::int32_t>(cpuSelected.size(), -7));
        gpu_->indexerTopK(a, stream_);
        EXPECT_EQ(download(dSelected), cpuSelected) << "scoreRows " << scoreRows;
        if (scoreRows == rows)
        {
            // Every visible score (cancelling signed dots, unequal block scales)
            // is the same binary32 word on both backends.
            const auto gpuScores = download(dScores);
            for (int r = 0; r < rows; ++r)
                for (int b = 0; b < (pos[r] + 1) / 4; ++b)
                {
                    const std::size_t at = static_cast<std::size_t>(r) * poolRows + b;
                    ASSERT_EQ(qat::bitsOf(gpuScores[at]), qat::bitsOf(cpuScores[at])) << "row " << r << " block " << b;
                }
        }
    }
}

TEST_F(DeepSeekV4CUDAKernels, SparseAttentionMatchesCPUAndChunking)
{
    const int heads = 4, window = 8, seq = 40, ratio = 4, topK = 4;
    const PackedKVEntryFormat format{512, 64};
    const RopePolicy rope = RopePolicy::make(64, 10000.0f, 0, 1.0f, 32.0f, 1.0f);
    auto q = randomVector(static_cast<std::size_t>(seq) * heads * 512, 41);
    auto kv = randomVector(static_cast<std::size_t>(seq) * 512, 42);
    for (int p = 0; p < seq; ++p) qat::apply(ActivationQuant::FP8NopeBF16Rope, &kv[p * 512], 512, 64);
    const int blocks = seq / ratio;
    std::vector<std::uint8_t> pool(static_cast<std::size_t>(blocks) * format.entryBytes());
    auto poolValues = randomVector(static_cast<std::size_t>(blocks) * 512, 43);
    for (int b = 0; b < blocks; ++b)
        qat::packKVEntry(format, &poolValues[b * 512], &pool[static_cast<std::size_t>(b) * format.entryBytes()]);
    std::vector<std::int32_t> selected(static_cast<std::size_t>(seq) * topK);
    for (int p = 0; p < seq; ++p)
        for (int i = 0; i < topK; ++i) selected[p * topK + i] = i < (p + 1) / ratio ? ((p + 1) / ratio - 1 - i) : -1;
    std::vector<float> sinks{0.3f, -0.4f, 0.0f, 1.0f};

    const auto runCPU = [&](int chunk) {
        std::vector<std::uint8_t> ring(static_cast<std::size_t>(window) * format.entryBytes(), 0xEE);
        std::vector<float> out(q.size());
        for (int start = 0; start < seq; start += chunk)
        {
            const int rows = std::min(chunk, seq - start);
            std::vector<std::int32_t> pos(rows);
            for (int r = 0; r < rows; ++r) pos[r] = start + r;
            cpu_->sparseAttention({{rows, nullptr}, heads, format, window, ratio, topK, CompressedAccess::Selected,
                                   1.0f / std::sqrt(512.0f), rope, pos.data(), &q[start * heads * 512], &kv[start * 512],
                                   ring.data(), pool.data(), &selected[start * topK], sinks.data(),
                                   &out[start * heads * 512]}, nullptr);
            cpu_->rawStore({{rows, nullptr}, format, window, pos.data(), &kv[start * 512], ring.data()}, nullptr);
        }
        return out;
    };
    DeviceBuffer<float> dQ(q.size()), dKV(kv.size()), dSinks(sinks.size()), dOut(q.size());
    DeviceBuffer<std::uint8_t> dPool(pool.size()), dRing(static_cast<std::size_t>(window) * format.entryBytes());
    DeviceBuffer<std::int32_t> dSelected(selected.size()), dPos(seq);
    upload(dQ, q);
    upload(dKV, kv);
    upload(dSinks, sinks);
    upload(dPool, pool);
    upload(dSelected, selected);
    const auto runGPU = [&](int chunk) {
        for (int start = 0; start < seq; start += chunk)
        {
            const int rows = std::min(chunk, seq - start);
            std::vector<std::int32_t> pos(rows);
            for (int r = 0; r < rows; ++r) pos[r] = start + r;
            upload(dPos, pos);
            gpu_->sparseAttention({{rows, nullptr}, heads, format, window, ratio, topK, CompressedAccess::Selected,
                                   1.0f / std::sqrt(512.0f), rope, dPos.data, dQ.data + start * heads * 512,
                                   dKV.data + start * 512, dRing.data, dPool.data, dSelected.data + start * topK,
                                   dSinks.data, dOut.data + start * heads * 512}, stream_);
            gpu_->rawStore({{rows, nullptr}, format, window, dPos.data, dKV.data + start * 512, dRing.data}, stream_);
        }
        return download(dOut);
    };
    const auto cpu = runCPU(seq);
    const auto gpuWhole = runGPU(seq);
    for (std::size_t i = 0; i < cpu.size(); ++i) ASSERT_NEAR(gpuWhole[i], cpu[i], 2e-5f * (1.0f + std::fabs(cpu[i])));
    for (int chunk : {1, 3, 9})
    {
        const auto gpu = runGPU(chunk);
        for (std::size_t i = 0; i < cpu.size(); ++i) ASSERT_EQ(gpu[i], gpuWhole[i]) << "chunk " << chunk;
    }
}

TEST_F(DeepSeekV4CUDAKernels, ImageSpanAttentionAndMergeMatchCPU)
{
    // Vision: an image block wider than the window attends bidirectionally on
    // device exactly as on CPU; span-preserving chunkings are bit-identical and
    // the sentinel merge is a byte copy.
    const int heads = 4, window = 8, seq = 48, ratio = 4;
    const PackedKVEntryFormat format{512, 64};
    const RopePolicy rope = RopePolicy::make(64, 10000.0f, 0, 1.0f, 32.0f, 1.0f);
    const ImageSentinelIds ids{1000, 5};
    std::vector<std::int32_t> tokens(seq, 7);
    tokens[10] = 1001; // compressor-alignment pad: outside the span
    tokens[11] = 1000;
    for (int p = 12; p < 34; ++p) tokens[p] = (p % 7 == 0) ? 1003 : 1002;
    tokens[34] = 1004;
    auto q = randomVector(static_cast<std::size_t>(seq) * heads * 512, 71);
    auto kv = randomVector(static_cast<std::size_t>(seq) * 512, 72);
    for (int p = 0; p < seq; ++p) qat::apply(ActivationQuant::FP8NopeBF16Rope, &kv[p * 512], 512, 64);
    const int blocks = seq / ratio;
    std::vector<std::uint8_t> pool(static_cast<std::size_t>(blocks) * format.entryBytes());
    auto poolValues = randomVector(static_cast<std::size_t>(blocks) * 512, 73);
    for (int b = 0; b < blocks; ++b)
        qat::packKVEntry(format, &poolValues[b * 512], &pool[static_cast<std::size_t>(b) * format.entryBytes()]);
    std::vector<float> sinks{0.3f, -0.4f, 0.0f, 1.0f};
    const std::vector<std::vector<int>> cuts{{0, seq}, {0, 11, 35, seq}, {0, 5, 11, 35, 40, seq}};

    const auto runCPU = [&](const std::vector<int> &cut) {
        std::vector<std::uint8_t> ring(static_cast<std::size_t>(window) * format.entryBytes(), 0xEE);
        std::vector<float> out(q.size());
        for (std::size_t c = 0; c + 1 < cut.size(); ++c)
        {
            const int start = cut[c], rows = cut[c + 1] - cut[c];
            std::vector<std::int32_t> pos(rows);
            for (int r = 0; r < rows; ++r) pos[r] = start + r;
            cpu_->sparseAttention({{rows, nullptr}, heads, format, window, ratio, 0, CompressedAccess::All,
                                   1.0f / std::sqrt(512.0f), rope, pos.data(), &q[start * heads * 512], &kv[start * 512],
                                   ring.data(), pool.data(), nullptr, sinks.data(), &out[start * heads * 512],
                                   &tokens[start], ids}, nullptr);
            cpu_->rawStore({{rows, nullptr}, format, window, pos.data(), &kv[start * 512], ring.data()}, nullptr);
        }
        return out;
    };
    DeviceBuffer<float> dQ(q.size()), dKV(kv.size()), dSinks(sinks.size()), dOut(q.size());
    DeviceBuffer<std::uint8_t> dPool(pool.size()), dRing(static_cast<std::size_t>(window) * format.entryBytes());
    DeviceBuffer<std::int32_t> dPos(seq), dTokens(seq);
    upload(dQ, q);
    upload(dKV, kv);
    upload(dSinks, sinks);
    upload(dPool, pool);
    upload(dTokens, tokens);
    const auto runGPU = [&](const std::vector<int> &cut) {
        for (std::size_t c = 0; c + 1 < cut.size(); ++c)
        {
            const int start = cut[c], rows = cut[c + 1] - cut[c];
            std::vector<std::int32_t> pos(rows);
            for (int r = 0; r < rows; ++r) pos[r] = start + r;
            upload(dPos, pos);
            gpu_->sparseAttention({{rows, nullptr}, heads, format, window, ratio, 0, CompressedAccess::All,
                                   1.0f / std::sqrt(512.0f), rope, dPos.data, dQ.data + start * heads * 512,
                                   dKV.data + start * 512, dRing.data, dPool.data, nullptr, dSinks.data,
                                   dOut.data + start * heads * 512, dTokens.data + start, ids}, stream_);
            gpu_->rawStore({{rows, nullptr}, format, window, dPos.data, dKV.data + start * 512, dRing.data}, stream_);
        }
        return download(dOut);
    };
    const auto cpu = runCPU(cuts[0]);
    const auto gpuWhole = runGPU(cuts[0]);
    for (std::size_t i = 0; i < cpu.size(); ++i) ASSERT_NEAR(gpuWhole[i], cpu[i], 2e-5f * (1.0f + std::fabs(cpu[i])));
    for (const auto &cut : cuts)
    {
        const auto gpu = runGPU(cut);
        for (std::size_t i = 0; i < cpu.size(); ++i) ASSERT_EQ(gpu[i], gpuWhole[i]);
        const auto chunkedCPU = runCPU(cut);
        for (std::size_t i = 0; i < cpu.size(); ++i) ASSERT_EQ(chunkedCPU[i], cpu[i]);
    }

    // Merge: only sentinel rows with a nonzero map entry take their bank row.
    const int width = 96, bankRows = 6, mapRows = 64;
    auto bank = randomVector(static_cast<std::size_t>(bankRows) * width, 74);
    std::vector<std::int32_t> map(mapRows, 0);
    for (int p = 11; p <= 34; ++p) map[p] = 1 + (p % bankRows);
    map[3] = 2; // text token at a mapped position keeps its embedding
    std::vector<std::int32_t> pos(seq);
    for (int r = 0; r < seq; ++r) pos[r] = r;
    const auto embedding = randomVector(static_cast<std::size_t>(seq) * width, 75);
    auto expected = embedding;
    cpu_->imageMerge({{seq, nullptr}, width, bankRows, mapRows, ids, tokens.data(), pos.data(), bank.data(), map.data(),
                      expected.data()}, nullptr);
    DeviceBuffer<float> dBank(bank.size()), dEmb(embedding.size());
    DeviceBuffer<std::int32_t> dMap(map.size()), dAll(seq);
    upload(dBank, bank);
    upload(dMap, map);
    upload(dAll, pos);
    upload(dEmb, embedding);
    gpu_->imageMerge({{seq, nullptr}, width, bankRows, mapRows, ids, dTokens.data, dAll.data, dBank.data, dMap.data,
                      dEmb.data}, stream_);
    EXPECT_EQ(download(dEmb), expected);
    EXPECT_NE(expected, embedding);
}

TEST_F(DeepSeekV4CUDAKernels, FrontierOnlyCacheOwnsNoConversionScratchAndAdvancesOnce)
{
    using namespace llaminar2;
    HybridKVCacheConfig hybrid;
    hybrid.layer_types.assign(43, kHybridModelStateLayerType);
    hybrid.n_heads = 64;
    llaminar::v2::kernels::KVCacheConfig config;
    config.precision = ActivationPrecision::FP16;
    config.device = DeviceId::cuda(0);
    config.num_layers = 43;
    config.batch_size = 1;
    config.max_seq_len = 1 << 20;
    config.n_kv_heads = 1;
    config.head_dim = 512;
    config.hybrid_config = &hybrid;
    auto cache = llaminar::v2::kernels::KernelFactory::createKVCache(config);
    ASSERT_NE(cache, nullptr);
    auto *hybridView = dynamic_cast<IHybridKVCache *>(cache.get());
    ASSERT_NE(hybridView, nullptr);
    EXPECT_EQ(hybridView->kvLayerCount(), 0);
    EXPECT_EQ(hybridView->gdnLayerCount(), 0);
    // No attention layer: no full-context FP32 K/V conversion scratch (~4 GiB at 1M).
    auto *consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(consumer, nullptr);
    EXPECT_TRUE(consumer->getWorkspaceRequirements(1 << 20, 1, 0).empty());

    IKVCache::StateResetContext reset;
    reset.execution_stream = stream_;
    reset.reason = "deepseek4_frontier_replay";
    ASSERT_TRUE(cache->resetRequestState(reset));
    const int *count = cache->deviceSequenceCachedTokenCountPtr(0);
    ASSERT_NE(count, nullptr);
    DeviceBuffer<std::int32_t> realRows(1);
    upload(realRows, std::vector<std::int32_t>{5});
    const auto frontier = [&] {
        int host = -1;
        EXPECT_EQ(cudaMemcpyAsync(&host, count, sizeof(int), cudaMemcpyDeviceToHost, stream_), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        return host;
    };
    EXPECT_EQ(frontier(), 0);
    for (int replay = 0; replay < 2; ++replay)
    {
        // Prefill of 8 captured rows with 5 real rows, then two decode steps.
        for (int layer = 0; layer < 43; ++layer)
            ASSERT_TRUE(cache->advanceRecurrentSequenceState(layer, 1, realRows.data, 8, stream_));
        EXPECT_EQ(frontier(), 5);
        for (int step = 0; step < 2; ++step)
            ASSERT_TRUE(cache->advanceRecurrentSequenceState(42, 1, nullptr, 1, stream_));
        EXPECT_EQ(frontier(), 7);
        ASSERT_TRUE(cache->resetRequestState(reset));
        EXPECT_EQ(frontier(), 0) << "request reset resets data, not topology";
    }
}

TEST_F(DeepSeekV4CUDAKernels, QATKeepsSubnormalsDespiteFastMathFTZ)
{
    // Production CUDA is compiled with --use_fast_math --ftz=true. A normal
    // input whose Hadamard + FP4 QAT result is subnormal must survive on the
    // device exactly as on the CPU: [2^-124, 0, ...] (64 wide) rotates to 64
    // copies of 2^-127 = E2M1 0.5 at the E8M0 floor scale 2^-126.
    RopePolicy noRope;
    noRope.ropeDim = 0;
    std::vector<float> data(64, 0.0f);
    data[0] = 0x1p-124f;
    auto cpu = data;
    HeadNormRopeArgs a{{1, nullptr}, 1, 64, false, 1e-6f, nullptr, noRope, false, ActivationQuant::HadamardFP4,
                       nullptr, cpu.data()};
    cpu_->headNormRope(a, nullptr);
    for (const float v : cpu) ASSERT_EQ(v, 0x1p-127f);
    DeviceBuffer<float> dData(data.size());
    a.data = upload(dData, data);
    gpu_->headNormRope(a, stream_);
    const auto gpu = download(dData);
    for (std::size_t i = 0; i < gpu.size(); ++i) ASSERT_EQ(qat::bitsOf(gpu[i]), qat::bitsOf(cpu[i])) << i;

    // The packed minimum-scale record (scale byte 1, code 0.5) decodes and
    // scores to subnormals on the device: query 2^-7 per channel gives block
    // sums 2^-3 and a score of 2 * 2^-3 * 2^-126 = 2^-128. A flushing device
    // would tie it with the zero key at block 0 and select the lower id.
    const PackedIndexerKeyFormat format{64};
    std::vector<std::uint8_t> pool(2 * static_cast<std::size_t>(format.entryBytes()));
    const std::vector<float> zeros(64, 0.0f);
    qat::packIndexerKey(format, zeros.data(), pool.data());
    qat::packIndexerKey(format, cpu.data(), pool.data() + format.entryBytes());
    ASSERT_EQ(pool[format.entryBytes() + format.scaleOffset()], 1);
    ASSERT_EQ(qat::decodeIndexerChannel(format, pool.data() + format.entryBytes(), 5), 0x1p-127f);
    std::vector<float> query(64, 0x1p-7f), weights{1.0f};
    std::vector<std::int32_t> pos{7}, cpuSelected(1), gpuSelected;
    std::vector<float> cpuScores(2, NAN);
    IndexerTopKArgs k{{1, nullptr}, 1, format, 4, 1, 2, 1, pos.data(), query.data(), weights.data(), 1.0f,
                      pool.data(), cpuScores.data(), cpuSelected.data()};
    cpu_->indexerTopK(k, nullptr);
    EXPECT_EQ(cpuScores[1], 0x1p-128f);
    EXPECT_EQ(cpuSelected[0], 1);
    DeviceBuffer<std::int32_t> dPos(1), dSelected(1);
    DeviceBuffer<float> dQuery(64), dWeights(1), dScores(2);
    DeviceBuffer<std::uint8_t> dPool(pool.size());
    k.positions = upload(dPos, pos);
    k.query = upload(dQuery, query);
    k.weights = upload(dWeights, weights);
    k.pool = upload(dPool, pool);
    k.scores = dScores.data;
    k.selected = dSelected.data;
    gpu_->indexerTopK(k, stream_);
    const auto gpuScores = download(dScores);
    EXPECT_EQ(qat::bitsOf(gpuScores[1]), qat::bitsOf(cpuScores[1]));
    EXPECT_EQ(download(dSelected)[0], 1);
}
