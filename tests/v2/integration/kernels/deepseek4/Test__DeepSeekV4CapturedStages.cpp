/**
 * @file Test__DeepSeekV4CapturedStages.cpp
 * @brief Captured replay of the production DeepSeek-V4 bank stages and frontier publication.
 *
 * One layer's CompressorPool -> CompressorStore -> SparseAttention -> RawStore
 * -> AdvanceSequenceFrontier is recorded once through DeepSeekV4PrimitiveStage
 * (arena-style tensors, a device live-row count and device positions) and
 * replayed over a padded chunk schedule with full, partial and zero live rows.
 * Each replay must:
 *  - leave rows beyond the live count untouched and a zero-live replay a no-op
 *    for every bank, output and the cache frontier;
 *  - advance the frontier-only cache by exactly the live rows;
 *  - publish packed pools, window ring and attention outputs identical to one
 *    eager whole-prompt execution (chunk invariance through the carried state);
 *  - reproduce the same bytes after a request reset (data reset, same topology).
 */
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/deepseek4/DeepSeekV4PrimitiveStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/deepseek4/DeepSeekV4Kernels.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::deepseek4;

namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> randomVector(std::size_t n, unsigned seed, float scale = 1.0f)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

constexpr int kCapacity = 6, kSeq = 13, kHeads = 2, kDim = 128, kRope = 64, kWindow = 4, kRatio = 4;
constexpr int kWidth = 2 * kDim, kStates = 2 * kRatio, kPoolRows = 8;
} // namespace

TEST(DeepSeekV4CapturedStages, PaddedChunkReplayMatchesWholePromptAndAdvancesFrontierOnce)
{
    const DeviceId device = DeviceId::cuda(0);
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal) GTEST_FAIL() << "CUDA suite requires a CUDA device";
    auto context = IDeviceContext::create(device, 1);
    require(bool(context), "device context creation failed");
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        void *stream = ExplicitGPUStream(worker.defaultStream()).get();
        const auto upload = [&](void *dst, const void *src, std::size_t bytes) {
            require(backend->hostToDeviceOnStream(dst, src, bytes, device.ordinal, stream), "upload failed");
            require(backend->synchronizeStream(stream, device.ordinal), "upload join failed");
        };
        const auto download = [&](void *dst, const void *src, std::size_t bytes) {
            require(backend->deviceToHostOnStream(dst, src, bytes, device.ordinal, stream), "download failed");
            require(backend->synchronizeStream(stream, device.ordinal), "download join failed");
        };
        const PackedKVEntryFormat format{kDim, kRope};
        const std::size_t words = static_cast<std::size_t>(format.entryWords());
        const RopePolicy rope = RopePolicy::make(kRope, 160000.0f, 65536, 16.0f, 32.0f, 1.0f);
        const CompressorGeometry geometry{kRatio, kDim, kPoolRows, CompressedStream::Attention, kRope};

        // Per-position inputs: QAT kv, compressor projections and queries.
        auto kvAll = randomVector(static_cast<std::size_t>(kSeq) * kDim, 1);
        for (int p = 0; p < kSeq; ++p) qat::apply(ActivationQuant::FP8NopeBF16Rope, &kvAll[p * kDim], kDim, kRope);
        const auto compKVAll = randomVector(static_cast<std::size_t>(kSeq) * kWidth, 2);
        const auto compScoreAll = randomVector(static_cast<std::size_t>(kSeq) * kWidth, 3);
        const auto qAll = randomVector(static_cast<std::size_t>(kSeq) * kHeads * kDim, 4);

        FP32Tensor kv({kCapacity, kDim}, DeviceId::cpu()), compKV({kCapacity, kWidth}, DeviceId::cpu()),
            compScore({kCapacity, kWidth}, DeviceId::cpu()), q({kCapacity, kHeads * kDim}, DeviceId::cpu()),
            out({kCapacity, kHeads * kDim}, DeviceId::cpu()), stateKV({kStates, kWidth}, DeviceId::cpu()),
            stateScore({kStates, kWidth}, DeviceId::cpu()), ape({kRatio, kWidth}, DeviceId::cpu()),
            gain({kDim}, DeviceId::cpu()), sinks({kHeads}, DeviceId::cpu());
        INT32Tensor pool({kPoolRows * words}), ring({kWindow * words}), positions({kCapacity}), live({1});
        for (TensorBase *t : std::vector<TensorBase *>{&kv, &compKV, &compScore, &q, &out, &stateKV, &stateScore,
                                                       &ape, &gain, &sinks, &pool, &ring, &positions, &live})
            std::memset(t->raw_mutable_data(), 0, t->size_bytes());
        const auto apeValues = randomVector(static_cast<std::size_t>(kRatio) * kWidth, 5, 0.5f);
        std::memcpy(ape.mutable_data(), apeValues.data(), apeValues.size() * sizeof(float));
        for (int c = 0; c < kDim; ++c) gain.mutable_data()[c] = 1.0f + 0.01f * c;
        sinks.mutable_data()[0] = 0.3f;
        sinks.mutable_data()[1] = -0.4f;
        for (TensorBase *t : std::vector<TensorBase *>{&kv, &compKV, &compScore, &q, &out, &stateKV, &stateScore,
                                                       &ape, &gain, &sinks, &pool, &ring, &positions, &live})
        {
            TransferEngine::prepareDeviceInput(t, device, stream);
            TransferEngine::requireDeviceInput(t, device, stream);
        }

        HybridKVCacheConfig hybrid;
        hybrid.layer_types = {kHybridModelStateLayerType};
        hybrid.n_heads = kHeads;
        llaminar::v2::kernels::KVCacheConfig cacheConfig;
        cacheConfig.precision = ActivationPrecision::FP16;
        cacheConfig.device = device;
        cacheConfig.num_layers = 1;
        cacheConfig.batch_size = 1;
        cacheConfig.max_seq_len = 64;
        cacheConfig.n_kv_heads = 1;
        cacheConfig.head_dim = kDim;
        cacheConfig.hybrid_config = &hybrid;
        auto cache = llaminar::v2::kernels::KernelFactory::createKVCache(cacheConfig);
        require(bool(cache), "frontier-only cache creation failed");
        IKVCache::StateResetContext reset;
        reset.execution_stream = stream;
        reset.reason = "deepseek4_captured_stage_replay";
        require(cache->resetRequestState(reset), "cache reset failed");

        using Stage = DeepSeekV4PrimitiveStage;
        using Float = Qwen4ExpFloatBinding;
        using Int = Qwen4ExpIntBinding;
        using Weight = Qwen4ExpImmutableWeightBinding;
        const Qwen4ExpStageRows rows{kCapacity, Int(live, BufferId::PREFILL_CHUNK_GEOMETRY)};
        const Qwen4ExpIntInput pos{Int(positions, BufferId::REQUEST_TOKEN_IDS)};
        const auto make = [&](Stage::Operation operation) {
            auto stage = std::make_unique<Stage>(Stage::Params{.device_id = device, .rows = rows,
                                                               .operation = std::move(operation)});
            stage->setGPUStream(stream);
            return stage;
        };
        std::vector<std::unique_ptr<Stage>> stages;
        stages.push_back(make(Stage::CompressorPool{geometry, pos, Float(compKV, BufferId::DSV4_COMP_KV),
            Float(compScore, BufferId::DSV4_COMP_SCORE), Weight(ape), Float(stateKV, BufferId::DSV4_COMP_RING_KV),
            Float(stateScore, BufferId::DSV4_COMP_RING_SCORE), Weight(gain), 1e-6f, rope,
            Int(pool, BufferId::DSV4_COMP_POOL)}));
        stages.push_back(make(Stage::CompressorStore{geometry, pos, Float(compKV, BufferId::DSV4_COMP_KV),
            Float(compScore, BufferId::DSV4_COMP_SCORE), Weight(ape), Float(stateKV, BufferId::DSV4_COMP_RING_KV),
            Float(stateScore, BufferId::DSV4_COMP_RING_SCORE)}));
        stages.push_back(make(Stage::SparseAttention{kHeads, format, kWindow, kRatio, 0, kPoolRows,
            CompressedAccess::All, 1.0f / 16.0f, rope, pos, Float(q, BufferId::DSV4_Q), Float(kv, BufferId::K_PROJ),
            Int(ring, BufferId::DSV4_RAW_RING), Int(pool, BufferId::DSV4_COMP_POOL), std::nullopt, Weight(sinks),
            Float(out, BufferId::DSV4_ATTN)}));
        stages.push_back(make(Stage::RawStore{format, kWindow, pos, Float(kv, BufferId::K_PROJ),
            Int(ring, BufferId::DSV4_RAW_RING)}));
        stages.push_back(make(Stage::AdvanceSequenceFrontier{cache.get(), 0, pos}));

        auto graph = worker.createGraphCapture(stream);
        require(bool(graph), "graph owner missing");
        {
            ScopedBackendGraphCapture capture(worker, *graph, "DeepSeek-V4 bank stages");
            require(capture.begin(), "capture begin failed");
            for (auto &stage : stages) require(stage->execute(context.get()), "stage recording failed");
            capture.finish();
        }
        require(graph->instantiate(), "graph instantiation failed");

        const auto frontier = [&] {
            int value = -1;
            download(&value, cache->deviceSequenceCachedTokenCountPtr(0), sizeof(int));
            return value;
        };
        const float sentinel = -12345.0f;
        struct Result { std::vector<std::int32_t> pool, ring; std::vector<float> out; };
        // Schedule (start, live): full, partial (block straddles), zero, singles.
        const std::vector<std::pair<int, int>> schedule{{0, 6}, {6, 5}, {11, 0}, {11, 1}, {12, 1}};
        const auto runRequest = [&]() -> Result {
            Result result{std::vector<std::int32_t>(pool.numel()), std::vector<std::int32_t>(ring.numel()),
                          std::vector<float>(static_cast<std::size_t>(kSeq) * kHeads * kDim, sentinel)};
            int advanced = 0;
            for (const auto &[start, count] : schedule)
            {
                std::vector<std::int32_t> hostPos(kCapacity);
                for (int r = 0; r < kCapacity; ++r) hostPos[r] = start + r;
                std::vector<float> hostKV(kCapacity * kDim, 0.0f), hostCKV(kCapacity * kWidth, 0.0f),
                    hostCS(kCapacity * kWidth, 0.0f), hostQ(kCapacity * kHeads * kDim, 0.0f),
                    hostOut(kCapacity * kHeads * kDim, sentinel);
                for (int r = 0; r < kCapacity && start + r < kSeq; ++r)
                {
                    std::memcpy(&hostKV[r * kDim], &kvAll[(start + r) * kDim], kDim * sizeof(float));
                    std::memcpy(&hostCKV[r * kWidth], &compKVAll[(start + r) * kWidth], kWidth * sizeof(float));
                    std::memcpy(&hostCS[r * kWidth], &compScoreAll[(start + r) * kWidth], kWidth * sizeof(float));
                    std::memcpy(&hostQ[r * kHeads * kDim], &qAll[(start + r) * kHeads * kDim],
                                kHeads * kDim * sizeof(float));
                }
                upload(positions.gpu_data_ptr(), hostPos.data(), hostPos.size() * 4);
                upload(live.gpu_data_ptr(), &count, 4);
                upload(kv.gpu_data_ptr(), hostKV.data(), hostKV.size() * 4);
                upload(compKV.gpu_data_ptr(), hostCKV.data(), hostCKV.size() * 4);
                upload(compScore.gpu_data_ptr(), hostCS.data(), hostCS.size() * 4);
                upload(q.gpu_data_ptr(), hostQ.data(), hostQ.size() * 4);
                upload(out.gpu_data_ptr(), hostOut.data(), hostOut.size() * 4);
                std::vector<std::int32_t> poolBefore(pool.numel()), ringBefore(ring.numel());
                download(poolBefore.data(), pool.gpu_data_ptr(), pool.size_bytes());
                download(ringBefore.data(), ring.gpu_data_ptr(), ring.size_bytes());
                require(graph->launch(), "replay failed");
                advanced += count;
                EXPECT_EQ(frontier(), advanced) << "start " << start << " live " << count;
                download(hostOut.data(), out.gpu_data_ptr(), hostOut.size() * 4);
                bool untouched = true;
                for (std::size_t i = static_cast<std::size_t>(count) * kHeads * kDim; i < hostOut.size(); ++i)
                    untouched &= hostOut[i] == sentinel;
                EXPECT_TRUE(untouched) << "row beyond live count written, start " << start;
                for (int r = 0; r < count; ++r)
                    std::memcpy(&result.out[(start + r) * kHeads * kDim], &hostOut[r * kHeads * kDim],
                                kHeads * kDim * sizeof(float));
                download(result.pool.data(), pool.gpu_data_ptr(), pool.size_bytes());
                download(result.ring.data(), ring.gpu_data_ptr(), ring.size_bytes());
                if (count == 0)
                {
                    EXPECT_EQ(result.pool, poolBefore) << "zero-live replay mutated the pool";
                    EXPECT_EQ(result.ring, ringBefore) << "zero-live replay mutated the ring";
                }
            }
            return result;
        };

        const auto first = runRequest();
        require(cache->resetRequestState(reset), "request reset failed");
        EXPECT_EQ(frontier(), 0);
        const auto second = runRequest();
        EXPECT_EQ(second.pool, first.pool) << "request reset must reproduce the same bytes";
        EXPECT_EQ(second.ring, first.ring);
        EXPECT_EQ(std::memcmp(second.out.data(), first.out.data(), first.out.size() * 4), 0);

        // Oracle: one eager whole-prompt execution of the same kernels.
        auto kernels = llaminar::v2::kernels::KernelFactory::createDeepSeekV4(device);
        const auto devAlloc = [&](std::size_t bytes) {
            auto owner = TransferEngine::instance().allocateDeviceTransferBuffer(bytes, device);
            require(owner && owner->isBound(), "oracle allocation failed");
            return owner;
        };
        auto oPos = devAlloc(kSeq * 4), oKV = devAlloc(kvAll.size() * 4), oCKV = devAlloc(compKVAll.size() * 4),
             oCS = devAlloc(compScoreAll.size() * 4), oQ = devAlloc(qAll.size() * 4), oOut = devAlloc(qAll.size() * 4),
             oSK = devAlloc(stateKV.size_bytes()), oSS = devAlloc(stateScore.size_bytes()),
             oPool = devAlloc(pool.size_bytes()), oRing = devAlloc(ring.size_bytes());
        std::vector<std::int32_t> wholePos(kSeq);
        for (int p = 0; p < kSeq; ++p) wholePos[p] = p;
        upload(oPos->mutableDeviceData(), wholePos.data(), kSeq * 4);
        upload(oKV->mutableDeviceData(), kvAll.data(), kvAll.size() * 4);
        upload(oCKV->mutableDeviceData(), compKVAll.data(), compKVAll.size() * 4);
        upload(oCS->mutableDeviceData(), compScoreAll.data(), compScoreAll.size() * 4);
        upload(oQ->mutableDeviceData(), qAll.data(), qAll.size() * 4);
        const std::vector<std::int32_t> zeroPool(pool.numel(), 0), zeroRing(ring.numel(), 0);
        upload(oPool->mutableDeviceData(), zeroPool.data(), pool.size_bytes());
        upload(oRing->mutableDeviceData(), zeroRing.data(), ring.size_bytes());
        const auto *p32 = static_cast<const std::int32_t *>(oPos->mutableDeviceData());
        const auto f = [](const std::shared_ptr<DeviceTransferBuffer> &b) { return static_cast<float *>(b->mutableDeviceData()); };
        const auto u8 = [](const std::shared_ptr<DeviceTransferBuffer> &b) { return static_cast<std::uint8_t *>(b->mutableDeviceData()); };
        const auto *apeDev = static_cast<const float *>(ape.gpu_data_ptr());
        kernels->compressorPool({{kSeq, nullptr}, geometry, p32, f(oCKV), f(oCS), apeDev, f(oSK), f(oSS),
                                 static_cast<const float *>(gain.gpu_data_ptr()), 1e-6f, rope, u8(oPool)}, stream);
        kernels->compressorStore({{kSeq, nullptr}, geometry, p32, f(oCKV), f(oCS), apeDev, f(oSK), f(oSS)}, stream);
        kernels->sparseAttention({{kSeq, nullptr}, kHeads, format, kWindow, kRatio, 0, CompressedAccess::All,
                                  1.0f / 16.0f, rope, p32, f(oQ), f(oKV), u8(oRing), u8(oPool), nullptr,
                                  static_cast<const float *>(sinks.gpu_data_ptr()), f(oOut)}, stream);
        kernels->rawStore({{kSeq, nullptr}, format, kWindow, p32, f(oKV), u8(oRing)}, stream);
        std::vector<std::int32_t> wholePool(pool.numel()), wholeRing(ring.numel());
        std::vector<float> wholeOut(qAll.size());
        download(wholePool.data(), oPool->mutableDeviceData(), pool.size_bytes());
        download(wholeRing.data(), oRing->mutableDeviceData(), ring.size_bytes());
        download(wholeOut.data(), oOut->mutableDeviceData(), wholeOut.size() * 4);
        EXPECT_EQ(first.pool, wholePool) << "chunked captured pool differs from the whole prompt";
        EXPECT_EQ(first.ring, wholeRing) << "window ring differs from the whole prompt";
        EXPECT_EQ(std::memcmp(first.out.data(), wholeOut.data(), wholeOut.size() * 4), 0)
            << "chunked captured attention differs from the whole prompt";
        graph.reset(); // Retire captured pointers before their tensors.
    }).get();
}
