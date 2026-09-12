/**
 * @file AttentionKeyCachePartitionProof.h
 * @brief Shared captured CUDA/ROCm proof of prefix-independent key encoding.
 *
 * Cold prefill, a split prefill and device-hot prefix restoration must produce
 * identical native bytes. This checks the production cache APIs, including the
 * request-local anchor, rather than comparing two decoders of the same payload.
 * All input and export storage is prepared before capture; host observation is
 * confined to result collection after a complete graph replay.
 */
#pragma once

#include <gtest/gtest.h>
#include "KVCacheTestWorkspace.h"
#include "AttentionKeyQ8DeviceTestCommon.h"
#include "../../utils/TestTensorFactory.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/IKVCache.h"
#include "kernels/kvcache/TurboQuantKVMode.h"
#include "tensors/GpuTensorView.h"

#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2::test
{
/**
 * @brief Compare complete native cache payloads for every supported AQ8 value codec.
 * @tparam Cache Backend's production compressed ring implementation.
 * @param device Physical device owning tensors, stream and graphs.
 * @param stream Explicit test stream, retained until all graphs/caches retire.
 * @param make_graph Construct the backend's RAII capture on that exact stream.
 * @param synchronize Test-only completed-result boundary on the same stream.
 */
template <class Cache, class MakeGraph, class Synchronize>
void proveAttentionKeyCachePartitions(DeviceId device, void *stream,
                                     MakeGraph make_graph, Synchronize synchronize)
{
    constexpr int total = 158;
    constexpr int heads = 2;
    for (int capacity : {256, 31})
    for (int dimension : {64, 128, 256})
    for (auto mode : {TurboQuantKVMode::AQ8_K_Q8_1_V,
                      TurboQuantKVMode::AQ8_K_TQ4_V,
                      TurboQuantKVMode::AQ8_K_TQ8_V})
    {
        SCOPED_TRACE(::testing::Message() << device.toString() << '/' << capacity << '/' << dimension
                                        << '/' << turboQuantKVModeName(mode));
        const size_t width = heads * dimension;
        TurboQuantContext context(dimension, 42);
        Cache cold(1, 1, capacity, heads, dimension, &context, 0, mode);
        Cache prefix(1, 1, capacity, heads, dimension, &context, 0, mode);
        Cache restored(1, 1, capacity, heads, dimension, &context, 0, mode);
        KVCacheTestWorkspaceBinding cold_workspace(cold, device);
        KVCacheTestWorkspaceBinding prefix_workspace(prefix, device);
        KVCacheTestWorkspaceBinding restored_workspace(restored, device);
        auto key = TestTensorFactory::createFP32({total, width});
        auto value = TestTensorFactory::createFP32({total, width});
        AttentionKeyQ8XorShift32 random(53);
        for (size_t i = 0; i < key->numel(); ++i)
        {
            // Biased, non-constant keys expose a chunk-mean basis immediately.
            key->mutable_data()[i] = 12.0f + 3.0f * random.symmetricUnit();
            value->mutable_data()[i] = random.symmetricUnit();
        }
        ASSERT_TRUE(key->ensureOnDevice(device, stream));
        ASSERT_TRUE(value->ensureOnDevice(device, stream));
        const int retained = std::min(total, capacity);
        const auto layout = cold.logicalBlockLayout(0, retained);
        auto block_k = TestTensorFactory::createFP32Zeros({(layout.k_bytes + 3) / 4});
        auto block_v = TestTensorFactory::createFP32Zeros({(layout.v_bytes + 3) / 4});
        ASSERT_TRUE(block_k->ensureOnDevice(device, stream));
        ASSERT_TRUE(block_v->ensureOnDevice(device, stream));
        ASSERT_TRUE(synchronize());

        std::vector<int> splits;
        for (int rows = 1; rows <= 16; ++rows)
            splits.push_back(rows);
        splits.insert(splits.end(), {31, 64, 139});
        for (int split : splits)
        {
            if (split > capacity)
                continue; // The exported prefix must fit the configured ring.
            SCOPED_TRACE(split);
            GpuTensorView suffix_k(static_cast<float *>(key->gpu_data_ptr()) + split * width,
                                  total - split, width, TensorType::FP32, device);
            GpuTensorView suffix_v(static_cast<float *>(value->gpu_data_ptr()) + split * width,
                                  total - split, width, TensorType::FP32, device);
            const IKVCache::KVCacheLogicalBlockDescriptor empty{
                0, 0, 0, 0, stream, IKVCache::KVCacheLogicalBlockPayloadDomain::Device};
            const IKVCache::KVCacheLogicalBlockDescriptor part{
                0, 0, 0, split, stream, IKVCache::KVCacheLogicalBlockPayloadDomain::Device};
            auto graph = make_graph();
            ASSERT_TRUE(graph->beginCapture());
            bool submitted = false;
            {
                GraphCaptureGuard guard;
                // Reset is itself ordered device work. The second replay starts
                // with non-empty caches and must establish the same fresh basis.
                submitted = cold.importLogicalBlock(empty, nullptr, nullptr) &&
                    prefix.importLogicalBlock(empty, nullptr, nullptr) &&
                    restored.importLogicalBlock(empty, nullptr, nullptr) &&
                    cold.appendWithStream(0, 0, key.get(), value.get(), total, stream) &&
                    prefix.appendWithStream(0, 0, key.get(), value.get(), split, stream) &&
                    prefix.exportLogicalBlock(part, block_k->gpu_data_ptr(), block_v->gpu_data_ptr()) &&
                    restored.importLogicalBlock(part, block_k->gpu_data_ptr(), block_v->gpu_data_ptr()) &&
                    restored.appendWithStream(0, 0, &suffix_k, &suffix_v, total - split, stream);
            }
            ASSERT_TRUE(graph->endCapture());
            ASSERT_TRUE(submitted);
            ASSERT_TRUE(graph->instantiate());
            for (int replay = 0; replay < 2; ++replay)
            {
                SCOPED_TRACE(replay);
                ASSERT_TRUE(graph->launch());
                ASSERT_TRUE(synchronize());
                std::vector<uint8_t> cold_k(layout.k_bytes), cold_v(layout.v_bytes);
                std::vector<uint8_t> restored_k(layout.k_bytes), restored_v(layout.v_bytes);
                const IKVCache::KVCacheLogicalBlockDescriptor all{0, 0, 0, retained, stream};
                ASSERT_TRUE(cold.exportLogicalBlock(all, cold_k.data(), cold_v.data()));
                ASSERT_TRUE(restored.exportLogicalBlock(all, restored_k.data(), restored_v.data()));
                // This independent basis assertion prevents two identically
                // wrong publication paths from certifying one another.
                EXPECT_EQ(std::memcmp(cold_k.data(), key->data(), width * sizeof(float)), 0)
                    << "Initial cache basis must not depend on the first chunk's width";
                EXPECT_EQ(std::memcmp(cold_k.data(), restored_k.data(), layout.k_bytes), 0)
                    << "Cold and prefix-restored native keys differ";
                EXPECT_EQ(std::memcmp(cold_v.data(), restored_v.data(), layout.v_bytes), 0)
                    << "Cold and prefix-restored native values differ";
            }
        }
    }
}
} // namespace llaminar2::test
