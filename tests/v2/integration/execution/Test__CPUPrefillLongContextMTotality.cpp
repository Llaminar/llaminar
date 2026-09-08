/**
 * @file Test__CPUPrefillLongContextMTotality.cpp
 * @brief CPU long-context prefill chunk and KV-cache M-totality integration.
 *
 * This suite exercises a 256K logical prompt plus a one-token tail through the
 * same 4096-row chunk policy used by GPU prefill. The cache probe uses the
 * Qwen2.5-0.5B KV geometry (two 64-value KV heads) and native Q8_1 storage, but
 * a single layer keeps the integration gate compact. The purpose is to prove
 * scheduler offsets, native cache addressing, and real-row advancement across
 * the 256K boundary; running a complete 0.5B transformer for 256K rows would
 * add enormous compute without strengthening those invariants.
 */

#include <gtest/gtest.h>

#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kPrefillBucketRows = 4096;
        constexpr int kLongContextRows = 256 * 1024 + 1;
        constexpr int kQwen25_05BHiddenSize = 896;
        constexpr int kQwen25_05BKVHeads = 2;
        constexpr int kQwen25_05BHeadDim = 64;
        constexpr int kQwen25_05BKVDim =
            kQwen25_05BKVHeads * kQwen25_05BHeadDim;

        /** @return A deterministic one-rank MPI context for CPU cache policy. */
        MPIContext singleRankContext()
        {
            return MPIContext(0, 1, MPI_COMM_WORLD);
        }

        /**
         * @brief Create one reusable 4096-row Qwen2.5-0.5B K or V source tensor.
         *
         * Every row has a distinct, exactly reproducible pattern. Reusing this
         * tensor for each chunk makes token zero and the final one-token tail
         * mathematically identical, allowing a native-byte comparison after
         * Q8_1 encoding while still making adjacent rows distinguishable.
         */
        std::shared_ptr<Q8_1Tensor> makeKVRows(float base)
        {
            const std::vector<size_t> shape{
                static_cast<size_t>(kPrefillBucketRows),
                static_cast<size_t>(kQwen25_05BKVDim)};
            std::vector<float> values(
                static_cast<size_t>(kPrefillBucketRows) * kQwen25_05BKVDim);
            for (int row = 0; row < kPrefillBucketRows; ++row)
            {
                for (int column = 0; column < kQwen25_05BKVDim; ++column)
                {
                    values[static_cast<size_t>(row) * kQwen25_05BKVDim +
                           static_cast<size_t>(column)] =
                        base +
                        static_cast<float>((row * 17 + column * 3) % 127) /
                            32.0f;
                }
            }
            return Q8_1Tensor::quantize_from_fp32(values.data(), shape);
        }

        /** Export one native-precision logical token from a CPU KV cache. */
        std::pair<std::vector<uint8_t>, std::vector<uint8_t>> exportToken(
            const IKVCache &cache,
            int logical_token)
        {
            const auto layout = cache.logicalBlockLayout(
                /*global_layer=*/0,
                /*token_count=*/1);
            EXPECT_FALSE(layout.device_resident);
            EXPECT_EQ(layout.k_precision, ActivationPrecision::Q8_1);
            EXPECT_EQ(layout.v_precision, ActivationPrecision::Q8_1);
            EXPECT_GT(layout.k_bytes, 0u);
            EXPECT_GT(layout.v_bytes, 0u);

            std::vector<uint8_t> k_bytes(layout.k_bytes);
            std::vector<uint8_t> v_bytes(layout.v_bytes);
            const IKVCache::KVCacheLogicalBlockDescriptor descriptor{
                .layer = 0,
                .seq_idx = 0,
                .logical_token_start = logical_token,
                .token_count = 1,
                .stream = nullptr,
                .payload_domain =
                    IKVCache::KVCacheLogicalBlockPayloadDomain::Host,
            };
            EXPECT_TRUE(cache.exportLogicalBlock(
                descriptor,
                k_bytes.data(),
                v_bytes.data()));
            return {std::move(k_bytes), std::move(v_bytes)};
        }
    }

    /**
     * @brief Append 256K plus a padded-tail boundary with no M holes.
     *
     * The scheduler must produce 64 full buckets and one real row in the tail.
     * IKVCache then receives only each plan's real row count. Native-byte
     * equality between the first and terminal tokens proves that the final row
     * landed at logical index 262144 and was encoded identically; the cached
     * count proves that the 4095 padding rows were never committed.
     */
    TEST(Test__CPUPrefillLongContextMTotality,
         Qwen25_05BGeometry256KPlusTailUsesEveryRealRowExactlyOnce)
    {
        static_assert(kQwen25_05BHiddenSize == 896);
        static_assert(kQwen25_05BKVDim == 128);

        PrefillChunkSchedulerPolicy policy;
        policy.bucket_sizes = {kPrefillBucketRows};
        policy.fixed_chunk_real_tokens = kPrefillBucketRows;
        policy.real_token_count = kLongContextRows;
        const PrefillChunkSchedule schedule =
            planPrefillChunkSchedule(policy);
        ASSERT_TRUE(schedule) << schedule.error;
        ASSERT_EQ(schedule.chunks.size(), 65u);

        int expected_offset = 0;
        uint64_t total_real_rows = 0;
        for (size_t index = 0; index < schedule.chunks.size(); ++index)
        {
            const auto &chunk = schedule.chunks[index];
            SCOPED_TRACE(index);
            EXPECT_EQ(chunk.chunk_index, static_cast<int>(index));
            EXPECT_EQ(chunk.token_offset, expected_offset);
            EXPECT_EQ(chunk.bucket_seq_len, kPrefillBucketRows);
            EXPECT_EQ(
                chunk.real_count,
                index + 1 == schedule.chunks.size()
                    ? 1
                    : kPrefillBucketRows);
            expected_offset += chunk.real_count;
            total_real_rows += static_cast<uint64_t>(chunk.real_count);
        }
        ASSERT_EQ(expected_offset, kLongContextRows);
        ASSERT_EQ(total_real_rows, static_cast<uint64_t>(kLongContextRows));

        MPIContext mpi = singleRankContext();
        llaminar::v2::kernels::KVCacheConfig config;
        config.precision = ActivationPrecision::Q8_1;
        config.device = DeviceId::cpu();
        config.num_layers = 1;
        config.batch_size = 1;
        config.max_seq_len = kLongContextRows;
        config.n_kv_heads = kQwen25_05BKVHeads;
        config.head_dim = kQwen25_05BHeadDim;
        config.mpi_ctx = &mpi;
        auto cache =
            llaminar::v2::kernels::KernelFactory::createCPUKVCache(config);
        ASSERT_NE(cache, nullptr);

        const auto k_rows = makeKVRows(0.25f);
        const auto v_rows = makeKVRows(-0.5f);
        for (const auto &chunk : schedule.chunks)
        {
            ASSERT_TRUE(cache->append(
                /*layer_idx=*/0,
                /*seq_idx=*/0,
                k_rows.get(),
                v_rows.get(),
                chunk.real_count))
                << "chunk=" << chunk.chunk_index
                << " offset=" << chunk.token_offset
                << " real_rows=" << chunk.real_count;
        }

        EXPECT_EQ(cache->get_cached_tokens(0, 0), kLongContextRows);
        const auto state = cache->sequenceState(0, 0);
        EXPECT_EQ(state.cached_tokens, kLongContextRows);
        EXPECT_FALSE(state.wrapped);

        const auto first = exportToken(*cache, 0);
        const auto terminal = exportToken(*cache, kLongContextRows - 1);
        EXPECT_EQ(first.first, terminal.first);
        EXPECT_EQ(first.second, terminal.second);

        const auto penultimate = exportToken(*cache, kLongContextRows - 2);
        EXPECT_NE(penultimate.first, terminal.first)
            << "The last full-bucket row and one-token tail must remain distinct.";
        EXPECT_NE(penultimate.second, terminal.second)
            << "The last full-bucket V row and one-token tail must remain distinct.";
    }
}
