/**
 * @file Test__CPUQ16CacheRange.cpp
 * @brief Native Q16 cache range, prefix-byte identity and accumulator safety.
 *
 * These model-free tests use production append and attention stages. Small
 * values must retain Q16 precision without a model-wide range assumption, and
 * full-range native keys must not overflow the integer score reduction. The
 * expected attention is a separate FP64 equation, not another native kernel.
 */
#include <gtest/gtest.h>

#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUKVCache.h"
#include "kernels/cpu/rotation/ActivationRotation.h"
#include "utils/MPIContext.h"
#include "../../utils/TestTensorFactory.h"
#include "../../utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace llaminar2::test
{
namespace
{
using llaminar::v2::kernels::KernelFactory;
using llaminar::v2::kernels::KVCacheConfig;

/**
 * @brief Real Q16 append owns a local scale in every physical block.
 *
 * The Qwen-wide fixed range previously rounded small values to zero. Compare
 * the actual cache against original projections and compare whole/serial
 * publication plus prefix round trips as exact native bytes.
 */
TEST(CPUQ16CacheRange, IndependentBlockScalesPreserveSmallValuesAndPrefixBytes)
{
    MPIContext mpi(0, 1, MPI_COMM_WORLD);
    constexpr int rows = 9;
    for (const int dim : {32, 64, 128, 256})
    {
        for (const float magnitude : {0.001f, 1.0f, 1024.0f})
        {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim << " magnitude=" << magnitude);
            KVCacheConfig config;
            config.precision = ActivationPrecision::Q16_1;
            config.num_layers = 1;
            config.max_seq_len = rows + 3;
            config.n_kv_heads = 1;
            config.head_dim = dim;
            config.mpi_ctx = &mpi;
            auto whole = KernelFactory::createCPUKVCache(config);
            auto serial = KernelFactory::createCPUKVCache(config);
            auto restored = KernelFactory::createCPUKVCache(config);
            auto key = TestTensorFactory::createFP32Random({rows, size_t(dim)}, -magnitude, magnitude, 17);
            auto value = TestTensorFactory::createFP32Random({rows, size_t(dim)}, -magnitude / 100, magnitude / 100, 29);

            const auto append = [&](ICPUKVCache &cache, int begin, int count)
            {
                auto k = TestTensorFactory::createFP32({size_t(count), size_t(dim)});
                auto v = TestTensorFactory::createFP32(k->shape());
                std::copy_n(key->data() + begin * dim, count * dim, k->mutable_data());
                std::copy_n(value->data() + begin * dim, count * dim, v->mutable_data());
                KVCacheAppendStage::Params params;
                params.K = k.get();
                params.V = v.get();
                params.kv_cache = &cache;
                params.num_tokens = count;
                params.seq_len = count;
                params.head_dim = dim;
                // Existing model hints must not erase a native block's range.
                params.kv_cache_scale_k = 1024.0f;
                params.kv_cache_scale_v = 256.0f;
                KVCacheAppendStage stage(params);
                ASSERT_TRUE(stage.execute(nullptr));
            };
            ASSERT_NO_FATAL_FAILURE(append(*whole, 0, rows));
            for (int row = 0; row < rows; ++row)
                ASSERT_NO_FATAL_FAILURE(append(*serial, row, 1));

            const ITensor *native_k = nullptr, *native_v = nullptr;
            int count = 0;
            ASSERT_TRUE(whole->get_kv(0, 0, &native_k, &native_v, &count));
            ASSERT_EQ(count, rows);
            const auto *k16 = dynamic_cast<const Q16_1Tensor *>(native_k);
            const auto *v16 = dynamic_cast<const Q16_1Tensor *>(native_v);
            ASSERT_NE(k16, nullptr);
            ASSERT_NE(v16, nullptr);
            std::vector<float> decoded(dim);
            for (int row = 0; row < rows; ++row)
            {
                k16->to_fp32_row(row, decoded.data());
                for (int col = 0; col < dim; ++col)
                    ASSERT_NEAR(decoded[col], key->data()[row * dim + col], magnitude * 4e-5f);
                v16->to_fp32_row(row, decoded.data());
                for (int col = 0; col < dim; ++col)
                    ASSERT_NEAR(decoded[col], value->data()[row * dim + col], magnitude * 4e-7f);
            }

            const IKVCache::KVCacheLogicalBlockDescriptor block{0, 0, 0, rows, nullptr};
            const auto layout = whole->logicalBlockLayout(0, rows);
            std::vector<uint8_t> keys(layout.k_bytes), values(layout.v_bytes);
            std::vector<uint8_t> other_keys(layout.k_bytes), other_values(layout.v_bytes);
            ASSERT_TRUE(whole->exportLogicalBlock(block, keys.data(), values.data()));
            ASSERT_TRUE(serial->exportLogicalBlock(block, other_keys.data(), other_values.data()));
            EXPECT_EQ(keys, other_keys);
            EXPECT_EQ(values, other_values);
            ASSERT_TRUE(restored->importLogicalBlock(block, keys.data(), values.data()));
            ASSERT_TRUE(restored->exportLogicalBlock(block, other_keys.data(), other_values.data()));
            EXPECT_EQ(keys, other_keys);
            EXPECT_EQ(values, other_values);
        }
    }
}

/**
 * @brief Request batching cannot change rotation, scales, or native cache bytes.
 *
 * Two heads expose head-major storage and two requests expose request slicing.
 * Also feed pre-encoded native operands: native V must remain a byte copy,
 * not undergo a second rotation or a lossy decode/re-encode cycle.
 */
TEST(CPUQ16CacheRange, RotatedBatchedAndNativePublicationMatchSeparateRequests)
{
    MPIContext mpi(0, 1, MPI_COMM_WORLD);
    constexpr int requests = 2, rows = 9, heads = 2;
    for (const int dim : {32, 64, 128, 256})
    {
        SCOPED_TRACE(dim);
        const size_t width = heads * dim;
        KVCacheConfig config;
        config.precision = ActivationPrecision::Q16_1;
        config.num_layers = 1;
        config.batch_size = requests;
        config.max_seq_len = rows + 7;
        config.n_kv_heads = heads;
        config.head_dim = dim;
        config.mpi_ctx = &mpi;
        auto batched = KernelFactory::createCPUKVCache(config);
        auto separate = KernelFactory::createCPUKVCache(config);
        auto native = KernelFactory::createCPUKVCache(config);
        ActivationRotation rotation(width, dim, 42);
        auto key = TestTensorFactory::createFP32Random({requests * rows, width}, -0.1f, 0.1f, 81);
        auto value = TestTensorFactory::createFP32Random(key->shape(), -0.01f, 0.01f, 91);
        auto decoded = TestTensorFactory::createFP32(key->shape());
        const auto publish = [&](ICPUKVCache &cache, const ITensor &k,
                                 const ITensor &v, int batch, int request,
                                 FP32Tensor *output)
        {
            KVCacheAppendStage::Params params;
            params.K = &k;
            params.V = &v;
            params.kv_cache = &cache;
            params.num_tokens = batch * rows;
            params.batch_size = batch;
            params.seq_idx = request;
            params.seq_len = rows;
            params.head_dim = dim;
            params.kv_rotation = &rotation;
            params.V_dequant_out = output;
            KVCacheAppendStage stage(params);
            ASSERT_TRUE(stage.execute(nullptr));
        };
        ASSERT_NO_FATAL_FAILURE(publish(*batched, *key, *value, requests, 0, decoded.get()));
        for (int request = 0; request < requests; ++request)
        {
            auto k = TestTensorFactory::createFP32({rows, width});
            auto v = TestTensorFactory::createFP32(k->shape());
            std::copy_n(key->data() + request * rows * width, rows * width, k->mutable_data());
            std::copy_n(value->data() + request * rows * width, rows * width, v->mutable_data());
            ASSERT_NO_FATAL_FAILURE(publish(*separate, *k, *v, 1, request, nullptr));
        }

        std::vector<float> k_rotated(key->data(), key->data() + key->numel());
        std::vector<float> v_rotated(value->data(), value->data() + value->numel());
        rotation.rotate_rows_inplace(k_rotated.data(), requests * rows, width);
        rotation.rotate_rows_inplace(v_rotated.data(), requests * rows, width);
        Q16_1Tensor encoded_k(key->shape(), optimal_q16_block_size(dim));
        Q16_1Tensor encoded_v(value->shape(), optimal_q16_block_size(dim));
        ASSERT_TRUE(encoded_k.copyFrom_fp32(k_rotated.data()));
        ASSERT_TRUE(encoded_v.copyFrom_fp32(v_rotated.data()));
        ASSERT_NO_FATAL_FAILURE(publish(*native, encoded_k, encoded_v, requests, 0, nullptr));
        for (int row = 0; row < requests * rows; ++row)
            for (size_t col = 0; col < width; ++col)
                ASSERT_EQ(decoded->data()[row * width + col], encoded_v.dequant_element(row, col));

        const auto layout = batched->logicalBlockLayout(0, rows);
        for (int request = 0; request < requests; ++request)
        {
            const IKVCache::KVCacheLogicalBlockDescriptor block{0, request, 0, rows, nullptr};
            std::vector<uint8_t> k(layout.k_bytes), v(layout.v_bytes);
            std::vector<uint8_t> expected_k(layout.k_bytes), expected_v(layout.v_bytes);
            ASSERT_TRUE(batched->exportLogicalBlock(block, expected_k.data(), expected_v.data()));
            for (const auto *cache : {separate.get(), native.get()})
            {
                ASSERT_TRUE(cache->exportLogicalBlock(block, k.data(), v.data()));
                EXPECT_EQ(k, expected_k);
                EXPECT_EQ(v, expected_v);
            }
        }
    }
}

/**
 * @brief Full-range Q16 blocks cannot reverse softmax preference by overflowing.
 *
 * Constant signed rows maximize integer accumulation and have independently
 * predictable scores. Every physical block size and grouped runtime width
 * reaches the same production attention stage as ordinary one-row decode.
 */
TEST(CPUQ16CacheRange, FullRangeKeysKeepEveryGroupedScoreReductionInRange)
{
    constexpr int kv_rows = 4;
    constexpr std::array<float, kv_rows> key_values{1.0f, -1.0f, 0.75f, -0.75f};
    constexpr std::array<float, kv_rows> value_values{1.0f, -1.0f, 0.25f, -0.25f};
    std::vector<int> widths(kGroupedVerifierRuntimeRows.begin(), kGroupedVerifierRuntimeRows.end());
    widths.insert(widths.begin(), 1);
    for (const auto block : {Q16BlockSize::BLOCK_32, Q16BlockSize::BLOCK_64, Q16BlockSize::BLOCK_128})
    {
        for (const int dim : {64, 128, 256})
        {
            if (static_cast<int>(block) > dim) continue;
            std::vector<float> k(kv_rows * dim), v(kv_rows * dim);
            for (int row = 0; row < kv_rows; ++row)
            {
                std::fill_n(k.data() + row * dim, dim, key_values[row]);
                std::fill_n(v.data() + row * dim, dim, value_values[row]);
            }
            Q16_1Tensor keys({kv_rows, size_t(dim)}, block);
            Q16_1Tensor values({kv_rows, size_t(dim)}, block);
            ASSERT_TRUE(keys.copyFrom_fp32(k.data()));
            ASSERT_TRUE(values.copyFrom_fp32(v.data()));
            double weighted = 0.0, denominator = 0.0;
            for (int row = 0; row < kv_rows; ++row)
            {
                const double probability = std::exp((key_values[row] - 1.0) * std::sqrt(double(dim)));
                weighted += probability * value_values[row];
                denominator += probability;
            }
            const float expected = static_cast<float>(weighted / denominator);
            for (const int rows : widths)
            {
                SCOPED_TRACE(::testing::Message() << "block=" << int(block) << " dim=" << dim << " M=" << rows);
                auto query = TestTensorFactory::createFP32({size_t(rows), size_t(dim)});
                auto output = TestTensorFactory::createFP32(query->shape());
                std::fill_n(query->mutable_data(), query->numel(), 1.0f);
                AttentionComputeStage::Params params;
                params.Q = query.get();
                params.K = &keys;
                params.V = &values;
                params.output = output.get();
                params.seq_len = rows;
                params.kv_len = kv_rows;
                params.n_heads = 1;
                params.n_kv_heads = 1;
                params.head_dim = dim;
                params.causal = false;
                AttentionComputeStage stage(params);
                ASSERT_TRUE(stage.execute(nullptr));
                for (size_t i = 0; i < output->numel(); ++i)
                    ASSERT_NEAR(output->data()[i], expected, 2e-5f);
            }
        }
    }
}
} // namespace
} // namespace llaminar2::test
