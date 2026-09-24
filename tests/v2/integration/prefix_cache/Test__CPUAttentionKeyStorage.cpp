/**
 * @file Test__CPUAttentionKeyStorage.cpp
 * @brief Native AQ8 storage, immutable geometry and complete-basis copy proofs.
 *
 * The independent scalar codec authors each native payload. CPU SIMD decoding
 * then reads through the actual tensor abstraction for both physical layouts.
 * Whole-plane copies must retain their request anchor; generic FP32 access and
 * layout relabeling are forbidden so neither can hide an incompatible reader.
 */
#include <gtest/gtest.h>

#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../utils/TestTensorFactory.h"
#include "tensors/AttentionKeyQ8Tensor.h"
#include "kernels/kvcache/AttentionKeyQ8Reference.h"

#include <array>
#include <cstring>
#include <limits>

namespace llaminar2::test
{
namespace
{
using llaminar::v2::kernels::KernelFactory;
using llaminar::v2::kernels::KVCacheConfig;

/** @brief Serialize the complete logical key/value state, including its basis. */
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> logicalBytes(ICPUKVCache &cache)
{
    const int rows = cache.get_cached_tokens(0, 0);
    const auto layout = cache.logicalBlockLayout(0, rows);
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> bytes{
        std::vector<uint8_t>(layout.k_bytes), std::vector<uint8_t>(layout.v_bytes)};
    if (!cache.exportLogicalBlock({0, 0, 0, rows, nullptr}, bytes.first.data(), bytes.second.data()))
        throw std::runtime_error("native cache export failed");
    return bytes;
}

/** @brief Native factory, serial/grouped append, wrap and exact physical BOM. */
TEST(CPUAttentionKeyStorage, AllCompressedFactoriesPreserveSerialGroupedBytesAndAccounting)
{
    MPIContext mpi(0, 1, MPI_COMM_WORLD);
    for (int dimension : {64, 128, 256})
    {
        TurboQuantContext context(dimension, 42);
        for (auto precision : {ActivationPrecision::Q8_1, ActivationPrecision::TQ4, ActivationPrecision::TQ8})
            for (auto layout : {KVCacheLayoutMode::POSITION_MAJOR, KVCacheLayoutMode::HEAD_MAJOR})
            {
                KVCacheConfig config;
                config.precision = precision;
                config.num_layers = 1;
                config.batch_size = 1;
                config.max_seq_len = 19;
                config.n_kv_heads = 6;
                config.local_n_kv_heads = 2;
                config.kv_head_start = 2;
                config.head_dim = dimension;
                config.layout_mode = layout;
                config.mpi_ctx = &mpi;
                config.turboquant_ctx = &context;
                for (int prior : {0, 7, 19})
                    for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 64})
                    {
                        SCOPED_TRACE(::testing::Message() << dimension << '/' << static_cast<int>(precision)
                            << '/' << static_cast<int>(layout) << '/' << prior << '/' << rows);
                        auto grouped = KernelFactory::createCPUKVCache(config);
                        auto serial = KernelFactory::createCPUKVCache(config);
                        ASSERT_EQ(grouped->k_precision(), ActivationPrecision::AQ8);
                        ASSERT_EQ(grouped->v_precision(), precision);
                        const size_t width = 2 * dimension;
                        auto keys = TestTensorFactory::createFP32Random({static_cast<size_t>(prior + rows), width}, -3, 3, 91);
                        auto values = TestTensorFactory::createFP32Random(keys->shape(), -1, 1, 97);
                        auto one_k = TestTensorFactory::createFP32({1, width});
                        auto one_v = TestTensorFactory::createFP32({1, width});
                        if (prior > 0)
                        {
                            ASSERT_TRUE(grouped->append(0, 0, keys.get(), values.get(), prior));
                            const auto bytes = logicalBytes(*grouped);
                            ASSERT_TRUE(serial->importLogicalBlock({0, 0, 0, prior, nullptr}, bytes.first.data(), bytes.second.data()));
                        }
                        auto batch_k = TestTensorFactory::createFP32({static_cast<size_t>(rows), width});
                        auto batch_v = TestTensorFactory::createFP32(batch_k->shape());
                        std::memcpy(batch_k->mutable_data(), keys->data() + prior * width, rows * width * sizeof(float));
                        std::memcpy(batch_v->mutable_data(), values->data() + prior * width, rows * width * sizeof(float));
                        if (rows == 1)
                            ASSERT_TRUE(grouped->append(0, 0, batch_k.get(), batch_v.get(), rows));
                        else
                            ASSERT_TRUE(grouped->appendVerifierRowsDecodeEquivalent(0, 0, batch_k.get(), batch_v.get(), rows, nullptr));
                        for (int row = 0; row < rows; ++row)
                        {
                            std::memcpy(one_k->mutable_data(), batch_k->data() + row * width, width * sizeof(float));
                            std::memcpy(one_v->mutable_data(), batch_v->data() + row * width, width * sizeof(float));
                            ASSERT_TRUE(serial->append(0, 0, one_k.get(), one_v.get(), 1));
                        }
                        EXPECT_EQ(grouped->ring_head(0), serial->ring_head(0));
                        EXPECT_EQ(logicalBytes(*grouped), logicalBytes(*serial));
                        EXPECT_EQ(grouped->get_k(0)->size_bytes() + grouped->get_v(0)->size_bytes(), config.estimateBytes());
                    }
            }
    }
}

/** @brief Cold prefill and prefix-restored chunks must select the same key basis. */
TEST(CPUAttentionKeyStorage, InitialPrefillSegmentationCannotChangeNativeBytes)
{
    MPIContext mpi(0, 1, MPI_COMM_WORLD);
    TurboQuantContext context(64, 42);
    for (auto precision : {ActivationPrecision::Q8_1, ActivationPrecision::TQ4, ActivationPrecision::TQ8})
    {
        auto cold = createCPURingKVCache(precision, mpi, 1, 1, 256, 2, 64,
            DeviceId::cpu(), KVCacheLayoutMode::POSITION_MAJOR, &context);
        auto segmented = createCPURingKVCache(precision, mpi, 1, 1, 256, 2, 64,
            DeviceId::cpu(), KVCacheLayoutMode::POSITION_MAJOR, &context);
        auto key = TestTensorFactory::createFP32Random({158, 128}, -3, 3, 53);
        auto value = TestTensorFactory::createFP32Random({158, 128}, -1, 1, 59);
        auto suffix_k = TestTensorFactory::createFP32({19, 128});
        auto suffix_v = TestTensorFactory::createFP32(suffix_k->shape());
        std::memcpy(suffix_k->mutable_data(), key->data() + 139 * 128, 19 * 128 * sizeof(float));
        std::memcpy(suffix_v->mutable_data(), value->data() + 139 * 128, 19 * 128 * sizeof(float));
        ASSERT_TRUE(cold->append(0, 0, key.get(), value.get(), 158));
        ASSERT_TRUE(segmented->append(0, 0, key.get(), value.get(), 139));
        ASSERT_TRUE(segmented->append(0, 0, suffix_k.get(), suffix_v.get(), 19));
        EXPECT_EQ(logicalBytes(*cold), logicalBytes(*segmented));
    }
}

/** @brief Restore rejects malformed or foreign bases before touching live bytes. */
TEST(CPUAttentionKeyStorage, PrefixBasisIsImmutableUntilResetAndInvalidImportsDoNotMutate)
{
    MPIContext mpi(0, 1, MPI_COMM_WORLD);
    TurboQuantContext context(64, 42);
    for (auto precision : {ActivationPrecision::Q8_1, ActivationPrecision::TQ4, ActivationPrecision::TQ8})
    {
        auto cache = createCPURingKVCache(precision, mpi, 1, 1, 32, 2, 64,
            DeviceId::cpu(), KVCacheLayoutMode::POSITION_MAJOR, &context);
        auto key = TestTensorFactory::createFP32Random({7, 128}, -3, 3, 53);
        auto value = TestTensorFactory::createFP32Random({7, 128}, -1, 1, 59);
        ASSERT_TRUE(cache->append(0, 0, key.get(), value.get(), 7));
        const auto original = logicalBytes(*cache);
        const auto descriptor = IKVCache::KVCacheLogicalBlockDescriptor{0, 0, 0, 7, nullptr};
        for (int corruption = 0; corruption < 4; ++corruption)
        {
            auto malformed = original.first;
            ASSERT_GT(malformed.size(), 128 * sizeof(float) + sizeof(float));
            float invalid = corruption == 0 ? std::numeric_limits<float>::quiet_NaN() : -1.0f;
            if (corruption < 2)
                std::memcpy(malformed.data() + (corruption == 0 ? 0 : 128 * sizeof(float)), &invalid, sizeof(float));
            else if (corruption == 2)
                malformed[128 * sizeof(float) + sizeof(float)] = 128; // Reserved signed code.
            else
                malformed[0] ^= 1; // A different finite basis cannot relabel live keys.
            EXPECT_FALSE(cache->importLogicalBlock(descriptor, malformed.data(), original.second.data()));
            EXPECT_EQ(logicalBytes(*cache), original);
        }
        const auto *address = cache->get_k(0)->raw_data();
        ASSERT_TRUE(cache->truncateSequence(0, 0, nullptr));
        ASSERT_TRUE(cache->importLogicalBlock(descriptor, original.first.data(), original.second.data()));
        EXPECT_EQ(cache->get_k(0)->raw_data(), address);
        EXPECT_EQ(logicalBytes(*cache), original);
    }
}

/** @brief Build native bytes independently and verify every diagnostic span. */
template <int D>
void verifyStorage(TensorLayout layout)
{
    constexpr std::size_t positions = 17, heads = 3;
    AttentionKeyQ8Tensor tensor(positions, heads, D, layout);
    const std::size_t extent = heads * D * sizeof(float) + positions * heads * sizeof(AttentionKeyQ8Block<D>);
    ASSERT_EQ(tensor.size_bytes(), extent);
    ASSERT_EQ(tensor.native_type(), TensorType::AQ8);
    ASSERT_EQ(tensor.layout(), layout);
    auto *raw = static_cast<uint8_t *>(tensor.raw_mutable_data());
    const auto original_address = raw;
    auto *basis = reinterpret_cast<float *>(raw);
    for (std::size_t coordinate = 0; coordinate < heads * D; ++coordinate)
        basis[coordinate] = static_cast<float>(coordinate % 23) * 0.125f;
    auto *blocks = reinterpret_cast<AttentionKeyQ8Block<D> *>(raw + tensor.anchor_bytes());
    std::vector<float> expected(tensor.numel());
    for (std::size_t position = 0; position < positions; ++position)
    {
        for (std::size_t head = 0; head < heads; ++head)
        {
            std::array<float, D> residual{}, decoded{};
            for (int coordinate = 0; coordinate < D; ++coordinate)
                residual[coordinate] = static_cast<float>(static_cast<int>((position * 7 + head * 13 + coordinate) % 31) - 15) * 0.03125f;
            residual[position % D] = position % 2 ? -128.0f : 128.0f;
            const auto block = tensor.block_index(position, head);
            attentionKeyQ8QuantizeReference<D>(residual, blocks[block]);
            attentionKeyQ8DequantizeReference<D>(blocks[block], decoded);
            for (int coordinate = 0; coordinate < D; ++coordinate)
                expected[block * D + coordinate] = decoded[coordinate] + basis[head * D + coordinate];
        }
    }
    std::vector<float> actual(tensor.numel());
    tensor.to_fp32(actual.data());
    ASSERT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
    for (std::size_t row = 0; row < tensor.shape()[0]; ++row)
    {
        tensor.to_fp32_row(row, actual.data());
        ASSERT_EQ(std::memcmp(expected.data() + row * tensor.shape()[1], actual.data(), tensor.shape()[1] * sizeof(float)), 0);
    }
    // Cross head and token boundaries, including empty/end spans. These checks
    // catch accidental use of a neighboring head's anchor after byte slicing.
    for (std::size_t offset = 0; offset <= tensor.numel(); offset += D - 1)
    {
        const auto count = std::min<std::size_t>(3 * D + 1, tensor.numel() - offset);
        tensor.to_fp32_span(offset, count, actual.data());
        ASSERT_EQ(std::memcmp(expected.data() + offset, actual.data(), count * sizeof(float)), 0);
    }
    tensor.to_fp32_span(tensor.numel(), 0, nullptr);
    EXPECT_EQ(tensor.raw_data(), original_address);
    EXPECT_EQ(tensor.size_bytes(), extent);
    EXPECT_THROW(tensor.data(), std::logic_error);
    EXPECT_THROW(tensor.mutable_data(), std::logic_error);
    EXPECT_THROW(tensor.createGemm(), std::logic_error);
    EXPECT_THROW(tensor.create_view({1, D}, 0), std::logic_error);

    AttentionKeyQ8Tensor copy(positions, heads, D, layout);
    auto *copy_address = copy.raw_mutable_data();
    ASSERT_TRUE(copy.copyFrom(&tensor));
    EXPECT_EQ(copy.raw_data(), copy_address);
    EXPECT_EQ(std::memcmp(copy.raw_data(), tensor.raw_data(), extent), 0);
    copy.to_fp32(actual.data());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
}

TEST(CPUAttentionKeyStorage, EveryHeadWidthAndLayoutRetainsBasisAndBytes)
{
    for (const auto layout : {TensorLayout::KV_POS_HEAD_DIM, TensorLayout::KV_HEAD_POS_DIM})
    {
        verifyStorage<64>(layout); verifyStorage<128>(layout); verifyStorage<256>(layout);
    }
}

TEST(CPUAttentionKeyStorage, GeometryMismatchCannotMutateDestination)
{
    AttentionKeyQ8Tensor destination(17, 2, 64, TensorLayout::KV_POS_HEAD_DIM);
    std::memset(destination.raw_mutable_data(), 0x5a, destination.size_bytes());
    std::vector<uint8_t> before(destination.size_bytes());
    std::memcpy(before.data(), destination.raw_data(), before.size());
    for (const auto layout : {TensorLayout::KV_POS_HEAD_DIM, TensorLayout::KV_HEAD_POS_DIM})
    {
        AttentionKeyQ8Tensor wrong_heads(17, 3, 64, layout);
        AttentionKeyQ8Tensor wrong_positions(16, 2, 64, layout);
        AttentionKeyQ8Tensor wrong_width(17, 2, 128, layout);
        EXPECT_FALSE(destination.copyFrom(&wrong_heads));
        EXPECT_FALSE(destination.copyFrom(&wrong_positions));
        EXPECT_FALSE(destination.copyFrom(&wrong_width));
    }
    AttentionKeyQ8Tensor wrong_layout(17, 2, 64, TensorLayout::KV_HEAD_POS_DIM);
    EXPECT_FALSE(destination.copyFrom(&wrong_layout));
    EXPECT_THROW(destination.setLayout(TensorLayout::KV_HEAD_POS_DIM), std::invalid_argument);
    EXPECT_NO_THROW(destination.setLayout(TensorLayout::KV_POS_HEAD_DIM));
    EXPECT_EQ(std::memcmp(before.data(), destination.raw_data(), before.size()), 0);
}

TEST(CPUAttentionKeyStorage, InvalidGeometryAndDiagnosticRangesFailBeforeAccess)
{
    const auto layout = TensorLayout::KV_POS_HEAD_DIM;
    EXPECT_THROW(AttentionKeyQ8Tensor(0, 2, 64, layout), std::invalid_argument);
    EXPECT_THROW(AttentionKeyQ8Tensor(1, 0, 64, layout), std::invalid_argument);
    EXPECT_THROW(AttentionKeyQ8Tensor(1, 2, 32, layout), std::invalid_argument);
    EXPECT_THROW(AttentionKeyQ8Tensor(1, 2, 64, TensorLayout::UNKNOWN), std::invalid_argument);
    EXPECT_THROW(AttentionKeyQ8Tensor(std::numeric_limits<std::size_t>::max(), 2, 64, layout), std::overflow_error);
    EXPECT_THROW(AttentionKeyQ8Tensor(1, std::numeric_limits<std::size_t>::max(), 64, layout), std::overflow_error);
    AttentionKeyQ8Tensor tensor(1, 2, 64, layout);
    std::array<float, 128> destination{};
    EXPECT_THROW(tensor.to_fp32_span(128, 1, destination.data()), std::out_of_range);
    EXPECT_THROW(tensor.to_fp32_span(129, 0, destination.data()), std::out_of_range);
    EXPECT_THROW(tensor.to_fp32_span(0, 1, nullptr), std::out_of_range);
    EXPECT_THROW(tensor.to_fp32_row(1, destination.data()), std::out_of_range);
    EXPECT_THROW(tensor.block_index(1, 0), std::out_of_range);
    EXPECT_THROW(tensor.block_index(0, 2), std::out_of_range);
}
} // namespace
} // namespace llaminar2::test
