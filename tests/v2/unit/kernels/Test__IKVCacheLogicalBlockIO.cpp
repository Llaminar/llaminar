#include <gtest/gtest.h>

#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"
#include "utils/PerfStatsCollector.h"
#include "../../utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    MPIContext testMPI()
    {
        return MPIContext(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<FP32Tensor> taggedFP32(int rows, int cols, float base)
    {
        auto tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
        float *data = tensor->mutable_data();
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                    base + static_cast<float>(r * 10 + c);
            }
        }
        return tensor;
    }

    std::vector<uint8_t> taggedBytes(size_t bytes, uint8_t seed)
    {
        std::vector<uint8_t> out(bytes);
        for (size_t i = 0; i < bytes; ++i)
        {
            out[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>(i % 251u));
        }
        return out;
    }

    std::shared_ptr<Q16_1Tensor> taggedQ16Tensor(const std::vector<size_t> &shape,
                                                 Q16BlockSize block_size,
                                                 uint8_t seed)
    {
        auto tensor = std::make_shared<Q16_1Tensor>(shape, block_size);
        auto *bytes = static_cast<uint8_t *>(tensor->raw_mutable_data());
        if (!bytes)
        {
            ADD_FAILURE() << "Q16 test tensor did not expose mutable raw data";
            return tensor;
        }
        for (size_t i = 0; i < tensor->size_bytes(); ++i)
        {
            bytes[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>((i * 17u) % 251u));
        }
        return tensor;
    }

    std::shared_ptr<Q16_1Tensor> q16PositionRowFromHeadMajor(
        const Q16_1Tensor &head_major,
        int verifier_row,
        int verifier_rows,
        int heads,
        int head_dim)
    {
        auto row = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(heads * head_dim)},
            head_major.q16_block_size());
        const size_t head_bytes = q16_block_size_bytes(head_major.q16_block_size());
        const auto *src = static_cast<const uint8_t *>(head_major.raw_data());
        auto *dst = static_cast<uint8_t *>(row->raw_mutable_data());
        EXPECT_NE(src, nullptr);
        EXPECT_NE(dst, nullptr);
        for (int h = 0; h < heads; ++h)
        {
            const size_t src_row = static_cast<size_t>(h) *
                                       static_cast<size_t>(verifier_rows) +
                                   static_cast<size_t>(verifier_row);
            std::memcpy(dst + static_cast<size_t>(h) * head_bytes,
                        src + src_row * head_bytes,
                        head_bytes);
        }
        return row;
    }

    void expectFP32Rows(const std::vector<float> &actual, int rows, int cols,
                        const std::vector<float> &expected_first_values)
    {
        ASSERT_EQ(static_cast<size_t>(rows * cols), actual.size());
        ASSERT_EQ(static_cast<size_t>(rows), expected_first_values.size());
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                EXPECT_FLOAT_EQ(actual[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)],
                                expected_first_values[static_cast<size_t>(r)] + static_cast<float>(c));
            }
        }
    }

    /** @brief Enable perfstats for grouped KV publication route assertions. */
    class ScopedPerfStats
    {
    public:
        ScopedPerfStats()
        {
            const char *old_value = std::getenv("LLAMINAR_PERF_STATS_SUMMARY");
            if (old_value)
            {
                had_old_value_ = true;
                old_value_ = old_value;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_old_value_)
                setenv("LLAMINAR_PERF_STATS_SUMMARY", old_value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

    private:
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /**
     * @brief Allocate a cache-format tensor and fill every storage byte.
     *
     * Grouped append is a storage publication operation, so arbitrary tagged
     * native bytes are a stronger witness than numerically valid values: the
     * test detects any omitted block, layout-stride error, or format-specific
     * padding mistake without involving conversion kernels.
     */
    template <ActivationPrecision Precision>
    std::shared_ptr<typename detail::CPUKVCacheTensor<Precision>::Type>
    taggedCacheTensor(TensorFactory &factory,
                      size_t rows,
                      size_t cols,
                      int head_dim,
                      uint8_t seed)
    {
        using Trait = detail::CPUKVCacheTensor<Precision>;
        auto tensor = Trait::allocate(factory, rows, cols, head_dim, DeviceId::cpu());
        auto *bytes = static_cast<uint8_t *>(tensor->raw_mutable_data());
        EXPECT_NE(bytes, nullptr);
        if (!bytes)
            return tensor;
        for (size_t index = 0; index < tensor->size_bytes(); ++index)
        {
            bytes[index] = static_cast<uint8_t>(
                seed + static_cast<uint8_t>((index * 29u + index / 7u) % 251u));
        }
        return tensor;
    }

    /**
     * @brief Extract one logical verifier position into a position-major row.
     */
    template <ActivationPrecision Precision>
    std::shared_ptr<typename detail::CPUKVCacheTensor<Precision>::Type>
    verifierPositionRow(TensorFactory &factory,
                        const typename detail::CPUKVCacheTensor<Precision>::Type &source,
                        int verifier_row,
                        int verifier_rows,
                        int heads,
                        int head_dim,
                        bool source_head_major)
    {
        using Trait = detail::CPUKVCacheTensor<Precision>;
        const int kv_dim = heads * head_dim;
        auto row = Trait::allocate(
            factory, 1, static_cast<size_t>(kv_dim), head_dim, DeviceId::cpu());
        const size_t row_bytes = Trait::row_bytes(row.get(), kv_dim, head_dim);
        const size_t head_bytes = Trait::head_bytes(row.get(), kv_dim, head_dim);
        const auto *src = static_cast<const uint8_t *>(source.raw_data());
        auto *dst = static_cast<uint8_t *>(row->raw_mutable_data());
        EXPECT_NE(src, nullptr);
        EXPECT_NE(dst, nullptr);
        if (!src || !dst)
            return row;

        if (!source_head_major)
        {
            std::memcpy(dst,
                        src + static_cast<size_t>(verifier_row) * row_bytes,
                        row_bytes);
            return row;
        }

        for (int head = 0; head < heads; ++head)
        {
            const size_t source_row = static_cast<size_t>(head) * verifier_rows +
                                      static_cast<size_t>(verifier_row);
            std::memcpy(dst + static_cast<size_t>(head) * head_bytes,
                        src + source_row * head_bytes,
                        head_bytes);
        }
        return row;
    }

    /** @brief Assert that grouped append used one metadata commit for this matrix cell. */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void expectGroupedAppendCounter(int verifier_rows,
                                    bool source_head_major,
                                    KVCacheLayoutMode destination_layout)
    {
        const std::string expected_source = source_head_major ? "head_major" : "position_major";
        const std::string expected_destination =
            destination_layout == KVCacheLayoutMode::HEAD_MAJOR
                ? "head_major"
                : "position_major";
        bool found = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_kv_cache_grouped_verifier_append_calls"}))
        {
            auto tag_equals = [&](const char *name, const std::string &expected)
            {
                const auto it = record.tags.find(name);
                return it != record.tags.end() && it->second == expected;
            };
            found = found ||
                    (tag_equals("k_format", activationPrecisionToString(KPrecision)) &&
                     tag_equals("v_format", activationPrecisionToString(VPrecision)) &&
                     tag_equals("verifier_rows", std::to_string(verifier_rows)) &&
                     tag_equals("source_k_layout", expected_source) &&
                     tag_equals("source_v_layout", expected_source) &&
                     tag_equals("destination_layout", expected_destination) &&
                     tag_equals("commit_policy", "single_grouped_metadata_commit"));
        }
        EXPECT_TRUE(found)
            << "CPU grouped KV append did not publish the expected route for K="
            << activationPrecisionToString(KPrecision)
            << " V=" << activationPrecisionToString(VPrecision)
            << " M=" << verifier_rows
            << " source=" << expected_source
            << " destination=" << expected_destination << "\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cpu_kv_cache_grouped_verifier_append_calls"}, 20);
    }

    /**
     * @brief Run one K/V format through all grouped append layouts and depths.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision = KPrecision>
    void runGroupedVerifierAppendFormat(const char *format_label)
    {
        constexpr int heads = 2;
        constexpr int head_dim = 64;
        constexpr int kv_dim = heads * head_dim;
        constexpr int prefix_rows = 3;
        constexpr int max_seq_len = 5;

        MPIContext mpi = testMPI();
        TensorFactory factory(mpi);
        uint8_t seed = 17;

        for (KVCacheLayoutMode destination_layout : {
                 KVCacheLayoutMode::POSITION_MAJOR,
                 KVCacheLayoutMode::HEAD_MAJOR})
        {
            for (bool source_head_major : {false, true})
            {
                for (int verifier_rows : test::kGroupedVerifierRuntimeRows)
                {
                    SCOPED_TRACE(std::string(format_label) +
                                 " M=" + std::to_string(verifier_rows) +
                                 " source=" + (source_head_major ? "head" : "position") +
                                 " destination=" +
                                 (destination_layout == KVCacheLayoutMode::HEAD_MAJOR
                                      ? "head"
                                      : "position"));

                    CPURingKVCache<KPrecision, VPrecision> serial(
                        mpi, 1, 1, max_seq_len, heads, head_dim,
                        DeviceId::cpu(), destination_layout);
                    CPURingKVCache<KPrecision, VPrecision> grouped(
                        mpi, 1, 1, max_seq_len, heads, head_dim,
                        DeviceId::cpu(), destination_layout);

                    auto prefix_k = taggedCacheTensor<KPrecision>(
                        factory, prefix_rows, kv_dim, head_dim, seed++);
                    auto prefix_v = taggedCacheTensor<VPrecision>(
                        factory, prefix_rows, kv_dim, head_dim, seed++);
                    ASSERT_TRUE(serial.append_kv(
                        0, 0, prefix_k.get(), prefix_v.get(), prefix_rows));
                    ASSERT_TRUE(grouped.append_kv(
                        0, 0, prefix_k.get(), prefix_v.get(), prefix_rows));

                    const size_t source_rows = source_head_major
                                                   ? static_cast<size_t>(heads * verifier_rows)
                                                   : static_cast<size_t>(verifier_rows);
                    const size_t source_cols = source_head_major
                                                   ? static_cast<size_t>(head_dim)
                                                   : static_cast<size_t>(kv_dim);
                    auto verifier_k = taggedCacheTensor<KPrecision>(
                        factory, source_rows, source_cols, head_dim, seed++);
                    auto verifier_v = taggedCacheTensor<VPrecision>(
                        factory, source_rows, source_cols, head_dim, seed++);

                    for (int row_index = 0; row_index < verifier_rows; ++row_index)
                    {
                        auto k_row = verifierPositionRow<KPrecision>(
                            factory, *verifier_k, row_index, verifier_rows,
                            heads, head_dim, source_head_major);
                        auto v_row = verifierPositionRow<VPrecision>(
                            factory, *verifier_v, row_index, verifier_rows,
                            heads, head_dim, source_head_major);
                        ASSERT_TRUE(serial.append_kv(
                            0, 0, k_row.get(), v_row.get(), 1));
                    }

                    PerfStatsCollector::reset();
                    ASSERT_TRUE(grouped.appendVerifierRowsDecodeEquivalent(
                        0, 0, verifier_k.get(), verifier_v.get(), verifier_rows));
                    expectGroupedAppendCounter<KPrecision, VPrecision>(
                        verifier_rows, source_head_major, destination_layout);

                    const auto serial_state = serial.sequenceState(0, 0);
                    const auto grouped_state = grouped.sequenceState(0, 0);
                    EXPECT_EQ(grouped_state.cached_tokens, serial_state.cached_tokens);
                    EXPECT_EQ(grouped_state.implementation_head, serial_state.implementation_head);
                    EXPECT_EQ(grouped_state.wrapped, serial_state.wrapped);

                    const auto layout = serial.logicalBlockLayout(
                        0, serial_state.cached_tokens);
                    std::vector<uint8_t> serial_k(layout.k_bytes, 0);
                    std::vector<uint8_t> serial_v(layout.v_bytes, 0);
                    std::vector<uint8_t> grouped_k(layout.k_bytes, 0);
                    std::vector<uint8_t> grouped_v(layout.v_bytes, 0);
                    const IKVCache::KVCacheLogicalBlockDescriptor descriptor{
                        0, 0, 0, serial_state.cached_tokens, nullptr};
                    ASSERT_TRUE(serial.exportLogicalBlock(
                        descriptor, serial_k.data(), serial_v.data()));
                    ASSERT_TRUE(grouped.exportLogicalBlock(
                        descriptor, grouped_k.data(), grouped_v.data()));
                    EXPECT_EQ(grouped_k, serial_k);
                    EXPECT_EQ(grouped_v, serial_v);
                }
            }
        }
    }
} // namespace

TEST(Test__IKVCacheLogicalBlockIO, EmptyExportImportContract)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, 1, 2, DeviceId::cpu());

    const auto layout = cache.logicalBlockLayout(0, 0);
    EXPECT_EQ(layout.k_precision, ActivationPrecision::FP32);
    EXPECT_EQ(layout.v_precision, ActivationPrecision::FP32);
    EXPECT_EQ(layout.layout, TensorLayout::KV_POS_HEAD_DIM);
    EXPECT_EQ(layout.local_kv_heads, 1);
    EXPECT_EQ(layout.kv_head_start, 0);
    EXPECT_EQ(layout.head_dim, 2);
    EXPECT_EQ(layout.k_bytes, 0u);
    EXPECT_EQ(layout.v_bytes, 0u);
    EXPECT_FALSE(layout.device_resident);

    const IKVCache::KVCacheLogicalBlockDescriptor empty_desc{0, 0, 0, 0, nullptr};
    EXPECT_TRUE(cache.exportLogicalBlock(empty_desc, nullptr, nullptr));
    EXPECT_TRUE(cache.importLogicalBlock(empty_desc, nullptr, nullptr));
    EXPECT_TRUE(cache.truncateSequence(0, 0));
    EXPECT_FALSE(cache.truncateSequence(0, 1));

    auto state = cache.sequenceState(0, 0);
    EXPECT_EQ(state.cached_tokens, 0);
    EXPECT_EQ(state.implementation_head, 0);
    EXPECT_FALSE(state.wrapped);
}

TEST(Test__IKVCacheLogicalBlockIO, PositionMajorExportsNonWrappedLogicalSlice)
{
    constexpr int KV_DIM = 2;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, 1, KV_DIM, DeviceId::cpu());
    auto k = taggedFP32(4, KV_DIM, 100.0f);
    auto v = taggedFP32(4, KV_DIM, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 4));

    const auto layout = cache.logicalBlockLayout(0, 2);
    ASSERT_EQ(layout.k_bytes, 2u * KV_DIM * sizeof(float));
    std::vector<float> out_k(2 * KV_DIM, 0.0f);
    std::vector<float> out_v(2 * KV_DIM, 0.0f);

    const IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 1, 2, nullptr};
    ASSERT_TRUE(cache.exportLogicalBlock(desc, out_k.data(), out_v.data()));

    expectFP32Rows(out_k, 2, KV_DIM, {110.0f, 120.0f});
    expectFP32Rows(out_v, 2, KV_DIM, {210.0f, 220.0f});

    const auto state = cache.sequenceState(0, 0);
    EXPECT_EQ(state.cached_tokens, 4);
    EXPECT_EQ(state.implementation_head, 0);
    EXPECT_FALSE(state.wrapped);
}

TEST(Test__IKVCacheLogicalBlockIO, PositionMajorExportsWrappedLogicalOrder)
{
    constexpr int KV_DIM = 2;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, 1, KV_DIM, DeviceId::cpu());
    auto k0 = taggedFP32(4, KV_DIM, 100.0f);
    auto v0 = taggedFP32(4, KV_DIM, 200.0f);
    auto k1 = taggedFP32(2, KV_DIM, 140.0f);
    auto v1 = taggedFP32(2, KV_DIM, 240.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 2));

    auto state = cache.sequenceState(0, 0);
    ASSERT_EQ(state.cached_tokens, 4);
    ASSERT_EQ(state.implementation_head, 2);
    EXPECT_TRUE(state.wrapped);

    std::vector<float> out_k(4 * KV_DIM, 0.0f);
    std::vector<float> out_v(4 * KV_DIM, 0.0f);
    const IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 0, 4, nullptr};
    ASSERT_TRUE(cache.exportLogicalBlock(desc, out_k.data(), out_v.data()));

    expectFP32Rows(out_k, 4, KV_DIM, {120.0f, 130.0f, 140.0f, 150.0f});
    expectFP32Rows(out_v, 4, KV_DIM, {220.0f, 230.0f, 240.0f, 250.0f});
}

TEST(Test__IKVCacheLogicalBlockIO, ImportsIntoEmptyAndAfterRequestReset)
{
    constexpr int KV_DIM = 2;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, 1, KV_DIM, DeviceId::cpu());

    std::vector<float> k_payload = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v_payload = {10.0f, 20.0f, 30.0f, 40.0f};
    const IKVCache::KVCacheLogicalBlockDescriptor two_tokens{0, 0, 0, 2, nullptr};
    ASSERT_TRUE(cache.importLogicalBlock(two_tokens, k_payload.data(), v_payload.data()));
    EXPECT_EQ(cache.sequenceState(0, 0).cached_tokens, 2);

    std::vector<float> out_k(2 * KV_DIM, 0.0f);
    std::vector<float> out_v(2 * KV_DIM, 0.0f);
    ASSERT_TRUE(cache.exportLogicalBlock(two_tokens, out_k.data(), out_v.data()));
    EXPECT_EQ(out_k, k_payload);
    EXPECT_EQ(out_v, v_payload);

    ASSERT_TRUE(cache.resetRequestState(
        IKVCache::StateResetContext::testReinitialization(nullptr)));
    std::vector<float> k_payload_2 = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    std::vector<float> v_payload_2 = {50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f};
    const IKVCache::KVCacheLogicalBlockDescriptor three_tokens{0, 0, 0, 3, nullptr};
    ASSERT_TRUE(cache.importLogicalBlock(three_tokens, k_payload_2.data(), v_payload_2.data()));

    out_k.assign(3 * KV_DIM, 0.0f);
    out_v.assign(3 * KV_DIM, 0.0f);
    ASSERT_TRUE(cache.exportLogicalBlock(three_tokens, out_k.data(), out_v.data()));
    EXPECT_EQ(out_k, k_payload_2);
    EXPECT_EQ(out_v, v_payload_2);
}

TEST(Test__IKVCacheLogicalBlockIO, ImportsLogicalBlocksSequentiallyAtNonZeroOffsets)
{
    constexpr int KV_DIM = 2;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 6, 1, KV_DIM, DeviceId::cpu());

    std::vector<float> first_k = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> first_v = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> second_k = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> second_v = {50.0f, 60.0f, 70.0f, 80.0f};

    const IKVCache::KVCacheLogicalBlockDescriptor first{0, 0, 0, 2, nullptr};
    const IKVCache::KVCacheLogicalBlockDescriptor second{0, 0, 2, 2, nullptr};
    ASSERT_TRUE(cache.importLogicalBlock(first, first_k.data(), first_v.data()));
    ASSERT_TRUE(cache.importLogicalBlock(second, second_k.data(), second_v.data()));

    EXPECT_EQ(cache.sequenceState(0, 0).cached_tokens, 4);
    EXPECT_EQ(cache.sequenceState(0, 0).implementation_head, 0);

    std::vector<float> out_k(4 * KV_DIM, 0.0f);
    std::vector<float> out_v(4 * KV_DIM, 0.0f);
    const IKVCache::KVCacheLogicalBlockDescriptor all{0, 0, 0, 4, nullptr};
    ASSERT_TRUE(cache.exportLogicalBlock(all, out_k.data(), out_v.data()));

    const std::vector<float> expected_k = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> expected_v = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    EXPECT_EQ(out_k, expected_k);
    EXPECT_EQ(out_v, expected_v);
}

TEST(Test__IKVCacheLogicalBlockIO, TruncateKeepsOldestPrefixAndRejectsLongerLength)
{
    constexpr int KV_DIM = 2;
    CPURingKVCacheFP32 cache(testMPI(), 2, 1, 4, 1, KV_DIM, DeviceId::cpu());
    auto k0 = taggedFP32(4, KV_DIM, 100.0f);
    auto v0 = taggedFP32(4, KV_DIM, 200.0f);
    auto k1 = taggedFP32(2, KV_DIM, 140.0f);
    auto v1 = taggedFP32(2, KV_DIM, 240.0f);
    for (int layer = 0; layer < 2; ++layer)
    {
        ASSERT_TRUE(cache.append_kv(layer, 0, k0.get(), v0.get(), 4));
        ASSERT_TRUE(cache.append_kv(layer, 0, k1.get(), v1.get(), 2));
    }

    ASSERT_TRUE(cache.truncateSequence(0, 2));
    EXPECT_EQ(cache.sequenceState(0, 0).cached_tokens, 2);
    EXPECT_EQ(cache.sequenceState(1, 0).cached_tokens, 2);
    EXPECT_FALSE(cache.truncateSequence(0, 3));

    std::vector<float> out_k(2 * KV_DIM, 0.0f);
    std::vector<float> out_v(2 * KV_DIM, 0.0f);
    const IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 0, 2, nullptr};
    ASSERT_TRUE(cache.exportLogicalBlock(desc, out_k.data(), out_v.data()));
    expectFP32Rows(out_k, 2, KV_DIM, {120.0f, 130.0f});
    expectFP32Rows(out_v, 2, KV_DIM, {220.0f, 230.0f});

    ASSERT_TRUE(cache.truncateSequence(0, 0));
    EXPECT_EQ(cache.sequenceState(0, 0).cached_tokens, 0);
    EXPECT_EQ(cache.sequenceState(0, 0).implementation_head, 0);
}

TEST(Test__IKVCacheLogicalBlockIO, HeadMajorExportsAndImportsPackedByHead)
{
    constexpr int HEADS = 2;
    constexpr int HEAD_DIM = 2;
    constexpr int KV_DIM = HEADS * HEAD_DIM;
    CPURingKVCacheFP32 source(testMPI(), 1, 1, 4, HEADS, HEAD_DIM, DeviceId::cpu(),
                              KVCacheLayoutMode::HEAD_MAJOR);

    auto k = std::make_shared<FP32Tensor>(std::vector<size_t>{2, KV_DIM});
    auto v = std::make_shared<FP32Tensor>(std::vector<size_t>{2, KV_DIM});
    float *kd = k->mutable_data();
    float *vd = v->mutable_data();
    // token0 h0=[1,2] h1=[10,20]; token1 h0=[3,4] h1=[30,40]
    const float k_values[] = {1.0f, 2.0f, 10.0f, 20.0f, 3.0f, 4.0f, 30.0f, 40.0f};
    const float v_values[] = {101.0f, 102.0f, 110.0f, 120.0f, 103.0f, 104.0f, 130.0f, 140.0f};
    std::copy(k_values, k_values + 8, kd);
    std::copy(v_values, v_values + 8, vd);
    ASSERT_TRUE(source.append_kv(0, 0, k.get(), v.get(), 2));

    const auto layout = source.logicalBlockLayout(0, 2);
    ASSERT_EQ(layout.layout, TensorLayout::KV_HEAD_POS_DIM);
    ASSERT_EQ(layout.k_bytes, HEADS * 2u * HEAD_DIM * sizeof(float));

    std::vector<float> out_k(layout.k_bytes / sizeof(float), 0.0f);
    std::vector<float> out_v(layout.v_bytes / sizeof(float), 0.0f);
    const IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 0, 2, nullptr};
    ASSERT_TRUE(source.exportLogicalBlock(desc, out_k.data(), out_v.data()));

    const std::vector<float> expected_k = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    const std::vector<float> expected_v = {101.0f, 102.0f, 103.0f, 104.0f, 110.0f, 120.0f, 130.0f, 140.0f};
    EXPECT_EQ(out_k, expected_k);
    EXPECT_EQ(out_v, expected_v);

    CPURingKVCacheFP32 target(testMPI(), 1, 1, 4, HEADS, HEAD_DIM, DeviceId::cpu(),
                              KVCacheLayoutMode::HEAD_MAJOR);
    ASSERT_TRUE(target.importLogicalBlock(desc, out_k.data(), out_v.data()));
    std::vector<float> roundtrip_k(out_k.size(), 0.0f);
    std::vector<float> roundtrip_v(out_v.size(), 0.0f);
    ASSERT_TRUE(target.exportLogicalBlock(desc, roundtrip_k.data(), roundtrip_v.data()));
    EXPECT_EQ(roundtrip_k, out_k);
    EXPECT_EQ(roundtrip_v, out_v);
}

TEST(Test__IKVCacheLogicalBlockIO, ReportsShardedKVHeadMetadata)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8,
                             /*n_kv_heads=*/4, /*local_n_kv_heads=*/2,
                             /*kv_head_start=*/2, /*head_dim=*/4, DeviceId::cpu());

    const auto layout = cache.logicalBlockLayout(0, 3);
    EXPECT_EQ(layout.local_kv_heads, 2);
    EXPECT_EQ(layout.kv_head_start, 2);
    EXPECT_EQ(layout.head_dim, 4);
    EXPECT_EQ(layout.k_bytes, 3u * 2u * 4u * sizeof(float));
    EXPECT_FALSE(layout.device_resident);
}

TEST(Test__IKVCacheLogicalBlockIO, Q16HeadMajorLogicalBlockRoundTripsRawBytes)
{
    CPURingKVCacheQ16_1 cache(testMPI(), 1, 1, 4, 2, 64, DeviceId::cpu(),
                              KVCacheLayoutMode::HEAD_MAJOR);
    const auto layout = cache.logicalBlockLayout(0, 2);
    ASSERT_EQ(layout.k_precision, ActivationPrecision::Q16_1);
    ASSERT_EQ(layout.v_precision, ActivationPrecision::Q16_1);
    ASSERT_EQ(layout.layout, TensorLayout::KV_HEAD_POS_DIM);
    ASSERT_EQ(layout.k_bytes, 2u * 2u * q16_block_size_bytes(Q16BlockSize::BLOCK_64));
    ASSERT_EQ(layout.v_bytes, layout.k_bytes);

    auto k_payload = taggedBytes(layout.k_bytes, 7);
    auto v_payload = taggedBytes(layout.v_bytes, 91);
    const IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 0, 2, nullptr};
    ASSERT_TRUE(cache.importLogicalBlock(desc, k_payload.data(), v_payload.data()));

    std::vector<uint8_t> out_k(layout.k_bytes, 0);
    std::vector<uint8_t> out_v(layout.v_bytes, 0);
    ASSERT_TRUE(cache.exportLogicalBlock(desc, out_k.data(), out_v.data()));
    EXPECT_EQ(out_k, k_payload);
    EXPECT_EQ(out_v, v_payload);
}

TEST(Test__IKVCacheLogicalBlockIO, Q16HeadMajorGroupedVerifierAppendMatchesSerialDecode)
{
    constexpr int HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int MAX_SEQ = 4;
    constexpr int PREFIX = 3;
    constexpr int M = 3;

    CPURingKVCacheQ16_1 serial(testMPI(), 1, 1, MAX_SEQ, HEADS, HEAD_DIM, DeviceId::cpu(),
                               KVCacheLayoutMode::HEAD_MAJOR);
    CPURingKVCacheQ16_1 grouped(testMPI(), 1, 1, MAX_SEQ, HEADS, HEAD_DIM, DeviceId::cpu(),
                                KVCacheLayoutMode::HEAD_MAJOR);

    const auto prefix_layout = serial.logicalBlockLayout(0, PREFIX);
    auto prefix_k = taggedBytes(prefix_layout.k_bytes, 13);
    auto prefix_v = taggedBytes(prefix_layout.v_bytes, 89);
    const IKVCache::KVCacheLogicalBlockDescriptor prefix_desc{0, 0, 0, PREFIX, nullptr};
    ASSERT_TRUE(serial.importLogicalBlock(prefix_desc, prefix_k.data(), prefix_v.data()));
    ASSERT_TRUE(grouped.importLogicalBlock(prefix_desc, prefix_k.data(), prefix_v.data()));

    auto verifier_k = taggedQ16Tensor({static_cast<size_t>(HEADS * M), HEAD_DIM},
                                      Q16BlockSize::BLOCK_64,
                                      41);
    auto verifier_v = taggedQ16Tensor({static_cast<size_t>(HEADS * M), HEAD_DIM},
                                      Q16BlockSize::BLOCK_64,
                                      177);

    for (int row = 0; row < M; ++row)
    {
        auto k_row = q16PositionRowFromHeadMajor(*verifier_k, row, M, HEADS, HEAD_DIM);
        auto v_row = q16PositionRowFromHeadMajor(*verifier_v, row, M, HEADS, HEAD_DIM);
        ASSERT_TRUE(serial.append_kv(0, 0, k_row.get(), v_row.get(), 1));
    }

    ASSERT_TRUE(grouped.appendVerifierRowsDecodeEquivalent(
        0, 0, verifier_k.get(), verifier_v.get(), M));

    const auto serial_state = serial.sequenceState(0, 0);
    const auto grouped_state = grouped.sequenceState(0, 0);
    EXPECT_EQ(grouped_state.cached_tokens, serial_state.cached_tokens);
    EXPECT_EQ(grouped_state.implementation_head, serial_state.implementation_head);
    EXPECT_EQ(grouped_state.wrapped, serial_state.wrapped);

    const auto full_layout = serial.logicalBlockLayout(0, serial_state.cached_tokens);
    std::vector<uint8_t> serial_k(full_layout.k_bytes, 0);
    std::vector<uint8_t> serial_v(full_layout.v_bytes, 0);
    std::vector<uint8_t> grouped_k(full_layout.k_bytes, 0);
    std::vector<uint8_t> grouped_v(full_layout.v_bytes, 0);

    const IKVCache::KVCacheLogicalBlockDescriptor full_desc{
        0, 0, 0, serial_state.cached_tokens, nullptr};
    ASSERT_TRUE(serial.exportLogicalBlock(full_desc, serial_k.data(), serial_v.data()));
    ASSERT_TRUE(grouped.exportLogicalBlock(full_desc, grouped_k.data(), grouped_v.data()));
    EXPECT_EQ(grouped_k, serial_k);
    EXPECT_EQ(grouped_v, serial_v);
}

/**
 * @brief Sweep every CPU KV storage format through grouped verifier publication.
 *
 * Each format is exercised for M=2,3,4, position-major and head-major source
 * tensors, position-major and head-major cache destinations, and both wrapped
 * and non-wrapped ring transitions. The serial witness appends one logical row
 * at a time through ordinary decode; the grouped side uses exactly one
 * appendVerifierRowsDecodeEquivalent() call and must publish identical raw
 * cache bytes and ring metadata.
 */
TEST(Test__IKVCacheLogicalBlockIO, AllCPUFormatsGroupedVerifierAppendRuntimeMMatchesSerialDecodeBytes)
{
    ScopedPerfStats perfstats;

    runGroupedVerifierAppendFormat<ActivationPrecision::FP32>("FP32/FP32");
    runGroupedVerifierAppendFormat<ActivationPrecision::BF16>("BF16/BF16");
    runGroupedVerifierAppendFormat<ActivationPrecision::FP16>("FP16/FP16");
    runGroupedVerifierAppendFormat<ActivationPrecision::Q8_1>("Q8_1/Q8_1");
    runGroupedVerifierAppendFormat<ActivationPrecision::Q16_1>("Q16_1/Q16_1");
    runGroupedVerifierAppendFormat<ActivationPrecision::TQ4>("TQ4/TQ4");
    runGroupedVerifierAppendFormat<ActivationPrecision::TQ8>("TQ8/TQ8");
    runGroupedVerifierAppendFormat<ActivationPrecision::TQ8, ActivationPrecision::TQ4>(
        "TQ8/TQ4");
}

TEST(Test__IKVCacheLogicalBlockIO, SplitTQAsymmetricLogicalBlockRoundTripsRawBytes)
{
    CPURingKVCacheTQ cache(testMPI(), 1, 1, 4, 2, 64, DeviceId::cpu(),
                           KVCacheLayoutMode::POSITION_MAJOR);
    const auto layout = cache.logicalBlockLayout(0, 3);
    ASSERT_EQ(layout.k_precision, ActivationPrecision::TQ8);
    ASSERT_EQ(layout.v_precision, ActivationPrecision::TQ4);
    ASSERT_EQ(layout.layout, TensorLayout::KV_POS_HEAD_DIM);
    ASSERT_EQ(layout.k_bytes, 3u * 2u * sizeof(TQ8Block_64));
    ASSERT_EQ(layout.v_bytes, 3u * 2u * sizeof(TQ4Block_64));

    auto k_payload = taggedBytes(layout.k_bytes, 11);
    auto v_payload = taggedBytes(layout.v_bytes, 193);
    const IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 0, 3, nullptr};
    ASSERT_TRUE(cache.importLogicalBlock(desc, k_payload.data(), v_payload.data()));

    std::vector<uint8_t> out_k(layout.k_bytes, 0);
    std::vector<uint8_t> out_v(layout.v_bytes, 0);
    ASSERT_TRUE(cache.exportLogicalBlock(desc, out_k.data(), out_v.data()));
    EXPECT_EQ(out_k, k_payload);
    EXPECT_EQ(out_v, v_payload);
}
