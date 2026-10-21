/**
 * @file Test__QwenSparseAttention.cpp
 * @brief Device-free regressions for QSA causal pooling, selection, and rollback.
 *
 * Tests exercise production CPU entrypoints with dirty persistent scratch, every
 * tail length, the 2048-token threshold, nonzero RoPE origins, and acceptance
 * across a complete-block boundary. Grouped results are compared byte-for-byte
 * with serial transactions, including the native selected-index attention path.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "../../integration/kernels/qwen4exp/QwenSparseNativeCodecFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>

namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Restore the caller's OpenMP worker policy after explicit thread-count regressions. */
struct ThreadPolicy
{
    int prior = omp_get_max_threads();
    /** @brief Restore setup-owned worker count after the focused test. */
    ~ThreadPolicy() { omp_set_num_threads(prior); }
};
/** @brief Compact CPU fixture with independent mutable cache and transaction state. */
struct SparseFixture
{
    QwenSparseReplay replay;
    std::vector<float> query, raw, partial, pools, norm, cosine, sine, prepared, scores;
    std::vector<std::int32_t> selected, count;
    QwenSparseIndexer p;
    /** @brief Bind admitted storage and deterministic nonzero input/cache values. */
    SparseFixture(int prefix, int rows, int capacity, int origin = 0)
        : replay{prefix, rows, origin}, query(rows * 4 * 128), raw(rows * 128), partial(4 * 128),
          pools(((capacity + 3) / 4) * 128), norm(128, 1.0f),
          cosine((capacity + origin) * 32, 1.0f), sine((capacity + origin) * 32, 0.0f),
          prepared(rows * 4 * 128, NAN), scores(rows * ((capacity + 3) / 4), NAN),
          selected(rows * QwenSparseSelected, -77), count(rows, -77)
    {
        for (std::size_t i = 0; i < query.size(); ++i) query[i] = std::sin(float(i + 1) * 0.13f);
        for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = std::cos(float(i + 1) * 0.073f);
        for (std::size_t i = 0; i < partial.size(); ++i) partial[i] = std::sin(float(i + 2) * 0.033f);
        for (std::size_t i = 0; i < pools.size(); ++i) pools[i] = std::cos(float(i + 2) * 0.041f);
        p = {.replay = &replay, .query = query.data(), .raw_keys = raw.data(), .partial_keys = partial.data(),
             .pooled_keys = pools.data(), .query_norm = norm.data(), .key_norm = norm.data(),
             .cosine = cosine.data(), .sine = sine.data(), .prepared_query = prepared.data(),
             .scores = scores.data(), .selected = selected.data(), .selected_count = count.data(),
             .row_capacity = rows, .token_capacity = capacity, .block_capacity = (capacity + 3) / 4,
             .position_capacity = capacity + origin};
    }
};

TEST(QwenSparseAttention, ExplicitRepresentationAndStaticBindingValidation)
{
    for (unsigned bits = 0; bits <= 65535; ++bits)
    {
        const float value = detail::decodeHalf(static_cast<std::uint16_t>(bits));
        if (std::isnan(value)) continue;
        ASSERT_EQ(detail::encodeHalf(value), bits);
    }
    EXPECT_EQ(detail::roundRepresentation(1.00390625f, QwenSparseScalar::BF16), 1.0f);
    EXPECT_EQ(detail::roundRepresentation(1.01171875f, QwenSparseScalar::BF16), 1.015625f);
    SparseFixture f(0, 1, 16);
    EXPECT_NO_THROW(validateQwenSparseIndexer(f.p));
    f.p.block_capacity = 1;
    EXPECT_THROW(validateQwenSparseIndexer(f.p), std::invalid_argument);
    f.p.block_capacity = 4;
    f.p.representation = static_cast<QwenSparseScalar>(99);
    EXPECT_THROW(validateQwenSparseIndexer(f.p), std::invalid_argument);
}

TEST(QwenSparseAttention, PoolBeforeNormAndFirstTokenRotaryPosition)
{
    SparseFixture f(3, 2, 16, 7);
    std::fill(f.query.begin(), f.query.end(), 0.0f);
    for (int member = 0; member < 3; ++member)
        for (int d = 0; d < 128; ++d) f.partial[member * 128 + d] = float(member + 1) * float(d + 1) * 0.001f;
    for (int d = 0; d < 128; ++d) f.raw[d] = float(d + 1) * 0.004f;
    // Only absolute position seven is rotated by ninety degrees. A block-end
    // or transaction-local position silently produces a different pool here.
    std::fill_n(f.cosine.data() + 7 * 32, 32, 0.0f);
    std::fill_n(f.sine.data() + 7 * 32, 32, 1.0f);
    for (auto representation : {QwenSparseScalar::FP32, QwenSparseScalar::FP16, QwenSparseScalar::BF16})
    {
        f.p.representation = representation;
        cpuQwenSparseIndexer(f.p);
        float pooled[128], sum = 0;
        for (int d = 0; d < 128; ++d)
        {
            float value = detail::roundRepresentation(f.partial[d], representation);
            value += detail::roundRepresentation(f.partial[128 + d], representation);
            value += detail::roundRepresentation(f.partial[256 + d], representation);
            value += detail::roundRepresentation(f.raw[d], representation);
            pooled[d] = detail::roundRepresentation(value * 0.25f, representation);
            sum += pooled[d] * pooled[d];
        }
        const float inverse = 1.0f / std::sqrt(sum / 128.0f + 1e-6f);
        for (int d = 0; d < 128; ++d)
        {
            const int source = d < 32 ? d + 32 : d < 64 ? d - 32 : d;
            const float expected = detail::roundRepresentation(pooled[source] * inverse, representation) * (d < 32 ? -1 : 1);
            EXPECT_NEAR(f.pools[d], expected, representation == QwenSparseScalar::FP32 ? 2e-6f : 0.0f);
        }
    }
}

TEST(QwenSparseAttention, EveryTailThresholdTiesAndDirtyLargeToSmallReplay)
{
    SparseFixture f(2047, 9, 4096);
    std::fill(f.query.begin(), f.query.end(), 0.0f);
    cpuQwenSparseIndexer(f.p);
    for (int row = 0; row < 9; ++row)
    {
        const int visible = 2048 + row;
        const int expected = std::min(visible / 4, 512) * 4 + visible % 4;
        ASSERT_EQ(f.count[row], expected);
        for (int i = 0; i < 2048; ++i) EXPECT_EQ(f.selected[row * QwenSparseSelected + i], i);
        for (int i = 2048; i < expected; ++i)
            EXPECT_EQ(f.selected[row * QwenSparseSelected + i], visible / 4 * 4 + i - 2048);
    }
    f.replay.prefix_tokens = 0;
    f.replay.live_rows = 1;
    cpuQwenSparseIndexer(f.p);
    EXPECT_EQ(f.count[0], 1);
    EXPECT_EQ(f.selected[0], 0);
    for (int row = 1; row < 9; ++row) EXPECT_EQ(f.count[row], 0);
    f.replay.live_rows = 0;
    cpuQwenSparseIndexer(f.p);
    EXPECT_TRUE(std::all_of(f.count.begin(), f.count.end(), [](int n) { return n == 0; }));
}

TEST(QwenSparseAttention, GroupedIndexerMatchesSerialTransactionsEveryRepresentation)
{
    ThreadPolicy thread_policy;
    for (auto representation : {QwenSparseScalar::FP32, QwenSparseScalar::FP16, QwenSparseScalar::BF16})
    for (int prefix : {3, 2047, 2050})
    {
        SparseFixture grouped(prefix, 15, 4096, 11);
        SparseFixture serial(prefix, 15, 4096, 11);
        grouped.p.representation = serial.p.representation = representation;
        omp_set_num_threads(3);
        cpuQwenSparseIndexer(grouped.p);
        for (int row = 0; row < 15; ++row)
        {
            serial.replay.live_rows = 1;
            serial.p.row_capacity = 1;
            serial.p.query = serial.query.data() + row * 4 * 128;
            serial.p.raw_keys = serial.raw.data() + row * 128;
            cpuQwenSparseIndexer(serial.p);
            ASSERT_EQ(grouped.count[row], serial.count[0]);
            EXPECT_EQ(std::memcmp(grouped.selected.data() + row * QwenSparseSelected, serial.selected.data(),
                                  serial.count[0] * sizeof(std::int32_t)), 0);
            EXPECT_EQ(std::memcmp(grouped.prepared.data() + row * 4 * 128, serial.prepared.data(), 4 * 128 * sizeof(float)), 0);
            const int accepted = 1;
            cpuQwenSparseCommitPartial({.replay = &serial.replay, .prior_partial = serial.partial.data(),
                .raw_keys = serial.p.raw_keys, .acceptance = AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,1,&serial.replay.live_rows),&accepted),
                .partial_keys = serial.partial.data()});
            ++serial.replay.prefix_tokens;
        }
        EXPECT_EQ(std::memcmp(grouped.pools.data(), serial.pools.data(),
                              ((prefix + 15) / 4) * 128 * sizeof(float)), 0);
    }
}

TEST(QwenSparseAttention, RollbackAcrossBlockBoundaryOverwritesRejectedPools)
{
    SparseFixture f(3, 6, 32);
    const auto original_partial = f.partial;
    cpuQwenSparseIndexer(f.p);
    const int accepted = 2;
    cpuQwenSparseCommitPartial({.replay = &f.replay, .prior_partial = f.partial.data(), .raw_keys = f.raw.data(),
        .acceptance = AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,6,&f.replay.live_rows),&accepted), .partial_keys = f.partial.data()});
    EXPECT_EQ(std::memcmp(f.partial.data(), f.raw.data() + 128, 128 * sizeof(float)), 0);
    f.replay.prefix_tokens = 5;
    f.replay.live_rows = 3;
    std::fill(f.raw.begin(), f.raw.end(), 3.0f);
    cpuQwenSparseIndexer(f.p);
    SparseFixture clean(5, 3, 32);
    clean.partial = f.partial;
    std::fill(clean.raw.begin(), clean.raw.end(), 3.0f);
    cpuQwenSparseIndexer(clean.p);
    EXPECT_EQ(std::memcmp(f.pools.data() + 128, clean.pools.data() + 128, 128 * sizeof(float)), 0);
    const int rejected = 0;
    f.replay.prefix_tokens = 3;
    std::copy(original_partial.begin(), original_partial.end(), f.partial.begin());
    cpuQwenSparseCommitPartial({.replay = &f.replay, .prior_partial = f.partial.data(), .raw_keys = f.raw.data(),
        .acceptance = AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,6,&f.replay.live_rows),&rejected), .partial_keys = f.partial.data()});
    EXPECT_EQ(f.partial, original_partial);
}

/** @test Ordinary zero acceptance is a no-op on a distinct dirty destination, not prior restoration. */
TEST(QwenSparseAttention, OrdinaryTailNoOpAndExactAcceptedPrefix)
{
    ThreadPolicy restore;omp_set_num_threads(1);
    constexpr int capacity=3,width=128;
    for(int prefix : {0,1,2,3,5,7})for(int live : {3,1,0,2,3})for(int accepted=0;accepted<=live;++accepted)
    {
        SCOPED_TRACE(prefix);
        SCOPED_TRACE(live);
        SCOPED_TRACE(accepted);
        SparseFixture f(prefix,capacity,16);f.replay.live_rows=live;
        for(std::size_t i=0;i<f.partial.size();++i)f.partial[i]=-float(i+1);
        for(std::size_t i=0;i<f.raw.size();++i)f.raw[i]=float(i+1001);
        const auto prior=f.partial,raw=f.raw;
        std::vector<float> destination(4*width,-24680.0f),expected=destination;
        if(live>0 && accepted>0)
        {
            std::vector<float> suffix(prior.begin(),prior.begin()+(prefix%4)*width);
            suffix.insert(suffix.end(),raw.begin(),raw.begin()+accepted*width);
            const auto complete=(suffix.size()/(4*width))*(4*width);
            std::copy(suffix.begin()+complete,suffix.end(),expected.begin());
        }
        cpuQwenSparseCommitPartial({.replay=&f.replay,.prior_partial=f.partial.data(),.raw_keys=f.raw.data(),
            .acceptance=AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,capacity,&f.replay.live_rows),&accepted),
            .partial_keys=destination.data()});
        EXPECT_EQ(std::memcmp(destination.data(),expected.data(),destination.size()*sizeof(float)),0);
        EXPECT_EQ(f.partial,prior);EXPECT_EQ(f.raw,raw);
        EXPECT_EQ(f.replay.prefix_tokens,prefix);EXPECT_EQ(f.replay.live_rows,live);
    }
}

TEST(QwenSparseAttention, NativeSelectedAttentionIsGroupedSerialByteEquivalent)
{
    constexpr int rows = 7, capacity = 16;
    SparseFixture f(3, rows, capacity);
    cpuQwenSparseIndexer(f.p);
    std::vector<float> query(rows * 24 * 256), gate(query.size()), output(query.size()), serial(query.size());
    std::vector<float> keys(capacity * 2 * 256), values(keys.size());
    for (std::size_t i = 0; i < query.size(); ++i) { query[i] = std::sin(float(i) * .01f); gate[i] = std::cos(float(i) * .02f); }
    for (std::size_t i = 0; i < keys.size(); ++i) { keys[i] = std::cos(float(i) * .01f); values[i] = std::sin(float(i) * .03f); }
    for (auto representation : {QwenSparseScalar::FP32, QwenSparseScalar::FP16, QwenSparseScalar::BF16})
    {
        std::vector<std::uint16_t> key16(keys.size()), value16(values.size());
        if (representation != QwenSparseScalar::FP32)
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                key16[i] = representation == QwenSparseScalar::FP16 ? detail::encodeHalf(keys[i]) : std::uint16_t(detail::floatBits(detail::roundRepresentation(keys[i], representation)) >> 16);
                value16[i] = representation == QwenSparseScalar::FP16 ? detail::encodeHalf(values[i]) : std::uint16_t(detail::floatBits(detail::roundRepresentation(values[i], representation)) >> 16);
            }
        const std::size_t scalar = representation == QwenSparseScalar::FP32 ? 4 : 2;
        QwenSparseAttention p{.replay = &f.replay, .query = query.data(),
            .kv = {.keys = {.data = representation == QwenSparseScalar::FP32 ? static_cast<void *>(keys.data()) : key16.data(),
                           .token_stride_bytes = 512 * scalar, .head_stride_bytes = 256 * scalar,
                           .codec = representation == QwenSparseScalar::FP32 ? TensorType::FP32 : representation == QwenSparseScalar::FP16 ? TensorType::FP16 : TensorType::BF16},
                   .values = {.data = representation == QwenSparseScalar::FP32 ? static_cast<void *>(values.data()) : value16.data(),
                           .token_stride_bytes = 512 * scalar, .head_stride_bytes = 256 * scalar,
                           .codec = representation == QwenSparseScalar::FP32 ? TensorType::FP32 : representation == QwenSparseScalar::FP16 ? TensorType::FP16 : TensorType::BF16}},
            .selected = f.selected.data(), .selected_count = f.count.data(), .gate = gate.data(), .output = output.data(),
            .row_capacity = rows, .token_capacity = capacity};
        cpuQwenSparseAttention(p);
        for (int row = 0; row < rows; ++row)
        {
            QwenSparseReplay state{3 + row, 1, 0};
            p.replay = &state; p.query = query.data() + row * 24 * 256; p.gate = gate.data() + row * 24 * 256;
            p.selected = f.selected.data() + row * QwenSparseSelected; p.selected_count = f.count.data() + row;
            p.output = serial.data() + row * 24 * 256; p.row_capacity = 1;
            cpuQwenSparseAttention(p);
        }
        EXPECT_EQ(std::memcmp(output.data(), serial.data(), output.size() * sizeof(float)), 0);
    }
}

/** @test Odd three-query shards retain every original global KV mapping and exact native decode. */
TEST(QwenSparseAttention, ReplicatedKVOddQueryIntervalsMatchFullOriginalHeads)
{
    ThreadPolicy thread_policy;
    omp_set_num_threads(3);
    constexpr int rows = 2, tokens = 8, local_heads = 3, dimension = 256;
    SparseFixture f(3, rows, tokens);
    cpuQwenSparseIndexer(f.p);
    std::vector<float> query(rows * 24 * dimension), gate(query.size()), full(query.size());
    for (std::size_t i = 0; i < query.size(); ++i)
    {
        query[i] = static_cast<float>(static_cast<int>(i % 113) - 56) / 64;
        gate[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 16;
    }
    for (const auto codec : {TensorType::FP32, TensorType::FP16, TensorType::BF16,
            TensorType::Q8_1, TensorType::Q16_1, TensorType::AQ8, TensorType::TQ4, TensorType::TQ8})
    {
        SCOPED_TRACE(static_cast<int>(codec));
        // AQ8 is a key-only codec; preserve its actual supported FP32 value pair.
        test::NativeCodecFixture keys(codec, tokens), values(codec == TensorType::AQ8 ? TensorType::FP32 : codec, tokens);
        QwenSparseAttention complete{.replay = &f.replay, .query = query.data(),
            .kv = {.keys = keys.bank(), .values = values.bank()}, .selected = f.selected.data(),
            .selected_count = f.count.data(), .gate = gate.data(), .output = full.data(),
            .row_capacity = rows, .token_capacity = tokens};
        ASSERT_NO_THROW(cpuQwenSparseAttention(complete));
        std::vector<float> local_query(rows * local_heads * dimension), local_gate(local_query.size()), actual(local_query.size());
        for (const int first : {0, 3, 6, 9, 12, 15, 18, 21})
        {
            SCOPED_TRACE(first);
            for (int row = 0; row < rows; ++row)
            {
                std::copy_n(query.data() + (row * 24 + first) * dimension, local_heads * dimension,
                    local_query.data() + row * local_heads * dimension);
                std::copy_n(gate.data() + (row * 24 + first) * dimension, local_heads * dimension,
                    local_gate.data() + row * local_heads * dimension);
            }
            auto local = complete;
            local.query = local_query.data(); local.gate = local_gate.data(); local.output = actual.data();
            local.query_heads = local_heads; local.query_head_start = first;
            std::fill(actual.begin(), actual.end(), -91.0f);
            ASSERT_NO_THROW(cpuQwenSparseAttention(local));
            for (int row = 0; row < rows; ++row)
                EXPECT_EQ(std::memcmp(actual.data() + row * local_heads * dimension,
                    full.data() + (row * 24 + first) * dimension, local_heads * dimension * sizeof(float)), 0);
            local.query_head_start = 22; // Global query interval would extend beyond original24.
            std::fill(actual.begin(), actual.end(), -91.0f);
            EXPECT_THROW(cpuQwenSparseAttention(local), std::invalid_argument);
            EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](float v) { return v == -91.0f; }));
        }
    }
}

/** @test Partial/empty CPU launches preserve inactive output and actual-read counters byte for byte. */
TEST(QwenSparseAttention, InactiveRowsPreserveOutputAndTrafficAcrossDirtyReplays)
{
    ThreadPolicy thread_policy;
    constexpr int rows=3,tokens=8,heads=24,width=heads*256;
    SparseFixture f(3,rows,tokens);
    omp_set_num_threads(1);
    cpuQwenSparseIndexer(f.p);
    const auto selectedCounts=f.count;
    test::NativeCodecFixture keys(TensorType::FP16,tokens),values(TensorType::FP16,tokens);
    std::vector<float> query(rows*width),gate(query.size()),baseline(query.size()),actual(query.size());
    for(std::size_t i=0;i<query.size();++i) {
        query[i]=float(int(i%127)-63)/64.0f;
        gate[i]=float(int(i%31)-15)/16.0f;
    }
    std::vector<QwenSparseAttentionObservation> observations(rows*heads);
    QwenSparseAttention p{.replay=&f.replay,.query=query.data(),
        .kv={.keys=keys.bank(),.values=values.bank(),.addressing=QwenSparseKVAddressing::FixedLogicalView,
             .logical_view={.logical_row_origin=6,.physical_row_capacity=tokens}},
        .selected=f.selected.data(),.selected_count=f.count.data(),.output=baseline.data(),
        .row_capacity=rows,.token_capacity=tokens};
    for(bool gated:{false,true}) {
        p.gate=gated ? gate.data() : nullptr;
        p.output=baseline.data();p.observations=nullptr;
        f.replay.live_rows=rows;std::copy(selectedCounts.begin(),selectedCounts.end(),f.count.begin());
        cpuQwenSparseAttention(p);
        for(int workers:{1,3}) {
            omp_set_num_threads(workers);
            std::fill(actual.begin(),actual.end(),-923.25f);
            std::memset(observations.data(),0xa5,observations.size()*sizeof(observations[0]));
            p.output=actual.data();p.observations=observations.data();
            // Partial first use proves poison preservation; growth followed by
            // shrink/empty replay proves previous live values are retained too.
            for(int live:{1,rows,1,0,2,0}) {
                SCOPED_TRACE(::testing::Message()<<"gated="<<gated<<" workers="<<workers<<" live="<<live);
                auto expected=actual;
                const auto inactive=std::vector<unsigned char>(
                    reinterpret_cast<const unsigned char *>(observations.data()),
                    reinterpret_cast<const unsigned char *>(observations.data()+observations.size()));
                std::copy_n(baseline.data(),live*width,expected.data());
                f.replay.live_rows=live;
                for(int row=0;row<rows;++row)f.count[row]=row<live ? selectedCounts[row] : -987;
                ASSERT_NO_THROW(cpuQwenSparseAttention(p));
                EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
                const auto offset=std::size_t(live)*heads*sizeof(observations[0]);
                EXPECT_EQ(std::memcmp(reinterpret_cast<const unsigned char *>(observations.data())+offset,
                    inactive.data()+offset,inactive.size()-offset),0);
                for(int row=0;row<live;++row)for(int head=0;head<heads;++head) {
                    const auto &traffic=observations[row*heads+head];
                    EXPECT_EQ(traffic.key_elements,std::uint64_t(selectedCounts[row])*256);
                    EXPECT_EQ(traffic.value_elements,std::uint64_t(selectedCounts[row])*256);
                    EXPECT_EQ(traffic.selected_tokens,selectedCounts[row]);
                }
            }
        }
    }
}

TEST(QwenSparseAttention, EveryNativeCodecAndQ16BlockUsesCanonicalRingAndBasis)
{
    ThreadPolicy thread_policy;
    constexpr int tokens = 8;
    SparseFixture f(5, 2, tokens);
    cpuQwenSparseIndexer(f.p);
    std::vector<float> query(2 * 6144), actual(query.size()), expected(query.size());
    for (std::size_t i = 0; i < query.size(); ++i) query[i] = float(int(i % 127) - 63) / 64;
    const auto codecs = {TensorType::FP32, TensorType::FP16, TensorType::BF16, TensorType::Q8_1,
                         TensorType::Q16_1, TensorType::AQ8, TensorType::TQ4, TensorType::TQ8};
    for (auto key_codec : codecs)
    for (auto value_codec : codecs)
    {
        if (value_codec == TensorType::AQ8) continue;
        for (int block_size : {32, 64, 128})
        {
            if (block_size != 32 && key_codec != TensorType::Q16_1 && value_codec != TensorType::Q16_1) continue;
            test::NativeCodecFixture keys(key_codec, tokens, block_size), values(value_codec, tokens, block_size);
            QwenSparseAttention p{.replay = &f.replay, .query = query.data(),
                .kv = {.keys = keys.bank(), .values = values.bank(), .addressing = QwenSparseKVAddressing::FixedLogicalView,
                       .logical_view = {.logical_row_origin = 6, .physical_row_capacity = tokens}},
                .selected = f.selected.data(), .selected_count = f.count.data(), .output = actual.data(),
                .row_capacity = 2, .token_capacity = tokens};
            omp_set_num_threads(4);
            cpuQwenSparseAttention(p);
            std::vector<float> serial(actual.size());
            omp_set_num_threads(1);
            for (int row = 0; row < 2; ++row)
            {
                QwenSparseReplay state{5 + row, 1, 0};
                auto one = p;
                one.replay = &state; one.query = query.data() + row * 6144;
                one.selected = f.selected.data() + row * QwenSparseSelected;
                one.selected_count = f.count.data() + row; one.output = serial.data() + row * 6144; one.row_capacity = 1;
                cpuQwenSparseAttention(one);
            }
            ASSERT_EQ(std::memcmp(actual.data(), serial.data(), actual.size() * sizeof(float)), 0);
            p.kv.keys = keys.reference(true); p.kv.values = values.reference(); p.output = expected.data();
            cpuQwenSparseAttention(p);
            for (std::size_t i = 0; i < actual.size(); ++i) ASSERT_NEAR(actual[i], expected[i], 2e-5f);
            // Every installed native codec/block geometry obeys the same live
            // write extent; no inactive read-counter reset may erase a sibling.
            p.kv.keys=keys.bank();p.kv.values=values.bank();p.output=actual.data();
            std::vector<QwenSparseAttentionObservation> observations(2*24);
            std::memset(observations.data(),0xa5,observations.size()*sizeof(observations[0]));
            const auto poisoned=std::vector<unsigned char>(
                reinterpret_cast<const unsigned char *>(observations.data()),
                reinterpret_cast<const unsigned char *>(observations.data()+observations.size()));
            p.observations=observations.data();f.replay.live_rows=1;
            ASSERT_NO_THROW(cpuQwenSparseAttention(p));
            EXPECT_EQ(std::memcmp(actual.data(),serial.data(),actual.size()*sizeof(float)),0);
            constexpr auto inactiveOffset=24*sizeof(QwenSparseAttentionObservation);
            EXPECT_EQ(std::memcmp(reinterpret_cast<const unsigned char *>(observations.data())+inactiveOffset,
                poisoned.data()+inactiveOffset,inactiveOffset),0);
            for(int head=0;head<24;++head) {
                EXPECT_EQ(observations[head].key_elements,std::uint64_t(f.count[0])*256);
                EXPECT_EQ(observations[head].value_elements,std::uint64_t(f.count[0])*256);
                EXPECT_EQ(observations[head].selected_tokens,f.count[0]);
            }
            const auto beforeEmpty=std::vector<unsigned char>(
                reinterpret_cast<const unsigned char *>(observations.data()),
                reinterpret_cast<const unsigned char *>(observations.data()+observations.size()));
            f.replay.live_rows=0;
            ASSERT_NO_THROW(cpuQwenSparseAttention(p));
            EXPECT_EQ(std::memcmp(actual.data(),serial.data(),actual.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(observations.data(),beforeEmpty.data(),beforeEmpty.size()),0);
            f.replay.live_rows=2;
        }
    }
}
} // namespace
} // namespace llaminar2::qwen4exp

namespace llaminar2::qwen4exp
{
/** @test Actual CPU producer score bytes and selection are unchanged by each explicit candidate. */
TEST(QwenSparseAttention, BoundedCandidateFullIndexerKeepsScoreBytes)
{
    ThreadPolicy workers; omp_set_num_threads(1);
    for (auto policy : {QwenSparsePoolSelection::Bounded256_512,
                       QwenSparsePoolSelection::Bounded512_512,
                       QwenSparsePoolSelection::Bounded1024_1024})
        for (auto representation : {QwenSparseScalar::FP32, QwenSparseScalar::FP16, QwenSparseScalar::BF16})
        {
            SparseFixture installed(4093, 9, 4102), candidate(4093, 9, 4102);
            installed.p.representation = representation; candidate.p.representation = representation;
            candidate.p.pool_selection = policy;
            cpuQwenSparseIndexer(installed.p); cpuQwenSparseIndexer(candidate.p);
            EXPECT_EQ(std::memcmp(installed.scores.data(), candidate.scores.data(),
                installed.scores.size() * sizeof(float)), 0);
            EXPECT_EQ(installed.selected, candidate.selected); EXPECT_EQ(installed.count, candidate.count);
            EXPECT_EQ(std::memcmp(installed.prepared.data(), candidate.prepared.data(),
                installed.prepared.size() * sizeof(float)), 0);
        }
}
} // namespace llaminar2::qwen4exp
