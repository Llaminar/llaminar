/**
 * @file Test__QwenSparseAttention.cpp
 * @brief Device-free regressions for QSA causal pooling, selection, and rollback.
 *
 * Tests exercise production CPU entrypoints with dirty persistent scratch, every
 * tail length, the 2048-token threshold, nonzero RoPE origins, and acceptance
 * across a complete-block boundary. Grouped results are compared byte-for-byte
 * with serial transactions, including the native selected-index attention path.
 * Canonical consumer-local compact KV slices additionally cover every logical
 * degree1..8; these are CPU operation tests, not multi-device model certification.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "../../integration/kernels/qwen4exp/QwenSparseNativeCodecFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>
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

/** @test Combined CTA capacity is checked independently of live replay counts. */
TEST(QwenSparseAttention, FusedPreparationGridAndCrossRoleAliases)
{
    for(const int rows:{1,16,33,2048})
    {
        const auto grid=qwenSparsePreparationGrid(rows);
        EXPECT_EQ(grid.pool_blocks,(rows+6)/4);EXPECT_EQ(grid.query_blocks,rows*4);
        EXPECT_EQ(grid.total_blocks,grid.pool_blocks+grid.query_blocks);
    }
    EXPECT_THROW((void)qwenSparsePreparationGrid(0),std::invalid_argument);
    EXPECT_THROW((void)qwenSparsePreparationGrid(std::numeric_limits<int>::max()),std::invalid_argument);
    for (int rows : {1,16})
        EXPECT_EQ(qwenSparsePreparationGrid(rows).program,QwenSparsePreparationProgram::FusedGrouped);
    for (int rows : {17,33,512,65536})
        EXPECT_EQ(qwenSparsePreparationGrid(rows).program,QwenSparsePreparationProgram::SeparatePrefill);
    EXPECT_NO_THROW((void)qwenSparsePreparationGrid(505290269));
    EXPECT_THROW((void)qwenSparsePreparationGrid(505290270),std::invalid_argument);
    SparseFixture f(3,16,4096);
    const auto original=f.p;
    EXPECT_NO_THROW(validateQwenSparseIndexer(f.p)); // Shared immutable norm is legal.
    const auto reject=[&](QwenSparseIndexer p){EXPECT_THROW(validateQwenSparseIndexer(p),std::invalid_argument);};
    f.p.pooled_keys=const_cast<float*>(f.p.query)+1;reject(f.p);
    f.p=original;f.p.pooled_keys=const_cast<float*>(f.p.query_norm);reject(f.p);
    f.p=original;f.p.prepared_query=const_cast<float*>(f.p.raw_keys)+1;reject(f.p);
    f.p=original;f.p.prepared_query=const_cast<float*>(f.p.partial_keys);reject(f.p);
    f.p=original;f.p.prepared_query=const_cast<float*>(f.p.key_norm);reject(f.p);
    f.p=original;f.p.prepared_query=f.p.pooled_keys+1;reject(f.p);
    f.p=original;f.p.prepared_query=const_cast<float*>(f.p.cosine);reject(f.p);
    f.p=original;f.p.pooled_keys=reinterpret_cast<float*>(const_cast<QwenSparseReplay*>(f.p.replay));reject(f.p);
    f.p=original;f.p.raw_keys=f.p.query;EXPECT_NO_THROW(validateQwenSparseIndexer(f.p)); // Read/read alias.
    f.p=original;f.p.prepared_query=const_cast<float*>(f.p.query);EXPECT_NO_THROW(validateQwenSparseIndexer(f.p)); // Same-head in-place query preparation.
    QwenSparseTransactionPrior codec{};codec.row_capacity=16;
    f.p=original;f.p.transaction_prior=&codec;f.p.replay=&codec.replay;f.p.partial_keys=codec.partial;
    EXPECT_NO_THROW(validateQwenSparseIndexer(f.p)); // Canonical prior-header/tail owner.
    f.p=original;f.p.prepared_query=reinterpret_cast<float*>(UINTPTR_MAX-3);reject(f.p);
    f.p=original;EXPECT_NO_THROW(validateQwenSparseIndexer(f.p));
}

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

/** @test Compact bank fixture accepts only authenticated original-consumer assignment authority. */
TEST(QwenSparseAttention, CompactConsumerKVRejectsUndeclaredOrForeignGeometry)
{
    test::NativeCodecFixture source(TensorType::FP16,1,32,false);
    const auto legacy=TensorParallelConfig::equalSplit(4,24,2,10240,248320);
    EXPECT_THROW((test::ConsumerLocalCodecFixture(source,legacy,0)),std::invalid_argument);
    const auto undeclared=TensorParallelConfig::equalSplit(4,24,2,10240,248320,std::nullopt,KVHeadCoveragePolicy::QueryInterval);
    EXPECT_THROW((test::ConsumerLocalCodecFixture(source,undeclared,0)),std::invalid_argument);
    const auto grain=QueryHeadPartitionDeclaration::fromOriginalConsumers(24,{{16,48}});
    const auto actual=TensorParallelConfig::equalSplit(4,24,2,10240,248320,std::nullopt,KVHeadCoveragePolicy::QueryInterval,grain);
    EXPECT_THROW((test::ConsumerLocalCodecFixture(source,actual,-1)),std::out_of_range);
    EXPECT_THROW((test::ConsumerLocalCodecFixture(source,actual,4)),std::out_of_range);
}

/** @test Canonical consumer-local native KV bytes match full original heads for every degree1..8. */
TEST(QwenSparseAttention, CompactConsumerKVEveryDegreeMatchesFullOriginalBytes)
{
    ThreadPolicy thread_policy;omp_set_num_threads(3);
    constexpr int rows=3,tokens=8,dimension=256,full_width=24*dimension;
    QwenSparseReplay replay{5,rows,17};
    std::vector<float> query(rows*full_width),gate(query.size()),full(query.size());
    for(std::size_t i=0;i<query.size();++i)
    {query[i]=float(int(i%113)-56)/64.f;gate[i]=float(int(i%23)-11)/16.f;}
    std::vector<std::int32_t> selected(rows*QwenSparseSelected,-99),counts{3,4,5};
    for(int row=0;row<rows;++row)for(int i=0;i<counts[row];++i)selected[row*QwenSparseSelected+i]=i;
    const auto grain=QueryHeadPartitionDeclaration::fromOriginalConsumers(24,{{16,48}});
    for(const auto codec:{TensorType::FP32,TensorType::FP16,TensorType::BF16,TensorType::Q8_1,
            TensorType::Q16_1,TensorType::AQ8,TensorType::TQ4,TensorType::TQ8})
    for(int block:{32,64,128})
    {
        if(block!=32 && codec!=TensorType::Q16_1)continue;
        SCOPED_TRACE(::testing::Message()<<"codec="<<int(codec)<<" block="<<block);
        test::NativeCodecFixture keys(codec,tokens,block,false),values(codec==TensorType::AQ8?TensorType::FP32:codec,tokens,block,false);
        QwenSparseAttention original{.replay=&replay,.query=query.data(),
            .kv={.keys=keys.bank(),.values=values.bank(),.addressing=QwenSparseKVAddressing::FixedLogicalView,
                 .logical_view={.logical_row_origin=6,.physical_row_capacity=tokens}},
            .selected=selected.data(),.selected_count=counts.data(),.gate=gate.data(),.output=full.data(),
            .row_capacity=rows,.token_capacity=tokens};
        replay.live_rows=rows;for(int row=0;row<rows;++row)counts[row]=row+3;cpuQwenSparseAttention(original);
        for(int degree=1;degree<=8;++degree)
        {
            const auto config=TensorParallelConfig::equalSplit(degree,24,2,10240,248320,
                std::nullopt,KVHeadCoveragePolicy::QueryInterval,grain);
            for(int rank=0;rank<degree;++rank)
            {
                const auto &a=config.forRank(rank);const int width=a.head_count*dimension;
                SCOPED_TRACE(::testing::Message()<<"degree="<<degree<<" rank="<<rank);
                test::ConsumerLocalCodecFixture local_keys(keys,config,rank),local_values(values,config,rank);
                ASSERT_EQ(local_keys.storage.size(),std::size_t(tokens*a.kv_head_count)*keys.head_bytes);
                ASSERT_EQ(local_values.storage.size(),std::size_t(tokens*a.kv_head_count)*values.head_bytes);
                ASSERT_EQ(local_keys.bank().token_stride_bytes,std::size_t(a.kv_head_count)*keys.head_bytes);
                ASSERT_EQ(local_values.bank().token_stride_bytes,std::size_t(a.kv_head_count)*values.head_bytes);
                for(int token=0;token<tokens;++token)
                {
                    EXPECT_EQ(std::memcmp(local_keys.storage.data()+std::size_t(token*a.kv_head_count)*keys.head_bytes,
                        keys.storage.data()+std::size_t(token*2+a.kv_head_start)*keys.head_bytes,std::size_t(a.kv_head_count)*keys.head_bytes),0);
                    EXPECT_EQ(std::memcmp(local_values.storage.data()+std::size_t(token*a.kv_head_count)*values.head_bytes,
                        values.storage.data()+std::size_t(token*2+a.kv_head_start)*values.head_bytes,std::size_t(a.kv_head_count)*values.head_bytes),0);
                }
                std::vector<float> local_query(rows*width),local_gate(local_query.size()),actual(local_query.size()+16,-923.25f),expected=actual;
                for(int row=0;row<rows;++row)
                {
                    std::copy_n(query.data()+row*full_width+a.head_start*dimension,width,local_query.data()+row*width);
                    std::copy_n(gate.data()+row*full_width+a.head_start*dimension,width,local_gate.data()+row*width);
                }
                std::vector<QwenSparseAttentionObservation> observations(rows*a.head_count);
                std::memset(observations.data(),0xa5,observations.size()*sizeof(observations[0]));
                auto local=original;local.query=local_query.data();local.gate=local_gate.data();local.output=actual.data();
                local.kv.keys=local_keys.bank();local.kv.values=local_values.bank();local.observations=observations.data();
                local.query_heads=a.head_count;local.query_head_start=a.head_start;
                local.kv_heads=a.kv_head_count;local.kv_head_start=a.kv_head_start;
                for(int history=0;history<20;++history)
                {
                    const int live=std::array{1,3,1,0,2}[std::size_t(history)%5];replay.live_rows=live;
                    for(int row=0;row<rows;++row)counts[row]=row<live?row+3:-987;
                    const auto before=std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(observations.data()),
                        reinterpret_cast<const unsigned char*>(observations.data()+observations.size()));
                    for(int row=0;row<live;++row)std::copy_n(full.data()+row*full_width+a.head_start*dimension,width,expected.data()+row*width);
                    ASSERT_NO_THROW(cpuQwenSparseAttention(local));
                    EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0)<<"history="<<history;
                    const auto inactive=std::size_t(live*a.head_count)*sizeof(observations[0]);
                    EXPECT_EQ(std::memcmp(reinterpret_cast<const unsigned char*>(observations.data())+inactive,before.data()+inactive,before.size()-inactive),0);
                    for(int row=0;row<live;++row)for(int head=0;head<a.head_count;++head)
                    {
                        const auto &traffic=observations[row*a.head_count+head];
                        EXPECT_EQ(traffic.key_elements,std::uint64_t(counts[row])*dimension);
                        EXPECT_EQ(traffic.value_elements,std::uint64_t(counts[row])*dimension);
                        EXPECT_EQ(traffic.selected_tokens,counts[row]);
                    }
                }
            }
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
            // Compact head0 is sufficient to witness that actual candidate
            // membership feeds the unchanged real CPU attention consumer.
            test::NativeCodecFixture keys(TensorType::FP16,4102),values(TensorType::FP16,4102);
            std::vector<float> query(256,0.125f),a(query.size()),b(query.size());
            QwenSparseReplay coreReplay=installed.replay;coreReplay.live_rows=1;
            QwenSparseAttention attention{.replay=&coreReplay,.query=query.data(),
                .kv={.keys=keys.bank(),.values=values.bank()},.selected=installed.selected.data(),
                .selected_count=installed.count.data(),.output=a.data(),.row_capacity=1,.token_capacity=4102,.query_heads=1,.kv_heads=1};
            ASSERT_NO_THROW(validateQwenSparseAttention(attention));
            ASSERT_LE(coreReplay.prefix_tokens+attention.row_capacity,attention.token_capacity);
            for(int live:{9,1,0,9}){
                installed.replay.live_rows=candidate.replay.live_rows=live;
                cpuQwenSparseIndexer(installed.p);cpuQwenSparseIndexer(candidate.p);
                const int firstVisible=coreReplay.prefix_tokens+1;
                const int firstCount=live?std::min(firstVisible/4,512)*4+firstVisible%4:0;
                ASSERT_EQ(installed.count[0],firstCount);ASSERT_EQ(candidate.count[0],firstCount);
                std::fill(a.begin(),a.end(),-77.f);std::fill(b.begin(),b.end(),-77.f);
                coreReplay.live_rows=std::min(live,1);attention.selected=installed.selected.data();attention.selected_count=installed.count.data();attention.output=a.data();
                cpuQwenSparseAttention(attention);
                attention.selected=candidate.selected.data();attention.selected_count=candidate.count.data();attention.output=b.data();
                cpuQwenSparseAttention(attention);
                EXPECT_EQ(std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),0);
                if(!live){EXPECT_TRUE(std::all_of(a.begin(),a.end(),[](float value){return value==-77.f;}));
                    EXPECT_TRUE(std::all_of(b.begin(),b.end(),[](float value){return value==-77.f;}));}
            }
        }
}
} // namespace llaminar2::qwen4exp

namespace llaminar2::qwen4exp
{
/**
 * @test Per-row query admission preserves actual CPU indexer and attention results.
 *
 * Every replay retains its original frontier and physical row capacity. The
 * ungated producer receives every live input row; the gated producer receives
 * NaN query poison whenever ranking is unnecessary. Its prepared-query poison
 * must survive byte-for-byte, while keys, selections and native attention agree
 * with the ordinary producer. Reusing both fixtures catches stale large/empty
 * state without changing the invocation's arithmetic or observation policy.
 * This substantial native CPU sweep is opt-in through its explicit Integration
 * preflight registration and does not execute in the ordinary Unit suite.
 */
TEST(QwenSparseAttention, DISABLED_RankingOnlyQueryPreparationPreservesNativeResults)
{
    ThreadPolicy workers;omp_set_num_threads(3);
    constexpr int tokens=2080,width=256;
    for(const auto selection:{QwenSparsePoolSelection::Installed,QwenSparsePoolSelection::Bounded1024_1024})
    for(const auto representation:{QwenSparseScalar::FP32,QwenSparseScalar::FP16,QwenSparseScalar::BF16})
    {
        const auto codec=representation==QwenSparseScalar::FP32 ? TensorType::FP32 :
            representation==QwenSparseScalar::FP16 ? TensorType::FP16 : TensorType::BF16;
        test::NativeCodecFixture keys(codec,tokens),values(codec,tokens);
        for(const int rows:{1,3,16})
        {
            SCOPED_TRACE(::testing::Message()<<"representation="<<int(representation)<<" M="<<rows);
            SparseFixture ordinary(2052-rows,rows,tokens,17),gated(2052-rows,rows,tokens,17),serial(2050,1,tokens,17);
            ordinary.p.pool_selection=gated.p.pool_selection=serial.p.pool_selection=selection;
            serial.p.representation=representation;
            ordinary.p.representation=gated.p.representation=representation;
            ordinary.p.query_demand=QwenSparseQueryDemand::AllLive;
            gated.p.query_demand=QwenSparseQueryDemand::RankingOnly;
            const auto originalQuery=ordinary.query;
            std::vector<float> coreQuery(size_t(rows)*width),coreGate(coreQuery.size());
            std::vector<float> expected(coreQuery.size()),actual(coreQuery.size());
            for(size_t i=0;i<coreQuery.size();++i) {
                coreQuery[i]=float(int(i%29)-14)/32.f;coreGate[i]=float(int(i%17)-8)/16.f;
            }
            auto attentionFor=[&](SparseFixture &f,std::vector<float> &output) {
                return QwenSparseAttention{.replay=&f.replay,.query=coreQuery.data(),
                    .kv={.keys=keys.bank(),.values=values.bank()},.selected=f.selected.data(),
                    .selected_count=f.count.data(),.gate=coreGate.data(),.output=output.data(),
                    .row_capacity=rows,.token_capacity=tokens,.query_heads=1};
            };
            const int partial=std::min(rows,2);
            int iteration=0;
            for(const auto replay:std::array{QwenSparseReplay{2060,rows,17},
                    QwenSparseReplay{2051-rows,rows,17},QwenSparseReplay{2060-rows,rows,17},
                    QwenSparseReplay{3,std::min(rows,3),17},QwenSparseReplay{2060,0,17},
                    QwenSparseReplay{2050,rows,17},QwenSparseReplay{2051,1,17},QwenSparseReplay{0,0,17},
                    QwenSparseReplay{2051-partial,partial,17},QwenSparseReplay{2052-partial,partial,17},
                    QwenSparseReplay{2060,rows,17}})
            {
                SCOPED_TRACE(::testing::Message()<<"prefix="<<replay.prefix_tokens<<" live="<<replay.live_rows);
                const auto needed=[&](int row){return row<replay.live_rows&&(replay.prefix_tokens+row+1)/4>512;};
                const size_t liveWords=size_t(replay.live_rows)*512;
                const std::uint32_t preparedBits=0x7fc12000u+static_cast<unsigned>(iteration++);
                float preparedPoison;std::memcpy(&preparedPoison,&preparedBits,sizeof(preparedPoison));
                const float outputPoison=-8123.25f;
                const auto priorPools=gated.pools;
                const auto readonlyRaw=gated.raw,readonlyPartial=gated.partial;
                ordinary.replay=gated.replay=replay;
                std::fill(ordinary.query.begin(),ordinary.query.end(),NAN);
                std::copy_n(originalQuery.begin(),liveWords,ordinary.query.begin());
                std::fill(gated.query.begin(),gated.query.end(),NAN);
                for(int row=0;row<replay.live_rows;++row)if(needed(row))
                    std::copy_n(originalQuery.begin()+std::size_t(row)*512,512,gated.query.begin()+std::size_t(row)*512);
                const auto readonlyQuery=gated.query;
                for(auto *f:{&ordinary,&gated}) {
                    std::fill(f->prepared.begin(),f->prepared.end(),preparedPoison);
                    std::fill(f->scores.begin(),f->scores.end(),-1357.f);
                    std::fill(f->selected.begin(),f->selected.end(),-77);
                    std::fill(f->count.begin(),f->count.end(),-77);
                }
                ASSERT_NO_THROW(cpuQwenSparseIndexer(ordinary.p));
                ASSERT_NO_THROW(cpuQwenSparseIndexer(gated.p));
                EXPECT_EQ(ordinary.replay.prefix_tokens,replay.prefix_tokens);
                EXPECT_EQ(gated.replay.prefix_tokens,replay.prefix_tokens);
                EXPECT_EQ(ordinary.replay.live_rows,replay.live_rows);EXPECT_EQ(gated.replay.live_rows,replay.live_rows);
                EXPECT_TRUE(std::all_of(ordinary.prepared.begin(),ordinary.prepared.begin()+liveWords,
                    [](float value){return std::isfinite(value);}));
                for(int row=0;row<rows;++row) {
                    const auto offset=std::size_t(row)*512;
                    if(needed(row))EXPECT_EQ(std::memcmp(ordinary.prepared.data()+offset,gated.prepared.data()+offset,512*sizeof(float)),0);
                    else for(int d=0;d<512;++d)EXPECT_EQ(detail::floatBits(gated.prepared[offset+d]),preparedBits);
                }
                EXPECT_EQ(std::memcmp(ordinary.pools.data(),gated.pools.data(),ordinary.pools.size()*sizeof(float)),0);
                for(int block=0;block<gated.p.block_capacity;++block)
                    if(block<replay.prefix_tokens/4||block>=(replay.prefix_tokens+replay.live_rows)/4)
                        EXPECT_EQ(std::memcmp(gated.pools.data()+std::size_t(block)*128,
                            priorPools.data()+std::size_t(block)*128,128*sizeof(float)),0);
                EXPECT_EQ(std::memcmp(gated.query.data(),readonlyQuery.data(),readonlyQuery.size()*sizeof(float)),0);
                EXPECT_EQ(std::memcmp(gated.raw.data(),readonlyRaw.data(),readonlyRaw.size()*sizeof(float)),0);
                EXPECT_EQ(std::memcmp(gated.partial.data(),readonlyPartial.data(),readonlyPartial.size()*sizeof(float)),0);
                EXPECT_EQ(std::memcmp(ordinary.scores.data(),gated.scores.data(),ordinary.scores.size()*sizeof(float)),0);
                EXPECT_EQ(ordinary.selected,gated.selected);EXPECT_EQ(ordinary.count,gated.count);
                for(int row=0;row<rows;++row) {
                    const int visible=replay.prefix_tokens+row+1;
                    EXPECT_EQ(gated.count[row],row<replay.live_rows ? std::min(visible/4,512)*4+visible%4 : 0);
                }
                std::fill(expected.begin(),expected.end(),outputPoison);std::fill(actual.begin(),actual.end(),outputPoison);
                ASSERT_NO_THROW(cpuQwenSparseAttention(attentionFor(ordinary,expected)));
                ASSERT_NO_THROW(cpuQwenSparseAttention(attentionFor(gated,actual)));
                EXPECT_EQ(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)),0);
                EXPECT_TRUE(std::all_of(actual.begin()+size_t(replay.live_rows)*width,actual.end(),
                    [&](float value){return value==outputPoison;}));
                std::copy(priorPools.begin(),priorPools.end(),serial.pools.begin());
                std::copy(readonlyPartial.begin(),readonlyPartial.end(),serial.partial.begin());
                for(int row=0;row<replay.live_rows;++row) {
                    serial.replay={replay.prefix_tokens+row,1,replay.position_origin};
                    std::copy_n(originalQuery.begin()+std::size_t(row)*512,512,serial.query.begin());
                    std::copy_n(ordinary.raw.begin()+std::size_t(row)*128,128,serial.raw.begin());
                    std::fill(serial.selected.begin(),serial.selected.end(),-77);
                    std::fill(serial.count.begin(),serial.count.end(),-77);
                    ASSERT_NO_THROW(cpuQwenSparseIndexer(serial.p));
                    if(needed(row))EXPECT_EQ(std::memcmp(gated.prepared.data()+std::size_t(row)*512,serial.prepared.data(),512*sizeof(float)),0);
                    EXPECT_EQ(gated.count[row],serial.count[0]);
                    EXPECT_EQ(std::memcmp(gated.selected.data()+std::size_t(row)*QwenSparseSelected,
                        serial.selected.data(),QwenSparseSelected*sizeof(std::int32_t)),0);
                    ASSERT_NO_THROW(cpuQwenSparseCommitPartial({.replay=&serial.replay,.prior_partial=serial.partial.data(),
                        .raw_keys=serial.raw.data(),.acceptance=AcceptedRequestPrefixes::fixed(
                            DeviceRequestRowRanges::deviceCounted(1,1,&serial.replay.live_rows),1),
                        .partial_keys=serial.partial.data()}));
                }
                EXPECT_EQ(std::memcmp(gated.pools.data(),serial.pools.data(),gated.pools.size()*sizeof(float)),0);
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
    }
}
} // namespace llaminar2::qwen4exp
