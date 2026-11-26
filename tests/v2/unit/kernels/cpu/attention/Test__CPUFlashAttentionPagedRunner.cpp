/**
 * @file Test__CPUFlashAttentionPagedRunner.cpp
 * @brief P2 CPU vertical: pooled pages through the actual table/snapshot
 * operation view, including capacity one, odd/mixed counts and replacement.
 *
 * The production attention entry `compute_paged_snapshot_fp32` consumes a
 * retained `CPUFP32PagedKVTable::Snapshot`; these tests prove that real path
 * (not only the raw page fixture) matches the contiguous numerical control
 * under capacity one, long-over-equal-share, mixed request lengths and
 * permutation. No contiguous adapter or full-history gather exists here.
 */
#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "v2/kernels/cpu/attention/PagedKVArchive.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief One exact capacity-one pool + three-bank table on one authority. */
struct RunnerFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::shared_ptr<PhysicalPagePool> pool;
    std::shared_ptr<CPUFP32PagedKVTable> table;
    RunnerFixture(int page_tokens, int max_pages, int dimension = 128)
    {
        const auto device = DeviceId::cpu();
        // Frame bytes cover the largest geometry tested (kv_heads=2, dim=128).
        const size_t frame_bytes = static_cast<size_t>(page_tokens) * 2 * 2 *
            static_cast<size_t>(dimension) * sizeof(float);
        const PhysicalPagePoolGeometry pool_geometry{static_cast<size_t>(max_pages),
            frame_bytes, 2, static_cast<size_t>(max_pages)};
        const CPUFP32PagedKVTableGeometry table_geometry{
            static_cast<size_t>(max_pages), 4, 4, page_tokens, 64, 0};
        const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
            CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
        table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    }
};

/** @brief Publish one request's page-set through the table under a single grant. */
CPUFP32PagedKVTable::Snapshot publishRunnerSet(RunnerFixture &fx, const std::vector<float> &k,
    const std::vector<float> &v, int page_tokens, int heads, int dimension)
{
    const size_t width = static_cast<size_t>(heads * dimension);
    const size_t tokens = k.size() / width;
    const size_t page_count = (tokens + static_cast<size_t>(page_tokens) - 1) / static_cast<size_t>(page_tokens);
    auto request = fx.pool->tryAdmit(page_count);
    if (!request) throw std::logic_error("Runner fixture request refused");
    auto update = fx.table->tryBeginUpdate();
    if (!update) throw std::logic_error("Runner fixture table refused");
    for (size_t page = 0; page < page_count; ++page)
    {
        const size_t first = page * static_cast<size_t>(page_tokens);
        const size_t count = std::min(static_cast<size_t>(page_tokens), tokens - first);
        std::vector<float> fixture(count * 2 * width);
        for (size_t row = 0; row < count; ++row)
        {
            std::copy_n(k.data() + (first + row) * width, width, fixture.data() + row * 2 * width);
            std::copy_n(v.data() + (first + row) * width, width, fixture.data() + row * 2 * width + width);
        }
        const auto frame = request->assignFrame();
        auto write = request->prepareAppend(frame, fixture.size() * sizeof(float));
        if (!write) throw std::logic_error("Runner fixture write refused");
        write->copyFrom(std::as_bytes(std::span(fixture)));
        update->append(CPUFP32KVPage(write->commit(), heads, dimension));
    }
    return update->commit();
}
}

/** @test Capacity one, odd/short and long-over-equal-share sets match contiguous bytes. */
TEST(CPUFlashAttentionPagedRunner, SnapshotViewMatchesContiguousAcrossRunnerGeometries)
{
    constexpr int page_tokens = 16;
    for (const int heads : {1, 4})
    for (const int dimension : {64, 128})
    for (const int tokens : {1, 7, 16, 33, 150})
    {
        SCOPED_TRACE(::testing::Message() << "heads=" << heads << " dim=" << dimension << " tokens=" << tokens);
        const int kv_heads = heads == 1 ? 1 : 2;
        const size_t width = static_cast<size_t>(kv_heads * dimension);
        const size_t page_count = (static_cast<size_t>(tokens) + page_tokens - 1) / page_tokens;
        RunnerFixture fx(page_tokens, static_cast<int>(page_count) + 2);
        std::mt19937 random(7);
        std::uniform_real_distribution<float> values(-0.5f, 0.5f);
        std::vector<float> k(static_cast<size_t>(tokens) * width), v(k.size());
        for (auto *bank : {&k, &v}) for (auto &value : *bank) value = values(random);
        const int queries = 1;
        std::vector<float> q(static_cast<size_t>(queries * heads * dimension));
        for (auto &value : q) value = values(random);
        auto snapshot = publishRunnerSet(fx, k, v, page_tokens, kv_heads, dimension);
        // Request/table holders may retire; the snapshot retains every referent.
        fx.pool.reset();
        fx.table.reset();
        std::vector<float> expected(q.size(), -99.0f), actual(q.size(), -88.0f);
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.compute_decode(q.data(), k.data(), v.data(), expected.data(), queries,
            tokens, heads, kv_heads, dimension, true, tokens - queries));
        ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(q, snapshot, actual, queries, heads, true,
            -1, tokens - queries));
        EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
    }
}

/** @test Replacement publishes a newer table version; old readers keep their original bytes. */
TEST(CPUFlashAttentionPagedRunner, ReplacementRetainsOriginalSnapshotBytes)
{
    constexpr int page_tokens = 16;
    constexpr int heads = 1, dimension = 64, tokens = 20;
    const size_t width = static_cast<size_t>(heads * dimension);
    const size_t page_count = 2;
    RunnerFixture fx(page_tokens, static_cast<int>(page_count) * 2 + 1);
    std::mt19937 random(11);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    std::vector<float> k(static_cast<size_t>(tokens) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &value : *bank) value = values(random);
    std::vector<float> q(static_cast<size_t>(heads * dimension));
    for (auto &value : q) value = values(random);
    auto old = publishRunnerSet(fx, k, v, page_tokens, heads, dimension);
    auto old_reader = old.tryBorrow();
    ASSERT_TRUE(old_reader);
    // New request replaces the slot with different bytes.
    std::vector<float> k2(k), v2(v);
    for (auto &value : k2) value = -value;
    auto newest = publishRunnerSet(fx, k2, v2, page_tokens, heads, dimension);
    EXPECT_GT(newest.version(), old.version());
    std::vector<float> actual(q.size(), -88.0f);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(q, *old_reader, actual, 1, heads, true, -1, tokens - 1));
    // Contiguous control for the original bytes.
    std::vector<float> expected(q.size(), -99.0f);
    ASSERT_TRUE(kernel.compute_decode(q.data(), k.data(), v.data(), expected.data(), 1,
        tokens, heads, heads, dimension, true, tokens - 1));
    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
}

/** @test Unavailable grant growth refuses before any table element is published. */
TEST(CPUFlashAttentionPagedRunner, InsufficientGrowthRefusesBeforePublication)
{
    constexpr int page_tokens = 16;
    RunnerFixture fx(page_tokens, 1); // Only one frame: two requests cannot both fit.
    std::vector<float> k(16 * 64), v(16 * 64);
    auto one = publishRunnerSet(fx, k, v, page_tokens, 1, 64);
    EXPECT_NO_THROW(one.view());
    // A second request needs a frame but the pool has none left -> refused.
    auto grant = fx.pool->tryAdmit(1);
    EXPECT_FALSE(grant);
}

/** @test Two mixed-length requests and long-over-equal-share stay byte-exact, plus cancellation. */
TEST(CPUFlashAttentionPagedRunner, MixedLengthsAndLongOverEqualShareRemainByteExact)
{
    constexpr int page_tokens = 16;
    constexpr int heads = 1, dimension = 64;
    constexpr int short_tokens = 7, long_tokens = 150; // Short + long-over-equal-share.
    const size_t short_frames = (static_cast<size_t>(short_tokens) + page_tokens - 1) / page_tokens;
    const size_t long_frames = (static_cast<size_t>(long_tokens) + page_tokens - 1) / page_tokens;
    const int max_pages = static_cast<int>(short_frames + long_frames + 1);
    RunnerFixture fx(page_tokens, max_pages, dimension);
    std::mt19937 random(23);
    std::uniform_real_distribution<float> values(-0.6f, 0.6f);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::vector<float> k_short(static_cast<size_t>(short_tokens) * width), v_short(k_short.size());
    std::vector<float> k_long(static_cast<size_t>(long_tokens) * width), v_long(k_long.size());
    for (auto *bank : {&k_short, &v_short, &k_long, &v_long})
        for (auto &value : *bank) value = values(random);
    std::vector<float> q_short(static_cast<size_t>(heads * dimension)), q_long(q_short.size());
    for (auto &value : q_short) value = values(random);
    for (auto &value : q_long) value = values(random);

    auto short_snapshot = publishRunnerSet(fx, k_short, v_short, page_tokens, heads, dimension);
    auto long_snapshot = publishRunnerSet(fx, k_long, v_long, page_tokens, heads, dimension);
    fx.pool.reset();
    fx.table.reset(); // Snapshot retention is proof.

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    std::vector<float> expected_short(q_short.size(), -99.0f), actual_short(q_short.size(), -88.0f);
    std::vector<float> expected_long(q_long.size(), -99.0f), actual_long(q_long.size(), -88.0f);
    ASSERT_TRUE(kernel.compute_decode(q_short.data(), k_short.data(), v_short.data(), expected_short.data(),
        1, short_tokens, heads, heads, dimension, true, short_tokens - 1));
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(q_short, short_snapshot, actual_short, 1, heads,
        true, -1, short_tokens - 1));
    EXPECT_EQ(std::memcmp(expected_short.data(), actual_short.data(), actual_short.size() * sizeof(float)), 0);
    ASSERT_TRUE(kernel.compute_decode(q_long.data(), k_long.data(), v_long.data(), expected_long.data(),
        1, long_tokens, heads, heads, dimension, true, long_tokens - 1));
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(q_long, long_snapshot, actual_long, 1, heads,
        true, -1, long_tokens - 1));
    EXPECT_EQ(std::memcmp(expected_long.data(), actual_long.data(), actual_long.size() * sizeof(float)), 0);
}

/** @test Effective-use ordering on a REAL attachment (plan P3): reordered receipts, eviction after final ack. */
TEST(CPUFlashAttentionPagedRunner, EffectiveUseNoCopyHitOrdersEvictionAfterFinalAck)
{
    constexpr int page_tokens = 16;
    constexpr int heads = 1, dimension = 64;
    RunnerFixture fx(page_tokens, 4, dimension);
    std::mt19937 random(11);
    std::uniform_real_distribution<float> values(-0.9f, 0.9f);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::vector<float> k(static_cast<size_t>(64) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &value : *bank) value = values(random);
    std::vector<float> q(width);
    for (auto &value : q) value = values(random);
    auto snapshot = publishRunnerSet(fx, k, v, page_tokens, heads, dimension);
    const auto incarnation = snapshot.version();
    // Two consumers attach to the SAME immutable publication (no-copy hit);
    // each touch mints its own owned lease and ordinal.
    auto first = fx.table->touchEffectiveUse(snapshot);
    auto second = fx.table->touchEffectiveUse(snapshot);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    // A live incarnation cannot be retired early: refuses and records the
    // retirement request (completed by the final discharge).
    EXPECT_FALSE(fx.table->retireEffectiveUseReplica(incarnation)); // Live: refuse.
    // Reordered acknowledgement: acking the NEWER lease first must not evict
    // while the FIRST is still live.
    EXPECT_TRUE(second->acknowledge());
    EXPECT_FALSE(fx.table->effectiveUseMayEvict(incarnation));
    EXPECT_TRUE(first->acknowledge());
    EXPECT_TRUE(fx.table->effectiveUseMayEvict(incarnation));
    // Late-ack retry: the final discharge already completed the retirement,
    // so a second retire is an idempotent refusal (no duplicate transition).
    EXPECT_FALSE(fx.table->retireEffectiveUseReplica(incarnation));
    // Release the publication so its frames can be reused by the next
    // publication; the retired record slot was already reclaimed.
    snapshot = {};
    auto grant = fx.pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    auto next = fx.table->tryBeginUpdate();
    ASSERT_TRUE(next);
    std::vector<float> fixture(static_cast<size_t>(page_tokens) * 2 * width);
    for (size_t row = 0; row < static_cast<size_t>(page_tokens); ++row)
    {
        std::copy_n(k.data() + row * width, width, fixture.data() + row * 2 * width);
        std::copy_n(v.data() + row * width, width, fixture.data() + row * 2 * width + width);
    }
    const auto frame = grant->assignFrame();
    auto write = grant->prepareAppend(frame, fixture.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(fixture)));
    next->append(CPUFP32KVPage(write->commit(), heads, dimension));
    auto fresh = next->commit();
    EXPECT_GT(fresh.version(), snapshot.version());
}

/** @test A shared forward of two independent pooled requests matches per-request oracles. */
TEST(CPUFlashAttentionPagedRunner, TwoPooledRequestsShareOneForwardByteExactPerRequest)
{
    constexpr int page_tokens = 16;
    constexpr int heads = 1, dimension = 64;
    constexpr int short_tokens = 7, long_tokens = 120;
    RunnerFixture fx(page_tokens, 16, dimension);
    std::mt19937 random(47);
    std::uniform_real_distribution<float> values(-0.8f, 0.8f);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::vector<float> k_short(static_cast<size_t>(short_tokens) * width), v_short(k_short.size());
    std::vector<float> k_long(static_cast<size_t>(long_tokens) * width), v_long(k_long.size());
    for (auto *bank : {&k_short, &v_short, &k_long, &v_long})
        for (auto &value : *bank) value = values(random);
    std::vector<float> q_short(width), q_long(width);
    for (auto &value : q_short) value = values(random);
    for (auto &value : q_long) value = values(random);
    // Two independent publications on the SAME table: the shared forward
    // consumes both snapshots in one invocation without draining a cohort.
    auto short_snapshot = publishRunnerSet(fx, k_short, v_short, page_tokens, heads, dimension);
    auto long_snapshot = publishRunnerSet(fx, k_long, v_long, page_tokens, heads, dimension);
    fx.pool.reset();
    fx.table.reset(); // Snapshot retention is the only owner.
    // Oracle: the equivalent grouped decode (two rows, each against its own
    // pooled KV source). The kernel entry is per-request; the runner-level
    // grouped forward calls both in one step and stays byte-identical.
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    std::vector<float> expected_short(q_short.size(), -99.0f), actual_short(q_short.size(), -88.0f);
    std::vector<float> expected_long(q_long.size(), -99.0f), actual_long(q_long.size(), -88.0f);
    ASSERT_TRUE(kernel.compute_decode(q_short.data(), k_short.data(), v_short.data(), expected_short.data(),
        1, short_tokens, heads, heads, dimension, true, short_tokens - 1));
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(q_short, short_snapshot, actual_short, 1, heads,
        true, -1, short_tokens - 1));
    ASSERT_TRUE(kernel.compute_decode(q_long.data(), k_long.data(), v_long.data(), expected_long.data(),
        1, long_tokens, heads, heads, dimension, true, long_tokens - 1));
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(q_long, long_snapshot, actual_long, 1, heads,
        true, -1, long_tokens - 1));
    EXPECT_EQ(std::memcmp(expected_short.data(), actual_short.data(), actual_short.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(expected_long.data(), actual_long.data(), actual_long.size() * sizeof(float)), 0);
}

/** @test Truncate -> export -> re-attach import stays byte-exact vs contiguous. */
TEST(CPUFlashAttentionPagedRunner, TruncateExportReattachStaysByteExact)
{
    constexpr int page_tokens = 16;
    constexpr int heads = 1, dimension = 64;
    constexpr int source_tokens = 80, visible = 33; // Truncated mid-page.
    RunnerFixture fx(page_tokens, 8, dimension);
    std::mt19937 random(31);
    std::uniform_real_distribution<float> values(-0.7f, 0.7f);
    const size_t width = static_cast<size_t>(heads * dimension);
    std::vector<float> k(static_cast<size_t>(source_tokens) * width), v(k.size());
    for (auto *bank : {&k, &v}) for (auto &value : *bank) value = values(random);
    std::vector<float> q(static_cast<size_t>(width));
    for (auto &value : q) value = values(random);
    // Publish the full 80-token set, then truncate to the first 33 tokens.
    auto snapshot = publishRunnerSet(fx, k, v, page_tokens, heads, dimension);
    auto clipped = snapshot.view().prefix(static_cast<std::size_t>(visible));
    EXPECT_EQ(clipped.rows(), visible);
    // Export the truncated view to an archive, then import it into a fresh
    // request on the SAME table through the production attach path.
    auto archive = PagedKVArchive::fromView(clipped);
    auto grant = fx.pool->tryAdmit(3);
    ASSERT_TRUE(grant);
    auto update = fx.table->tryBeginUpdate();
    ASSERT_TRUE(update);
    update->attachArchivedPrefix(archive, *grant, page_tokens);
    auto reattached = update->commit();
    fx.pool.reset();
    fx.table.reset();
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    std::vector<float> expected(q.size(), -99.0f), actual(q.size(), -88.0f);
    ASSERT_TRUE(kernel.compute_decode(q.data(), k.data(), v.data(), expected.data(), 1,
        visible, heads, heads, dimension, true, visible - 1));
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(q, reattached, actual, 1, heads,
        true, -1, visible - 1));
    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
}
