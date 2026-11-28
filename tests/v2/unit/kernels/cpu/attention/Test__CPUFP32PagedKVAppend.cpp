/**
 * @file Test__CPUFP32PagedKVAppend.cpp
 * @brief Native paged FP32 producer, exact byte completion and immutable-tail contracts.
 *
 * Real PMA frames receive separated K/V rows directly. Tests compare every row
 * bit and execute optimized paged attention against the contiguous production
 * control. Malformed production never publishes partial byte extents.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/attention/CPUFP32PagedKVAppend.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <array>
#include <cstring>
#include <bit>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief Exact PMA backing fixture; neither request capacity nor payload is mirrored. */
struct AppendOwner
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::shared_ptr<PhysicalPagePool> pool;
    PhysicalPagePool::Grant grant;
    /** @brief Admit three fungible frames and one private request under the actual root BOM. */
    explicit AppendOwner(size_t frame_bytes = 4096)
    {
        const auto device = DeviceId::cpu();
        const PhysicalPagePoolGeometry geometry{3, frame_bytes, 1, 3};
        const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool = authority->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
        grant = std::move(*pool->tryAdmit(3));
    }
};
}

/** @test Actual separated K/V inputs and a retained partial tail produce exact optimized attention operands. */
TEST(CPUFP32PagedKVAppend, DirectRowsAndPartialTailPreserveNativeAttentionBytes)
{
    AppendOwner owner;
    alignas(64) std::array<float, 512> keys, values;
    alignas(64) std::array<float, 256> query{}, expected{}, actual{};
    for (size_t i = 0; i < keys.size(); ++i)
    {
        keys[i] = (static_cast<int>(i % 17) - 8) * 0.03125f;
        values[i] = (static_cast<int>(i % 23) - 11) * 0.0625f;
    }
    for (size_t i = 0; i < query.size(); ++i) query[i] = (static_cast<int>(i % 11) - 5) * 0.03125f;
    auto tail = appendCPUFP32KVPage(owner.grant, owner.grant.assignFrame(), {
        .keys=std::span(keys).first(256), .values=std::span(values).first(256),
        .heads=2, .dimension=64, .page_tokens=4});
    ASSERT_TRUE(tail);
    auto full = appendCPUFP32KVPage(owner.grant, owner.grant.assignFrame(), {
        .keys=std::span(keys).subspan(256), .values=std::span(values).subspan(256),
        .heads=2, .dimension=64, .page_tokens=4, .tail=std::cref(*tail)});
    ASSERT_TRUE(full);
    EXPECT_EQ(tail->rows(), 2);
    EXPECT_EQ(full->rows(), 4);
    for (int row = 0; row < 4; ++row)
    {
        EXPECT_EQ(std::memcmp(full->key(row, 0), keys.data() + row * 128, 128 * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(full->value(row, 0), values.data() + row * 128, 128 * sizeof(float)), 0);
    }
    CPUFP32PagedKVView view(std::span<const CPUFP32KVPage>(&*full, 1), 4);
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    ASSERT_TRUE(kernel.compute_decode(query.data(), keys.data(), values.data(), expected.data(), 1,
        4, 4, 2, 64, true, 3));
    ASSERT_TRUE(kernel.compute_paged_fp32(query, view, actual, 1, 4, true, -1, 3));
    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)), 0);
}

/** @test Positive, odd and full/partial page geometries retain every original FP32 bit without padding. */
TEST(CPUFP32PagedKVAppend, ExactGeometryAndBitPreservationSweep)
{
    for (const int block : {1, 3, 16})
    for (const int heads : {1, 3})
    for (const int dimension : {1, 7, 64})
    for (const int prior : {0, block / 2, block - 1})
    for (const int added : {1, block - prior})
    {
        const auto layout = KVNativeStorageLayout::resolve(KVCacheFamily::AttentionOnly,
            heads, dimension, "fp32", DeviceId::cpu());
        AppendOwner owner(layout.pagePayload(block).totalBytes());
        const size_t width = static_cast<size_t>(heads * dimension);
        std::vector<float> keys(static_cast<size_t>(block) * width), values(keys.size());
        for (size_t i = 0; i < keys.size(); ++i)
        {
            keys[i] = std::bit_cast<float>(i % 13 == 0 ? 0x80000000u : 0x3f000000u + static_cast<uint32_t>(i % 127));
            values[i] = std::bit_cast<float>(i % 11 == 0 ? 0x7fc00001u : 0xbf000000u + static_cast<uint32_t>(i % 127));
        }
        std::optional<CPUFP32KVPage> tail;
        if (prior)
        {
            tail = appendCPUFP32KVPage(owner.grant, owner.grant.assignFrame(), {
                .keys=std::span(keys).first(prior * width), .values=std::span(values).first(prior * width),
                .heads=heads, .dimension=dimension, .page_tokens=block});
            ASSERT_TRUE(tail);
        }
        auto result = appendCPUFP32KVPage(owner.grant, owner.grant.assignFrame(), {
            .keys=std::span(keys).subspan(prior * width, added * width),
            .values=std::span(values).subspan(prior * width, added * width),
            .heads=heads, .dimension=dimension, .page_tokens=block,
            .tail=tail ? std::optional(std::cref(*tail)) : std::nullopt});
        ASSERT_TRUE(result);
        ASSERT_EQ(result->rows(), prior + added);
        EXPECT_EQ(result->storage().size(), layout.pagePayload(prior + added).totalBytes());
        for (int row = 0; row < prior + added; ++row)
        {
            EXPECT_EQ(std::memcmp(result->key(row, 0), keys.data() + row * width, width * sizeof(float)), 0);
            EXPECT_EQ(std::memcmp(result->value(row, 0), values.data() + row * width, width * sizeof(float)), 0);
        }
        if (tail) EXPECT_EQ(tail->rows(), prior);
    }
}

/** @test Unaligned physical frames fail before producing a native publication, rather than selecting another frame. */
TEST(CPUFP32PagedKVAppend, RejectsActualMisalignedDestination)
{
    AppendOwner owner(513);
    (void)owner.grant.assignFrame();
    const auto unaligned = owner.grant.assignFrame();
    std::array<float, 64> keys{}, values{};
    EXPECT_THROW(appendCPUFP32KVPage(owner.grant, unaligned, {
        .keys=keys, .values=values, .heads=1, .dimension=64, .page_tokens=1}), std::invalid_argument);
    EXPECT_TRUE(owner.grant.prepareFreshWrite(unaligned, 512)); // No partial extent was committed.
}

/** @test Incremental CPU writes cannot publish incomplete extents, overrun them or survive cancellation. */
TEST(CPUFP32PagedKVAppend, SequentialCompletionAndCancellationRequireActualOriginalWrites)
{
    AppendOwner owner;
    const auto frame = owner.grant.assignFrame();
    auto plan = owner.grant.prepareFreshWrite(frame, 32);
    ASSERT_TRUE(plan);
    std::array<std::byte, 32> bytes{};
    plan->copyNext(std::span(bytes).first(16));
    EXPECT_THROW(plan->commit(), std::logic_error);
    EXPECT_THROW(plan->copyFrom(bytes), std::logic_error);
    EXPECT_THROW(plan->copyNext({}), std::logic_error);
    EXPECT_THROW(plan->copyNext(std::span(bytes).first(17)), std::logic_error);
    owner.grant = {};
    EXPECT_THROW(plan->copyNext(std::span(bytes).first(16)), std::logic_error);
    EXPECT_THROW(plan->commit(), std::logic_error);
    EXPECT_FALSE(owner.pool->tryAdmit(3));
    plan.reset();
    EXPECT_TRUE(owner.pool->tryAdmit(3));
    AppendOwner completed_owner;
    auto completed = completed_owner.grant.prepareFreshWrite(completed_owner.grant.assignFrame(), 32);
    ASSERT_TRUE(completed);
    completed->copyNext(std::span(bytes).first(16));
    auto transferred = std::move(*completed);
    EXPECT_THROW(completed->commit(), std::logic_error);
    transferred.copyNext(std::span(bytes).subspan(16));
    completed_owner.grant = {};
    EXPECT_THROW(transferred.commit(), std::logic_error); // All bytes exist, but their owner was canceled.
}

/** @test A recycled destination never overwrites an aliasing source or accepts another PMA's tail. */
TEST(CPUFP32PagedKVAppend, AliasAndForeignTailRefusalLeaveDestinationUnpublished)
{
    AppendOwner owner;
    alignas(64) std::array<float, 64> keys{}, values{};
    const auto old = owner.grant.assignFrame();
    auto page = appendCPUFP32KVPage(owner.grant, old, {
        .keys=keys, .values=values, .heads=1, .dimension=64, .page_tokens=2});
    ASSERT_TRUE(page);
    {
        AppendOwner foreign;
        const auto frame = foreign.grant.assignFrame();
        EXPECT_THROW(appendCPUFP32KVPage(foreign.grant, frame, {
            .keys=keys, .values=values, .heads=1, .dimension=64, .page_tokens=2,
            .tail=std::cref(*page)}), std::invalid_argument);
        EXPECT_TRUE(foreign.grant.prepareFreshWrite(frame, 512));
    }
    // Deliberately malformed escaped address: it is compared, never dereferenced.
    const std::span<const float> alias(page->key(0, 0), keys.size());
    page.reset();
    owner.grant.discardFrame(old);
    const auto fresh = owner.grant.assignFrame();
    ASSERT_EQ(fresh.index, old.index);
    EXPECT_THROW(appendCPUFP32KVPage(owner.grant, fresh, {
        .keys=alias, .values=values, .heads=1, .dimension=64, .page_tokens=2}), std::invalid_argument);
    EXPECT_TRUE(owner.grant.prepareFreshWrite(fresh, 512));
}
