/**
 * @file Test__CPUFP32PagedKVTable.cpp
 * @brief Native CPU attention retains exact immutable metadata banks and payload owners.
 *
 * Lifecycle/geometry checks are not recorded replay. They exercise bounded
 * metadata pressure, old/new native reads, aborted publication and actual
 * storage release, without a model, GPU, contiguous gather or hot allocation.
 */
#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include <gtest/gtest.h>
#include <vector>

using namespace llaminar2;

/** @test Metadata construction is exactly admitted and rolls back all roots on a short budget. */
TEST(CPUFP32PagedKVTable, MetadataGeometryAndPartialConstructionRollback)
{
    const auto device = DeviceId::cpu();
    EXPECT_THROW(CPUFP32PagedKVTable::physicalBytes({}, device), std::invalid_argument);
    for (const size_t banks : {1u, 3u, 5u})
    {
        const CPUFP32PagedKVTableGeometry geometry{3, banks, 2, 16};
        const auto bytes = CPUFP32PagedKVTable::physicalBytes(geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes - 1);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        EXPECT_THROW(CPUFP32PagedKVTable::create(authority, device, geometry), std::exception);
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::KVCache,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}

/** @test Two published table versions remain immutable until their original borrowers retire. */
TEST(CPUFP32PagedKVTable, PublishedBanksRetainOriginalNativeOperands)
{
    constexpr int dimension = 64;
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry pool_geometry{3, 2 * 2 * dimension * sizeof(float), 3, 1};
    const CPUFP32PagedKVTableGeometry table_geometry{1, 2, 2, 2};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    const auto page = [&](float value) {
        auto grant = pool->tryAdmit(1);
        if (!grant) throw std::logic_error("Fixture page grant refused");
        const auto frame = grant->assignFrame();
        std::vector<float> input(4 * dimension, 0.0f);
        for (int token = 0; token < 2; ++token)
            std::fill_n(input.data() + token * 2 * dimension + dimension, dimension, value);
        auto write = grant->prepareAppend(frame, input.size() * sizeof(float));
        if (!write) throw std::logic_error("Fixture page write refused");
        write->copyFrom(std::as_bytes(std::span(input)));
        return CPUFP32KVPage(write->commit(), 1, dimension);
    };
    auto first = table->tryBeginUpdate();
    ASSERT_TRUE(first);
    first->append(page(4.0f));
    auto old = first->commit();
    auto reader = old.tryBorrow();
    ASSERT_TRUE(reader);
    EXPECT_FALSE(old.tryBorrow());
    auto next = table->tryBeginUpdate();
    ASSERT_TRUE(next);
    next->append(page(9.0f));
    auto current = next->commit();
    EXPECT_GT(current.version(), old.version());
    EXPECT_FALSE(table->tryBeginUpdate());
    table->closeBorrowers(old);
    old = {};
    EXPECT_THROW(old.view(), std::logic_error);
    EXPECT_FALSE(reader->tryBorrow()); // Below the reader limit, but this version's frontier is closed.
    EXPECT_FALSE(table->tryBeginUpdate());
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    std::vector<float> query(dimension, 0.0f), output(dimension, -1.0f);
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, *reader, output, 1, 1, false));
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float value) { return value == 4.0f; }));
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, current, output, 1, 1, false));
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float value) { return value == 9.0f; }));
    reader.reset();
    auto aborted = table->tryBeginUpdate();
    ASSERT_TRUE(aborted);
    EXPECT_THROW(aborted->commit(), std::invalid_argument); // No dummy page/empty compute publication.
    aborted.reset();
    auto replacement = table->tryBeginUpdate();
    ASSERT_TRUE(replacement);
    replacement->append(page(7.0f));
    auto newest = replacement->commit();
    EXPECT_GT(newest.version(), current.version() + 1); // Aborted version is not recycled.
    EXPECT_FALSE(table->tryBeginUpdate());
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, current, output, 1, 1, false));
    EXPECT_EQ(output.front(), 9.0f);
    {
        const auto foreign_bytes = CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
        PhysicalMemoryPlanBuilder foreign_plan;
        foreign_plan.add({.world_rank=0, .device=device, .total_bytes=foreign_bytes, .admission_available_bytes=foreign_bytes},
            PhysicalMemoryOwner::KVCache, foreign_bytes);
        auto foreign_authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(foreign_plan.build()), 0);
        auto foreign_table = CPUFP32PagedKVTable::create(foreign_authority, device, table_geometry);
        auto foreign_update = foreign_table->tryBeginUpdate();
        ASSERT_TRUE(foreign_update);
        auto original_page = page(11.0f);
        EXPECT_THROW(foreign_update->append(std::move(original_page)), std::invalid_argument);
        EXPECT_EQ(original_page.rows(), 2); // Refusal did not consume the original owner.
        EXPECT_EQ(original_page.value(0, 0)[0], 11.0f);
    }
    table.reset(); pool.reset();
    EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), bytes);
    current = {};
    ASSERT_TRUE(kernel.compute_paged_snapshot_fp32(query, newest, output, 1, 1, false));
    EXPECT_EQ(output.front(), 7.0f);
    newest = {};
    EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}
