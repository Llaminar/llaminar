/**
 * @file Test__PhysicalMemoryRankHostPool.cpp
 * @brief Joint admission of a rank's CPU and integrated (unified-memory) GPUs.
 *
 * On GB10-class hardware the CPU allocator and the CUDA allocator draw on one
 * LPDDR pool. Each allocator fitting its own sample must not be enough: the
 * plan has to reject CPU60+GPU60 against a pool of 100, while discrete GPUs
 * keep fully independent capacity. These tests drive the canonical
 * PhysicalMemoryAuthority surface only; they are device-free.
 */
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/PlanningObservedResource.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace llaminar2;

namespace
{
    /** @brief Host allocator observation. */
    PhysicalMemoryResource host(int rank, std::size_t total, std::size_t available)
    {
        return {.world_rank = rank, .device = DeviceId::cpu(), .total_bytes = total,
                .admission_available_bytes = available};
    }

    /** @brief Integrated GPU sharing the rank host pool. */
    PhysicalMemoryResource pooled(int rank, DeviceId device, std::size_t total, std::size_t available,
                                  std::size_t pool)
    {
        return {.world_rank = rank, .device = device, .total_bytes = total,
                .admission_available_bytes = available,
                .backing = PhysicalMemoryBacking::RankHostPool, .pool_available_bytes = pool};
    }

    /** @brief Discrete GPU with dedicated memory. */
    PhysicalMemoryResource discrete(int rank, DeviceId device, std::size_t total, std::size_t available)
    {
        return {.world_rank = rank, .device = device, .total_bytes = total,
                .admission_available_bytes = available};
    }
}

/** @test CPU60 + integrated GPU60 over a pool of 100 is rejected although each fits alone. */
TEST(PhysicalMemoryRankHostPool, JointDemandOverSharedPoolIsRejected)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(0, 128u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u)
        .add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u);
    const auto plan = builder.build();
    for (const auto &bom : plan.resources())
        EXPECT_TRUE(bom.fits()) << bom.summary();
    EXPECT_FALSE(plan.fits());
    const auto violation = plan.rankHostPoolViolation();
    ASSERT_TRUE(violation.has_value());
    EXPECT_NE(violation->find("incremental=120"), std::string::npos) << *violation;
    EXPECT_NE(violation->find("pool_available=100"), std::string::npos) << *violation;
    EXPECT_THROW(PhysicalMemoryPlanAdmissionCertificate{plan}, PhysicalMemoryCapacityExhausted);
}

/** @test The same demand on a discrete GPU is admitted: dedicated memory is not pooled. */
TEST(PhysicalMemoryRankHostPool, DiscreteGpuKeepsIndependentCapacity)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(0, 128u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u)
        .add(discrete(0, DeviceId::cuda(0), 128u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u);
    const auto plan = builder.build();
    EXPECT_TRUE(plan.fits());
    EXPECT_FALSE(plan.rankHostPoolViolation().has_value());
    EXPECT_NO_THROW(PhysicalMemoryPlanAdmissionCertificate{plan});
}

/** @test Joint demand that fits is admitted, at the exact boundary. */
TEST(PhysicalMemoryRankHostPool, JointDemandAtPoolBoundaryIsAdmitted)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(0, 128u, 100u), PhysicalMemoryOwner::KVCache, 40u)
        .add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u);
    EXPECT_TRUE(builder.build().fits());
}

/** @test Retained credit is excluded once per allocator, not credited twice to the pool. */
TEST(PhysicalMemoryRankHostPool, ResidencyCreditCountsOnceInJointDemand)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(0, 128u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u, 20u)
        .add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u);
    const auto plan = builder.build();
    EXPECT_TRUE(plan.fits()) << plan.summary(); // 40 + 60 = 100
    PhysicalMemoryPlanBuilder over;
    over.add(host(0, 128u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 61u, 20u)
        .add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 60u);
    EXPECT_FALSE(over.build().fits());
}

/** @test The smallest pool observation bounds the pool; capacities are never summed. */
TEST(PhysicalMemoryRankHostPool, TwoPooledGpusUseSmallestPoolSampleWithoutSumming)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::KVCache, 50u)
        .add(pooled(0, DeviceId::cuda(1), 128u, 100u, 90u), PhysicalMemoryOwner::KVCache, 41u);
    EXPECT_FALSE(builder.build().fits()); // 91 > min(100, 90)
    PhysicalMemoryPlanBuilder fits;
    fits.add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::KVCache, 50u)
        .add(pooled(0, DeviceId::cuda(1), 128u, 100u, 90u), PhysicalMemoryOwner::KVCache, 40u);
    EXPECT_TRUE(fits.build().fits());
}

/** @test A user cap on the CPU allocator does not shrink the GPU's pool limit. */
TEST(PhysicalMemoryRankHostPool, PerAllocatorCapAndPoolLimitAreIndependent)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(0, 128u, 20u), PhysicalMemoryOwner::KVCache, 20u)
        .add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 80u);
    EXPECT_TRUE(builder.build().fits());
    PhysicalMemoryPlanBuilder capped;
    capped.add(host(0, 128u, 20u), PhysicalMemoryOwner::KVCache, 21u)
        .add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 10u);
    EXPECT_FALSE(capped.build().fits()) << "The CPU cap still applies to the CPU allocator itself";
}

/** @test Pools are per rank: another rank's host demand is not charged to this pool. */
TEST(PhysicalMemoryRankHostPool, PoolsAreRankScoped)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(1, 128u, 100u), PhysicalMemoryOwner::KVCache, 90u)
        .add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::PrimaryModelWeights, 90u);
    EXPECT_TRUE(builder.build().fits());
}

/** @test Two contributors that disagree on backing for one allocator are rejected. */
TEST(PhysicalMemoryRankHostPool, ConflictingBackingObservationsAreRejected)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::KVCache, 10u);
    EXPECT_THROW(builder.add(discrete(0, DeviceId::cuda(0), 128u, 100u), PhysicalMemoryOwner::KVCache, 10u),
                 std::invalid_argument);
}

/** @test Backing coherence: only GPUs can be pooled; dedicated resources carry no pool figure. */
TEST(PhysicalMemoryRankHostPool, BackingValidationRejectsIncoherentResources)
{
    auto cpu_pooled = host(0, 128u, 100u);
    cpu_pooled.backing = PhysicalMemoryBacking::RankHostPool;
    cpu_pooled.pool_available_bytes = 100u;
    EXPECT_FALSE(cpu_pooled.valid());
    auto dedicated_with_pool = discrete(0, DeviceId::cuda(0), 128u, 100u);
    dedicated_with_pool.pool_available_bytes = 1u;
    EXPECT_FALSE(dedicated_with_pool.valid());
    EXPECT_TRUE(pooled(0, DeviceId::cuda(0), 128u, 0u, 0u).valid()) << "An exhausted pool is a legitimate observation";
    EXPECT_THROW((void)physicalMemoryBackingFromWire(7u), std::invalid_argument);
}

/** @test A retained envelope rejects a later plan that changes physical backing. */
TEST(PhysicalMemoryRankHostPool, RetainedEnvelopeRejectsBackingChange)
{
    PhysicalMemoryPlanBuilder admitted;
    admitted.add(pooled(0, DeviceId::cuda(0), 128u, 100u, 100u), PhysicalMemoryOwner::KVCache, 10u);
    PhysicalMemoryPlanBuilder required;
    required.add(discrete(0, DeviceId::cuda(0), 128u, 100u), PhysicalMemoryOwner::KVCache, 10u);
    const auto mismatch = admitted.build().requiredFootprintMismatch(required.build());
    ASSERT_TRUE(mismatch.has_value());
    EXPECT_NE(mismatch->find("backing"), std::string::npos) << *mismatch;
}

/** @test Measured runtime context on a pooled GPU protects the CPU's unmaterialized demand too. */
TEST(PhysicalMemoryRankHostPool, RuntimeContextProtectsPoolMatesPendingDemand)
{
    const auto gpu = DeviceId::cuda(0);
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(0, 1000u, 800u), PhysicalMemoryOwner::KVCache, 300u)
        .add(pooled(0, gpu, 1000u, 800u, 800u), PhysicalMemoryOwner::PrimaryModelWeights, 200u);
    PhysicalMemoryAuthority memory(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    // Fresh driver figure on an integrated GPU samples the shared pool.
    const auto fresh = pooled(0, gpu, 1000u, 800u, 800u);
    // Pending demand is GPU 200 + CPU 300 = 500 against 800 available.
    EXPECT_THROW((void)memory.claimMeasuredRuntimeContextStorage(fresh, 301u), PhysicalMemoryCapacityExhausted);
    auto context = memory.claimMeasuredRuntimeContextStorage(fresh, 300u);
    EXPECT_EQ(context.bytes(), 300u);
}

/** @test The same scenario on a discrete GPU protects only that GPU's own demand. */
TEST(PhysicalMemoryRankHostPool, RuntimeContextOnDiscreteGpuIgnoresHostDemand)
{
    const auto gpu = DeviceId::cuda(0);
    PhysicalMemoryPlanBuilder builder;
    builder.add(host(0, 1000u, 800u), PhysicalMemoryOwner::KVCache, 300u)
        .add(discrete(0, gpu, 1000u, 800u), PhysicalMemoryOwner::PrimaryModelWeights, 200u);
    PhysicalMemoryAuthority memory(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    auto context = memory.claimMeasuredRuntimeContextStorage(discrete(0, gpu, 1000u, 800u), 600u);
    EXPECT_EQ(context.bytes(), 600u);
}

/** @test Inventory projection attaches the discovered backing and uncapped pool sample. */
TEST(PhysicalMemoryRankHostPool, InventoryProjectionCarriesIntegratedBacking)
{
    RankInventory rank;
    rank.rank = 0;
    rank.cpu.type = DeviceType::CPU;
    rank.cpu.memory_bytes = 128u;
    rank.cpu.free_memory_bytes = 100u;
    DeviceInfo integrated;
    integrated.type = DeviceType::CUDA;
    integrated.local_device_id = 0;
    integrated.memory_bytes = 128u;
    integrated.free_memory_bytes = 97u;
    integrated.memory_backing = PhysicalMemoryBacking::RankHostPool;
    DeviceInfo dedicated = integrated;
    dedicated.local_device_id = 1;
    dedicated.memory_backing = PhysicalMemoryBacking::Dedicated;
    rank.gpus = {integrated, dedicated};

    const auto pooled_gpu = planningObservedResource(rank, DeviceId::cuda(0));
    EXPECT_EQ(pooled_gpu.backing, PhysicalMemoryBacking::RankHostPool);
    EXPECT_EQ(pooled_gpu.pool_available_bytes, 97u);
    const auto discrete_gpu = planningObservedResource(rank, DeviceId::cuda(1));
    EXPECT_EQ(discrete_gpu.backing, PhysicalMemoryBacking::Dedicated);
    EXPECT_EQ(discrete_gpu.pool_available_bytes, 0u);
    const auto cpu = planningObservedResource(rank, DeviceId::cpu());
    EXPECT_EQ(cpu.backing, PhysicalMemoryBacking::Dedicated);
}
