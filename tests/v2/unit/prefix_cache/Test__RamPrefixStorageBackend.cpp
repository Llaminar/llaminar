/**
 * @file Test__RamPrefixStorageBackend.cpp
 * @brief Unit coverage for typed RAM prefix payloads and readiness ownership.
 */

#include <gtest/gtest.h>

#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

using namespace llaminar2;

namespace
{
    PrefixPayloadLayout makeLayout(size_t k_bytes = 16, size_t v_bytes = 16)
    {
        PrefixPayloadLayout layout;
        layout.block_size = 2;
        layout.fa_layers = 1;
        layout.total_layers = 1;
        layout.bytes_per_fa_layer_k = k_bytes;
        layout.bytes_per_fa_layer_v = v_bytes;
        return layout;
    }

    PrefixCacheKey keyFor(int block)
    {
        return makePrefixCacheKey(0xfeed, 0, block, block * 2, {block, block + 1});
    }

    /** @brief Admit one readable CPU prefix-tier capacity for focused tests. */
    std::shared_ptr<PhysicalMemoryAuthority> makePrefixAuthority(
        size_t prefix_bytes)
    {
        PhysicalMemoryPlanBuilder builder;
        builder.add(
            PhysicalMemoryResource{
                .world_rank = 0,
                .device = DeviceId::cpu(),
                .total_bytes = 4096u,
                .admission_available_bytes = 4096u,
            },
            PhysicalMemoryOwner::PrefixHostTier,
            prefix_bytes);
        auto admission = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
        return std::make_shared<PhysicalMemoryAuthority>(
            std::move(admission), 0);
    }
} // namespace

TEST(Test__RamPrefixStorageBackend, AllocatesTypedPayloadSegmentsWithinBudget)
{
    RamPrefixStorageBackend backend(256);
    PrefixPayloadLayout layout = makeLayout();
    layout.includes_terminal_hidden = true;
    layout.terminal_hidden_bytes = 8;

    auto handle = backend.allocate(keyFor(0), layout);
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(handle.total_bytes, 40u);
    EXPECT_EQ(backend.usedBytes(), 40u);
    ASSERT_NE(handle.kvKData(), nullptr);
    ASSERT_NE(handle.kvVData(), nullptr);
    ASSERT_NE(handle.terminal_hidden, nullptr);

    std::fill(handle.kvKData(), handle.kvKData() + handle.kvKBytes(), 0x11);
    std::fill(handle.kvVData(), handle.kvVData() + handle.kvVBytes(), 0x22);
    EXPECT_EQ(handle.kv_storage->front(), 0x11);
    EXPECT_EQ(handle.kv_storage->at(handle.kvKBytes()), 0x22);

    EXPECT_TRUE(backend.release(handle));
    EXPECT_EQ(backend.usedBytes(), 0u);
    EXPECT_FALSE(backend.release(handle));
}

TEST(Test__RamPrefixStorageBackend, AllocatesHybridPayloadSegment)
{
    RamPrefixStorageBackend backend(256);
    PrefixPayloadLayout layout = makeLayout();
    layout.includes_hybrid_state = true;
    layout.hybrid_host_state_bytes = 10;
    layout.hybrid_state_bytes = 10;

    auto handle = backend.allocate(keyFor(0), layout);
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(handle.total_bytes, 42u);
    ASSERT_NE(handle.kvKData(), nullptr);
    ASSERT_NE(handle.kvVData(), nullptr);
    ASSERT_NE(handle.hybrid_payload, nullptr);
    ASSERT_NE(handle.hybrid_storage, nullptr);
    EXPECT_EQ(handle.hybrid_storage->size(), 10u);
    EXPECT_FALSE(handle.has_hybrid_state);
}

TEST(Test__RamPrefixStorageBackend, RejectsBlocksThatDoNotFit)
{
    RamPrefixStorageBackend backend(31);
    EXPECT_FALSE(backend.canStore(32));

    auto handle = backend.allocate(keyFor(0), makeLayout());
    EXPECT_FALSE(handle.valid());
    EXPECT_EQ(backend.usedBytes(), 0u);
}

TEST(Test__RamPrefixStorageBackend,
     ProductionCapacityAndPayloadClaimsFollowPhysicalLifetime)
{
    auto authority = makePrefixAuthority(96u);
    std::string error;
    auto backend = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 96u, authority, &error);
    ASSERT_NE(backend, nullptr) << error;
    EXPECT_TRUE(backend->accounted());
    EXPECT_EQ(
        authority->reservedBytes(
            DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier),
        96u);

    auto handle = backend->allocate(keyFor(0), makeLayout());
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrefixHostTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        32u);

    // Cache retirement only removes the key. A request-held copy must retain
    // both its backing vectors and the exact ledger claim.
    PrefixBlockHandle retained = handle;
    ASSERT_TRUE(backend->release(handle));
    handle = {};
    backend.reset();
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrefixHostTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        32u);
    EXPECT_EQ(
        authority->reservedBytes(
            DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier),
        96u);

    retained = {};
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrefixHostTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        0u);
    EXPECT_EQ(
        authority->reservedBytes(
            DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier),
        0u);
}

TEST(Test__RamPrefixStorageBackend,
     ModelRuntimeStateUsesTheSameBoundedHostTier)
{
    auto authority = makePrefixAuthority(64u);
    auto backend = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 64u, authority);
    ASSERT_NE(backend, nullptr);
    auto handle = backend->allocate(keyFor(0), makeLayout());
    ASSERT_TRUE(handle.valid());

    auto runtime_state =
        std::make_shared<std::vector<uint8_t>>(16u, uint8_t{0x5a});
    ASSERT_TRUE(backend->attachModelRuntimeState(
        &handle, std::move(runtime_state)));
    EXPECT_EQ(handle.total_bytes, 48u);
    EXPECT_EQ(backend->usedBytes(), 48u);
    EXPECT_TRUE(handle.has_model_runtime_state);
    ASSERT_NE(handle.model_runtime_state_storage, nullptr);
    EXPECT_EQ(handle.model_runtime_state_storage->front(), 0x5a);
    EXPECT_EQ(
        authority->claimedBytes(
            DeviceId::cpu(),
            PhysicalMemoryOwner::PrefixHostTier,
            PhysicalMemoryMaterializationKind::NewAllocation),
        48u);
}

TEST(Test__RamPrefixStorageBackend,
     IndependentPoolsCannotSpendTheSameAdmittedCapacity)
{
    auto authority = makePrefixAuthority(128u);
    auto first = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 96u, authority);
    ASSERT_NE(first, nullptr);

    std::string error;
    auto conflicting = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 64u, authority, &error);
    EXPECT_EQ(conflicting, nullptr);
    EXPECT_NE(error.find("exceeds admitted"), std::string::npos);
}

/**
 * @brief Proves readiness publication is a two-phase, single-use contract.
 *
 * Unit tests never execute GPU work. Opaque non-null values stand in for an
 * event and stream so this test can validate the ownership state machine
 * without constructing a backend context.
 */
TEST(Test__RamPrefixStorageBackend, PayloadReadinessRequiresPreparationBeforePublication)
{
    PrefixPayloadReadiness readiness;
    auto event_owner = std::make_shared<uint8_t>(0u);
    void *stream = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1234u));

    EXPECT_FALSE(readiness.prepared());
    EXPECT_FALSE(readiness.published());
    EXPECT_FALSE(readiness.publishRecorded())
        << "An unprepared event must never become visible to consumers.";

    ASSERT_TRUE(readiness.prepare(
        std::static_pointer_cast<void>(event_owner),
        DeviceId::cuda(0),
        stream));
    EXPECT_TRUE(readiness.prepared());
    EXPECT_FALSE(readiness.published());
    EXPECT_EQ(readiness.event(), event_owner.get());
    EXPECT_EQ(readiness.producerDevice(), DeviceId::cuda(0));
    EXPECT_EQ(readiness.producerStream(), stream);

    EXPECT_FALSE(readiness.prepare(
        std::static_pointer_cast<void>(event_owner),
        DeviceId::cuda(0),
        stream))
        << "A handle has exactly one producer event and stream.";
    ASSERT_TRUE(readiness.publishRecorded());
    EXPECT_TRUE(readiness.published());
    EXPECT_FALSE(readiness.publishRecorded())
        << "Publishing one recorded event twice would hide lifecycle misuse.";
}
