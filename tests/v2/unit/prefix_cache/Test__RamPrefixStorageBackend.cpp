/**
 * @file Test__RamPrefixStorageBackend.cpp
 * @brief Device-free coverage for RAM archives, physical claims and section placement.
 *
 * Fragmented restores keep their actual read sections alive while admitting
 * new archives into unrelated free ranges. Failed section BOMs must preserve
 * all existing owners and capacity without a partial placement publication.
 */

#include <gtest/gtest.h>

#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixHostArena.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <random>
#include <string>
#include <thread>

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

    /** @brief Device-free backing exercises the identical production placement authority. */
    std::shared_ptr<PrefixHostArena> makeArena(size_t bytes)
    {
        auto storage = std::make_shared<std::vector<uint8_t>>(bytes, 0xa5);
        return PrefixHostArena::create(
            std::shared_ptr<void>(storage, storage->data()), bytes, {});
    }
} // namespace

/** @test Final aliases retain the backing and its sole canonical physical claim. */
TEST(Test__RamPrefixStorageBackend, ArenaPhysicalLifetimeOutlastsBackendAndAliases)
{
    constexpr size_t capacity = 256u;
    auto authority = makePrefixAuthority(capacity);
    auto reservation = authority->reserveNewAllocations(
        DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier, capacity);
    auto claim = std::make_shared<PhysicalMemorySuballocationLease>(
        reservation.claimAllocation(capacity));
    bool freed = false;
    auto storage = std::shared_ptr<std::vector<uint8_t>>(
        new std::vector<uint8_t>(capacity), [&](auto *pointer) {
            EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier,
                PhysicalMemoryMaterializationKind::NewAllocation), capacity);
            freed = true;
            delete pointer;
        });
    auto arena = PrefixHostArena::create(
        std::shared_ptr<void>(storage, storage->data()), capacity, std::move(claim));
    storage.reset();
    auto range = arena->acquire(capacity);
    ASSERT_NE(range, nullptr);
    auto alias = range;
    range.reset();
    EXPECT_EQ(arena->availableBytes(), 0u);
    arena.reset();
    EXPECT_FALSE(freed);
    alias.reset();
    EXPECT_TRUE(freed);
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test Nonadjacent holes and retained aliases cannot satisfy a larger archive. */
TEST(Test__RamPrefixStorageBackend, ArenaFragmentationCoalescesOnlyReleasedNeighbors)
{
    auto arena = makeArena(256u);
    std::array<std::shared_ptr<void>, 4> ranges;
    for (auto &range : ranges)
    {
        range = arena->acquire(64u);
        ASSERT_NE(range, nullptr);
    }
    auto first_alias = ranges[0];
    void *const middle = ranges[1].get();
    ranges[0].reset();
    ranges[2].reset();
    EXPECT_EQ(arena->availableBytes(), 64u);
    EXPECT_EQ(arena->acquire(128u), nullptr);
    ranges[1].reset();
    auto joined = arena->acquire(128u);
    ASSERT_NE(joined, nullptr);
    EXPECT_EQ(joined.get(), middle);
    EXPECT_EQ(arena->availableBytes(), 0u);
    first_alias.reset();
    joined.reset();
    ranges[3].reset();
    EXPECT_EQ(arena->availableBytes(), 256u);
    EXPECT_NE(arena->acquire(256u), nullptr);
}

/** @test Pending row readers permit fragmented section reuse without overwriting their bytes. */
TEST(Test__RamPrefixStorageBackend, ArenaIndependentSectionsReuseFragmentedRestoreHoles)
{
    constexpr size_t capacity = 40u;
    constexpr std::array<size_t, 6> sections{4u, 8u, 0u, 0u, 0u, 0u};
    auto arena = makeArena(capacity);
    for (unsigned iteration = 0u; iteration < 20u; ++iteration)
    {
        SCOPED_TRACE(iteration);
        std::array<std::vector<std::shared_ptr<void>>, 3> archives;
        for (auto &archive : archives)
        {
            ASSERT_TRUE(arena->canAcquireSections(sections));
            archive = arena->acquireSections(sections);
            ASSERT_EQ(archive.size(), sections.size());
            std::memset(archive[0].get(), 0xb6, sections[0]);
            std::memset(archive[1].get(), 0x8b, sections[1]);
            for (size_t section = 2u; section < sections.size(); ++section)
                EXPECT_EQ(archive[section], nullptr);
        }
        // Earlier checkpoints supply only KV rows. The terminal checkpoint's
        // recurrent section remains a real reader and cannot be reclaimed.
        archives[0][1].reset();
        archives[1][1].reset();
        EXPECT_EQ(arena->availableStorageBytes(), 20u);
        EXPECT_EQ(arena->availableBytes(), 8u);

        // The aggregate fits but one nine-byte section cannot fit any hole.
        // No early four-byte placement may survive this rejected transaction.
        constexpr std::array<size_t, 2> impossible{4u, 9u};
        EXPECT_FALSE(arena->canAcquireSections(impossible));
        EXPECT_TRUE(arena->acquireSections(impossible).empty());
        EXPECT_EQ(arena->availableStorageBytes(), 20u);
        EXPECT_EQ(arena->availableBytes(), 8u);
        ASSERT_TRUE(arena->canAcquireSections(sections));
        auto next = arena->acquireSections(sections);
        ASSERT_EQ(next.size(), sections.size());
        std::memset(next[0].get(), 0x3c, sections[0]);
        std::memset(next[1].get(), 0x71, sections[1]);
        for (const auto &archive : archives)
        {
            const auto *bytes = static_cast<const uint8_t *>(archive[0].get());
            EXPECT_TRUE(std::all_of(bytes, bytes + sections[0],
                [](uint8_t byte) { return byte == 0xb6; }));
        }
        const auto *terminal = static_cast<const uint8_t *>(archives.back()[1].get());
        EXPECT_TRUE(std::all_of(terminal, terminal + sections[1],
            [](uint8_t byte) { return byte == 0x8b; }));
        EXPECT_EQ(arena->availableStorageBytes(), 8u);
        archives = {};
        next.clear();
        EXPECT_EQ(arena->availableStorageBytes(), capacity);
        EXPECT_EQ(arena->availableBytes(), capacity);
    }
}

/** @test Empty and overflowing section geometries never manufacture physical ownership. */
TEST(Test__RamPrefixStorageBackend, ArenaSectionGeometryRejectsOverflowWithoutPartialLeases)
{
    auto arena = makeArena(13u);
    constexpr std::array<size_t, 6> empty{};
    ASSERT_TRUE(arena->canAcquireSections(empty));
    const auto empty_owners = arena->acquireSections(empty);
    ASSERT_EQ(empty_owners.size(), empty.size());
    for (const auto &owner : empty_owners) EXPECT_EQ(owner, nullptr);
    constexpr std::array<size_t, 3> overflow{1u, std::numeric_limits<size_t>::max(), 1u};
    EXPECT_FALSE(arena->canAcquireSections(overflow));
    EXPECT_TRUE(arena->acquireSections(overflow).empty());
    EXPECT_EQ(arena->availableStorageBytes(), 13u);
    constexpr std::array<size_t, 3> odd{3u, 0u, 10u};
    auto owners = arena->acquireSections(odd);
    ASSERT_EQ(owners.size(), odd.size());
    EXPECT_EQ(owners[1], nullptr);
    EXPECT_EQ(arena->availableStorageBytes(), 0u);
    owners.clear();
    EXPECT_EQ(arena->availableBytes(), 13u);
}

/** @test Randomized odd geometries retain exact byte capacity and disjoint live ranges. */
TEST(Test__RamPrefixStorageBackend, ArenaRandomizedLargeSmallAliasReuse)
{
    constexpr size_t capacity = 4096u;
    auto arena = makeArena(capacity);
    struct LiveRange { std::shared_ptr<void> owner; size_t bytes; uint8_t value; };
    std::vector<LiveRange> live;
    std::mt19937 random(0x771af);
    for (unsigned iteration = 0u; iteration < 4000u; ++iteration)
    {
        SCOPED_TRACE(iteration);
        for (const auto &range : live)
        {
            const auto *begin = static_cast<const uint8_t *>(range.owner.get());
            ASSERT_TRUE(std::all_of(begin, begin + range.bytes,
                [&](uint8_t value) { return value == range.value; }));
        }
        if (!live.empty() && random() % 3u == 0u)
        {
            live.erase(live.begin() + random() % live.size());
            continue;
        }
        const size_t bytes = 1u + random() % 521u;
        auto owner = arena->acquire(bytes);
        if (!owner)
        {
            if (!live.empty()) live.erase(live.begin() + random() % live.size());
            continue;
        }
        const auto address = reinterpret_cast<uintptr_t>(owner.get());
        for (const auto &range : live)
        {
            const auto other = reinterpret_cast<uintptr_t>(range.owner.get());
            ASSERT_TRUE(address + bytes <= other || other + range.bytes <= address);
        }
        const auto value = static_cast<uint8_t>(iteration);
        std::memset(owner.get(), value, bytes);
        live.push_back({owner, bytes, value});
        if (random() % 4u == 0u) live.push_back({std::move(owner), bytes, value});
    }
    live.clear();
    EXPECT_EQ(arena->availableBytes(), capacity);
    auto whole = arena->acquire(capacity);
    ASSERT_NE(whole, nullptr);
    std::memset(whole.get(), 0x39, capacity);
    whole.reset();
    auto small = arena->acquire(1u);
    ASSERT_NE(small, nullptr);
    EXPECT_EQ(*static_cast<uint8_t *>(small.get()), 0x39)
        << "Placement must not silently clear complete-overwrite payloads";
}

/** @test Independent disk/request owners may retire range aliases concurrently. */
TEST(Test__RamPrefixStorageBackend, ArenaConcurrentLastAliasRetirement)
{
    auto arena = makeArena(4096u);
    constexpr unsigned workers = 8u;
    std::barrier rendezvous(workers);
    std::array<std::jthread, workers> threads;
    for (unsigned worker = 0; worker < workers; ++worker)
        threads[worker] = std::jthread([&, worker] {
            for (unsigned replay = 0; replay < 200u; ++replay)
            {
                auto range = arena->acquire(17u + worker * 7u);
                EXPECT_NE(range, nullptr);
                auto alias = range;
                if (range) std::memset(range.get(), worker + 1u, 17u + worker * 7u);
                range.reset();
                rendezvous.arrive_and_wait();
                if (alias)
                {
                    const auto *begin = static_cast<const uint8_t *>(alias.get());
                    EXPECT_TRUE(std::all_of(begin, begin + 17u + worker * 7u,
                        [&](uint8_t value) { return value == worker + 1u; }));
                }
                alias.reset();
                rendezvous.arrive_and_wait();
            }
        });
    for (auto &thread : threads) thread.join();
    EXPECT_EQ(arena->availableBytes(), 4096u);
}

/** @test A GPU archive can never lazily materialize through an unaccounted test constructor. */
TEST(Test__RamPrefixStorageBackend, ArenaRequiresGPUSetupAdmission)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        RamPrefixStorageBackend backend(device, 64u);
        EXPECT_EQ(backend.availableAllocationBytes(), 0u);
        std::string error;
        EXPECT_FALSE(backend.allocateWithDiagnostics(keyFor(0), makeLayout(), &error).valid());
        EXPECT_NE(error.find("admitted persistent arena"), std::string::npos);
    }
    EXPECT_THROW(PrefixHostArena::create({}, 32u, {}), std::invalid_argument);
    auto arena = makeArena(32u);
    EXPECT_EQ(arena->acquire(0u), nullptr);
    EXPECT_EQ(arena->acquire(std::numeric_limits<size_t>::max()), nullptr);
    auto odd = arena->acquire(13u);
    ASSERT_NE(odd, nullptr);
    EXPECT_EQ(arena->availableBytes(), 19u);
    EXPECT_NE(arena->acquire(19u), nullptr);
}

/** @test An evicted handle cannot release or mutate a newer incarnation of the same key. */
TEST(Test__RamPrefixStorageBackend, ArenaRejectsStaleArchiveIncarnations)
{
    RamPrefixStorageBackend backend(128u);
    auto old = backend.allocate(keyFor(0), makeLayout());
    ASSERT_TRUE(old.valid());
    ASSERT_TRUE(backend.release(old));
    auto current = backend.allocate(keyFor(0), makeLayout());
    ASSERT_TRUE(current.valid());
    EXPECT_FALSE(backend.attachModelRuntimeState(&old,
        std::make_shared<std::vector<uint8_t>>(3u, 0x44)));
    EXPECT_FALSE(backend.release(old));
    EXPECT_EQ(backend.usedBytes(), 32u);
    EXPECT_TRUE(backend.release(current));
}

/** @test Malformed section arithmetic cannot turn a tiny arena lease into an oversized write. */
TEST(Test__RamPrefixStorageBackend, ArenaRejectsPayloadExtentOverflow)
{
    RamPrefixStorageBackend backend(128u);
    auto layout = makeLayout(std::numeric_limits<size_t>::max(), 33u);
    EXPECT_FALSE(backend.allocate(keyFor(0), layout).valid());
    layout = makeLayout(size_t{1} << (sizeof(size_t) * 8u - 1u), 16u);
    layout.fa_layers = 2;
    EXPECT_FALSE(backend.allocate(keyFor(1), layout).valid());
    layout = makeLayout();
    layout.includes_hybrid_state = true;
    layout.hybrid_state_bytes = std::numeric_limits<size_t>::max() - 15u;
    EXPECT_FALSE(backend.allocate(keyFor(2), layout).valid());
    EXPECT_EQ(backend.usedBytes(), 0u);
}

/** @test Missing publication is fatal before backing can be returned to its allocator. */
TEST(Test__RamPrefixStorageBackend, ArenaUnpublishedProducerFailsClosed)
{
    EXPECT_DEATH({
        auto arena = makeArena(32u);
        auto readiness = std::make_shared<PrefixPayloadReadiness>();
        auto event = std::make_shared<uint8_t>(0u);
        (void)readiness->prepare(event, DeviceId::cuda(0), reinterpret_cast<void *>(uintptr_t{1}));
        auto range = arena->acquire(32u, readiness);
        range.reset();
        arena.reset();
    }, "unsafe final backing retirement");
}

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

/** @brief Passive occupancy follows exact backend keys, including runtime sections and rejected mutations. */
TEST(Test__RamPrefixStorageBackend, TelemetryOccupancyTracksOnlyCommittedKeys)
{
    RamPrefixStorageBackend backend(100);
    const auto telemetry = backend.telemetry();
    EXPECT_EQ(telemetry->snapshot().used_bytes, 0u);
    auto handle = backend.allocate(keyFor(0), makeLayout());
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(telemetry->snapshot().capacity_bytes, 100u);
    EXPECT_EQ(telemetry->snapshot().used_bytes, 32u);
    EXPECT_EQ(telemetry->snapshot().entries, 1u);
    ASSERT_TRUE(backend.attachModelRuntimeState(&handle,
        std::make_shared<std::vector<uint8_t>>(17, 0x42)));
    const auto with_runtime = telemetry->snapshot();
    EXPECT_EQ(with_runtime.used_bytes, 49u);
    EXPECT_FALSE(backend.allocate(keyFor(0), makeLayout()).valid());
    EXPECT_FALSE(backend.allocate(keyFor(1), makeLayout(100, 100)).valid());
    EXPECT_EQ(telemetry->snapshot().revision, with_runtime.revision);
    EXPECT_TRUE(backend.release(handle));
    EXPECT_EQ(telemetry->snapshot().used_bytes, 0u);
    EXPECT_EQ(telemetry->snapshot().entries, 0u);
    const auto released = telemetry->snapshot().revision;
    EXPECT_FALSE(backend.release(handle));
    EXPECT_EQ(telemetry->snapshot().revision, released);
    // Observations describe indexed payloads, not external aliases. Polling
    // cannot reclaim or inspect this still-readable consumer's bytes.
    ASSERT_NE(handle.kv_storage, nullptr);
    EXPECT_EQ(handle.model_runtime_state_storage->data()[0], 0x42);
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

/** @test Recurrent checkpoints are real bounded payloads, not zero-byte KV placeholders. */
TEST(Test__RamPrefixStorageBackend, RecurrentOnlyLayoutOwnsNoAttentionBytes)
{
    auto layout = makeLayout(0, 0);
    layout.fa_layers = 0;
    layout.gdn_layers = 1;
    layout.hybrid_host_state_bytes = layout.hybrid_state_bytes = 32;
    layout.includes_hybrid_state = true;
    ASSERT_TRUE(layout.hasRestorableMainState());
    EXPECT_EQ(layout.organization(), PrefixPayloadOrganization::RecurrentCheckpoint);
    RamPrefixStorageBackend backend(32);
    auto handle = backend.allocate(keyFor(0), layout);
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(handle.kvKData(), nullptr);
    EXPECT_EQ(handle.kvVData(), nullptr);
    ASSERT_NE(handle.hybrid_payload, nullptr);
    EXPECT_EQ(backend.usedBytes(), 32u);
    EXPECT_FALSE(backend.allocate(keyFor(1), layout).valid());

    layout.includes_hybrid_state = false;
    EXPECT_FALSE(layout.hasRestorableMainState());
    layout.includes_hybrid_state = true;
    layout.includes_mtp_state = true;
    layout.mtp_kv_bytes = 16;
    EXPECT_EQ(layout.organization(), PrefixPayloadOrganization::AttentionBlocks)
        << "A terminal participant still needs its shifted attention block chain";
    layout.hybrid_state_bytes = 31;
    EXPECT_FALSE(layout.hasRestorableMainState()) << "Reject incomplete recurrent geometry";
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

/** @test Logical eviction cannot reissue bytes still leased by a request. */
TEST(Test__RamPrefixStorageBackend,
     RequestAliasKeepsPhysicalArchiveCapacityBusy)
{
    auto authority = makePrefixAuthority(64u);
    auto backend = RamPrefixStorageBackend::create(
        DeviceId::cpu(), 64u, authority);
    ASSERT_NE(backend, nullptr);

    auto first = backend->allocate(keyFor(0), makeLayout());
    auto second = backend->allocate(keyFor(1), makeLayout());
    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    PrefixBlockHandle request_alias = first;
    ASSERT_TRUE(backend->release(first));
    first = {};

    EXPECT_EQ(backend->usedBytes(), 32u);
    EXPECT_FALSE(backend->canStore(32u));
    std::string error;
    EXPECT_FALSE(backend->allocateWithDiagnostics(
        keyFor(2), makeLayout(), &error).valid());
    EXPECT_NE(error.find("physical RAM prefix tier capacity busy"),
              std::string::npos);
    EXPECT_EQ(authority->claimedBytes(
                  DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier,
                  PhysicalMemoryMaterializationKind::NewAllocation),
              64u);

    request_alias = {};
    EXPECT_TRUE(backend->canStore(32u));
    EXPECT_TRUE(backend->allocate(keyFor(2), makeLayout()).valid());
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
    bool complete = false;
    EXPECT_TRUE(readiness.queryComplete(&complete));
    EXPECT_TRUE(complete);

    ASSERT_TRUE(readiness.prepare(
        std::static_pointer_cast<void>(event_owner),
        DeviceId::cuda(0),
        stream));
    EXPECT_TRUE(readiness.prepared());
    EXPECT_FALSE(readiness.published());
    EXPECT_EQ(readiness.event(), event_owner.get());
    EXPECT_EQ(readiness.producerDevice(), DeviceId::cuda(0));
    EXPECT_EQ(readiness.producerStream(), stream);
    EXPECT_FALSE(readiness.queryComplete(&complete));
    EXPECT_FALSE(complete) << "Prepared-but-unpublished is not a reusable range";

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
