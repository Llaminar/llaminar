/**
 * @file Test__HostImmutableRowProvider.cpp
 * @brief Native-file/PMA tests for the closed whole-tier and adaptive host owners.
 *
 * Fixtures construct actual admitted source owners and inspect original native
 * bytes, exact successful observation alternatives and last-owner retirement.
 * No loader/model/backend startup, device execution, fake provider or ledger.
 */
#include "loaders/HostImmutableRowProvider.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <type_traits>

namespace llaminar2::test
{
namespace
{
using Provider = HostImmutableRowProvider;
static_assert(std::is_same_v<decltype(std::declval<const HostImmutableRowOwnerObservation &>().authority()),
    const std::shared_ptr<const PhysicalMemoryAuthority> &>);
using Storage = ImmutableRowStorage;
using Cache = CpuAdaptiveImmutableRowCache;
using Geometry = CpuAdaptiveImmutableRowCacheGeometry;
const PhysicalMemoryAllocatorIdentity hostId{0, DeviceId::cpu()};
/** @return Canonical observed host resource; no free-capacity subtraction. */
PhysicalMemoryResource hostResource() { return {0, DeviceId::cpu(), 1ull << 34, 1ull << 34}; }

/** @brief Small original native file, including cross-window rows and nonzero source offset. */
struct NativeFile
{
    int fd = -1;
    std::string path;
    std::vector<std::byte> original;
    ImmutableRowTableSource source;
    /** @brief Write distinguishable native bits verbatim, without a quantizer or payload hash. */
    NativeFile()
    {
        char name[] = "/tmp/llaminar-host-provider-XXXXXX";
        fd = ::mkstemp(name);
        if (fd < 0) throw std::runtime_error("Cannot create native provider fixture");
        path = name;
        source = {.path = path, .tensor_name = "fixture.ple", .file_offset = 37,
            .rows = 128, .row_elements = 160, .format = GGUFTensorType::IQ4_NL};
        original.resize(source.file_offset + source.tensorBytes());
        for (std::size_t i = 0; i < original.size(); ++i) original[i] = std::byte((i * 13 + i / 31) % 251);
        if (::write(fd, original.data(), original.size()) != static_cast<ssize_t>(original.size()))
        {
            ::close(fd); fd = -1; ::unlink(path.c_str());
            throw std::runtime_error("Cannot write native provider fixture");
        }
    }
    /** @brief Source metadata remains present until all provider handles in the test retire. */
    ~NativeFile() { if (fd >= 0) ::close(fd); ::unlink(path.c_str()); }
};
/** @return Existing adaptive host-domain proof, including legal hosted GPU consumers. */
Geometry geometry(const NativeFile &file, std::size_t slots = 2)
{
    const std::array tiers{Storage::RAM, Storage::MemoryMapped};
    const auto layout = ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(file.source.format, file.source.rows, file.source.rowBytes()),
        ImmutableRowTableTiers::of(tiers), ImmutableRowTableResidency::Adaptive,
        {{{hostId, Storage::RAM}, 0, slots, {}}}, {{0, DeviceId::cuda(0)}}, {},
        {{hostId, ImmutableRowTableStoragePolicy::forStorage(Storage::MemoryMapped, 4096, 2)}});
    return Geometry::fromLayout(layout, hostId, 8);
}
/** @return Sole actual PMA for the supplied complete source-owned plan. */
std::shared_ptr<PhysicalMemoryAuthority> memory(const PhysicalMemoryPlan &plan)
{
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan), 0);
}
/** @return Whole-source owner admitted by its exact existing storage factory. */
std::unique_ptr<ImmutableRowTable> whole(const NativeFile &file, Storage storage,
    std::shared_ptr<PhysicalMemoryAuthority> &authority)
{
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(storage, 4096, 2);
    PhysicalMemoryPlanBuilder plan;
    plan.add(hostResource(), PhysicalMemoryOwner::ImmutableTableStorage, ImmutableRowTable::physicalBytes(file.source, policy));
    authority = memory(plan.build());
    return std::make_unique<ImmutableRowTable>(file.source, policy, DeviceId::cpu(), authority,
        ImmutableRowTableHostPlacement::taskPolicy());
}
/** @return One actual arm, without another table or reconstructed allocation count. */
std::shared_ptr<Provider> provider(const NativeFile &file, HostImmutableRowMode mode,
    std::shared_ptr<PhysicalMemoryAuthority> &authority)
{
    if (mode != HostImmutableRowMode::AdaptiveRAMSSD)
        return Provider::fromWholeTable(whole(file,
            mode == HostImmutableRowMode::WholeRAM ? Storage::RAM : Storage::MemoryMapped, authority));
    const auto shape = geometry(file); authority = memory(shape.bom(hostResource()));
    return Provider::fromAdaptiveCache(Cache::create(file.source, shape, authority));
}
/** @brief Prove exact source bytes and duplicate order from the original file. */
void expectRows(const NativeFile &file, std::span<const std::int64_t> ids, std::span<const std::byte> out)
{
    ASSERT_EQ(out.size(), ids.size() * file.source.rowBytes());
    for (std::size_t i = 0; i < ids.size(); ++i)
        EXPECT_EQ(std::memcmp(out.data() + i * file.source.rowBytes(),
            file.original.data() + file.source.file_offset + ids[i] * file.source.rowBytes(), file.source.rowBytes()), 0);
}
/** @return Actual currently claimed bytes under one owner. */
std::size_t claimed(const std::shared_ptr<PhysicalMemoryAuthority> &authority, PhysicalMemoryOwner owner)
{
    return authority->claimedBytes(DeviceId::cpu(), owner, PhysicalMemoryMaterializationKind::NewAllocation);
}
constexpr std::array modes{HostImmutableRowMode::WholeRAM,
    HostImmutableRowMode::WholeMemoryMapped, HostImmutableRowMode::AdaptiveRAMSSD};
}

/** @test Whole-tier observations retain actual RAM/mmap mode and exact per-call native work. */
TEST(HostImmutableRowProvider, WholeArmsRetainSourceModeBytesAndOriginalPerCallWork)
{
    NativeFile file;
    for (auto storage : {Storage::RAM, Storage::MemoryMapped})
    {
        std::shared_ptr<PhysicalMemoryAuthority> authority;
        auto owner = whole(file, storage, authority);
        const auto identity = owner->identity(); const auto physical = owner->physicalBytes();
        auto service = Provider::fromWholeTable(std::move(owner));
        EXPECT_FALSE(owner); EXPECT_TRUE(authority->rankComplete());
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), physical);
        EXPECT_EQ(service->source(), file.source); EXPECT_EQ(service->identity(), identity);
        EXPECT_EQ(service->mode(), storage == Storage::RAM ? HostImmutableRowMode::WholeRAM : HostImmutableRowMode::WholeMemoryMapped);
        const std::array<std::int64_t, 4> ids{45, 0, 45, 3}; std::array<std::byte, 360> out{};
        const auto observation = service->gather(ids, out); expectRows(file, ids, out);
        ASSERT_TRUE(std::holds_alternative<WholeImmutableRowGatherObservation>(observation));
        const auto &work = std::get<WholeImmutableRowGatherObservation>(observation);
        EXPECT_EQ(work.storage, storage); EXPECT_EQ(work.rows, 4u); EXPECT_EQ(work.bytes, out.size());
        if (storage == Storage::RAM) EXPECT_EQ(work.windows.windowHits + work.windows.windowMisses, 0u);
        else EXPECT_GT(work.windows.windowHits + work.windows.windowMisses, 0u);
        service->authenticateSource();
        const auto empty = std::get<WholeImmutableRowGatherObservation>(service->gather({}, {}));
        EXPECT_EQ(empty.storage, storage); EXPECT_EQ(empty.rows, 0u); EXPECT_EQ(empty.bytes, 0u);
        service.reset(); EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), 0u);
    }
}

/** @test Adoption/readiness are not gathers; adaptive observations and cache order survive unmodified. */
TEST(HostImmutableRowProvider, AdaptiveArmPreservesObservationAndReadinessHasNoPolicyWork)
{
    NativeFile file; const auto shape = geometry(file); auto authority = memory(shape.bom(hostResource()));
    auto service = Provider::fromAdaptiveCache(Cache::create(file.source, shape, authority));
    EXPECT_EQ(service->mode(), HostImmutableRowMode::AdaptiveRAMSSD); EXPECT_TRUE(authority->rankComplete());
    service->authenticateSource(); service->authenticateSource();
    const std::array<std::int64_t, 3> first{5, 5, 2}; std::array<std::byte, 270> out{};
    const auto observation = service->gather(first, out); expectRows(file, first, out);
    ASSERT_TRUE(std::holds_alternative<CpuImmutableRowGatherObservation>(observation));
    const auto &cold = std::get<CpuImmutableRowGatherObservation>(observation);
    EXPECT_EQ(cold.policy_ordinal, 0u); EXPECT_EQ(cold.rows, 3u); EXPECT_EQ(cold.bytes, 270u);
    EXPECT_EQ(cold.source_rows, 2u); EXPECT_EQ(cold.duplicate_rows, 1u); EXPECT_EQ(cold.ram_rows, 0u);
    service->authenticateSource();
    const std::array<std::int64_t, 2> second{5, 2};
    const auto warm = std::get<CpuImmutableRowGatherObservation>(service->gather(second, std::span(out).first(180)));
    expectRows(file, second, std::span(out).first(180));
    EXPECT_EQ(warm.policy_ordinal, 1u); EXPECT_EQ(warm.ram_rows, 2u); EXPECT_EQ(warm.source_rows, 0u);
    EXPECT_EQ(service->mode(), HostImmutableRowMode::AdaptiveRAMSSD);
}

/** @test Zero optional capacity remains an adaptive owner; explicit SSD responses do not relabel it. */
TEST(HostImmutableRowProvider, ZeroSlotArmIsStillAdaptiveAndCompletesExactDuplicates)
{
    NativeFile file; const auto shape = geometry(file, 0); auto authority = memory(shape.bom(hostResource()));
    auto service = Provider::fromAdaptiveCache(Cache::create(file.source, shape, authority));
    const std::array<std::int64_t, 3> ids{2, 2, 45}; std::array<std::byte, 270> out{};
    const auto work = std::get<CpuImmutableRowGatherObservation>(service->gather(ids, out)); expectRows(file, ids, out);
    EXPECT_EQ(service->mode(), HostImmutableRowMode::AdaptiveRAMSSD); EXPECT_EQ(shape.storageBytes(), 0u);
    EXPECT_EQ(work.source_rows, 2u); EXPECT_EQ(work.duplicate_rows, 1u); EXPECT_EQ(work.admissions, 0u);
}

/** @test Invalid frames and provider/source failure return no successful alternative or native response. */
TEST(HostImmutableRowProvider, FailuresPreserveOriginalOwnerValidationAndRejectNullOrStaleAdoption)
{
    EXPECT_THROW(Provider::fromWholeTable(nullptr), std::invalid_argument);
    EXPECT_THROW(Provider::fromAdaptiveCache(nullptr), std::invalid_argument);
    for (auto mode : modes)
    {
        NativeFile file; std::shared_ptr<PhysicalMemoryAuthority> authority; auto service = provider(file, mode, authority);
        const std::array<std::int64_t, 2> bad{0, 128}, good{0, 45}; std::array<std::byte, 180> out;
        out.fill(std::byte{0x63}); const auto before = out;
        EXPECT_THROW(service->gather(bad, out), std::out_of_range); EXPECT_EQ(out, before);
        const auto observation = service->gather(good, out); expectRows(file, good, out);
        if (mode == HostImmutableRowMode::AdaptiveRAMSSD)
        {
            EXPECT_EQ(std::get<CpuImmutableRowGatherObservation>(observation).policy_ordinal, 0u);
        }
        ASSERT_EQ(::ftruncate(file.fd, file.original.size() + 1), 0);
        EXPECT_THROW(service->authenticateSource(), std::runtime_error);
        out.fill(std::byte{0x63}); EXPECT_THROW(service->gather(good, out), std::runtime_error); EXPECT_EQ(out, before);
        service.reset(); EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), 0u);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), 0u);
    }
    for (auto mode : modes)
    {
        NativeFile file; std::shared_ptr<PhysicalMemoryAuthority> authority;
        if (mode == HostImmutableRowMode::AdaptiveRAMSSD)
        {
            const auto shape = geometry(file); authority = memory(shape.bom(hostResource()));
            auto owner = Cache::create(file.source, shape, authority);
            ASSERT_EQ(::ftruncate(file.fd, file.original.size() + 1), 0);
            EXPECT_THROW(Provider::fromAdaptiveCache(std::move(owner)), std::runtime_error); EXPECT_FALSE(owner);
        }
        else
        {
            auto owner = whole(file, mode == HostImmutableRowMode::WholeRAM ? Storage::RAM : Storage::MemoryMapped, authority);
            ASSERT_EQ(::ftruncate(file.fd, file.original.size() + 1), 0);
            EXPECT_THROW(Provider::fromWholeTable(std::move(owner)), std::runtime_error); EXPECT_FALSE(owner);
        }
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), 0u);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), 0u);
    }
}

/** @test Service/stage shared handles retain the exact original PMA claims through the terminal owner. */
TEST(HostImmutableRowProvider, LastSharedProviderRetiresTheSolePhysicalOwners)
{
    for (auto mode : modes)
    {
        NativeFile file; std::shared_ptr<PhysicalMemoryAuthority> authority; auto service = provider(file, mode, authority);
        const auto original_storage = claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage);
        const auto original_execution = claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging);
        auto terminal_owner = service; service.reset();
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), original_storage);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), original_execution);
        const std::array<std::int64_t, 1> ids{45}; std::array<std::byte, 90> out{};
        terminal_owner->gather(ids, out); expectRows(file, ids, out);
        terminal_owner.reset();
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), 0u);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), 0u);
    }
}
/** @test A provider reports its actual leases, excluding unrelated claims on the same PMA owners. */
TEST(HostImmutableRowProvider, RetainedOwnerObservationUsesOnlyActualWholeOwnerLeases)
{
    NativeFile file;
    for (auto storage : {Storage::RAM, Storage::MemoryMapped})
    {
        const auto policy = ImmutableRowTableStoragePolicy::forStorage(storage, 4096, 2);
        const auto expected = ImmutableRowTable::physicalBytes(file.source, policy);
        PhysicalMemoryPlanBuilder plan;
        plan.add(hostResource(), PhysicalMemoryOwner::ImmutableTableStorage, expected + 4096);
        plan.add(hostResource(), PhysicalMemoryOwner::ImmutableLookupStaging, 8192);
        auto authority = memory(plan.build());
        auto foreign_storage = authority->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableTableStorage, 4096);
        auto foreign_staging = authority->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableLookupStaging, 8192);
        // Exact locality is an immutable policy observation. A file mapping
        // makes no physical page-cache NUMA promise and does not bind a node.
        const auto locality = storage == Storage::MemoryMapped
            ? ImmutableRowTableHostPlacement::onNode(7) : ImmutableRowTableHostPlacement::taskPolicy();
        auto table = std::make_unique<ImmutableRowTable>(file.source, policy, DeviceId::cpu(), authority, locality);
        const auto own_bytes = table->physicalBytes();
        const auto direct = table->retainedOwnerObservation();
        EXPECT_EQ(direct.authority().get(), authority.get()); EXPECT_EQ(direct.allocator(), hostId);
        EXPECT_EQ(direct.hostPlacement(), locality); EXPECT_EQ(direct.tableStorageBytes(), own_bytes);
        EXPECT_EQ(direct.lookupStagingBytes(), 0u);
        auto service = Provider::fromWholeTable(std::move(table));
        const auto observed = service->retainedOwnerObservation();
        EXPECT_EQ(observed.authority().get(), authority.get()); EXPECT_EQ(observed.allocator(), hostId);
        EXPECT_EQ(observed.hostPlacement(), locality); EXPECT_EQ(observed.tableStorageBytes(), own_bytes);
        EXPECT_EQ(observed.lookupStagingBytes(), 0u);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), own_bytes + foreign_storage.bytes());
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), foreign_staging.bytes());
        service.reset();
        // A retained observation keeps only const authority identity, not the
        // table lease or an extra right to reuse storage after owner release.
        EXPECT_EQ(observed.authority().get(), authority.get());
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), foreign_storage.bytes());
    }
}

/** @test Zero/nonzero cache arms expose both real owner lines without gather/policy side effects. */
TEST(HostImmutableRowProvider, RetainedOwnerObservationPreservesAdaptiveSplitAndSharedLifetime)
{
    NativeFile file;
    for (const std::size_t slots : {0u, 2u})
    {
        const auto shape = geometry(file, slots);
        auto authority = memory(shape.bom(hostResource()));
        auto service = Provider::fromAdaptiveCache(Cache::create(file.source, shape, authority));
        const auto observed = service->retainedOwnerObservation();
        EXPECT_EQ(observed.authority().get(), authority.get()); EXPECT_EQ(observed.allocator(), hostId);
        EXPECT_EQ(observed.hostPlacement(), ImmutableRowTableHostPlacement::taskPolicy());
        EXPECT_EQ(observed.tableStorageBytes(), claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage));
        EXPECT_EQ(observed.lookupStagingBytes(), claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging));
        EXPECT_GT(observed.tableStorageBytes(), 0u); EXPECT_GT(observed.lookupStagingBytes(), 0u);
        service->retainedOwnerObservation();
        const std::array<std::int64_t, 1> ids{5}; std::array<std::byte, 90> out{};
        EXPECT_EQ(std::get<CpuImmutableRowGatherObservation>(service->gather(ids, out)).policy_ordinal, 0u);
        auto terminal = service; service.reset();
        const auto retained = terminal->retainedOwnerObservation();
        EXPECT_EQ(retained.tableStorageBytes(), observed.tableStorageBytes());
        EXPECT_EQ(retained.lookupStagingBytes(), observed.lookupStagingBytes());
        terminal.reset();
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), 0u);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), 0u);
        EXPECT_NE(observed.authority(), nullptr); // This snapshot is not a backing pin.
    }
}

/** @test A stale source cannot be authenticated for reuse by either provider arm. */
TEST(HostImmutableRowProvider, RetainedOwnerObservationRejectsStaleSourceWithoutPublishingSuccess)
{
    for (auto mode : modes)
    {
        NativeFile file; std::shared_ptr<PhysicalMemoryAuthority> authority;
        auto service = provider(file, mode, authority);
        const auto retained = service->retainedOwnerObservation();
        ASSERT_EQ(::ftruncate(file.fd, file.original.size() + 1), 0);
        EXPECT_THROW(service->retainedOwnerObservation(), std::runtime_error);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), retained.tableStorageBytes());
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), retained.lookupStagingBytes());
        if (mode == HostImmutableRowMode::AdaptiveRAMSSD)
        {
            EXPECT_THROW(service->retainedOwnerObservation(), std::runtime_error);
        }
        service.reset();
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableTableStorage), 0u);
        EXPECT_EQ(claimed(authority, PhysicalMemoryOwner::ImmutableLookupStaging), 0u);
    }
}
} // namespace llaminar2::test
