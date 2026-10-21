/**
 * @file Test__ImmutableRowTable.cpp
 * @brief Native row parity, bounded mmap lifecycle and exact physical ownership.
 *
 * Small private files cover all exported codecs without models or accelerators.
 * Every read is checked against original bytes, including duplicate IDs, rows
 * crossing mapping boundaries, eviction and concurrent clients. Mutation and
 * malformed admission fail before a destination can become partially valid.
 */
#include "loaders/ImmutableRowTable.h"
#include "loaders/ModelContext.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "planning/MemoryPlanner.h"
#include "planning/PlanningModelMetadata.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <thread>
#include <tuple>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace llaminar2;

namespace
{
/** @brief Private deterministic source with an intentionally unaligned tensor start. */
class RowFile
{
public:
    /** @brief Populate one small file with bytewise distinguishable native rows. */
    RowFile(std::size_t row_bytes, std::size_t rows)
        : bytes(offset + row_bytes * rows)
    {
        char name[] = "/tmp/llaminar-row-table-XXXXXX";
        fd = ::mkstemp(name);
        if (fd < 0)
            throw std::runtime_error("row table fixture mkstemp failed");
        path = name;
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = std::byte((i * 17 + i / 89) % 251);
        if (::write(fd, bytes.data(), bytes.size()) != static_cast<ssize_t>(bytes.size()))
            throw std::runtime_error("row table fixture write failed");
    }
    /** @brief Retire the fixture after all row-table owners. */
    ~RowFile() { ::close(fd); ::unlink(path.c_str()); }
    static constexpr std::size_t offset = 37;
    int fd = -1;
    std::string path;
    std::vector<std::byte> bytes;
};

/** @brief Single canonical CPU admission, with no parallel test-only balance. */
std::shared_ptr<PhysicalMemoryAuthority> authority(std::size_t bytes, DeviceId host_device = DeviceId::cpu())
{
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank = 0, .device = host_device,
              .total_bytes = 1024u * 1024u * 1024u,
              .admission_available_bytes = 1024u * 1024u * 1024u},
             PhysicalMemoryOwner::ImmutableTableStorage, bytes);
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
}

/** @brief Resolve host test storage through the same canonical candidate planner as serving. */
ImmutableRowTablePlacementSelection hostPlacement(const ImmutableRowTableSource &src,
    const ImmutableRowTableStoragePolicy &policy,
    ImmutableRowTableHostPlacement locality=ImmutableRowTableHostPlacement::taskPolicy())
{
    const PhysicalMemoryResource host{.world_rank=0,.device=DeviceId::cpu(),
        .total_bytes=1u<<30,.admission_available_bytes=1u<<30};
    PhysicalMemoryPlanBuilder higher; higher.add(host,PhysicalMemoryOwner::PrimaryModelWeights,0);
    const std::array consumers{host};
    return ImmutableRowTablePlacementPlanner::select(
        {.storage=policy.storage()==ImmutableRowStorage::RAM ? ImmutableRowStoragePreference::RAM :
            ImmutableRowStoragePreference::MemoryMapped,
         .window_bytes=policy.windowBytes(),.window_count=policy.windowCount()},
        ImmutableRowTablePlacementInput::forConsumers(
            ImmutableRowTableExtent::fromNative(src.format,src.rows,src.rowBytes()),host,consumers,locality),
        higher.build(),{});
}

/** @brief Exact native directory; file labels do not participate in format selection. */
ImmutableRowTableSource source(const RowFile &file, std::uint64_t rows,
                              GGUFTensorType format = GGUFTensorType::IQ4_NL,
                              std::uint64_t elements = 160)
{
    return {.path = file.path, .tensor_name = "fixture.ple.weight",
            .file_offset = RowFile::offset, .rows = rows,
            .row_elements = elements, .format = format};
}

/** @brief Verify original bytes for every selected row in request order. */
void expectRows(const RowFile &file, std::size_t row_bytes,
                std::span<const std::int64_t> ids, std::span<const std::byte> bytes)
{
    for (std::size_t i = 0; i < ids.size(); ++i)
        EXPECT_EQ(std::memcmp(bytes.data() + i * row_bytes,
            file.bytes.data() + RowFile::offset + ids[i] * row_bytes, row_bytes), 0);
}

/** @return All public observations, including metadata authentication and mapping state. */
auto allTableObservations(const ImmutableRowTableStats &s)
{
    return std::tuple{s.storage,s.table_file_bytes,s.physical_budget_bytes,s.mapped_virtual_bytes,
        s.requests,s.rows,s.copied_bytes,s.source_metadata_checks,
        s.window_hits,s.window_misses,s.window_evictions};
}
}

/** @test Allocated overlapping ID/output storage cannot overwrite later valid IDs during rereads. */
TEST(ImmutableRowTable, ContiguousGatherRejectsAliasedIdsBeforeAnyMutation)
{
    constexpr std::size_t RowBytes=16,Rows=4;
    RowFile file(RowBytes,Rows);
    // With the old gather, row0's zeros overwrite later IDs with valid0. Every
    // old read remains in range, producing wrong rows rather than a crash.
    for(std::size_t row=0;row<Rows;++row)
        std::fill_n(file.bytes.begin()+RowFile::offset+row*RowBytes,RowBytes,std::byte(row));
    ASSERT_EQ(::pwrite(file.fd,file.bytes.data(),file.bytes.size(),0),static_cast<ssize_t>(file.bytes.size()));
    const auto src=source(file,Rows,GGUFTensorType::F32,4);
    for(const auto mode:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const auto policy=ImmutableRowTableStoragePolicy::forStorage(mode,4096,2);
        ImmutableRowTable table(src,policy,DeviceId::cpu(),authority(ImmutableRowTable::physicalBytes(src,policy)),
            ImmutableRowTableHostPlacement::taskPolicy());
        for(const std::size_t offset:{0u,8u})
        {
            SCOPED_TRACE(offset);
            std::array<std::int64_t,10> storage{};
            const std::array<std::int64_t,4> ids{0,1,2,3};std::copy(ids.begin(),ids.end(),storage.begin());
            const auto saved=storage;const auto before=table.stats();
            auto output=std::as_writable_bytes(std::span(storage)).subspan(offset,Rows*RowBytes);
            EXPECT_THROW(table.gather({storage.data(),ids.size()},output),std::invalid_argument);
            EXPECT_EQ(storage,saved);
            EXPECT_EQ(allTableObservations(table.stats()),allTableObservations(before));
        }
    }
}

/** @test Disjoint allocations, both adjacency directions and zero live rows keep their exact semantics. */
TEST(ImmutableRowTable, ContiguousGatherAcceptsDisjointAdjacentAndEmptySpans)
{
    constexpr std::size_t RowBytes=16,Rows=4;
    RowFile file(RowBytes,Rows);const auto src=source(file,Rows,GGUFTensorType::F32,4);
    const std::array<std::int64_t,4> ids{3,0,2,1};
    for(const auto mode:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const auto policy=ImmutableRowTableStoragePolicy::forStorage(mode,4096,2);
        ImmutableRowTable table(src,policy,DeviceId::cpu(),authority(ImmutableRowTable::physicalBytes(src,policy)),
            ImmutableRowTableHostPlacement::taskPolicy());
        std::array<std::byte,Rows*RowBytes> disjoint{};table.gather(ids,disjoint);expectRows(file,RowBytes,ids,disjoint);
        std::array<std::int64_t,12> adjacent{};
        std::copy(ids.begin(),ids.end(),adjacent.begin());
        auto after_ids=std::as_writable_bytes(std::span(adjacent)).subspan(ids.size()*sizeof(std::int64_t),Rows*RowBytes);
        table.gather({adjacent.data(),ids.size()},after_ids);expectRows(file,RowBytes,ids,after_ids);
        std::copy(ids.begin(),ids.end(),adjacent.begin()+8);
        auto before_ids=std::as_writable_bytes(std::span(adjacent)).first(Rows*RowBytes);
        table.gather({adjacent.data()+8,ids.size()},before_ids);expectRows(file,RowBytes,ids,before_ids);
        const auto before=table.stats();const auto saved=adjacent;
        const auto empty=table.gather({adjacent.data(),0},std::as_writable_bytes(std::span(adjacent)).first(0));
        EXPECT_EQ(adjacent,saved);EXPECT_EQ(empty.windowHits+empty.windowMisses+empty.windowEvictions,0u);
        EXPECT_EQ(table.stats().requests,before.requests+1);
        EXPECT_EQ(table.stats().source_metadata_checks,before.source_metadata_checks+1);
        EXPECT_EQ(table.stats().rows,before.rows);EXPECT_EQ(table.stats().copied_bytes,before.copied_bytes);
        EXPECT_EQ(table.stats().window_hits,before.window_hits);EXPECT_EQ(table.stats().window_misses,before.window_misses);
        EXPECT_EQ(table.stats().window_evictions,before.window_evictions);
        EXPECT_EQ(table.stats().requests,4u);EXPECT_EQ(table.stats().rows,12u);
        EXPECT_EQ(table.stats().copied_bytes,3*Rows*RowBytes);
        const auto unchanged=table.stats();const auto saved_disjoint=disjoint;
        EXPECT_THROW(table.gather(ids,{}),std::invalid_argument);
        EXPECT_THROW(table.gather({},disjoint),std::invalid_argument);
        EXPECT_EQ(adjacent,saved);
        EXPECT_EQ(disjoint,saved_disjoint);
        EXPECT_EQ(allTableObservations(table.stats()),allTableObservations(unchanged));
    }
}

TEST(ImmutableRowTable, EveryNativeFormatPreservesBytesInBothModes)
{
    constexpr std::array formats{
        GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16,
        GGUFTensorType::Q4_0, GGUFTensorType::Q4_1, GGUFTensorType::Q5_0,
        GGUFTensorType::Q5_1, GGUFTensorType::Q8_0, GGUFTensorType::Q2_K,
        GGUFTensorType::Q3_K, GGUFTensorType::Q4_K, GGUFTensorType::Q5_K,
        GGUFTensorType::Q6_K, GGUFTensorType::Q8_K, GGUFTensorType::IQ1_S,
        GGUFTensorType::IQ1_M, GGUFTensorType::IQ2_XXS, GGUFTensorType::IQ2_XS,
        GGUFTensorType::IQ2_S, GGUFTensorType::IQ3_XXS, GGUFTensorType::IQ3_S,
        GGUFTensorType::IQ4_NL, GGUFTensorType::IQ4_XS};
    for (const auto format : formats)
    {
        GGUFTensorInfo info{};
        info.type = format;
        const auto block = info.isQuantized() ? info.getBlockSize() : 1;
        const auto row_bytes = 256 / block * info.getTypeSize();
        RowFile file(row_bytes, 97);
        for (const auto mode : {ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped})
        {
            SCOPED_TRACE(static_cast<int>(format));
            SCOPED_TRACE(static_cast<int>(mode));
            const auto src = source(file, 97, format, 256);
            const auto policy = ImmutableRowTableStoragePolicy::forStorage(mode, 4096, 2);
            auto memory = authority(ImmutableRowTable::physicalBytes(src, policy));
            ImmutableRowTable table(src, policy, DeviceId::cpu(), memory,ImmutableRowTableHostPlacement::taskPolicy());
            const std::array<std::int64_t, 8> ids{96, 0, 48, 3, 96, 4, 47, 95};
            std::vector<std::byte> output(ids.size() * row_bytes);
            table.gather(ids, output);
            expectRows(file, row_bytes, ids, output);
            EXPECT_EQ(table.stats().copied_bytes, output.size());
            EXPECT_EQ(table.stats().rows, ids.size());
            EXPECT_LE(table.stats().mapped_virtual_bytes, 8192u);
        }
    }
}

TEST(ImmutableRowTable, MmapCrossWindowRowsAndLruEvictionStayBounded)
{
    RowFile file(90, 500);
    const auto src = source(file, 500);
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4096, 2);
    const auto planned = ImmutableRowTable::physicalBytes(src, policy);
    auto memory = authority(planned);
    {
        ImmutableRowTable table(src, policy, DeviceId::cpu(), memory,ImmutableRowTableHostPlacement::taskPolicy());
        EXPECT_EQ(table.stats().mapped_virtual_bytes, 0u);
        const std::array<std::int64_t, 8> ids{45, 0, 45, 200, 499, 45, 1, 200};
        std::vector<std::byte> output(ids.size() * 90);
        for (int replay = 0; replay < 20; ++replay)
        {
            table.gather(ids, output);
            expectRows(file, 90, ids, output);
            EXPECT_LE(table.stats().mapped_virtual_bytes, 8192u);
        }
        EXPECT_GT(table.stats().window_hits, 0u);
        EXPECT_GT(table.stats().window_evictions, 0u);
        EXPECT_EQ(table.stats().rows, ids.size() * 20);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation), planned);
    }
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

TEST(ImmutableRowTable, InvalidIdsAndExtentsCannotPartiallyPublish)
{
    RowFile file(90, 4);
    const auto src = source(file, 4);
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4096, 1);
    ImmutableRowTable table(src, policy, DeviceId::cpu(), authority(ImmutableRowTable::physicalBytes(src, policy)),ImmutableRowTableHostPlacement::taskPolicy());
    std::vector<std::byte> output(180, std::byte{0x6a});
    const auto before = output;
    for (const auto invalid : {-1ll, 4ll, std::numeric_limits<long long>::max()})
    {
        const std::array<std::int64_t, 2> ids{0, invalid};
        EXPECT_THROW(table.gather(ids, output), std::out_of_range);
        EXPECT_EQ(output, before);
    }
    const std::array<std::int64_t, 1> valid{0};
    EXPECT_THROW(table.gather(valid, output), std::invalid_argument);
    EXPECT_EQ(table.stats().requests, 0u);
}

TEST(ImmutableRowTable, ChangedTruncatedAndReplacedFilesFailBeforeReading)
{
    for (const auto mode : {ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped})
    {
        for (int mutation = 0; mutation < 3; ++mutation)
        {
            RowFile file(90, 20);
            const auto src = source(file, 20);
            const auto policy = ImmutableRowTableStoragePolicy::forStorage(mode, 4096, 1);
            ImmutableRowTable table(src, policy, DeviceId::cpu(), authority(ImmutableRowTable::physicalBytes(src, policy)),ImmutableRowTableHostPlacement::taskPolicy());
            const std::array<std::int64_t, 1> ids{19};
            std::array<std::byte, 90> output{};
            table.gather(ids, output);
            if (mutation == 0)
            {
                const timespec times[2]{{1, 0}, {1, 0}};
                ASSERT_EQ(::futimens(file.fd, times), 0);
            }
            else if (mutation == 1)
                ASSERT_EQ(::ftruncate(file.fd, 3), 0);
            else
            {
                RowFile replacement(90, 20);
                ASSERT_EQ(::rename(replacement.path.c_str(), file.path.c_str()), 0);
            }
            output.fill(std::byte{0x33});
            const auto before = output;
            EXPECT_THROW(table.gather(ids, output), std::runtime_error);
            EXPECT_EQ(output, before);
        }
    }
}

TEST(ImmutableRowTable, AdmissionGeometryAndFileExtentAreAuthenticated)
{
    RowFile file(90, 20);
    auto src = source(file, 20);
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4096, 1);
    const auto planned = ImmutableRowTable::physicalBytes(src, policy);
    EXPECT_THROW(ImmutableRowTable(src, policy, DeviceId::cpu(), nullptr,ImmutableRowTableHostPlacement::taskPolicy()), std::invalid_argument);
    EXPECT_THROW(ImmutableRowTable(src, policy, DeviceId::cpu(), authority(planned - 1),ImmutableRowTableHostPlacement::taskPolicy()), std::exception);
    auto memory = authority(planned);
    src.rows = 21;
    EXPECT_THROW(ImmutableRowTable(src, policy, DeviceId::cpu(), memory,ImmutableRowTableHostPlacement::taskPolicy()), std::invalid_argument);
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    src = source(file, 20);
    src.row_elements = 159;
    EXPECT_THROW(src.rowBytes(), std::invalid_argument);
    src.row_elements = std::numeric_limits<std::uint64_t>::max() - 31;
    EXPECT_THROW(src.tensorBytes(), std::invalid_argument);
    src = source(file, 20);
    EXPECT_THROW(ImmutableRowTable::physicalBytes(src, ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4097, 1)),
        std::invalid_argument);
    EXPECT_THROW(ImmutableRowTable::physicalBytes(src, ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4096, 0)),
        std::invalid_argument);
}

TEST(ImmutableRowTable, ConcurrentClientsCannotEvictAnActiveCopy)
{
    RowFile file(90, 500);
    const auto src = source(file, 500);
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4096, 1);
    ImmutableRowTable table(src, policy, DeviceId::cpu(), authority(ImmutableRowTable::physicalBytes(src, policy)),ImmutableRowTableHostPlacement::taskPolicy());
    std::vector<std::thread> clients;
    for (int client = 0; client < 4; ++client)
        clients.emplace_back([&, client]
        {
            std::vector<std::byte> output(270);
            for (int turn = 0; turn < 20; ++turn)
            {
                const std::array<std::int64_t, 3> ids{client, 499 - turn, 45};
                table.gather(ids, output);
                expectRows(file, 90, ids, output);
            }
        });
    for (auto &client : clients)
        client.join();
    EXPECT_EQ(table.stats().requests, 80u);
    EXPECT_EQ(table.stats().rows, 240u);
    EXPECT_LE(table.stats().mapped_virtual_bytes, 4096u);
}

/** @test The model/rank facade shares one admitted table and rejects implicit storage changes. */
TEST(ImmutableRowTable, ModelContextOwnsOneTypedTableAcrossAllLocalConsumers)
{
    RowFile file(90, 3);
    // Directory-only injection keeps this ownership test device-free; source
    // bytes and exact metadata are still consumed by production table code.
    auto context = ModelContext::createForTesting(file.path, nullptr, 1, false);
    auto &directory = const_cast<GGUFModel &>(context->model());
    directory.split_paths = {file.path};
    directory.data_offset = RowFile::offset;
    directory.tensors.push_back({.name = "per_layer_token_embd.weight",
        .dimensions = {3, 160}, .type = GGUFTensorType::IQ4_NL,
        .offset = 0, .size_bytes = 270, .split_idx = 0});
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4096, 2);
    const auto selected=hostPlacement(source(file,3),policy);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(selected.physical_plan),0);
    const TransferEngine transfers;
    EXPECT_FALSE(context->hostImmutableRowProvider());
    context->prepareImmutableRowTable(*selected.placement,memory,{},transfers);
    const auto first=context->hostImmutableRowProvider(); ASSERT_TRUE(first);
    context->prepareImmutableRowTable(*selected.placement,memory,{},transfers);
    EXPECT_EQ(context->hostImmutableRowProvider().get(),first.get());
    EXPECT_EQ(context->loader()->ggufDirectory(),&context->model());
    const std::array<std::int64_t,3> ids{2,0,2};
    std::array<std::byte,270> output;
    first->gather(ids,output); expectRows(file,90,ids,output);
    const auto other=hostPlacement(source(file,3),
        ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM,4096,2));
    EXPECT_THROW(context->prepareImmutableRowTable(*other.placement,memory,{},transfers),std::logic_error);
    // A different locality is a different sealed setup identity even when its
    // mode, source, allocator and every physical byte are otherwise identical.
    const auto different_locality=hostPlacement(source(file,3),policy,ImmutableRowTableHostPlacement::onNode(0));
    EXPECT_THROW(context->prepareImmutableRowTable(*different_locality.placement,memory,{},transfers),std::logic_error);
    auto foreign=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(selected.physical_plan),0);
    EXPECT_THROW(context->prepareImmutableRowTable(*selected.placement,foreign,{},transfers),std::logic_error);

}

/** @test Whole-host RAM follows task first-touch without claiming an exact physical node. */
TEST(ImmutableRowTable, WholeHostTaskPolicyPreservesCanonicalAllocator)
{
    RowFile file(90,3); const auto src=source(file,3);
    const auto policy=ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM,4096,2);
    const auto bytes=ImmutableRowTable::physicalBytes(src,policy);
    auto memory=authority(bytes,DeviceId::cpu());
    ImmutableRowTable table(src,policy,DeviceId::cpu(),memory,
        ImmutableRowTableHostPlacement::fromObservedNUMANode(-1));
    EXPECT_EQ(table.hostDevice(),DeviceId::cpu());
    EXPECT_TRUE(table.hostPlacement().usesTaskPolicy());
    EXPECT_FALSE(table.hostPlacement().numaNode());
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),bytes);
    const std::array<std::int64_t,2> ids{2,0};std::array<std::byte,180> output;
    table.gather(ids,output);expectRows(file,90,ids,output);
}

/** @test Explicit NUMA1 locality never changes the canonical CPU0 physical ledger. */
TEST(ImmutableRowTable, ExplicitNumaPlacementPreservesCanonicalPhysicalOwnership)
{
    RowFile file(90, 3);
    for (const auto mode : {ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped})
    {
        const auto src = source(file, 3);
        const auto policy = ImmutableRowTableStoragePolicy::forStorage(mode, 4096, 2);
        const auto planned = ImmutableRowTable::physicalBytes(src, policy);
        auto memory = authority(planned, DeviceId::cpu());
        {
            ImmutableRowTable table(src, policy, DeviceId::cpu(), memory,ImmutableRowTableHostPlacement::onNode(1));
            EXPECT_EQ(table.hostDevice(), DeviceId::cpu());
            EXPECT_EQ(table.hostPlacement(), ImmutableRowTableHostPlacement::onNode(1));
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation), planned);
            const std::array<std::int64_t, 2> ids{2, 0};
            std::array<std::byte, 180> bytes;
            table.gather(ids, bytes);
            expectRows(file, 90, ids, bytes);
        }
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_THROW(ImmutableRowTable(src, policy, DeviceId{DeviceType::CPU, 1}, memory,ImmutableRowTableHostPlacement::onNode(1)), std::exception);
        EXPECT_THROW(ImmutableRowTable(src, policy, DeviceId::cuda(0), memory,ImmutableRowTableHostPlacement::onNode(1)), std::invalid_argument);
    }
}

/**
 * @test EOF mappings report actual virtual pages while readable rows and PMA stay exact.
 *
 * On a4KiB-page host this is an8224-byte source: the last mapping contains
 *32readable file bytes but occupies4096virtual bytes. Full and partial windows
 * are hit/evicted/remapped under one- and two-slot policies without changing
 * the existing descriptor-directory admission or admitting page padding as data.
 */
TEST(ImmutableRowTable, EndOfFileWindowReportsPageRoundedVirtualExtentWithoutReadablePadding)
{
    const auto native_page=::sysconf(_SC_PAGESIZE);
    ASSERT_GT(native_page,0);const auto page=static_cast<std::size_t>(native_page);
    ASSERT_EQ(page%32,0u);
    const auto rows=2*page/32+1;
    RowFile file(32,rows);
    // The shared fixture normally has an unaligned prefix. Truncate before
    // immutable admission and bind offset0 for the exact EOF32-byte case.
    ASSERT_EQ(::ftruncate(file.fd,static_cast<off_t>(rows*32)),0);
    auto src=source(file,rows,GGUFTensorType::F32,8);src.file_offset=0;
    for(const std::size_t slots:{1u,2u})
    {
        SCOPED_TRACE("slots="+std::to_string(slots));
        const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,page,slots);
        const auto planned=ImmutableRowTable::physicalBytes(src,policy);auto memory=authority(planned);
        ImmutableRowTable table(src,policy,DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy());
        EXPECT_EQ(table.stats().mapped_virtual_bytes,0u);
        std::array<std::byte,48> destination;
        std::size_t calls=0;
        for(const auto id:{rows-1,rows-1,std::size_t{0},page/32,rows-1})
        {
            destination.fill(std::byte{0xa5});const std::array<std::int64_t,1> ids{static_cast<std::int64_t>(id)};
            table.gather(ids,std::span<std::byte>(destination).first(32));++calls;
            EXPECT_EQ(std::memcmp(destination.data(),file.bytes.data()+id*32,32),0);
            EXPECT_TRUE(std::all_of(destination.begin()+32,destination.end(),[](std::byte b){return b==std::byte{0xa5};}));
            const auto expected_pages=slots==1||calls<=2?1u:2u;
            EXPECT_EQ(table.stats().mapped_virtual_bytes,expected_pages*page);
            EXPECT_EQ(table.stats().physical_budget_bytes,planned);
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),planned);
        }
        const auto before=table.stats();const std::array<std::int64_t,1> invalid{static_cast<std::int64_t>(rows)};
        EXPECT_THROW(table.gather(invalid,std::span<std::byte>(destination).first(32)),std::out_of_range);
        EXPECT_EQ(table.stats().mapped_virtual_bytes,before.mapped_virtual_bytes);
        EXPECT_EQ(table.stats().rows,5u);EXPECT_EQ(table.stats().copied_bytes,5u*32);
        EXPECT_EQ(table.stats().window_hits,1u);EXPECT_EQ(table.stats().window_misses,4u);
        EXPECT_EQ(table.stats().window_evictions,slots==1?3u:2u);
    }
}

namespace {
/** @brief The model API cannot expose a raw table to bypass adaptive ownership or truthful observations. */
template<class T>concept HasRawImmutableTable=requires(const T &value){value.immutableRowTable();};
static_assert(!HasRawImmutableTable<IModelContext> && !HasRawImmutableTable<ModelContext>);
/** @return The canonical complete-source admission request for a fixed compact CPU continuation. */
ImmutableRowTablePlacementSelection contextAdaptivePlacement(const ModelMemoryProfile &profile,std::uint64_t maximum_rows,
    std::size_t cache_bytes,std::uint64_t aging=1048576)
{
    DevicePlanConfig c;c.world_rank=0;c.device=DeviceId::cpu();c.device_total_bytes=512ull<<30;c.device_free_bytes=500ull<<30;
    // The complete CPU BOM requires this participant's published execution
    // geometry even though this source/provider test never executes a GEMM.
    c.device_compute_units=8;c.cpu_execution=test::kSyntheticCPUExecutionGeometry;
    c.max_seq_len=64;c.activation_seq_len=2;c.first_layer=0;c.last_layer=47;
    c.ngram_execution_policy=ImmutableRowTablePolicy{ImmutableRowStoragePreference::HostAdaptiveRAMSSD,4096,2};
    c.ngram_execution_policy->host_adaptive={maximum_rows,cache_bytes,{.aging_accesses=aging}};
    const std::vector<std::vector<DevicePlanConfig>> groups{{c}};
    const auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,*c.ngram_execution_policy,groups,{},
        ImmutableRowTableHostPlacement::taskPolicy());
    if(!request)throw std::logic_error("Actual PLE consumer omitted by canonical admission");
    return ImmutableRowTablePlacementPlanner::select(*request,MemoryPlanner::plan(profile,{c}).physicalPlan());
}
}
/** @test Actual ModelContext publishes one native adaptive provider and validates its retained PMA/source/request authority. */
TEST(ImmutableRowTable, ModelContextAdaptiveProviderHasOneOwnerAndNoRawTableEscape)
{
    // This is a sparse GGUF metadata file. Only the bounded PLE rows below are
    // read; ordinary weights are never materialized and no inference is claimed.
    llaminar2::test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    for(const std::size_t slots:{0u,2u})
    {
        auto context=ModelContext::createForTesting(file.path(),nullptr,48,false);
        const_cast<GGUFModel &>(context->model())=source.loader().getModel();
        const auto selected=contextAdaptivePlacement(profile,slots,slots?1u<<20:0u);
        ASSERT_TRUE(selected.placement->isHostAdaptive());
        const auto &p=*selected.placement;
        const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(p.hostAdaptiveLayout(),p.allocators().front(),
            p.hostAdaptiveMaximumLiveRows(),p.hostAdaptivePolicy());
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(selected.physical_plan),0);
        const TransferEngine transfers;
        EXPECT_FALSE(context->hostImmutableRowProvider());
        context->prepareImmutableRowTable(p,memory,{},transfers);
        context->requireImmutableRowTablePlacementComplete(p,memory);
        auto first=context->hostImmutableRowProvider();ASSERT_TRUE(first);
        EXPECT_EQ(first->mode(),HostImmutableRowMode::AdaptiveRAMSSD);
        EXPECT_THROW((void)first->wholeStorage(),std::logic_error);
        EXPECT_EQ(first->source().format,static_cast<GGUFTensorType>(profile.flash_next->ple_table_gguf_type));
        EXPECT_EQ(first->source().rows,profile.flash_next->ple_table_rows);
        EXPECT_EQ(first->maximumLiveRows(),p.hostAdaptiveMaximumLiveRows());
        const auto observed=first->retainedOwnerObservation();
        EXPECT_EQ(observed.authority().get(),memory.get());EXPECT_EQ(observed.allocator(),p.allocators().front());
        EXPECT_EQ(observed.tableStorageBytes(),geometry.sourceBytes()+geometry.storageBytes());
        EXPECT_EQ(observed.lookupStagingBytes(),geometry.executionBytes());
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),observed.tableStorageBytes());
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),observed.lookupStagingBytes());
        // Native source holes contain original zero bytes. This proves copying,
        // duplicate order and owner work, not decoded model mathematics.
        const std::array<std::int64_t,3> ids{2,0,2};std::vector<std::byte> output(3*first->source().rowBytes(),std::byte{0x71});
        const auto cold=std::get<CpuImmutableRowGatherObservation>(first->gather(ids,output));
        EXPECT_EQ(cold.policy_ordinal,0u);EXPECT_EQ(cold.source_rows,2u);EXPECT_EQ(cold.duplicate_rows,1u);
        EXPECT_TRUE(std::all_of(output.begin(),output.end(),[](auto b){return b==std::byte{0};}));
        context->prepareImmutableRowTable(p,memory,{},transfers);
        context->requireImmutableRowTablePlacementComplete(p,memory);
        EXPECT_EQ(context->hostImmutableRowProvider(),first);
        const auto warm=std::get<CpuImmutableRowGatherObservation>(first->gather(ids,output));
        EXPECT_EQ(warm.policy_ordinal,1u);EXPECT_EQ(warm.duplicate_rows,1u);
        EXPECT_EQ(warm.source_rows,slots?0u:2u);EXPECT_EQ(warm.ram_rows,slots?2u:0u);
        EXPECT_EQ(first->retainedOwnerObservation().tableStorageBytes(),observed.tableStorageBytes());
        auto foreign=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(selected.physical_plan),0);
        EXPECT_THROW(context->prepareImmutableRowTable(p,foreign,{},transfers),std::logic_error);
        EXPECT_EQ(foreign->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        const auto changed=contextAdaptivePlacement(profile,slots+1,0);
        EXPECT_THROW(context->prepareImmutableRowTable(*changed.placement,memory,{},transfers),std::logic_error);
        context.reset(); // The real provider, not the context facade, retains both physical leases.
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),observed.lookupStagingBytes());
        first.reset();
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
