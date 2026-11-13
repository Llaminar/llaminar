/**
 * @file Test__ImmutableRowTableNUMA.cpp
 * @brief Actual owned-RAM NUMA placement and pre-read capability rejection.
 *
 * The shared NUMA authority supplies a one-page OS capability control. A private
 * sparse source never enters the page cache before construction. Successful
 * construction is checked through the kernel's physical-page query, independent
 * of the admitted resource identity. Rejection must leave native source pages
 * cold. No GPU, model, payload checksum, page migration or timing gate is used.
 */
#include "loaders/ImmutableRowTable.h"
#include "memory/NUMAAllocator.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <fstream>
#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>

namespace llaminar2::test
{
namespace
{
/** @brief An unread native source with no initial physical file-cache pages. */
class SparseRows final
{
public:
    static constexpr size_t Bytes=32u*1024u*1024u;
    std::string path;
    /** @brief Create only file metadata; native rows are never written or read. */
    SparseRows()
    {
        char pattern[]="/tmp/llaminar-ram-numa-XXXXXX";int fd=::mkstemp(pattern);
        if(fd<0)throw std::runtime_error("cannot create native RAM placement source");
        path=pattern;const auto status=::ftruncate(fd,Bytes);::close(fd);
        if(status){::unlink(path.c_str());throw std::runtime_error("cannot size placement source");}
    }
    /** @brief Retire the fixture after all exact owner and observation mappings. */
    ~SparseRows(){::unlink(path.c_str());}
    /** @return Native F32 page rows, preserving literal zero bytes as live data. */
    ImmutableRowTableSource source() const
    {
        return {.path=path,.tensor_name="fixture.ple.weight",.rows=Bytes/4096,
                .row_elements=1024,.format=GGUFTensorType::F32};
    }
    /** @return Whether the distant native source page remains physically cold. */
    bool cold() const
    {
        int fd=::open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("cannot inspect native source");
        void*view=::mmap(nullptr,Bytes,PROT_READ,MAP_SHARED,fd,0);::close(fd);
        if(view==MAP_FAILED)throw std::runtime_error("cannot map placement observation");
        unsigned char resident=0;const auto status=::mincore(static_cast<char*>(view)+Bytes-4096,4096,&resident);
        ::munmap(view,Bytes);if(status)throw std::runtime_error("cannot query native source residency");
        return !(resident&1u);
    }
};
/** @brief Admit the exact same RAM owner that construction must materialize. */
std::shared_ptr<PhysicalMemoryAuthority> authority(const ImmutableRowTableSource &source,DeviceId device)
{
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM,4096,1);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0,.device=device,.total_bytes=1ull<<30,.admission_available_bytes=1ull<<30},
             PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(source,policy));
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
}
/** @brief Preserve the caller's exact thread policy while forcing a contrasting source socket. */
class SourceSocket final
{
    cpu_set_t affinity_{};std::array<unsigned long,16> nodes_{};int mode_=MPOL_DEFAULT;
public:
    /** @brief Establish an actual source socket, independently from the table's target node. */
    SourceSocket(int cpu,int node)
    {
        if(::sched_getaffinity(0,sizeof(affinity_),&affinity_) ||
           ::get_mempolicy(&mode_,nodes_.data(),nodes_.size()*64,nullptr,0))
            throw std::runtime_error("cannot capture source socket state");
        cpu_set_t target;CPU_ZERO(&target);CPU_SET(cpu,&target);
        if(::sched_setaffinity(0,sizeof(target),&target))throw std::runtime_error("cannot bind source CPU");
        auto selected=nodes_;selected.fill(0);selected.at(node/64)|=1ul<<(node%64);
        if(::set_mempolicy(MPOL_BIND,selected.data(),selected.size()*64))
        {
            ::sched_setaffinity(0,sizeof(affinity_),&affinity_);
            throw std::runtime_error("cannot bind source memory policy");
        }
    }
    /** @brief Restore exact incoming thread state after the owned table has retired. */
    ~SourceSocket()
    {
        ::set_mempolicy(mode_,mode_==MPOL_DEFAULT?nullptr:nodes_.data(),mode_==MPOL_DEFAULT?0:nodes_.size()*64);
        ::sched_setaffinity(0,sizeof(affinity_),&affinity_);
    }
};
/** @brief A kernel-observed anonymous read-only virtual interval. */
struct Range{uintptr_t begin,end;bool operator==(const Range&) const=default;};
/** @return Exact anonymous immutable VMAs; no payload address is inferred from a model role. */
std::vector<Range> immutableAnonymousRanges()
{
    std::ifstream maps("/proc/self/maps");if(!maps)throw std::runtime_error("cannot inspect process mappings");
    std::vector<Range> result;std::string line;
    while(std::getline(maps,line))
    {
        std::istringstream fields(line);std::string interval,permissions,offset,device,inode,path;
        fields>>interval>>permissions>>offset>>device>>inode;std::getline(fields,path);
        if(permissions!="r--p" || path.find_first_not_of(' ')!=std::string::npos)continue;
        const auto dash=interval.find('-');result.push_back({std::stoull(interval.substr(0,dash),nullptr,16),
                                                           std::stoull(interval.substr(dash+1),nullptr,16)});
    }
    return result;
}
/** @return Newly materialized immutable intervals, excluding every pre-existing page. */
std::vector<Range> addedRanges(const std::vector<Range>&before,const std::vector<Range>&after)
{
    std::vector<Range> result;
    for(const auto range:after)
    {
        auto cursor=range.begin;
        for(const auto old:before)
        {
            if(old.end<=cursor || old.begin>=range.end)continue;
            if(old.begin>cursor)result.push_back({cursor,old.begin});
            cursor=std::max(cursor,old.end);
        }
        if(cursor<range.end)result.push_back({cursor,range.end});
    }
    return result;
}

/** @test Actual owned RAM resides on its admitted node even when pread runs elsewhere. */
TEST(ImmutableRowTableNUMA, OwnedRAMIsCertifiedOrRejectedBeforeNativeRead)
{
    auto &numa=NUMAAllocator::instance();
    const int cpu=::sched_getcpu();ASSERT_GE(cpu,0);
    const int source_node=::numa_node_of_cpu(cpu);
    // Explicit NUMA0 and NUMA1 cases share CPU0 accounting and either certify real placement or
    // reject through the same OS authority before reading a fresh source.
    for(const int target:{0,1})
    {
    SCOPED_TRACE(target);SparseRows file;
    const DeviceId device=DeviceId::cpu();
    const auto source=file.source();auto memory=authority(source,device);
    void *control=::mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    ASSERT_NE(control,MAP_FAILED);
    const bool capable=numa.prepareExternalReceiveRangeOnNode(control,4096,target);
    ::munmap(control,4096);
    const auto policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM,4096,1);
    if(!capable)
    {
        EXPECT_THROW(ImmutableRowTable(source,policy,device,memory,ImmutableRowTableHostPlacement::onNode(target)),std::runtime_error);
        EXPECT_TRUE(file.cold());continue;
    }
    SourceSocket placement(cpu,source_node);
    const auto before=immutableAnonymousRanges();
    {
        ImmutableRowTable table(source,policy,device,memory,ImmutableRowTableHostPlacement::onNode(target));
        auto added=addedRanges(before,immutableAnonymousRanges());
        added.erase(std::remove_if(added.begin(),added.end(),[](auto r){return r.end-r.begin!=SparseRows::Bytes;}),added.end());
        ASSERT_EQ(added.size(),1u) << "One exact, separately owned RAM mapping must become immutable";
        for(const auto offset:{size_t{0},SparseRows::Bytes/2,SparseRows::Bytes-4096})
            EXPECT_EQ(numa.getNUMANodeForAddress(reinterpret_cast<void*>(added[0].begin+offset)),target);
        EXPECT_EQ(table.hostDevice(),device);
        EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,
                  PhysicalMemoryMaterializationKind::NewAllocation),SparseRows::Bytes);
    }
    EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,
              PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}

/** @test A fictitious RAM node cannot cause native source reads or retain its lease. */
TEST(ImmutableRowTableNUMA, InvalidOwnedNodeRejectsBeforeReadingPayload)
{
    SparseRows file;const auto source=file.source();const DeviceId device=DeviceId::cpu();
    auto memory=authority(source,device);
    EXPECT_THROW(ImmutableRowTable(source,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM,4096,1),device,memory,ImmutableRowTableHostPlacement::onNode(INT_MAX)),std::runtime_error);
    EXPECT_TRUE(file.cold());
    EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,
              PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
} // namespace
} // namespace llaminar2::test
