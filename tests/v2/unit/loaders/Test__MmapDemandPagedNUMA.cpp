/**
 * @file Test__MmapDemandPagedNUMA.cpp
 * @brief Device-free live residency proof for sparse NUMA file mappings.
 *
 * A sparse file starts with no resident data pages. The OS capability control
 * distinguishes an authenticated bound mapping from a fatal placement failure;
 * neither case may touch the file. No models, accelerators, timing threshold,
 * payload checksum, or test-only execution substitute are used.
 */
#include "loaders/MmapRegion.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <climits>
#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <unistd.h>

namespace llaminar2::test
{
namespace
{
/** @brief Anonymous local sparse source, physically empty until a live page is faulted. */
class SparseFile final
{
public:
    std::string path;
    static constexpr size_t Bytes=16*1024*1024;
    /** @brief Create an unread sparse source, without writing or warming its pages. */
    SparseFile()
    {
        char pattern[]="/tmp/qfn-demand-paged-numa-XXXXXX";
        const auto fd=::mkstemp(pattern);
        if(fd<0)throw std::runtime_error("Cannot create sparse mapping test source");
        path=pattern;const auto status=::ftruncate(fd,Bytes);::close(fd);
        if(status){::unlink(path.c_str());throw std::runtime_error("Cannot size sparse mapping test source");}
    }
    /** @brief Retire the private source without retaining any fixture state. */
    ~SparseFile(){::unlink(path.c_str());}
    SparseFile(const SparseFile&)=delete;
    SparseFile &operator=(const SparseFile&)=delete;
};
/** @brief Probe the real OS binding permission on one anonymous control page, never the source file. */
bool canBind(int node)
{
    if(numa_available()<0)return false;
    const auto bytes=static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    auto *memory=::mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(memory==MAP_FAILED)throw std::runtime_error("Cannot create NUMA permission control");
    auto *mask=numa_allocate_nodemask();
    if(!mask){::munmap(memory,bytes);throw std::runtime_error("Cannot create NUMA permission mask");}
    numa_bitmask_clearall(mask);numa_bitmask_setbit(mask,node);
    const auto status=::mbind(memory,bytes,MPOL_BIND,mask->maskp,mask->size,0);
    numa_free_nodemask(mask);::munmap(memory,bytes);return status==0;
}
/** @brief Count actual resident source pages without faulting any of their bytes. */
size_t residentPages(const MmapRegion &mapping)
{
    const auto page=static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    std::vector<unsigned char> residency((mapping.size()+page-1)/page);
    if(::mincore(const_cast<uint8_t*>(mapping.data()),mapping.size(),residency.data()))
        throw std::runtime_error("Cannot query sparse source residency");
    return std::count_if(residency.begin(),residency.end(),[](auto flags){return (flags&1)!=0;});
}
/** @test A NUMA hint cannot convert a demand-paged mapping into a whole-file first-touch. */
TEST(MmapDemandPagedNUMA, BoundOrRejectedSourceNeverPrefaultsUnselectedPages)
{
    SparseFile file;
    const int node=numa_available()<0?0:numa_max_node();
    const auto allowed=canBind(node);
    auto mapping=MmapRegion::create(file.path,node,false,MmapRegion::PrefaultPolicy::DemandPaged);
    EXPECT_EQ(bool(mapping),allowed) << "NUMA placement must obey the OS binding result rather than bypass it";
    if(mapping)
    {
        EXPECT_FALSE(mapping->wasEagerPrefaulted());
        EXPECT_EQ(residentPages(*mapping),0u);
    }
    // This independent unbound control observes file-cache residency even if
    // strict NUMA placement was rejected before publication of a mapping.
    auto observation=MmapRegion::create(file.path,-1,false,MmapRegion::PrefaultPolicy::DemandPaged);
    ASSERT_NE(observation,nullptr);EXPECT_EQ(residentPages(*observation),0u);
}
/** @test Invalid demand-paged NUMA ownership fails before reading any source page. */
TEST(MmapDemandPagedNUMA, RejectsInvalidNodeBeforePublishingOrTouchingSource)
{
    SparseFile file;
    auto mapping=MmapRegion::create(file.path,INT_MAX,false,MmapRegion::PrefaultPolicy::DemandPaged);
    EXPECT_EQ(mapping,nullptr);
    auto observation=MmapRegion::create(file.path,-1,false,MmapRegion::PrefaultPolicy::DemandPaged);
    ASSERT_NE(observation,nullptr);EXPECT_EQ(residentPages(*observation),0u);
}
} // namespace
} // namespace llaminar2::test
