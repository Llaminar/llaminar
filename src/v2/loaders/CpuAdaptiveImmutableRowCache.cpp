/**
 * @file CpuAdaptiveImmutableRowCache.cpp
 * @brief Exact PMA-backed native host caching with bounded hashed lookup and deterministic replacement.
 *
 * The private SSD source authenticates original bytes even for warm cache hits.
 * Miss responses use caller storage. Pins remove slots from a fixed eligibility
 * heap, permitting immediate all-pinned progress without allocation or waiting.
 * Lazy frequency aging follows published logical accesses, never a wall clock.
 * Any source-authentication failure bars future owner/pin reads permanently.
 * Ordinary facade retirement alone keeps existing pinned bytes readable.
 * Exclusive subordinate invalidation detaches the resident lookup before ACK;
 * old pins retain their exact bytes until the last reader releases the slot.
 * The existing processor owns channel sequence/admission authentication. Its
 * completed grants keep bounded CPU read records, never a GPU location shadow.
 */
#include "CpuAdaptiveImmutableRowCache.h"
#include "memory/NUMAAllocator.h"
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <sys/mman.h>
#include <unistd.h>
namespace llaminar2 {
namespace {
constexpr auto None=std::numeric_limits<std::uint32_t>::max();
/** @brief One bounded native slot; indices/grants never point to an unpriced secondary container. */
struct Slot {
    std::uint64_t row=0,generation=0,touched=0;
    std::uint32_t pins=0,heap_position=None,previous=None,next=None;
    bool ready=false,protected_region=false,retiring=false;
};
/** @brief Lazily aged bounded count-min cell; no source-row-sized history exists. */
struct Frequency {std::uint32_t epoch=0;std::uint16_t count=0;};
/** @brief Per-response dedup entry; stamped batches avoid clearing physical capacity per call. */
struct Request {std::uint64_t row=0,stamp=0;std::uint32_t first=0;};
/** @brief One initial resident/duplicate response position, independent of later policy mutation. */
struct ResponsePlan {std::uint32_t first=0,initial_slot=None;};
/** @brief Bounded outstanding destination tokens only; never a persistent GPU residency directory. */
struct GrantTarget {std::uint64_t token=0,row=0,stamp=0;};
/** @brief Immutable CPU read-grant rows retained only until the existing channel consumption ACK. */
struct GrantedRow {std::int64_t row=0;ImmutableRowReadIntent intent=ImmutableRowReadIntent::Read;};
/** @return Checked byte/index addition, never a free-memory calculation. */
std::size_t plus(std::size_t a,std::size_t b) {
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::overflow_error("CPU row cache byte sum overflows");
    return a+b;
}
/** @return Checked lifetime byte/window observation, failing instead of publishing wrapped work. */
std::uint64_t countPlus(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a)throw std::overflow_error("CPU row cache observation counter overflows");
    return a+b;
}
/** @return Checked native/array extent. */
std::size_t times(std::size_t a,std::size_t b) {
    if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::overflow_error("CPU row cache array overflows");
    return a*b;
}
/** @return Checked aligned offset; alignment owns bytes, not communication volume. */
std::size_t align(std::size_t n,std::size_t a) {return times(plus(n,a-1)/a,a);}
/** @return Existing host page authority for anonymous physical ownership. */
std::size_t page() {const auto n=::sysconf(_SC_PAGESIZE);if(n<=0)throw std::runtime_error("Cannot observe cache page size");return static_cast<std::size_t>(n);}
/** @return Half-loaded fixed hash capacity, without truncating the uint32 slot sentinel. */
std::size_t hashCount(std::size_t n) {
    if(!n)return 0;
    const auto wanted=times(n,2);
    if(wanted>std::bit_floor(std::numeric_limits<std::size_t>::max()))throw std::overflow_error("CPU row cache hash capacity overflows");
    return std::bit_ceil(wanted);
}
/** @return Fixed avalanche hash; neither key ranges nor source-home quotas determine placement. */
std::uint64_t hash(std::uint64_t n) {n^=n>>30;n*=0xbf58476d1ce4e5b9ULL;n^=n>>27;n*=0x94d049bb133111ebULL;return n^(n>>31);}
/** @return Whether two live byte spans overlap, rejecting wrapped pointer arithmetic. */
bool overlaps(const void *a,std::size_t na,const void *b,std::size_t nb) {
    if(!na || !nb)return false;
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(na>std::numeric_limits<std::uintptr_t>::max()-x || nb>std::numeric_limits<std::uintptr_t>::max()-y)
        throw std::invalid_argument("CPU row cache caller span wraps address space");
    return x<y+nb && y<x+na;
}
/** @brief One exact anonymous allocation, unmapped before its sole PMA lease retires. */
struct Block {
    PhysicalMemoryAllocationLease lease;
    void *data=MAP_FAILED;std::size_t bytes=0;
    ~Block(){if(data!=MAP_FAILED && ::munmap(data,bytes))std::terminate();}
    /** @brief Materialize only a canonical admitted owner; exact NUMA precedes every first touch. */
    void allocate(const std::shared_ptr<PhysicalMemoryAuthority> &memory,DeviceId host,
        PhysicalMemoryOwner owner,std::size_t n,ImmutableRowTableHostPlacement locality) {
        if(!n)return;
        lease=memory->claimNewAllocation(host,owner,n);bytes=n;
        data=::mmap(nullptr,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(data==MAP_FAILED)throw std::runtime_error("CPU row cache mmap failed: "+std::string(std::strerror(errno)));
        if(const auto node=locality.numaNode();node &&
            !NUMAAllocator::instance().prepareExternalReceiveRangeOnNode(data,n,*node))
            throw std::runtime_error("CPU row cache exact NUMA placement rejected");
    }
};
}
CpuAdaptiveImmutableRowCacheGeometry CpuAdaptiveImmutableRowCacheGeometry::fromLayout(
    const ImmutableRowTableLayoutGeometry &layout,PhysicalMemoryAllocatorIdentity host,
    std::size_t maximum_live_rows,CpuImmutableRowCachePolicy policy) {
    return project(layout,host,maximum_live_rows,policy,std::nullopt);
}
CpuAdaptiveImmutableRowCacheGeometry CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(
    const ImmutableRowTableLayoutGeometry &layout,PhysicalMemoryAllocatorIdentity host,
    PhysicalMemoryAllocatorIdentity requester,CpuImmutableRowCachePolicy policy) {
    if(requester.world_rank!=host.world_rank || !requester.device.is_gpu() ||
        layout.consumers().size()!=1 || layout.consumers().front()!=requester ||
        !layout.maximumInflightRows() || layout.maximumInflightRows()>std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Subordinate CPU source requires exact one local GPU requester and bounded live rows");
    std::uint64_t gpu_slots=0;
    for(const auto &owner:layout.owners())if(owner.key.tier==ImmutableRowStorage::VRAM) {
        if(owner.key.allocator!=requester || owner.fixed_rows)
            throw std::invalid_argument("Subordinate CPU source has a foreign GPU owner");
        gpu_slots=countPlus(gpu_slots,owner.cache_rows);
    }
    if((gpu_slots && layout.gpuCoordinator()!=requester) || (!gpu_slots && layout.gpuCoordinator()))
        throw std::invalid_argument("Subordinate CPU source requester differs from the actual GPU coordinator");
    if(layout.sourceServices().size()!=1 || layout.sourceServices().front().allocator!=host)
        throw std::invalid_argument("Subordinate CPU source requires its sole original rank-local SSD service");
    return project(layout,host,static_cast<std::size_t>(layout.maximumInflightRows()),policy,requester);
}
CpuAdaptiveImmutableRowCacheGeometry CpuAdaptiveImmutableRowCacheGeometry::project(
    const ImmutableRowTableLayoutGeometry &layout,PhysicalMemoryAllocatorIdentity host,
    std::size_t maximum_live_rows,CpuImmutableRowCachePolicy policy,
    std::optional<PhysicalMemoryAllocatorIdentity> requester) {
    if(host.world_rank<0 || !host.device.is_cpu() || !maximum_live_rows || maximum_live_rows>=None ||
        !policy.aging_accesses || !policy.protected_denominator || policy.protected_numerator>policy.protected_denominator)
        throw std::invalid_argument("CPU row cache requires exact host, representable live rows and valid policy");
    if(layout.residency()!=ImmutableRowTableResidency::Adaptive || !layout.tiers().contains(ImmutableRowStorage::MemoryMapped) ||
        ((!requester) && (layout.gpuCoordinator() || layout.gpuDirectoryRecordBound())) ||
        !layout.replicaGroups().empty() || !layout.fixedRanges().empty())
        throw std::invalid_argument("CPU row cache requires a host-coordinated adaptive SSD domain, not GPU grants or fixed copies");
    // A resolved layout's request bound is the sole capacity authority. Raw
    // component geometry may leave it unspecified and bind it here once.
    if(layout.maximumInflightRows() && layout.maximumInflightRows()!=maximum_live_rows)
        throw std::invalid_argument("CPU row cache live-row capacity differs from its declared request envelope");
    const ImmutableRowTableSourceService *service=nullptr;
    for(const auto &s:layout.sourceServices())if(s.allocator==host)service=&s;
    if(!service)throw std::invalid_argument("CPU row cache has no exact local original SSD source service");
    CpuAdaptiveImmutableRowCacheGeometry g(layout.extent(),host,service->windows,
        ImmutableRowTableHostPlacement::taskPolicy(),policy);
    g.requester_=requester;
    std::optional<std::size_t> limit;
    for(const auto &o:layout.owners()) {
        if(o.key.tier==ImmutableRowStorage::VRAM) {
            if(!requester && (o.cache_rows || o.fixed_rows))throw std::invalid_argument("CPU row cache cannot bypass a GPU residency authority");
            continue;
        }
        if(o.key.allocator!=host) {
            if(o.cache_rows || o.fixed_rows)throw std::invalid_argument("CPU row cache cannot own another host's RAM pool");
            continue;
        }
        if(o.cache_rows>=None)throw std::overflow_error("CPU row cache slots exceed uint32 grant encoding");
        g.slots_=static_cast<std::size_t>(o.cache_rows);g.locality_=o.host_placement;limit=o.physical_byte_limit;
    }
    g.maximum_live_rows_=maximum_live_rows;g.index_count_=hashCount(g.slots_);
    // Four disjoint lanes prevent a tiny cache from incrementing the same
    // two cells repeatedly and losing every candidate/resident distinction.
    // The bounded minimum and every lane are included in optional storage.
    g.sketch_count_=g.slots_?times(std::max<std::size_t>(16,g.index_count_),4):0;
    g.request_count_=hashCount(maximum_live_rows);
    g.protected_limit_=times(g.slots_,policy.protected_numerator)/policy.protected_denominator;
    if(g.slots_>1)g.protected_limit_=std::min(g.protected_limit_,g.slots_-1);
    g.index_offset_=align(times(g.slots_,sizeof(Slot)),alignof(std::uint32_t));
    g.heap_offset_=align(plus(g.index_offset_,times(g.index_count_,sizeof(std::uint32_t))),alignof(std::uint32_t));
    g.sketch_offset_=align(plus(g.heap_offset_,times(g.slots_,sizeof(std::uint32_t))),alignof(Frequency));
    g.payload_offset_=align(plus(g.sketch_offset_,times(g.sketch_count_,sizeof(Frequency))),64);
    const auto native=times(g.slots_,g.extent_.rowBytes());
    g.storage_bytes_=g.slots_?align(plus(g.payload_offset_,native),page()):0;
    g.plan_offset_=align(times(g.request_count_,sizeof(Request)),alignof(ResponsePlan));
    g.scatter_offset_=align(plus(g.plan_offset_,times(maximum_live_rows,sizeof(ResponsePlan))),alignof(ImmutableRowTableScatterRow));
    const auto scatter_end=plus(g.scatter_offset_,times(maximum_live_rows,sizeof(ImmutableRowTableScatterRow)));
    g.grant_offset_=align(scatter_end,alignof(GrantTarget));
    g.read_rows_offset_=align(plus(g.grant_offset_,requester?times(g.request_count_,sizeof(GrantTarget)):0),alignof(GrantedRow));
    g.execution_bytes_=align(requester?plus(g.read_rows_offset_,times(maximum_live_rows,sizeof(GrantedRow))):scatter_end,page());
    g.source_bytes_=ImmutableRowTable::physicalBytesForNativeExtent(g.extent_.nativeBytes(),g.source_policy_);
    if(limit && g.storage_bytes_>*limit)throw std::invalid_argument("CPU row cache payload and ALL metadata exceed RAM owner byte limit");
    return g;
}
PhysicalMemoryPlan CpuAdaptiveImmutableRowCacheGeometry::mandatoryBOM(PhysicalMemoryResource resource) const {
    if(!resource.valid() || PhysicalMemoryAllocatorIdentity{resource.world_rank,resource.device}!=host_)
        throw std::invalid_argument("CPU row cache BOM uses a foreign observed host resource");
    PhysicalMemoryPlanBuilder b;b.add(resource,PhysicalMemoryOwner::ImmutableTableStorage,source_bytes_);
    b.add(resource,PhysicalMemoryOwner::ImmutableLookupStaging,execution_bytes_);return b.build();
}
PhysicalMemoryPlan CpuAdaptiveImmutableRowCacheGeometry::cacheBOM(PhysicalMemoryResource resource) const {
    if(!resource.valid() || PhysicalMemoryAllocatorIdentity{resource.world_rank,resource.device}!=host_)
        throw std::invalid_argument("CPU row cache BOM uses a foreign observed host resource");
    PhysicalMemoryPlanBuilder b;b.add(resource,PhysicalMemoryOwner::ImmutableTableStorage,storage_bytes_);return b.build();
}
PhysicalMemoryPlan CpuAdaptiveImmutableRowCacheGeometry::bom(PhysicalMemoryResource resource) const {
    PhysicalMemoryPlanBuilder b(mandatoryBOM(resource));
    const auto cache=cacheBOM(resource);
    for(const auto &item:cache.resources())b.add(item);
    return b.build();
}
struct CpuAdaptiveImmutableRowCache::Impl {
    CpuAdaptiveImmutableRowCacheGeometry geometry;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::unique_ptr<ImmutableRowTable> source;
    Block storage,execution;
    mutable std::mutex mutex;
    CpuImmutableRowCacheStats counters;
    std::uint64_t next_policy_ordinal=0,clock=0;
    std::uint32_t filled=0,free_head=None,heap_size=0,protected_head=None,protected_tail=None,protected_count=0;
    const qwen4exp::ImmutableLookupHostProcessor *subordinate_channel=nullptr;
    bool retired=false,failed=false;
    /** @brief Retain one proven immutable geometry; source creation/allocation complete before facade publication. */
    Impl(CpuAdaptiveImmutableRowCacheGeometry g,std::shared_ptr<PhysicalMemoryAuthority> m):geometry(std::move(g)),memory(std::move(m)){}
    /** @return Pre-admitted slot records. */
    Slot *slots() const {return static_cast<Slot*>(storage.data);}
    /** @return Pre-admitted resident hash, storing slot indices instead of duplicating source rows. */
    std::uint32_t *index() const {return reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(storage.data)+geometry.index_offset_);}
    /** @return Fixed eligibility heap containing exactly unpinned ready slots. */
    std::uint32_t *heap() const {return reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(storage.data)+geometry.heap_offset_);}
    /** @return Bounded lazy frequency sketch. */
    Frequency *sketch() const {return reinterpret_cast<Frequency*>(static_cast<std::byte*>(storage.data)+geometry.sketch_offset_);}
    /** @return Exact original row payload region, with no row padding. */
    std::byte *data(std::uint32_t slot) const {return static_cast<std::byte*>(storage.data)+geometry.payload_offset_+slot*geometry.extent_.rowBytes();}
    /** @brief Reject all new work after failure/retirement; existing pin bytes remain separately valid. */
    void requireOpen() const {if(retired || failed)throw std::runtime_error("CPU row cache is retired or failed");}
    /** @return Slot index or None, never allocating or scanning source metadata. */
    std::uint32_t find(std::uint64_t row) const {
        if(!geometry.slots_)return None;
        const auto mask=geometry.index_count_-1;auto at=static_cast<std::size_t>(hash(row))&mask;
        for(std::size_t n=0;n<geometry.index_count_;++n,at=(at+1)&mask) {
            const auto slot=index()[at];if(slot==None)return None;if(slots()[slot].row==row)return slot;
        }
        throw std::logic_error("CPU row cache resident index lost an empty slot");
    }
    /** @brief Publish one ready native slot in the half-loaded resident index. */
    void insert(std::uint32_t slot) {
        const auto mask=geometry.index_count_-1;auto at=static_cast<std::size_t>(hash(slots()[slot].row))&mask;
        for(std::size_t n=0;n<geometry.index_count_;++n,at=(at+1)&mask)
            if(index()[at]==None){index()[at]=slot;return;}
        throw std::logic_error("CPU row cache resident index cannot insert a complete row");
    }
    /** @brief Backward-shift deletion prevents tombstones growing with cache churn. */
    void erase(std::uint32_t slot) {
        const auto mask=geometry.index_count_-1;auto hole=static_cast<std::size_t>(hash(slots()[slot].row))&mask;
        std::size_t probes=0;
        while(index()[hole]!=slot) {
            if(++probes==geometry.index_count_)throw std::logic_error("CPU row cache erased a foreign slot");
            hole=(hole+1)&mask;
        }
        for(auto at=(hole+1)&mask;index()[at]!=None;at=(at+1)&mask) {
            const auto home=static_cast<std::size_t>(hash(slots()[index()[at]].row))&mask;
            if(((at-home)&mask)>((hole-home)&mask)){index()[hole]=index()[at];hole=at;}
        }
        index()[hole]=None;
    }
    /** @return Deterministic eligible victim order: probation, logical recency, original row, stable slot. */
    bool less(std::uint32_t a,std::uint32_t b) const {
        const auto &x=slots()[a],&y=slots()[b];
        return std::tuple{x.protected_region,x.touched,x.row,a}<std::tuple{y.protected_region,y.touched,y.row,b};
    }
    /** @brief Restore a changed heap key without allocations or completion-time recency. */
    void fix(std::uint32_t at) {
        while(at && less(heap()[at],heap()[(at-1)/2])) {
            std::swap(heap()[at],heap()[(at-1)/2]);slots()[heap()[at]].heap_position=at;at=(at-1)/2;slots()[heap()[at]].heap_position=at;
        }
        for(;;) {
            auto child=static_cast<std::size_t>(at)*2+1;if(child>=heap_size)break;
            if(child+1<heap_size && less(heap()[child+1],heap()[child]))++child;
            if(!less(heap()[child],heap()[at]))break;
            std::swap(heap()[child],heap()[at]);slots()[heap()[at]].heap_position=at;at=static_cast<std::uint32_t>(child);slots()[heap()[at]].heap_position=at;
        }
    }
    /** @brief Add one unpinned eligible row; caller owns slot state under the mutex. */
    void eligible(std::uint32_t slot) {const auto at=heap_size++;heap()[at]=slot;slots()[slot].heap_position=at;fix(at);}
    /** @brief Remove a pinned/replaced row without searching other slots. */
    void ineligible(std::uint32_t slot) {
        const auto at=slots()[slot].heap_position;if(at==None)throw std::logic_error("CPU row cache heap lost slot");
        slots()[slot].heap_position=None;--heap_size;
        if(at<heap_size){heap()[at]=heap()[heap_size];slots()[heap()[at]].heap_position=at;fix(at);}
    }
    /** @brief Remove one protected-list member, including pinned rows, without changing logical age. */
    void unlink(std::uint32_t slot) {
        auto &s=slots()[slot];
        if(s.previous!=None)slots()[s.previous].next=s.next;else protected_head=s.next;
        if(s.next!=None)slots()[s.next].previous=s.previous;else protected_tail=s.previous;
        s.previous=s.next=None;
    }
    /** @brief Observe actual logical reuse and maintain bounded probation/protected policy. */
    void touch(std::uint32_t slot) {
        auto &s=slots()[slot];s.touched=clock;
        if(s.protected_region)unlink(slot);else{s.protected_region=true;++protected_count;}
        s.previous=None;s.next=protected_head;if(protected_head!=None)slots()[protected_head].previous=slot;
        else protected_tail=slot;
        protected_head=slot;
        if(!s.pins)fix(s.heap_position);
        if(protected_count>geometry.protected_limit_) {
            const auto victim=protected_tail;unlink(victim);slots()[victim].protected_region=false;--protected_count;
            if(!slots()[victim].pins)fix(slots()[victim].heap_position);
        }
    }
    /** @return Lazily aged count-min estimate; observation is deterministic in published access order. */
    std::uint16_t frequency(std::uint64_t row,bool observe) {
        if(!geometry.sketch_count_)return 0;
        const auto epoch=static_cast<std::uint32_t>(clock/geometry.policy_.aging_accesses);
        std::uint16_t result=std::numeric_limits<std::uint16_t>::max();
        for(std::uint64_t lane=0;lane<4;++lane) {
            const auto width=geometry.sketch_count_/4;
            auto &cell=sketch()[lane*width+(hash(row+lane*0x9e3779b97f4a7c15ULL)&(width-1))];
            const auto elapsed=epoch-cell.epoch;cell.count=static_cast<std::uint16_t>(cell.count>>std::min<std::uint32_t>(elapsed,16));cell.epoch=epoch;
            if(observe && cell.count!=std::numeric_limits<std::uint16_t>::max())++cell.count;
            result=std::min(result,cell.count);
        }
        return result;
    }
    /** @brief Recycle only detached, unpinned bytes, preserving generation until the next actual fill. */
    void recycle(std::uint32_t slot) {
        auto &s=slots()[slot];
        if(!s.ready || !s.retiring || s.pins || s.heap_position!=None)
            throw std::logic_error("CPU subordinate source cannot recycle live reader backing");
        s.ready=false;s.retiring=false;s.next=free_head;free_head=slot;
    }
    /** @brief Invalidate new reads before acknowledgement; existing pins keep their exact old generation. */
    void invalidate(std::uint64_t row) {
        const auto slot=find(row);if(slot==None)return;
        if(counters.invalidations==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("CPU subordinate invalidation observation exhausted");
        auto &s=slots()[slot];erase(slot);
        if(!s.pins)ineligible(slot);
        if(s.protected_region){unlink(slot);--protected_count;s.protected_region=false;}
        s.retiring=true;--counters.resident_rows;++counters.invalidations;
        if(!s.pins)recycle(slot);
    }
    /** @brief Optionally retain an already complete response; failure never publishes half-filled bytes. */
    void admit(std::uint64_t row,const std::byte *response) {
        if(!geometry.slots_)return;
        std::uint32_t slot;
        bool new_resident=true;
        if(free_head!=None){slot=free_head;free_head=slots()[slot].next;}
        else if(filled<geometry.slots_)slot=filled++;
        else {
            if(!heap_size){++counters.all_pinned_responses;return;}
            slot=heap()[0];
            if(frequency(row,false)<=frequency(slots()[slot].row,false)){++counters.admission_rejections;return;}
            ineligible(slot);erase(slot);
            if(slots()[slot].protected_region){unlink(slot);--protected_count;}
            ++counters.evictions;new_resident=false;
        }
        auto &s=slots()[slot];
        if(s.generation==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("CPU row slot generation exhausted");
        std::memcpy(data(slot),response,geometry.extent_.rowBytes());
        const auto generation=s.generation+1;s=Slot{};s.row=row;s.generation=generation;s.touched=clock;s.ready=true;
        insert(slot);eligible(slot);++counters.admissions;if(new_resident)++counters.resident_rows;
    }
};
std::unique_ptr<CpuAdaptiveImmutableRowCache> CpuAdaptiveImmutableRowCache::create(ImmutableRowTableSource source,
    CpuAdaptiveImmutableRowCacheGeometry geometry,std::shared_ptr<PhysicalMemoryAuthority> memory) {
    if(!memory || memory->worldRank()!=geometry.host_.world_rank || !memory->contains(geometry.host_.device) ||
        source.rows!=geometry.extent_.rows() || source.format!=geometry.extent_.format() || source.rowBytes()!=geometry.extent_.rowBytes())
        throw std::invalid_argument("CPU row cache source or canonical physical authority differs from admitted geometry");
    auto state=std::make_shared<Impl>(std::move(geometry),std::move(memory));const auto &g=state->geometry;
    state->source=std::make_unique<ImmutableRowTable>(std::move(source),g.source_policy_,g.host_.device,state->memory,g.locality_);
    // Authenticate the named/opened source before materializing optional cache
    // or execution arrays, then authenticate again before facade publication.
    state->source->authenticateSource();
    state->storage.allocate(state->memory,g.host_.device,PhysicalMemoryOwner::ImmutableTableStorage,g.storage_bytes_,g.locality_);
    state->execution.allocate(state->memory,g.host_.device,PhysicalMemoryOwner::ImmutableLookupStaging,g.execution_bytes_,g.locality_);
    for(std::size_t i=0;i<g.slots_;++i)std::construct_at(state->slots()+i);
    for(std::size_t i=0;i<g.index_count_;++i)std::construct_at(state->index()+i,None);
    for(std::size_t i=0;i<g.slots_;++i)std::construct_at(state->heap()+i,None);
    for(std::size_t i=0;i<g.sketch_count_;++i)std::construct_at(state->sketch()+i);
    for(std::size_t i=0;i<g.request_count_;++i)std::construct_at(static_cast<Request*>(state->execution.data)+i);
    auto *plans=reinterpret_cast<ResponsePlan*>(static_cast<std::byte*>(state->execution.data)+g.plan_offset_);
    auto *targets=reinterpret_cast<ImmutableRowTableScatterRow*>(static_cast<std::byte*>(state->execution.data)+g.scatter_offset_);
    for(std::size_t i=0;i<g.maximum_live_rows_;++i){std::construct_at(plans+i);std::construct_at(targets+i);}
    if(g.requester_) {
        auto *grants=reinterpret_cast<GrantTarget*>(static_cast<std::byte*>(state->execution.data)+g.grant_offset_);
        for(std::size_t i=0;i<g.request_count_;++i)std::construct_at(grants+i);
        auto *rows=reinterpret_cast<GrantedRow*>(static_cast<std::byte*>(state->execution.data)+g.read_rows_offset_);
        for(std::size_t i=0;i<g.maximum_live_rows_;++i)std::construct_at(rows+i);
    }
    state->source->authenticateSource();return std::unique_ptr<CpuAdaptiveImmutableRowCache>(new CpuAdaptiveImmutableRowCache(std::move(state)));
}
CpuAdaptiveImmutableRowCache::~CpuAdaptiveImmutableRowCache(){retire();}
void CpuAdaptiveImmutableRowCache::authenticateSource() const {
    auto &s=*impl_;std::lock_guard lock(s.mutex);s.requireOpen();
    try {s.source->authenticateSource();}
    catch(...) {s.failed=true;throw;}
}
std::optional<PhysicalMemoryAllocatorIdentity> CpuAdaptiveImmutableRowCache::subordinateRequester() const noexcept {
    return impl_->geometry.requester_;
}
void CpuAdaptiveImmutableRowCache::bindSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel) {
    auto &s=*impl_;std::lock_guard lock(s.mutex);s.requireOpen();
    if(!channel || !s.geometry.requester_ || s.subordinate_channel)
        throw std::invalid_argument("Subordinate CPU source requires its sole actual processor binding");
    try{s.source->authenticateSource();}catch(...){s.failed=true;throw;}
    s.subordinate_channel=channel;
}
void CpuAdaptiveImmutableRowCache::retireSubordinate(const qwen4exp::ImmutableLookupHostProcessor *channel) noexcept {
    auto &s=*impl_;std::lock_guard lock(s.mutex);
    if(s.subordinate_channel==channel)s.retired=true;
}
CpuAdaptiveImmutableRowCache::ReadGrant CpuAdaptiveImmutableRowCache::authorizeSubordinate(
    const qwen4exp::ImmutableLookupHostProcessor *channel,std::uint64_t generation,std::uint64_t sequence,
    std::span<const std::int64_t> ids,std::span<const ImmutableRowSubordinateIntent> intents,
    std::span<ImmutableRowInvalidationAcknowledgement> acknowledgements) {
    auto &s=*impl_;std::lock_guard lock(s.mutex);s.requireOpen();
    if(!s.geometry.requester_ || s.subordinate_channel!=channel || !generation || !sequence ||
        ids.size()!=intents.size() || ids.size()!=acknowledgements.size() || ids.size()>s.geometry.maximum_live_rows_)
        throw std::invalid_argument("Subordinate CPU source requires the original acquired processor frame");
    return ReadGrant(impl_,channel,generation,sequence,ids,intents,acknowledgements);
}
CpuImmutableRowGatherObservation CpuAdaptiveImmutableRowCache::gather(std::span<const std::int64_t> ids,std::span<std::byte> destination) {
    return gatherImpl(ids,destination,nullptr);
}
CpuImmutableRowGatherObservation CpuAdaptiveImmutableRowCache::gatherImpl(
    std::span<const std::int64_t> ids,std::span<std::byte> destination,const ReadGrant *grant) {
    auto &s=*impl_;std::lock_guard lock(s.mutex);s.requireOpen();const auto &g=s.geometry;
    if(static_cast<bool>(g.requester_)!=static_cast<bool>(grant) ||
        (grant && (grant->owner_.get()!=impl_.get() || grant->channel_!=s.subordinate_channel ||
         grant->ids_.data()!=ids.data() || grant->ids_.size()!=ids.size())))
        throw std::invalid_argument("Subordinate CPU source forbids independent or foreign reads");
    if(s.next_policy_ordinal==std::numeric_limits<std::uint64_t>::max() ||
        ids.size()>g.maximum_live_rows_ || destination.size()!=times(ids.size(),g.extent_.rowBytes()) ||
        destination.size()>std::numeric_limits<std::uint64_t>::max()-s.counters.completed_bytes ||
        ids.size()>std::numeric_limits<std::uint64_t>::max()-s.clock ||
        (s.clock+ids.size())/g.policy_.aging_accesses>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("CPU row cache live capacity, policy ordinal or epoch is invalid");
    for(auto id:ids)if(id<0 || static_cast<std::uint64_t>(id)>=g.extent_.rows())throw std::out_of_range("CPU row cache source ID is invalid");
    if(overlaps(ids.data(),ids.size_bytes(),destination.data(),destination.size()) ||
        overlaps(destination.data(),destination.size(),s.storage.data,s.storage.bytes) ||
        overlaps(destination.data(),destination.size(),s.execution.data,s.execution.bytes))
        throw std::invalid_argument("CPU row cache caller destination aliases IDs or internal storage");
    if(grant) {
        if(overlaps(grant->intents_.data(),grant->intents_.size_bytes(),destination.data(),destination.size()) ||
            overlaps(grant->acknowledgements_.data(),grant->acknowledgements_.size_bytes(),destination.data(),destination.size()) ||
            overlaps(grant->acknowledgements_.data(),grant->acknowledgements_.size_bytes(),ids.data(),ids.size_bytes()) ||
            overlaps(grant->acknowledgements_.data(),grant->acknowledgements_.size_bytes(),grant->intents_.data(),grant->intents_.size_bytes()))
            throw std::invalid_argument("Subordinate CPU response metadata aliases live payload");
        for(const auto &intent:grant->intents_) {
            if((intent.intent!=ImmutableRowReadIntent::Read && intent.intent!=ImmutableRowReadIntent::ReadAndInvalidate) ||
                ((intent.intent==ImmutableRowReadIntent::Read)!=(intent.destination_grant==0)))
                throw std::invalid_argument("Subordinate CPU read intent or destination grant is invalid");
        }
    }
    const auto before=s.counters;
    {
        auto *requests=static_cast<Request*>(s.execution.data);
        auto *plans=reinterpret_cast<ResponsePlan*>(static_cast<std::byte*>(s.execution.data)+g.plan_offset_);
        auto *targets=reinterpret_cast<ImmutableRowTableScatterRow*>(static_cast<std::byte*>(s.execution.data)+g.scatter_offset_);
        // This is private scratch freshness, not a caller/channel authority.
        // A successful ordinal is published only after source validation/copies.
        const auto stamp=s.next_policy_ordinal+1;
        auto *grants=grant?reinterpret_cast<GrantTarget*>(static_cast<std::byte*>(s.execution.data)+g.grant_offset_):nullptr;
        const auto rejectGrant=[&](const char *message) {
            // Invalid frame rejection preserves cache/output/order and leaves
            // the next attempt's private scratch stamp fresh.
            for(std::size_t cell=0;cell<g.request_count_;++cell) {
                if(requests[cell].stamp==stamp)requests[cell].stamp=0;
                if(grants && grants[cell].stamp==stamp)grants[cell].stamp=0;
            }
            throw std::invalid_argument(message);
        };
        std::size_t misses=0;
        // Resolve the complete initial read set without changing policy or output.
        for(std::size_t i=0;i<ids.size();++i) {
            const auto row=static_cast<std::uint64_t>(ids[i]);auto at=hash(row)&(g.request_count_-1);
            while(requests[at].stamp==stamp && requests[at].row!=row)at=(at+1)&(g.request_count_-1);
            if(requests[at].stamp==stamp) {
                const auto first=requests[at].first;
                if(grant && (grant->intents_[first].intent!=grant->intents_[i].intent ||
                    grant->intents_[first].destination_grant!=grant->intents_[i].destination_grant))
                {
                    rejectGrant("Subordinate duplicate rows carry conflicting promotion grants");
                }
                plans[i]={first,None};
            }
            else {
                const auto resident=s.find(row);plans[i]={static_cast<std::uint32_t>(i),resident};
                requests[at]={row,stamp,static_cast<std::uint32_t>(i)};
                if(resident==None)targets[misses++]={ids[i],i};
            }
            if(grant && grant->intents_[i].intent==ImmutableRowReadIntent::ReadAndInvalidate) {
                const auto token=grant->intents_[i].destination_grant;
                auto target=hash(token)&(g.request_count_-1);
                while(grants[target].stamp==stamp && grants[target].token!=token)
                    target=(target+1)&(g.request_count_-1);
                if(grants[target].stamp==stamp && grants[target].row!=row)
                    rejectGrant("Subordinate destination grant names two different source rows");
                grants[target]={token,row,stamp};
            }
        }
        try {
        // The existing source owns one metadata-authenticated transaction, even
        // for zero misses. Response holes are not copied or counted as I/O.
        const auto observed=s.source->gatherScattered({targets,misses},destination);
        s.counters.source_rows=countPlus(s.counters.source_rows,misses);
        s.counters.source_window_hits=countPlus(s.counters.source_window_hits,observed.windowHits);
        s.counters.source_window_misses=countPlus(s.counters.source_window_misses,observed.windowMisses);
        s.counters.source_window_evictions=countPlus(s.counters.source_window_evictions,observed.windowEvictions);
        // Copy ALL initial hits before an earlier logical admission can evict
        // a later hit's backing. No extra native payload scratch is necessary.
        for(std::size_t i=0;i<ids.size();++i) {
            if(plans[i].first==i && plans[i].initial_slot!=None) {
                std::memcpy(destination.data()+i*g.extent_.rowBytes(),s.data(plans[i].initial_slot),g.extent_.rowBytes());
                ++s.counters.ram_rows;
            }
        }
        for(std::size_t i=0;i<ids.size();++i) {
            const auto row=static_cast<std::uint64_t>(ids[i]);auto *out=destination.data()+i*g.extent_.rowBytes();
            if(plans[i].first!=i) {
                std::memcpy(out,destination.data()+plans[i].first*g.extent_.rowBytes(),g.extent_.rowBytes());
                ++s.counters.duplicate_rows;
            }
            ++s.clock;
            if(grant && grant->intents_[i].intent==ImmutableRowReadIntent::ReadAndInvalidate) {
                if(plans[i].first==i)s.invalidate(row);
            } else {
                s.frequency(row,true);const auto resident=s.find(row);
                if(resident!=None)s.touch(resident);else s.admit(row,out);
            }
        }
        // ACKs are written only after every fill, duplicate and invalidation succeeds.
        if(grant) {
            auto *rows=reinterpret_cast<GrantedRow*>(static_cast<std::byte*>(s.execution.data)+g.read_rows_offset_);
            for(std::size_t i=0;i<ids.size();++i) {
                rows[i]={ids[i],grant->intents_[i].intent};
                grant->acknowledgements_[i].destination_grant=grant->intents_[i].destination_grant;
            }
        }
        ++s.next_policy_ordinal;++s.counters.completed_batches;s.counters.completed_rows+=ids.size();s.counters.completed_bytes+=destination.size();
        } catch(...) {s.failed=true;++s.counters.failures;throw;}
    }
    const auto &a=s.counters;
    return {ids.size(),destination.size(),a.ram_rows-before.ram_rows,a.source_rows-before.source_rows,
        a.duplicate_rows-before.duplicate_rows,a.admissions-before.admissions,a.evictions-before.evictions,
        a.admission_rejections-before.admission_rejections,a.all_pinned_responses-before.all_pinned_responses,
        {a.source_window_hits-before.source_window_hits,a.source_window_misses-before.source_window_misses,
         a.source_window_evictions-before.source_window_evictions},s.next_policy_ordinal-1,a.invalidations-before.invalidations};
}
std::optional<CpuAdaptiveImmutableRowCache::Pin> CpuAdaptiveImmutableRowCache::tryPinResident(std::int64_t row) {
    return pinImpl(row,nullptr);
}
std::optional<CpuAdaptiveImmutableRowCache::Pin> CpuAdaptiveImmutableRowCache::pinSubordinate(
    const ReadGrant &grant,std::size_t response_row) {
    if(response_row>=grant.row_count_)throw std::out_of_range("Subordinate source pin names a foreign response row");
    return pinImpl(static_cast<std::int64_t>(response_row),&grant);
}
std::optional<CpuAdaptiveImmutableRowCache::Pin> CpuAdaptiveImmutableRowCache::pinImpl(std::int64_t row,const ReadGrant *grant) {
    auto &s=*impl_;std::lock_guard lock(s.mutex);s.requireOpen();
    if(static_cast<bool>(s.geometry.requester_)!=static_cast<bool>(grant) ||
        (grant && (grant->owner_.get()!=impl_.get() || grant->channel_!=s.subordinate_channel)))
        throw std::invalid_argument("Subordinate CPU source forbids independent or foreign pins");
    if(grant) {
        const auto *rows=reinterpret_cast<const GrantedRow*>(static_cast<const std::byte*>(s.execution.data)+s.geometry.read_rows_offset_);
        const auto &record=rows[static_cast<std::size_t>(row)];
        if(record.intent!=ImmutableRowReadIntent::Read)
            throw std::invalid_argument("Invalidated promotion rows cannot publish new RAM read pins");
        row=record.row;
    }
    if(row<0 || static_cast<std::uint64_t>(row)>=s.geometry.extent_.rows())throw std::out_of_range("CPU row cache pin ID invalid");
    // A transient source-path failure is still a failed immutable owner. A
    // later path restoration must not reopen this same residency generation.
    try {s.source->authenticateSource();}
    catch(...) {s.failed=true;throw;}
    const auto slot=s.find(static_cast<std::uint64_t>(row));if(slot==None)return std::nullopt;
    auto &record=s.slots()[slot];if(record.pins==std::numeric_limits<std::uint32_t>::max())throw std::overflow_error("CPU row cache pin count exhausted");
    if(!record.pins){s.ineligible(slot);++s.counters.pinned_rows;}++record.pins;
    return Pin(impl_,slot,record.row,record.generation);
}
std::size_t CpuAdaptiveImmutableRowCache::maximumLiveRows() const noexcept {return impl_->geometry.maximum_live_rows_;}
void CpuAdaptiveImmutableRowCache::retire() noexcept {if(impl_){std::lock_guard lock(impl_->mutex);impl_->retired=true;}}
HostImmutableRowOwnerObservation CpuAdaptiveImmutableRowCache::retainedOwnerObservation() const
{
    auto &s = *impl_;
    std::lock_guard lock(s.mutex);
    s.requireOpen();
    try
    {
        const auto source = s.source->retainedOwnerObservation();
        if (source.authority().get() != s.memory.get() ||
            source.allocator() != s.geometry.host_ || source.hostPlacement() != s.geometry.locality_ ||
            source.lookupStagingBytes() ||
            (s.storage.lease.valid() && s.storage.lease.owner() != PhysicalMemoryOwner::ImmutableTableStorage) ||
            !s.execution.lease.valid() || s.execution.lease.owner() != PhysicalMemoryOwner::ImmutableLookupStaging)
            throw std::logic_error("CPU row cache has foreign or missing live PMA owner leases");
        // A zero-slot pool has no storage lease. The real SSD lease remains
        // present, and mandatory request arrays retain their own staging lease.
        return {source.authority(), source.allocator(), source.hostPlacement(),
            plus(source.tableStorageBytes(), s.storage.lease.bytes()), s.execution.lease.bytes()};
    }
    catch (...)
    {
        s.failed = true;
        throw;
    }
}
CpuImmutableRowCacheStats CpuAdaptiveImmutableRowCache::stats() const {std::lock_guard lock(impl_->mutex);return impl_->counters;}
ImmutableRowTableStats CpuAdaptiveImmutableRowCache::sourceStats() const {std::lock_guard lock(impl_->mutex);return impl_->source->stats();}
const ImmutableRowTableSource &CpuAdaptiveImmutableRowCache::source() const noexcept {return impl_->source->source();}
const ImmutableRowFileIdentity &CpuAdaptiveImmutableRowCache::identity() const noexcept {return impl_->source->identity();}
CpuAdaptiveImmutableRowCache::Pin::~Pin(){release();}
CpuAdaptiveImmutableRowCache::Pin::Pin(Pin &&other) noexcept
    :owner_(std::move(other.owner_)),slot_(other.slot_),row_(other.row_),generation_(other.generation_){}
CpuAdaptiveImmutableRowCache::Pin &CpuAdaptiveImmutableRowCache::Pin::operator=(Pin &&other) noexcept {
    if(this!=&other){release();owner_=std::move(other.owner_);slot_=other.slot_;row_=other.row_;generation_=other.generation_;}return *this;
}
void CpuAdaptiveImmutableRowCache::Pin::release() noexcept {
    if(!owner_)return;
    auto state=std::move(owner_);std::lock_guard lock(state->mutex);auto &s=state->slots()[slot_];
    if(!s.ready || s.row!=row_ || s.generation!=generation_ || !s.pins)std::terminate();
    if(!--s.pins){
        --state->counters.pinned_rows;
        if(s.retiring){try{state->recycle(slot_);}catch(...){std::terminate();}}
        else state->eligible(slot_);
    }
}
/**
 * @brief Read a live pinned generation after ordinary retirement, never after source failure.
 * @return Exact original native row retained by this generation-matched lease.
 * @throws std::invalid_argument For a moved-from pin.
 * @throws std::runtime_error For an already failed owner or stale source.
 * @throws std::exception For source metadata I/O/observation failure, which poisons this shared owner.
 * @throws std::logic_error For an invalidated internal slot generation.
 */
std::span<const std::byte> CpuAdaptiveImmutableRowCache::Pin::bytes() const {
    if(!owner_)throw std::invalid_argument("Moved-from CPU row cache pin");
    std::lock_guard lock(owner_->mutex);
    // Do not use requireOpen(): normal retirement deliberately preserves this
    // existing grant. Failure is different and poisons every retained grant.
    if(owner_->failed)throw std::runtime_error("CPU row cache is failed");
    try {owner_->source->authenticateSource();}
    catch(...) {owner_->failed=true;throw;}
    const auto &s=owner_->slots()[slot_];
    if(!s.ready || !s.pins || s.row!=row_ || s.generation!=generation_)throw std::logic_error("Stale CPU row cache grant");
    return {owner_->data(slot_),owner_->geometry.extent_.rowBytes()};
}
} // namespace llaminar2
