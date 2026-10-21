/**
 * @file Test__CpuAdaptiveImmutableRowCache.cpp
 * @brief Native-file exactness, bounded progress, pin retirement and PMA tests without models/devices.
 *
 * Small original-format files exercise the actual SSD service, same source
 * identity, actual physical authority and complete cache implementation. No
 * backend startup, GPU, model source, prepared weights or alternate ledger.
 */
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <barrier>
#include <exception>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <random>
#include <thread>
#include <tuple>
#include <unistd.h>
namespace llaminar2::test {
namespace {
using Cache=CpuAdaptiveImmutableRowCache;
using Geometry=CpuAdaptiveImmutableRowCacheGeometry;
using Storage=ImmutableRowStorage;
using Id=PhysicalMemoryAllocatorIdentity;
const Id Host{0,DeviceId::cpu()},GPU{0,DeviceId::cuda(0)};
/** @brief Small distinguishable original-native file with a deliberately unaligned tensor offset. */
struct File {
    int fd=-1;std::string path;std::vector<std::byte> original;ImmutableRowTableSource source;
    /** @brief Write only fixture bytes, keeping arbitrary native payload bits verbatim. */
    File(GGUFTensorType type=GGUFTensorType::IQ4_NL,std::size_t elements=160,std::size_t rows=128) {
        char name[]="/tmp/llaminar-adaptive-row-XXXXXX";fd=::mkstemp(name);
        if(fd<0)throw std::runtime_error("Cannot create private native row file");
        path=name;
        source={.path=path,.tensor_name="fixture.ple",.file_offset=37,.rows=rows,.row_elements=elements,.format=type};
        original.resize(37+rows*source.rowBytes());
        for(std::size_t i=0;i<original.size();++i)original[i]=std::byte((i*13+i/31)%251);
        if(::write(fd,original.data(),original.size())!=static_cast<ssize_t>(original.size()))throw std::runtime_error("Cannot populate native fixture");
    }
    ~File(){if(fd>=0)::close(fd);::unlink(path.c_str());}
};
/** @return Exact canonical host observation; no capacity subtraction occurs in the test. */
PhysicalMemoryResource host(){return {0,DeviceId::cpu(),1ull<<34,1ull<<34};}
/** @return Explicit source-owned layout, permitting hosted GPU consumers without a GPU cache domain. */
ImmutableRowTableLayoutGeometry layout(const File &f,std::size_t slots,bool hosted=false,
    std::optional<std::size_t> limit=std::nullopt,std::uint64_t maximum_inflight_rows=0) {
    const std::array tiers{Storage::VRAM,Storage::RAM,Storage::MemoryMapped};
    const std::vector<ImmutableRowTableLayoutOwner> owners{{{Host,Storage::RAM},0,slots,limit},{{GPU,Storage::VRAM},0,0,0}};
    return ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(f.source.format,f.source.rows,f.source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,{hosted?GPU:Host},{},
        {{Host,ImmutableRowTableStoragePolicy::forStorage(Storage::MemoryMapped,4096,2)}},{},std::nullopt,maximum_inflight_rows);
}
/** @return One topology certificate and sole rank materialization authority. */
std::shared_ptr<PhysicalMemoryAuthority> authority(const Geometry &g) {
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(g.bom(host())),0);
}
/** @brief Compare original row bytes in exact caller order, including duplicates/cross-window rows. */
void expectRows(const File &f,std::span<const std::int64_t> ids,std::span<const std::byte> bytes) {
    ASSERT_EQ(bytes.size(),ids.size()*f.source.rowBytes());
    for(std::size_t i=0;i<ids.size();++i)EXPECT_EQ(std::memcmp(bytes.data()+i*f.source.rowBytes(),
        f.original.data()+f.source.file_offset+ids[i]*f.source.rowBytes(),f.source.rowBytes()),0);
}
/** @return Actual claimed owner bytes, never a duplicated test balance. */
std::size_t claimed(const std::shared_ptr<PhysicalMemoryAuthority> &p,PhysicalMemoryOwner o) {
    return p->claimedBytes(DeviceId::cpu(),o,PhysicalMemoryMaterializationKind::NewAllocation);
}
/** @return Complete observable cache work/occupancy, excluding private mutable state. */
auto cacheWork(const CpuImmutableRowCacheStats &s) {
    return std::tuple{s.completed_batches,s.completed_rows,s.completed_bytes,s.failures,
        s.ram_rows,s.source_rows,s.duplicate_rows,s.admissions,s.evictions,s.admission_rejections,
        s.all_pinned_responses,s.source_window_hits,s.source_window_misses,s.source_window_evictions,
        s.resident_rows,s.pinned_rows};
}
/** @return Source work/extent excluding the intentional completed-authentication counter. */
auto sourceWork(const ImmutableRowTableStats &s) {
    return std::tuple{s.storage,s.table_file_bytes,s.physical_budget_bytes,s.mapped_virtual_bytes,
        s.requests,s.rows,s.copied_bytes,s.window_hits,s.window_misses,s.window_evictions};
}
}
TEST(CpuAdaptiveImmutableRowCache, ReadyReuseAuthenticationDoesNotGatherOrChangeWarmPolicy) {
    File f;const auto g=Geometry::fromLayout(layout(f,2,true),Host,2);
    auto p=authority(g);auto c=Cache::create(f.source,g,p);
    auto control=Cache::create(f.source,g,authority(g));
    const auto cold=c->stats();const auto cold_source=c->sourceStats();
    const auto storage=claimed(p,PhysicalMemoryOwner::ImmutableTableStorage);
    const auto execution=claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging);
    c->authenticateSource();
    EXPECT_EQ(cacheWork(c->stats()),cacheWork(cold));
    EXPECT_EQ(sourceWork(c->sourceStats()),sourceWork(cold_source));
    EXPECT_EQ(c->sourceStats().source_metadata_checks,cold_source.source_metadata_checks+1);
    const std::array<std::int64_t,2> seed{0,1};std::array<std::byte,180> out{},expected{};
    EXPECT_EQ(c->gather(seed,out).policy_ordinal,0u);
    EXPECT_EQ(control->gather(seed,expected).policy_ordinal,0u);EXPECT_EQ(out,expected);
    const auto warm=c->stats();const auto warm_source=c->sourceStats();const auto identity=c->identity();
    for(int repeat=0;repeat<3;++repeat)c->authenticateSource();
    EXPECT_EQ(cacheWork(c->stats()),cacheWork(warm));
    EXPECT_EQ(sourceWork(c->sourceStats()),sourceWork(warm_source));
    EXPECT_EQ(c->sourceStats().source_metadata_checks,warm_source.source_metadata_checks+3);
    EXPECT_EQ(c->identity(),identity);EXPECT_EQ(c->source(),f.source);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),storage);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),execution);
    const std::array<std::int64_t,7> churn{2,3,0,1,2,3,0};
    for(std::size_t step=0;step<churn.size();++step) {
        const auto ids=std::span(churn).subspan(step,1);
        const auto actual=c->gather(ids,std::span(out).first(90));
        const auto baseline=control->gather(ids,std::span(expected).first(90));
        EXPECT_EQ(actual.policy_ordinal,step+1);EXPECT_EQ(actual.policy_ordinal,baseline.policy_ordinal);
        EXPECT_EQ(out,expected);EXPECT_EQ(cacheWork(c->stats()),cacheWork(control->stats()));
        for(std::int64_t row=0;row<4;++row) {
            auto a=c->tryPinResident(row),b=control->tryPinResident(row);
            ASSERT_EQ(a.has_value(),b.has_value());
            if(a && b) {EXPECT_EQ(a->generation(),b->generation());EXPECT_EQ(a->row(),b->row());}
        }
    }
}
TEST(CpuAdaptiveImmutableRowCache, RetiredReuseAuthenticationRejectsAndKeepsOutstandingPinLifetime) {
    File f;const auto g=Geometry::fromLayout(layout(f,1),Host,1);auto p=authority(g);auto c=Cache::create(f.source,g,p);
    const std::array<std::int64_t,1> ids{7};std::array<std::byte,90> out{};c->gather(ids,out);
    auto pin=c->tryPinResident(7);ASSERT_TRUE(pin);const auto generation=pin->generation();
    c->retire();const auto work=c->stats();const auto source=c->sourceStats();
    EXPECT_THROW(c->authenticateSource(),std::runtime_error);
    EXPECT_EQ(cacheWork(c->stats()),cacheWork(work));
    EXPECT_EQ(sourceWork(c->sourceStats()),sourceWork(source));
    EXPECT_EQ(c->sourceStats().source_metadata_checks,source.source_metadata_checks);
    EXPECT_EQ(pin->generation(),generation);expectRows(f,ids,pin->bytes());
    c.reset();EXPECT_GT(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
    expectRows(f,ids,pin->bytes());pin.reset();
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),0u);
}
TEST(CpuAdaptiveImmutableRowCache, StaleReuseAuthenticationPoisonsWithoutClaimingGatherWork) {
    File f;const auto g=Geometry::fromLayout(layout(f,1),Host,1);auto c=Cache::create(f.source,g,authority(g));
    const std::array<std::int64_t,1> ids{0};std::array<std::byte,90> out{};c->gather(ids,out);
    auto pin=c->tryPinResident(0);ASSERT_TRUE(pin);const auto generation=pin->generation();
    const auto work=c->stats();const auto source=c->sourceStats();
    ASSERT_EQ(::ftruncate(f.fd,f.original.size()+1),0);
    EXPECT_THROW(c->authenticateSource(),std::runtime_error);
    EXPECT_EQ(cacheWork(c->stats()),cacheWork(work));
    EXPECT_EQ(sourceWork(c->sourceStats()),sourceWork(source));
    EXPECT_EQ(c->sourceStats().source_metadata_checks,source.source_metadata_checks+1);
    const auto after=c->sourceStats();out.fill(std::byte{0x79});const auto untouched=out;
    EXPECT_THROW(c->authenticateSource(),std::runtime_error);
    EXPECT_THROW(c->gather(ids,out),std::runtime_error);EXPECT_EQ(out,untouched);
    EXPECT_THROW(c->tryPinResident(0),std::runtime_error);
    EXPECT_EQ(cacheWork(c->stats()),cacheWork(work));
    EXPECT_EQ(c->sourceStats().source_metadata_checks,after.source_metadata_checks);
    EXPECT_EQ(pin->generation(),generation);EXPECT_THROW(pin->bytes(),std::runtime_error);
}
TEST(CpuAdaptiveImmutableRowCache, FailedGatherCannotBeReauthenticatedForReuse) {
    File f;const auto g=Geometry::fromLayout(layout(f,0),Host,1);auto c=Cache::create(f.source,g,authority(g));
    const std::array<std::int64_t,1> ids{0};std::array<std::byte,90> out{};
    ASSERT_EQ(::ftruncate(f.fd,f.original.size()+1),0);
    EXPECT_THROW(c->gather(ids,out),std::runtime_error);EXPECT_EQ(c->stats().failures,1u);
    const auto work=c->stats();const auto source=c->sourceStats();
    EXPECT_THROW(c->retainedOwnerObservation(),std::runtime_error);
    EXPECT_THROW(c->authenticateSource(),std::runtime_error);
    EXPECT_EQ(cacheWork(c->stats()),cacheWork(work));
    EXPECT_EQ(c->sourceStats().source_metadata_checks,source.source_metadata_checks);
}
TEST(CpuAdaptiveImmutableRowCache, SourceScatterPreservesHolesAndCountsOnlySelectedNativeRows) {
    File f;
    for(auto storage:{Storage::RAM,Storage::MemoryMapped}) {
        const auto policy=ImmutableRowTableStoragePolicy::forStorage(storage,4096,2);
        PhysicalMemoryPlanBuilder b;b.add(host(),PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(f.source,policy));
        auto p=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(b.build()),0);
        ImmutableRowTable source(f.source,policy,DeviceId::cpu(),p,ImmutableRowTableHostPlacement::taskPolicy());
        std::array<std::byte,720> out;out.fill(std::byte{0x63});
        const std::array<ImmutableRowTableScatterRow,3> rows{{{45,1},{3,4},{45,7}}};
        const auto before=source.stats();source.gatherScattered(rows,out);const auto after=source.stats();
        EXPECT_EQ(after.source_metadata_checks-before.source_metadata_checks,2u);
        EXPECT_EQ(after.rows-before.rows,3u);EXPECT_EQ(after.copied_bytes-before.copied_bytes,270u);
        for(std::size_t i=0;i<8;++i) {
            if(i==1 || i==4 || i==7) {
                const auto id=i==4?3:45;
                EXPECT_EQ(std::memcmp(out.data()+i*90,f.original.data()+37+id*90,90),0);
            } else for(auto byte:std::span(out).subspan(i*90,90))EXPECT_EQ(byte,std::byte{0x63});
        }
        const auto saved=out;const auto previous=source.stats();source.gatherScattered({},out);
        EXPECT_EQ(out,saved);EXPECT_EQ(source.stats().source_metadata_checks-previous.source_metadata_checks,2u);
        EXPECT_EQ(source.stats().copied_bytes,after.copied_bytes);
        const std::array<ImmutableRowTableScatterRow,2> overlap{{{0,2},{1,2}}},reverse{{{0,4},{1,1}}},bad{{{0,0},{128,1}}};
        EXPECT_THROW(source.gatherScattered(overlap,out),std::invalid_argument);
        EXPECT_THROW(source.gatherScattered(reverse,out),std::invalid_argument);
        EXPECT_THROW(source.gatherScattered(bad,out),std::out_of_range);
        EXPECT_EQ(out,saved);
        alignas(ImmutableRowTableScatterRow) std::array<std::byte,180> aliased{};
        auto *target=std::construct_at(reinterpret_cast<ImmutableRowTableScatterRow*>(aliased.data()),ImmutableRowTableScatterRow{0,0});
        EXPECT_THROW(source.gatherScattered({target,1},aliased),std::invalid_argument);
    }
}
TEST(CpuAdaptiveImmutableRowCache, OneSourceTransactionAuthenticatesBeforeAllResponsesRegardlessOfMissCount) {
    File f(GGUFTensorType::IQ4_NL,160,8192);const auto g=Geometry::fromLayout(layout(f,0),Host,8192);
    auto c=Cache::create(f.source,g,authority(g));std::vector<std::int64_t> ids(8192);std::vector<std::byte> out(8192*90);
    for(std::size_t i=0;i<ids.size();++i)ids[i]=i;
    const auto before=c->sourceStats();const auto observed=c->gather(ids,out);expectRows(f,ids,out);const auto after=c->sourceStats();
    EXPECT_EQ(observed.source_rows,8192u);EXPECT_EQ(after.requests-before.requests,1u);
    EXPECT_EQ(after.source_metadata_checks-before.source_metadata_checks,2u);
    EXPECT_EQ(after.copied_bytes-before.copied_bytes,out.size());
    const auto preempty=c->sourceStats();c->gather({},{});
    EXPECT_EQ(c->sourceStats().source_metadata_checks-preempty.source_metadata_checks,2u);
}
TEST(CpuAdaptiveImmutableRowCache, MissDuplicatesAndLaterInitialHitSurviveEarlierAdmission) {
    File f;const auto g=Geometry::fromLayout(layout(f,1,true),Host,4);auto c=Cache::create(f.source,g,authority(g));
    std::array<std::byte,270> out{};const std::array<std::int64_t,1> seed{0};c->gather(seed,std::span(out).first(90));
    const std::array<std::int64_t,3> ids{9,9,0};const auto before=c->sourceStats();
    const auto observed=c->gather(ids,out);expectRows(f,ids,out);
    EXPECT_EQ(observed.ram_rows,1u);EXPECT_EQ(observed.source_rows,1u);EXPECT_EQ(observed.duplicate_rows,1u);
    EXPECT_EQ(c->sourceStats().source_metadata_checks-before.source_metadata_checks,2u);
    EXPECT_GE(observed.evictions,1u);
}
TEST(CpuAdaptiveImmutableRowCache, ConcurrentClientsReceiveOnlyOwnerMintedLinearizedPolicyOrdinals) {
    File f;const auto g=Geometry::fromLayout(layout(f,4,true),Host,2);auto c=Cache::create(f.source,g,authority(g));
    constexpr std::size_t Clients=4,Batches=16;
    std::barrier start(static_cast<std::ptrdiff_t>(Clients));
    std::array<std::array<std::uint64_t,Batches>,Clients> ordinals{};
    std::array<std::exception_ptr,Clients> failures{};std::array<bool,Clients> exact{};
    std::vector<std::thread> workers;
    for(std::size_t client=0;client<Clients;++client)workers.emplace_back([&,client] {
        start.arrive_and_wait();
        try {
            exact[client]=true;const std::array<std::int64_t,2> ids{static_cast<std::int64_t>(client),static_cast<std::int64_t>(client+8)};
            std::array<std::byte,180> out{};
            for(std::size_t batch=0;batch<Batches;++batch) {
                const auto observation=c->gather(ids,out);ordinals[client][batch]=observation.policy_ordinal;
                for(std::size_t row=0;row<ids.size();++row)
                    if(std::memcmp(out.data()+row*90,f.original.data()+37+ids[row]*90,90)!=0)exact[client]=false;
            }
        } catch(...) {failures[client]=std::current_exception();}
    });
    for(auto &worker:workers)worker.join();
    std::array<std::uint64_t,Clients*Batches> all{};
    for(std::size_t client=0;client<Clients;++client) {
        ASSERT_FALSE(failures[client]);EXPECT_TRUE(exact[client]);
        for(std::size_t batch=0;batch<Batches;++batch) {
            all[client*Batches+batch]=ordinals[client][batch];
            if(batch) {EXPECT_LT(ordinals[client][batch-1],ordinals[client][batch]);}
        }
    }
    std::sort(all.begin(),all.end());for(std::size_t i=0;i<all.size();++i)EXPECT_EQ(all[i],i);
    EXPECT_EQ(c->gather({},{}).policy_ordinal,Clients*Batches);
    EXPECT_EQ(c->stats().completed_batches,Clients*Batches+1);
}
TEST(CpuAdaptiveImmutableRowCache, ExactBOMPhasesAndActualMaterializationRetireOnce) {
    File f;const auto g=Geometry::fromLayout(layout(f,7),Host,32);
    const auto fixed=g.mandatoryBOM(host()),optional=g.cacheBOM(host()),all=g.bom(host());
    ASSERT_EQ(fixed.resources().size(),1u);ASSERT_EQ(optional.resources().size(),1u);
    EXPECT_EQ(fixed.resources()[0].bytes(PhysicalMemoryOwner::ImmutableTableStorage),g.sourceBytes());
    EXPECT_EQ(fixed.resources()[0].bytes(PhysicalMemoryOwner::ImmutableLookupStaging),g.executionBytes());
    EXPECT_EQ(optional.resources()[0].bytes(PhysicalMemoryOwner::ImmutableTableStorage),g.storageBytes());
    EXPECT_EQ(all.totalBytes(),fixed.totalBytes()+optional.totalBytes());
    EXPECT_GT(g.storageBytes(),7*90u);auto p=authority(g);
    {auto c=Cache::create(f.source,g,p);EXPECT_TRUE(p->rankComplete());
     EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),g.sourceBytes()+g.storageBytes());
     EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),g.executionBytes());}
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),0u);
}
TEST(CpuAdaptiveImmutableRowCache, EveryOriginalNativeFormatAndCrossWindowRowPreservesBytes) {
    constexpr std::array formats{GGUFTensorType::F32,GGUFTensorType::F16,GGUFTensorType::BF16,
        GGUFTensorType::Q4_0,GGUFTensorType::Q4_1,GGUFTensorType::Q5_0,GGUFTensorType::Q5_1,GGUFTensorType::Q8_0,
        GGUFTensorType::Q2_K,GGUFTensorType::Q3_K,GGUFTensorType::Q4_K,GGUFTensorType::Q5_K,GGUFTensorType::Q6_K,
        GGUFTensorType::Q8_K,GGUFTensorType::IQ1_S,GGUFTensorType::IQ1_M,GGUFTensorType::IQ2_XXS,
        GGUFTensorType::IQ2_XS,GGUFTensorType::IQ2_S,GGUFTensorType::IQ3_XXS,GGUFTensorType::IQ3_S,
        GGUFTensorType::IQ4_NL,GGUFTensorType::IQ4_XS};
    for(auto type:formats) {
        SCOPED_TRACE(static_cast<int>(type));File f(type,4096,16);const auto g=Geometry::fromLayout(layout(f,3),Host,8);
        auto c=Cache::create(f.source,g,authority(g));const std::array<std::int64_t,8> ids{15,0,4,15,4,7,15,0};
        std::vector<std::byte> out(ids.size()*f.source.rowBytes());
        for(std::uint64_t i=0;i<3;++i){const auto observed=c->gather(ids,out);expectRows(f,ids,out);EXPECT_EQ(observed.rows,8u);}
    }
}
TEST(CpuAdaptiveImmutableRowCache, ZeroCacheDeduplicatesIntoCallerResponseWithoutNativePayload) {
    File f;const auto g=Geometry::fromLayout(layout(f,0,true),Host,8);auto p=authority(g);
    EXPECT_EQ(g.storageBytes(),0u);auto c=Cache::create(f.source,g,p);
    const std::array<std::int64_t,5> ids{5,2,5,5,2};std::array<std::byte,450> out{};
    const auto observed=c->gather(ids,out);expectRows(f,ids,out);
    EXPECT_EQ(observed.source_rows,2u);EXPECT_EQ(observed.duplicate_rows,3u);EXPECT_EQ(observed.admissions,0u);
    EXPECT_EQ(observed.windows.windowHits+observed.windows.windowMisses,2u);EXPECT_EQ(c->stats().resident_rows,0u);
    EXPECT_FALSE(c->tryPinResident(2));EXPECT_TRUE(p->rankComplete());
}
TEST(CpuAdaptiveImmutableRowCache, AllPinnedCompletesLargerRequestAndKeepsExactGenerations) {
    File f;const auto g=Geometry::fromLayout(layout(f,2),Host,8);auto c=Cache::create(f.source,g,authority(g));
    std::array<std::byte,360> out{};const std::array<std::int64_t,2> seed{0,1};c->gather(seed,std::span(out).first(180));
    auto a=c->tryPinResident(0),b=c->tryPinResident(1);ASSERT_TRUE(a);ASSERT_TRUE(b);
    const auto generation=a->generation();const std::array<std::int64_t,4> ids{2,2,3,2};
    const auto observed=c->gather(ids,out);expectRows(f,ids,out);EXPECT_EQ(observed.source_rows,2u);
    EXPECT_EQ(observed.all_pinned_responses,4u);EXPECT_EQ(observed.evictions,0u);EXPECT_EQ(c->stats().pinned_rows,2u);
    EXPECT_EQ(a->generation(),generation);expectRows(f,std::span(seed).first(1),a->bytes());
}
TEST(CpuAdaptiveImmutableRowCache, PinsOutliveFacadeAndRetainSourceAndBothPhysicalOwners) {
    File f;const auto g=Geometry::fromLayout(layout(f,1),Host,1);auto p=authority(g);auto c=Cache::create(f.source,g,p);
    const std::array<std::int64_t,1> ids{7};std::array<std::byte,90> out{};c->gather(ids,out);
    auto pin=c->tryPinResident(7);ASSERT_TRUE(pin);auto moved=std::move(*pin);
    EXPECT_THROW(pin->bytes(),std::invalid_argument);
    c->retire();
    EXPECT_THROW(c->gather(ids,out),std::runtime_error);
    c.reset();EXPECT_GT(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);expectRows(f,ids,moved.bytes());
    {auto final=std::move(moved);EXPECT_THROW(moved.bytes(),std::invalid_argument);expectRows(f,ids,final.bytes());}
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),0u);
}
TEST(CpuAdaptiveImmutableRowCache, BadFramesAndCallerAliasesCannotPartiallyPublishOrConsumeOwnerOrdinal) {
    File f;const auto g=Geometry::fromLayout(layout(f,2),Host,2);auto c=Cache::create(f.source,g,authority(g));
    std::array<std::byte,180> out;out.fill(std::byte{0x5a});const auto before=out;
    const std::array<std::int64_t,2> bad{0,128},good{0,1};
    EXPECT_THROW(c->gather(bad,out),std::out_of_range);EXPECT_EQ(out,before);
    EXPECT_THROW(c->gather(good,std::span(out).first(90)),std::invalid_argument);EXPECT_EQ(out,before);
    EXPECT_EQ(c->stats().completed_batches,0u);EXPECT_EQ(c->gather(good,out).policy_ordinal,0u);expectRows(f,good,out);
    auto pin=c->tryPinResident(0);ASSERT_TRUE(pin);
    EXPECT_THROW(c->gather(std::span(good).first(1),{const_cast<std::byte*>(pin->bytes().data()),90}),std::invalid_argument);
    EXPECT_EQ(c->stats().completed_batches,1u);EXPECT_EQ(c->gather({},{}).policy_ordinal,1u);EXPECT_EQ(c->stats().completed_batches,2u);
}
TEST(CpuAdaptiveImmutableRowCache, WarmHitsAndOutstandingPinsAuthenticateOriginalFileIdentity) {
    File f;const auto g=Geometry::fromLayout(layout(f,2),Host,1);auto c=Cache::create(f.source,g,authority(g));
    const std::array<std::int64_t,1> ids{0};std::array<std::byte,90> out{};c->gather(ids,out);
    auto pin=c->tryPinResident(0);ASSERT_TRUE(pin);ASSERT_EQ(::ftruncate(f.fd,f.original.size()+1),0);
    out.fill(std::byte{0x6b});const auto before=out;
    EXPECT_THROW(c->gather(ids,out),std::runtime_error);EXPECT_EQ(out,before);
    EXPECT_THROW(pin->bytes(),std::runtime_error);EXPECT_EQ(c->stats().completed_batches,1u);EXPECT_EQ(c->stats().failures,1u);
    EXPECT_THROW(c->gather(ids,out),std::runtime_error);
}
TEST(CpuAdaptiveImmutableRowCache, NativeMetadataBudgetAndHostDomainCannotBeBypassed) {
    File f;
    EXPECT_THROW(Geometry::fromLayout(layout(f,2,false,180),Host,4),std::invalid_argument);
    const std::array tier{Storage::VRAM,Storage::RAM,Storage::MemoryMapped};
    const auto gpuDomain=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(f.source.format,128,90),
        ImmutableRowTableTiers::of(tier),ImmutableRowTableResidency::Adaptive,
        {{{Host,Storage::RAM},0,2,{}},{{GPU,Storage::VRAM},0,2,{}}},{GPU},{},
        {{Host,ImmutableRowTableStoragePolicy::forStorage(Storage::MemoryMapped,4096,2)}},{},GPU,4);
    EXPECT_THROW(Geometry::fromLayout(gpuDomain,Host,4),std::invalid_argument);
    EXPECT_THROW(Geometry::fromLayout(layout(f,2),Host,0),std::invalid_argument);
    const auto g=Geometry::fromLayout(layout(f,2),Host,4);auto wrong=host();wrong.world_rank=1;
    EXPECT_THROW(g.bom(wrong),std::invalid_argument);
    auto p=authority(g);auto invalid=f.source;invalid.row_elements=32;
    EXPECT_THROW(Cache::create(invalid,g,p),std::invalid_argument);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
}
/** @test The retained host request envelope, not a second caller capacity, prices execution scratch. */
TEST(CpuAdaptiveImmutableRowCache, DeclaredRequestEnvelopeCannotBeRepriced) {
    File f;
    for(const auto slots:{0u,2u,128u}) for(const bool hosted:{false,true}) {
        const auto declared=layout(f,slots,hosted,std::nullopt,4);
        const auto geometry=Geometry::fromLayout(declared,Host,4);
        auto owner=Cache::create(f.source,geometry,authority(geometry));
        EXPECT_EQ(owner->maximumLiveRows(),4u);
        EXPECT_EQ(geometry.slots(),slots);
        EXPECT_EQ(declared.gpuDirectoryRecordBound(),0u);
        for(const auto other:{1u,3u,5u}) {
            EXPECT_THROW(Geometry::fromLayout(declared,Host,other),std::invalid_argument);
        }
    }
}
TEST(CpuAdaptiveImmutableRowCache, PmaRejectsMissingCacheOrExecutionBeforeAnyOwnerEscapes) {
    File f;const auto g=Geometry::fromLayout(layout(f,3),Host,4);
    for(const bool omit_execution:{false,true}) {
        PhysicalMemoryPlanBuilder b;
        b.add(host(),PhysicalMemoryOwner::ImmutableTableStorage,g.sourceBytes()+(omit_execution?g.storageBytes():0));
        b.add(host(),PhysicalMemoryOwner::ImmutableLookupStaging,omit_execution?0:g.executionBytes());
        auto p=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(b.build()),0);
        EXPECT_ANY_THROW(Cache::create(f.source,g,p));
        EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
        EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),0u);
    }
}
TEST(CpuAdaptiveImmutableRowCache, ChurnMatchesOriginalBytesAndNeverGrowsPhysicalClaims) {
    File f;const auto g=Geometry::fromLayout(layout(f,7),Host,32,{.aging_accesses=16});auto p=authority(g);
    auto c=Cache::create(f.source,g,p);std::mt19937 rng(931);std::array<std::int64_t,32> ids{};std::array<std::byte,2880> out{};
    for(std::uint64_t batch=0;batch<300;++batch) {
        for(auto &id:ids)id=rng()%128;
        const auto observed=c->gather(ids,out);expectRows(f,ids,out);
        EXPECT_EQ(observed.ram_rows+observed.source_rows+observed.duplicate_rows,32u);
        EXPECT_LE(c->stats().resident_rows,7u);EXPECT_EQ(c->stats().pinned_rows,0u);
        EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),g.sourceBytes()+g.storageBytes());
    }
    EXPECT_GT(c->stats().evictions,0u);EXPECT_GT(c->stats().admission_rejections,0u);
}
TEST(CpuAdaptiveImmutableRowCache, ReusedHotRowsSurviveSinglePassScanUnderExplicitCandidatePolicy) {
    File f;const auto g=Geometry::fromLayout(layout(f,4),Host,8);auto c=Cache::create(f.source,g,authority(g));
    std::array<std::byte,360> out{};const std::array<std::int64_t,4> seed{0,1,2,3};c->gather(seed,out);
    const std::array<std::int64_t,4> hot{0,1,0,1};
    for(std::uint64_t batch=1;batch<=8;++batch)c->gather(hot,out);
    for(std::int64_t row=4;row<80;++row){const std::array id{row};c->gather(id,std::span(out).first(90));}
    EXPECT_TRUE(c->tryPinResident(0));EXPECT_TRUE(c->tryPinResident(1));
    EXPECT_GT(c->stats().admission_rejections,0u);
}
/** @test Read-only owner snapshots exclude unrelated claims and do not prolong backing past final pin release. */
TEST(CpuAdaptiveImmutableRowCache, RetainedOwnerObservationUsesActualLeasesAndRespectsPinRetirement) {
    File f;const auto g=Geometry::fromLayout(layout(f,1,true),Host,2);
    PhysicalMemoryPlanBuilder plan(g.bom(host()));
    plan.add(host(),PhysicalMemoryOwner::ImmutableTableStorage,4096);
    plan.add(host(),PhysicalMemoryOwner::ImmutableLookupStaging,8192);
    auto p=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    auto extra_storage=p->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,4096);
    auto extra_staging=p->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,8192);
    auto c=Cache::create(f.source,g,p);
    const auto before=cacheWork(c->stats());const auto source=sourceWork(c->sourceStats());
    const auto checks=c->sourceStats().source_metadata_checks;
    const auto observed=c->retainedOwnerObservation();
    EXPECT_EQ(observed.authority().get(),p.get());EXPECT_EQ(observed.allocator(),Host);
    EXPECT_EQ(observed.hostPlacement(),ImmutableRowTableHostPlacement::taskPolicy());
    EXPECT_EQ(observed.tableStorageBytes()+extra_storage.bytes(),claimed(p,PhysicalMemoryOwner::ImmutableTableStorage));
    EXPECT_EQ(observed.lookupStagingBytes()+extra_staging.bytes(),claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging));
    EXPECT_EQ(cacheWork(c->stats()),before);EXPECT_EQ(sourceWork(c->sourceStats()),source);
    EXPECT_EQ(c->sourceStats().source_metadata_checks,checks+1);
    const std::array<std::int64_t,1> ids{7};std::array<std::byte,90> out{};
    EXPECT_EQ(c->gather(ids,out).policy_ordinal,0u);
    auto pin=c->tryPinResident(7);ASSERT_TRUE(pin);
    c->retire();const auto retired=cacheWork(c->stats());
    const auto retired_checks=c->sourceStats().source_metadata_checks;
    EXPECT_THROW(c->retainedOwnerObservation(),std::runtime_error);
    EXPECT_EQ(cacheWork(c->stats()),retired);
    EXPECT_EQ(c->sourceStats().source_metadata_checks,retired_checks);
    c.reset();expectRows(f,ids,pin->bytes());
    EXPECT_EQ(observed.tableStorageBytes()+extra_storage.bytes(),claimed(p,PhysicalMemoryOwner::ImmutableTableStorage));
    EXPECT_EQ(observed.lookupStagingBytes()+extra_staging.bytes(),claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging));
    pin.reset();
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),extra_storage.bytes());
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),extra_staging.bytes());
}

/** @test Failed source observation poisons reuse without minting successful or failed-gather work. */
TEST(CpuAdaptiveImmutableRowCache, RetainedOwnerObservationPoisonsStaleReuseWithoutGatherObservations) {
    File f;const auto g=Geometry::fromLayout(layout(f,0),Host,1);auto p=authority(g);auto c=Cache::create(f.source,g,p);
    const auto own=c->retainedOwnerObservation();const auto before=cacheWork(c->stats());
    ASSERT_EQ(::ftruncate(f.fd,f.original.size()+1),0);
    EXPECT_THROW(c->retainedOwnerObservation(),std::runtime_error);
    EXPECT_EQ(cacheWork(c->stats()),before);
    const auto checks=c->sourceStats().source_metadata_checks;
    EXPECT_THROW(c->retainedOwnerObservation(),std::runtime_error);
    EXPECT_THROW(c->authenticateSource(),std::runtime_error);
    EXPECT_EQ(c->sourceStats().source_metadata_checks,checks);
    EXPECT_EQ(cacheWork(c->stats()),before);
    EXPECT_EQ(own.tableStorageBytes(),claimed(p,PhysicalMemoryOwner::ImmutableTableStorage));
    EXPECT_EQ(own.lookupStagingBytes(),claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging));
    c.reset();EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),0u);
}

namespace {
/**
 * @brief Temporarily remove only a containing namespace, preserving the source inode metadata.
 *
 * Renaming the file itself changes ctime and cannot prove recovery of the exact
 * original source. Moving its parent leaves the child's identity unchanged;
 * restoring that parent makes the exact named/opened file authentic again.
 */
class ReversibleSourceNamespace final {
public:
    /** @brief Put the fixture under a privately owned parent before any source owner authenticates it. */
    explicit ReversibleSourceNamespace(File &file):file_(file),original_(file.path) {
        char name[]="/tmp/llaminar-row-pin-parent-XXXXXX";
        const char *created=::mkdtemp(name);
        if(!created)throw std::runtime_error("Cannot create private pin source parent");
        directory_=created;hidden_=directory_+".held";
        const auto relocated=directory_+"/rows";
        if(::rename(original_.c_str(),relocated.c_str())) {
            ::rmdir(directory_.c_str());throw std::runtime_error("Cannot place native source in private parent");
        }
        file_.path=relocated;file_.source.path=relocated;
    }
    /** @brief Restore the exact path before File retires its open descriptor and original fixture bytes. */
    ~ReversibleSourceNamespace() {
        if(hidden_now_ && ::rename(hidden_.c_str(),directory_.c_str()))std::terminate();
        if(::rename(file_.path.c_str(),original_.c_str()))std::terminate();
        file_.path=original_;file_.source.path=original_;
        if(::rmdir(directory_.c_str()))std::terminate();
    }
    ReversibleSourceNamespace(const ReversibleSourceNamespace &)=delete;
    ReversibleSourceNamespace &operator=(const ReversibleSourceNamespace &)=delete;
    /** @brief Make stat(original child path) fail without changing any child payload/identity field. */
    void hide() {
        if(hidden_now_ || ::rename(directory_.c_str(),hidden_.c_str()))throw std::runtime_error("Cannot hide private source parent");
        hidden_now_=true;
    }
    /** @brief Restore the same original child inode; an independent owner must authenticate it successfully. */
    void restore() {
        if(!hidden_now_ || ::rename(hidden_.c_str(),directory_.c_str()))throw std::runtime_error("Cannot restore private source parent");
        hidden_now_=false;
    }
private:
    File &file_;std::string original_,directory_,hidden_;bool hidden_now_=false;
};
/** @brief Check that pin-only authentication faults cannot mint a later successful owner operation. */
void pinAuthenticationFailureIsSticky(bool through_existing_pin) {
    File f;ReversibleSourceNamespace source_path(f);
    const auto g=Geometry::fromLayout(layout(f,1),Host,1);auto p=authority(g);
    auto c=Cache::create(f.source,g,p),control=Cache::create(f.source,g,authority(g));
    const std::array<std::int64_t,1> ids{7};std::array<std::byte,90> out{};
    ASSERT_EQ(c->gather(ids,out).policy_ordinal,0u);
    auto pin=c->tryPinResident(7);ASSERT_TRUE(pin);const auto generation=pin->generation();
    const auto before=cacheWork(c->stats());const auto source_before=sourceWork(c->sourceStats());
    const auto storage=claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),execution=claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging);
    source_path.hide();
    if(through_existing_pin) {EXPECT_THROW(pin->bytes(),std::runtime_error);}
    else {EXPECT_THROW(c->tryPinResident(7),std::runtime_error);}
    EXPECT_EQ(cacheWork(c->stats()),before);EXPECT_EQ(sourceWork(c->sourceStats()),source_before);
    const auto checks=c->sourceStats().source_metadata_checks;
    source_path.restore();
    // This positive authenticates real opened/named metadata AFTER recovery;
    // stale source metadata cannot accidentally satisfy the fault-state test.
    ASSERT_NO_THROW(control->authenticateSource());
    EXPECT_EQ(control->identity(),c->identity());
    out.fill(std::byte{0x73});const auto untouched=out;
    EXPECT_THROW(c->authenticateSource(),std::runtime_error);
    EXPECT_THROW(c->retainedOwnerObservation(),std::runtime_error);
    EXPECT_THROW(c->tryPinResident(7),std::runtime_error);
    EXPECT_THROW(c->gather(ids,out),std::runtime_error);EXPECT_EQ(out,untouched);
    EXPECT_THROW(pin->bytes(),std::runtime_error);
    EXPECT_EQ(pin->generation(),generation);EXPECT_EQ(cacheWork(c->stats()),before);
    EXPECT_EQ(c->sourceStats().source_metadata_checks,checks);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),storage);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),execution);
    c.reset();EXPECT_THROW(pin->bytes(),std::runtime_error);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),storage);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),execution);
    pin.reset();EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),0u);
}
}
/** @test A failed attempt to acquire another resident pin permanently invalidates the shared owner. */
TEST(CpuAdaptiveImmutableRowCache, TryPinAuthenticationFailureIsStickyAfterOriginalPathReturns) {
    pinAuthenticationFailureIsSticky(false);
}
/** @test A failure observed only through an existing pin bars later owner work and all retained pin reads. */
TEST(CpuAdaptiveImmutableRowCache, PinBytesAuthenticationFailureIsStickyAfterOriginalPathReturns) {
    pinAuthenticationFailureIsSticky(true);
}
/** @test Facade retirement preserves a valid pin, while a subsequent source failure remains terminal. */
TEST(CpuAdaptiveImmutableRowCache, RetiredPinAuthenticationFailureCannotRecoverWhenOriginalPathReturns) {
    File f;ReversibleSourceNamespace source_path(f);
    const auto g=Geometry::fromLayout(layout(f,1),Host,1);auto p=authority(g);
    auto c=Cache::create(f.source,g,p),control=Cache::create(f.source,g,authority(g));
    const std::array<std::int64_t,1> ids{7};std::array<std::byte,90> out{};c->gather(ids,out);
    auto pin=c->tryPinResident(7);ASSERT_TRUE(pin);const auto generation=pin->generation();
    c.reset();expectRows(f,ids,pin->bytes());
    const auto storage=claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),execution=claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging);
    source_path.hide();EXPECT_THROW(pin->bytes(),std::runtime_error);source_path.restore();
    ASSERT_NO_THROW(control->authenticateSource());
    EXPECT_THROW(pin->bytes(),std::runtime_error);EXPECT_EQ(pin->generation(),generation);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),storage);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),execution);
    pin.reset();EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableTableStorage),0u);
    EXPECT_EQ(claimed(p,PhysicalMemoryOwner::ImmutableLookupStaging),0u);
}
} // namespace llaminar2::test
