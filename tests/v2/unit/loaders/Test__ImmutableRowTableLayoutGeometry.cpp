/**
 * @file Test__ImmutableRowTableLayoutGeometry.cpp
 * @brief Device-free original-native range, pool and no-SSD contract regressions.
 *
 * No model, file access, backend initialization, allocation authority or runtime
 * row directory is used. Tiny explicit intervals supply independent coverage
 * and native-byte equations; large extents prove source-size-independent metadata.
 * Cross-vendor declarations prove geometry only, not native transport or runtime eligibility.
 */
#include "loaders/ImmutableRowTableLayoutGeometry.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <limits>
namespace llaminar2::test {
namespace {
using Storage=ImmutableRowStorage;
using Residency=ImmutableRowTableResidency;
using Role=ImmutableRowTableCopyRole;
using Id=PhysicalMemoryAllocatorIdentity;
using Owner=ImmutableRowTableOwnerKey;
using Layout=ImmutableRowTableLayoutGeometry;
/** @return Exact native IQ4_NL row extent without creating any tensor or reading its payload. */
ImmutableRowTableExtent extent(std::uint64_t rows=10,std::size_t bytes=90) {
    return ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(20),rows,bytes);
}
/** @return An exact setup allocator, including foreign-rank coverage in tests. */
Id gpu(int ordinal=0,int rank=0) {return {rank,DeviceId::cuda(ordinal)};}
/** @return Exact host allocator, distinct from NUMA placement policy. */
Id cpu(int rank=0) {return {rank,DeviceId::cpu()};}
/** @return Checked enabled runtime-tier set. */
ImmutableRowTableTiers tiers(std::initializer_list<Storage> values) {
    return ImmutableRowTableTiers::of(std::span(values.begin(),values.size()));
}
/** @return Only the existing concrete bounded mapping geometry. */
ImmutableRowTableSourceService service(Id id=cpu()) {
    return {id,ImmutableRowTableStoragePolicy::forStorage(Storage::MemoryMapped,65536,1)};
}
const Owner v0{gpu(),Storage::VRAM},v1{gpu(1),Storage::VRAM},ram{cpu(),Storage::RAM};
/** @return Minimal complete fixed native table, independent of consumer ownership. */
Layout fixed() {
    return Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,10,0,{}}},{gpu()},
        {{0,10,ram,Role::Primary}},{});
}
/** @return One bounded adaptive GPU pool with original SSD service. */
Layout adaptive(std::uint64_t slots=2,std::uint64_t inflight=3) {
    return Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,slots,{}}},{gpu()}, {},{service()}, {},gpu(),inflight);
}
}
TEST(ImmutableRowTableLayoutGeometry, AllSevenTierMasksAreExactAndInvalidSetsReject) {
    const std::array all{Storage::RAM,Storage::MemoryMapped,Storage::VRAM};
    for(unsigned mask=1;mask<8;++mask) {
        std::vector<Storage> selected;
        for(unsigned i=0;i<3;++i) if(mask&(1u<<i))selected.push_back(all[i]);
        const auto actual=ImmutableRowTableTiers::of(selected);
        EXPECT_EQ(actual.bits(),mask);
        for(unsigned i=0;i<3;++i) EXPECT_EQ(actual.contains(all[i]),bool(mask&(1u<<i)));
    }
    EXPECT_THROW(tiers({}),std::invalid_argument);
    EXPECT_THROW(tiers({Storage::RAM,Storage::RAM}),std::invalid_argument);
    EXPECT_THROW(tiers({static_cast<Storage>(255)}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, EveryFixedMaskHasActualOwnersAndOriginalCoverage) {
    for(unsigned mask=1;mask<8;++mask) {
        std::vector<Storage> selected;std::vector<ImmutableRowTableLayoutOwner> owners;
        std::vector<ImmutableRowTableFixedRange> ranges;std::vector<ImmutableRowTableSourceService> services;
        if(mask&1)selected.push_back(Storage::RAM);
        if(mask&2){selected.push_back(Storage::MemoryMapped);services.push_back(service());}
        if(mask&4)selected.push_back(Storage::VRAM);
        if(mask&1){const auto n=mask&4?5:10;owners.push_back({ram,static_cast<std::uint64_t>(n),0,{}});ranges.push_back({0,static_cast<std::uint64_t>(n),ram,Role::Primary});}
        if(mask&4){const auto start=mask&1?5:0;owners.push_back({v0,static_cast<std::uint64_t>(10-start),0,{}});ranges.push_back({static_cast<std::uint64_t>(start),static_cast<std::uint64_t>(10-start),v0,Role::Primary});}
        const auto result=Layout::checked(extent(),ImmutableRowTableTiers::of(selected),Residency::Fixed,
            owners,{gpu(7,3)},ranges,services);
        EXPECT_EQ(result.extent(),extent());EXPECT_EQ(result.gpuDirectoryRecordBound(),0u);
        EXPECT_EQ(result.consumers().front(),gpu(7,3)); // Consumers need not be payload owners or same rank.
    }
}
TEST(ImmutableRowTableLayoutGeometry, NoSSDGapOverlapAndFalseReplicaCoverageReject) {
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,9,0,{}}},{cpu()},
        {{0,9,ram,Role::Primary}},{}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,10,0,{}}},{cpu()},
        {{0,10,ram,Role::Replica}},{}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM}),Residency::Fixed,{{v0,6,0,{}},{v1,5,0,{}}},{gpu()},
        {{0,6,v0,Role::Primary},{5,5,v1,Role::Primary}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, ExplicitFixedReplicasHaveIndependentPayloadCapacity) {
    const auto result=Layout::checked(extent(),tiers({Storage::VRAM}),Residency::Fixed,
        {{v0,10,0,900},{v1,10,0,900}},{gpu(3)},{{0,10,v0,Role::Primary},{0,10,v1,Role::Replica}},{});
    EXPECT_EQ(result.nativePayloadBytes(v0),900u);EXPECT_EQ(result.nativePayloadBytes(v1),900u);
    EXPECT_EQ(result.fixedRanges().size(),2u);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM}),Residency::Fixed,{{v0,10,0,899}},{gpu()},
        {{0,10,v0,Role::Primary}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, SameOwnerDuplicateCopyAndUnpricedFixedCapacityReject) {
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,10,0,{}}},{cpu()},
        {{0,8,ram,Role::Primary},{6,2,ram,Role::Replica}},{}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,10,0,{}}},{cpu()},
        {{0,9,ram,Role::Primary}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, AdaptivePoolsAreUnrestrictedAndDirectoryBoundUsesActualSlots) {
    const auto result=Layout::checked(extent(320001536),tiers({Storage::VRAM,Storage::RAM,Storage::MemoryMapped}),
        Residency::Adaptive,{{v0,0,2,180},{v1,0,3,270},{ram,0,4,360}},{gpu(7),cpu()}, {},{service()}, {},gpu(),5);
    EXPECT_TRUE(result.fixedRanges().empty());EXPECT_EQ(result.gpuDirectoryRecordBound(),10u);
    EXPECT_EQ(result.owners().size(),3u);EXPECT_EQ(result.extent().nativeBytes(),28800138240ull);
    EXPECT_EQ(result.nativePayloadBytes(v1),270u); // Any source row can occupy any of these three slots.
}
TEST(ImmutableRowTableLayoutGeometry, AllowedNativeTiersSurviveZeroFitWithoutPolicySubstitution) {
    const auto allowed=tiers({Storage::VRAM,Storage::RAM,Storage::MemoryMapped});
    for(const bool declare_empty:{false,true}) {
        const std::vector<ImmutableRowTableLayoutOwner> owners=declare_empty?
            std::vector<ImmutableRowTableLayoutOwner>{{v0,0,0,0},{ram,0,0,0}}:
            std::vector<ImmutableRowTableLayoutOwner>{};
        for(const auto policy:{Residency::Fixed,Residency::Adaptive}) {
            const auto result=Layout::checked(extent(),allowed,policy,owners,{gpu()}, {},{service()});
            EXPECT_EQ(result.tiers(),allowed);EXPECT_EQ(result.tiers().bits(),7u);
            EXPECT_EQ(result.owners().size(),owners.size());EXPECT_EQ(result.gpuDirectoryRecordBound(),0u);
            EXPECT_FALSE(result.gpuCoordinator());
            if(declare_empty){EXPECT_EQ(result.nativePayloadBytes(v0),0u);EXPECT_EQ(result.nativePayloadBytes(ram),0u);}
        }
    }
    // An allowed but empty GPU tier does not become an invented GPU payload owner.
    const auto resident=Layout::checked(extent(),tiers({Storage::VRAM,Storage::RAM}),Residency::Fixed,
        {{v0,0,0,0},{ram,10,0,900}},{gpu()},{{0,10,ram,Role::Primary}},{});
    EXPECT_TRUE(resident.tiers().contains(Storage::VRAM));EXPECT_EQ(resident.nativePayloadBytes(v0),0u);
    EXPECT_EQ(resident.nativePayloadBytes(ram),900u);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,0,0},{v1,0,2,180}},{gpu()}, {},{service()}, {{{v0,v1},2}},gpu(),1),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, SSDPrimaryMayBackExplicitNativeReplicaOnlyRanges) {
    const auto result=Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Fixed,
        {{v0,3,0,270}},{gpu()},{{2,3,v0,Role::Replica}},{service()});
    EXPECT_EQ(result.nativePayloadBytes(v0),270u);ASSERT_EQ(result.fixedRanges().size(),1u);
    EXPECT_EQ(result.fixedRanges().front().role,Role::Replica);EXPECT_EQ(result.sourceServices().size(),1u);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Fixed,
        {{v0,3,0,270}},{gpu()},{{2,3,v0,Role::Replica}},{}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM}),Residency::Fixed,
        {{v0,3,0,270}},{gpu()},{{2,3,v0,Role::Replica}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, ZeroOptionalCacheUsesOnlyExplicitSSDService) {
    const auto result=Layout::checked(extent(320001536),tiers({Storage::MemoryMapped}),Residency::Adaptive,
        {},{gpu()}, {},{service()});
    EXPECT_TRUE(result.owners().empty());EXPECT_TRUE(result.fixedRanges().empty());
    EXPECT_EQ(result.sourceServices().size(),1u);EXPECT_EQ(result.gpuDirectoryRecordBound(),0u);
}
TEST(ImmutableRowTableLayoutGeometry, AdaptiveNoSSDAndSourceHomeQuotasRejectPrecisely) {
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM}),Residency::Adaptive,
        {{v0,0,10,{}}},{gpu()}, {},{}, {},gpu(),1),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,2,{}}},{gpu()}, {{0,2,v0,Role::Primary}},{service()}, {},gpu(),1),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, AdaptiveGroupsAreExplicitAndDoNotAssignSourceRanges) {
    const auto result=Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,2,{}},{v1,0,3,{}}},{gpu()}, {},{service()}, {{{v0,v1},2}},gpu(),1);
    ASSERT_EQ(result.replicaGroups().size(),1u);EXPECT_EQ(result.gpuDirectoryRecordBound(),6u);
    EXPECT_TRUE(result.fixedRanges().empty());
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,2,{}},{v1,0,3,{}}},{gpu()}, {},{service()}, {{{v0,v0},2}},gpu(),1),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, SSDServiceIsExplicitAndDisabledSourceCannotBeSmuggledIn) {
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::MemoryMapped}),Residency::Fixed,{}, {cpu()}, {},{}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,10,0,{}}}, {cpu()},
        {{0,10,ram,Role::Primary}},{service()}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::MemoryMapped}),Residency::Fixed,{}, {cpu()}, {},
        {{gpu(),ImmutableRowTableStoragePolicy::forStorage(Storage::MemoryMapped)}}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, ExactEndpointAndCoordinatorIdentityAreMandatory) {
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,10,0,{}}}, {cpu(),cpu()},
        {{0,10,ram,Role::Primary}},{}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,2,{}}},{gpu()}, {},{service()}, {},gpu(3),1),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,2,{}}},{gpu()}, {},{service()}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{{gpu(),Storage::RAM},10,0,{}}},
        {cpu()},{{0,10,{gpu(),Storage::RAM},Role::Primary}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, HostAdaptivePoolsOwnNoGPUDirectory) {
    const auto result=Layout::checked(extent(),tiers({Storage::RAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{ram,0,2,{}}},{gpu()}, {},{service()});
    EXPECT_EQ(result.gpuDirectoryRecordBound(),0u);EXPECT_FALSE(result.gpuCoordinator());
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{ram,0,2,{}}},{gpu()}, {},{service()}, {},gpu(),1),std::invalid_argument);
}
/** @test Host request scratch retains its declared bound without inventing GPU residency records. */
TEST(ImmutableRowTableLayoutGeometry, HostAdaptiveRequestBoundsDoNotCreateGPUDirectory) {
    for(const auto slots:{0u,2u,10u}) for(const auto inflight:{1u,16u,8192u})
        for(const auto consumer:{cpu(),gpu()}) {
            const auto result=Layout::checked(extent(),tiers({Storage::RAM,Storage::MemoryMapped}),Residency::Adaptive,
                {{ram,0,slots,{}}},{consumer}, {},{service()}, {},std::nullopt,inflight);
            EXPECT_EQ(result.maximumInflightRows(),inflight);
            EXPECT_EQ(result.gpuDirectoryRecordBound(),0u);
            EXPECT_FALSE(result.gpuCoordinator());
            EXPECT_EQ(result.nativePayloadBytes(ram),slots*90u);
            EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM,Storage::MemoryMapped}),Residency::Adaptive,
                {{ram,0,slots,{}}},{consumer}, {},{service()}, {},gpu(),inflight),std::invalid_argument);
        }
}
TEST(ImmutableRowTableLayoutGeometry, PooledCountsOneTwoFourEightUseAggregateSlotsAndExactRanks) {
    for(const int count:{1,2,4,8}) {
        std::vector<ImmutableRowTableLayoutOwner> owners;
        std::vector<Id> consumers;
        for(int i=0;i<count;++i) {
            owners.push_back({{gpu(i,i%2),Storage::VRAM},0,static_cast<std::uint64_t>(i+1),{}});
            consumers.push_back(gpu(i,i%2));
        }
        const auto result=Layout::checked(extent(1000000000),tiers({Storage::VRAM,Storage::MemoryMapped}),
            Residency::Adaptive,owners,consumers,{}, {service()}, {},gpu(),3);
        EXPECT_EQ(result.owners().size(),static_cast<std::size_t>(count));
        EXPECT_EQ(result.gpuDirectoryRecordBound(),static_cast<std::uint64_t>(count*(count+1)/2+3));
        EXPECT_TRUE(result.fixedRanges().empty());
    }
}
TEST(ImmutableRowTableLayoutGeometry, DeclarationIdentityIncludesAllPolicyAndCapacityInputs) {
    const auto a=adaptive(2,3),b=adaptive(3,3),c=adaptive(2,4);
    EXPECT_EQ(a,adaptive(2,3));EXPECT_NE(a,b);EXPECT_NE(a,c);
    auto malformed=v0;malformed.allocator.world_rank=-1;
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM}),Residency::Fixed,{{malformed,10,0,{}}},
        {gpu()},{{0,10,malformed,Role::Primary}},{}),std::invalid_argument);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),static_cast<Residency>(99),{{ram,10,0,{}}},
        {cpu()},{{0,10,ram,Role::Primary}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, CheckedNativeAndDirectoryOverflowHaveNoSilentWrap) {
    const auto n=std::numeric_limits<std::uint64_t>::max();
    EXPECT_THROW(extent(n,2),std::overflow_error);
    EXPECT_THROW(Layout::checked(extent(n,1),tiers({Storage::VRAM,Storage::MemoryMapped}),Residency::Adaptive,
        {{v0,0,n,{}}},{gpu()}, {},{service()}, {},gpu(),1),std::overflow_error);
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::RAM}),Residency::Fixed,{{ram,10,0,{}}},{cpu()},
        {{n,2,ram,Role::Primary}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, HostLocalityIsExplicitAndIndependentOfAllocatorOrdinal) {
    const std::array hosts{cpu()},consumers{gpu()};
    const auto result=Layout::wholeTier(ImmutableRowTableStoragePolicy::forStorage(Storage::RAM),extent(),hosts,
        consumers,ImmutableRowTableHostPlacement::onNode(7));
    ASSERT_EQ(result.owners().size(),1u);
    EXPECT_EQ(result.owners().front().host_placement.numaNode(),7);
    EXPECT_EQ(result.owners().front().key.allocator,cpu());
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM}),Residency::Fixed,
        {{v0,10,0,{},ImmutableRowTableHostPlacement::onNode(7)}},{gpu()},{{0,10,v0,Role::Primary}},{}),std::invalid_argument);
}
TEST(ImmutableRowTableLayoutGeometry, WholeTierAdapterPreservesWorkingStorageAndConsumers) {
    const std::array hosts{cpu()}; const std::array gpus{gpu(),gpu(1)};
    const auto source=extent();
    for(const auto storage:{Storage::RAM,Storage::MemoryMapped,Storage::VRAM}) {
        const auto owners=storage==Storage::VRAM?std::span<const Id>(gpus):std::span<const Id>(hosts);
        const auto result=Layout::wholeTier(ImmutableRowTableStoragePolicy::forStorage(storage),source,owners,gpus);
        EXPECT_EQ(result.extent(),source);EXPECT_EQ(result.consumers().size(),2u);
        EXPECT_EQ(result.gpuDirectoryRecordBound(),0u);EXPECT_EQ(result.residency(),Residency::Fixed);
        if(storage==Storage::MemoryMapped){EXPECT_EQ(result.sourceServices().size(),1u);EXPECT_TRUE(result.owners().empty());}
        else for(auto owner:owners)EXPECT_EQ(result.nativePayloadBytes({owner,storage}),900u);
    }
    EXPECT_THROW(fixed().nativePayloadBytes(v0),std::invalid_argument);
    EXPECT_EQ(adaptive().maximumInflightRows(),3u);
}

/** @test Equal vendor-local ordinals remain distinct physical owners across independent consumers. */
TEST(ImmutableRowTableLayoutGeometry, MixedVendorOwnersAndConsumersRemainIndependent) {
    const Id cuda0{0,DeviceId::cuda(0)},rocm0{0,DeviceId::rocm(0)};
    const Id storage_only{2,DeviceId::rocm(3)},consumer_only{3,DeviceId::cuda(1)};
    const Id host_source{4,DeviceId::cpu()};
    const Owner cuda{cuda0,Storage::VRAM},rocm{rocm0,Storage::VRAM};
    const Owner empty{storage_only,Storage::VRAM},host{host_source,Storage::RAM};
    ASSERT_NE(cuda0,rocm0);
    const auto allowed=tiers({Storage::VRAM,Storage::RAM,Storage::MemoryMapped});
    const std::vector<ImmutableRowTableLayoutOwner> owners{
        {cuda,0,2,180},{rocm,0,5,450},{empty,0,0,0},{host,0,3,270}};
    const std::vector<Id> consumers{cuda0,rocm0,cpu(1),consumer_only};
    const auto result=Layout::checked(extent(),allowed,Residency::Adaptive,owners,consumers,
        {},{service(host_source)}, {},storage_only,7);
    EXPECT_EQ(result.owners().size(),4u);
    EXPECT_EQ(result.consumers().size(),consumers.size());
    EXPECT_TRUE(std::ranges::equal(result.consumers(),consumers));
    EXPECT_EQ(result.gpuCoordinator(),storage_only);
    EXPECT_EQ(result.sourceServices().front().allocator,host_source);
    EXPECT_EQ(result.nativePayloadBytes(cuda),180u);
    EXPECT_EQ(result.nativePayloadBytes(rocm),450u);
    EXPECT_EQ(result.nativePayloadBytes(empty),0u);
    EXPECT_EQ(result.nativePayloadBytes(host),270u);
    EXPECT_EQ(result.gpuDirectoryRecordBound(),14u); // GPU slots (2+5+0) plus seven live rows; RAM slots are not GPU directory records.
    EXPECT_EQ(result.maximumInflightRows(),7u);
    EXPECT_TRUE(result.fixedRanges().empty());
    EXPECT_EQ(result,Layout::checked(extent(),allowed,Residency::Adaptive,owners,consumers,
        {},{service(host_source)}, {},storage_only,7));
    EXPECT_NE(result,Layout::checked(extent(),allowed,Residency::Adaptive,owners,consumers,
        {},{service(host_source)}, {},rocm0,7));
    auto changed=owners;
    changed[0].physical_byte_limit=181;
    EXPECT_NE(result,Layout::checked(extent(),allowed,Residency::Adaptive,changed,consumers,
        {},{service(host_source)}, {},storage_only,7));
    EXPECT_NE(result,Layout::checked(extent(),allowed,Residency::Adaptive,owners,consumers,
        {},{service(cpu(5))}, {},storage_only,7));
    EXPECT_THROW(Layout::checked(extent(),allowed,Residency::Adaptive,owners,consumers,
        {},{service(host_source)}, {},Id{8,DeviceId::rocm(3)},7),std::invalid_argument);
}

/** @test Native fixed ranges may span vendors/ranks without mirroring the table on consumers. */
TEST(ImmutableRowTableLayoutGeometry, MixedVendorFixedRangesRetainExactNativeCoverage) {
    const Owner cuda{{0,DeviceId::cuda(0)},Storage::VRAM};
    const Owner rocm{{1,DeviceId::rocm(0)},Storage::VRAM};
    const Owner host{{2,DeviceId::cpu()},Storage::RAM};
    const std::vector<ImmutableRowTableLayoutOwner> owners{{cuda,2,0,180},{rocm,5,0,450},{host,3,0,270}};
    const std::vector<ImmutableRowTableFixedRange> ranges{
        {0,2,cuda,Role::Primary},{2,5,rocm,Role::Primary},{7,3,host,Role::Primary}};
    const std::vector<Id> consumers{cpu(3),{4,DeviceId::cuda(1)},{5,DeviceId::rocm(2)}};
    const auto result=Layout::checked(extent(),tiers({Storage::VRAM,Storage::RAM}),Residency::Fixed,
        owners,consumers,ranges,{});
    EXPECT_EQ(result.nativePayloadBytes(cuda)+result.nativePayloadBytes(rocm)+result.nativePayloadBytes(host),900u);
    EXPECT_EQ(result.gpuDirectoryRecordBound(),0u);
    EXPECT_EQ(result.fixedRanges().size(),ranges.size());
    EXPECT_TRUE(std::ranges::equal(result.fixedRanges(),ranges));
    EXPECT_EQ(result.consumers().size(),consumers.size());
    EXPECT_TRUE(std::ranges::equal(result.consumers(),consumers));
    auto collision=owners;
    collision[1].key=cuda;
    EXPECT_THROW(Layout::checked(extent(),tiers({Storage::VRAM,Storage::RAM}),Residency::Fixed,
        collision,consumers,ranges,{}),std::invalid_argument);
}
} // namespace llaminar2::test
