/**
 * @file Test__ImmutableLookupSubordinate.cpp
 * @brief Source-only authored CPU proofs for the actual subordinate processor/cache authority.
 *
 * Tiny original native files, exact source-owned PMA plans and real host processor
 * acquisitions exercise the capability. Wire publications below simulate the
 * documented GPU producer; they do not certify GPU reservation/copy/publication.
 * No proof factory, substitute provider, device startup or table hash is used.
 */
#include "execution/qwen4exp/ImmutableLookupService.h"
#include <gtest/gtest.h>
#include "../../../mocks/MockBackend.h"
#include <chrono>
#include <thread>
#include <cstdio>
#include <limits>
#include <sys/resource.h>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <new>
#include <vector>
#include <unistd.h>

namespace llaminar2::qwen4exp {
/** @brief Deterministic interleaving at the real serial borrow boundary; no capability constructor or copied pin logic. */
struct ImmutableLookupSubordinateTestAccess final {
    /** @brief Inspect only joined CPU-owned state; callers have already awaited this actual service worker. */
    static bool ownsAcquiredGrant(const ImmutableLookupService &service) {
        if(!service.worker_.idle())throw std::logic_error("Grant observation requires joined worker");
        return service.processor_.subordinateGrant_.has_value();
    }
    /** @brief Insert a wire retirement/refill after the production acquire, before its production completion. */
    template<class Mutation>
    static auto borrowAcross(ImmutableLookupHostProcessor &processor,std::size_t row,Mutation mutate) {
        const auto borrow=processor.acquireSubordinateRead();
        mutate();
        return processor.finishSubordinateRead(borrow,row);
    }
};
}

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace {
using Intent=ImmutableRowSubordinateIntent;
using Operation=ImmutableRowReadIntent;
const PhysicalMemoryAllocatorIdentity hostId{0,DeviceId::cpu()};
/** @return Complete canonical host observation; no independent free-memory balance. */
PhysicalMemoryResource hostResource(){return {0,DeviceId::cpu(),1ull<<30,1ull<<30};}
/** @brief Retain a small nonzero-offset original native tensor through every source owner. */
struct NativeFile {
    int fd=-1;std::string path;std::vector<std::byte> bytes;ImmutableRowTableSource source;
    /** @brief Author distinguishable original IQ4_NL bytes without quantization or a payload hash. */
    NativeFile(){
        char name[]="/tmp/llaminar-subordinate-source-XXXXXX";fd=::mkstemp(name);
        if(fd<0)throw std::runtime_error("Cannot create subordinate source fixture");
        path=name;source={.path=path,.tensor_name="fixture.ple",.file_offset=37,
            .rows=128,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
        bytes.resize(source.file_offset+source.tensorBytes());
        for(std::size_t i=0;i<bytes.size();++i)bytes[i]=std::byte((i*13+i/31)%251);
        if(::write(fd,bytes.data(),bytes.size())!=static_cast<ssize_t>(bytes.size())) {
            ::close(fd);fd=-1;::unlink(path.c_str());
            throw std::runtime_error("Cannot write subordinate source fixture");
        }
    }
    /** @brief Remove only this authored source after all fixture owners have retired. */
    ~NativeFile(){if(fd>=0)::close(fd);::unlink(path.c_str());}
};
/** @return Checked source/endpoint geometry with legal zero-capacity native endpoints. */
ImmutableRowTableLayoutGeometry layoutFor(const NativeFile &file,DeviceId gpu,std::size_t ramSlots,std::size_t gpuSlots){
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    const PhysicalMemoryAllocatorIdentity requester{0,gpu};
    return ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(file.source.format,file.source.rows,file.source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{hostId,ImmutableRowStorage::RAM},0,ramSlots,{}},{{requester,ImmutableRowStorage::VRAM},0,gpuSlots,{}}},
        {requester},{},{{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},
        {},gpuSlots?std::optional(requester):std::nullopt,8);
}
/** @brief Matching owner for this actual aligned CPU protocol allocation. */
struct AlignedDelete {/** @brief Release only the corresponding admitted test bytes. */
    void operator()(std::byte *p)const noexcept{::operator delete(p,std::align_val_t{64});}};
/** @brief Actual CPU component, exact PMA leases and existing acquired-request processor. */
struct Lane {
    NativeFile file;
    ImmutableLookupLayout layout;
    CpuAdaptiveImmutableRowCacheGeometry geometry;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<HostImmutableRowProvider> provider;
    CpuAdaptiveImmutableRowCache *cache=nullptr; // Retained pre-adoption identity, never a production source escape.
    PhysicalMemoryAllocationLease channelLease;
    std::unique_ptr<std::byte,AlignedDelete> bytes;
    std::atomic<ImmutableLookupStatus> fatal{ImmutableLookupStatus::Success};
    std::atomic<bool> cancelled{false};
    std::unique_ptr<ImmutableLookupHostProcessor> processor;
    /** @brief Admit and materialize only CPU source/cache/scratch/channel owners; GPU slots are metadata here. */
    Lane(DeviceId gpu=DeviceId::cuda(0),std::size_t ramSlots=2,std::size_t gpuSlots=2)
        :layout(ImmutableLookupLayout::createSubordinate(8,file.source.rowBytes())),
         geometry(CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layoutFor(file,gpu,ramSlots,gpuSlots),hostId,{0,gpu})) {
        PhysicalMemoryPlanBuilder plan(geometry.bom(hostResource()));
        plan.add(hostResource(),PhysicalMemoryOwner::ImmutableLookupStaging,layout.bytes());
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        auto owner=CpuAdaptiveImmutableRowCache::create(file.source,geometry,memory);cache=owner.get();
        provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(std::move(owner));
        channelLease=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,layout.bytes());
        bytes.reset(static_cast<std::byte*>(::operator new(layout.bytes(),std::align_val_t{64})));
        std::fill_n(bytes.get(),layout.bytes(),std::byte{0xa5});
        processor=std::make_unique<ImmutableLookupHostProcessor>(*provider,layout,
            std::span<std::byte>{bytes.get(),layout.bytes()},17,&fatal,&cancelled);
    }
    /** @brief Retire ingress before its provider or channel, matching actual retained service ownership. */
    ~Lane(){processor.reset();provider.reset();bytes.reset();channelLease={};}
    /** @return Original request frame; tests mutate only the declared wire boundary. */
    ImmutableLookupRequest &request(){return *reinterpret_cast<ImmutableLookupRequest*>(bytes.get()+layout.requestOffset());}
    /** @return Exact response header; padding is never treated as traffic. */
    ImmutableLookupCompletion &completion(){return *reinterpret_cast<ImmutableLookupCompletion*>(bytes.get()+layout.completionOffset());}
    /** @return Existing response retirement metadata, never a new sequence ledger. */
    ImmutableLookupAcknowledgement &ack(){return *reinterpret_cast<ImmutableLookupAcknowledgement*>(bytes.get()+layout.acknowledgementOffset());}
    /** @return Full admitted payload for inactive-suffix poison checks only. */
    std::span<std::byte> payload(){return {bytes.get()+layout.payloadOffset(),layout.maximumIds()*layout.rowBytes()};}
    /** @return Full admitted ACK metadata capacity for untouched-suffix checks only. */
    std::span<ImmutableRowInvalidationAcknowledgement> invalidations(){return {
        reinterpret_cast<ImmutableRowInvalidationAcknowledgement*>(bytes.get()+layout.invalidationsOffset()),layout.maximumIds()};}
    /** @brief Simulate only an immutable GPU compact-ID/intent publication, with exact live extents. */
    void publish(std::uint64_t sequence,std::uint64_t generation,std::span<const std::int64_t> ids,std::span<const Intent> intents){
        if(ids.size()!=intents.size() || ids.size()>layout.maximumIds())throw std::invalid_argument("Invalid test frame");
        if(!ids.empty()){
            std::memcpy(bytes.get()+layout.idsOffset(),ids.data(),ids.size_bytes());
            std::memcpy(bytes.get()+layout.intentsOffset(),intents.data(),intents.size_bytes());
        }
        ImmutableLookupRequest frame;frame.sequenceComplement=~sequence;
        frame.generation=generation;frame.generationComplement=~generation;frame.table=17;frame.tableComplement=~frame.table;
        frame.idCount=ids.size();frame.idCountComplement=~frame.idCount;frame.idBytes=ids.size_bytes();frame.idBytesComplement=~frame.idBytes;
        frame.rowBytes=layout.rowBytes();frame.rowBytesComplement=~frame.rowBytes;
        frame.protocol=layout.protocol();frame.protocolComplement=~frame.protocol;
        std::memcpy(reinterpret_cast<std::byte*>(&request())+8,reinterpret_cast<const std::byte*>(&frame)+8,kLookupRequestLiveBytes-8);
        std::atomic_ref(request().publication).store(sequence,std::memory_order_release);
    }
    /** @brief Publish the actual consumer's response-retirement word. */
    void consume(std::uint64_t sequence){std::atomic_ref(ack().consumedSequence).store(sequence,std::memory_order_release);}
    /** @brief Compare native output directly against the authored source, preserving duplicates. */
    void expectRows(std::span<const std::int64_t> ids){
        for(std::size_t i=0;i<ids.size();++i)EXPECT_EQ(std::memcmp(payload().data()+i*layout.rowBytes(),
            file.bytes.data()+file.source.file_offset+ids[i]*layout.rowBytes(),layout.rowBytes()),0);
        for(std::size_t i=ids.size()*layout.rowBytes();i<payload().size();++i)EXPECT_EQ(payload()[i],std::byte{0xa5});
    }
    /** @brief Fill only test payload/ACK storage after its previous consumer retired. */
    void poison(){std::fill(payload().begin(),payload().end(),std::byte{0xa5});
        std::memset(invalidations().data(),0xa5,invalidations().size_bytes());}
};
}

/** @test Exact endpoints and zero GPU residency preserve subordinate mode without a fake coordinator. */
TEST(ImmutableLookupSubordinate, GeometryAuthenticatesEndpointsAndZeroGpuCapacity){
    NativeFile file;
    for(auto gpu:{DeviceId::cuda(0),DeviceId::rocm(0)})for(auto gpuSlots:{0u,2u}){
        const auto layout=layoutFor(file,gpu,0,gpuSlots);
        EXPECT_EQ(layout.gpuCoordinator().has_value(),gpuSlots!=0);
        EXPECT_NO_THROW((void)CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,{0,gpu}));
        if(gpuSlots){EXPECT_THROW((void)CpuAdaptiveImmutableRowCacheGeometry::fromLayout(layout,hostId,8),std::invalid_argument);}
        else {EXPECT_NO_THROW((void)CpuAdaptiveImmutableRowCacheGeometry::fromLayout(layout,hostId,8));}
        EXPECT_THROW((void)CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,{0,DeviceId::cpu()}),std::invalid_argument);
        EXPECT_THROW((void)CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,{1,gpu}),std::invalid_argument);
        EXPECT_THROW((void)CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,{0,DeviceId(gpu.type,1)}),std::invalid_argument);
    }
}

/** @test The same provider cannot escape to independent gather/pin or an ordinary channel. */
TEST(ImmutableLookupSubordinate, ClosedOwnerRejectsIndependentReadsAndWrongChannelMode){
    Lane lane;const std::array<std::int64_t,1> ids{3};std::array<std::byte,90> out{};
    EXPECT_THROW((void)lane.provider->gather(ids,out),std::invalid_argument);
    EXPECT_THROW((void)lane.cache->gather(ids,out),std::invalid_argument);
    EXPECT_THROW((void)lane.cache->tryPinResident(3),std::invalid_argument);
    EXPECT_TRUE(lane.memory->rankComplete());
    const auto ordinary=ImmutableLookupLayout::create(8,lane.layout.rowBytes());
    EXPECT_THROW((void)ImmutableLookupHostProcessor(*lane.provider,ordinary,
        std::span<std::byte>{lane.bytes.get(),lane.layout.bytes()},19),std::invalid_argument);
    EXPECT_THROW((void)ImmutableLookupHostProcessor(*lane.provider,lane.layout,
        std::span<std::byte>{lane.bytes.get(),lane.layout.bytes()},19),std::invalid_argument);
    EXPECT_EQ(lane.cache->stats().completed_batches,0u);
}

/** @test Fill then detach precedes ACK; old pins retain backing, all-pinned responses progress and slot ABA cannot reuse live bytes. */
TEST(ImmutableLookupSubordinate, PromotionDetachesBeforeAckAndRetainsOldPinnedBacking){
    Lane lane;const std::array<std::int64_t,3> seed{3,3,4};const std::array<Intent,3> reads{};
    lane.publish(1,1,seed,reads);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);lane.expectRows(seed);
    auto old=lane.processor->borrowSubordinateResponseRow(0);auto other=lane.processor->borrowSubordinateResponseRow(2);
    ASSERT_TRUE(old);ASSERT_TRUE(other);const auto oldGeneration=old->generation();
    lane.consume(1);lane.poison();
    const std::array<std::int64_t,2> promote{3,3};const std::array<Intent,2> intents{{{Operation::ReadAndInvalidate,99},{Operation::ReadAndInvalidate,99}}};
    lane.publish(2,2,promote,intents);ASSERT_EQ(lane.processor->poll(2),ImmutableLookupPoll::Completed);lane.expectRows(promote);
    ASSERT_EQ(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::Success));
    EXPECT_EQ(lane.cache->stats().invalidations,1u);EXPECT_EQ(lane.cache->stats().resident_rows,1u);
    EXPECT_EQ(lane.invalidations()[0].destination_grant,99u);EXPECT_EQ(lane.invalidations()[1].destination_grant,99u);
    EXPECT_EQ(lane.invalidations()[2].destination_grant,0xa5a5a5a5a5a5a5a5ULL);
    EXPECT_THROW((void)lane.processor->borrowSubordinateResponseRow(0),std::invalid_argument);
    EXPECT_EQ(std::memcmp(old->bytes().data(),lane.file.bytes.data()+lane.file.source.file_offset+3*lane.layout.rowBytes(),lane.layout.rowBytes()),0);
    lane.consume(2);lane.poison();const std::array<std::int64_t,1> miss{7};const std::array<Intent,1> read{};
    lane.publish(3,3,miss,read);ASSERT_EQ(lane.processor->poll(3),ImmutableLookupPoll::Completed);lane.expectRows(miss);
    EXPECT_FALSE(lane.processor->borrowSubordinateResponseRow(0));EXPECT_EQ(lane.cache->stats().all_pinned_responses,1u);
    EXPECT_EQ(old->generation(),oldGeneration);old.reset();lane.consume(3);lane.poison();
    lane.publish(4,4,miss,read);ASSERT_EQ(lane.processor->poll(4),ImmutableLookupPoll::Completed);lane.expectRows(miss);
    auto replacement=lane.processor->borrowSubordinateResponseRow(0);ASSERT_TRUE(replacement);EXPECT_GT(replacement->generation(),oldGeneration);
    EXPECT_EQ(lane.cache->stats().resident_rows,2u);
    const auto before=lane.provider->retainedOwnerObservation();lane.processor.reset();lane.provider.reset();
    EXPECT_EQ(other->bytes().size(),lane.layout.rowBytes());EXPECT_EQ(replacement->bytes().size(),lane.layout.rowBytes());
    EXPECT_EQ(lane.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),before.tableStorageBytes());
    EXPECT_EQ(lane.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
        PhysicalMemoryMaterializationKind::NewAllocation),before.lookupStagingBytes()+lane.channelLease.bytes());
    other.reset();replacement.reset();
    EXPECT_EQ(lane.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(lane.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
        PhysicalMemoryMaterializationKind::NewAllocation),lane.channelLease.bytes());
}

/** @test Zero GPU/RAM capacities retain this explicit mode and exact live traffic, including empty batches and duplicates. */
TEST(ImmutableLookupSubordinate, ZeroSlotsDuplicatesAndEmptyHaveExactLiveExtents){
    for(auto gpu:{DeviceId::cuda(0),DeviceId::rocm(0)})for(auto gpuSlots:{0u,2u}){
        Lane lane(gpu,0,gpuSlots);const std::array<std::int64_t,4> ids{45,3,45,127};
        const std::array<Intent,4> intents{{{Operation::ReadAndInvalidate,101},{Operation::Read,0},{Operation::ReadAndInvalidate,101},{Operation::ReadAndInvalidate,103}}};
        lane.publish(1,1,ids,intents);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);lane.expectRows(ids);
        EXPECT_EQ(lane.cache->stats().source_rows,3u);EXPECT_EQ(lane.cache->stats().duplicate_rows,1u);
        EXPECT_EQ(lane.cache->stats().resident_rows,0u);EXPECT_EQ(lane.cache->stats().invalidations,0u);
        EXPECT_EQ(lane.invalidations()[0].destination_grant,101u);EXPECT_EQ(lane.invalidations()[1].destination_grant,0u);
        EXPECT_EQ(lane.invalidations()[2].destination_grant,101u);EXPECT_EQ(lane.invalidations()[3].destination_grant,103u);
        auto stats=lane.processor->stats();EXPECT_EQ(stats.nativeBytes,ids.size()*lane.layout.rowBytes());
        EXPECT_EQ(stats.idBytes,ids.size()*sizeof(std::int64_t));
        EXPECT_EQ(stats.metadataBytes,kLookupRequestLiveBytes+kLookupCompletionLiveBytes+ids.size()*(sizeof(Intent)+sizeof(ImmutableRowInvalidationAcknowledgement)));
        EXPECT_EQ(lane.provider->mode(),HostImmutableRowMode::AdaptiveRAMSSD);
        lane.consume(1);lane.poison();lane.publish(2,2,{},{});ASSERT_EQ(lane.processor->poll(2),ImmutableLookupPoll::Completed);lane.expectRows({});
        const auto empty=lane.processor->stats();EXPECT_EQ(empty.nativeBytes,stats.nativeBytes);
        EXPECT_EQ(empty.metadataBytes-stats.metadataBytes,kLookupRequestLiveBytes+kLookupCompletionLiveBytes);
        EXPECT_EQ(ImmutableLookupProtocol::validateCompletion(lane.completion(),2,2,2,17,0,lane.layout.rowBytes(),lane.layout.protocol()),ImmutableLookupStatus::Success);
        EXPECT_EQ(ImmutableLookupProtocol::validateCompletion(lane.completion(),2,2,2,17,0,lane.layout.rowBytes()),ImmutableLookupStatus::InvalidProtocol);
    }
}

/** @test All malformed bodies reject before native output, ACK, source/cache work or policy mutation. */
TEST(ImmutableLookupSubordinate, InvalidFramesCannotMutatePayloadOrCpuResidency){
    for(int variant=0;variant<14;++variant){
        SCOPED_TRACE(variant);Lane lane;std::array<std::int64_t,2> ids{3,4};std::array<Intent,2> intents{};
        switch(variant){
        case 0:intents[1].intent=static_cast<Operation>(77);break;
        case 1:intents[1].destination_grant=9;break;
        case 2:intents[1].intent=Operation::ReadAndInvalidate;break;
        case 3:ids[1]=3;intents[1]={Operation::ReadAndInvalidate,9};break;
        case 4:ids[1]=3;intents={{{Operation::ReadAndInvalidate,9},{Operation::ReadAndInvalidate,10}}};break;
        case 5:intents={{{Operation::ReadAndInvalidate,9},{Operation::ReadAndInvalidate,9}}};break;
        case 6:ids[1]=-1;break;
        case 7:ids[1]=128;break;
        default:break;
        }
        lane.publish(1,1,ids,intents);
        if(variant==8){lane.request().protocol=kImmutableLookupProtocol;lane.request().protocolComplement=~kImmutableLookupProtocol;}
        if(variant==9){lane.request().table=29;lane.request().tableComplement=~std::uint64_t{29};}
        if(variant==10){lane.request().generation=2;lane.request().generationComplement=~std::uint64_t{2};}
        if(variant==11){lane.request().idCount=9;lane.request().idCountComplement=~std::uint64_t{9};lane.request().idBytes=72;lane.request().idBytesComplement=~std::uint64_t{72};}
        if(variant==12){lane.request().rowBytes=1;lane.request().rowBytesComplement=~std::uint64_t{1};}
        if(variant==13)lane.request().sequenceComplement=0;
        const auto before=lane.cache->stats();const auto source=lane.provider->sourceStats();
        ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);
        EXPECT_NE(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::Success));
        lane.expectRows({});for(const auto &ack:lane.invalidations())EXPECT_EQ(ack.destination_grant,0xa5a5a5a5a5a5a5a5ULL);
        EXPECT_EQ(lane.cache->stats().completed_batches,before.completed_batches);EXPECT_EQ(lane.cache->stats().failures,before.failures);
        EXPECT_EQ(lane.cache->stats().resident_rows,before.resident_rows);EXPECT_EQ(lane.cache->stats().invalidations,before.invalidations);
        EXPECT_EQ(lane.provider->sourceStats().source_metadata_checks,source.source_metadata_checks);
        EXPECT_NO_THROW(lane.provider->authenticateSource()); // Validation failure is a failed channel, not an I/O-poisoned source.
        EXPECT_THROW((void)lane.processor->poll(1),std::runtime_error);
    }
}

/** @test Future/regressed consumed ACKs cannot release a different outstanding batch; premature reuse cannot fill. */
TEST(ImmutableLookupSubordinate, StaleAcknowledgementsAndUnconsumedReuseAreFatal){
    for(auto badAck:{0u,2u}){
        Lane lane;const std::array<std::int64_t,1> id{3};const std::array<Intent,1> intent{};
        lane.publish(1,1,id,intent);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);
        lane.consume(badAck);lane.publish(2,2,id,intent);
        const auto before=lane.cache->stats().completed_batches;
        EXPECT_THROW((void)lane.processor->poll(2),std::runtime_error);
        EXPECT_EQ(lane.cache->stats().completed_batches,before);
        EXPECT_THROW((void)lane.processor->borrowSubordinateResponseRow(0),std::logic_error);
    }
    Lane lane;const std::array<std::int64_t,1> id{3};const std::array<Intent,1> intent{};
    lane.publish(1,1,id,intent);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);lane.consume(1);
    lane.publish(2,2,id,intent);ASSERT_EQ(lane.processor->poll(2),ImmutableLookupPoll::Completed);
    lane.consume(0);std::atomic_ref(lane.ack().closedGeneration).store(2,std::memory_order_release);
    EXPECT_THROW((void)lane.processor->closed(2),std::runtime_error);
}

/** @test Cancellation bars later capabilities without claiming CPU cancellation can prevent an already-visible GPU completion. */
TEST(ImmutableLookupSubordinate, CancellationIsStickyAndPreservesOldReaderBacking){
    Lane lane;const std::array<std::int64_t,1> id{3};const std::array<Intent,1> intent{};
    lane.publish(1,1,id,intent);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);
    auto old=lane.processor->borrowSubordinateResponseRow(0);ASSERT_TRUE(old);
    lane.cancelled.store(true,std::memory_order_release);
    EXPECT_THROW((void)lane.processor->borrowSubordinateResponseRow(0),std::logic_error);
    EXPECT_THROW((void)lane.processor->poll(1),std::runtime_error);
    lane.cancelled.store(false,std::memory_order_release);EXPECT_THROW((void)lane.processor->poll(1),std::runtime_error);
    EXPECT_EQ(old->bytes().size(),lane.layout.rowBytes());EXPECT_EQ(lane.cache->stats().completed_batches,1u);
    EXPECT_EQ(lane.processor->lastStatus(),ImmutableLookupStatus::Cancelled);
}

/** @test Source failure poisons this same CPU authority; old warm pins and future requests cannot bypass it. */
TEST(ImmutableLookupSubordinate, FailedSourcePublishesNoPromotionAckAndPoisonsPins){
    Lane lane;const std::array<std::int64_t,1> id{3};const std::array<Intent,1> read{};
    lane.publish(1,1,id,read);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);
    auto pin=lane.processor->borrowSubordinateResponseRow(0);ASSERT_TRUE(pin);lane.consume(1);lane.poison();
    ASSERT_EQ(::ftruncate(lane.file.fd,0),0);const std::array<Intent,1> promote{{{Operation::ReadAndInvalidate,123}}};
    lane.publish(2,2,id,promote);ASSERT_EQ(lane.processor->poll(2),ImmutableLookupPoll::Completed);
    EXPECT_EQ(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::StorageFailure));
    EXPECT_EQ(lane.invalidations()[0].destination_grant,0xa5a5a5a5a5a5a5a5ULL);lane.expectRows({});
    EXPECT_EQ(lane.cache->stats().completed_batches,1u);EXPECT_EQ(lane.cache->stats().invalidations,0u);
    EXPECT_THROW((void)pin->bytes(),std::runtime_error);
    EXPECT_THROW(lane.provider->authenticateSource(),std::runtime_error);
    EXPECT_THROW((void)lane.processor->poll(2),std::runtime_error);
}

/** @test A deterministic GPU ACK/refill between the actual acquire and pin operations cannot change the borrowed source row. */
TEST(ImmutableLookupSubordinate, AcquiredGrantUsesOwnedRowsAcrossAckAndMappedFrameOverwrite){
    Lane lane;const std::array<std::int64_t,2> seed{3,4};const std::array<Intent,2> reads{};
    lane.publish(1,1,seed,reads);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);
    auto original=lane.processor->borrowSubordinateResponseRow(0);ASSERT_TRUE(original);
    const auto generation=original->generation();
    auto old=ImmutableLookupSubordinateTestAccess::borrowAcross(*lane.processor,0,[&]{
        // Even an illicit early shared-frame overwrite cannot alter CPU grant metadata.
        auto *ids=reinterpret_cast<std::int64_t*>(lane.bytes.get()+lane.layout.idsOffset());ids[0]=4;
        auto *intents=reinterpret_cast<Intent*>(lane.bytes.get()+lane.layout.intentsOffset());intents[0]={Operation::ReadAndInvalidate,999};
    });
    ASSERT_TRUE(old);EXPECT_EQ(old->row(),3u);EXPECT_EQ(old->generation(),generation);
    EXPECT_THROW((void)ImmutableLookupSubordinateTestAccess::borrowAcross(*lane.processor,0,[&]{
        lane.consume(1);const std::array<std::int64_t,2> next{7,7};const std::array<Intent,2> read{};
        lane.publish(2,2,next,read);
    }),std::logic_error);
    EXPECT_EQ(original->row(),3u);EXPECT_EQ(original->generation(),generation);
    EXPECT_EQ(original->bytes().size(),lane.layout.rowBytes());
    EXPECT_EQ(lane.cache->stats().pinned_rows,1u); // No tentative leaked pin or redirected row7 grant.
    EXPECT_EQ(lane.processor->poll(2),ImmutableLookupPoll::Completed);
    auto next=lane.processor->borrowSubordinateResponseRow(0);ASSERT_TRUE(next);EXPECT_EQ(next->row(),7u);
    EXPECT_EQ(original->row(),3u);EXPECT_EQ(original->generation(),generation);
}

/** @test An acquired response cannot be reused under a later admission until its original generation closes its read grant. */
TEST(ImmutableLookupSubordinate, OutstandingGrantRejectsChangedAdmissionBeforeAnyNewPublication){
    Lane lane;const std::array<std::int64_t,1> id{3};const std::array<Intent,1> read{};
    lane.publish(1,1,id,read);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);
    EXPECT_THROW((void)lane.processor->poll(2),std::runtime_error);
    EXPECT_EQ(lane.processor->lastStatus(),ImmutableLookupStatus::StaleGeneration);
    EXPECT_THROW((void)lane.processor->borrowSubordinateResponseRow(0),std::logic_error);
    EXPECT_EQ(lane.cache->stats().completed_batches,1u);
}

/** @test A later successful CPU poll may reuse its bounded row bank but cannot turn an old acquire into a new row permit. */
TEST(ImmutableLookupSubordinate, OldHostBorrowRejectsAfterNewSuccessfulPollReusesGrantedRowBank){
    Lane lane;const std::array<std::int64_t,1> first{3};const std::array<Intent,1> read{};
    lane.publish(1,1,first,read);ASSERT_EQ(lane.processor->poll(1),ImmutableLookupPoll::Completed);
    auto original=lane.processor->borrowSubordinateResponseRow(0);ASSERT_TRUE(original);
    const auto oldGeneration=original->generation();
    EXPECT_THROW((void)ImmutableLookupSubordinateTestAccess::borrowAcross(*lane.processor,0,[&]{
        lane.consume(1);const std::array<std::int64_t,1> second{7};lane.poison();
        lane.publish(2,2,second,read);EXPECT_EQ(lane.processor->poll(2),ImmutableLookupPoll::Completed);
    }),std::logic_error);
    auto current=lane.processor->borrowSubordinateResponseRow(0);ASSERT_TRUE(current);EXPECT_EQ(current->row(),7u);
    EXPECT_EQ(original->row(),3u);EXPECT_EQ(original->generation(),oldGeneration);
    EXPECT_EQ(original->bytes().size(),lane.layout.rowBytes());
}

/** @test Exact mandatory staging remains a canonical PMA obligation, including zero native RAM capacity. */
TEST(ImmutableLookupSubordinate, SourceAndGrantScratchRequireExactPmaAndNoForeignSource){
    NativeFile file;
    for(auto ramSlots:{0u,2u}){
        const auto layout=layoutFor(file,DeviceId::cuda(0),ramSlots,0);
        const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,{0,DeviceId::cuda(0)});
        PhysicalMemoryPlanBuilder shortPlan;
        shortPlan.add(hostResource(),PhysicalMemoryOwner::ImmutableTableStorage,geometry.storageBytes()+geometry.sourceBytes());
        shortPlan.add(hostResource(),PhysicalMemoryOwner::ImmutableLookupStaging,geometry.executionBytes()-1);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(shortPlan.build()),0);
        EXPECT_THROW((void)CpuAdaptiveImmutableRowCache::create(file.source,geometry,memory),std::exception);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        auto foreign=file.source;foreign.rows-=1;
        EXPECT_THROW((void)CpuAdaptiveImmutableRowCache::create(foreign,geometry,memory),std::invalid_argument);
    }
}

namespace {
/** @brief CPU-only mapped allocator; no vendor runtime, kernel or GPU memory query is used. */
class WorkerMappedBackend final : public llaminar2::test::MockBackend {
public:
    WorkerMappedBackend():MockBackend(DeviceType::CUDA) {}
    /** @brief Materialize the real TransferEngine request with the wire's required alignment. */
    void *allocateMapped(std::size_t bytes,int,void **alias) override {
        if(address)throw std::logic_error("Worker fixture allocated its one mapped channel twice");
        address=std::aligned_alloc(64,(bytes+63)/64*64);
        if(alias)*alias=address;
        return address;
    }
    /** @brief Count actual final mapping destruction, independently of CPU worker completion. */
    void freeMapped(void *pointer,int) override {
        if(pointer!=address)std::terminate();
        std::free(address);address=nullptr;++frees;
    }
    void *address=nullptr;unsigned frees=0;
};
WorkerMappedBackend *workerBackend=nullptr;
/** @return One explicitly installed hardware-free mapping endpoint. */
IBackend *resolveWorkerMapping(DeviceId device) {
    return device==DeviceId::cuda(0)?workerBackend:nullptr;
}
/** @brief Actual PMA provider, service, mapped backing and persistent worker with simulated GPU wire publications. */
struct ServiceLane {
    WorkerMappedBackend backend;
    TransferEngine transfers{&resolveWorkerMapping};
    NativeFile file;
    ImmutableLookupLayout layout=ImmutableLookupLayout::createSubordinate(8,file.source.rowBytes());
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<HostImmutableRowProvider> provider;
    std::shared_ptr<ImmutableLookupService> service;
    std::optional<ImmutableSubordinateLookupBinding> binding;
    /** @brief Use the production service create/arm/poll/retire/await path with original allocator identities. */
    explicit ServiceLane(std::size_t gpuSlots=2,std::size_t ramSlots=2) {
        if(workerBackend)throw std::logic_error("Overlapping serial mapping fixture");
        workerBackend=&backend;
        try {
            const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(
                layoutFor(file,DeviceId::cuda(0),ramSlots,gpuSlots),hostId,{0,DeviceId::cuda(0)});
            PhysicalMemoryPlanBuilder plan(geometry.bom(hostResource()));
            plan.add(hostResource(),PhysicalMemoryOwner::ImmutableLookupStaging,ImmutableLookupService::hostBytes(layout));
            memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
            provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(file.source,geometry,memory));
            const std::array endpoints{DeviceId::cuda(0)};
            service=ImmutableLookupService::create(provider,layout,endpoints,memory,DeviceId::cpu(),transfers);
            binding.emplace(service->bindSubordinate(endpoints[0]));
        } catch(...) {workerBackend=nullptr;throw;}
    }
    /** @brief Final captured binding retires before its mapped backend and source fixture. */
    ~ServiceLane(){binding.reset();service.reset();provider.reset();workerBackend=nullptr;}
    /** @return Exact shared metadata retained by the original source binding. */
    ImmutableLookupAcknowledgement &ack(){return *binding->deviceView().base.acknowledgement;}
    /** @return Original acquired-completion channel, never a copied successful observation. */
    ImmutableLookupCompletion &completion(){return *binding->deviceView().base.completion;}
    /** @brief Release an exact live request through the real protocol2 ABI. */
    void publish(std::uint64_t sequence,std::uint64_t generation,std::span<const std::int64_t> ids,
                 std::span<const Intent> intents) {
        auto &view=binding->deviceView();
        if(ids.size()!=intents.size() || ids.size()>layout.maximumIds())throw std::invalid_argument("Invalid service fixture frame");
        if(!ids.empty()) {
            std::memcpy(view.base.ids,ids.data(),ids.size_bytes());
            std::memcpy(view.intents,intents.data(),intents.size_bytes());
        }
        ImmutableLookupRequest request;
        request.sequenceComplement=~sequence;request.generation=generation;request.generationComplement=~generation;
        request.table=view.base.table;request.tableComplement=~request.table;
        request.idCount=ids.size();request.idCountComplement=~request.idCount;
        request.idBytes=ids.size_bytes();request.idBytesComplement=~request.idBytes;
        request.rowBytes=layout.rowBytes();request.rowBytesComplement=~request.rowBytes;
        request.protocol=layout.protocol();request.protocolComplement=~request.protocol;
        std::memcpy(reinterpret_cast<std::byte*>(view.base.request)+8,reinterpret_cast<const std::byte*>(&request)+8,kLookupRequestLiveBytes-8);
        std::atomic_ref(view.base.request->publication).store(sequence,std::memory_order_release);
    }
    /** @brief Acquire one actual worker publication with a bounded test-only wait. */
    static void wait(std::uint64_t &word,std::uint64_t expected) {
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(std::atomic_ref(word).load(std::memory_order_acquire)!=expected) {
            if(std::chrono::steady_clock::now()>=until)throw std::runtime_error("Actual immutable worker publication did not retire");
            std::this_thread::yield();
        }
    }
    /** @brief Simulate complete consumer retirement and graph terminal, never before actual completion. */
    void close(std::uint64_t sequence,std::uint64_t generation) {
        std::atomic_ref(ack().consumedSequence).store(sequence,std::memory_order_release);
        std::atomic_ref(ack().closedGeneration).store(generation,std::memory_order_release);
    }
};
}

/** @test Cancellation without any acquired completion releases an exact worker witness and keeps captured backing alive. */
TEST(ImmutableLookupSubordinate, WorkerCancelBeforePublicationAndBeforePollRetiresExactInvocation) {
    for(auto gpuSlots:{0u,2u})for(bool publishAfterCancel:{false,true}) {
        ServiceLane lane(gpuSlots);
        const auto retained=lane.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation);
        ASSERT_GE(retained,ImmutableLookupService::hostBytes(lane.layout));
        const auto ticket=lane.service->arm(17);
        EXPECT_FALSE(lane.service->cancel({ticket.generation+1,ticket.invocation}));
        EXPECT_FALSE(lane.service->cancel({ticket.generation,ticket.invocation+1}));
        ASSERT_TRUE(lane.service->cancel(ticket));
        // A queued GPU producer can publish after CPU cancellation. Wait for
        // actual retirement first so the before-poll interleaving is exact,
        // independent of host thread scheduling, and cannot fabricate a reply.
        ServiceLane::wait(lane.ack().retiredInvocation,ticket.invocation);
        if(publishAfterCancel) {
            const std::array<std::int64_t,1> ids{3};const std::array<Intent,1> read{};
            lane.publish(1,ticket.generation,ids,read);
        }
        ServiceLane::wait(lane.ack().retiredInvocation,ticket.invocation);
        EXPECT_TRUE(ImmutableLookupProtocol::matchesInvocation(lane.ack().retiredInvocation,
            lane.ack().retiredGeneration,ticket.generation,ticket.invocation));
        EXPECT_EQ(lane.completion().publication,0u); // Retirement never manufactures Consumed.
        EXPECT_EQ(lane.ack().consumedSequence,0u);
        EXPECT_FALSE(lane.service->cancel(ticket)); // Retirement rejects late writes, including await's missing-close path.
        EXPECT_FALSE(lane.service->await(ticket));
        EXPECT_EQ(lane.service->stats().terminalBytes,4*sizeof(std::uint64_t));
        EXPECT_FALSE(ImmutableLookupSubordinateTestAccess::ownsAcquiredGrant(*lane.service));
        EXPECT_FALSE(lane.service->cancel(ticket));
        lane.service.reset();lane.provider.reset();
        EXPECT_EQ(lane.backend.frees,0u); // Captured binding, not worker retirement, owns backing.
        EXPECT_EQ(lane.ack().retiredGeneration,ticket.generation);
        EXPECT_EQ(lane.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),retained);
        lane.binding.reset();EXPECT_EQ(lane.backend.frees,1u);
        EXPECT_EQ(lane.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}

/** @test A completed but unconsumed source grant retires on cancellation; original invalidation tokens remain immutable. */
TEST(ImmutableLookupSubordinate, WorkerCancelAcquiredRequestDropsGrantWithoutFakeConsumption) {
    for(auto gpuSlots:{0u,2u}) {
        ServiceLane lane(gpuSlots);const auto ticket=lane.service->arm(2);
        const std::array<std::int64_t,2> ids{3,4};
        const std::array<Intent,2> intents{{{Operation::Read,0},{Operation::ReadAndInvalidate,0xfedcba9876543210ULL}}};
        lane.publish(1,ticket.generation,ids,intents);ServiceLane::wait(lane.completion().publication,1);
        ASSERT_EQ(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::Success));
        EXPECT_EQ(lane.binding->deviceView().acknowledgements[0].destination_grant,0u);
        EXPECT_EQ(lane.binding->deviceView().acknowledgements[1].destination_grant,intents[1].destination_grant);
        ASSERT_TRUE(lane.service->cancel(ticket));ServiceLane::wait(lane.ack().retiredInvocation,ticket.invocation);
        EXPECT_FALSE(lane.service->await(ticket));
        EXPECT_FALSE(ImmutableLookupSubordinateTestAccess::ownsAcquiredGrant(*lane.service));
        EXPECT_EQ(lane.ack().consumedSequence,0u);
        EXPECT_EQ(lane.completion().publication,1u);
        EXPECT_EQ(lane.binding->deviceView().acknowledgements[1].destination_grant,intents[1].destination_grant);
    }
}

/** @test Request reset reuses one worker/channel and rejects old cancellation/retirement pairs across large-to-small frames. */
TEST(ImmutableLookupSubordinate, WorkerLargeSmallResetRejectsStaleCancellationAndRetirement) {
    ServiceLane lane(0);ImmutableLookupService::Ticket previous{};
    std::uint64_t sequence=0;
    for(const auto count:{8u,1u,0u,3u}) {
        const auto ticket=lane.service->arm(100+sequence);
        if(previous.valid()) {
            EXPECT_FALSE(lane.service->cancel(previous));
            EXPECT_FALSE(ImmutableLookupProtocol::matchesInvocation(lane.ack().retiredInvocation,
                lane.ack().retiredGeneration,ticket.generation,ticket.invocation));
            EXPECT_FALSE(ImmutableLookupProtocol::matchesInvocation(ticket.invocation,
                previous.generation,ticket.generation,ticket.invocation));
        }
        std::array<std::int64_t,8> ids{0,1,2,3,4,5,6,7};std::array<Intent,8> intents{};
        ++sequence;lane.publish(sequence,ticket.generation,std::span(ids).first(count),std::span(intents).first(count));
        ServiceLane::wait(lane.completion().publication,sequence);
        EXPECT_EQ(lane.completion().idCount,count);EXPECT_EQ(lane.completion().nativeBytes,count*lane.layout.rowBytes());
        lane.close(sequence,ticket.generation);
        EXPECT_TRUE(lane.service->await(ticket));
        EXPECT_FALSE(ImmutableLookupSubordinateTestAccess::ownsAcquiredGrant(*lane.service));
        EXPECT_TRUE(ImmutableLookupProtocol::matchesInvocation(lane.ack().retiredInvocation,
            lane.ack().retiredGeneration,ticket.generation,ticket.invocation));
        previous=ticket;
    }
    EXPECT_EQ(lane.backend.frees,0u);
}

/** @test Concurrent publication/cancellation permits either truthful completion or absence, and never invents consumption. */
TEST(ImmutableLookupSubordinate, WorkerCancelPublicationRaceHasOnlyTruthfulTerminalOutcomes) {
    for(unsigned repeat=0;repeat<20;++repeat) {
        ServiceLane lane(repeat%2?0:2);const auto ticket=lane.service->arm(9);
        const std::array<std::int64_t,2> ids{11,12};
        const std::array<Intent,2> intents{{{Operation::Read,0},{Operation::ReadAndInvalidate,0x123456789abcdef0ULL}}};
        lane.publish(1,ticket.generation,ids,intents);
        ASSERT_TRUE(lane.service->cancel(ticket));
        ServiceLane::wait(lane.ack().retiredInvocation,ticket.invocation);
        EXPECT_EQ(lane.ack().retiredGeneration,ticket.generation);
        const auto acquired=std::atomic_ref(lane.completion().publication).load(std::memory_order_acquire);
        if(acquired) {
            const auto status=ImmutableLookupProtocol::validateCompletion(lane.completion(),acquired,1,ticket.generation,
                lane.binding->deviceView().base.table,ids.size(),lane.layout.rowBytes(),lane.layout.protocol());
            EXPECT_TRUE(status==ImmutableLookupStatus::Success || status==ImmutableLookupStatus::Cancelled);
            EXPECT_EQ(lane.completion().nativeBytes,status==ImmutableLookupStatus::Success?ids.size()*lane.layout.rowBytes():0u);
        }
        EXPECT_EQ(lane.ack().consumedSequence,0u);
        EXPECT_FALSE(lane.service->await(ticket));
        EXPECT_FALSE(ImmutableLookupSubordinateTestAccess::ownsAcquiredGrant(*lane.service));
        EXPECT_EQ(lane.binding->deviceView().intents[1].destination_grant,intents[1].destination_grant);
    }
}

/** @test The actual SSD worker can copy a payload prefix and fail without releasing any promotion ACK or successful bytes. */
TEST(ImmutableLookupSubordinate, WorkerFailedPartialPayloadRetainsOriginalIntentAndBacking) {
    struct DeathStyle {
        std::string previous=GTEST_FLAG_GET(death_test_style);
        /** @brief Isolate the address-space bound from other test workers in a fresh process. */
        DeathStyle(){GTEST_FLAG_SET(death_test_style,"threadsafe");}
        /** @brief Restore process-wide GTest policy. */
        ~DeathStyle(){GTEST_FLAG_SET(death_test_style,previous);}
    } deathStyle;
    ASSERT_EXIT(([] {
        ServiceLane lane(0,0);
        const std::array<std::int64_t,1> first{0};const std::array<Intent,1> read{};
        const auto warm=lane.service->arm(1);lane.publish(1,1,first,read);
        ServiceLane::wait(lane.completion().publication,1);lane.close(1,1);
        if(!lane.service->await(warm))::_exit(2);
        auto &view=lane.binding->deviceView();
        std::memset(view.base.nativeRows,0xa5,2*lane.layout.rowBytes());
        std::memset(view.acknowledgements,0xa5,2*sizeof(ImmutableRowInvalidationAcknowledgement));
        const std::array<std::int64_t,2> ids{0,64};
        const std::array<Intent,2> intents{{{Operation::ReadAndInvalidate,0x100000001ULL},
                                          {Operation::ReadAndInvalidate,0xfedcba9876543210ULL}}};
        const auto ticket=lane.service->arm(2);
        rlimit saved{};
        if(::getrlimit(RLIMIT_AS,&saved)!=0)::_exit(3);
        auto *statm=std::fopen("/proc/self/statm","r");unsigned long long pages=0;
        if(!statm)::_exit(4);
        const auto parsed=std::fscanf(statm,"%llu",&pages);std::fclose(statm);
        const auto pageBytes=::sysconf(_SC_PAGESIZE);
        if(parsed!=1 || pageBytes<=0 || pages>std::numeric_limits<rlim_t>::max()/static_cast<unsigned long long>(pageBytes))::_exit(5);
        rlimit bounded=saved;bounded.rlim_cur=pages*static_cast<rlim_t>(pageBytes);
        if(::setrlimit(RLIMIT_AS,&bounded)!=0)::_exit(6);
        lane.publish(2,2,ids,intents);
        ServiceLane::wait(lane.ack().retiredInvocation,ticket.invocation);
        if(::setrlimit(RLIMIT_AS,&saved)!=0)std::terminate();
        EXPECT_EQ(lane.completion().publication,2u);
        EXPECT_EQ(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::StorageFailure));
        EXPECT_EQ(lane.completion().nativeBytes,0u);
        EXPECT_EQ(std::memcmp(view.base.nativeRows,lane.file.bytes.data()+lane.file.source.file_offset,lane.layout.rowBytes()),0);
        for(std::size_t i=lane.layout.rowBytes();i<2*lane.layout.rowBytes();++i)EXPECT_EQ(view.base.nativeRows[i],std::byte{0xa5});
        for(std::size_t i=0;i<2;++i) {
            EXPECT_EQ(view.acknowledgements[i].destination_grant,0xa5a5a5a5a5a5a5a5ULL);
            EXPECT_EQ(view.intents[i].destination_grant,intents[i].destination_grant);
        }
        lane.close(2,2);EXPECT_FALSE(lane.service->await(ticket));
        EXPECT_FALSE(ImmutableLookupSubordinateTestAccess::ownsAcquiredGrant(*lane.service));
        lane.service.reset();lane.provider.reset();EXPECT_EQ(lane.backend.frees,0u);
        lane.binding.reset();EXPECT_EQ(lane.backend.frees,1u);
        ::_exit(::testing::Test::HasFailure()?1:0);
    }()),::testing::ExitedWithCode(0),"");
}

/** @test Late cancellation before await cannot publish uncounted metadata after the actual CPU worker retires. */
TEST(ImmutableLookupSubordinate, WorkerLateCancelBeforeAwaitRejectsWithoutPublication) {
    ServiceLane lane(0);const auto ticket=lane.service->arm(71);
    const std::array<std::int64_t,1> ids{7};const std::array<Intent,1> read{};
    lane.publish(1,ticket.generation,ids,read);ServiceLane::wait(lane.completion().publication,1);
    ASSERT_EQ(lane.completion().status,static_cast<std::uint64_t>(ImmutableLookupStatus::Success));
    lane.close(1,ticket.generation);ServiceLane::wait(lane.ack().retiredInvocation,ticket.invocation);
    EXPECT_FALSE(lane.service->cancel(ticket));
    EXPECT_FALSE(lane.service->cancel(ticket));
    EXPECT_EQ(lane.ack().cancelledInvocation,0u);EXPECT_EQ(lane.ack().cancelledGeneration,0u);
    EXPECT_TRUE(lane.service->await(ticket));
    EXPECT_EQ(lane.service->stats().terminalBytes,3*sizeof(std::uint64_t));
    EXPECT_FALSE(ImmutableLookupSubordinateTestAccess::ownsAcquiredGrant(*lane.service));
    const auto next=lane.service->arm(72);
    EXPECT_FALSE(lane.service->cancel(ticket));
    EXPECT_TRUE(lane.service->cancel(next));ServiceLane::wait(lane.ack().retiredInvocation,next.invocation);
    EXPECT_FALSE(lane.service->cancel(next));EXPECT_FALSE(lane.service->await(next));
    EXPECT_EQ(lane.service->stats().terminalBytes,7*sizeof(std::uint64_t));
}
