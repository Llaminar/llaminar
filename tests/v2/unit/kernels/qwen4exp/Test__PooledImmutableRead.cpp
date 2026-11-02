/**
 * @file Test__PooledImmutableRead.cpp
 * @brief Actual hit-read/body replay over original decoded CPU phase roots.
 *
 * The original physical/coordinator operations issue grants and Ready. Test
 * transport observations supply acquired bytes/counts; CPU byte loops execute
 * the same address functions as native leaves, and certify no GPU transport,
 * source provider, PMA materialization or native factory capability.
 */
#include "kernels/qwen4exp/PooledImmutableReadOps.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "fixtures/qwen4exp/PooledImmutableReadFixture.h"
#include <bit>
namespace llaminar2::test {
namespace {
/** @brief Deliver one original owner request into its actual remote receive/count banks. */
std::vector<p::Grant> receiveHits(Pool& pool,std::uint32_t coordinate,const n::SourceFrontier& source){
    auto& owner=*pool.owners[coordinate];auto& x=owner.args;std::vector<p::Grant> records;
    for(std::uint32_t unique=0;unique<pool.view().frame->unique_rows;++unique)
        if(pool.view().unique[unique].phase==p::UniquePhase::Read && pool.view().unique[unique].grant.owner==coordinate)
            records.push_back(pool.view().unique[unique].grant);
    auto* ingress=n::at<n::OwnerIngress>(x.native,x.plan.native.ingress);
    ingress->invocation=*pool.invocation();ingress->record_bytes=records.size()*sizeof(p::Grant);ingress->payload_bytes=~std::uint64_t{0};
    if(!records.empty())std::memcpy(x.frame+x.plan.frame.commands,records.data(),records.size()*sizeof(p::Grant));
    x.input={&ingress->invocation,x.plan.native.reply_capacity?x.frame+x.plan.frame.commands:nullptr,
        x.plan.native.reply_capacity?&ingress->record_bytes:nullptr,nullptr,nullptr,nullptr,n::SourceCondition::AfterSuccess,&source};
    return records;
}
/** @brief Execute the actual native byte-address body for precisely its produced live payload. */
void copyHitPrefix(const n::OwnerArgs& a){
    const auto bytes=n::at<n::OwnerIngress>(a.native,a.plan.native.ingress)->payload_bytes;
    if(n::progress(a).delivery==n::Delivery::Validated)
        for(std::uint64_t byte=0;byte<bytes;++byte)a.native[a.plan.native.source_payload+byte]=*n::hitReadByte(a,byte);
}
}
TEST(PooledImmutableRead, ActualOwnerHitsPreserveOriginalGrantsNativeBytesAndC0){
    Pool pool(std::array{3u,0u,3u});pool.complete(std::array<std::uint64_t,4>{101,102,103,104});
    pool.begin(std::array<std::uint64_t,6>{104,101,104,105,102,101});pool.wave<false>();pool.wave<true>();
    pool.finish([&](const n::SourceFrontier& source){
        std::uint64_t total=0;
        for(std::uint32_t coordinate=0;coordinate<pool.owners.size();++coordinate){
            auto& x=pool.owners[coordinate]->args;const auto records=receiveHits(pool,coordinate,source);
            const auto pins=n::ownerView<a::Slot64>(x).frame->initial_pin_count;
            std::vector<p::Grant> original(n::at<p::Grant>(x.frame,x.plan.frame.initial_pins),n::at<p::Grant>(x.frame,x.plan.frame.initial_pins)+pins);
            ASSERT_TRUE((n::owner<n::OwnerAction::ReadRows,a::Slot64>(x)));
            const auto bytes=n::at<n::OwnerIngress>(x.native,x.plan.native.ingress)->payload_bytes;
            EXPECT_EQ(bytes,records.size()*x.plan.row_bytes);total+=bytes;copyHitPrefix(x);
            ASSERT_TRUE(n::completeHitRead<a::Slot64>(x));
            const auto& receipt=*n::at<n::HitReadReceipt>(x.native,x.plan.native.header);
            EXPECT_EQ(receipt.version,n::Version);EXPECT_EQ(receipt.kind,n::OwnerHeaderKind::HitRead);
            EXPECT_EQ(receipt.record.status,a::Status::Success);EXPECT_EQ(n::progress(x).header_bytes,64u);
            for(std::size_t row=0;row<records.size();++row)for(std::size_t byte=0;byte<x.plan.row_bytes;++byte)
                EXPECT_EQ(x.native[x.plan.native.source_payload+row*x.plan.row_bytes+byte],std::byte((records[row].row+byte)&255));
            if(pins)EXPECT_EQ(std::memcmp(original.data(),x.frame+x.plan.frame.initial_pins,pins*sizeof(p::Grant)),0);
            x.input={pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
        }
        EXPECT_GT(total,0u);
    });
}
namespace {
namespace sr=q::pooled_source;
/** @brief Decoded retained-source observation and actual original source geometry; never a provider substitute. */
struct ReadReplay {
    Pool& pool;
    q::PooledImmutableSourceGeometry geometry;
    Bytes frame,native;
    int healthy=0,error=0;
    sr::DeviceArgs args;
    static q::PooledImmutableSourceGeometry declare(Pool& pool,GGUFTensorType format,std::uint32_t columns=160){
        std::vector<q::PooledImmutableOwnerNativePlan> owners;for(const auto& owner:pool.owners)owners.push_back(owner->plan);
        return q::PooledImmutableSourceGeometry::fromPlans(owners,pool.plan,
            {"recorded-source","rows",0,pool.args.plan.policy.sourceRows(),columns,format});
    }
    ReadReplay(Pool& x,const n::SourceFrontier& frontier,GGUFTensorType format,std::uint32_t columns=160):pool(x),geometry(declare(x,format,columns)),
        frame(geometry.parameters().offsets.frame_bytes),native(geometry.parameters().offsets.native_bytes),
        args{geometry.parameters(),pool.owners.front()->args,pool.args,frame.data(),native.data(),{},&healthy,&error}{
        const auto local=args.plan.local_owner;if(local!=a::None)args.owner=pool.owners[local]->args;
        // This is the recorded actual source/Ready boundary: payload bytes and
        // source-miss coordinates are observations, not another executing controller.
        sr::protocol(args)={};sr::progress(args)={};sr::frontier(args)=frontier;sr::stage(args)=sr::ResponseStage::Consumed;
        sr::protocol(args).admissionGeneration=sr::protocol(args).generation=17;sr::invocation(args)=3;
        sr::protocol(args).issuedSequence=sr::protocol(args).consumedSequence=frontier.source.sequence;
        const auto participants=pool.args.plan.policy.participants();
        sr::progress(args).publication_headers=sr::progress(args).publication_acquired=sr::progress(args).ready_headers=participants;
        auto* coordinates=n::at<std::uint32_t>(args.frame,args.plan.offsets.coordinates);
        std::uint32_t misses=0;
        for(std::uint32_t unique=0;unique<pool.view().frame->unique_rows;++unique){
            const auto& row=pool.view().unique[unique];coordinates[unique]=a::None;
            if(row.phase!=p::UniquePhase::Read){coordinates[unique]=misses;
                for(std::size_t byte=0;byte<args.plan.owner.row_bytes;++byte)
                    args.frame[args.plan.offsets.responses+std::size_t{misses}*args.plan.owner.row_bytes+byte]=std::byte((row.row+byte)&255);
                ++misses;
            }
        }
        sr::protocol(args).idCount=misses;sr::protocol(args).nativeBytes=std::uint64_t{misses}*args.plan.owner.row_bytes;
    }
    /** @brief Replay exact acquired remote original header/count/payload, then actual import address/commit bodies. */
    void import(std::uint32_t owner){
        if(owner==args.plan.local_owner){
            Pool::require(sr::prepareLocalReadImport<a::Slot64>(args,owner));
            const auto bytes=sr::localHitReadCopyBytes(args,owner);
            for(std::uint64_t byte=0;byte<bytes;++byte)*sr::hitReadDestination(args,byte)=*sr::hitReadSource(args,owner,byte);
            Pool::require(sr::completeLocalReadImport<a::Slot64>(args,owner));return;
        }
        auto& x=pool.owners[owner]->args;auto* ingress=n::at<n::OwnerIngress>(x.native,x.plan.native.ingress);
        const auto slice=sr::hitReadSlice(args,owner);ingress->invocation=*pool.invocation();ingress->record_bytes=slice.live_bytes;
        if(slice.live_bytes)std::memcpy(x.frame+x.plan.frame.commands,sr::hitReadRequests(args,owner),slice.live_bytes);
        x.input={&ingress->invocation,x.plan.native.reply_capacity?x.frame+x.plan.frame.commands:nullptr,
            x.plan.native.reply_capacity?&ingress->record_bytes:nullptr,nullptr,nullptr,nullptr,n::SourceCondition::AfterSuccess,&sr::frontier(args)};
        Pool::require(n::owner<n::OwnerAction::ReadRows,a::Slot64>(x));copyHitPrefix(x);(void)n::completeHitRead<a::Slot64>(x);
        std::memcpy(args.native+args.plan.offsets.seal_receipt,x.native+x.plan.native.header,sizeof(n::HitReadReceipt));
        const auto bytes=ingress->payload_bytes;*n::at<std::uint64_t>(args.native,args.plan.offsets.payload_bytes)=bytes;
        if(bytes)std::memcpy(args.frame+args.plan.offsets.payload,x.native+x.plan.native.source_payload,bytes);
        Pool::require(sr::validateReadImport(args,owner));
        if(sr::readImportCopies(args,owner))for(std::uint64_t byte=0;byte<bytes;++byte)*sr::hitReadDestination(args,byte)=*sr::hitReadSource(args,owner,byte);
        Pool::require(sr::completeReadImport(args,owner));x.input={pool.invocation(),nullptr,nullptr,nullptr,nullptr,nullptr};
    }
};
/** @brief Compare assembled original positions against the unchanged production decoder over original native bytes. */
template<TensorType Format>void originalOrder(const ReadReplay& replay){
    const auto& a=replay.args;const auto view=n::coordinatorView(a.coordinator);NativeEmbeddingIQTables tables;
    std::vector<std::byte> expected(a.plan.owner.row_bytes);
    for(std::uint32_t original=0;original<view.frame->original_ids;++original){
        const auto unique=view.original_to_unique[original];const auto row=view.unique[unique].row;
        for(std::size_t byte=0;byte<expected.size();++byte)expected[byte]=std::byte((row+byte)&255);
        const auto* actual=sr::readableRow<a::Slot64>(a,unique);ASSERT_NE(actual,nullptr);
        ASSERT_EQ(std::memcmp(actual,expected.data(),expected.size()),0);
        for(std::uint32_t column=0;column<a.plan.row_elements;++column){
            const auto value=q::lookup_detail::decode<Format>(actual,column,tables),oracle=q::lookup_detail::decode<Format>(expected.data(),column,tables);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(value),std::bit_cast<std::uint32_t>(oracle));
        }
    }
}
}
TEST(PooledImmutableRead, ActualSourceAssemblyPreservesPrivateMissesQ8AndOriginalDuplicateOrder){
    for(const auto format:{GGUFTensorType::IQ4_NL,GGUFTensorType::Q8_0})
    for(const auto capacities:{std::vector<std::uint32_t>{2},std::vector<std::uint32_t>{3,0},std::vector<std::uint32_t>{2,0,2,1},std::vector<std::uint32_t>{1,0,1,0,1,0,1,0}})
    for(const bool local:{false,true}){
        const auto bytes=format==GGUFTensorType::IQ4_NL?90u:170u;
        Pool pool(capacities,64,{65536,1,2},local?std::optional<std::size_t>(0):std::nullopt,format,bytes);
        pool.complete(std::array<std::uint64_t,4>{101,102,103,104});
        const std::array queries{std::vector<std::uint64_t>{104,101,104,105,102,101,106,107},
            std::vector<std::uint64_t>{105,105},std::vector<std::uint64_t>{},std::vector<std::uint64_t>{108,109,110}};
        for(const auto& query:queries){
            pool.begin(query);pool.wave<false>();pool.wave<true>();
            pool.finish([&](const n::SourceFrontier& frontier){
                ReadReplay replay(pool,frontier,format);auto& args=replay.args;
                const auto io=sr::protocol(args);const auto misses=io.nativeBytes;
                std::vector<std::byte> originalMiss(args.frame+args.plan.offsets.responses,args.frame+args.plan.offsets.responses+misses);
                ASSERT_TRUE(sr::planReads(args));
                for(std::uint32_t owner=0;owner<capacities.size();++owner)replay.import(owner);
                ASSERT_TRUE(sr::sealReadRows(args));EXPECT_EQ(sr::stage(args),sr::ResponseStage::RowsReady);
                EXPECT_EQ(std::memcmp(&io,&sr::protocol(args),sizeof(io)),0);
                if(misses)EXPECT_EQ(std::memcmp(originalMiss.data(),args.frame+args.plan.offsets.responses,misses),0);
                if(format==GGUFTensorType::IQ4_NL)originalOrder<TensorType::IQ4_NL>(replay);else originalOrder<TensorType::Q8_0>(replay);
            });
        }
    }
}

TEST(PooledImmutableRead, MalformedOwnerPrefixesRejectBeforePayloadOrPinMutation){
    for(const unsigned fault:{0u,1u,2u,3u,4u,5u}){
        Pool pool(std::array{4u,0u});pool.complete(std::array<std::uint64_t,3>{101,102,103});
        pool.begin(std::array<std::uint64_t,3>{103,101,103});pool.wave<false>();pool.wave<true>();
        struct RejectionObserved {};
        EXPECT_THROW(pool.finish([&](const n::SourceFrontier& frontier){
            auto& x=pool.owners[0]->args;const auto records=receiveHits(pool,0,frontier);ASSERT_FALSE(records.empty());
            const auto payload=x.plan.native.source_payload;std::vector<std::byte> before(x.native+payload,x.native+payload+x.plan.native.reply_capacity*x.plan.row_bytes);
            const auto pins=n::ownerView<a::Slot64>(x).frame->initial_pin_count;std::vector<p::Grant> original(n::at<p::Grant>(x.frame,x.plan.frame.initial_pins),n::at<p::Grant>(x.frame,x.plan.frame.initial_pins)+pins);
            auto* ingress=n::at<n::OwnerIngress>(x.native,x.plan.native.ingress);std::uint64_t foreign=ingress->record_bytes;
            auto saved=*n::at<p::Grant>(x.frame,x.plan.frame.commands);auto source=frontier;
            if(fault==0)++ingress->record_bytes;
            if(fault==1)ingress->record_bytes=std::uint64_t{x.plan.native.reply_capacity+1}*sizeof(p::Grant);
            if(fault==2)x.input.record_bytes=&foreign;
            if(fault==3)n::at<p::Grant>(x.frame,x.plan.frame.commands)->generation++;
            if(fault==4){source.interval.sequence++;x.input.outcome=&source;}
            if(fault==5)x.input.records=x.frame+x.plan.frame.initial_pins;
            EXPECT_FALSE((n::owner<n::OwnerAction::ReadRows,a::Slot64>(x)));
            EXPECT_EQ(ingress->payload_bytes,0u);EXPECT_EQ(std::memcmp(before.data(),x.native+payload,before.size()),0);
            if(pins)EXPECT_EQ(std::memcmp(original.data(),x.frame+x.plan.frame.initial_pins,pins*sizeof(p::Grant)),0);
            // A malformed parser transaction is fatal in native execution;
            // stop this CPU observation before the success-only fixture finish.
            throw RejectionObserved{};
        }),RejectionObserved);
    }
}
TEST(PooledImmutableRead, StaleOrNominallyForeignImportsRejectAtomicallyBeforeMapPublication){
    for(const unsigned fault:{0u,1u,2u,3u,4u,5u,6u}){
        Pool pool(std::array{3u,0u});pool.complete(std::array<std::uint64_t,2>{101,102});pool.begin(std::array<std::uint64_t,3>{102,101,103});pool.wave<false>();pool.wave<true>();
        pool.finish([&](const n::SourceFrontier& frontier){
            ReadReplay replay(pool,frontier,GGUFTensorType::IQ4_NL);auto& a=replay.args;ASSERT_TRUE(sr::planReads(a));
            // Decode the actual first packet once, then rewind only recorded
            // acquired input/cursor state for the hostile parser observation.
            auto coordinates=std::vector<std::byte>(a.frame+a.plan.offsets.coordinates,a.frame+a.plan.offsets.coordinates+pool.view().frame->unique_rows*4);
            const auto rows=sr::progress(a).read_rows;replay.import(0);sr::progress(a).read_owners=0;sr::progress(a).read_rows=rows;
            std::memcpy(a.frame+a.plan.offsets.coordinates,coordinates.data(),coordinates.size());
            auto* receipt=n::at<n::HitReadReceipt>(a.native,a.plan.offsets.seal_receipt);const auto saved=*receipt;
            auto* bytes=n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes);const auto count=*bytes;
            if(fault==0)receipt->version--;
            if(fault==1)receipt->kind=n::OwnerHeaderKind::SourceSeal;
            if(fault==2)receipt->record.interval.sequence++;
            if(fault==3)receipt->record.owner++;
            if(fault==4)++*bytes;
            if(fault==5)receipt->record.source.generation++;
            if(fault==6)receipt->record.status=a::Status::Cancelled;
            EXPECT_FALSE(sr::validateReadImport(a,0));EXPECT_FALSE(sr::completeReadImport(a,0));
            EXPECT_EQ(sr::progress(a).read_owners,0u);EXPECT_EQ(sr::progress(a).read_rows,rows);
            EXPECT_EQ(std::memcmp(coordinates.data(),a.frame+a.plan.offsets.coordinates,coordinates.size()),0);
            *receipt=saved;*bytes=count;ASSERT_TRUE(sr::completeReadImport(a,0));replay.import(1);ASSERT_TRUE(sr::sealReadRows(a));
        });
    }
}
TEST(PooledImmutableRead, CancelledReadDrainsAllFixedMetadataWithoutPublishingRowsOrChangingSourceIO){
    for(const bool cancelAfterPlan:{false,true})for(const bool local:{false,true}){
        Pool pool(std::array{3u,0u},64,{65536,1,2},local?std::optional<std::size_t>(0):std::nullopt);pool.complete(std::array<std::uint64_t,2>{101,102});
        pool.begin(std::array<std::uint64_t,3>{102,101,103});pool.wave<false>();pool.wave<true>();
        pool.finish([&](const n::SourceFrontier& frontier){ReadReplay replay(pool,frontier,GGUFTensorType::IQ4_NL);auto& a=replay.args;
            const auto ids=sr::protocol(a).idCount,issued=sr::protocol(a).issuedSequence,consumed=sr::protocol(a).consumedSequence;
            if(!cancelAfterPlan){replay.error=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);replay.healthy=0;}ASSERT_TRUE(sr::planReads(a));if(cancelAfterPlan){{replay.error=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);replay.healthy=0;}sr::reject(a,q::ImmutableLookupStatus::Cancelled);}
            for(unsigned owner=0;owner<2;++owner)replay.import(owner);ASSERT_TRUE(sr::sealReadRows(a));
            EXPECT_EQ(sr::stage(a),sr::ResponseStage::ReadPlanned);EXPECT_EQ(sr::progress(a).read_owners,2u);
            EXPECT_EQ(sr::protocol(a).idCount,ids);EXPECT_EQ(sr::protocol(a).issuedSequence,issued);EXPECT_EQ(sr::protocol(a).consumedSequence,consumed);
            EXPECT_EQ(sr::readableRow<a::Slot64>(a,0),nullptr);
        });
    }
}
TEST(PooledImmutableRead, OriginalSourceRootOffsetsAndBOMStayWithinExistingNativePadding){
    for(const auto capacities:{std::vector<std::uint32_t>{0},std::vector<std::uint32_t>{2},std::vector<std::uint32_t>{2,0,3,1}})
    for(const bool local:{false,true}){
        Pool pool(capacities,64,{65536,1,2},local?std::optional<std::size_t>(0):std::nullopt);
        const ImmutableRowTableSource source{"recorded-source","rows",0,pool.args.plan.policy.sourceRows(),160,GGUFTensorType::IQ4_NL};
        const auto geometry=ReadReplay::declare(pool,GGUFTensorType::IQ4_NL);const auto before=q::PooledImmutableSourceGeometry::rootOffsets(pool.physical,pool.geometry,source);
        EXPECT_EQ(before,geometry.parameters().offsets);EXPECT_EQ(before.progress,160u);EXPECT_EQ(before.observation,192u);EXPECT_LE(before.progress+sizeof(sr::SourceProgress),before.observation);
        EXPECT_EQ(before.native_bytes,local && capacities.size()==1?320u:384u);
        const auto allocator=pool.physical.coordinator();const PhysicalMemoryResource resource{allocator.world_rank,allocator.device,1ull<<30,1ull<<30};
        EXPECT_EQ(geometry.mandatoryBOM(resource).totalBytes(),q::PooledImmutableSourceGeometry::mandatoryBOM(pool.physical,pool.geometry,source,resource).totalBytes());
        EXPECT_EQ(geometry.mandatoryBOM(resource).totalBytes(),before.frame_bytes+before.native_bytes);
    }
}

TEST(PooledImmutableRead, AllNativeFormatsRetainExactQuantizedRowsAndCodebookBytes){
    const std::array formats{GGUFTensorType::F32,GGUFTensorType::F16,GGUFTensorType::BF16,
        GGUFTensorType::Q4_0,GGUFTensorType::Q4_1,GGUFTensorType::Q5_0,GGUFTensorType::Q5_1,GGUFTensorType::Q8_0,
        GGUFTensorType::Q2_K,GGUFTensorType::Q3_K,GGUFTensorType::Q4_K,GGUFTensorType::Q5_K,GGUFTensorType::Q6_K,GGUFTensorType::Q8_K,
        GGUFTensorType::IQ1_S,GGUFTensorType::IQ1_M,GGUFTensorType::IQ2_XXS,GGUFTensorType::IQ2_XS,GGUFTensorType::IQ2_S,
        GGUFTensorType::IQ3_XXS,GGUFTensorType::IQ3_S,GGUFTensorType::IQ4_NL,GGUFTensorType::IQ4_XS};
    for(const auto format:formats){const ImmutableRowTableSource source{"recorded","rows",0,0x100000100ULL,256,format};
        Pool pool(std::array{2u,0u,1u},64,{65536,1,2},std::nullopt,format,source.rowBytes());pool.complete(std::array<std::uint64_t,3>{101,102,103});
        pool.begin(std::array<std::uint64_t,5>{103,101,103,104,105});pool.wave<false>();pool.wave<true>();pool.finish([&](const n::SourceFrontier& frontier){
            ReadReplay replay(pool,frontier,format,256);auto& args=replay.args;ASSERT_TRUE(sr::planReads(args));for(unsigned owner=0;owner<3;++owner)replay.import(owner);ASSERT_TRUE(sr::sealReadRows(args));
            const auto view=pool.view();for(std::uint32_t unique=0;unique<view.frame->unique_rows;++unique){const auto* row=sr::readableRow<a::Slot64>(args,unique);ASSERT_NE(row,nullptr);for(size_t byte=0;byte<source.rowBytes();++byte)EXPECT_EQ(row[byte],std::byte((view.unique[unique].row+byte)&255));}
        });
    }
}

/** @brief Public remote packing cannot manufacture absent local ingress/payload roots. */
TEST(PooledImmutableRead, PublicNativeReadRejectsLocalWithoutIngressAndKeepsSource17DirectLocal){
    for(const uint32_t capacity:{0u,3u}){
        Pool local(std::array{capacity},64,{65536,1,2},0);const auto& localPlan=local.owners.front()->plan.parameters();
        EXPECT_EQ(localPlan.native.ingress,0u);EXPECT_EQ(localPlan.native.source_payload,0u);
        EXPECT_THROW(q::pooledNativeSchema(n::OwnerAction::ReadRows,localPlan,n::SourceCondition::AfterSuccess),std::invalid_argument);
        Pool remote(std::array{capacity});const auto& remotePlan=remote.owners.front()->plan.parameters();ASSERT_NE(remotePlan.native.ingress,0u);
        EXPECT_NO_THROW(q::pooledNativeSchema(n::OwnerAction::ReadRows,remotePlan,n::SourceCondition::AfterSuccess));
        local.complete(std::array<std::uint64_t,2>{101,102});local.begin(std::array<std::uint64_t,3>{102,101,103});local.wave<false>();local.wave<true>();
        local.finish([&](const n::SourceFrontier& frontier){ReadReplay replay(local,frontier,GGUFTensorType::IQ4_NL);auto& args=replay.args;
            ASSERT_TRUE(sr::planReads(args));ASSERT_TRUE(sr::prepareLocalReadImport<a::Slot64>(args,0));
            const auto bytes=sr::localHitReadCopyBytes(args,0);for(std::uint64_t byte=0;byte<bytes;++byte)*sr::hitReadDestination(args,byte)=*sr::hitReadSource(args,0,byte);
            ASSERT_TRUE(sr::completeLocalReadImport<a::Slot64>(args,0));ASSERT_TRUE(sr::sealReadRows(args));originalOrder<TensorType::IQ4_NL>(replay);
        });
    }
}

}

namespace llaminar2::test {
/** @test Actual pooled predicate follows first error across prefill, response admission and ordered request reset. */
TEST(PooledImmutableRead, CanonicalFirstErrorAllowsColdPrefillAndStaysStickyAcrossResponse){
    using namespace sampling_math;
    std::array<int,kDeviceGenerationControlCount> control{};
    Pool pool(std::array{1u});pool.begin(std::array<std::uint64_t,1>{101});pool.wave<false>();pool.wave<true>();
    pool.finish([&](const n::SourceFrontier& original){
    ReadReplay replay(pool,original,GGUFTensorType::IQ4_NL);auto args=replay.args;
    args.healthy=control.data()+kDeviceGenerationControlOk;args.error=control.data()+kDeviceGenerationControlErrorCode;
    for(unsigned replay=0;replay<20;++replay){
        control.fill(0);const auto cold=control;
        EXPECT_TRUE(sr::forwardValid(args));EXPECT_EQ(control,cold);
        DeviceGenerationInitialization init{};init.forward_failure_policy=DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
        ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::ordinary(),control.data(),DeviceGenerationLeadingRowDisposition::PendingResponse,init));
        EXPECT_TRUE(sr::forwardValid(args));EXPECT_EQ(*args.healthy,1);
        *args.error=static_cast<int>(DeviceGenerationError::ImmutableStorageFailure);*args.healthy=0;
        EXPECT_FALSE(sr::forwardValid(args));
        EXPECT_FALSE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::ordinary(),control.data(),DeviceGenerationLeadingRowDisposition::PendingResponse,init));
        EXPECT_EQ(*args.error,static_cast<int>(DeviceGenerationError::ImmutableStorageFailure));
        EXPECT_FALSE(sr::forwardValid(args));
    }
    args.error=nullptr;EXPECT_FALSE(sr::forwardValid(args));
    });
}
}
