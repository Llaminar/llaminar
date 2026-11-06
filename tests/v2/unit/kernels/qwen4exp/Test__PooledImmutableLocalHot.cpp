/**
 * @file Test__PooledImmutableLocalHot.cpp
 * @brief Replay actual local-frame and canonical cache Operations without devices or source authority.
 *
 * Finite coalesced inputs are recorded controller inputs. Native graph/sort,
 * transport and physical source qualification remain separate explicit tests.
 */
#include "kernels/qwen4exp/PooledImmutableLocalHotOps.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTerminal.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include "planning/ActivationMemoryEstimator.h"
#include "../../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotGeometry.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotReadyProtocol.h"
#include <atomic>
#include <array>
#include <type_traits>
#include "loaders/ImmutableRowTable.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>
namespace llaminar2::qwen4exp {
namespace {
namespace a=adaptive_lookup;
namespace h=local_hot;
namespace t=local_hot_transport;
/** @brief Actual production structs over bounded host replay arrays, never a materialization permit. */
template<class Slot> struct Replay {
    static constexpr std::uint32_t Maximum=32;
    std::vector<Slot> slots;
    std::vector<a::RadixNode> nodes;
    a::OwnerControl owner{};
    a::FrameControl frame{};
    h::FrameIdentity identity{};
    h::Counters counters{};
    std::vector<std::uint32_t> epochs;
    std::vector<std::uint16_t> counts;
    std::vector<a::UniqueRecord> unique=std::vector<a::UniqueRecord>(Maximum);
    std::vector<std::uint32_t> inverse=std::vector<std::uint32_t>(Maximum);
    std::vector<std::uint64_t> misses=std::vector<std::uint64_t>(Maximum);
    std::vector<a::KeyIndex> sorted=std::vector<a::KeyIndex>(Maximum);
    std::vector<std::byte> payload,response;
    std::size_t row_bytes;
    std::uint32_t lane_width=0;
    std::uint64_t source_rows;
    t::ResultHeader result{};
    std::uint64_t header_bytes=0,native_bytes=0;
    /** @brief Cold initialize only the real native metadata algorithm. */
    Replay(std::uint32_t capacity,std::size_t bytes=16,std::uint64_t rows=1024)
        :slots(capacity),nodes(capacity?capacity-1:0),payload(capacity*bytes),response(Maximum*bytes),
          row_bytes(bytes),source_rows(rows)
    {
        if(capacity){lane_width=1;while(lane_width<capacity)lane_width*=2;}
        epochs.resize(4*lane_width);counts.resize(4*lane_width);
        auto c=a::Operations<Slot>(metadata());
        for(std::uint32_t i=0;i<capacity;++i)c.initializeSlot(i);
        for(std::uint32_t i=0;i+1<capacity;++i)c.initializeNode(i);
        c.initializeOwner();c.initializeFrame();
        t::SourceAdmissionHeader admitted;admitted.table=7;admitted.source_storage_generation=11;
        admitted.source_admission=23;admitted.source_invocation=37;
        if(!operations().resetRequest(admitted))throw std::logic_error("Recorded original request reset rejected");
    }
    /** @return The original metadata authority consumed by unchanged cache Operations. */
    a::Metadata<Slot> metadata(){return {slots.data(),nodes.data(),&owner,&frame,epochs.data(),counts.data(),
        std::uint32_t(slots.size()),std::uint32_t(slots.size()/2),lane_width,source_rows,65536};}
    /** @return Exact decoded frame/root aliases for actual production mechanics. */
    h::View<Slot> view(){return {metadata(),&identity,&counters,unique.data(),inverse.data(),misses.data(),sorted.data(),
        payload.data(),response.data(),Maximum,row_bytes,row_bytes,7,11,19};}
    /** @return Actual production wrapper retaining the same original arrays. */
    h::Operations<Slot> operations(){return h::Operations<Slot>(view());}
    /** @brief Supply finite recorded coalescer outputs with stable first-original unique order. */
    void query(std::span<const std::uint64_t> ids,PooledImmutableLocalHotAccessIntent intent=PooledImmutableLocalHotAccessIntent::CurrentInvocation)
    {
        auto op=operations();ASSERT_TRUE(op.begin(intent,std::uint32_t(ids.size())));
        std::uint32_t u=0;
        for(std::uint32_t i=0;i<ids.size();++i){
            std::uint32_t found=u;
            for(std::uint32_t j=0;j<u;++j)if(unique[j].row==ids[i]){found=j;break;}
            if(found==u){unique[u++]={ids[i],0,i,a::None,a::None,0};}
            inverse[i]=found;sorted[i]={ids[i],i,0};
        }
        std::stable_sort(sorted.begin(),sorted.begin()+ids.size(),[](auto x,auto y){return x.row<y.row;});
        ASSERT_TRUE(op.coalesced(u));
        for(std::uint32_t i=0;i<u;++i)ASSERT_TRUE(op.probeUnique(i));
        ASSERT_TRUE(op.pinAndCompact());
    }
    /** @return Deterministic native byte for recorded immutable row data, independent of format decoding. */
    static std::byte byte(std::uint64_t row,std::size_t at){return std::byte((row*29+at*7+3)&255);}
    /** @return Original own compact request, without granting native execution/source authority. */
    t::RequestHeader request()const {
        t::RequestHeader q;q.query=identity.query;q.miss_rows=frame.source_rows;q.original_ids=std::uint32_t(frame.original_ids);
        q.source_admission=23;q.source_invocation=37;
        q.consumer=0;q.native_format=std::uint32_t(GGUFTensorType::F32);q.native_row_bytes=row_bytes;q.intent=identity.intent;return q;
    }
    /** @return Original declared reader envelope; current root lifetime is explicit. */
    t::ConsumerGeometry consumer()const {return {0,1,Maximum,Maximum,source_rows,row_bytes,std::uint32_t(GGUFTensorType::F32),19};}
    /** @brief Replay a completed native result in the actual receipt validator. */
    bool receive(bool short_payload=false) {
        const auto q=request();result={};result.query=q.query;result.pool_interval={7,23,31};
        result.miss_rows=q.miss_rows;result.native_bytes=q.miss_rows*row_bytes;result.consumer=0;
        result.native_format=q.native_format;result.intent=q.intent;
        for(std::uint32_t i=0;i<q.miss_rows;++i)for(std::size_t b=0;b<row_bytes;++b)
            response[i*row_bytes+b]=byte(misses[i],b);
        header_bytes=sizeof(result);native_bytes=result.native_bytes-(short_payload && result.native_bytes?1:0);
        return operations().acceptNativeResult(q,consumer(),{&result,response.data(),&header_bytes,&native_bytes,response.size()});
    }
    /** @brief Copy exactly requested native bytes using the same production source/destination selection. */
    void fill(std::size_t limit=SIZE_MAX) {
        auto op=operations();std::size_t written=0;
        for(std::uint32_t u=0;u<frame.unique_rows;++u)for(std::size_t b=0;b<row_bytes;++b){
            const auto* from=op.fillSource(u,b);if(!from)continue;
            auto* to=op.fillDestination(u,b);ASSERT_NE(to,nullptr);
            if(written++<limit)*to=*from;
        }
    }
    /** @brief Replay actual local native row reads before the genuine completion transition. */
    void read(){
        auto op=operations();
        for(std::uint32_t i=0;i<frame.original_ids;++i){
            const auto* row=op.readableOriginal(i);ASSERT_NE(row,nullptr);
            for(std::size_t b=0;b<row_bytes;++b)ASSERT_EQ(row[b],byte(unique[inverse[i]].row,b));
        }
        ASSERT_TRUE(op.completeRead(identity.query));
    }
    /** @brief Actual source-independent local read points at exact immutable bytes. */
    void expectRows(std::span<const std::uint64_t> ids){
        auto op=operations();for(std::uint32_t i=0;i<ids.size();++i){
            const auto* row=op.readableOriginal(i);ASSERT_NE(row,nullptr);
            for(std::size_t b=0;b<row_bytes;++b)ASSERT_EQ(row[b],byte(ids[i],b));
        }
        if(!(identity.flags&h::ReadCompleted))read();
    }
    /** @brief Complete actual consumption/fill/retirement, preserving the model-lifetime ordinal. */
    void consumeRetire(){if(!(identity.flags&h::ReadCompleted))read();auto op=operations();ASSERT_TRUE(op.consumeDemand(identity.query));fill();ASSERT_TRUE(op.completeFills());
        const auto retired=op.retire(identity.query);EXPECT_EQ(retired.physical,decltype(retired.physical)::Retired);
        EXPECT_EQ(retired.outcome,decltype(retired.outcome)::Consumed);EXPECT_EQ(frame.held_pins,0u);}
};
/** @brief Shared recorded full sixteen-head query, with stable repeats. */
std::array<std::uint64_t,16> repeated(std::uint64_t row){std::array<std::uint64_t,16>x{};x.fill(row);return x;}
}
/** @test All-hit local data never depends on a lower response; policy credits consumption exactly once. */
TEST(PooledImmutableLocalHot, DemandAndAllHitReplayPreserveNativeBytesAndOrdinal)
{
    for(const auto capacity:{0u,1u,4u}){
        Replay<a::Slot32> r(capacity);const auto ids=repeated(5);
        for(int replay=0;replay<20;++replay){
            const auto clock=r.owner.logical_clock;r.query(ids);EXPECT_EQ(r.identity.query.sequence,std::uint64_t(replay+1));
            EXPECT_EQ(r.owner.logical_clock,clock);
            if(r.frame.source_rows)ASSERT_TRUE(r.receive());
            else {std::fill(r.response.begin(),r.response.end(),std::byte{0xdd});r.native_bytes=0;r.header_bytes=0;}
            r.expectRows(ids);r.consumeRetire();
            EXPECT_EQ(r.owner.logical_clock,capacity?clock+ids.size():clock);
        }
        r.query({});EXPECT_EQ(r.frame.source_rows,0u);r.consumeRetire();EXPECT_EQ(r.identity.query.sequence,21u);
    }
}
/** @test A future frame can be adopted without another query, and discard never changes hot policy. */
TEST(PooledImmutableLocalHot, FutureIntentAdoptionAndDiscardNeverCreditsUnconsumedWork)
{
    Replay<a::Slot32> r(2);const auto ids=repeated(8);r.query(ids);ASSERT_TRUE(r.receive());r.consumeRetire();
    for(int replay=0;replay<20;++replay){
        const auto owner=r.owner;const auto counts=r.counts;const auto epochs=r.epochs;
        r.query(ids,PooledImmutableLocalHotAccessIntent::FutureReadAhead);ASSERT_EQ(r.frame.source_rows,0u);
        auto op=r.operations();ASSERT_TRUE(op.discard(r.identity.query));const auto retired=op.retire(r.identity.query);
        EXPECT_EQ(retired.outcome,decltype(retired.outcome)::Discarded);EXPECT_EQ(std::memcmp(&owner,&r.owner,sizeof(owner)),0);
        EXPECT_EQ(r.counts,counts);EXPECT_EQ(r.epochs,epochs);EXPECT_EQ(r.frame.held_pins,0u);
        r.query(ids,PooledImmutableLocalHotAccessIntent::FutureReadAhead);const auto ordinal=r.identity.query.sequence;
        op=r.operations();ASSERT_TRUE(op.adopt(r.identity.query));r.expectRows(ids);r.consumeRetire();
        EXPECT_EQ(r.identity.query.sequence,ordinal);
    }
}
/** @test Partial reservations/copies/Ready rows drain physical pins without successful semantic retirement. */
TEST(PooledImmutableLocalHot, PartialConsumeCancellationDrainsActualPinsAndPreservesFirstFailure)
{
    for(const auto capacity:{0u,1u,4u})for(const auto copied:{std::size_t{0},std::size_t{1},std::size_t{17}}){
        Replay<a::Slot32> r(capacity);std::array<std::uint64_t,16> ids{};for(unsigned i=0;i<ids.size();++i)ids[i]=i%3;
        r.query(ids);ASSERT_TRUE(r.receive());r.read();auto op=r.operations();ASSERT_TRUE(op.consumeDemand(r.identity.query));r.fill(copied);
        EXPECT_LE(r.frame.held_pins,std::min(capacity,3u));ASSERT_TRUE(op.cancel(r.identity.query,a::Status::SourceFailure));
        ASSERT_TRUE(op.cancel(r.identity.query,a::Status::Cancelled));const auto retired=op.retire(r.identity.query);
        EXPECT_EQ(retired.physical,decltype(retired.physical)::Retired);EXPECT_EQ(retired.outcome,decltype(retired.outcome)::Failed);
        EXPECT_EQ(retired.status,a::Status::SourceFailure);EXPECT_EQ(r.frame.held_pins,0u);
        EXPECT_FALSE(op.begin(PooledImmutableLocalHotAccessIntent::CurrentInvocation,16));
    }
}
/** @test Original row64 keys, declined admissions, partial/empty counts and fatal ordinal overflow stay exact. */
TEST(PooledImmutableLocalHot, Row64AndMalformedReceiptReplayFailClosed)
{
    Replay<a::Slot64> r(2,16,0x100000002ULL);const auto ids=repeated(0x100000001ULL);
    r.query(ids);ASSERT_TRUE(r.receive());r.expectRows(ids);r.consumeRetire();r.query(ids);
    ASSERT_EQ(r.frame.source_rows,0u);r.expectRows(ids);r.consumeRetire();
    r.identity.query.sequence=UINT64_MAX;EXPECT_FALSE(r.operations().begin(PooledImmutableLocalHotAccessIntent::CurrentInvocation,0));
    EXPECT_EQ(r.identity.terminal_error,a::Status::Exhausted);
    Replay<a::Slot32> bad(0);bad.query(repeated(9));EXPECT_FALSE(bad.receive(true));EXPECT_EQ(bad.frame.status,a::Status::SourceFailure);
    ASSERT_TRUE(bad.operations().cancel(bad.identity.query));const auto retired=bad.operations().retire(bad.identity.query);
    EXPECT_EQ(retired.outcome,decltype(retired.outcome)::Failed);EXPECT_EQ(retired.status,a::Status::SourceFailure);
}

/** @test Controller frontiers reject early policy credit, duplicate consumption and concurrent frame rearm. */
TEST(PooledImmutableLocalHot, OriginalReadConsumptionFrontiersFailClosed)
{
    for (unsigned replay = 0; replay < 20; ++replay) {
        Replay<a::Slot32> early(2);
        early.query(repeated(3));
        ASSERT_TRUE(early.receive());
        auto op = early.operations();
        EXPECT_FALSE(op.consumeDemand(early.identity.query));
        EXPECT_EQ(early.owner.logical_clock, 0u);
        ASSERT_TRUE(op.cancel(early.identity.query));
        EXPECT_EQ(op.retire(early.identity.query).physical, h::Operations<a::Slot32>::Retirement::Physical::Retired);

        Replay<a::Slot32> duplicate(2);
        duplicate.query(repeated(4));
        ASSERT_TRUE(duplicate.receive());
        duplicate.read();
        auto duplicateOp = duplicate.operations();
        ASSERT_TRUE(duplicateOp.consumeDemand(duplicate.identity.query));
        EXPECT_FALSE(duplicateOp.consumeDemand(duplicate.identity.query));
        ASSERT_TRUE(duplicateOp.cancel(duplicate.identity.query));
        EXPECT_EQ(duplicateOp.retire(duplicate.identity.query).physical, h::Operations<a::Slot32>::Retirement::Physical::Retired);

        Replay<a::Slot32> active(2);
        active.query(repeated(5));
        const auto query = active.identity.query;
        EXPECT_FALSE(active.operations().begin(PooledImmutableLocalHotAccessIntent::CurrentInvocation, 16));
        EXPECT_TRUE(t::same(query, active.identity.query));
        ASSERT_TRUE(active.operations().cancel(query));
        EXPECT_EQ(active.operations().retire(query).physical, h::Operations<a::Slot32>::Retirement::Physical::Retired);
    }
}
/** @test Only a genuinely newer original request resets a physically retired normal cancellation. */
TEST(PooledImmutableLocalHot, RequestResetPreservesHotRowsAndRejectsIncompleteRetirement)
{
    for (unsigned replay = 0; replay < 20; ++replay) {
        Replay<a::Slot32> r(2);
        const auto ids = repeated(7);
        r.query(ids); ASSERT_TRUE(r.receive()); r.consumeRetire();
        const auto policy = r.owner;
        const auto sequence = r.identity.query.sequence;
        r.query(ids); ASSERT_EQ(r.frame.source_rows, 0u);
        auto op = r.operations();
        ASSERT_TRUE(op.cancel(r.identity.query));
        ASSERT_EQ(op.retire(r.identity.query).outcome, h::Operations<a::Slot32>::Retirement::Outcome::Cancelled);
        t::SourceAdmissionHeader next;
        next.table=7; next.source_storage_generation=11; next.source_admission=24; next.source_invocation=38;
        ASSERT_TRUE(op.resetRequest(next));
        EXPECT_EQ(std::memcmp(&policy,&r.owner,sizeof(policy)),0);
        EXPECT_EQ(r.identity.query.sequence,sequence+1);
        r.query(ids); EXPECT_EQ(r.frame.source_rows,0u); r.consumeRetire();
        EXPECT_EQ(r.identity.query.sequence,sequence+2);
        EXPECT_EQ(r.identity.source_request_invocation,38u);
        EXPECT_FALSE(op.resetRequest(next));

        Replay<a::Slot32> pending(2);
        pending.query(ids);
        EXPECT_FALSE(pending.operations().resetRequest(next));
        EXPECT_EQ(pending.identity.source_request_invocation,37u);

        Replay<a::Slot32> failed(2);
        failed.query(ids); ASSERT_TRUE(failed.operations().cancel(failed.identity.query,a::Status::SourceFailure));
        ASSERT_EQ(failed.operations().retire(failed.identity.query).physical,h::Operations<a::Slot32>::Retirement::Physical::Retired);
        EXPECT_FALSE(failed.operations().resetRequest(next));
        EXPECT_EQ(failed.identity.terminal_error,a::Status::SourceFailure);
    }
}
/** @test Every original native format and both source-key widths retain exact separate local caps and fixed C0 floors. */
TEST(PooledImmutableLocalHot, GeometryKeepsOriginalNativeFormatsAndSeparateCapacity)
{
    std::vector<GGUFTensorType> formats;
    for (unsigned code=0;code<=30;++code) {
        const auto format=static_cast<GGUFTensorType>(code);
        try { (void)ggufToTensorType(format); formats.push_back(format); }
        catch (const std::exception&) {
            const std::array<unsigned,8> unsupported{4,5,9,24,25,26,27,28};
            EXPECT_NE(std::find(unsupported.begin(),unsupported.end(),code),unsupported.end());
        }
    }
    ASSERT_EQ(formats.size(),23u);
    for(const auto format:formats){
        ImmutableRowTableSource source;source.format=format;source.rows=4096;source.row_elements=256;
        const auto zero=PooledImmutableLocalHotGeometry::declare(source,{0,DeviceId::cuda(0)},32,0,{65536,3,4});
        const auto hot=PooledImmutableLocalHotGeometry::declare(source,{0,DeviceId::cuda(0)},32,4,{65536,3,4});
        EXPECT_EQ(zero.physicalBytes(),0u);EXPECT_EQ(hot.rowBytes(),source.rowBytes());EXPECT_GT(hot.physicalBytes(),4*source.rowBytes());
        EXPECT_EQ(zero.frameOffsets(),hot.frameOffsets());EXPECT_EQ(zero.scratchOffsets(),hot.scratchOffsets());
        EXPECT_EQ(hot.maximumOriginalIds(),32u);EXPECT_EQ(hot.maximumCompactMissRows(),32u);
        EXPECT_THROW(PooledImmutableLocalHotGeometry::declare(source,{0,DeviceId::cuda(0)},17,1,{65536,3,4}),std::invalid_argument);
    }
}
} // namespace llaminar2::qwen4exp

namespace llaminar2::qwen4exp {
/** @test The actual native reset policy never erases cancellation before or between its acquire/CAS. */
TEST(PooledImmutableLocalHot, ReadyResetPreservesAbsorbingFailureWithoutRetry)
{
    for (unsigned replay = 0; replay < 20; ++replay) {
        for (const auto original : std::array<std::uint64_t,5>{0,1,2,3,UINT64_MAX}) {
            for (const bool cancel : {false, true}) {
                std::atomic<std::uint64_t> word(original);
                unsigned reads = 0, exchanges = 0;
                const bool accepted = local_hot::resetReadyForProbe(
                    [&] { ++reads; return word.load(std::memory_order_acquire); },
                    [&](std::uint64_t expected, std::uint64_t value) {
                        ++exchanges;
                        if (cancel) word.store(UINT64_MAX, std::memory_order_release);
                        return word.compare_exchange_strong(expected, value, std::memory_order_acq_rel);
                    });
                EXPECT_EQ(reads, 1u);
                EXPECT_EQ(exchanges, original <= 2 ? 1u : 0u);
                EXPECT_EQ(accepted, original <= 2 && !cancel);
                EXPECT_EQ(word.load(), accepted ? 0u : (cancel && original <= 2 ? UINT64_MAX : original));
            }
        }
    }
}

/** @test Terminal snapshots require real reader retirement and never clear warmed lifetime counters. */
TEST(PooledImmutableLocalHot, TerminalSnapshotsPreserveRequestResetAndRejectDirtyFrames)
{
    Replay<a::Slot32> r(2);const auto ids=repeated(7);
    r.query(ids);
    h::TerminalObservation unchanged;unchanged.status=h::TerminalObservationStatus::Failed;const auto before=unchanged;
    EXPECT_FALSE(h::terminalObservationDelta(h::snapshotTerminalCounters(r.counters,r.identity,r.frame.held_pins,r.frame.status),{},unchanged));
    EXPECT_EQ(std::memcmp(&unchanged,&before,sizeof(before)),0);
    ASSERT_TRUE(r.receive());r.consumeRetire();
    const auto first=h::snapshotTerminalCounters(r.counters,r.identity,r.frame.held_pins,r.frame.status);
    h::TerminalObservation delta;ASSERT_TRUE(h::terminalObservationDelta(first,{},delta));
    EXPECT_EQ(delta.values[0],16u);EXPECT_EQ(delta.values[1],0u);EXPECT_EQ(delta.values[8],r.frame.promotions);
    const auto counters=r.counters;const auto warmed=r.owner;
    t::SourceAdmissionHeader next;next.table=7;next.source_storage_generation=11;next.source_admission=24;next.source_invocation=38;
    ASSERT_TRUE(r.operations().resetRequest(next));EXPECT_EQ(r.counters,counters);EXPECT_EQ(std::memcmp(&warmed,&r.owner,sizeof(warmed)),0);
    r.query(ids);ASSERT_EQ(r.frame.source_rows,0u);r.consumeRetire();
    const auto second=h::snapshotTerminalCounters(r.counters,r.identity,r.frame.held_pins,r.frame.status);
    ASSERT_TRUE(h::terminalObservationDelta(second,first,delta));
    EXPECT_EQ(delta.values[0],16u);EXPECT_EQ(delta.values[1],16u);EXPECT_EQ(delta.values[2],16u);
    EXPECT_EQ(delta.values[8],0u);EXPECT_EQ(delta.values[9],0u);
    ASSERT_TRUE(h::terminalObservationDelta(second,second,delta));for(auto value:delta.values)EXPECT_EQ(value,0u);
    EXPECT_FALSE(h::terminalObservationDelta(first,second,delta));
    r.identity.terminal_error=a::Status::SourceFailure;
    EXPECT_FALSE(h::terminalObservationDelta(h::snapshotTerminalCounters(r.counters,r.identity,0,r.frame.status),second,delta));
}
/** @test Every actual frame promotion/erase survives frame reuse in its original producer total. */
TEST(PooledImmutableLocalHot, TerminalChurnAccumulatesActualTransitionsAcrossFrames)
{
    Replay<a::Slot32> r(1);std::uint64_t promotions=0,evictions=0;
    for(unsigned frame=0;frame<20;++frame){
        r.query(repeated(5+frame%2));if(r.frame.source_rows)ASSERT_TRUE(r.receive());r.consumeRetire();
        promotions+=r.frame.promotions;evictions+=r.frame.evictions;
        EXPECT_EQ(r.counters.promotions,promotions);EXPECT_EQ(r.counters.evictions,evictions);
    }
    EXPECT_GT(promotions,0u);
    const auto observed=h::snapshotTerminalCounters(r.counters,r.identity,0,r.frame.status);
    EXPECT_EQ(observed.values[8],promotions);EXPECT_EQ(observed.values[9],evictions);
    // Original L1 rows and compact L2 rows are different denominators.
    QwenAdaptiveCounterTotals pool{1,3,1,1,1,16,0,0,0};
    EXPECT_TRUE(validQwenAdaptiveCounters(pool,16,32,QwenAdaptiveQuerySemantics::CompactNativeMisses));
    EXPECT_FALSE(validQwenAdaptiveCounters(pool,16,32,QwenAdaptiveQuerySemantics::OriginalNgramRows));
    EXPECT_EQ(observed.values[0],20u*16u);EXPECT_NE(observed.values[0],pool.completed_live_ids);
}
/** @test Live completed-byte arithmetic, malformed terminal shapes and overflow fail before fabricated reporting. */
TEST(PooledImmutableLocalHot, TerminalLiveWireArithmeticAndControlExtentAreExact)
{
    std::uint64_t bytes=0;ASSERT_TRUE(h::accumulateCompletedWireCounter(bytes,0));EXPECT_EQ(bytes,0u);
    for(auto live:{0u,1u,3u,17u,0u})ASSERT_TRUE(h::accumulateCompletedWireCounter(bytes,std::uint64_t(live)*8));
    EXPECT_EQ(bytes,21u*8u);const auto actual=bytes;
    EXPECT_FALSE(h::accumulateCompletedWireCounter(bytes,UINT64_MAX));EXPECT_EQ(bytes,actual);
    for(int requests=1;requests<=8;++requests)
        EXPECT_EQ(h::terminalObservationBytes(requests,sampling_math::kDeviceGenerationControlCount),
            std::size_t(requests)*108u*sizeof(int)+128u);
    EXPECT_THROW(h::terminalObservationBytes(0,108),std::invalid_argument);
    EXPECT_THROW(h::terminalObservationBytes(1,109),std::invalid_argument);
    h::TerminalObservation wire;wire.values[10]=24;wire.values[12]=88;wire.values[13]=96;
    EXPECT_TRUE(h::validTerminalWireDelta(wire,h::TerminalReceiveRole::Coordinator,16));
    EXPECT_FALSE(h::validTerminalWireDelta(wire,h::TerminalReceiveRole::Reader,16));
    --wire.values[13];EXPECT_FALSE(h::validTerminalWireDelta(wire,h::TerminalReceiveRole::Coordinator,16));
    wire={};wire.values[11]=48;wire.values[12]=96;wire.values[14]=40;
    EXPECT_TRUE(h::validTerminalWireDelta(wire,h::TerminalReceiveRole::Reader,16));
    ++wire.values[11];EXPECT_FALSE(h::validTerminalWireDelta(wire,h::TerminalReceiveRole::Reader,16));
    static_assert(sampling_math::kDeviceGenerationControlCount==108);
}
/** @test The actual storage-owned observer survives replacement consumers and never republishes cumulative work. */
TEST(PooledImmutableLocalHot, TerminalStorageLifetimeRejectsStaleAndDuplicateWork)
{
    auto storage=std::make_shared<h::TerminalObservationAccumulator>(h::TerminalReceiveRole::Coordinator,16);
    h::TerminalObservation first;first.values[0]=16;first.values[3]=16;first.values[8]=1;
    first.values[10]=8;first.values[12]=88;first.values[13]=96;
    {auto firstRunnerObserver=storage;const auto published=firstRunnerObserver->observe(first);
        EXPECT_TRUE(published.since_storage_initialization);EXPECT_EQ(published.delta.values[0],16u);}
    auto replacementRunnerObserver=storage;auto second=first;
    second.values[0]=32;second.values[1]=16;second.values[2]=16;second.values[3]=32;
    second.values[12]=176;second.values[13]=192;
    auto published=replacementRunnerObserver->observe(second);
    EXPECT_FALSE(published.since_storage_initialization);EXPECT_EQ(published.delta.values[0],16u);
    EXPECT_EQ(published.delta.values[1],16u);EXPECT_EQ(published.delta.values[8],0u);
    published=storage->observe(second);for(auto value:published.delta.values)EXPECT_EQ(value,0u);
    EXPECT_THROW(storage->observe(first),std::logic_error);
    auto failed=second;failed.status=h::TerminalObservationStatus::Failed;EXPECT_THROW(storage->observe(failed),std::logic_error);
    published=storage->observe(second);for(auto value:published.delta.values)EXPECT_EQ(value,0u);
    // Terminal observers can race only in HOST reporting state. No mutex is
    // held across a device operation; the same high snapshot is credited once.
    h::TerminalObservationAccumulator raced(h::TerminalReceiveRole::Coordinator,16);
    std::array<std::uint64_t,2> observed{};
    std::thread a([&]{try{observed[0]=raced.observe(first).delta.values[0];}catch(const std::logic_error&){};});
    std::thread b([&]{try{observed[1]=raced.observe(second).delta.values[0];}catch(const std::logic_error&){};});
    a.join();b.join();EXPECT_EQ(observed[0]+observed[1],32u);
}
/** @test Original activation admission prices precisely the same separate terminal arena registration. */
TEST(PooledImmutableLocalHot, TerminalArenaAdmissionUsesOriginalRequestCapacity)
{
    const auto profile=ModelMemoryProfile::fromGGUF(llaminar2::test::qwen4MetadataFixture(false));
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})for(int requests:{1,3,8}){
        ActivationGraphMemoryGeometry geometry;
        geometry.ngram_lookup=ImmutableRowLookupExecution::AdaptiveGPUCache;geometry.batch_size=1;
        geometry.resident_graph_rows=16;geometry.context_tokens=64;geometry.local_d_ff=640;
        geometry.local_n_heads=24;geometry.local_n_kv_heads=2;geometry.local_vocab=profile.vocab_size;
        geometry.first_layer=0;geometry.last_layer=47;geometry.total_shards=1;geometry.generation_request_capacity=requests;
        const auto original=ActivationMemoryEstimator::immutableLookupArenaBytes(profile,geometry,device);
        const auto fullOriginal=ActivationMemoryEstimator::estimate(profile,geometry,device);
        geometry.local_hot_terminal_observation=true;
        const auto actual=h::terminalObservationBytes(requests,sampling_math::kDeviceGenerationControlCount);
        EXPECT_EQ(ActivationMemoryEstimator::immutableLookupArenaBytes(profile,geometry,device),original+actual);
        EXPECT_EQ(ActivationMemoryEstimator::estimate(profile,geometry,device),fullOriginal+actual);
        geometry.ngram_lookup=ImmutableRowLookupExecution::ResidentGPU;
        const auto ordinary=ActivationMemoryEstimator::estimate(profile,geometry,device);
        geometry.local_hot_terminal_observation=false;
        EXPECT_EQ(ActivationMemoryEstimator::estimate(profile,geometry,device),ordinary);
    }
}
} // namespace llaminar2::qwen4exp

namespace llaminar2::qwen4exp {
/** @brief Exercise the SAME native forward predicate for prefill, generation and absorbing failures. */
TEST(PooledImmutableLocalHot, OriginalForwardControllerAdmission)
{
    for (unsigned replay = 0; replay < 20; ++replay) {
        EXPECT_TRUE(local_hot::validForwardController(0, 0));
        EXPECT_TRUE(local_hot::validForwardController(1, 0));
        for (int ok : {-1, 2, 7, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()})
            EXPECT_FALSE(local_hot::validForwardController(ok, 0));
        for (int error : {-1, 1, 7, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()})
            for (int ok : {0, 1}) EXPECT_FALSE(local_hot::validForwardController(ok, error));
    }
}
}

namespace llaminar2::qwen4exp {
namespace {
/**
 * @brief Snapshot actual object bytes without depending on copy constructors preserving padding.
 * @tparam T Original trivially copyable production metadata record.
 * @param original Record whose complete object representation is observed.
 * @return Exact immutable bytes, including the original padding representation.
 */
template<class T>
auto localHotObjectBytes(const T& original)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<std::byte, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &original, bytes.size());
    return bytes;
}

/**
 * @brief Preserve warmed demand state across a speculative mixed scan with no source completion.
 * @tparam Slot Original Slot32 or Slot64 physical cache record.
 * @param base Native row-key offset selecting the original key-width domain.
 * @param source_rows Exact bound containing every replayed source row.
 */
template<class Slot>
void replayFutureMissDiscard(std::uint64_t base, std::uint64_t source_rows)
{
    for (const auto capacity : {1u, 2u, 4u}) {
        Replay<Slot> r(capacity, 16, source_rows);
        const auto hot = repeated(base + 5);
        for (unsigned warm = 0; warm < 20; ++warm) {
            r.query(hot);
            if (r.frame.source_rows) ASSERT_TRUE(r.receive());
            r.consumeRetire();
        }
        for (unsigned replay = 0; replay < 20; ++replay) {
            const auto owner = localHotObjectBytes(r.owner);
            std::vector<std::array<std::byte, sizeof(Slot)>> slots;
            for (const auto& slot : r.slots) slots.push_back(localHotObjectBytes(slot));
            std::vector<std::array<std::byte, sizeof(a::RadixNode)>> nodes;
            for (const auto& node : r.nodes) nodes.push_back(localHotObjectBytes(node));
            const auto payload = r.payload;
            const auto counts = r.counts;
            const auto epochs = r.epochs;
            const auto counters = r.counters;
            std::array<std::uint64_t, 32> scan{};
            for (unsigned i = 0; i < scan.size(); ++i) scan[i] = base + 100 + i;
            scan[0] = hot.front();
            r.query(scan, PooledImmutableLocalHotAccessIntent::FutureReadAhead);
            ASSERT_EQ(r.frame.source_rows, 31u);
            ASSERT_EQ(r.frame.held_pins, 1u);
            ASSERT_EQ(r.identity.disposition, h::Disposition::Probed);
            EXPECT_EQ(localHotObjectBytes(r.owner), owner);
            EXPECT_EQ(r.counts, counts);
            EXPECT_EQ(r.epochs, epochs);
            for (const auto& slot : r.slots)
                EXPECT_NE(a::PhysicalSlot<Slot>::state(slot), a::SlotState::Reserved);
            auto op = r.operations();
            // No future wire receipt is manufactured: the actual shared protocol
            // still rejects FutureReadAhead. Discard must not spend hot capacity.
            ASSERT_TRUE(op.discard(r.identity.query));
            const auto retired = op.retire(r.identity.query);
            EXPECT_EQ(retired.outcome, decltype(retired.outcome)::Discarded);
            EXPECT_EQ(retired.physical, decltype(retired.physical)::Retired);
            EXPECT_EQ(r.frame.held_pins, 0u);
            EXPECT_EQ(localHotObjectBytes(r.owner), owner);
            for (std::size_t i = 0; i < slots.size(); ++i)
                EXPECT_EQ(localHotObjectBytes(r.slots[i]), slots[i]);
            for (std::size_t i = 0; i < nodes.size(); ++i)
                EXPECT_EQ(localHotObjectBytes(r.nodes[i]), nodes[i]);
            EXPECT_EQ(r.payload, payload);
            EXPECT_EQ(r.counts, counts);
            EXPECT_EQ(r.epochs, epochs);
            EXPECT_EQ(r.counters, counters);
            r.query(hot);
            EXPECT_EQ(r.frame.source_rows, 0u);
            r.expectRows(hot);
            r.consumeRetire();
        }
    }
}

/**
 * @brief A future miss becomes ordinary demand before receipt; only actual read earns policy credit.
 * @tparam Slot Original Slot32 or Slot64 physical cache record.
 * @param base Native row-key offset selecting the original key-width domain.
 * @param source_rows Exact bound containing every replayed source row.
 */
template<class Slot>
void replayLateAdoptedMiss(std::uint64_t base, std::uint64_t source_rows)
{
    Replay<Slot> r(2, 16, source_rows);
    const auto hot = repeated(base + 5);
    r.query(hot); ASSERT_TRUE(r.receive()); r.consumeRetire();
    for (unsigned replay = 0; replay < 20; ++replay) {
        auto input = hot;
        input.back() = base + 100 + replay;
        r.query(input, PooledImmutableLocalHotAccessIntent::FutureReadAhead);
        ASSERT_EQ(r.frame.source_rows, 1u);
        ASSERT_EQ(r.frame.held_pins, 1u);
        const auto original_query = r.identity.query;
        const auto clock = r.owner.logical_clock;
        const auto counters = r.counters;
        auto op = r.operations();
        ASSERT_TRUE(op.adopt(original_query));
        EXPECT_TRUE(t::same(original_query, r.identity.query));
        EXPECT_EQ(r.identity.intent, PooledImmutableLocalHotAccessIntent::CurrentInvocation);
        EXPECT_EQ(r.owner.logical_clock, clock);
        EXPECT_EQ(r.counters, counters);
        ASSERT_TRUE(r.receive()); // Actual current-demand receipt, never a future tier-prefetch claim.
        EXPECT_EQ(r.counters.future_late_native_rows, counters.future_late_native_rows + 1);
        EXPECT_EQ(r.counters.future_useful_native_rows, counters.future_useful_native_rows);
        EXPECT_EQ(r.owner.logical_clock, clock);
        r.read();
        EXPECT_EQ(r.owner.logical_clock, clock);
        EXPECT_EQ(r.counters.consumed_original_ids, counters.consumed_original_ids);
        ASSERT_TRUE(op.consumeDemand(original_query));
        EXPECT_EQ(r.owner.logical_clock, clock + input.size());
        r.fill(); ASSERT_TRUE(op.completeFills());
        EXPECT_EQ(r.counters.consumed_original_ids, counters.consumed_original_ids + input.size());
        EXPECT_EQ(r.counters.future_useful_native_rows, counters.future_useful_native_rows + 1);
        EXPECT_EQ(r.counters.future_on_time_native_rows, counters.future_on_time_native_rows);
        EXPECT_EQ(r.counters.future_wasted_native_rows, counters.future_wasted_native_rows);
        const auto retired = op.retire(original_query);
        EXPECT_EQ(retired.outcome, decltype(retired.outcome)::Consumed);
        EXPECT_EQ(r.frame.held_pins, 0u);
        r.query(hot);
        ASSERT_EQ(r.frame.source_rows, 0u);
        r.consumeRetire();
    }
}

/**
 * @brief Replay a finite disjoint consumed scan through the same policy after warming a decode row.
 * @tparam Slot Original Slot32 or Slot64 physical cache record.
 * @param base Native row-key offset selecting the original key-width domain.
 * @param source_rows Exact bound containing every replayed source row.
 */
template<class Slot>
void replayConsumedScanProtection(std::uint64_t base, std::uint64_t source_rows)
{
    Replay<Slot> r(4, 16, source_rows);
    const auto hot = repeated(base + 5);
    for (unsigned warm = 0; warm < 20; ++warm) {
        r.query(hot);
        if (r.frame.source_rows) ASSERT_TRUE(r.receive());
        r.consumeRetire();
    }
    for (unsigned replay = 0; replay < 20; ++replay) {
        std::array<std::uint64_t, 32> scan{};
        for (unsigned i = 0; i < scan.size(); ++i) scan[i] = base + 100 + replay * scan.size() + i;
        r.query(scan); ASSERT_TRUE(r.receive()); r.consumeRetire();
        r.query(hot);
        ASSERT_EQ(r.frame.source_rows, 0u);
        r.expectRows(hot); r.consumeRetire();
    }
}
} // namespace

/** @test An unconsumed speculative scan cannot reserve/evict/credit warmed demand rows. */
TEST(PooledImmutableLocalHot, FutureMissDiscardPreservesDemandPolicyAndNativeBytes)
{
    replayFutureMissDiscard<a::Slot32>(0, 1024);
    replayFutureMissDiscard<a::Slot64>(0x100000000ULL, 0x100000400ULL);
}

/** @test Delayed adoption reuses the original query and receives policy credit only after LocalRead. */
TEST(PooledImmutableLocalHot, FutureLateMissAdoptionCreditsActualDemandExactlyOnce)
{
    replayLateAdoptedMiss<a::Slot32>(0, 1024);
    replayLateAdoptedMiss<a::Slot64>(0x100000000ULL, 0x100000400ULL);
}

/** @test One recorded finite consumed prefill scan preserves the repeatedly consumed decode row. */
TEST(PooledImmutableLocalHot, RecordedConsumedScanPreservesHotDecodeRow)
{
    replayConsumedScanProtection<a::Slot32>(0, 1024);
    replayConsumedScanProtection<a::Slot64>(0x100000000ULL, 0x100000400ULL);
}
} // namespace llaminar2::qwen4exp
