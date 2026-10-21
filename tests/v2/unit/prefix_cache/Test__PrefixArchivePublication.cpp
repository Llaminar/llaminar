/**
 * @file Test__PrefixArchivePublication.cpp
 * @brief Device-free partial-copy failure, exact event and physical owner retirement tests.
 *
 * Deferred native-copy mocks retain the original addresses until completion.
 * Real PrefixHostArena ranges/readiness state prove that an exceptional archive
 * cannot recycle its destination early, copy padded capacity or invent a host
 * synchronization. No model, accelerator or alternate numerical oracle is used.
 */
#include "execution/prefix_cache/PrefixArchivePublicationScope.h"
#include "execution/prefix_cache/PrefixArchiveCheckpointReadiness.h"
#include "execution/prefix_cache/PrefixHostArena.h"
#include "../../mocks/MockBackend.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief Queue exact mock copies and expose one explicit producer completion. */
class ArchiveBackend final : public llaminar2::test::MockBackend
{
public:
    /** @brief No driver discovery: the backend is one host-only native family. */
    ArchiveBackend() : MockBackend(DeviceType::CUDA) {}
    struct Copy {void *destination;const void *source;size_t bytes;};
    std::vector<Copy> copies;
    bool completed=false;
    bool fail_record=false;
    size_t records=0,host_waits=0,stream_waits=0;
    void *event=nullptr,*stream=nullptr;
    /** @brief A successful submission retains raw addresses without observing completion. */
    bool deviceCopyAsync(void *dst,const void *src,size_t bytes,int,void *operation_stream) override
    { EXPECT_EQ(operation_stream,stream);copies.push_back({dst,src,bytes});return true; }
    /** @brief Record the exact borrowed producer edge; never complete pending bytes here. */
    bool recordEvent(void *value,int ordinal,void *producer) override
    { EXPECT_EQ(ordinal,0);event=value;EXPECT_EQ(producer,stream);++records;return !fail_record; }
    /** @brief Query carries no blocking or data movement. */
    bool queryEvent(void *value,int ordinal,bool *ready) override
    { EXPECT_EQ(value,event);EXPECT_EQ(ordinal,0);*ready=completed;return true; }
    /** @brief A wait can occur only at explicit final owner teardown in these tests. */
    bool waitForEvent(void *value,int) override {EXPECT_EQ(value,event);++host_waits;return completed;}
    /** @brief Record an exact consumer stream edge without completing the mock producer. */
    bool streamWaitEvent(void *,void *value,int) override {EXPECT_EQ(value,event);++stream_waits;return true;}
    /** @brief Explicit diagnostic completion executes only accepted live copy extents. */
    void complete()
    { for(const auto &copy:copies)std::memcpy(copy.destination,copy.source,copy.bytes);completed=true; }
};

/** @test A later section throws after a queued copy; the arena and event retain its exact destination. */
TEST(PrefixArchivePublication, PartialFailurePublishesBeforeLeaseUnwindAndRetainsOnlyLiveExtent)
{
    ArchiveBackend backend;backend.stream=reinterpret_cast<void*>(uintptr_t{0x1234});
    auto backing=std::make_shared<std::array<uint8_t,512>>();backing->fill(0xa5);
    auto arena=PrefixHostArena::create(std::shared_ptr<void>(backing,backing->data()),backing->size(),{});
    auto readiness=std::make_shared<PrefixPayloadReadiness>();
    auto event=std::make_shared<uint8_t>(0);std::weak_ptr<void> event_lifetime=event;
    ASSERT_TRUE(readiness->prepare(event,DeviceId::cuda(0),backend.stream,backend));
    std::vector<std::shared_ptr<void>> retained_sources;
    auto source=std::make_shared<std::array<uint8_t,128>>();source->fill(0x37);
    std::weak_ptr<void> source_lifetime=source;
    EXPECT_THROW({
        auto destination=arena->acquire(512,readiness);ASSERT_TRUE(destination);
        auto source_owner=source;source.reset();
        PrefixArchivePublicationScope completion([&](PrefixArchiveCompletion reason) {
            EXPECT_EQ(reason,PrefixArchiveCompletion::Abandon);
            if(!backend.recordEvent(readiness->event(),0,backend.stream))return false;
            retained_sources.push_back(source_owner);
            return readiness->publishRecorded();
        });
        ASSERT_TRUE(backend.deviceCopyAsync(destination.get(),source_owner->data(),128,0,backend.stream));
        throw std::runtime_error("injected later archive section failure");
    },std::runtime_error);
    EXPECT_EQ(backend.records,1u);EXPECT_EQ(backend.host_waits,0u);EXPECT_EQ(backend.stream_waits,0u);
    EXPECT_TRUE(readiness->published());EXPECT_FALSE(source_lifetime.expired());
    readiness.reset();event.reset();
    EXPECT_FALSE(event_lifetime.expired());EXPECT_EQ(arena->availableBytes(),0u);
    EXPECT_TRUE(std::all_of(backing->begin(),backing->end(),[](uint8_t v){return v==0xa5;}));
    ASSERT_EQ(backend.copies.size(),1u);EXPECT_EQ(backend.copies.front().bytes,128u);
    backend.complete();retained_sources.clear();
    EXPECT_EQ(arena->availableBytes(),512u);EXPECT_TRUE(event_lifetime.expired());EXPECT_TRUE(source_lifetime.expired());
    for(size_t i=0;i<backing->size();++i)EXPECT_EQ((*backing)[i],i<128?0x37:0xa5)<<i;
}

/** @test A committed earlier handle remains leased when a later handle fails before aggregate publication. */
TEST(PrefixArchivePublication, EarlierCommittedHandleSurvivesLaterHandleFailure)
{
    ArchiveBackend first_backend,second_backend;
    first_backend.stream=second_backend.stream=reinterpret_cast<void*>(uintptr_t{0x1234});
    auto backing=std::make_shared<std::array<uint8_t,1024>>();backing->fill(0xa5);
    auto arena=PrefixHostArena::create(std::shared_ptr<void>(backing,backing->data()),backing->size(),{});
    std::vector<std::shared_ptr<void>> pending_uses;
    std::array<std::weak_ptr<void>,2> events,sources;
    auto copy_handle=[&](ArchiveBackend &backend,size_t index,bool fail) {
        auto readiness=std::make_shared<PrefixPayloadReadiness>();
        auto event=std::make_shared<uint8_t>(0);events[index]=event;
        ASSERT_TRUE(readiness->prepare(event,DeviceId::cuda(0),backend.stream,backend));
        auto destination=arena->acquire(512,readiness);ASSERT_TRUE(destination);
        auto source=std::make_shared<std::array<uint8_t,128>>();source->fill(index?0x49:0x37);sources[index]=source;
        PrefixArchivePublicationScope completion([&](PrefixArchiveCompletion reason) {
            EXPECT_EQ(reason,fail?PrefixArchiveCompletion::Abandon:PrefixArchiveCompletion::Commit);
            if(!backend.recordEvent(readiness->event(),0,backend.stream))return false;
            // Existing pending-use authority retains both successful and abandoned
            // local handles until their exact completion; aggregate success is later.
            pending_uses.push_back(destination);pending_uses.push_back(source);
            pending_uses.push_back(std::shared_ptr<void>(readiness,readiness->event()));
            return readiness->publishRecorded();
        });
        ASSERT_TRUE(backend.deviceCopyAsync(destination.get(),source->data(),128,0,backend.stream));
        if(fail)throw std::runtime_error("injected second handle failure");
        completion.commit();
    };
    copy_handle(first_backend,0,false);
    EXPECT_THROW(copy_handle(second_backend,1,true),std::runtime_error);
    EXPECT_EQ(first_backend.records,1u);EXPECT_EQ(second_backend.records,1u);
    EXPECT_EQ(arena->availableBytes(),0u);EXPECT_EQ(pending_uses.size(),6u);
    for(size_t i=0;i<2;++i){EXPECT_FALSE(events[i].expired());EXPECT_FALSE(sources[i].expired());}
    EXPECT_EQ(first_backend.host_waits+second_backend.host_waits,0u);
    first_backend.complete();second_backend.complete();pending_uses.clear();
    EXPECT_EQ(arena->availableBytes(),1024u);
    for(size_t i=0;i<2;++i){EXPECT_TRUE(events[i].expired());EXPECT_TRUE(sources[i].expired());}
    for(size_t i=0;i<backing->size();++i)EXPECT_EQ((*backing)[i],i<128?0x37:(i>=512&&i<640?0x49:0xa5))<<i;
}

/** @test Auxiliary-only main and shifted handles preflight one event without inventing recurrent storage. */
TEST(PrefixArchivePublication, AuxiliaryOnlyMainAndShiftedArchivesBindOneCompletionOwner)
{
    ArchiveBackend backend;backend.stream=reinterpret_cast<void*>(uintptr_t{0x1234});
    for(int depth:{0,1,3}) {
        PrefixBlockHandle handle;handle.key.block_index=depth;
        handle.layout.main_auxiliary={CacheAuxiliaryStateKind::QwenFlashNext,8,12,true};
        handle.payload_readiness=std::make_shared<PrefixPayloadReadiness>();
        auto event=std::make_shared<uint8_t>(0);size_t preparations=0;
        ASSERT_TRUE(preparePrefixArchiveCheckpointReadiness(
            handle,DeviceId::cuda(0),backend.stream,backend,[&] {
                ++preparations;
                return handle.payload_readiness->prepare(event,DeviceId::cuda(0),backend.stream,backend);
            }));
        EXPECT_EQ(preparations,1u);
        EXPECT_EQ(handle.live_checkpoint_ready_event.get(),event.get());
        EXPECT_EQ(handle.payload_readiness->event(),event.get());
        EXPECT_FALSE(handle.device_hybrid_allocation);EXPECT_FALSE(handle.device_terminal_hidden_allocation);
        EXPECT_FALSE(handle.payload_readiness->published());
        EXPECT_FALSE(handle.payload_readiness->waitOnHost());
    }
}

/** @test A recurrent slot event is reused, while missing preparation cannot authorize an auxiliary snapshot. */
TEST(PrefixArchivePublication, CheckpointPreparationReusesSlotAndRejectsUnpreparedAuthority)
{
    ArchiveBackend backend;backend.stream=reinterpret_cast<void*>(uintptr_t{0x1234});
    PrefixBlockHandle handle;handle.layout.main_auxiliary={CacheAuxiliaryStateKind::QwenFlashNext,8,12,true};
    handle.payload_readiness=std::make_shared<PrefixPayloadReadiness>();
    handle.live_checkpoint_ready_event=std::make_shared<uint8_t>(0);
    EXPECT_TRUE(preparePrefixArchiveCheckpointReadiness(handle,DeviceId::cuda(0),backend.stream,backend,[] {
        ADD_FAILURE()<<"preadmitted event must not allocate a second event";return false;
    }));
    EXPECT_EQ(handle.payload_readiness->event(),handle.live_checkpoint_ready_event.get());
    PrefixBlockHandle missing;missing.layout=handle.layout;
    missing.payload_readiness=std::make_shared<PrefixPayloadReadiness>();
    EXPECT_FALSE(preparePrefixArchiveCheckpointReadiness(missing,DeviceId::cuda(0),backend.stream,backend,[] {return true;}));
    EXPECT_FALSE(missing.live_checkpoint_ready_event);
}

/** @test All consumption APIs reject the same prepared-but-unpublished state without querying a backend. */
TEST(PrefixArchivePublication, PreparedStateCannotAuthorizeHostOrStreamConsumption)
{
    ArchiveBackend backend;backend.stream=reinterpret_cast<void*>(uintptr_t{0x1234});
    PrefixPayloadReadiness readiness;auto event=std::make_shared<uint8_t>(0);
    EXPECT_TRUE(readiness.waitOnHost());EXPECT_TRUE(readiness.waitOnStream(backend.stream));
    ASSERT_TRUE(readiness.prepare(event,DeviceId::cuda(0),backend.stream,backend));
    EXPECT_FALSE(readiness.waitOnHost());EXPECT_FALSE(readiness.waitOnStream(backend.stream));
    EXPECT_FALSE(readiness.waitOnStream(reinterpret_cast<void*>(uintptr_t{0x4567})));
    bool ready=true;EXPECT_FALSE(readiness.queryComplete(&ready));EXPECT_FALSE(ready);
    EXPECT_EQ(backend.host_waits,0u);EXPECT_EQ(backend.stream_waits,0u);EXPECT_EQ(backend.records,0u);
}

/** @test Both explicit success and early-return abandonment publish once using the original stream. */
TEST(PrefixArchivePublication, CommitAndEarlyReturnHaveOneCompletionEdge)
{
    for(bool commit:{false,true}) {
        size_t calls=0;
        const auto transaction=[&] {
            PrefixArchivePublicationScope scope([&](PrefixArchiveCompletion reason) {
                ++calls;EXPECT_EQ(reason,commit?PrefixArchiveCompletion::Commit:PrefixArchiveCompletion::Abandon);return true;
            });
            if(!commit)return;
            scope.commit();
        };
        transaction();EXPECT_EQ(calls,1u);
    }
}

/** @test Event failure or a throwing publication cannot return into destination cleanup. */
TEST(PrefixArchivePublication, CompletionFailureIsFatalBeforeOwnerRelease)
{
    EXPECT_DEATH({PrefixArchivePublicationScope scope([](PrefixArchiveCompletion){return false;});},"terminate");
    EXPECT_DEATH({PrefixArchivePublicationScope scope([](PrefixArchiveCompletion)->bool{throw std::runtime_error("event failure");});},"terminate");
    EXPECT_DEATH({PrefixArchivePublicationScope scope([](PrefixArchiveCompletion){return true;});scope.commit();scope.commit();},"terminate");
}

/** @test Event preparation rejects a foreign backend before any source/destination is exposed. */
TEST(PrefixArchivePublication, PreparedEventBindsItsExactNativeBackend)
{
    ArchiveBackend backend;auto event=std::make_shared<uint8_t>(0);PrefixPayloadReadiness readiness;
    EXPECT_FALSE(readiness.prepare(event,DeviceId::rocm(0),reinterpret_cast<void*>(uintptr_t{1}),backend));
    EXPECT_FALSE(readiness.prepared());
}
} // namespace
