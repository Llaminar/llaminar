/**
 * @file RankLocalPublisherStateReplay.inc
 * @brief Replay the actual rank-local publisher protocol with recorded native outcomes.
 *
 * The private helper is the same state owner consumed by the real adapter. No
 * model, device, prepared-family permission or native-ready runner is invented.
 * Callback observations prove submission/retirement ordering only; real endpoint
 * capture, bytes and device completion remain separate native obligations.
 */
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
namespace llaminar2::test {
namespace {
using LocalPublicationState = MoEOverlayParticipantNativeRetirementReplayAccess::PublicationState;
/** @return An exact same-rank original protocol descriptor without any native capability. */
MoEOverlayInferenceTransactionProtocol::Config localPublicationConfig(){
 return {.topology={.workspace_generation=71,.topology_fingerprint_low=0x1122334455667788ULL,
  .topology_fingerprint_high=0x8877665544332211ULL,.source_world_rank=0,.target_world_rank=0},
  .slot_count=4,.max_request_count=1,.max_rows_per_request=4,.max_mtp_draft_depth=3};
}
/** @return Actual ordinary or grouped-verifier descriptor for the declared depth. */
MoEOverlayInferenceExecutionDescriptor localPublicationDescriptor(int depth,std::uint64_t step,std::uint64_t epoch){
 return {.graph_role=depth?MoEOverlayInferenceGraphRole::MTPGroupedVerifier:MoEOverlayInferenceGraphRole::MainDecode,
  .logical_step_id=step,.placement_epoch=epoch,.request_count=1,.logical_rows_per_request=depth+1,
  .physical_rows_per_request=depth+1,.draft_depth=depth?depth:-1,.sidecar_depth=-1};
}
}
/** @test Exact ticket identity gates one submit/join, bounded slot reuse and changing real verifier widths. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherRetiresExactIdentityAcrossCommands){
 LocalPublicationState state(localPublicationConfig());std::string error;unsigned submissions=0,joins=0;
 for(std::uint64_t generation=1;generation<=4;++generation){
  const MoEOverlayInferenceCommandIdentity request{generation,generation,41+generation};
  ASSERT_TRUE(state.begin(request,&error))<<error;EXPECT_FALSE(state.begin(request,&error));
  std::uint64_t ordinal=0;
  for(const int depth:{0,1,3,2}){
   const auto descriptor=localPublicationDescriptor(depth,100+ordinal,request.initial_placement_epoch);
   auto handle=state.publish(descriptor,[&](const auto& ticket,std::string*){
    ++submissions;EXPECT_EQ(ticket.topology_fingerprint_low,state.topology().topology_fingerprint_low);
    EXPECT_EQ(ticket.target_world_rank,ticket.source_world_rank);EXPECT_EQ(ticket.logical_rows_per_request,depth+1);
    return true;
   });
   ASSERT_TRUE(handle.ok)<<handle.error;EXPECT_EQ(handle.ticket.transaction_ordinal,++ordinal);
   ASSERT_NE(state.activeTicket(),nullptr);EXPECT_EQ(*state.activeTicket(),handle.ticket);
   EXPECT_FALSE(state.complete(request.initial_placement_epoch,0,&error));
   EXPECT_FALSE(state.begin(request,&error)) << "A live original command cannot be replaced";
   auto foreign=handle;++foreign.ticket.workspace_generation;
   EXPECT_FALSE(state.retire(foreign,[&](std::string*){++joins;return true;},&error));
   ASSERT_TRUE(state.retire(handle,[&](std::string*){++joins;return true;},&error))<<error;
   EXPECT_FALSE(state.active());EXPECT_EQ(state.activeTicket(),nullptr);
   EXPECT_FALSE(state.retire(handle,[&](std::string*){++joins;return true;},&error));
  }
  ASSERT_TRUE(state.complete(request.initial_placement_epoch,0,&error))<<error;
  EXPECT_FALSE(state.begin(request,&error));
 }
 EXPECT_EQ(submissions,16u);EXPECT_EQ(joins,16u);
}
/** @test A partial native submission retains its original slot until an actual successful drain; abort is absorbing. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherPartialSubmitRequiresOriginalDrain){
 for(const bool throwing:{false,true}){
  LocalPublicationState state(localPublicationConfig());std::string error;unsigned submits=0,drains=0;
  ASSERT_TRUE(state.begin({9,3,41},&error));
  const auto handle=state.publish(localPublicationDescriptor(1,73,41),[&](const auto&,std::string* failure){
   ++submits;if(throwing)throw std::runtime_error("recorded partial submission");
   *failure="recorded partial submission";return false;
  });
  EXPECT_FALSE(handle.ok);EXPECT_EQ(handle.error,"recorded partial submission");ASSERT_TRUE(state.active());
  const auto original=*state.activeTicket();
  EXPECT_FALSE(state.complete(41,0,&error));
  EXPECT_FALSE(state.retire(handle,[](std::string*){ADD_FAILURE()<<"Failed publication cannot become a successful join";return true;},&error));
  EXPECT_FALSE(state.abort(41,7,[&](std::string* failure){++drains;*failure="recorded missing native proof";return false;},&error));
  ASSERT_NE(state.activeTicket(),nullptr);EXPECT_EQ(*state.activeTicket(),original);
  EXPECT_FALSE(state.abort(41,7,[&](std::string*)->bool{++drains;throw std::runtime_error("recorded thrown drain");},&error));
  EXPECT_EQ(error,"recorded thrown drain");ASSERT_TRUE(state.active());
  ASSERT_TRUE(state.abort(41,7,[&](std::string*){++drains;return true;},&error))<<error;
  EXPECT_FALSE(state.active());EXPECT_FALSE(state.complete(41,0,&error));EXPECT_FALSE(state.begin({10,4,41},&error));
  EXPECT_TRUE(state.abort(41,7,[&](std::string*){++drains;return true;},&error));
  EXPECT_EQ(submits,1u);EXPECT_EQ(drains,3u);
 }
}
/** @test False/thrown native joins cannot later publish success; cancellation attempts every original peer. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherJoinFailureStaysFailedUntilAllPeerDrain){
 for(const bool throwing:{false,true}){
  LocalPublicationState state(localPublicationConfig());std::string error;ASSERT_TRUE(state.begin({9,3,41},&error));
  const auto handle=state.publish(localPublicationDescriptor(3,73,41),[](const auto&,std::string*){return true;});ASSERT_TRUE(handle.ok);
  unsigned joins=0;EXPECT_FALSE(state.retire(handle,[&](std::string* failure){++joins;if(throwing)throw std::runtime_error("recorded missing terminal");*failure="recorded missing terminal";return false;},&error));
  EXPECT_EQ(error,"recorded missing terminal");ASSERT_TRUE(state.active());
  EXPECT_FALSE(state.retire(handle,[&](std::string*){++joins;return true;},&error));EXPECT_EQ(joins,1u);
  const std::array originalPeers{3,1,0,2};std::vector<int> visited;unsigned cancels=0;
  const auto drain=[&](bool fail,std::string* failure){return MoEOverlayParticipantNativeRetirementReplayAccess::join(originalPeers,
   [&](int peer){visited.push_back(peer);if(fail&&peer==1)throw std::logic_error("recorded peer1 terminal missing");},
   [&]()noexcept{++cancels;},failure);};
  EXPECT_FALSE(state.abort(41,7,[&](std::string* failure){return drain(true,failure);},&error));
  EXPECT_EQ(visited,(std::vector<int>(originalPeers.begin(),originalPeers.end())));EXPECT_EQ(cancels,1u);ASSERT_TRUE(state.active());
  visited.clear();ASSERT_TRUE(state.abort(41,7,[&](std::string* failure){return drain(false,failure);},&error))<<error;
  EXPECT_EQ(visited,(std::vector<int>(originalPeers.begin(),originalPeers.end())));EXPECT_FALSE(state.active());
  EXPECT_FALSE(state.complete(41,0,&error));EXPECT_FALSE(state.begin({10,4,41},&error));
 }
}
/** @test Invalid cancellation is authenticated before touching the active original native transaction. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherRejectsStaleAbortBeforeNativeDrain){
 std::string error;unsigned drains=0;
 const auto forbidden=[&](std::string*){++drains;return true;};
 {
  LocalPublicationState absent(localPublicationConfig());
  EXPECT_FALSE(absent.abort(41,7,forbidden,&error));EXPECT_EQ(drains,0u);
  EXPECT_TRUE(absent.begin({9,3,41},&error))<<"Rejected absent-command cancellation cannot poison later admission";
 }
 LocalPublicationState state(localPublicationConfig());ASSERT_TRUE(state.begin({9,3,41},&error))<<error;
 const auto handle=state.publish(localPublicationDescriptor(1,73,43),[](const auto&,std::string*){return true;});ASSERT_TRUE(handle.ok)<<handle.error;
 for(const auto [epoch,code]:{std::pair<std::uint64_t,int>{40,7},{42,7},{43,0},{43,-1}}){
  EXPECT_FALSE(state.abort(epoch,code,forbidden,&error));ASSERT_NE(state.activeTicket(),nullptr);
  EXPECT_EQ(*state.activeTicket(),handle.ticket);EXPECT_EQ(drains,0u);
 }
 ASSERT_TRUE(state.retire(handle,[](std::string*){return true;},&error))<<error;
 ASSERT_TRUE(state.complete(43,0,&error))<<error;
 EXPECT_TRUE(state.begin({10,4,44},&error))<<error;
}
} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/**
 * @brief Run recorded original queue outcomes through the real private worker/state/join guards.
 * @param outcome Zero succeeds, one returns SDK-false, two throws the original worker failure.
 * @note No native event, model-ready permission or GPU success is fabricated.
 */
void replayQueuedLocalPublication(int outcome){
 LocalPublicationState state(localPublicationConfig());std::string error;ASSERT_TRUE(state.begin({9,3,41},&error));
 std::promise<void> release;const auto gate=release.get_future().share();std::array<std::future<void>,2> queued;
 std::atomic<unsigned> immediateCancels{0};unsigned consumed=0;std::vector<int> nativeVisited;
 const auto originalCancel=[&]()noexcept{immediateCancels.fetch_add(1,std::memory_order_release);};
 const auto control=std::this_thread::get_id();const std::array originalNativePeers{0,1,2,3};
 bool released=false;
 struct RecordedQueueLifetime{
  std::promise<void>& release;bool& released;std::array<std::future<void>,2>& tasks;
  ~RecordedQueueLifetime(){if(!released)release.set_value();for(auto& task:tasks)if(task.valid())task.wait();}
 } recordedLifetime{release,released,queued};
 const auto handle=state.publish(localPublicationDescriptor(1,73,41),[&](const auto&,std::string*){
  queued[0]=std::async(std::launch::async,[&]{
   if(!MoEOverlayParticipantNativeRetirementReplayAccess::enqueue([]{return true;},originalCancel))
    throw std::runtime_error("recorded successful peer unexpectedly failed");
  });
  queued[1]=std::async(std::launch::async,[&]{
   EXPECT_NE(std::this_thread::get_id(),control);gate.wait();
   if(!MoEOverlayParticipantNativeRetirementReplayAccess::enqueue([&]{
    if(outcome==2)throw std::runtime_error("recorded delayed enqueue failure");
    return outcome==0;
   },originalCancel))throw std::runtime_error("recorded delayed enqueue failure");
  });
  return true;
 });
 ASSERT_TRUE(handle.ok)<<handle.error;EXPECT_TRUE(state.active());
 EXPECT_EQ(queued[1].wait_for(std::chrono::seconds(0)),std::future_status::timeout)
  <<"Publication returns while the original queued worker is still delayed";
 EXPECT_FALSE(state.complete(41,0,&error));EXPECT_EQ(consumed,0u);
 release.set_value();released=true;ASSERT_EQ(queued[1].wait_for(std::chrono::seconds(2)),std::future_status::ready);
 // This check PRECEDES future.get/native terminal join. A false/throwing
 // producer must cancel its original source before that source could wait.
 EXPECT_EQ(immediateCancels.load(std::memory_order_acquire),outcome?1u:0u);
 const auto joinQueues=[&](std::string* failure){
  const bool queuesComplete=MoEOverlayParticipantNativeRetirementReplayAccess::joinMutable(queued,[&](auto& future){++consumed;future.get();},
   []()noexcept{},failure);
  if(!queuesComplete)return false;
  return MoEOverlayParticipantNativeRetirementReplayAccess::join(originalNativePeers,
   [&](int peer){nativeVisited.push_back(peer);},[]()noexcept{},failure);
 };
 const bool retired=state.retire(handle,joinQueues,&error);EXPECT_EQ(consumed,2u);
 if(!outcome){
  ASSERT_TRUE(retired)<<error;EXPECT_EQ(nativeVisited,(std::vector<int>{0,1,2,3}));EXPECT_FALSE(state.active());EXPECT_TRUE(state.complete(41,0,&error));
 }else{
  EXPECT_FALSE(retired);EXPECT_EQ(error,"recorded delayed enqueue failure");ASSERT_TRUE(state.active());
  EXPECT_FALSE(state.complete(41,0,&error));EXPECT_FALSE(state.begin({10,4,41},&error));
  const auto drain=[&](bool missing,std::string* failure){return MoEOverlayParticipantNativeRetirementReplayAccess::join(originalNativePeers,
   [&](int peer){nativeVisited.push_back(peer);if(missing&&peer==1)throw std::logic_error("recorded missing original native terminal");},
   []()noexcept{},failure);};
  EXPECT_FALSE(state.abort(41,7,[&](std::string* failure){return drain(true,failure);},&error));
  EXPECT_EQ(nativeVisited,(std::vector<int>{0,1,2,3}));ASSERT_TRUE(state.active());
  nativeVisited.clear();ASSERT_TRUE(state.abort(41,7,[&](std::string* failure){return drain(false,failure);},&error));
  EXPECT_EQ(nativeVisited,(std::vector<int>{0,1,2,3}));EXPECT_FALSE(state.active());
  EXPECT_FALSE(state.complete(41,0,&error));EXPECT_FALSE(state.begin({10,4,41},&error));
 }
}
}
/** @test Original queued successful peers stay owned until every queue and terminal duty is consumed. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherKeepsQueuedSuccessUntilOriginalJoins){replayQueuedLocalPublication(0);}
/** @test Delayed false/throw outcomes cancel before joining, retain successful peers and never authorize reuse. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherCancelsDelayedFailureBeforeAnyJoin){
 replayQueuedLocalPublication(1);replayQueuedLocalPublication(2);
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test Actual publish/state/join guards cannot permit continuation while an original SDK enqueue remains pending. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalOriginalEnqueuesCompleteBeforeContinuationReplay)
{
    using Access=MoEOverlayParticipantNativeRetirementReplayAccess;
    enum class FirstBoundary {HostJoin,PublicationReturned};
    for (const int outcome : {0,1,2}) for (unsigned repeat=0;repeat<20;++repeat) {
        LocalPublicationState state(localPublicationConfig());std::string error;
        ASSERT_TRUE(state.begin({9,3,41},&error));
        std::promise<void> release;auto gate=release.get_future().share();
        std::array<std::future<void>,2> queued;
        std::promise<FirstBoundary> first;auto first_boundary=first.get_future();
        std::atomic_flag boundary_reported=ATOMIC_FLAG_INIT;
        const auto report=[&](FirstBoundary value) {
            if (!boundary_reported.test_and_set(std::memory_order_acq_rel)) first.set_value(value);
        };
        std::atomic<unsigned> cancelled{0};unsigned consumed=0,native_terminal_observations=0;
        const auto cancel=[&]() noexcept {cancelled.fetch_add(1,std::memory_order_release);};
        bool released=false;
        struct OriginalQueuedLifetime {
            std::promise<void>& release;bool& released;std::array<std::future<void>,2>& queued;
            ~OriginalQueuedLifetime(){if(!released)release.set_value();for(auto& future:queued)if(future.valid())future.wait();}
        } lifetime{release,released,queued};
        auto publication=std::async(std::launch::async,[&] {
            auto result=state.publishOriginalFollowers(localPublicationDescriptor(1,73,41),[&](const auto&,std::string*) {
                queued[0]=std::async(std::launch::async,[]{});
                queued[1]=std::async(std::launch::async,[&] {
                    gate.wait();
                    if (!Access::enqueue([&] {
                        if (outcome==2) throw std::runtime_error("original delayed SDK enqueue failure");
                        return outcome==0;
                    },cancel)) throw std::runtime_error("original delayed SDK enqueue failure");
                });
                return true;
            },[&](std::string* failure) {
                report(FirstBoundary::HostJoin);
                return Access::joinMutable(queued,[&](auto& future){++consumed;future.get();},cancel,failure);
            });
            report(FirstBoundary::PublicationReturned);return result;
        });
        // This is a causal checkpoint, not a timing/sleep oracle. The original
        // queue-only implementation reports PublicationReturned first instead.
        EXPECT_EQ(first_boundary.get(),FirstBoundary::HostJoin);
        EXPECT_EQ(publication.wait_for(std::chrono::seconds(0)),std::future_status::timeout);
        release.set_value();released=true;
        const auto result=publication.get();EXPECT_EQ(consumed,2u);
        EXPECT_EQ(native_terminal_observations,0u) << "Publication collects host SDK results without GPU terminal observation";
        if (outcome==0) {
            ASSERT_TRUE(result.ok)<<result.error;EXPECT_EQ(cancelled.load(),0u);
            EXPECT_TRUE(state.retire(result,[&](std::string*){++native_terminal_observations;return true;},&error));
            EXPECT_EQ(native_terminal_observations,1u);
        } else {
            EXPECT_FALSE(result.ok);EXPECT_EQ(result.error,"original delayed SDK enqueue failure");
            EXPECT_GT(cancelled.load(),0u);
            EXPECT_TRUE(state.abort(41,7,[](std::string*){return true;},&error));
        }
    }
}
}

namespace llaminar2::test
{
/** @test Original fixed protocol slots retain each draft/verifier handle until exact return retirement. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherRetainsOriginalMTPRing)
{
    LocalPublicationState state(localPublicationConfig());
    std::string error;
    ASSERT_EQ(state.slotCount(), 4u);
    for (std::uint64_t request = 1; request <= 20; ++request)
    {
        ASSERT_TRUE(state.begin({request, 1, 41}, &error)) << error;
        std::uint64_t ordinal = 0;
        for (const int graphs : {2, 4, 3, 2, 4})
        {
            std::array<MoEOverlayPublishedInferenceTransaction, 4> handles;
            for (int index = 0; index < graphs; ++index)
            {
                auto descriptor = localPublicationDescriptor(index + 1 == graphs ? graphs - 1 : 0, ordinal + index, 41);
                if (index + 1 < graphs)
                {
                    descriptor.graph_role = MoEOverlayInferenceGraphRole::MTPDraft;
                    descriptor.draft_depth = graphs - 1;
                    descriptor.sidecar_depth = index;
                }
                handles[index] = state.publishOriginalFollowers(descriptor,
                    [&](const auto &ticket, std::string *)
                    {
                        EXPECT_EQ(ticket.transaction_ordinal, ++ordinal);
                        return true;
                    }, [](std::string *) { return true; });
                ASSERT_TRUE(handles[index].ok) << handles[index].error;
                ASSERT_TRUE(state.activeSlot(handles[index].ticket));
                EXPECT_EQ(*state.activeSlot(handles[index].ticket), handles[index].slot_index);
            }
            EXPECT_FALSE(state.complete(41, 0, &error));
            for (int index = 0; index < graphs; ++index)
            {
                auto foreign = handles[index];
                ++foreign.ticket.command_id;
                EXPECT_FALSE(state.retire(foreign, [](std::string *) { ADD_FAILURE(); return true; }, &error));
                ASSERT_TRUE(state.retire(handles[index], [](std::string *) { return true; }, &error)) << error;
                EXPECT_FALSE(state.activeSlot(handles[index].ticket));
            }
            EXPECT_FALSE(state.active());
        }
        ASSERT_TRUE(state.complete(41, 0, &error)) << error;
    }
}

/** @test A later partial enqueue retains every original slot until all native drains are proven. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherMTPRingRetainsPartialFailure)
{
    LocalPublicationState state(localPublicationConfig());
    std::string error;
    ASSERT_TRUE(state.begin({9, 1, 41}, &error));
    std::array<MoEOverlayPublishedInferenceTransaction, 3> handles;
    for (std::size_t index = 0; index < handles.size(); ++index)
    {
        handles[index] = state.publish(localPublicationDescriptor(0, index, 41),
            [index](const auto &, std::string *failure)
            {
                if (index == 2)
                {
                    *failure = "original later enqueue failure";
                    return false;
                }
                return true;
            });
        EXPECT_EQ(handles[index].ok, index != 2);
    }
    ASSERT_TRUE(state.active());
    for (const auto &handle : handles)
        ASSERT_TRUE(state.activeSlot(handle.ticket));
    std::size_t drains = 0;
    EXPECT_FALSE(state.abort(41, 7, [&](std::string *failure)
    {
        for (std::size_t index = 0; index < state.slotCount(); ++index)
            if (state.activeHandle(index))
                ++drains;
        *failure = "original missing native proof";
        return false;
    }, &error));
    EXPECT_EQ(error, "original missing native proof");
    EXPECT_EQ(drains, 3u);
    for (const auto &handle : handles)
        ASSERT_TRUE(state.activeSlot(handle.ticket));
    ASSERT_TRUE(state.abort(41, 7, [&](std::string *)
    {
        for (std::size_t index = 0; index < state.slotCount(); ++index)
            if (state.activeHandle(index))
                ++drains;
        return true;
    }, &error)) << error;
    EXPECT_EQ(drains, 6u);
    EXPECT_FALSE(state.active());
    EXPECT_FALSE(state.begin({10, 1, 41}, &error));
}

/** @test Exhaustion cannot allocate or overwrite a fifth handle beyond the original ring. */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherMTPRingRejectsCapacityOverrun)
{
    LocalPublicationState state(localPublicationConfig());
    std::string error;
    ASSERT_TRUE(state.begin({9, 1, 41}, &error));
    std::size_t submissions = 0;
    for (std::size_t index = 0; index < state.slotCount(); ++index)
    {
        const auto handle = state.publish(localPublicationDescriptor(0, index, 41),
            [&](const auto &, std::string *) { ++submissions; return true; });
        ASSERT_TRUE(handle.ok) << handle.error;
    }
    EXPECT_FALSE(state.publish(localPublicationDescriptor(0, 5, 41),
        [&](const auto &, std::string *) { ++submissions; return true; }).ok);
    EXPECT_EQ(submissions, state.slotCount());
    ASSERT_TRUE(state.abort(41, 7, [](std::string *) { return true; }, &error)) << error;
    EXPECT_FALSE(state.active());
}

} // namespace llaminar2::test

namespace llaminar2::test
{
/**
 * @test Exact native terminal observation never erases a failed original source cancellation.
 * This exercises the actual cancellation publisher and original submission drain
 * guard with recorded native-proof state; it creates no native owner or permit.
 */
TEST(Test__MoEOverlayInferenceTransaction, RankLocalCallablePublisherAbortRetainsSourceFailureAfterNativeProof)
{
    using Access = MoEOverlayParticipantNativeRetirementReplayAccess;
    LocalPublicationState state(localPublicationConfig());
    std::string error;
    ASSERT_TRUE(state.begin({9, 1, 41}, &error));
    const auto handle = state.publish(localPublicationDescriptor(0, 0, 41),
        [](const auto &, std::string *) { return true; });
    ASSERT_TRUE(handle.ok) << handle.error;
    std::mutex mutex;
    std::exception_ptr source_failure;
    Access::cancelSource([] { throw std::runtime_error("original source retirement refusal"); }, mutex, source_failure);
    ASSERT_TRUE(source_failure);
    const auto original_failure = source_failure;
    bool native_joined = false;
    int native_drains = 0;
    const std::string expected = "Original pooled source cancellation failed: original source retirement refusal";
    const auto abort_drain = [&](std::string *diagnostic)
    {
        return Access::drainCancellation(native_joined, mutex, source_failure,
            [&](std::string *failure)
            {
                ++native_drains;
                native_joined = true;
                *failure = expected;
                return false;
            }, diagnostic);
    };
    EXPECT_FALSE(state.abort(41, 7, abort_drain, &error));
    ASSERT_TRUE(native_joined);
    EXPECT_EQ(error, expected);
    EXPECT_EQ(native_drains, 1);
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        EXPECT_FALSE(state.abort(41, 7, abort_drain, &error));
        EXPECT_EQ(error, expected);
        EXPECT_EQ(native_drains, 1) << "An already observed event cannot be waited again";
        EXPECT_TRUE(state.activeSlot(handle.ticket));
        EXPECT_EQ(source_failure, original_failure);
    }
    std::exception_ptr successful_source;
    EXPECT_TRUE(Access::drainCancellation(true, mutex, successful_source,
        [](std::string *) { ADD_FAILURE() << "Already observed native terminal was waited again"; return false; }, &error));
}
} // namespace llaminar2::test
