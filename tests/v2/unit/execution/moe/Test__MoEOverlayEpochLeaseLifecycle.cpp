/**
 * @file Test__MoEOverlayEpochLeaseLifecycle.cpp
 * @brief Adversarial CPU-only tests for overlay epoch host submission ownership.
 *
 * The production lifecycle is shared by inference and background maintenance.
 * These tests deliberately interleave the two host workers while performing no
 * GPU work, locking down the rule that a partial acquire/release recipe is never
 * a published semantic state.
 */

#include "execution/moe/MoEOverlayEpochLeaseLifecycle.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace llaminar2::test
{
    namespace
    {
        using State = MoEOverlayEpochLeaseState;

        static_assert(
            !std::is_move_constructible_v<
                MoEOverlayEpochLeaseLifecycle::Submission>);
        static_assert(
            !std::is_move_assignable_v<
                MoEOverlayEpochLeaseLifecycle::Submission>);

        /** Stable fake stream identity; lifecycle tests never dereference it. */
        int release_stream_storage = 0;

        /** @return Non-null stream identity used by typed release receipts. */
        void *releaseStream() noexcept
        {
            return &release_stream_storage;
        }

        /** Publish the ordinary external-forward starting point for a test. */
        void publishExternalForward(MoEOverlayEpochLeaseLifecycle &lifecycle)
        {
            auto submission = lifecycle.beginSubmission();
            ASSERT_EQ(submission.state(), State::Idle);
            ASSERT_TRUE(submission.commit(State::ExternalForward));
        }
    } // namespace

    /** @brief KV-only sidecars cannot pin placement before main admission. */
    TEST(MoEOverlayEpochLeaseLifecycle, SidecarResidencyFollowsExpertAccess)
    {
        using Owner = MoEOverlaySidecarEpochOwnership;
        for (const bool coordinated : {false, true})
        {
            EXPECT_EQ(moeOverlaySidecarEpochOwnership(
                          MTPSidecarCaptureRole::KVOnly, coordinated),
                      Owner::NoExpertAccess);
            for (const auto role : {MTPSidecarCaptureRole::Full,
                                    MTPSidecarCaptureRole::Chained})
            {
                EXPECT_EQ(moeOverlaySidecarEpochOwnership(role, coordinated),
                          coordinated ? Owner::GraphSequence
                                      : Owner::ExternalReader);
            }
        }
        EXPECT_THROW(moeOverlaySidecarEpochOwnership(
                         static_cast<MTPSidecarCaptureRole>(255), true),
                     std::logic_error);
    }

    /**
     * @brief A failed enqueue leaves the last complete semantic owner unchanged.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         AbandonedSubmissionPublishesNoTemporaryState)
    {
        MoEOverlayEpochLeaseLifecycle lifecycle;
        publishExternalForward(lifecycle);

        {
            auto release = lifecycle.beginSubmission();
            ASSERT_EQ(release.state(), State::ExternalForward);
            /* Model a backend enqueue failure by leaving without commit. */
        }

        EXPECT_EQ(lifecycle.load(), State::ExternalForward);
        auto release = lifecycle.beginSubmission();
        ASSERT_TRUE(release.publishRelease(releaseStream()));
        EXPECT_EQ(lifecycle.load(), State::ReleasePublished);
        EXPECT_EQ(
            release.publishedReleaseProducerStream(),
            releaseStream());
    }

    /**
     * @brief Invalid ownership edges cannot be published through the typed API.
     */
    TEST(MoEOverlayEpochLeaseLifecycle, RejectsIllegalSemanticTransitions)
    {
        MoEOverlayEpochLeaseLifecycle lifecycle;
        auto idle = lifecycle.beginSubmission();
        EXPECT_FALSE(idle.commit(State::HostedChildForward));
        EXPECT_EQ(lifecycle.load(), State::Idle);
    }

    /**
     * @brief A release state cannot exist without its exact ordering payload.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         PublishedReleaseRequiresNonNullTypedReceipt)
    {
        MoEOverlayEpochLeaseLifecycle lifecycle;
        publishExternalForward(lifecycle);

        {
            auto invalid = lifecycle.beginSubmission();
            EXPECT_FALSE(invalid.commit(State::ReleasePublished));
            EXPECT_FALSE(invalid.publishRelease(nullptr));
            EXPECT_EQ(
                invalid.publishedReleaseProducerStream(),
                nullptr);
        }
        EXPECT_EQ(lifecycle.load(), State::ExternalForward);

        auto release = lifecycle.beginSubmission();
        ASSERT_TRUE(release.publishRelease(releaseStream()));
        EXPECT_EQ(lifecycle.load(), State::ReleasePublished);
        EXPECT_EQ(
            release.publishedReleaseProducerStream(),
            releaseStream());
    }

    /**
     * @brief A forward may atomically consume a reader acquired by its prelude.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         ExistingExternalReaderBecomesForwardOwnerWithoutReleaseWindow)
    {
        MoEOverlayEpochLeaseLifecycle lifecycle;
        {
            auto acquire = lifecycle.beginSubmission();
            ASSERT_TRUE(acquire.commit(State::ExternalReader));
        }
        {
            auto forward = lifecycle.beginSubmission();
            ASSERT_EQ(forward.state(), State::ExternalReader);
            ASSERT_TRUE(forward.commit(State::ExternalForward));
        }
        EXPECT_EQ(lifecycle.load(), State::ExternalForward);
    }

    /**
     * @brief Background observation never steals any live inference lease.
     *
     * This is exhaustive over the public state enum.  Most importantly, an
     * ExternalReader is an admitted request waiting for its forward submission,
     * so the observer must defer rather than clearing the device ticket.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         BackgroundObservationDefersToEveryInferenceOwner)
    {
        using Action = MoEOverlayEpochObservationAction;
        EXPECT_EQ(
            moeOverlayEpochObservationAction(State::Idle),
            Action::SubmitAtIdle);
        EXPECT_EQ(
            moeOverlayEpochObservationAction(State::ReleasePublished),
            Action::WaitForPublishedRelease);

        constexpr State live_owners[]{
            State::ExternalReader,
            State::ExternalSequence,
            State::HostedParent,
            State::CapturedMainForward,
            State::ExternalForward,
            State::HostedChildForward,
        };
        for (const State owner : live_owners)
        {
            EXPECT_EQ(
                moeOverlayEpochObservationAction(owner),
                Action::DeferToInferenceOwner);
        }

        MoEOverlayEpochLeaseLifecycle lifecycle;
        {
            auto admission = lifecycle.beginSubmission();
            ASSERT_TRUE(admission.commit(State::ExternalReader));
        }
        ASSERT_EQ(
            moeOverlayEpochObservationAction(lifecycle.load()),
            Action::DeferToInferenceOwner);
        EXPECT_EQ(lifecycle.load(), State::ExternalReader);

        /* The admitted inference still owns the only legal promotion. */
        auto forward = lifecycle.beginSubmission();
        ASSERT_TRUE(forward.commit(State::ExternalForward));
        EXPECT_EQ(lifecycle.load(), State::ExternalForward);
    }

    /**
     * @brief A placement writer closes or joins exactly one completed boundary.
     *
     * The classification is exhaustive over the public state enum.  In
     * particular, a self-contained forward's `ReleasePublished` receipt is
     * joined rather than interpreted as an ambient reader to release again.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         PlacementMaintenanceClaimDistinguishesCloseFromJoin)
    {
        using Action = MoEOverlayEpochPlacementMaintenanceAction;
        EXPECT_EQ(
            moeOverlayEpochPlacementMaintenanceAction(State::ExternalReader),
            Action::CloseExternalReader);
        EXPECT_EQ(
            moeOverlayEpochPlacementMaintenanceAction(State::ReleasePublished),
            Action::JoinPublishedRelease);
        EXPECT_EQ(
            moeOverlayEpochPlacementMaintenanceAction(State::Idle),
            Action::RejectMissingBoundary);

        constexpr State live_non_reader_owners[]{
            State::ExternalSequence,
            State::HostedParent,
            State::CapturedMainForward,
            State::ExternalForward,
            State::HostedChildForward,
        };
        for (const State owner : live_non_reader_owners)
        {
            EXPECT_EQ(
                moeOverlayEpochPlacementMaintenanceAction(owner),
                Action::RejectLiveInferenceOwner);
        }

        EXPECT_STREQ(
            moeOverlayEpochMaintenanceBoundarySourceName(
                MoEOverlayEpochMaintenanceBoundarySource::
                    GraphLaunchDependency),
            "graph_launch_dependency");
        EXPECT_STREQ(
            moeOverlayEpochMaintenanceBoundarySourceName(
                MoEOverlayEpochMaintenanceBoundarySource::
                    AuthenticatedDispatchTicket),
            "authenticated_dispatch_ticket");
    }

    /**
     * @brief Forward ownership follows transaction shape, never topology.
     *
     * An ordinary main graph is a complete captured transaction under both
     * host- and device-authoritative placement. Auxiliary graphs are members of
     * a larger sequence and therefore retain the explicit external envelope.
     * This distinction removes a backend scheduling edge without changing the
     * authority that selects or publishes placement.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         ForwardSubmissionPolicyIsTotalAndTransactionOwned)
    {
        using Policy = MoEOverlayForwardEpochSubmissionPolicy;

        for (const bool main_inference : {false, true})
        {
            EXPECT_EQ(
                moeOverlayForwardEpochSubmissionPolicy(
                    /*has_epoch_binding=*/false,
                    main_inference),
                Policy::Unbound);
        }

        EXPECT_EQ(
            moeOverlayForwardEpochSubmissionPolicy(
                /*has_epoch_binding=*/true,
                /*main_inference=*/true),
            Policy::CapturedMainTransaction);
        EXPECT_EQ(
            moeOverlayForwardEpochSubmissionPolicy(
                /*has_epoch_binding=*/true,
                /*main_inference=*/false),
            Policy::RetainedPerForwardTransaction);
    }

    /**
     * @brief The next graph observes a complete release, never release submission.
     *
     * The first worker holds the host enqueue authority while modeling the event
     * wait, retained release launch, and event record.  The second worker begins
     * the next graph concurrently.  It may briefly wait for that host recipe, but
     * once admitted it sees the immutable release receipt and can enqueue its GPU
     * event wait; neither worker waits for device completion.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         ConcurrentReleaseAndAcquireFormOneCompleteEventEdge)
    {
        using namespace std::chrono_literals;

        MoEOverlayEpochLeaseLifecycle lifecycle;
        publishExternalForward(lifecycle);

        std::promise<void> release_submission_open;
        std::promise<void> release_gate;
        std::shared_future<void> allow_release_commit =
            release_gate.get_future().share();
        std::promise<void> acquire_attempted;
        std::promise<State> acquire_observed_state;

        std::thread release_worker(
            [&]
            {
                auto release = lifecycle.beginSubmission();
                if (release.state() != State::ExternalForward)
                {
                    release_submission_open.set_exception(
                        std::make_exception_ptr(
                            std::runtime_error("release lost external forward")));
                    return;
                }
                release_submission_open.set_value();
                allow_release_commit.wait();
                if (!release.publishRelease(releaseStream()))
                    std::terminate();
            });

        release_submission_open.get_future().get();
        std::thread acquire_worker(
            [&]
            {
                acquire_attempted.set_value();
                auto acquire = lifecycle.beginSubmission();
                acquire_observed_state.set_value(acquire.state());
                if (!acquire.commit(State::ExternalForward))
                    std::terminate();
            });

        acquire_attempted.get_future().wait();
        auto observed = acquire_observed_state.get_future();
        EXPECT_EQ(observed.wait_for(20ms), std::future_status::timeout);
        EXPECT_EQ(lifecycle.load(), State::ExternalForward);

        release_gate.set_value();
        release_worker.join();
        ASSERT_EQ(observed.get(), State::ReleasePublished);
        acquire_worker.join();
        EXPECT_EQ(lifecycle.load(), State::ExternalForward);
    }


    /**
     * @brief Maintenance cannot observe a release while acquire retires it.
     *
     * This is the exact host interleaving exposed by the real 122B mixed-vendor
     * Dynamic campaign: maintenance saw the old atomic `ReleasePublished`
     * state while the next acquire had already cleared a separately stored raw
     * producer-stream pointer.  A typed observation now holds the same
     * submission authority as acquire, so it sees either the complete receipt
     * or the complete successor owner and never a torn pair.
     */
    TEST(MoEOverlayEpochLeaseLifecycle,
         ConcurrentObservationAndAcquireCannotTearReleaseReceipt)
    {
        using namespace std::chrono_literals;

        MoEOverlayEpochLeaseLifecycle lifecycle;
        publishExternalForward(lifecycle);
        {
            auto release = lifecycle.beginSubmission();
            ASSERT_TRUE(release.publishRelease(releaseStream()));
        }

        std::promise<void> acquire_attempted;
        std::promise<State> acquire_observed_state;
        std::promise<void *> acquire_observed_stream;
        std::thread acquire_worker;
        auto observed_state = acquire_observed_state.get_future();
        auto observed_stream = acquire_observed_stream.get_future();

        {
            auto maintenance_observation = lifecycle.beginSubmission();
            ASSERT_EQ(
                maintenance_observation.state(),
                State::ReleasePublished);
            ASSERT_EQ(
                maintenance_observation.publishedReleaseProducerStream(),
                releaseStream());

            acquire_worker = std::thread(
                [&]
                {
                    acquire_attempted.set_value();
                    auto acquire = lifecycle.beginSubmission();
                    acquire_observed_state.set_value(acquire.state());
                    acquire_observed_stream.set_value(
                        acquire.publishedReleaseProducerStream());
                    if (!acquire.commit(State::ExternalForward))
                        std::terminate();
                });

            acquire_attempted.get_future().wait();
            EXPECT_EQ(
                observed_state.wait_for(20ms),
                std::future_status::timeout);
            EXPECT_EQ(lifecycle.load(), State::ReleasePublished);
        }

        ASSERT_EQ(observed_state.get(), State::ReleasePublished);
        EXPECT_EQ(observed_stream.get(), releaseStream());
        acquire_worker.join();
        EXPECT_EQ(lifecycle.load(), State::ExternalForward);

        auto successor = lifecycle.beginSubmission();
        EXPECT_EQ(successor.state(), State::ExternalForward);
        EXPECT_EQ(successor.publishedReleaseProducerStream(), nullptr);
    }
    /** @test Exact terminal publication is repeatable across changes of stream and request reset. */
    TEST(MoEOverlayEpochLeaseLifecycle, CompleteCapturedHostedParentReplaysAndResets)
    {
        MoEOverlayEpochLeaseLifecycle lifecycle;
        int other_stream_storage = 0;
        void* const other_stream = &other_stream_storage;
        void* previous_stream = nullptr;
        for (int replay = 0; replay < 20; ++replay) {
            SCOPED_TRACE(replay);
            void* const stream = replay % 2 == 0 ? releaseStream() : other_stream;
            int calls = 0;
            ASSERT_TRUE(lifecycle.submitCompleteCapturedHostedParent(stream,
                [&](void* actual) {
                    EXPECT_EQ(actual, stream);
                    EXPECT_NE(previous_stream, nullptr);
                    EXPECT_EQ(lifecycle.load(), State::ReleasePublished);
                    ++calls;
                    return true;
                },
                [&](void* actual) {
                    EXPECT_EQ(actual, stream);
                    EXPECT_EQ(lifecycle.load(), State::CapturedHostedParent);
                    EXPECT_EQ(calls, previous_stream ? 1 : 0);
                    ++calls;
                    return true;
                },
                [&](void* actual) {
                    EXPECT_EQ(actual, stream);
                    EXPECT_EQ(lifecycle.load(), State::CapturedHostedParent);
                    ++calls;
                    return true;
                }));
            EXPECT_EQ(calls, previous_stream ? 3 : 2);
            auto terminal = lifecycle.beginSubmission();
            EXPECT_EQ(terminal.state(), State::ReleasePublished);
            EXPECT_EQ(terminal.publishedReleaseProducerStream(), stream);
            previous_stream = stream;
            if (replay % 4 == 3) {
                ASSERT_TRUE(terminal.commit(State::Idle));
                previous_stream = nullptr;
            }
        }
    }

    /** @test Competing host submissions cannot borrow, release or duplicate an admitted complete parent. */
    TEST(MoEOverlayEpochLeaseLifecycle, CompleteCapturedHostedParentRejectsConcurrentOwners)
    {
        MoEOverlayEpochLeaseLifecycle lifecycle;
        std::promise<void> entered;
        std::promise<void> resume;
        auto worker = std::async(std::launch::async, [&] {
            return lifecycle.submitCompleteCapturedHostedParent(releaseStream(),
                [](void*) { return false; },
                [&](void*) {
                    entered.set_value();
                    resume.get_future().wait();
                    return true;
                },
                [](void*) { return true; });
        });
        entered.get_future().wait();
        {
            auto competitor = lifecycle.beginSubmission();
            EXPECT_EQ(competitor.state(), State::CapturedHostedParent);
            EXPECT_EQ(competitor.publishedReleaseProducerStream(), nullptr);
            EXPECT_FALSE(competitor.commit(State::HostedChildForward));
            EXPECT_FALSE(competitor.publishRelease(releaseStream()));
            EXPECT_EQ(moeOverlayEpochObservationAction(competitor.state()),
                MoEOverlayEpochObservationAction::DeferToInferenceOwner);
            EXPECT_EQ(moeOverlayEpochPlacementMaintenanceAction(competitor.state()),
                MoEOverlayEpochPlacementMaintenanceAction::RejectLiveInferenceOwner);
        }
        int unexpected_calls = 0;
        const auto unexpected = [&](void*) { ++unexpected_calls; return true; };
        EXPECT_FALSE(lifecycle.submitCompleteCapturedHostedParent(
            releaseStream(), unexpected, unexpected, unexpected));
        EXPECT_EQ(unexpected_calls, 0);
        resume.set_value();
        ASSERT_TRUE(worker.get());
        EXPECT_EQ(lifecycle.load(), State::ReleasePublished);
    }

    /** @test A rejected predecessor wait preserves its receipt, and failed launch publishes no release. */
    TEST(MoEOverlayEpochLeaseLifecycle, CompleteCapturedHostedParentFailurePreservesOwnership)
    {
        MoEOverlayEpochLeaseLifecycle lifecycle;
        publishExternalForward(lifecycle);
        {
            auto prior = lifecycle.beginSubmission();
            ASSERT_TRUE(prior.publishRelease(releaseStream()));
        }
        int calls = 0;
        const auto succeeds = [&](void*) { ++calls; return true; };
        EXPECT_FALSE(lifecycle.submitCompleteCapturedHostedParent(
            nullptr, succeeds, succeeds, succeeds));
        for (std::uintptr_t default_stream : {1u, 2u})
            EXPECT_FALSE(lifecycle.submitCompleteCapturedHostedParent(
                reinterpret_cast<void*>(default_stream), succeeds, succeeds, succeeds));
        EXPECT_FALSE(lifecycle.submitCompleteCapturedHostedParent(
            releaseStream(), [](void*) { return false; }, succeeds, succeeds));
        EXPECT_EQ(calls, 0);
        {
            auto prior = lifecycle.beginSubmission();
            EXPECT_EQ(prior.state(), State::ReleasePublished);
            EXPECT_EQ(prior.publishedReleaseProducerStream(), releaseStream());
        }
        EXPECT_THROW((void)lifecycle.submitCompleteCapturedHostedParent(
            releaseStream(), succeeds, [](void*) { return false; }, succeeds), std::runtime_error);
        EXPECT_EQ(calls, 1);
        auto unfinished = lifecycle.beginSubmission();
        EXPECT_EQ(unfinished.state(), State::CapturedHostedParent);
        EXPECT_EQ(unfinished.publishedReleaseProducerStream(), nullptr);
        EXPECT_FALSE(unfinished.publishRelease(releaseStream()));
    }

    /** @test A graph already enqueued without its exact terminal event is fatal. */
    TEST(MoEOverlayEpochLeaseLifecycle, CompleteCapturedHostedParentRequiresRecordedTerminal)
    {
        EXPECT_DEATH({
            MoEOverlayEpochLeaseLifecycle lifecycle;
            (void)lifecycle.submitCompleteCapturedHostedParent(releaseStream(),
                [](void*) { return true; }, [](void*) { return true; }, [](void*) { return false; });
        }, "");
    }

} // namespace llaminar2::test
