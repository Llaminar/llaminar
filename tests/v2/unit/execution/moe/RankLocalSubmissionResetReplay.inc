/**
 * @file RankLocalSubmissionResetReplay.inc
 * @brief Replay original queued worker/admission metadata through the production reset guard.
 *
 * These records prove metadata ordering, not native readiness or GPU completion.
 * A worker-written flag is counted so a pending enqueue cannot be observed as
 * though a false launch flag proved there was no native work.
 */
namespace llaminar2::test
{
    TEST(Test__MoEOverlayInferenceTransaction,
         RankLocalCallablePublisherResetRetainsQueuedAndUnprovenOriginalWork)
    {
        struct ObservedFlag
        {
            bool value = false;
            int *reads = nullptr;
            operator bool() const { ++*reads; return value; }
        };
        struct Enqueue
        {
            bool queued = true;
            bool joined = false;
            ObservedFlag launch_attempted;
        };
        struct Runtime { MoEOverlayInferenceBoundaryReceipt inference_boundary_receipt; };
        struct Boundary
        {
            Runtime *runtime = nullptr;
            std::uint64_t generation = 0;
            ObservedFlag graph_submitted;
        };
        Runtime runtime;
        ASSERT_EQ(runtime.inference_boundary_receipt.beginSubmission(), 1u);
        int attempt_reads = 0, submitted_reads = 0;
        std::array enqueues{Enqueue{true, false, {false, &attempt_reads}}};
        std::array boundaries{Boundary{&runtime, 1u, {false, &submitted_reads}}};
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, false), std::logic_error);
        EXPECT_EQ(attempt_reads, 0);
        EXPECT_EQ(submitted_reads, 0) << "Queued workers may still publish the actual launch flag";

        enqueues[0].joined = true;
        enqueues[0].launch_attempted.value = true;
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, false), std::logic_error);
        EXPECT_EQ(submitted_reads, 0);
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true), std::logic_error) << "Unlaunched admission still needs rollback";
        ASSERT_TRUE(runtime.inference_boundary_receipt.cancelUnsubmitted(1u));
        enqueues[0].launch_attempted.value = false;
        EXPECT_NO_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, false));

        ASSERT_EQ(runtime.inference_boundary_receipt.beginSubmission(), 1u);
        enqueues[0].launch_attempted.value = true;
        boundaries[0].graph_submitted.value = true;
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true), std::logic_error)
            << "Proven failed work cannot become a successful reusable submission";
        ASSERT_TRUE(runtime.inference_boundary_receipt.completeSubmission(1u));
        EXPECT_NO_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true));
        // Each successful native/event receipt advances the SAME runtime's
        // monotonic prefix. Reusing an older original host slot is safe even
        // after newer receipts complete, while its own native proof is mandatory.
        for (std::uint64_t generation = 2; generation <= 8; ++generation)
        {
            ASSERT_EQ(runtime.inference_boundary_receipt.beginSubmission(), generation);
            EXPECT_NO_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
                enqueues, boundaries, true)) << "Newer pending work does not revoke an older exact receipt";
            EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
                enqueues, boundaries, false), std::logic_error);
            ASSERT_TRUE(runtime.inference_boundary_receipt.completeSubmission(generation));
            EXPECT_NO_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
                enqueues, boundaries, true)) << "Later successful completions preserve the original prefix";
        }

        // A failed/unlaunched receipt needs its exact rollback, not a guessed
        // completion from another admission. This process-terminal branch is
        // deliberately distinct from a successfully submitted original graph.
        enqueues[0].launch_attempted.value = false;
        boundaries[0].graph_submitted.value = false;
        boundaries[0].generation = 9u;
        ASSERT_EQ(runtime.inference_boundary_receipt.beginSubmission(), 9u);
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, false), std::logic_error);
        ASSERT_TRUE(runtime.inference_boundary_receipt.cancelUnsubmitted(9u));
        EXPECT_NO_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, false));
        ASSERT_EQ(runtime.inference_boundary_receipt.beginSubmission(), 9u);
        ASSERT_TRUE(runtime.inference_boundary_receipt.completeSubmission(9u));
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, false), std::logic_error) << "Later work is not the old unlaunched rollback identity";
        enqueues[0].launch_attempted.value = true;
        boundaries[0].graph_submitted.value = true;
        boundaries[0].generation = 10u;
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true), std::logic_error) << "Native proof cannot supply a missing receipt";
        ASSERT_EQ(runtime.inference_boundary_receipt.beginSubmission(), 10u);
        ASSERT_EQ(runtime.inference_boundary_receipt.beginSubmission(), 11u);
        EXPECT_FALSE(runtime.inference_boundary_receipt.completeSubmission(11u))
            << "A newer completion cannot skip the original failed/uncompleted generation";
        EXPECT_EQ(runtime.inference_boundary_receipt.completedGeneration(), 9u);
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true), std::logic_error);
        boundaries[0].generation = 1u;
        boundaries[0].runtime = nullptr;
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true), std::logic_error);
        boundaries[0].runtime = &runtime;
        boundaries[0].generation = 0u;
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true), std::logic_error);
    }
}
