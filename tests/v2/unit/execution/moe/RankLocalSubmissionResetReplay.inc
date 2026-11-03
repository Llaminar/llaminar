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
        boundaries[0].generation = 0u;
        EXPECT_THROW(MoEOverlayParticipantNativeRetirementReplayAccess::requireRetired(
            enqueues, boundaries, true), std::logic_error);
    }
}
