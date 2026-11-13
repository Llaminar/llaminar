/**
 * @file PreparedProjectionIntersectionContract.inc
 * @brief Device-free replay of actual checked projection publication and operation drain.
 *
 * Scripted physical completion is a protocol oracle, not a native ACK or device
 * readiness claim. The production wrapper/coverage owner execute unchanged.
 */
namespace
{
    struct IntersectionOperationState
    {
        bool finalized = false;
        bool ready = false;
        bool failed = false;
        bool aborted = false;
        bool drained = false;
        unsigned polls = 0;
        ExpertTierProjectionTransferMeasurement measurement{3, 64, 7, 2, 3, 1};
    };

    /** @brief Script a finalized recipient whose original outer completion is delayed. */
    class IntersectionPhysicalOperation final : public IMoEOverlayTierTransferOperation
    {
    public:
        explicit IntersectionPhysicalOperation(std::shared_ptr<IntersectionOperationState> state)
            : state_(std::move(state))
        {
        }
        MoEOverlayResidencyWaveProgress poll(std::string *error) noexcept override
        {
            ++state_->polls;
            state_->finalized = true;
            if (state_->failed)
            {
                if (error)
                    *error = "original physical completion failed";
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            return state_->ready ? MoEOverlayResidencyWaveProgress::Ready
                                 : MoEOverlayResidencyWaveProgress::Pending;
        }
        void abort() noexcept override
        {
            state_->aborted = true;
        }
        MoEOverlayResidencyWaveProgress pollAbort(std::string *) noexcept override
        {
            return state_->drained ? MoEOverlayResidencyWaveProgress::Ready
                                   : MoEOverlayResidencyWaveProgress::Pending;
        }
        std::optional<ExpertTierProjectionTransferMeasurement>
        completedMeasurement() const noexcept override
        {
            return state_->ready && !state_->aborted ? std::optional{state_->measurement}
                                                     : std::nullopt;
        }

    private:
        std::shared_ptr<IntersectionOperationState> state_;
    };

    using ProjectionRectangle = MoEExpertProjectionOwnership::Projection;
    /** @brief Two independent source strides tile one shifted original destination. */
    std::vector<ExpertTierWeightProjectionIntersection> splitProjectionCoverage()
    {
        const ProjectionRectangle destination{
            MoEProjectionResidence::ParticipantInputSlice, 12, 16, 2, 6, 4, 8};
        const ProjectionRectangle first{
            MoEProjectionResidence::ParticipantInputSlice, 12, 16, 0, 10, 2, 6};
        const ProjectionRectangle second{
            MoEProjectionResidence::ParticipantInputSlice, 12, 16, 1, 9, 8, 6};
        return {ExpertTierWeightProjectionIntersection::between(first, destination),
                ExpertTierWeightProjectionIntersection::between(second, destination)};
    }

    /** @brief Retain all required sibling roles so completePayload isolates the tested gate. */
    std::shared_ptr<MoEOverlayPreparedExpertArrival> readySiblingArrival()
    {
        auto arrival = std::make_shared<MoEOverlayPreparedExpertArrival>();
        EXPECT_TRUE(arrival->publish(ExpertTierWeightProjection::Up, payload(800).up()));
        EXPECT_TRUE(arrival->publish(ExpertTierWeightProjection::Down, payload(900).down()));
        return arrival;
    }
} // namespace

TEST(PreparedProjectionIntersectionContract,
     CompleteDisjointCoverageRejectsGapsOverlapAndForeignCoordinates)
{
    auto arrival = readySiblingArrival();
    auto engine = payload(100).gate();
    auto provider = [engine] { return engine; };
    auto pieces = splitProjectionCoverage();
    EXPECT_NO_THROW(MoEOverlayPreparedProjectionCompletion(
        pieces, arrival, ExpertTierWeightProjection::Gate, provider));
    auto gap = pieces;
    gap.pop_back();
    EXPECT_THROW(MoEOverlayPreparedProjectionCompletion(gap, arrival,
                                                        ExpertTierWeightProjection::Gate, provider),
                 std::invalid_argument);
    auto duplicate = pieces;
    duplicate.push_back(pieces.front());
    EXPECT_THROW(MoEOverlayPreparedProjectionCompletion(duplicate, arrival,
                                                        ExpertTierWeightProjection::Gate, provider),
                 std::invalid_argument);
    const ProjectionRectangle foreign{MoEProjectionResidence::ExpertOwner, 12, 16, 0, 12, 0, 16};
    auto changed = pieces;
    changed[1] = ExpertTierWeightProjectionIntersection::between(foreign, foreign);
    EXPECT_THROW(MoEOverlayPreparedProjectionCompletion(changed, arrival,
                                                        ExpertTierWeightProjection::Gate, provider),
                 std::invalid_argument);
    EXPECT_THROW(MoEOverlayPreparedProjectionCompletion({}, arrival,
                                                        ExpertTierWeightProjection::Gate, provider),
                 std::invalid_argument);
    auto completion = std::make_shared<MoEOverlayPreparedProjectionCompletion>(
        pieces, arrival, ExpertTierWeightProjection::Gate, provider);
    MoEOverlayPreparedProjectionOperation first(std::make_unique<ReadyProjectionTransfer>(),
                                                completion, 0);
    EXPECT_THROW(MoEOverlayPreparedProjectionOperation(std::make_unique<ReadyProjectionTransfer>(),
                                                       completion, 0),
                 std::invalid_argument);
    EXPECT_THROW(MoEOverlayPreparedProjectionOperation(std::make_unique<ReadyProjectionTransfer>(),
                                                       completion, 2),
                 std::invalid_argument);
}

TEST(PreparedProjectionIntersectionContract,
     FinalizedStorageWaitsForEveryOriginalOuterCompletionAndKeepsPieceTiming)
{
    for (int replay = 0; replay < 3; ++replay)
    {
        auto arrival = readySiblingArrival();
        auto engine = payload(100).gate();
        unsigned provider_calls = 0;
        auto completion = std::make_shared<MoEOverlayPreparedProjectionCompletion>(
            splitProjectionCoverage(), arrival, ExpertTierWeightProjection::Gate,
            [engine, &provider_calls]
            {
                ++provider_calls;
                return engine;
            });
        auto a = std::make_shared<IntersectionOperationState>();
        auto b = std::make_shared<IntersectionOperationState>();
        a->measurement.sequence = 41;
        b->measurement.sequence = 62;
        MoEOverlayPreparedProjectionOperation first(
            std::make_unique<IntersectionPhysicalOperation>(a), completion, 0);
        MoEOverlayPreparedProjectionOperation last(
            std::make_unique<IntersectionPhysicalOperation>(b), completion, 1);
        std::string error;
        MoEOverlayPreparedExpertPayload result;
        EXPECT_EQ(last.poll(&error), MoEOverlayResidencyWaveProgress::Pending);
        EXPECT_TRUE(b->finalized);
        EXPECT_FALSE(arrival->completePayload(result));
        EXPECT_EQ(provider_calls, 0u);
        a->ready = true;
        EXPECT_EQ(first.poll(&error), MoEOverlayResidencyWaveProgress::Ready);
        EXPECT_FALSE(arrival->completePayload(result));
        EXPECT_EQ(first.completedMeasurement(), a->measurement);
        EXPECT_EQ(last.completedMeasurement(), std::nullopt);
        b->ready = true;
        EXPECT_EQ(last.poll(&error), MoEOverlayResidencyWaveProgress::Ready);
        ASSERT_TRUE(arrival->completePayload(result)) << error;
        EXPECT_EQ(result.gate(), engine);
        EXPECT_EQ(last.completedMeasurement(), b->measurement);
        EXPECT_EQ(provider_calls, 1u);
        EXPECT_EQ(last.poll(&error), MoEOverlayResidencyWaveProgress::Ready);
        EXPECT_EQ(provider_calls, 1u);
    }
}

TEST(PreparedProjectionIntersectionContract,
     FailedOrAbortedFinalPiecePoisonsArrivalAndEachOriginalOperationDrains)
{
    for (const bool fail_final : {false, true})
    {
        auto arrival = readySiblingArrival();
        auto engine = payload(100).gate();
        unsigned provider_calls = 0;
        auto completion = std::make_shared<MoEOverlayPreparedProjectionCompletion>(
            splitProjectionCoverage(), arrival, ExpertTierWeightProjection::Gate,
            [engine, &provider_calls]
            {
                ++provider_calls;
                return engine;
            });
        auto a = std::make_shared<IntersectionOperationState>();
        auto b = std::make_shared<IntersectionOperationState>();
        a->ready = true;
        MoEOverlayPreparedProjectionOperation first(
            std::make_unique<IntersectionPhysicalOperation>(a), completion, 0);
        MoEOverlayPreparedProjectionOperation last(
            std::make_unique<IntersectionPhysicalOperation>(b), completion, 1);
        std::string error;
        EXPECT_EQ(first.poll(&error), MoEOverlayResidencyWaveProgress::Ready);
        if (fail_final)
        {
            b->failed = true;
            EXPECT_EQ(last.poll(&error), MoEOverlayResidencyWaveProgress::Failed);
            EXPECT_EQ(error, "original physical completion failed");
        }
        last.abort();
        first.abort();
        EXPECT_TRUE(a->aborted);
        EXPECT_TRUE(b->aborted);
        EXPECT_EQ(last.pollAbort(&error), MoEOverlayResidencyWaveProgress::Pending);
        a->drained = true;
        EXPECT_EQ(first.pollAbort(&error), MoEOverlayResidencyWaveProgress::Ready);
        EXPECT_EQ(last.pollAbort(&error), MoEOverlayResidencyWaveProgress::Pending);
        b->drained = true;
        EXPECT_EQ(last.pollAbort(&error), MoEOverlayResidencyWaveProgress::Ready);
        MoEOverlayPreparedExpertPayload result;
        EXPECT_FALSE(arrival->completePayload(result, &error));
        EXPECT_EQ(provider_calls, 0u);
        EXPECT_EQ(first.completedMeasurement(), std::nullopt);
    }
}
