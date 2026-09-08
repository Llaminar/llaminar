/**
 * @file Test__MoEGroupedVerifierHistogramBoundarySet.cpp
 * @brief Model-free integration regressions for MTP MoE history boundaries.
 *
 * These tests lock down the production graph-discovery state machine that sits
 * between grouped-verifier lowering and captured accepted-state publication.
 * They deliberately use opaque stream identities and no device runtime: CUDA
 * and ROCm integration suites separately prove the exact stream's kernels and
 * event edges, while this suite proves topology and lifecycle totality before a
 * real-weight parity campaign pays model-loading cost.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEOverlayEconomyProfileComposer.h"
#include "execution/moe/MoEGroupedVerifierHistogramBoundarySet.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /**
         * @brief Minimal typed stage used to exercise boundary discovery.
         *
         * Publication methods are never invoked by the discovery state machine;
         * their counters still make an accidental hot-path call visible.
         */
        class RecordingBoundaryStage final
            : public IMoEGroupedVerifierHistogramPublisher
        {
        public:
            MoEGroupedVerifierHistogramRole role =
                MoEGroupedVerifierHistogramRole::NotOwner;
            int layer = 0;
            void *stream = nullptr;
            int prepare_calls = 0;
            int enqueue_calls = 0;

            /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
            [[nodiscard]] MoEGroupedVerifierHistogramRole
            groupedVerifierHistogramRole() const noexcept override
            {
                return role;
            }

            /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
            [[nodiscard]] int
            groupedVerifierHistogramLayerIndex() const noexcept override
            {
                return layer;
            }

            /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
            [[nodiscard]] std::string_view
            groupedVerifierHistogramPublisherName() const noexcept override
            {
                return "recording_boundary";
            }

            /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
            [[nodiscard]] void *
            groupedVerifierHistogramPublicationStream() const override
            {
                return stream;
            }

            /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
            bool prepareGroupedVerifierHistogramProducer(void *) override
            {
                ++prepare_calls;
                return true;
            }

            /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
            bool transitionGroupedVerifierHistogramProducerCapture(
                void *,
                RuntimeHistogramProducerCaptureTransition) override
            {
                return true;
            }

            /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
            bool enqueueCommittedGroupedVerifierHistograms(
                const int32_t *,
                const int32_t *,
                int,
                int,
                void *) override
            {
                ++enqueue_calls;
                return true;
            }
        };
    }

    /**
     * @brief Static and Dynamic lowering select distinct explicit roles.
     */
    TEST(MoEGroupedVerifierHistogramBoundarySet,
         RoleSelectionIsTotalAcrossOwnershipAndMaintenance)
    {
        EXPECT_EQ(
            selectMoEGroupedVerifierHistogramRole(false, false),
            MoEGroupedVerifierHistogramRole::NotOwner);
        EXPECT_EQ(
            selectMoEGroupedVerifierHistogramRole(false, true),
            MoEGroupedVerifierHistogramRole::NotOwner);
        EXPECT_EQ(
            selectMoEGroupedVerifierHistogramRole(true, false),
            MoEGroupedVerifierHistogramRole::StaticNoPublication);
        EXPECT_EQ(
            selectMoEGroupedVerifierHistogramRole(true, true),
            MoEGroupedVerifierHistogramRole::DeferredAcceptedRows);
    }

    /**
     * @brief Static resolves as a deliberate publication-free transaction.
     */
    TEST(MoEGroupedVerifierHistogramBoundarySet,
         StaticBoundaryResolvesWithoutPublisherOrStream)
    {
        RecordingBoundaryStage router;
        RecordingBoundaryStage expert;
        expert.role = MoEGroupedVerifierHistogramRole::StaticNoPublication;

        MoEGroupedVerifierHistogramBoundarySet boundaries;
        std::string error;
        ASSERT_TRUE(boundaries.registerRoutedLayer(0, "layer0_router", &error))
            << error;
        ASSERT_TRUE(boundaries.registerStage(&router, "layer0_router", &error))
            << error;
        ASSERT_TRUE(boundaries.registerStage(&expert, "layer0_expert", &error))
            << error;

        MoEGroupedVerifierHistogramBoundaryResolution resolution;
        ASSERT_TRUE(boundaries.resolve(resolution, &error)) << error;
        EXPECT_TRUE(resolution.publishers.empty());
        EXPECT_EQ(resolution.publication_stream, nullptr);
        EXPECT_EQ(
            boundaries.state(),
            MoEGroupedVerifierHistogramBoundarySet::State::Resolved);
        EXPECT_EQ(expert.prepare_calls, 0);
        EXPECT_EQ(expert.enqueue_calls, 0);

        EXPECT_FALSE(
            boundaries.registerRoutedLayer(1, "late_router", &error));
        EXPECT_NE(error.find("after verifier boundary discovery"),
                  std::string::npos);
    }

    /**
     * @brief Dynamic resolves one ordered publisher per layer on one stream.
     */
    TEST(MoEGroupedVerifierHistogramBoundarySet,
         DeferredBoundariesResolveInLayerOrderOnSharedStream)
    {
        void *const stream = reinterpret_cast<void *>(uintptr_t{0xCAFE00});
        RecordingBoundaryStage layer_one;
        layer_one.role =
            MoEGroupedVerifierHistogramRole::DeferredAcceptedRows;
        layer_one.layer = 1;
        layer_one.stream = stream;
        RecordingBoundaryStage layer_zero = layer_one;
        layer_zero.layer = 0;

        MoEGroupedVerifierHistogramBoundarySet boundaries;
        std::string error;
        ASSERT_TRUE(boundaries.registerRoutedLayer(1, "layer1_router", &error));
        ASSERT_TRUE(boundaries.registerRoutedLayer(0, "layer0_router", &error));
        ASSERT_TRUE(boundaries.registerStage(&layer_one, "layer1_expert", &error));
        ASSERT_TRUE(boundaries.registerStage(&layer_zero, "layer0_expert", &error));

        MoEGroupedVerifierHistogramBoundaryResolution resolution;
        ASSERT_TRUE(boundaries.resolve(resolution, &error)) << error;
        ASSERT_EQ(resolution.publishers.size(), 2u);
        EXPECT_EQ(resolution.publishers[0], &layer_zero);
        EXPECT_EQ(resolution.publishers[1], &layer_one);
        EXPECT_EQ(resolution.publication_stream, stream);
    }

    /**
     * @brief Missing, duplicate, or stream-incoherent ownership fails closed.
     */
    TEST(MoEGroupedVerifierHistogramBoundarySet,
         InvalidLifecycleTransitionsBecomeTerminalFaults)
    {
        void *const stream_a = reinterpret_cast<void *>(uintptr_t{0xA000});
        void *const stream_b = reinterpret_cast<void *>(uintptr_t{0xB000});

        RecordingBoundaryStage missing_stream;
        missing_stream.role =
            MoEGroupedVerifierHistogramRole::DeferredAcceptedRows;
        MoEGroupedVerifierHistogramBoundarySet missing_stream_set;
        std::string error;
        ASSERT_TRUE(missing_stream_set.registerRoutedLayer(
            0, "layer0_router", &error));
        EXPECT_FALSE(missing_stream_set.registerStage(
            &missing_stream, "layer0_expert", &error));
        EXPECT_EQ(
            missing_stream_set.state(),
            MoEGroupedVerifierHistogramBoundarySet::State::Faulted);

        RecordingBoundaryStage first;
        first.role = MoEGroupedVerifierHistogramRole::DeferredAcceptedRows;
        first.stream = stream_a;
        RecordingBoundaryStage second = first;
        second.layer = 1;
        second.stream = stream_b;
        MoEGroupedVerifierHistogramBoundarySet foreign_stream_set;
        ASSERT_TRUE(foreign_stream_set.registerRoutedLayer(
            0, "layer0_router", &error));
        ASSERT_TRUE(foreign_stream_set.registerRoutedLayer(
            1, "layer1_router", &error));
        ASSERT_TRUE(foreign_stream_set.registerStage(
            &first, "layer0_expert", &error));
        EXPECT_FALSE(foreign_stream_set.registerStage(
            &second, "layer1_expert", &error));

        RecordingBoundaryStage static_with_stream;
        static_with_stream.role =
            MoEGroupedVerifierHistogramRole::StaticNoPublication;
        static_with_stream.stream = stream_a;
        MoEGroupedVerifierHistogramBoundarySet static_stream_set;
        ASSERT_TRUE(static_stream_set.registerRoutedLayer(
            0, "layer0_router", &error));
        EXPECT_FALSE(static_stream_set.registerStage(
            &static_with_stream, "layer0_expert", &error));

        MoEGroupedVerifierHistogramBoundarySet missing_boundary_set;
        ASSERT_TRUE(missing_boundary_set.registerRoutedLayer(
            0, "layer0_router", &error));
        MoEGroupedVerifierHistogramBoundaryResolution resolution;
        EXPECT_FALSE(missing_boundary_set.resolve(resolution, &error));
        EXPECT_EQ(
            missing_boundary_set.state(),
            MoEGroupedVerifierHistogramBoundarySet::State::Faulted);
    }

    /**
     * @brief Fixed-depth MTP retains main-model serial catch-up service.
     *
     * Production can execute a one-row main-model decode transaction before a
     * grouped verifier transaction, notably for short tails and terminal
     * catch-up. Feed that observation through the economy normalizer: it must
     * remain legal graph evidence without becoming a recurring service cost or
     * blocking fixed-depth MTP certification. Predictor-only retained layers
     * remain grouped-verifier-only and are checked independently below.
     */
    TEST(MoEGroupedVerifierHistogramBoundarySet,
         RetainedMTPTopologyAcceptsMainDecodeCatchupEvidence)
    {
        const auto topology =
            ExpertHistogramProductionTopology::forRetainedExecution(
                /*retained_layer_count=*/2,
                /*main_inference_layer_count=*/1,
                ExpertHistogramServingRegime::PositiveDepthMTP);

        std::vector<MoEOverlayParticipantLayerServiceTotals> totals{
            {
                .participant_id = 4,
                .layer = 0,
                .total_nanoseconds = {100, 200, 300},
                .activation_count = {1, 2, 3},
                .sample_count = {1, 1, 1},
            },
            {
                .participant_id = 4,
                .layer = 1,
                .total_nanoseconds = {0, 0, 400},
                .activation_count = {0, 0, 4},
                .sample_count = {0, 0, 1},
            },
        };

        const auto normalized =
            MoEOverlayEconomyProfileComposer::normalizeServiceTotals(
                totals, topology);
        ASSERT_EQ(normalized.size(), 2u);
        EXPECT_EQ(
            normalized[0].nanoseconds_per_activation,
            (std::array<std::uint64_t, 3>{0, 100, 100}));
        EXPECT_EQ(
            normalized[1].nanoseconds_per_activation,
            (std::array<std::uint64_t, 3>{0, 0, 100}));

        // An auxiliary predictor layer still cannot masquerade as serial
        // decode; only the main-model catch-up coordinate is reachable.
        totals[1].total_nanoseconds[0] = 1;
        totals[1].activation_count[0] = 1;
        totals[1].sample_count[0] = 1;
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::normalizeServiceTotals(
                totals, topology),
            std::invalid_argument);
    }
}
