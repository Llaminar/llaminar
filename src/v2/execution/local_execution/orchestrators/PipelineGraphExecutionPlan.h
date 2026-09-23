/**
 * @file PipelineGraphExecutionPlan.h
 * @brief Typed lifecycle for a participant-local pipeline graph transaction.
 *
 * A LocalPP pipeline is one ordered production transaction composed from
 * participant-local compute graphs and explicit activation-transfer
 * boundaries. CPU stages own declarative host graphs; GPU stages own retained
 * native executable families. This type freezes that order once, identifies
 * the only topology in which coordinator segmentation is legal, and prevents
 * inference from claiming a replay before every native segment is resident.
 */

#pragma once

#include "../../../backends/DeviceId.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    /** @brief Execution authority owned by one ordered pipeline segment. */
    enum class PipelineGraphSegmentExecution : std::uint8_t
    {
        HostDeclarative = 0, ///< CPU graph executed by its host participant.
        NativeDeviceExecutable, ///< GPU graph backed by a retained native executable.
    };

    /** @brief Setup state of the immutable pipeline execution plan. */
    enum class PipelineGraphExecutionLifecycle : std::uint8_t
    {
        Declared = 0, ///< Stage order is frozen but native families are not certified.
        Materialized, ///< Every declared native segment is resident and replay-ready.
    };

    /**
     * @brief One execution domain in pipeline order, including every TP member.
     *
     * Membership preserves the compiled collective order: index zero is the
     * domain leader, not the lowest physical ordinal. A domain contains one
     * backend; heterogeneous movement is an edge between domains. This is a
     * frozen projection of the rank plan, not another topology discoverer.
     */
    struct PipelineGraphExecutionSegment
    {
        std::size_t stage_index = 0; ///< Contiguous LocalPP stage coordinate.
        std::vector<DeviceId> participants; ///< Ordered execution identities; CPU NUMA ownership remains in the rank plan/context.
        PipelineGraphSegmentExecution execution =
            PipelineGraphSegmentExecution::HostDeclarative;

        /** @return Domain leader, or invalid for an incomplete declaration. */
        [[nodiscard]] DeviceId primaryDevice() const noexcept
        {
            return participants.empty() ? DeviceId::invalid() : participants.front();
        }

        /** @return Whether members are concrete and same-backend, with no duplicate GPU. */
        [[nodiscard]] bool valid() const noexcept
        {
            const auto primary = primaryDevice();
            if (!((execution ==
                         PipelineGraphSegmentExecution::HostDeclarative &&
                     primary.is_cpu()) ||
                    (execution == PipelineGraphSegmentExecution::
                                      NativeDeviceExecutable &&
                     primary.is_gpu())))
                return false;
            for (std::size_t i = 0; i < participants.size(); ++i)
            {
                // DeviceId deliberately collapses CPU NUMA endpoints to the
                // host execution kind. Their physical membership remains in
                // the existing TP context/rank plan, not a new NUMA ledger here.
                if (!participants[i].is_valid() || participants[i].type != primary.type ||
                    (primary.is_gpu() &&
                     std::find(participants.begin(), participants.begin() + i, participants[i]) !=
                        participants.begin() + i))
                    return false;
            }
            return true;
        }
    };

    /**
     * @brief Immutable pipeline order plus its one-way materialization edge.
     *
     * The plan deliberately contains no runner pointers. RankOrchestrator owns
     * both the runners and this value, rechecks their identities at setup, and
     * then uses the stage coordinates here to drive every transaction. A
     * heterogeneous boundary exists only when adjacent stages change device
     * backend; homogeneous LocalPP therefore cannot accidentally advertise the
     * segmentation exception.
     */
    class PipelineGraphExecutionPlan
    {
    public:
        /**
         * @brief Freeze and validate one non-empty contiguous segment order.
         * @param segments Participant-local segments in execution order.
         * @throws std::invalid_argument when an owner or coordinate is invalid.
         */
        explicit PipelineGraphExecutionPlan(
            std::vector<PipelineGraphExecutionSegment> segments)
            : segments_(std::move(segments))
        {
            std::string error;
            if (!validate(&error))
            {
                throw std::invalid_argument(
                    error.empty()
                        ? "Pipeline graph execution plan is invalid"
                        : error);
            }
        }

        /** @return Ordered immutable participant-local segments. */
        [[nodiscard]] const std::vector<PipelineGraphExecutionSegment> &
        segments() const noexcept
        {
            return segments_;
        }

        /** @return Number of ordered domains, not the number of native device graphs. */
        [[nodiscard]] std::size_t segmentCount() const noexcept
        {
            return segments_.size();
        }

        /** @return Number of participant-local graphs, retaining nested TP width. */
        [[nodiscard]] std::size_t participantCount() const noexcept
        {
            std::size_t count = 0;
            for (const auto &segment : segments_)
                count += segment.participants.size();
            return count;
        }

        /** @return Number of retained GPU executable segments. */
        [[nodiscard]] std::size_t nativeSegmentCount() const noexcept
        {
            std::size_t count = 0;
            for (const auto &segment : segments_)
            {
                count += segment.execution ==
                                 PipelineGraphSegmentExecution::
                                     NativeDeviceExecutable
                             ? 1u
                             : 0u;
            }
            return count;
        }

        /** @return Number of declarative CPU graph segments. */
        [[nodiscard]] std::size_t hostSegmentCount() const noexcept
        {
            return segmentCount() - nativeSegmentCount();
        }

        /**
         * @return True only when an adjacent transfer crosses backend kinds.
         *
         * Different ordinals of the same GPU backend remain homogeneous. Their
         * participant-local graph protocol may not claim the heterogeneous
         * segmentation exception merely because a pipeline transfer exists.
         */
        [[nodiscard]] bool hasHeterogeneousBoundary() const noexcept
        {
            for (std::size_t index = 1; index < segments_.size(); ++index)
            {
                if (segments_[index - 1u].primaryDevice().type !=
                    segments_[index].primaryDevice().type)
                {
                    return true;
                }
            }
            return false;
        }

        /** @return Current one-way setup state. */
        [[nodiscard]] PipelineGraphExecutionLifecycle lifecycle() const noexcept
        {
            return lifecycle_;
        }

        /** @return Whether every native executable named by this plan is ready. */
        [[nodiscard]] bool materialized() const noexcept
        {
            return lifecycle_ ==
                   PipelineGraphExecutionLifecycle::Materialized;
        }

        /**
         * @brief Commit the setup edge after exact native-stage materialization.
         * @param materialized_native_segments Native children that returned ready.
         * @param error Optional precise mismatch diagnostic.
         * @return True for the exact transition or its identical idempotent replay.
         */
        bool markMaterialized(
            std::size_t materialized_native_segments,
            std::string *error = nullptr) noexcept
        {
            const std::size_t expected = nativeSegmentCount();
            if (materialized_native_segments != expected)
            {
                if (error)
                {
                    *error =
                        "Pipeline graph materialization covered " +
                        std::to_string(materialized_native_segments) +
                        " native segments but the frozen plan requires " +
                        std::to_string(expected);
                }
                return false;
            }
            lifecycle_ = PipelineGraphExecutionLifecycle::Materialized;
            if (error)
                error->clear();
            return true;
        }

        /**
         * @brief Validate one completed coordinator transaction against the plan.
         * @param completed_segments Participant stages successfully executed.
         * @param error Optional precise lifecycle or cardinality diagnostic.
         * @return True only after setup and an exact complete stage traversal.
         */
        [[nodiscard]] bool certifiesReplay(
            std::size_t completed_segments,
            std::string *error = nullptr) const noexcept
        {
            if (!materialized())
            {
                if (error)
                    *error = "Pipeline graph replay preceded serving-family materialization";
                return false;
            }
            if (completed_segments != segmentCount())
            {
                if (error)
                {
                    *error =
                        "Pipeline graph transaction completed " +
                        std::to_string(completed_segments) +
                        " segments but the frozen plan requires " +
                        std::to_string(segmentCount());
                }
                return false;
            }
            if (error)
                error->clear();
            return true;
        }

    private:
        /** @brief Validate owner kinds and exact contiguous stage coordinates. */
        [[nodiscard]] bool validate(std::string *error) const noexcept
        {
            if (segments_.empty())
            {
                if (error)
                    *error = "Pipeline graph execution plan has no segments";
                return false;
            }
            for (std::size_t index = 0; index < segments_.size(); ++index)
            {
                const auto &segment = segments_[index];
                if (segment.stage_index != index)
                {
                    if (error)
                    {
                        *error =
                            "Pipeline graph execution plan has a non-contiguous stage coordinate at index " +
                            std::to_string(index);
                    }
                    return false;
                }
                if (!segment.valid())
                {
                    if (error)
                    {
                        *error =
                            "Pipeline graph execution plan has an invalid owner at stage " +
                            std::to_string(index);
                    }
                    return false;
                }
            }
            if (error)
                error->clear();
            return true;
        }

        /** Frozen participant-local stages in exact transaction order. */
        std::vector<PipelineGraphExecutionSegment> segments_;

        /** One-way setup edge owned by this plan. */
        PipelineGraphExecutionLifecycle lifecycle_ =
            PipelineGraphExecutionLifecycle::Declared;
    };

} // namespace llaminar2
