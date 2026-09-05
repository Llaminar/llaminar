/**
 * @file MoEOverlayRetainedActivationTransaction.cpp
 * @brief Native graph assembly for retained node-local activation epochs.
 */

#include "MoEOverlayRetainedActivationTransaction.h"

#include "backends/IGPUGraphCapture.h"
#include "execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Validate topology, manifest, mapped aliases, and signal banks. */
        void validateCommon(
            const IGPUGraphCapture &destination,
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::span<const std::int32_t> manifest_layers,
            std::size_t stage_count)
        {
            if (!lane.valid() || !lane.device.is_gpu() ||
                !destination.executionStream() || destination.hasExecutable() ||
                destination.nodeCount() != 0u || manifest_layers.empty() ||
                manifest_layers.size() != stage_count || !lane.control_host ||
                lane.control_host->channel.stage_count != stage_count ||
                lane.control_host->channel.target_participant_id !=
                    lane.target_participant_id ||
                lane.control_host->channel.stage_manifest_digest !=
                    MoEOverlayActivationEpochProtocol::stageManifestDigest(
                        manifest_layers))
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay activation transaction has an incomplete lane, destination, or stage manifest");
            }

            for (std::uint32_t bank = 0u;
                 bank < kMoEOverlayActivationBufferCount;
                 ++bank)
            {
                if (!lane.mapped_region->contains(
                        lane.dispatch_signal_offsets[bank],
                        sizeof(std::uint64_t)) ||
                    !lane.mapped_region->contains(
                        lane.return_signal_offsets[bank],
                        sizeof(std::uint64_t)))
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay activation transaction signal lies outside its mapped region");
                }
            }
        }

        /** @return Capture-ready fragment identity for one exact endpoint graph. */
        bool validFragment(const IGPUGraphCapture *fragment) noexcept
        {
            return fragment && fragment->executionStream() &&
                   fragment->nodeCount() != 0u;
        }

        /** @return Stable diagnostic name stored for the duration of lowering. */
        const char *appendName(
            std::deque<std::string> &names,
            const char *operation,
            std::uint32_t stage_ordinal,
            std::int32_t model_layer_index)
        {
            names.push_back(
                std::string(operation) + "_stage" +
                std::to_string(stage_ordinal) + "_layer" +
                std::to_string(model_layer_index));
            return names.back().c_str();
        }

        /** @return Stable identity for one planner-selected mapped lane. */
        const void *laneIdentity(
            const MoEOverlayMappedActivationDeviceLane &lane) noexcept
        {
            return lane.control_host;
        }

        /** @brief Validate one lane against the complete retained family. */
        void validateFanoutLane(
            const IGPUGraphCapture &destination,
            const MoEOverlayMappedActivationDeviceLane &lane,
            std::span<const std::int32_t> manifest_layers,
            DeviceId continuation_device)
        {
            validateCommon(
                destination, lane, manifest_layers, manifest_layers.size());
            if (lane.device != continuation_device)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay fanout lane is not mapped on the continuation graph device");
            }
        }

        /** @return true when every ordered child fragment is non-empty and bound. */
        bool validFragments(
            std::span<const IGPUGraphCapture *const> fragments) noexcept
        {
            return std::all_of(
                fragments.begin(),
                fragments.end(),
                [](const IGPUGraphCapture *fragment)
                {
                    return validFragment(fragment);
                });
        }

        /** @brief Append one named sequence of ordinary captured child graphs. */
        void appendFragments(
            std::vector<MappedTimelineTransactionStep> &steps,
            std::deque<std::string> &names,
            const char *operation,
            std::uint32_t stage_ordinal,
            std::int32_t model_layer_index,
            std::span<const IGPUGraphCapture *const> fragments)
        {
            for (std::size_t fragment_index = 0u;
                 fragment_index < fragments.size();
                 ++fragment_index)
            {
                names.push_back(
                    std::string(operation) + "_stage" +
                    std::to_string(stage_ordinal) + "_layer" +
                    std::to_string(model_layer_index) + "_fragment" +
                    std::to_string(fragment_index));
                steps.emplace_back(MappedTimelineCapturedFragment{
                    .name = names.back().c_str(),
                    .capture = fragments[fragment_index],
                });
            }
        }

        /** @brief Typed packet frontier encoded by one declarative graph stage. */
        enum class PacketMarkerKind : std::uint8_t
        {
            ContinuationDispatch,
            ContinuationReturn,
            FollowerDispatch,
            FollowerReturn,
        };

        /** @brief Immutable packet identity recovered from a captured child. */
        struct PacketMarker
        {
            PacketMarkerKind kind = PacketMarkerKind::ContinuationDispatch;
            std::uint32_t stage_ordinal = 0u;
            std::int32_t model_layer_index = -1;
            MoEOverlayMappedActivationDeviceLane lane;
            std::size_t stage_index = 0u;
        };

        /** @brief Capture unit plus every typed packet frontier it contains. */
        struct ClassifiedCaptureUnit
        {
            const IGPUGraphCapture *capture = nullptr;
            std::size_t stage_count = 0u;
            std::vector<PacketMarker> markers;
        };

        /** @brief Recover packet markers without interpreting backend or topology. */
        std::vector<ClassifiedCaptureUnit> classifyCaptureUnits(
            const ComputeGraph &graph,
            std::span<const MoEOverlayRetainedCaptureUnit> units)
        {
            if (units.empty())
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay graph lowering requires at least one captured unit");
            }

            std::vector<ClassifiedCaptureUnit> classified;
            classified.reserve(units.size());
            for (const auto &unit : units)
            {
                if (!validFragment(unit.capture) || unit.stage_names.empty())
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay graph lowering received an empty or invalid capture unit");
                }

                ClassifiedCaptureUnit result{
                    .capture = unit.capture,
                    .stage_count = unit.stage_names.size(),
                };
                for (std::size_t stage_index = 0u;
                     stage_index < unit.stage_names.size();
                     ++stage_index)
                {
                    const std::string &stage_name = unit.stage_names[stage_index];
                    const ComputeNode *const node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        throw std::invalid_argument(
                            "Retained ExpertOverlay capture unit names an unresolved graph stage: " +
                            stage_name);
                    }

                    const auto append = [&](PacketMarkerKind kind,
                                            const auto &params)
                    {
                        result.markers.push_back(PacketMarker{
                            .kind = kind,
                            .stage_ordinal = params.stage_ordinal,
                            .model_layer_index = params.model_layer_index,
                            .lane = params.lane,
                            .stage_index = stage_index,
                        });
                    };
                    if (const auto *stage = dynamic_cast<const
                            MoEOverlayActivationDispatchPackStage *>(
                            node->stage.get()))
                    {
                        append(
                            PacketMarkerKind::ContinuationDispatch,
                            stage->getParams());
                    }
                    else if (const auto *stage = dynamic_cast<const
                                 MoEOverlayActivationReturnConsumeStage *>(
                                 node->stage.get()))
                    {
                        append(
                            PacketMarkerKind::ContinuationReturn,
                            stage->getParams());
                    }
                    else if (const auto *stage = dynamic_cast<const
                                 MoEOverlayActivationDispatchConsumeStage *>(
                                 node->stage.get()))
                    {
                        append(
                            PacketMarkerKind::FollowerDispatch,
                            stage->getParams());
                    }
                    else if (const auto *stage = dynamic_cast<const
                                 MoEOverlayActivationReturnPackStage *>(
                                 node->stage.get()))
                    {
                        append(
                            PacketMarkerKind::FollowerReturn,
                            stage->getParams());
                    }
                }
                classified.push_back(std::move(result));
            }
            return classified;
        }

        /** @return Complete manifest width authenticated by one packet lane. */
        std::size_t laneStageCount(
            const MoEOverlayMappedActivationDeviceLane &lane)
        {
            if (!lane.valid() || !lane.control_host ||
                lane.control_host->channel.stage_count == 0u)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay packet stage has an incomplete mapped lane");
            }
            return lane.control_host->channel.stage_count;
        }

        /** @brief Reject a packet marker that does not name the expected stage. */
        void requireMarkerIdentity(
            const PacketMarker &marker,
            PacketMarkerKind kind,
            std::size_t ordinal,
            std::int32_t layer)
        {
            if (marker.kind != kind ||
                marker.stage_ordinal != ordinal ||
                marker.model_layer_index != layer)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay packet frontier changed operation, ordinal, or model layer");
            }
        }
    } // namespace

    void MoEOverlayRetainedActivationTransaction::buildContinuation(
        IGPUGraphCapture &destination,
        const MoEOverlayMappedActivationDeviceLane &lane,
        std::span<const std::int32_t> manifest_layers,
        std::span<const MoEOverlayContinuationActivationStage> stages)
    {
        validateCommon(destination, lane, manifest_layers, stages.size());

        std::deque<std::string> names;
        std::vector<MappedTimelineTransactionStep> steps;
        steps.reserve(1u + stages.size() * 4u);
        steps.emplace_back(MappedTimelineWait64{
            .name = appendName(names, "scheduler_admission_wait", 0u, -1),
            .region = lane.mapped_region.get(),
            .signal_offset = lane.admission_signal_offset,
            .value = kMoEOverlayActivationAdmissionTimeline,
        });

        for (std::size_t index = 0; index < stages.size(); ++index)
        {
            const auto ordinal = static_cast<std::uint32_t>(index);
            const auto &stage = stages[index];
            if (stage.model_layer_index != manifest_layers[index] ||
                !validFragment(stage.dispatch_fragment) ||
                !validFragment(stage.return_fragment))
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay continuation fragment disagrees with its ordered manifest");
            }
            const std::uint32_t bank =
                moeOverlayActivationBufferIndex(ordinal);
            const std::uint64_t timeline =
                moeOverlayActivationLeasedTimelineValue(
                    moeOverlayActivationBufferVisit(ordinal));
            if (timeline == 0u)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay continuation stage exceeds the timeline visit range");
            }

            steps.emplace_back(MappedTimelineCapturedFragment{
                .name = appendName(
                    names, "continuation_dispatch_pack", ordinal,
                    stage.model_layer_index),
                .capture = stage.dispatch_fragment,
            });
            steps.emplace_back(MappedTimelinePublish64{
                .name = appendName(
                    names, "dispatch_publication", ordinal,
                    stage.model_layer_index),
                .region = lane.mapped_region.get(),
                .signal_offset = lane.dispatch_signal_offsets[bank],
                .value = timeline,
            });
            steps.emplace_back(MappedTimelineWait64{
                .name = appendName(
                    names, "return_wait", ordinal,
                    stage.model_layer_index),
                .region = lane.mapped_region.get(),
                .signal_offset = lane.return_signal_offsets[bank],
                .value = timeline,
            });
            steps.emplace_back(MappedTimelineCapturedFragment{
                .name = appendName(
                    names, "continuation_return_consume", ordinal,
                    stage.model_layer_index),
                .capture = stage.return_fragment,
            });
        }

        TransferEngine::instance().buildMappedTimelineTransaction(
            destination, steps, lane.device);
    }

    void MoEOverlayRetainedActivationTransaction::buildContinuationFanout(
        IGPUGraphCapture &destination,
        std::span<const std::int32_t> manifest_layers,
        std::span<const MoEOverlayContinuationActivationFanoutStage> stages)
    {
        if (!destination.executionStream() || destination.hasExecutable() ||
            destination.nodeCount() != 0u || manifest_layers.empty() ||
            stages.size() != manifest_layers.size() || stages.empty() ||
            stages.front().lanes.empty())
        {
            throw std::invalid_argument(
                "Retained ExpertOverlay fanout transaction has an incomplete destination, manifest, or lane set");
        }

        /*
         * The first stage freezes the topology-owned canonical lane order.  A
         * later layer may carry zero live rows for a lane, but it may not remove
         * that lane: the follower's retained graph still performs the matching
         * stage-ordinal wait and zero-work publication.
         */
        const DeviceId continuation_device = stages.front().lanes.front().lane.device;
        if (!continuation_device.is_gpu())
        {
            throw std::invalid_argument(
                "Retained ExpertOverlay fanout transaction requires one planner-selected GPU continuation device");
        }
        std::vector<const void *> canonical_lane_identities;
        canonical_lane_identities.reserve(stages.front().lanes.size());
        std::unordered_set<const void *> unique_lanes;
        for (const auto &lane_stage : stages.front().lanes)
        {
            validateFanoutLane(
                destination, lane_stage.lane, manifest_layers,
                continuation_device);
            const void *const identity = laneIdentity(lane_stage.lane);
            if (!identity || !unique_lanes.insert(identity).second)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay fanout transaction repeats one mapped lane");
            }
            canonical_lane_identities.push_back(identity);
        }

        std::deque<std::string> names;
        std::vector<MappedTimelineTransactionStep> steps;
        const std::size_t lane_count = canonical_lane_identities.size();
        steps.reserve(
            lane_count + stages.size() * (lane_count * 4u + 3u));

        for (std::size_t lane_index = 0; lane_index < lane_count; ++lane_index)
        {
            const auto &lane = stages.front().lanes[lane_index].lane;
            steps.emplace_back(MappedTimelineWait64{
                .name = appendName(
                    names, "scheduler_admission_wait_lane",
                    static_cast<std::uint32_t>(lane_index),
                    lane.target_participant_id),
                .region = lane.mapped_region.get(),
                .signal_offset = lane.admission_signal_offset,
                .value = kMoEOverlayActivationAdmissionTimeline,
            });
        }

        for (std::size_t index = 0; index < stages.size(); ++index)
        {
            const auto ordinal = static_cast<std::uint32_t>(index);
            const auto &stage = stages[index];
            if (stage.model_layer_index != manifest_layers[index] ||
                stage.lanes.size() != lane_count ||
                !validFragments(stage.prefix_fragments) ||
                !validFragments(stage.overlap_fragments) ||
                !validFragments(stage.suffix_fragments))
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay fanout stage disagrees with its manifest or dense fragments");
            }
            appendFragments(
                steps,
                names,
                "continuation_prefix",
                ordinal,
                stage.model_layer_index,
                stage.prefix_fragments);

            const std::uint32_t bank =
                moeOverlayActivationBufferIndex(ordinal);
            const std::uint64_t timeline =
                moeOverlayActivationLeasedTimelineValue(
                    moeOverlayActivationBufferVisit(ordinal));
            if (timeline == 0u)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay fanout stage exceeds the timeline visit range");
            }

            for (std::size_t lane_index = 0; lane_index < lane_count;
                 ++lane_index)
            {
                const auto &lane_stage = stage.lanes[lane_index];
                validateFanoutLane(
                    destination, lane_stage.lane, manifest_layers,
                    continuation_device);
                if (laneIdentity(lane_stage.lane) !=
                        canonical_lane_identities[lane_index] ||
                    !validFragment(lane_stage.dispatch_fragment) ||
                    !validFragment(lane_stage.return_fragment))
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay fanout lane order or fragment identity changed between model stages");
                }
                steps.emplace_back(MappedTimelineCapturedFragment{
                    .name = appendName(
                        names, "continuation_dispatch_pack_lane",
                        ordinal,
                        lane_stage.lane.target_participant_id),
                    .capture = lane_stage.dispatch_fragment,
                });
                steps.emplace_back(MappedTimelinePublish64{
                    .name = appendName(
                        names, "dispatch_publication_lane", ordinal,
                        lane_stage.lane.target_participant_id),
                    .region = lane_stage.lane.mapped_region.get(),
                    .signal_offset =
                        lane_stage.lane.dispatch_signal_offsets[bank],
                    .value = timeline,
                });
            }

            appendFragments(
                steps,
                names,
                "continuation_local_overlap",
                ordinal,
                stage.model_layer_index,
                stage.overlap_fragments);

            for (std::size_t lane_index = 0; lane_index < lane_count;
                 ++lane_index)
            {
                const auto &lane_stage = stage.lanes[lane_index];
                steps.emplace_back(MappedTimelineWait64{
                    .name = appendName(
                        names, "return_wait_lane", ordinal,
                        lane_stage.lane.target_participant_id),
                    .region = lane_stage.lane.mapped_region.get(),
                    .signal_offset =
                        lane_stage.lane.return_signal_offsets[bank],
                    .value = timeline,
                });
                steps.emplace_back(MappedTimelineCapturedFragment{
                    .name = appendName(
                        names, "continuation_return_consume_lane", ordinal,
                        lane_stage.lane.target_participant_id),
                    .capture = lane_stage.return_fragment,
                });
            }

            appendFragments(
                steps,
                names,
                "continuation_suffix",
                ordinal,
                stage.model_layer_index,
                stage.suffix_fragments);
        }

        TransferEngine::instance().buildMappedTimelineTransaction(
            destination, steps, continuation_device);
    }

    void MoEOverlayRetainedActivationTransaction::buildFollower(
        IGPUGraphCapture &destination,
        const MoEOverlayMappedActivationDeviceLane &lane,
        std::span<const std::int32_t> manifest_layers,
        std::span<const MoEOverlayFollowerActivationStage> stages)
    {
        validateCommon(destination, lane, manifest_layers, stages.size());

        std::deque<std::string> names;
        std::vector<MappedTimelineTransactionStep> steps;
        steps.reserve(1u + stages.size() * 3u);
        steps.emplace_back(MappedTimelineWait64{
            .name = appendName(names, "scheduler_admission_wait", 0u, -1),
            .region = lane.mapped_region.get(),
            .signal_offset = lane.admission_signal_offset,
            .value = kMoEOverlayActivationAdmissionTimeline,
        });

        for (std::size_t index = 0; index < stages.size(); ++index)
        {
            const auto ordinal = static_cast<std::uint32_t>(index);
            const auto &stage = stages[index];
            if (stage.model_layer_index != manifest_layers[index] ||
                stage.compute_fragments.empty() ||
                !validFragments(stage.prefix_fragments) ||
                !validFragments(stage.compute_fragments) ||
                !validFragments(stage.suffix_fragments))
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay follower fragment disagrees with its ordered manifest");
            }
            const std::uint32_t bank =
                moeOverlayActivationBufferIndex(ordinal);
            const std::uint64_t timeline =
                moeOverlayActivationLeasedTimelineValue(
                    moeOverlayActivationBufferVisit(ordinal));
            if (timeline == 0u)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay follower stage exceeds the timeline visit range");
            }

            appendFragments(
                steps,
                names,
                "follower_prefix",
                ordinal,
                stage.model_layer_index,
                stage.prefix_fragments);

            steps.emplace_back(MappedTimelineWait64{
                .name = appendName(
                    names, "dispatch_wait", ordinal,
                    stage.model_layer_index),
                .region = lane.mapped_region.get(),
                .signal_offset = lane.dispatch_signal_offsets[bank],
                .value = timeline,
            });
            appendFragments(
                steps,
                names,
                "follower_compute_and_return_pack",
                ordinal,
                stage.model_layer_index,
                stage.compute_fragments);
            steps.emplace_back(MappedTimelinePublish64{
                .name = appendName(
                    names, "return_publication", ordinal,
                    stage.model_layer_index),
                .region = lane.mapped_region.get(),
                .signal_offset = lane.return_signal_offsets[bank],
                .value = timeline,
            });
            appendFragments(
                steps,
                names,
                "follower_suffix",
                ordinal,
                stage.model_layer_index,
                stage.suffix_fragments);
        }

        TransferEngine::instance().buildMappedTimelineTransaction(
            destination, steps, lane.device);
    }

    void MoEOverlayRetainedActivationTransaction::
        buildContinuationFromCapturedUnits(
            IGPUGraphCapture &destination,
            const ComputeGraph &graph,
            std::span<const MoEOverlayRetainedCaptureUnit> units)
    {
        const auto classified = classifyCaptureUnits(graph, units);
        const auto first_packet = std::find_if(
            classified.begin(),
            classified.end(),
            [](const ClassifiedCaptureUnit &unit)
            {
                return !unit.markers.empty();
            });
        if (first_packet == classified.end() ||
            first_packet->markers.size() != 1u ||
            first_packet->markers.front().kind !=
                PacketMarkerKind::ContinuationDispatch)
        {
            throw std::invalid_argument(
                "Retained ExpertOverlay continuation graph has no leading dispatch-packet frontier");
        }

        const std::size_t stage_count =
            laneStageCount(first_packet->markers.front().lane);
        std::vector<std::int32_t> manifest_layers(stage_count, -1);
        std::vector<MoEOverlayContinuationActivationFanoutStage> stages(
            stage_count);
        std::vector<const void *> canonical_lanes;
        std::size_t cursor = 0u;

        for (std::size_t ordinal = 0u; ordinal < stage_count; ++ordinal)
        {
            auto &stage = stages[ordinal];

            // Dense work preceding the first dispatch remains device-owned.
            while (cursor < classified.size() &&
                   classified[cursor].markers.empty())
            {
                stage.prefix_fragments.push_back(
                    classified[cursor].capture);
                ++cursor;
            }
            if (cursor >= classified.size())
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay continuation graph ended before its dispatch frontiers");
            }

            std::vector<const void *> dispatch_lanes;
            while (cursor < classified.size())
            {
                const auto &unit = classified[cursor];
                if (unit.markers.size() != 1u ||
                    unit.markers.front().kind !=
                        PacketMarkerKind::ContinuationDispatch)
                {
                    break;
                }
                if (unit.markers.front().stage_index + 1u !=
                    unit.stage_count)
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay continuation dispatch packet must be the final stage in its native capture unit");
                }
                const PacketMarker &marker = unit.markers.front();
                if (marker.stage_ordinal != ordinal ||
                    marker.model_layer_index < 0)
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay continuation dispatch order diverged from its manifest");
                }
                if (manifest_layers[ordinal] < 0)
                {
                    manifest_layers[ordinal] = marker.model_layer_index;
                    stage.model_layer_index = marker.model_layer_index;
                }
                requireMarkerIdentity(
                    marker,
                    PacketMarkerKind::ContinuationDispatch,
                    ordinal,
                    manifest_layers[ordinal]);
                if (laneStageCount(marker.lane) != stage_count)
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay continuation lanes disagree on graph-family width");
                }

                const void *const identity = laneIdentity(marker.lane);
                if (!identity ||
                    std::find(
                        dispatch_lanes.begin(),
                        dispatch_lanes.end(),
                        identity) != dispatch_lanes.end())
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay continuation dispatch repeats one planner lane");
                }
                dispatch_lanes.push_back(identity);
                stage.lanes.push_back({
                    .lane = marker.lane,
                    .dispatch_fragment = unit.capture,
                });
                ++cursor;
            }
            if (dispatch_lanes.empty())
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay continuation stage has no dispatch lanes");
            }
            if (ordinal == 0u)
                canonical_lanes = dispatch_lanes;
            else if (dispatch_lanes != canonical_lanes)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay continuation lane set/order changed between model stages");
            }

            // Local/shared work is queued only after every lane is published.
            while (cursor < classified.size() &&
                   classified[cursor].markers.empty())
            {
                stage.overlap_fragments.push_back(
                    classified[cursor].capture);
                ++cursor;
            }

            std::vector<const void *> return_lanes;
            while (cursor < classified.size())
            {
                const auto &unit = classified[cursor];
                if (unit.markers.size() != 1u ||
                    unit.markers.front().kind !=
                        PacketMarkerKind::ContinuationReturn)
                {
                    break;
                }
                if (unit.markers.front().stage_index != 0u)
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay continuation return consume must be the first stage in its native capture unit");
                }
                const PacketMarker &marker = unit.markers.front();
                requireMarkerIdentity(
                    marker,
                    PacketMarkerKind::ContinuationReturn,
                    ordinal,
                    manifest_layers[ordinal]);
                const void *const identity = laneIdentity(marker.lane);
                return_lanes.push_back(identity);
                if (return_lanes.size() > stage.lanes.size() ||
                    identity != canonical_lanes[return_lanes.size() - 1u])
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay continuation return fold is not in canonical planner order");
                }
                stage.lanes[return_lanes.size() - 1u].return_fragment =
                    unit.capture;
                ++cursor;
            }
            if (return_lanes != canonical_lanes)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay continuation stage does not return every planner lane");
            }

            /*
             * Everything until the next packet frontier is ordinary dense work
             * after the canonical fold. Assigning it to this stage's suffix is
             * order-equivalent to calling it the next stage's prefix and avoids
             * inventing a model-specific layer-boundary marker.
             */
            while (cursor < classified.size() &&
                   classified[cursor].markers.empty())
            {
                stage.suffix_fragments.push_back(
                    classified[cursor].capture);
                ++cursor;
            }
        }

        if (cursor != classified.size())
        {
            throw std::invalid_argument(
                "Retained ExpertOverlay continuation graph contains extra or follower-only packet frontiers");
        }
        buildContinuationFanout(
            destination, manifest_layers, stages);
    }

    void MoEOverlayRetainedActivationTransaction::
        buildFollowerFromCapturedUnits(
            IGPUGraphCapture &destination,
            const ComputeGraph &graph,
            std::span<const MoEOverlayRetainedCaptureUnit> units)
    {
        const auto classified = classifyCaptureUnits(graph, units);
        const auto first_packet = std::find_if(
            classified.begin(),
            classified.end(),
            [](const ClassifiedCaptureUnit &unit)
            {
                return !unit.markers.empty();
            });
        if (first_packet == classified.end())
        {
            throw std::invalid_argument(
                "Retained ExpertOverlay follower graph has no packet frontier");
        }
        const auto first_dispatch = std::find_if(
            first_packet->markers.begin(),
            first_packet->markers.end(),
            [](const PacketMarker &marker)
            {
                return marker.kind == PacketMarkerKind::FollowerDispatch;
            });
        if (first_dispatch == first_packet->markers.end())
        {
            throw std::invalid_argument(
                "Retained ExpertOverlay follower graph has no leading dispatch-consume frontier");
        }

        const std::size_t stage_count = laneStageCount(first_dispatch->lane);
        std::vector<std::int32_t> manifest_layers(stage_count, -1);
        std::vector<MoEOverlayFollowerActivationStage> stages(stage_count);
        MoEOverlayMappedActivationDeviceLane canonical_lane;
        std::size_t cursor = 0u;

        for (std::size_t ordinal = 0u; ordinal < stage_count; ++ordinal)
        {
            auto &stage = stages[ordinal];
            while (cursor < classified.size() &&
                   classified[cursor].markers.empty())
            {
                stage.prefix_fragments.push_back(
                    classified[cursor].capture);
                ++cursor;
            }
            if (cursor >= classified.size())
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay follower graph ended before dispatch consume");
            }

            bool saw_dispatch = false;
            bool saw_return = false;
            MoEOverlayMappedActivationDeviceLane stage_lane;
            while (cursor < classified.size() && !saw_return)
            {
                const auto &unit = classified[cursor];
                if (unit.markers.empty())
                {
                    if (!saw_dispatch)
                    {
                        throw std::invalid_argument(
                            "Retained ExpertOverlay follower compute preceded its dispatch consume");
                    }
                }
                for (const PacketMarker &marker : unit.markers)
                {
                    if (marker.kind == PacketMarkerKind::FollowerDispatch)
                    {
                        if (saw_dispatch || marker.stage_index != 0u ||
                            marker.stage_ordinal != ordinal ||
                            marker.model_layer_index < 0)
                        {
                            throw std::invalid_argument(
                                "Retained ExpertOverlay follower dispatch consume is duplicated, fused after compute, or out of order");
                        }
                        saw_dispatch = true;
                        stage_lane = marker.lane;
                        manifest_layers[ordinal] =
                            marker.model_layer_index;
                        stage.model_layer_index =
                            marker.model_layer_index;
                    }
                    else if (marker.kind == PacketMarkerKind::FollowerReturn)
                    {
                        if (!saw_dispatch || saw_return ||
                            marker.stage_index + 1u != unit.stage_count)
                        {
                            throw std::invalid_argument(
                                "Retained ExpertOverlay follower return pack is duplicated, precedes dispatch, or is fused before later compute");
                        }
                        requireMarkerIdentity(
                            marker,
                            PacketMarkerKind::FollowerReturn,
                            ordinal,
                            manifest_layers[ordinal]);
                        if (laneIdentity(marker.lane) !=
                            laneIdentity(stage_lane))
                        {
                            throw std::invalid_argument(
                                "Retained ExpertOverlay follower dispatch and return use different planner lanes");
                        }
                        saw_return = true;
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Retained ExpertOverlay follower graph contains continuation-only packet stages");
                    }
                }
                if (!saw_dispatch)
                {
                    throw std::invalid_argument(
                        "Retained ExpertOverlay follower compute unit has no dispatch consume");
                }
                stage.compute_fragments.push_back(unit.capture);
                ++cursor;
            }
            if (!saw_dispatch || !saw_return)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay follower stage has an incomplete consume/compute/pack body");
            }
            if (laneStageCount(stage_lane) != stage_count)
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay follower lane changed graph-family width");
            }
            if (ordinal == 0u)
                canonical_lane = stage_lane;
            else if (laneIdentity(stage_lane) != laneIdentity(canonical_lane))
            {
                throw std::invalid_argument(
                    "Retained ExpertOverlay follower lane changed between model stages");
            }

            while (cursor < classified.size() &&
                   classified[cursor].markers.empty())
            {
                stage.suffix_fragments.push_back(
                    classified[cursor].capture);
                ++cursor;
            }
        }

        if (cursor != classified.size())
        {
            throw std::invalid_argument(
                "Retained ExpertOverlay follower graph contains extra packet frontiers");
        }
        buildFollower(
            destination, canonical_lane, manifest_layers, stages);
    }

    void MoEOverlayRetainedActivationTransaction::
        buildStageOwnedTransactionFromCapturedUnits(
            IGPUGraphCapture &destination,
            const ComputeGraph &graph,
            std::span<const MoEOverlayRetainedCaptureUnit> units)
    {
        if (!requiresHeterogeneousTicketSegmentation(
                graph.nativeCaptureEnvelope()))
        {
            throw std::invalid_argument(
                "Stage-owned retained composition requires a typed heterogeneous ticket envelope");
        }

        std::vector<std::string> expected_stages;
        expected_stages.reserve(graph.getExecutionOrder().size());
        for (const std::string &stage_name : graph.getExecutionOrder())
        {
            const ComputeNode *const node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                throw std::invalid_argument(
                    "Stage-owned retained composition found an unresolved graph stage: " +
                    stage_name);
            }
            if (node->stage->isManualGraphBoundary() ||
                node->stage->isPassiveGraphCaptureNoOp())
            {
                continue;
            }
            expected_stages.push_back(stage_name);
        }

        std::vector<std::string> captured_stages;
        captured_stages.reserve(expected_stages.size());
        DeviceId device = DeviceId::invalid();
        for (const auto &unit : units)
        {
            for (const std::string &stage_name : unit.stage_names)
            {
                const ComputeNode *const node = graph.getNode(stage_name);
                if (!node || !node->stage ||
                    node->stage->isManualGraphBoundary() ||
                    node->stage->isPassiveGraphCaptureNoOp() ||
                    !node->device.is_gpu() ||
                    node->stage->device() != node->device)
                {
                    throw std::invalid_argument(
                        "Stage-owned retained composition contains an invalid captured stage: " +
                        stage_name);
                }
                if (!device.is_valid())
                    device = node->device;
                else if (node->device != device)
                {
                    throw std::invalid_argument(
                        "Stage-owned retained composition cannot combine multiple GPU endpoints");
                }
                captured_stages.push_back(stage_name);
            }
        }
        if (!device.is_gpu() || captured_stages != expected_stages)
        {
            throw std::invalid_argument(
                "Stage-owned retained composition does not exactly cover the ordered GPU graph");
        }

        std::deque<std::string> names;
        std::vector<MappedTimelineTransactionStep> steps;
        steps.reserve(units.size());
        for (std::size_t index = 0u; index < units.size(); ++index)
        {
            std::string name =
                "stage_owned_capture_unit_" + std::to_string(index);
            if (!units[index].stage_names.empty())
            {
                name += ":" + units[index].stage_names.front();
                if (units[index].stage_names.size() > 1u)
                {
                    name += ".." + units[index].stage_names.back();
                }
            }
            names.push_back(std::move(name));
            steps.emplace_back(MappedTimelineCapturedFragment{
                .name = names.back().c_str(),
                .capture = units[index].capture,
            });
        }
        TransferEngine::instance().buildMappedTimelineTransaction(
            destination, steps, device);
    }
} // namespace llaminar2
