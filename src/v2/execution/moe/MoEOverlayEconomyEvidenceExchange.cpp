/**
 * @file MoEOverlayEconomyEvidenceExchange.cpp
 * @brief Pure validation and conservative merge of distributed economy rows.
 */

#include "MoEOverlayEconomyEvidenceExchange.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @return Whether two rows name one exact physical migration. */
        bool sameMigrationIdentity(
            const MoEOverlayCompletedMigrationMeasurement &left,
            const MoEOverlayCompletedMigrationMeasurement &right) noexcept
        {
            return left.expected_epoch == right.expected_epoch &&
                   left.candidate_epoch == right.candidate_epoch &&
                   left.source_participant == right.source_participant &&
                   left.destination_participant ==
                       right.destination_participant &&
                   left.layer == right.layer && left.expert == right.expert;
        }

        /** @return Non-negative interference without unsigned underflow. */
        std::uint64_t interference(
            const MoEOverlayCalibrationAttemptEvidence &attempt) noexcept
        {
            return attempt.concurrent_nanoseconds >
                           attempt.baseline_nanoseconds
                       ? attempt.concurrent_nanoseconds -
                             attempt.baseline_nanoseconds
                       : 0;
        }
    } // namespace

    bool MoEOverlayMigrationProfileEvidence::valid() const noexcept
    {
        if (profile_sequence == 0 || !coordinate.valid() ||
            coordinate.source_participant >=
                coordinate.destination_participant ||
            local_measurements.size() != 2)
        {
            return false;
        }
        const auto &forward = local_measurements[0];
        const auto &reverse = local_measurements[1];
        return forward.valid() && reverse.valid() &&
               forward.source_participant ==
                   coordinate.source_participant &&
               forward.destination_participant ==
                   coordinate.destination_participant &&
               reverse.source_participant ==
                   coordinate.destination_participant &&
               reverse.destination_participant ==
                   coordinate.source_participant &&
               forward.layer == coordinate.layer &&
               reverse.layer == coordinate.layer;
    }

    bool MoEOverlayMigrationProfileResult::valid() const noexcept
    {
        if (measurements.size() != 2)
            return false;
        return measurements[0].valid() && measurements[1].valid() &&
               measurements[0].source_participant ==
                   measurements[1].destination_participant &&
               measurements[0].destination_participant ==
                   measurements[1].source_participant &&
               measurements[0].layer == measurements[1].layer;
    }

    MoEOverlayMigrationProfileResult
    MoEOverlayEconomyEvidenceMerger::mergeMigrationProfile(
        const std::vector<MoEOverlayMigrationProfileEvidence> &ranks)
    {
        if (ranks.empty() || !ranks.front().valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay migration-profile merge requires valid rank evidence");
        }
        std::vector<std::vector<MoEOverlayCompletedMigrationMeasurement>>
            local_rows;
        local_rows.reserve(ranks.size());
        for (const auto &rank : ranks)
        {
            if (!rank.valid() ||
                rank.profile_sequence != ranks.front().profile_sequence ||
                rank.coordinate != ranks.front().coordinate)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration-profile ranks disagree on identity");
            }
            local_rows.push_back(rank.local_measurements);
        }
        MoEOverlayMigrationProfileResult result{
            .measurements =
                MoEOverlayMigrationMeasurementMerger::merge(local_rows),
        };
        if (!result.valid())
        {
            throw std::logic_error(
                "ExpertOverlay migration-profile merge produced incomplete rows");
        }
        return result;
    }

    bool MoEOverlayCalibrationReadiness::valid() const noexcept
    {
        return kind ==
                   MoEOverlayCalibrationReadinessKind::BaselineDeviceComplete &&
               calibration_sequence != 0 && coordinate.valid() &&
               coordinate.source_participant <
                   coordinate.destination_participant &&
               static_cast<std::size_t>(source) <
                   kExpertHistogramProductionSourceCount &&
               workload.valid() && workload.source == source;
    }

    bool MoEOverlayCalibrationAttemptEvidence::valid() const noexcept
    {
        if (calibration_sequence == 0 || !coordinate.valid() ||
            coordinate.source_participant >=
                coordinate.destination_participant ||
            static_cast<std::size_t>(source) >=
                kExpertHistogramProductionSourceCount ||
            !workload.valid() || workload.source != source ||
            baseline_nanoseconds == 0 || local_measurements.size() != 2 ||
            (exact_overlap && concurrent_nanoseconds == 0))
        {
            return false;
        }
        const auto &forward = local_measurements[0];
        const auto &reverse = local_measurements[1];
        return forward.valid() && reverse.valid() &&
               forward.source_participant ==
                   coordinate.source_participant &&
               forward.destination_participant ==
                   coordinate.destination_participant &&
               reverse.source_participant ==
                   coordinate.destination_participant &&
               reverse.destination_participant ==
                   coordinate.source_participant &&
               forward.layer == coordinate.layer &&
               reverse.layer == coordinate.layer;
    }

    bool MoEOverlayCalibrationAttemptResult::valid() const noexcept
    {
        if (!accepted)
        {
            return measurements.empty() && baseline_nanoseconds == 0 &&
                   concurrent_nanoseconds == 0;
        }
        return baseline_nanoseconds != 0 && concurrent_nanoseconds != 0 &&
               measurements.size() == 2 &&
               std::all_of(
                   measurements.begin(),
                   measurements.end(),
                   [](const auto &measurement)
                   {
                       return measurement.valid() &&
                              std::all_of(
                                  measurement.projections.begin(),
                                  measurement.projections.end(),
                                  [](const auto &projection)
                                  {
                                      return projection.has_value() &&
                                             projection->valid();
                                  });
                   });
    }

    MoEOverlayCalibrationAttemptResult
    MoEOverlayEconomyEvidenceMerger::mergeAttempt(
        const std::vector<MoEOverlayCalibrationAttemptEvidence> &ranks)
    {
        if (ranks.empty() || !ranks.front().valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay attempt merge requires valid rank evidence");
        }
        const auto &first = ranks.front();
        bool accepted = first.exact_overlap;
        std::size_t maximum_interference_rank = 0;
        std::uint64_t maximum_interference = interference(first);
        for (std::size_t rank = 0; rank < ranks.size(); ++rank)
        {
            const auto &candidate = ranks[rank];
            if (!candidate.valid() ||
                candidate.calibration_sequence !=
                    first.calibration_sequence ||
                candidate.coordinate != first.coordinate ||
                candidate.source != first.source ||
                candidate.workload != first.workload)
            {
                throw std::invalid_argument(
                    "ExpertOverlay attempt ranks disagree on calibration or workload identity");
            }
            for (std::size_t migration = 0; migration < 2; ++migration)
            {
                if (!sameMigrationIdentity(
                        first.local_measurements[migration],
                        candidate.local_measurements[migration]))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay attempt ranks disagree on migration identity");
                }
            }
            accepted = accepted && candidate.exact_overlap;
            const std::uint64_t candidate_interference =
                interference(candidate);
            if (candidate_interference > maximum_interference)
            {
                maximum_interference = candidate_interference;
                maximum_interference_rank = rank;
            }
        }
        if (!accepted)
            return {};

        MoEOverlayCalibrationAttemptResult result;
        result.accepted = true;
        result.measurements =
            MoEOverlayMigrationMeasurementMerger::merge(
                [&ranks]
                {
                    std::vector<std::vector<
                        MoEOverlayCompletedMigrationMeasurement>> rows;
                    rows.reserve(ranks.size());
                    for (const auto &rank : ranks)
                        rows.push_back(rank.local_measurements);
                    return rows;
                }());
        result.baseline_nanoseconds =
            ranks[maximum_interference_rank].baseline_nanoseconds;
        result.concurrent_nanoseconds =
            ranks[maximum_interference_rank].concurrent_nanoseconds;
        if (!result.valid())
        {
            throw std::logic_error(
                "ExpertOverlay attempt merge produced an invalid result");
        }
        return result;
    }

    std::vector<MoEOverlayParticipantLayerServiceTotals>
    MoEOverlayEconomyEvidenceMerger::mergeService(
        const std::vector<std::vector<
            MoEOverlayParticipantLayerServiceTotals>> &rank_rows,
        const MoEExpertOwnerMap &owner_map,
        const ExpertHistogramProductionTopology &production_topology)
    {
        if (rank_rows.empty() || owner_map.participants().empty() ||
            !production_topology.valid() ||
            production_topology.layerCount() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument(
                "ExpertOverlay service merge requires rank, participant, and layer geometry");
        }
        const int num_layers =
            static_cast<int>(production_topology.layerCount());

        using Coordinate = std::pair<int, int>;
        std::map<Coordinate, MoEOverlayParticipantLayerServiceTotals> merged;
        for (std::size_t rank = 0; rank < rank_rows.size(); ++rank)
        {
            for (const auto &row : rank_rows[rank])
            {
                const auto *participant =
                    owner_map.participantForId(row.participant_id);
                if (!row.valid() || !participant ||
                    !participant->world_rank_known ||
                    participant->world_rank != static_cast<int>(rank) ||
                    row.layer < 0 || row.layer >= num_layers)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service ranks supplied a malformed, out-of-range, or wrongly owned row");
                }
                for (std::size_t phase = 0;
                     phase < kExpertHistogramProductionSourceCount;
                     ++phase)
                {
                    if (!production_topology.reachable(row.layer, phase) &&
                        row.sample_count[phase] != 0)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay service ranks supplied evidence for an unreachable layer/phase coordinate");
                    }
                }
                if (!merged.emplace(
                            Coordinate{row.participant_id, row.layer}, row)
                         .second)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service ranks supplied a duplicate participant/layer row");
                }
            }
        }

        std::vector<MoEOverlayParticipantLayerServiceTotals> result;
        result.reserve(
            owner_map.participants().size() *
            static_cast<std::size_t>(num_layers));
        std::vector<int> participant_ids;
        participant_ids.reserve(owner_map.participants().size());
        for (const auto &participant : owner_map.participants())
            participant_ids.push_back(participant.participant_id);
        std::sort(participant_ids.begin(), participant_ids.end());
        for (const int participant : participant_ids)
        {
            for (int layer = 0; layer < num_layers; ++layer)
            {
                const auto found = merged.find({participant, layer});
                if (found == merged.end())
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service ranks omitted a participant/layer row");
                }
                result.push_back(found->second);
            }
        }
        return result;
    }
} // namespace llaminar2
