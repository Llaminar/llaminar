/**
 * @file PlanningInterconnectCostMatrix.cpp
 * @brief Strict physical binding and bounded directed physical peer-copy curves.
 *
 * All arithmetic consumes completed sampler receipts. Explicit workload rounds
 * preserve live byte direction and concurrency without inventing collective
 * algorithms, link rates, topology advantages or a second capacity authority.
 */
#include "PlanningInterconnectCostMatrix.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @return Median of the three validated raw intervals; no sample is discarded from evidence. */
        double median(const PlanningLocalTPPairObservation &observation)
        {
            std::array<double, PlanningLocalTPPairSampling::kSamples> values{};
            std::copy(observation.seconds_per_transfer.begin(), observation.seconds_per_transfer.end(), values.begin());
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        }
    }

    PlanningInterconnectCostMatrix::PlanningInterconnectCostMatrix(PlanningInterconnectIdentity identity,
        std::span<const PlanningLocalTPPairObservation> observations, std::span<const uint8_t> direct_peer_access)
        : identity_(std::move(identity)), observations_(observations.begin(), observations.end()),
          direct_peer_access_(direct_peer_access.begin(), direct_peer_access.end())
    {
        if (identity_.discovery_rank < 0 || identity_.physical_node < 0 || identity_.uuids.empty() ||
            (identity_.backend != DeviceType::CUDA && identity_.backend != DeviceType::ROCm))
            throw std::invalid_argument("Interconnect matrix requires original rank/node and homogeneous GPU identity");
        std::set<std::string> uuids;
        for (const auto &uuid : identity_.uuids)
            if (uuid.find_first_not_of(" \r\n\t") == std::string::npos || !uuids.insert(uuid).second)
                throw std::invalid_argument("Interconnect matrix requires distinct nonempty physical UUIDs");
        for (size_t index = 0; index < identity_.uuids.size(); ++index)
            indices_.emplace(identity_.uuids[index], index);

        using Pair = std::pair<size_t, size_t>;
        std::map<Pair, std::set<size_t>> anchors;
        const size_t count = identity_.uuids.size();
        if (count > std::numeric_limits<size_t>::max() / count || direct_peer_access_.size() != count * count)
            throw std::invalid_argument("Physical peer-copy mask requires exact source-major NxN geometry");
        for (size_t source = 0; source < count; ++source)
            for (size_t destination = 0; destination < count; ++destination)
                if (direct_peer_access_[source * count + destination] > 1 ||
                    (source == destination && direct_peer_access_[source * count + destination] != 0))
                    throw std::invalid_argument("Physical peer-copy mask requires bits and zero self edges");
        for (const auto &sample : observations_)
        {
            if (sample.source_index >= count || sample.destination_index >= count ||
                sample.source_index == sample.destination_index || !sample.payload_bytes ||
                sample.seconds_per_transfer.size() != PlanningLocalTPPairSampling::kSamples ||
                !direct_peer_access_[sample.source_index * count + sample.destination_index] || !sample.graph_nodes ||
                !anchors[{sample.source_index, sample.destination_index}].insert(sample.payload_bytes).second)
                throw std::invalid_argument("Interconnect matrix has duplicate or incompatible directed sample geometry");
            for (const double seconds : sample.seconds_per_transfer)
                if (!std::isfinite(seconds) || seconds <= 0)
                    throw std::invalid_argument("Interconnect matrix requires completed positive finite clock samples");
            (void)median(sample);
        }
        if (count == 1)
        {
            if (!observations_.empty()) throw std::invalid_argument("Singleton interconnect matrix has no transfers");
            return;
        }
        const std::set<size_t> *basis = nullptr;
        for (size_t source = 0; source < count; ++source)
            for (size_t destination = 0; destination < count; ++destination)
                if (direct_peer_access_[source * count + destination])
                {
                    const auto found = anchors.find({source, destination});
                    if (found == anchors.end() || found->second.empty())
                        throw std::invalid_argument("Interconnect matrix is missing a measured direction");
                    if (basis && *basis != found->second)
                        throw std::invalid_argument("Interconnect directions require one common measured payload basis");
                    basis = &found->second;
                }
        for (size_t index = 0; index < observations_.size(); ++index)
        {
            const auto &sample = observations_[index];
            curves_[{sample.source_index, sample.destination_index}].push_back(index);
        }
        for (auto &entry : curves_)
        {
            auto &curve = entry.second;
            std::sort(curve.begin(), curve.end(), [&](size_t left, size_t right) {
                return observations_[left].payload_bytes < observations_[right].payload_bytes;
            });
        }
    }

    void PlanningInterconnectCostMatrix::requireIdentity(const PlanningInterconnectIdentity &expected) const
    {
        if (expected != identity_)
            throw std::invalid_argument("Interconnect observation identity differs from expected discovery rank/node/backend/UUID order");
    }

    bool PlanningInterconnectCostMatrix::hasDirectPeer(const std::string &source_uuid,
        const std::string &destination_uuid) const
    {
        return direct_peer_access_[indexOf(source_uuid) * identity_.uuids.size() + indexOf(destination_uuid)] != 0;
    }

    size_t PlanningInterconnectCostMatrix::indexOf(const std::string &uuid) const
    {
        const auto found = indices_.find(uuid);
        if (found == indices_.end())
            throw std::invalid_argument("Interconnect demand names an absent physical UUID");
        return found->second;
    }

    PlanningInterconnectService PlanningInterconnectCostMatrix::directedService(
        const PlanningInterconnectIdentity &expected, const std::string &source_uuid,
        const std::string &destination_uuid, size_t payload_bytes) const
    {
        requireIdentity(expected);
        const auto source = indexOf(source_uuid), destination = indexOf(destination_uuid);
        if (source == destination || !payload_bytes)
            throw std::invalid_argument("Interconnect demand requires distinct endpoints and positive live bytes");
        if (!hasDirectPeer(source_uuid, destination_uuid))
            throw std::invalid_argument("Physical peer-copy curve has no enabled edge; host-staged service is separate evidence");
        const PlanningLocalTPPairObservation *lower = nullptr, *upper = nullptr;
        const auto &curve = curves_.at({source, destination});
        const auto bracket = std::lower_bound(curve.begin(), curve.end(), payload_bytes,
            [&](size_t index, size_t bytes) { return observations_[index].payload_bytes < bytes; });
        if (bracket != curve.end()) upper = &observations_[*bracket];
        if (upper && upper->payload_bytes == payload_bytes) lower = upper;
        else if (bracket != curve.begin()) lower = &observations_[*std::prev(bracket)];
        if (!lower || !upper)
            throw std::invalid_argument("Interconnect payload lies outside its measured directed anchors");
        double seconds = median(*lower);
        if (lower != upper)
        {
            const double fraction = static_cast<double>(payload_bytes - lower->payload_bytes) /
                static_cast<double>(upper->payload_bytes - lower->payload_bytes);
            // A convex combination retains measured nonmonotonic curves without
            // subtracting nearly equal clocks or inventing a positive slope.
            seconds = (1 - fraction) * seconds + fraction * median(*upper);
        }
        return {seconds, lower->payload_bytes, upper->payload_bytes};
    }

    PlanningInterconnectWorkloadScore PlanningInterconnectCostMatrix::groupWorkloadScore(
        const PlanningInterconnectIdentity &expected, std::span<const std::string> group,
        const PlanningInterconnectWorkload &workload) const
    {
        requireIdentity(expected);
        if (group.empty()) throw std::invalid_argument("Interconnect group must contain a positive number of physical GPUs");
        std::set<std::string> members;
        for (const auto &uuid : group)
        {
            (void)indexOf(uuid);
            if (!members.insert(uuid).second) throw std::invalid_argument("Interconnect group repeats a physical GPU");
        }
        const auto phaseScore = [&](const PlanningInterconnectPhaseDemand &phase)
        {
            if (phase.rounds.empty() != (phase.occurrences == 0) && group.size() != 1)
                throw std::invalid_argument("Interconnect phase requires both explicit rounds and positive occurrences");
            if (group.size() == 1 && !phase.rounds.empty())
                throw std::invalid_argument("Singleton interconnect group cannot contain transfer rounds");
            double seconds = 0;
            for (const auto &round : phase.rounds)
            {
                if (round.empty()) throw std::invalid_argument("Interconnect schedule has an empty transfer round");
                std::set<std::string> senders, receivers;
                double critical = 0;
                for (const auto &edge : round)
                {
                    if (!members.contains(edge.source_uuid) || !members.contains(edge.destination_uuid) ||
                        !senders.insert(edge.source_uuid).second || !receivers.insert(edge.destination_uuid).second)
                        throw std::invalid_argument("Interconnect round escapes its group or overlaps a sender/receiver");
                    critical = std::max(critical, directedService(expected,
                        edge.source_uuid, edge.destination_uuid, edge.payload_bytes).seconds);
                }
                seconds += critical;
            }
            seconds *= static_cast<double>(phase.occurrences);
            if (!std::isfinite(seconds)) throw std::overflow_error("Interconnect workload score overflow");
            return seconds;
        };
        PlanningInterconnectWorkloadScore result{phaseScore(workload.prefill), phaseScore(workload.decode)};
        if (!std::isfinite(result.seconds())) throw std::overflow_error("Interconnect total workload score overflow");
        if (group.size() > 1 && !result.seconds())
            throw std::invalid_argument("Multi-GPU placement selection requires explicit live communication demand");
        return result;
    }
}
