/**
 * @file PlanningCommunicationCost.cpp
 * @brief Conservative payload composition without inventing link symmetry or collective speedups.
 *
 * Costing preserves the physical namespace from discovery and keeps protocols
 * disjoint. Predictions consume actual message bytes, never reserved workspace
 * extents. A small measured basis is reused across candidates; querying it does
 * not initialize a backend, send packets or launch a model warmup.
 */
#include "PlanningCommunicationCost.h"
#include <algorithm>
#include <array>
#include <climits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        /** @brief Reject identity/protocol gaps instead of pricing them as zero or another transport. */
        void require(bool condition, const char *detail)
        {
            if (!condition) throw std::invalid_argument(std::string("Planning communication cost: ") + detail);
        }

        /** @return A stable unordered identity used for membership only, never native communicator order. */
        std::vector<std::string> physicalSet(std::vector<std::string> uuids)
        {
            std::sort(uuids.begin(), uuids.end());
            require(!uuids.empty() && !uuids.front().empty() &&
                std::adjacent_find(uuids.begin(), uuids.end()) == uuids.end(), "missing or repeated physical GPU");
            return uuids;
        }

        /** @return Control/outbound/return coordinate; symmetric-only observations do not identify directions. */
        size_t exchangeCoordinate(const PlanningMPITransferRequest &request)
        {
            require(request.request_bytes > 0 && request.reply_bytes > 0 &&
                request.request_bytes <= INT_MAX && request.reply_bytes <= INT_MAX &&
                request.initiator != request.responder, "invalid directed MPI sample");
            if (request.request_bytes == 1 && request.reply_bytes == 1) return 0;
            if (request.reply_bytes == 1) return 1;
            if (request.request_bytes == 1) return 2;
            throw std::invalid_argument("Planning communication cost requires independent outbound/return samples");
        }
    }

    const RankInventory &PlanningCommunicationCost::rank(int discovery_rank) const
    {
        require(discovery_rank >= 0 && discovery_rank < inventory_.world_size &&
            inventory_.ranks.at(discovery_rank).rank == discovery_rank, "unknown discovery rank");
        return inventory_.ranks.at(discovery_rank);
    }

    const DeviceInfo &PlanningCommunicationCost::gpu(int discovery_rank, DeviceId device) const
    {
        require(device.is_gpu(), "native collective requires GPU endpoints");
        const auto &devices = rank(discovery_rank).gpus;
        const DeviceInfo *found = nullptr;
        for (const auto &candidate : devices)
            if (candidate.type == device.type && candidate.local_device_id == device.ordinal)
            {
                require(!found && !candidate.uuid.empty(), "duplicate or incomplete ordinal identity");
                found = &candidate;
            }
        require(found, "GPU ordinal is not visible on this discovery rank");
        return *found;
    }

    PlanningCommunicationCost::PlanningCommunicationCost(const ClusterInventory &inventory,
        std::span<const PlanningNativeCollectiveService> native,
        std::span<const PlanningMPITransferObservation> mpi, std::span<const PlanningHostDeviceObservations> host_device)
        : inventory_(inventory), native_(native.begin(), native.end()), mpi_(mpi.begin(), mpi.end()),
          host_device_(host_device.begin(), host_device.end())
    {
        require(inventory_.world_size > 0 && inventory_.ranks.size() == size_t(inventory_.world_size),
            "incomplete discovery membership");
        for (int index = 0; index < inventory_.world_size; ++index)
            (void)inventory_.connectionBetweenRanks(index, index);
        using NativeKey = std::tuple<int, DeviceType, std::vector<std::string>, PlanningAllreducePrecision>;
        std::set<NativeKey> seen;
        for (const auto &service : native_)
        {
            PlanningCommunicationService::validateNative(service);
            const auto &sample = service.sample;
            require(rank(sample.discovery_rank).node_id == sample.physical_node &&
                sample.uuids.size() == sample.request.devices().size(), "native physical membership mismatch");
            for (size_t index = 0; index < sample.uuids.size(); ++index)
                require(gpu(sample.discovery_rank, sample.request.devices()[index]).uuid == sample.uuids[index],
                    "native physical GPU/order substitution");
            (void)physicalSet(sample.uuids);
            require(seen.emplace(sample.physical_node, sample.request.devices().front().type,
                sample.uuids, sample.request.precision()).second, "duplicate native ordered group/precision");
        }
        std::map<std::pair<int, int>, std::array<bool, 3>> coordinates;
        for (const auto &sample : mpi_)
        {
            const auto &request = sample.request();
            require(sample.topology() == inventory_.connectionBetweenRanks(request.initiator, request.responder),
                "MPI sample belongs to another physical membership");
            auto &present = coordinates[{request.initiator, request.responder}][exchangeCoordinate(request)];
            require(!present, "duplicate MPI direction/payload coordinate");
            present = true;
        }
        for (const auto &[pair, present] : coordinates)
            require(std::all_of(present.begin(), present.end(), [](bool found) { return found; }),
                "incomplete MPI control/outbound/return basis");
        std::set<std::tuple<int, DeviceType, std::string>> host_coordinates;
        for (const auto &observed : host_device_)
        {
            PlanningHostDeviceMeasurement::validate(observed);
            const auto &request = observed.request;
            require(request == PlanningHostDeviceRequest::fromInventory(rank(request.rank()), request.device(), request.bulkBytes()),
                "GPU/host receipt belongs to another first-touch or physical endpoint scope");
            require(host_coordinates.emplace(request.rank(), request.device().type, request.uuid()).second,
                "duplicate GPU/host physical sample for one first-touch rank");
        }
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::nativeAllreduce(int discovery_rank,
        std::span<const DeviceId> devices, int width, int rows, PlanningAllreducePrecision precision) const
    {
        require(!devices.empty() && width > 0 && rows > 0 &&
            size_t(width) <= size_t(INT_MAX) / sizeof(float) / size_t(rows), "invalid native query geometry");
        require(precision == PlanningAllreducePrecision::FP32 || precision == PlanningAllreducePrecision::FP16,
            "unsupported native wire precision");
        const auto &owner = rank(discovery_rank);
        std::vector<std::string> uuids;
        for (const auto device : devices)
        {
            require(device.type == devices.front().type, "mixed backend group is not a native allreduce");
            uuids.push_back(gpu(discovery_rank, device).uuid);
        }
        const auto query_set = physicalSet(uuids);
        if (devices.size() == 1)
            return {0, PlanningCommunicationBasis::NoCommunication, "single physical GPU; no interconnect operation"};

        const PlanningNativeCollectiveService *selected = nullptr;
        // Exact order first, then the same physical set, then the smallest
        // containing group. Rank IDs are only a stable final tie-breaker.
        std::tuple<int, size_t, int> selected_key{3, SIZE_MAX, INT_MAX};
        for (const auto &service : native_)
        {
            const auto &sample = service.sample;
            if (sample.physical_node != owner.node_id || sample.request.devices().front().type != devices.front().type ||
                sample.request.precision() != precision) continue;
            const auto sample_set = physicalSet(sample.uuids);
            if (!std::includes(sample_set.begin(), sample_set.end(), query_set.begin(), query_set.end())) continue;
            const int relationship = sample.uuids == uuids ? 0 : sample_set == query_set ? 1 : 2;
            const auto key = std::tuple{relationship, sample.uuids.size(), sample.discovery_rank};
            if (key < selected_key) { selected = &service; selected_key = key; }
        }
        require(selected, "no measured containing native group with this physical membership and precision");
        const auto &phases = selected->observation.phases;
        const PlanningPayloadServiceCurve curve(phases[0].payload_bytes, phases[0].secondsPerCollective(),
            phases[1].payload_bytes, phases[1].secondsPerCollective());
        const size_t payload = size_t(width) * rows * (precision == PlanningAllreducePrecision::FP16 ? 2 : 4);
        const auto basis = std::get<0>(selected_key) == 0 ? PlanningCommunicationBasis::SameProtocolPayloadCurve :
            std::get<0>(selected_key) == 1 ? PlanningCommunicationBasis::ReorderedNativeGroup :
                PlanningCommunicationBasis::ContainingNativeGroup;
        return {curve.seconds(payload), basis,
            "native collective payload curve; node=" + std::to_string(owner.node_id) +
            "; measured-degree=" + std::to_string(selected->sample.uuids.size()) +
            "; query-degree=" + std::to_string(devices.size()) +
            "; full-group cost, no degree discount; subset/reorder is a proxy, not measured candidate throughput"};
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::mpiExchange(const PlanningMPITransferRequest &request) const
    {
        require(request.request_bytes > 0 && request.reply_bytes > 0 && request.request_bytes <= INT_MAX &&
            request.reply_bytes <= INT_MAX && request.initiator != request.responder, "invalid MPI query geometry");
        const auto topology = inventory_.connectionBetweenRanks(request.initiator, request.responder);
        std::array<const PlanningMPITransferObservation *, 3> samples{};
        for (const auto &sample : mpi_)
            if (sample.request().initiator == request.initiator && sample.request().responder == request.responder)
                samples[exchangeCoordinate(sample.request())] = &sample;
        require(std::all_of(samples.begin(), samples.end(), [](auto *sample) { return sample != nullptr; }),
            "missing exact directed MPI basis; reverse, native or mapped service cannot substitute");
        const double control = samples[0]->secondsPerExchange();
        const PlanningPayloadServiceCurve outbound(1, control, samples[1]->request().request_bytes,
            samples[1]->secondsPerExchange());
        const PlanningPayloadServiceCurve inbound(1, control, samples[2]->request().reply_bytes,
            samples[2]->secondsPerExchange());
        // Remove the common control RTT from each payload increment, then add
        // it exactly once. Neither increment is advertised as a one-way latency.
        const double seconds = planningSerialSeconds(std::array{control,
            outbound.seconds(request.request_bytes) - control, inbound.seconds(request.reply_bytes) - control});
        return {seconds, PlanningCommunicationBasis::SameProtocolPayloadCurve,
            std::string("host MPI ordered request/reply curve; ") +
            (topology.locality() == RankConnectionLocality::CrossNode ? "cross-node" : "same-node") +
            "; separate outbound/return service, one control RTT; excludes GPU DMA and mapped activation access"};
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::hostTransfer(int gpu_rank, DeviceId device, int host_rank,
        PlanningHostTransferMechanism mechanism, MappedTransferDirection direction, size_t payload) const
    {
        const auto topology = inventory_.connectionBetweenRanks(gpu_rank, host_rank);
        require(topology.locality() != RankConnectionLocality::CrossNode, "mapped host pages cannot cross physical machines");
        require(payload > 0 && payload <= INT_MAX &&
            (mechanism == PlanningHostTransferMechanism::DMA || mechanism == PlanningHostTransferMechanism::MappedKernel) &&
            (direction == MappedTransferDirection::HostToDevice || direction == MappedTransferDirection::DeviceToHost),
            "invalid GPU/host query mechanism/direction/payload");
        const auto &endpoint = gpu(gpu_rank, device);
        const auto found = std::find_if(host_device_.begin(), host_device_.end(), [&](const auto &observed) {
            return observed.request.rank() == host_rank && observed.request.device().type == device.type &&
                observed.request.uuid() == endpoint.uuid;
        });
        require(found != host_device_.end(), "missing exact GPU and host first-touch scope; another rank is not a substitute");
        const PlanningHostDevicePhaseObservation *small = nullptr, *large = nullptr;
        for (const auto &phase : found->phases)
            if (phase.mechanism == mechanism && phase.direction == direction)
            {
                if (phase.payload_bytes == 1) small = &phase;
                else large = &phase;
            }
        require(small && large, "incomplete GPU/host service coordinate");
        const double invocations = PlanningExecutionMeasurement::kTimedInvocations;
        const PlanningPayloadServiceCurve curve(1, small->service.elapsedSeconds() / invocations,
            large->payload_bytes, large->service.elapsedSeconds() / invocations);
        return {curve.seconds(payload), PlanningCommunicationBasis::SameProtocolPayloadCurve,
            "GPU-host byte primitive; host-rank=" + std::to_string(host_rank) +
            "; host-numa=" + std::to_string(found->request.hostNumaNode()) +
            "; mechanism=" + std::to_string(int(mechanism)) + "; direction=" + std::to_string(int(direction)) +
            "; excludes packet routing/protocol waits and MPI; no inferred NUMA relocation"};
    }
}
