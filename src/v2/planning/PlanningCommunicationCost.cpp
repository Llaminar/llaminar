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
#include "PlanningPeerCopyAccess.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
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
        {
            (void)inventory_.connectionBetweenRanks(index, index);
            pcie_.emplace(index, PlanningPCIeCommunicationCost(rank(index)));
        }
        using NativeKey = std::tuple<int, int, DeviceType, std::vector<std::string>, PlanningAllreducePrecision>;
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
            require(seen.emplace(sample.discovery_rank, sample.physical_node, sample.request.devices().front().type,
                sample.uuids, sample.request.precision()).second, "duplicate native ordered group/precision");
            if (sample.pair_sampling == PlanningPairSampling::Enabled)
            {
                const auto anchors = sample.request.pairPayloadAnchors();
                require(!anchors.empty(), "physical peer-copy matrix has no bounded payload basis");
                PlanningInterconnectIdentity identity{sample.discovery_rank, sample.physical_node,
                    sample.request.devices().front().type, sample.uuids};
                require(service.observation.pair_sampling.direct_peer_access ==
                    planningDirectedPeerCopyAccess(rank(sample.discovery_rank), sample.request.devices()),
                    "physical peer-copy eligibility differs from original directed inventory");
                require(physical_peers_.emplace(std::pair{identity.discovery_rank, identity.backend},
                    PhysicalPeerMatrix{PlanningInterconnectCostMatrix(identity,
                        service.observation.pair_sampling.directed_pairs,
                        service.observation.pair_sampling.direct_peer_access), anchors.front(), anchors.back()}).second,
                    "duplicate original observer/backend pair matrix");
            }
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
        std::set<std::tuple<int, int, DeviceType, std::string>> host_coordinates;
        for (const auto &observed : host_device_)
        {
            PlanningHostDeviceMeasurement::validate(observed);
            const auto &request = observed.request;
            require(request == PlanningHostDeviceRequest::fromInventory(inventory_,
                {request.gpuRank(), request.device()}, {request.hostRank()}, request.bulkBytes()),
                "GPU/host receipt belongs to another first-touch or physical endpoint scope");
            require(host_coordinates.emplace(request.gpuRank(), request.hostRank(), request.device().type, request.uuid()).second,
                "duplicate GPU executor/host first-touch sample");
        }
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::nativeAllreduce(int discovery_rank,
        std::span<const DeviceId> devices, int width, int rows, PlanningAllreducePrecision precision) const
    {
        return priceNativeAllreduce(discovery_rank, devices, width, rows, precision, false).prediction;
    }

    PlanningCommunicationCost::NativeAllreducePrice PlanningCommunicationCost::priceNativeAllreduce(int discovery_rank,
        std::span<const DeviceId> devices, int width, int rows, PlanningAllreducePrecision precision,
        bool exact_observer) const
    {
        require(!devices.empty() && width > 0 && rows > 0 &&
            size_t(width) <= size_t(INT_MAX) / sizeof(float) / size_t(rows), "invalid native query geometry");
        require(precision == PlanningAllreducePrecision::FP32 || precision == PlanningAllreducePrecision::FP16,
            "unsupported native wire precision");
        const auto reference = nativeReference(discovery_rank, devices, precision, exact_observer);
        if (!reference.service)
            return {{0, PlanningCommunicationBasis::NoCommunication, "single physical GPU; no interconnect operation"}, nullptr, 0};
        const auto &selected = *reference.service;
        const auto &phases = selected.observation.phases;
        const PlanningPayloadServiceCurve curve(phases[0].payload_bytes, phases[0].secondsPerCollective(),
            phases[1].payload_bytes, phases[1].secondsPerCollective());
        const size_t payload = size_t(width) * rows * (precision == PlanningAllreducePrecision::FP16 ? 2 : 4);
        return {{curve.seconds(payload), reference.relationship,
            "native collective payload curve; node=" + std::to_string(rank(discovery_rank).node_id) +
            "; measured-degree=" + std::to_string(selected.sample.uuids.size()) +
            "; query-degree=" + std::to_string(devices.size()) +
            "; full-group cost, no degree discount; subset/reorder is a proxy, not measured candidate throughput"},
            &selected, payload};
    }

    PlanningCommunicationCost::NativeReference PlanningCommunicationCost::nativeReference(int discovery_rank,
        std::span<const DeviceId> devices, std::optional<PlanningAllreducePrecision> precision, bool exact_observer) const
    {
        require(!devices.empty(), "native query requires original endpoints");
        const auto &owner = rank(discovery_rank);
        std::vector<std::string> uuids;
        for (const auto device : devices)
        {
            require(device.type == devices.front().type, "mixed backend group is not a native collective");
            uuids.push_back(gpu(discovery_rank, device).uuid);
        }
        const auto query_set = physicalSet(uuids);
        if (devices.size() == 1)
            return {nullptr, PlanningCommunicationBasis::NoCommunication};

        const PlanningNativeCollectiveService *selected = nullptr;
        // Exact order first, then the same physical set, then the smallest
        // containing group. Rank IDs are only a stable final tie-breaker.
        std::tuple<int, size_t, int, int> selected_key{3, SIZE_MAX, INT_MAX, INT_MAX};
        for (const auto &service : native_)
        {
            const auto &sample = service.sample;
            if (sample.mode != PlanningLocalTPMeasurementMode::NativeCollective ||
                (exact_observer && sample.discovery_rank != discovery_rank) ||
                sample.physical_node != owner.node_id || sample.request.devices().front().type != devices.front().type ||
                (precision && sample.request.precision() != *precision)) continue;
            const auto sample_set = physicalSet(sample.uuids);
            if (!std::includes(sample_set.begin(), sample_set.end(), query_set.begin(), query_set.end())) continue;
            const int relationship = sample.uuids == uuids ? 0 : sample_set == query_set ? 1 : 2;
            // Gather bytes have no FP wire type. Prefer the first FP32 sample
            // deterministically when both receipts observed the same group;
            // either contains the identical integer-byte gather operation.
            const auto key = std::tuple{relationship, sample.uuids.size(), sample.discovery_rank,
                static_cast<int>(sample.request.precision())};
            if (key < selected_key) { selected = &service; selected_key = key; }
        }
        require(selected, "no measured containing native group with this physical membership and precision");
        const auto basis = std::get<0>(selected_key) == 0 ? PlanningCommunicationBasis::SameProtocolPayloadCurve :
            std::get<0>(selected_key) == 1 ? PlanningCommunicationBasis::ReorderedNativeGroup :
                PlanningCommunicationBasis::ContainingNativeGroup;
        return {selected, basis};
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::calibratedNativeAllreduce(int discovery_rank,
        std::span<const DeviceId> devices, int width, int rows, PlanningAllreducePrecision precision) const
    {
        const auto native = priceNativeAllreduce(discovery_rank, devices, width, rows, precision, true);
        if (!native.reference) return native.prediction;
        const auto &reference = *native.reference;
        const auto &cuts = pcie_.at(discovery_rank);
        const auto reference_cut = cuts.allreduce(reference.sample.request.devices(), native.payload);
        const auto candidate_cut = cuts.allreduce(devices, native.payload);
        require(reference_cut.seconds <= native.prediction.seconds,
            "native allreduce curve is below its observed PCIe wire floor; inconsistent timing/topology evidence");
        const auto original_pair = physicalGroupPlacement(discovery_rank, reference.sample.request.devices(), native.payload);
        const auto candidate_pair = physicalGroupPlacement(discovery_rank, devices, native.payload);
        const double original_pressure = std::max(original_pair.seconds, reference_cut.seconds);
        const double candidate_pressure = std::max(candidate_pair.seconds, candidate_cut.seconds);
        const auto &phases = reference.observation.phases;
        const PlanningPayloadServiceCurve curve(phases[0].payload_bytes, phases[0].secondsPerCollective(),
            phases[1].payload_bytes, phases[1].secondsPerCollective());
        // The smallest positive native sample is an observed floor proxy, not
        // a fitted zero-byte intercept. Copy latency cannot discount it.
        const double floor = curve.seconds(1);
        const double seconds = std::max(candidate_cut.seconds,
            floor + (native.prediction.seconds - floor) * candidate_pressure / original_pressure);
        require(std::isfinite(seconds) && seconds > 0, "calibrated native allreduce has invalid service");
        return {seconds, PlanningCommunicationBasis::CalibratedNativeAllreduceProxy,
            native.prediction.evidence + "; calibrated native floor=" + std::to_string(floor) +
            "; payload-excess pressure ratio=" + std::to_string(candidate_pressure / original_pressure) +
            "; fixed protocol floor retained; candidate-pcie-floor=" + std::to_string(candidate_cut.seconds) +
            "; known-cuts=" + std::to_string(candidate_cut.known_cuts) +
            "; unknown-cuts=" + std::to_string(candidate_cut.unknown_cuts) +
            "; topology-complete=" + std::to_string(candidate_cut.topology_complete) +
            "; PCIe negotiated wire bounds exclude unknown fabric contention; original=" + original_pair.evidence +
            "; candidate=" + candidate_pair.evidence};
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::calibratedNativeAllgather(int discovery_rank,
        std::span<const DeviceId> devices, size_t bytes_per_owner) const
    {
        require(bytes_per_owner > 0, "fixed allgather requires positive live source bytes");
        return calibratedGather(discovery_rank, devices, std::vector<size_t>(devices.size(), bytes_per_owner),
            PlanningNativeGatherProtocol::FixedAllgather);
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::calibratedNativeCountedAllgather(int discovery_rank,
        std::span<const DeviceId> devices, std::span<const size_t> owner_live_bytes) const
    {
        return calibratedGather(discovery_rank, devices, owner_live_bytes,
            PlanningNativeGatherProtocol::CountedAllgather);
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::calibratedGather(int discovery_rank,
        std::span<const DeviceId> devices, std::span<const size_t> owner_live_bytes,
        PlanningNativeGatherProtocol protocol) const
    {
        require(!devices.empty() && owner_live_bytes.size() == devices.size(),
            "native gather requires one original fragment per participant");
        const bool counted = protocol == PlanningNativeGatherProtocol::CountedAllgather;
        require(counted || protocol == PlanningNativeGatherProtocol::FixedAllgather, "unknown native gather protocol");
        const size_t control_bytes = counted ? sizeof(uint64_t) : 0;
        std::vector<size_t> wire_bytes;
        wire_bytes.reserve(devices.size());
        size_t total = 0;
        for (const auto bytes : owner_live_bytes)
        {
            require(counted || (bytes > 0 && bytes == owner_live_bytes.front()),
                "fixed native allgather cannot price unequal or absent fragments");
            if (bytes > SIZE_MAX - control_bytes || bytes > SIZE_MAX - total)
                throw std::overflow_error("Native gather live extent overflow");
            total += bytes;
            wire_bytes.push_back(bytes + control_bytes);
        }
        const auto reference = nativeReference(discovery_rank, devices, {}, true);
        const auto &cuts = pcie_.at(discovery_rank);
        // Counted zero payload is a real invocation. Each root still publishes
        // its uint64 extent; the native frame/completion cost lives in its
        // measured zero-payload anchor, not in invented nonzero data.
        const auto cut = cuts.allgather(devices, wire_bytes);
        if (!reference.service)
            return {0, PlanningCommunicationBasis::NoCommunication, "single physical GPU; no allgather"};
        const auto &sample = *reference.service;
        std::array<const PlanningLocalTPGatherPhaseObservation *, 2> phases{};
        size_t coordinate = 0;
        for (const auto &phase : sample.observation.gather_phases)
            if (phase.protocol == protocol)
            {
                require(coordinate < phases.size(), "duplicate native gather payload coordinate");
                phases[coordinate++] = &phase;
            }
        require(coordinate == phases.size(), "missing actual native gather protocol observations");
        const size_t bulk_bytes = phases[1]->payload_bytes;
        if (bulk_bytes > SIZE_MAX - control_bytes)
            throw std::overflow_error("Native gather measured extent overflow");
        const PlanningPayloadServiceCurve curve(phases[0]->payload_bytes + control_bytes,
            phases[0]->secondsPerCollective(), bulk_bytes + control_bytes, phases[1]->secondsPerCollective());
        const double reference_floor = curve.seconds(phases[0]->payload_bytes + control_bytes);
        const double reference_degree = sample.sample.request.devices().size();
        double floor = reference_floor;
        double nominal = 0;
        if (counted)
        {
            // The production implementation explicitly executes one ordered
            // broadcast per owner. Use that loop's observed mean, including
            // an empty owner's control. Unequal roots and changed membership
            // remain estimates, not newly measured native invocations.
            floor *= static_cast<double>(devices.size()) / reference_degree;
            for (const auto bytes : wire_bytes) nominal += curve.seconds(bytes) / reference_degree;
        }
        else nominal = curve.seconds(owner_live_bytes.front());

        double pressure_ratio = 1;
        if (reference.relationship == PlanningCommunicationBasis::ContainingNativeGroup)
        {
            const auto &original = sample.sample.request.devices();
            // Only the bulk anchor determines the payload-rate adjustment.
            // Pricing a native control packet from tiny-copy launch latency
            // amplified timer noise once per layer and favored arbitrary mixes.
            const auto pressure = [&](std::span<const DeviceId> members) {
                const double peer = physicalGroupPlacement(discovery_rank, members, bulk_bytes).seconds;
                const double wire = counted ? cuts.allreduce(members, bulk_bytes).seconds :
                    cuts.allgather(members, std::vector<size_t>(members.size(), bulk_bytes)).seconds;
                return std::max(peer, wire);
            };
            pressure_ratio = pressure(devices) / pressure(original);
        }
        const std::vector<size_t> reference_wire(sample.sample.request.devices().size(), bulk_bytes + control_bytes);
        const auto reference_cut = cuts.allgather(sample.sample.request.devices(), reference_wire);
        require(reference_cut.seconds <= curve.seconds(bulk_bytes + control_bytes),
            "native gather curve is below its observed PCIe wire floor; inconsistent timing/topology evidence");
        const double seconds = std::max(cut.seconds, floor + std::max(0.0, nominal - floor) * pressure_ratio);
        require(std::isfinite(seconds) && seconds > 0, "calibrated native gather has invalid service");
        return {seconds, counted ? PlanningCommunicationBasis::CalibratedNativeCountedAllgatherProxy :
                PlanningCommunicationBasis::CalibratedNativeAllgatherProxy,
            std::string(counted ? "native counted gather: extent, payload, completion and frame validation" :
                "native fixed allgather; equal DeviceRows uses this payload-service estimate, count binding unmeasured") +
            "; measured-degree=" + std::to_string(sample.sample.request.devices().size()) +
            "; query-degree=" + std::to_string(devices.size()) +
            "; total-live-bytes=" + std::to_string(total) +
            "; control-bytes-per-owner=" + std::to_string(control_bytes) +
            "; bulk-pressure-ratio=" + std::to_string(pressure_ratio) +
            "; native control floor retained; shared-pcie-floor=" + std::to_string(cut.seconds) +
            "; known-cuts=" + std::to_string(cut.known_cuts) + "; unknown-cuts=" + std::to_string(cut.unknown_cuts) +
            "; topology-complete=" + std::to_string(cut.topology_complete) +
            (counted ? "; changed degree/unequal fragments use measured root mean with amortized frame cost" :
                "; containing group retains full native control cost, no degree discount") +
            "; subset/reorder is a payload-service proxy, not measured candidate throughput; unique fragments cross each cut once"};
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::physicalPeerTransfer(int discovery_rank,
        DeviceId source, DeviceId destination, size_t payload) const
    {
        require(payload > 0 && source != destination && source.type == destination.type,
            "physical peer placement requires distinct homogeneous endpoints and positive live bytes");
        const auto &sender = gpu(discovery_rank, source);
        const auto &receiver = gpu(discovery_rank, destination);
        const auto found = physical_peers_.find({discovery_rank, source.type});
        require(found != physical_peers_.end(), "missing exact original observer/backend physical peer-copy matrix");
        const auto &basis = found->second;
        const bool direct = basis.matrix.hasDirectPeer(sender.uuid, receiver.uuid);
        return {physicalEdgeSeconds(basis, source, destination, sender.uuid, receiver.uuid, payload),
            direct ? PlanningCommunicationBasis::PhysicalDirectedPeerPlacementProxy :
                PlanningCommunicationBasis::PhysicalHostStagedPlacementProxy,
            direct ? directPeerEvidence(basis, sender.uuid, receiver.uuid, payload) :
                "physical nonpeer copy placement proxy; observer=" + std::to_string(discovery_rank) +
                "; source=" + sender.uuid + "; destination=" + receiver.uuid + "; live-bytes=" + std::to_string(payload) +
                "; composed same-rank-host DMA source D2H then destination H2D; measured host services, not native RCCL peer or collective latency"};
    }

    double PlanningCommunicationCost::physicalEdgeSeconds(const PhysicalPeerMatrix &basis,
        DeviceId source, DeviceId destination, const std::string &source_uuid,
        const std::string &destination_uuid, size_t payload) const
    {
        if (basis.matrix.hasDirectPeer(source_uuid, destination_uuid))
            return directPeerSeconds(basis, source_uuid, destination_uuid, payload);
        const int observer = basis.matrix.identity().discovery_rank;
        const double seconds = hostTransferPrice(observer, source, observer, PlanningHostTransferMechanism::DMA,
            MappedTransferDirection::DeviceToHost, payload).seconds +
            hostTransferPrice(observer, destination, observer, PlanningHostTransferMechanism::DMA,
                MappedTransferDirection::HostToDevice, payload).seconds;
        require(std::isfinite(seconds) && seconds > 0, "physical nonpeer host composition has no finite positive service");
        return seconds;
    }

    double PlanningCommunicationCost::directPeerSeconds(const PhysicalPeerMatrix &basis,
        const std::string &source_uuid, const std::string &destination_uuid, size_t payload)
    {
        const auto &identity = basis.matrix.identity();
        const auto bounded = [&](size_t bytes) {
            return basis.matrix.directedService(identity, source_uuid, destination_uuid,
                std::max(bytes, basis.minimum_payload)).seconds;
        };
        const size_t full_chunks = payload / basis.maximum_payload;
        const size_t remainder = payload % basis.maximum_payload;
        double seconds = payload <= basis.maximum_payload ? bounded(payload) :
            static_cast<double>(full_chunks) * bounded(basis.maximum_payload) + (remainder ? bounded(remainder) : 0);
        if (!std::isfinite(seconds) || seconds <= 0)
            throw std::overflow_error("Physical direct-peer placement proxy has no finite positive duration");
        return seconds;
    }

    std::string PlanningCommunicationCost::directPeerEvidence(const PhysicalPeerMatrix &basis,
        const std::string &source_uuid, const std::string &destination_uuid, size_t payload)
    {
        const auto &identity = basis.matrix.identity();
        return "physical destination-owned direct-peer copy placement proxy; observer=" + std::to_string(identity.discovery_rank) +
            "; node=" + std::to_string(identity.physical_node) + "; source=" + source_uuid +
            "; destination=" + destination_uuid + "; live-bytes=" + std::to_string(payload) +
            "; measured-min-bytes=" + std::to_string(basis.minimum_payload) +
            "; measured-max-bytes=" + std::to_string(basis.maximum_payload) +
            "; anchor-estimator=median-of-three-observed-services" +
            "; full-max-chunks=" + std::to_string(payload > basis.maximum_payload ? payload / basis.maximum_payload : 0) +
            "; tail-live-bytes=" + std::to_string(payload > basis.maximum_payload ? payload % basis.maximum_payload : payload) +
            "; sub-anchor uses measured minimum service; larger payload uses sequential measured-size chunks" +
            "; packetization is a selection surrogate, not actual graph schedule or collective latency";
    }

    PlanningCommunicationPrediction PlanningCommunicationCost::physicalGroupPlacement(int discovery_rank,
        std::span<const DeviceId> devices, size_t payload) const
    {
        require(!devices.empty() && payload > 0, "physical group placement requires positive degree and live bytes");
        (void)rank(discovery_rank);
        std::vector<std::string> uuids;
        uuids.reserve(devices.size());
        for (const auto device : devices)
        {
            require(device.type == devices.front().type, "physical group placement cannot mix backends");
            uuids.push_back(gpu(discovery_rank, device).uuid);
        }
        (void)physicalSet(uuids);
        if (devices.size() == 1)
            return {0, PlanningCommunicationBasis::NoCommunication, "single physical GPU; no pair placement service"};
        const auto found = physical_peers_.find({discovery_rank, devices.front().type});
        require(found != physical_peers_.end(), "missing exact original observer/backend physical peer-copy matrix");
        const auto &basis = found->second;
        double seconds = 0;
        size_t critical_source = 0, critical_destination = 1;
        for (size_t source = 0; source < uuids.size(); ++source)
            for (size_t destination = 0; destination < uuids.size(); ++destination)
                if (source != destination)
                {
                    const double edge_seconds = physicalEdgeSeconds(basis, devices[source], devices[destination],
                        uuids[source], uuids[destination], payload);
                    if (edge_seconds > seconds)
                    {
                        seconds = edge_seconds;
                        critical_source = source;
                        critical_destination = destination;
                    }
                }
        return {seconds, PlanningCommunicationBasis::PhysicalGroupWorstPeerPlacementProxy,
            "common physical-edge worst-service placement proxy; degree=" + std::to_string(devices.size()) +
            "; no inferred ring/tree/all-to-all schedule; not collective latency or a performance bound; critical=" +
                physicalPeerTransfer(discovery_rank, devices[critical_source], devices[critical_destination], payload).evidence};
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
        const auto price = hostTransferPrice(gpu_rank, device, host_rank, mechanism, direction, payload);
        return {price.seconds, PlanningCommunicationBasis::SameProtocolPayloadCurve,
            "GPU-host byte primitive; gpu-executor-rank=" + std::to_string(gpu_rank) +
            "; host-rank=" + std::to_string(host_rank) + "; host-numa=" + std::to_string(price.numa) +
            "; mechanism=" + std::to_string(int(mechanism)) + "; direction=" + std::to_string(int(direction)) +
            "; excludes packet routing/protocol waits and MPI; no inferred NUMA relocation"};
    }

    PlanningCommunicationCost::HostTransferPrice PlanningCommunicationCost::hostTransferPrice(
        int gpu_rank, DeviceId device, int host_rank, PlanningHostTransferMechanism mechanism,
        MappedTransferDirection direction, size_t payload) const
    {
        const auto topology = inventory_.connectionBetweenRanks(gpu_rank, host_rank);
        require(topology.locality() != RankConnectionLocality::CrossNode, "mapped host pages cannot cross physical machines");
        require(payload > 0 && payload <= INT_MAX &&
            (mechanism == PlanningHostTransferMechanism::DMA || mechanism == PlanningHostTransferMechanism::MappedKernel) &&
            (direction == MappedTransferDirection::HostToDevice || direction == MappedTransferDirection::DeviceToHost),
            "invalid GPU/host query mechanism/direction/payload");
        const auto &endpoint = gpu(gpu_rank, device);
        const auto found = std::find_if(host_device_.begin(), host_device_.end(), [&](const auto &observed) {
            return observed.request.gpuRank() == gpu_rank && observed.request.hostRank() == host_rank &&
                observed.request.device().type == device.type &&
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
        return {curve.seconds(payload), found->request.hostNumaNode()};
    }
}
