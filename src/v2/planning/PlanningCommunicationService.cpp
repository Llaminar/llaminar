/**
 * @file PlanningCommunicationService.cpp
 * @brief Failure-atomic collection of bounded native and host-MPI communication evidence.
 *
 * One discovery transaction authenticates the entire sample basis before PMA
 * admission. Native group workers run only within a node's designated reporter
 * round; MPI follows after every retained native borrower retires. Group and
 * packet observations remain separate, and all numerical times are completed
 * observations rather than capability flags or assumed hardware specifications.
 */
#include "PlanningCommunicationService.h"
#include "PlanningKernelServiceCatalog.h"
#include "PlanningObservedResource.h"
#include "interfaces/IMPIContext.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <climits>
#include <map>
#include <set>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        using Json = nlohmann::json;
        constexpr auto kSchema = "llaminar.planning-communication.v3";

        /** @brief Reject incomplete evidence before it can influence selection. */
        void require(bool condition, const char *detail)
        {
            if (!condition) throw std::invalid_argument(std::string("Planning communication: ") + detail);
        }

        /** @return Strict nonnegative integer, excluding floating/truncated JSON values. */
        size_t integer(const Json &value, size_t maximum = SIZE_MAX)
        {
            require(value.is_number_integer() && (value.is_number_unsigned() || value.get<int64_t>() >= 0),
                "expected nonnegative integer");
            require(value.get<uint64_t>() <= maximum, "integer outside supported range");
            return value.get<size_t>();
        }

        /** @return Native protocol and complete ordered payload geometry. */
        Json identity(const PlanningLocalTPRequest &request)
        {
            Json devices = Json::array();
            for (const auto device : request.devices()) devices.push_back({int(device.type), device.ordinal});
            return {{"devices", devices}, {"width", request.hiddenWidth()}, {"rows", request.prefillRows()},
                {"precision", int(request.precision())}, {"backend", int(request.backend())}};
        }

        /** @return Complete physical host/GPU identity, including rank-owned first touch. */
        Json identity(const PlanningHostDeviceRequest &request)
        {
            return {{"rank", request.rank()}, {"node", request.physicalNode()}, {"host_numa", request.hostNumaNode()},
                {"device", {int(request.device().type), request.device().ordinal}}, {"uuid", request.uuid()},
                {"bulk_bytes", request.bulkBytes()}};
        }

        /** @return Wire bytes for immutable control/receipt metadata, never transfer payloads. */
        std::vector<uint8_t> bytes(const Json &value)
        {
            const auto text = value.dump();
            return {text.begin(), text.end()};
        }

        /** @brief Validate every phase and ordered endpoint, not merely a positive aggregate time. */
        void validate(const PlanningLocalTPObservations &observed, const PlanningNativeCollectiveSample &sample)
        {
            require(identity(observed.request) == identity(sample.request) && observed.phases.size() == 2,
                "native request or phase substitution");
            for (size_t index = 0; index < observed.phases.size(); ++index)
            {
                const auto &phase = observed.phases[index];
                const int rows = index ? sample.request.prefillRows() : 1;
                require(phase.rows == rows && phase.payload_bytes == sample.request.payloadBytes(rows) &&
                    phase.endpoints.size() == sample.request.devices().size(), "wrong native payload or degree");
                for (size_t endpoint = 0; endpoint < phase.endpoints.size(); ++endpoint)
                    require(phase.endpoints[endpoint].device == sample.request.devices()[endpoint],
                        "native endpoint order/ordinal substitution");
                (void)phase.secondsPerCollective(); // Also rejects zero graphs, NaN and incomplete completion.
            }
        }

        /** @return One per-rank mutually exclusive PMA envelope for the complete observation batch. */
        std::shared_ptr<PhysicalMemoryAuthority> admit(const ClusterInventory &inventory, int rank,
            const PlanningCommunicationSamplePlan &plan)
        {
            const auto &observed = inventory.ranks.at(rank);
            const auto host = planningObservedResource(observed, DeviceId::cpu());
            std::vector<PhysicalMemoryPlan> alternatives;
            for (const auto &sample : plan.native())
            {
                if (sample.discovery_rank != rank) continue;
                PhysicalMemoryPlanBuilder builder;
                std::vector<PhysicalMemoryResource> resources;
                for (const auto device : sample.request.devices())
                    resources.push_back(planningObservedResource(observed, device));
                PlanningLocalTPMeasurement::contributeMemory(sample.request, host, resources, builder);
                alternatives.push_back(builder.build());
            }
            for (const auto &sample : plan.hostDevice())
            {
                if (sample.rank() != rank) continue;
                PhysicalMemoryPlanBuilder builder;
                PlanningHostDeviceMeasurement::contributeMemory(sample, host,
                    planningObservedResource(observed, sample.device()), builder);
                alternatives.push_back(builder.build());
            }
            PhysicalMemoryPlanBuilder packets;
            packets.add(host, PhysicalMemoryOwner::ActivationTransportStaging,
                PlanningMPITransferMeasurement::workspaceBytes(plan.mpi(), rank, inventory.world_size));
            alternatives.push_back(packets.build());
            PhysicalMemoryPlanBuilder aggregate;
            aggregate.addMutuallyExclusive(alternatives);
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(aggregate.build()), rank);
        }
    }

    PlanningCommunicationSamplePlan PlanningCommunicationSamplePlan::resolve(const ClusterInventory &inventory,
        const AutomaticOrchestrationRequest &request, int width, int rows,
        std::span<const PlanningAllreducePrecision> precisions)
    {
        require(width > 0 && rows > 0 && size_t(width) <= size_t(INT_MAX) / sizeof(float) / size_t(rows),
            "activation payload must be positive and MPI-representable");
        std::set<PlanningAllreducePrecision> unique;
        for (const auto precision : precisions)
            require((precision == PlanningAllreducePrecision::FP32 || precision == PlanningAllreducePrecision::FP16) &&
                unique.insert(precision).second, "duplicate or unsupported native precision");
        require(!unique.empty(), "missing native precision policy");
        // Reuse the canonical alias/physical membership checks without fresh
        // discovery. A CPU staging process is not itself an eligible endpoint.
        (void)PlanningKernelServiceCatalog::observers(inventory, request);
        PlanningCommunicationSamplePlan result;
        for (const auto &rank : inventory.ranks) result.rank_nodes_.push_back(rank.node_id);
        if (!request.allows(OrchestrationStrategy::TensorParallel) &&
            !request.allows(OrchestrationStrategy::PipelineParallel) &&
            !request.allows(OrchestrationStrategy::ExpertOverlay)) return result;

        using Key = std::tuple<int, DeviceType, std::vector<std::string>>;
        /** @brief A common visible group and deterministic affinity-preferred process owner. */
        struct Group
        {
            int rank, affinity;
            std::vector<DeviceId> devices;
            std::vector<std::string> uuids;
        };
        std::map<Key, Group> groups;
        std::vector<int> participants;
        for (const auto &rank : inventory.ranks)
        {
            bool compute = request.allows(DeviceType::CPU) && rank.cpu.memory_bytes > 0 && rank.cpu_cores > 0;
            for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            {
                if (!request.allows(backend)) continue;
                std::vector<const DeviceInfo *> devices;
                for (const auto &gpu : rank.gpus) if (gpu.type == backend) devices.push_back(&gpu);
                compute |= !devices.empty();
                if (devices.size() < 2) continue;
                // Match the candidate compiler's ordinal order. Sorting the
                // actual communicator by UUID would measure another ring and
                // miss the production coordinator pool on subsequent serving.
                // Only the deduplication key uses the unordered physical set.
                std::sort(devices.begin(), devices.end(), [](const auto *a, const auto *b) {
                    return a->local_device_id < b->local_device_id;
                });
                Group group{rank.rank, 0, {}, {}};
                for (const auto *gpu : devices)
                {
                    group.uuids.push_back(gpu->uuid);
                    group.devices.emplace_back(backend, gpu->local_device_id);
                    group.affinity += gpu->numa_node >= 0 && gpu->numa_node == rank.cpu.numa_node;
                }
                auto uuids = group.uuids;
                std::sort(uuids.begin(), uuids.end());
                require(std::adjacent_find(uuids.begin(), uuids.end()) == uuids.end(),
                    "one native group cannot contain a physical GPU twice");
                const Key key{rank.node_id, backend, std::move(uuids)};
                const auto previous = groups.find(key);
                if (previous == groups.end()) groups.emplace(key, std::move(group));
                else if (group.affinity > previous->second.affinity) previous->second = std::move(group);
            }
            if (compute) participants.push_back(rank.rank);
        }
        for (const auto &[key, group] : groups)
        {
            const auto &[node, backend, uuids] = key;
            const bool covered = std::any_of(groups.begin(), groups.end(), [&](const auto &other) {
                const auto &[other_node, other_backend, other_uuids] = other.first;
                return node == other_node && backend == other_backend && uuids.size() < other_uuids.size() &&
                    std::includes(other_uuids.begin(), other_uuids.end(), uuids.begin(), uuids.end());
            });
            if (covered) continue;
            for (const auto precision : unique)
                result.native_.push_back({group.rank, node, group.uuids, PlanningLocalTPRequest(group.devices, width, rows, precision)});
        }
        // Each sample is a completed round trip. Changing one payload at a
        // time separates outbound and return service without inventing a
        // symmetric half-RTT or inferring physical locality from timings.
        const size_t payload = size_t(width) * rows * sizeof(float);
        for (const int from : participants)
            for (const int to : participants)
                if (from != to)
                {
                    result.mpi_.push_back({from, to, 1, 1});
                    result.mpi_.push_back({from, to, payload, 1});
                    result.mpi_.push_back({from, to, 1, payload});
                }
        // Native homogeneous TP does not use host-GPU activation staging. PP
        // and overlay searches do. Preserve every observing rank's host scope:
        // the same GPU with pages first-touched on another socket is not the
        // same link sample, even when both ranks see identical GPU ordinals.
        if (request.allows(OrchestrationStrategy::PipelineParallel) || request.allows(OrchestrationStrategy::ExpertOverlay))
            for (const auto &rank : inventory.ranks)
            {
                std::vector<DeviceId> devices;
                for (const auto &gpu : rank.gpus)
                    if (request.allows(gpu.type)) devices.emplace_back(gpu.type, gpu.local_device_id);
                std::sort(devices.begin(), devices.end(), [](auto a, auto b) {
                    return std::tuple{a.type, a.ordinal} < std::tuple{b.type, b.ordinal};
                });
                for (auto device : devices)
                    result.host_device_.push_back(PlanningHostDeviceRequest::fromInventory(rank, device, payload));
            }
        return result;
    }

    void PlanningCommunicationService::validateNative(const PlanningNativeCollectiveService &service)
    {
        validate(service.observation, service.sample);
    }

    std::vector<uint8_t> PlanningCommunicationSamplePlan::serialize() const
    {
        Json wire{{"schema", kSchema}, {"rank_nodes", rank_nodes_}, {"native", Json::array()},
            {"mpi", Json::array()}, {"host_device", Json::array()}};
        for (const auto &sample : native_)
            wire["native"].push_back({{"rank", sample.discovery_rank}, {"node", sample.physical_node},
                {"uuids", sample.uuids}, {"request", identity(sample.request)}});
        for (const auto &sample : mpi_)
            wire["mpi"].push_back({sample.initiator, sample.responder, sample.request_bytes, sample.reply_bytes});
        for (const auto &sample : host_device_) wire["host_device"].push_back(identity(sample));
        return bytes(wire);
    }

    std::vector<uint8_t> PlanningCommunicationService::encodeLocal(const PlanningCommunicationSamplePlan &plan,
        std::span<const std::pair<size_t, PlanningLocalTPObservations>> observations,
        std::span<const std::pair<size_t, PlanningHostDeviceObservations>> host_device)
    {
        Json wire{{"schema", kSchema}, {"plan", Json::parse(plan.serialize())},
            {"observations", Json::array()}, {"host_device", Json::array()}};
        std::set<size_t> seen;
        for (const auto &[index, observed] : observations)
        {
            require(index < plan.native().size() && seen.insert(index).second, "duplicate/unknown native sample");
            validate(observed, plan.native()[index]);
            Json phases = Json::array();
            for (const auto &phase : observed.phases)
            {
                Json endpoints = Json::array();
                for (const auto &endpoint : phase.endpoints)
                    endpoints.push_back({int(endpoint.device.type), endpoint.device.ordinal,
                        endpoint.graph_nodes, endpoint.seconds_per_collective});
                phases.push_back({{"rows", phase.rows}, {"bytes", phase.payload_bytes}, {"endpoints", endpoints}});
            }
            wire["observations"].push_back({{"index", index}, {"phases", phases}});
        }
        seen.clear();
        for (const auto &[index, observed] : host_device)
        {
            require(index < plan.hostDevice().size() && seen.insert(index).second &&
                observed.request == plan.hostDevice()[index], "foreign/duplicate host-device sample");
            PlanningHostDeviceMeasurement::validate(observed);
            Json phases = Json::array();
            for (const auto &phase : observed.phases)
                phases.push_back({int(phase.mechanism), int(phase.direction), phase.payload_bytes, phase.graph_nodes,
                    phase.service.completedWork(), phase.service.elapsedSeconds(), phase.service.provenance()});
            wire["host_device"].push_back({{"index", index}, {"phases", phases}});
        }
        return bytes(wire);
    }

    PlanningLocalCommunicationEvidence PlanningCommunicationService::acceptLocal(
        const PlanningCommunicationSamplePlan &plan, int world_size, std::span<const RankPlanningSample> receipts)
    {
        require(world_size > 0 && receipts.size() == size_t(world_size), "incomplete discovery receipts");
        const auto identity_wire = Json::parse(plan.serialize());
        require(identity_wire.at("rank_nodes").size() == size_t(world_size), "foreign discovery size");
        std::set<int> ranks;
        std::map<size_t, PlanningNativeCollectiveService> completed;
        std::map<size_t, PlanningHostDeviceObservations> completed_host;
        for (const auto &receipt : receipts)
        {
            require(receipt.discovery_rank >= 0 && receipt.discovery_rank < world_size &&
                ranks.insert(receipt.discovery_rank).second, "duplicate/foreign receipt rank");
            const auto wire = Json::parse(receipt.bytes);
            require(wire.is_object() && wire.size() == 4 && wire.at("schema") == kSchema &&
                wire.at("plan") == identity_wire && wire.at("observations").is_array() && wire.at("host_device").is_array(),
                "foreign local communication evidence identity");
            for (const auto &entry : wire.at("observations"))
            {
                require(entry.is_object() && entry.size() == 2 && entry.at("phases").is_array() &&
                    entry.at("phases").size() == 2, "incomplete native observation");
                const size_t index = integer(entry.at("index"));
                require(index < plan.native().size() && !completed.contains(index), "duplicate/foreign native index");
                const auto &sample = plan.native()[index];
                require(sample.discovery_rank == receipt.discovery_rank, "native result from wrong process owner");
                PlanningLocalTPObservations observed{sample.request, {}};
                for (const auto &phase : entry.at("phases"))
                {
                    require(phase.is_object() && phase.size() == 3 && phase.at("endpoints").is_array(), "malformed native phase");
                    PlanningLocalTPPhaseObservation result{int(integer(phase.at("rows"), INT_MAX)), integer(phase.at("bytes")), {}};
                    for (const auto &endpoint : phase.at("endpoints"))
                    {
                        require(endpoint.is_array() && endpoint.size() == 4, "malformed native endpoint");
                        const auto type = static_cast<DeviceType>(integer(endpoint[0], int(DeviceType::Metal)));
                        result.endpoints.push_back({{type, int(integer(endpoint[1], INT_MAX))},
                            integer(endpoint[2]), endpoint[3].get<double>()});
                    }
                    observed.phases.push_back(std::move(result));
                }
                validate(observed, sample);
                completed.emplace(index, PlanningNativeCollectiveService{sample, std::move(observed)});
            }
            for (const auto &entry : wire.at("host_device"))
            {
                require(entry.is_object() && entry.size() == 2 && entry.at("phases").is_array() &&
                    entry.at("phases").size() == 8, "incomplete host-device observation");
                const size_t index = integer(entry.at("index"));
                require(index < plan.hostDevice().size() && !completed_host.contains(index), "duplicate/foreign host-device index");
                const auto &sample = plan.hostDevice()[index];
                require(sample.rank() == receipt.discovery_rank, "host-device result from wrong first-touch process");
                PlanningHostDeviceObservations observed{sample, {}};
                for (const auto &phase : entry.at("phases"))
                {
                    require(phase.is_array() && phase.size() == 7, "malformed host-device phase");
                    observed.phases.push_back({static_cast<PlanningHostTransferMechanism>(integer(phase[0], INT_MAX)),
                        static_cast<MappedTransferDirection>(integer(phase[1], UINT8_MAX)), integer(phase[2]), integer(phase[3]),
                        PlanningServiceObservation(PlanningWorkUnit::Bytes, phase[4].get<double>(),
                            phase[5].get<double>(), phase[6].get<std::string>())});
                }
                PlanningHostDeviceMeasurement::validate(observed);
                completed_host.emplace(index, std::move(observed));
            }
        }
        require(completed.size() == plan.native().size(), "missing native group evidence");
        require(completed_host.size() == plan.hostDevice().size(), "missing host-device evidence");
        PlanningLocalCommunicationEvidence result;
        for (auto &[index, record] : completed) result.native.push_back(std::move(record));
        for (auto &[index, record] : completed_host) result.host_device.push_back(std::move(record));
        return result;
    }

    std::optional<PlanningCommunicationService> PlanningCommunicationService::collect(const std::shared_ptr<IMPIContext> &mpi,
        const ClusterInventory &inventory, const AutomaticOrchestrationRequest &request,
        int width, int rows, std::span<const PlanningAllreducePrecision> precisions)
    {
        std::optional<PlanningCommunicationSamplePlan> plan;
        std::shared_ptr<PhysicalMemoryAuthority> memory;
        const int rank = mpi ? mpi->rank() : 0;
        int node_slot = 0, rounds = 1;
        exchangePlanningSamples(mpi, [&](int count) {
            require(count == inventory.world_size, "discovery communicator size mismatch");
            const auto agreed = PlanningCommunicationSamplePlan::resolve(inventory, request, width, rows, precisions).serialize();
            return std::vector<std::vector<uint8_t>>(count, agreed);
        }, [&](auto envelope) {
            require(!mpi || (mpi->clusterInventory() && mpi->clusterInventory().get() == &inventory),
                "collection must use context-owned inventory");
            plan = PlanningCommunicationSamplePlan::resolve(inventory, request, width, rows, precisions);
            require(plan->serialize() == std::vector<uint8_t>(envelope.begin(), envelope.end()), "sample basis disagrees across ranks");
            memory = admit(inventory, rank, *plan);
            std::map<int, int> slots;
            for (const auto &member : inventory.ranks)
            {
                const int slot = slots[member.node_id]++;
                if (member.rank == rank) node_slot = slot;
                rounds = std::max(rounds, slot + 1);
            }
            return std::vector<uint8_t>{1};
        }, [](auto) {});
        std::vector<std::pair<size_t, PlanningLocalTPObservations>> local;
        std::vector<std::pair<size_t, PlanningHostDeviceObservations>> local_host;
        PlanningLocalCommunicationEvidence completed;
        for (int round = 0; round < rounds; ++round)
            exchangePlanningSamples(mpi, [&](int count) {
                return std::vector<std::vector<uint8_t>>(count, bytes(round));
            }, [&](auto envelope) {
                require(Json::parse(envelope) == round, "native observer round disagreement");
                if (node_slot == round)
                {
                    for (size_t index = 0; index < plan->native().size(); ++index)
                    {
                        const auto &sample = plan->native()[index];
                        if (sample.discovery_rank == rank)
                            local.emplace_back(index, PlanningLocalTPMeasurement::measure(sample.request, memory));
                    }
                    for (size_t index = 0; index < plan->hostDevice().size(); ++index)
                    {
                        const auto &sample = plan->hostDevice()[index];
                        if (sample.rank() == rank)
                            local_host.emplace_back(index, PlanningHostDeviceMeasurement::measure(sample, memory));
                    }
                }
                return encodeLocal(*plan, local, local_host);
            }, [&](auto receipts) {
                if (round + 1 == rounds) completed = acceptLocal(*plan, inventory.world_size, receipts);
            });
        std::vector<PlanningMPITransferObservation> packets;
        if (!plan->mpi().empty())
            packets = PlanningMPITransferMeasurement::measure(mpi, memory, DeviceId::cpu(), [&] { return plan->mpi(); });
        // All resource owners are already retired inside each primitive. This
        // final consensus includes root result construction, not another probe.
        std::optional<PlanningCommunicationService> result;
        acceptPlanningCostPreparation(mpi, [&] {
            if (!mpi || mpi->is_root())
                result = PlanningCommunicationService(std::move(completed.native), std::move(packets), std::move(completed.host_device));
        });
        return result;
    }
}
