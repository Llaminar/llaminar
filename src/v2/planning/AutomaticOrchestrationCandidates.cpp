/**
 * @file AutomaticOrchestrationCandidates.cpp
 * @brief Observed endpoint pools expressed through existing domain/role types.
 *
 * Physical endpoints and discovery ranks travel together until one proposal
 * selects its execution membership. Domains are then rebased once; CPU NUMA
 * IDs and GPU ordinals never undergo rank arithmetic. The model compiler, not
 * this enumerator, decides whether each proposed graph can execute. Memory and
 * performance admission remain separate and cannot short-circuit on first fit.
 */
#include "planning/AutomaticOrchestrationCandidates.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/RankHardwareOwnership.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief One observed endpoint and its owning discovery process. */
        struct Endpoint
        {
            int rank;
            int node;
            GlobalDeviceAddress address;
            std::string physical_identity;
        };
        using Pool = std::vector<Endpoint>;

        /** @return Whether two domains would execute on any same physical endpoint. */
        bool overlaps(const Pool &left, const Pool &right)
        {
            return std::any_of(left.begin(), left.end(), [&](const auto &a) {
                return std::any_of(right.begin(), right.end(), [&](const auto &b) {
                    return a.node == b.node && a.physical_identity == b.physical_identity;
                });
            });
        }

        /**
         * @brief Create the exact selected namespace, preserving first domain authority.
         * @param pools Domain participants in semantic role order.
         * @param inventory Discovery namespace to project without rediscovery.
         * @return Membership whose rank zero owns the first domain's first endpoint.
         */
        ExecutionRankMembership membershipFor(const std::vector<Pool> &pools,
                                              const ClusterInventory &inventory)
        {
            std::vector<int> ranks;
            for (const auto &pool : pools)
                for (const auto &endpoint : pool)
                    if (std::find(ranks.begin(), ranks.end(), endpoint.rank) == ranks.end())
                        ranks.push_back(endpoint.rank);
            return ExecutionRankMembership(inventory, std::move(ranks));
        }

        /**
         * @brief Express one pool using the existing canonical domain contract.
         * @param pool Discovery-bound endpoints; never empty.
         * @param membership Exact selected namespace for the whole proposal.
         * @param name Role-local name without vendor or priority semantics.
         * @return Explicit physical scope and participant ownership, no collective guess.
         */
        ExecutionDomainDefinition domainFor(const Pool &pool,
            const ExecutionRankMembership &membership, std::string name)
        {
            ExecutionDomainDefinition domain;
            domain.name = std::move(name);
            std::set<int> ranks, nodes;
            for (const auto &endpoint : pool)
            {
                const int rank = membership.executionRank(endpoint.rank).value();
                ranks.insert(rank);
                nodes.insert(endpoint.node);
                domain.participants.push_back(endpoint.address);
                domain.ranks.push_back(rank);
            }
            domain.owner_rank = domain.ranks.front();
            domain.scope = pool.size() == 1 ? ExecutionDomainScope::SINGLE :
                ranks.size() == 1 ? ExecutionDomainScope::RANK_LOCAL :
                nodes.size() == 1 ? ExecutionDomainScope::NODE_LOCAL : ExecutionDomainScope::GLOBAL;
            // Rank-local domains have one owner, not an independent per-device
            // communicator. Repeated cross-rank owners remain participant-indexed.
            if (domain.scope == ExecutionDomainScope::RANK_LOCAL) domain.ranks.clear();
            return domain;
        }

        /** @return Private apply-only copy; all serving/numerical policies survive. */
        OrchestrationConfig explicitRequest(const OrchestrationConfig &request)
        {
            auto result = request;
            result.planning_mode = OrchestrationPlanningMode::Apply;
            result.automatic_planning = {};
            result.config_file_path.clear();
            return result;
        }

        /**
         * @brief Append a pool once, deduplicating topology rather than memory totals.
         * @param pools Retained building blocks, not compiled candidate products.
         * @param pool Endpoints in canonical order.
         */
        void addPool(std::vector<Pool> &pools, Pool pool)
        {
            if (pool.empty()) return;
            const auto same = [&](const Pool &other) {
                return other.size() == pool.size() && std::equal(other.begin(), other.end(), pool.begin(),
                    [](const auto &a, const auto &b) {
                        return a.rank == b.rank && a.node == b.node && a.physical_identity == b.physical_identity;
                    });
            };
            if (std::none_of(pools.begin(), pools.end(), same)) pools.push_back(std::move(pool));
        }

        /**
         * @brief Enumerate GPU subsets without a machine-specific cardinality cap.
         * @param endpoints One rank's observed same-backend devices, sorted by ordinal.
         * @param offset Next endpoint considered for inclusion.
         * @param current Working subset, restored before returning.
         * @param pools Destination building blocks; no inference resources are created.
         */
        void gpuSubsets(const Pool &endpoints, size_t offset, Pool &current, std::vector<Pool> &pools)
        {
            for (size_t index = offset; index < endpoints.size(); ++index)
            {
                current.push_back(endpoints[index]);
                addPool(pools, current);
                gpuSubsets(endpoints, index + 1, current, pools);
                current.pop_back();
            }
        }

        /**
         * @brief Build observed CPU pools and homogeneous rank-local GPU subsets.
         * @param inventory Authenticated discovery records.
         * @param policy Hard backend constraints; no performance hints.
         * @return Pools retaining physical identity and exact discovery ownership.
         * @throws std::invalid_argument for missing/duplicate physical endpoint identity.
         */
        std::vector<Pool> endpointPools(const ClusterInventory &inventory,
                                      const AutomaticOrchestrationRequest &policy)
        {
            std::vector<Pool> pools;
            const bool needs_domains = policy.allows(OrchestrationStrategy::TensorParallel) ||
                policy.allows(OrchestrationStrategy::PipelineParallel) || policy.allows(OrchestrationStrategy::ExpertOverlay);
            std::map<int, Pool> node_cpus;
            std::set<std::pair<int, int>> cpu_nodes;
            for (const auto &rank : inventory.ranks)
            {
                if (rank.hostname.empty()) throw std::invalid_argument("Automatic planning requires observed hostnames");
                if (policy.allows(DeviceType::CPU))
                {
                    std::set<int> numa_nodes;
                    if (rank.cpu.numa_node >= 0) numa_nodes.insert(rank.cpu.numa_node);
                    else for (const auto &socket : rank.cpu_socket_info)
                        if (rankOwnsCPUNode(rank, socket.numa_node)) numa_nodes.insert(socket.numa_node);
                    if (numa_nodes.empty()) throw std::invalid_argument("Automatic planning requires observed CPU NUMA ownership");
                    for (const int numa : numa_nodes)
                    {
                        Endpoint cpu{rank.rank, rank.node_id,
                            GlobalDeviceAddress::fromLocalDeviceId(DeviceId(DeviceType::CPU, numa), rank.hostname, numa),
                            "cpu:" + std::to_string(numa)};
                        addPool(pools, {cpu});
                        // Repeated rank visibility is an alternate process owner,
                        // not another socket or another RAM capacity contribution.
                        if (cpu_nodes.emplace(rank.node_id, numa).second)
                            node_cpus[rank.node_id].push_back(std::move(cpu));
                    }
                }
                for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
                {
                    if (!policy.allows(backend)) continue;
                    Pool devices;
                    std::set<int> ordinals;
                    std::set<std::string> uuids;
                    for (const auto &gpu : rank.gpus)
                    {
                        if (gpu.type != backend) continue;
                        if (gpu.local_device_id < 0 || gpu.uuid.empty() ||
                            !ordinals.insert(gpu.local_device_id).second || !uuids.insert(gpu.uuid).second)
                            throw std::invalid_argument("Automatic planning requires unique observed GPU ordinals and UUIDs");
                        devices.push_back({rank.rank, rank.node_id,
                            GlobalDeviceAddress::fromLocalDeviceId(DeviceId(backend, gpu.local_device_id),
                                rank.hostname, gpu.numa_node),
                            std::string(deviceTypeToString(backend)) + ":" + gpu.uuid});
                    }
                    std::sort(devices.begin(), devices.end(), [](const auto &a, const auto &b) {
                        return a.address.device_ordinal < b.address.device_ordinal;
                    });
                    Pool current;
                    if (needs_domains) gpuSubsets(devices, 0, current, pools);
                    else for (const auto &device : devices) addPool(pools, {device});
                }
            }
            Pool cluster_cpus;
            for (const auto &[node, cpus] : node_cpus)
            {
                (void)node;
                addPool(pools, cpus);
                cluster_cpus.insert(cluster_cpus.end(), cpus.begin(), cpus.end());
            }
            addPool(pools, std::move(cluster_cpus));
            return pools;
        }

        /**
         * @brief Build explicit MoE domains without choosing a quota or physical reserve.
         * @param request Policy to retain, including Static/Dynamic and MTP capacity.
         * @param pools Continuation followed by optional expert-only domains; one pool expresses MoE TP.
         * @param membership Selected rank namespace.
         * @return Existing normalized overlay config, ready for model-aware compilation.
         */
        OrchestrationConfig overlayConfig(const OrchestrationConfig &request,
            const std::vector<Pool> &pools, const ExecutionRankMembership &membership)
        {
            auto config = explicitRequest(request);
            auto overlay = std::make_shared<MoERoutedExpertPlacementPlan>();
            overlay->enabled = true;
            overlay->owner_order = request.routed_expert_owner_order;
            overlay->residency_policy = request.moe_rebalance.mode == MoERebalanceRuntimeMode::Off
                ? RoutedExpertResidencyPolicy::StaticById : RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            overlay->continuation_dense_policy_intent = MoEContinuationDensePolicyIntent::Automatic;
            if (pools.size() == 1)
            {
                // A one-domain TP proposal is still TP, including across
                // ranks. Only expert ownership is normalized to the universal
                // authority; do not turn its dense trunk into replicated work.
                overlay->continuation_dense_policy_intent = MoEContinuationDensePolicyIntent::Explicit;
                overlay->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
            }
            for (size_t index = 0; index < pools.size(); ++index)
            {
                const auto name = "domain_" + std::to_string(index);
                overlay->domains.push_back(RoutedExpertDomain::fromExecutionDomainDefinition(
                    domainFor(pools[index], membership, name)));
                overlay->routed_tiers.push_back({.name = "tier_" + std::to_string(index),
                    .domain = name, .priority = static_cast<int>(index)});
            }
            config.moe_routed_expert_plan = std::move(overlay);
            // Roles, dense TP defaults and final-coverage selection belong to
            // shared normalization, just as they do for compact CLI tiers.
            const auto errors = normalizeMoERoutedExpertPlacementDomains(config);
            if (!errors.empty())
            {
                std::string message = "Automatic overlay construction failed";
                for (const auto &error : errors) message += "\n  - " + error;
                throw std::invalid_argument(message);
            }
            return config;
        }
    }

    void visitAutomaticOrchestrationCandidates(const OrchestrationConfig &request,
        const PlanningModelMetadata &model, const ClusterInventory &inventory,
        const std::function<void(AutomaticOrchestrationCandidate)> &visit)
    {
        const auto intent = resolveOrchestrationIntent(request);
        const auto *policy = std::get_if<AutomaticOrchestrationRequest>(&intent);
        if (!policy || !visit) throw std::invalid_argument("Candidate enumeration requires automatic intent and a visitor");
        // Reuse the canonical namespace validator before touching any records.
        std::vector<int> all_ranks(inventory.ranks.size());
        std::iota(all_ranks.begin(), all_ranks.end(), 0);
        const ExecutionRankMembership validated(inventory, std::move(all_ranks));
        (void)validated;
        // Membership contains only ranks owning the proposal's compute pools,
        // not idle discovery/control ranks. Compare physical node identities,
        // never hostfile aliases or process counts: one host can own many ranks.
        // Original node IDs may be sparse. Their indexed summary extent is
        // not the number of actual hosts; compare the observed identity sets.
        std::set<int> required_nodes;
        for (const auto &rank : inventory.ranks) required_nodes.insert(rank.node_id);
        const auto publish = [&](AutomaticOrchestrationCandidate proposal) {
            if (policy->options().host_participation == AutomaticHostParticipation::AllDiscovered)
            {
                std::set<int> nodes;
                for (const int rank : proposal.membership.discoveryRanks())
                    nodes.insert(inventory.ranks.at(rank).node_id);
                if (nodes != required_nodes) return;
            }
            visit(std::move(proposal));
        };
        const auto pools = endpointPools(inventory, *policy);
        for (const auto &pool : pools)
        {
            const auto strategy = pool.size() == 1 ? OrchestrationStrategy::SingleDevice : OrchestrationStrategy::TensorParallel;
            if (!policy->allows(strategy)) continue;
            auto membership = membershipFor({pool}, inventory);
            auto config = explicitRequest(request);
            if (pool.size() == 1)
            {
                config.device_for_this_rank = pool.front().address;
                // This is an observed endpoint, not the user's unbound CPU
                // shorthand. Preserve that binding after its discovery rank
                // becomes rank zero of a single-process execution context.
                config.device_for_this_rank_numa_explicit =
                    pool.front().address.isCPU() && pool.front().address.hasValidNuma();
            }
            else
            {
                const auto domain = domainFor(pool, membership, "compute");
                if (domain.scope == ExecutionDomainScope::RANK_LOCAL)
                {
                    // A whole-model local TP proposal is not a one-stage
                    // pipeline. The shared compiler must be free to install
                    // the sole ExpertOverlay authority for a MoE model.
                    config.tp_devices = domain.participants;
                }
                else if (model.memoryProfile().expert_count > 0)
                {
                    // The legacy one-stage PP spelling loses the cross-rank
                    // MoE owner map. Emit the same explicit single-domain
                    // authority consumed by authored topology configurations.
                    config = overlayConfig(request, {pool}, membership);
                }
                else
                {
                    config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain));
                    config.pp_stage_definitions.push_back({0, domain.name, 0, model.mainLayerCount() - 1});
                }
            }
            publish({strategy, std::move(membership), std::move(config)});
        }
        const bool pipelines = policy->allows(OrchestrationStrategy::PipelineParallel);
        const bool overlays = model.memoryProfile().expert_count > 0 && policy->allows(OrchestrationStrategy::ExpertOverlay);
        if (!pipelines && !overlays) return;
        using PoolKey = std::vector<std::pair<int, std::string>>;
        std::set<std::vector<PoolKey>> emitted_overlays;
        for (const auto &first : pools)
            for (const auto &second : pools)
            {
                if (overlaps(first, second)) continue;
                const std::vector<Pool> domains{first, second};
                if (pipelines)
                {
                    // Every legal split competes on complete BOM/cost evidence.
                    // Equal layer counts are not a proxy for equal physical bytes.
                    for (int split = 1; split < model.mainLayerCount(); ++split)
                    {
                        auto membership = membershipFor(domains, inventory);
                        auto config = explicitRequest(request);
                        for (size_t index = 0; index < domains.size(); ++index)
                        {
                            const auto domain = domainFor(domains[index], membership, "stage_" + std::to_string(index));
                            config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain));
                            config.pp_stage_definitions.push_back({static_cast<int>(index), domain.name,
                                index == 0 ? 0 : split, index == 0 ? split - 1 : model.mainLayerCount() - 1});
                        }
                        publish({OrchestrationStrategy::PipelineParallel, std::move(membership), std::move(config)});
                    }
                }
                if (!overlays ||
                    first.front().address.device_type == DeviceType::CPU) continue;
                auto overlay_domains = domains;
                if (second.front().address.device_type == DeviceType::CPU)
                {
                    auto &cpu_pool = overlay_domains[1];
                    const int continuation_node = first.front().node;
                    if (std::any_of(cpu_pool.begin(), cpu_pool.end(), [&](const auto &cpu) { return cpu.node != continuation_node; }))
                        std::erase_if(cpu_pool, [&](const auto &cpu) { return cpu.node == continuation_node; });
                }
                // Removing the continuation node from a cluster CPU pool can
                // produce a pool already visited as a node-only CPU choice.
                // Publish that same topology once, not twice with different
                // enumeration provenance or duplicated admission/probe work.
                std::vector<PoolKey> key;
                for (const auto &pool : overlay_domains)
                {
                    PoolKey participants;
                    for (const auto &endpoint : pool)
                        participants.emplace_back(endpoint.rank, endpoint.physical_identity);
                    key.push_back(std::move(participants));
                }
                if (!emitted_overlays.insert(std::move(key)).second) continue;
                auto membership = membershipFor(overlay_domains, inventory);
                auto config = overlayConfig(request, overlay_domains, membership);
                publish({OrchestrationStrategy::ExpertOverlay, std::move(membership), std::move(config)});
            }
    }
}
