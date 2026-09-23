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
#include "config/GDNHeadAssignment.h"
#include "config/TensorParallelConfig.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/RankHardwareOwnership.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
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

        /**
         * @brief Apply cardinality to disjoint physical pools before compiling candidates.
         * @param policy Validated public automatic filters.
         * @param strategy Complete candidate family, not an individual domain's role.
         * @param pools Disjoint pools; repeated discovery visibility is already deduplicated.
         * @return Whether this membership satisfies every hard compute constraint.
         */
        bool permitsPools(const AutomaticOrchestrationRequest &policy,
                          OrchestrationStrategy strategy, const std::vector<Pool> &pools)
        {
            std::vector<DeviceType> backends;
            for (const auto &pool : pools)
                for (const auto &endpoint : pool)
                    backends.push_back(endpoint.address.device_type);
            return policy.allows(strategy, backends);
        }

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
         * @brief Require a real participant executor between distinct GPU tiers.
         * @param domains Continuation pool followed by the expert-only pool.
         * @return Whether the existing graph lowering can execute both pools.
         *
         * Two different GPU tiers on one MPI rank are not a distributed
         * overlay: the continuation graph would try to lower a foreign GPU's
         * expert work into its own device graph. Visibility of both vendors
         * from one process is not proof of an executable graph boundary.
         * Different ranks provide the participant executor and sparse edge.
         * CPU expert tiers have their own host execution path on the same rank.
         */
        bool hasExecutableOverlayBoundary(const std::vector<Pool> &domains)
        {
            if (domains.size() != 2 || domains[0].empty() || domains[1].empty())
                throw std::invalid_argument("Automatic overlay boundary requires two nonempty domains");
            if (domains[1].front().address.isCPU()) return true;
            std::set<int> continuation_ranks;
            for (const auto &endpoint : domains[0]) continuation_ranks.insert(endpoint.rank);
            return std::any_of(domains[1].begin(), domains[1].end(),
                [&](const Endpoint &endpoint) { return !continuation_ranks.contains(endpoint.rank); });
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
         * @brief Return whether a TP pool preserves the model's GDN ownership relation.
         *
         * Automatic search may discard an optional topology which cannot ever
         * compile without changing model arithmetic.  The test deliberately
         * uses TensorParallelConfig rather than division by pool size because
         * its proportional split is the exact geometry later used by loading
         * and graph construction, including GQA alignment and remainders.
         *
         * @param profile Immutable model geometry.
         * @param pool Proposed homogeneous or CPU tensor-parallel participants.
         * @return @c true when every participant owns an integral GDN key-head
         *         interval, or when the model has no GDN layers.
         * @throws std::invalid_argument for incomplete or invalid GDN geometry.
         */
        bool preservesGDNHeadOwnership(const ModelMemoryProfile &profile, const Pool &pool)
        {
            if (pool.size() <= 1 ||
                (profile.gdn_group_count == 0 && profile.gdn_time_step_rank == 0))
            {
                return true;
            }
            if (profile.gdn_group_count <= 0 || profile.gdn_time_step_rank <= 0)
            {
                throw std::invalid_argument(
                    "Automatic planning requires both GDN key and value head counts");
            }

            const auto sharding = TensorParallelConfig::equalSplit(
                static_cast<int>(pool.size()), profile.n_heads, profile.n_kv_heads,
                profile.d_ff, profile.vocab_size);
            return std::all_of(sharding.assignments().begin(), sharding.assignments().end(),
                [&](const DeviceShardingAssignment &assignment) {
                    return GDNHeadAssignment::hasIntegralKeyHeadBoundaries(
                        profile.gdn_group_count, profile.gdn_time_step_rank,
                        assignment.head_start, assignment.head_count, profile.n_heads);
                });
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
            const std::vector<Pool> &pools, const ExecutionRankMembership &membership,
            std::optional<DenseParallelPolicy> dense_policy = std::nullopt)
        {
            auto config = explicitRequest(request);
            auto overlay = std::make_shared<MoERoutedExpertPlacementPlan>();
            overlay->enabled = true;
            overlay->owner_order = request.routed_expert_owner_order;
            overlay->residency_policy =
                defaultRoutedExpertResidencyPolicy(
                    request.moe_rebalance.mode);
            overlay->continuation_dense_policy_intent = dense_policy
                ? MoEContinuationDensePolicyIntent::Resolved
                : MoEContinuationDensePolicyIntent::Automatic;
            if (dense_policy)
                overlay->continuation_domain_spec.setDensePolicy(*dense_policy);
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
        const auto publishOverlay = [&](OrchestrationStrategy strategy,
                                        const std::vector<Pool> &domains,
                                        ExecutionRankMembership membership) {
            auto default_config = overlayConfig(request, domains, membership);
            const auto default_policy = default_config.moe_routed_expert_plan
                ->continuation_domain_spec.effectiveDensePolicy();
            publish({strategy, membership, std::move(default_config)});
            if (default_policy == DenseParallelPolicy::PrefillTensorParallelDecodeReplicated)
            {
                // Full decode replicas are faster only when their complete
                // physical BOM fits. Admit all-phase TP independently so the
                // planner can choose it when replicas crowd out migration or
                // expert residency; neither policy is a hidden runtime fallback.
                auto tp_config = overlayConfig(request, domains, membership,
                    DenseParallelPolicy::TensorParallel);
                publish({strategy, std::move(membership), std::move(tp_config)});
            }
        };
        const auto pools = endpointPools(inventory, *policy);
        for (const auto &pool : pools)
        {
            const auto strategy = pool.size() == 1 ? OrchestrationStrategy::SingleDevice : OrchestrationStrategy::TensorParallel;
            if (!permitsPools(*policy, strategy, {pool})) continue;
            if (strategy == OrchestrationStrategy::TensorParallel &&
                !preservesGDNHeadOwnership(model.memoryProfile(), pool)) continue;
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
                if (model.memoryProfile().expert_count > 0)
                {
                    /* MoE always enters through its sole ExpertOverlay
                     * authority before admission, including one rank-local
                     * tier. This lets the physical memory authority account
                     * phase-specific dense replicas instead of discovering
                     * them only after a simple-TP candidate was admitted. */
                    publishOverlay(strategy, {pool}, std::move(membership));
                    continue;
                }
                if (domain.scope == ExecutionDomainScope::RANK_LOCAL)
                {
                    // A whole-model local TP proposal is not a one-stage
                    // pipeline. The shared compiler must be free to install
                    // the sole ExpertOverlay authority for a MoE model.
                    config.tp_devices = domain.participants;
                }
                else
                {
                    config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain));
                    config.pp_stage_definitions.push_back({0, domain.name, 0, model.mainLayerCount() - 1});
                }
            }
            publish({strategy, std::move(membership), std::move(config)});
        }
        const bool routed_model = model.memoryProfile().expert_count > 0;
        /*
         * A routed model may only leave automatic discovery with one complete
         * ExpertOverlay authority.  The ordinary PP candidate below describes
         * dense layer ownership but has no routed-expert domains, tiers, or
         * continuation role.  Publishing it would defer that missing topology
         * until rank compilation, where normalization must (correctly) reject
         * an implicit cross-rank/pipeline authority.  Do not advertise that
         * incomplete execution form as an automatic candidate.  Authored MoE
         * PP remains responsible for supplying its explicit overlay topology.
         */
        const bool pipelines =
            !routed_model &&
            policy->allows(OrchestrationStrategy::PipelineParallel);
        const bool overlays =
            routed_model &&
            policy->allows(OrchestrationStrategy::ExpertOverlay);
        if (!pipelines && !overlays) return;
        using PoolKey = std::vector<std::pair<int, std::string>>;
        std::set<std::vector<PoolKey>> emitted_overlays;
        for (const auto &first : pools)
            for (const auto &second : pools)
            {
                if (overlaps(first, second)) continue;
                const std::vector<Pool> domains{first, second};
                if (pipelines && permitsPools(*policy, OrchestrationStrategy::PipelineParallel, domains))
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
                if (!preservesGDNHeadOwnership(model.memoryProfile(), first)) continue;
                auto overlay_domains = domains;
                if (second.front().address.device_type == DeviceType::CPU)
                {
                    auto &cpu_pool = overlay_domains[1];
                    const int continuation_node = first.front().node;
                    if (std::any_of(cpu_pool.begin(), cpu_pool.end(), [&](const auto &cpu) { return cpu.node != continuation_node; }))
                        std::erase_if(cpu_pool, [&](const auto &cpu) { return cpu.node == continuation_node; });
                }
                // Apply counts after remote-only CPU projection. Counting the
                // original pool would admit an idle local socket as remote work.
                if (!permitsPools(*policy, OrchestrationStrategy::ExpertOverlay, overlay_domains)) continue;
                if (!hasExecutableOverlayBoundary(overlay_domains)) continue;
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
                publishOverlay(OrchestrationStrategy::ExpertOverlay,
                    overlay_domains, std::move(membership));
            }
    }
}
