/**
 * @file MoEExpertOverlayExecutionPlan.h
 * @brief Rank-local role contract for MoE expert overlay orchestration.
 *
 * The runtime plan resolves domain/device descriptors. This execution plan
 * answers the orchestration question for one MPI rank: whether it builds a
 * continuation graph shard or serves only auxiliary expert-overlay domains.
 * A single-device/LocalTP continuation has one graph-building command root. A
 * NodeTP continuation builds one symmetric dense shard on every domain
 * rank while retaining exactly one coordinated command and artifact authority.
 */

#pragma once

#include "MoEExpertOverlayRuntimePlan.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    struct ClusterInventory;

    enum class OverlayRankRole
    {
        ContinuationRoot,        ///< Sole command/logit/artifact authority.
        ContinuationParticipant, ///< Non-root dense NodeTP graph shard.
        LocalAcceleratorParticipant, ///< Routed accelerator endpoint on this rank.
        CpuFallbackParticipant,  ///< Routed CPU endpoint on this rank.
        RemoteExpertParticipant, ///< Routed endpoint outside the continuation domain.
        RelayOnly,               ///< Rank owns no graph or expert endpoint.
    };

    const char *toString(OverlayRankRole role);

    /**
     * @brief Exclusive production execution lifecycle assigned to one rank.
     *
     * Topology roles above are composable: a continuation rank may also own
     * local routed experts.  The execution kind is deliberately not
     * composable.  It identifies the one command/graph protocol the rank must
     * enter, so a dense NodeTP peer cannot accidentally be treated as an
     * expert-only retained-graph follower.
     */
    enum class OverlayRankExecutionKind
    {
        ContinuationAuthority, ///< Dense graph plus command/artifact authority.
        ContinuationPeer,      ///< Dense NodeTP shard following normal commands.
        ExpertOnlyFollower,    ///< Retained sparse graph following signed tickets.
        RelayOnly,             ///< No model graph participates on this rank.
    };

    /** @brief Stable diagnostic spelling for a rank execution lifecycle. */
    const char *toString(OverlayRankExecutionKind kind);

    struct OverlayRankPlan
    {
        int world_rank = -1;
        OverlayRankRole role = OverlayRankRole::RelayOnly;
        OverlayRankExecutionKind execution_kind =
            OverlayRankExecutionKind::RelayOnly;
        std::vector<OverlayRankRole> roles;
        std::vector<std::string> owned_domains;
        std::vector<std::string> root_weight_domains;
        std::vector<std::string> shared_expert_weight_domains;
        std::vector<std::string> accelerator_routed_expert_domains;
        std::vector<std::string> cpu_fallback_expert_domains;
        std::vector<std::string> worker_fallback_expert_domains;
        std::vector<DeviceId> local_devices;
        bool loads_tokenizer = false;
        bool loads_worker_tokenizer_state = false;
        bool loads_full_model_metadata = false;
        bool loads_root_weights = false;
        bool loads_shared_expert_weights = false;
        bool loads_accelerator_routed_experts = false;
        bool loads_cpu_fallback_experts = false;
        bool loads_worker_fallback_experts = false;
        bool loads_expert_weights = false;

        bool hasRole(OverlayRankRole role) const;
        bool ownsDomain(const std::string &domain_name) const;
        bool hasLocalDevice(DeviceId device) const;

        /** @brief Whether this rank owns one dense continuation graph shard. */
        bool ownsContinuationGraph() const;

        /** @brief Whether this rank publishes expert-only transaction tickets. */
        bool ownsTransactionAuthority() const;

        /** @brief Whether this rank consumes retained expert-graph tickets. */
        bool usesExpertTransactionFollower() const;
    };

    struct MoEExpertOverlayExecutionPlanResolverOptions
    {
        int current_world_rank = 0;

        /// MPI world size. When <= 0, the resolver infers the minimum world
        /// size needed by explicit owner/rank fields and current_world_rank.
        int world_size = 0;
    };

    struct MoEExpertOverlayExecutionPlan
    {
        int world_size = 1;
        std::string continuation_domain;
        std::string base_model_domain;
        std::string shared_expert_domain;
        int continuation_root_rank = -1;
        std::vector<MoEOverlayRuntimeDomain> domains;
        std::vector<OverlayRankPlan> rank_plans;
        OverlayRankPlan current_rank;

        const OverlayRankPlan &currentRankPlan() const { return current_rank; }
        const OverlayRankPlan *rankPlanFor(int world_rank) const;
        /**
         * @brief Return the complete ordered set of dense-continuation ranks.
         *
         * Prefix/KV state, logits, sampling, and public request admission live
         * only on ranks that own a continuation graph.  Consumers use this
         * set to construct domain-scoped control collectives without pulling
         * expert-only transaction followers into a dense-state protocol.
         *
         * @return Unique ascending MPI world ranks owning continuation graphs.
         */
        [[nodiscard]] std::vector<int> continuationWorldRanks() const;
        bool ownsContinuationGraph() const
        {
            return current_rank.ownsContinuationGraph();
        }
        std::string diagnostics() const;
    };

    MoEExpertOverlayExecutionPlan buildMoEExpertOverlayExecutionPlan(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        int world_size = 0);

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoERoutedExpertPlacementPlan> plan,
        int current_world_rank);

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoERoutedExpertPlacementPlan> plan,
        const MoEExpertOverlayExecutionPlanResolverOptions &options);

    /**
     * @brief Bind rank-agnostic overlay participants to discovered hardware.
     *
     * Explicit `world_ranks` constrain inventory matching. Bound `LOCAL`
     * domains collapse those per-participant matches into one `owner_rank` and
     * leave `world_ranks` empty because all devices belong to one MPI rank.
     * `SINGLE` and `NODE_LOCAL` domains retain their participant-rank bindings;
     * repeated NodeTP ranks mean one process owns multiple physical devices.
     * `AUTO` is resolved after discovery to SINGLE, RANK_LOCAL, or NODE_LOCAL,
     * so moving accelerators between sockets does not require a config edit.
     * Domains
     * with neither ranks nor an owner are matched to the rank inventory that
     * currently exposes each participant: GPU type/ordinal follows actual
     * NUMA-filtered visibility, while an explicitly numbered CPU NUMA node maps
     * to the corresponding node-local MPI rank. This allows accelerators to be
     * moved between sockets without changing model configuration.
     *
     * @param plan Immutable requested placement plan.
     * @param inventory Cluster-wide rank/device inventory gathered at startup.
     * @return A copied plan with deterministic world-rank and owner bindings.
     * @throws std::invalid_argument when requested hardware is absent,
     *         ambiguous after locality tie-breaking, or incompatible with the
     *         domain scope.
     */
    std::shared_ptr<MoERoutedExpertPlacementPlan>
    bindMoEExpertOverlayPlanToClusterInventory(
        const MoERoutedExpertPlacementPlan &plan,
        const ClusterInventory &inventory);

} // namespace llaminar2
