/**
 * @file MoERebalanceController.h
 * @brief Orchestrates MoE decode histogram tracking and socket-aware rebalancing
 *
 * Lifecycle:
 * 1. Created at graph-build time with model config
 * 2. Histogram pointer passed to MoEExpertComputeStage params for recording
 * 3. After each decode step, caller checks shouldRebalance()
 * 4. If true, caller calls rebalance() which proposes + applies swaps
 * 5. Updated placement is available for next decode step
 */

#pragma once

#include "DecodeExpertHistogram.h"
#include "MoELayeredExpertOwnership.h"
#include "SocketAwareRebalancer.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /// Describes resident hot-expert replicas in a routed-expert domain.
    ///
    /// Base ownership is a complete per-layer table: exactly one participant
    /// owns every `(layer, expert)` pair. Hot replicas are additional resident
    /// copies and are likewise scoped by layer and participant. The aggregate
    /// `replicated_in_any_layer` vector is a derived diagnostic index only; it
    /// never participates in dispatch or ownership decisions.
    struct ExpertReplicaSet
    {
        std::string domain_id; ///< Routed-expert domain this replica set belongs to.
        MoELayeredExpertOwnership base_ownership;
        std::vector<bool> replicated_in_any_layer; ///< Derived `[expert]` diagnostic index.
        int num_replicated = 0; ///< Count of layer/expert/participant replica slots.

        /// [layer][expert][participant] true when participant has a non-owner
        /// resident replica for that layer/expert. Owner residency is implicit
        /// through base_ownership and must not be represented here.
        std::vector<std::vector<std::vector<bool>>> replica_participants_by_layer;

        /// Pre-built prefill mask: expert_mask[e] && ownership check baked in.
        /// When non-empty, prefill path uses this single-lookup mask instead of
        /// repeated replica and owner lookups per expert.
        /// Built once per socket at rebalance time.
        std::vector<bool> prefill_mask; ///< [expert_id] for this socket

        /// Build prefill mask from expert_mask + ownership for a specific socket.
        /// Call after masks and layered base ownership are finalized.
        void buildPrefillMask(
            int my_socket_id,
            const std::vector<bool> &expert_mask,
            int layer_idx = -1);

        bool hasLayerReplicaPlacement() const;
        int participantCount() const { return base_ownership.participantCount(); }
        int ownerParticipant(int layer_idx, int expert_id) const
        {
            return base_ownership.owner(layer_idx, expert_id);
        }
        bool hasReplicaOnParticipant(int layer_idx, int expert_id, int participant_id) const;
        bool isReplicatedForLayer(int layer_idx, int expert_id) const;
        void setReplicaOnParticipant(int layer_idx, int expert_id, int participant_id, bool enabled = true);
        void rebuildAggregateReplicaFlags();

        /// Compare replica placement only. Socket-specific prefill_mask is ignored.
        bool sameReplicaPlacement(const ExpertReplicaSet &other) const;

        /// Return the subset of this replica set that was not already resident
        /// in previous with the same owner socket. Used to avoid re-sending
        /// unchanged hot replicas every rebalance window.
        ExpertReplicaSet arrivalsSince(const ExpertReplicaSet &previous) const;

        /// For a given token's top-k routing, determine which experts this
        /// socket should compute. Deterministic across all ranks given the
        /// same inputs (no communication needed).
        ///
        /// @param expert_indices  The top-k expert IDs for this token
        /// @param expert_weights  The top-k weights (for tie-breaking)
        /// @param top_k           Number of routed experts
        /// @param my_socket_id    This rank's socket index
        /// @param expert_mask     Full mask of experts available on this rank
        /// @param[out] compute_here  Output: compute_here[k]=true if this socket
        ///                           should compute expert_indices[k]
        void assignForToken(
            const int *expert_indices,
            const float *expert_weights,
            int top_k,
            int my_socket_id,
            const std::vector<bool> &expert_mask,
            bool *compute_here,
            int layer_idx = -1) const;
    };

} // namespace llaminar2 (forward decl block)

namespace llaminar2
{

    enum class MoERebalanceMode
    {
        OFF,     ///< No histogram tracking, no rebalancing
        OBSERVE, ///< Track histograms but don't rebalance (for profiling)
        DYNAMIC  ///< Track histograms and dynamically rebalance
    };

    enum class MoERebalanceDecisionReason
    {
        ModeOff,
        DynamicDisabledForDomain,
        SingleParticipantObserveOnly,
        WindowNotFull,
        Ready,
    };

    const char *toString(MoERebalanceDecisionReason reason);

    struct MoERebalanceDecision
    {
        bool ready = false;
        MoERebalanceDecisionReason reason = MoERebalanceDecisionReason::ModeOff;
    };

    class MoERebalanceController
    {
    public:
        struct Config
        {
            std::string domain_id = "single";
            MoERebalanceMode mode = MoERebalanceMode::OFF;
            int num_layers = 0;
            int num_experts = 0;
            int top_k = 0;
            int window_size = 256;
            int token_boundary_layer_idx = -1;       ///< Routed layer that advances decode-token windows
            int max_window_size = 4096;                ///< Cap for adaptive growth (0 = no adaptive growth)
            float window_growth_factor = 1.5f;         ///< Multiply window_size by this after each rebalance
            int max_replicas = 0;                      ///< Max replica slots per participant (0 = disabled)
            std::vector<DeviceId> sockets;             ///< Domain participants, e.g. {cpu:0, cpu:1}
            MoELayeredExpertOwnership initial_ownership; ///< [layer][expert] -> participant
            SocketRebalanceConfig rebalance_config;
        };

        explicit MoERebalanceController(Config config);

        /// Get histogram pointer for MoEExpertComputeStage params (nullptr if OFF)
        DecodeExpertHistogram *histogram() { return histogram_.get(); }

        /// Check if rebalancing should be attempted (window full + mode == DYNAMIC)
        bool shouldRebalance() const;

        /// Check rebalance readiness with an explicit Phase 8 rollout reason.
        MoERebalanceDecision rebalanceDecision() const;

        /**
         * @brief Propose, validate, and install one layered ownership update.
         *
         * The returned entries identify only layer/expert owners that changed.
         * An empty result means no candidate met both the per-layer planner
         * policy and the aggregate non-regression contract.
         */
        std::vector<MoELayeredExpertOwnershipChange> rebalance();

        /// Complete authoritative `(layer, expert) -> participant` ownership.
        const MoELayeredExpertOwnership &currentOwnership() const
        {
            return current_ownership_;
        }

        /// Number of participants in this rebalance domain.
        int participantCount() const { return static_cast<int>(config_.sockets.size()); }

        /// Devices that back the participants in this rebalance domain.
        const std::vector<DeviceId> &participantDevices() const { return config_.sockets; }

        /// Get the rebalance mode
        MoERebalanceMode mode() const { return config_.mode; }

        /// Get the originally requested mode before domain safety downgrades.
        MoERebalanceMode requestedMode() const { return requested_mode_; }

        /// Rebalance domain id this controller owns.
        const std::string &domainId() const { return config_.domain_id; }

        /// Get the number of MoE layers
        int numLayers() const { return config_.num_layers; }

        /// Get total routed experts per MoE layer
        int numExperts() const { return config_.num_experts; }

        /// Get routed experts selected per token
        int topK() const { return config_.top_k; }

        /// Get max hot expert replica slots per participant/rank.
        int maxReplicasPerSocket() const { return config_.max_replicas; }

        /// Get the ownership-swap rebalancer policy configured for this domain.
        const SocketRebalanceConfig &rebalanceConfig() const { return config_.rebalance_config; }

        /// Load-spread stats scored before the most recent rebalance policy decision.
        const ExpertLoadImbalanceStats &lastImbalanceBefore() const { return last_imbalance_before_; }

        /// Load-spread stats scored after the most recent rebalance policy decision.
        const ExpertLoadImbalanceStats &lastImbalanceAfter() const { return last_imbalance_after_; }

        /// Mean normalized spread reduction from the last policy decision.
        double lastAverageSpreadImprovement() const
        {
            if (!last_imbalance_before_.valid || !last_imbalance_after_.valid)
                return 0.0;
            return last_imbalance_before_.average_spread - last_imbalance_after_.average_spread;
        }

        /// Get total rebalances performed
        int totalRebalances() const { return total_rebalances_; }

        /// Get total capacity-preserving ownership swap pairs installed.
        int totalSwapPairs() const { return total_swap_pairs_; }

        /// Get total layer/expert owner entries changed across all updates.
        int totalOwnershipChanges() const { return total_ownership_changes_; }

        /// Get the domain placement epoch used by prefix and graph-cache keys.
        uint64_t placementEpoch() const { return placement_epoch_; }

        /// Get duration of last applyExpertMasks (VNNI prep) in milliseconds
        double lastPrepDurationMs() const { return last_prep_duration_ms_; }

        /// Record the prep duration from applyExpertMasks (set by DGO)
        void recordPrepDuration(double ms) { last_prep_duration_ms_ = ms; }

        /// Reset the current histogram window after applying a placement update
        /// that does not move base expert ownership, such as additive replicas.
        void resetRebalanceWindow();

        /// Log current histogram summary (for OBSERVE mode)
        void logHistogramSummary() const;

        /// Get the legacy human-readable profiling summary.
        /// Includes: histogram stats, rebalance timing, expert movement counts.
        std::string getProfilingSummary() const;

        /// Compute per-layer expert masks for a given participant/rank.
        /// Returns a vector of num_layers expert masks (each size num_experts).
        /// expert_mask[layer][expert] == true means this rank computes that expert.
        /// When replicas are active, the mask includes both owned and replicated experts.
        std::vector<std::vector<bool>> computeExpertMasks(int socket_id) const;

        /// Domain/participant vocabulary alias for routed-expert call sites.
        std::vector<std::vector<bool>> computeExpertMasksForParticipant(int participant_id) const
        {
            return computeExpertMasks(participant_id);
        }

        /// Compute expert masks for all sockets with a bounded GPU routed-expert cache.
        /// The hottest experts per layer are placed on GPU sockets up to
        /// gpu_cache_experts_per_layer; all remaining experts are placed on CPU sockets.
        /// If the bounded heterogeneous-cache policy is not applicable, the
        /// authoritative installed ownership/replica masks are returned unchanged.
        std::vector<std::vector<std::vector<bool>>> computeGpuCacheExpertMasks(
            int gpu_cache_experts_per_layer) const;

        /// Propose layer/expert replica slots based on histogram data.
        /// For each participant, chooses hot layer/expert slots owned by other
        /// participants, up to max_replicas_per_socket slots for that target.
        /// Returns empty set if no replicas are beneficial.
        ExpertReplicaSet proposeReplicas(int max_replicas_per_socket);

        /// Domain/participant vocabulary alias for replica planning.
        ExpertReplicaSet proposeReplicasForParticipants(int max_replicas_per_participant)
        {
            return proposeReplicas(max_replicas_per_participant);
        }

        /// Get the current active replica set (empty if no replicas configured).
        const ExpertReplicaSet &currentReplicas() const { return current_replicas_; }

        /// Whether expert replication is active.
        bool hasReplicas() const { return current_replicas_.num_replicated > 0; }

    private:
        MoERebalanceMode requested_mode_ = MoERebalanceMode::OFF;
        Config config_;
        std::unique_ptr<DecodeExpertHistogram> histogram_;
        std::unique_ptr<SocketAwareRebalancer> rebalancer_;
        MoELayeredExpertOwnership current_ownership_;
        int total_rebalances_ = 0;
        int total_swap_pairs_ = 0;
        int total_ownership_changes_ = 0;
        double last_prep_duration_ms_ = 0.0;
        uint64_t placement_epoch_ = 0;
        ExpertLoadImbalanceStats last_imbalance_before_;
        ExpertLoadImbalanceStats last_imbalance_after_;
        int current_window_size_ = 0;       ///< Tracks effective window size for adaptive growth
        ExpertReplicaSet current_replicas_; ///< Active replica set

        void growWindowIfAdaptive();
    };

    /// Pick the controller that should own graph-side routed expert telemetry
    /// and runtime rebalance decisions when multiple domains are present.
    MoERebalanceController *selectActiveMoERebalanceController(
        const std::vector<MoERebalanceController *> &controllers);

    MoERebalanceController *selectActiveMoERebalanceController(
        const std::vector<std::unique_ptr<MoERebalanceController>> &controllers);

} // namespace llaminar2
