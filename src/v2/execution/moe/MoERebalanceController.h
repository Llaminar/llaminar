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
#include "SocketAwareRebalancer.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /// Describes which experts are replicated across sockets.
    /// Both sockets have GEMM engines for replicated experts;
    /// per-token dynamic dispatch decides which socket computes each one.
    struct ExpertReplicaSet
    {
        std::string domain_id;          ///< ExpertParallel domain this replica set belongs to.
        std::vector<bool> is_replicated; ///< [num_experts] true if on both sockets
        std::vector<int> owner_socket;   ///< [num_experts] primary owner socket
        int num_replicated = 0;          ///< Count of replicated experts
        int num_sockets = 0;             ///< Cached socket count (computed once from owner_socket)

        /// Pre-built prefill mask: expert_mask[e] && ownership check baked in.
        /// When non-empty, prefill path uses this single-lookup mask instead of
        /// the multi-branch is_replicated + owner_socket check per expert.
        /// Built once per socket at rebalance time.
        std::vector<bool> prefill_mask; ///< [expert_id] for this socket

        /// Build prefill mask from expert_mask + ownership for a specific socket.
        /// Call after rebalance when masks and owner_socket are finalized.
        void buildPrefillMask(int my_socket_id, const std::vector<bool> &expert_mask);

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
            bool *compute_here) const;
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
            int max_window_size = 4096;                ///< Cap for adaptive growth (0 = no adaptive growth)
            float window_growth_factor = 1.5f;         ///< Multiply window_size by this after each rebalance
            int max_replicas = 0;                      ///< Max experts to replicate per socket (0 = disabled)
            std::vector<DeviceId> sockets;             ///< e.g. {cpu:0, cpu:1}
            std::vector<int> initial_expert_to_socket; ///< [num_experts]
            SocketRebalanceConfig rebalance_config;
        };

        explicit MoERebalanceController(Config config);

        /// Get histogram pointer for MoEExpertComputeStage params (nullptr if OFF)
        DecodeExpertHistogram *histogram() { return histogram_.get(); }

        /// Check if rebalancing should be attempted (window full + mode == DYNAMIC)
        bool shouldRebalance() const;

        /// Check rebalance readiness with an explicit Phase 8 rollout reason.
        MoERebalanceDecision rebalanceDecision() const;

        /// Propose and apply rebalance (swap-based, global placement).
        /// Returns the new expert_to_socket mapping.
        /// Returns empty vector if no rebalancing was done.
        /// The caller is responsible for updating MoEExpertComputeStage local_expert_start/count.
        std::vector<int> rebalance();

        /// Compute optimal per-layer partition using LPT (Longest Processing Time First).
        /// Each layer gets its own independently-optimized expert assignment.
        /// This is more effective than swap-based rebalancing because layer routing
        /// patterns differ — an expert popular in layer 5 may be rare in layer 20.
        void rebalanceLPT();

        /// Get current global expert-to-socket mapping (used by swap-based rebalance)
        const std::vector<int> &currentPlacement() const { return current_placement_; }

        /// Domain/participant vocabulary alias for new ExpertParallel call sites.
        const std::vector<int> &currentParticipantPlacement() const { return current_placement_; }

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

        /// Get max hot expert replicas per socket/rank.
        int maxReplicasPerSocket() const { return config_.max_replicas; }

        /// Get total rebalances performed
        int totalRebalances() const { return total_rebalances_; }

        /// Get total swaps performed across all rebalances
        int totalSwaps() const { return total_swaps_; }

        /// Get the domain placement epoch used by prefix and graph-cache keys.
        uint64_t placementEpoch() const { return placement_epoch_; }

        /// Get duration of last rebalanceLPT() call in milliseconds
        double lastRebalanceDurationMs() const { return last_rebalance_duration_ms_; }

        /// Get number of experts moved in last rebalanceLPT() call
        int lastExpertsMoved() const { return last_experts_moved_; }

        /// Get duration of last applyExpertMasks (VNNI prep) in milliseconds
        double lastPrepDurationMs() const { return last_prep_duration_ms_; }

        /// Record the prep duration from applyExpertMasks (set by DGO)
        void recordPrepDuration(double ms) { last_prep_duration_ms_ = ms; }

        /// Reset the current histogram window after applying a placement update
        /// that does not move base expert ownership, such as additive replicas.
        void resetRebalanceWindow();

        /// Log current histogram summary (for OBSERVE mode)
        void logHistogramSummary() const;

        /// Get a structured profiling summary string for LLAMINAR_PROFILING output.
        /// Includes: histogram stats, rebalance timing, expert movement counts.
        std::string getProfilingSummary() const;

        /// Compute per-layer expert masks for a given socket/rank.
        /// Returns a vector of num_layers expert masks (each size num_experts).
        /// expert_mask[layer][expert] == true means this rank computes that expert.
        /// After rebalanceLPT(), uses per-layer placement. Otherwise uses global placement.
        /// When replicas are active, the mask includes both owned and replicated experts.
        std::vector<std::vector<bool>> computeExpertMasks(int socket_id) const;

        /// Domain/participant vocabulary alias for new ExpertParallel call sites.
        std::vector<std::vector<bool>> computeExpertMasksForParticipant(int participant_id) const
        {
            return computeExpertMasks(participant_id);
        }

        /// Compute expert masks for all sockets with a bounded GPU routed-expert cache.
        /// The hottest experts per layer are placed on GPU sockets up to
        /// gpu_cache_experts_per_layer; all remaining experts are placed on CPU sockets.
        /// If the topology does not contain both GPU and CPU sockets, falls back to
        /// computeExpertMasks() for each socket.
        std::vector<std::vector<std::vector<bool>>> computeGpuCacheExpertMasks(
            int gpu_cache_experts_per_layer) const;

        /// Propose experts to replicate across sockets based on histogram data.
        /// Identifies the top-N hottest experts on each socket and proposes
        /// replicating them on the other socket. max_replicas_per_socket controls
        /// how many experts each socket gets as replicas.
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

        /// Update replica set owner_socket to match current placement.
        /// Must be called after rebalance() if replicas are active,
        /// since rebalance swaps change which socket owns each expert.
        void syncReplicaPlacement()
        {
            if (current_replicas_.num_replicated > 0)
                current_replicas_.owner_socket = current_placement_;
        }

    private:
        MoERebalanceMode requested_mode_ = MoERebalanceMode::OFF;
        Config config_;
        std::unique_ptr<DecodeExpertHistogram> histogram_;
        std::unique_ptr<SocketAwareRebalancer> rebalancer_;
        std::vector<int> current_placement_;                ///< Global placement (swap-based)
        std::vector<std::vector<int>> per_layer_placement_; ///< Per-layer placement (LPT-based)
        bool use_per_layer_placement_ = false;              ///< True after rebalanceLPT()
        int total_rebalances_ = 0;
        int total_swaps_ = 0;
        double last_rebalance_duration_ms_ = 0.0;
        int last_experts_moved_ = 0;
        double last_prep_duration_ms_ = 0.0;
        uint64_t placement_epoch_ = 0;
        float last_avg_imbalance_before_ = 0.0f;
        float last_avg_imbalance_after_ = 0.0f;
        float last_worst_imbalance_before_ = 0.0f;
        int last_worst_layer_before_ = 0;
        int current_window_size_ = 0;       ///< Tracks effective window size for adaptive growth
        ExpertReplicaSet current_replicas_; ///< Active replica set

        void growWindowIfAdaptive();
    };

} // namespace llaminar2
