/**
 * @file PrefixCacheStats.h
 * @brief Typed cumulative and request-local prefix-cache observations.
 *
 * Cumulative counters describe cache economy, while request summaries expose
 * the exact admission and completion facts needed by serving diagnostics and
 * parity tests.  Neither structure owns cache policy or mutable inference
 * state; the prefix-cache implementation remains the sole authority.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace llaminar2
{

    struct PrefixCacheStats
    {
        uint64_t lookups = 0;
        uint64_t hits = 0;
        uint64_t partial_hits = 0;
        uint64_t misses = 0;
        uint64_t matched_blocks = 0;
        uint64_t matched_tokens = 0;
        uint64_t stores = 0;
        uint64_t inserts = 0;
        uint64_t evictions = 0;
        uint64_t promotions = 0;
        uint64_t ram_to_disk_demotions = 0;
        uint64_t device_hot_promotions = 0;
        uint64_t device_hot_repromotions = 0;
        uint64_t device_hot_evictions = 0;
        uint64_t disk_evictions = 0;
        uint64_t device_hot_direct_hits = 0;
        uint64_t disk_hydrations = 0;
        uint64_t terminal_state_hits = 0;
        uint64_t fingerprint_rebases = 0;
        uint64_t fingerprint_invalidated_ram_entries = 0;
        uint64_t fingerprint_invalidated_device_entries = 0;
        uint64_t fingerprint_unindexed_disk_entries = 0;
        uint64_t disk_write_failures = 0;
        uint64_t disk_read_failures = 0;
        uint64_t ram_bytes = 0;
        uint64_t device_bytes = 0;
        uint64_t device_hot_bytes = 0;
        uint64_t disk_bytes = 0;
        uint64_t hybrid_state_bytes = 0;
        uint64_t mtp_state_bytes = 0;
        uint64_t bypasses = 0;
        uint64_t unsupported_backend_bypasses = 0;
        uint64_t fingerprint_bypasses = 0;
        uint64_t terminal_state_bypasses = 0;
    };

    struct MTPStats
    {
        uint64_t draft_steps = 0;
        uint64_t accepted_tokens = 0;
        uint64_t rejected_tokens = 0;
        uint64_t rollbacks = 0;
        uint64_t bypasses = 0;
        uint64_t verifier_runs = 0;
        uint64_t verifier_token_count = 0;
        int last_transaction_draft_depth = 0;
        int last_transaction_emitted_token_count = 0;
        uint64_t stochastic_accept_tests = 0;
        uint64_t stochastic_accepts = 0;
        uint64_t stochastic_residual_samples = 0;
        uint64_t stochastic_terminal_samples = 0;
        uint64_t transaction_commits = 0;
        uint64_t transaction_rollbacks = 0;
        uint64_t transaction_validation_failures = 0;
        uint64_t unsafe_verifier_state_rejections = 0;
        uint64_t depth_policy_windows = 0;
        uint64_t depth_policy_updates = 0;
        uint64_t depth_policy_promotions = 0;
        uint64_t depth_policy_demotions = 0;
        uint64_t depth_policy_observe_recommendations = 0;
        int current_depth = 0;
        int min_depth = 0;
        int max_depth = 0;
    };

    struct PrefillChunkStats
    {
        uint64_t schedules = 0;
        uint64_t successful_schedules = 0;
        uint64_t chunks = 0;
        uint64_t real_tokens = 0;
        uint64_t padded_tokens = 0;
        uint64_t failures = 0;
    };

    /**
     * @brief Immutable outcome of the most recent prefix-cache request.
     *
     * ExpertOverlay movement may publish while an already admitted request is
     * executing.  The request remains correct under its RCU lease, but its
     * completed state cannot be archived under a newer placement fingerprint.
     * The two movement epochs make that interval explicit without asking a
     * diagnostic counter to reconstruct lifecycle state after the fact.
     */
    struct PrefixCacheRequestSummary
    {
        bool enabled = false;
        bool bypassed = false;
        std::string bypass_reason;
        bool hit = false;
        bool partial_hit = false;
        int requested_tokens = 0;
        int matched_tokens = 0;
        int matched_blocks = 0;
        bool terminal_logits_restored = false;
        bool terminal_hidden_restored = false;
        bool mtp_state_restored = false;
        bool hybrid_state_restored = false;
        std::string storage_tier = "none";
        /** Placement epoch selected by the coordinated prefix lookup. */
        uint64_t admission_movement_epoch = 0;
        /** Live placement epoch sampled after prefix harvest completed. */
        uint64_t completion_movement_epoch = 0;

        /**
         * @return Whether placement publication crossed this request.
         *
         * A true result is the typed reason that a successful inference may
         * deliberately discard its prefix archive and make the next lookup a
         * cold miss.
         */
        [[nodiscard]] bool crossedMovementEpoch() const noexcept
        {
            return completion_movement_epoch > admission_movement_epoch;
        }

        /**
         * @brief Determine whether movement preceded another request's lookup.
         *
         * Movement that crosses this request can discard its harvest. Movement
         * between this completion and the next admission can invalidate an
         * already archived entry. Movement that begins only after the next
         * request was admitted cannot explain that request's lookup result;
         * its admitted prefix state remains protected by the request lease.
         *
         * @param next Immutable summary of the later request.
         * @return Whether movement could have invalidated this request's
         *         archive before @p next performed its coordinated lookup.
         */
        [[nodiscard]] bool movementPrecededAdmissionOf(
            const PrefixCacheRequestSummary &next) const noexcept
        {
            return crossedMovementEpoch() ||
                   next.admission_movement_epoch >
                       completion_movement_epoch;
        }
    };

    struct MTPRequestSummary
    {
        bool enabled = false;
        bool bypassed = false;
        std::string bypass_reason;
        std::string verify_mode = "greedy";
        bool stochastic_verify = false;
        bool adaptive_depth_enabled = false;
        std::string depth_policy_mode = "fixed";
        int current_depth = 0;
        int min_depth = 0;
        int max_depth = 0;
        uint64_t depth_policy_updates = 0;
        std::string last_depth_policy_reason;
        uint64_t draft_steps = 0;
        uint64_t accepted_tokens = 0;
        uint64_t rejected_tokens = 0;
        uint64_t rollbacks = 0;
        double acceptance_rate = 0.0;
        uint64_t stochastic_accept_tests = 0;
        uint64_t stochastic_accepts = 0;
        uint64_t stochastic_residual_samples = 0;
        uint64_t stochastic_terminal_samples = 0;
        double stochastic_acceptance_rate = 0.0;
    };

} // namespace llaminar2
