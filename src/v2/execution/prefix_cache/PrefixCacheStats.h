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
        uint64_t disk_hydrations = 0;
        uint64_t terminal_state_hits = 0;
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
    };

    struct MTPRequestSummary
    {
        bool enabled = false;
        bool bypassed = false;
        std::string bypass_reason;
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
    };

} // namespace llaminar2
