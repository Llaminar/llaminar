#pragma once

#include <cstddef>
#include <cstdint>

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
    };

} // namespace llaminar2
