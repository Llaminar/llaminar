/**
 * @file StartupBanner.h
 * @brief Consolidated startup information display for Llaminar.
 *
 * Renders cluster topology, inference configuration, model info, and
 * preflight check results as a single cohesive block, replacing the
 * previous scattered LOG_INFO lines from many different source files.
 *
 * @author David Sanftenberg
 * @date May 2026
 */
#pragma once

#include <string>
#include <vector>

namespace llaminar2
{
    struct ClusterInventory;
    struct OrchestrationConfig;
    class ModelContext;
    struct RankExecutionPlan;

    /// Preflight check result for display in the startup banner.
    struct PreflightCheckResult
    {
        std::string name; // e.g. "Host RAM (weight staging)"
        bool passed{true};
        std::string detail;     // e.g. "20.7 GB / 725 GB avail"
        std::string mitigation; // non-empty only on failure
    };

    /// All data needed to render the startup banner.
    struct StartupBannerData
    {
        // Phase 1: Cluster topology
        const ClusterInventory *cluster{nullptr};
        int threads_per_rank{0};
        std::string bind_policy; // e.g. "socket", "core"

        // Phase 2: Inference configuration
        std::string device_description; // e.g. "CPU (2 NUMA sockets, GLOBAL TP degree=2)"
        std::string parallelism;        // e.g. "TP=2 (UPI shmem_spin) · PP=1"
        std::string precision;          // e.g. "Activations: FP32 · KV Cache: Q16_1"
        std::string context_length;     // e.g. "4096 / 262144 (model max)"
        std::string backend;            // e.g. "CPU (OneDNN/AVX-512 + VNNI)"

        // Phase 3: Model
        std::string model_filename; // basename only
        std::string model_size;     // e.g. "21.2 GB"
        std::string architecture;   // e.g. "qwen35moe (40 layers, 64 experts, top-4)"
        std::string quantization;   // e.g. "Q4_K (experts) · Q6_K (attention)"
        std::string vocab;          // e.g. "151,936 tokens"
        std::string thinking;       // e.g. "Enabled (<think>...</think>)" or ""

        // Phase 4: Preflight checks
        std::vector<PreflightCheckResult> preflight_checks;
    };

    namespace StartupBanner
    {
        /// Render the full startup banner to a string (for LOG_INFO output).
        /// The string contains embedded ANSI color codes when `use_color` is true.
        std::string render(const StartupBannerData &data, bool use_color);

        /// Detect whether color output should be enabled (TTY + no NO_COLOR).
        bool shouldUseColor();
    }
} // namespace llaminar2
