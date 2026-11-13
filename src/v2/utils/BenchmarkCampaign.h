/**
 * @file BenchmarkCampaign.h
 * @brief Ordered request-policy benchmarks over one admitted production runtime.
 *
 * Campaigns contain one immutable sampling cohort and a shared natural-prompt
 * table. Every cell is validated before the first reset/request. Physical
 * admission, model preparation, inference and timing retain their existing
 * owners; this interface only selects an idle request policy and persists the
 * ordinary BenchmarkRunner result after each cell.
 */
#pragma once
#include "BenchmarkRunner.h"
#include "execution/runner/IOrchestrationRunner.h"
#include <filesystem>

namespace llaminar2
{
/** @brief One immutable prompt identity and train/holdout membership. */
struct BenchmarkCampaignPrompt
{
    std::string id;
    std::string partition;
    PreparedBenchmarkInput input;
};
/** @brief One explicit request policy, never a second topology/axis expander. */
struct BenchmarkCampaignCell
{
    std::string id;
    std::size_t prompt_index;
    MTPRequestPolicy policy;
};
/** @brief Fully validated immutable campaign; no caller may mutate it after tokenization. */
class PreparedBenchmarkCampaign final
{
public:
    /** @return Fresh result directory reserved only when execution begins. */
    const std::filesystem::path &outputDirectory() const noexcept { return output_; }
    /** @return Digest of the exact input JSON bytes, independent of pretty printing. */
    const std::string &documentDigest() const noexcept { return digest_; }
    /** @return Owned prompt table; each tokenizer result was produced exactly once. */
    const std::vector<BenchmarkCampaignPrompt> &prompts() const noexcept { return prompts_; }
    /** @return Explicit authored request order. */
    const std::vector<BenchmarkCampaignCell> &cells() const noexcept { return cells_; }
    /** @return Resolved physical runtime and immutable cohort request parameters. */
    const OrchestrationConfig &configuration() const noexcept { return config_; }
private:
    friend PreparedBenchmarkCampaign prepareBenchmarkCampaign(const OrchestrationConfig &, BenchmarkRunner &);
    PreparedBenchmarkCampaign() = default;
    std::filesystem::path output_;
    std::string digest_;
    OrchestrationConfig config_;
    std::vector<BenchmarkCampaignPrompt> prompts_;
    std::vector<BenchmarkCampaignCell> cells_;
};
/** @brief Terminal outcome; completed result files survive any later cell failure. */
struct BenchmarkCampaignOutcome
{
    bool success = false;
    std::size_t completed_cells = 0;
    std::string failure_reason;
};
/**
 * @brief Parse, resolve and validate all inputs without submitting inference.
 * @param config Topology-resolved startup config containing the campaign path.
 * @param benchmark Canonical tokenizer/input preparation owner.
 * @return Immutable ordered campaign, including exact natural token IDs.
 * @throws std::invalid_argument For unknown/duplicate keys, invalid laws/policies,
 *         conflicting prompt identities, output collisions or context overflow.
 */
PreparedBenchmarkCampaign prepareBenchmarkCampaign(const OrchestrationConfig &config, BenchmarkRunner &benchmark);
/**
 * @brief Run admitted request cells through the unchanged production timing path.
 * @param campaign Immutable validated inputs.
 * @param runner Sole initialized production request/reset/policy authority.
 * @param benchmark Existing BenchmarkRunner sharing this runner through its adapter.
 * @return Success only after every cell and the atomic terminal summary are written.
 */
BenchmarkCampaignOutcome runBenchmarkCampaign(const PreparedBenchmarkCampaign &campaign,
    IOrchestrationRunner &runner, BenchmarkRunner &benchmark);
}
