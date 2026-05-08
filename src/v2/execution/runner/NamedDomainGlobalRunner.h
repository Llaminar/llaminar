/**
 * @file NamedDomainGlobalRunner.h
 * @brief IOrchestrationRunner for mixed local/cross-rank named-domain PP configs.
 *
 * Created by OrchestrationRunnerFactory when the config contains named domains
 * with PP stages that span multiple MPI ranks (scope=node_local, scope=global,
 * or explicit_ranks with >1 entry).
 *
 * initialize() performs:
 *   1. MPI context acquisition
 *   2. Model metadata read (header only) for layer count
 *   3. ClusterInventory gathering
 *   4. GlobalPPTopology construction via ExecutionPlanBuilder::buildGlobalPPTopology()
 *   5. DomainCommunicatorRegistry initialization for global TP stages
 *   6. Per-rank GlobalPPRankPlan derivation
 *   7. ModelContext loading (full weights)
 *   8. Per-execute-stage StageRunnerEntry construction via StageRunnerFactory
 *   9. GlobalOrchestrator + GlobalOrchestratorRunner construction and delegation
 *
 * Phase 5 scope: config/factory integration.  Phase 6 will add Qwen 3.5 MoE
 * parity migration and expert-mask routing through the domain registry.
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#pragma once

#include "IOrchestrationRunner.h"
#include "../../config/OrchestrationConfig.h"
#include "../mpi_orchestration/IExecutionPlanBuilder.h"
#include "../global/GlobalOrchestratorRunner.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief IOrchestrationRunner for mixed local/cross-rank named-domain PP.
     *
     * Delegates all inference operations to an inner GlobalOrchestratorRunner
     * that is built lazily in initialize().
     */
    class NamedDomainGlobalRunner : public IOrchestrationRunner
    {
    public:
        // ==================================================================
        // Construction
        // ==================================================================

        NamedDomainGlobalRunner(
            OrchestrationConfig config,
            std::unique_ptr<IExecutionPlanBuilder> plan_builder);

        ~NamedDomainGlobalRunner() override;

        // Non-copyable
        NamedDomainGlobalRunner(const NamedDomainGlobalRunner &) = delete;
        NamedDomainGlobalRunner &operator=(const NamedDomainGlobalRunner &) = delete;

        // ==================================================================
        // Detection helper
        // ==================================================================

        /**
         * @brief Return true when this runner should be used for the given config.
         *
         * True when config has named domains with PP stages and any domain is
         * explicitly non-local (scope=node_local, scope=global) or has more
         * than one entry in explicit_ranks.
         *
         * Does not require world_size so it works without a live MPI session.
         */
        static bool shouldUse(const OrchestrationConfig &config);

        // ==================================================================
        // IOrchestrationRunner: Lifecycle
        // ==================================================================

        bool initialize() override;
        void shutdown() override;

        // ==================================================================
        // IOrchestrationRunner: Inference
        // ==================================================================

        bool prefill(const std::vector<int32_t> &tokens) override;
        GenerationResult decodeStep() override;
        GenerationResult generate(
            const std::vector<int32_t> &prompt_tokens,
            int max_new_tokens,
            const SamplingParams &sampling) override;

        // ==================================================================
        // IOrchestrationRunner: Configuration / Status
        // ==================================================================

        const RankExecutionPlan &executionPlan() const override;
        const OrchestrationConfig &config() const override;
        bool isInitialized() const override;
        const std::string &lastError() const override;

        // ==================================================================
        // IOrchestrationRunner: Stats
        // ==================================================================

        int vocabSize() const override;
        int currentPosition() const override;
        void clearCache() override;
        const float *lastLogits() const override;
        void setStopTokens(const std::vector<int32_t> &stop_tokens) override;
        std::shared_ptr<ITokenizer> tokenizer() const override;
        const std::string &architecture() const override;

        // ==================================================================
        // IOrchestrationRunner: Snapshot
        // ==================================================================

        void enableSnapshotCapture(const std::string &output_dir) override;
        void disableSnapshotCapture() override;
        void clearSnapshots() override;
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;
        std::vector<std::string> getSnapshotKeys() const override;

    private:
        OrchestrationConfig config_;
        std::unique_ptr<IExecutionPlanBuilder> plan_builder_;

        // Inner runner — created in initialize()
        std::unique_ptr<GlobalOrchestratorRunner> inner_;

        bool initialized_ = false;
        std::string last_error_;
        RankExecutionPlan empty_plan_;

        bool setError(const std::string &msg);
    };

} // namespace llaminar2
