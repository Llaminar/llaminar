/**
 * @file NamedDomainGlobalRunner.h
 * @brief IOrchestrationRunner for mixed local/cross-rank named-domain PP configs.
 *
 * Created by OrchestrationRunnerFactory when named PP stages span multiple MPI
 * ranks. Cross-rank ownership may be expressed either inside one node/global
 * domain or by consecutive rank-local stages owned by different ranks.
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
 *   9. GlobalOrchestrator adoption by the common OrchestrationRunner lifecycle
 *
 * The wrapper owns topology/model setup only. Request lifecycle is deliberately
 * not duplicated here: prefix restore, sampling, snapshots, forced decode, MTP,
 * and reset are all delegated to the ordinary production lifecycle owner.
 *
 * @author David Sanftenberg
 * @date May 2026
 */

#pragma once

#include "IOrchestrationRunner.h"
#include "../../config/OrchestrationConfig.h"
#include "../mpi_orchestration/IExecutionPlanBuilder.h"
#include "OrchestrationRunner.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief IOrchestrationRunner for mixed local/cross-rank named-domain PP.
     *
     * Builds the physical GlobalOrchestrator lazily, then adopts it into the
     * ordinary OrchestrationRunner. This preserves one prefix, sampling,
     * forced-token, snapshot, and request-reset lifecycle for every topology.
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
         * True when config has named PP stages and their combined owners span
         * ranks. This includes a non-local domain, a multi-rank explicit list,
         * or distinct rank-local owner values on consecutive stages.
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
        GenerationResult forceDecodeToken(int32_t token) override;
        GenerationResult generate(
            const std::vector<int32_t> &prompt_tokens,
            int max_new_tokens,
            const SamplingParams &sampling) override;
        void setDecodeStepTokenBudget(int max_tokens) override;

        // ==================================================================
        // IOrchestrationRunner: Configuration / Status
        // ==================================================================

        const RankExecutionPlan &executionPlan() const override;
        const OrchestrationConfig &config() const override;
        bool configureMTPRequestPolicy(
            const MTPRequestPolicy &policy) override;
        [[nodiscard]] MTPRequestPolicy mtpRequestPolicy() const override;
        bool isInitialized() const override;
        const std::string &lastError() const override;

        // ==================================================================
        // IOrchestrationRunner: Stats
        // ==================================================================

        int vocabSize() const override;
        int currentPosition() const override;
        void clearCache() override;
        bool purgePrefixCache() override;
        /** @return Nonintrusive request observations from the common owner. */
        RequestRuntimeSummary requestRuntimeSummary() const override;
        PrefixRuntimeStateSnapshot prefixStateProbe() const override;
        DeviceId primaryDeviceId() const override;
        const float *lastLogits() const override;
        void setStopTokens(const std::vector<int32_t> &stop_tokens) override;
        /** @brief Install the request sampling policy on the common lifecycle owner. */
        void setSamplingParams(const SamplingParams &params) override;
        /** @return Model-recommended sampling policy from the common owner. */
        SamplingParams getRecommendedSamplingParams() const override;
        std::shared_ptr<ITokenizer> tokenizer() const override;
        const std::string &architecture() const override;
        /** @brief Return the retained model authority used by every local stage. */
        const IModelContext *modelContextForDiagnostics() const override;

        /** @brief Delegate the coordinated follower loop to the common owner. */
        void runMPIWorkerLoop() override;
        /** @brief Delegate a nonterminal retained-runner session boundary. */
        bool yieldMPIWorkersForRetainedRunner() override;
        /** @brief Delegate terminal coordinated shutdown. */
        void shutdownMPIWorkers() override;
        /** @brief Delegate coordinated-mode admission. */
        void setMPICoordinatedMode(bool enabled) override;
        /** @return Coordinated request authority selected by the inner runner. */
        int coordinatedRootRank() const override;

        // ==================================================================
        // IOrchestrationRunner: Snapshot
        // ==================================================================

        void enableSnapshotCapture(const std::string &output_dir) override;
        /** @brief Retain and publish the exact pre-initialization snapshot filter. */
        void setSnapshotCaptureFilter(
            const std::vector<std::string> &keys) override;
        void disableSnapshotCapture() override;
        void clearSnapshots() override;
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;
        std::vector<std::string> getSnapshotKeys() const override;

    private:
        OrchestrationConfig config_;
        std::unique_ptr<IExecutionPlanBuilder> plan_builder_;

        // One common request lifecycle around the cross-rank physical runner.
        std::unique_ptr<OrchestrationRunner> inner_;

        /** Tokenizer/model identity retained from the rank-local model authority. */
        std::shared_ptr<ITokenizer> tokenizer_;
        std::shared_ptr<ModelContext> model_context_;
        std::string architecture_name_;

        /** Pre-initialization policies accepted by the public runner surface. */
        bool snapshot_capture_enabled_ = false;
        std::string snapshot_output_dir_;
        std::vector<std::string> snapshot_capture_filter_;
        SamplingParams active_sampling_params_;
        std::vector<int32_t> stop_tokens_;

        bool initialized_ = false;
        std::string last_error_;
        RankExecutionPlan empty_plan_;

        bool setError(const std::string &msg);
    };

} // namespace llaminar2
