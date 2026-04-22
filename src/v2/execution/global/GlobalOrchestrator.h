/**
 * @file GlobalOrchestrator.h
 * @brief Cross-machine MPI cluster inference orchestrator
 *
 * Top tier of the three-tier orchestration stack:
 *   GlobalOrchestrator       (cross-rank: MPI PP + Global TP)
 *     └─ RankOrchestrator    (per-rank: local devices)
 *          └─ DeviceGraphOrchestrator  (per-device: graph execution)
 *
 * Implements IInferenceRunner, so it is transparent to callers
 * (ChatCompletionHandler, BenchmarkMode, etc.). One instance per MPI rank;
 * each rank consults its own GlobalPPRankPlan.
 *
 * Phase 1 scope:
 * - Pure global-TP (all ranks, all layers) — pass-through to RankOrchestrator
 * - Pure global-PP (disjoint layer ranges) — MPI send/recv of activations
 * - Tail-rank sampling with MPI_Bcast of token
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../local_execution/orchestrators/IRankOrchestrator.h"
#include "../global_pp/GlobalPPTopology.h"
#include "../global_pp/GlobalPPRankPlan.h"
#include "../../interfaces/IMPIContext.h"
#include "../../collective/ITPContext.h"
#include "../../utils/Sampler.h"

#include <memory>
#include <string>

namespace llaminar2
{

    class TensorBase;

    /**
     * @brief Cross-machine MPI cluster inference orchestrator
     *
     * Coordinates global pipeline parallelism (PP) and/or global tensor
     * parallelism (TP) across MPI ranks, delegating per-rank execution to
     * an IRankOrchestrator (or IInferenceRunner for single-device stages).
     *
     * Execution model:
     * - Each rank owns one GlobalOrchestrator instance
     * - Each rank independently executes its GlobalPPRankPlan
     * - Stages alternate: EXECUTE_STAGE → TRANSFER → EXECUTE_STAGE → ...
     * - Only the pipeline tail rank produces valid logits
     * - The tail rank samples and broadcasts the token to all ranks
     */
    class GlobalOrchestrator : public IInferenceRunner
    {
    public:
        // =================================================================
        // Configuration
        // =================================================================

        struct Config
        {
            // Cluster topology
            GlobalPPTopology topology;       ///< Cluster-wide stage layout
            int rank = 0;                    ///< This MPI rank
            int world_size = 1;              ///< Total MPI ranks

            // MPI context (not owned)
            IMPIContext *mpi_ctx = nullptr;

            // Global TP context (optional, not owned)
            ITPContext *global_tp_ctx = nullptr;

            // Per-rank local runner (ownership transferred)
            std::unique_ptr<IInferenceRunner> rank_runner;

            // Model metadata
            int vocab_size = 0;              ///< Full vocabulary size
            int d_model = 0;                 ///< Hidden state dimension
            std::string architecture_name = "unknown";
        };

        // =================================================================
        // Construction / Destruction
        // =================================================================

        /**
         * @brief Construct from configuration
         *
         * Builds the rank's execution plan from the topology, allocates
         * activation transfer buffers (for PP), and takes ownership of
         * the rank runner.
         *
         * @param config Configuration (rank_runner ownership transferred)
         * @throws std::invalid_argument if config is invalid
         */
        explicit GlobalOrchestrator(Config config);

        ~GlobalOrchestrator() override = default;

        // Non-copyable, non-movable (owns unique_ptrs)
        GlobalOrchestrator(const GlobalOrchestrator &) = delete;
        GlobalOrchestrator &operator=(const GlobalOrchestrator &) = delete;
        GlobalOrchestrator(GlobalOrchestrator &&) = delete;
        GlobalOrchestrator &operator=(GlobalOrchestrator &&) = delete;

        // =================================================================
        // IInferenceRunner — Core Inference API
        // =================================================================

        bool forward(const int *tokens, int seq_len) override;
        const float *logits() const override;
        int vocab_size() const override;
        void clear_cache() override;
        int get_position() const override;
        ExecutionPath executionPath() const override;
        const char *architecture() const override;

        // =================================================================
        // IInferenceRunner — GPU-side Sampling
        // =================================================================

        /**
         * @brief Greedy sampling with cross-rank broadcast
         *
         * On the tail rank: delegates to rank_runner_->sampleGreedyOnDevice().
         * On all ranks: MPI_Bcast the sampled token from tail to all others.
         *
         * @return Token ID on all ranks
         */
        int sampleGreedyOnDevice() override;

        /**
         * @brief Full sampling with cross-rank broadcast
         *
         * On the tail rank: delegates to rank_runner_->sampleOnDevice(params).
         * On all ranks: MPI_Bcast the sampled token from tail to all others.
         *
         * @return Token ID on all ranks
         */
        int sampleOnDevice(const SamplingParams &params) override;

        // =================================================================
        // IInferenceRunner — Logits Gather Control
        // =================================================================

        void setSkipLogitsGatherDecode(bool skip) override;
        void setSkipLogitsGatherPrefill(bool skip) override;

        // =================================================================
        // IInferenceRunner — Timeline/Profiling
        // =================================================================

        void setSuppressTimeline(bool suppress) override;
        void setAccumulatePrefill(bool accumulate) override;
        void flushStageTimeline() override;

        // =================================================================
        // IInferenceRunner — Hidden State API (Pipeline Parallelism)
        // =================================================================

        TensorBase *getHiddenState() override;
        const TensorBase *getHiddenState() const override;
        void setHiddenState(TensorBase *hidden_state) override;
        bool hasHiddenStateInput() const override;
        void clearHiddenStateInput() override;

        // =================================================================
        // IInferenceRunner — Snapshot API (delegate to rank runner)
        // =================================================================

        void enableSnapshotCapture(const std::string &output_dir) override;
        void disableSnapshotCapture() override;
        void clearSnapshots() override;
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;
        SnapshotInfo getSnapshotWithShape(const std::string &key) const override;
        std::vector<std::string> getSnapshotKeys() const override;

        // =================================================================
        // IInferenceRunner — Device & Logits Local
        // =================================================================

        DeviceId primaryDeviceId() const override;
        bool hasLogitsLocal() const override;
        LogitsLocalInfo getLogitsLocalInfo() const override;

        // =================================================================
        // Query API
        // =================================================================

        /** @brief True if this rank runs the embedding layer */
        bool isPipelineHead() const;

        /** @brief True if this rank runs the LM head */
        bool isPipelineTail() const;

        /** @brief Number of PP stages in the topology */
        int pipelineDepth() const;

        /** @brief This rank's execution plan */
        const GlobalPPRankPlan &rankPlan() const;

        /** @brief The cluster topology */
        const GlobalPPTopology &topology() const;

    private:
        // =================================================================
        // Internal Execution
        // =================================================================

        /**
         * @brief Execute a single EXECUTE_STAGE step
         *
         * Delegates to the rank runner's forward(), passing tokens only
         * for the pipeline head stage; other stages use hidden state input.
         */
        bool executeStage(const RankStageAction &action,
                          const int *tokens, int seq_len);

        /**
         * @brief Execute a single TRANSFER step (MPI Send or Recv)
         *
         * Uses synchronous MPI_Send / MPI_Recv for activation transfer.
         */
        bool executeTransfer(const RankTransferAction &action);

        /**
         * @brief Find the tail rank (the rank that has the LM head)
         */
        int findTailRank() const;

        // =================================================================
        // State
        // =================================================================

        Config config_;
        GlobalPPRankPlan rank_plan_;

        // Per-rank local execution (owned)
        std::unique_ptr<IInferenceRunner> rank_runner_;

        // Activation transfer buffer (for PP send/recv)
        std::shared_ptr<TensorBase> activation_buffer_;

        // Cached queries
        int tail_rank_ = 0;
        bool is_pipeline_head_ = false;
        bool is_pipeline_tail_ = false;

        int last_seq_len_ = 0;   ///< Sequence length from last forward() call (for buffer sizing)

        /// Ensure activation buffer can hold seq_len * d_model elements
        void ensureActivationBufferCapacity(size_t num_elements);
    };

} // namespace llaminar2
