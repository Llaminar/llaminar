/**
 * @file PipelineRunner.h
 * @brief Cross-rank pipeline runner for distributed PP
 *
 * PipelineRunner implements IInferenceRunner for cross-rank pipeline parallelism.
 * It wraps a sequence of stage runners with MPI transfers between them.
 *
 * Only the stage owned by this rank actually computes; other stages
 * are represented as stubs for the transfer protocol.
 *
 * Execution flow for 2-stage pipeline (rank 0 owns stage 0):
 * 1. forward() called on rank 0
 * 2. Execute stage 0 (embedding + first layers)
 * 3. MPI_Send hidden state to rank 1
 * 4. MPI_Recv logits from rank 1 (or wait for completion)
 * 5. Return logits
 *
 * For rank 1:
 * 1. forward() called (with tokens, but we only need for position tracking)
 * 2. MPI_Recv hidden state from rank 0
 * 3. Execute stage 1 (remaining layers + lm_head)
 * 4. MPI_Send logits back to rank 0 (or just continue)
 * 5. Return logits (local)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "TransferSpec.h"
#include <memory>
#include <vector>
#include <string>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class FP32Tensor;

    /**
     * @brief Cross-rank pipeline runner for distributed PP
     *
     * Implements IInferenceRunner for cross-rank pipeline parallelism.
     * Each rank owns one stage of the pipeline and participates in
     * MPI send/recv for inter-stage transfers.
     */
    class PipelineRunner : public IInferenceRunner
    {
    public:
        // =====================================================================
        // Configuration Types
        // =====================================================================

        /**
         * @brief Configuration for a pipeline stage
         *
         * Describes a single stage in the distributed pipeline.
         * The runner is only non-null for stages owned by this rank.
         */
        struct StageInfo
        {
            /// Unique stage index in the pipeline (0-based)
            int stage_index = 0;

            /// MPI rank that owns this stage
            int owning_rank = 0;

            /// Runner for this stage (nullptr if this rank doesn't own it)
            std::unique_ptr<IInferenceRunner> runner;

            /// First layer this stage processes (inclusive, 0-based)
            int first_layer = 0;

            /// Last layer this stage processes (inclusive, 0-based)
            int last_layer = 0;

            /// Does this stage have the embedding layer?
            bool has_embedding = false;

            /// Does this stage have the LM head?
            bool has_lm_head = false;
        };

        /**
         * @brief Transfer specification between pipeline stages
         *
         * Describes how to transfer activations between adjacent stages.
         */
        struct TransferInfo
        {
            /// From stage index
            int from_stage = 0;

            /// To stage index
            int to_stage = 0;

            /// MPI tag for this transfer
            int mpi_tag = 0;

            /// Transfer mechanism
            TransferSpec::Mechanism mechanism = TransferSpec::Mechanism::MPI_INTERHOST;

            /// Sender rank (-1 for local transfer)
            int sender_rank = 0;

            /// Receiver rank (-1 for local transfer)
            int receiver_rank = 0;
        };

        // =====================================================================
        // Construction
        // =====================================================================

        /**
         * @brief Construct pipeline runner
         *
         * @param my_rank This rank's MPI rank
         * @param world_size Total MPI ranks in the pipeline
         * @param stages Stage configurations (one per pipeline stage)
         * @param transfers Transfer specs between adjacent stages
         * @param hidden_dim Dimension of hidden state for intermediate buffers
         * @param vocab_size Model vocabulary size
         */
        PipelineRunner(
            int my_rank,
            int world_size,
            std::vector<StageInfo> stages,
            std::vector<TransferInfo> transfers,
            int hidden_dim,
            int vocab_size);

        ~PipelineRunner() override;

        // Move-only
        PipelineRunner(PipelineRunner &&) noexcept = default;
        PipelineRunner &operator=(PipelineRunner &&) noexcept = default;
        PipelineRunner(const PipelineRunner &) = delete;
        PipelineRunner &operator=(const PipelineRunner &) = delete;

        // =====================================================================
        // IInferenceRunner Interface
        // =====================================================================

        /**
         * @brief Run forward pass
         *
         * Each rank:
         * 1. If first stage: Process tokens through embedding + layers
         *    else: Receive hidden state from previous rank
         * 2. Execute local layers
         * 3. If last stage: Return logits
         *    else: Send hidden state to next rank
         *
         * @param tokens Token IDs (only used by embedding stage)
         * @param seq_len Sequence length
         * @return true if forward succeeded
         */
        bool forward(const int *tokens, int seq_len) override;

        /**
         * @brief Get logits from last forward pass
         *
         * Returns logits only if this rank owns the final stage (has_lm_head).
         * For other ranks, returns nullptr (they don't have logits).
         *
         * @return Pointer to logits [vocab_size], or nullptr
         */
        const float *logits() const override;

        /**
         * @brief Clear KV cache on all owned stages
         */
        void clear_cache() override;

        /**
         * @brief Get current position in sequence
         */
        int get_position() const override;

        /**
         * @brief Get vocabulary size
         */
        int vocab_size() const override;

        void enableSnapshotCapture(const std::string &output_dir = "") override;
        void disableSnapshotCapture() override;
        void clearSnapshots() override;
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;
        SnapshotInfo getSnapshotWithShape(const std::string &key) const override;
        std::vector<std::string> getSnapshotKeys() const override;

        /**
         * @brief Get execution path type
         */
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }

        /**
         * @brief Get architecture name
         */
        const char *architecture() const override { return "pipeline"; }

        // =====================================================================
        // Pipeline-Specific API
        // =====================================================================

        /**
         * @brief Get the stage index this rank owns
         *
         * @return Stage index (0-based), or -1 if this rank owns no stage
         */
        int myStageIndex() const { return my_stage_index_; }

        /**
         * @brief Get hidden state from last forward
         *
         * Used for chaining to the next stage (via send).
         *
         * @return Pointer to hidden state tensor, or nullptr
         */
        const TensorBase *getHiddenState() const;

        /**
         * @brief Set hidden state (received from previous stage)
         *
         * Used to inject hidden state before executing local layers.
         *
         * @param hidden Hidden state tensor
         */
        void setHiddenState(std::unique_ptr<TensorBase> hidden);

        /**
         * @brief Get number of stages in the pipeline
         */
        int numStages() const { return static_cast<int>(stages_.size()); }

        /**
         * @brief Get number of transfers between stages
         */
        int numTransfers() const { return static_cast<int>(transfers_.size()); }

        /**
         * @brief Get this rank's MPI rank
         */
        int myRank() const { return my_rank_; }

        /**
         * @brief Get total MPI world size
         */
        int worldSize() const { return world_size_; }

        /**
         * @brief Check if this rank owns the embedding stage
         */
        bool hasEmbedding() const;

        /**
         * @brief Check if this rank owns the LM head stage
         */
        bool hasLMHead() const;

    private:
        int my_rank_;
        int world_size_;
        std::vector<StageInfo> stages_;
        std::vector<TransferInfo> transfers_;
        int hidden_dim_;
        int vocab_size_;

        // Which stage this rank owns (-1 if none)
        int my_stage_index_ = -1;

        // Transfer buffer for hidden state
        std::unique_ptr<TensorBase> hidden_buffer_;

        // Logits buffer (only valid for final stage)
        mutable std::vector<float> logits_buffer_;

        // Position counter
        int position_ = 0;

        // ─── Internal Helpers ───

        /**
         * @brief Find which stage this rank owns
         */
        void findMyStage();

        /**
         * @brief Execute my stage's forward pass
         */
        void executeMyStage(const int *tokens, int seq_len);

        /**
         * @brief Send hidden state to next stage
         */
        void sendHiddenState(int to_stage_index);

        /**
         * @brief Receive hidden state from previous stage
         */
        void recvHiddenState(int from_stage_index);

        /**
         * @brief Send logits to requesting rank (for broadcast scenarios)
         */
        void sendLogits(int to_rank);

        /**
         * @brief Receive logits from final stage
         */
        void recvLogits(int from_rank);

        /**
         * @brief Get transfer info for a specific edge
         */
        const TransferInfo *getTransfer(int from_stage, int to_stage) const;
    };

} // namespace llaminar2
