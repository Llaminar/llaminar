/**
 * @file PipelineRunner.cpp
 * @brief Implementation of cross-rank pipeline runner
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "PipelineRunner.h"
#include "../../tensors/TensorClasses.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <set>
#include <stdexcept>

#include <mpi.h>

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    PipelineRunner::PipelineRunner(
        int my_rank,
        int world_size,
        std::vector<StageInfo> stages,
        std::vector<TransferInfo> transfers,
        int hidden_dim,
        int vocab_size)
        : my_rank_(my_rank),
          world_size_(world_size),
          stages_(std::move(stages)),
          transfers_(std::move(transfers)),
          hidden_dim_(hidden_dim),
          vocab_size_(vocab_size)
    {
        // Validate inputs
        if (stages_.empty())
        {
            throw std::invalid_argument("PipelineRunner: stages cannot be empty");
        }

        if (my_rank < 0 || my_rank >= world_size)
        {
            throw std::invalid_argument("PipelineRunner: my_rank out of range");
        }

        if (hidden_dim <= 0)
        {
            throw std::invalid_argument("PipelineRunner: hidden_dim must be positive");
        }

        if (vocab_size <= 0)
        {
            throw std::invalid_argument("PipelineRunner: vocab_size must be positive");
        }

        // Validate transfer count
        if (stages_.size() > 1 && transfers_.size() != stages_.size() - 1)
        {
            throw std::invalid_argument("PipelineRunner: transfers count must be stages-1");
        }

        // Find which stage this rank owns
        findMyStage();

        // Pre-size logits buffer if we own the final stage
        if (hasLMHead())
        {
            logits_buffer_.resize(vocab_size);
        }

        LOG_DEBUG("PipelineRunner: rank " << my_rank_ << " owns stage " << my_stage_index_
                                          << ", hidden_dim=" << hidden_dim_ << ", vocab_size=" << vocab_size_);
    }

    PipelineRunner::~PipelineRunner() = default;

    // =========================================================================
    // IInferenceRunner Interface
    // =========================================================================

    bool PipelineRunner::forward(const int *tokens, int seq_len)
    {
        // Each rank participates in the pipeline:
        // 1. First stage: Uses tokens directly
        // 2. Other stages: Receive hidden state from previous stage
        // 3. Execute local layers
        // 4. Send hidden state to next stage (or output logits if last)

        if (my_stage_index_ < 0)
        {
            // This rank doesn't own any stage - unusual but handle gracefully
            LOG_WARN("PipelineRunner::forward: rank " << my_rank_ << " owns no stage");
            position_ += seq_len;
            return true;
        }

        // Check if we need to receive hidden state from previous stage
        if (my_stage_index_ > 0)
        {
            recvHiddenState(my_stage_index_ - 1);
        }

        // Execute our stage
        executeMyStage(tokens, seq_len);

        // Send hidden state to next stage (if not the last)
        if (my_stage_index_ < static_cast<int>(stages_.size()) - 1)
        {
            sendHiddenState(my_stage_index_ + 1);
        }

        // Update position
        position_ += seq_len;

        return true;
    }

    const float *PipelineRunner::logits() const
    {
        // Only the final stage has logits
        if (!hasLMHead())
        {
            return nullptr;
        }

        // Get logits from the stage runner
        if (my_stage_index_ >= 0 && stages_[my_stage_index_].runner)
        {
            return stages_[my_stage_index_].runner->logits();
        }

        // Fall back to cached logits buffer
        if (!logits_buffer_.empty())
        {
            return logits_buffer_.data();
        }

        return nullptr;
    }

    void PipelineRunner::clear_cache()
    {
        // Clear cache on the stage we own
        if (my_stage_index_ >= 0 && stages_[my_stage_index_].runner)
        {
            stages_[my_stage_index_].runner->clear_cache();
        }

        // Reset position
        position_ = 0;
    }

    int PipelineRunner::get_position() const
    {
        // Return position from our stage runner if available
        if (my_stage_index_ >= 0 && stages_[my_stage_index_].runner)
        {
            return stages_[my_stage_index_].runner->get_position();
        }

        // Fall back to our tracked position
        return position_;
    }

    int PipelineRunner::vocab_size() const
    {
        return vocab_size_;
    }

    void PipelineRunner::enableSnapshotCapture(const std::string &output_dir)
    {
        for (auto &stage : stages_)
        {
            if (stage.runner)
                stage.runner->enableSnapshotCapture(output_dir);
        }
    }

    void PipelineRunner::disableSnapshotCapture()
    {
        for (auto &stage : stages_)
        {
            if (stage.runner)
                stage.runner->disableSnapshotCapture();
        }
    }

    void PipelineRunner::clearSnapshots()
    {
        for (auto &stage : stages_)
        {
            if (stage.runner)
                stage.runner->clearSnapshots();
        }
    }

    const float *PipelineRunner::getSnapshot(const std::string &key, size_t &out_size) const
    {
        for (const auto &stage : stages_)
        {
            if (!stage.runner)
                continue;
            const float *data = stage.runner->getSnapshot(key, out_size);
            if (data)
                return data;
        }
        out_size = 0;
        return nullptr;
    }

    SnapshotInfo PipelineRunner::getSnapshotWithShape(const std::string &key) const
    {
        for (const auto &stage : stages_)
        {
            if (!stage.runner)
                continue;
            auto info = stage.runner->getSnapshotWithShape(key);
            if (info.data)
                return info;
        }
        return {};
    }

    std::vector<std::string> PipelineRunner::getSnapshotKeys() const
    {
        std::set<std::string> all_keys;
        for (const auto &stage : stages_)
        {
            if (!stage.runner)
                continue;
            auto keys = stage.runner->getSnapshotKeys();
            all_keys.insert(keys.begin(), keys.end());
        }
        return std::vector<std::string>(all_keys.begin(), all_keys.end());
    }

    // =========================================================================
    // Pipeline-Specific API
    // =========================================================================

    const TensorBase *PipelineRunner::getHiddenState() const
    {
        // Get hidden state from the stage we own
        if (my_stage_index_ >= 0 && stages_[my_stage_index_].runner)
        {
            return stages_[my_stage_index_].runner->getHiddenState();
        }

        return nullptr;
    }

    void PipelineRunner::setHiddenState(std::unique_ptr<TensorBase> hidden)
    {
        hidden_buffer_ = std::move(hidden);

        // If we have a runner, set its hidden state
        if (my_stage_index_ >= 0 && stages_[my_stage_index_].runner && hidden_buffer_)
        {
            stages_[my_stage_index_].runner->setHiddenState(hidden_buffer_.get());
        }
    }

    bool PipelineRunner::hasEmbedding() const
    {
        if (my_stage_index_ < 0)
            return false;
        return stages_[my_stage_index_].has_embedding;
    }

    bool PipelineRunner::hasLMHead() const
    {
        if (my_stage_index_ < 0)
            return false;
        return stages_[my_stage_index_].has_lm_head;
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    void PipelineRunner::findMyStage()
    {
        my_stage_index_ = -1;
        for (size_t i = 0; i < stages_.size(); ++i)
        {
            if (stages_[i].owning_rank == my_rank_)
            {
                my_stage_index_ = static_cast<int>(i);
                break;
            }
        }
    }

    void PipelineRunner::executeMyStage(const int *tokens, int seq_len)
    {
        if (my_stage_index_ < 0)
            return;

        auto &stage = stages_[my_stage_index_];
        if (!stage.runner)
        {
            LOG_WARN("PipelineRunner: stage " << my_stage_index_ << " has no runner");
            return;
        }

        // If we have a hidden buffer (received from previous stage),
        // set it on the runner before forward
        if (hidden_buffer_)
        {
            stage.runner->setHiddenState(hidden_buffer_.get());
        }

        // Execute forward
        bool ok = stage.runner->forward(tokens, seq_len);
        if (!ok)
        {
            LOG_ERROR("PipelineRunner: stage " << my_stage_index_ << " forward failed");
        }
    }

    void PipelineRunner::sendHiddenState(int to_stage_index)
    {
        // Get transfer info
        const auto *transfer = getTransfer(my_stage_index_, to_stage_index);
        if (!transfer)
        {
            LOG_WARN("PipelineRunner: no transfer info for " << my_stage_index_ << " -> " << to_stage_index);
            return;
        }

        // For LOCAL_PP, no MPI needed
        if (transfer->mechanism == TransferSpec::Mechanism::LOCAL_PP)
        {
            LOG_DEBUG("PipelineRunner: LOCAL_PP transfer (no MPI send)");
            return;
        }

        // Get hidden state to send
        const TensorBase *hidden = getHiddenState();
        if (!hidden)
        {
            LOG_ERROR("PipelineRunner: no hidden state to send");
            return;
        }

        // Send via MPI
        LOG_DEBUG("PipelineRunner: MPI_Send hidden state to rank " << transfer->receiver_rank
                                                                   << ", tag=" << transfer->mpi_tag);

        const float *data = hidden->data();
        int count = static_cast<int>(hidden->numel());

        MPI_Send(data, count, MPI_FLOAT, transfer->receiver_rank, transfer->mpi_tag, MPI_COMM_WORLD);
    }

    void PipelineRunner::recvHiddenState(int from_stage_index)
    {
        // Get transfer info
        const auto *transfer = getTransfer(from_stage_index, my_stage_index_);
        if (!transfer)
        {
            LOG_WARN("PipelineRunner: no transfer info for " << from_stage_index << " -> " << my_stage_index_);
            return;
        }

        // For LOCAL_PP, no MPI needed - hidden state should already be set
        if (transfer->mechanism == TransferSpec::Mechanism::LOCAL_PP)
        {
            LOG_DEBUG("PipelineRunner: LOCAL_PP transfer (no MPI recv)");
            return;
        }

        // Allocate buffer if needed
        // TODO: Pre-allocate based on max sequence length
        int buffer_size = hidden_dim_; // Simplified - should be seq_len * hidden_dim

        if (!hidden_buffer_)
        {
            hidden_buffer_ = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(hidden_dim_)});
        }

        LOG_DEBUG("PipelineRunner: MPI_Recv hidden state from rank " << transfer->sender_rank
                                                                     << ", tag=" << transfer->mpi_tag);

        float *data = hidden_buffer_->mutable_data();
        MPI_Status status;
        MPI_Recv(data, buffer_size, MPI_FLOAT, transfer->sender_rank, transfer->mpi_tag,
                 MPI_COMM_WORLD, &status);
    }

    void PipelineRunner::sendLogits(int to_rank)
    {
        const float *data = logits();
        if (!data)
        {
            LOG_ERROR("PipelineRunner: no logits to send");
            return;
        }

        // Use a special tag for logits
        int tag = 10000; // Reserved tag range for logits

        LOG_DEBUG("PipelineRunner: MPI_Send logits to rank " << to_rank);
        MPI_Send(data, vocab_size_, MPI_FLOAT, to_rank, tag, MPI_COMM_WORLD);
    }

    void PipelineRunner::recvLogits(int from_rank)
    {
        logits_buffer_.resize(vocab_size_);

        int tag = 10000; // Reserved tag range for logits

        LOG_DEBUG("PipelineRunner: MPI_Recv logits from rank " << from_rank);
        MPI_Status status;
        MPI_Recv(logits_buffer_.data(), vocab_size_, MPI_FLOAT, from_rank, tag,
                 MPI_COMM_WORLD, &status);
    }

    const PipelineRunner::TransferInfo *PipelineRunner::getTransfer(int from_stage, int to_stage) const
    {
        for (const auto &t : transfers_)
        {
            if (t.from_stage == from_stage && t.to_stage == to_stage)
            {
                return &t;
            }
        }
        return nullptr;
    }

} // namespace llaminar2
