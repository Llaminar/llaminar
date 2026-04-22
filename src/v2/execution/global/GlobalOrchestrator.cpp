/**
 * @file GlobalOrchestrator.cpp
 * @brief Implementation of cross-machine MPI cluster inference orchestrator
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include "GlobalOrchestrator.h"
#include "../global_pp/GlobalPPRankPlanBuilder.h"
#include "../../tensors/TensorClasses.h"
#include "../../utils/Logger.h"

#include <stdexcept>
#include <cassert>

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    GlobalOrchestrator::GlobalOrchestrator(Config config)
        : config_(std::move(config))
    {
        // Validate required fields
        if (!config_.mpi_ctx)
        {
            throw std::invalid_argument("GlobalOrchestrator: mpi_ctx is required");
        }
        if (!config_.rank_runner)
        {
            throw std::invalid_argument("GlobalOrchestrator: rank_runner is required");
        }
        if (config_.topology.stages.empty())
        {
            throw std::invalid_argument("GlobalOrchestrator: topology must have at least one stage");
        }
        if (config_.rank < 0 || config_.rank >= config_.world_size)
        {
            throw std::invalid_argument("GlobalOrchestrator: rank out of range [0, world_size)");
        }
        if (config_.vocab_size <= 0)
        {
            throw std::invalid_argument("GlobalOrchestrator: vocab_size must be positive");
        }
        if (config_.d_model <= 0)
        {
            throw std::invalid_argument("GlobalOrchestrator: d_model must be positive");
        }

        // Take ownership of rank runner
        rank_runner_ = std::move(config_.rank_runner);

        // Build this rank's execution plan from topology
        rank_plan_ = GlobalPPRankPlanBuilder::build(config_.topology, config_.rank);

        LOG_INFO("GlobalOrchestrator: rank " << config_.rank << "/" << config_.world_size
                 << ", " << config_.topology.numStages() << " PP stages, "
                 << rank_plan_.steps.size() << " execution steps");
        LOG_DEBUG("GlobalOrchestrator plan:\n" << rank_plan_.toString());

        // Cache pipeline head/tail
        tail_rank_ = findTailRank();
        for (const auto *action : rank_plan_.executeStages())
        {
            if (action->has_embedding)
                is_pipeline_head_ = true;
            if (action->has_lm_head)
                is_pipeline_tail_ = true;
        }

        // Allocate activation transfer buffer for PP (if we have any transfers)
        if (!rank_plan_.transferActions().empty() && config_.d_model > 0)
        {
            // Buffer size: max_seq_len * d_model floats
            // Use a conservative max_seq_len estimate; will be resized on demand
            // For now: create a 1D FP32 tensor sized to d_model (single token decode)
            // The actual transfer count is determined at runtime from the hidden state
            activation_buffer_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(config_.d_model)});

            LOG_DEBUG("GlobalOrchestrator: allocated activation buffer ["
                      << config_.d_model << "] for PP transfers");
        }

        LOG_INFO("GlobalOrchestrator: pipeline_head=" << is_pipeline_head_
                 << " pipeline_tail=" << is_pipeline_tail_
                 << " tail_rank=" << tail_rank_);
    }

    // =========================================================================
    // Core Inference API
    // =========================================================================

    bool GlobalOrchestrator::forward(const int *tokens, int seq_len)
    {
        last_seq_len_ = seq_len;

        for (const auto &step : rank_plan_.steps)
        {
            switch (step.type)
            {
            case GlobalPPRankPlan::Step::Type::EXECUTE_STAGE:
            {
                if (step.stage_action.role == RankStageAction::Role::EXECUTE)
                {
                    if (!executeStage(step.stage_action, tokens, seq_len))
                    {
                        LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                                  << " failed to execute stage " << step.stage_action.stage_id);
                        return false;
                    }
                }
                // IDLE role: nothing to do
                break;
            }
            case GlobalPPRankPlan::Step::Type::TRANSFER:
            {
                if (step.transfer_action.direction != RankTransferAction::Direction::NONE)
                {
                    if (!executeTransfer(step.transfer_action))
                    {
                        LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                                  << " failed transfer (peer=" << step.transfer_action.peer_rank
                                  << " tag=" << step.transfer_action.mpi_tag << ")");
                        return false;
                    }
                }
                break;
            }
            }
        }
        return true;
    }

    const float *GlobalOrchestrator::logits() const
    {
        // Only the tail rank (with LM head) has valid logits
        if (is_pipeline_tail_)
        {
            return rank_runner_->logits();
        }
        return nullptr;
    }

    int GlobalOrchestrator::vocab_size() const
    {
        return config_.vocab_size;
    }

    void GlobalOrchestrator::clear_cache()
    {
        rank_runner_->clear_cache();
        // Synchronize across ranks to ensure all caches are cleared
        config_.mpi_ctx->barrier();
    }

    int GlobalOrchestrator::get_position() const
    {
        return rank_runner_->get_position();
    }

    ExecutionPath GlobalOrchestrator::executionPath() const
    {
        return rank_runner_->executionPath();
    }

    const char *GlobalOrchestrator::architecture() const
    {
        return config_.architecture_name.c_str();
    }

    // =========================================================================
    // GPU-side Sampling with Cross-Rank Broadcast
    // =========================================================================

    int GlobalOrchestrator::sampleGreedyOnDevice()
    {
        int32_t token = -1;

        if (is_pipeline_tail_)
        {
            // Tail rank: sample locally
            token = rank_runner_->sampleGreedyOnDevice();
            if (token < 0)
            {
                // Fallback to CPU sampling
                const float *log = logits();
                if (log)
                {
                    // Simple argmax fallback
                    token = 0;
                    float best = log[0];
                    for (int i = 1; i < config_.vocab_size; ++i)
                    {
                        if (log[i] > best)
                        {
                            best = log[i];
                            token = i;
                        }
                    }
                }
            }
        }

        // Broadcast sampled token from tail rank to all ranks
        config_.mpi_ctx->broadcast_int32(&token, 1, tail_rank_);
        return token;
    }

    int GlobalOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        int32_t token = -1;

        if (is_pipeline_tail_)
        {
            token = rank_runner_->sampleOnDevice(params);
            if (token < 0)
            {
                // Fallback: greedy
                token = sampleGreedyOnDevice();
                // Note: sampleGreedyOnDevice already broadcasts, so return directly
                return token;
            }
        }

        // Broadcast sampled token from tail rank to all ranks
        config_.mpi_ctx->broadcast_int32(&token, 1, tail_rank_);
        return token;
    }

    // =========================================================================
    // Logits Gather Control (delegate)
    // =========================================================================

    void GlobalOrchestrator::setSkipLogitsGatherDecode(bool skip)
    {
        rank_runner_->setSkipLogitsGatherDecode(skip);
    }

    void GlobalOrchestrator::setSkipLogitsGatherPrefill(bool skip)
    {
        rank_runner_->setSkipLogitsGatherPrefill(skip);
    }

    // =========================================================================
    // Timeline/Profiling (delegate)
    // =========================================================================

    void GlobalOrchestrator::setSuppressTimeline(bool suppress)
    {
        rank_runner_->setSuppressTimeline(suppress);
    }

    void GlobalOrchestrator::setAccumulatePrefill(bool accumulate)
    {
        rank_runner_->setAccumulatePrefill(accumulate);
    }

    void GlobalOrchestrator::flushStageTimeline()
    {
        rank_runner_->flushStageTimeline();
    }

    // =========================================================================
    // Hidden State API (delegate)
    // =========================================================================

    TensorBase *GlobalOrchestrator::getHiddenState()
    {
        return rank_runner_->getHiddenState();
    }

    const TensorBase *GlobalOrchestrator::getHiddenState() const
    {
        return rank_runner_->getHiddenState();
    }

    void GlobalOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        rank_runner_->setHiddenState(hidden_state);
    }

    bool GlobalOrchestrator::hasHiddenStateInput() const
    {
        return rank_runner_->hasHiddenStateInput();
    }

    void GlobalOrchestrator::clearHiddenStateInput()
    {
        rank_runner_->clearHiddenStateInput();
    }

    // =========================================================================
    // Snapshot API (delegate)
    // =========================================================================

    void GlobalOrchestrator::enableSnapshotCapture(const std::string &output_dir)
    {
        rank_runner_->enableSnapshotCapture(output_dir);
    }

    void GlobalOrchestrator::disableSnapshotCapture()
    {
        rank_runner_->disableSnapshotCapture();
    }

    void GlobalOrchestrator::clearSnapshots()
    {
        rank_runner_->clearSnapshots();
    }

    const float *GlobalOrchestrator::getSnapshot(const std::string &key, size_t &out_size) const
    {
        return rank_runner_->getSnapshot(key, out_size);
    }

    SnapshotInfo GlobalOrchestrator::getSnapshotWithShape(const std::string &key) const
    {
        return rank_runner_->getSnapshotWithShape(key);
    }

    std::vector<std::string> GlobalOrchestrator::getSnapshotKeys() const
    {
        return rank_runner_->getSnapshotKeys();
    }

    // =========================================================================
    // Device & Logits Local (delegate)
    // =========================================================================

    DeviceId GlobalOrchestrator::primaryDeviceId() const
    {
        return rank_runner_->primaryDeviceId();
    }

    bool GlobalOrchestrator::hasLogitsLocal() const
    {
        if (is_pipeline_tail_)
            return rank_runner_->hasLogitsLocal();
        return false;
    }

    LogitsLocalInfo GlobalOrchestrator::getLogitsLocalInfo() const
    {
        if (is_pipeline_tail_)
            return rank_runner_->getLogitsLocalInfo();
        return {};
    }

    // =========================================================================
    // Query API
    // =========================================================================

    bool GlobalOrchestrator::isPipelineHead() const { return is_pipeline_head_; }
    bool GlobalOrchestrator::isPipelineTail() const { return is_pipeline_tail_; }
    int GlobalOrchestrator::pipelineDepth() const { return config_.topology.numStages(); }
    const GlobalPPRankPlan &GlobalOrchestrator::rankPlan() const { return rank_plan_; }
    const GlobalPPTopology &GlobalOrchestrator::topology() const { return config_.topology; }

    // =========================================================================
    // Internal: Execute Stage
    // =========================================================================

    bool GlobalOrchestrator::executeStage(const RankStageAction &action,
                                          const int *tokens, int seq_len)
    {
        if (action.has_embedding)
        {
            // Pipeline head: pass tokens to rank runner for embedding + layers
            return rank_runner_->forward(tokens, seq_len);
        }
        else
        {
            // Middle/tail stage: hidden state already set via transfer
            // rank_runner should have had setHiddenState() called after the
            // preceding RECV transfer step.
            return rank_runner_->forward(tokens, seq_len);
        }
    }

    // =========================================================================
    // Internal: Execute Transfer
    // =========================================================================

    bool GlobalOrchestrator::executeTransfer(const RankTransferAction &action)
    {
        if (action.direction == RankTransferAction::Direction::NONE)
            return true;

        if (action.direction == RankTransferAction::Direction::SEND)
        {
            // Get hidden state from rank runner
            TensorBase *hidden = rank_runner_->getHiddenState();
            if (!hidden)
            {
                LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                          << " has no hidden state to send");
                return false;
            }

            // Send via IMPIContext wrapper (data() triggers GPU→host sync if needed)
            const float *send_data = hidden->data();
            size_t count = hidden->numel();

            LOG_DEBUG("GlobalOrchestrator: rank " << config_.rank
                      << " SEND " << count << " floats to rank " << action.peer_rank
                      << " tag=" << action.mpi_tag);

            config_.mpi_ctx->sendFloat(send_data, count, action.peer_rank, action.mpi_tag);
            return true;
        }
        else // RECV
        {
            // Ensure activation buffer is sized for current sequence
            size_t needed = static_cast<size_t>(last_seq_len_) * config_.d_model;
            if (needed == 0) needed = static_cast<size_t>(config_.d_model); // fallback for single token
            ensureActivationBufferCapacity(needed);

            TensorBase *recv_tensor = activation_buffer_.get();
            if (!recv_tensor)
            {
                LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                          << " has no activation buffer for RECV");
                return false;
            }

            float *recv_data = recv_tensor->mutable_data();
            size_t count = recv_tensor->numel();

            LOG_DEBUG("GlobalOrchestrator: rank " << config_.rank
                      << " RECV " << count << " floats from rank " << action.peer_rank
                      << " tag=" << action.mpi_tag);

            config_.mpi_ctx->recvFloat(recv_data, count, action.peer_rank, action.mpi_tag, nullptr);

            // Pass received hidden state to rank runner for the next stage
            rank_runner_->setHiddenState(recv_tensor);
            return true;
        }
    }

    void GlobalOrchestrator::ensureActivationBufferCapacity(size_t num_elements)
    {
        if (!activation_buffer_ || activation_buffer_->numel() < num_elements)
        {
            activation_buffer_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{num_elements});
            LOG_DEBUG("GlobalOrchestrator: resized activation buffer to " << num_elements << " elements");
        }
    }

    // =========================================================================
    // Internal: Find Tail Rank
    // =========================================================================

    int GlobalOrchestrator::findTailRank() const
    {
        for (const auto &stage : config_.topology.stages)
        {
            if (stage.has_lm_head)
            {
                if (stage.is_global_tp)
                {
                    // For global TP stages, all participating ranks have the LM head.
                    // Use the first participating rank as the "primary" tail for
                    // broadcasting purposes.
                    if (!stage.participating_ranks.empty())
                        return stage.participating_ranks[0];
                }
                return stage.owning_rank;
            }
        }
        // Fallback: last stage's owner
        if (!config_.topology.stages.empty())
        {
            const auto &last = config_.topology.stages.back();
            return last.is_global_tp && !last.participating_ranks.empty()
                       ? last.participating_ranks[0]
                       : last.owning_rank;
        }
        return 0;
    }

} // namespace llaminar2
