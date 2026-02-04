/**
 * @file PPExecutionStrategy.cpp
 * @brief Pipeline Parallelism execution strategy implementation
 *
 * @see PPExecutionStrategy.h
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "PPExecutionStrategy.h"
#include "../orchestrators/DeviceGraphOrchestrator.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{

    PPExecutionStrategy::PPExecutionStrategy(ILocalPPContext *pp_ctx)
        : pp_ctx_(pp_ctx)
    {
        if (!pp_ctx_)
        {
            throw std::invalid_argument("PPExecutionStrategy: pp_ctx cannot be null");
        }

        LOG_INFO("PPExecutionStrategy: Created with " << pp_ctx_->numStages() << " stages");
    }

    bool PPExecutionStrategy::executeForward(
        std::vector<DeviceGraphOrchestrator *> &runners,
        const int *tokens,
        int seq_len)
    {
        if (runners.empty())
        {
            LOG_ERROR("PPExecutionStrategy::executeForward: No runners provided");
            return false;
        }

        const int num_stages = static_cast<int>(runners.size());

        // Validate runner count matches PP context
        if (num_stages != pp_ctx_->numStages())
        {
            LOG_ERROR("PPExecutionStrategy::executeForward: Runner count ("
                      << num_stages << ") doesn't match PP stage count ("
                      << pp_ctx_->numStages() << ")");
            return false;
        }

        LOG_DEBUG("PPExecutionStrategy: Starting forward pass with "
                  << num_stages << " stages, seq_len=" << seq_len);

        for (int stage = 0; stage < num_stages; ++stage)
        {
            auto *runner = runners[stage];
            if (!runner)
            {
                LOG_ERROR("PPExecutionStrategy: Null runner at stage " << stage);
                return false;
            }

            LOG_DEBUG("PPExecutionStrategy: Executing stage " << stage
                                                              << " on device " << runner->device().to_string());

            // Execute this stage's forward pass
            bool success;
            if (stage == 0)
            {
                // First stage: embed tokens and process layers
                success = runner->forward(tokens, seq_len);
            }
            else
            {
                // Later stages: continue from hidden state
                // The hidden buffer was set by the previous transfer
                success = runner->forwardFromHidden(seq_len);
            }

            if (!success)
            {
                LOG_ERROR("PPExecutionStrategy: Stage " << stage << " forward failed");
                return false;
            }

            // Transfer activations to next stage (if not last)
            if (stage < num_stages - 1)
            {
                auto &state = runner->inferenceState();
                TensorBase *hidden = state.hidden.get();

                if (!hidden)
                {
                    LOG_ERROR("PPExecutionStrategy: Stage " << stage << " has no hidden state");
                    return false;
                }

                LOG_DEBUG("PPExecutionStrategy: Transferring activations from stage "
                          << stage << " to stage " << (stage + 1)
                          << " [" << hidden->numel() << " elements]");

                // Transfer via PP context
                if (!pp_ctx_->transfer(hidden, stage, stage + 1))
                {
                    LOG_ERROR("PPExecutionStrategy: Transfer from stage "
                              << stage << " to " << (stage + 1) << " failed");
                    return false;
                }

                // Point next stage's input to transferred hidden
                // The transfer may have:
                // - Copied data to a buffer on the next device
                // - Made the original buffer accessible to the next device
                //
                // In either case, the next runner needs to know where its input is.
                // This is handled by DeviceGraphOrchestrator::setInputHidden()
                runners[stage + 1]->setInputHidden(hidden);
            }
        }

        LOG_DEBUG("PPExecutionStrategy: Forward pass completed successfully");
        return true;
    }

    bool PPExecutionStrategy::gatherLogits(
        std::vector<DeviceGraphOrchestrator *> &runners,
        TensorBase *output_buffer,
        size_t seq_len)
    {
        if (runners.empty())
        {
            LOG_ERROR("PPExecutionStrategy::gatherLogits: No runners");
            return false;
        }

        if (!output_buffer)
        {
            LOG_ERROR("PPExecutionStrategy::gatherLogits: Null output buffer");
            return false;
        }

        // Logits come from the last stage (which has the LM head)
        auto *last_runner = runners.back();
        if (!last_runner)
        {
            LOG_ERROR("PPExecutionStrategy::gatherLogits: Null last runner");
            return false;
        }

        const float *stage_logits = last_runner->logits();
        if (!stage_logits)
        {
            LOG_ERROR("PPExecutionStrategy::gatherLogits: Last stage has no logits");
            return false;
        }

        // Copy logits to output buffer
        int vocab = last_runner->vocab_size();
        size_t copy_bytes = seq_len * static_cast<size_t>(vocab) * sizeof(float);

        LOG_DEBUG("PPExecutionStrategy::gatherLogits: Copying "
                  << seq_len << "x" << vocab << " logits (" << copy_bytes << " bytes)");

        std::memcpy(output_buffer->mutable_data(), stage_logits, copy_bytes);

        return true;
    }

    void PPExecutionStrategy::clearCaches(
        std::vector<DeviceGraphOrchestrator *> &runners)
    {
        LOG_DEBUG("PPExecutionStrategy::clearCaches: Clearing " << runners.size() << " stages");

        for (auto *runner : runners)
        {
            if (runner)
            {
                runner->clear_cache();
            }
        }
    }

    int PPExecutionStrategy::getPosition(
        const std::vector<DeviceGraphOrchestrator *> &runners) const
    {
        // All stages should have the same position
        // Return from first non-null runner
        for (const auto *runner : runners)
        {
            if (runner)
            {
                return runner->get_position();
            }
        }
        return 0;
    }

    std::string PPExecutionStrategy::description() const
    {
        int stages = pp_ctx_ ? pp_ctx_->numStages() : 0;
        return std::to_string(stages) + "-stage PP with sequential execution";
    }

} // namespace llaminar2
