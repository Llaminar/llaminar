/**
 * @file ModelExecutor.cpp
 * @brief Base implementation of model-level execution orchestration
 * @author David Sanftenberg
 * @date December 2025
 */

#include "ModelExecutor.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../backends/ComputeBackend.h"
#include <chrono>

namespace llaminar2
{

    // =============================================================================
    // Constructor
    // =============================================================================

    ModelExecutor::ModelExecutor(std::shared_ptr<ModelContext> model_ctx,
                                 std::shared_ptr<IMPIContext> mpi_ctx,
                                 const ModelExecutorConfig &config)
        : model_ctx_(std::move(model_ctx)),
          mpi_ctx_(std::move(mpi_ctx)),
          config_(config),
          layer_executor_(config.layer_config)
    {
        LOG_DEBUG("[ModelExecutor] Initialized with n_layers=" << config_.n_layers
                                                              << ", max_batch_size=" << config_.max_batch_size
                                                              << ", max_seq_len=" << config_.max_seq_len);

        // Propagate MPI rank to layer executor for stage dumping
        if (mpi_ctx_)
        {
            layer_executor_.setMPIRank(mpi_ctx_->rank());
        }
    }

    // =============================================================================
    // Device Context Management
    // =============================================================================

    IDeviceContext *ModelExecutor::getDeviceContext(DeviceId device)
    {
        auto it = device_contexts_.find(device);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }

        // Create new context
        auto ctx = IDeviceContext::create(device);
        if (!ctx)
        {
            LOG_ERROR("[ModelExecutor] Failed to create device context for device " << device.to_string());
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device] = std::move(ctx);
        return raw_ptr;
    }

    // =============================================================================
    // Forward Execution
    // =============================================================================

    bool ModelExecutor::executeForward(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Validate input
        if (!input.token_ids && !input.batches)
        {
            LOG_ERROR("[ModelExecutor] No token input provided");
            return false;
        }

        if (input.seq_len <= 0)
        {
            LOG_ERROR("[ModelExecutor] Invalid sequence length: " << input.seq_len);
            return false;
        }

        LOG_TRACE("[ModelExecutor] executeForward: batch_size=" << input.batch_size
                                                                << ", seq_len=" << input.seq_len
                                                                << ", device=" << input.device.to_string());

        // Build and execute full forward graph
        ComputeGraph graph = buildFullForwardGraph(input, output);

        if (graph.size() == 0)
        {
            LOG_ERROR("[ModelExecutor] Empty forward graph");
            return false;
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(input.device);
        if (!ctx)
        {
            return false;
        }

        // Execute the graph
        bool success = execute(graph, ctx);

        // Update statistics
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats_.total_time_ms += total_ms;
        stats_.total_forward_passes++;
        stats_.total_tokens_processed += input.batch_size * input.seq_len;

        LOG_TRACE("[ModelExecutor] Forward pass completed in " << total_ms << " ms");

        return success;
    }

    bool ModelExecutor::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        return layer_executor_.execute(graph, ctx);
    }

    // =============================================================================
    // Phase Execution with Timing
    // =============================================================================

    bool ModelExecutor::executeEmbedding(const ForwardInput &input, TensorBase *output_hidden)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ComputeGraph graph = buildEmbeddingGraph(input, output_hidden);
        IDeviceContext *ctx = getDeviceContext(input.device);

        if (!ctx)
        {
            return false;
        }

        bool success = execute(graph, ctx);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats_.embedding_time_ms += ms;

        return success;
    }

    bool ModelExecutor::executeTransformerLayers(
        TensorBase *hidden,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (config_.per_layer_graphs)
        {
            // Execute layers separately (for debugging)
            for (int i = 0; i < config_.n_layers; ++i)
            {
                ComputeGraph graph = buildLayerGraph(i, hidden, kv_cache, position_ids, device);
                IDeviceContext *ctx = getDeviceContext(device);

                if (!ctx || !execute(graph, ctx))
                {
                    LOG_ERROR("[ModelExecutor] Layer " << i << " failed");
                    return false;
                }
            }
        }
        else
        {
            // Execute all layers as single graph
            ComputeGraph graph = buildTransformerLayersGraph(hidden, kv_cache, position_ids, device);
            IDeviceContext *ctx = getDeviceContext(device);

            if (!ctx || !execute(graph, ctx))
            {
                return false;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats_.layers_time_ms += ms;

        return true;
    }

    bool ModelExecutor::executeLMHead(TensorBase *hidden, TensorBase *logits, int total_tokens, DeviceId device,
                                      TensorBase *logits_local)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ComputeGraph graph = buildLMHeadGraph(hidden, logits, total_tokens, device, logits_local);
        IDeviceContext *ctx = getDeviceContext(device);

        if (!ctx)
        {
            return false;
        }

        bool success = execute(graph, ctx);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        stats_.lm_head_time_ms += ms;

        return success;
    }

    // =============================================================================
    // State Management
    // =============================================================================

    void ModelExecutor::clearCache()
    {
        LOG_DEBUG("[ModelExecutor] Clearing cache");
        // Derived classes should override to clear KV cache
    }

    void ModelExecutor::recordPhaseTime(const std::string &phase, double ms)
    {
        if (phase == "embedding")
        {
            stats_.embedding_time_ms += ms;
        }
        else if (phase == "layers")
        {
            stats_.layers_time_ms += ms;
        }
        else if (phase == "lm_head")
        {
            stats_.lm_head_time_ms += ms;
        }
    }

} // namespace llaminar2
