/**
 * @file Qwen2Pipeline.cpp
 * @brief Qwen 2.x transformer pipeline implementation
 *
 * Greenfield V2 implementation with:
 * - Direct kernel orchestration (no operator layer)
 * - Streaming dequant in kernels (no slab cache)
 * - Per-tensor device placement
 * - Selective BF16 for bandwidth-bound ops
 *
 * @author David Sanftenberg
 */

#include "Qwen2Pipeline.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugAssert.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/KernelProfiler.h"
#include "../PipelineFactory.h"
#include "../../loaders/ModelLoader.h"
#include "../../tensors/TensorFactory.h"
#include "../../backends/ComputeBackend.h"
#include "../../utils/BatchPaddingUtils.h"
#include "../../kernels/cpu/ops/CPURMSNormKernelT.h"
#include "../../kernels/cpu/gemm_v4/FusedGEMM.h"
#include "../../tensors/SIMDHelpers.h"
#include "../ops/EmbeddingOp.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <omp.h>

namespace llaminar2
{
    // =============================================================================
    // Factory Registration
    // =============================================================================

    /**
     * @brief Creator function for Qwen2Pipeline
     */
    static std::unique_ptr<PipelineBase> createQwen2(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        const PipelineConfig &config)
    {
        // Factory doesn't have placement_map yet
        // Use batch_size from config (defaults to 1 in PipelineConfig)
        return std::make_unique<Qwen2Pipeline>(model_ctx, mpi_ctx, device_idx, nullptr, config, config.batch_size);
    }

    /**
     * @brief Register Qwen2Pipeline with factory
     *
     * Made public so tests can force registration if needed
     */
    void ensureQwen2Registration()
    {
        static bool registered = false;
        if (!registered)
        {
            PipelineFactory::instance().registerCreator("qwen2", &createQwen2);
            registered = true;
        }
    }

    /**
     * @brief Automatic registration at startup
     */
    __attribute__((constructor)) static void initQwen2()
    {
        ensureQwen2Registration();
    }

    // =============================================================================
    // Pipeline Implementation
    // =============================================================================

    Qwen2Pipeline::Qwen2Pipeline(std::shared_ptr<ModelContext> model_ctx,
                                 std::shared_ptr<MPIContext> mpi_ctx,
                                 int device_idx,
                                 std::shared_ptr<WeightPlacementMap> placement_map,
                                 const PipelineConfig &config,
                                 int batch_size)
        : PipelineBase(model_ctx, mpi_ctx, device_idx, placement_map, config),
          batch_size_(batch_size)
    {
        LOG_INFO("Initializing Qwen 2.x pipeline (batch_size=" << batch_size << ")");

        // Initialize typed ops for the configured activation precision (Q8_1, BF16, etc.)
        initializeTypedOps();

        // Read architecture from GGUF metadata
        const GGUFModel &model = model_ctx_->model();
        n_layers_ = static_cast<int>(model.block_count);
        d_model_ = static_cast<int>(model.embedding_length);
        vocab_size_ = static_cast<int>(model.vocab_size);
        n_heads_ = static_cast<int>(model.head_count);
        n_kv_heads_ = static_cast<int>(model.head_count_kv);

        // Calculate head_dim from d_model and n_heads
        head_dim_ = d_model_ / n_heads_;

        // Read FFN intermediate size from metadata
        if (model.hasMetadata("qwen2.feed_forward_length"))
        {
            d_ff_ = static_cast<int>(model.metadata.at("qwen2.feed_forward_length").asUInt32());
        }
        else
        {
            // Fallback: typical ratio for Qwen models
            d_ff_ = d_model_ * 4;
            LOG_WARN("Warning: feed_forward_length not in metadata, using " << d_ff_);
        }

        LOG_DEBUG("Architecture: " << n_layers_ << " layers, "
                                   << d_model_ << " d_model, " << vocab_size_ << " vocab");
        LOG_DEBUG("Attention: " << n_heads_ << " heads, "
                                << n_kv_heads_ << " KV heads (GQA), " << head_dim_ << " head_dim");
        LOG_DEBUG("FFN: " << d_ff_ << " intermediate_size (SwiGLU)");

        // =============================================================================
        // FFN Column-Parallel Sharding Setup (Phase 4b-1)
        // =============================================================================
        // When weight sharding is enabled, Gate/Up are column-parallel:
        // - Each rank produces [seq, d_ff_local] where d_ff_local = d_ff / world_size
        // - Down projection (row-parallel) receives sharded input
        // - Final allreduce sums partial Down outputs
        // =============================================================================
        bool has_mpi = mpi_ctx_ && mpi_ctx_->world_size() > 1;
        bool has_weight_manager = model_ctx_->weightManager() != nullptr;
        bool is_sharded = has_weight_manager &&
                          model_ctx_->weightManager()->strategy() == WeightDistributionStrategy::SHARDED;

        LOG_DEBUG("FFN column-parallel check: has_mpi=" << has_mpi
                                                        << ", has_weight_manager=" << has_weight_manager
                                                        << ", is_sharded=" << is_sharded);

        if (has_mpi && is_sharded)
        {
            int world_size = mpi_ctx_->world_size();
            if (d_ff_ % world_size != 0)
            {
                throw std::runtime_error("FFN intermediate size (" + std::to_string(d_ff_) +
                                         ") not divisible by world_size (" + std::to_string(world_size) + ")");
            }
            d_ff_local_ = d_ff_ / world_size;
            ffn_column_parallel_ = true;
            LOG_INFO("FFN column-parallel enabled: d_ff=" << d_ff_ << " -> d_ff_local=" << d_ff_local_
                                                          << " per rank (world_size=" << world_size << ")");
        }
        else
        {
            d_ff_local_ = d_ff_;
            ffn_column_parallel_ = false;
        }

        // Weights are loaded lazily via getLayerWeight() and model_ctx_->getWeight()
        // Resize layer weights vector for lazy loading
        layers_.resize(n_layers_);

        // =============================================================================
        // Generic Initialization (PipelineBase handles device/MPI/KV cache setup)
        // =============================================================================

        // Initialize infrastructure with batched workspace buffers
        initializeInfrastructureBatched();

        // Override KV cache with batched version
        // Use effective MPI context (user-provided or default single-rank)
        const MPIContext &effective_mpi_ctx = mpi_ctx_ ? *mpi_ctx_ : *default_mpi_ctx_;
        std::vector<int> attention_devices = detectAttentionDevices(n_layers_);
        kv_cache_batched_ = std::make_shared<BatchedKVCache>(
            effective_mpi_ctx, n_layers_, batch_size_, config.max_seq_len, n_kv_heads_, head_dim_, attention_devices);
        LOG_DEBUG("Initialized batched KV cache: batch_size=" << batch_size_
                                                              << ", max_seq_len=" << config.max_seq_len);

        // =============================================================================
        // Fused Attention + Wo Kernel (Phase 5: experimental, optional)
        // =============================================================================
        // When enabled via config.use_fused_attention, use fused kernel that:
        // 1. Eliminates context quantization round-trip (FP32 → Q8_1 → FP32)
        // 2. Improves cache locality (context stays in registers through projection)
        // Only supported with Q8_1 activation precision currently.
        // =============================================================================
        if (config_.use_fused_attention && config_.activation_precision == ActivationPrecision::Q8_1)
        {
            FusedAttentionWoKernel::Config fused_config;
            fused_config.num_heads = n_heads_;
            fused_config.num_kv_heads = n_kv_heads_;
            fused_config.head_dim = head_dim_;
            fused_config.d_model = d_model_;
            // Use REFERENCE backend initially (JIT doesn't apply Wo projection yet)
            fused_config.backend = FusedAttentionBackend::REFERENCE;

            fused_attn_wo_kernel_ = std::make_unique<FusedAttentionWoKernel>(fused_config);
            LOG_INFO("Fused attention+Wo kernel enabled (backend=REFERENCE)");
        }
        else if (config_.use_fused_attention)
        {
            LOG_WARN("Fused attention requires Q8_1 activation precision, using unfused path");
        }

        LOG_DEBUG("Pipeline initialized (weights loaded on-demand)");
    }

    void Qwen2Pipeline::initializeInfrastructureBatched()
    {
        // Use max_seq_len from runtime configuration
        int max_seq_len = config_.max_seq_len;

        // Device infrastructure with batch_size for workspace mask allocation
        initializeDeviceInfrastructure(max_seq_len, batch_size_);

        // MPI strategy configuration
        configureMPIStrategy();

        // KV cache initialization
        initializeKVCache(max_seq_len);

        LOG_DEBUG("Pipeline infrastructure initialized (max_seq_len=" << max_seq_len
                                                                      << ", batch_size=" << batch_size_ << ")");
    }

    // =============================================================================
    // Multi-Device Infrastructure (implements PipelineBase abstract methods)
    // =============================================================================

    std::vector<std::string> Qwen2Pipeline::getAllWeightNames() const
    {
        std::vector<std::string> weight_names;

        // Embedding
        weight_names.push_back("token_embd.weight");

        // Layer weights
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            weight_names.push_back(prefix + "attn_q.weight");
            weight_names.push_back(prefix + "attn_k.weight");
            weight_names.push_back(prefix + "attn_v.weight");
            weight_names.push_back(prefix + "attn_output.weight");
            weight_names.push_back(prefix + "attn_norm.weight");
            weight_names.push_back(prefix + "ffn_gate.weight");
            weight_names.push_back(prefix + "ffn_up.weight");
            weight_names.push_back(prefix + "ffn_down.weight");
            weight_names.push_back(prefix + "ffn_norm.weight");
        }

        // Output weights
        weight_names.push_back("output_norm.weight");
        weight_names.push_back("output.weight");

        return weight_names;
    }

    ActivationBuffers Qwen2Pipeline::createBuffersForDevice(int device_idx, int max_seq_len)
    {
        ActivationBuffers buffers;
        // Size buffers for batch_size * max_seq_len (flattened batch dimension)
        int effective_max = batch_size_ * max_seq_len;
        buffers.max_seq_len = effective_max;

        // Get configured activation precision
        auto precision = config_.activation_precision;

        // Helper lambda to create activation tensors via inherited tensor_factory_
        auto createActivation = [&](const std::vector<size_t> &shape) -> std::shared_ptr<TensorBase>
        {
            return tensor_factory_->createActivation(shape, precision, device_idx);
        };

        // Helper to create FP32 tensors (for operations that need high precision)
        auto createFP32 = [&](const std::vector<size_t> &shape) -> std::shared_ptr<TensorBase>
        {
            return tensor_factory_->createActivation(shape, ActivationPrecision::FP32, device_idx);
        };

        // Residual (d_model) - sized for batch
        buffers.residual = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)});

        // Normalization buffer (shared across attention and FFN) - sized for batch
        buffers.normalized = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)});

        // Attention buffers (Qwen-specific dimensions) - sized for batch
        // Q/K/V can be low precision (Q8_1 supports dequant via data() for attention input)
        buffers.Q = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_heads_ * head_dim_)});
        buffers.K = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)});
        buffers.V = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)});

        // attn_output: For Q8_1 mode, the fused attention kernel performs softmax
        // internally in FP32 and outputs Q8_1 blocks. This avoids a dequant→requant
        // round trip that would add quantization noise. For other precisions, the
        // attention computation uses FP32 throughout.
        buffers.attn_output = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_heads_ * head_dim_)});

        buffers.attn_proj = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)});

        // FFN buffers (Qwen-specific d_ff_) - sized for batch
        // Use d_ff_local_ for column-parallel sharding (each rank produces local output)
        int ffn_dim = ffn_column_parallel_ ? d_ff_local_ : d_ff_;
        buffers.gate = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(ffn_dim)});
        buffers.up = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(ffn_dim)});
        buffers.ffn_output = createActivation(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)});

        return buffers;
    }

    bool Qwen2Pipeline::forward(const int *tokens, int seq_len)
    {
        // Legacy single-sequence interface: wrap as batch_size=1
        std::vector<int> token_vec(tokens, tokens + seq_len);
        return forward_batch(std::vector<std::vector<int>>{token_vec});
    }

    bool Qwen2Pipeline::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        DEBUG_ASSERT(static_cast<int>(token_batches.size()) == batch_size_,
                     "Expected batch_size=" << batch_size_ << ", got " << token_batches.size());

        // Initialize per-sequence position counters if needed
        if (static_cast<int>(current_positions_.size()) != batch_size_)
        {
            current_positions_.assign(batch_size_, 0);
            LOG_DEBUG("[Position Init] Initialized current_positions_ to size " << batch_size_ << " (all zeros)");
        }
        else
        {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < current_positions_.size(); ++i)
            {
                ss << current_positions_[i] << (i < current_positions_.size() - 1 ? ", " : "");
            }
            ss << "]";
            LOG_TRACE("[Position State] current_positions_: " << ss.str());
        }

        // Pad sequences to uniform length
        auto padded = createPaddedBatch(token_batches, /*pad_token_id=*/0);
        padded_seq_len_ = padded.max_length;
        sequence_lengths_ = padded.actual_lengths;
        int effective_seq_len = batch_size_ * padded_seq_len_;

        LOG_TRACE("Forward pass: batch_size=" << batch_size_
                                              << ", padded_seq_len=" << padded_seq_len_
                                              << ", effective_seq_len=" << effective_seq_len);

        // Allocate activation tensors if needed (sized for batch)
        if (!current_hidden_ || static_cast<int>(current_hidden_->shape()[0]) != effective_seq_len)
        {
            current_hidden_ = tensor_factory_->createActivation(
                {static_cast<size_t>(effective_seq_len), static_cast<size_t>(d_model_)},
                config_.activation_precision, device_idx_);
            LOG_TRACE("Allocated hidden states: "
                      << effective_seq_len << " x " << d_model_ << " on device " << device_idx_);
        }

        // Validate hidden state dimensions
        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "hidden_allocation");

        // Batch embedding lookup
        if (!embedding_batch(token_batches, current_hidden_.get()))
        {
            LOG_ERROR("Embedding batch failed");
            return false;
        }

        // Capture embedding output
        CAPTURE_SNAPSHOT("EMBEDDING", current_hidden_.get());

        // Validate after embedding
        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_embedding");

        // DEBUG: Print embedding values for last position
        static const bool debug_layers = std::getenv("LLAMINAR_DEBUG_LAYERS") != nullptr;
        if (debug_layers && (!mpi_ctx_ || mpi_ctx_->rank() == 0))
        {
            const float *hidden = current_hidden_->fp32_data(); // Explicit dequant for debug
            int last_pos = effective_seq_len - 1;
            LOG_DEBUG("[DEBUG] Embedding output (position " << last_pos << "), first 10 values:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_TRACE("  hidden[" << i << "] = " << hidden[last_pos * d_model_ + i]);
            }
        }

        // Process all transformer layers
        for (int i = 0; i < n_layers_; ++i)
        {
            if (!transformer_layer(i, effective_seq_len))
            {
                LOG_ERROR("Layer " << i << " failed");
                return false;
            }

            // DEBUG: Print hidden state after each layer
            if (debug_layers && (!mpi_ctx_ || mpi_ctx_->rank() == 0) && i == 0)
            {
                const float *hidden = current_hidden_->fp32_data(); // Explicit dequant for debug
                int last_pos = effective_seq_len - 1;
                LOG_TRACE("[DEBUG] After layer " << i << " (position " << last_pos << "), first 10 values:");
                for (int j = 0; j < 10; ++j)
                {
                    LOG_TRACE("  hidden[" << j << "] = " << hidden[last_pos * d_model_ + j]);
                }
            }

            // Validate hidden state dimensions unchanged between layers
            VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_layer_" + std::to_string(i));
        }

        // Final normalization
        auto final_norm = getFinalNorm();
        VALIDATE_POINTER(final_norm, "final norm");

        // Apply final RMSNorm using PipelineBase::rms_norm()
        TRY_OP(rms_norm(
            current_hidden_.get(), // input
            final_norm.get(),      // weight
            current_hidden_.get(), // output (in-place)
            effective_seq_len, d_model_,
            1e-6f, // epsilon
            "FINAL_NORM", device_idx_));

        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_final_norm");

        // LM head projection (batched)
        if (!lm_head_batch(current_hidden_.get(), effective_seq_len))
        {
            LOG_ERROR("LM head batch failed");
            return false;
        }

        // Update per-sequence positions for next incremental decode step
        for (int b = 0; b < batch_size_; ++b)
        {
            current_positions_[b] += sequence_lengths_[b]; // Increment by actual sequence length
        }

        return true;
    }

    bool Qwen2Pipeline::transformer_layer(int layer_idx, int effective_seq_len)
    {
        LOG_TRACE("Processing layer " << layer_idx);

        // Get layer weights (loaded lazily on first access)
        auto &layer = getLayerWeights(layer_idx);

        // Attention block
        if (!attention_block(layer, layer_idx, effective_seq_len))
        {
            LOG_ERROR("Attention block failed in layer " << layer_idx);
            return false;
        }

        // FFN block
        if (!ffn_block(layer, layer_idx, effective_seq_len))
        {
            LOG_ERROR("FFN block failed in layer " << layer_idx);
            return false;
        }

        return true;
    }

    // =============================================================================
    // Phase 5: Device Orchestration - Lazy Weight Transfers
    // =============================================================================

    bool Qwen2Pipeline::ensureAttentionWeightsOnDevice(const LayerWeights &layer, int target_device)
    {
        // CPU path: no transfer needed
        // Note: target_device < 0 is legacy CPU convention
        // target_device >= 0 can also be CPU if DeviceManager enumerated it that way
        if (target_device < 0)
        {
            return true;
        }

        // Check if target device is actually a CPU (DeviceManager style)
        auto &dm = DeviceManager::instance();
        if (static_cast<size_t>(target_device) < dm.devices().size())
        {
            const auto &dev = dm.devices()[target_device];
            if (dev.type == ComputeBackendType::CPU_OPENBLAS ||
                dev.type == ComputeBackendType::CPU_MKL)
            {
                // CPU device - no GPU transfer needed
                return true;
            }
        }

        // Lazy transfer attention weights to GPU
        // ensureOnDevice() is a no-op if already on target device
        bool success = true;

        if (layer.wq && !layer.wq->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer wq to device " << target_device);
            success = false;
        }
        if (layer.wk && !layer.wk->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer wk to device " << target_device);
            success = false;
        }
        if (layer.wv && !layer.wv->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer wv to device " << target_device);
            success = false;
        }
        if (layer.wo && !layer.wo->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer wo to device " << target_device);
            success = false;
        }
        if (layer.attn_norm && !layer.attn_norm->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer attn_norm to device " << target_device);
            success = false;
        }

        // Optional biases
        if (layer.q_bias && !layer.q_bias->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer q_bias to device " << target_device);
            success = false;
        }
        if (layer.k_bias && !layer.k_bias->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer k_bias to device " << target_device);
            success = false;
        }
        if (layer.v_bias && !layer.v_bias->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer v_bias to device " << target_device);
            success = false;
        }

        return success;
    }

    bool Qwen2Pipeline::ensureFFNWeightsOnDevice(const LayerWeights &layer, int target_device)
    {
        // CPU path: no transfer needed
        // Note: target_device < 0 is legacy CPU convention
        // target_device >= 0 can also be CPU if DeviceManager enumerated it that way
        if (target_device < 0)
        {
            return true;
        }

        // Check if target device is actually a CPU (DeviceManager style)
        auto &dm = DeviceManager::instance();
        if (static_cast<size_t>(target_device) < dm.devices().size())
        {
            const auto &dev = dm.devices()[target_device];
            if (dev.type == ComputeBackendType::CPU_OPENBLAS ||
                dev.type == ComputeBackendType::CPU_MKL)
            {
                // CPU device - no GPU transfer needed
                return true;
            }
        }

        // Lazy transfer FFN weights to GPU
        // ensureOnDevice() is a no-op if already on target device
        bool success = true;

        if (layer.gate_proj && !layer.gate_proj->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer gate_proj to device " << target_device);
            success = false;
        }
        if (layer.up_proj && !layer.up_proj->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer up_proj to device " << target_device);
            success = false;
        }
        if (layer.down_proj && !layer.down_proj->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer down_proj to device " << target_device);
            success = false;
        }
        if (layer.ffn_norm && !layer.ffn_norm->ensureOnDevice(target_device))
        {
            LOG_ERROR("Failed to transfer ffn_norm to device " << target_device);
            success = false;
        }

        return success;
    }

    bool Qwen2Pipeline::attention_block(const LayerWeights &layer, int layer_idx, int effective_seq_len)
    {
        // Determine execution device based on weight placement
        int attn_device = placement_map_ ? getWeightDevice("attn_q", -1) : device_idx_;

        // Phase 5: Lazy transfer weights to target device (no-op if already there or CPU)
        if (!ensureAttentionWeightsOnDevice(layer, attn_device))
        {
            LOG_ERROR("Failed to ensure attention weights on device " << attn_device);
            return false;
        }

        // Prepare input activation for execution on attention device
        TensorBase *input_hidden = current_hidden_.get();
        if (placement_map_ && current_hidden_->device_index() != attn_device)
        {
            input_hidden = prepareActivationForDevice(current_hidden_.get(), attn_device, "attention_input");
            if (!input_hidden)
            {
                LOG_ERROR("Failed to prepare activation for attention device");
                return false;
            }
        }

        // Get device-appropriate buffers
        auto &buffers = placement_map_ ? getBuffersForDevice(attn_device) : activation_buffers_;
        std::string layer_prefix = "layer" + std::to_string(layer_idx);

        // Save residual for later
        LOG_TRACE("[attention_block] layer=" << layer_idx
                                             << " residual=" << reinterpret_cast<void *>(buffers.residual.get())
                                             << " normalized=" << reinterpret_cast<void *>(buffers.normalized.get()));
        TRY_OP(save_residual(input_hidden, buffers.residual.get(), effective_seq_len, d_model_));

        // 1. Pre-attention RMSNorm
        TRY_OP(rms_norm(
            buffers.residual.get(), layer.attn_norm.get(), buffers.normalized.get(),
            effective_seq_len, d_model_, 1e-6f,
            layer_prefix + "_ATTENTION_NORM", attn_device));

        // DEBUG: Print normalized output for layer 0
        if (layer_idx == 0 && (!mpi_ctx_ || mpi_ctx_->rank() == 0))
        {
            const float *norm_data = buffers.normalized->fp32_data(); // Explicit dequantization for debug output
            int last_pos = effective_seq_len - 1;
            LOG_TRACE("Layer 0 after RMSNorm (seq_len=" << effective_seq_len
                                                        << ", last_pos=" << last_pos << ", tensor_type=" << static_cast<int>(buffers.normalized->native_type())
                                                        << ", first 10 values): " << norm_data[last_pos * d_model_ + 0]
                                                        << ", " << norm_data[last_pos * d_model_ + 1]
                                                        << ", " << norm_data[last_pos * d_model_ + 2]
                                                        << ", " << norm_data[last_pos * d_model_ + 3]
                                                        << ", " << norm_data[last_pos * d_model_ + 4]);
        }

        // 2. Fused Q/K/V projections
        if (!layer.qkv_fused)
        {
            layer.qkv_fused = std::make_unique<FusedGEMM>(
                layer.wq.get(), layer.wk.get(), layer.wv.get());
        }

        // Extract bias pointers (nullptr if model doesn't have biases)
        const float *q_bias_ptr = nullptr;
        const float *k_bias_ptr = nullptr;
        const float *v_bias_ptr = nullptr;

        if (layer.q_bias)
        {
            auto *q_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.q_bias.get());
            if (q_bias_fp32)
                q_bias_ptr = q_bias_fp32->data();
        }
        if (layer.k_bias)
        {
            auto *k_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.k_bias.get());
            if (k_bias_fp32)
                k_bias_ptr = k_bias_fp32->data();
        }
        if (layer.v_bias)
        {
            auto *v_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.v_bias.get());
            if (v_bias_fp32)
                v_bias_ptr = v_bias_fp32->data();
        }

        // Execute Q/K/V projection based on activation precision
        if (config_.activation_precision == ActivationPrecision::Q8_1)
        {
            // Q8_1 path: Output directly to Q8_1 blocks with optional bias
            // Bias is added to FP32 GEMM result before Q8_1 requantization

            // Get Q8_1 block pointers from activation tensors
            auto *Q_q8_1 = dynamic_cast<Q8_1Tensor *>(buffers.Q.get());
            auto *K_q8_1 = dynamic_cast<Q8_1Tensor *>(buffers.K.get());
            auto *V_q8_1 = dynamic_cast<Q8_1Tensor *>(buffers.V.get());

            if (!Q_q8_1 || !K_q8_1 || !V_q8_1)
            {
                LOG_ERROR("Q8_1 activation precision configured but buffers are not Q8_1Tensor");
                return false;
            }

            // Check if normalized buffer is Q8_1 - if so, use Q8_1→Q8_1 path to avoid double quantization
            auto *normalized_q8_1 = dynamic_cast<Q8_1Tensor *>(buffers.normalized.get());
            if (normalized_q8_1)
            {
                // Pure Q8_1 path: Q8_1 input → Q8_1 output (no double quantization)
                LOG_TRACE("Layer " << layer_idx << " QKV: Using Q8_1→Q8_1 path (avoiding double quantization)");
                VALIDATE_OP(layer.qkv_fused->execute_q8_1_to_q8_1(
                                normalized_q8_1->q8_1_blocks(),
                                Q_q8_1->mutable_q8_1_blocks(),
                                K_q8_1->mutable_q8_1_blocks(),
                                V_q8_1->mutable_q8_1_blocks(),
                                q_bias_ptr, k_bias_ptr, v_bias_ptr,
                                effective_seq_len,
                                n_heads_ * head_dim_, n_kv_heads_ * head_dim_,
                                d_model_,
                                mpi_ctx_.get(), attn_device),
                            "Fused Q/K/V projection (Q8_1→Q8_1)");
            }
            else
            {
                // FP32 input → Q8_1 output path
                VALIDATE_OP(layer.qkv_fused->execute_to_q8_1(
                                buffers.normalized->data(),
                                Q_q8_1->mutable_q8_1_blocks(),
                                K_q8_1->mutable_q8_1_blocks(),
                                V_q8_1->mutable_q8_1_blocks(),
                                q_bias_ptr, k_bias_ptr, v_bias_ptr,
                                effective_seq_len,
                                n_heads_ * head_dim_, n_kv_heads_ * head_dim_,
                                d_model_,
                                mpi_ctx_.get(), attn_device),
                            "Fused Q/K/V projection (FP32→Q8_1)");
            }
        }
        else
        {
            // FP32/BF16/FP16 path: Standard execute
            VALIDATE_OP(layer.qkv_fused->execute(
                            buffers.normalized->data(),
                            buffers.Q->mutable_data(),
                            buffers.K->mutable_data(),
                            buffers.V->mutable_data(),
                            q_bias_ptr, k_bias_ptr, v_bias_ptr,
                            effective_seq_len,
                            n_heads_ * head_dim_, n_kv_heads_ * head_dim_, n_kv_heads_ * head_dim_,
                            d_model_,
                            mpi_ctx_.get(), attn_device),
                        "Fused Q/K/V projection");
        }

        // Capture Q/K/V projections
        capture_snapshot(layer_prefix + "_Q_PROJECTION", buffers.Q.get(), effective_seq_len, n_heads_ * head_dim_);
        capture_snapshot(layer_prefix + "_K_PROJECTION", buffers.K.get(), effective_seq_len, n_kv_heads_ * head_dim_);
        capture_snapshot(layer_prefix + "_V_PROJECTION", buffers.V.get(), effective_seq_len, n_kv_heads_ * head_dim_);

        // DEBUG: Print Q projection output for layer 0 (works for both FP32 and Q8_1)
        if (layer_idx == 0 && (!mpi_ctx_ || mpi_ctx_->rank() == 0))
        {
            const float *q_data = buffers.Q->fp32_data(); // Explicit dequantization for debug output
            int last_pos = effective_seq_len - 1;
            int q_dim = n_heads_ * head_dim_;
            LOG_TRACE("Layer 0 Q projection (tensor_type=" << static_cast<int>(buffers.Q->native_type())
                                                           << ", last_pos=" << last_pos << ", first 5 values): "
                                                           << q_data[last_pos * q_dim + 0] << ", " << q_data[last_pos * q_dim + 1]
                                                           << ", " << q_data[last_pos * q_dim + 2] << ", " << q_data[last_pos * q_dim + 3]
                                                           << ", " << q_data[last_pos * q_dim + 4]);

            // Also print K and V
            const float *k_data = buffers.K->fp32_data();
            const float *v_data = buffers.V->fp32_data();
            int kv_dim = n_kv_heads_ * head_dim_;
            LOG_TRACE("Layer 0 K projection (first 5 values): "
                      << k_data[last_pos * kv_dim + 0] << ", " << k_data[last_pos * kv_dim + 1]
                      << ", " << k_data[last_pos * kv_dim + 2] << ", " << k_data[last_pos * kv_dim + 3]
                      << ", " << k_data[last_pos * kv_dim + 4]);
            LOG_TRACE("Layer 0 V projection (first 5 values): "
                      << v_data[last_pos * kv_dim + 0] << ", " << v_data[last_pos * kv_dim + 1]
                      << ", " << v_data[last_pos * kv_dim + 2] << ", " << v_data[last_pos * kv_dim + 3]
                      << ", " << v_data[last_pos * kv_dim + 4]);
        }

        // 3. Apply RoPE to Q and K
        std::vector<int> position_ids(effective_seq_len);
        for (int b = 0; b < batch_size_; ++b)
        {
            int actual_len = (batch_size_ == 1) ? padded_seq_len_ : sequence_lengths_[b];
            for (int i = 0; i < padded_seq_len_; ++i)
            {
                position_ids[b * padded_seq_len_ + i] = (i < actual_len) ? current_positions_[b] + i : -1;
            }
        }

        TRY_OP(apply_rope(
            buffers.Q.get(), buffers.K.get(), position_ids.data(),
            effective_seq_len, n_heads_, n_kv_heads_, head_dim_,
            model_ctx_->model().rope_theta,
            layer_prefix, attn_device));

        // DEBUG: Print Q after RoPE for layer 0 (works for both FP32 and Q8_1)
        if (layer_idx == 0 && (!mpi_ctx_ || mpi_ctx_->rank() == 0))
        {
            const float *q_data = buffers.Q->fp32_data(); // Explicit dequantization for debug output
            int last_pos = effective_seq_len - 1;
            int q_dim = n_heads_ * head_dim_;
            LOG_TRACE("Layer 0 Q after RoPE (tensor_type=" << static_cast<int>(buffers.Q->native_type())
                                                           << ", first 5 values): "
                                                           << q_data[last_pos * q_dim + 0] << ", " << q_data[last_pos * q_dim + 1]
                                                           << ", " << q_data[last_pos * q_dim + 2] << ", " << q_data[last_pos * q_dim + 3]
                                                           << ", " << q_data[last_pos * q_dim + 4]);
        }

        // 4. Update KV cache and compute attention
        // For single sequence: append new K/V to cache, then use full cache for attention
        // For batched: append per-sequence K/V, then use per-sequence cached K/V
        TensorBase *K_for_attention = buffers.K.get();
        TensorBase *V_for_attention = buffers.V.get();
        int kv_seq_len = effective_seq_len; // Default: K/V has same seq_len as Q
        bool use_kv_cache = false;

        // Debug: Log KV cache state for layer 0
        if (layer_idx == 0 && mpi_ctx_ && mpi_ctx_->rank() == 0)
        {
            LOG_TRACE("KV Cache state: kv_cache_=" << (kv_cache_ ? "yes" : "no")
                                                   << ", batch_size_=" << batch_size_
                                                   << ", effective_seq_len=" << effective_seq_len);
        }

        if (kv_cache_ && batch_size_ == 1)
        {
            // Single-sequence KV cache path
            // December 2025: Typed KV cache handles format conversion internally.
            // Pass the K/V buffers directly with explicit token count.
            // For Q8_1 cache: stores Q8_1 data natively (no dequant/requant cycle!)
            // For FP32 cache: stores FP32 data (dequantizes Q8_1 input if needed)

            // Append K/V to cache - the typed cache handles precision conversion
            // Pass effective_seq_len explicitly since activation buffers may be larger
            if (!kv_cache_->append_kv(layer_idx, buffers.K.get(), buffers.V.get(), effective_seq_len))
            {
                LOG_ERROR("Failed to append K/V to cache for layer " << layer_idx);
                return false;
            }

            // Get full cached K/V for attention
            auto cached_K = kv_cache_->get_k(layer_idx);
            auto cached_V = kv_cache_->get_v(layer_idx);
            int cached_tokens = kv_cache_->get_cached_tokens(layer_idx);

            if (cached_K && cached_V && cached_tokens > 0)
            {
                K_for_attention = cached_K.get();
                V_for_attention = cached_V.get();
                kv_seq_len = cached_tokens;
                use_kv_cache = true;

                // Debug: Log KV cache usage for layer 0
                if (layer_idx == 0 && mpi_ctx_ && mpi_ctx_->rank() == 0)
                {
                    LOG_TRACE("KV Cache using cached K/V: cached_tokens=" << cached_tokens
                                                                          << ", Q tokens=" << effective_seq_len
                                                                          << ", cache precision=" << static_cast<int>(kv_cache_->precision()));
                }

                LOG_TRACE("[KVCache] Layer " << layer_idx << ": using cached K/V with "
                                             << cached_tokens << " tokens for attention (Q has " << effective_seq_len << " tokens)");
            }
        }

        // Compute attention + Wo projection
        // Two paths: fused kernel (when enabled) or unfused (separate attention + GEMM)

        if (fused_attn_wo_kernel_)
        {
            // =================================================================
            // FUSED PATH: Attention + Wo in single kernel
            // Benefits:
            // 1. Eliminates context quantization round-trip (FP32 → Q8_1 → FP32)
            // 2. Better cache locality (context stays in registers through projection)
            // 3. Single pass over V and Wo (reduced memory bandwidth)
            // =================================================================

            // The fused kernel takes Q8_1 inputs (Q, K, V) and Q8_1 Wo weight,
            // outputs directly to FP32 attn_proj (skipping intermediate attn_output)
            int position_offset = 0;
            if (use_kv_cache && kv_seq_len > effective_seq_len)
            {
                // Decode mode: position offset for causal mask
                position_offset = kv_seq_len - effective_seq_len;
            }

            bool fused_success = fused_attn_wo_kernel_->compute(
                buffers.Q.get(),
                K_for_attention,
                V_for_attention,
                layer.wo.get(),
                buffers.attn_proj.get(), // Output directly to projection buffer (skip attn_output)
                effective_seq_len,
                kv_seq_len,
                /*causal=*/true,
                position_offset);

            if (!fused_success)
            {
                LOG_ERROR("Fused attention+Wo kernel failed at layer " << layer_idx);
                return false;
            }

            capture_snapshot(layer_prefix + "_ATTENTION_CONTEXT", buffers.attn_proj.get(),
                             effective_seq_len, d_model_);
            capture_snapshot(layer_prefix + "_ATTENTION_OUTPUT", buffers.attn_proj.get(),
                             effective_seq_len, d_model_);
        }
        else
        {
            // =================================================================
            // UNFUSED PATH: Separate attention + Wo GEMM (original flow)
            // =================================================================

            // Compute attention
            if (use_kv_cache && kv_seq_len != effective_seq_len)
            {
                // Decode path: Q has fewer tokens than cached K/V
                // Use compute_attention_with_kv_cache which handles asymmetric lengths
                TRY_OP(compute_attention_with_kv_cache(
                    buffers.Q.get(), K_for_attention, V_for_attention, buffers.attn_output.get(),
                    effective_seq_len, kv_seq_len, n_heads_, n_kv_heads_, head_dim_,
                    batch_size_, sequence_lengths_,
                    /*causal=*/true, layer_prefix + "_ATTENTION_CONTEXT"));
            }
            else
            {
                // Prefill path or no KV cache: Q/K/V have same length
                TRY_OP(compute_attention(
                    buffers.Q.get(), K_for_attention, V_for_attention, buffers.attn_output.get(),
                    effective_seq_len, n_heads_, n_kv_heads_, head_dim_,
                    batch_size_, sequence_lengths_, padded_seq_len_,
                    /*causal=*/true, layer_prefix + "_ATTENTION_CONTEXT"));
            }

            // DEBUG: Print attention output for layer 0
            if (layer_idx == 0 && mpi_ctx_ && mpi_ctx_->rank() == 0)
            {
                auto *attn_out_fp32 = dynamic_cast<FP32Tensor *>(buffers.attn_output.get());
                if (attn_out_fp32)
                {
                    const float *attn_data = attn_out_fp32->data();
                    int last_pos = effective_seq_len - 1;
                    int attn_dim = n_heads_ * head_dim_;
                    std::ostringstream oss;
                    oss << "Layer 0 attention output (last position, first 10 values): ";
                    for (int i = 0; i < std::min(10, attn_dim); ++i)
                    {
                        oss << attn_data[last_pos * attn_dim + i] << ", ";
                    }
                    LOG_TRACE(oss.str());
                }
            }

            // 5. Output projection (row-parallel: needs allreduce for tensor parallelism)
            // Wo projection: [seq, n_heads*head_dim] @ [d_model, n_heads*head_dim].T -> [seq, d_model]
            // In tensor parallelism: attention output is partitioned by heads, so Wo sees partitioned input
            TRY_OP(project_row_parallel(
                buffers.attn_output.get(), layer.wo.get(), buffers.attn_proj.get(),
                effective_seq_len, d_model_, n_heads_ * head_dim_,
                layer_prefix + "_ATTENTION_OUTPUT", attn_device));
        }

        // DEBUG: Print output projection for layer 0
        if (layer_idx == 0 && mpi_ctx_ && mpi_ctx_->rank() == 0)
        {
            auto *attn_proj_fp32 = dynamic_cast<FP32Tensor *>(buffers.attn_proj.get());
            if (attn_proj_fp32)
            {
                const float *proj_data = attn_proj_fp32->data();
                int last_pos = effective_seq_len - 1;
                std::ostringstream oss;
                oss << "Layer 0 output projection (last position, first 10 values): ";
                for (int i = 0; i < std::min(10, d_model_); ++i)
                {
                    oss << proj_data[last_pos * d_model_ + i] << ", ";
                }
                LOG_TRACE(oss.str());
            }
        }

        // 6. Residual connection
        TRY_OP(add_residual(
            buffers.residual.get(), buffers.attn_proj.get(), current_hidden_.get(),
            batch_size_, padded_seq_len_, d_model_,
            sequence_lengths_,
            layer_prefix + "_ATTENTION_RESIDUAL"));

        // Update device index if using heterogeneous execution
        if (placement_map_)
        {
            current_hidden_->set_device(attn_device);
        }

        return true;
    }

    bool Qwen2Pipeline::ffn_block(const LayerWeights &layer, int layer_idx, int effective_seq_len)
    {
        // Determine execution device based on weight placement
        int ffn_device = placement_map_ ? getWeightDevice("ffn_gate", -1) : device_idx_;

        // Phase 5: Lazy transfer weights to target device (no-op if already there or CPU)
        if (!ensureFFNWeightsOnDevice(layer, ffn_device))
        {
            LOG_ERROR("Failed to ensure FFN weights on device " << ffn_device);
            return false;
        }

        // Prepare input activation for execution on FFN device
        TensorBase *input_hidden = current_hidden_.get();
        if (placement_map_ && current_hidden_->device_index() != ffn_device)
        {
            input_hidden = prepareActivationForDevice(current_hidden_.get(), ffn_device, "ffn_input");
            if (!input_hidden)
            {
                LOG_ERROR("Failed to prepare activation for FFN device");
                return false;
            }
        }

        // Get device-appropriate buffers
        auto &buffers = placement_map_ ? getBuffersForDevice(ffn_device) : activation_buffers_;
        std::string layer_prefix = "layer" + std::to_string(layer_idx);

        // Save residual for later
        TRY_OP(save_residual(input_hidden, buffers.residual.get(), effective_seq_len, d_model_));

        // Capture FFN input residual (before FFN processing)
        capture_snapshot(layer_prefix + "_FFN_INPUT_RESIDUAL", buffers.residual.get(), effective_seq_len, d_model_);

        // 1. Pre-FFN RMSNorm
        TRY_OP(rms_norm(
            buffers.residual.get(), layer.ffn_norm.get(), buffers.normalized.get(),
            effective_seq_len, d_model_, 1e-6f,
            layer_prefix + "_FFN_NORM", ffn_device));

        // 2. Fused Gate/Up projections
        // Use d_ff_local_ for column-parallel (each rank produces local output)
        int ffn_output_dim = ffn_column_parallel_ ? d_ff_local_ : d_ff_;

        if (!layer.gate_up_fused)
        {
            layer.gate_up_fused = std::make_unique<FusedGEMM>(
                layer.gate_proj.get(), layer.up_proj.get());
        }

        // For Q8_1 activation mode, use Q8_1 output path
        auto *gate_q8_1 = dynamic_cast<Q8_1Tensor *>(buffers.gate.get());
        auto *up_q8_1 = dynamic_cast<Q8_1Tensor *>(buffers.up.get());

        if (gate_q8_1 && up_q8_1)
        {
            // Check if normalized buffer is Q8_1 - if so, use Q8_1→Q8_1 path to avoid double quantization
            auto *normalized_q8_1 = dynamic_cast<Q8_1Tensor *>(buffers.normalized.get());
            if (normalized_q8_1)
            {
                // Pure Q8_1 path: Q8_1 input → Q8_1 output (no double quantization)
                VALIDATE_OP(layer.gate_up_fused->execute_q8_1_to_q8_1(
                                normalized_q8_1->q8_1_blocks(),
                                {{gate_q8_1->mutable_q8_1_blocks(), nullptr, ffn_output_dim, "gate"},
                                 {up_q8_1->mutable_q8_1_blocks(), nullptr, ffn_output_dim, "up"}},
                                effective_seq_len, d_model_,
                                mpi_ctx_.get(), ffn_device),
                            "Fused Gate/Up projection (Q8_1→Q8_1)");
            }
            else
            {
                // FP32 input → Q8_1 output path
                VALIDATE_OP(layer.gate_up_fused->execute_to_q8_1(
                                buffers.normalized->data(),
                                {{gate_q8_1->mutable_q8_1_blocks(), nullptr, ffn_output_dim, "gate"},
                                 {up_q8_1->mutable_q8_1_blocks(), nullptr, ffn_output_dim, "up"}},
                                effective_seq_len, d_model_,
                                mpi_ctx_.get(), ffn_device),
                            "Fused Gate/Up projection (FP32→Q8_1)");
            }
        }
        else
        {
            // FP32 output path: use standard execute with FP32 output pointers
            VALIDATE_OP(layer.gate_up_fused->execute(
                            buffers.normalized->data(),
                            {{buffers.gate->mutable_data(), nullptr, ffn_output_dim, "gate"},
                             {buffers.up->mutable_data(), nullptr, ffn_output_dim, "up", nullptr, false}},
                            effective_seq_len, d_model_,
                            mpi_ctx_.get(), ffn_device),
                        "Fused Gate/Up projection (FP32)");
        }

        // Capture gate/up projections (with actual output dimension)
        capture_snapshot(layer_prefix + "_FFN_GATE", buffers.gate.get(), effective_seq_len, ffn_output_dim);
        capture_snapshot(layer_prefix + "_FFN_UP", buffers.up.get(), effective_seq_len, ffn_output_dim);

        // 3. Apply SwiGLU (operates on local FFN dimension)
        TRY_OP(swiglu(
            buffers.gate.get(), buffers.up.get(), buffers.up.get(),
            effective_seq_len, ffn_output_dim,
            layer_prefix + "_FFN_SWIGLU", ffn_device));

        // 4. Down projection
        // With column-parallel Gate/Up: input is [seq, d_ff_local], Down weight is [d_model, d_ff_local]
        // Use project_column_parallel which does GEMM + allreduce-sum
        if (ffn_column_parallel_)
        {
            TRY_OP(project_column_parallel(
                buffers.up.get(), layer.down_proj.get(), buffers.ffn_output.get(),
                effective_seq_len, d_model_, ffn_output_dim,
                layer_prefix + "_FFN_DOWN", ffn_device));
        }
        else
        {
            // Non-sharded path: standard row-parallel (or full GEMM)
            TRY_OP(project_row_parallel(
                buffers.up.get(), layer.down_proj.get(), buffers.ffn_output.get(),
                effective_seq_len, d_model_, d_ff_,
                layer_prefix + "_FFN_DOWN", ffn_device));
        }

        // 5. Residual connection
        TRY_OP(add_residual(
            buffers.residual.get(), buffers.ffn_output.get(), current_hidden_.get(),
            batch_size_, padded_seq_len_, d_model_,
            sequence_lengths_,
            layer_prefix + "_FFN_RESIDUAL"));

        // Update device index if using heterogeneous execution
        if (placement_map_)
        {
            current_hidden_->set_device(ffn_device);
        }

        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_ffn_residual");

        return true;
    }

    // =============================================================================
    // Lazy Weight Accessors
    // =============================================================================

    std::shared_ptr<TensorBase> Qwen2Pipeline::getEmbeddingTable()
    {
        if (!embedding_table_)
        {
            auto raw_embed = model_ctx_->getWeight("token_embd.weight", device_idx_);
            if (!raw_embed)
            {
                LOG_ERROR("[Qwen2Pipeline] Failed to load embedding table");
                return nullptr;
            }

            // GGUF stores embedding as [d_model, vocab_size] but we need [vocab_size, d_model]
            // for efficient row-wise lookup. Transpose the dequantized data.
            const auto &shape = raw_embed->shape();

            // Debug: Log actual shape
            std::stringstream shape_str;
            shape_str << "[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i > 0)
                    shape_str << ", ";
                shape_str << shape[i];
            }
            shape_str << "]";
            LOG_DEBUG("[Qwen2Pipeline] Raw embedding shape: " << shape_str.str()
                                                              << ", d_model_=" << d_model_ << ", vocab_size_=" << vocab_size_);

            if (shape.size() == 2 && shape[0] == static_cast<size_t>(d_model_) && shape[1] == static_cast<size_t>(vocab_size_))
            {
                LOG_DEBUG("[Qwen2Pipeline] Transposing embedding table from [" << shape[0] << ", " << shape[1]
                                                                               << "] to [" << shape[1] << ", " << shape[0] << "]");

                // Create FP32 tensor with transposed shape [vocab_size, d_model]
                embedding_table_ = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(vocab_size_), static_cast<size_t>(d_model_)},
                    device_idx_);

                // Transpose: dst[v, d] = src[d, v]
                const float *src = raw_embed->data();
                float *dst = embedding_table_->mutable_data();

#pragma omp parallel for
                for (int v = 0; v < vocab_size_; ++v)
                {
                    for (int d = 0; d < d_model_; ++d)
                    {
                        dst[v * d_model_ + d] = src[d * vocab_size_ + v];
                    }
                }
            }
            else
            {
                // Already in correct shape or unexpected shape - use as-is
                embedding_table_ = raw_embed;
            }
        }
        return embedding_table_;
    }

    std::shared_ptr<TensorBase> Qwen2Pipeline::getFinalNorm()
    {
        if (!final_norm_)
        {
            final_norm_ = model_ctx_->getWeight("output_norm.weight", device_idx_);
        }
        return final_norm_;
    }

    std::shared_ptr<TensorBase> Qwen2Pipeline::getLMHead()
    {
        if (!lm_head_)
        {
            // Check if model has explicit output.weight (not tied embeddings)
            if (model_ctx_->hasTensor("output.weight"))
            {
                lm_head_ = model_ctx_->getWeight("output.weight", device_idx_);
            }
            else
            {
                // Model uses tied embeddings - LM head shares weights with token embeddings
                // Use the transposed embedding table (already [vocab_size, d_model])
                LOG_INFO("[Qwen2Pipeline] Using tied embeddings for LM head (no output.weight tensor)");
                lm_head_ = getEmbeddingTable();

                if (lm_head_)
                {
                    const auto &shape = lm_head_->shape();
                    LOG_DEBUG("[Qwen2Pipeline] Using embedding table as LM head: ["
                              << shape[0] << ", " << shape[1] << "]");
                }
            }
        }
        return lm_head_;
    }

    Qwen2Pipeline::LayerWeights &Qwen2Pipeline::getLayerWeights(int layer_idx)
    {
        auto &layer = layers_[layer_idx];
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        // Lazy load on first access
        if (!layer.wq)
        {
            layer.wq = model_ctx_->getWeight(prefix + "attn_q.weight", device_idx_);

            // DEBUG: Log weight type and shape
            if (layer.wq)
            {
                const auto &shape = layer.wq->shape();
                std::string type_name;
                switch (layer.wq->native_type())
                {
                case TensorType::FP32:
                    type_name = "FP32";
                    break;
                case TensorType::FP16:
                    type_name = "FP16";
                    break;
                case TensorType::Q4_0:
                    type_name = "Q4_0";
                    break;
                case TensorType::Q6_K:
                    type_name = "Q6_K";
                    break;
                case TensorType::Q8_0:
                    type_name = "Q8_0";
                    break;
                default:
                    type_name = "UNKNOWN";
                    break;
                }
                LOG_TRACE("[DEBUG] Layer " << layer_idx << " wq: type=" << type_name
                                           << ", shape=[" << shape[0] << ", " << shape[1] << "]");
            }

            layer.wk = model_ctx_->getWeight(prefix + "attn_k.weight", device_idx_);
            layer.wv = model_ctx_->getWeight(prefix + "attn_v.weight", device_idx_);
            layer.wo = model_ctx_->getWeight(prefix + "attn_output.weight", device_idx_);
            layer.attn_norm = model_ctx_->getWeight(prefix + "attn_norm.weight", device_idx_);
            layer.gate_proj = model_ctx_->getWeight(prefix + "ffn_gate.weight", device_idx_);
            layer.up_proj = model_ctx_->getWeight(prefix + "ffn_up.weight", device_idx_);
            layer.down_proj = model_ctx_->getWeight(prefix + "ffn_down.weight", device_idx_);
            layer.ffn_norm = model_ctx_->getWeight(prefix + "ffn_norm.weight", device_idx_);

            // Load Q/K/V biases (optional - not all models have them)
            layer.q_bias = model_ctx_->getWeight(prefix + "attn_q.bias", device_idx_);
            layer.k_bias = model_ctx_->getWeight(prefix + "attn_k.bias", device_idx_);
            layer.v_bias = model_ctx_->getWeight(prefix + "attn_v.bias", device_idx_);

            if (layer.q_bias)
            {
                LOG_TRACE("[DEBUG] Layer " << layer_idx << " has Q/K/V biases");
            }
        }

        return layer;
    }

    std::shared_ptr<TensorBase> Qwen2Pipeline::get_layer_weight(
        int layer_idx, const std::string &weight_name)
    {
        DEBUG_ASSERT_RANGE(layer_idx, 0, n_layers_, "Invalid layer index");

        const auto &layer = layers_[layer_idx];

        if (weight_name == "wq")
            return layer.wq;
        if (weight_name == "wk")
            return layer.wk;
        if (weight_name == "wv")
            return layer.wv;
        if (weight_name == "wo")
            return layer.wo;
        if (weight_name == "gate")
            return layer.gate_proj;
        if (weight_name == "up")
            return layer.up_proj;
        if (weight_name == "down")
            return layer.down_proj;
        if (weight_name == "attn_norm")
            return layer.attn_norm;
        if (weight_name == "ffn_norm")
            return layer.ffn_norm;

        LOG_ERROR("Unknown weight name: " << weight_name);
        return nullptr;
    }

    // =============================================================================
    // Batch-Aware Helper Methods
    // =============================================================================

    bool Qwen2Pipeline::embedding_batch(const std::vector<std::vector<int>> &token_batches, TensorBase *output)
    {
        KERNEL_PROFILE_SCOPE(KernelType::EMBEDDING);

        auto embed_table = getEmbeddingTable();
        VALIDATE_POINTER(embed_table, "embedding table");

        // Use EmbeddingOp for typed tensor support (FP32 and Q8_1)
        EmbeddingOp embedding_op;
        if (!embedding_op(embed_table.get(), token_batches, padded_seq_len_, d_model_, output))
        {
            LOG_ERROR("EmbeddingOp failed");
            return false;
        }

        // Debug logging
        static const bool debug_embedding = std::getenv("LLAMINAR_DEBUG_EMBEDDING") != nullptr;
        if (debug_embedding && mpi_ctx_ && mpi_ctx_->rank() == 0)
        {
            const float *embed_data = embed_table->data();
            int test_tokens[] = {785, 9906};
            for (int token_id : test_tokens)
            {
                const float *emb = embed_data + token_id * d_model_;
                LOG_DEBUG("[DEBUG] Embedding table for token " << token_id << ":");
                for (int i = 0; i < 20; ++i)
                {
                    LOG_DEBUG("  emb[" << i << "] = " << emb[i]);
                }
            }
        }

        return true;
    }

    bool Qwen2Pipeline::lm_head_batch(TensorBase *hidden, int effective_seq_len)
    {
        // Allocate logits buffer if needed
        // CRITICAL: Logits must ALWAYS be FP32 regardless of activation precision.
        // Q8_1/BF16/FP16 logits would lose precision needed for accurate sampling
        // (logits can span a wide range, e.g., -100 to +100, which Q8_1 cannot represent accurately).
        if (!logits_buffer_ || static_cast<int>(logits_buffer_->shape()[0]) != effective_seq_len)
        {
            logits_buffer_ = tensor_factory_->createActivation(
                {static_cast<size_t>(effective_seq_len), static_cast<size_t>(vocab_size_)},
                ActivationPrecision::FP32, device_idx_); // Always FP32 for sampling accuracy
            LOG_TRACE("Allocated logits buffer (FP32): "
                      << effective_seq_len << " x " << vocab_size_ << " on device " << device_idx_);
        }

        VALIDATE_TENSOR(logits_buffer_, spec_logits(effective_seq_len), "logits_allocation");

        auto lm_head = getLMHead();
        VALIDATE_POINTER(lm_head, "LM head");

        // LM head projection using PipelineBase::project()
        // logits = hidden @ lm_head^T
        // hidden: [effective_seq_len, d_model], lm_head: [vocab_size, d_model]
        // output: [effective_seq_len, vocab_size]
        TRY_OP(project(
            hidden, lm_head.get(), logits_buffer_.get(),
            effective_seq_len, vocab_size_, d_model_,
            "LM_HEAD", device_idx_));

        // DEBUG: Check logits after LM head
        {
            const float *logits = logits_buffer_->data();
            float min_val = logits[0], max_val = logits[0];
            for (int i = 0; i < 100; ++i)
            {
                min_val = std::min(min_val, logits[i]);
                max_val = std::max(max_val, logits[i]);
            }
            LOG_TRACE("[LM_HEAD] After projection: logits[0:100] min=" << min_val
                                                                       << " max=" << max_val << " first=" << logits[0]);

            // Also check the hidden state input
            const float *hidden_data = hidden->fp32_data(); // Explicit dequantization for debug output
            float h_min = hidden_data[0], h_max = hidden_data[0];
            for (int i = 0; i < 100; ++i)
            {
                h_min = std::min(h_min, hidden_data[i]);
                h_max = std::max(h_max, hidden_data[i]);
            }
            LOG_TRACE("[LM_HEAD] Hidden input: hidden[0:100] min=" << h_min
                                                                   << " max=" << h_max << " first=" << hidden_data[0]);
        }

        VALIDATE_TENSOR(logits_buffer_, spec_logits(effective_seq_len), "after_lm_head");

        return true;
    }

    const float *Qwen2Pipeline::logits() const
    {
        // Return logits for LAST token of first sequence (what sampling needs)
        return getLastTokenLogits(0);
    }

    const float *Qwen2Pipeline::getLogits(int seq_idx) const
    {
        if (!logits_buffer_)
        {
            return nullptr;
        }

        DEBUG_ASSERT_RANGE(seq_idx, 0, batch_size_, "Invalid sequence index");

        // Return pointer to logits for requested sequence
        // Layout: [batch_size * padded_seq_len, vocab_size]
        // For sequence seq_idx, logits start at row (seq_idx * padded_seq_len)
        return logits_buffer_->data() + (seq_idx * padded_seq_len_ * vocab_size_);
    }

    const float *Qwen2Pipeline::getLastTokenLogits(int seq_idx) const
    {
        if (!logits_buffer_)
        {
            return nullptr;
        }

        DEBUG_ASSERT_RANGE(seq_idx, 0, batch_size_, "Invalid sequence index");

        // Return pointer to logits for the LAST token of the requested sequence
        // Layout: [batch_size * padded_seq_len, vocab_size]
        // For sequence seq_idx, the last token's logits are at row:
        //   (seq_idx * padded_seq_len) + (sequence_length - 1)
        // For single-sequence decode steps, padded_seq_len_ = 1, so last row = row 0
        int seq_length = (seq_idx < static_cast<int>(sequence_lengths_.size()))
                             ? sequence_lengths_[seq_idx]
                             : padded_seq_len_;
        int last_pos = seq_idx * padded_seq_len_ + (seq_length - 1);

        // Compute the actual buffer size from the logits buffer shape
        int logits_rows = static_cast<int>(logits_buffer_->shape()[0]);

        LOG_TRACE("[getLastTokenLogits] seq_idx=" << seq_idx
                                                  << ", padded_seq_len=" << padded_seq_len_
                                                  << ", seq_length=" << seq_length
                                                  << ", last_pos=" << last_pos
                                                  << ", logits_buffer_rows=" << logits_rows);

        // Sanity check: ensure last_pos is within buffer bounds
        if (last_pos >= logits_rows)
        {
            LOG_WARN("[getLastTokenLogits] last_pos=" << last_pos
                                                      << " exceeds logits_buffer_rows=" << logits_rows
                                                      << ", clamping to " << (logits_rows - 1));
            last_pos = logits_rows - 1;
        }

        return logits_buffer_->data() + (last_pos * vocab_size_);
    }

} // namespace llaminar2
