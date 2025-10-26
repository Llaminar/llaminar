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
#include "../PipelineFactory.h"
#include "../../loaders/ModelLoader.h"
#include "../../tensors/TensorFactory.h"
#include "../../utils/BatchPaddingUtils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <set>
#include <algorithm>

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
        // Factory doesn't have placement_map yet (Phase 4.2 integration)
        // Default batch_size=1 for backward compatibility
        return std::make_unique<Qwen2Pipeline>(model_ctx, mpi_ctx, device_idx, nullptr, config, 1);
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
            LOG_INFO("Warning: feed_forward_length not in metadata, using " << d_ff_);
        }

        LOG_INFO("Architecture: " << n_layers_ << " layers, "
                                  << d_model_ << " d_model, " << vocab_size_ << " vocab");
        LOG_INFO("Attention: " << n_heads_ << " heads, "
                               << n_kv_heads_ << " KV heads (GQA), " << head_dim_ << " head_dim");
        LOG_INFO("FFN: " << d_ff_ << " intermediate_size (SwiGLU)");

        // Weights are loaded lazily via getLayerWeight() and model_ctx_->getWeight()
        // Resize layer weights vector for lazy loading
        layers_.resize(n_layers_);

        // =============================================================================
        // Generic Initialization (PipelineBase handles device/MPI/KV cache setup)
        // =============================================================================

        initializeInfrastructure();

        // Override KV cache with batched version
        std::vector<int> attention_devices = detectAttentionDevices(n_layers_);
        kv_cache_batched_ = std::make_shared<BatchedKVCache>(
            n_layers_, batch_size_, config.max_seq_len, n_kv_heads_, head_dim_, attention_devices);
        LOG_INFO("Initialized batched KV cache: batch_size=" << batch_size_
                                                             << ", max_seq_len=" << config.max_seq_len);

        LOG_INFO("Pipeline initialized (weights loaded on-demand)");
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

        // Residual (d_model) - sized for batch
        buffers.residual = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx);

        // Normalization buffer (shared across attention and FFN) - sized for batch
        buffers.normalized = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx);

        // Attention buffers (Qwen-specific dimensions) - sized for batch
        buffers.Q = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(n_heads_ * head_dim_)},
            device_idx);
        buffers.K = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)},
            device_idx);
        buffers.V = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)},
            device_idx);
        buffers.attn_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(n_heads_ * head_dim_)},
            device_idx);
        buffers.attn_proj = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx);

        // FFN buffers (Qwen-specific d_ff_) - sized for batch
        buffers.gate = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(d_ff_)},
            device_idx);
        buffers.up = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(d_ff_)},
            device_idx);
        buffers.ffn_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx);

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
            LOG_DEBUG("[Position State] current_positions_: "
                      << "[" << current_positions_[0] << ", " << current_positions_[1] << "]");
        }

        // Pad sequences to uniform length
        auto padded = createPaddedBatch(token_batches, /*pad_token_id=*/0);
        padded_seq_len_ = padded.max_length;
        sequence_lengths_ = padded.actual_lengths;
        int effective_seq_len = batch_size_ * padded_seq_len_;

        LOG_INFO("Forward pass: batch_size=" << batch_size_
                                             << ", padded_seq_len=" << padded_seq_len_
                                             << ", effective_seq_len=" << effective_seq_len);

        // Allocate activation tensors if needed (sized for batch)
        if (!current_hidden_ || static_cast<int>(current_hidden_->shape()[0]) != effective_seq_len)
        {
            current_hidden_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(effective_seq_len), static_cast<size_t>(d_model_)},
                device_idx_);
            LOG_INFO("Allocated hidden states: "
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

        // Validate after embedding
        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_embedding");

        // Process all transformer layers
        for (int i = 0; i < n_layers_; ++i)
        {
            if (!transformer_layer(i, effective_seq_len))
            {
                LOG_ERROR("Layer " << i << " failed");
                return false;
            }

            // Validate hidden state dimensions unchanged between layers
            VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_layer_" + std::to_string(i));
        }

        // Final normalization
        auto final_norm = getFinalNorm();
        VALIDATE_POINTER(final_norm, "final norm");
        VALIDATE_KERNEL(norm_kernel, final_norm->createRMSNorm(), "RMSNorm kernel");
        VALIDATE_OP(norm_kernel->apply(
                        current_hidden_->data(), final_norm->data(), current_hidden_->mutable_data(),
                        effective_seq_len, d_model_, 1e-6f, false, mpi_ctx_.get(), device_idx_),
                    "Final RMSNorm");

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
        LOG_INFO("Processing layer " << layer_idx);

        // Get layer weights (loaded lazily on first access)
        auto &layer = getLayerWeights(layer_idx);

        // Attention block
        if (!attention_block(layer, layer_idx, effective_seq_len))
        {
            LOG_ERROR("Attention block failed in layer " << layer_idx);
            return false;
        }

        // FFN block
        if (!ffn_block(layer, effective_seq_len))
        {
            LOG_ERROR("FFN block failed in layer " << layer_idx);
            return false;
        }

        return true;
    }

    bool Qwen2Pipeline::attention_block(const LayerWeights &layer, int layer_idx, int effective_seq_len)
    {
        // Phase 4.3: Determine execution device based on weight placement
        // All attention weights should be on same device (enforced by placement strategies)
        int attn_device = placement_map_ ? getWeightDevice("attn_q", -1) : device_idx_;

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

        // Get device-appropriate buffers (Phase 4.1)
        auto &buffers = placement_map_ ? getBuffersForDevice(attn_device) : activation_buffers_;

        // Validate input dimensions
        VALIDATE_TENSOR_PTR(input_hidden, spec_hidden(effective_seq_len), "attn_input");
        VALIDATE_TENSOR_PTR(layer.attn_norm.get(), spec_norm_gamma(), "attn_norm_weight");

        // Validate effective_seq_len fits in pre-allocated buffers
        DEBUG_ASSERT(effective_seq_len <= buffers.max_seq_len,
                     "effective_seq_len (" << effective_seq_len << ") exceeds max_seq_len (" << buffers.max_seq_len << ")");

        // Save residual for later (use device-appropriate buffer)
        std::memcpy(buffers.residual->mutable_data(), input_hidden->data(),
                    effective_seq_len * d_model_ * sizeof(float));

        // Reuse pre-allocated normalized buffer (no allocation in hot path!)
        auto normalized_hidden = buffers.normalized;

        // Copy input to normalized_hidden for in-place normalization
        std::memcpy(normalized_hidden->mutable_data(), input_hidden->data(),
                    effective_seq_len * d_model_ * sizeof(float));

        // 1. Pre-attention RMSNorm
        VALIDATE_KERNEL(norm_kernel, layer.attn_norm->createRMSNorm(), "RMSNorm kernel");
        VALIDATE_OP(norm_kernel->apply(
                        normalized_hidden->data(), layer.attn_norm->data(), normalized_hidden->mutable_data(),
                        effective_seq_len, d_model_, 1e-6f, false, mpi_ctx_.get(), attn_device),
                    "Attention norm");
        VALIDATE_TENSOR_BUFFER(normalized_hidden, spec_hidden(effective_seq_len), "after_attn_norm");

        // 2. Q/K/V projections (use device-appropriate buffers)
        VALIDATE_KERNEL(q_gemm, layer.wq->createGemm(), "Q GEMM kernel");
        VALIDATE_KERNEL(k_gemm, layer.wk->createGemm(), "K GEMM kernel");
        VALIDATE_KERNEL(v_gemm, layer.wv->createGemm(), "V GEMM kernel");

        // Q = hidden @ wq^T
        VALIDATE_OP(q_gemm->multiply(
                        normalized_hidden->data(), buffers.Q->mutable_data(),
                        effective_seq_len, n_heads_ * head_dim_, d_model_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                    "Q projection");
        VALIDATE_TENSOR_BUFFER(buffers.Q, spec_q(effective_seq_len), "after_q_proj");

        // K = hidden @ wk^T
        VALIDATE_OP(k_gemm->multiply(
                        normalized_hidden->data(), buffers.K->mutable_data(),
                        effective_seq_len, n_kv_heads_ * head_dim_, d_model_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                    "K projection");
        VALIDATE_TENSOR_BUFFER(buffers.K, spec_kv(effective_seq_len), "after_k_proj");

        // V = hidden @ wv^T
        VALIDATE_OP(v_gemm->multiply(
                        normalized_hidden->data(), buffers.V->mutable_data(),
                        effective_seq_len, n_kv_heads_ * head_dim_, d_model_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                    "V projection");
        VALIDATE_TENSOR_BUFFER(buffers.V, spec_kv(effective_seq_len), "after_v_proj");

        // 3. Apply RoPE to Q and K
        // Position IDs for batched input (per-sequence position tracking)
        std::vector<int> position_ids(effective_seq_len);
        for (int b = 0; b < batch_size_; ++b)
        {
            for (int i = 0; i < padded_seq_len_; ++i)
            {
                // Each sequence has independent position counter
                position_ids[b * padded_seq_len_ + i] = current_positions_[b] + i;
            }
        }

        // DEBUG: Log position IDs for first layer
        if (layer_idx == 0 && mpi_ctx_->rank() == 0)
        {
            if (batch_size_ == 1)
            {
                LOG_DEBUG("[RoPE Debug] Layer " << layer_idx << " (batch_size=1) position_ids: "
                                                << "seq0=[" << position_ids[0] << "," << position_ids[1] << "]");
            }
            else if (batch_size_ == 2)
            {
                LOG_DEBUG("[RoPE Debug] Layer " << layer_idx << " (batch_size=2) position_ids: "
                                                << "seq0=[" << position_ids[0] << "," << position_ids[1] << "], "
                                                << "seq1=[" << position_ids[2] << "," << position_ids[3] << "]");
            }
        }

        VALIDATE_KERNEL(rope_kernel, layer.wq->createRoPE(), "RoPE kernel"); // Any weight can create RoPE kernel
        VALIDATE_OP(rope_kernel->apply(
                        buffers.Q->mutable_data(), buffers.K->mutable_data(), position_ids.data(),
                        effective_seq_len, n_heads_, n_kv_heads_, head_dim_,
                        false, mpi_ctx_.get(), attn_device),
                    "RoPE application");
        VALIDATE_TENSOR_BUFFER(buffers.Q, spec_q(effective_seq_len), "after_rope_q");
        VALIDATE_TENSOR_BUFFER(buffers.K, spec_kv(effective_seq_len), "after_rope_k");

        // 4. GQA attention computation (MPI-aware, batch-aware)
        // Dispatches to tensor-parallel if mpi_strategy_ == TensorParallel
        VALIDATE_OP(attention_gqa_mpi(
                        buffers.Q.get(), buffers.K.get(),
                        buffers.V.get(), buffers.attn_output.get(),
                        n_heads_, n_kv_heads_, head_dim_,
                        /*causal=*/true, /*window_size=*/-1,
                        batch_size_, &sequence_lengths_),
                    "GQA attention");

        VALIDATE_TENSOR_BUFFER(buffers.attn_output, spec_q(effective_seq_len), "after_attention");

        // 5. Output projection (reuse attn_proj buffer)
        VALIDATE_KERNEL(o_gemm, layer.wo->createGemm(), "output GEMM kernel");
        VALIDATE_OP(o_gemm->multiply(
                        buffers.attn_output->data(), buffers.attn_proj->mutable_data(),
                        effective_seq_len, d_model_, n_heads_ * head_dim_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                    "Output projection");
        VALIDATE_TENSOR_BUFFER(buffers.attn_proj, spec_hidden(effective_seq_len), "after_attn_out_proj");

        // 6. Residual connection - write back to current_hidden_
        // Note: If multi-device, result stays on attn_device and is stored in current_hidden_
        for (size_t i = 0; i < effective_seq_len * d_model_; ++i)
        {
            current_hidden_->mutable_data()[i] = buffers.residual->data()[i] + buffers.attn_proj->data()[i];
        }

        // Update current_hidden_ device index to reflect where computation happened
        if (placement_map_)
        {
            current_hidden_->set_device(attn_device);
        }

        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_attn_residual");

        return true;
    }

    bool Qwen2Pipeline::ffn_block(const LayerWeights &layer, int effective_seq_len)
    {
        // Phase 4.3: Determine execution device based on weight placement
        int ffn_device = placement_map_ ? getWeightDevice("ffn_gate", -1) : device_idx_;

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

        // Validate input dimensions
        VALIDATE_TENSOR_PTR(input_hidden, spec_hidden(effective_seq_len), "ffn_input");
        VALIDATE_TENSOR_PTR(layer.ffn_norm.get(), spec_norm_gamma(), "ffn_norm_weight");

        // Save residual for later (use device-appropriate buffer)
        std::memcpy(buffers.residual->mutable_data(), input_hidden->data(),
                    effective_seq_len * d_model_ * sizeof(float));

        // Reuse pre-allocated normalized buffer (no allocation in hot path!)
        auto normalized_hidden = buffers.normalized;

        // Copy input to normalized_hidden for in-place normalization
        std::memcpy(normalized_hidden->mutable_data(), input_hidden->data(),
                    effective_seq_len * d_model_ * sizeof(float));

        // 1. Pre-FFN RMSNorm
        VALIDATE_KERNEL(norm_kernel, layer.ffn_norm->createRMSNorm(), "RMSNorm kernel");
        VALIDATE_OP(norm_kernel->apply(
                        normalized_hidden->data(), layer.ffn_norm->data(), normalized_hidden->mutable_data(),
                        effective_seq_len, d_model_, 1e-6f, false, mpi_ctx_.get(), ffn_device),
                    "FFN norm");
        VALIDATE_TENSOR_BUFFER(normalized_hidden, spec_hidden(effective_seq_len), "after_ffn_norm");

        // 2. Gate and up projections (use device-appropriate buffers)
        VALIDATE_KERNEL(gate_gemm, layer.gate_proj->createGemm(), "gate GEMM kernel");
        VALIDATE_KERNEL(up_gemm, layer.up_proj->createGemm(), "up GEMM kernel");

        // gate = hidden @ gate_proj^T
        VALIDATE_OP(gate_gemm->multiply(
                        normalized_hidden->data(), buffers.gate->mutable_data(),
                        effective_seq_len, d_ff_, d_model_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), ffn_device),
                    "Gate projection");
        VALIDATE_TENSOR_BUFFER(buffers.gate, spec_ffn_gate_up(effective_seq_len), "after_gate_proj");

        // up = hidden @ up_proj^T
        VALIDATE_OP(up_gemm->multiply(
                        normalized_hidden->data(), buffers.up->mutable_data(),
                        effective_seq_len, d_ff_, d_model_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), ffn_device),
                    "Up projection");
        VALIDATE_TENSOR_BUFFER(buffers.up, spec_ffn_gate_up(effective_seq_len), "after_up_proj");

        // 3. SwiGLU activation (up buffer reused for output)
        VALIDATE_KERNEL(swiglu_kernel, layer.gate_proj->createSwiGLU(), "SwiGLU kernel");
        VALIDATE_OP(swiglu_kernel->apply(
                        buffers.gate->data(), buffers.up->data(),
                        buffers.up->mutable_data(),
                        effective_seq_len, d_ff_, false, mpi_ctx_.get(), ffn_device),
                    "SwiGLU activation");
        VALIDATE_TENSOR_BUFFER(buffers.up, spec_ffn_intermediate(effective_seq_len), "after_swiglu");

        // 4. Down projection (reuse ffn_output buffer)
        VALIDATE_KERNEL(down_gemm, layer.down_proj->createGemm(), "down GEMM kernel");

        // ffn_output = up @ down_proj^T
        VALIDATE_OP(down_gemm->multiply(
                        buffers.up->data(), buffers.ffn_output->mutable_data(),
                        effective_seq_len, d_model_, d_ff_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), ffn_device),
                    "Down projection");
        VALIDATE_TENSOR_BUFFER(buffers.ffn_output, spec_hidden(effective_seq_len), "after_down_proj");

        // 5. Residual connection - write back to current_hidden_
        for (size_t i = 0; i < effective_seq_len * d_model_; ++i)
        {
            current_hidden_->mutable_data()[i] = buffers.residual->data()[i] + buffers.ffn_output->data()[i];
        }

        // Update current_hidden_ device index to reflect where computation happened
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
            embedding_table_ = model_ctx_->getWeight("token_embd.weight", device_idx_);
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
            lm_head_ = model_ctx_->getWeight("output.weight", device_idx_);
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
            layer.wk = model_ctx_->getWeight(prefix + "attn_k.weight", device_idx_);
            layer.wv = model_ctx_->getWeight(prefix + "attn_v.weight", device_idx_);
            layer.wo = model_ctx_->getWeight(prefix + "attn_output.weight", device_idx_);
            layer.attn_norm = model_ctx_->getWeight(prefix + "attn_norm.weight", device_idx_);
            layer.gate_proj = model_ctx_->getWeight(prefix + "ffn_gate.weight", device_idx_);
            layer.up_proj = model_ctx_->getWeight(prefix + "ffn_up.weight", device_idx_);
            layer.down_proj = model_ctx_->getWeight(prefix + "ffn_down.weight", device_idx_);
            layer.ffn_norm = model_ctx_->getWeight(prefix + "ffn_norm.weight", device_idx_);
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
        auto embed_table = getEmbeddingTable();
        VALIDATE_POINTER(embed_table, "embedding table");

        const float *embed_data = embed_table->data();
        float *output_data = output->mutable_data();

        // Process each sequence in the batch
        int global_idx = 0;
        for (int b = 0; b < batch_size_; ++b)
        {
            const auto &tokens = token_batches[b];
            int seq_len = tokens.size();

            // Lookup embeddings for this sequence
            for (int i = 0; i < seq_len; ++i)
            {
                int token_id = tokens[i];
                DEBUG_ASSERT_RANGE(token_id, 0, vocab_size_,
                                   "Invalid token at batch=" << b << ", pos=" << i);
                std::memcpy(output_data + global_idx * d_model_,
                            embed_data + token_id * d_model_,
                            d_model_ * sizeof(float));
                global_idx++;
            }

            // Pad remaining positions with zeros (or pad token embedding)
            for (int i = seq_len; i < padded_seq_len_; ++i)
            {
                std::memset(output_data + global_idx * d_model_, 0, d_model_ * sizeof(float));
                global_idx++;
            }
        }

        return true;
    }

    bool Qwen2Pipeline::lm_head_batch(TensorBase *hidden, int effective_seq_len)
    {
        // Allocate logits buffer if needed
        if (!logits_buffer_ || static_cast<int>(logits_buffer_->shape()[0]) != effective_seq_len)
        {
            logits_buffer_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(effective_seq_len), static_cast<size_t>(vocab_size_)},
                device_idx_);
            LOG_INFO("Allocated logits buffer: "
                     << effective_seq_len << " x " << vocab_size_ << " on device " << device_idx_);
        }

        VALIDATE_TENSOR(logits_buffer_, spec_logits(effective_seq_len), "logits_allocation");

        auto lm_head = getLMHead();
        VALIDATE_POINTER(lm_head, "LM head");
        VALIDATE_KERNEL(lm_gemm, lm_head->createGemm(), "LM head GEMM kernel");

        // LM head: logits = hidden @ lm_head^T
        // hidden: [effective_seq_len, d_model], lm_head: [vocab_size, d_model]
        // output: [effective_seq_len, vocab_size]
        VALIDATE_OP(lm_gemm->multiply(
                        hidden->data(), logits_buffer_->mutable_data(),
                        effective_seq_len, vocab_size_, d_model_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_),
                    "LM head projection");

        VALIDATE_TENSOR(logits_buffer_, spec_logits(effective_seq_len), "after_lm_head");
        return true;
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

} // namespace llaminar2
