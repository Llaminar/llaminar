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
#include "../PipelineFactory.h"
#include "../../loaders/ModelLoader.h"
#include "../../tensors/TensorFactory.h"
#include "../../utils/BatchPaddingUtils.h"
#include "../../kernels/cpu/ops/CPURMSNormKernelT.h"
#include "../../kernels/cpu/gemm_v4/FusedGEMM.h"
#include "../../kernels/cpu/fused/FusedDequantSwiGLU.h"
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

        // Initialize infrastructure with batched workspace buffers
        initializeInfrastructureBatched();

        // Override KV cache with batched version
        std::vector<int> attention_devices = detectAttentionDevices(n_layers_);
        kv_cache_batched_ = std::make_shared<BatchedKVCache>(
            n_layers_, batch_size_, config.max_seq_len, n_kv_heads_, head_dim_, attention_devices);
        LOG_INFO("Initialized batched KV cache: batch_size=" << batch_size_
                                                             << ", max_seq_len=" << config.max_seq_len);

        LOG_INFO("Pipeline initialized (weights loaded on-demand)");
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

        LOG_INFO("Pipeline infrastructure initialized (max_seq_len=" << max_seq_len
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

    // Helper: Create activation tensor with configured precision
    namespace
    {
        std::shared_ptr<TensorBase> createActivationTensor(
            const std::vector<size_t> &shape,
            int device_idx,
            ActivationPrecision precision)
        {
            switch (precision)
            {
            case ActivationPrecision::FP32:
                return std::make_shared<FP32Tensor>(shape, device_idx);

            case ActivationPrecision::BF16:
                return std::make_shared<BF16Tensor>(shape);

            case ActivationPrecision::FP16:
                return std::make_shared<FP16Tensor>(shape);

            case ActivationPrecision::INT32:
                return std::make_shared<INT32Tensor>(shape);

            default:
                LOG_ERROR("Unknown activation precision, defaulting to FP32");
                return std::make_shared<FP32Tensor>(shape, device_idx);
            }
        }
    } // anonymous namespace

    ActivationBuffers Qwen2Pipeline::createBuffersForDevice(int device_idx, int max_seq_len)
    {
        ActivationBuffers buffers;
        // Size buffers for batch_size * max_seq_len (flattened batch dimension)
        int effective_max = batch_size_ * max_seq_len;
        buffers.max_seq_len = effective_max;

        // Get configured activation precision
        auto precision = config_.activation_precision;

        // Residual (d_model) - sized for batch
        buffers.residual = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx, precision);

        // Normalization buffer (shared across attention and FFN) - sized for batch
        buffers.normalized = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx, precision);

        // Attention buffers (Qwen-specific dimensions) - sized for batch
        buffers.Q = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_heads_ * head_dim_)},
            device_idx, precision);
        buffers.K = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)},
            device_idx, precision);
        buffers.V = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)},
            device_idx, precision);
        buffers.attn_output = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(n_heads_ * head_dim_)},
            device_idx, precision);
        buffers.attn_proj = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx, precision);

        // FFN buffers (Qwen-specific d_ff_) - sized for batch
        buffers.gate = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_ff_)},
            device_idx, precision);
        buffers.up = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_ff_)},
            device_idx, precision);
        buffers.ffn_output = createActivationTensor(
            {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
            device_idx, precision);

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
            LOG_DEBUG("[Position State] current_positions_: " << ss.str());
        }

        // Pad sequences to uniform length
        auto padded = createPaddedBatch(token_batches, /*pad_token_id=*/0);
        padded_seq_len_ = padded.max_length;
        sequence_lengths_ = padded.actual_lengths;
        int effective_seq_len = batch_size_ * padded_seq_len_;

        LOG_DEBUG("Forward pass: batch_size=" << batch_size_
                                              << ", padded_seq_len=" << padded_seq_len_
                                              << ", effective_seq_len=" << effective_seq_len);

        // Allocate activation tensors if needed (sized for batch)
        if (!current_hidden_ || static_cast<int>(current_hidden_->shape()[0]) != effective_seq_len)
        {
            current_hidden_ = createActivationTensor(
                {static_cast<size_t>(effective_seq_len), static_cast<size_t>(d_model_)},
                device_idx_, config_.activation_precision);
            LOG_DEBUG("Allocated hidden states: "
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
        LOG_DEBUG("Processing layer " << layer_idx);

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

    bool Qwen2Pipeline::attention_block(const LayerWeights &layer, int layer_idx, int effective_seq_len)
    {
        // Determine execution device based on weight placement
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

        // Get device-appropriate buffers
        auto &buffers = placement_map_ ? getBuffersForDevice(attn_device) : activation_buffers_;
        std::string layer_prefix = "layer" + std::to_string(layer_idx);

        // Save residual for later
        TRY_OP(save_residual(input_hidden, buffers.residual.get(), effective_seq_len, d_model_));

        // 1. Pre-attention RMSNorm
        TRY_OP(rms_norm(
            buffers.residual.get(), layer.attn_norm.get(), buffers.normalized.get(),
            effective_seq_len, d_model_, 1e-6f,
            layer_prefix + "_ATTENTION_NORM", attn_device));

        // 2. Fused Q/K/V projections
        if (!layer.qkv_fused)
        {
            layer.qkv_fused = std::make_unique<FusedGEMM>(
                layer.wq.get(), layer.wk.get(), layer.wv.get());
        }

        VALIDATE_OP(layer.qkv_fused->execute(
                        buffers.normalized->data(),
                        buffers.Q->mutable_data(),
                        buffers.K->mutable_data(),
                        buffers.V->mutable_data(),
                        nullptr, nullptr, nullptr,
                        effective_seq_len,
                        n_heads_ * head_dim_, n_kv_heads_ * head_dim_, n_kv_heads_ * head_dim_,
                        d_model_,
                        mpi_ctx_.get(), attn_device),
                    "Fused Q/K/V projection");

        // Capture Q/K/V projections
        capture_snapshot(layer_prefix + "_Q_PROJECTION", buffers.Q.get(), effective_seq_len, n_heads_ * head_dim_);
        capture_snapshot(layer_prefix + "_K_PROJECTION", buffers.K.get(), effective_seq_len, n_kv_heads_ * head_dim_);
        capture_snapshot(layer_prefix + "_V_PROJECTION", buffers.V.get(), effective_seq_len, n_kv_heads_ * head_dim_);

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

        // 4. GQA attention
        TRY_OP(compute_attention(
            buffers.Q.get(), buffers.K.get(), buffers.V.get(), buffers.attn_output.get(),
            effective_seq_len, n_heads_, n_kv_heads_, head_dim_,
            batch_size_, sequence_lengths_, padded_seq_len_,
            /*causal=*/false, layer_prefix + "_ATTENTION_CONTEXT"));

        // 5. Output projection
        TRY_OP(project(
            buffers.attn_output.get(), layer.wo.get(), buffers.attn_proj.get(),
            effective_seq_len, d_model_, n_heads_ * head_dim_,
            layer_prefix + "_ATTENTION_OUTPUT", attn_device));

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

        // 1. Pre-FFN RMSNorm
        TRY_OP(rms_norm(
            buffers.residual.get(), layer.ffn_norm.get(), buffers.normalized.get(),
            effective_seq_len, d_model_, 1e-6f,
            layer_prefix + "_FFN_NORM", ffn_device));

        // 2. Fused Gate/Up projections
        if (!layer.gate_up_fused)
        {
            layer.gate_up_fused = std::make_unique<FusedGEMM>(
                layer.gate_proj.get(), layer.up_proj.get());
        }

        VALIDATE_OP(layer.gate_up_fused->execute(
                        buffers.normalized->data(),
                        {{buffers.gate->mutable_data(), nullptr, d_ff_, "gate"},
                         {buffers.up->mutable_data(), nullptr, d_ff_, "up", nullptr, false}},
                        effective_seq_len, d_model_,
                        mpi_ctx_.get(), ffn_device),
                    "Fused Gate/Up projection");

        // Capture gate/up projections
        capture_snapshot(layer_prefix + "_FFN_GATE", buffers.gate.get(), effective_seq_len, d_ff_);
        capture_snapshot(layer_prefix + "_FFN_UP", buffers.up.get(), effective_seq_len, d_ff_);

        // 3. Apply SwiGLU
        TRY_OP(swiglu(
            buffers.gate.get(), buffers.up.get(), buffers.up.get(),
            effective_seq_len, d_ff_,
            layer_prefix + "_FFN_SWIGLU", ffn_device));

        // 4. Down projection
        TRY_OP(project(
            buffers.up.get(), layer.down_proj.get(), buffers.ffn_output.get(),
            effective_seq_len, d_model_, d_ff_,
            layer_prefix + "_FFN_DOWN", ffn_device));

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
                LOG_DEBUG("[DEBUG] Layer " << layer_idx << " wq: type=" << type_name
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

        static const bool debug_batch = std::getenv("LLAMINAR_DEBUG_BATCH") != nullptr;

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
            logits_buffer_ = createActivationTensor(
                {static_cast<size_t>(effective_seq_len), static_cast<size_t>(vocab_size_)},
                device_idx_, config_.activation_precision);
            LOG_INFO("Allocated logits buffer: "
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
