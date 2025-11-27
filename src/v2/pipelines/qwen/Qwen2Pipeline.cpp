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
#include "../../kernels/cpu/CPURMSNormKernelT.h"
#include "../../kernels/cpu/CPUSwiGLUKernelT.h"
#include "../../kernels/cpu/fused/FusedGEMM.h"
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

// Helper macro for snapshot capture (element_count() is protected)
#ifdef ENABLE_PIPELINE_SNAPSHOTS
#define CAPTURE_SNAPSHOT(key, tensor_ptr)                     \
    do                                                        \
    {                                                         \
        const auto &_shape = (tensor_ptr)->shape();           \
        size_t _numel = 1;                                    \
        for (auto _dim : _shape)                              \
            _numel *= _dim;                                   \
        captureSnapshot((key), (tensor_ptr)->data(), _numel); \
    } while (0)

// Helper macro for snapshot capture with view (for buffers larger than actual data)
#define CAPTURE_SNAPSHOT_VIEW(key, tensor_ptr, rows, cols)                                              \
    do                                                                                                  \
    {                                                                                                   \
        auto _view = (tensor_ptr)->create_view({static_cast<size_t>(rows), static_cast<size_t>(cols)}); \
        if (_view)                                                                                      \
        {                                                                                               \
            const auto &_shape = _view->shape();                                                        \
            size_t _numel = 1;                                                                          \
            for (auto _dim : _shape)                                                                    \
                _numel *= _dim;                                                                         \
            captureSnapshot((key), _view->data(), _numel);                                              \
        }                                                                                               \
    } while (0)
#else
#define CAPTURE_SNAPSHOT(key, tensor_ptr) \
    do                                    \
    {                                     \
    } while (0)
#define CAPTURE_SNAPSHOT_VIEW(key, tensor_ptr, rows, cols) \
    do                                                     \
    {                                                      \
    } while (0)
#endif

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

        // Phase 4.1: Device infrastructure with batch_size for workspace mask allocation
        initializeDeviceInfrastructure(max_seq_len, batch_size_);

        // Phase 2: MPI strategy configuration
        configureMPIStrategy();

        // Phase 3: KV cache initialization
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

        // Apply RMSNorm using clean tensor interface (tensor knows its own type)
        auto *activation_tensor = dynamic_cast<IActivationTensor *>(current_hidden_.get());
        VALIDATE_POINTER(activation_tensor, "activation tensor for RMSNorm");

        // Use kernel pattern: tensor creates kernel, kernel executes
        auto rmsnorm_kernel = activation_tensor->createRMSNorm();
        VALIDATE_POINTER(rmsnorm_kernel, "RMSNorm kernel");

        VALIDATE_OP(rmsnorm_kernel->apply(
                        current_hidden_->data(),         // input
                        final_norm->data(),              // weight (gamma)
                        current_hidden_->mutable_data(), // output (in-place)
                        effective_seq_len, d_model_,
                        1e-6f, // epsilon
                        false, // normalize_gamma
                        mpi_ctx_.get(), device_idx_),
                    "Final RMSNorm");

        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_final_norm");

        // Capture final norm output
        CAPTURE_SNAPSHOT("FINAL_NORM", current_hidden_.get());

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
        static const bool debug_batch_env = std::getenv("LLAMINAR_DEBUG_BATCH") != nullptr;

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
        auto *fp32_activation = dynamic_cast<IActivationTensor *>(normalized_hidden.get());
        VALIDATE_POINTER(fp32_activation, "FP32 activation tensor for RMSNorm");

        auto rmsnorm_kernel = fp32_activation->createRMSNorm();
        VALIDATE_POINTER(rmsnorm_kernel, "RMSNorm kernel");

        VALIDATE_OP(rmsnorm_kernel->apply(
                        buffers.residual->data(),          // input
                        layer.attn_norm->data(),           // weight
                        normalized_hidden->mutable_data(), // output
                        effective_seq_len, d_model_,
                        1e-6f, // epsilon
                        false, // normalize_gamma
                        mpi_ctx_.get(), attn_device),
                    "Attention norm");
        VALIDATE_TENSOR_BUFFER(normalized_hidden, spec_hidden(effective_seq_len), "after_attn_norm");

        // Capture attention norm output (use view to get only valid data, not entire buffer)
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_ATTENTION_NORM",
                              normalized_hidden, effective_seq_len, d_model_);

        // 2. Q/K/V projections (FUSED: quantize activations once, reuse for all 3 projections)
        // Lazily initialize the fused QKV kernel if not already done
        if (!layer.qkv_fused)
        {
            layer.qkv_fused = std::make_unique<FusedGEMM>(
                layer.wq.get(), layer.wk.get(), layer.wv.get());
        }

        // Execute fused Q/K/V projection
        // Q: [seq, n_heads * head_dim], K: [seq, n_kv_heads * head_dim], V: [seq, n_kv_heads * head_dim]
        VALIDATE_OP(layer.qkv_fused->execute(
                        normalized_hidden->data(),
                        buffers.Q->mutable_data(),
                        buffers.K->mutable_data(),
                        buffers.V->mutable_data(),
                        nullptr, nullptr, nullptr, // No biases
                        effective_seq_len,
                        n_heads_ * head_dim_,    // n1 = Q dimension
                        n_kv_heads_ * head_dim_, // n2 = K dimension
                        n_kv_heads_ * head_dim_, // n3 = V dimension
                        d_model_,                // k = input dimension
                        mpi_ctx_.get(), attn_device),
                    "Fused Q/K/V projection");

        VALIDATE_TENSOR_BUFFER(buffers.Q, spec_q(effective_seq_len), "after_q_proj");
        VALIDATE_TENSOR_BUFFER(buffers.K, spec_kv(effective_seq_len), "after_k_proj");
        VALIDATE_TENSOR_BUFFER(buffers.V, spec_kv(effective_seq_len), "after_v_proj");

        // Capture Q projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_Q_PROJECTION", buffers.Q, effective_seq_len, n_heads_ * head_dim_);

        // Capture K projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_K_PROJECTION", buffers.K, effective_seq_len, n_kv_heads_ * head_dim_);

        // Capture V projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_V_PROJECTION", buffers.V, effective_seq_len, n_kv_heads_ * head_dim_);

        // 3. Apply RoPE to Q and K
        // Position IDs for batched input (per-sequence position tracking)
        // CRITICAL: Only set position IDs for actual tokens, not padding
        std::vector<int> position_ids(effective_seq_len);
        for (int b = 0; b < batch_size_; ++b)
        {
            int actual_len = (batch_size_ == 1) ? padded_seq_len_ : sequence_lengths_[b];
            for (int i = 0; i < padded_seq_len_; ++i)
            {
                if (i < actual_len)
                {
                    // Real token: use actual position
                    position_ids[b * padded_seq_len_ + i] = current_positions_[b] + i;
                }
                else
                {
                    // Padding token: use -1 to signal "skip RoPE"
                    position_ids[b * padded_seq_len_ + i] = -1;
                }
            }
        }

        // Apply RoPE using clean tensor interface (Q is the activation tensor, K is passed as parameter)
        auto *activation_tensor_q = dynamic_cast<IActivationTensor *>(buffers.Q.get());
        VALIDATE_POINTER(activation_tensor_q, "activation tensor for RoPE");

        VALIDATE_OP(activation_tensor_q->applyRoPE(
                        buffers.K->mutable_data(), position_ids.data(),
                        effective_seq_len, n_heads_, n_kv_heads_, head_dim_,
                        model_ctx_->model().rope_theta,
                        false, mpi_ctx_.get(), attn_device),
                    "RoPE application");
        VALIDATE_TENSOR_BUFFER(buffers.Q, spec_q(effective_seq_len), "after_rope_q");
        VALIDATE_TENSOR_BUFFER(buffers.K, spec_kv(effective_seq_len), "after_rope_k");

        // Capture Q and K after RoPE
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_Q_ROPE", buffers.Q, effective_seq_len, n_heads_ * head_dim_);
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_K_ROPE", buffers.K, effective_seq_len, n_kv_heads_ * head_dim_);

        // 4. GQA attention computation (MPI-aware, batch-aware)
        // Create views of Q/K/V buffers with actual effective_seq_len to avoid pre-allocated buffer size mismatch
        auto Q_view = buffers.Q->create_view({static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_heads_ * head_dim_)});
        auto K_view = buffers.K->create_view({static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)});
        auto V_view = buffers.V->create_view({static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)});
        auto attn_out_view = buffers.attn_output->create_view({static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_heads_ * head_dim_)});

        if (!Q_view || !K_view || !V_view || !attn_out_view)
        {
            LOG_ERROR("[Qwen2Pipeline] Failed to create tensor views for attention");
            return false;
        }

        // Dispatches to tensor-parallel if mpi_strategy_ == TensorParallel
        // NOTE: Using causal=false to match PyTorch reference for E2E parity testing
        // In production inference, causal=true should be used for autoregressive generation

        // Only pass sequence_lengths if there's actual padding (any sequence shorter than padded_seq_len)
        // For equal-length batches, no mask is needed when causal=false
        const std::vector<int> *seq_lens_ptr = nullptr;
        if (batch_size_ > 1)
        {
            bool has_padding = false;
            for (int b = 0; b < batch_size_; ++b)
            {
                if (sequence_lengths_[b] < padded_seq_len_)
                {
                    has_padding = true;
                    break;
                }
            }
            if (has_padding)
            {
                seq_lens_ptr = &sequence_lengths_;
            }
        }

        VALIDATE_OP(attention_gqa_mpi(
                        Q_view.get(), K_view.get(),
                        V_view.get(), attn_out_view.get(),
                        n_heads_, n_kv_heads_, head_dim_,
                        /*causal=*/false, /*window_size=*/-1,
                        batch_size_, seq_lens_ptr), // Pass sequence lengths only if padding exists
                    "GQA attention");

        VALIDATE_TENSOR_BUFFER(buffers.attn_output, spec_q(effective_seq_len), "after_attention");

        // Capture attention context (before output projection)
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_ATTENTION_CONTEXT", buffers.attn_output, effective_seq_len, n_heads_ * head_dim_);

        // DEBUG: Track attention output (after attention mechanism, before output projection)
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *attn_data = buffers.attn_output->data();
            size_t seq1_offset = 4 * n_heads_ * head_dim_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After ATTENTION (before output proj), Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << attn_data[seq1_offset + i]);
            }
        }

        // 5. Output projection (reuse attn_proj buffer)
        VALIDATE_KERNEL(o_gemm, layer.wo->createGemm(), "output GEMM kernel");
        VALIDATE_OP(o_gemm->multiply_activations(
                        buffers.attn_output->data(), nullptr, buffers.attn_proj->mutable_data(),
                        effective_seq_len, d_model_, n_heads_ * head_dim_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                    "Output projection");
        VALIDATE_TENSOR_BUFFER(buffers.attn_proj, spec_hidden(effective_seq_len), "after_attn_out_proj");

        // Capture attention output projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_ATTENTION_OUTPUT", buffers.attn_proj, effective_seq_len, d_model_);

        // 6. Residual connection - write back to current_hidden_
        // Note: If multi-device, result stays on attn_device and is stored in current_hidden_
        // CRITICAL: Process all batch sequences, not just effective_seq_len
        // CRITICAL FIX: Zero out padding rows to prevent NaN propagation to next layer
        const size_t residual_elements = batch_size_ * padded_seq_len_ * d_model_;

#pragma omp parallel for
        for (size_t i = 0; i < residual_elements; ++i)
        {
            size_t token_idx = i / d_model_;
            size_t batch_idx = token_idx / padded_seq_len_;
            size_t seq_idx = token_idx % padded_seq_len_;

            if (seq_idx >= sequence_lengths_[batch_idx])
            {
                current_hidden_->mutable_data()[i] = 0.0f;
            }
            else
            {
                current_hidden_->mutable_data()[i] = buffers.residual->data()[i] + buffers.attn_proj->data()[i];
            }
        }

        // Update current_hidden_ device index to reflect where computation happened
        if (placement_map_)
        {
            current_hidden_->set_device(attn_device);
        }

        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_attn_residual");

        // Capture attention residual output
        CAPTURE_SNAPSHOT("layer" + std::to_string(layer_idx) + "_ATTENTION_RESIDUAL", current_hidden_.get());

        return true;
    }

    bool Qwen2Pipeline::ffn_block(const LayerWeights &layer, int layer_idx, int effective_seq_len)
    {
        static const bool debug_batch_env = std::getenv("LLAMINAR_DEBUG_BATCH") != nullptr;

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
        auto *fp32_activation = dynamic_cast<IActivationTensor *>(normalized_hidden.get());
        VALIDATE_POINTER(fp32_activation, "FP32 activation tensor for FFN RMSNorm");

        auto rmsnorm_kernel = fp32_activation->createRMSNorm();
        VALIDATE_POINTER(rmsnorm_kernel, "RMSNorm kernel");

        VALIDATE_OP(rmsnorm_kernel->apply(
                        buffers.residual->data(),          // input
                        layer.ffn_norm->data(),            // weight
                        normalized_hidden->mutable_data(), // output
                        effective_seq_len, d_model_,
                        1e-6f, // epsilon
                        false, // normalize_gamma
                        mpi_ctx_.get(), ffn_device),
                    "FFN norm");
        VALIDATE_TENSOR_BUFFER(normalized_hidden, spec_hidden(effective_seq_len), "after_ffn_norm");

        // Capture FFN norm output
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_NORM", normalized_hidden, effective_seq_len, d_model_);

        // 2. Gate and up projections (FUSED: quantize activations once, reuse for both projections)
        // Lazily initialize the fused gate/up kernel if not already done
        if (!layer.gate_up_fused)
        {
            layer.gate_up_fused = std::make_unique<FusedGEMM>(
                layer.gate_proj.get(), layer.up_proj.get());
        }

        // Execute fused gate/up projection (Raw projections, no SwiGLU fusion yet)
        // We compute raw projections first to allow capturing snapshots of FFN_GATE and FFN_UP
        // for parity debugging. SwiGLU is applied manually afterwards.
        VALIDATE_OP(layer.gate_up_fused->execute(
                        normalized_hidden->data(),
                        {{buffers.gate->mutable_data(), nullptr, d_ff_, "gate"},
                         {buffers.up->mutable_data(), nullptr, d_ff_, "up", nullptr, false}}, // do_swiglu=false
                        effective_seq_len, d_model_,
                        mpi_ctx_.get(), ffn_device),
                    "Fused Gate/Up projection");

        VALIDATE_TENSOR_BUFFER(buffers.gate, spec_ffn_gate_up(effective_seq_len), "after_gate_proj");
        VALIDATE_TENSOR_BUFFER(buffers.up, spec_ffn_gate_up(effective_seq_len), "after_up_proj");

        // Capture gate projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_GATE",
                              buffers.gate, effective_seq_len, d_ff_);

        // Health check: Gate projection output
        CHECK_NUMERICAL_HEALTH(("layer" + std::to_string(layer_idx) + "_FFN_GATE").c_str(),
                               buffers.gate->data(), effective_seq_len * d_ff_);

        // Capture up projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_UP",
                              buffers.up, effective_seq_len, d_ff_);

        // Apply SwiGLU manually
        // Qwen2 SwiGLU formulation: output = gate * silu(up)
        // Note: This differs from standard SwiGLU (swish(gate) * up)
        // We pass buffers.gate as the "gate" param (linear term) and buffers.up as "up" param (activated term)
        static CPUSwiGLUKernelT<FP32Tensor> swiglu_kernel;
        VALIDATE_OP(swiglu_kernel.apply(
                        buffers.gate->data(),       // Linear term
                        buffers.up->data(),         // Activated term (silu applied to this)
                        buffers.up->mutable_data(), // Output (in-place in up buffer)
                        effective_seq_len, d_ff_,
                        false, // no residual
                        mpi_ctx_.get(), ffn_device),
                    "SwiGLU activation");

        // Capture SwiGLU output (same as UP now, but keeping key for compatibility)
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_SWIGLU", buffers.up, effective_seq_len, d_ff_);

        // Health check: SwiGLU activation output (critical for numerical stability)
        CHECK_NUMERICAL_HEALTH(("layer" + std::to_string(layer_idx) + "_FFN_SWIGLU").c_str(),
                               buffers.up->data(), effective_seq_len * d_ff_);

        // 4. Down projection (reuse ffn_output buffer)
        VALIDATE_KERNEL(down_gemm, layer.down_proj->createGemm(), "down GEMM kernel");

        // ffn_output = up @ down_proj^T
        VALIDATE_OP(down_gemm->multiply_activations(
                        buffers.up->data(), nullptr, buffers.ffn_output->mutable_data(),
                        effective_seq_len, d_model_, d_ff_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), ffn_device),
                    "Down projection");
        VALIDATE_TENSOR_BUFFER(buffers.ffn_output, spec_hidden(effective_seq_len), "after_down_proj");

        // Capture down projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_DOWN", buffers.ffn_output, effective_seq_len, d_model_);

        // Health check: FFN output before residual (final dequantized output from INT8 path)
        CHECK_NUMERICAL_HEALTH(("layer" + std::to_string(layer_idx) + "_FFN_DOWN").c_str(),
                               buffers.ffn_output->data(), effective_seq_len * d_model_);

        // 5. Residual connection - write back to current_hidden_
        const size_t residual_elements = batch_size_ * padded_seq_len_ * d_model_;

#pragma omp parallel for
        for (size_t i = 0; i < residual_elements; ++i)
        {
            size_t token_idx = i / d_model_;
            size_t batch_idx = token_idx / padded_seq_len_;
            size_t seq_idx = token_idx % padded_seq_len_;

            if (seq_idx >= sequence_lengths_[batch_idx])
            {
                current_hidden_->mutable_data()[i] = 0.0f;
            }
            else
            {
                current_hidden_->mutable_data()[i] = buffers.residual->data()[i] + buffers.ffn_output->data()[i];
            }
        }

        // Update current_hidden_ device index to reflect where computation happened
        if (placement_map_)
        {
            current_hidden_->set_device(ffn_device);
        }

        VALIDATE_TENSOR(current_hidden_, spec_hidden(effective_seq_len), "after_ffn_residual");

        // Capture FFN residual output
        CAPTURE_SNAPSHOT("layer" + std::to_string(layer_idx) + "_FFN_RESIDUAL", current_hidden_.get());

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
        VALIDATE_KERNEL(lm_gemm, lm_head->createGemm(), "LM head GEMM kernel");

        // LM head: logits = hidden @ lm_head^T
        // hidden: [effective_seq_len, d_model], lm_head: [vocab_size, d_model]
        // output: [effective_seq_len, vocab_size]
        VALIDATE_OP(lm_gemm->multiply_activations(
                        hidden->data(), nullptr, logits_buffer_->mutable_data(),
                        effective_seq_len, vocab_size_, d_model_,
                        true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_),
                    "LM head projection");

        VALIDATE_TENSOR(logits_buffer_, spec_logits(effective_seq_len), "after_lm_head");

        // Capture LM head logits
        CAPTURE_SNAPSHOT("LM_HEAD", logits_buffer_.get());

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
