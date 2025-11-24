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
#include "../../kernels/cpu/fused/FusedDualGEMM.h"
#include "../../kernels/cpu/fused/FusedTripleGEMM.h"
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

            case ActivationPrecision::INT8:
                return std::make_shared<INT8Tensor>(shape);

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

        // INT8 quantization buffers (for FusedRMSNormQuantize output)
        // Only allocate if using INT8 activation precision
        if (precision == ActivationPrecision::INT8)
        {
            buffers.normalized_int8 = createActivationTensor(
                {static_cast<size_t>(effective_max), static_cast<size_t>(d_model_)},
                device_idx, ActivationPrecision::INT8);
            buffers.normalized_scales.resize(effective_max); // Per-row scales
        }
        else
        {
            buffers.normalized_int8 = nullptr; // Explicitly null when not using INT8
            buffers.normalized_scales.clear();
        }

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

        // INT32 attention buffers (for FusedTripleGEMM output - raw INT8×INT8→INT32 accumulators)
        // Always allocate when using INT8 precision (needed for Phase 2 fused kernels)
        if (precision == ActivationPrecision::INT8)
        {
            buffers.Q_int32 = createActivationTensor(
                {static_cast<size_t>(effective_max), static_cast<size_t>(n_heads_ * head_dim_)},
                device_idx, ActivationPrecision::INT32);
            buffers.K_int32 = createActivationTensor(
                {static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)},
                device_idx, ActivationPrecision::INT32);
            buffers.V_int32 = createActivationTensor(
                {static_cast<size_t>(effective_max), static_cast<size_t>(n_kv_heads_ * head_dim_)},
                device_idx, ActivationPrecision::INT32);
        }
        else
        {
            buffers.Q_int32 = nullptr;
            buffers.K_int32 = nullptr;
            buffers.V_int32 = nullptr;
        }

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

        // INT32 FFN buffers (for FusedDualGEMM output - raw INT8×INT8→INT32 accumulators)
        // Always allocate when using INT8 precision (needed for Phase 2 fused kernels)
        if (precision == ActivationPrecision::INT8)
        {
            buffers.gate_int32 = createActivationTensor(
                {static_cast<size_t>(effective_max), static_cast<size_t>(d_ff_)},
                device_idx, ActivationPrecision::INT32);
            buffers.up_int32 = createActivationTensor(
                {static_cast<size_t>(effective_max), static_cast<size_t>(d_ff_)},
                device_idx, ActivationPrecision::INT32);
        }
        else
        {
            buffers.gate_int32 = nullptr;
            buffers.up_int32 = nullptr;
        }

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

        // Debug: Print actual token batch lengths
        LOG_ERROR("[BATCH DEBUG] batch_size_=" << batch_size_ << ", padded_seq_len_=" << padded_seq_len_
                                               << ", effective_seq_len=" << effective_seq_len);
        for (int i = 0; i < batch_size_; ++i)
        {
            LOG_ERROR("[BATCH DEBUG]   Sequence " << i << ": " << token_batches[i].size() << " tokens");
        }

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

        // DEBUG: Track final norm output
        static const bool debug_batch_env = std::getenv("LLAMINAR_DEBUG_BATCH") != nullptr;
        if (debug_batch_env && batch_size_ == 2)
        {
            const float *norm_data = current_hidden_->data();
            size_t seq1_offset = 4 * d_model_; // Seq1 starts at position 4
            LOG_ERROR("[FINAL_NORM] Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << norm_data[seq1_offset + i]);
            }
        }

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

        // 1. Pre-attention RMSNorm + INT8 Quantization (fused for efficiency)
        // Use FusedRMSNormQuantize to eliminate redundant FP32→INT8 quantization in GEMM

        // Check if FusedRMSNormQuantize is available and INT8 buffers allocated
        bool use_fused_quantize = (buffers.normalized_int8 != nullptr) &&
                                  (!buffers.normalized_scales.empty());

        if (use_fused_quantize)
        {
            // Fused path: RMSNorm + INT8 quantization in single kernel
            // Use kernel from INT8 output tensor (which provides FusedRMSNormQuantize)
            auto *int8_activation = dynamic_cast<IActivationTensor *>(buffers.normalized_int8.get());
            VALIDATE_POINTER(int8_activation, "INT8 activation tensor for fused RMSNorm");

            auto rmsnorm_kernel = int8_activation->createRMSNorm();
            VALIDATE_POINTER(rmsnorm_kernel, "Fused RMSNorm kernel");

            // CRITICAL: Process ALL batch rows, not just effective_seq_len
            const int total_rows = batch_size_ * padded_seq_len_;

            // Get INT8 buffer directly (don't use dangerous mutable_data() cast)
            auto *int8_tensor_ptr = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            VALIDATE_POINTER(int8_tensor_ptr, "INT8 tensor for quantized output");

            VALIDATE_OP(rmsnorm_kernel->execute(
                            buffers.residual->data(),             // input (FP32)
                            layer.attn_norm->data(),              // weight
                            int8_tensor_ptr->mutable_int8_data(), // output (INT8)
                            buffers.normalized_scales.data(),     // scales
                            total_rows, d_model_,
                            1e-6f, mpi_ctx_.get(), attn_device),
                        "Attention norm (fused INT8)");
            VALIDATE_TENSOR_BUFFER(buffers.normalized_int8,
                                   TensorSpec({static_cast<size_t>(effective_seq_len),
                                               static_cast<size_t>(d_model_)},
                                              "normalized_int8"),
                                   "after_fused_attn_norm_quantize");

            // Register per-row scales with INT8 tensor (critical for correct dequantization)
            auto *int8_tensor = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            if (int8_tensor)
            {
                int8_tensor->set_row_scales(buffers.normalized_scales.data(), total_rows);

                // Debug: Verify scale sanity
                float min_scale = 1e9f, max_scale = -1e9f;
                for (int r = 0; r < total_rows; ++r)
                {
                    min_scale = std::min(min_scale, buffers.normalized_scales[r]);
                    max_scale = std::max(max_scale, buffers.normalized_scales[r]);
                }
                LOG_DEBUG("[FusedQuantize] Attn norm L" << layer_idx
                                                        << " scales: min=" << min_scale << " max=" << max_scale
                                                        << " total_rows=" << total_rows << " effective_seq_len=" << effective_seq_len);
            }

            // Capture INT8 normalized output (convert to FP32 for snapshot compatibility)
            // CRITICAL FIX: Dequantize only effective_seq_len rows (not total_rows)
            // BUG: Previously used total_rows, which accessed uninitialized scales beyond effective_seq_len,
            //      causing garbage in normalized_hidden that propagated through QKV projections
            auto *int8_tensor_data = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            if (int8_tensor_data)
            {
                const int8_t *int8_data = int8_tensor_data->int8_data();
                for (int r = 0; r < effective_seq_len; ++r) // ← FIX: Use effective_seq_len
                {
                    const float row_scale = buffers.normalized_scales[r];
                    for (int c = 0; c < d_model_; ++c)
                    {
                        const size_t idx = r * d_model_ + c;
                        normalized_hidden->mutable_data()[idx] =
                            static_cast<float>(int8_data[idx]) * row_scale;
                    }
                }
            }
        }
        else
        {
            // Unfused path (fallback): Separate RMSNorm + implicit quantization in GEMM
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
        }

        // Capture attention norm output (use view to get only valid data, not entire buffer)
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_ATTENTION_NORM",
                              normalized_hidden, effective_seq_len, d_model_);

        // DEBUG: Track divergence after attention norm
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *norm_data = normalized_hidden->data();
            size_t seq1_offset = 4 * d_model_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After ATTN_NORM, Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << norm_data[seq1_offset + i]);
            }
        }

        // 2. Q/K/V projections - PHASE 2 FUSED PATH
        // Use FusedTripleGEMM to eliminate redundant quantization:
        //   Old: Quant(norm) → q_gemm + Quant(norm) → k_gemm + Quant(norm) → v_gemm → Dequant(Q) + Dequant(K) + Dequant(V)
        //   New: Quant(norm) → FusedTripleGEMM → [q_int32, k_int32, v_int32] → Dequant → [Q_fp32, K_fp32, V_fp32]
        if (use_fused_quantize)
        {
            // Phase 2: Fused path with FusedTripleGEMM + manual dequantization
            // Create fused kernel from INT8 activation tensor
            auto *int8_activation = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            VALIDATE_POINTER(int8_activation, "INT8 activation tensor for FusedTripleGEMM");

            auto fused_triple_gemm = int8_activation->createFusedTripleGemm(
                layer.wq.get(), layer.wk.get(), layer.wv.get());
            VALIDATE_POINTER(fused_triple_gemm, "FusedTripleGEMM kernel");

            // Get INT32 buffers from activation buffers
            auto *q_int32 = dynamic_cast<INT32Tensor *>(buffers.Q_int32.get());
            auto *k_int32 = dynamic_cast<INT32Tensor *>(buffers.K_int32.get());
            auto *v_int32 = dynamic_cast<INT32Tensor *>(buffers.V_int32.get());
            VALIDATE_POINTER(q_int32, "INT32 Q buffer");
            VALIDATE_POINTER(k_int32, "INT32 K buffer");
            VALIDATE_POINTER(v_int32, "INT32 V buffer");

            // Get column scales from weight tensors (for INT32 dequantization)
            auto *q_weight_int8 = dynamic_cast<INT8Tensor *>(layer.wq.get());
            auto *k_weight_int8 = dynamic_cast<INT8Tensor *>(layer.wk.get());
            auto *v_weight_int8 = dynamic_cast<INT8Tensor *>(layer.wv.get());
            VALIDATE_POINTER(q_weight_int8, "INT8 Q weight");
            VALIDATE_POINTER(k_weight_int8, "INT8 K weight");
            VALIDATE_POINTER(v_weight_int8, "INT8 V weight");

            const float *q_col_scales = q_weight_int8->col_scales();
            const float *k_col_scales = k_weight_int8->col_scales();
            const float *v_col_scales = v_weight_int8->col_scales();
            VALIDATE_POINTER(q_col_scales, "Q weight column scales");
            VALIDATE_POINTER(k_col_scales, "K weight column scales");
            VALIDATE_POINTER(v_col_scales, "V weight column scales");

            // Execute FusedTripleGEMM: normalized_fp32 → [q_int32, k_int32, v_int32]
            // Note: FusedTripleGEMM internally quantizes FP32 input to INT8, then performs GEMMs
            // We pass normalized FP32 buffer (not INT8), as kernel handles quantization
            const int n_q = n_heads_ * head_dim_;
            const int n_kv = n_kv_heads_ * head_dim_;

            // FusedTripleGEMM now supports GQA (different Q vs K/V dimensions)
            VALIDATE_OP(fused_triple_gemm->execute(
                            normalized_hidden->data(),        // Input FP32 [m, k]
                            q_int32->mutable_int32_data(),    // Output Q INT32 [m, n_q]
                            k_int32->mutable_int32_data(),    // Output K INT32 [m, n_kv]
                            v_int32->mutable_int32_data(),    // Output V INT32 [m, n_kv]
                            buffers.normalized_scales.data(), // Output row scales [m]
                            effective_seq_len,                // m (batch size)
                            n_q,                              // n_q (Q output features)
                            n_kv,                             // n_kv (K/V output features, may differ for GQA)
                            d_model_),                        // k (input features)
                        "FusedTripleGEMM (Q+K+V projections)");

            LOG_DEBUG("[Phase2] FusedTripleGEMM completed, dequantizing Q/K/V INT32 → FP32");

// Dequantize Q: INT32 → FP32 (row_scales * col_scales)
#pragma omp parallel for collapse(2)
            for (int r = 0; r < effective_seq_len; ++r)
            {
                for (int c = 0; c < n_q; ++c)
                {
                    const size_t idx = r * n_q + c;
                    buffers.Q->mutable_data()[idx] =
                        static_cast<float>(q_int32->int32_data()[idx]) *
                        buffers.normalized_scales[r] * q_col_scales[c];
                }
            }

// Dequantize K: INT32 → FP32 (row_scales * col_scales)
#pragma omp parallel for collapse(2)
            for (int r = 0; r < effective_seq_len; ++r)
            {
                for (int c = 0; c < n_kv; ++c)
                {
                    const size_t idx = r * n_kv + c;
                    buffers.K->mutable_data()[idx] =
                        static_cast<float>(k_int32->int32_data()[idx]) *
                        buffers.normalized_scales[r] * k_col_scales[c];
                }
            }

// Dequantize V: INT32 → FP32 (row_scales * col_scales)
#pragma omp parallel for collapse(2)
            for (int r = 0; r < effective_seq_len; ++r)
            {
                for (int c = 0; c < n_kv; ++c)
                {
                    const size_t idx = r * n_kv + c;
                    buffers.V->mutable_data()[idx] =
                        static_cast<float>(v_int32->int32_data()[idx]) *
                        buffers.normalized_scales[r] * v_col_scales[c];
                }
            }

            LOG_DEBUG("[Phase2] Q/K/V dequantization completed");
            VALIDATE_TENSOR_BUFFER(buffers.Q, spec_q(effective_seq_len), "after_q_proj");
            VALIDATE_TENSOR_BUFFER(buffers.K, spec_kv(effective_seq_len), "after_k_proj");
            VALIDATE_TENSOR_BUFFER(buffers.V, spec_kv(effective_seq_len), "after_v_proj");
        }
        else
        {
            // Phase 1: Unfused path (fallback for non-INT8 precision)
            VALIDATE_KERNEL(q_gemm, layer.wq->createGemm(), "Q GEMM kernel");
            VALIDATE_KERNEL(k_gemm, layer.wk->createGemm(), "K GEMM kernel");
            VALIDATE_KERNEL(v_gemm, layer.wv->createGemm(), "V GEMM kernel");

            // Q = hidden @ wq^T
            VALIDATE_OP(q_gemm->multiply_activations(
                            normalized_hidden->data(), nullptr, buffers.Q->mutable_data(),
                            effective_seq_len, n_heads_ * head_dim_, d_model_,
                            true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                        "Q projection");
            VALIDATE_TENSOR_BUFFER(buffers.Q, spec_q(effective_seq_len), "after_q_proj");

            // K = hidden @ wk^T
            VALIDATE_OP(k_gemm->multiply_activations(
                            normalized_hidden->data(), nullptr, buffers.K->mutable_data(),
                            effective_seq_len, n_kv_heads_ * head_dim_, d_model_,
                            true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                        "K projection");
            VALIDATE_TENSOR_BUFFER(buffers.K, spec_kv(effective_seq_len), "after_k_proj");

            // V = hidden @ wv^T
            VALIDATE_OP(v_gemm->multiply_activations(
                            normalized_hidden->data(), nullptr, buffers.V->mutable_data(),
                            effective_seq_len, n_kv_heads_ * head_dim_, d_model_,
                            true, 1.0f, 0.0f, mpi_ctx_.get(), attn_device),
                        "V projection");
            VALIDATE_TENSOR_BUFFER(buffers.V, spec_kv(effective_seq_len), "after_v_proj");
        }

        // Capture Q projection
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_Q_PROJECTION", buffers.Q, effective_seq_len, n_heads_ * head_dim_);

        // DEBUG: Track Q projection output
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *q_data = buffers.Q->data();
            size_t seq1_offset = 4 * n_heads_ * head_dim_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After Q_PROJ, Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << q_data[seq1_offset + i]);
            }
        }

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

        // DEBUG: Log position IDs for first layer
        if (layer_idx == 0 && (!mpi_ctx_ || mpi_ctx_->rank() == 0))
        {
            if (debug_batch_env)
            {
                LOG_ERROR("[RoPE Layer " << layer_idx << "] Position IDs:");
                for (int b = 0; b < batch_size_; ++b)
                {
                    std::stringstream ss;
                    ss << "  Batch " << b << ": [";
                    for (int i = 0; i < std::min(6, padded_seq_len_); ++i)
                    {
                        ss << position_ids[b * padded_seq_len_ + i];
                        if (i < std::min(6, padded_seq_len_) - 1)
                            ss << ", ";
                    }
                    if (padded_seq_len_ > 6)
                        ss << ", ...";
                    ss << "]";
                    LOG_ERROR(ss.str());
                }
            }

            if (batch_size_ == 1)
            {
                if (position_ids.size() >= 2)
                {
                    LOG_DEBUG("[RoPE Debug] Layer " << layer_idx << " (batch_size=1) position_ids: "
                                                    << "seq0=[" << position_ids[0] << "," << position_ids[1] << "]");
                }
                else if (position_ids.size() == 1)
                {
                    LOG_DEBUG("[RoPE Debug] Layer " << layer_idx << " (batch_size=1) position_ids: "
                                                    << "seq0=[" << position_ids[0] << "]");
                }
            }
            else if (batch_size_ == 2 && position_ids.size() >= 4)
            {
                LOG_DEBUG("[RoPE Debug] Layer " << layer_idx << " (batch_size=2) position_ids: "
                                                << "seq0=[" << position_ids[0] << "," << position_ids[1] << "], "
                                                << "seq1=[" << position_ids[2] << "," << position_ids[3] << "]");
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

        // DEBUG: Track after attention residual connection
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *residual_data = current_hidden_->data();
            size_t seq1_offset = 4 * d_model_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After ATTN_RESIDUAL, Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << residual_data[seq1_offset + i]);
            }
        }

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

        // 1. Pre-FFN RMSNorm + INT8 Quantization (fused for efficiency)

        // Check if FusedRMSNormQuantize is available and INT8 buffers allocated
        bool use_fused_quantize = (buffers.normalized_int8 != nullptr) &&
                                  (!buffers.normalized_scales.empty());

        if (use_fused_quantize)
        {
            // Fused path: RMSNorm + INT8 quantization in single kernel
            // Use kernel from INT8 output tensor (which provides FusedRMSNormQuantize)
            auto *int8_activation = dynamic_cast<IActivationTensor *>(buffers.normalized_int8.get());
            VALIDATE_POINTER(int8_activation, "INT8 activation tensor for fused FFN RMSNorm");

            auto rmsnorm_kernel = int8_activation->createRMSNorm();
            VALIDATE_POINTER(rmsnorm_kernel, "Fused RMSNorm kernel");

            // CRITICAL: Process ALL batch rows, not just effective_seq_len
            const int total_rows = batch_size_ * padded_seq_len_;

            // Get INT8 buffer directly (don't use dangerous mutable_data() cast)
            auto *int8_tensor_ptr = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            VALIDATE_POINTER(int8_tensor_ptr, "INT8 tensor for quantized output");

            VALIDATE_OP(rmsnorm_kernel->execute(
                            buffers.residual->data(),             // input (FP32)
                            layer.ffn_norm->data(),               // weight
                            int8_tensor_ptr->mutable_int8_data(), // output (INT8)
                            buffers.normalized_scales.data(),     // scales
                            total_rows, d_model_,
                            1e-6f, mpi_ctx_.get(), ffn_device),
                        "FFN norm (fused INT8)");
            VALIDATE_TENSOR_BUFFER(buffers.normalized_int8,
                                   TensorSpec({static_cast<size_t>(effective_seq_len),
                                               static_cast<size_t>(d_model_)},
                                              "normalized_int8"),
                                   "after_fused_ffn_norm_quantize");

            // Register per-row scales with INT8 tensor (critical for correct dequantization)
            auto *int8_tensor = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            if (int8_tensor)
            {
                int8_tensor->set_row_scales(buffers.normalized_scales.data(), total_rows);

                // Debug: Verify scale sanity
                float min_scale = 1e9f, max_scale = -1e9f;
                for (int r = 0; r < total_rows; ++r)
                {
                    min_scale = std::min(min_scale, buffers.normalized_scales[r]);
                    max_scale = std::max(max_scale, buffers.normalized_scales[r]);
                }
                LOG_DEBUG("[FusedQuantize] FFN norm L" << layer_idx
                                                       << " scales: min=" << min_scale << " max=" << max_scale
                                                       << " total_rows=" << total_rows << " effective_seq_len=" << effective_seq_len);
            }

            // Capture INT8 normalized output (convert to FP32 for snapshot compatibility)
            // CRITICAL FIX: Dequantize only effective_seq_len rows (not total_rows)
            // BUG: Previously used total_rows, which accessed uninitialized scales beyond effective_seq_len,
            //      causing garbage in normalized_hidden that propagated through FFN projections
            auto *int8_tensor_data = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            if (int8_tensor_data)
            {
                const int8_t *int8_data = int8_tensor_data->int8_data();
                for (int r = 0; r < effective_seq_len; ++r) // ← FIX: Use effective_seq_len
                {
                    const float row_scale = buffers.normalized_scales[r];
                    for (int c = 0; c < d_model_; ++c)
                    {
                        const size_t idx = r * d_model_ + c;
                        normalized_hidden->mutable_data()[idx] =
                            static_cast<float>(int8_data[idx]) * row_scale;
                    }
                }
            }
        }
        else
        {
            // Unfused path (fallback): Separate RMSNorm + implicit quantization in GEMM
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
        }

        // Capture FFN norm output
        CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_NORM", normalized_hidden, effective_seq_len, d_model_);

        // DEBUG: Track FFN norm output
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *norm_data = normalized_hidden->data();
            size_t seq1_offset = 4 * d_model_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After FFN_NORM, Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << norm_data[seq1_offset + i]);
            }
        }

        // 2. Gate and up projections - PHASE 2 FUSED PATH
        // Use FusedDualGEMM to eliminate redundant quantization:
        //   Old: Quant(norm) → gate_gemm + Quant(norm) → up_gemm → Dequant(gate) + Dequant(up) → SwiGLU
        //   New: Quant(norm) → FusedDualGEMM → [gate_int32, up_int32] → FusedDequantSwiGLU → output_fp32
        if (use_fused_quantize)
        {
            // Phase 2: Fused path with FusedDualGEMM + FusedDequantSwiGLU
            // Create fused kernel from INT8 activation tensor
            auto *int8_activation = dynamic_cast<INT8Tensor *>(buffers.normalized_int8.get());
            VALIDATE_POINTER(int8_activation, "INT8 activation tensor for FusedDualGEMM");

            auto fused_dual_gemm = int8_activation->createFusedDualGemm(
                layer.gate_proj.get(), layer.up_proj.get());
            VALIDATE_POINTER(fused_dual_gemm, "FusedDualGEMM kernel");

            // Get INT32 buffers from activation buffers
            auto *gate_int32 = dynamic_cast<INT32Tensor *>(buffers.gate_int32.get());
            auto *up_int32 = dynamic_cast<INT32Tensor *>(buffers.up_int32.get());
            VALIDATE_POINTER(gate_int32, "INT32 gate buffer");
            VALIDATE_POINTER(up_int32, "INT32 up buffer");

            // Get column scales from weight tensors (for INT32 dequantization)
            auto *gate_weight_int8 = dynamic_cast<INT8Tensor *>(layer.gate_proj.get());
            auto *up_weight_int8 = dynamic_cast<INT8Tensor *>(layer.up_proj.get());
            VALIDATE_POINTER(gate_weight_int8, "INT8 gate weight");
            VALIDATE_POINTER(up_weight_int8, "INT8 up weight");

            const float *gate_col_scales = gate_weight_int8->col_scales();
            const float *up_col_scales = up_weight_int8->col_scales();
            VALIDATE_POINTER(gate_col_scales, "Gate weight column scales");
            VALIDATE_POINTER(up_col_scales, "Up weight column scales");

            // Execute FusedDualGEMM: normalized_fp32 → [gate_int32, up_int32]
            // Note: FusedDualGEMM internally quantizes FP32 input to INT8, then performs GEMMs
            // We pass normalized FP32 buffer (not INT8), as kernel handles quantization
            VALIDATE_OP(fused_dual_gemm->execute(
                            normalized_hidden->data(),           // Input FP32 [m, k]
                            gate_int32->mutable_int32_data(),    // Output INT32 [m, n]
                            up_int32->mutable_int32_data(),      // Output INT32 [m, n]
                            buffers.normalized_scales.data(),    // Output row scales [m]
                            effective_seq_len, d_ff_, d_model_), // Dimensions
                        "FusedDualGEMM (gate+up projections)");

            // Health check: INT32 accumulator outputs
            LOG_DEBUG("[Phase2] FusedDualGEMM completed, checking INT32 outputs");

            // Execute FusedDequantSwiGLU: [gate_int32, up_int32] → up_fp32 (with SwiGLU activation)
            FusedDequantSwiGLU fused_dequant_swiglu;
            VALIDATE_OP(fused_dequant_swiglu.execute(
                            gate_int32->int32_data(),         // Gate INT32 [m, n]
                            up_int32->int32_data(),           // Up INT32 [m, n]
                            buffers.up->mutable_data(),       // Output FP32 [m, n] (reuse up buffer)
                            buffers.normalized_scales.data(), // Row scales [m]
                            gate_col_scales,                  // Gate column scales [n]
                            up_col_scales,                    // Up column scales [n]
                            effective_seq_len, d_ff_),        // Dimensions
                        "FusedDequantSwiGLU (dequant+activation)");

            // Capture gate projection (dequantize INT32 for snapshot)
            // This is for parity testing - production code only stores INT32
            for (int r = 0; r < effective_seq_len; ++r)
            {
                const float row_scale = buffers.normalized_scales[r];
                for (int c = 0; c < d_ff_; ++c)
                {
                    const size_t idx = r * d_ff_ + c;
                    buffers.gate->mutable_data()[idx] =
                        static_cast<float>(gate_int32->int32_data()[idx]) * row_scale * gate_col_scales[c];
                }
            }
            CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_GATE",
                                  buffers.gate, effective_seq_len, d_ff_);

            LOG_DEBUG("[Phase2] FusedDequantSwiGLU completed");
        }
        else
        {
            // Phase 1: Unfused path (fallback for non-INT8 precision)
            VALIDATE_KERNEL(gate_gemm, layer.gate_proj->createGemm(), "gate GEMM kernel");
            VALIDATE_KERNEL(up_gemm, layer.up_proj->createGemm(), "up GEMM kernel");

            // gate = hidden @ gate_proj^T
            VALIDATE_OP(gate_gemm->multiply_activations(
                            normalized_hidden->data(), nullptr, buffers.gate->mutable_data(),
                            effective_seq_len, d_ff_, d_model_,
                            true, 1.0f, 0.0f, mpi_ctx_.get(), ffn_device),
                        "Gate projection");
            VALIDATE_TENSOR_BUFFER(buffers.gate, spec_ffn_gate_up(effective_seq_len), "after_gate_proj");

            // Capture gate projection
            CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_GATE",
                                  buffers.gate, effective_seq_len, d_ff_);

            // Health check: Gate projection output
            CHECK_NUMERICAL_HEALTH(("layer" + std::to_string(layer_idx) + "_FFN_GATE").c_str(),
                                   buffers.gate->data(), effective_seq_len * d_ff_);

            // up = hidden @ up_proj^T
            VALIDATE_OP(up_gemm->multiply_activations(
                            normalized_hidden->data(), nullptr, buffers.up->mutable_data(),
                            effective_seq_len, d_ff_, d_model_,
                            true, 1.0f, 0.0f, mpi_ctx_.get(), ffn_device),
                        "Up projection");
            VALIDATE_TENSOR_BUFFER(buffers.up, spec_ffn_gate_up(effective_seq_len), "after_up_proj");

            // Capture up projection
            CAPTURE_SNAPSHOT_VIEW("layer" + std::to_string(layer_idx) + "_FFN_UP",
                                  buffers.up, effective_seq_len, d_ff_);

            // SwiGLU activation (up buffer reused for output)
            auto *activation_tensor_up = dynamic_cast<IActivationTensor *>(buffers.up.get());
            VALIDATE_POINTER(activation_tensor_up, "activation tensor for SwiGLU kernel");
            VALIDATE_KERNEL(swiglu_kernel, activation_tensor_up->createSwiGLU(), "SwiGLU kernel");

            VALIDATE_OP(swiglu_kernel->apply(
                            buffers.gate->data(), buffers.up->data(),
                            buffers.up->mutable_data(),
                            effective_seq_len, d_ff_, false, mpi_ctx_.get(), ffn_device),
                        "SwiGLU activation");
        }
        VALIDATE_TENSOR_BUFFER(buffers.up, spec_ffn_intermediate(effective_seq_len), "after_swiglu");

        // Capture SwiGLU output
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

        // DEBUG: Track FFN down projection output
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *down_data = buffers.ffn_output->data();
            size_t seq1_offset = 4 * d_model_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After FFN_DOWN, Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << down_data[seq1_offset + i]);
            }
        }

        // Health check: FFN output before residual (final dequantized output from INT8 path)
        CHECK_NUMERICAL_HEALTH(("layer" + std::to_string(layer_idx) + "_FFN_DOWN").c_str(),
                               buffers.ffn_output->data(), effective_seq_len * d_model_);

        // 5. Residual connection - write back to current_hidden_
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

        // DEBUG: Track after FFN residual connection (end of layer)
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *residual_data = current_hidden_->data();
            size_t seq1_offset = 4 * d_model_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After FFN_RESIDUAL (END OF LAYER), Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << residual_data[seq1_offset + i]);
            }
            LOG_ERROR("[Layer " << layer_idx << "] ========================================");
        }

        // DEBUG: Track after FFN residual connection (end of layer)
        if (debug_batch_env && layer_idx < 3 && batch_size_ == 2)
        {
            const float *residual_data = current_hidden_->data();
            size_t seq1_offset = 4 * d_model_; // Seq1 starts at position 4
            LOG_ERROR("[Layer " << layer_idx << "] After FFN_RESIDUAL (END OF LAYER), Seq1 token0 [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << residual_data[seq1_offset + i]);
            }
            LOG_ERROR("[Layer " << layer_idx << "] ========================================");
        }

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

            if (debug_batch && mpi_ctx_->rank() == 0)
            {
                LOG_ERROR("[embedding_batch] Batch " << b << ": seq_len=" << seq_len
                                                     << ", padded_seq_len=" << padded_seq_len_);
            }

            // Lookup embeddings for this sequence
            for (int i = 0; i < seq_len; ++i)
            {
                int token_id = tokens[i];
                DEBUG_ASSERT_RANGE(token_id, 0, vocab_size_,
                                   "Invalid token at batch=" << b << ", pos=" << i);
                std::memcpy(output_data + global_idx * d_model_,
                            embed_data + token_id * d_model_,
                            d_model_ * sizeof(float));

                if (debug_batch && mpi_ctx_->rank() == 0 && i < 2)
                {
                    LOG_ERROR("[embedding_batch] Batch " << b << ", token " << i
                                                         << " (id=" << token_id << "): global_idx=" << global_idx
                                                         << ", first_value=" << output_data[global_idx * d_model_]);
                }

                global_idx++;
            }

            // Pad remaining positions with zeros (or pad token embedding)
            for (int i = seq_len; i < padded_seq_len_; ++i)
            {
                std::memset(output_data + global_idx * d_model_, 0, d_model_ * sizeof(float));

                if (debug_batch && mpi_ctx_->rank() == 0 && i == seq_len)
                {
                    LOG_ERROR("[embedding_batch] Batch " << b << ", padding starts at global_idx=" << global_idx);
                }

                global_idx++;
            }
        }

        if (debug_batch && mpi_ctx_->rank() == 0)
        {
            LOG_ERROR("[embedding_batch] Total positions filled: " << global_idx
                                                                   << " (expected: " << (batch_size_ * padded_seq_len_) << ")");
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

        // DEBUG: Track LM head output (logits)
        static const bool debug_batch_env = std::getenv("LLAMINAR_DEBUG_BATCH") != nullptr;
        if (debug_batch_env && batch_size_ == 2)
        {
            const float *logits_data = logits_buffer_->data();

            // Show position 4 (first token of Seq1)
            size_t pos4_offset = 4 * vocab_size_;
            LOG_ERROR("[LM_HEAD] Position 4 (Seq1 token0) logits [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << logits_data[pos4_offset + i]);
            }

            // Show position 5 (second token of Seq1)
            size_t pos5_offset = 5 * vocab_size_;
            LOG_ERROR("[LM_HEAD] Position 5 (Seq1 token1) logits [0:10]:");
            for (int i = 0; i < 10; ++i)
            {
                LOG_ERROR("  [" << i << "] = " << logits_data[pos5_offset + i]);
            }
        }

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
