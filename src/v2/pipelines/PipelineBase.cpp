/**
 * @file PipelineBase.cpp
 * @brief Base pipeline implementation
 * @author David Sanftenberg
 */

#include "../utils/Logger.h"
#include "../utils/DebugAssert.h"
#include "PipelineBase.h"
#include "AttentionUtils.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/Tensors.h"
#include "../kernels/cpu/primitives/SoftmaxPrimitives.h"
#include "../kernels/cpu/primitives/RoPEPrimitives.h"
#include "../kernels/cpu/primitives/RMSNormPrimitives.h"
#include "../kernels/cpu/FP32StandaloneGemm.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <set>
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{

    PipelineBase::PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                               std::shared_ptr<MPIContext> mpi_ctx,
                               int device_idx,
                               std::shared_ptr<WeightPlacementMap> placement_map,
                               const PipelineConfig &config)
        : model_ctx_(model_ctx), mpi_ctx_(mpi_ctx), device_idx_(device_idx), config_(config), placement_map_(placement_map)
    {
        if (!model_ctx_)
        {
            throw std::runtime_error("PipelineBase: model_ctx cannot be null");
        }

        model_path_ = model_ctx_->path();

        LOG_INFO("[PipelineBase] Initializing with model: " << model_path_);
        LOG_INFO("[PipelineBase] Runtime config: max_seq_len=" << config_.max_seq_len
                                                               << ", n_threads=" << config_.n_threads << ", batch_size=" << config_.batch_size);

        if (mpi_ctx_)
        {
            std::cout << "[PipelineBase] MPI context provided, rank "
                      << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size() << "\n";
        }

        if (device_idx_ >= 0)
        {
            LOG_INFO("[PipelineBase] Device index: " << device_idx_ << " (GPU)\n");
            // TODO Phase 4: GPU tensor support
        }
        else
        {
            LOG_INFO("[PipelineBase] Device index: " << device_idx_ << " (CPU)\n");
        }

        // Create default placement map if not provided (all weights on device_idx_)
        if (!placement_map_)
        {
            std::cout << "[PipelineBase] No placement map provided, creating default (all on device "
                      << device_idx_ << ")\n";
            placement_map_ = std::make_shared<WeightPlacementMap>(device_idx_);
        }

        // Generic pipeline initialization (derived classes have already set architecture params)
        // Derived constructors must set n_layers_, n_heads_, n_kv_heads_, head_dim_, d_model_
        // BEFORE calling initializeInfrastructure()
    }

    void PipelineBase::initializeInfrastructure()
    {
        // Use max_seq_len from runtime configuration
        int max_seq_len = config_.max_seq_len;

        // Phase 4.1: Device infrastructure (device discovery, buffer allocation)
        initializeDeviceInfrastructure(max_seq_len);

        // Phase 2: MPI strategy configuration (auto-select or validate)
        configureMPIStrategy();

        // Phase 3: KV cache initialization (uses attention device placement)
        initializeKVCache(max_seq_len);

        LOG_INFO("Pipeline infrastructure initialized (max_seq_len=" << max_seq_len << ")");
    }
    const float *PipelineBase::logits() const
    {
        DEBUG_ASSERT_NOT_NULL(logits_.get(), "logits() called before forward()");
        return logits_->data();
    }

    bool PipelineBase::attention_gqa(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        int batch_size, const std::vector<int> *sequence_lengths)
    {
        // Validate inputs
        if (!Q || !K || !V || !output)
        {
            LOG_ERROR("[PipelineBase] attention_gqa: null pointer\n");
            return false;
        }

        if (n_heads % n_kv_heads != 0)
        {
            std::cerr << "[PipelineBase] attention_gqa: n_heads (" << n_heads
                      << ") must be divisible by n_kv_heads (" << n_kv_heads << ")\n";
            return false;
        }

        // Infer seq_len from Q shape: [seq_len, n_heads * head_dim]
        const auto &q_shape = Q->shape();
        if (q_shape.size() != 2)
        {
            LOG_ERROR("[PipelineBase] attention_gqa: Q must be 2D\n");
            return false;
        }
        int seq_len = static_cast<int>(q_shape[0]);

        // Get tensor data pointers
        const float *Q_data = Q->data();
        const float *K_data = K->data();
        const float *V_data = V->data();
        float *output_data = output->mutable_data();

        // Broadcast K/V heads to match Q heads (if needed)
        std::vector<float> K_broadcast;
        std::vector<float> V_broadcast;
        const float *K_expanded = K_data;
        const float *V_expanded = V_data;

        if (n_kv_heads < n_heads)
        {
            // Need to broadcast K/V
            K_broadcast.resize(seq_len * n_heads * head_dim);
            V_broadcast.resize(seq_len * n_heads * head_dim);

            attention_utils::broadcast_kv_heads(
                K_data, K_broadcast.data(),
                seq_len, n_heads, n_kv_heads, head_dim);

            attention_utils::broadcast_kv_heads(
                V_data, V_broadcast.data(),
                seq_len, n_heads, n_kv_heads, head_dim);

            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // Create temporary FP32 tensors for scores computation
        // scores: [n_heads, seq_len, seq_len]
        auto scores_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)});

        float *scores = scores_tensor->mutable_data();

        // Compute attention scores per head: Q @ K^T
        // We'll process each head separately to handle the strided layout
        for (int h = 0; h < n_heads; ++h)
        {
            // Extract contiguous head data for Q and K
            std::vector<float> Q_h(seq_len * head_dim);
            std::vector<float> K_h(seq_len * head_dim);

            for (int s = 0; s < seq_len; ++s)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    Q_h[s * head_dim + d] = Q_data[s * n_heads * head_dim + h * head_dim + d];
                    K_h[s * head_dim + d] = K_expanded[s * n_heads * head_dim + h * head_dim + d];
                }
            }

            // GEMM: scores[h] = Q_h @ K_h^T
            // Q_h: [seq_len, head_dim], K_h: [seq_len, head_dim]
            // scores[h]: [seq_len, seq_len]
            float *scores_h = scores + h * seq_len * seq_len;

            LOG_DEBUG("[Baseline] Head " << h << ": Q_h[0]=" << Q_h[0]
                                         << " K_h[0]=" << K_h[0]);

            if (!FP32StandaloneGemm::multiply_with_b(
                    Q_h.data(), K_h.data(), scores_h,
                    seq_len, seq_len, head_dim,
                    true, 1.0f, 0.0f))
            {
                LOG_ERROR("[PipelineBase] attention_gqa: Q·K^T GEMM failed for head " << h);
                return false;
            }

            LOG_DEBUG("[Baseline] Head " << h << ": scores[0]=" << scores_h[0]
                                         << " scores[" << (seq_len * seq_len - 1) << "]=" << scores_h[seq_len * seq_len - 1]);
        }

        // Scale scores by 1/sqrt(head_dim) - vectorized
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        LOG_DEBUG("[Baseline] Scaling scores by " << scale);
#pragma omp parallel for if (n_heads * seq_len * seq_len > 8192)
        for (int i = 0; i < n_heads * seq_len * seq_len; ++i)
        {
            scores[i] *= scale;
        }
        LOG_DEBUG("[Baseline] After scaling: scores[0]=" << scores[0]);

        // Apply causal mask (if enabled)
        if (causal)
        {
            LOG_DEBUG("[Baseline] Applying causal mask (batch_size=" << batch_size << ")");
            std::vector<float> mask(seq_len * seq_len);

            if (batch_size == 1)
            {
                // Single sequence: standard causal mask
                attention_utils::create_causal_mask(mask.data(), seq_len, window_size);
            }
            else
            {
                // Batched sequences: block-diagonal causal mask with padding
                int padded_seq_len = seq_len / batch_size;
                const int *seq_lens_ptr = sequence_lengths ? sequence_lengths->data() : nullptr;
                attention_utils::create_batch_causal_mask(
                    mask.data(), batch_size, padded_seq_len, seq_lens_ptr, window_size);
            }

            // Apply mask to each head separately
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len;
                attention_utils::apply_attention_mask(scores_h, mask.data(), seq_len, seq_len);
            }
            LOG_DEBUG("[Baseline] After causal mask: scores[0]=" << scores[0]
                                                                 << " scores[" << (seq_len - 1) << "]=" << scores[seq_len - 1]);
        }

        // Apply softmax - using vectorized primitives
        primitives::SoftmaxRowArgs softmax_args;
        softmax_args.causal = false; // Already masked above if needed
        softmax_args.scale = 1.0f;   // Already scaled above
        softmax_args.rows = n_heads * seq_len;
        softmax_args.cols = seq_len;
        softmax_args.scores = scores;

        primitives::softmax_row_major_vectorized(softmax_args);
        LOG_DEBUG("[Baseline] After softmax: scores[0]=" << scores[0]
                                                         << " scores[" << (seq_len - 1) << "]=" << scores[seq_len - 1]);

        // Compute context: scores @ V
        std::memset(output_data, 0, seq_len * n_heads * head_dim * sizeof(float));

        for (int h = 0; h < n_heads; ++h)
        {
            // Extract contiguous V_h data
            std::vector<float> V_h(seq_len * head_dim);
            for (int s = 0; s < seq_len; ++s)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    V_h[s * head_dim + d] = V_expanded[s * n_heads * head_dim + h * head_dim + d];
                }
            }

            // Temporary contiguous output for this head
            std::vector<float> context_h(seq_len * head_dim);

            // GEMM: context_h = scores[h] @ V_h
            const float *scores_h = scores + h * seq_len * seq_len;

            if (!FP32StandaloneGemm::multiply_with_b(
                    scores_h, V_h.data(), context_h.data(),
                    seq_len, head_dim, seq_len,
                    false, 1.0f, 0.0f))
            {
                LOG_ERROR("[PipelineBase] attention_gqa: scores·V GEMM failed for head " << h);
                return false;
            }

            // Write back to strided output
            for (int s = 0; s < seq_len; ++s)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    output_data[s * n_heads * head_dim + h * head_dim + d] = context_h[s * head_dim + d];
                }
            }
        }

        return true;
    }

    bool PipelineBase::attention_gqa_batch(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        const std::vector<int> &actual_lengths,
        int batch_size, int seq_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size)
    {
        // Validate inputs
        if (!Q || !K || !V || !output)
        {
            LOG_ERROR("[PipelineBase] attention_gqa_batch: null pointer");
            return false;
        }

        if (n_heads % n_kv_heads != 0)
        {
            LOG_ERROR("[PipelineBase] attention_gqa_batch: n_heads (" << n_heads
                                                                      << ") must be divisible by n_kv_heads (" << n_kv_heads << ")");
            return false;
        }

        if (static_cast<int>(actual_lengths.size()) != batch_size)
        {
            LOG_ERROR("[PipelineBase] attention_gqa_batch: actual_lengths size ("
                      << actual_lengths.size() << ") != batch_size (" << batch_size << ")");
            return false;
        }

        // Validate tensor shapes
        const auto &q_shape = Q->shape();
        if (q_shape.size() != 2)
        {
            LOG_ERROR("[PipelineBase] attention_gqa_batch: Q must be 2D");
            return false;
        }

        const int total_seq_len = batch_size * seq_len;
        if (static_cast<int>(q_shape[0]) != total_seq_len)
        {
            LOG_ERROR("[PipelineBase] attention_gqa_batch: Q shape[0] ("
                      << q_shape[0] << ") != batch_size * seq_len (" << total_seq_len << ")");
            return false;
        }

        // Get tensor data pointers
        const float *Q_data = Q->data();
        const float *K_data = K->data();
        const float *V_data = V->data();
        float *output_data = output->mutable_data();

        // Broadcast K/V heads if needed (GQA)
        std::vector<float> K_broadcast;
        std::vector<float> V_broadcast;
        const float *K_expanded = K_data;
        const float *V_expanded = V_data;

        if (n_kv_heads < n_heads)
        {
            // Need to broadcast K/V
            K_broadcast.resize(total_seq_len * n_heads * head_dim);
            V_broadcast.resize(total_seq_len * n_heads * head_dim);

            attention_utils::broadcast_kv_heads(
                K_data, K_broadcast.data(),
                total_seq_len, n_heads, n_kv_heads, head_dim);

            attention_utils::broadcast_kv_heads(
                V_data, V_broadcast.data(),
                total_seq_len, n_heads, n_kv_heads, head_dim);

            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // Create combined causal + padding mask [batch_size, seq_len, seq_len]
        std::vector<float> combined_mask(batch_size * seq_len * seq_len);
        attention_utils::create_combined_batch_mask(
            combined_mask.data(),
            batch_size,
            seq_len,
            actual_lengths.data(),
            causal,
            window_size);

        // Create temporary tensor for scores: [batch_size * n_heads, seq_len, seq_len]
        auto scores_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(batch_size * n_heads * seq_len), static_cast<size_t>(seq_len)});
        float *scores = scores_tensor->mutable_data();

        // Compute attention scores per batch and head: Q @ K^T
        // Process each batch and head separately
        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                // Extract contiguous head data for this batch
                std::vector<float> Q_bh(seq_len * head_dim);
                std::vector<float> K_bh(seq_len * head_dim);

                const int batch_offset = b * seq_len;
                for (int s = 0; s < seq_len; ++s)
                {
                    for (int d = 0; d < head_dim; ++d)
                    {
                        const int global_pos = (batch_offset + s) * n_heads * head_dim + h * head_dim + d;
                        Q_bh[s * head_dim + d] = Q_data[global_pos];
                        K_bh[s * head_dim + d] = K_expanded[global_pos];
                    }
                }

                // GEMM: scores[b,h] = Q_bh @ K_bh^T
                // Q_bh: [seq_len, head_dim], K_bh: [seq_len, head_dim]
                // scores[b,h]: [seq_len, seq_len]
                float *scores_bh = scores + (b * n_heads + h) * seq_len * seq_len;

                if (!FP32StandaloneGemm::multiply_with_b(
                        Q_bh.data(), K_bh.data(), scores_bh,
                        seq_len, seq_len, head_dim,
                        true, 1.0f, 0.0f))
                {
                    LOG_ERROR("[PipelineBase] attention_gqa_batch: Q·K^T GEMM failed for batch " << b << " head " << h);
                    return false;
                }
            }
        }

        // Scale scores by 1/sqrt(head_dim)
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
#pragma omp parallel for if (batch_size * n_heads * seq_len * seq_len > 8192)
        for (int i = 0; i < batch_size * n_heads * seq_len * seq_len; ++i)
        {
            scores[i] *= scale;
        }

        // Apply combined causal + padding mask
        // Broadcast mask across heads: mask[b, seq_len, seq_len] -> all heads for batch b
        for (int b = 0; b < batch_size; ++b)
        {
            const float *mask_b = combined_mask.data() + b * seq_len * seq_len;
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_bh = scores + (b * n_heads + h) * seq_len * seq_len;
                attention_utils::apply_attention_mask(scores_bh, mask_b, seq_len, seq_len);
            }
        }

        // Apply softmax per batch and head
        primitives::SoftmaxRowArgs softmax_args;
        softmax_args.causal = false; // Already masked above
        softmax_args.scale = 1.0f;   // Already scaled above
        softmax_args.rows = batch_size * n_heads * seq_len;
        softmax_args.cols = seq_len;
        softmax_args.scores = scores;

        primitives::softmax_row_major_vectorized(softmax_args);

        // Compute context: scores @ V
        std::memset(output_data, 0, total_seq_len * n_heads * head_dim * sizeof(float));

        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                // Extract contiguous V_bh data
                std::vector<float> V_bh(seq_len * head_dim);
                const int batch_offset = b * seq_len;
                for (int s = 0; s < seq_len; ++s)
                {
                    for (int d = 0; d < head_dim; ++d)
                    {
                        const int global_pos = (batch_offset + s) * n_heads * head_dim + h * head_dim + d;
                        V_bh[s * head_dim + d] = V_expanded[global_pos];
                    }
                }

                // Temporary contiguous output for this batch-head
                std::vector<float> context_bh(seq_len * head_dim);

                // GEMM: context_bh = scores[b,h] @ V_bh
                const float *scores_bh = scores + (b * n_heads + h) * seq_len * seq_len;

                if (!FP32StandaloneGemm::multiply_with_b(
                        scores_bh, V_bh.data(), context_bh.data(),
                        seq_len, head_dim, seq_len,
                        false, 1.0f, 0.0f))
                {
                    LOG_ERROR("[PipelineBase] attention_gqa_batch: scores·V GEMM failed for batch " << b << " head " << h);
                    return false;
                }

                // Write back to strided output
                for (int s = 0; s < seq_len; ++s)
                {
                    for (int d = 0; d < head_dim; ++d)
                    {
                        const int global_pos = (batch_offset + s) * n_heads * head_dim + h * head_dim + d;
                        output_data[global_pos] = context_bh[s * head_dim + d];
                    }
                }
            }
        }

        return true;
    }

    // =============================================================================
    // Multi-Device Infrastructure (Phase 4)
    // =============================================================================

    std::vector<int> PipelineBase::discoverActiveDevices()
    {
        std::set<int> device_set;

        // Get all weight names from derived class (architecture-specific)
        std::vector<std::string> weight_names = getAllWeightNames();

        // Query placement map for each weight
        for (const auto &weight_name : weight_names)
        {
            // Try to extract layer index from weight name (e.g., "blk.5.attn_q.weight" -> 5)
            // This is a heuristic - some weights don't have layer indices
            int layer_idx = -1;
            size_t blk_pos = weight_name.find("blk.");
            if (blk_pos != std::string::npos)
            {
                size_t dot_pos = weight_name.find('.', blk_pos + 4);
                if (dot_pos != std::string::npos)
                {
                    std::string layer_str = weight_name.substr(blk_pos + 4, dot_pos - (blk_pos + 4));
                    try
                    {
                        layer_idx = std::stoi(layer_str);
                    }
                    catch (...)
                    {
                        // Not a valid layer index, keep -1
                    }
                }
            }

            int device = placement_map_->getDeviceForWeight(weight_name, layer_idx);
            device_set.insert(device);
        }

        // Convert set to sorted vector
        std::vector<int> devices(device_set.begin(), device_set.end());
        std::sort(devices.begin(), devices.end());

        return devices;
    }

    // =============================================================================
    // Phase 3: MoE Device Placement Helpers
    // =============================================================================

    std::vector<int> PipelineBase::detectAttentionDevices(int n_layers) const
    {
        std::vector<int> attention_devices(n_layers);

        for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
        {
            // Query placement map for attention block device
            // Uses Phase 2 block-level method (getAttentionDevice)
            attention_devices[layer_idx] = placement_map_->getAttentionDevice(layer_idx);
        }

        return attention_devices;
    }

    std::vector<int> PipelineBase::detectFFNDevices(int n_layers) const
    {
        std::vector<int> ffn_devices(n_layers);

        for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
        {
            // Query placement map for FFN block device
            // Uses Phase 2 block-level method (getFFNDevice)
            ffn_devices[layer_idx] = placement_map_->getFFNDevice(layer_idx);
        }

        return ffn_devices;
    }

    ActivationBuffers &PipelineBase::getBuffersForDevice(int device_idx)
    {
        // Check if we already have buffers for this device
        auto it = buffers_per_device_.find(device_idx);
        if (it != buffers_per_device_.end())
        {
            return it->second;
        }

        // Lazy allocation: create buffers for this device
        LOG_INFO("[PipelineBase] Lazy allocating buffers for device " << device_idx);

        // Determine max_seq_len from existing buffers (or use default)
        int max_seq_len = 2048; // Default
        if (!buffers_per_device_.empty())
        {
            max_seq_len = buffers_per_device_.begin()->second.max_seq_len;
        }

        // Call derived class to create buffers with architecture-specific dimensions
        ActivationBuffers buffers = createBuffersForDevice(device_idx, max_seq_len);

        // Insert into map
        auto [inserted_it, success] = buffers_per_device_.emplace(device_idx, std::move(buffers));
        if (!success)
        {
            throw std::runtime_error("Failed to insert buffers for device " + std::to_string(device_idx));
        }

        return inserted_it->second;
    }

    int PipelineBase::getWeightDevice(const std::string &weight_name, int layer_idx) const
    {
        return placement_map_->getDeviceForWeight(weight_name, layer_idx);
    }

    TensorBase *PipelineBase::prepareActivationForDevice(TensorBase *activation, int target_device, const std::string &context)
    {
        if (!activation)
        {
            LOG_ERROR("[PipelineBase] prepareActivationForDevice: null activation for " << context);
            return nullptr;
        }

        int current_device = activation->device_index();

        // Fast path: already on target device
        if (current_device == target_device)
        {
            return activation;
        }

        // Transfer required
        std::cout << "[PipelineBase] [" << context << "] Transferring activation from device "
                  << current_device << " to device " << target_device << "\n";

        // Get target device's buffers
        ActivationBuffers &target_buffers = const_cast<PipelineBase *>(this)->getBuffersForDevice(target_device);

        // Use residual buffer as staging area
        TensorBase *staging = target_buffers.residual.get();

        // Compute element counts from shapes
        auto compute_element_count = [](const std::vector<size_t> &shape) -> size_t
        {
            size_t count = 1;
            for (auto dim : shape)
                count *= dim;
            return count;
        };

        size_t staging_count = compute_element_count(staging->shape());
        size_t activation_count = compute_element_count(activation->shape());

        // Validate staging buffer size
        if (staging_count < activation_count)
        {
            std::cerr << "[PipelineBase] Staging buffer too small: " << staging_count
                      << " < " << activation_count << "\n";
            return nullptr;
        }

        // Perform transfer
        if (!staging->copyFrom(activation))
        {
            LOG_ERROR("[PipelineBase] Failed to transfer activation to device " << target_device);
            return nullptr;
        }

        // Update staging buffer's device index
        staging->set_device(target_device);

        return staging;
    }

    // =============================================================================
    // MPI-Aware Attention Methods
    // =============================================================================

    bool PipelineBase::attention_gqa_mpi(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        int batch_size, const std::vector<int> *sequence_lengths)
    {
        // Fast path: No MPI or single-rank execution
        if (!mpi_ctx_ || mpi_ctx_->world_size() == 1 || mpi_strategy_ == MPIStrategy::None)
        {
            return attention_gqa(Q, K, V, output, n_heads, n_kv_heads, head_dim, causal, window_size, batch_size, sequence_lengths);
        }

        // Dispatch based on MPI strategy
        switch (mpi_strategy_)
        {
        case MPIStrategy::TensorParallel:
            return attention_gqa_tensor_parallel(Q, K, V, output, n_heads, n_kv_heads, head_dim, causal, window_size, batch_size, sequence_lengths);

        case MPIStrategy::SequenceParallel:
            // TODO: Implement sequence-parallel attention (Phase 6)
            LOG_ERROR("[PipelineBase] SequenceParallel attention not yet implemented");
            return false;

        case MPIStrategy::PipelineParallel:
            // Pipeline-parallel doesn't change attention (distributes layers instead)
            return attention_gqa(Q, K, V, output, n_heads, n_kv_heads, head_dim, causal, window_size, batch_size, sequence_lengths);

        case MPIStrategy::Hybrid:
            // TODO: Implement hybrid strategy (Phase 6)
            LOG_ERROR("[PipelineBase] Hybrid strategy not yet implemented");
            return false;

        default:
            LOG_ERROR("[PipelineBase] Unknown MPI strategy: " << static_cast<int>(mpi_strategy_));
            return false;
        }
    }

    bool PipelineBase::attention_gqa_tensor_parallel(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        int batch_size, const std::vector<int> *sequence_lengths)
    {
        // Validate MPI context
        if (!mpi_ctx_)
        {
            LOG_ERROR("[PipelineBase] Tensor-parallel attention requires MPI context");
            return false;
        }

        int rank = mpi_ctx_->rank();
        int world_size = mpi_ctx_->world_size();

        // Validate inputs
        if (!Q || !K || !V || !output)
        {
            LOG_ERROR("[PipelineBase] attention_gqa_tensor_parallel: null pointer");
            return false;
        }

        if (n_heads % n_kv_heads != 0)
        {
            LOG_ERROR("[PipelineBase] n_heads (" << n_heads << ") must be divisible by n_kv_heads (" << n_kv_heads << ")");
            return false;
        }

        // Validate divisibility (should have been checked in constructor, but double-check)
        if (n_heads % world_size != 0)
        {
            LOG_ERROR("[PipelineBase] Tensor-parallel requires n_heads (" << n_heads
                                                                          << ") divisible by world_size (" << world_size << ")");
            return false;
        }

        // Infer dimensions from Q shape and batch_size
        const auto &q_shape = Q->shape();
        if (q_shape.size() != 2)
        {
            LOG_ERROR("[PipelineBase] Q must be 2D, got " << q_shape.size() << "D");
            return false;
        }
        int total_tokens = static_cast<int>(q_shape[0]);

        // Compute per-batch sequence length
        int effective_batch_size = (batch_size > 0) ? batch_size : 1;
        int seq_len = total_tokens / effective_batch_size;

        if (total_tokens % effective_batch_size != 0)
        {
            LOG_ERROR("[PipelineBase] total_tokens (" << total_tokens
                                                      << ") not divisible by batch_size (" << effective_batch_size << ")");
            return false;
        }

        LOG_DEBUG("[MPI TP] Batch-aware attention: total_tokens=" << total_tokens
                                                                  << " batch_size=" << effective_batch_size
                                                                  << " seq_len_per_batch=" << seq_len);

        int padded_seq_len = seq_len;

        // 1. Distribute attention heads across ranks
        auto [start_head, local_n_heads] = getHeadDistribution(n_heads);

        if (mpi_config_.verbose_logging && rank == 0)
        {
            LOG_INFO("[MPI TensorParallel] Attention: n_heads=" << n_heads
                                                                << ", world_size=" << world_size
                                                                << ", local_n_heads=" << local_n_heads);
        }

        if (rank == 0 || mpi_config_.verbose_logging)
        {
            LOG_INFO("[MPI TensorParallel] Rank " << rank << "/" << world_size
                                                  << ": Computing heads [" << start_head << ", " << (start_head + local_n_heads - 1) << "]");
        }

        // Get tensor data pointers
        const float *Q_data = Q->data();
        const float *K_data = K->data();
        const float *V_data = V->data();
        float *output_data = output->mutable_data();

        // 2. Broadcast K/V heads if needed (GQA)
        std::vector<float> K_broadcast;
        std::vector<float> V_broadcast;
        const float *K_expanded = K_data;
        const float *V_expanded = V_data;

        if (n_kv_heads < n_heads)
        {
            K_broadcast.resize(total_tokens * n_heads * head_dim);
            V_broadcast.resize(total_tokens * n_heads * head_dim);

            attention_utils::broadcast_kv_heads(
                K_data, K_broadcast.data(),
                total_tokens, n_heads, n_kv_heads, head_dim);

            attention_utils::broadcast_kv_heads(
                V_data, V_broadcast.data(),
                total_tokens, n_heads, n_kv_heads, head_dim);

            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // 3. Allocate local output buffer for this rank's heads
        // [total_tokens, local_n_heads * head_dim] - use total_tokens to cover all batches
        std::vector<float> local_output(total_tokens * local_n_heads * head_dim, 0.0f);

        // Temporary scores for local heads: [local_n_heads, total_tokens, total_tokens]
        auto local_scores_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(local_n_heads * total_tokens), static_cast<size_t>(total_tokens)});
        float *local_scores = local_scores_tensor->mutable_data();

        // 4. Compute attention for local heads only
        for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
        {
            size_t global_h = start_head + local_h;

            // Extract contiguous Q_h and K_h for this head
            std::vector<float> Q_h(total_tokens * head_dim);
            std::vector<float> K_h(total_tokens * head_dim);

            for (int t = 0; t < total_tokens; ++t)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    Q_h[t * head_dim + d] = Q_data[t * n_heads * head_dim + global_h * head_dim + d];
                    K_h[t * head_dim + d] = K_expanded[t * n_heads * head_dim + global_h * head_dim + d];
                }
            }

            // GEMM: scores[local_h] = Q_h @ K_h^T
            float *scores_h = local_scores + local_h * total_tokens * total_tokens;

            LOG_DEBUG("[MPI TP] Rank " << rank << " Head " << global_h << " (local " << local_h << "): Q[0]="
                                       << Q_h[0] << " K[0]=" << K_h[0]);

            if (!FP32StandaloneGemm::multiply_with_b(
                    Q_h.data(), K_h.data(), scores_h,
                    total_tokens, total_tokens, head_dim,
                    true, 1.0f, 0.0f))
            {
                LOG_ERROR("[PipelineBase] Q·K^T GEMM failed for local head " << local_h);
                return false;
            }

            LOG_DEBUG("[MPI TP] Rank " << rank << " Head " << global_h << ": scores[0]=" << scores_h[0]
                                       << " scores[" << (total_tokens * total_tokens - 1) << "]=" << scores_h[total_tokens * total_tokens - 1]);
        }

        // 5. Scale scores by 1/sqrt(head_dim) - vectorized
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        LOG_DEBUG("[MPI TP] Rank " << rank << ": Scaling scores by " << scale
                                   << " (1/sqrt(" << head_dim << "))");
#pragma omp parallel for if (local_n_heads * total_tokens * total_tokens > 8192)
        for (int i = 0; i < local_n_heads * total_tokens * total_tokens; ++i)
        {
            local_scores[i] *= scale;
        }
        LOG_DEBUG("[MPI TP] Rank " << rank << " after scaling: scores[0]=" << local_scores[0]
                                   << " scores[" << (local_n_heads * total_tokens * total_tokens - 1) << "]="
                                   << local_scores[local_n_heads * total_tokens * total_tokens - 1]);

        // 6. Apply causal mask (if enabled)
        if (causal)
        {
            LOG_DEBUG("[MPI TP] Rank " << rank << ": Applying causal mask (batch_size=" << batch_size << ")");
            std::vector<float> mask(total_tokens * total_tokens);

            if (effective_batch_size == 1)
            {
                // Single sequence: standard causal mask
                attention_utils::create_causal_mask(mask.data(), total_tokens, window_size);
            }
            else
            {
                // Batched sequences: block-diagonal causal mask with padding
                const int *seq_lens_ptr = sequence_lengths ? sequence_lengths->data() : nullptr;
                attention_utils::create_batch_causal_mask(
                    mask.data(), effective_batch_size, padded_seq_len, seq_lens_ptr, window_size);
            }

            for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
            {
                float *scores_h = local_scores + local_h * total_tokens * total_tokens;
                attention_utils::apply_attention_mask(scores_h, mask.data(), total_tokens, total_tokens);
            }
            LOG_DEBUG("[MPI TP] Rank " << rank << " after masking: scores[0]=" << local_scores[0]
                                       << " scores[" << (total_tokens - 1) << "]=" << local_scores[total_tokens - 1]);
        }

        // 7. Softmax over local scores - using vectorized primitives
        primitives::SoftmaxRowArgs softmax_args;
        softmax_args.causal = false; // Already masked in step 6
        softmax_args.scale = 1.0f;   // Already scaled in step 5
        softmax_args.rows = local_n_heads * total_tokens;
        softmax_args.cols = total_tokens;
        softmax_args.scores = local_scores;

        // Use vectorized row-major softmax (AVX512/AVX2/scalar fallback)
        primitives::softmax_row_major_vectorized(softmax_args);

        LOG_DEBUG("[MPI TP] Rank " << rank << " after softmax: scores[0]=" << local_scores[0]
                                   << " scores[" << (total_tokens - 1) << "]=" << local_scores[total_tokens - 1]
                                   << " (should sum to ~1.0 per row)");

        if (mpi_config_.verbose_logging && rank == 0)
        {
            LOG_INFO("[MPI TensorParallel] Applied vectorized softmax to " << local_n_heads << " heads");
        }

        // 8. Compute context: scores @ V for local heads
        for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
        {
            size_t global_h = start_head + local_h;

            // Extract contiguous V_h for this head
            std::vector<float> V_h(total_tokens * head_dim);
            for (int t = 0; t < total_tokens; ++t)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    V_h[t * head_dim + d] = V_expanded[t * n_heads * head_dim + global_h * head_dim + d];
                }
            }

            // Temporary contiguous context for this head
            std::vector<float> context_h(total_tokens * head_dim);

            const float *scores_h = local_scores + local_h * total_tokens * total_tokens;

            if (!FP32StandaloneGemm::multiply_with_b(
                    scores_h, V_h.data(), context_h.data(),
                    total_tokens, head_dim, total_tokens,
                    false, 1.0f, 0.0f))
            {
                LOG_ERROR("[PipelineBase] scores·V GEMM failed for local head " << local_h);
                return false;
            }

            LOG_DEBUG("[MPI TP] Rank " << rank << " Head " << global_h << ": context[0]=" << context_h[0]
                                       << " V[0]=" << V_h[0]);

            // Write to local_output buffer (contiguous for this rank's heads)
            for (int t = 0; t < total_tokens; ++t)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    local_output[t * local_n_heads * head_dim + local_h * head_dim + d] = context_h[t * head_dim + d];
                }
            }
        }

        LOG_DEBUG("[MPI TP] Rank " << rank << ": local_output[0]=" << local_output[0]
                                   << " local_output[" << (total_tokens * local_n_heads * head_dim - 1) << "]="
                                   << local_output[total_tokens * local_n_heads * head_dim - 1]);

        // 9. Allreduce: Sum local outputs from all ranks
        // Each rank contributes its local heads at the correct global offset
        // Use total_tokens (not seq_len) to cover all batches

        // Zero-initialize global output
        std::memset(output_data, 0, total_tokens * n_heads * head_dim * sizeof(float));

        // Create temporary buffer for allreduce (each rank's contribution)
        std::vector<float> send_buffer(total_tokens * n_heads * head_dim, 0.0f);

        // Copy local heads to correct position in send buffer
        // Iterate over ALL tokens (including all batches)
        for (int t = 0; t < total_tokens; ++t)
        {
            for (size_t local_h = 0; local_h < local_n_heads; ++local_h)
            {
                size_t global_h = start_head + local_h;
                for (int d = 0; d < head_dim; ++d)
                {
                    send_buffer[t * n_heads * head_dim + global_h * head_dim + d] =
                        local_output[t * local_n_heads * head_dim + local_h * head_dim + d];
                }
            }
        }

        LOG_DEBUG("[MPI TP] Rank " << rank << ": send_buffer[0]=" << send_buffer[0]
                                   << " (global head 0 position, rank computes heads " << start_head
                                   << "-" << (start_head + local_n_heads - 1) << ")");
        LOG_DEBUG("[MPI TP] Rank " << rank << ": send_buffer[" << (start_head * head_dim) << "]="
                                   << send_buffer[start_head * head_dim]
                                   << " (first element of head " << start_head << ")");

        LOG_DEBUG("[MPI TP] Rank " << rank << " BEFORE allreduce: send_buffer[100]=" << send_buffer[100]
                                   << " send_buffer[1000]=" << send_buffer[1000]
                                   << " send_buffer[8000]=" << send_buffer[8000]);

        // Allreduce: Sum contributions from all ranks
        // Use total_tokens (not seq_len) to cover all batches
        mpi_ctx_->allreduce_sum(
            send_buffer.data(),
            output_data,
            total_tokens * n_heads * head_dim);

        LOG_DEBUG("[MPI TP] Rank " << rank << " AFTER allreduce: output[0]=" << output_data[0]
                                   << " output[100]=" << output_data[100]
                                   << " output[1000]=" << output_data[1000]
                                   << " output[8000]=" << output_data[8000]
                                   << " output[" << (total_tokens * n_heads * head_dim - 1) << "]="
                                   << output_data[total_tokens * n_heads * head_dim - 1]);

        // 10. Barrier to ensure all ranks complete
        mpi_ctx_->barrier();

        return true;
    }

    // =============================================================================
    // MPI Strategy Management
    // =============================================================================

    MPIStrategy PipelineBase::selectOptimalStrategy()
    {
        // No MPI or single rank
        if (!mpi_ctx_ || mpi_ctx_->world_size() == 1)
        {
            return MPIStrategy::None;
        }

        int world_size = mpi_ctx_->world_size();
        int rank = mpi_ctx_->rank();

        // Try tensor-parallel first (most common, best performance)
        if (validateStrategy(MPIStrategy::TensorParallel))
        {
            if (rank == 0)
            {
                LOG_INFO("[MPI Strategy] Selected TensorParallel (n_heads=" << n_heads_
                                                                            << " divisible by world_size=" << world_size << ")");
            }
            return MPIStrategy::TensorParallel;
        }

        // Fallback to pipeline-parallel
        if (validateStrategy(MPIStrategy::PipelineParallel))
        {
            if (rank == 0)
            {
                LOG_INFO("[MPI Strategy] Selected PipelineParallel (n_layers=" << n_layers_
                                                                               << " divisible by world_size=" << world_size << ")");
            }
            return MPIStrategy::PipelineParallel;
        }

        // No valid strategy found
        if (rank == 0)
        {
            LOG_WARN("[MPI Strategy] No valid strategy found (n_heads=" << n_heads_
                                                                        << ", n_layers=" << n_layers_
                                                                        << ", world_size=" << world_size
                                                                        << "). Using single-rank execution.");
        }

        return MPIStrategy::None;
    }

    bool PipelineBase::validateStrategy(MPIStrategy strategy)
    {
        if (!mpi_ctx_)
            return false;

        int world_size = mpi_ctx_->world_size();

        switch (strategy)
        {
        case MPIStrategy::None:
            return true; // Always valid

        case MPIStrategy::TensorParallel:
            // Requires n_heads divisible by world_size
            if (n_heads_ == 0)
            {
                LOG_WARN("[MPI Validation] n_heads not yet set (called too early?)");
                return false;
            }
            if (n_heads_ % world_size != 0)
            {
                LOG_WARN("[MPI Validation] TensorParallel requires n_heads (" << n_heads_
                                                                              << ") divisible by world_size (" << world_size << ")");
                return false;
            }
            return true;

        case MPIStrategy::PipelineParallel:
            // Requires n_layers divisible by world_size
            if (n_layers_ == 0)
            {
                LOG_WARN("[MPI Validation] n_layers not yet set (called too early?)");
                return false;
            }
            if (n_layers_ % world_size != 0)
            {
                LOG_WARN("[MPI Validation] PipelineParallel requires n_layers (" << n_layers_
                                                                                 << ") divisible by world_size (" << world_size << ")");
                return false;
            }
            return true;

        case MPIStrategy::SequenceParallel:
            // Always valid (can split any sequence length)
            return true;

        case MPIStrategy::Hybrid:
            // TODO: Implement hybrid validation (Phase 6)
            LOG_WARN("[MPI Validation] Hybrid strategy not yet implemented");
            return false;

        default:
            LOG_ERROR("[MPI Validation] Unknown strategy: " << static_cast<int>(strategy));
            return false;
        }
    }

    // =============================================================================
    // MPI Distribution Helpers
    // =============================================================================

    std::pair<size_t, size_t> PipelineBase::getHeadDistribution(int n_heads)
    {
        if (!mpi_ctx_)
        {
            return {0, static_cast<size_t>(n_heads)};
        }

        return mpi_ctx_->get_local_slice(static_cast<size_t>(n_heads));
    }

    std::pair<size_t, size_t> PipelineBase::getLayerDistribution(int n_layers)
    {
        if (!mpi_ctx_)
        {
            return {0, static_cast<size_t>(n_layers)};
        }

        return mpi_ctx_->get_local_slice(static_cast<size_t>(n_layers));
    }

    std::pair<size_t, size_t> PipelineBase::getTokenDistribution(int seq_len)
    {
        if (!mpi_ctx_)
        {
            return {0, static_cast<size_t>(seq_len)};
        }

        return mpi_ctx_->get_local_slice(static_cast<size_t>(seq_len));
    }

    // =============================================================================
    // Generic Initialization (extracted from Qwen2Pipeline)
    // =============================================================================

    void PipelineBase::initializeDeviceInfrastructure(int max_seq_len)
    {
        // Phase 4.1: Discover which devices are used by this rank
        active_devices_ = discoverActiveDevices();
        std::stringstream devices_str;
        devices_str << "Active devices for this rank: [";
        for (size_t i = 0; i < active_devices_.size(); ++i)
        {
            if (i > 0)
                devices_str << ", ";
            devices_str << active_devices_[i];
        }
        devices_str << "]";
        LOG_INFO(devices_str.str());

        if (active_devices_.size() == 1 && active_devices_[0] == device_idx_)
        {
            // Single-device mode: use legacy path for backward compat
            LOG_INFO("Single-device mode (device " << device_idx_ << ")");
            // Derived class must allocate buffers via createBuffersForDevice
            activation_buffers_ = createBuffersForDevice(device_idx_, max_seq_len);
        }
        else
        {
            // Multi-device mode: allocate buffer pool per device
            LOG_INFO("Multi-device mode: allocating buffers for "
                     << active_devices_.size() << " devices");
            for (int dev_idx : active_devices_)
            {
                // Lazy allocation happens in getBuffersForDevice(), just ensure they exist
                ActivationBuffers &buffers = getBuffersForDevice(dev_idx);
                // Log memory usage (estimated, architecture-specific)
                // Derived class can override for accurate calculation
                (void)buffers; // Suppress unused warning
            }
            // For backward compat, point activation_buffers_ to primary device
            activation_buffers_ = buffers_per_device_[device_idx_];
        }
    }

    void PipelineBase::configureMPIStrategy()
    {
        // Configure MPI strategy if using multi-rank execution
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            // Use default MPIConfig (TensorParallel with auto_select=true)
            mpi_config_ = defaultMPIConfig();

            if (mpi_config_.auto_select)
            {
                mpi_strategy_ = selectOptimalStrategy();
            }
            else
            {
                // User specified a strategy - validate it
                mpi_strategy_ = mpi_config_.strategy;
                if (!validateStrategy(mpi_strategy_))
                {
                    LOG_WARN("[PipelineBase] User-specified strategy '"
                             << strategyName(mpi_strategy_) << "' invalid, using fallback");
                    mpi_strategy_ = mpi_config_.fallback_strategy;
                }
            }

            if (mpi_ctx_->rank() == 0)
            {
                LOG_INFO("[PipelineBase] MPI Strategy: " << strategyName(mpi_strategy_)
                                                         << " (rank " << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size() << ")");

                // Log strategy-specific info (virtual, can be overridden)
                logMPIStrategyInfo();
            }
        }
        else
        {
            mpi_strategy_ = MPIStrategy::None;
            if (mpi_ctx_)
            {
                LOG_INFO("[PipelineBase] Single-rank MPI execution (world_size=1)");
            }
        }
    }

    void PipelineBase::logMPIStrategyInfo()
    {
        // Default implementation: log tensor-parallel head distribution
        if (mpi_strategy_ == MPIStrategy::TensorParallel && n_heads_ > 0)
        {
            auto [start_head, local_n_heads] = getHeadDistribution(n_heads_);
            LOG_INFO("[PipelineBase] Tensor-parallel: " << local_n_heads
                                                        << " heads per rank (total: " << n_heads_ << ")");
        }
    }

    void PipelineBase::initializeKVCache(int max_seq_len)
    {
        DEBUG_ASSERT(n_layers_ > 0, "n_layers_ must be set before calling initializeKVCache");
        DEBUG_ASSERT(n_kv_heads_ > 0, "n_kv_heads_ must be set before calling initializeKVCache");
        DEBUG_ASSERT(head_dim_ > 0, "head_dim_ must be set before calling initializeKVCache");

        // Phase 3: Use placement map to detect attention devices per layer
        std::vector<int> attention_devices = detectAttentionDevices(n_layers_);
        kv_cache_ = std::make_shared<KVCache>(n_layers_, max_seq_len, n_kv_heads_, head_dim_, attention_devices);
        current_positions_.clear(); // Will be resized to batch_size in forward_batch()

        LOG_INFO("Initialized KV cache: " << n_layers_ << " layers, "
                                          << max_seq_len << " max_seq_len, "
                                          << n_kv_heads_ << " KV heads, " << head_dim_ << " head_dim");
    }

} // namespace llaminar2
