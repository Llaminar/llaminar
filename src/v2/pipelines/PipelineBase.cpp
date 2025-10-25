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
                               std::shared_ptr<WeightPlacementMap> placement_map)
        : model_ctx_(model_ctx), mpi_ctx_(mpi_ctx), device_idx_(device_idx), placement_map_(placement_map)
    {
        if (!model_ctx_)
        {
            throw std::runtime_error("PipelineBase: model_ctx cannot be null");
        }

        model_path_ = model_ctx_->path();

        LOG_INFO("[PipelineBase] Initializing with model: " << model_path_);

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
    }

    const float *PipelineBase::logits() const
    {
        DEBUG_ASSERT_NOT_NULL(logits_.get(), "logits() called before forward()");
        return logits_->data();
    }

    bool PipelineBase::attention_gqa(
        TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size)
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

        // Get GEMM kernel from Q tensor (any FP32 tensor can provide kernels)
        auto gemm_kernel = Q->createGemm();
        if (!gemm_kernel)
        {
            LOG_ERROR("[PipelineBase] attention_gqa: failed to create GEMM kernel\n");
            return false;
        }

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

            if (!gemm_kernel->multiply(
                    Q_h.data(), scores_h,
                    seq_len, seq_len, head_dim,
                    true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
            {
                LOG_ERROR("[PipelineBase] attention_gqa: Q·K^T GEMM failed for head " << h);
                return false;
            }
        }

        // Scale scores by 1/sqrt(head_dim)
        attention_utils::scale_attention_scores(
            scores, n_heads * seq_len * seq_len, head_dim);

        // Apply causal mask (if enabled)
        if (causal)
        {
            std::vector<float> mask(seq_len * seq_len);
            attention_utils::create_causal_mask(mask.data(), seq_len, window_size);

            // Apply mask to each head
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len;
                attention_utils::apply_attention_mask(scores_h, mask.data(), seq_len, seq_len);
            }
        }

        // Softmax over scores using kernel from scores_tensor
        auto softmax_kernel = scores_tensor->createSoftmax();
        if (!softmax_kernel)
        {
            LOG_ERROR("[PipelineBase] attention_gqa: failed to create Softmax kernel\n");
            return false;
        }

        // Apply softmax per head
        for (int h = 0; h < n_heads; ++h)
        {
            float *scores_h = scores + h * seq_len * seq_len;

            if (!softmax_kernel->apply(
                    scores_h, scores_h,
                    seq_len, seq_len,
                    false, mpi_ctx_.get(), device_idx_))
            {
                LOG_ERROR("[PipelineBase] attention_gqa: softmax failed for head " << h);
                return false;
            }
        }

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

            if (!gemm_kernel->multiply(
                    scores_h, context_h.data(),
                    seq_len, head_dim, seq_len,
                    false, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
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

} // namespace llaminar2
