/**
 * @file MoEFFNStage.cpp
 * @brief Implementation of unified MoE FFN, shared expert, and shared expert gate stages
 */

#include "MoEFFNStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace llaminar2
{
    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    // =========================================================================
    // Helper: Dequantize a single row from a 3D quantized expert tensor
    // =========================================================================

    namespace
    {
        /// Get the byte stride per expert for a 3D quantized expert tensor.
        /// GGUF 3D layout (no swap during loading):
        ///   shape[0] = ne[0] = fastest-varying = cols_per_expert
        ///   shape[1] = ne[1] = middle          = rows_per_expert
        ///   shape[2] = ne[2] = slowest-varying  = num_experts
        struct ExpertTensorInfo
        {
            const void *raw_blocks = nullptr;
            size_t block_byte_size = 0;      ///< sizeof(BlockType)
            size_t block_element_size = 0;    ///< elements per block (256 for Q4_K, Q6_K)
            size_t blocks_per_row = 0;        ///< cols / block_element_size
            size_t rows_per_expert = 0;       ///< shape[1] = ne[1]
            size_t cols_per_expert = 0;       ///< shape[0] = ne[0]
            TensorType tensor_type = TensorType::FP32;
        };

        bool getExpertTensorInfo(TensorBase *tensor, ExpertTensorInfo &info)
        {
            if (!tensor)
                return false;

            const auto &shape = tensor->shape();
            if (shape.size() != 3)
            {
                LOG_ERROR("[MoE] Expert tensor must be 3D, got " << shape.size() << "D");
                return false;
            }

            info.raw_blocks = tensor->raw_data();
            // GGUF 3D: shape = [ne[0], ne[1], ne[2]] = [cols, rows, num_experts]
            // ne[0] is fastest-varying (contiguous), ne[2] is slowest (outermost)
            info.cols_per_expert = shape[0];  // ne[0] = K dimension (d_model or intermediate)
            info.rows_per_expert = shape[1];  // ne[1] = rows per expert slice
            info.tensor_type = tensor->native_type();

            switch (info.tensor_type)
            {
            case TensorType::Q4_K:
                info.block_byte_size = sizeof(Q4_KBlock);
                info.block_element_size = Q4_KBlock::BLOCK_SIZE;
                break;
            case TensorType::Q5_K:
                info.block_byte_size = sizeof(Q5_KBlock);
                info.block_element_size = Q5_KBlock::BLOCK_SIZE;
                break;
            case TensorType::Q6_K:
                info.block_byte_size = sizeof(Q6_KBlock);
                info.block_element_size = Q6_KBlock::BLOCK_SIZE;
                break;
            default:
                LOG_ERROR("[MoE] Unsupported expert tensor type: " << static_cast<int>(info.tensor_type));
                return false;
            }

            info.blocks_per_row = info.cols_per_expert / info.block_element_size;
            return true;
        }

        /// Dequantize a full row from a 3D expert tensor
        /// @param info Expert tensor metadata
        /// @param expert_id Which expert (shape[2] / ne[2] index)
        /// @param row Row within the expert slice (shape[1] / ne[1] index)
        /// @param output Output FP32 buffer, must be >= cols_per_expert elements
        void dequantizeExpertRow(const ExpertTensorInfo &info, int expert_id, int row, float *output)
        {
            const size_t blocks_per_expert = info.rows_per_expert * info.blocks_per_row;
            const size_t block_offset = static_cast<size_t>(expert_id) * blocks_per_expert
                                        + static_cast<size_t>(row) * info.blocks_per_row;

            switch (info.tensor_type)
            {
            case TensorType::Q4_K:
            {
                const auto *blocks = static_cast<const Q4_KBlock *>(info.raw_blocks);
                for (size_t b = 0; b < info.blocks_per_row; ++b)
                {
                    Q4_KTensor::decodeBlock(blocks[block_offset + b],
                                            output + b * info.block_element_size);
                }
                break;
            }
            case TensorType::Q5_K:
            {
                const auto *blocks = static_cast<const Q5_KBlock *>(info.raw_blocks);
                for (size_t b = 0; b < info.blocks_per_row; ++b)
                {
                    Q5_KTensor::decodeBlock(blocks[block_offset + b],
                                            output + b * info.block_element_size);
                }
                break;
            }
            case TensorType::Q6_K:
            {
                const auto *blocks = static_cast<const Q6_KBlock *>(info.raw_blocks);
                for (size_t b = 0; b < info.blocks_per_row; ++b)
                {
                    Q6_KTensor::decodeBlock(blocks[block_offset + b],
                                            output + b * info.block_element_size);
                }
                break;
            }
            default:
                break;
            }
        }

    } // anonymous namespace

    // =========================================================================
    // MoEFFNStage — Unified Router + Expert FFN + Combine
    // =========================================================================

    MoEFFNStage::MoEFFNStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool MoEFFNStage::executeRouting(
        const float *hidden, int seq_len, int d_model,
        const float *gate_w, int num_experts, int top_k,
        std::vector<int> &expert_indices,
        std::vector<float> &expert_weights) const
    {
        // Output: expert_indices[seq_len * top_k], expert_weights[seq_len * top_k]
        expert_indices.resize(seq_len * top_k);
        expert_weights.resize(seq_len * top_k);

        // Stash raw router logits for snapshot capture [seq_len, num_experts]
        router_logits_.resize(static_cast<size_t>(seq_len) * num_experts);

        // For each token: compute router logits, softmax, top-k selection
        for (int t = 0; t < seq_len; ++t)
        {
            const float *h = hidden + t * d_model;

            // Compute router logits: gate_w[e] · h for each expert
            std::vector<float> logits(num_experts);
            for (int e = 0; e < num_experts; ++e)
            {
                float dot = 0.0f;
                const float *w = gate_w + e * d_model;
                for (int d = 0; d < d_model; ++d)
                {
                    dot += w[d] * h[d];
                }
                logits[e] = dot;
            }

            // Softmax
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (int e = 0; e < num_experts; ++e)
            {
                logits[e] = std::exp(logits[e] - max_logit);
                sum_exp += logits[e];
            }
            for (int e = 0; e < num_experts; ++e)
            {
                logits[e] /= sum_exp;
            }

            // Stash post-softmax probabilities (matches PyTorch gate output[0])
            std::copy(logits.begin(), logits.end(),
                      router_logits_.begin() + static_cast<size_t>(t) * num_experts);

            // Top-k selection
            std::vector<int> indices(num_experts);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                              [&logits](int a, int b)
                              { return logits[a] > logits[b]; });

            // Normalize top-k weights
            float topk_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
            {
                topk_sum += logits[indices[k]];
            }

            for (int k = 0; k < top_k; ++k)
            {
                expert_indices[t * top_k + k] = indices[k];
                expert_weights[t * top_k + k] = params_.norm_topk_prob
                                                     ? logits[indices[k]] / topk_sum
                                                     : logits[indices[k]];
            }
        }

        return true;
    }

    void MoEFFNStage::stashRoutingResults(
        const std::vector<int> &expert_indices,
        const std::vector<float> &expert_weights,
        int seq_len, int top_k) const
    {
        const size_t n = static_cast<size_t>(seq_len) * top_k;
        routing_indices_f32_.resize(n);
        routing_weights_.resize(n);
        for (size_t i = 0; i < n; ++i)
            routing_indices_f32_[i] = static_cast<float>(expert_indices[i]);
        std::copy(expert_weights.begin(), expert_weights.end(), routing_weights_.begin());

        // Invalidate cached dump info so snapshot callback sees the routing data
        invalidateDumpInfoCache();
    }

    bool MoEFFNStage::executeExpertFFN(
        const float *input_tokens, int num_tokens, int d_model,
        const float *gate_w, const float *up_w, const float *down_w,
        int intermediate, float *output) const
    {
        // SwiGLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
        // gate_w: [intermediate, d_model]
        // up_w:   [intermediate, d_model]
        // down_w: [d_model, intermediate]

        std::vector<float> gate_buf(num_tokens * intermediate);
        std::vector<float> up_buf(num_tokens * intermediate);

        // Gate and Up projections
        for (int t = 0; t < num_tokens; ++t)
        {
            const float *x = input_tokens + t * d_model;

            // Gate projection
            for (int i = 0; i < intermediate; ++i)
            {
                float dot = 0.0f;
                const float *w = gate_w + i * d_model;
                for (int d = 0; d < d_model; ++d)
                {
                    dot += w[d] * x[d];
                }
                gate_buf[t * intermediate + i] = dot;
            }

            // Up projection
            for (int i = 0; i < intermediate; ++i)
            {
                float dot = 0.0f;
                const float *w = up_w + i * d_model;
                for (int d = 0; d < d_model; ++d)
                {
                    dot += w[d] * x[d];
                }
                up_buf[t * intermediate + i] = dot;
            }
        }

        // SwiGLU activation: silu(gate) * up
        for (int i = 0; i < num_tokens * intermediate; ++i)
        {
            float g = gate_buf[i];
            float silu_g = g / (1.0f + std::exp(-g)); // silu(x) = x * sigmoid(x)
            gate_buf[i] = silu_g * up_buf[i];
        }

        // Down projection
        for (int t = 0; t < num_tokens; ++t)
        {
            for (int d = 0; d < d_model; ++d)
            {
                float dot = 0.0f;
                const float *w = down_w + d * intermediate;
                for (int i = 0; i < intermediate; ++i)
                {
                    dot += w[i] * gate_buf[t * intermediate + i];
                }
                output[t * d_model + d] = dot;
            }
        }

        return true;
    }

    bool MoEFFNStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEFFNStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_weights || !params_.output)
        {
            LOG_ERROR("[MoEFFNStage] Null input/gate/output tensor");
            return false;
        }

        if (!params_.gate_exps || !params_.up_exps || !params_.down_exps)
        {
            LOG_ERROR("[MoEFFNStage] Null expert weight tensors");
            return false;
        }

        // Dispatch to GPU path if we have expert views and are on a GPU device
        const bool on_gpu = (ctx->backendType() == ComputeBackendType::GPU_CUDA ||
                             ctx->backendType() == ComputeBackendType::GPU_ROCM);
        if (on_gpu && !params_.expert_gate_views.empty())
        {
            return executeGPU(ctx);
        }

        return executeCPU(ctx);
    }

    bool MoEFFNStage::executeCPU(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        const float *hidden = params_.input->data();
        const float *gate_w = params_.gate_weights->data();
        float *output = params_.output->mutable_data();

        // Prepare expert tensor info for 3D quantized weight access
        ExpertTensorInfo gate_info, up_info, down_info;
        if (!getExpertTensorInfo(params_.gate_exps, gate_info) ||
            !getExpertTensorInfo(params_.up_exps, up_info) ||
            !getExpertTensorInfo(params_.down_exps, down_info))
        {
            LOG_ERROR("[MoEFFNStage] Failed to get expert tensor info");
            return false;
        }

        // Step 1: Routing — softmax top-k
        std::vector<int> expert_indices;
        std::vector<float> expert_weights_vec;
        if (!executeRouting(hidden, seq_len, d_model, gate_w, num_experts, top_k,
                            expert_indices, expert_weights_vec))
        {
            LOG_ERROR("[MoEFFNStage] Routing failed");
            return false;
        }
        stashRoutingResults(expert_indices, expert_weights_vec, seq_len, top_k);

        // Step 2: Zero output
        std::memset(output, 0, static_cast<size_t>(seq_len) * d_model * sizeof(float));

        // Step 3: For each token, execute top-k expert FFNs and combine
        // Thread-local scratch buffers for dequantized weight rows and intermediate results
        for (int t = 0; t < seq_len; ++t)
        {
            const float *token_input = hidden + t * d_model;
            float *token_output = output + t * d_model;

            // Scratch buffers (per-row dequant + intermediate activations)
            std::vector<float> row_buf(std::max(d_model, intermediate));
            std::vector<float> gate_buf(intermediate);
            std::vector<float> up_buf(intermediate);
            std::vector<float> expert_output(d_model);

            for (int k = 0; k < top_k; ++k)
            {
                int expert_id = expert_indices[t * top_k + k];
                float weight = expert_weights_vec[t * top_k + k];

                // Gate projection: gate_exps[expert_id] × input → gate_buf
                // gate_exps[expert_id] shape: [intermediate, d_model]
                for (int i = 0; i < intermediate; ++i)
                {
                    dequantizeExpertRow(gate_info, expert_id, i, row_buf.data());
                    float dot = 0.0f;
                    for (int d = 0; d < d_model; ++d)
                        dot += row_buf[d] * token_input[d];
                    gate_buf[i] = dot;
                }

                // Up projection: up_exps[expert_id] × input → up_buf
                for (int i = 0; i < intermediate; ++i)
                {
                    dequantizeExpertRow(up_info, expert_id, i, row_buf.data());
                    float dot = 0.0f;
                    for (int d = 0; d < d_model; ++d)
                        dot += row_buf[d] * token_input[d];
                    up_buf[i] = dot;
                }

                // SwiGLU activation: silu(gate) * up
                for (int i = 0; i < intermediate; ++i)
                {
                    float g = gate_buf[i];
                    float silu_g = g / (1.0f + std::exp(-g));
                    gate_buf[i] = silu_g * up_buf[i];
                }

                // Down projection: down_exps[expert_id] × activated → expert_output
                // down_exps[expert_id] shape: [d_model, intermediate]
                for (int d = 0; d < d_model; ++d)
                {
                    dequantizeExpertRow(down_info, expert_id, d, row_buf.data());
                    float dot = 0.0f;
                    for (int i = 0; i < intermediate; ++i)
                        dot += row_buf[i] * gate_buf[i];
                    expert_output[d] = dot;
                }

                // Accumulate weighted output
                for (int d = 0; d < d_model; ++d)
                {
                    token_output[d] += weight * expert_output[d];
                }
            }
        }

        LOG_DEBUG("[MoEFFNStage] Processed " << seq_len << " tokens, "
                                              << top_k << " experts per token");
        return true;
    }

    // =========================================================================
    // MoEFFNStage::executeGPU — GPU execution via KernelFactory GEMM kernels
    // =========================================================================

    bool MoEFFNStage::executeGPU(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        // Router logits are computed on device via GEMM (hidden @ gate_weights^T)
        // gate_weights is FP32 [num_experts, d_model], so we can use it directly.
        // For now, routing stays on CPU (copy hidden to host, route, copy back).
        // This is fine because routing is O(seq_len * num_experts * d_model) — small
        // compared to expert FFN work O(seq_len * top_k * d_model * intermediate * 3).
        const float *hidden = params_.input->data();
        const float *gate_w = params_.gate_weights->data();
        float *output = params_.output->mutable_data();

        // Step 1: Routing — softmax top-k (CPU)
        std::vector<int> expert_indices;
        std::vector<float> expert_weights_vec;
        if (!executeRouting(hidden, seq_len, d_model, gate_w, num_experts, top_k,
                            expert_indices, expert_weights_vec))
        {
            LOG_ERROR("[MoEFFNStage] GPU routing failed");
            return false;
        }
        stashRoutingResults(expert_indices, expert_weights_vec, seq_len, top_k);

        // Step 2: Zero output
        std::memset(output, 0, static_cast<size_t>(seq_len) * d_model * sizeof(float));

        // Step 3: Group tokens by expert for batched execution
        // expert_token_lists[expert_id] = list of (token_idx, routing_weight)
        std::vector<std::vector<std::pair<int, float>>> expert_token_lists(num_experts);
        for (int t = 0; t < seq_len; ++t)
        {
            for (int k = 0; k < top_k; ++k)
            {
                int expert_id = expert_indices[t * top_k + k];
                float weight = expert_weights_vec[t * top_k + k];
                expert_token_lists[expert_id].emplace_back(t, weight);
            }
        }

        // Step 4: Execute each active expert's FFN via GEMM kernels
        // For each expert with assigned tokens, we:
        //   a) Gather token inputs into a contiguous batch
        //   b) gate_proj: batch @ expert_gate_view^T → gate_out [num_tokens, intermediate]
        //   c) up_proj: batch @ expert_up_view^T → up_out [num_tokens, intermediate]
        //   d) SwiGLU: silu(gate_out) * up_out
        //   e) down_proj: activated @ expert_down_view^T → expert_out [num_tokens, d_model]
        //   f) Scatter weighted results back to output
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &token_list = expert_token_lists[expert_id];
            if (token_list.empty()) continue;

            const int num_tokens = static_cast<int>(token_list.size());

            // Gather tokens into contiguous batch
            std::vector<float> batch_input(num_tokens * d_model);
            for (int i = 0; i < num_tokens; ++i)
            {
                const float *src = hidden + token_list[i].first * d_model;
                std::copy(src, src + d_model, batch_input.data() + i * d_model);
            }

            // Get expert 2D tensor views
            TensorBase *gate_view = params_.expert_gate_views[expert_id].get();
            TensorBase *up_view = params_.expert_up_views[expert_id].get();
            TensorBase *down_view = params_.expert_down_views[expert_id].get();

            // Get GEMM engines via KernelFactory for this expert's weight views
            auto *gate_prepared = KernelFactory::getOrCreatePreparedGemmWeights(gate_view, params_.device_id);
            auto *up_prepared = KernelFactory::getOrCreatePreparedGemmWeights(up_view, params_.device_id);
            auto *down_prepared = KernelFactory::getOrCreatePreparedGemmWeights(down_view, params_.device_id);

            ITensorGemm *gate_gemm = KernelFactory::getOrCreateGemmEngine(gate_prepared);
            ITensorGemm *up_gemm = KernelFactory::getOrCreateGemmEngine(up_prepared);
            ITensorGemm *down_gemm = KernelFactory::getOrCreateGemmEngine(down_prepared);

            // Allocate FP32 scratch for gate/up/down projections
            // Create temporary FP32 tensors with proper shape
            auto batch_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(d_model)});
            std::copy(batch_input.data(), batch_input.data() + num_tokens * d_model,
                      batch_tensor->mutable_data());

            auto gate_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(intermediate)});
            auto up_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(intermediate)});
            auto output_tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(d_model)});

            // Gate projection: [num_tokens, d_model] × [intermediate, d_model]^T → [num_tokens, intermediate]
            gate_gemm->multiply_tensor(
                batch_tensor.get(), gate_tensor.get(),
                num_tokens, intermediate, d_model);

            // Up projection: same dimensions
            up_gemm->multiply_tensor(
                batch_tensor.get(), up_tensor.get(),
                num_tokens, intermediate, d_model);

            // SwiGLU activation: silu(gate) * up (in-place on gate_tensor)
            float *g = gate_tensor->mutable_data();
            const float *u = up_tensor->data();
            for (int i = 0; i < num_tokens * intermediate; ++i)
            {
                float gv = g[i];
                g[i] = (gv / (1.0f + std::exp(-gv))) * u[i];
            }

            // Down projection with SwiGLU'd input: [num_tokens, intermediate] → [num_tokens, d_model]
            down_gemm->multiply_tensor(
                gate_tensor.get(), output_tensor.get(),
                num_tokens, d_model, intermediate);

            // Scatter weighted results back to output
            const float *exp_out = output_tensor->data();
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_idx = token_list[i].first;
                float weight = token_list[i].second;
                float *dst = output + token_idx * d_model;
                const float *src = exp_out + i * d_model;
                for (int d = 0; d < d_model; ++d)
                {
                    dst[d] += weight * src[d];
                }
            }
        }

        LOG_DEBUG("[MoEFFNStage] GPU processed " << seq_len << " tokens, "
                                                  << top_k << " experts per token");
        return true;
    }

    // =========================================================================
    // MoEFFNStage::extractExpertViews — Create 2D views from 3D packed tensors
    // =========================================================================

    bool MoEFFNStage::extractExpertViews(Params &params)
    {
        if (!params.gate_exps || !params.up_exps || !params.down_exps)
        {
            LOG_ERROR("[MoEFFNStage] Cannot extract views: null expert tensors");
            return false;
        }

        const int num_experts = params.num_experts;
        if (num_experts <= 0)
        {
            LOG_ERROR("[MoEFFNStage] Invalid num_experts: " << num_experts);
            return false;
        }

        params.expert_gate_views.resize(num_experts);
        params.expert_up_views.resize(num_experts);
        params.expert_down_views.resize(num_experts);

        // Extract 2D views for each expert.
        // GGUF 3D: shape = [ne[0], ne[1], ne[2]] = [cols, rows, num_experts]
        // Each expert's 2D slice is [rows, cols] at element offset = expert_id * rows * cols.
        // create_view() handles 3D→2D slicing internally.
        auto extract_views = [](TensorBase *tensor_3d, int n_experts,
                                std::vector<std::shared_ptr<TensorBase>> &views) -> bool
        {
            const auto &shape = tensor_3d->shape();
            if (shape.size() != 3)
            {
                LOG_ERROR("[MoE] Expert tensor must be 3D, got " << shape.size() << "D");
                return false;
            }

            // GGUF 3D: shape[0]=ne[0]=cols (fastest), shape[1]=ne[1]=rows, shape[2]=ne[2]=experts (slowest)
            size_t cols = shape[0];
            size_t rows = shape[1];
            size_t elements_per_expert = rows * cols;

            for (int e = 0; e < n_experts; ++e)
            {
                size_t element_offset = static_cast<size_t>(e) * elements_per_expert;
                std::vector<size_t> view_shape = {rows, cols};
                auto view = tensor_3d->create_view(view_shape, element_offset);
                if (!view)
                {
                    LOG_ERROR("[MoE] Failed to create view for expert " << e);
                    return false;
                }
                views[e] = std::move(view);
            }
            return true;
        };

        if (!extract_views(params.gate_exps, num_experts, params.expert_gate_views))
            return false;
        if (!extract_views(params.up_exps, num_experts, params.expert_up_views))
            return false;
        if (!extract_views(params.down_exps, num_experts, params.expert_down_views))
            return false;

        LOG_INFO("[MoEFFNStage] Extracted " << num_experts
                 << " expert 2D views per weight tensor");
        return true;
    }

    size_t MoEFFNStage::estimatedFlops() const
    {
        // Per token: top_k experts × (gate + up + down projections)
        // gate/up: d_model × intermediate
        // down: intermediate × d_model
        size_t per_expert = static_cast<size_t>(6) * params_.d_model * params_.expert_intermediate;
        return static_cast<size_t>(params_.seq_len) * params_.top_k * per_expert;
    }

    bool MoEFFNStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return !params_.expert_gate_views.empty();
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return !params_.expert_gate_views.empty();
#endif
        default:
            return false;
        }
    }

    StageBufferRequirements MoEFFNStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        return reqs;
    }

    StageDumpInfo MoEFFNStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_weights)
            info.addWeight("gate_weights", params_.gate_weights);
        if (params_.gate_exps)
            info.addWeight("gate_exps", params_.gate_exps);
        if (params_.up_exps)
            info.addWeight("up_exps", params_.up_exps);
        if (params_.down_exps)
            info.addWeight("down_exps", params_.down_exps);
        if (params_.output)
            info.addOutput("output", params_.output, params_.seq_len, params_.d_model);

        // Routing data (stashed during execute) — outputs[1..3]
        if (!router_logits_.empty())
            info.addOutput("router_logits", router_logits_.data(),
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.num_experts));
        if (!routing_indices_f32_.empty())
            info.addOutput("routing_indices", routing_indices_f32_.data(),
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.top_k));
        if (!routing_weights_.empty())
            info.addOutput("routing_weights", routing_weights_.data(),
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.top_k));

        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("expert_intermediate", params_.expert_intermediate);
        return info;
    }

    // =========================================================================
    // SharedExpertFFNStage — Dense SwiGLU on shared expert
    // =========================================================================

    SharedExpertFFNStage::SharedExpertFFNStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool SharedExpertFFNStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SharedExpertFFNStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_w || !params_.up_w || !params_.down_w || !params_.output)
        {
            LOG_ERROR("[SharedExpertFFNStage] Null tensor parameter");
            return false;
        }

        const bool on_gpu = (ctx->backendType() == ComputeBackendType::GPU_CUDA ||
                             ctx->backendType() == ComputeBackendType::GPU_ROCM);

        if (on_gpu)
        {
            return executeGPU_SharedExpert(ctx);
        }

        return executeCPU_SharedExpert(ctx);
    }

    bool SharedExpertFFNStage::executeCPU_SharedExpert(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int intermediate = params_.intermediate;

        const float *input = params_.input->data();
        const float *gate_w = params_.gate_w->data();
        const float *up_w = params_.up_w->data();
        const float *down_w = params_.down_w->data();
        float *output = params_.output->mutable_data();

        // SwiGLU FFN: output = down(silu(gate(x)) * up(x))
        auto do_work = [=]()
        {
            // Allocate thread-local buffers
            std::vector<float> gate_buf(intermediate);
            std::vector<float> up_buf(intermediate);

            #pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const float *x = input + t * d_model;

                // Gate projection: gate_w [intermediate, d_model] × x [d_model]
                for (int i = 0; i < intermediate; ++i)
                {
                    float dot = 0.0f;
                    const float *w = gate_w + i * d_model;
                    for (int d = 0; d < d_model; ++d)
                    {
                        dot += w[d] * x[d];
                    }
                    gate_buf[i] = dot;
                }

                // Up projection
                for (int i = 0; i < intermediate; ++i)
                {
                    float dot = 0.0f;
                    const float *w = up_w + i * d_model;
                    for (int d = 0; d < d_model; ++d)
                    {
                        dot += w[d] * x[d];
                    }
                    up_buf[i] = dot;
                }

                // SwiGLU: silu(gate) * up
                for (int i = 0; i < intermediate; ++i)
                {
                    float g = gate_buf[i];
                    float silu_g = g / (1.0f + std::exp(-g));
                    gate_buf[i] = silu_g * up_buf[i];
                }

                // Down projection: down_w [d_model, intermediate] × activated [intermediate]
                float *out = output + t * d_model;
                for (int d = 0; d < d_model; ++d)
                {
                    float dot = 0.0f;
                    const float *w = down_w + d * intermediate;
                    for (int i = 0; i < intermediate; ++i)
                    {
                        dot += w[i] * gate_buf[i];
                    }
                    out[d] = dot;
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

    bool SharedExpertFFNStage::executeGPU_SharedExpert(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int intermediate = params_.intermediate;

        // Use KernelFactory GEMM for gate/up/down projections
        // Shared expert weights are regular 2D tensors (not 3D packed)
        auto *gate_prepared = KernelFactory::getOrCreatePreparedGemmWeights(
            params_.gate_w, params_.device_id);
        auto *up_prepared = KernelFactory::getOrCreatePreparedGemmWeights(
            params_.up_w, params_.device_id);
        auto *down_prepared = KernelFactory::getOrCreatePreparedGemmWeights(
            params_.down_w, params_.device_id);

        ITensorGemm *gate_gemm = KernelFactory::getOrCreateGemmEngine(gate_prepared);
        ITensorGemm *up_gemm = KernelFactory::getOrCreateGemmEngine(up_prepared);
        ITensorGemm *down_gemm = KernelFactory::getOrCreateGemmEngine(down_prepared);

        // Allocate scratch tensors for intermediate activations
        auto gate_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
        auto up_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});

        // Gate projection: [seq_len, d_model] × [intermediate, d_model]^T → [seq_len, intermediate]
        gate_gemm->multiply_tensor(
            params_.input, gate_tensor.get(),
            seq_len, intermediate, d_model);

        // Up projection
        up_gemm->multiply_tensor(
            params_.input, up_tensor.get(),
            seq_len, intermediate, d_model);

        // SwiGLU activation: silu(gate) * up
        float *g = gate_tensor->mutable_data();
        const float *u = up_tensor->data();
        for (int i = 0; i < seq_len * intermediate; ++i)
        {
            float gv = g[i];
            g[i] = (gv / (1.0f + std::exp(-gv))) * u[i];
        }

        // Down projection: [seq_len, intermediate] × [d_model, intermediate]^T → [seq_len, d_model]
        down_gemm->multiply_tensor(
            gate_tensor.get(), params_.output,
            seq_len, d_model, intermediate);

        return true;
    }

    size_t SharedExpertFFNStage::estimatedFlops() const
    {
        return static_cast<size_t>(6) * params_.seq_len * params_.d_model * params_.intermediate;
    }

    bool SharedExpertFFNStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageBufferRequirements SharedExpertFFNStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        return reqs;
    }

    StageDumpInfo SharedExpertFFNStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_w) info.addWeight("gate_w", params_.gate_w);
        if (params_.up_w) info.addWeight("up_w", params_.up_w);
        if (params_.down_w) info.addWeight("down_w", params_.down_w);
        if (params_.output)
            info.addOutput("output", params_.output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("intermediate", params_.intermediate);
        return info;
    }

    // =========================================================================
    // SharedExpertGateStage — Sigmoid gating on shared expert output
    // =========================================================================

    SharedExpertGateStage::SharedExpertGateStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool SharedExpertGateStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SharedExpertGateStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_inp || !params_.shared_output)
        {
            LOG_ERROR("[SharedExpertGateStage] Null tensor parameter");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;

        const float *input = params_.input->data();
        const float *gate_inp = params_.gate_inp->data();
        float *shared = params_.shared_output->mutable_data();

        // For each token:
        // g = sigmoid(dot(gate_inp, input[t]))
        // shared_output[t] *= g
        auto do_work = [=]()
        {
            #pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const float *x = input + t * d_model;

                // Compute gate value: dot product + sigmoid
                float dot = 0.0f;
                for (int d = 0; d < d_model; ++d)
                {
                    dot += gate_inp[d] * x[d];
                }
                float gate = 1.0f / (1.0f + std::exp(-dot)); // sigmoid

                // Apply gate: shared_output[t] *= gate
                float *out = shared + t * d_model;
                for (int d = 0; d < d_model; ++d)
                {
                    out[d] *= gate;
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

    size_t SharedExpertGateStage::estimatedFlops() const
    {
        // dot product + sigmoid + elementwise multiply
        return static_cast<size_t>(params_.seq_len) * (2 * params_.d_model + params_.d_model);
    }

    bool SharedExpertGateStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageBufferRequirements SharedExpertGateStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.shared_output)
            reqs.addOutput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
        return reqs;
    }

    StageDumpInfo SharedExpertGateStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_inp) info.addWeight("gate_inp", params_.gate_inp);
        if (params_.shared_output)
            info.addOutput("shared_output", params_.shared_output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        return info;
    }

} // namespace llaminar2
