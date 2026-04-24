/**
 * @file MoEFFNStage.cpp
 * @brief Implementation of unified MoE FFN, shared expert, and shared expert gate stages
 */

#include "MoEFFNStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../loaders/MmapRegion.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#ifdef HAVE_CUDA
#include "../../../kernels/cuda/gemm/CUDAWeightPacker.h"
#include "../../../kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif

#ifdef HAVE_ROCM
#include "../../../kernels/rocm/ROCmWeightPacker.h"
#include "../../../kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#endif

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace llaminar2
{
    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    namespace
    {
        /// Execute SwiGLU activation + Down projection via fused kernel when available,
        /// falling back to IMoEKernel::swiGLU + separate GEMM when not (e.g., FP32 weights).
        /// @param gate_tensor [m, intermediate] — gate projection output (modified in-place on fallback)
        /// @param up_tensor [m, intermediate] — up projection output
        /// @param output [m, n] — final output
        /// @param down_gemm GEMM engine for down projection
        /// @param moe_kernel MoE kernel for SwiGLU fallback
        /// @param m sequence length / batch size
        /// @param n output dimension (d_model)
        /// @param intermediate intermediate dimension
        void fusedSwigluDown(
            FP32Tensor *gate_tensor, FP32Tensor *up_tensor, TensorBase *output,
            ITensorGemm *down_gemm, IMoEKernel *moe_kernel,
            int m, int n, int intermediate)
        {
            // Try fused path first (quantized GEMM engines support this)
            if (down_gemm->multiply_tensor_with_fused_swiglu(
                    gate_tensor, up_tensor, output,
                    m, n, intermediate))
            {
                return;
            }

            // Fallback: SwiGLU via device-agnostic kernel, then separate down GEMM
            float *g = gate_tensor->mutable_data();
            const float *u = up_tensor->data();
            const int count = m * intermediate;
            moe_kernel->swiGLU(g, u, count);
            down_gemm->multiply_tensor(
                gate_tensor, output,
                m, n, intermediate);
        }
    } // anonymous namespace

    // =========================================================================
    // MoEFFNStage — Unified Router + Expert FFN + Combine
    // =========================================================================

    MoEFFNStage::MoEFFNStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    IMoEKernel *MoEFFNStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return moe_kernel_;
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

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        if (params_.expert_gate_views.empty())
        {
            LOG_ERROR("[MoEFFNStage] Requires pre-extracted expert views. "
                      "Call extractExpertViews() at graph build time.");
            return false;
        }

        const float *hidden = params_.input->data();
        const float *gate_w = params_.gate_weights->data();
        float *output = params_.output->mutable_data();

        // Get device-appropriate MoE kernel for routing/gather/scatter
        IMoEKernel *kernel = ensureMoEKernel();

        // Step 1: Routing — softmax top-k via device kernel
        MoERoutingResult routing;
        if (!kernel->route(hidden, gate_w, seq_len, d_model,
                           num_experts, top_k, params_.norm_topk_prob, routing))
        {
            LOG_ERROR("[MoEFFNStage] Routing failed");
            return false;
        }

        // Stash routing data for snapshot capture
        router_logits_ = std::move(routing.router_logits);
        stashRoutingResults(routing.expert_indices, routing.expert_weights, seq_len, top_k);

        // Step 2: Zero output
        std::memset(output, 0, static_cast<size_t>(seq_len) * d_model * sizeof(float));

        // Step 3: Group tokens by expert for batched GEMM execution
        std::vector<std::vector<std::pair<int, float>>> expert_token_lists(num_experts);
        for (int t = 0; t < seq_len; ++t)
        {
            for (int k = 0; k < top_k; ++k)
            {
                int expert_id = routing.expert_indices[t * top_k + k];
                float weight = routing.expert_weights[t * top_k + k];
                expert_token_lists[expert_id].emplace_back(t, weight);
            }
        }

        // Ensure GEMM engines are cached (lazy init on first call)
        ensureGemmEnginesCached();

        // Ensure scratch buffers have enough capacity for largest expert batch
        int max_batch = 0;
        for (const auto &tl : expert_token_lists)
            max_batch = std::max(max_batch, static_cast<int>(tl.size()));

        if (max_batch > 0 && max_batch > scratch_capacity_)
        {
            scratch_batch_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(d_model)});
            scratch_gate_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(intermediate)});
            scratch_up_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(intermediate)});
            scratch_out_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(max_batch), static_cast<size_t>(d_model)});
            scratch_capacity_ = max_batch;
        }

        // Step 4: Execute each active expert (reusing cached engines + scratch)
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &token_list = expert_token_lists[expert_id];
            if (token_list.empty())
                continue;

            const int num_tokens = static_cast<int>(token_list.size());

            // Build token indices and weights arrays for kernel calls
            std::vector<int> token_indices(num_tokens);
            std::vector<float> token_weights(num_tokens);
            for (int i = 0; i < num_tokens; ++i)
            {
                token_indices[i] = token_list[i].first;
                token_weights[i] = token_list[i].second;
            }

            // Gather tokens into reusable scratch batch via kernel
            kernel->gatherTokenBatch(
                hidden, scratch_batch_->mutable_data(),
                token_indices.data(), num_tokens, d_model);

            // Use cached GEMM engines (device-agnostic via ITensorGemm)
            ITensorGemm *gate_gemm = cached_gate_gemm_[expert_id];
            ITensorGemm *up_gemm = cached_up_gemm_[expert_id];
            ITensorGemm *down_gemm = cached_down_gemm_[expert_id];

            // Gate+Up projections via fused multi-projection (quantizes input once)
            std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                {gate_gemm, scratch_gate_.get(), intermediate, nullptr, "gate"},
                {up_gemm, scratch_up_.get(), intermediate, nullptr, "up"}};
            gate_gemm->multiply_fused_tensor(
                scratch_batch_.get(), projections,
                num_tokens, d_model);

            // SwiGLU+Down via fused kernel with fallback through MoE kernel
            fusedSwigluDown(
                scratch_gate_.get(), scratch_up_.get(), scratch_out_.get(),
                down_gemm, kernel, num_tokens, d_model, intermediate);

            // Scatter weighted results back via kernel
            kernel->scatterAddWeighted(
                output, scratch_out_->data(),
                token_indices.data(), token_weights.data(),
                num_tokens, d_model);
        }

        LOG_DEBUG("[MoEFFNStage] Processed " << seq_len << " tokens via GEMM kernels, "
                                             << top_k << " experts per token");
        return true;
    }

    void MoEFFNStage::ensureGemmEnginesCached() const
    {
        if (!cached_gate_gemm_.empty())
            return;

        const int num_experts = params_.num_experts;

        // Use pre-resolved engines from graph build time if available
        if (!params_.prepared_gate_gemm.empty())
        {
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            return;
        }

        // Fallback: resolve on first call (triggers VNNI repacking — slow)
        LOG_WARN("[MoEFFNStage] GEMM engines not pre-resolved; "
                 "call prepareExpertGemmEngines() at graph build time for better perf");

        cached_gate_gemm_.resize(num_experts);
        cached_up_gemm_.resize(num_experts);
        cached_down_gemm_.resize(num_experts);

        for (int e = 0; e < num_experts; ++e)
        {
            auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_gate_views[e].get(), params_.device_id);
            auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_up_views[e].get(), params_.device_id);
            auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
                params_.expert_down_views[e].get(), params_.device_id);
            cached_gate_gemm_[e] = KernelFactory::getOrCreateGemmEngine(gp);
            cached_up_gemm_[e] = KernelFactory::getOrCreateGemmEngine(up);
            cached_down_gemm_[e] = KernelFactory::getOrCreateGemmEngine(dp);
        }
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

    bool MoEFFNStage::prepareExpertGemmEngines(Params &params)
    {
        const int num_experts = params.num_experts;
        if (params.expert_gate_views.empty() ||
            static_cast<int>(params.expert_gate_views.size()) != num_experts)
        {
            LOG_ERROR("[MoEFFNStage] prepareExpertGemmEngines: call extractExpertViews() first");
            return false;
        }

        params.prepared_gate_gemm.resize(num_experts);
        params.prepared_up_gemm.resize(num_experts);
        params.prepared_down_gemm.resize(num_experts);

        LOG_INFO("[MoEFFNStage] Preparing GEMM engines for " << num_experts
                 << " experts (3 weights each = " << (num_experts * 3) << " total)...");

#ifdef HAVE_CUDA
        if (params.device_id.is_cuda())
        {
            return prepareExpertGemmEnginesCUDA(params);
        }
#endif
#ifdef HAVE_ROCM
        if (params.device_id.is_rocm())
        {
            return prepareExpertGemmEnginesROCm(params);
        }
#endif

        // CPU path: use KernelFactory (existing behavior)
        for (int e = 0; e < num_experts; ++e)
        {
            auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(
                params.expert_gate_views[e].get(), params.device_id);
            auto *up = KernelFactory::getOrCreatePreparedGemmWeights(
                params.expert_up_views[e].get(), params.device_id);
            auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(
                params.expert_down_views[e].get(), params.device_id);

            if (!gp || !up || !dp)
            {
                LOG_ERROR("[MoEFFNStage] Failed to prepare GEMM weights for expert " << e);
                return false;
            }

            params.prepared_gate_gemm[e] = KernelFactory::getOrCreateGemmEngine(gp);
            params.prepared_up_gemm[e] = KernelFactory::getOrCreateGemmEngine(up);
            params.prepared_down_gemm[e] = KernelFactory::getOrCreateGemmEngine(dp);

            if (!params.prepared_gate_gemm[e] || !params.prepared_up_gemm[e] || !params.prepared_down_gemm[e])
            {
                LOG_ERROR("[MoEFFNStage] Failed to create GEMM engine for expert " << e);
                return false;
            }
        }

        LOG_INFO("[MoEFFNStage] All " << (num_experts * 3) << " expert GEMM engines prepared (CPU/KernelFactory path)");

        // Release mmap pages backing the raw expert weight data.
        // The VNNI interleaved engines now own their own copy — the original
        // mmap data is never accessed again. Releasing per-layer reduces peak RSS
        // by ~500 MB/layer instead of waiting for a bulk release at the end.
        {
            size_t released = 0;
            if (params.gate_exps)
                released += MmapRegion::adviseDontneedRange(params.gate_exps->raw_data(), params.gate_exps->size_bytes());
            if (params.up_exps)
                released += MmapRegion::adviseDontneedRange(params.up_exps->raw_data(), params.up_exps->size_bytes());
            if (params.down_exps)
                released += MmapRegion::adviseDontneedRange(params.down_exps->raw_data(), params.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEFFNStage] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after engine packing");
        }

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
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    void SharedExpertFFNStage::ensureGemmEnginesCached() const
    {
        if (cached_gate_gemm_)
            return;
        auto *gp = KernelFactory::getOrCreatePreparedGemmWeights(params_.gate_w, params_.device_id);
        auto *up = KernelFactory::getOrCreatePreparedGemmWeights(params_.up_w, params_.device_id);
        auto *dp = KernelFactory::getOrCreatePreparedGemmWeights(params_.down_w, params_.device_id);
        cached_gate_gemm_ = KernelFactory::getOrCreateGemmEngine(gp);
        cached_up_gemm_ = KernelFactory::getOrCreateGemmEngine(up);
        cached_down_gemm_ = KernelFactory::getOrCreateGemmEngine(dp);
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

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int intermediate = params_.intermediate;

        // Cache GEMM engines on first call
        ensureGemmEnginesCached();

        // Ensure scratch buffers are large enough
        if (seq_len > scratch_seq_len_)
        {
            scratch_gate_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
            scratch_up_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(intermediate)});
            scratch_seq_len_ = seq_len;
        }

        // Gate+Up projections via fused multi-projection (quantizes input once)
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_.get(), intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_.get(), intermediate, nullptr, "shared_up"}};
        cached_gate_gemm_->multiply_fused_tensor(
            params_.input, projections,
            seq_len, d_model);

        // SwiGLU+Down via fused kernel with MoE kernel fallback
        IMoEKernel *kernel = ensureMoEKernel();
        fusedSwigluDown(
            scratch_gate_.get(), scratch_up_.get(), params_.output,
            cached_down_gemm_, kernel, seq_len, d_model, intermediate);

        return true;
    }

    IMoEKernel *SharedExpertFFNStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return moe_kernel_;
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
        if (params_.gate_w)
            info.addWeight("gate_w", params_.gate_w);
        if (params_.up_w)
            info.addWeight("up_w", params_.up_w);
        if (params_.down_w)
            info.addWeight("down_w", params_.down_w);
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
        : IComputeStage(params.device_id), params_(std::move(params))
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

        // Delegate sigmoid gating to device-agnostic MoE kernel
        IMoEKernel *kernel = ensureMoEKernel();
        kernel->sharedExpertGate(input, gate_inp, shared, seq_len, d_model);

        return true;
    }

    IMoEKernel *SharedExpertGateStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return moe_kernel_;
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
        if (params_.gate_inp)
            info.addWeight("gate_inp", params_.gate_inp);
        if (params_.shared_output)
            info.addOutput("shared_output", params_.shared_output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        return info;
    }

    // =========================================================================
    // MoE batch packing helpers (CUDA / ROCm)
    // =========================================================================

#ifdef HAVE_CUDA
    bool MoEFFNStage::prepareExpertGemmEnginesCUDA(Params &params)
    {
        using namespace llaminar2::cuda;
        const int num_experts = params.num_experts;
        const int cuda_id = params.device_id.cuda_ordinal();

        // Helper: batch-pack one weight group and create per-expert kernels
        auto batchPackAndCreateKernels = [&](
            const std::vector<std::shared_ptr<TensorBase>> &views,
            std::vector<ITensorGemm *> &out_gemms,
            std::shared_ptr<void> &out_lifetime,
            const char *label) -> bool
        {
            const int rows = static_cast<int>(views[0]->rows());
            const int K = static_cast<int>(views[0]->cols());

            auto batch = packMoEExpertsCUDA(views, num_experts, rows);
            if (!batch)
            {
                LOG_ERROR("[MoEFFNStage::CUDA] Failed to batch-pack " << label);
                return false;
            }

            if (!batch->uploadToDevice(cuda_id))
            {
                LOG_ERROR("[MoEFFNStage::CUDA] Failed to upload " << label << " to device " << cuda_id);
                return false;
            }

            for (int e = 0; e < num_experts; ++e)
            {
                auto expert_ptrs = batch->getExpertDevicePointers(cuda_id, e);
                auto kernel = std::make_shared<llaminar2::cuda::CUDAQuantisedGemmKernel>(
                    rows, K, cuda_id,
                    expert_ptrs.d_vnni, expert_ptrs.d_scales,
                    expert_ptrs.d_mins, expert_ptrs.d_emins,
                    batch->codebook_id, static_cast<uint32_t>(batch->blocks_per_row),
                    batch);
                out_gemms[e] = kernel.get();
                params.moe_owned_kernels.push_back(std::move(kernel));
            }

            batch->freeHostBuffers();
            out_lifetime = std::move(batch);
            return true;
        };

        if (!batchPackAndCreateKernels(params.expert_gate_views, params.prepared_gate_gemm,
                                       params.moe_packed_gate_lifetime, "gate"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_up_views, params.prepared_up_gemm,
                                       params.moe_packed_up_lifetime, "up"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_down_views, params.prepared_down_gemm,
                                       params.moe_packed_down_lifetime, "down"))
            return false;

        LOG_INFO("[MoEFFNStage] All " << (num_experts * 3)
                 << " expert GEMM engines prepared (CUDA batch path, 3 GPU allocs)");

        // Release mmap pages for raw expert weights (now uploaded to GPU)
        {
            size_t released = 0;
            if (params.gate_exps)
                released += MmapRegion::adviseDontneedRange(params.gate_exps->raw_data(), params.gate_exps->size_bytes());
            if (params.up_exps)
                released += MmapRegion::adviseDontneedRange(params.up_exps->raw_data(), params.up_exps->size_bytes());
            if (params.down_exps)
                released += MmapRegion::adviseDontneedRange(params.down_exps->raw_data(), params.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEFFNStage] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after CUDA packing");
        }

        return true;
    }
#endif // HAVE_CUDA

#ifdef HAVE_ROCM
    bool MoEFFNStage::prepareExpertGemmEnginesROCm(Params &params)
    {
        using namespace llaminar2::rocm;
        const int num_experts = params.num_experts;
        const int rocm_id = params.device_id.rocm_ordinal();

        auto batchPackAndCreateKernels = [&](
            const std::vector<std::shared_ptr<TensorBase>> &views,
            std::vector<ITensorGemm *> &out_gemms,
            std::shared_ptr<void> &out_lifetime,
            const char *label) -> bool
        {
            const int rows = static_cast<int>(views[0]->rows());
            const int K = static_cast<int>(views[0]->cols());

            auto batch = packMoEExpertsROCm(views, num_experts, rows);
            if (!batch)
            {
                LOG_ERROR("[MoEFFNStage::ROCm] Failed to batch-pack " << label);
                return false;
            }

            if (!batch->uploadToDevice(rocm_id))
            {
                LOG_ERROR("[MoEFFNStage::ROCm] Failed to upload " << label << " to device " << rocm_id);
                return false;
            }

            for (int e = 0; e < num_experts; ++e)
            {
                auto expert_ptrs = batch->getExpertDevicePointers(rocm_id, e);
                auto kernel = std::make_shared<llaminar2::rocm::ROCmQuantisedGemmKernel>(
                    rows, K, rocm_id,
                    expert_ptrs.d_native_vnni,
                    expert_ptrs.d_native_scales,
                    expert_ptrs.d_native_mins,
                    expert_ptrs.d_native_emins,
                    batch->codebook_id, static_cast<uint32_t>(batch->blocks_per_row),
                    batch);
                out_gemms[e] = kernel.get();
                params.moe_owned_kernels.push_back(std::move(kernel));
            }

            batch->freeHostBuffers();
            out_lifetime = std::move(batch);
            return true;
        };

        if (!batchPackAndCreateKernels(params.expert_gate_views, params.prepared_gate_gemm,
                                       params.moe_packed_gate_lifetime, "gate"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_up_views, params.prepared_up_gemm,
                                       params.moe_packed_up_lifetime, "up"))
            return false;
        if (!batchPackAndCreateKernels(params.expert_down_views, params.prepared_down_gemm,
                                       params.moe_packed_down_lifetime, "down"))
            return false;

        LOG_INFO("[MoEFFNStage] All " << (num_experts * 3)
                 << " expert GEMM engines prepared (ROCm batch path, 3 GPU allocs)");

        // Release mmap pages for raw expert weights (now uploaded to GPU)
        {
            size_t released = 0;
            if (params.gate_exps)
                released += MmapRegion::adviseDontneedRange(params.gate_exps->raw_data(), params.gate_exps->size_bytes());
            if (params.up_exps)
                released += MmapRegion::adviseDontneedRange(params.up_exps->raw_data(), params.up_exps->size_bytes());
            if (params.down_exps)
                released += MmapRegion::adviseDontneedRange(params.down_exps->raw_data(), params.down_exps->size_bytes());
            if (released > 0)
                LOG_DEBUG("[MoEFFNStage] Advised " << (released >> 20) << " MB of mmap pages DONTNEED after ROCm packing");
        }

        return true;
    }
#endif // HAVE_ROCM

} // namespace llaminar2
