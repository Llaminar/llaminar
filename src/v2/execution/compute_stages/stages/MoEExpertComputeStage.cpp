/**
 * @file MoEExpertComputeStage.cpp
 * @brief Implementation of unified MoE FFN, shared expert, and shared expert gate stages
 */

#include "MoEExpertComputeStage.h"
#include "../../../execution/moe/DecodeExpertHistogram.h"
#include "../../../execution/moe/ExpertWeightTransfer.h"
#include "../../../execution/moe/MoEExpertWeightService.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/cpu/primitives/VectorPrimitives.h"
#include "../../../kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
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
    // MoEExpertComputeStage — Unified Router + Expert FFN + Combine
    // =========================================================================

    MoEExpertComputeStage::MoEExpertComputeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool MoEExpertComputeStage::updateExpertMask(const std::vector<bool>& mask) {
        if (static_cast<int>(mask.size()) != params_.num_experts) {
            LOG_ERROR("[MoEExpertComputeStage] Expert mask size " << mask.size()
                      << " != num_experts " << params_.num_experts);
            return false;
        }
        params_.expert_mask = mask;
        return true;
    }

    ExpertWeightBlobs MoEExpertComputeStage::detachAndSerializeExpert(int expert_id) {
        auto ctx = buildWeightContext();
        return MoEExpertWeightService::detachAndSerializeExpert(ctx, expert_id);
    }

    ExpertWeightBlobs MoEExpertComputeStage::serializeExpert(int expert_id) const {
        auto ctx = const_cast<MoEExpertComputeStage*>(this)->buildWeightContext();
        return MoEExpertWeightService::serializeExpert(ctx, expert_id);
    }

    size_t MoEExpertComputeStage::releaseRawExpertWeights() {
        auto ctx = buildWeightContext();
        size_t freed = MoEExpertWeightService::releaseRawWeights(ctx);
        raw_weights_released_ = true;
        return freed;
    }

    // ── Phased rebalance API ─────────────────────────────────────────────

    std::vector<const TensorBase*> MoEExpertComputeStage::releaseDepartedExperts(
        const std::vector<bool>& new_mask) {
        auto ctx = buildWeightContext();
        return MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    bool MoEExpertComputeStage::registerAndPrepareNewExperts(
        const std::vector<bool>& new_mask,
        const std::unordered_map<int, ExpertWeightBlobs>* received_weights) {
        auto ctx = buildWeightContext();
        return MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, received_weights);
    }

    void MoEExpertComputeStage::applyExpertMask(const std::vector<bool>& new_mask) {
        params_.expert_mask = new_mask;
        cached_gate_gemm_.clear();
        cached_up_gemm_.clear();
        cached_down_gemm_.clear();
    }

    MoEWeightContext MoEExpertComputeStage::buildWeightContext() {
        return MoEWeightContext{
            params_.device_id,
            params_.num_experts,
            params_.expert_intermediate,
            params_.d_model,
            params_.local_expert_start,
            params_.local_expert_count,
            params_.layer_idx,
            params_.expert_mask,
            params_.gate_exps,
            params_.up_exps,
            params_.down_exps,
            params_.expert_gate_views,
            params_.expert_up_views,
            params_.expert_down_views,
            params_.prepared_gate_gemm,
            params_.prepared_up_gemm,
            params_.prepared_down_gemm,
            params_.moe_owned_kernels,
            params_.moe_packed_gate_lifetime,
            params_.moe_packed_up_lifetime,
            params_.moe_packed_down_lifetime
        };
    }

    IMoEKernel *MoEExpertComputeStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return moe_kernel_;
    }

    bool MoEExpertComputeStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.routing_indices || !params_.routing_weights || !params_.output)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null input/routing_indices/routing_weights/output tensor");
            return false;
        }

        if (!raw_weights_released_ && (!params_.gate_exps || !params_.up_exps || !params_.down_exps))
        {
            LOG_ERROR("[MoEExpertComputeStage] Null expert weight tensors");
            return false;
        }

        // Fast path for decode (seq_len=1): eliminates gather/scatter overhead,
        // vector allocations, and expert_token_lists construction.
        if (params_.seq_len == 1)
        {
            return executeSingleToken(ctx);
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        if (params_.expert_gate_views.empty())
        {
            LOG_ERROR("[MoEExpertComputeStage] Requires pre-extracted expert views. "
                      "Call extractExpertViews() at graph build time.");
            return false;
        }

        const float *hidden = params_.input->data();
        float *output = params_.output->mutable_data();

        // Read pre-computed routing results from MoERoutingStage
        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();

        // Get device-appropriate MoE kernel for gather/scatter
        IMoEKernel *kernel = ensureMoEKernel();

        // Step 2: Zero output
        std::memset(output, 0, static_cast<size_t>(seq_len) * d_model * sizeof(float));

        // Step 3: Group tokens by expert for batched GEMM execution.
        // With EP, we only process experts in our local range, but still
        // build the full routing map so scratch sizing is correct.
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        std::vector<std::vector<std::pair<int, float>>> expert_token_lists(num_experts);

        // During prefill, replicated experts must only run on their owner
        // socket.  If both sockets process the same expert, the allreduce
        // will double-count that expert's contribution (correctness bug)
        // and waste ~12% compute (performance bug).
        //
        // Use pre-built prefill mask when available (zero per-expert overhead).
        // Falls back to multi-branch check if mask isn't built yet.
        const bool has_prefill_mask = !params_.replica_set.prefill_mask.empty();
        const std::vector<bool>& prefill_mask_ref = params_.replica_set.prefill_mask;
        const bool has_replicas = params_.replica_set.num_replicated > 0;

        for (int t = 0; t < seq_len; ++t)
        {
            for (int k = 0; k < top_k; ++k)
            {
                int expert_id = static_cast<int>(routing_idx_data[t * top_k + k]);
                float weight = routing_wt_data[t * top_k + k];
                // With EP or dynamic mask, only accumulate tokens for local experts
                bool is_local;
                if (has_prefill_mask)
                {
                    // Pre-built mask: single lookup, no branches
                    is_local = prefill_mask_ref[expert_id];
                }
                else if (!params_.expert_mask.empty())
                {
                    is_local = params_.expert_mask[expert_id];
                    // Replicated experts: only owner socket processes during prefill
                    if (is_local && has_replicas &&
                        params_.replica_set.is_replicated[expert_id] &&
                        params_.replica_set.owner_socket[expert_id] != params_.my_socket_id)
                    {
                        is_local = false;
                    }
                }
                else
                    is_local = (expert_id >= local_start && expert_id < local_end);
                if (is_local)
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

        LOG_TRACE("[MoEExpertComputeStage] Processed " << seq_len << " tokens via GEMM kernels, "
                                             << top_k << " experts per token");
        return true;
    }

    // =========================================================================
    // MoEExpertComputeStage::executeSingleToken — Optimized decode path (seq_len=1)
    //
    // Eliminates per-expert overhead:
    // - No gather (input IS the single token)
    // - No scatter (direct weighted accumulation into output)
    // - No vector allocations (stack arrays for top_k ≤ 16)
    // - No expert_token_lists grouping
    // - Reuses a single pair of scratch buffers across all experts
    // =========================================================================

    bool MoEExpertComputeStage::executeSingleToken(IDeviceContext *ctx)
    {
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        if (params_.expert_gate_views.empty())
        {
            LOG_ERROR("[MoEExpertComputeStage] Requires pre-extracted expert views.");
            return false;
        }

        const float *hidden = params_.input->data();
        float *output = params_.output->mutable_data();

        // Read pre-computed routing results from MoERoutingStage
        if (!params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null routing_indices or routing_weights (single-token)");
            return false;
        }
        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();
        IMoEKernel *kernel = ensureMoEKernel();

        // Zero output
        std::memset(output, 0, static_cast<size_t>(d_model) * sizeof(float));

        // EP range
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        // Ensure GEMM engines are cached
        ensureGemmEnginesCached();

        // Ensure batch scratch buffers for gate+up (one per top-k expert).
        // All experts' gate+up are fused into a single OMP region, so we need
        // all outputs to exist simultaneously.
        if (static_cast<int>(scratch_gate_batch_.size()) < top_k)
        {
            scratch_gate_batch_.resize(top_k);
            scratch_up_batch_.resize(top_k);
            for (int i = 0; i < top_k; ++i)
            {
                scratch_gate_batch_[i] = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{1, static_cast<size_t>(intermediate)});
                scratch_up_batch_[i] = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{1, static_cast<size_t>(intermediate)});
            }
        }
        // Scratch for down projection output (reused per expert)
        if (!scratch_out_)
        {
            scratch_out_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{1, static_cast<size_t>(d_model)});
        }

        // Use input tensor directly (no gather needed for 1 token)
        const TensorBase *input_tensor = params_.input;

        // ---------------------------------------------------------------
        // Phase 1: Batch all experts' gate+up into ONE fused GEMV call.
        // This quantizes the input to Q8_1 once (not 8×) and uses a single
        // OMP parallel region (not 8×), saving ~7×(2µs quant + 6µs OMP)
        // = ~56µs per layer × 36 MoE layers = ~2ms per decode token.
        // ---------------------------------------------------------------
        struct ActiveExpert { int expert_id; float weight; int batch_idx; };
        ActiveExpert active_experts[16]; // stack-allocated, max top_k
        int num_active = 0;

        batch_projections_.clear();
        batch_projections_.reserve(top_k * 2);

        // Per-token dynamic dispatch for replicated experts.
        // When replicas are active, use ExpertReplicaSet::assignForToken()
        // to deterministically decide which socket computes each expert.
        bool compute_here[16]; // stack-allocated, max top_k

        // Convert float indices to int for replica dispatch
        int routing_int_indices[16]; // stack-allocated, max top_k
        for (int k = 0; k < top_k; ++k)
            routing_int_indices[k] = static_cast<int>(routing_idx_data[k]);

        if (params_.replica_set.num_replicated > 0)
        {
            params_.replica_set.assignForToken(
                routing_int_indices,
                routing_wt_data,
                top_k,
                params_.my_socket_id,
                params_.expert_mask,
                compute_here);
        }
        else
        {
            // No replicas — use simple mask/range check
            for (int k = 0; k < top_k; ++k)
            {
                const int expert_id = routing_int_indices[k];
                if (!params_.expert_mask.empty())
                    compute_here[k] = params_.expert_mask[expert_id];
                else
                    compute_here[k] = (expert_id >= local_start && expert_id < local_end);
            }
        }

        for (int k = 0; k < top_k; ++k)
        {
            if (!compute_here[k]) continue;

            const int expert_id = routing_int_indices[k];

            ITensorGemm *gate_gemm = cached_gate_gemm_[expert_id];
            ITensorGemm *up_gemm = cached_up_gemm_[expert_id];

            if (!gate_gemm || !up_gemm)
            {
                LOG_ERROR("[MoEExpertComputeStage] FATAL: Null gate/up GEMM engine for expert "
                    << expert_id << " (layer " << params_.layer_idx
                    << ", mask=" << (params_.expert_mask.empty() ? -1 : (int)params_.expert_mask[expert_id])
                    << ", replicated=" << params_.replica_set.is_replicated[expert_id]
                    << ", prepared_gate=" << (bool)params_.prepared_gate_gemm[expert_id] << ")");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            batch_projections_.push_back(
                {gate_gemm, scratch_gate_batch_[num_active].get(), intermediate, nullptr, "gate"});
            batch_projections_.push_back(
                {up_gemm, scratch_up_batch_[num_active].get(), intermediate, nullptr, "up"});

            active_experts[num_active] = {expert_id, routing_wt_data[k], num_active};
            num_active++;
        }

        // Single fused call: quantize once + single OMP region for all gate+up
        if (num_active > 0)
        {
            batch_projections_[0].kernel->multiply_fused_tensor(
                input_tensor, batch_projections_, /*m=*/1, d_model);
        }

        // ---------------------------------------------------------------
        // Phase 2: Fused SwiGLU + Down projection + weighted accumulate
        //
        // Strategy: apply SwiGLU for all experts first, then fuse all
        // down projections into a single OMP region. This saves
        // (num_active-1) OMP fork/join cycles (~8µs each) and improves
        // load balance via nowait (128 total chunks vs 4×32).
        // Falls back to sequential if fused path unavailable.
        // ---------------------------------------------------------------

        // Ensure per-expert output buffers for fused approach
        if (static_cast<int>(scratch_down_batch_.size()) < num_active)
        {
            scratch_down_batch_.resize(num_active);
            for (int i = 0; i < num_active; ++i)
            {
                if (!scratch_down_batch_[i])
                    scratch_down_batch_[i] = std::make_shared<FP32Tensor>(
                        std::vector<size_t>{1, static_cast<size_t>(d_model)});
            }
        }

        // Validate down GEMM engines
        for (int i = 0; i < num_active; ++i)
        {
            if (!cached_down_gemm_[active_experts[i].expert_id])
            {
                LOG_ERROR("[MoEExpertComputeStage] FATAL: Null down GEMM engine for expert "
                    << active_experts[i].expert_id << " (layer " << params_.layer_idx << ")");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Phase 2a: Apply SwiGLU for all experts (serial, ~0.1µs each)
        for (int i = 0; i < num_active; ++i)
        {
            const auto &info = active_experts[i];
            const float *gate_fp32 = scratch_gate_batch_[info.batch_idx]->data();
            const float *up_fp32 = scratch_up_batch_[info.batch_idx]->data();
            swiglu_scratch_batch_.resize(std::max(swiglu_scratch_batch_.size(),
                                                  static_cast<size_t>(num_active)));
            if (static_cast<int>(swiglu_scratch_batch_[i].size()) < intermediate)
                swiglu_scratch_batch_[i].resize(intermediate);

            primitives::compute_swiglu_serial(gate_fp32, up_fp32,
                                              swiglu_scratch_batch_[i].data(), intermediate);
        }

        // Phase 2b: Try fused multi-input down projections
        bool fused_ok = false;
        if (num_active >= 2)
        {
            ITensorGemm::FusedExpertDownDesc down_descs[16];
            for (int i = 0; i < num_active && i < 16; ++i)
            {
                const auto &info = active_experts[i];
                down_descs[i].kernel = cached_down_gemm_[info.expert_id];
                down_descs[i].input = swiglu_scratch_batch_[i].data();
                down_descs[i].output = scratch_down_batch_[i]->mutable_data();
                down_descs[i].n = d_model;
            }
            fused_ok = cached_down_gemm_[active_experts[0].expert_id]
                           ->multiply_fused_expert_down(down_descs, num_active, 1, intermediate);
        }

        if (fused_ok)
        {
            // Phase 2c: Weighted accumulate all outputs
            for (int i = 0; i < num_active; ++i)
            {
                const auto &info = active_experts[i];
                primitives::vec_axpy(output, scratch_down_batch_[i]->data(),
                                     info.weight, d_model);
            }
        }
        else
        {
            // Fallback: sequential per-expert SwiGLU + Down + accumulate
            float *scratch_out_ptr = scratch_out_->mutable_data();
            for (int i = 0; i < num_active; ++i)
            {
                const auto &info = active_experts[i];
                ITensorGemm *down_gemm = cached_down_gemm_[info.expert_id];

                fusedSwigluDown(
                    scratch_gate_batch_[info.batch_idx].get(),
                    scratch_up_batch_[info.batch_idx].get(),
                    scratch_out_.get(),
                    down_gemm, kernel, /*m=*/1, d_model, intermediate);

                primitives::vec_axpy(output, scratch_out_ptr, info.weight, d_model);
            }
        }

        LOG_TRACE("[MoEExpertComputeStage] Single-token decode (batched gate+up): " << num_active << " experts");
        return true;
    }

    void MoEExpertComputeStage::ensureGemmEnginesCached() const
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
        LOG_WARN("[MoEExpertComputeStage] GEMM engines not pre-resolved; "
                 "call prepareExpertGemmEngines() at graph build time for better perf");

        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        cached_gate_gemm_.resize(num_experts, nullptr);
        cached_up_gemm_.resize(num_experts, nullptr);
        cached_down_gemm_.resize(num_experts, nullptr);

        for (int e = local_start; e < local_end; ++e)
        {
            if (!params_.expert_gate_views[e])
                continue;
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
    // MoEExpertComputeStage::extractExpertViews — Delegates to MoEExpertWeightService
    // =========================================================================

    bool MoEExpertComputeStage::extractExpertViews(Params &params)
    {
        MoEWeightContext ctx{
            params.device_id,
            params.num_experts,
            params.expert_intermediate,
            params.d_model,
            params.local_expert_start,
            params.local_expert_count,
            params.layer_idx,
            params.expert_mask,
            params.gate_exps,
            params.up_exps,
            params.down_exps,
            params.expert_gate_views,
            params.expert_up_views,
            params.expert_down_views,
            params.prepared_gate_gemm,
            params.prepared_up_gemm,
            params.prepared_down_gemm,
            params.moe_owned_kernels,
            params.moe_packed_gate_lifetime,
            params.moe_packed_up_lifetime,
            params.moe_packed_down_lifetime
        };
        return MoEExpertWeightService::extractExpertViews(ctx);
    }

    bool MoEExpertComputeStage::prepareExpertGemmEngines(Params &params)
    {
        MoEWeightContext ctx{
            params.device_id,
            params.num_experts,
            params.expert_intermediate,
            params.d_model,
            params.local_expert_start,
            params.local_expert_count,
            params.layer_idx,
            params.expert_mask,
            params.gate_exps,
            params.up_exps,
            params.down_exps,
            params.expert_gate_views,
            params.expert_up_views,
            params.expert_down_views,
            params.prepared_gate_gemm,
            params.prepared_up_gemm,
            params.prepared_down_gemm,
            params.moe_owned_kernels,
            params.moe_packed_gate_lifetime,
            params.moe_packed_up_lifetime,
            params.moe_packed_down_lifetime
        };
        return MoEExpertWeightService::prepareGemmEngines(ctx);
    }

    size_t MoEExpertComputeStage::estimatedFlops() const
    {
        // Per token: top_k experts × (gate + up + down projections)
        // gate/up: d_model × intermediate
        // down: intermediate × d_model
        size_t per_expert = static_cast<size_t>(6) * params_.d_model * params_.expert_intermediate;
        return static_cast<size_t>(params_.seq_len) * params_.top_k * per_expert;
    }

    bool MoEExpertComputeStage::supportsBackend(ComputeBackendType backend) const
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

    StageBufferRequirements MoEExpertComputeStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        return reqs;
    }

    StageDumpInfo MoEExpertComputeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.routing_indices)
            info.addInput("routing_indices", params_.routing_indices,
                          static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        if (params_.routing_weights)
            info.addInput("routing_weights", params_.routing_weights,
                          static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        if (params_.gate_exps)
            info.addWeight("gate_exps", params_.gate_exps);
        if (params_.up_exps)
            info.addWeight("up_exps", params_.up_exps);
        if (params_.down_exps)
            info.addWeight("down_exps", params_.down_exps);
        if (params_.output)
            info.addOutput("output", params_.output, params_.seq_len, params_.d_model);

        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("expert_intermediate", params_.expert_intermediate);
        info.addScalarInt("local_expert_start", params_.local_expert_start);
        info.addScalarInt("local_expert_count", params_.local_expert_count);
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

} // namespace llaminar2
