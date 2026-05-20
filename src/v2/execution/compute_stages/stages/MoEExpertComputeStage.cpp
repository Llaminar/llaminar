/**
 * @file MoEExpertComputeStage.cpp
 * @brief Implementation of unified MoE FFN, shared expert, and shared expert gate stages
 */

#include "MoEExpertComputeStage.h"
#include "../../../execution/moe/DecodeExpertHistogram.h"
#include "../../../execution/moe/ExpertWeightTransfer.h"
#include "../../../execution/moe/ExpertWeightPayloadProvider.h"
#include "../../../execution/moe/MoEExpertWeightService.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/cpu/primitives/VectorPrimitives.h"
#include "../../../kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
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
        /// Create an FP32 scratch tensor with GPU memory pre-allocated when
        /// running on a GPU device.  Without this, scratch tensors are HOST_ONLY
        /// and multiply_fused_tensor() fails when it calls gpu_data_ptr().
        std::shared_ptr<FP32Tensor> makeScratchFP32(
            size_t rows, size_t cols, DeviceId device)
        {
            auto t = std::make_shared<FP32Tensor>(
                std::vector<size_t>{rows, cols});
            if (device.is_gpu())
                t->allocateOnDevice(device);
            return t;
        }
        void markGpuTensorWritten(TensorBase *output, DeviceId device, void *stream)
        {
            if (!output || !device.is_gpu())
                return;
            output->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE, device, stream);
        }

        /// Execute SwiGLU activation + Down projection via fused kernel when available,
        /// falling back to IMoEKernel::swiGLUFromTensors + separate GEMM when not (e.g., FP32 weights).
        /// Fully device-agnostic — tensor-aware kernel methods handle CPU/GPU dispatch.
        bool fusedSwigluDown(
            FP32Tensor *gate_tensor, FP32Tensor *up_tensor, TensorBase *output,
            ITensorGemm *down_gemm, IMoEKernel *moe_kernel,
            int m, int n, int intermediate,
            DeviceId device, void *stream)
        {
            // Try fused path first (quantized GEMM engines support this)
            if (down_gemm->multiply_tensor_with_fused_swiglu(
                    gate_tensor, up_tensor, output,
                    m, n, intermediate))
            {
                markGpuTensorWritten(output, device, stream);
                return true;
            }

            // Fallback: SwiGLU via tensor-aware kernel, then separate down GEMM.
            const int count = m * intermediate;
            moe_kernel->swiGLUFromTensors(gate_tensor, up_tensor, count);
            if (device.is_gpu() && gate_tensor->needsUpload() &&
                !gate_tensor->ensureOnDevice(device, stream))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to upload SwiGLU fallback output to "
                          << device.to_string());
                return false;
            }

            const bool ok = down_gemm->multiply_tensor(
                gate_tensor, output,
                m, n, intermediate);
            if (ok)
                markGpuTensorWritten(output, device, stream);
            return ok;
        }

        void markStandaloneGpuOutputWritten(TensorBase *output, DeviceId device, void *stream)
        {
            markGpuTensorWritten(output, device, stream);
        }
    } // anonymous namespace

    // =========================================================================
    // MoEExpertComputeStage — Unified Router + Expert FFN + Combine
    // =========================================================================

    MoEExpertComputeStage::MoEExpertComputeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.moe_runtime_table && params_.layer_idx >= 0)
        {
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
            if (params_.device_id.is_rocm() && params_.seq_len == 1 &&
                debugEnv().rocm.moe_grouped_decode && debugEnv().rocm.moe_device_routed_decode)
            {
                moe_runtime_table_initialized_ = runtimeTableHasActiveGroupedDecodeBank() ||
                                                 initializeMoERuntimeTableForGroupedDecode();
            }
            else
            {
                moe_runtime_table_initialized_ = runtimeTableHasActiveGroupedDecodeBank();
            }
        }
    }

    bool MoEExpertComputeStage::validatePreparedWeights(std::string *error) const
    {
        auto fail = [error](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };

        const bool has_slab_ref = params_.gate_slab_ref.has_value() ||
                                  params_.up_slab_ref.has_value() ||
                                  params_.down_slab_ref.has_value();
        if (!has_slab_ref)
        {
            if (error)
                error->clear();
            return true;
        }

        if (!params_.prepared_store)
            return fail("PreparedWeightStore is required for MoE expert slab refs");
        if (!params_.gate_slab_ref.has_value() ||
            !params_.up_slab_ref.has_value() ||
            !params_.down_slab_ref.has_value())
        {
            return fail("gate/up/down ExpertSlabRefs must be provided together");
        }

        auto check_slab = [&](const char *name, const ExpertSlabRef &ref)
        {
            const auto availability = params_.prepared_store->expertAvailabilityMask(ref);
            if (availability.empty())
                return fail(std::string("PreparedWeightStore does not contain expert slab for ") + name);
            if (params_.num_experts > 0 && availability.size() != static_cast<size_t>(params_.num_experts))
            {
                return fail(std::string("expert slab size mismatch for ") + name +
                            ": got " + std::to_string(availability.size()) +
                            ", expected " + std::to_string(params_.num_experts));
            }
            return true;
        };

        if (!check_slab("gate", *params_.gate_slab_ref))
            return false;
        if (!check_slab("up", *params_.up_slab_ref))
            return false;
        if (!check_slab("down", *params_.down_slab_ref))
            return false;

        if (error)
            error->clear();
        return true;
    }

    bool MoEExpertComputeStage::updateExpertMask(const std::vector<bool> &mask)
    {
        if (static_cast<int>(mask.size()) != params_.num_experts)
        {
            LOG_ERROR("[MoEExpertComputeStage] Expert mask size " << mask.size()
                                                                  << " != num_experts " << params_.num_experts);
            return false;
        }
        params_.expert_mask = mask;
        grouped_gateup_desc_table_id_ = -1;
        grouped_gateup_desc_table_num_experts_ = 0;
        grouped_gateup_desc_table_d_model_ = 0;
        grouped_gateup_desc_table_intermediate_ = 0;
        grouped_down_desc_table_id_ = -1;
        grouped_down_desc_table_num_experts_ = 0;
        grouped_down_desc_table_d_model_ = 0;
        grouped_down_desc_table_intermediate_ = 0;
        return true;
    }

    ExpertWeightBlobs MoEExpertComputeStage::detachAndSerializeExpert(int expert_id)
    {
        auto ctx = buildWeightContext();
        return MoEExpertWeightService::detachAndSerializeExpert(ctx, expert_id);
    }

    ExpertWeightBlobs MoEExpertComputeStage::serializeExpert(int expert_id) const
    {
        auto ctx = const_cast<MoEExpertComputeStage *>(this)->buildWeightContext();
        return MoEExpertWeightService::serializeExpert(ctx, expert_id);
    }

    size_t MoEExpertComputeStage::releaseRawExpertWeights()
    {
        auto ctx = buildWeightContext();
        size_t freed = MoEExpertWeightService::releaseRawWeights(ctx);
        raw_weights_released_ = true;
        return freed;
    }

    // ── Phased rebalance API ─────────────────────────────────────────────

    std::vector<const TensorBase *> MoEExpertComputeStage::releaseDepartedExperts(
        const std::vector<bool> &new_mask)
    {
        auto ctx = buildWeightContext();
        return MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    }

    bool MoEExpertComputeStage::registerAndPrepareNewExperts(
        const std::vector<bool> &new_mask,
        const std::unordered_map<int, ExpertWeightBlobs> *received_weights)
    {
        auto ctx = buildWeightContext();
        const bool ok = MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, received_weights);
        if (ok)
        {
            grouped_gateup_desc_table_id_ = -1;
            grouped_gateup_desc_table_num_experts_ = 0;
            grouped_gateup_desc_table_d_model_ = 0;
            grouped_gateup_desc_table_intermediate_ = 0;
            grouped_down_desc_table_id_ = -1;
            grouped_down_desc_table_num_experts_ = 0;
            grouped_down_desc_table_d_model_ = 0;
            grouped_down_desc_table_intermediate_ = 0;
            moe_runtime_table_initialized_ = false;
        }
        return ok;
    }

    void MoEExpertComputeStage::applyExpertMask(const std::vector<bool> &new_mask)
    {
        params_.expert_mask = new_mask;
        cached_gate_gemm_.clear();
        cached_up_gemm_.clear();
        cached_down_gemm_.clear();
        grouped_gateup_desc_table_id_ = -1;
        grouped_gateup_desc_table_num_experts_ = 0;
        grouped_gateup_desc_table_d_model_ = 0;
        grouped_gateup_desc_table_intermediate_ = 0;
        grouped_down_desc_table_id_ = -1;
        grouped_down_desc_table_num_experts_ = 0;
        grouped_down_desc_table_d_model_ = 0;
        grouped_down_desc_table_intermediate_ = 0;
        moe_runtime_table_initialized_ = false;
    }

    MoEWeightContext MoEExpertComputeStage::buildWeightContext()
    {
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
            params_.moe_packed_down_lifetime,
            payload_provider_,
            params_.prepared_store,
            params_.expert_registry,
            params_.gate_slab_ref,
            params_.up_slab_ref,
            params_.down_slab_ref};
    }

    IMoEKernel *MoEExpertComputeStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return bindStageStream(moe_kernel_);
    }

    bool MoEExpertComputeStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.output)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null input/output tensor");
            return false;
        }

        if (params_.device_id.is_gpu() && !params_.output_registered_in_arena)
        {
            auto *output_tensor = dynamic_cast<TensorBase *>(params_.output);
            if (!output_tensor || !output_tensor->ensureOnDevice(params_.device_id))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to allocate standalone output on "
                          << params_.device_id.to_string() << " for layer " << params_.layer_idx);
                return false;
            }
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

        if (!params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null routing_indices or routing_weights");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;
        const bool is_gpu = params_.device_id.is_gpu();

        const bool has_prepared_expert_state =
            (!params_.prepared_gate_gemm.empty() &&
             params_.prepared_gate_gemm.size() == static_cast<size_t>(num_experts)) ||
            (params_.prepared_store &&
             params_.gate_slab_ref.has_value() &&
             params_.up_slab_ref.has_value() &&
             params_.down_slab_ref.has_value());
        if (params_.expert_gate_views.empty() && !has_prepared_expert_state)
        {
            LOG_ERROR("[MoEExpertComputeStage] Requires pre-extracted expert views or prepared expert slabs. "
                      "Call extractExpertViews()/prepareExpertGemmEngines() at graph build time.");
            return false;
        }

        // Get device-appropriate MoE kernel for gather/scatter
        IMoEKernel *kernel = ensureMoEKernel();

        // Zero the output buffer via tensor-aware kernel (works for both CPU and GPU)
        const size_t output_bytes = static_cast<size_t>(seq_len) * d_model * sizeof(float);
        kernel->zeroBuffer(params_.output, output_bytes);

        // Expert Parallelism: determine which experts this rank processes
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        const bool has_prefill_mask = !params_.replica_set.prefill_mask.empty();
        const std::vector<bool> &prefill_mask_ref = params_.replica_set.prefill_mask;
        const bool has_replicas = params_.replica_set.num_replicated > 0;

        // =====================================================================
        // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
        // All 5 kernels launched with zero host sync, counts stay on device.
        // This is the ONLY ROCm prefill path — no per-expert fallback.
        // =====================================================================
        if (is_gpu && canUseFixedTopologyGroupedPrefill())
        {
            // Ensure descriptor tables are built (lazy, only first call)
            bool tables_ready = true;
            if (grouped_gateup_desc_table_id_ < 0 || grouped_down_desc_table_id_ < 0)
            {
                if (static_cast<int>(all_expert_ids_.size()) != num_experts)
                {
                    all_expert_ids_.resize(static_cast<size_t>(num_experts));
                    std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
                }
                tables_ready = ensureGemmEnginesForExperts(all_expert_ids_) &&
                               ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate) &&
                               ensureGroupedDownDescriptorTable(kernel, d_model, intermediate);
            }

            if (!tables_ready)
            {
                LOG_ERROR("[MoEExpertComputeStage] Grouped prefill: descriptor table build failed, layer "
                          << params_.layer_idx);
                return false;
            }

            if (!executeFixedTopologyGroupedPrefill(kernel, seq_len))
            {
                LOG_ERROR("[MoEExpertComputeStage] Grouped prefill pipeline failed, layer "
                          << params_.layer_idx);
                return false;
            }

            if (params_.device_id.is_gpu())
                markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
            return true;
        }

        // ROCm prefill MUST use grouped path — fail hard if conditions not met
        if (params_.device_id.is_rocm() && params_.seq_len > 1)
        {
            LOG_ERROR("[MoEExpertComputeStage] ROCm prefill (seq_len=" << params_.seq_len
                      << ") requires grouped prefill but conditions not met: "
                      << "moe_grouped_prefill=" << debugEnv().rocm.moe_grouped_prefill
                      << ", fullOwnership=" << hasFullLocalExpertOwnership()
                      << ", allEnabled=" << expertMaskAllEnabled()
                      << ", replicas=" << params_.replica_set.num_replicated
                      << ", layer=" << params_.layer_idx);
            return false;
        }

        // =====================================================================
        // GPU prefill path: grouping + gather/scatter stay on device
        // Avoids D2H of routing tensors and CPU grouping O(seq_len * top_k)
        // =====================================================================
        if (is_gpu && kernel->prepareExpertGroups(
                          params_.routing_indices, params_.routing_weights,
                          seq_len, num_experts, top_k))
        {

            // Scratch sizing based on max local expert token count
            int max_batch = 0;
            std::vector<int> active_local_experts;
            for (int e = 0; e < num_experts; ++e)
            {
                bool is_local;
                if (has_prefill_mask)
                    is_local = prefill_mask_ref[e];
                else if (!params_.expert_mask.empty())
                {
                    is_local = params_.expert_mask[e];
                    if (is_local && has_replicas &&
                        params_.replica_set.is_replicated[e] &&
                        params_.replica_set.owner_socket[e] != params_.my_socket_id)
                        is_local = false;
                }
                else
                    is_local = (e >= local_start && e < local_end);
                if (is_local)
                {
                    if (kernel->getExpertTokenCount(e) > 0)
                        active_local_experts.push_back(e);
                    max_batch = std::max(max_batch, kernel->getExpertTokenCount(e));
                }
            }

            if (!ensureGemmEnginesForExperts(active_local_experts))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to prepare GPU GEMM engines for active experts");
                return false;
            }

            if (max_batch > 0 && max_batch > scratch_capacity_)
            {
                scratch_batch_ = makeScratchFP32(max_batch, d_model, params_.device_id);
                scratch_gate_ = makeScratchFP32(max_batch, intermediate, params_.device_id);
                scratch_up_ = makeScratchFP32(max_batch, intermediate, params_.device_id);
                scratch_out_ = makeScratchFP32(max_batch, d_model, params_.device_id);
                scratch_capacity_ = max_batch;
            }

            for (int expert_id = 0; expert_id < num_experts; ++expert_id)
            {
                int count = kernel->getExpertTokenCount(expert_id);
                if (count == 0)
                    continue;

                // Same locality check as CPU path
                bool is_local;
                if (has_prefill_mask)
                    is_local = prefill_mask_ref[expert_id];
                else if (!params_.expert_mask.empty())
                {
                    is_local = params_.expert_mask[expert_id];
                    if (is_local && has_replicas &&
                        params_.replica_set.is_replicated[expert_id] &&
                        params_.replica_set.owner_socket[expert_id] != params_.my_socket_id)
                        is_local = false;
                }
                else
                    is_local = (expert_id >= local_start && expert_id < local_end);
                if (!is_local)
                    continue;

                kernel->gatherExpertBatch(
                    params_.input, scratch_batch_.get(), expert_id, d_model);
                if (scratch_batch_->needsUpload() &&
                    !scratch_batch_->ensureOnDevice(params_.device_id, gpuStream()))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Failed to upload gathered CUDA expert batch for expert "
                              << expert_id << " layer " << params_.layer_idx);
                    return false;
                }

                ITensorGemm *gate_gemm = cached_gate_gemm_[expert_id];
                ITensorGemm *up_gemm = cached_up_gemm_[expert_id];
                ITensorGemm *down_gemm = cached_down_gemm_[expert_id];

                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {gate_gemm, scratch_gate_.get(), intermediate, nullptr, "gate"},
                    {up_gemm, scratch_up_.get(), intermediate, nullptr, "up"}};
                if (!gate_gemm->multiply_fused_tensor(
                        scratch_batch_.get(), projections, count, d_model))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CUDA gate/up projection failed for expert "
                              << expert_id << " layer " << params_.layer_idx);
                    return false;
                }
                for (const auto &projection : projections)
                    markGpuTensorWritten(projection.output, params_.device_id, gpuStream());

                if (!fusedSwigluDown(
                        scratch_gate_.get(), scratch_up_.get(), scratch_out_.get(),
                        down_gemm, kernel, count, d_model, intermediate,
                        params_.device_id, gpuStream()))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CUDA SwiGLU/down projection failed for expert "
                              << expert_id << " layer " << params_.layer_idx);
                    return false;
                }

                kernel->scatterExpertResults(
                    params_.output, scratch_out_.get(), expert_id, d_model);
            }

            if (params_.output->needsUpload() &&
                !params_.output->ensureOnDevice(params_.device_id, gpuStream()))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to upload CUDA routed expert output for layer "
                          << params_.layer_idx);
                return false;
            }
            if (!params_.output_registered_in_arena)
                markStandaloneGpuOutputWritten(params_.output, params_.device_id, gpuStream());

            LOG_TRACE("[MoEExpertComputeStage] GPU prefill: " << seq_len << " tokens, "
                                                              << top_k << " experts per token");
            return true;
        }

        // =====================================================================
        // CPU prefill path: D2H routing data + host-side grouping
        // =====================================================================

        // Read pre-computed routing results from MoERoutingStage.
        // These are small (seq_len * top_k floats each), so D2H is acceptable.
        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();

        // Step 3: Group tokens by expert for batched GEMM execution.
        // With EP, we only process experts in our local range, but still
        // build the full routing map so scratch sizing is correct.

        std::vector<std::vector<std::pair<int, float>>> expert_token_lists(num_experts);

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

        // Ensure GEMM engine pointers are copied into the stage-local cache and
        // validate every active expert before execution.
        int max_batch = 0;
        std::vector<int> active_local_experts;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &tl = expert_token_lists[expert_id];
            if (!tl.empty())
                active_local_experts.push_back(expert_id);
            max_batch = std::max(max_batch, static_cast<int>(tl.size()));
        }

        if (!ensureGemmEnginesForExperts(active_local_experts))
        {
            LOG_ERROR("[MoEExpertComputeStage] Missing prepared GEMM engines for active prefill experts");
            return false;
        }

        // Ensure scratch buffers have enough capacity for largest expert batch
        if (max_batch > 0 && max_batch > scratch_capacity_)
        {
            scratch_batch_ = makeScratchFP32(max_batch, d_model, params_.device_id);
            scratch_gate_ = makeScratchFP32(max_batch, intermediate, params_.device_id);
            scratch_up_ = makeScratchFP32(max_batch, intermediate, params_.device_id);
            scratch_out_ = makeScratchFP32(max_batch, d_model, params_.device_id);
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

            // Gather tokens into reusable scratch batch via tensor-aware kernel.
            // CPU: kernel reads data()/mutable_data().
            // GPU: kernel uploads indices to device staging, gathers on device.
            kernel->gatherTokenBatchFromTensors(
                params_.input, scratch_batch_.get(),
                token_indices.data(), num_tokens, d_model);
            if (is_gpu && scratch_batch_->needsUpload() &&
                !scratch_batch_->ensureOnDevice(params_.device_id, gpuStream()))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to upload gathered expert batch for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }

            // Use cached GEMM engines (device-agnostic via ITensorGemm)
            ITensorGemm *gate_gemm = cached_gate_gemm_[expert_id];
            ITensorGemm *up_gemm = cached_up_gemm_[expert_id];
            ITensorGemm *down_gemm = cached_down_gemm_[expert_id];

            // Gate+Up projections via fused multi-projection (quantizes input once)
            std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                {gate_gemm, scratch_gate_.get(), intermediate, nullptr, "gate"},
                {up_gemm, scratch_up_.get(), intermediate, nullptr, "up"}};
            if (!gate_gemm->multiply_fused_tensor(
                    scratch_batch_.get(), projections,
                    num_tokens, d_model))
            {
                LOG_ERROR("[MoEExpertComputeStage] Gate/up projection failed for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }
            if (is_gpu)
            {
                for (const auto &projection : projections)
                    markGpuTensorWritten(projection.output, params_.device_id, gpuStream());
            }

            // SwiGLU+Down via fused kernel with fallback through MoE kernel
            if (!fusedSwigluDown(
                    scratch_gate_.get(), scratch_up_.get(), scratch_out_.get(),
                    down_gemm, kernel, num_tokens, d_model, intermediate,
                    params_.device_id, gpuStream()))
            {
                LOG_ERROR("[MoEExpertComputeStage] SwiGLU/down projection failed for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }

            // Scatter weighted results back via tensor-aware kernel.
            kernel->scatterAddWeightedFromTensors(
                params_.output, scratch_out_.get(),
                token_indices.data(), token_weights.data(),
                num_tokens, d_model);
        }

        LOG_TRACE("[MoEExpertComputeStage] Processed " << seq_len << " tokens via GEMM kernels, "
                                                       << top_k << " experts per token");
        if (is_gpu && params_.output->needsUpload() &&
            !params_.output->ensureOnDevice(params_.device_id, gpuStream()))
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to upload routed expert output for layer "
                      << params_.layer_idx);
            return false;
        }
        if (is_gpu && !params_.output_registered_in_arena)
            markStandaloneGpuOutputWritten(params_.output, params_.device_id, gpuStream());
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
        const bool is_gpu = params_.device_id.is_gpu();

        const bool has_prepared_expert_state =
            (!params_.prepared_gate_gemm.empty() &&
             params_.prepared_gate_gemm.size() == static_cast<size_t>(num_experts)) ||
            (params_.prepared_store &&
             params_.gate_slab_ref.has_value() &&
             params_.up_slab_ref.has_value() &&
             params_.down_slab_ref.has_value());
        if (params_.expert_gate_views.empty() && !has_prepared_expert_state)
        {
            LOG_ERROR("[MoEExpertComputeStage] Requires pre-extracted expert views or prepared expert slabs.");
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();

        // Zero output via tensor-aware kernel (works for both CPU and GPU)
        kernel->zeroBuffer(params_.output, static_cast<size_t>(d_model) * sizeof(float));

        // EP range
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        // Ensure batch scratch buffers for gate+up (one per top-k expert).
        // All experts' gate+up are fused into a single OMP region, so we need
        // all outputs to exist simultaneously.
        if (static_cast<int>(scratch_gate_batch_.size()) < top_k)
        {
            scratch_gate_batch_.resize(top_k);
            scratch_up_batch_.resize(top_k);
            for (int i = 0; i < top_k; ++i)
            {
                scratch_gate_batch_[i] = makeScratchFP32(1, intermediate, params_.device_id);
                scratch_up_batch_[i] = makeScratchFP32(1, intermediate, params_.device_id);
            }
        }
        // Scratch for down projection output (reused per expert)
        if (!scratch_out_)
        {
            scratch_out_ = makeScratchFP32(1, d_model, params_.device_id);
        }

        // Use input tensor directly (no gather needed for 1 token)
        const TensorBase *input_tensor = params_.input;

        if (params_.device_id.is_rocm() && params_.moe_runtime_table && params_.layer_idx >= 0 && !moe_runtime_layer_)
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
        if (params_.device_id.is_rocm() && !moe_runtime_table_initialized_ &&
            debugEnv().rocm.moe_grouped_decode && debugEnv().rocm.moe_device_routed_decode)
        {
            moe_runtime_table_initialized_ = runtimeTableHasActiveGroupedDecodeBank() ||
                                             initializeMoERuntimeTableForGroupedDecode();
        }

        // Snapshot-enabled builds keep legacy routing tensors authoritative for
        // parity dumps, so MoERoutingStage does not populate the runtime top-k
        // table. Consuming that table here would use stale/zero routing data.
#if defined(ENABLE_PIPELINE_SNAPSHOTS)
        const bool can_try_device_routed_decode = false;
#else
        const bool can_try_device_routed_decode =
            params_.device_id.is_rocm() &&
            debugEnv().rocm.moe_grouped_decode &&
            debugEnv().rocm.moe_device_routed_decode &&
            params_.moe_runtime_table &&
            moe_runtime_layer_ &&
            moe_runtime_table_initialized_ &&
            runtimeTableHasActiveGroupedDecodeBank() &&
            top_k > 0 && top_k <= 16 &&
            params_.replica_set.num_replicated == 0 &&
            hasFullLocalExpertOwnership() &&
            expertMaskAllEnabled();
#endif

        if (can_try_device_routed_decode)
        {
            bool device_routed_done = false;
            const bool have_grouped_tables =
                grouped_gateup_desc_table_id_ >= 0 &&
                grouped_gateup_desc_table_num_experts_ == num_experts &&
                grouped_gateup_desc_table_d_model_ == d_model &&
                grouped_gateup_desc_table_intermediate_ == intermediate &&
                grouped_down_desc_table_id_ >= 0 &&
                grouped_down_desc_table_num_experts_ == num_experts &&
                grouped_down_desc_table_d_model_ == d_model &&
                grouped_down_desc_table_intermediate_ == intermediate;

            bool grouped_tables_ready = have_grouped_tables;
            if (!grouped_tables_ready)
            {
                if (static_cast<int>(all_expert_ids_.size()) != num_experts)
                {
                    all_expert_ids_.resize(static_cast<size_t>(num_experts));
                    std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
                }

                grouped_tables_ready = ensureGemmEnginesForExperts(all_expert_ids_) &&
                                       ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate) &&
                                       ensureGroupedDownDescriptorTable(kernel, d_model, intermediate);
            }

            if (grouped_tables_ready)
            {
                ITensor *gate_outputs[16] = {};
                ITensor *up_outputs[16] = {};
                for (int k = 0; k < top_k; ++k)
                {
                    gate_outputs[k] = scratch_gate_batch_[k].get();
                    up_outputs[k] = scratch_up_batch_[k].get();
                }

                const bool gateup_done = kernel->groupedExpertGateUpDecodeFromRuntime(
                    moe_runtime_layer_,
                    input_tensor,
                    grouped_gateup_desc_table_id_,
                    top_k,
                    gate_outputs,
                    up_outputs,
                    d_model,
                    intermediate);

                if (gateup_done)
                {
                    device_routed_done = kernel->groupedExpertDownDecodeFromRuntime(
                        gate_outputs,
                        up_outputs,
                        moe_runtime_layer_,
                        grouped_down_desc_table_id_,
                        top_k,
                        params_.output,
                        d_model,
                        intermediate);
                }

                if (!device_routed_done)
                {
                    LOG_DEBUG("[MoEExpertComputeStage] Device-routed ROCm decode path unavailable for layer "
                              << params_.layer_idx << "; using host-routed fallback");
                }
            }

            if (device_routed_done)
            {
                if (params_.output->needsUpload() &&
                    !params_.output->ensureOnDevice(params_.device_id, gpuStream()))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Failed to upload device-routed decode output for layer "
                              << params_.layer_idx);
                    return false;
                }
                return true;
            }

            kernel->zeroBuffer(params_.output, static_cast<size_t>(d_model) * sizeof(float));
        }

        if (!params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null routing_indices or routing_weights (single-token)");
            return false;
        }

        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();

        // ---------------------------------------------------------------
        // Phase 1: Batch all experts' gate+up into ONE fused GEMV call.
        // This quantizes the input to Q8_1 once (not 8×) and uses a single
        // OMP parallel region (not 8×), saving ~7×(2µs quant + 6µs OMP)
        // = ~56µs per layer × 36 MoE layers = ~2ms per decode token.
        // ---------------------------------------------------------------
        struct ActiveExpert
        {
            int expert_id;
            float weight;
            int batch_idx;
        };
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

        if (is_gpu)
        {
            std::vector<int> active_expert_ids;
            active_expert_ids.reserve(top_k);
            for (int k = 0; k < top_k; ++k)
                if (compute_here[k])
                    active_expert_ids.push_back(routing_int_indices[k]);
            if (!ensureGemmEnginesForExperts(active_expert_ids))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to prepare GPU GEMM engines for decode experts");
                return false;
            }
        }
        else
        {
            ensureGemmEnginesCached();
        }

        for (int k = 0; k < top_k; ++k)
        {
            if (!compute_here[k])
                continue;

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
            if (!batch_projections_[0].kernel->multiply_fused_tensor(
                    input_tensor, batch_projections_, /*m=*/1, d_model))
            {
                LOG_ERROR("[MoEExpertComputeStage] Decode gate/up batched projection failed for layer "
                          << params_.layer_idx);
                return false;
            }
            if (is_gpu)
            {
                for (const auto &projection : batch_projections_)
                    markGpuTensorWritten(projection.output, params_.device_id, gpuStream());
            }
        }

        // ---------------------------------------------------------------
        // Phase 2: Fused SwiGLU + Down projection + weighted accumulate
        //
        // GPU path: per-expert fusedSwigluDown (tensor-based, runs on GPU)
        //           + GPU weightedAdd (avoids all D2H transfers).
        // CPU path: batch SwiGLU + fused multi-input down projections
        //           + vec_axpy accumulation in a single OMP region.
        // ---------------------------------------------------------------

        if (is_gpu)
        {
            bool grouped_down_done = false;
            if (debugEnv().rocm.moe_grouped_decode && num_active > 0 && num_active <= 16)
            {
                ITensor *gate_tensors[16] = {};
                ITensor *up_tensors[16] = {};
                int grouped_expert_ids[16] = {};
                float grouped_weights[16] = {};
                DeviceNativeVNNIMatrixDesc down_descs[16] = {};
                bool grouped_supported = true;

                for (int i = 0; i < num_active; ++i)
                {
                    const auto &info = active_experts[i];
                    ITensorGemm *down_gemm = cached_down_gemm_[info.expert_id];
                    if (!down_gemm)
                    {
                        LOG_ERROR("[MoEExpertComputeStage] FATAL: Null down GEMM engine for expert "
                                  << info.expert_id << " (layer " << params_.layer_idx << ")");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    DeviceNativeVNNIMatrixDesc desc;
                    if (!down_gemm->exportNativeVNNIMatrixDesc(desc) ||
                        desc.n != d_model || desc.k != intermediate)
                    {
                        grouped_supported = false;
                        break;
                    }

                    gate_tensors[i] = scratch_gate_batch_[info.batch_idx].get();
                    up_tensors[i] = scratch_up_batch_[info.batch_idx].get();
                    grouped_expert_ids[i] = info.expert_id;
                    grouped_weights[i] = info.weight;
                    down_descs[i] = desc;
                }

                if (grouped_supported)
                {
                    grouped_down_done = kernel->groupedExpertDownDecode(
                        gate_tensors,
                        up_tensors,
                        grouped_expert_ids,
                        grouped_weights,
                        down_descs,
                        num_active,
                        params_.output,
                        d_model,
                        intermediate);
                    if (!grouped_down_done)
                    {
                        LOG_DEBUG("[MoEExpertComputeStage] Grouped ROCm decode down path unavailable for layer "
                                  << params_.layer_idx << "; using per-expert fallback");
                    }
                }
            }

            if (!grouped_down_done)
            {
                // GPU Phase 2 fallback: sequential per-expert SwiGLU+Down on GPU + GPU accumulate.
                // fusedSwigluDown uses multiply_tensor_with_fused_swiglu (tensor-based,
                // executed on GPU by ROCmQuantisedGemmKernel).  No D2H transfers occur.
                if (!scratch_out_)
                    scratch_out_ = makeScratchFP32(1, d_model, params_.device_id);

                for (int i = 0; i < num_active; ++i)
                {
                    const auto &info = active_experts[i];
                    ITensorGemm *down_gemm = cached_down_gemm_[info.expert_id];

                    if (!down_gemm)
                    {
                        LOG_ERROR("[MoEExpertComputeStage] FATAL: Null down GEMM engine for expert "
                                  << info.expert_id << " (layer " << params_.layer_idx << ")");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    // Fused SwiGLU + Down on GPU (tensor-based).
                    // After Phase 1, scratch_gate/up are DEVICE_AUTHORITATIVE.
                    // fusedSwigluDown primary path calls multiply_tensor_with_fused_swiglu
                    // which handles tensor coherence internally.
                    if (!fusedSwigluDown(
                            scratch_gate_batch_[info.batch_idx].get(),
                            scratch_up_batch_[info.batch_idx].get(),
                            scratch_out_.get(),
                            down_gemm, kernel, /*m=*/1, d_model, intermediate,
                            params_.device_id, gpuStream()))
                    {
                        LOG_ERROR("[MoEExpertComputeStage] CUDA decode SwiGLU/down projection failed for expert "
                                  << info.expert_id << " layer " << params_.layer_idx);
                        return false;
                    }

                    // GPU weighted accumulate: output += weight * scratch_out
                    kernel->weightedAddFromTensors(
                        params_.output, scratch_out_.get(), info.weight, d_model);
                }
            }
            if (params_.output->needsUpload() &&
                !params_.output->ensureOnDevice(params_.device_id, gpuStream()))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to upload CUDA decode routed expert output for layer "
                          << params_.layer_idx);
                return false;
            }
        }
        else
        {
            // CPU Phase 2: batch SwiGLU + fused down projections + vec_axpy

            float *output = params_.output->mutable_data();

            // Ensure per-expert output buffers for fused approach
            if (static_cast<int>(scratch_down_batch_.size()) < num_active)
            {
                scratch_down_batch_.resize(num_active);
                for (int i = 0; i < num_active; ++i)
                {
                    if (!scratch_down_batch_[i])
                        scratch_down_batch_[i] = makeScratchFP32(1, d_model, params_.device_id);
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

                    if (!fusedSwigluDown(
                            scratch_gate_batch_[info.batch_idx].get(),
                            scratch_up_batch_[info.batch_idx].get(),
                            scratch_out_.get(),
                            down_gemm, kernel, /*m=*/1, d_model, intermediate,
                            params_.device_id, gpuStream()))
                    {
                        LOG_ERROR("[MoEExpertComputeStage] Decode SwiGLU/down projection failed for expert "
                                  << info.expert_id << " layer " << params_.layer_idx);
                        return false;
                    }

                    primitives::vec_axpy(output, scratch_out_ptr, info.weight, d_model);
                }
            }

        } // end CPU Phase 2 else block

        LOG_TRACE("[MoEExpertComputeStage] Single-token decode (batched gate+up): " << num_active << " experts");
        if (is_gpu && !params_.output_registered_in_arena)
            markStandaloneGpuOutputWritten(params_.output, params_.device_id, gpuStream());
        return true;
    }

    void MoEExpertComputeStage::ensureGemmEnginesCached()
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

        // Phase C: Resolve from PreparedWeightStore if slab refs are cached
        if (params_.prepared_store && params_.gate_slab_ref.has_value())
        {
            cached_gate_gemm_.resize(num_experts, nullptr);
            cached_up_gemm_.resize(num_experts, nullptr);
            cached_down_gemm_.resize(num_experts, nullptr);

            const int local_start = params_.local_expert_start;
            const int local_count = (params_.local_expert_count < 0)
                                        ? num_experts
                                        : params_.local_expert_count;
            const int local_end = local_start + local_count;

            for (int e = local_start; e < local_end; ++e)
            {
                if (!params_.expert_mask.empty() && !params_.expert_mask[e])
                    continue;
                cached_gate_gemm_[e] = params_.prepared_store->expertGemmKernel(*params_.gate_slab_ref, e);
                cached_up_gemm_[e] = params_.prepared_store->expertGemmKernel(*params_.up_slab_ref, e);
                cached_down_gemm_[e] = params_.prepared_store->expertGemmKernel(*params_.down_slab_ref, e);
            }

            LOG_DEBUG("[MoEExpertComputeStage] Resolved GEMM engines from PreparedWeightStore"
                      << " (layer " << params_.layer_idx << ")");
            return;
        }

        // All local experts must be prepared at graph-build time.
        // If we reach here, it means prepareExpertGemmEngines() was not called.
        LOG_ERROR("[MoEExpertComputeStage] GEMM engines not pre-resolved for layer "
                  << params_.layer_idx << ". All experts must be prepared at graph build time. "
                  << "Ensure prepareExpertGemmEngines() is called during graph construction.");

        // Initialize empty cache to avoid repeated error logging
        cached_gate_gemm_.resize(num_experts, nullptr);
        cached_up_gemm_.resize(num_experts, nullptr);
        cached_down_gemm_.resize(num_experts, nullptr);
    }

    bool MoEExpertComputeStage::ensureGemmEnginesForExperts(const std::vector<int> &expert_ids)
    {
        const int num_experts = params_.num_experts;
        if (expert_ids.empty())
            return true;

        if (!params_.device_id.is_gpu())
        {
            ensureGemmEnginesCached();
            for (int expert_id : expert_ids)
            {
                if (expert_id < 0 || expert_id >= num_experts)
                {
                    LOG_ERROR("[MoEExpertComputeStage] Invalid expert id " << expert_id
                                                                           << " for layer " << params_.layer_idx);
                    return false;
                }
                if (cached_gate_gemm_.size() <= static_cast<size_t>(expert_id) ||
                    cached_up_gemm_.size() <= static_cast<size_t>(expert_id) ||
                    cached_down_gemm_.size() <= static_cast<size_t>(expert_id) ||
                    !cached_gate_gemm_[expert_id] ||
                    !cached_up_gemm_[expert_id] ||
                    !cached_down_gemm_[expert_id])
                {
                    LOG_ERROR("[MoEExpertComputeStage] Missing prepared CPU GEMM engine for expert "
                              << expert_id << " layer " << params_.layer_idx
                              << "; CPU expert repack fallback is disabled");
                    return false;
                }
            }
            return true;
        }

        if (params_.prepared_gate_gemm.size() != static_cast<size_t>(num_experts))
        {
            params_.prepared_gate_gemm.assign(num_experts, nullptr);
            params_.prepared_up_gemm.assign(num_experts, nullptr);
            params_.prepared_down_gemm.assign(num_experts, nullptr);
        }

        std::vector<bool> needed(num_experts, false);
        bool has_missing = false;
        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= num_experts)
            {
                LOG_ERROR("[MoEExpertComputeStage] Invalid expert id " << expert_id
                                                                       << " for layer " << params_.layer_idx);
                return false;
            }
            needed[expert_id] = true;
            if (!params_.prepared_gate_gemm[expert_id] ||
                !params_.prepared_up_gemm[expert_id] ||
                !params_.prepared_down_gemm[expert_id])
            {
                has_missing = true;
            }
        }

        if (has_missing)
        {
            // Missing experts MUST come from transfer blobs (dynamic rebalancing).
            // No fallback to raw host tensor data — all local experts are prepared
            // upfront at graph-build time. Missing engines at this point means either
            // a newly-arrived expert via rebalancing (must have a transfer blob) or
            // an initialization error.
            if (!payload_provider_)
            {
                LOG_ERROR("[MoEExpertComputeStage] Missing GEMM engines for layer "
                          << params_.layer_idx << " but no payload provider available. "
                          << "All local experts must be prepared at graph-build time.");
                return false;
            }

            auto provider_payloads = payload_provider_->payloadsForLayer(params_.layer_idx);
            if (provider_payloads.empty())
            {
                LOG_ERROR("[MoEExpertComputeStage] Missing GEMM engines for layer "
                          << params_.layer_idx << " and no transfer blobs available. "
                          << "Ensure the unified registry prepared all local experts, "
                          << "or dynamic expert transfer blobs were registered.");
                return false;
            }

            auto ctx = buildWeightContext();
            if (!MoEExpertWeightService::registerAndPrepareNewExperts(ctx, needed, &provider_payloads))
                return false;
        }

        cached_gate_gemm_ = params_.prepared_gate_gemm;
        cached_up_gemm_ = params_.prepared_up_gemm;
        cached_down_gemm_ = params_.prepared_down_gemm;

        auto bind_if_needed = [this](ITensorGemm *gemm)
        {
            if (!gemm)
                return;
            gemm->setGPUStream(gpuStream());
            if (bound_workspace_)
            {
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer && !consumer->hasWorkspace())
                    consumer->bindWorkspace(bound_workspace_);
            }
        };

        for (int expert_id : expert_ids)
        {
            if (!cached_gate_gemm_[expert_id] ||
                !cached_up_gemm_[expert_id] ||
                !cached_down_gemm_[expert_id])
            {
                LOG_ERROR("[MoEExpertComputeStage] Missing prepared GPU GEMM engine for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }
            bind_if_needed(cached_gate_gemm_[expert_id]);
            bind_if_needed(cached_up_gemm_[expert_id]);
            bind_if_needed(cached_down_gemm_[expert_id]);
        }
        return true;
    }

    bool MoEExpertComputeStage::ensureGroupedGateUpDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || params_.num_experts <= 0 || d_model <= 0 || intermediate <= 0)
            return false;

        if (grouped_gateup_desc_table_id_ >= 0 &&
            grouped_gateup_desc_table_num_experts_ == params_.num_experts &&
            grouped_gateup_desc_table_d_model_ == d_model &&
            grouped_gateup_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        std::vector<DeviceNativeVNNIMatrixDesc> gate_descs(static_cast<size_t>(params_.num_experts));
        std::vector<DeviceNativeVNNIMatrixDesc> up_descs(static_cast<size_t>(params_.num_experts));
        int valid_descs = 0;
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (cached_gate_gemm_.size() <= static_cast<size_t>(expert_id) ||
                cached_up_gemm_.size() <= static_cast<size_t>(expert_id) ||
                !cached_gate_gemm_[static_cast<size_t>(expert_id)] ||
                !cached_up_gemm_[static_cast<size_t>(expert_id)])
            {
                continue;
            }

            DeviceNativeVNNIMatrixDesc gate_desc;
            DeviceNativeVNNIMatrixDesc up_desc;
            if (!cached_gate_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(gate_desc) ||
                !cached_up_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(up_desc) ||
                gate_desc.n != intermediate || gate_desc.k != d_model ||
                up_desc.n != intermediate || up_desc.k != d_model)
            {
                LOG_DEBUG("[MoEExpertComputeStage] Unable to export grouped gate/up descriptor for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }

            gate_descs[static_cast<size_t>(expert_id)] = gate_desc;
            up_descs[static_cast<size_t>(expert_id)] = up_desc;
            ++valid_descs;
        }

        if (valid_descs == 0)
            return false;

        const int table_id = kernel->uploadGroupedExpertGateUpDescriptorTables(
            gate_descs.data(), up_descs.data(), params_.num_experts, d_model, intermediate);
        if (table_id < 0)
            return false;

        grouped_gateup_desc_table_id_ = table_id;
        grouped_gateup_desc_table_num_experts_ = params_.num_experts;
        grouped_gateup_desc_table_d_model_ = d_model;
        grouped_gateup_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool MoEExpertComputeStage::ensureGroupedDownDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || params_.num_experts <= 0 || d_model <= 0 || intermediate <= 0)
            return false;

        if (grouped_down_desc_table_id_ >= 0 &&
            grouped_down_desc_table_num_experts_ == params_.num_experts &&
            grouped_down_desc_table_d_model_ == d_model &&
            grouped_down_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        std::vector<DeviceNativeVNNIMatrixDesc> down_descs(static_cast<size_t>(params_.num_experts));
        int valid_descs = 0;
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (cached_down_gemm_.size() <= static_cast<size_t>(expert_id) ||
                !cached_down_gemm_[static_cast<size_t>(expert_id)])
            {
                continue;
            }

            DeviceNativeVNNIMatrixDesc desc;
            if (!cached_down_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(desc) ||
                desc.n != d_model || desc.k != intermediate)
            {
                LOG_DEBUG("[MoEExpertComputeStage] Unable to export grouped down descriptor for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }

            down_descs[static_cast<size_t>(expert_id)] = desc;
            ++valid_descs;
        }

        if (valid_descs == 0)
            return false;

        const int table_id = kernel->uploadGroupedExpertDownDescriptorTable(
            down_descs.data(), params_.num_experts, d_model, intermediate);
        if (table_id < 0)
            return false;

        grouped_down_desc_table_id_ = table_id;
        grouped_down_desc_table_num_experts_ = params_.num_experts;
        grouped_down_desc_table_d_model_ = d_model;
        grouped_down_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool MoEExpertComputeStage::initializeMoERuntimeTableForGroupedDecode()
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 ||
            params_.num_experts <= 0 || params_.top_k <= 0 ||
            !params_.device_id.is_rocm())
        {
            return false;
        }

        try
        {
            if (!moe_runtime_layer_)
                moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
            if (runtimeTableHasActiveGroupedDecodeBank())
                return true;

            const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            if (state.active_epoch != 0)
            {
                LOG_ERROR("[MoEExpertComputeStage] Invalid active MoE runtime decode bank for layer "
                          << params_.layer_idx << "; refusing to overwrite non-zero epoch "
                          << state.active_epoch);
                return false;
            }

            if (!hasFullLocalExpertOwnership() || !expertMaskAllEnabled())
                return false;

            if (static_cast<int>(all_expert_ids_.size()) != params_.num_experts)
            {
                all_expert_ids_.resize(static_cast<size_t>(params_.num_experts));
                std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
            }
            if (!ensureGemmEnginesForExperts(all_expert_ids_))
                return false;

            MoEPlacementUpdate update;
            update.epoch = 1;
            update.expert_count = static_cast<uint32_t>(params_.num_experts);
            update.experts.resize(static_cast<size_t>(params_.num_experts));
            update.local_compute_mask.assign(static_cast<size_t>(params_.num_experts), 1u);
            update.replica_role.assign(static_cast<size_t>(params_.num_experts),
                                       static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));

            const uint32_t local_flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                          DeviceMoEExpertFlags::Resident |
                                                          DeviceMoEExpertFlags::PreferredOwner |
                                                          DeviceMoEExpertFlags::LocalCompute);

            for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
            {
                DeviceMoEExpertDescriptor desc;
                if (!cached_gate_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(desc.gate) ||
                    !cached_up_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(desc.up) ||
                    !cached_down_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(desc.down))
                {
                    LOG_DEBUG("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                              << params_.layer_idx << " expert " << expert_id
                              << ": prepared GEMM engines did not export native-VNNI descriptors");
                    return false;
                }

                desc.logical_expert_id = expert_id;
                desc.owner_participant = 0;
                desc.local_slot = expert_id;
                desc.flags = local_flags;
                update.experts[static_cast<size_t>(expert_id)] = desc;
            }

            params_.moe_runtime_table->prepareInactiveBank(params_.layer_idx, update);
            params_.moe_runtime_table->flipActiveBank(params_.layer_idx, update.epoch, gpuStream());
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
            return runtimeTableHasActiveGroupedDecodeBank();
        }
        catch (const std::exception &ex)
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to initialize MoE runtime decode bank for layer "
                      << params_.layer_idx << ": " << ex.what());
            return false;
        }
    }

    bool MoEExpertComputeStage::initializeMoERuntimeTableForGroupedPrefill()
    {
        return false;
    }

    bool MoEExpertComputeStage::initializeFixedTopologyGroupedPrefill()
    {
        return false;
    }

    bool MoEExpertComputeStage::runtimeTableHasActiveGroupedDecodeBank() const
    {
        const DeviceMoEPlacementBank *bank = activeRuntimePlacementBank();
        if (!bank || !moe_runtime_layer_)
            return false;

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            const auto mask = bank->local_compute_mask[static_cast<size_t>(expert_id)];
            if (mask != 1u)
                return false;

            const auto &expert = bank->experts[static_cast<size_t>(expert_id)];
            if (expert.logical_expert_id != expert_id ||
                expert.local_slot < 0 ||
                !expert.gate.valid() ||
                !expert.up.valid() ||
                !expert.down.valid() ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Valid) ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Resident) ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::LocalCompute))
            {
                return false;
            }
        }
        return true;
    }

    bool MoEExpertComputeStage::canUseRuntimePrefillGrouping() const
    {
        return false;
    }

    bool MoEExpertComputeStage::canUseFixedTopologyGroupedPrefill() const
    {
        return params_.device_id.is_rocm() &&
               debugEnv().rocm.moe_grouped_prefill &&
               params_.seq_len > 1 &&
               hasFullLocalExpertOwnership() &&
               expertMaskAllEnabled() &&
               params_.replica_set.num_replicated == 0;
    }

    bool MoEExpertComputeStage::executeFixedTopologyGroupedPrefill(IMoEKernel *kernel, int max_tokens) const
    {
        (void)max_tokens;
        if (!kernel)
            return false;

        const int seq_len = params_.seq_len;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int d_model = params_.d_model;
        const int intermediate = params_.expert_intermediate;

        // Async grouping (no D2H, no sync)
        if (!kernel->prepareExpertGroupsAsync(
                params_.routing_indices, params_.routing_weights,
                seq_len, num_experts, top_k))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      "prepareExpertGroupsAsync failed");
            return false;
        }

        // Execute the full grouped pipeline (5 kernel launches, zero sync)
        if (!kernel->executeGroupedPrefillPipeline(
                params_.input, params_.output,
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                seq_len, d_model, intermediate,
                num_experts, top_k))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      "grouped prefill pipeline failed");
            return false;
        }

        return true;
    }

    bool MoEExpertComputeStage::isDeviceRoutedDecodeGraphCapturable() const
    {
        return false;
    }

    bool MoEExpertComputeStage::isFixedTopologyPrefillGraphCapturable() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || !defined(HAVE_ROCM)
        return false;
#else
        // Fixed-topology grouped prefill is capturable when all buffers are
        // pre-allocated from warmup and the stage uses the grouped pipeline.
        // The MoE kernel must have sufficient grouping + scratch capacity.
        if (!params_.device_id.is_rocm() || params_.seq_len <= 1)
            return false;

        const auto &rocm = debugEnv().rocm;
        if (!rocm.moe_grouped_prefill)
            return false;

        // Require full local expert ownership, no masks, no replicas
        if (!hasFullLocalExpertOwnership() || !expertMaskAllEnabled())
            return false;
        if (params_.replica_set.num_replicated != 0)
            return false;

        // Must have all prepared GEMM engines ready
        if (!hasAllPreparedExpertGemmEngines())
            return false;

        // MoE kernel must exist and have pre-allocated grouping + scratch
        if (!moe_kernel_)
            return false;

        return true;
#endif
    }

    bool MoEExpertComputeStage::hasFullLocalExpertOwnership() const
    {
        const int local_count = params_.local_expert_count < 0 ? params_.num_experts : params_.local_expert_count;
        return params_.local_expert_start == 0 && local_count == params_.num_experts;
    }

    bool MoEExpertComputeStage::expertMaskAllEnabled() const
    {
        return params_.expert_mask.empty() ||
               (params_.expert_mask.size() == static_cast<size_t>(params_.num_experts) &&
                std::all_of(params_.expert_mask.begin(), params_.expert_mask.end(),
                            [](bool enabled)
                            { return enabled; }));
    }

    bool MoEExpertComputeStage::hasAllPreparedExpertGemmEngines() const
    {
        if (params_.prepared_gate_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_up_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_down_gemm.size() != static_cast<size_t>(params_.num_experts))
        {
            return false;
        }
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (!params_.prepared_gate_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_up_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_down_gemm[static_cast<size_t>(expert_id)])
            {
                return false;
            }
        }
        return true;
    }

    bool MoEExpertComputeStage::hasGroupedDecodeDescriptorExportSupport() const
    {
        if (!hasAllPreparedExpertGemmEngines())
            return false;

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            DeviceNativeVNNIMatrixDesc gate;
            DeviceNativeVNNIMatrixDesc up;
            DeviceNativeVNNIMatrixDesc down;
            if (!params_.prepared_gate_gemm[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(gate) ||
                !params_.prepared_up_gemm[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(up) ||
                !params_.prepared_down_gemm[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(down) ||
                gate.n != params_.expert_intermediate || gate.k != params_.d_model ||
                up.n != params_.expert_intermediate || up.k != params_.d_model ||
                down.n != params_.d_model || down.k != params_.expert_intermediate)
            {
                return false;
            }
        }
        return true;
    }

    const DeviceMoEPlacementBank *MoEExpertComputeStage::activeRuntimePlacementBank() const
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 || params_.num_experts <= 0)
            return nullptr;

        const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
        if (state.active_bank > 1 ||
            state.active_epoch == 0 ||
            state.expert_count != static_cast<uint32_t>(params_.num_experts) ||
            state.top_k != static_cast<uint32_t>(params_.top_k))
        {
            return nullptr;
        }

        const auto &bank = state.banks[state.active_bank];
        if (bank.epoch != state.active_epoch ||
            bank.expert_count != static_cast<uint32_t>(params_.num_experts))
        {
            return nullptr;
        }
        return &bank;
    }

    bool MoEExpertComputeStage::runtimeLocalComputeEnabled(const DeviceMoEPlacementBank *bank, int expert_id) const
    {
        if (!bank || expert_id < 0 || expert_id >= params_.num_experts)
            return false;
        return bank->local_compute_mask[static_cast<size_t>(expert_id)] != 0u;
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
            params.moe_packed_down_lifetime};
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
            params.moe_packed_down_lifetime,
            nullptr,
            params.prepared_store,
            params.expert_registry,
            params.gate_slab_ref,
            params.up_slab_ref,
            params.down_slab_ref};
        bool ok = MoEExpertWeightService::prepareGemmEngines(ctx);
        // Phase C: Copy slab refs back to params for rebalance reuse
        params.gate_slab_ref = ctx.gate_slab_ref;
        params.up_slab_ref = ctx.up_slab_ref;
        params.down_slab_ref = ctx.down_slab_ref;
        return ok;
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

    bool MoEExpertComputeStage::isGraphCapturable() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || !defined(HAVE_ROCM)
        return false;
#else
        // Device-routed grouped decode path: after warmup, all descriptor tables
        // are built and execution is pure kernel launches reading routing info
        // from the device-resident MoE runtime table.
        const auto &rocm = debugEnv().rocm;
        if (params_.device_id.is_rocm() &&
            params_.seq_len == 1 &&
            rocm.moe_grouped_decode &&
            rocm.moe_device_routed_decode &&
            params_.moe_runtime_table != nullptr &&
            params_.top_k > 0 && params_.top_k <= 16 &&
            params_.replica_set.num_replicated == 0 &&
            hasFullLocalExpertOwnership() &&
            expertMaskAllEnabled())
            return true;

        // Fixed-topology grouped prefill path
        return isFixedTopologyPrefillGraphCapturable();
#endif
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

    StageBufferContract MoEExpertComputeStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        contract.addInput(params_.routing_indices_buffer_id);
        contract.addInput(params_.routing_weights_buffer_id);
        if (params_.output_registered_in_arena)
            contract.addOutput(params_.output_buffer_id);

        return contract;
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
    // MoEExpertComputeStage — IWorkspaceConsumer Implementation
    // =========================================================================

    WorkspaceRequirements MoEExpertComputeStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        // All expert GEMM engines use shared buffer names (not per-instance),
        // so requirements from any one engine represent all of them.
        const auto &engines = params_.prepared_gate_gemm.empty()
                                  ? cached_gate_gemm_
                                  : params_.prepared_gate_gemm;
        for (auto *gemm : engines)
        {
            if (!gemm)
                continue;
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
            if (consumer)
                return consumer->getWorkspaceRequirements(m, n, k);
        }
        return WorkspaceRequirements{};
    }

    void MoEExpertComputeStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        // Bind workspace to ALL expert GEMM engines (gate, up, down for each expert)
        auto bindAll = [workspace](const std::vector<ITensorGemm *> &engines)
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    consumer->bindWorkspace(workspace);
            }
        };

        const auto &gate = params_.prepared_gate_gemm.empty() ? cached_gate_gemm_ : params_.prepared_gate_gemm;
        const auto &up = params_.prepared_up_gemm.empty() ? cached_up_gemm_ : params_.prepared_up_gemm;
        const auto &down = params_.prepared_down_gemm.empty() ? cached_down_gemm_ : params_.prepared_down_gemm;

        bindAll(gate);
        bindAll(up);
        bindAll(down);

        bound_workspace_ = workspace;
        LOG_DEBUG("[MoEExpertComputeStage] Bound workspace to "
                  << gate.size() + up.size() + down.size() << " expert GEMM engines");
    }

    void MoEExpertComputeStage::unbindWorkspace()
    {
        auto unbindAll = [](const std::vector<ITensorGemm *> &engines)
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    consumer->unbindWorkspace();
            }
        };

        const auto &gate = params_.prepared_gate_gemm.empty() ? cached_gate_gemm_ : params_.prepared_gate_gemm;
        const auto &up = params_.prepared_up_gemm.empty() ? cached_up_gemm_ : params_.prepared_up_gemm;
        const auto &down = params_.prepared_down_gemm.empty() ? cached_down_gemm_ : params_.prepared_down_gemm;

        unbindAll(gate);
        unbindAll(up);
        unbindAll(down);

        bound_workspace_ = nullptr;
    }

    bool MoEExpertComputeStage::hasWorkspace() const
    {
        return bound_workspace_ != nullptr;
    }

    DeviceWorkspaceManager *MoEExpertComputeStage::getWorkspace() const
    {
        return bound_workspace_;
    }

    // =========================================================================
    // SharedExpertFFNStage — Dense SwiGLU on shared expert
    // =========================================================================

    SharedExpertFFNStage::SharedExpertFFNStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool SharedExpertFFNStage::validatePreparedWeights(std::string *error) const
    {
        auto fail = [error](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };

        if (!params_.gate_w && !params_.up_w && !params_.down_w)
        {
            if (error)
                error->clear();
            return true;
        }

        if (!params_.prepared_store)
            return fail("PreparedWeightStore is required for SharedExpertFFNStage weights");

        auto check = [&](const char *name, const TensorBase *weight, const std::optional<PreparedWeightRef> &ref)
        {
            if (!weight)
                return true;
            if (!ref.has_value())
                return fail(std::string("missing PreparedWeightRef for ") + name);
            if (!params_.prepared_store->contains(ref.value()))
                return fail(std::string("PreparedWeightStore does not contain ref for ") + name);
            return true;
        };

        if (!check("gate_w", params_.gate_w, params_.prepared_ref_gate))
            return false;
        if (!check("up_w", params_.up_w, params_.prepared_ref_up))
            return false;
        if (!check("down_w", params_.down_w, params_.prepared_ref_down))
            return false;

        if (error)
            error->clear();
        return true;
    }

    void SharedExpertFFNStage::ensureGemmEnginesCached() const
    {
        if (cached_gate_gemm_)
            return;

        if (!params_.prepared_store ||
            !params_.prepared_ref_gate.has_value() ||
            !params_.prepared_ref_up.has_value() ||
            !params_.prepared_ref_down.has_value())
        {
            LOG_ERROR("[SharedExpertFFNStage] PreparedWeightStore and gate/up/down PreparedWeightRefs are required");
            return;
        }

        cached_gate_gemm_ = params_.prepared_store->gemmKernel(params_.prepared_ref_gate.value());
        cached_up_gemm_ = params_.prepared_store->gemmKernel(params_.prepared_ref_up.value());
        cached_down_gemm_ = params_.prepared_store->gemmKernel(params_.prepared_ref_down.value());
        if (!cached_gate_gemm_ || !cached_up_gemm_ || !cached_down_gemm_)
        {
            LOG_ERROR("[SharedExpertFFNStage] PreparedWeightRefs were provided but shared expert kernel(s) were missing from PreparedWeightStore. "
                      "gate="
                      << (void *)cached_gate_gemm_ << " up=" << (void *)cached_up_gemm_
                      << " down=" << (void *)cached_down_gemm_);
            return;
        }

        auto bind_if_needed = [this](ITensorGemm *gemm)
        {
            if (!gemm)
                return;
            gemm->setGPUStream(gpuStream());
            if (bound_workspace_)
            {
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer && !consumer->hasWorkspace())
                    consumer->bindWorkspace(bound_workspace_);
            }
        };
        bind_if_needed(cached_gate_gemm_);
        bind_if_needed(cached_up_gemm_);
        bind_if_needed(cached_down_gemm_);
    }

    bool SharedExpertFFNStage::ensureSharedGroupedGateUpDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        if (!kernel || !cached_gate_gemm_ || !cached_up_gemm_ || d_model <= 0 || intermediate <= 0)
            return false;

        if (shared_grouped_gateup_desc_table_id_ >= 0 &&
            shared_grouped_gateup_desc_table_d_model_ == d_model &&
            shared_grouped_gateup_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        DeviceNativeVNNIMatrixDesc gate_desc;
        DeviceNativeVNNIMatrixDesc up_desc;
        if (!cached_gate_gemm_->exportNativeVNNIMatrixDesc(gate_desc) ||
            !cached_up_gemm_->exportNativeVNNIMatrixDesc(up_desc) ||
            gate_desc.n != intermediate || gate_desc.k != d_model ||
            up_desc.n != intermediate || up_desc.k != d_model)
        {
            return false;
        }

        const int table_id = kernel->uploadGroupedExpertGateUpDescriptorTables(
            &gate_desc, &up_desc, 1, d_model, intermediate);
        if (table_id < 0)
            return false;

        shared_grouped_gateup_desc_table_id_ = table_id;
        shared_grouped_gateup_desc_table_d_model_ = d_model;
        shared_grouped_gateup_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool SharedExpertFFNStage::ensureSharedGroupedDownDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        if (!kernel || !cached_down_gemm_ || d_model <= 0 || intermediate <= 0)
            return false;

        if (shared_grouped_down_desc_table_id_ >= 0 &&
            shared_grouped_down_desc_table_d_model_ == d_model &&
            shared_grouped_down_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        DeviceNativeVNNIMatrixDesc down_desc;
        if (!cached_down_gemm_->exportNativeVNNIMatrixDesc(down_desc) ||
            down_desc.n != d_model || down_desc.k != intermediate)
        {
            return false;
        }

        const int table_id = kernel->uploadGroupedExpertDownDescriptorTable(
            &down_desc, 1, d_model, intermediate);
        if (table_id < 0)
            return false;

        shared_grouped_down_desc_table_id_ = table_id;
        shared_grouped_down_desc_table_d_model_ = d_model;
        shared_grouped_down_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool SharedExpertFFNStage::tryGroupedDecode(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        if (!params_.device_id.is_rocm() || params_.seq_len != 1 ||
            !debugEnv().rocm.shared_expert_grouped_decode)
        {
            return false;
        }

        if (!ensureSharedGroupedGateUpDescriptorTable(kernel, d_model, intermediate) ||
            !ensureSharedGroupedDownDescriptorTable(kernel, d_model, intermediate))
        {
            return false;
        }

        constexpr int expert_id = 0;
        constexpr float expert_weight = 1.0f;
        ITensor *gate_outputs[1] = {scratch_gate_.get()};
        ITensor *up_outputs[1] = {scratch_up_.get()};
        if (!kernel->groupedExpertGateUpDecodeFromTable(
                params_.input,
                &expert_id,
                shared_grouped_gateup_desc_table_id_,
                1,
                gate_outputs,
                up_outputs,
                d_model,
                intermediate))
        {
            return false;
        }

        ITensor *gate_tensors[1] = {scratch_gate_.get()};
        ITensor *up_tensors[1] = {scratch_up_.get()};
        return kernel->groupedExpertDownDecodeFromTable(
            gate_tensors,
            up_tensors,
            &expert_id,
            &expert_weight,
            shared_grouped_down_desc_table_id_,
            1,
            params_.output,
            d_model,
            intermediate);
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
        if (!cached_gate_gemm_ || !cached_up_gemm_ || !cached_down_gemm_)
        {
            LOG_ERROR("[SharedExpertFFNStage] Missing shared expert GEMM engine");
            return false;
        }
        cached_gate_gemm_->setGPUStream(gpuStream());
        cached_up_gemm_->setGPUStream(gpuStream());
        cached_down_gemm_->setGPUStream(gpuStream());

        // Ensure scratch buffers are large enough
        if (seq_len > scratch_seq_len_)
        {
            scratch_gate_ = makeScratchFP32(seq_len, intermediate, params_.device_id);
            scratch_up_ = makeScratchFP32(seq_len, intermediate, params_.device_id);
            scratch_seq_len_ = seq_len;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (tryGroupedDecode(kernel, d_model, intermediate))
            return true;

        // Gate+Up projections via fused multi-projection (quantizes input once)
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_.get(), intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_.get(), intermediate, nullptr, "shared_up"}};
        if (!cached_gate_gemm_->multiply_fused_tensor(
                params_.input, projections,
                seq_len, d_model))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared gate/up projection failed");
            return false;
        }
        for (const auto &projection : projections)
            markGpuTensorWritten(projection.output, params_.device_id, gpuStream());

        // SwiGLU+Down via fused kernel with MoE kernel fallback
        if (!fusedSwigluDown(
                scratch_gate_.get(), scratch_up_.get(), params_.output,
                cached_down_gemm_, kernel, seq_len, d_model, intermediate,
                params_.device_id, gpuStream()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared SwiGLU/down projection failed");
            return false;
        }

        return true;
    }

    IMoEKernel *SharedExpertFFNStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return bindStageStream(moe_kernel_);
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

    bool SharedExpertFFNStage::isGraphCapturable() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || !defined(HAVE_ROCM)
        return false;
#else
        // After warmup: GEMM engines cached, scratch buffers allocated,
        // all operations are pure kernel launches (fused gate+up, swiglu, down).
        // Capturable for both decode and prefill once scratch is pre-sized.
        if (!params_.device_id.is_rocm())
            return false;
        // Decode: always ready after warmup (scratch not used for seq_len==1 grouped path)
        if (params_.seq_len == 1)
            return true;
        // Prefill: scratch must be pre-allocated to at least current seq_len
        return scratch_seq_len_ >= params_.seq_len && moe_kernel_ != nullptr;
#endif
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

    StageBufferContract SharedExpertFFNStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        contract.addOutput(params_.output_buffer_id);

        // Weights are model weights, not arena-managed
        if (params_.gate_w)
            contract.addWeight(params_.gate_w);
        if (params_.up_w)
            contract.addWeight(params_.up_w);
        if (params_.down_w)
            contract.addWeight(params_.down_w);

        return contract;
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
    // SharedExpertFFNStage — IWorkspaceConsumer Implementation
    // =========================================================================

    WorkspaceRequirements SharedExpertFFNStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        auto *self = const_cast<SharedExpertFFNStage *>(this);
        self->ensureGemmEnginesCached();

        WorkspaceRequirements combined;
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            combined.merge(c->getWorkspaceRequirements(m, n, k));
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            combined.merge(c->getWorkspaceRequirements(m, n, k));
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
            combined.merge(c->getWorkspaceRequirements(m, n, k));
        return combined;
    }

    void SharedExpertFFNStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        ensureGemmEnginesCached();

        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            c->bindWorkspace(workspace);
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            c->bindWorkspace(workspace);
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
            c->bindWorkspace(workspace);

        bound_workspace_ = workspace;
        LOG_DEBUG("[SharedExpertFFNStage] Bound workspace to gate/up/down GEMM engines");
    }

    void SharedExpertFFNStage::unbindWorkspace()
    {
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            c->unbindWorkspace();
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            c->unbindWorkspace();
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
            c->unbindWorkspace();

        bound_workspace_ = nullptr;
    }

    bool SharedExpertFFNStage::hasWorkspace() const
    {
        return bound_workspace_ != nullptr;
    }

    DeviceWorkspaceManager *SharedExpertFFNStage::getWorkspace() const
    {
        return bound_workspace_;
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

        // Delegate sigmoid gating to device-appropriate MoE kernel.
        // Tensor-aware API handles CPU/GPU dispatch internally.
        IMoEKernel *kernel = ensureMoEKernel();

        kernel->sharedExpertGateFromTensors(
            params_.input, params_.gate_inp, params_.shared_output,
            seq_len, d_model);

        if (params_.device_id.is_gpu() && params_.shared_output->needsUpload())
        {
            if (!params_.shared_output->ensureOnDevice(params_.device_id, gpuStream()))
            {
                LOG_ERROR("[SharedExpertGateStage] Failed to upload gated shared expert output to "
                          << params_.device_id.to_string());
                return false;
            }
        }

        return true;
    }

    IMoEKernel *SharedExpertGateStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        return bindStageStream(moe_kernel_);
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

    bool SharedExpertGateStage::isGraphCapturable() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || !defined(HAVE_ROCM)
        return false;
#else
        // Single kernel call (sigmoid gating) — pure kernel launch after warmup.
        // Capturable for both decode (seq_len==1) and prefill (seq_len>1) on ROCm
        // because the stage is just a kernel launch with stable device pointers.
        // MoE kernel must already be cached (from warmup execution).
        return params_.device_id.is_rocm() && moe_kernel_ != nullptr;
#endif
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

    StageBufferContract SharedExpertGateStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        contract.addInOut(params_.output_buffer_id);

        // Gate vector is a model weight, not arena-managed
        if (params_.gate_inp)
            contract.addWeight(params_.gate_inp);

        return contract;
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
