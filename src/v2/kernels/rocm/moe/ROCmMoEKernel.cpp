/**
 * @file ROCmMoEKernel.cpp
 * @brief ROCm MoE kernel implementation — calls extern "C" HIP bridges
 *
 * Separating .hip and .cpp allows hipcc to compile only the HIP code
 * without encountering issues with MPI or other complex C++ headers.
 */

#include "ROCmMoEKernel.h"
#include "../gemm/HipBLASGemmKernel.h"
#include "../../../execution/moe/MoERuntimeTable.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/DeviceId.h"
#include "../../../tensors/ITensor.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/ROCmKernelProfiler.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
    constexpr size_t kGroupedDecodeGateLogitsSharedCapBytes = 48 * 1024;

    bool setMoEDevice(int device_ordinal, const char *context)
    {
        hipError_t err = hipSetDevice(device_ordinal);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] hipSetDevice(" << device_ordinal
                                                      << ") failed in " << context << ": " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    /// @brief Returns whether grouped MoE kernels instantiate this native-VNNI codebook.
    bool groupedDecodeSupportsCodebook(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:  // Q4_0
        case 4:  // IQ4_NL / IQ4_XS
        case 5:  // Q4_1 / Q4_K
        case 6:  // Q5_0
        case 7:  // Q5_1 / Q5_K
        case 8:  // Q6_K
        case 9:  // Q3_K
        case 10: // Q2_K
        case 11: // IQ3_S
        case 12: // IQ3_XXS
        case 13: // IQ2_S
        case 14: // IQ2_XS
        case 15: // IQ2_XXS
        case 16: // IQ1_S
        case 17: // IQ1_M
        case 19: // Q8_0
            return true;
        default:
            return false;
        }
    }

    /// @brief Returns whether the descriptor's mins pointer is required for this format.
    bool groupedDecodeRequiresMins(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 5:  // Q4_1 / Q4_K min correction
        case 7:  // Q5_1 / Q5_K min correction
        case 8:  // Q6_K high half scale
        case 9:  // Q3_K high half scale
        case 10: // Q2_K high half scale
        case 13: // IQ2_S high half scale
        case 14: // IQ2_XS high half scale
        case 16: // IQ1_S min correction
        case 17: // IQ1_M high half scale
            return true;
        default:
            return false;
        }
    }

    /// @brief Returns whether the descriptor's emins pointer is required for this format.
    bool groupedDecodeRequiresEmins(uint8_t codebook_id)
    {
        return codebook_id == 10; // Q2_K stores embedded min correction separately.
    }

    bool validateGroupedDownDesc(
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc,
        int d_model,
        int intermediate,
        uint8_t codebook_id)
    {
        return desc.valid() && desc.n == d_model && desc.k == intermediate &&
               desc.blocks_per_row == static_cast<uint32_t>(intermediate / 32) &&
               desc.codebook_id == codebook_id &&
               (!groupedDecodeRequiresMins(desc.codebook_id) || desc.mins) &&
               (!groupedDecodeRequiresEmins(desc.codebook_id) || desc.emins);
    }

    bool validateGroupedGateUpDesc(
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc,
        int d_model,
        int intermediate,
        uint8_t codebook_id)
    {
        return desc.valid() && desc.n == intermediate && desc.k == d_model &&
               desc.blocks_per_row == static_cast<uint32_t>(d_model / 32) &&
               desc.codebook_id == codebook_id &&
               (!groupedDecodeRequiresMins(desc.codebook_id) || desc.mins) &&
               (!groupedDecodeRequiresEmins(desc.codebook_id) || desc.emins);
    }

    const int *runtimeTopKExpertIdsDevice(const llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        const auto *base = reinterpret_cast<const char *>(runtime_layer);
        return reinterpret_cast<const int *>(
            base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_expert_ids));
    }

    const float *runtimeTopKWeightsDevice(const llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        const auto *base = reinterpret_cast<const char *>(runtime_layer);
        return reinterpret_cast<const float *>(
            base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_weights));
    }
}

// Forward-declare extern "C" bridge functions (defined in ROCmMoEKernels.hip)
extern "C"
{
    bool hipMoE_gate_logits_single_token(
        const float *hidden, const float *gate_weights, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_small_m(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_kpart(
        const float *hidden, const float *gate_weights, float *logits,
        float *partials,
        int d_model, int num_experts, int k_partitions,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_kpart_partials(
        const float *hidden, const float *gate_weights, float *partials,
        int d_model, int num_experts, int k_partitions,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_grouped(
        const float *hidden, const float *gate_weights, float *logits,
        int d_model, int num_experts, size_t shared_mem_bytes,
        int device_idx, void *stream);

    bool hipMoE_fp32_to_fp16(
        const float *input, void *output_fp16, int count,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_fp16_weights(
        const float *hidden, const void *gate_weights_fp16, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_bf16_weights(
        const float *hidden, const void *gate_weights_bf16, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_quantize_router_gate_q8(
        const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_gate_logits_single_token_q8_weights(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_softmax_topk(
        float *logits,
        int *expert_indices, float *expert_weights,
        int seq_len, int num_experts, int top_k,
        bool normalize_weights,
        int device_idx, void *stream);

    bool hipMoE_softmax_topk_decode_runtime(
        float *logits,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        int device_idx, void *stream);

    bool hipMoE_softmax_topk_decode_runtime_wave64(
        const float *logits,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        int device_idx, void *stream);

    bool hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
        const float *partials,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int k_partitions, int top_k,
        bool normalize_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        int device_idx, void *stream);

    bool hipMoE_decode_route_select_runtime(
        const int *expert_indices,
        const float *expert_weights,
        void *runtime,
        float *legacy_indices,
        float *legacy_weights,
        int num_experts, int top_k,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        int device_idx, void *stream);

    bool hipMoE_gather_tokens(
        const float *hidden, float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_scatter_add(
        float *output, const float *expert_output,
        const int *token_indices, const float *weights,
        int num_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_shared_expert_gate(
        const float *input, const float *gate_inp,
        float *shared_output, float *gate_scratch,
        int seq_len, int d_model,
        int device_idx, void *stream);

    bool hipMoE_shared_expert_gate_decode_fused(
        const float *input, const float *gate_inp,
        float *shared_output, int d_model,
        int device_idx, void *stream);

    bool hipMoE_swiglu(
        float *gate, const float *up,
        int count,
        int device_idx, void *stream);

    bool hipMoE_weighted_add(
        float *output, const float *input,
        float weight, int count,
        int device_idx, void *stream);

    // Phase 2: Histogram + Expert Mask bridges
    bool hipMoE_histogram_record(
        const int *routing_indices, unsigned long long *histogram,
        int seq_len, int num_experts, int top_k, int layer_idx,
        int device_idx, void *stream);

    bool hipMoE_apply_expert_mask(
        float *routing_weights, const int *routing_indices,
        const bool *expert_mask,
        int seq_len, int top_k,
        int device_idx, void *stream);

    bool hipMoE_histogram_reset(
        unsigned long long *histogram, int layer_idx, int num_experts,
        int device_idx, void *stream);

    // Phase 3: Token grouping bridges
    bool hipMoE_count_per_expert(
        const int *routing_indices, int *expert_counts,
        int total_slots, int num_experts,
        int device_idx, void *stream);

    bool hipMoE_exclusive_scan(
        int *expert_counts, int *expert_offsets,
        int num_experts,
        int device_idx, void *stream);

    bool hipMoE_max_expert_count(
        const int *expert_counts, int *d_max_out,
        int num_experts,
        int device_idx, void *stream);

    bool hipMoE_scatter_tokens(
        const int *routing_indices, const float *routing_weights,
        const int *expert_offsets, int *write_heads,
        int *grouped_token_indices, float *grouped_weights,
        int total_slots, int num_experts, int top_k,
        int device_idx, void *stream);

    bool hipMoE_group_tokens_small_float(
        const float *routing_indices, const float *routing_weights,
        int *expert_counts, int *expert_offsets,
        int *grouped_token_indices, float *grouped_weights,
        int *active_expert_ids,
        int total_slots, int num_experts, int top_k,
        int max_active_experts,
        int device_idx, void *stream);

    // Phase 4: Tensor-aware utility bridges
    bool hipMoE_int_to_float(
        const int *d_input, float *d_output, int count,
        int device_idx, void *stream);

    bool hipMoE_float_to_int(
        const float *d_input, int *d_output, int count,
        int device_idx, void *stream);

    bool hipMoE_group_prefill_routes_runtime(
        const float *routing_indices, const float *routing_weights,
        void *runtime,
        int current_slots, int max_slots, int num_experts, int top_k,
        int device_idx, void *stream);

    bool hipMoE_prefill_gather_expert_runtime(
        const void *runtime,
        const float *hidden,
        float *batch_buffer,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool hipMoE_prefill_scatter_expert_runtime(
        float *output,
        const float *expert_output,
        const void *runtime,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool rocmMoE_grouped_swiglu_down_native_vnni_decode(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_descs,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool rocmMoE_grouped_gate_up_native_vnni_decode_table(
        const float *d_hidden,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart(
        const float *d_hidden,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream);

    bool rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int N,
        int K,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool rocmMoE_grouped_prefill_pipeline(
        const float *d_hidden,
        const void *d_gate_desc_table,
        const void *d_up_desc_table,
        const void *d_down_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        const int *d_group_token_indices,
        const float *d_group_weights,
        const int *d_active_expert_ids,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_scratch_down_out,
        float *d_output,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int active_expert_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        int device_id,
        void *stream);
}

namespace
{
    bool tryGroupedDecodeGateLogits(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream)
    {
        if (!llaminar2::debugEnv().rocm.moe_grouped_decode_router)
            return false;

        const size_t shared_mem_bytes = static_cast<size_t>(d_model) * sizeof(float);
        if (num_experts <= 1 || shared_mem_bytes > kGroupedDecodeGateLogitsSharedCapBytes)
            return false;

        return hipMoE_gate_logits_single_token_grouped(
            hidden, gate_weights, logits,
            d_model, num_experts, shared_mem_bytes,
            device_ordinal, stream);
    }

    bool launchDecodeGateLogits(
        const float *hidden,
        const float *gate_weights,
        float *logits,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream,
        const char *context)
    {
        if (tryGroupedDecodeGateLogits(hidden, gate_weights, logits,
                                       d_model, num_experts,
                                       device_ordinal, stream))
        {
            return true;
        }

        if (!hipMoE_gate_logits_single_token(hidden, gate_weights, logits,
                                             d_model, num_experts,
                                             device_ordinal, stream))
        {
            LOG_ERROR("[" << context << "] single-token gate logits kernel failed");
            return false;
        }
        return true;
    }

    bool launchDecodeGateLogitsForGateType(
        const float *hidden,
        const void *gate_weights,
        llaminar2::TensorType gate_type,
        float *logits,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream,
        const char *context)
    {
        switch (gate_type)
        {
        case llaminar2::TensorType::FP32:
            return launchDecodeGateLogits(
                hidden, static_cast<const float *>(gate_weights), logits,
                d_model, num_experts, device_ordinal, stream, context);
        case llaminar2::TensorType::FP16:
            if (!hipMoE_gate_logits_single_token_fp16_weights(
                    hidden, gate_weights, logits,
                    d_model, num_experts, device_ordinal, stream))
            {
                LOG_ERROR("[" << context << "] FP16 gate logits kernel failed");
                return false;
            }
            return true;
        case llaminar2::TensorType::BF16:
            if (!hipMoE_gate_logits_single_token_bf16_weights(
                    hidden, gate_weights, logits,
                    d_model, num_experts, device_ordinal, stream))
            {
                LOG_ERROR("[" << context << "] BF16 gate logits kernel failed");
                return false;
            }
            return true;
        default:
            LOG_ERROR("[" << context << "] unsupported router gate dtype "
                          << llaminar2::tensorTypeName(gate_type));
            return false;
        }
    }

    bool launchSmallMGateLogits(
        const float *hidden,
        const void *gate_weights,
        llaminar2::TensorType gate_type,
        float *logits,
        int seq_len,
        int d_model,
        int num_experts,
        int device_ordinal,
        void *stream)
    {
        if (seq_len < 2 || seq_len > 4 || gate_type != llaminar2::TensorType::FP32)
            return false;

        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] small-M rowwise router requires an explicit stream");
            return false;
        }

        const auto *gate_weights_fp32 = static_cast<const float *>(gate_weights);
        if (!hipMoE_gate_logits_small_m(
                hidden,
                gate_weights_fp32,
                logits,
                seq_len,
                d_model,
                num_experts,
                device_ordinal,
                stream))
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] small-M fused router kernel failed");
            return false;
        }

        return true;
    }
}

namespace llaminar2
{

    ROCmMoEKernel::ROCmMoEKernel(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_ordinal);
        ROCmKernelBase::setDeviceContext(&ctx);
        ROCmKernelBase::setGPUStream(ctx.defaultStream());

        // Create hipBLAS GEMM kernel using device context (shares hipBLAS handle)
        blas_gemm_ = std::make_unique<rocm::HipBLASGemmKernel>(&ctx);

        // Ensure hipBLAS runs on the same stream as our HIP kernels
        syncBlasStream();

        LOG_DEBUG("[ROCmMoEKernel] Created for ROCm device " << device_ordinal
                                                             << " stream=" << ROCmKernelBase::getStream());
    }

    void ROCmMoEKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        if (workspace_ == workspace)
            return;
        ROCmKernelBase::bindWorkspace(workspace);
        clearWorkspaceScratchBindings();
    }

    bool ROCmMoEKernel::bindWorkspaceBuffer(
        void **ptr,
        const char *name,
        size_t bytes,
        const char *context)
    {
        if (!ptr || !name || bytes == 0)
            return false;
        if (!validateROCmWorkspaceBinding(workspace_, device_ordinal_, "ROCmMoEKernel"))
        {
            LOG_ERROR("[ROCmMoEKernel] " << context
                                         << " requires graph-owned MoE workspace");
            return false;
        }
        void *buffer = workspace_->getBuffer(name);
        const size_t available = workspace_->getBufferSize(name);
        if (!buffer || available < bytes)
        {
            LOG_ERROR("[ROCmMoEKernel] " << context << " missing required workspace buffer '"
                                         << name << "' (need " << bytes << " bytes, have "
                                         << available << ")");
            return false;
        }
        *ptr = buffer;
        scratch_workspace_bound_ = true;
        return true;
    }

    void ROCmMoEKernel::clearWorkspaceScratchBindings() noexcept
    {
        d_write_heads_ = nullptr;
        d_staging_indices_ = nullptr;
        d_staging_weights_ = nullptr;
        d_grouped_gate_ptrs_ = nullptr;
        d_grouped_up_ptrs_ = nullptr;
        d_grouped_expert_ids_ = nullptr;
        d_grouped_decode_weights_ = nullptr;
        d_grouped_down_descs_ = nullptr;
        d_grouped_swiglu_int8_ = nullptr;
        d_grouped_swiglu_scales_ = nullptr;
        d_grouped_gate_output_ptrs_ = nullptr;
        d_grouped_up_output_ptrs_ = nullptr;
        d_grouped_gateup_expert_ids_ = nullptr;
        d_grouped_hidden_int8_ = nullptr;
        d_grouped_hidden_scales_ = nullptr;
        d_grouped_gateup_gate_partials_ = nullptr;
        d_grouped_gateup_up_partials_ = nullptr;
        d_shared_gate_scratch_ = nullptr;
        d_route_logits_ = nullptr;
        d_route_logits_partials_ = nullptr;
        d_router_q8_hidden_ = nullptr;
        d_router_q8_hidden_scales_ = nullptr;
        d_route_indices_ = nullptr;
        d_route_weights_ = nullptr;
        d_group_int_indices_ = nullptr;
        d_group_offsets_ = nullptr;
        d_group_counts_ = nullptr;
        d_group_max_tokens_ = nullptr;
        d_group_token_indices_ = nullptr;
        d_group_weights_ = nullptr;
        d_group_active_expert_ids_ = nullptr;
        d_prefill_A_int8_ = nullptr;
        d_prefill_A_scales_ = nullptr;
        d_prefill_gate_ = nullptr;
        d_prefill_up_ = nullptr;

        max_write_heads_experts_ = 0;
        staging_capacity_ = 0;
        grouped_decode_active_cap_ = 0;
        grouped_decode_intermediate_cap_ = 0;
        grouped_gateup_active_cap_ = 0;
        grouped_gateup_d_model_cap_ = 0;
        grouped_gateup_kpart_active_cap_ = 0;
        grouped_gateup_kpart_partitions_cap_ = 0;
        grouped_gateup_kpart_intermediate_cap_ = 0;
        shared_gate_scratch_capacity_ = 0;
        route_logits_capacity_ = 0;
        route_topk_capacity_ = 0;
        route_logits_partials_capacity_ = 0;
        router_q8_hidden_d_model_cap_ = 0;
        router_q8_hidden_blocks_cap_ = 0;
        group_active_expert_slots_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        scratch_workspace_bound_ = false;
    }

    ROCmMoEKernel::~ROCmMoEKernel()
    {
        (void)setMoEDevice(device_ordinal_, "destructor");
        if (scratch_workspace_bound_)
            clearWorkspaceScratchBindings();
        if (d_histogram_)
        {
            hipFree(d_histogram_);
            d_histogram_ = nullptr;
        }
        if (d_expert_mask_)
        {
            hipFree(d_expert_mask_);
            d_expert_mask_ = nullptr;
        }
        if (d_write_heads_)
        {
            hipFree(d_write_heads_);
            d_write_heads_ = nullptr;
        }
        if (d_staging_indices_)
        {
            hipFree(d_staging_indices_);
            d_staging_indices_ = nullptr;
        }
        if (d_staging_weights_)
        {
            hipFree(d_staging_weights_);
            d_staging_weights_ = nullptr;
        }
        if (d_grouped_gate_ptrs_)
        {
            hipFree(d_grouped_gate_ptrs_);
            d_grouped_gate_ptrs_ = nullptr;
        }
        if (d_grouped_up_ptrs_)
        {
            hipFree(d_grouped_up_ptrs_);
            d_grouped_up_ptrs_ = nullptr;
        }
        if (d_grouped_expert_ids_)
        {
            hipFree(d_grouped_expert_ids_);
            d_grouped_expert_ids_ = nullptr;
        }
        if (d_grouped_decode_weights_)
        {
            hipFree(d_grouped_decode_weights_);
            d_grouped_decode_weights_ = nullptr;
        }
        if (d_grouped_down_descs_)
        {
            hipFree(d_grouped_down_descs_);
            d_grouped_down_descs_ = nullptr;
        }
        if (d_grouped_swiglu_int8_)
        {
            hipFree(d_grouped_swiglu_int8_);
            d_grouped_swiglu_int8_ = nullptr;
        }
        if (d_grouped_swiglu_scales_)
        {
            hipFree(d_grouped_swiglu_scales_);
            d_grouped_swiglu_scales_ = nullptr;
        }
        if (d_grouped_gate_output_ptrs_)
        {
            hipFree(d_grouped_gate_output_ptrs_);
            d_grouped_gate_output_ptrs_ = nullptr;
        }
        if (d_grouped_up_output_ptrs_)
        {
            hipFree(d_grouped_up_output_ptrs_);
            d_grouped_up_output_ptrs_ = nullptr;
        }
        if (d_grouped_gateup_expert_ids_)
        {
            hipFree(d_grouped_gateup_expert_ids_);
            d_grouped_gateup_expert_ids_ = nullptr;
        }
        if (d_grouped_hidden_int8_)
        {
            hipFree(d_grouped_hidden_int8_);
            d_grouped_hidden_int8_ = nullptr;
        }
        if (d_grouped_hidden_scales_)
        {
            hipFree(d_grouped_hidden_scales_);
            d_grouped_hidden_scales_ = nullptr;
        }
        if (d_grouped_gateup_gate_partials_)
        {
            hipFree(d_grouped_gateup_gate_partials_);
            d_grouped_gateup_gate_partials_ = nullptr;
        }
        if (d_grouped_gateup_up_partials_)
        {
            hipFree(d_grouped_gateup_up_partials_);
            d_grouped_gateup_up_partials_ = nullptr;
        }
        if (d_shared_gate_scratch_)
        {
            hipFree(d_shared_gate_scratch_);
            d_shared_gate_scratch_ = nullptr;
        }
        if (d_route_logits_)
        {
            hipFree(d_route_logits_);
            d_route_logits_ = nullptr;
        }
        if (d_route_logits_partials_)
        {
            hipFree(d_route_logits_partials_);
            d_route_logits_partials_ = nullptr;
        }
        if (d_router_q8_hidden_)
        {
            hipFree(d_router_q8_hidden_);
            d_router_q8_hidden_ = nullptr;
        }
        if (d_router_q8_hidden_scales_)
        {
            hipFree(d_router_q8_hidden_scales_);
            d_router_q8_hidden_scales_ = nullptr;
        }
        if (d_route_indices_)
        {
            hipFree(d_route_indices_);
            d_route_indices_ = nullptr;
        }
        if (d_route_weights_)
        {
            hipFree(d_route_weights_);
            d_route_weights_ = nullptr;
        }
        if (d_group_int_indices_)
        {
            hipFree(d_group_int_indices_);
            d_group_int_indices_ = nullptr;
        }
        if (d_group_offsets_)
        {
            hipFree(d_group_offsets_);
            d_group_offsets_ = nullptr;
        }
        if (d_group_counts_)
        {
            hipFree(d_group_counts_);
            d_group_counts_ = nullptr;
        }
        if (d_group_max_tokens_)
        {
            hipFree(d_group_max_tokens_);
            d_group_max_tokens_ = nullptr;
        }
        if (d_group_token_indices_)
        {
            hipFree(d_group_token_indices_);
            d_group_token_indices_ = nullptr;
        }
        if (d_group_weights_)
        {
            hipFree(d_group_weights_);
            d_group_weights_ = nullptr;
        }
        if (d_group_active_expert_ids_)
        {
            hipFree(d_group_active_expert_ids_);
            d_group_active_expert_ids_ = nullptr;
        }
        for (auto &entry : router_fp16_gate_cache_)
        {
            if (entry.d_gate_weights_fp16)
            {
                (void)hipFree(entry.d_gate_weights_fp16);
                entry.d_gate_weights_fp16 = nullptr;
            }
        }
        router_fp16_gate_cache_.clear();
        for (auto &entry : router_q8_gate_cache_)
        {
            if (entry.d_gate_weights_q8)
            {
                (void)hipFree(entry.d_gate_weights_q8);
                entry.d_gate_weights_q8 = nullptr;
            }
            if (entry.d_gate_scales)
            {
                (void)hipFree(entry.d_gate_scales);
                entry.d_gate_scales = nullptr;
            }
        }
        router_q8_gate_cache_.clear();
        for (auto &entry : runtime_down_pointer_cache_)
        {
            if (entry.d_gate_ptrs)
            {
                (void)hipFree(entry.d_gate_ptrs);
                entry.d_gate_ptrs = nullptr;
            }
            if (entry.d_up_ptrs)
            {
                (void)hipFree(entry.d_up_ptrs);
                entry.d_up_ptrs = nullptr;
            }
        }
        for (auto &entry : runtime_gateup_pointer_cache_)
        {
            if (entry.d_gate_ptrs)
            {
                (void)hipFree(entry.d_gate_ptrs);
                entry.d_gate_ptrs = nullptr;
            }
            if (entry.d_up_ptrs)
            {
                (void)hipFree(entry.d_up_ptrs);
                entry.d_up_ptrs = nullptr;
            }
        }
        for (auto &table : grouped_down_desc_tables_)
        {
            if (table.device_descs)
            {
                hipFree(table.device_descs);
                table.device_descs = nullptr;
            }
        }
        for (auto &table : grouped_gateup_desc_tables_)
        {
            if (table.device_gate_descs)
            {
                hipFree(table.device_gate_descs);
                table.device_gate_descs = nullptr;
            }
            if (table.device_up_descs)
            {
                hipFree(table.device_up_descs);
                table.device_up_descs = nullptr;
            }
        }

        // Phase 5: grouped prefill scratch
        if (d_prefill_A_int8_)
        {
            hipFree(d_prefill_A_int8_);
            d_prefill_A_int8_ = nullptr;
        }
        if (d_prefill_A_scales_)
        {
            hipFree(d_prefill_A_scales_);
            d_prefill_A_scales_ = nullptr;
        }
        if (d_prefill_gate_)
        {
            hipFree(d_prefill_gate_);
            d_prefill_gate_ = nullptr;
        }
        if (d_prefill_up_)
        {
            hipFree(d_prefill_up_);
            d_prefill_up_ = nullptr;
        }
    }

    void ROCmMoEKernel::resetDynamicState()
    {
        if (!setMoEDevice(device_ordinal_, "resetDynamicState"))
            return;

        // Histogram counts are session-derived, but the allocation and mask
        // capacity are weight/model-shaped and may be referenced by captured
        // prefill graphs, so reset contents without changing device pointers.
        if (d_histogram_ && max_layers_ > 0 && max_experts_ > 0)
        {
            const size_t histogram_bytes = static_cast<size_t>(max_layers_) *
                                           static_cast<size_t>(max_experts_) *
                                           sizeof(uint64_t);
            hipError_t memset_err = hipMemset(d_histogram_, 0, histogram_bytes);
            if (memset_err != hipSuccess)
            {
                LOG_WARN("[ROCmMoEKernel::resetDynamicState] histogram reset failed: "
                         << hipGetErrorString(memset_err));
            }
        }

        // CPU-side grouping metadata mirrors the last request's routing table;
        // clearing it prevents legacy gather/scatter fallbacks from seeing old
        // expert counts after a cache clear.
        host_expert_counts_.clear();
        host_expert_offsets_.clear();
        host_grouped_indices_.clear();
        host_grouped_weights_.clear();
        prepared_num_experts_ = 0;
        group_active_expert_slots_ = 0;
    }

    void ROCmMoEKernel::syncBlasStream()
    {
        if (blas_gemm_)
            blas_gemm_->setStream(ROCmKernelBase::getStream());
    }

    bool ROCmMoEKernel::ensureSharedGateScratchCapacity(int seq_len)
    {
        if (seq_len <= shared_gate_scratch_capacity_)
            return true;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_shared_gate_scratch_),
                                 MoEWorkspaceBuffers::ROCM_SHARED_GATE,
                                 static_cast<size_t>(seq_len) * sizeof(float),
                                 "ensureSharedGateScratchCapacity"))
        {
            shared_gate_scratch_capacity_ = 0;
            return false;
        }
        shared_gate_scratch_capacity_ = seq_len;
        return true;
    }

    bool ROCmMoEKernel::ensureRouteBufferCapacity(size_t logits_count, size_t topk_count)
    {
        if (logits_count > route_logits_capacity_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_logits_),
                                     MoEWorkspaceBuffers::ROUTE_LOGITS,
                                     logits_count * sizeof(float),
                                     "ensureRouteBufferCapacity(logits)"))
            {
                route_logits_capacity_ = 0;
                return false;
            }
            route_logits_capacity_ = logits_count;
        }

        if (topk_count > route_topk_capacity_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_indices_),
                                     MoEWorkspaceBuffers::ROUTE_INDICES,
                                     topk_count * sizeof(int),
                                     "ensureRouteBufferCapacity(indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_weights_),
                                     MoEWorkspaceBuffers::ROUTE_WEIGHTS,
                                     topk_count * sizeof(float),
                                     "ensureRouteBufferCapacity(weights)"))
            {
                d_route_indices_ = nullptr;
                d_route_weights_ = nullptr;
                route_topk_capacity_ = 0;
                return false;
            }
            route_topk_capacity_ = topk_count;
        }

        if (topk_count == 0)
            return d_route_logits_ != nullptr;

        return d_route_logits_ && d_route_indices_ && d_route_weights_;
    }

    bool ROCmMoEKernel::ensureRouteLogitsPartialsCapacity(size_t partial_count)
    {
        if (partial_count == 0)
            return false;

        if (partial_count <= route_logits_partials_capacity_)
            return d_route_logits_partials_ != nullptr;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_logits_partials_),
                                 MoEWorkspaceBuffers::ROCM_ROUTE_LOGITS_PARTIALS,
                                 partial_count * sizeof(float),
                                 "ensureRouteLogitsPartialsCapacity"))
        {
            d_route_logits_partials_ = nullptr;
            route_logits_partials_capacity_ = 0;
            return false;
        }

        route_logits_partials_capacity_ = partial_count;
        return true;
    }

    bool ROCmMoEKernel::ensureRouterQ8HiddenScratchCapacity(int d_model)
    {
        if (d_model <= 0 || (d_model % 32) != 0)
            return false;

        const int blocks_per_row = d_model / 32;
        if (d_model <= router_q8_hidden_d_model_cap_ &&
            blocks_per_row <= router_q8_hidden_blocks_cap_ &&
            d_router_q8_hidden_ && d_router_q8_hidden_scales_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_router_q8_hidden_),
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_HIDDEN,
                                 static_cast<size_t>(d_model) * sizeof(int8_t),
                                 "ensureRouterQ8HiddenScratchCapacity(hidden)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_router_q8_hidden_scales_),
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_SCALES,
                                 static_cast<size_t>(blocks_per_row) * sizeof(float),
                                 "ensureRouterQ8HiddenScratchCapacity(scales)"))
        {
            d_router_q8_hidden_ = nullptr;
            d_router_q8_hidden_scales_ = nullptr;
            router_q8_hidden_d_model_cap_ = 0;
            router_q8_hidden_blocks_cap_ = 0;
            return false;
        }

        router_q8_hidden_d_model_cap_ = d_model;
        router_q8_hidden_blocks_cap_ = blocks_per_row;
        return true;
    }

    const ROCmMoEKernel::RouterQ8GateCacheEntry *ROCmMoEKernel::getOrCreateQ8RouterGateCache(
        ITensor *gate_weights,
        const float *gate_device_ptr,
        int d_model,
        int num_experts)
    {
        if (!debugEnv().rocm.moe_router_q8)
            return nullptr;
        if (!gate_weights || !gate_device_ptr || d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
            return nullptr;

        const auto tensor_key = reinterpret_cast<std::uintptr_t>(gate_weights);
        const auto device_ptr_key = reinterpret_cast<std::uintptr_t>(gate_device_ptr);
        for (const auto &entry : router_q8_gate_cache_)
        {
            if (entry.source_tensor == tensor_key &&
                entry.source_device_ptr == device_ptr_key &&
                entry.d_model == d_model &&
                entry.num_experts == num_experts &&
                entry.d_gate_weights_q8 && entry.d_gate_scales)
            {
                return &entry;
            }
        }

        const bool capture_active = isGraphCaptureActive() ||
                                    (deviceContext() && deviceContext()->isDeviceGraphCaptureActive());
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        void *stream = getStream();
        if (stream)
        {
            const hipError_t capture_err = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture_status);
            if (capture_err != hipSuccess)
                capture_status = hipStreamCaptureStatusNone;
        }
        if (capture_active || capture_status == hipStreamCaptureStatusActive)
        {
            LOG_DEBUG("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] cache miss during graph capture; falling back to non-Q8 router");
            return nullptr;
        }

        const size_t d_model_sz = static_cast<size_t>(d_model);
        const size_t experts_sz = static_cast<size_t>(num_experts);
        if (d_model_sz > std::numeric_limits<size_t>::max() / experts_sz)
        {
            LOG_WARN("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] router gate size overflow");
            return nullptr;
        }
        const size_t element_count = d_model_sz * experts_sz;
        const int blocks_per_row = d_model / 32;
        const size_t scale_count = static_cast<size_t>(num_experts) * static_cast<size_t>(blocks_per_row);

        if (!setMoEDevice(device_ordinal_, "getOrCreateQ8RouterGateCache"))
            return nullptr;

        for (auto it = router_q8_gate_cache_.begin(); it != router_q8_gate_cache_.end();)
        {
            if (it->source_tensor == tensor_key &&
                it->d_model == d_model &&
                it->num_experts == num_experts &&
                it->source_device_ptr != device_ptr_key)
            {
                if (it->d_gate_weights_q8)
                    (void)hipFree(it->d_gate_weights_q8);
                if (it->d_gate_scales)
                    (void)hipFree(it->d_gate_scales);
                it = router_q8_gate_cache_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        int8_t *d_gate_weights_q8 = nullptr;
        float *d_gate_scales = nullptr;
        hipError_t err = hipMalloc(&d_gate_weights_q8, element_count * sizeof(int8_t));
        if (err == hipSuccess)
        {
            err = hipMalloc(&d_gate_scales, scale_count * sizeof(float));
        }
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] hipMalloc Q8 router gate failed: "
                     << hipGetErrorString(err));
            if (d_gate_weights_q8)
                (void)hipFree(d_gate_weights_q8);
            return nullptr;
        }

        if (!hipMoE_quantize_router_gate_q8(gate_device_ptr, d_gate_weights_q8, d_gate_scales,
                                            d_model, num_experts,
                                            device_ordinal_, getStream()))
        {
            LOG_WARN("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] FP32->Q8 router gate conversion launch failed; falling back to non-Q8 router");
            (void)hipFree(d_gate_weights_q8);
            (void)hipFree(d_gate_scales);
            return nullptr;
        }

        RouterQ8GateCacheEntry entry{};
        entry.source_tensor = tensor_key;
        entry.source_device_ptr = device_ptr_key;
        entry.d_model = d_model;
        entry.num_experts = num_experts;
        entry.blocks_per_row = blocks_per_row;
        entry.element_count = element_count;
        entry.scale_count = scale_count;
        entry.d_gate_weights_q8 = d_gate_weights_q8;
        entry.d_gate_scales = d_gate_scales;
        router_q8_gate_cache_.push_back(entry);

        LOG_DEBUG("[ROCmMoEKernel] Cached Q8 router gate tensor="
                  << reinterpret_cast<const void *>(tensor_key)
                  << " source_device=" << reinterpret_cast<const void *>(device_ptr_key)
                  << " shape=[" << num_experts << "," << d_model << "] payload_bytes="
                  << element_count << " scale_bytes=" << (scale_count * sizeof(float)));
        return &router_q8_gate_cache_.back();
    }

    const void *ROCmMoEKernel::getOrCreateFP16RouterGateCache(
        ITensor *gate_weights,
        const float *gate_device_ptr,
        int d_model,
        int num_experts)
    {
        if (!debugEnv().rocm.moe_router_fp16)
            return nullptr;
        if (!gate_weights || !gate_device_ptr || d_model <= 0 || num_experts <= 0)
            return nullptr;

        const auto tensor_key = reinterpret_cast<std::uintptr_t>(gate_weights);
        const auto device_ptr_key = reinterpret_cast<std::uintptr_t>(gate_device_ptr);
        for (const auto &entry : router_fp16_gate_cache_)
        {
            if (entry.source_tensor == tensor_key &&
                entry.source_device_ptr == device_ptr_key &&
                entry.d_model == d_model &&
                entry.num_experts == num_experts &&
                entry.d_gate_weights_fp16)
            {
                return entry.d_gate_weights_fp16;
            }
        }

        const bool capture_active = isGraphCaptureActive() ||
                                    (deviceContext() && deviceContext()->isDeviceGraphCaptureActive());
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        void *stream = getStream();
        if (stream)
        {
            const hipError_t capture_err = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture_status);
            if (capture_err != hipSuccess)
                capture_status = hipStreamCaptureStatusNone;
        }
        if (capture_active || capture_status == hipStreamCaptureStatusActive)
        {
            LOG_DEBUG("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] cache miss during graph capture; falling back to FP32 router");
            return nullptr;
        }

        const size_t d_model_sz = static_cast<size_t>(d_model);
        const size_t experts_sz = static_cast<size_t>(num_experts);
        if (d_model_sz > std::numeric_limits<size_t>::max() / experts_sz)
        {
            LOG_WARN("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] router gate size overflow");
            return nullptr;
        }
        const size_t element_count = d_model_sz * experts_sz;
        if (element_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_WARN("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] router gate too large for conversion kernel: elements="
                     << element_count);
            return nullptr;
        }

        if (!setMoEDevice(device_ordinal_, "getOrCreateFP16RouterGateCache"))
            return nullptr;

        for (auto it = router_fp16_gate_cache_.begin(); it != router_fp16_gate_cache_.end();)
        {
            if (it->source_tensor == tensor_key &&
                it->d_model == d_model &&
                it->num_experts == num_experts &&
                it->source_device_ptr != device_ptr_key)
            {
                if (it->d_gate_weights_fp16)
                    (void)hipFree(it->d_gate_weights_fp16);
                it = router_fp16_gate_cache_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        void *d_gate_weights_fp16 = nullptr;
        hipError_t err = hipMalloc(&d_gate_weights_fp16, element_count * sizeof(uint16_t));
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] hipMalloc FP16 router gate failed: "
                     << hipGetErrorString(err));
            return nullptr;
        }

        if (!hipMoE_fp32_to_fp16(gate_device_ptr, d_gate_weights_fp16,
                                 static_cast<int>(element_count),
                                 device_ordinal_, getStream()))
        {
            LOG_WARN("[ROCmMoEKernel::getOrCreateFP16RouterGateCache] FP32->FP16 router gate conversion launch failed; falling back to FP32 router");
            (void)hipFree(d_gate_weights_fp16);
            return nullptr;
        }

        RouterFP16GateCacheEntry entry{};
        entry.source_tensor = tensor_key;
        entry.source_device_ptr = device_ptr_key;
        entry.d_model = d_model;
        entry.num_experts = num_experts;
        entry.element_count = element_count;
        entry.d_gate_weights_fp16 = d_gate_weights_fp16;
        router_fp16_gate_cache_.push_back(entry);

        LOG_DEBUG("[ROCmMoEKernel] Cached FP16 router gate tensor="
                  << reinterpret_cast<const void *>(tensor_key)
                  << " source_device=" << reinterpret_cast<const void *>(device_ptr_key)
                  << " shape=[" << num_experts << "," << d_model << "] bytes="
                  << (element_count * sizeof(uint16_t)));
        return d_gate_weights_fp16;
    }

    // =========================================================================
    // routeCore() — Shared GPU routing logic: gate GEMM + softmax + top-k.
    // Returns pointers into persistent device buffers owned by this kernel.
    // =========================================================================

    bool ROCmMoEKernel::routeCore(
        const float *hidden, const void *gate_weights, TensorType gate_type,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights, DeviceRouteBuffers &bufs)
    {
        bufs.logits_count = static_cast<size_t>(seq_len) * num_experts;
        bufs.topk_count = static_cast<size_t>(seq_len) * top_k;

        if (!ensureRouteBufferCapacity(bufs.logits_count, bufs.topk_count))
            return false;

        bufs.d_logits = d_route_logits_;
        bufs.d_indices = d_route_indices_;
        bufs.d_weights = d_route_weights_;

        const bool decode_single_token = (seq_len == 1);
        if (decode_single_token)
        {
            if (!launchDecodeGateLogitsForGateType(
                    hidden, gate_weights, gate_type, bufs.d_logits,
                    d_model, num_experts,
                    device_ordinal_, getStream(),
                    "ROCmMoEKernel::routeCore"))
            {
                bufs = {};
                return false;
            }
        }
        else if (seq_len >= 2 && seq_len <= 4 && gate_type == TensorType::FP32)
        {
            void *stream = getStream();
            if (!launchSmallMGateLogits(
                    hidden, gate_weights, gate_type, bufs.d_logits,
                    seq_len, d_model, num_experts,
                    device_ordinal_, stream))
            {
                bufs = {};
                return false;
            }

            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "rocm_moe_small_m_fused_router_calls",
                    1.0,
                    "moe",
                    DeviceId::rocm(device_ordinal_).to_string(),
                    PerfStatsCollector::Tags{
                        {"seq_len", std::to_string(seq_len)},
                        {"num_experts", std::to_string(num_experts)},
                        {"top_k", std::to_string(top_k)}});
            }
        }
        else if (gate_type != TensorType::FP32)
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] prefill routing requires FP32 gate weights; got "
                      << tensorTypeName(gate_type));
            bufs = {};
            return false;
        }
        else if (!blas_gemm_->execute(hidden, static_cast<const float *>(gate_weights), bufs.d_logits,
                                      seq_len, num_experts, d_model,
                                      /*transA=*/false, /*transB=*/true))
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] hipBLAS gate logits GEMM failed");
            bufs = {};
            return false;
        }

        // Softmax + top-k selection
        if (!hipMoE_softmax_topk(bufs.d_logits, bufs.d_indices, bufs.d_weights,
                                 seq_len, num_experts, top_k,
                                 normalize_weights,
                                 device_ordinal_, getStream()))
        {
            bufs = {};
            return false;
        }

        return true;
    }

    // =========================================================================
    // route() — Gate logits + softmax + top-k on GPU, results back to host
    // =========================================================================

    bool ROCmMoEKernel::route(
        const float *hidden,
        const float *gate_weights,
        int seq_len, int d_model,
        int num_experts, int top_k,
        bool normalize_weights,
        MoERoutingResult &result)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        DeviceRouteBuffers bufs;
        if (!routeCore(hidden, gate_weights, TensorType::FP32, seq_len, d_model,
                       num_experts, top_k, normalize_weights, bufs))
            return false;

        // D2H results
        result.expert_indices.resize(bufs.topk_count);
        result.expert_weights.resize(bufs.topk_count);
        result.router_logits.resize(bufs.logits_count);

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err;

        err = hipMemcpyAsync(result.router_logits.data(), bufs.d_logits,
                             bufs.logits_count * sizeof(float),
                             hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::route] D2H logits failed: " << hipGetErrorString(err));
            return false;
        }

        err = hipMemcpyAsync(result.expert_indices.data(), bufs.d_indices,
                             bufs.topk_count * sizeof(int),
                             hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::route] D2H indices failed: " << hipGetErrorString(err));
            return false;
        }

        err = hipMemcpyAsync(result.expert_weights.data(), bufs.d_weights,
                             bufs.topk_count * sizeof(float),
                             hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::route] D2H weights failed: " << hipGetErrorString(err));
            return false;
        }

        hipStreamSynchronize(stream);
        return true;
    }

    // =========================================================================
    // gatherTokenBatch() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::gatherTokenBatch(
        const float *hidden,
        float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "gatherTokenBatch"))
            return;

        if (!hipMoE_gather_tokens(hidden, batch_buffer, token_indices,
                                  num_tokens, d_model,
                                  device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatch] kernel launch failed");
        }
    }

    // =========================================================================
    // scatterAddWeighted() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::scatterAddWeighted(
        float *output,
        const float *expert_output,
        const int *token_indices,
        const float *weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "scatterAddWeighted"))
            return;

        if (!hipMoE_scatter_add(output, expert_output, token_indices, weights,
                                num_tokens, d_model,
                                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeighted] kernel launch failed");
        }
    }

    // =========================================================================
    // sharedExpertGate() — All pointers are device pointers
    //
    // Needs a small scratch buffer for gate values [seq_len floats].
    // Allocates on demand (small — at most a few KB).
    // =========================================================================

    void ROCmMoEKernel::sharedExpertGate(
        const float *input,
        const float *gate_inp,
        float *shared_output,
        int seq_len, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SHARED_GATE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || d_model <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "sharedExpertGate"))
            return;

        if (seq_len == 1)
        {
            if (!hipMoE_shared_expert_gate_decode_fused(input, gate_inp, shared_output,
                                                        d_model,
                                                        device_ordinal_, getStream()))
            {
                LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] fused decode kernel launch failed");
            }
            return;
        }

        if (!ensureSharedGateScratchCapacity(seq_len))
            return;

        if (!hipMoE_shared_expert_gate(input, gate_inp, shared_output, d_shared_gate_scratch_,
                                       seq_len, d_model,
                                       device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] kernel launch failed");
        }
    }

    // =========================================================================
    // swiGLU() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "swiGLU"))
            return;

        if (!hipMoE_swiglu(gate, up, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLU] kernel launch failed");
        }
    }

    void ROCmMoEKernel::weightedAdd(float *output, const float *input,
                                    float weight, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "weightedAdd"))
            return;

        if (!hipMoE_weighted_add(output, input, weight, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAdd] kernel launch failed");
        }
    }

    // =========================================================================
    // Phase 2: Device-resident histogram + expert mask
    // =========================================================================

    void ROCmMoEKernel::allocateHistogramBuffers(int num_layers, int num_experts)
    {
        // Already allocated with sufficient dimensions?
        if (d_histogram_ && max_layers_ >= num_layers && max_experts_ >= num_experts)
            return;

        if (!setMoEDevice(device_ordinal_, "allocateHistogramBuffers"))
            return;

        // Free old if dimensions grew
        if (d_histogram_)
        {
            hipFree(d_histogram_);
            d_histogram_ = nullptr;
        }

        max_layers_ = num_layers;
        max_experts_ = num_experts;

        const size_t total = static_cast<size_t>(max_layers_) * max_experts_;
        hipError_t err = hipMalloc(&d_histogram_, total * sizeof(uint64_t));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::allocateHistogramBuffers] hipMalloc histogram failed: "
                      << hipGetErrorString(err));
            d_histogram_ = nullptr;
            return;
        }

        err = hipMemset(d_histogram_, 0, total * sizeof(uint64_t));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::allocateHistogramBuffers] hipMemset failed: "
                      << hipGetErrorString(err));
        }

        LOG_DEBUG("[ROCmMoEKernel] Allocated histogram buffer: "
                  << max_layers_ << " layers × " << max_experts_ << " experts");
    }

    void ROCmMoEKernel::recordHistogramDevice(
        const int *d_routing_indices, int seq_len, int top_k, int layer_idx)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || top_k <= 0)
            return;

        // Lazy allocate — assume at least layer_idx+1 layers, 256 experts as initial guess
        const int min_experts = 256;
        if (!d_histogram_ || layer_idx >= max_layers_)
        {
            allocateHistogramBuffers(layer_idx + 1, (max_experts_ > 0) ? max_experts_ : min_experts);
        }
        if (!d_histogram_)
            return;

        if (!hipMoE_histogram_record(
                d_routing_indices,
                reinterpret_cast<unsigned long long *>(d_histogram_),
                seq_len, max_experts_, top_k, layer_idx,
                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::recordHistogramDevice] kernel launch failed");
        }
    }

    void ROCmMoEKernel::syncHistogramToHost(
        uint64_t *host_counts, int layer_idx, int num_experts)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!d_histogram_)
        {
            LOG_WARN("[ROCmMoEKernel::syncHistogramToHost] No histogram allocated");
            return;
        }
        if (layer_idx >= max_layers_ || num_experts > max_experts_)
        {
            LOG_ERROR("[ROCmMoEKernel::syncHistogramToHost] layer_idx=" << layer_idx
                                                                        << " or num_experts=" << num_experts << " out of range");
            return;
        }

        if (!setMoEDevice(device_ordinal_, "syncHistogramToHost"))
            return;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        const size_t offset = static_cast<size_t>(layer_idx) * max_experts_;

        hipError_t err = hipMemcpyAsync(
            host_counts,
            d_histogram_ + offset,
            num_experts * sizeof(uint64_t),
            hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::syncHistogramToHost] D2H copy failed: "
                      << hipGetErrorString(err));
            return;
        }

        hipStreamSynchronize(stream);
    }

    void ROCmMoEKernel::resetHistogramDevice(int layer_idx, int num_experts)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!d_histogram_)
        {
            LOG_WARN("[ROCmMoEKernel::resetHistogramDevice] No histogram allocated");
            return;
        }
        if (layer_idx >= max_layers_)
        {
            LOG_ERROR("[ROCmMoEKernel::resetHistogramDevice] layer_idx=" << layer_idx << " out of range");
            return;
        }

        if (!setMoEDevice(device_ordinal_, "resetHistogramDevice"))
            return;

        if (!hipMoE_histogram_reset(
                reinterpret_cast<unsigned long long *>(d_histogram_),
                layer_idx, num_experts,
                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::resetHistogramDevice] kernel launch failed");
        }
    }

    void ROCmMoEKernel::updateExpertMaskDevice(const bool *mask, int num_experts)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (num_experts <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "updateExpertMaskDevice"))
            return;

        // Allocate or reallocate if needed
        if (!d_expert_mask_ || num_experts > max_experts_)
        {
            if (d_expert_mask_)
                hipFree(d_expert_mask_);

            hipError_t err = hipMalloc(&d_expert_mask_, num_experts * sizeof(bool));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmMoEKernel::updateExpertMaskDevice] hipMalloc mask failed: "
                          << hipGetErrorString(err));
                d_expert_mask_ = nullptr;
                return;
            }
            // Update max_experts_ if the mask required more
            if (num_experts > max_experts_)
                max_experts_ = num_experts;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            d_expert_mask_, mask,
            num_experts * sizeof(bool),
            hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::updateExpertMaskDevice] H2D copy failed: "
                      << hipGetErrorString(err));
        }
    }

    void ROCmMoEKernel::applyExpertMaskDevice(
        float *d_routing_weights, const int *d_routing_indices,
        int seq_len, int top_k)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || top_k <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "applyExpertMaskDevice"))
            return;

        if (!d_expert_mask_)
        {
            LOG_WARN("[ROCmMoEKernel::applyExpertMaskDevice] No expert mask uploaded");
            return;
        }

        if (!hipMoE_apply_expert_mask(
                d_routing_weights, d_routing_indices,
                d_expert_mask_,
                seq_len, top_k,
                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::applyExpertMaskDevice] kernel launch failed");
        }
    }

    // =========================================================================
    // Phase 3: Device-side token grouping
    // =========================================================================

    bool ROCmMoEKernel::groupTokensByExpertDevice(
        const int *d_routing_indices,
        const float *d_routing_weights,
        int seq_len, int num_experts, int top_k,
        int *d_expert_offsets,
        int *d_expert_counts,
        int *d_grouped_token_indices,
        float *d_grouped_weights)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "groupTokensByExpertDevice"))
            return false;

        const int total_slots = seq_len * top_k;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err;

        // Step 1: Zero expert_counts
        err = hipMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync expert_counts failed: "
                      << hipGetErrorString(err));
            return false;
        }

        // Step 2: Count per expert
        if (!hipMoE_count_per_expert(d_routing_indices, d_expert_counts,
                                     total_slots, num_experts,
                                     device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] count_per_expert failed");
            return false;
        }

        // Step 3: Exclusive scan (expert_counts → expert_offsets)
        if (!hipMoE_exclusive_scan(d_expert_counts, d_expert_offsets,
                                   num_experts,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] exclusive_scan failed");
            return false;
        }

        // Step 4: Lazy-allocate write_heads scratch buffer
        if (!d_write_heads_ || max_write_heads_experts_ < num_experts)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_write_heads_),
                                     MoEWorkspaceBuffers::GROUP_WRITE_HEADS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "groupTokensByExpertDevice(write_heads)"))
            {
                d_write_heads_ = nullptr;
                max_write_heads_experts_ = 0;
                return false;
            }
            max_write_heads_experts_ = num_experts;
        }

        // Zero write_heads
        err = hipMemsetAsync(d_write_heads_, 0, num_experts * sizeof(int), stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] hipMemsetAsync write_heads failed: "
                      << hipGetErrorString(err));
            return false;
        }

        // Step 5: Scatter tokens into grouped arrays
        if (!hipMoE_scatter_tokens(d_routing_indices, d_routing_weights,
                                   d_expert_offsets, d_write_heads_,
                                   d_grouped_token_indices, d_grouped_weights,
                                   total_slots, num_experts, top_k,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] scatter_tokens failed");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Tensor-aware API overrides — GPU implementations
    //
    // These use gpu_data_ptr() for device-resident data and keep
    // all computation on device.  Small host-side arrays (token indices,
    // routing weights) are uploaded via cached staging buffers.
    // =========================================================================

    bool ROCmMoEKernel::ensureStagingCapacity(int count)
    {
        if (count <= staging_capacity_)
            return d_staging_indices_ && d_staging_weights_;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_staging_indices_),
                                 MoEWorkspaceBuffers::STAGING_INDICES,
                                 static_cast<size_t>(count) * sizeof(int),
                                 "ensureStagingCapacity(indices)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_staging_weights_),
                                 MoEWorkspaceBuffers::STAGING_WEIGHTS,
                                 static_cast<size_t>(count) * sizeof(float),
                                 "ensureStagingCapacity(weights)"))
        {
            d_staging_indices_ = nullptr;
            d_staging_weights_ = nullptr;
            staging_capacity_ = 0;
            return false;
        }
        staging_capacity_ = count;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedDecodeCapacity(int num_active, int intermediate)
    {
        if (num_active <= 0 || intermediate <= 0 || (intermediate % 32) != 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedDecodeCapacity"))
            return false;

        const int blocks_per_row = intermediate / 32;
        const bool pointer_capacity_ok = grouped_decode_active_cap_ >= num_active;
        const bool activation_capacity_ok = grouped_decode_intermediate_cap_ >= intermediate;
        if (pointer_capacity_ok && activation_capacity_ok &&
            d_grouped_gate_ptrs_ && d_grouped_up_ptrs_ && d_grouped_decode_weights_ &&
            d_grouped_expert_ids_ && d_grouped_down_descs_ &&
            d_grouped_swiglu_int8_ && d_grouped_swiglu_scales_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_GATE_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedDecodeCapacity(gate_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_UP_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedDecodeCapacity(up_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_expert_ids_),
                                 MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 "ensureGroupedDecodeCapacity(expert_ids)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_decode_weights_),
                                 MoEWorkspaceBuffers::DECODE_WEIGHTS,
                                 static_cast<size_t>(num_active) * sizeof(float),
                                 "ensureGroupedDecodeCapacity(weights)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_down_descs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_DOWN_DESCS,
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeVNNIMatrixDesc),
                                 "ensureGroupedDecodeCapacity(down_descs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_swiglu_int8_),
                                 MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,
                                 static_cast<size_t>(num_active) * intermediate * sizeof(int8_t),
                                 "ensureGroupedDecodeCapacity(swiglu_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_swiglu_scales_),
                                 MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
                                 static_cast<size_t>(num_active) * blocks_per_row * sizeof(float),
                                 "ensureGroupedDecodeCapacity(swiglu_scales)"))
        {
            d_grouped_gate_ptrs_ = nullptr;
            d_grouped_up_ptrs_ = nullptr;
            d_grouped_expert_ids_ = nullptr;
            d_grouped_decode_weights_ = nullptr;
            d_grouped_down_descs_ = nullptr;
            d_grouped_swiglu_int8_ = nullptr;
            d_grouped_swiglu_scales_ = nullptr;
            grouped_decode_active_cap_ = 0;
            grouped_decode_intermediate_cap_ = 0;
            return false;
        }

        grouped_decode_active_cap_ = num_active;
        grouped_decode_intermediate_cap_ = intermediate;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedGateUpCapacity(int num_active, int d_model)
    {
        if (num_active <= 0 || d_model <= 0 || (d_model % 32) != 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedGateUpCapacity"))
            return false;

        const int blocks_per_row = d_model / 32;
        if (grouped_gateup_active_cap_ >= num_active &&
            grouped_gateup_d_model_cap_ >= d_model &&
            d_grouped_gate_output_ptrs_ && d_grouped_up_output_ptrs_ &&
            d_grouped_gateup_expert_ids_ &&
            d_grouped_hidden_int8_ && d_grouped_hidden_scales_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_output_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_GATE_OUTPUT_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedGateUpCapacity(gate_output_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_output_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_UP_OUTPUT_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedGateUpCapacity(up_output_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gateup_expert_ids_),
                                 MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 "ensureGroupedGateUpCapacity(expert_ids)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_hidden_int8_),
                                 MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,
                                 static_cast<size_t>(d_model) * sizeof(int8_t),
                                 "ensureGroupedGateUpCapacity(hidden_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_hidden_scales_),
                                 MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
                                 static_cast<size_t>(blocks_per_row) * sizeof(float),
                                 "ensureGroupedGateUpCapacity(hidden_scales)"))
        {
            d_grouped_gate_output_ptrs_ = nullptr;
            d_grouped_up_output_ptrs_ = nullptr;
            d_grouped_gateup_expert_ids_ = nullptr;
            d_grouped_hidden_int8_ = nullptr;
            d_grouped_hidden_scales_ = nullptr;
            grouped_gateup_active_cap_ = 0;
            grouped_gateup_d_model_cap_ = 0;
            return false;
        }

        grouped_gateup_active_cap_ = num_active;
        grouped_gateup_d_model_cap_ = d_model;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedGateUpKPartScratchCapacity(int num_active, int k_partitions, int intermediate)
    {
        if (num_active <= 0 || intermediate <= 0 ||
            !(k_partitions == 2 || k_partitions == 4 || k_partitions == 8))
        {
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "ensureGroupedGateUpKPartScratchCapacity"))
            return false;

        if (d_grouped_gateup_gate_partials_ && d_grouped_gateup_up_partials_ &&
            grouped_gateup_kpart_active_cap_ >= num_active &&
            grouped_gateup_kpart_partitions_cap_ >= k_partitions &&
            grouped_gateup_kpart_intermediate_cap_ >= intermediate)
        {
            return true;
        }

        const size_t partial_count = static_cast<size_t>(num_active) *
                                     static_cast<size_t>(k_partitions) *
                                     static_cast<size_t>(intermediate);
        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gateup_gate_partials_),
                                 MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
                                 partial_count * sizeof(float),
                                 "ensureGroupedGateUpKPartScratchCapacity(gate)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gateup_up_partials_),
                                 MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                                 partial_count * sizeof(float),
                                 "ensureGroupedGateUpKPartScratchCapacity(up)"))
        {
            d_grouped_gateup_gate_partials_ = nullptr;
            d_grouped_gateup_up_partials_ = nullptr;
            grouped_gateup_kpart_active_cap_ = 0;
            grouped_gateup_kpart_partitions_cap_ = 0;
            grouped_gateup_kpart_intermediate_cap_ = 0;
            return false;
        }

        grouped_gateup_kpart_active_cap_ = num_active;
        grouped_gateup_kpart_partitions_cap_ = k_partitions;
        grouped_gateup_kpart_intermediate_cap_ = intermediate;
        return true;
    }

    bool ROCmMoEKernel::ensureRuntimeGateUpPointerArrays(
        int descriptor_table_id,
        int top_k,
        const std::array<float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        float ***d_gate_ptrs,
        float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || descriptor_table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "ensureRuntimeGateUpPointerArrays"))
            return false;

        for (auto &entry : runtime_gateup_pointer_cache_)
        {
            if (entry.descriptor_table_id != descriptor_table_id || entry.top_k != top_k)
                continue;

            bool matches = true;
            for (int slot = 0; slot < top_k; ++slot)
            {
                if (entry.gate_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]) ||
                    entry.up_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(up_ptrs[slot]))
                {
                    matches = false;
                    break;
                }
            }
            if (matches && entry.d_gate_ptrs && entry.d_up_ptrs)
            {
                *d_gate_ptrs = entry.d_gate_ptrs;
                *d_up_ptrs = entry.d_up_ptrs;
                return true;
            }
        }

        RuntimeGateUpPointerCacheEntry entry;
        entry.descriptor_table_id = descriptor_table_id;
        entry.top_k = top_k;
        for (int slot = 0; slot < top_k; ++slot)
        {
            entry.gate_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]);
            entry.up_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(up_ptrs[slot]);
        }

        hipError_t err = hipMalloc(&entry.d_gate_ptrs, static_cast<size_t>(top_k) * sizeof(float *));
        if (err == hipSuccess)
            err = hipMalloc(&entry.d_up_ptrs, static_cast<size_t>(top_k) * sizeof(float *));
        if (err == hipSuccess)
        {
            hipStream_t stream = static_cast<hipStream_t>(getStream());
            err = hipMemcpyAsync(entry.d_gate_ptrs, gate_ptrs.data(),
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
            if (err == hipSuccess)
                err = hipMemcpyAsync(entry.d_up_ptrs, up_ptrs.data(),
                                     static_cast<size_t>(top_k) * sizeof(float *),
                                     hipMemcpyHostToDevice, stream);
        }
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::ensureRuntimeGateUpPointerArrays] H2D pointer cache upload failed: "
                      << hipGetErrorString(err));
            if (entry.d_gate_ptrs)
                (void)hipFree(entry.d_gate_ptrs);
            if (entry.d_up_ptrs)
                (void)hipFree(entry.d_up_ptrs);
            return false;
        }

        runtime_gateup_pointer_cache_.push_back(entry);
        auto &cached = runtime_gateup_pointer_cache_.back();
        *d_gate_ptrs = cached.d_gate_ptrs;
        *d_up_ptrs = cached.d_up_ptrs;
        return true;
    }

    bool ROCmMoEKernel::ensureRuntimeDownPointerArrays(
        int descriptor_table_id,
        int top_k,
        const std::array<const float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<const float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        const float ***d_gate_ptrs,
        const float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || descriptor_table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "ensureRuntimeDownPointerArrays"))
            return false;

        for (auto &entry : runtime_down_pointer_cache_)
        {
            if (entry.descriptor_table_id != descriptor_table_id || entry.top_k != top_k)
                continue;

            bool matches = true;
            for (int slot = 0; slot < top_k; ++slot)
            {
                if (entry.gate_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]) ||
                    entry.up_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(up_ptrs[slot]))
                {
                    matches = false;
                    break;
                }
            }
            if (matches && entry.d_gate_ptrs && entry.d_up_ptrs)
            {
                *d_gate_ptrs = entry.d_gate_ptrs;
                *d_up_ptrs = entry.d_up_ptrs;
                return true;
            }
        }

        RuntimeDownPointerCacheEntry entry;
        entry.descriptor_table_id = descriptor_table_id;
        entry.top_k = top_k;
        for (int slot = 0; slot < top_k; ++slot)
        {
            entry.gate_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]);
            entry.up_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(up_ptrs[slot]);
        }

        hipError_t err = hipMalloc(&entry.d_gate_ptrs, static_cast<size_t>(top_k) * sizeof(float *));
        if (err == hipSuccess)
            err = hipMalloc(&entry.d_up_ptrs, static_cast<size_t>(top_k) * sizeof(float *));
        if (err == hipSuccess)
        {
            hipStream_t stream = static_cast<hipStream_t>(getStream());
            err = hipMemcpyAsync(entry.d_gate_ptrs, gate_ptrs.data(),
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
            if (err == hipSuccess)
                err = hipMemcpyAsync(entry.d_up_ptrs, up_ptrs.data(),
                                     static_cast<size_t>(top_k) * sizeof(float *),
                                     hipMemcpyHostToDevice, stream);
        }
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::ensureRuntimeDownPointerArrays] H2D pointer cache upload failed: "
                      << hipGetErrorString(err));
            if (entry.d_gate_ptrs)
                (void)hipFree(entry.d_gate_ptrs);
            if (entry.d_up_ptrs)
                (void)hipFree(entry.d_up_ptrs);
            return false;
        }

        runtime_down_pointer_cache_.push_back(entry);
        auto &cached = runtime_down_pointer_cache_.back();
        *d_gate_ptrs = cached.d_gate_ptrs;
        *d_up_ptrs = cached.d_up_ptrs;
        return true;
    }

    bool ROCmMoEKernel::routeWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensors] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        DeviceRouteBuffers bufs;
        if (!routeCore(h, g, gate_type, seq_len, d_model, num_experts, top_k,
                       normalize_weights, bufs))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err;

        // D2D: write routing results to output tensors on device.
        // Indices need int→float conversion; weights are a D2D copy.
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_idx || !d_wt)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensors] output tensors have no device allocation");
            return false;
        }

        float *h_idx = nullptr;
        float *h_wt = nullptr;
        const bool needs_decode_host_topk = (seq_len == 1);
#ifdef ENABLE_PIPELINE_SNAPSHOTS
        const bool needs_snapshot_host_topk = true;
#else
        const bool needs_snapshot_host_topk = false;
#endif
        const bool needs_host_topk = needs_decode_host_topk || needs_snapshot_host_topk;
        if (needs_host_topk)
        {
            h_idx = static_cast<float *>(output_indices->raw_mutable_data());
            h_wt = static_cast<float *>(output_weights->raw_mutable_data());
            if (!h_idx || !h_wt)
            {
                LOG_ERROR("[ROCmMoEKernel::routeWithTensors] output tensors have no host storage");
                return false;
            }
        }

        // int→float conversion kernel (indices are int on device, tensor stores float)
        hipMoE_int_to_float(bufs.d_indices, d_idx,
                            static_cast<int>(bufs.topk_count),
                            device_ordinal_, getStream());

        err = hipMemcpyAsync(d_wt, bufs.d_weights,
                             bufs.topk_count * sizeof(float),
                             hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensors] D2D weights failed: " << hipGetErrorString(err));
            return false;
        }

        host_result.expert_indices.clear();
        host_result.expert_weights.clear();
        host_result.router_logits.clear();

        if (needs_host_topk)
        {
            err = hipMemcpyAsync(h_idx, d_idx,
                                 bufs.topk_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmMoEKernel::routeWithTensors] D2H decode indices failed: " << hipGetErrorString(err));
                return false;
            }

            err = hipMemcpyAsync(h_wt, d_wt,
                                 bufs.topk_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmMoEKernel::routeWithTensors] D2H decode weights failed: " << hipGetErrorString(err));
                return false;
            }
        }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        host_result.router_logits.resize(bufs.logits_count);
        err = hipMemcpyAsync(host_result.router_logits.data(), bufs.d_logits,
                             bufs.logits_count * sizeof(float), hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensors] D2H snapshot logits failed: " << hipGetErrorString(err));
            return false;
        }
#endif

        if (needs_host_topk || !host_result.router_logits.empty())
        {
            err = hipStreamSynchronize(stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmMoEKernel::routeWithTensors] stream sync failed after routing D2H: " << hipGetErrorString(err));
                return false;
            }
        }

        if (needs_host_topk)
        {
            host_result.expert_indices.resize(bufs.topk_count);
            host_result.expert_weights.assign(h_wt, h_wt + bufs.topk_count);
            for (size_t i = 0; i < bufs.topk_count; ++i)
                host_result.expert_indices[i] = static_cast<int>(h_idx[i]);

            output_indices->transitionTo(TensorCoherenceState::SYNCED);
            output_weights->transitionTo(TensorCoherenceState::SYNCED);
        }
        else
        {
            output_indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            output_weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        }

        return true;
    }

    bool ROCmMoEKernel::decodeRouteSelect(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *gate_weights,
        int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !hidden || !gate_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] null runtime/input/gate tensor");
            return false;
        }
        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] invalid dimensions d_model=" << d_model
                                                                                       << " num_experts=" << num_experts
                                                                                       << " top_k=" << top_k);
            return false;
        }

#if LLAMINAR_ASSERTIONS_ACTIVE
        DeviceMoELayerRuntime runtime_snapshot{};
        hipError_t runtime_copy_err = hipMemcpy(&runtime_snapshot, runtime_layer,
                                                sizeof(DeviceMoELayerRuntime),
                                                hipMemcpyDeviceToHost);
        if (runtime_copy_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] runtime validation D2H failed: "
                      << hipGetErrorString(runtime_copy_err));
            return false;
        }
        if (runtime_snapshot.active_bank > 1u ||
            runtime_snapshot.active_epoch == 0u ||
            runtime_snapshot.expert_count != static_cast<uint32_t>(num_experts) ||
            runtime_snapshot.top_k != static_cast<uint32_t>(top_k) ||
            runtime_snapshot.banks[runtime_snapshot.active_bank].epoch != runtime_snapshot.active_epoch ||
            runtime_snapshot.banks[runtime_snapshot.active_bank].expert_count != static_cast<uint32_t>(num_experts))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] invalid runtime layer state"
                      << " active_bank=" << runtime_snapshot.active_bank
                      << " active_epoch=" << runtime_snapshot.active_epoch
                      << " expert_count=" << runtime_snapshot.expert_count
                      << " top_k=" << runtime_snapshot.top_k);
            return false;
        }
#endif

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts), /*topk_count=*/0))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] route logits scratch allocation failed");
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] legacy outputs requested without tensors");
                return false;
            }

            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
            if (!legacy_indices || !legacy_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] legacy output tensors have no device allocation");
                return false;
            }
        }

        bool logits_ready = false;
        bool runtime_ready = false;
        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        if (gate_is_fp32 && rocm_env.moe_router_q8)
        {
            if ((d_model % 32) == 0 && ensureRouterQ8HiddenScratchCapacity(d_model))
            {
                if (const auto *q8_gate = getOrCreateQ8RouterGateCache(
                        gate_weights, static_cast<const float *>(g), d_model, num_experts))
                {
                    logits_ready = hipMoE_gate_logits_single_token_q8_weights(
                        h, d_router_q8_hidden_, d_router_q8_hidden_scales_,
                        q8_gate->d_gate_weights_q8, q8_gate->d_gate_scales,
                        d_route_logits_, d_model, num_experts,
                        device_ordinal_, getStream());
                    if (!logits_ready)
                    {
                        LOG_WARN("[ROCmMoEKernel::decodeRouteSelect] Q8 router logits kernel failed; falling back to K-part/FP16/default router");
                    }
                }
                else
                {
                    LOG_DEBUG("[ROCmMoEKernel::decodeRouteSelect] Q8 router cache unavailable; falling back to K-part/FP16/default router");
                }
            }
            else
            {
                LOG_DEBUG("[ROCmMoEKernel::decodeRouteSelect] Q8 router unsupported or scratch unavailable for d_model="
                          << d_model << "; falling back to K-part/FP16/default router");
            }
        }

        if (!logits_ready && gate_is_fp32 && rocm_env.moe_router_kpart_decode)
        {
            const int k_partitions = rocm_env.moe_router_kparts;
            const size_t partial_count = static_cast<size_t>(num_experts) * static_cast<size_t>(k_partitions);
            if (ensureRouteLogitsPartialsCapacity(partial_count))
            {
                const bool partials_ready = hipMoE_gate_logits_single_token_kpart_partials(
                    h, static_cast<const float *>(g), d_route_logits_partials_,
                    d_model, num_experts, k_partitions,
                    device_ordinal_, getStream());
                if (partials_ready)
                {
                    runtime_ready = hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
                        d_route_logits_partials_,
                        static_cast<void *>(runtime_layer),
                        legacy_indices,
                        legacy_weights,
                        num_experts,
                        k_partitions,
                        top_k,
                        normalize_weights,
                        write_legacy_outputs,
                        update_runtime_histogram,
                        device_ordinal_,
                        getStream());
                    if (!runtime_ready)
                    {
                        LOG_WARN("[ROCmMoEKernel::decodeRouteSelect] fused K-part router runtime kernel failed; falling back to FP16/default router");
                    }
                }
                else
                {
                    LOG_WARN("[ROCmMoEKernel::decodeRouteSelect] K-part router logits kernel failed; falling back to FP16/default router");
                }
            }
            else
            {
                LOG_WARN("[ROCmMoEKernel::decodeRouteSelect] K-part router scratch unavailable; falling back to FP16/default router");
            }
        }

        if (!runtime_ready && !logits_ready && gate_is_fp32)
        {
            if (const void *g_fp16 = getOrCreateFP16RouterGateCache(
                    gate_weights, static_cast<const float *>(g), d_model, num_experts))
            {
                logits_ready = hipMoE_gate_logits_single_token_fp16_weights(
                    h, g_fp16, d_route_logits_,
                    d_model, num_experts,
                    device_ordinal_, getStream());
                if (!logits_ready)
                {
                    LOG_WARN("[ROCmMoEKernel::decodeRouteSelect] FP16 router logits kernel failed; falling back to FP32 router");
                }
            }
        }

        if (!runtime_ready && !logits_ready &&
            !launchDecodeGateLogitsForGateType(
                h, g, gate_type, d_route_logits_,
                d_model, num_experts,
                device_ordinal_, getStream(),
                "ROCmMoEKernel::decodeRouteSelect"))
        {
            return false;
        }

        if (!runtime_ready && rocm_env.moe_router_wave_topk && num_experts <= 256)
        {
            runtime_ready = hipMoE_softmax_topk_decode_runtime_wave64(
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                device_ordinal_,
                getStream());
            if (!runtime_ready)
            {
                LOG_WARN("[ROCmMoEKernel::decodeRouteSelect] wave64 decode softmax/top-k runtime kernel failed; falling back to default runtime top-k");
            }
        }

        if (!runtime_ready && !hipMoE_softmax_topk_decode_runtime(
                                  d_route_logits_,
                                  static_cast<void *>(runtime_layer),
                                  legacy_indices,
                                  legacy_weights,
                                  num_experts,
                                  top_k,
                                  normalize_weights,
                                  write_legacy_outputs,
                                  update_runtime_histogram,
                                  device_ordinal_,
                                  getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] decode softmax/top-k runtime kernel failed");
            return false;
        }

        if (write_legacy_outputs)
        {
            output_indices->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            output_weights->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        }

        return true;
    }

    void ROCmMoEKernel::zeroBuffer(ITensor *tensor, size_t bytes)
    {
        if (!setMoEDevice(device_ordinal_, "zeroBuffer"))
            return;

        void *ptr = tensor->gpu_data_ptr();
        if (!ptr)
        {
            LOG_ERROR("[ROCmMoEKernel::zeroBuffer] tensor has no device allocation");
            return;
        }
        hipError_t err = hipMemsetAsync(ptr, 0, bytes, static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::zeroBuffer] hipMemsetAsync failed: " << hipGetErrorString(err));
            return;
        }
        tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                             DeviceId::rocm(device_ordinal_));
    }

    void ROCmMoEKernel::gatherTokenBatchFromTensors(
        ITensor *hidden, ITensor *batch_buffer,
        const int *host_token_indices, int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "gatherTokenBatchFromTensors"))
            return;

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        float *b = static_cast<float *>(batch_buffer->gpu_data_ptr());

        if (!h || !b)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] null device pointer");
            return;
        }
        // Upload host token indices to device staging buffer
        if (!ensureStagingCapacity(num_tokens))
            return;

        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatchFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }

        gatherTokenBatch(h, b, d_staging_indices_, num_tokens, d_model);
        batch_buffer->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                                   DeviceId::rocm(device_ordinal_));
    }

    void ROCmMoEKernel::scatterAddWeightedFromTensors(
        ITensor *output, ITensor *expert_output,
        const int *host_token_indices, const float *host_weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "scatterAddWeightedFromTensors"))
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *e = static_cast<const float *>(expert_output->gpu_data_ptr());

        if (!o || !e)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] null device pointer");
            return;
        }
        // Upload host indices + weights to device staging
        if (!ensureStagingCapacity(num_tokens))
            return;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            d_staging_indices_, host_token_indices,
            num_tokens * sizeof(int), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D token indices failed: "
                      << hipGetErrorString(err));
            return;
        }
        err = hipMemcpyAsync(
            d_staging_weights_, host_weights,
            num_tokens * sizeof(float), hipMemcpyHostToDevice,
            stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeightedFromTensors] H2D weights failed: "
                      << hipGetErrorString(err));
            return;
        }

        scatterAddWeighted(o, e, d_staging_indices_, d_staging_weights_,
                           num_tokens, d_model);
        output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                             DeviceId::rocm(device_ordinal_));
    }

    void ROCmMoEKernel::sharedExpertGateFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model)
    {
        if (!setMoEDevice(device_ordinal_, "sharedExpertGateFromTensors"))
            return;

        const float *in = static_cast<const float *>(input->gpu_data_ptr());
        const float *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        float *so = static_cast<float *>(shared_output->gpu_data_ptr());

        if (!in || !gi || !so)
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGateFromTensors] null device pointer");
            return;
        }

        sharedExpertGate(in, gi, so, seq_len, d_model);
        shared_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                                    DeviceId::rocm(device_ordinal_));
    }

    void ROCmMoEKernel::swiGLUFromTensors(ITensor *gate, ITensor *up, int count)
    {
        if (!setMoEDevice(device_ordinal_, "swiGLUFromTensors"))
            return;

        float *g = static_cast<float *>(gate->gpu_data_ptr());
        float *u = static_cast<float *>(up->gpu_data_ptr());

        if (!g || !u)
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLUFromTensors] null device pointer");
            return;
        }

        swiGLU(g, u, count);
        gate->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                           DeviceId::rocm(device_ordinal_));
    }

    void ROCmMoEKernel::weightedAddFromTensors(
        ITensor *output, ITensor *input, float weight, int count)
    {
        if (!setMoEDevice(device_ordinal_, "weightedAddFromTensors"))
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *in = static_cast<const float *>(input->gpu_data_ptr());

        if (!o || !in)
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAddFromTensors] null device pointer");
            return;
        }

        weightedAdd(o, in, weight, count);
        output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                             DeviceId::rocm(device_ordinal_));
    }

    int ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable(
        const DeviceNativeVNNIMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs || num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (intermediate % 32) != 0)
        {
            return -1;
        }

        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertDownDescriptorTable"))
            return -1;

        uint8_t codebook_id = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (!desc.valid())
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }

            if (expert_id == 0)
            {
                codebook_id = desc.codebook_id;
                if (!groupedDecodeSupportsCodebook(codebook_id))
                {
                    LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] Unsupported native-VNNI codebook "
                              << static_cast<int>(codebook_id));
                    return -1;
                }
            }

            if (!validateGroupedDownDesc(desc, d_model, intermediate, codebook_id))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }
        }

        GroupedDownDescriptorTable table;
        table.host_descs.assign(down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;

        DeviceNativeVNNIMatrixDesc *device_descs = nullptr;
        hipError_t err = hipMalloc(&device_descs,
                                   static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] hipMalloc failed: "
                      << hipGetErrorString(err));
            return -1;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        err = hipMemcpyAsync(device_descs, table.host_descs.data(),
                             static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc),
                             hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::uploadGroupedExpertDownDescriptorTable] H2D descriptor table upload failed: "
                      << hipGetErrorString(err));
            (void)hipFree(device_descs);
            return -1;
        }

        table.device_descs = device_descs;
        table.valid = true;
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    int ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables(
        const DeviceNativeVNNIMatrixDesc *gate_descs,
        const DeviceNativeVNNIMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs || num_experts <= 0 || d_model <= 0 ||
            intermediate <= 0 || (d_model % 32) != 0)
        {
            return -1;
        }

        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertGateUpDescriptorTables"))
            return -1;

        uint8_t codebook_id = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            if (!gate_desc.valid() || !up_desc.valid())
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (gate_desc.codebook_id != up_desc.codebook_id)
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Gate/up codebook mismatch for expert "
                          << expert_id << ": gate=" << static_cast<int>(gate_desc.codebook_id)
                          << " up=" << static_cast<int>(up_desc.codebook_id));
                return -1;
            }

            if (expert_id == 0)
            {
                codebook_id = gate_desc.codebook_id;
                if (!groupedDecodeSupportsCodebook(codebook_id))
                {
                    LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Unsupported native-VNNI codebook "
                              << static_cast<int>(codebook_id));
                    return -1;
                }
            }

            if (!validateGroupedGateUpDesc(gate_desc, d_model, intermediate, codebook_id) ||
                !validateGroupedGateUpDesc(up_desc, d_model, intermediate, codebook_id))
            {
                LOG_DEBUG("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Invalid descriptor pair for expert "
                          << expert_id);
                return -1;
            }
        }

        GroupedGateUpDescriptorTable table;
        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;

        DeviceNativeVNNIMatrixDesc *device_gate_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *device_up_descs = nullptr;
        hipError_t err = hipMalloc(&device_gate_descs,
                                   static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc));
        if (err == hipSuccess)
            err = hipMalloc(&device_up_descs,
                            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] hipMalloc failed: "
                      << hipGetErrorString(err));
            if (device_gate_descs)
                (void)hipFree(device_gate_descs);
            if (device_up_descs)
                (void)hipFree(device_up_descs);
            return -1;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        err = hipMemcpyAsync(device_gate_descs, table.host_gate_descs.data(),
                             static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc),
                             hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(device_up_descs, table.host_up_descs.data(),
                                 static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::uploadGroupedExpertGateUpDescriptorTables] H2D descriptor upload failed: "
                      << hipGetErrorString(err));
            (void)hipFree(device_gate_descs);
            (void)hipFree(device_up_descs);
            return -1;
        }

        table.device_gate_descs = device_gate_descs;
        table.device_up_descs = device_up_descs;
        table.valid = true;
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromTable(
        const TensorBase *input,
        const int *expert_ids,
        int descriptor_table_id,
        int num_active,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || !expert_ids || descriptor_table_id < 0 || num_active <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16 || (d_model % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        for (int i = 0; i < num_active; ++i)
        {
            const int expert_id = expert_ids[i];
            if (expert_id < 0 || expert_id >= table.num_experts)
            {
                return false;
            }
        }

        if (!ensureGroupedGateUpCapacity(num_active, d_model))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromTable"))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden)
        {
            if (!const_cast<TensorBase *>(input)->ensureOnDevice(DeviceId::rocm(device_ordinal_), getStream()))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        if (!d_hidden)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] Input tensor has no device pointer");
            return false;
        }

        host_grouped_gate_output_ptrs_.assign(static_cast<size_t>(num_active), nullptr);
        host_grouped_up_output_ptrs_.assign(static_cast<size_t>(num_active), nullptr);
        host_grouped_gateup_expert_ids_.assign(expert_ids, expert_ids + num_active);
        for (int i = 0; i < num_active; ++i)
        {
            host_grouped_gate_output_ptrs_[i] = static_cast<float *>(gate_outputs[i]->gpu_data_ptr());
            host_grouped_up_output_ptrs_[i] = static_cast<float *>(up_outputs[i]->gpu_data_ptr());
            if (!host_grouped_gate_output_ptrs_[i] || !host_grouped_up_output_ptrs_[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] Null gate/up output pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_output_ptrs_, host_grouped_gate_output_ptrs_.data(),
                                        static_cast<size_t>(num_active) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_output_ptrs_, host_grouped_up_output_ptrs_.data(),
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_gateup_expert_ids_, host_grouped_gateup_expert_ids_.data(),
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromTable] H2D staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        const bool ok = rocmMoE_grouped_gate_up_native_vnni_decode_table(
            d_hidden,
            table.device_gate_descs,
            table.device_up_descs,
            d_grouped_gateup_expert_ids_,
            d_grouped_gate_output_ptrs_,
            d_grouped_up_output_ptrs_,
            d_grouped_hidden_int8_,
            d_grouped_hidden_scales_,
            num_active,
            intermediate,
            d_model,
            table.codebook_id,
            device_ordinal_,
            getStream());

        if (ok)
        {
            for (int i = 0; i < num_active; ++i)
            {
                gate_outputs[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
                up_outputs[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        int descriptor_table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!input || !routing_indices || descriptor_table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > 16 || (d_model % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedGateUpCapacity(top_k, d_model))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRouting"))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden)
        {
            if (!const_cast<TensorBase *>(input)->ensureOnDevice(DeviceId::rocm(device_ordinal_), getStream()))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        const float *d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_routing_indices)
        {
            if (!routing_indices->ensureOnDevice(DeviceId::rocm(device_ordinal_)))
                return false;
            d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        }
        if (!d_hidden || !d_routing_indices)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] Missing input/routing device pointer");
            return false;
        }

        host_grouped_gate_output_ptrs_.assign(static_cast<size_t>(top_k), nullptr);
        host_grouped_up_output_ptrs_.assign(static_cast<size_t>(top_k), nullptr);
        for (int i = 0; i < top_k; ++i)
        {
            host_grouped_gate_output_ptrs_[i] = static_cast<float *>(gate_outputs[i]->gpu_data_ptr());
            host_grouped_up_output_ptrs_[i] = static_cast<float *>(up_outputs[i]->gpu_data_ptr());
            if (!host_grouped_gate_output_ptrs_[i] || !host_grouped_up_output_ptrs_[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] Null gate/up output pointer for slot "
                          << i);
                return false;
            }
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_output_ptrs_, host_grouped_gate_output_ptrs_.data(),
                                        static_cast<size_t>(top_k) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_output_ptrs_, host_grouped_up_output_ptrs_.data(),
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRouting] H2D pointer staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        if (!hipMoE_float_to_int(
                d_routing_indices, d_grouped_gateup_expert_ids_, top_k,
                device_ordinal_, getStream()))
        {
            return false;
        }

        const bool ok = rocmMoE_grouped_gate_up_native_vnni_decode_table(
            d_hidden,
            table.device_gate_descs,
            table.device_up_descs,
            d_grouped_gateup_expert_ids_,
            d_grouped_gate_output_ptrs_,
            d_grouped_up_output_ptrs_,
            d_grouped_hidden_int8_,
            d_grouped_hidden_scales_,
            top_k,
            intermediate,
            d_model,
            table.codebook_id,
            device_ordinal_,
            getStream());

        if (ok)
        {
            for (int i = 0; i < top_k; ++i)
            {
                gate_outputs[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
                up_outputs[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int descriptor_table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_FFN, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !input || descriptor_table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) || (d_model % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[descriptor_table_id];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedGateUpCapacity(top_k, d_model))
            return false;

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRuntime"))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden)
        {
            if (!const_cast<TensorBase *>(input)->ensureOnDevice(DeviceId::rocm(device_ordinal_), getStream()))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }
        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        if (!d_hidden || !d_expert_ids)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime] Missing input/runtime device pointer");
            return false;
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_output_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_output_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_output_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_output_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_output_ptrs[slot] || !up_output_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime] Null gate/up output pointer for slot "
                          << slot);
                return false;
            }
        }

        float **d_gate_output_ptrs = nullptr;
        float **d_up_output_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(
                descriptor_table_id, top_k, gate_output_ptrs, up_output_ptrs,
                &d_gate_output_ptrs, &d_up_output_ptrs))
        {
            return false;
        }

        const int k_partitions = debugEnv().rocm.moe_gateup_kparts;
        const bool use_kpart_gateup =
            debugEnv().rocm.moe_gateup_kpart_decode &&
            top_k > 0 &&
            groupedDecodeSupportsCodebook(table.codebook_id) &&
            ensureGroupedGateUpKPartScratchCapacity(top_k, k_partitions, intermediate);

        bool ok = false;
        if (use_kpart_gateup)
        {
            ok = rocmMoE_grouped_gate_up_native_vnni_decode_table_kpart(
                d_hidden,
                table.device_gate_descs,
                table.device_up_descs,
                d_expert_ids,
                d_gate_output_ptrs,
                d_up_output_ptrs,
                d_grouped_hidden_int8_,
                d_grouped_hidden_scales_,
                d_grouped_gateup_gate_partials_,
                d_grouped_gateup_up_partials_,
                top_k,
                intermediate,
                d_model,
                table.codebook_id,
                k_partitions,
                device_ordinal_,
                getStream());
            if (!ok)
            {
                LOG_DEBUG("[ROCmMoEKernel::groupedExpertGateUpDecodeFromRuntime] K-partition gate/up path failed; falling back to serial path");
            }
        }

        if (!ok)
        {
            ok = rocmMoE_grouped_gate_up_native_vnni_decode_table(
                d_hidden,
                table.device_gate_descs,
                table.device_up_descs,
                d_expert_ids,
                d_gate_output_ptrs,
                d_up_output_ptrs,
                d_grouped_hidden_int8_,
                d_grouped_hidden_scales_,
                top_k,
                intermediate,
                d_model,
                table.codebook_id,
                device_ordinal_,
                getStream());
        }

        if (ok)
        {
            for (int i = 0; i < top_k; ++i)
            {
                gate_outputs[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
                up_outputs[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
        }
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromTable(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        int descriptor_table_id,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            !output || descriptor_table_id < 0 || num_active <= 0 ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16 || (intermediate % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        for (int i = 0; i < num_active; ++i)
        {
            const int expert_id = expert_ids[i];
            if (expert_id < 0 || expert_id >= table.num_experts)
            {
                return false;
            }
        }

        if (!ensureGroupedDecodeCapacity(num_active, intermediate))
            return false;

        const float *host_gate_ptrs[16] = {};
        const float *host_up_ptrs[16] = {};
        for (int i = 0; i < num_active; ++i)
        {
            host_gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            host_up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!host_gate_ptrs[i] || !host_up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromTable] Null gate/up device pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromTable] Output tensor has no device pointer");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_ptrs_, host_gate_ptrs,
                                        static_cast<size_t>(num_active) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_ptrs_, host_up_ptrs,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_expert_ids_, expert_ids,
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_decode_weights_, expert_weights,
                                 static_cast<size_t>(num_active) * sizeof(float),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromTable] H2D staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
            d_grouped_gate_ptrs_,
            d_grouped_up_ptrs_,
            table.device_descs,
            d_grouped_expert_ids_,
            d_grouped_decode_weights_,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            d_output,
            num_active,
            d_model,
            intermediate,
            table.codebook_id,
            device_ordinal_,
            getStream());

        if (ok)
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromRouting(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !routing_indices || !routing_weights ||
            !output || descriptor_table_id < 0 || top_k <= 0 ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > 16 || (intermediate % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedDecodeCapacity(top_k, intermediate))
            return false;

        const float *host_gate_ptrs[16] = {};
        const float *host_up_ptrs[16] = {};
        for (int i = 0; i < top_k; ++i)
        {
            host_gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            host_up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!host_gate_ptrs[i] || !host_up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] Null gate/up device pointer for slot "
                          << i);
                return false;
            }
        }

        const float *d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_routing_indices)
        {
            if (!routing_indices->ensureOnDevice(DeviceId::rocm(device_ordinal_)))
                return false;
            d_routing_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        }
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_weights)
        {
            if (!routing_weights->ensureOnDevice(DeviceId::rocm(device_ordinal_)))
                return false;
            d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_routing_indices || !d_weights || !d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] Missing routing/output device pointer");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_ptrs_, host_gate_ptrs,
                                        static_cast<size_t>(top_k) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_ptrs_, host_up_ptrs,
                                 static_cast<size_t>(top_k) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRouting] H2D pointer staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        if (!hipMoE_float_to_int(
                d_routing_indices, d_grouped_expert_ids_, top_k,
                device_ordinal_, getStream()))
        {
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
            d_grouped_gate_ptrs_,
            d_grouped_up_ptrs_,
            table.device_descs,
            d_grouped_expert_ids_,
            d_weights,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            d_output,
            top_k,
            d_model,
            intermediate,
            table.codebook_id,
            device_ordinal_,
            getStream());

        if (ok)
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecodeFromRuntime(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        DeviceMoELayerRuntime *runtime_layer,
        int descriptor_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !runtime_layer || !output ||
            descriptor_table_id < 0 || top_k <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) || (intermediate % 32) != 0)
            return false;
        if (descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[descriptor_table_id];
        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            return false;
        }

        if (!ensureGroupedDecodeCapacity(top_k, intermediate))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRuntime] Null gate/up device pointer for slot "
                          << slot);
                return false;
            }
        }

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        const float *d_weights = runtimeTopKWeightsDevice(runtime_layer);
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_expert_ids || !d_weights || !d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecodeFromRuntime] Missing runtime/output device pointer");
            return false;
        }

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(
                descriptor_table_id, top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        const bool use_parallel_down =
            debugEnv().rocm.moe_parallel_down_decode &&
            top_k > 1 &&
            groupedDecodeSupportsCodebook(table.codebook_id);

        const bool ok = use_parallel_down
                            ? rocmMoE_grouped_swiglu_down_native_vnni_decode_table_parallel(
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  table.device_descs,
                                  d_expert_ids,
                                  d_weights,
                                  d_grouped_swiglu_int8_,
                                  d_grouped_swiglu_scales_,
                                  d_output,
                                  top_k,
                                  d_model,
                                  intermediate,
                                  table.codebook_id,
                                  device_ordinal_,
                                  getStream())
                            : rocmMoE_grouped_swiglu_down_native_vnni_decode_table(
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  table.device_descs,
                                  d_expert_ids,
                                  d_weights,
                                  d_grouped_swiglu_int8_,
                                  d_grouped_swiglu_scales_,
                                  d_output,
                                  top_k,
                                  d_model,
                                  intermediate,
                                  table.codebook_id,
                                  device_ordinal_,
                                  getStream());

        if (ok)
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        return ok;
    }

    bool ROCmMoEKernel::groupedExpertDownDecode(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        const DeviceNativeVNNIMatrixDesc *down_descs,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            !down_descs || !output || num_active <= 0 || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (num_active > 16 || (intermediate % 32) != 0)
        {
            return false;
        }

        const uint8_t codebook_id = down_descs[0].codebook_id;
        for (int i = 0; i < num_active; ++i)
        {
            if (expert_ids[i] < 0 || !down_descs[i].valid() ||
                down_descs[i].n != d_model || down_descs[i].k != intermediate ||
                down_descs[i].blocks_per_row != static_cast<uint32_t>(intermediate / 32) ||
                down_descs[i].codebook_id != codebook_id ||
                (groupedDecodeRequiresMins(down_descs[i].codebook_id) && !down_descs[i].mins) ||
                (groupedDecodeRequiresEmins(down_descs[i].codebook_id) && !down_descs[i].emins))
            {
                return false;
            }
        }

        if (!groupedDecodeSupportsCodebook(codebook_id))
        {
            LOG_DEBUG("[ROCmMoEKernel::groupedExpertDownDecode] Unsupported native-VNNI codebook "
                      << static_cast<int>(codebook_id) << "; using fallback");
            return false;
        }

        if (!ensureGroupedDecodeCapacity(num_active, intermediate))
            return false;

        const float *host_gate_ptrs[16] = {};
        const float *host_up_ptrs[16] = {};
        for (int i = 0; i < num_active; ++i)
        {
            host_gate_ptrs[i] = static_cast<const float *>(gate_tensors[i]->gpu_data_ptr());
            host_up_ptrs[i] = static_cast<const float *>(up_tensors[i]->gpu_data_ptr());
            if (!host_gate_ptrs[i] || !host_up_ptrs[i])
            {
                LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Null gate/up device pointer for active slot "
                          << i << " expert=" << expert_ids[i]);
                return false;
            }
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] Output tensor has no device pointer");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_grouped_gate_ptrs_, host_gate_ptrs,
                                        static_cast<size_t>(num_active) * sizeof(float *),
                                        hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_up_ptrs_, host_up_ptrs,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_decode_weights_, expert_weights,
                                 static_cast<size_t>(num_active) * sizeof(float),
                                 hipMemcpyHostToDevice, stream);
        if (err == hipSuccess)
            err = hipMemcpyAsync(d_grouped_down_descs_, down_descs,
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeVNNIMatrixDesc),
                                 hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::groupedExpertDownDecode] H2D staging failed: "
                      << hipGetErrorString(err));
            return false;
        }

        const bool ok = rocmMoE_grouped_swiglu_down_native_vnni_decode(
            d_grouped_gate_ptrs_,
            d_grouped_up_ptrs_,
            d_grouped_down_descs_,
            d_grouped_decode_weights_,
            d_grouped_swiglu_int8_,
            d_grouped_swiglu_scales_,
            d_output,
            num_active,
            d_model,
            intermediate,
            codebook_id,
            device_ordinal_,
            getStream());

        if (ok)
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        return ok;
    }

    // =========================================================================
    // Phase 4: GPU-side expert dispatch for prefill
    // =========================================================================

    bool ROCmMoEKernel::groupPrefillRoutes(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices, ITensor *routing_weights,
        int current_tokens, int max_tokens,
        int num_experts, int top_k)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !routing_indices || !routing_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] null runtime or routing tensor");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] invalid dimensions current_tokens=" << current_tokens
                                                                                               << " max_tokens=" << max_tokens
                                                                                               << " num_experts=" << num_experts
                                                                                               << " top_k=" << top_k);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "groupPrefillRoutes"))
            return false;

        const float *d_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_indices || !d_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::groupPrefillRoutes] routing tensors have no device pointers");
            return false;
        }

        return hipMoE_group_prefill_routes_runtime(
            d_indices,
            d_weights,
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            device_ordinal_,
            getStream());
    }

    bool ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *batch_buffer,
        int expert_id, int max_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !hidden || !batch_buffer)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] invalid arguments expert_id=" << expert_id
                                                                                                          << " max_tokens=" << max_tokens
                                                                                                          << " d_model=" << d_model);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "gatherPrefillExpertBatchFromRuntime"))
            return false;

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_batch = static_cast<float *>(batch_buffer->gpu_data_ptr());
        if (!d_hidden || !d_batch)
        {
            LOG_ERROR("[ROCmMoEKernel::gatherPrefillExpertBatchFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = hipMoE_prefill_gather_expert_runtime(
            static_cast<const void *>(runtime_layer),
            d_hidden,
            d_batch,
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            getStream());
        if (ok)
            batch_buffer->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        return ok;
    }

    bool ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime(
        ITensor *output, ITensor *expert_results,
        DeviceMoELayerRuntime *runtime_layer,
        int expert_id, int max_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (!output || !expert_results || !runtime_layer)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] invalid arguments expert_id=" << expert_id
                                                                                                             << " max_tokens=" << max_tokens
                                                                                                             << " d_model=" << d_model);
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "scatterPrefillExpertResultsFromRuntime"))
            return false;

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const float *d_expert_output = static_cast<const float *>(expert_results->gpu_data_ptr());
        if (!d_output || !d_expert_output)
        {
            LOG_ERROR("[ROCmMoEKernel::scatterPrefillExpertResultsFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = hipMoE_prefill_scatter_expert_runtime(
            d_output,
            d_expert_output,
            static_cast<const void *>(runtime_layer),
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            getStream());
        if (ok)
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        return ok;
    }

    bool ROCmMoEKernel::prepareExpertGroups(
        ITensor *routing_indices, ITensor *routing_weights,
        int seq_len, int num_experts, int top_k)
    {
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "prepareExpertGroups"))
            return false;

        const int total_slots = seq_len * top_k;
        hipStream_t stream = static_cast<hipStream_t>(getStream());

        // 1. Ensure routing tensors are on device
        routing_indices->ensureOnDevice(DeviceId::rocm(device_ordinal_));
        routing_weights->ensureOnDevice(DeviceId::rocm(device_ordinal_));

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroups] null device pointers");
            return false;
        }

        // 2. Lazy-allocate grouping buffers
        if (total_slots > group_slots_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroups(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroups(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(total_slots) * sizeof(float),
                                     "prepareExpertGroups(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroups(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_active_expert_slots_ = 0;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        if (num_experts > group_experts_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroups(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroups(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareExpertGroups(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = num_experts;
        }

        // 3. Convert float indices → int on device
        if (!hipMoE_float_to_int(d_float_indices, d_group_int_indices_,
                                 total_slots, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroups] float_to_int failed");
            return false;
        }

        // 4. Group tokens by expert (counts, offsets, scatter)
        if (!groupTokensByExpertDevice(
                d_group_int_indices_, d_float_weights,
                seq_len, num_experts, top_k,
                d_group_offsets_, d_group_counts_,
                d_group_token_indices_, d_group_weights_))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroups] groupTokensByExpertDevice failed");
            return false;
        }

        // 5. D2H expert counts and offsets (small — num_experts ints each)
        host_expert_counts_.resize(num_experts);
        host_expert_offsets_.resize(num_experts);
        hipError_t copy_err = hipMemcpyAsync(host_expert_counts_.data(), d_group_counts_,
                                             num_experts * sizeof(int), hipMemcpyDeviceToHost, stream);
        if (copy_err == hipSuccess)
            copy_err = hipMemcpyAsync(host_expert_offsets_.data(), d_group_offsets_,
                                      num_experts * sizeof(int), hipMemcpyDeviceToHost, stream);
        if (copy_err == hipSuccess)
            copy_err = hipStreamSynchronize(stream);
        if (copy_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroups] D2H grouping metadata failed: "
                      << hipGetErrorString(copy_err));
            return false;
        }

        prepared_num_experts_ = num_experts;
        return true;
    }

    int ROCmMoEKernel::getExpertTokenCount(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= prepared_num_experts_)
            return 0;
        return host_expert_counts_[expert_id];
    }

    void ROCmMoEKernel::gatherExpertBatch(
        ITensor *hidden, ITensor *batch_buffer,
        int expert_id, int d_model)
    {
        int count = getExpertTokenCount(expert_id);
        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "gatherExpertBatch"))
            return;

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        float *b = static_cast<float *>(batch_buffer->gpu_data_ptr());
        int offset = host_expert_offsets_[expert_id];

        gatherTokenBatch(h, b, d_group_token_indices_ + offset, count, d_model);
    }

    void ROCmMoEKernel::scatterExpertResults(
        ITensor *output, ITensor *expert_results,
        int expert_id, int d_model)
    {
        int count = getExpertTokenCount(expert_id);
        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "scatterExpertResults"))
            return;

        float *o = static_cast<float *>(output->gpu_data_ptr());
        const float *r = static_cast<const float *>(expert_results->gpu_data_ptr());
        int offset = host_expert_offsets_[expert_id];

        scatterAddWeighted(o, r, d_group_token_indices_ + offset,
                           d_group_weights_ + offset, count, d_model);
    }

    // =========================================================================
    // Phase 5: Fully-grouped MoE prefill pipeline (graph-capturable)
    // =========================================================================

    bool ROCmMoEKernel::prepareExpertGroupsAsync(
        ITensor *routing_indices, ITensor *routing_weights,
        int seq_len, int num_experts, int top_k)
    {
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "prepareExpertGroupsAsync"))
            return false;

        const int total_slots = seq_len * top_k;

        // 1. Ensure routing tensors are on device
        routing_indices->ensureOnDevice(DeviceId::rocm(device_ordinal_));
        routing_weights->ensureOnDevice(DeviceId::rocm(device_ordinal_));

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] null device pointers");
            return false;
        }

        // 2. Lazy-allocate grouping buffers
        if (total_slots > group_slots_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_int_indices_),
                                     MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_int_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_token_indices_),
                                     MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_token_indices)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_weights_),
                                     MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                     static_cast<size_t>(total_slots) * sizeof(float),
                                     "prepareExpertGroupsAsync(group_weights)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_active_expert_ids_),
                                     MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                     static_cast<size_t>(total_slots) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_active_expert_ids)"))
            {
                d_group_int_indices_ = nullptr;
                d_group_token_indices_ = nullptr;
                d_group_weights_ = nullptr;
                d_group_active_expert_ids_ = nullptr;
                group_active_expert_slots_ = 0;
                group_slots_cap_ = 0;
                return false;
            }
            group_slots_cap_ = total_slots;
        }
        if (num_experts > group_experts_cap_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_offsets_),
                                     MoEWorkspaceBuffers::GROUP_OFFSETS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_offsets)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_counts_),
                                     MoEWorkspaceBuffers::GROUP_COUNTS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "prepareExpertGroupsAsync(group_counts)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_max_tokens_),
                                     MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                                     sizeof(int),
                                     "prepareExpertGroupsAsync(group_max_tokens)"))
            {
                d_group_offsets_ = nullptr;
                d_group_counts_ = nullptr;
                d_group_max_tokens_ = nullptr;
                group_experts_cap_ = 0;
                return false;
            }
            group_experts_cap_ = num_experts;
        }

        const bool use_small_grouping =
            total_slots <= 64 &&
            num_experts <= 256 &&
            top_k <= 16;
        if (use_small_grouping)
        {
            const int active_expert_slots = std::min(total_slots, num_experts);
            if (!hipMoE_group_tokens_small_float(
                    d_float_indices,
                    d_float_weights,
                    d_group_counts_,
                    d_group_offsets_,
                    d_group_token_indices_,
                    d_group_weights_,
                    d_group_active_expert_ids_,
                    total_slots,
                    num_experts,
                    top_k,
                    active_expert_slots,
                    device_ordinal_,
                    getStream()))
            {
                LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] small-M grouping failed");
                group_active_expert_slots_ = 0;
                return false;
            }
            group_active_expert_slots_ = active_expert_slots;

            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "rocm_moe_small_prefill_grouping_calls",
                    1.0,
                    "moe",
                    DeviceId::rocm(device_ordinal_).to_string(),
                    PerfStatsCollector::Tags{
                        {"total_slots", std::to_string(total_slots)},
                        {"num_experts", std::to_string(num_experts)},
                        {"top_k", std::to_string(top_k)}});
            }

            prepared_num_experts_ = num_experts;
            return true;
        }

        // 3. Convert float indices → int on device
        group_active_expert_slots_ = 0;
        if (!hipMoE_float_to_int(d_float_indices, d_group_int_indices_,
                                 total_slots, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] float_to_int failed");
            return false;
        }

        // 4. Group tokens by expert (counts, offsets, scatter) — all on device
        if (!groupTokensByExpertDevice(
                d_group_int_indices_, d_float_weights,
                seq_len, num_experts, top_k,
                d_group_offsets_, d_group_counts_,
                d_group_token_indices_, d_group_weights_))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] groupTokensByExpertDevice failed");
            return false;
        }

        // Launch device-side max-reduction over expert counts (async, no sync).
        // Result in d_group_max_tokens_ — consumed by GEMM grid early-exit.
        if (!hipMoE_max_expert_count(d_group_counts_, d_group_max_tokens_,
                                     num_experts, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::prepareExpertGroupsAsync] max_expert_count failed");
            return false;
        }

        // NO D2H copy, NO hipStreamSynchronize — data stays on device
        // for consumption by executeGroupedPrefillPipeline()

        prepared_num_experts_ = num_experts;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate)
    {
        if (!setMoEDevice(device_ordinal_, "ensureGroupedPrefillScratchCapacity"))
            return false;

        const bool need_realloc = (total_slots > prefill_slots_cap_ ||
                                   d_model > prefill_d_model_cap_ ||
                                   intermediate > prefill_intermediate_cap_);
        if (!need_realloc)
            return true;

        const int max_dim = (d_model > intermediate) ? d_model : intermediate;
        const int max_blocks = max_dim / 32;

        // K1 produces A_int8/scales and K2 consumes them before K3 starts.
        // Reuse the same buffers for K3's SwiGLU quantized output to avoid
        // holding a second full route-slot INT8 activation at prefill length.
        // K2 writes gate/up; K3 consumes both, then K4 can reuse gate as the
        // down-projection output buffer because it is sized for max(d_model,intermediate).
        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_A_int8_),
                                 MoEWorkspaceBuffers::PREFILL_A_INT8,
                                 static_cast<size_t>(total_slots) * max_dim * sizeof(int8_t),
                                 "ensureGroupedPrefillScratchCapacity(a_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_A_scales_),
                                 MoEWorkspaceBuffers::PREFILL_A_SCALES,
                                 static_cast<size_t>(total_slots) * max_blocks * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(a_scales)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_gate_),
                                 MoEWorkspaceBuffers::PREFILL_GATE,
                                 static_cast<size_t>(total_slots) * max_dim * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(gate)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_prefill_up_),
                                 MoEWorkspaceBuffers::PREFILL_UP,
                                 static_cast<size_t>(total_slots) * intermediate * sizeof(float),
                                 "ensureGroupedPrefillScratchCapacity(up)"))
        {
            d_prefill_A_int8_ = nullptr;
            d_prefill_A_scales_ = nullptr;
            d_prefill_gate_ = nullptr;
            d_prefill_up_ = nullptr;
            prefill_slots_cap_ = 0;
            prefill_d_model_cap_ = 0;
            prefill_intermediate_cap_ = 0;
            return false;
        }

        prefill_slots_cap_ = total_slots;
        prefill_d_model_cap_ = d_model;
        prefill_intermediate_cap_ = intermediate;
        return true;
    }

    bool ROCmMoEKernel::executeGroupedPrefillPipeline(
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k)
    {
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "executeGroupedPrefillPipeline"))
            return false;

        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] invalid gateup descriptor table id "
                      << gateup_desc_table_id);
            return false;
        }
        if (down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] invalid down descriptor table id "
                      << down_desc_table_id);
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];

        if (!gateup_table.valid || !down_table.valid)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] descriptor tables not valid");
            return false;
        }

        // Grid sizing for K2/K4 GEMM kernels.
        // Use seq_len as conservative upper bound — no host sync needed.
        // Each layer has independent routing (different gate weights), so
        // max_tokens_per_expert varies per layer and per input. A cache would
        // need (layer_id, input_hash) as key — not worth the complexity.
        // The K2/K4 kernels early-exit via per-expert d_group_counts check,
        // so overprovision only costs empty-block dispatch (~4ms / 437ms < 1%).
        // The device-side max (d_group_max_tokens_) is still computed async for
        // potential future use (e.g., separate-stream approach or graph capture).
        const int total_slots = seq_len * top_k;
        const int max_tokens_per_expert = seq_len;
        const int active_expert_slots = group_active_expert_slots_;
        const int *d_active_expert_ids =
            (active_expert_slots > 0) ? d_group_active_expert_ids_ : nullptr;
        if (active_expert_slots > 0 && !d_active_expert_ids)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] compact active expert list missing");
            return false;
        }

        // Ensure scratch buffers
        if (!ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate))
            return false;

        // Ensure hidden and output are on device
        hidden->ensureOnDevice(DeviceId::rocm(device_ordinal_));
        output->ensureOnDevice(DeviceId::rocm(device_ordinal_));

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_output = static_cast<float *>(output->gpu_data_ptr());

        if (isGraphCaptureActive() && (!d_hidden || !d_output))
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] null device pointers during graph capture");
            return false;
        }

        // Zero the output buffer (pre-zero requirement for scatter-add)
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t memset_err = hipMemsetAsync(d_output, 0, static_cast<size_t>(seq_len) * d_model * sizeof(float), stream);
        if (memset_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] output zero failed: "
                      << hipGetErrorString(memset_err));
            return false;
        }

        // Call the fully-grouped pipeline (5 kernel launches, zero sync)
        const bool ok = rocmMoE_grouped_prefill_pipeline(
            d_hidden,
            gateup_table.device_gate_descs,
            gateup_table.device_up_descs,
            down_table.device_descs,
            d_group_counts_,
            d_group_offsets_,
            d_group_token_indices_,
            d_group_weights_,
            d_active_expert_ids,
            d_prefill_A_int8_,
            d_prefill_A_scales_,
            d_prefill_gate_,
            d_prefill_up_,
            d_prefill_A_int8_,
            d_prefill_A_scales_,
            d_prefill_gate_,
            d_output,
            num_experts,
            d_model,
            intermediate,
            max_tokens_per_expert,
            total_slots,
            active_expert_slots,
            gateup_table.codebook_id,
            down_table.codebook_id,
            device_ordinal_,
            getStream());

        if (!ok)
        {
            LOG_ERROR("[ROCmMoEKernel::executeGroupedPrefillPipeline] pipeline failed for layer");
            return false;
        }

        output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                             DeviceId::rocm(device_ordinal_));
        if (PerfStatsCollector::isEnabled() && active_expert_slots > 0)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_grouped_prefill_active_expert_grid_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                PerfStatsCollector::Tags{
                    {"total_slots", std::to_string(total_slots)},
                    {"active_expert_slots", std::to_string(active_expert_slots)},
                    {"num_experts", std::to_string(num_experts)}});
        }
        return true;
    }

} // namespace llaminar2
