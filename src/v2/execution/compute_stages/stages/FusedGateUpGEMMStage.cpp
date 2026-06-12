/**
 * @file FusedGateUpGEMMStage.cpp
 * @brief Implementation of FusedGateUpGEMMStage
 */

#include "FusedGateUpGEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"
#include "../../../utils/GemmContext.h"
#include "../../../loaders/PreparedWeightStore.h"

namespace llaminar2
{

    // =============================================================================
    // FusedGateUpGEMMStage Implementation
    // =============================================================================

    FusedGateUpGEMMStage::FusedGateUpGEMMStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool FusedGateUpGEMMStage::validatePreparedWeights(std::string *error) const
    {
        if (!params_.w_gate && !params_.w_up)
            return true;
        if (!params_.prepared_store ||
            !params_.prepared_ref_gate.has_value() ||
            !params_.prepared_ref_up.has_value())
        {
            if (error) *error = "FusedGateUpGEMMStage requires PreparedWeightStore and gate/up PreparedWeightRefs";
            return false;
        }
        if (!params_.prepared_store->contains(params_.prepared_ref_gate.value()) ||
            !params_.prepared_store->contains(params_.prepared_ref_up.value()))
        {
            if (error) *error = "FusedGateUpGEMMStage has a PreparedWeightRef missing from PreparedWeightStore";
            return false;
        }
        return true;
    }

    ITensorFusedGateUpGemm *FusedGateUpGEMMStage::resolvePreparedKernel(const char *caller)
    {
        if (cached_kernel_)
            return cached_kernel_;

        if (!params_.prepared_store ||
            !params_.prepared_ref_gate.has_value() ||
            !params_.prepared_ref_up.has_value())
        {
            LOG_ERROR("[" << caller << "] PreparedWeightStore and gate/up PreparedWeightRefs are required");
            return nullptr;
        }

        cached_kernel_ = params_.prepared_store->fusedGateUpKernel(
            params_.prepared_ref_gate.value(),
            params_.prepared_ref_up.value());
        if (!cached_kernel_)
        {
            LOG_ERROR("[" << caller << "] PreparedWeightRefs were provided but no fused gate/up kernel was found in PreparedWeightStore");
        }
        return cached_kernel_;
    }

    bool FusedGateUpGEMMStage::execute(IDeviceContext *ctx)
    {
        ScopedGemmContext gemm_ctx(GemmContext::FFN);

        LOG_DEBUG("[FusedGateUpGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                       << " n_gate=" << params_.n_gate << " n_up=" << params_.n_up);

        if (!ctx)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.input)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null input");
            return false;
        }
        if (!params_.w_gate || !params_.w_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null weight tensor(s)");
            return false;
        }
        if (!params_.output_gate || !params_.output_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null output buffer(s)");
            return false;
        }
        if (params_.m <= 0 || params_.k <= 0 || params_.n_gate <= 0 || params_.n_up <= 0)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Invalid dimensions: m=" << params_.m
                                                                       << " k=" << params_.k
                                                                       << " n_gate=" << params_.n_gate
                                                                       << " n_up=" << params_.n_up);
            return false;
        }

        LOG_DEBUG("[FusedGateUpGEMMStage] input_type=" << params_.input->dtype_name()
                                                       << " output_gate_type=" << params_.output_gate->dtype_name()
                                                       << " output_up_type=" << params_.output_up->dtype_name());

        // Validate weight tensor types for diagnostics and dump coherence.
        auto *w_gate_base = requireTensorBase(params_.w_gate, "w_gate");
        auto *w_up_base = requireTensorBase(params_.w_up, "w_up");
        if (!w_gate_base || !w_up_base)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] GPU tensors not yet supported");
            return false;
        }

        // Cast input/output tensors before resolving the prepared kernel so
        // graph-declared arena buffer mismatches fail on the CPU side.
        auto *input_base = requireTensorBase(params_.input, "input");
        auto *output_gate_base = asTensorBase(params_.output_gate, "output_gate");
        auto *output_up_base = asTensorBase(params_.output_up, "output_up");

        if (!input_base || !output_gate_base || !output_up_base)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to cast input/output tensors");
            return false;
        }

        auto has_capacity = [](const TensorBase *tensor,
                               const char *name,
                               size_t rows,
                               size_t cols) -> bool
        {
            const auto &shape = tensor->shape();
            if (shape.size() < 2 || shape[0] < rows || shape[1] < cols)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] " << name
                                                    << " shape is too small: got ["
                                                    << (shape.size() > 0 ? shape[0] : 0) << ", "
                                                    << (shape.size() > 1 ? shape[1] : 0) << "], need at least ["
                                                    << rows << ", " << cols << "]");
                return false;
            }
            return true;
        };

        if (!has_capacity(input_base, "input", static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)) ||
            !has_capacity(output_gate_base, "output_gate", static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_gate)) ||
            !has_capacity(output_up_base, "output_up", static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_up)))
        {
            return false;
        }

        LOG_DEBUG("[FusedGateUpGEMMStage] Looking up kernel for gate=" << (void *)w_gate_base
                                                                       << " up=" << (void *)w_up_base << " device=" << params_.device_id.to_string());

        ITensorFusedGateUpGemm *fused_kernel = resolvePreparedKernel("FusedGateUpGEMMStage");

        LOG_DEBUG("[FusedGateUpGEMMStage] Got fused_kernel=" << (void *)fused_kernel);

        if (!fused_kernel)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to get fused Gate/Up kernel");
            return false;
        }
        fused_kernel->setGPUStream(gpuStream());

        // Check if we have bias - use appropriate execute method
        if (params_.bias_gate || params_.bias_up)
        {
            bool success = fused_kernel->execute_with_bias(
                input_base, output_gate_base, output_up_base,
                params_.bias_gate, params_.bias_up,
                params_.m, params_.k, params_.n_gate, params_.n_up,
                ctx, params_.device_id.toKernelDeviceIndex());

            if (!success)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] execute_with_bias failed");
                return false;
            }
        }
        else
        {
            bool success = fused_kernel->execute(
                input_base, output_gate_base, output_up_base,
                params_.m, params_.k, params_.n_gate, params_.n_up,
                ctx, params_.device_id.toKernelDeviceIndex());

            if (!success)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] execute failed");
                return false;
            }
        }

        LOG_DEBUG("[FusedGateUpGEMMStage] Complete");
        return true;
    }

    size_t FusedGateUpGEMMStage::estimatedFlops() const
    {
        // Two GEMMs: 2 * M * N * K each
        size_t flops_gate = static_cast<size_t>(2) * params_.m * params_.n_gate * params_.k;
        size_t flops_up = static_cast<size_t>(2) * params_.m * params_.n_up * params_.k;
        return flops_gate + flops_up;
    }

    size_t FusedGateUpGEMMStage::estimatedMemoryBytes() const
    {
        // Input: m * k reads (shared)
        size_t input_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);

        // Outputs: gate, up writes
        size_t output_bytes = static_cast<size_t>(params_.m) * (params_.n_gate + params_.n_up) * sizeof(float);

        // Weight reads (approximate - actual depends on quantization format)
        // Quantized weights are ~4-8 bits/element, but we estimate conservatively
        size_t weight_bytes = static_cast<size_t>(params_.k) * (params_.n_gate + params_.n_up) * sizeof(float) / 4;

        return input_bytes + output_bytes + weight_bytes;
    }

    bool FusedGateUpGEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // Uses ITensorGemm interface - device-agnostic
        // Actual device support depends on the kernel implementation returned by KernelFactory
        switch (backend)
        {
        case ComputeBackendType::CPU:

        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo FusedGateUpGEMMStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input (shared)
        info.addInput("input", params_.input, params_.m, params_.k);

        // Weight tensors
        info.addWeight("w_gate", params_.w_gate);
        info.addWeight("w_up", params_.w_up);

        // Outputs
        info.addOutput("output_gate", params_.output_gate, params_.m, params_.n_gate);
        info.addOutput("output_up", params_.output_up, params_.m, params_.n_up);

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        info.addScalarInt("n_gate", params_.n_gate);
        info.addScalarInt("n_up", params_.n_up);

        return info;
    }

    StageBufferRequirements FusedGateUpGEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.w_gate || !params_.w_up)
            return reqs; // Empty if tensors not set

        // Convert tensor types
        BufferTensorType input_type = toBufferTensorType(params_.input->native_type());
        BufferTensorType gate_type = toBufferTensorType(params_.w_gate->native_type());
        BufferTensorType up_type = toBufferTensorType(params_.w_up->native_type());

        // INPUT buffer (shared activation)
        reqs.addInput("input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, input_type);

        // WEIGHT buffers
        reqs.addWeight("w_gate", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_gate)}, gate_type);
        reqs.addWeight("w_up", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_up)}, up_type);

        // OUTPUT buffers
        BufferTensorType out_type = params_.output_gate
                                        ? toBufferTensorType(params_.output_gate->native_type())
                                        : BufferTensorType::FP32;
        reqs.addOutput("output_gate", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_gate)}, out_type);
        reqs.addOutput("output_up", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_up)}, out_type);

        // Optional biases
        if (params_.bias_gate)
        {
            reqs.addWeight("bias_gate", {static_cast<size_t>(params_.n_gate)}, BufferTensorType::FP32);
        }
        if (params_.bias_up)
        {
            reqs.addWeight("bias_up", {static_cast<size_t>(params_.n_up)}, BufferTensorType::FP32);
        }

        return reqs;
    }

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *FusedGateUpGEMMStage::getKernelAsWorkspaceConsumer()
    {
        // Check if we have valid weight tensors
        if (!params_.w_gate || !params_.w_up)
        {
            LOG_WARN("[FusedGateUpGEMMStage::getKernelAsWorkspaceConsumer] Weight tensors not set");
            return nullptr;
        }

        // Validate tensor types before resolving the prepared fused kernel.
        auto *w_gate_base = dynamic_cast<const TensorBase *>(params_.w_gate);
        auto *w_up_base = dynamic_cast<const TensorBase *>(params_.w_up);
        if (!w_gate_base || !w_up_base)
        {
            LOG_WARN("[FusedGateUpGEMMStage::getKernelAsWorkspaceConsumer] "
                     "Weight tensors are not TensorBase");
            return nullptr;
        }

        cached_kernel_ = resolvePreparedKernel("FusedGateUpGEMMStage::getKernelAsWorkspaceConsumer");

        if (!cached_kernel_)
        {
            LOG_WARN("[FusedGateUpGEMMStage::getKernelAsWorkspaceConsumer] "
                     "Failed to get fused kernel");
            return nullptr;
        }

        // The FusedGateUpGemmAdapter now implements IWorkspaceConsumer
        return dynamic_cast<IWorkspaceConsumer *>(cached_kernel_);
    }

    WorkspaceRequirements FusedGateUpGEMMStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        if (!params_.prepared_store ||
            !params_.prepared_ref_gate.has_value() ||
            !params_.prepared_ref_up.has_value())
        {
            LOG_TRACE("[FusedGateUpGEMMStage::getWorkspaceRequirements] PreparedWeightStore and gate/up refs are required");
            return {};
        }

        const int workspace_m = (m > 0) ? m : params_.m;
        const int workspace_k = (k > 0) ? k : params_.k;

        WorkspaceRequirements combined;
        auto mergeFrom = [&](const PreparedWeightRef &ref, int projection_n)
        {
            auto *gemm = params_.prepared_store->gemmKernel(ref);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
            if (!consumer)
                return;
            const int workspace_n = (projection_n > 0) ? projection_n : n;
            combined.merge(consumer->getWorkspaceRequirements(workspace_m, workspace_n, workspace_k));
        };

        mergeFrom(params_.prepared_ref_gate.value(), params_.n_gate);
        mergeFrom(params_.prepared_ref_up.value(), params_.n_up);
        addCudaConcurrentDecodeGemvSideStreamWorkspace(
            combined, params_.device_id, workspace_m, /*projection_count=*/2);
        return combined;
    }

    StageBufferContract FusedGateUpGEMMStage::bufferContract() const
    {
        if (!params_.input_buffer_id || !params_.output_gate_buffer_id ||
            !params_.output_up_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
            .addInput(*params_.input_buffer_id)
            .addOutput(*params_.output_gate_buffer_id)
            .addOutput(*params_.output_up_buffer_id);
        // Model weights are not arena-managed
        if (params_.w_gate)
            contract.addWeight(const_cast<ITensor *>(params_.w_gate));
        if (params_.w_up)
            contract.addWeight(const_cast<ITensor *>(params_.w_up));
        if (params_.bias_gate)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_gate)));
        if (params_.bias_up)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_up)));
        return contract;
    }

} // namespace llaminar2
