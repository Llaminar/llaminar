/**
 * @file FusedQKVGEMMStage.cpp
 * @brief Implementation of FusedQKVGEMMStage
 */

#include "FusedQKVGEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/GemmContext.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../loaders/PreparedWeightStore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace llaminar2
{
    namespace
    {
        std::shared_ptr<FP32Tensor> makeScratchFP32(size_t rows, size_t cols, DeviceId device, void *stream)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
            if (device.is_gpu())
                tensor->allocateOnDevice(device, stream);
            return tensor;
        }

        bool ensureScratchFP32(
            std::shared_ptr<FP32Tensor> &tensor,
            size_t rows,
            size_t cols,
            DeviceId device,
            void *stream,
            const char *name)
        {
            const std::vector<size_t> expected_shape{rows, cols};
            if (!tensor || tensor->shape() != expected_shape)
            {
                if (device.is_gpu() && isGraphCaptureActive())
                {
                    LOG_ERROR("[FusedQKVGEMMStage] Cannot allocate verifier scratch tensor '"
                              << name << "' during graph capture");
                    return false;
                }
                tensor = makeScratchFP32(rows, cols, device, stream);
            }

            if (device.is_gpu() && !tensor->gpu_data_ptr())
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[FusedQKVGEMMStage] Verifier scratch tensor '"
                              << name << "' was not device-resident before graph capture");
                    return false;
                }
                if (!tensor->allocateOnDevice(device, stream))
                {
                    LOG_ERROR("[FusedQKVGEMMStage] Failed to allocate verifier scratch tensor '"
                              << name << "' on " << device.to_string());
                    return false;
                }
            }
            return true;
        }

        void markGpuTensorWritten(TensorBase *output, DeviceId device, void *stream)
        {
            if (!output || !device.is_gpu())
                return;
            output->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE, device, stream);
        }

        bool copyFP32DeviceRow(
            TensorBase *dst,
            int dst_row,
            int dst_cols,
            const TensorBase *src,
            int src_row,
            int src_cols,
            int copy_cols,
            DeviceId device,
            void *stream,
            const char *label)
        {
            if (!stream)
            {
                LOG_ERROR("[FusedQKVGEMMStage] " << label
                                                  << " requires an explicit GPU stream");
                return false;
            }

            IBackend *backend = getBackendFor(device);
            if (!backend)
            {
                LOG_ERROR("[FusedQKVGEMMStage] No backend for " << device.to_string()
                                                                 << " while copying " << label);
                return false;
            }

            auto *dst_ptr = static_cast<float *>(dst ? dst->gpu_data_ptr() : nullptr);
            const auto *src_ptr = static_cast<const float *>(src ? src->gpu_data_ptr() : nullptr);
            if (!dst_ptr || !src_ptr)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Null device pointer while copying "
                          << label << " dst=" << static_cast<void *>(dst_ptr)
                          << " src=" << static_cast<const void *>(src_ptr));
                return false;
            }

            const size_t dst_offset = static_cast<size_t>(dst_row) * static_cast<size_t>(dst_cols);
            const size_t src_offset = static_cast<size_t>(src_row) * static_cast<size_t>(src_cols);
            const size_t bytes = static_cast<size_t>(copy_cols) * sizeof(float);
            const bool ok = backend->deviceCopyAsync(
                dst_ptr + dst_offset,
                src_ptr + src_offset,
                bytes,
                device.gpu_ordinal(),
                stream);
            if (!ok)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Device row copy failed for " << label
                                                                            << " bytes=" << bytes);
                return false;
            }
            return true;
        }
    }

    // =============================================================================
    // FusedQKVGEMMStage Implementation
    // =============================================================================

    FusedQKVGEMMStage::FusedQKVGEMMStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool FusedQKVGEMMStage::validatePreparedWeights(std::string *error) const
    {
        if (!params_.wq && !params_.wk && !params_.wv)
            return true;
        if (!params_.prepared_store ||
            !params_.prepared_ref_q.has_value() ||
            !params_.prepared_ref_k.has_value() ||
            !params_.prepared_ref_v.has_value())
        {
            if (error) *error = "FusedQKVGEMMStage requires PreparedWeightStore and Q/K/V PreparedWeightRefs";
            return false;
        }
        const bool has_q = params_.prepared_store->contains(params_.prepared_ref_q.value());
        const bool has_k = params_.prepared_store->contains(params_.prepared_ref_k.value());
        const bool has_v = params_.prepared_store->contains(params_.prepared_ref_v.value());
        if (!has_q || !has_k || !has_v)
        {
            if (error)
            {
                *error = "FusedQKVGEMMStage has a PreparedWeightRef missing from PreparedWeightStore"
                         " q=" + std::to_string(params_.prepared_ref_q->binding_id) + ":" + (has_q ? "1" : "0") +
                         " k=" + std::to_string(params_.prepared_ref_k->binding_id) + ":" + (has_k ? "1" : "0") +
                         " v=" + std::to_string(params_.prepared_ref_v->binding_id) + ":" + (has_v ? "1" : "0");
            }
            return false;
        }
        return true;
    }

    bool FusedQKVGEMMStage::resolveIndividualKernels(const char *caller)
    {
        if (cache_resolved_individual_)
            return cached_gemm_q_ && cached_gemm_k_ && cached_gemm_v_;

        if (!params_.prepared_store ||
            !params_.prepared_ref_q.has_value() ||
            !params_.prepared_ref_k.has_value() ||
            !params_.prepared_ref_v.has_value())
        {
            LOG_ERROR("[" << caller << "] PreparedWeightStore and Q/K/V PreparedWeightRefs are required");
            return false;
        }

        cached_gemm_q_ = params_.prepared_store->gemmKernel(params_.prepared_ref_q.value());
        cached_gemm_k_ = params_.prepared_store->gemmKernel(params_.prepared_ref_k.value());
        cached_gemm_v_ = params_.prepared_store->gemmKernel(params_.prepared_ref_v.value());

        if (!cached_gemm_q_ || !cached_gemm_k_ || !cached_gemm_v_)
        {
            LOG_ERROR("[" << caller << "] PreparedWeightRefs were provided but one or more Q/K/V GEMM kernels were missing from PreparedWeightStore"
                      << " q=" << static_cast<const void *>(cached_gemm_q_)
                      << " k=" << static_cast<const void *>(cached_gemm_k_)
                      << " v=" << static_cast<const void *>(cached_gemm_v_));
            return false;
        }

        cache_resolved_individual_ = true;
        return true;
    }

    bool FusedQKVGEMMStage::execute(IDeviceContext *ctx)
    {
        ScopedGemmContext gemm_ctx(GemmContext::ATTN);

        LOG_DEBUG("[FusedQKVGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                    << " n_q=" << params_.n_q << " n_k=" << params_.n_k << " n_v=" << params_.n_v
                                                    << " device=" << params_.device_id.to_string());

        // Log bias tensor pointers for multi-GPU debugging
        if (params_.bias_q || params_.bias_k || params_.bias_v)
        {
            LOG_DEBUG("[FusedQKVGEMMStage] BIAS POINTERS:"
                      << " bias_q=" << static_cast<const void *>(params_.bias_q ? params_.bias_q->raw_data() : nullptr)
                      << " bias_k=" << static_cast<const void *>(params_.bias_k ? params_.bias_k->raw_data() : nullptr)
                      << " bias_v=" << static_cast<const void *>(params_.bias_v ? params_.bias_v->raw_data() : nullptr)
                      << " stage_device=" << params_.device_id.to_string());
        }

        if (!ctx)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.input)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null input");
            return false;
        }

        if (!params_.wq || !params_.wk || !params_.wv)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null weight tensor(s)");
            return false;
        }
        if (!params_.output_q || !params_.output_k || !params_.output_v)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null output buffer(s)");
            return false;
        }
        if (params_.m <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Invalid dimensions: m=" << params_.m << " k=" << params_.k);
            return false;
        }

        if (!resolveIndividualKernels("FusedQKVGEMMStage"))
            return false;
        auto *gemm_q = cached_gemm_q_;
        auto *gemm_k = cached_gemm_k_;
        auto *gemm_v = cached_gemm_v_;
        if (gemm_q)
            gemm_q->setGPUStream(gpuStream());
        if (gemm_k)
            gemm_k->setGPUStream(gpuStream());
        if (gemm_v)
            gemm_v->setGPUStream(gpuStream());
        const bool gpu_execution = params_.device_id.is_gpu();
        LOG_DEBUG("[FusedQKVGEMMStage] device_id=" << params_.device_id.to_string()
                                                   << " is_gpu=" << gpu_execution);
        bool success = false;

        if (gpu_execution)
        {
            if (!gpuStream())
            {
                LOG_ERROR("[FusedQKVGEMMStage] GPU execution requires an explicit non-null stream");
                return false;
            }

            // GPU path: Use tensor-aware API - kernel handles device placement
            LOG_DEBUG("[FusedQKVGEMMStage] Using tensor-aware GPU path");

            // Cast ITensor* to TensorBase* for tensor-aware API
            auto *input_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.input));
            auto *output_q_base = dynamic_cast<TensorBase *>(params_.output_q);
            auto *output_k_base = dynamic_cast<TensorBase *>(params_.output_k);
            auto *output_v_base = dynamic_cast<TensorBase *>(params_.output_v);

            if (!input_base || !output_q_base || !output_k_base || !output_v_base)
            {
                LOG_ERROR("[FusedQKVGEMMStage] GPU path requires TensorBase-derived types");
                return false;
            }

            if (params_.force_decode_equivalent_verifier_prefill && params_.m > 1)
            {
                success = executeDecodeEquivalentVerifierPrefill(
                    input_base, output_q_base, output_k_base, output_v_base,
                    gemm_q, gemm_k, gemm_v);
            }
            else
            {
                // Build tensor projection descriptors
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {gemm_q, output_q_base, params_.n_q, params_.bias_q, "Q"},
                    {gemm_k, output_k_base, params_.n_k, params_.bias_k, "K"},
                    {gemm_v, output_v_base, params_.n_v, params_.bias_v, "V"}};

                // Use tensor-aware fused API - handles all device sync internally
                success = gemm_q->multiply_fused_tensor(
                    input_base,
                    projections,
                    params_.m,
                    params_.k,
                    nullptr,
                    bound_workspace_);
            }

            if (success)
            {
                LOG_DEBUG("[FusedQKVGEMMStage] GPU execution complete");
            }
        }
        else
        {
            // CPU path: Use tensor-aware API (same as GPU path)
            auto *input_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.input));
            auto *output_q_base = dynamic_cast<TensorBase *>(params_.output_q);
            auto *output_k_base = dynamic_cast<TensorBase *>(params_.output_k);
            auto *output_v_base = dynamic_cast<TensorBase *>(params_.output_v);

            if (!input_base || !output_q_base || !output_k_base || !output_v_base)
            {
                LOG_ERROR("[FusedQKVGEMMStage] CPU path requires TensorBase-derived types");
                return false;
            }

            LOG_DEBUG("[FusedQKVGEMMStage] input_type=" << params_.input->dtype_name()
                                                        << " output_type=" << params_.output_q->dtype_name());

            if (params_.force_decode_equivalent_verifier_prefill && params_.m > 1)
            {
                success = executeDecodeEquivalentVerifierPrefill(
                    input_base, output_q_base, output_k_base, output_v_base,
                    gemm_q, gemm_k, gemm_v);
            }
            else
            {
                // Build tensor projection descriptors
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {gemm_q, output_q_base, params_.n_q, params_.bias_q, "Q"},
                    {gemm_k, output_k_base, params_.n_k, params_.bias_k, "K"},
                    {gemm_v, output_v_base, params_.n_v, params_.bias_v, "V"}};

                success = gemm_q->multiply_fused_tensor(
                    input_base,
                    projections,
                    params_.m,
                    params_.k,
                    nullptr,
                    bound_workspace_);
            }

            if (success && Logger::getInstance().shouldLog(LogLevel::TRACE))
            {
                const float *q_data = output_q_base->data();
                LOG_TRACE("[FusedQKVGEMMStage] Q output[0:8]=" << std::setprecision(10)
                                                               << q_data[0] << "," << q_data[1] << ","
                                                               << q_data[2] << "," << q_data[3] << ","
                                                               << q_data[4] << "," << q_data[5] << ","
                                                               << q_data[6] << "," << q_data[7]
                                                               << " n_q=" << params_.n_q);
            }
        }

        if (!success)
        {
            LOG_ERROR("[FusedQKVGEMMStage] GEMM failed");
            return false;
        }

        LOG_DEBUG("[FusedQKVGEMMStage] Complete");
        return true;
    }

    bool FusedQKVGEMMStage::executeDecodeEquivalentVerifierPrefill(
        TensorBase *input_base,
        TensorBase *output_q_base,
        TensorBase *output_k_base,
        TensorBase *output_v_base,
        ITensorGemm *gemm_q,
        ITensorGemm *gemm_k,
        ITensorGemm *gemm_v)
    {
        if (params_.m > 4)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Decode-equivalent verifier prefill is only supported "
                      << "for tiny MTP verifier batches, got m=" << params_.m);
            return false;
        }

        const bool is_gpu = params_.device_id.is_gpu();
        void *stream = gpuStream();
        if (is_gpu && !stream)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Decode-equivalent GPU verifier prefill requires an explicit stream");
            return false;
        }

        if (!ensureScratchFP32(verifier_input_row_, 1, static_cast<size_t>(params_.k),
                               params_.device_id, stream, "input_row") ||
            !ensureScratchFP32(verifier_output_q_row_, 1, static_cast<size_t>(params_.n_q),
                               params_.device_id, stream, "output_q_row") ||
            !ensureScratchFP32(verifier_output_k_row_, 1, static_cast<size_t>(params_.n_k),
                               params_.device_id, stream, "output_k_row") ||
            !ensureScratchFP32(verifier_output_v_row_, 1, static_cast<size_t>(params_.n_v),
                               params_.device_id, stream, "output_v_row"))
        {
            return false;
        }

        const float *input_data = is_gpu ? nullptr : input_base->data();
        float *output_q_data = is_gpu ? nullptr : output_q_base->mutable_data();
        float *output_k_data = is_gpu ? nullptr : output_k_base->mutable_data();
        float *output_v_data = is_gpu ? nullptr : output_v_base->mutable_data();
        if (!is_gpu && (!input_data || !output_q_data || !output_k_data || !output_v_data))
        {
            LOG_ERROR("[FusedQKVGEMMStage] Decode-equivalent verifier prefill requires FP32 host-visible tensors");
            return false;
        }

        if (is_gpu)
        {
            if (!input_base->gpu_data_ptr())
            {
                LOG_ERROR("[FusedQKVGEMMStage] Decode-equivalent verifier input is not device-resident on "
                          << params_.device_id.to_string()
                          << "; StageBufferContract/BufferArena coherence must prepare graph inputs");
                return false;
            }
            if (!output_q_base->allocateOnDevice(params_.device_id, stream) ||
                !output_k_base->allocateOnDevice(params_.device_id, stream) ||
                !output_v_base->allocateOnDevice(params_.device_id, stream))
            {
                LOG_ERROR("[FusedQKVGEMMStage] Failed to prepare verifier QKV tensors on "
                          << params_.device_id.to_string());
                return false;
            }
        }

        bool success = true;
        for (int row = 0; row < params_.m; ++row)
        {
            if (is_gpu)
            {
                if (!copyFP32DeviceRow(
                        verifier_input_row_.get(), 0, params_.k,
                        input_base, row, params_.k,
                        params_.k, params_.device_id, stream,
                        "verifier_input_row"))
                {
                    success = false;
                    break;
                }
                markGpuTensorWritten(verifier_input_row_.get(), params_.device_id, stream);
            }
            else
            {
                // Keep the row operation on the exact M=1 path used by serial
                // decode. This avoids small-M routing drift in quantized GEMMs
                // while still publishing all verifier rows in one stage.
                std::copy(input_data + static_cast<size_t>(row) * params_.k,
                          input_data + static_cast<size_t>(row + 1) * params_.k,
                          verifier_input_row_->mutable_data());
            }

            std::vector<ITensorGemm::TensorProjectionDesc> row_projections = {
                {gemm_q, verifier_output_q_row_.get(), params_.n_q, params_.bias_q, "Q"},
                {gemm_k, verifier_output_k_row_.get(), params_.n_k, params_.bias_k, "K"},
                {gemm_v, verifier_output_v_row_.get(), params_.n_v, params_.bias_v, "V"}};

            const bool row_success = gemm_q->multiply_fused_tensor(
                verifier_input_row_.get(),
                row_projections,
                1,
                params_.k,
                nullptr,
                bound_workspace_);
            if (!row_success)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Decode-equivalent verifier row "
                          << row << " failed");
                success = false;
                break;
            }

            /*
             * Each fused row projection writes GPU scratch tensors that are
             * immediately copied into the batched verifier outputs. Make that
             * stream-local producer/consumer handoff explicit so future
             * coherence checks cannot read stale host shadows.
             */
            if (is_gpu)
            {
                markGpuTensorWritten(
                    verifier_output_q_row_.get(),
                    params_.device_id,
                    stream);
                markGpuTensorWritten(
                    verifier_output_k_row_.get(),
                    params_.device_id,
                    stream);
                markGpuTensorWritten(
                    verifier_output_v_row_.get(),
                    params_.device_id,
                    stream);
            }

            if (is_gpu)
            {
                if (!copyFP32DeviceRow(
                        output_q_base, row, params_.n_q,
                        verifier_output_q_row_.get(), 0, params_.n_q,
                        params_.n_q, params_.device_id, stream,
                        "verifier_output_q_row") ||
                    !copyFP32DeviceRow(
                        output_k_base, row, params_.n_k,
                        verifier_output_k_row_.get(), 0, params_.n_k,
                        params_.n_k, params_.device_id, stream,
                        "verifier_output_k_row") ||
                    !copyFP32DeviceRow(
                        output_v_base, row, params_.n_v,
                        verifier_output_v_row_.get(), 0, params_.n_v,
                        params_.n_v, params_.device_id, stream,
                        "verifier_output_v_row"))
                {
                    success = false;
                    break;
                }
            }
            else
            {
                std::copy(verifier_output_q_row_->data(),
                          verifier_output_q_row_->data() + params_.n_q,
                          output_q_data + static_cast<size_t>(row) * params_.n_q);
                std::copy(verifier_output_k_row_->data(),
                          verifier_output_k_row_->data() + params_.n_k,
                          output_k_data + static_cast<size_t>(row) * params_.n_k);
                std::copy(verifier_output_v_row_->data(),
                          verifier_output_v_row_->data() + params_.n_v,
                          output_v_data + static_cast<size_t>(row) * params_.n_v);
            }
        }

        if (success)
        {
            if (is_gpu)
            {
                markGpuTensorWritten(output_q_base, params_.device_id, stream);
                markGpuTensorWritten(output_k_base, params_.device_id, stream);
                markGpuTensorWritten(output_v_base, params_.device_id, stream);
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "qkv_decode_equivalent_verifier_prefill_rows",
                static_cast<double>(params_.m),
                {},
                params_.device_id.to_string(),
                {{"stage", "FusedQKVGEMM"}});
        }

        return success;
    }

    size_t FusedQKVGEMMStage::estimatedFlops() const
    {
        // Three GEMMs: 2 * M * N * K each
        size_t flops_q = static_cast<size_t>(2) * params_.m * params_.n_q * params_.k;
        size_t flops_k = static_cast<size_t>(2) * params_.m * params_.n_k * params_.k;
        size_t flops_v = static_cast<size_t>(2) * params_.m * params_.n_v * params_.k;
        return flops_q + flops_k + flops_v;
    }

    size_t FusedQKVGEMMStage::estimatedMemoryBytes() const
    {
        // Input: m * k reads (shared)
        size_t input_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);

        // Outputs: Q, K, V writes
        size_t output_bytes = static_cast<size_t>(params_.m) * (params_.n_q + params_.n_k + params_.n_v) * sizeof(float);

        // Weight reads (approximate - actual depends on quantization format)
        // Quantized weights are ~4-8 bits/element, but we estimate conservatively
        size_t weight_bytes = static_cast<size_t>(params_.k) * (params_.n_q + params_.n_k + params_.n_v) * sizeof(float) / 4;

        return input_bytes + output_bytes + weight_bytes;
    }

    bool FusedQKVGEMMStage::supportsBackend(ComputeBackendType backend) const
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

    StageDumpInfo FusedQKVGEMMStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input (shared)
        info.addInput("input", params_.input, params_.m, params_.k);

        // Weight tensors
        info.addWeight("wq", params_.wq);
        info.addWeight("wk", params_.wk);
        info.addWeight("wv", params_.wv);

        // Bias tensors (optional but needed for coherence)
        if (params_.bias_q)
        {
            info.addWeight("bias_q", params_.bias_q);
        }
        if (params_.bias_k)
        {
            info.addWeight("bias_k", params_.bias_k);
        }
        if (params_.bias_v)
        {
            info.addWeight("bias_v", params_.bias_v);
        }

        // Outputs
        info.addOutput("output_q", params_.output_q, params_.m, params_.n_q);
        info.addOutput("output_k", params_.output_k, params_.m, params_.n_k);
        info.addOutput("output_v", params_.output_v, params_.m, params_.n_v);

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        info.addScalarInt("n_q", params_.n_q);
        info.addScalarInt("n_k", params_.n_k);
        info.addScalarInt("n_v", params_.n_v);

        return info;
    }

    StageBufferRequirements FusedQKVGEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.wq || !params_.wk || !params_.wv)
            return reqs; // Empty if tensors not set

        // Convert tensor types
        BufferTensorType input_type = params_.input
                                          ? toBufferTensorType(params_.input->native_type())
                                          : BufferTensorType::FP32;
        BufferTensorType wq_type = toBufferTensorType(params_.wq->native_type());
        BufferTensorType wk_type = toBufferTensorType(params_.wk->native_type());
        BufferTensorType wv_type = toBufferTensorType(params_.wv->native_type());

        // INPUT buffer (shared activation)
        reqs.addInput("input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, input_type);

        // WEIGHT buffers
        reqs.addWeight("wq", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_q)}, wq_type);
        reqs.addWeight("wk", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_k)}, wk_type);
        reqs.addWeight("wv", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_v)}, wv_type);

        // OUTPUT buffers
        BufferTensorType out_type = params_.output_q
                                        ? toBufferTensorType(params_.output_q->native_type())
                                        : BufferTensorType::FP32;
        reqs.addOutput("output_q", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_q)}, out_type);
        reqs.addOutput("output_k", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_k)}, out_type);
        reqs.addOutput("output_v", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_v)}, out_type);

        // Optional biases
        if (params_.bias_q)
        {
            reqs.addWeight("bias_q", {static_cast<size_t>(params_.n_q)}, BufferTensorType::FP32);
        }
        if (params_.bias_k)
        {
            reqs.addWeight("bias_k", {static_cast<size_t>(params_.n_k)}, BufferTensorType::FP32);
        }
        if (params_.bias_v)
        {
            reqs.addWeight("bias_v", {static_cast<size_t>(params_.n_v)}, BufferTensorType::FP32);
        }

        return reqs;
    }

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *FusedQKVGEMMStage::getKernelAsWorkspaceConsumer()
    {
        if (!params_.wq)
        {
            LOG_WARN("[FusedQKVGEMMStage::getKernelAsWorkspaceConsumer] Q weight tensor not set");
            return nullptr;
        }

        auto *wq_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wq));
        if (!wq_base)
        {
            LOG_WARN("[FusedQKVGEMMStage::getKernelAsWorkspaceConsumer] Q weight tensor is not TensorBase");
            return nullptr;
        }

        if (!resolveIndividualKernels("FusedQKVGEMMStage::getKernelAsWorkspaceConsumer"))
            return nullptr;

        return dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_);
    }

    WorkspaceRequirements FusedQKVGEMMStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        // Ensure all three kernels are resolved so each projection contributes
        // its actual N/K shape before workspace requirements are merged.
        auto *self = const_cast<FusedQKVGEMMStage *>(this);
        if (!self->resolveIndividualKernels("FusedQKVGEMMStage::getWorkspaceRequirements"))
            return {};

        const int workspace_m = (m > 0) ? m : params_.m;
        const int workspace_k = (k > 0) ? k : params_.k;

        WorkspaceRequirements combined;
        if (auto *consumer_q = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
            combined.merge(consumer_q->getWorkspaceRequirements(
                workspace_m,
                params_.n_q > 0 ? params_.n_q : n,
                workspace_k));
        if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_k_))
            combined.merge(consumer_k->getWorkspaceRequirements(
                workspace_m,
                params_.n_k > 0 ? params_.n_k : n,
                workspace_k));
        if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_v_))
            combined.merge(consumer_v->getWorkspaceRequirements(
                workspace_m,
                params_.n_v > 0 ? params_.n_v : n,
                workspace_k));
        addCudaConcurrentDecodeGemvSideStreamWorkspace(
            combined, params_.device_id, workspace_m, /*projection_count=*/3);
        return combined;
    }

    void FusedQKVGEMMStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        // Bind workspace to ALL THREE kernels (Q, K, V)
        // Each kernel needs workspace for GPU execution
        if (!resolveIndividualKernels("FusedQKVGEMMStage::bindWorkspace"))
            return;

        // Bind workspace using cached kernels
        if (auto *consumer_q = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
        {
            consumer_q->bindWorkspace(workspace);
            LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to Q kernel");
        }
        if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_k_))
        {
            consumer_k->bindWorkspace(workspace);
            LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to K kernel");
        }
        if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_v_))
        {
            consumer_v->bindWorkspace(workspace);
            LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to V kernel");
        }

        // Store workspace reference for hasWorkspace()/getWorkspace()
        bound_workspace_ = workspace;
    }

    void FusedQKVGEMMStage::unbindWorkspace()
    {
        // Unbind workspace from cached kernels
        if (auto *consumer_q = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
            consumer_q->unbindWorkspace();
        if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_k_))
            consumer_k->unbindWorkspace();
        if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_v_))
            consumer_v->unbindWorkspace();

        bound_workspace_ = nullptr;
    }

    StageBufferContract FusedQKVGEMMStage::bufferContract() const
    {
        if (!params_.input_buffer_id || !params_.output_q_buffer_id ||
            !params_.output_k_buffer_id || !params_.output_v_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
                            .addInput(*params_.input_buffer_id)
                            .addOutput(*params_.output_q_buffer_id)
                            .addOutput(*params_.output_k_buffer_id)
                            .addOutput(*params_.output_v_buffer_id);
        // Model weights are not arena-managed
        if (params_.wq)
            contract.addWeight(const_cast<ITensor *>(params_.wq));
        if (params_.wk)
            contract.addWeight(const_cast<ITensor *>(params_.wk));
        if (params_.wv)
            contract.addWeight(const_cast<ITensor *>(params_.wv));
        if (params_.bias_q)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_q)));
        if (params_.bias_k)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_k)));
        if (params_.bias_v)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_v)));
        return contract;
    }

} // namespace llaminar2
