/**
 * @file FusedQKVGEMMStage.cpp
 * @brief Implements shared-activation Q/K/V and K/V projection transactions.
 *
 * Projection descriptors are resolved and allocated before graph execution.
 * The K/V-only policy deliberately omits every Q weight, output, prepared
 * binding, and ownership edge while retaining the backend's one-quantization
 * fused projection route.
 */

#include "FusedQKVGEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
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
    // =============================================================================
    // FusedQKVGEMMStage Implementation
    // =============================================================================

    FusedQKVGEMMStage::FusedQKVGEMMStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool FusedQKVGEMMStage::includesQuery() const noexcept
    {
        return params_.projection_set ==
               AttentionProjectionSet::QueryKeyValue;
    }

    size_t FusedQKVGEMMStage::projectionCount() const noexcept
    {
        return includesQuery() ? 3U : 2U;
    }

    ITensorGemm *FusedQKVGEMMStage::anchorKernel() const noexcept
    {
        return includesQuery() ? cached_gemm_q_ : cached_gemm_k_;
    }

    void FusedQKVGEMMStage::rebuildProjectionDescriptors()
    {
        cached_projections_.clear();
        cached_projections_.reserve(projectionCount());
        if (includesQuery())
        {
            cached_projections_.emplace_back(
                cached_gemm_q_,
                dynamic_cast<TensorBase *>(params_.output_q),
                params_.n_q,
                params_.bias_q,
                "Q");
        }
        cached_projections_.emplace_back(
            cached_gemm_k_,
            dynamic_cast<TensorBase *>(params_.output_k),
            params_.n_k,
            params_.bias_k,
            "K");
        cached_projections_.emplace_back(
            cached_gemm_v_,
            dynamic_cast<TensorBase *>(params_.output_v),
            params_.n_v,
            params_.bias_v,
            "V");
    }

    void FusedQKVGEMMStage::clearCachedGemmStreams()
    {
        if (cached_gemm_q_)
            cached_gemm_q_->clearGPUStreamBinding();
        if (cached_gemm_k_)
            cached_gemm_k_->clearGPUStreamBinding();
        if (cached_gemm_v_)
            cached_gemm_v_->clearGPUStreamBinding();
    }

    void FusedQKVGEMMStage::resetSessionState()
    {
        IComputeStage::resetSessionState();
        clearCachedGemmStreams();
    }

    void FusedQKVGEMMStage::resetSessionStatePreservingCapturedReplay()
    {
        IComputeStage::resetSessionStatePreservingCapturedReplay();
        clearCachedGemmStreams();
    }

    void FusedQKVGEMMStage::resetSessionStatePreservingLazyInitialization()
    {
        resetSessionStatePreservingCapturedReplay();
    }

    bool FusedQKVGEMMStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Graph launch preparation requires "
                      "the exact non-null producer stream");
            return false;
        }
        setGPUStream(stream);

        if (!resolveIndividualKernels("FusedQKVGEMMStage::prepareGraphLaunch"))
            return false;
        if (cached_gemm_q_)
            bindStageStream(cached_gemm_q_);
        bindStageStream(cached_gemm_k_);
        bindStageStream(cached_gemm_v_);

        const size_t projection_count = projectionCount();
        const bool q_ready =
            !cached_gemm_q_ ||
            cached_gemm_q_->prepareFusedProjectionGraphCapture(
                projection_count);
        const bool k_ready =
            cached_gemm_k_ &&
            cached_gemm_k_->prepareFusedProjectionGraphCapture(
                projection_count);
        const bool v_ready =
            cached_gemm_v_ &&
            cached_gemm_v_->prepareFusedProjectionGraphCapture(
                projection_count);
        if (!q_ready || !k_ready || !v_ready)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Failed to provision persistent "
                      "fused-projection resources before graph capture"
                      << " device=" << params_.device_id.toString()
                      << " projections=" << projection_count);
            return false;
        }
        return true;
    }

    bool FusedQKVGEMMStage::validatePreparedWeights(std::string *error) const
    {
        if ((!includesQuery() || !params_.wq) &&
            !params_.wk && !params_.wv)
            return true;
        if (!params_.prepared_store ||
            (includesQuery() && !params_.prepared_ref_q.has_value()) ||
            !params_.prepared_ref_k.has_value() ||
            !params_.prepared_ref_v.has_value())
        {
            if (error)
            {
                *error = includesQuery()
                             ? "FusedQKVGEMMStage requires PreparedWeightStore and Q/K/V PreparedWeightRefs"
                             : "FusedQKVGEMMStage K/V-only mode requires PreparedWeightStore and K/V PreparedWeightRefs";
            }
            return false;
        }
        const bool has_q = !includesQuery() ||
                           params_.prepared_store->contains(
                               params_.prepared_ref_q.value());
        const bool has_k = params_.prepared_store->contains(params_.prepared_ref_k.value());
        const bool has_v = params_.prepared_store->contains(params_.prepared_ref_v.value());
        if (!has_q || !has_k || !has_v)
        {
            if (error)
            {
                *error = "FusedQKVGEMMStage has a PreparedWeightRef missing from PreparedWeightStore"
                         " q=" + (includesQuery()
                                      ? std::to_string(params_.prepared_ref_q->binding_id) + ":" + (has_q ? "1" : "0")
                                      : std::string("not-requested")) +
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
            return (!includesQuery() || cached_gemm_q_) &&
                   cached_gemm_k_ && cached_gemm_v_;

        if (!params_.prepared_store ||
            (includesQuery() && !params_.prepared_ref_q.has_value()) ||
            !params_.prepared_ref_k.has_value() ||
            !params_.prepared_ref_v.has_value())
        {
            LOG_ERROR("[" << caller << "] PreparedWeightStore and "
                      << (includesQuery() ? "Q/K/V" : "K/V")
                      << " PreparedWeightRefs are required");
            return false;
        }

        cached_gemm_q_ = includesQuery()
                             ? params_.prepared_store->gemmKernel(
                                   params_.prepared_ref_q.value())
                             : nullptr;
        cached_gemm_k_ = params_.prepared_store->gemmKernel(params_.prepared_ref_k.value());
        cached_gemm_v_ = params_.prepared_store->gemmKernel(params_.prepared_ref_v.value());

        if ((includesQuery() && !cached_gemm_q_) ||
            !cached_gemm_k_ || !cached_gemm_v_)
        {
            LOG_ERROR("[" << caller << "] PreparedWeightRefs were provided but one or more Q/K/V GEMM kernels were missing from PreparedWeightStore"
                      << " q=" << static_cast<const void *>(cached_gemm_q_)
                      << " k=" << static_cast<const void *>(cached_gemm_k_)
                      << " v=" << static_cast<const void *>(cached_gemm_v_));
            return false;
        }

        rebuildProjectionDescriptors();
        cache_resolved_individual_ = true;
        return true;
    }

    bool FusedQKVGEMMStage::execute(IDeviceContext *ctx)
    {
        ScopedGemmContext gemm_ctx(GemmContext::ATTN);

        LOG_TRACE("[FusedQKVGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                    << " n_q=" << params_.n_q << " n_k=" << params_.n_k << " n_v=" << params_.n_v
                                                    << " projections=" << (includesQuery() ? "Q/K/V" : "K/V")
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

        if ((includesQuery() && !params_.wq) || !params_.wk || !params_.wv)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null "
                      << (includesQuery() ? "Q/K/V" : "K/V")
                      << " weight tensor(s)");
            return false;
        }
        if ((includesQuery() && !params_.output_q) ||
            !params_.output_k || !params_.output_v)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null "
                      << (includesQuery() ? "Q/K/V" : "K/V")
                      << " output buffer(s)");
            return false;
        }
        if (params_.projection_set == AttentionProjectionSet::KeyValueOnly &&
            (params_.wq || params_.output_q || params_.bias_q ||
             params_.n_q != 0 || params_.prepared_ref_q.has_value() ||
             params_.output_q_buffer_id.has_value()))
        {
            LOG_ERROR("[FusedQKVGEMMStage] K/V-only policy must not carry query weights, output, dimensions, bias, prepared refs, or buffer ownership");
            return false;
        }
        if (params_.m <= 0 || params_.k <= 0 ||
            params_.n_k <= 0 || params_.n_v <= 0 ||
            (includesQuery() && params_.n_q <= 0))
        {
            LOG_ERROR("[FusedQKVGEMMStage] Invalid dimensions: m=" << params_.m
                      << " k=" << params_.k
                      << " n_q=" << params_.n_q
                      << " n_k=" << params_.n_k
                      << " n_v=" << params_.n_v);
            return false;
        }

        if (!resolveIndividualKernels("FusedQKVGEMMStage"))
            return false;
        if (cached_gemm_q_)
            bindStageStream(cached_gemm_q_);
        bindStageStream(cached_gemm_k_);
        bindStageStream(cached_gemm_v_);

        const bool gpu_execution = params_.device_id.is_gpu();
        LOG_TRACE("[FusedQKVGEMMStage] device_id=" << params_.device_id.to_string()
                                                   << " is_gpu=" << gpu_execution);

        auto *input_base =
            dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.input));
        auto *output_q_base = includesQuery()
                                  ? dynamic_cast<TensorBase *>(params_.output_q)
                                  : nullptr;
        auto *output_k_base = dynamic_cast<TensorBase *>(params_.output_k);
        auto *output_v_base = dynamic_cast<TensorBase *>(params_.output_v);
        if (!input_base || (includesQuery() && !output_q_base) ||
            !output_k_base || !output_v_base)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Projection transaction requires TensorBase-derived input and outputs");
            return false;
        }

        bool success = false;
        if (params_.force_decode_equivalent_verifier_prefill && params_.m > 1)
        {
            success = executeDecodeEquivalentVerifierPrefill(input_base);
        }
        else
        {
            success = anchorKernel()->multiply_fused_tensor(
                input_base,
                cached_projections_,
                params_.m,
                params_.k,
                nullptr,
                bound_workspace_);
        }

        if (success && !gpu_execution &&
            Logger::getInstance().shouldLog(LogLevel::TRACE))
        {
            TensorBase *diagnostic_output =
                includesQuery() ? output_q_base : output_k_base;
            const float *data = diagnostic_output->data();
            LOG_TRACE("[FusedQKVGEMMStage] "
                      << (includesQuery() ? "Q" : "K")
                      << " output[0:8]=" << std::setprecision(10)
                      << data[0] << "," << data[1] << ","
                      << data[2] << "," << data[3] << ","
                      << data[4] << "," << data[5] << ","
                      << data[6] << "," << data[7]);
        }

        if (!success)
        {
            LOG_ERROR("[FusedQKVGEMMStage] GEMM failed");
            return false;
        }

        LOG_TRACE("[FusedQKVGEMMStage] Complete");
        return true;
    }

    bool FusedQKVGEMMStage::executeDecodeEquivalentVerifierPrefill(
        TensorBase *input_base)
    {
        const bool is_gpu = params_.device_id.is_gpu();
        void *stream = gpuStream();
        if (is_gpu && !stream)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Grouped verifier GPU projection requires an explicit stream");
            return false;
        }

        /*
         * Real verifier fast path: all rows and projections are grouped in the
         * kernel layer. Unsupported grouped implementations fail closed here;
         * production stage execution must never replay verifier rows one at a time.
         */
        const bool success = anchorKernel()->multiply_fused_verifier_rows_decode_equivalent(
            input_base,
            cached_projections_,
            params_.m,
            params_.k,
            nullptr,
            bound_workspace_);

        if (success)
        {
            if (is_gpu)
            {
                const StageGPUExecution gpu = gpuExecution();
                for (const auto &projection : cached_projections_)
                    gpu.publish(projection.output);
            }
            PerfStatsCollector::addCounter(
                "mtp",
                includesQuery()
                    ? "qkv_decode_equivalent_verifier_prefill_rows"
                    : "kv_decode_equivalent_prefill_rows",
                static_cast<double>(params_.m),
                {},
                params_.device_id.to_string(),
                {{"stage", includesQuery() ? "FusedQKVGEMM" : "FusedKVGEMM"},
                 {"route", "grouped"}});
        }
        else
        {
            LOG_ERROR("[FusedQKVGEMMStage] Grouped decode-equivalent "
                      << (includesQuery() ? "Q/K/V" : "K/V")
                      << " projection is unsupported or failed"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.m
                      << " k=" << params_.k);
        }

        return success;
    }

    size_t FusedQKVGEMMStage::estimatedFlops() const
    {
        const size_t flops_q = includesQuery()
                                   ? static_cast<size_t>(2) * params_.m *
                                         params_.n_q * params_.k
                                   : 0U;
        size_t flops_k = static_cast<size_t>(2) * params_.m * params_.n_k * params_.k;
        size_t flops_v = static_cast<size_t>(2) * params_.m * params_.n_v * params_.k;
        return flops_q + flops_k + flops_v;
    }

    size_t FusedQKVGEMMStage::estimatedMemoryBytes() const
    {
        // Input: m * k reads (shared)
        size_t input_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);

        const int projected_columns =
            (includesQuery() ? params_.n_q : 0) + params_.n_k + params_.n_v;
        size_t output_bytes = static_cast<size_t>(params_.m) *
                              projected_columns * sizeof(float);

        // Weight reads (approximate - actual depends on quantization format)
        // Quantized weights are ~4-8 bits/element, but we estimate conservatively
        size_t weight_bytes = static_cast<size_t>(params_.k) *
                              projected_columns * sizeof(float) / 4;

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
        if (includesQuery())
            info.addWeight("wq", params_.wq);
        info.addWeight("wk", params_.wk);
        info.addWeight("wv", params_.wv);

        // Bias tensors (optional but needed for coherence)
        if (includesQuery() && params_.bias_q)
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
        if (includesQuery())
            info.addOutput("output_q", params_.output_q, params_.m, params_.n_q);
        info.addOutput("output_k", params_.output_k, params_.m, params_.n_k);
        info.addOutput("output_v", params_.output_v, params_.m, params_.n_v);

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        if (includesQuery())
            info.addScalarInt("n_q", params_.n_q);
        info.addScalarInt("n_k", params_.n_k);
        info.addScalarInt("n_v", params_.n_v);

        return info;
    }

    StageBufferRequirements FusedQKVGEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || (includesQuery() && !params_.wq) ||
            !params_.wk || !params_.wv ||
            (includesQuery() && !params_.output_q) ||
            !params_.output_k || !params_.output_v)
            return reqs; // Empty if tensors not set

        // Convert tensor types
        BufferTensorType input_type = params_.input
                                          ? toBufferTensorType(params_.input->native_type())
                                          : BufferTensorType::FP32;
        BufferTensorType wq_type = includesQuery()
                                       ? toBufferTensorType(params_.wq->native_type())
                                       : BufferTensorType::FP32;
        BufferTensorType wk_type = toBufferTensorType(params_.wk->native_type());
        BufferTensorType wv_type = toBufferTensorType(params_.wv->native_type());

        // INPUT buffer (shared activation)
        reqs.addInput("input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, input_type);

        // WEIGHT buffers
        if (includesQuery())
        {
            reqs.addWeight(
                "wq",
                {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_q)},
                wq_type);
        }
        reqs.addWeight("wk", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_k)}, wk_type);
        reqs.addWeight("wv", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_v)}, wv_type);

        // OUTPUT buffers
        const ITensor *representative_output =
            includesQuery() ? params_.output_q : params_.output_k;
        BufferTensorType out_type = representative_output
                                        ? toBufferTensorType(representative_output->native_type())
                                        : BufferTensorType::FP32;
        if (includesQuery())
        {
            reqs.addOutput(
                "output_q",
                {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_q)},
                out_type);
        }
        reqs.addOutput("output_k", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_k)}, toBufferTensorType(params_.output_k->native_type()));
        reqs.addOutput("output_v", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_v)}, toBufferTensorType(params_.output_v->native_type()));

        // Optional biases
        if (includesQuery() && params_.bias_q)
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
        const ITensor *anchor_weight = includesQuery() ? params_.wq : params_.wk;
        if (!anchor_weight)
        {
            LOG_WARN("[FusedQKVGEMMStage::getKernelAsWorkspaceConsumer] Transaction anchor weight tensor not set");
            return nullptr;
        }

        auto *anchor_weight_base =
            dynamic_cast<TensorBase *>(const_cast<ITensor *>(anchor_weight));
        if (!anchor_weight_base)
        {
            LOG_WARN("[FusedQKVGEMMStage::getKernelAsWorkspaceConsumer] Transaction anchor weight is not TensorBase");
            return nullptr;
        }

        if (!resolveIndividualKernels("FusedQKVGEMMStage::getKernelAsWorkspaceConsumer"))
            return nullptr;

        return dynamic_cast<IWorkspaceConsumer *>(anchorKernel());
    }

    WorkspaceRequirements FusedQKVGEMMStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        // Resolve every active kernel so each projection contributes
        // its actual N/K shape before workspace requirements are merged.
        auto *self = const_cast<FusedQKVGEMMStage *>(this);
        if (!self->resolveIndividualKernels("FusedQKVGEMMStage::getWorkspaceRequirements"))
            return {};

        const int workspace_m = (m > 0) ? m : params_.m;
        const int workspace_k = (k > 0) ? k : params_.k;

        WorkspaceRequirements combined;
        if (includesQuery())
        {
            if (auto *consumer_q =
                    dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
            {
                combined.merge(consumer_q->getWorkspaceRequirements(
                    workspace_m,
                    params_.n_q > 0 ? params_.n_q : n,
                    workspace_k));
            }
        }
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

        std::vector<int> fused_columns;
        fused_columns.reserve(projectionCount());
        if (includesQuery())
            fused_columns.push_back(params_.n_q > 0 ? params_.n_q : n);
        fused_columns.push_back(params_.n_k > 0 ? params_.n_k : n);
        fused_columns.push_back(params_.n_v > 0 ? params_.n_v : n);
        if (auto *anchor = dynamic_cast<IWorkspaceConsumer *>(anchorKernel()))
        {
            anchor->appendFusedProjectionWorkspaceRequirements(
                combined, workspace_m, fused_columns, workspace_k);
        }
        addCudaConcurrentDecodeGemvSideStreamWorkspace(
            combined,
            params_.device_id,
            workspace_m,
            static_cast<int>(projectionCount()));
        return combined;
    }

    void FusedQKVGEMMStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        // Every active projection kernel shares the same persistent workspace.
        if (!resolveIndividualKernels("FusedQKVGEMMStage::bindWorkspace"))
            return;

        // Bind workspace using cached kernels
        if (includesQuery())
        {
            if (auto *consumer_q =
                    dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
            {
                consumer_q->bindWorkspace(workspace);
                LOG_TRACE("[FusedQKVGEMMStage] Bound workspace to Q kernel");
            }
        }
        if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_k_))
        {
            consumer_k->bindWorkspace(workspace);
            LOG_TRACE("[FusedQKVGEMMStage] Bound workspace to K kernel");
        }
        if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_v_))
        {
            consumer_v->bindWorkspace(workspace);
            LOG_TRACE("[FusedQKVGEMMStage] Bound workspace to V kernel");
        }

        // Store workspace reference for hasWorkspace()/getWorkspace()
        bound_workspace_ = workspace;
    }

    void FusedQKVGEMMStage::unbindWorkspace()
    {
        // Unbind workspace from cached kernels
        if (includesQuery())
        {
            if (auto *consumer_q =
                    dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
                consumer_q->unbindWorkspace();
        }
        if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_k_))
            consumer_k->unbindWorkspace();
        if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_v_))
            consumer_v->unbindWorkspace();

        bound_workspace_ = nullptr;
    }

    StageBufferContract FusedQKVGEMMStage::bufferContract() const
    {
        if (!params_.input_buffer_id ||
            (includesQuery() && !params_.output_q_buffer_id) ||
            !params_.output_k_buffer_id || !params_.output_v_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
                            .addInput(*params_.input_buffer_id);
        if (includesQuery())
            contract.addOutput(*params_.output_q_buffer_id);
        contract.addOutput(*params_.output_k_buffer_id)
            .addOutput(*params_.output_v_buffer_id);
        // Q/K/V kernels consume store-owned prepared representations.
        if (includesQuery() && params_.wq)
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.wq),
                params_.prepared_store,
                params_.prepared_ref_q.value_or(PreparedWeightRef{}));
        if (params_.wk)
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.wk),
                params_.prepared_store,
                params_.prepared_ref_k.value_or(PreparedWeightRef{}));
        if (params_.wv)
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.wv),
                params_.prepared_store,
                params_.prepared_ref_v.value_or(PreparedWeightRef{}));
        if (includesQuery() && params_.bias_q)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_q)));
        if (params_.bias_k)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_k)));
        if (params_.bias_v)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_v)));
        return contract;
    }

} // namespace llaminar2
