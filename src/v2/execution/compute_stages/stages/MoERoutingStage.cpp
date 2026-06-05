/**
 * @file MoERoutingStage.cpp
 * @brief Implementation of MoE routing stage (softmax top-k)
 */

#include "MoERoutingStage.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

#include <cmath>

namespace llaminar2
{

    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    namespace
    {
        bool ensureRoutingOutputOnStageDevice(TensorBase *tensor, DeviceId device, const char *name)
        {
            if (!device.is_gpu())
                return true;

            if (!tensor->ensureOnDevice(device))
            {
                LOG_ERROR("[MoERoutingStage] Failed to make routing output '" << name
                                                                              << "' available on " << device.to_string());
                return false;
            }

            if (!tensor->is_on_device(device))
            {
                LOG_ERROR("[MoERoutingStage] Routing output '" << name
                                                               << "' is not resident on " << device.to_string()
                                                               << " after routing");
                return false;
            }

            return true;
        }

        bool supportsGroupedPrefillExecutionBackend(DeviceId device)
        {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
            (void)device;
            return false;
#else
            if (!debugEnv().rocm.moe_grouped_prefill)
                return false;
#if defined(HAVE_ROCM)
            if (device.is_rocm())
                return true;
#endif
#if defined(HAVE_CUDA)
            if (device.is_cuda())
                return true;
#endif
            return false;
#endif
        }

        bool supportsGroupedPrefillGraphCaptureBackend(DeviceId device)
        {
#if defined(ENABLE_PIPELINE_SNAPSHOTS)
            (void)device;
            return false;
#else
            return supportsGroupedPrefillExecutionBackend(device);
#endif
        }

        bool supportsDeviceRoutedDecodeGraphCaptureBackend(DeviceId device)
        {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
            (void)device;
            return false;
#else
            const auto &rocm = debugEnv().rocm;
            if (!rocm.moe_grouped_decode || !rocm.moe_device_routed_decode)
                return false;
#if defined(HAVE_ROCM)
            if (device.is_rocm())
                return true;
#endif
#if defined(HAVE_CUDA)
            if (device.is_cuda())
                return true;
#endif
            return false;
#endif
        }
    } // namespace

    MoERoutingStage::MoERoutingStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.moe_runtime_table && params_.layer_idx >= 0)
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
    }

    IMoEKernel *MoERoutingStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
            moe_kernel_ = KernelFactory::getOrCreateMoEKernel(params_.device_id);
        auto *kernel = bindStageStream(moe_kernel_);
        if (bound_workspace_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
            {
                consumer->bindWorkspace(bound_workspace_);
            }
        }
        return kernel;
    }

    void MoERoutingStage::stashRoutingResults(
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

    void MoERoutingStage::recordRuntimeHistogramTokenBoundary() const
    {
        if (params_.decode_histogram && params_.layer_idx >= 0 && params_.seq_len == 1)
            params_.decode_histogram->recordTokenBoundary(params_.layer_idx);
    }

    bool MoERoutingStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoERoutingStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.gate_weights)
        {
            LOG_ERROR("[MoERoutingStage] Null input or gate_weights tensor");
            return false;
        }

        if (!params_.output_indices || !params_.output_weights)
        {
            LOG_ERROR("[MoERoutingStage] Null output_indices or output_weights tensor");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;

        // Delegate entirely to the kernel's tensor-aware API.
        // CPU: uses data()/mutable_data() — no device involvement.
        // GPU: routing runs on device, results written D2D to tensors,
        //      host_result populated via D2H for CPU-side expert dispatch.
        //      No intermediate H2D transfers.
        IMoEKernel *kernel = ensureMoEKernel();

        if (isDeviceRoutedDecodeGraphCapturable())
        {
            // The current grouped expert decode still consumes the legacy routing
            // tensors, so keep them device-resident while also filling runtime top-k.
            if (!kernel->decodeRouteSelect(
                    moe_runtime_layer_,
                    params_.input,
                    params_.gate_weights,
                    d_model,
                    num_experts,
                    top_k,
                    params_.norm_topk_prob,
                    params_.output_indices,
                    params_.output_weights,
                    /*write_legacy_outputs=*/true,
                    /*update_runtime_histogram=*/true))
            {
                LOG_ERROR("[MoERoutingStage] Runtime-table decode routing failed");
                return false;
            }

            if (!ensureRoutingOutputOnStageDevice(params_.output_indices, params_.device_id, "output_indices") ||
                !ensureRoutingOutputOnStageDevice(params_.output_weights, params_.device_id, "output_weights"))
            {
                return false;
            }

            LOG_TRACE("[MoERoutingStage] Runtime-routed single token to top-"
                      << top_k << " of " << num_experts << " experts");
            recordRuntimeHistogramTokenBoundary();
            return true;
        }

        if (params_.seq_len == 1 && params_.force_grouped_verifier_prefill_for_decode &&
            !isDeviceRoutedPrefillExecutionSupported())
        {
            LOG_ERROR("[MoERoutingStage] MTP verifier correction replay requested grouped prefill "
                      "routing for seq_len=1, but the device-routed prefill path is unavailable"
                      << " (device=" << params_.device_id.toString()
                      << ", grouped_prefill=" << debugEnv().rocm.moe_grouped_prefill
                      << ", d_model=" << params_.d_model
                      << ", num_experts=" << params_.num_experts
                      << ", top_k=" << params_.top_k
                      << ", input=" << (params_.input != nullptr)
                      << ", gate_weights=" << (params_.gate_weights != nullptr)
                      << ", output_indices=" << (params_.output_indices != nullptr)
                      << ", output_weights=" << (params_.output_weights != nullptr)
                      << ")");
            return false;
        }

        if (!kernel->routeWithTensors(
                params_.input, params_.gate_weights,
                seq_len, d_model, num_experts, top_k,
                params_.norm_topk_prob,
                params_.output_indices, params_.output_weights,
                cached_routing_))
        {
            LOG_ERROR("[MoERoutingStage] Routing failed");
            return false;
        }

        if (!ensureRoutingOutputOnStageDevice(params_.output_indices, params_.device_id, "output_indices") ||
            !ensureRoutingOutputOnStageDevice(params_.output_weights, params_.device_id, "output_weights"))
        {
            return false;
        }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        // Stash routing data for snapshot capture
        router_logits_ = std::move(cached_routing_.router_logits);
        stashRoutingResults(cached_routing_.expert_indices, cached_routing_.expert_weights, seq_len, top_k);
#endif

        // Record routing result in decode histogram (if tracking enabled)
        if (params_.decode_histogram && params_.layer_idx >= 0 && seq_len == 1)
        {
            if (cached_routing_.expert_indices.size() >= static_cast<size_t>(top_k) &&
                cached_routing_.expert_weights.size() >= static_cast<size_t>(top_k))
            {
                params_.decode_histogram->record(
                    params_.layer_idx,
                    cached_routing_.expert_indices.data(),
                    cached_routing_.expert_weights.data(),
                    top_k);
            }
        }

        LOG_TRACE("[MoERoutingStage] Routed " << seq_len << " tokens to top-"
                                              << top_k << " of " << num_experts << " experts");
        return true;
    }

    size_t MoERoutingStage::estimatedFlops() const
    {
        // Gate GEMV: seq_len * d_model * num_experts
        // Top-k: seq_len * num_experts * log2(num_experts)
        const size_t gate_flops = static_cast<size_t>(params_.seq_len) * params_.d_model * params_.num_experts;
        const int log2_experts = (params_.num_experts > 0)
                                     ? static_cast<int>(std::ceil(std::log2(params_.num_experts)))
                                     : 0;
        const size_t topk_flops = static_cast<size_t>(params_.seq_len) * params_.num_experts * log2_experts;
        return gate_flops + topk_flops;
    }

    bool MoERoutingStage::isGraphCapturable() const
    {
        if (params_.force_grouped_verifier_prefill_for_decode)
            return isDeviceRoutedPrefillGraphCapturable();

        return isDeviceRoutedDecodeGraphCapturable() ||
               isDeviceRoutedPrefillGraphCapturable();
    }

    bool MoERoutingStage::supportsWarmupDependentGraphCapture() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        const bool decode_supported =
            !params_.force_grouped_verifier_prefill_for_decode &&
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.seq_len == 1 &&
            params_.d_model > 0 &&
            params_.num_experts > 0 &&
            params_.top_k > 0 &&
            params_.top_k <= params_.num_experts &&
            params_.top_k <= DecodeExpertHistogram::MAX_TOP_K &&
            params_.input &&
            params_.gate_weights &&
            params_.output_indices &&
            params_.output_weights &&
            params_.moe_runtime_table &&
            params_.layer_idx >= 0;

        return decode_supported || isDeviceRoutedPrefillGraphCaptureSupported();
#endif
    }

    bool MoERoutingStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return isDeviceRoutedPrefillGraphCaptureSupported();
    }

    void MoERoutingStage::onGraphReplayed()
    {
        recordRuntimeHistogramTokenBoundary();
    }

    bool MoERoutingStage::needsOnGraphReplayed() const
    {
        return params_.decode_histogram != nullptr &&
               (isDeviceRoutedDecodeGraphCapturable() || isDeviceRoutedPrefillGraphCapturable());
    }

    bool MoERoutingStage::isDeviceRoutedDecodeGraphCapturable() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        // Runtime-table decode routing is capture-safe when the GPU backend
        // keeps top-k routing tensors device-resident. Snapshot builds still
        // require host top-k/logit materialization, but decode histograms are
        // merged lazily from DeviceMoELayerRuntime::decode_histogram.
        return supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
               !params_.force_grouped_verifier_prefill_for_decode &&
               params_.seq_len == 1 &&
               params_.d_model > 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               params_.top_k <= params_.num_experts &&
               params_.top_k <= DecodeExpertHistogram::MAX_TOP_K &&
               params_.input &&
               params_.gate_weights &&
               params_.output_indices &&
               params_.output_weights &&
               params_.moe_runtime_table &&
               hasInitializedRuntimeTableIfProvided();
#endif
    }

    bool MoERoutingStage::isDeviceRoutedPrefillExecutionSupported() const
    {
        const bool forced_decode_replay =
            params_.force_grouped_verifier_prefill_for_decode && params_.seq_len == 1;
        return supportsGroupedPrefillExecutionBackend(params_.device_id) &&
               (params_.seq_len > 1 || forced_decode_replay) &&
               params_.d_model > 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               params_.top_k <= params_.num_experts &&
               params_.input &&
               params_.gate_weights &&
               params_.output_indices &&
               params_.output_weights;
    }

    bool MoERoutingStage::isDeviceRoutedPrefillGraphCaptureSupported() const
    {
        // Cold padded-bucket preflight can run before ensureMoEKernel() has
        // been called. Validate the backend, shape, and tensor contract here;
        // isDeviceRoutedPrefillGraphCapturable() adds warmed-kernel readiness.
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               isDeviceRoutedPrefillExecutionSupported();
    }

    bool MoERoutingStage::isDeviceRoutedPrefillGraphCapturable() const
    {
        // Prefill routing is graph-capturable on supported GPU backends when the full path is
        // device-only and the lazy MoE kernel has already been resolved during
        // normal warmup. routeWithTensors() in non-snapshot Release builds does
        // no D2H and no backend stream synchronization, so data stays device-resident.
        return isDeviceRoutedPrefillGraphCaptureSupported() && moe_kernel_ != nullptr;
    }

    bool MoERoutingStage::hasInitializedRuntimeTableIfProvided() const
    {
        if (!params_.moe_runtime_table)
            return false;
        if (!moe_runtime_layer_ || params_.layer_idx < 0)
            return false;

        const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
        return state.active_bank <= 1 &&
               state.active_epoch > 0 &&
               state.expert_count == static_cast<uint32_t>(params_.num_experts) &&
               state.top_k == static_cast<uint32_t>(params_.top_k) &&
               state.banks[state.active_bank].epoch == state.active_epoch &&
               state.banks[state.active_bank].expert_count == static_cast<uint32_t>(params_.num_experts);
    }

    bool MoERoutingStage::supportsBackend(ComputeBackendType backend) const
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

    StageBufferRequirements MoERoutingStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output_indices)
            reqs.addOutput("output_indices", params_.output_indices->shape(), toBufferTensorType(params_.output_indices->native_type()));
        if (params_.output_weights)
            reqs.addOutput("output_weights", params_.output_weights->shape(), toBufferTensorType(params_.output_weights->native_type()));
        return reqs;
    }

    StageBufferContract MoERoutingStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        contract.addOutput(params_.output_indices_buffer_id);
        contract.addOutput(params_.output_weights_buffer_id);

        // Gate weights are model weights, not arena-managed
        if (params_.gate_weights)
            contract.addWeight(params_.gate_weights);

        return contract;
    }

    StageDumpInfo MoERoutingStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.gate_weights)
            info.addWeight("gate_weights", params_.gate_weights);

        // Routing outputs (stashed during execute for snapshots)
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

        // Output tensors
        if (params_.output_indices)
            info.addOutput("output_indices_tensor", params_.output_indices,
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.top_k));
        if (params_.output_weights)
            info.addOutput("output_weights_tensor", params_.output_weights,
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.top_k));

        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("top_k", params_.top_k);
        return info;
    }

    WorkspaceRequirements MoERoutingStage::getWorkspaceRequirements(int, int, int) const
    {
        if (!params_.device_id.is_cuda())
            return WorkspaceRequirements{};
        return MoEWorkspaceBuffers::routing(
            params_.seq_len,
            params_.num_experts,
            params_.top_k);
    }

    void MoERoutingStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        if (moe_kernel_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(moe_kernel_))
                consumer->bindWorkspace(workspace);
        }
    }

    void MoERoutingStage::unbindWorkspace()
    {
        bindWorkspace(nullptr);
    }

    bool MoERoutingStage::hasWorkspace() const
    {
        return bound_workspace_ != nullptr;
    }

    DeviceWorkspaceManager *MoERoutingStage::getWorkspace() const
    {
        return bound_workspace_;
    }

} // namespace llaminar2
