/**
 * @file MoERoutingStage.cpp
 * @brief Implementation of MoE routing stage (softmax top-k)
 */

#include "MoERoutingStage.h"
#include "MoEDeviceRebalanceStage.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <sstream>

#ifdef HAVE_CUDA
#include "../../../kernels/cuda/ops/CUDARowSelectKernels.h"
#endif

#ifdef HAVE_ROCM
#include "../../../kernels/rocm/ops/ROCmRowSelectKernels.h"
#endif

namespace llaminar2
{

    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    namespace
    {
        bool supportsGroupedPrefillExecutionBackend(DeviceId device)
        {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
            (void)device;
            return false;
#else
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
            return supportsGroupedPrefillExecutionBackend(device);
        }

        bool supportsDeviceRoutedDecodeGraphCaptureBackend(DeviceId device)
        {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
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

    struct MoERoutingStage::GpuEffectiveSeqLenState
    {
        DeviceId device = DeviceId::invalid();   ///< Device that owns device_effective_seq_len.
        int *host_effective_seq_len = nullptr;   ///< Pinned host scalar uploaded before capture/replay.
        int *device_effective_seq_len = nullptr; ///< Workspace scalar read by GPU routing kernels.
        bool device_value_uploaded = false;      ///< True when device scalar matches the host value.
    };

    MoERoutingStage::MoERoutingStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.moe_runtime_table && params_.layer_idx >= 0)
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
    }

    MoERoutingStage::~MoERoutingStage()
    {
        releaseGpuEffectiveSeqLenState();
    }

    void MoERoutingStage::resetSessionState()
    {
        IComputeStage::resetSessionState();
        routing_indices_f32_.clear();
        routing_weights_.clear();
        router_logits_.clear();
        cached_routing_ = MoERoutingResult{};
        prefill_effective_seq_len_ = 0;
        prefill_bucket_seq_len_ = 0;
        prefill_replay_params_set_ = false;
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
    }

    void MoERoutingStage::resetSessionStatePreservingCapturedReplay()
    {
        IComputeStage::resetSessionState();
        routing_indices_f32_.clear();
        routing_weights_.clear();
        router_logits_.clear();
        cached_routing_ = MoERoutingResult{};
        prefill_effective_seq_len_ = 0;
        prefill_bucket_seq_len_ = 0;
        prefill_replay_params_set_ = false;
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
    }

    void MoERoutingStage::resetSessionStatePreservingLazyInitialization()
    {
        resetSessionStatePreservingCapturedReplay();
    }

    void MoERoutingStage::invalidateKernelDynamicState()
    {
        IMoEKernel *kernel =
            params_.routed_pipeline_kernel_owner &&
                    params_.routed_pipeline_kernel_owner->kernel
                ? params_.routed_pipeline_kernel_owner->kernel.get()
                : owned_moe_kernel_.get();
        if (!kernel)
            return;

        kernel->resetDynamicState();
        kernel->clearGPUStreamBinding();
    }

    IMoEKernel *MoERoutingStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
        {
            if (params_.routed_pipeline_kernel_owner)
            {
                if (!params_.routed_pipeline_kernel_owner->kernel)
                {
                    params_.routed_pipeline_kernel_owner->kernel =
                        KernelFactory::createMoEKernel(params_.device_id);
                }
                moe_kernel_ = params_.routed_pipeline_kernel_owner->kernel.get();
            }
            else
            {
                owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
                moe_kernel_ = owned_moe_kernel_.get();
            }
        }
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

    int MoERoutingStage::effectivePrefillSeqLen() const
    {
        if (!prefill_replay_params_set_ || prefill_effective_seq_len_ <= 0)
            return params_.seq_len;
        return std::clamp(prefill_effective_seq_len_, 1, std::max(1, params_.seq_len));
    }

    void MoERoutingStage::updatePrefillReplayParams(const PrefillReplayParams &replay)
    {
        prefill_replay_params_set_ = true;
        prefill_bucket_seq_len_ = replay.bucket_seq_len > 0 ? replay.bucket_seq_len : params_.seq_len;
        const int real_seq_len = replay.real_seq_len > 0 ? replay.real_seq_len : params_.seq_len;
        prefill_effective_seq_len_ = std::clamp(real_seq_len, 1, std::max(1, params_.seq_len));
        refreshPinnedEffectiveSeqLen();
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
        if (params_.device_id.is_gpu() && hasGPUStream() && bound_workspace_)
            (void)(ensureGpuEffectiveSeqLenStateInitialized() && uploadGpuEffectiveSeqLen());
    }

    void MoERoutingStage::refreshPinnedEffectiveSeqLen()
    {
        if (gpu_effective_seq_len_state_ && gpu_effective_seq_len_state_->host_effective_seq_len)
            *gpu_effective_seq_len_state_->host_effective_seq_len = effectivePrefillSeqLen();
    }

    bool MoERoutingStage::ensureGpuEffectiveSeqLenStateInitialized()
    {
        if (!bound_workspace_ ||
            !bound_workspace_->hasBuffer(MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN) ||
            bound_workspace_->getBufferSize(MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN) < sizeof(int))
        {
            LOG_ERROR("[MoERoutingStage] Missing graph workspace buffer '"
                      << MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN
                      << "' for padded MoE prefill replay on "
                      << params_.device_id.toString());
            return false;
        }

        auto *device_effective_seq_len = static_cast<int *>(
            bound_workspace_->getBuffer(MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN));
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[MoERoutingStage] Graph workspace buffer '"
                      << MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN
                      << "' resolved to null on " << params_.device_id.toString());
            return false;
        }

        if (gpu_effective_seq_len_state_)
        {
            gpu_effective_seq_len_state_->device_effective_seq_len = device_effective_seq_len;
            return true;
        }

        auto state = std::make_unique<GpuEffectiveSeqLenState>();
        state->device = params_.device_id;
        state->device_effective_seq_len = device_effective_seq_len;

        bool allocated = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            allocated = cuda::allocateRowSelectHostParam(
                params_.device_id.cuda_ordinal(),
                &state->host_effective_seq_len);
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            allocated = rocm::allocateRowSelectHostParam(
                params_.device_id.rocm_ordinal(),
                &state->host_effective_seq_len);
#endif
        }

        if (!allocated || !state->host_effective_seq_len || !state->device_effective_seq_len)
        {
            LOG_ERROR("[MoERoutingStage] Failed to allocate pinned effective-length scalar for "
                      << params_.device_id.toString());
            return false;
        }

        gpu_effective_seq_len_state_ = std::move(state);
        refreshPinnedEffectiveSeqLen();
        return true;
    }

    bool MoERoutingStage::uploadGpuEffectiveSeqLen()
    {
        if (!gpu_effective_seq_len_state_)
            return false;
        refreshPinnedEffectiveSeqLen();

        if (isGraphCaptureActive())
        {
            if (!gpu_effective_seq_len_state_->device_value_uploaded)
            {
                LOG_ERROR("[MoERoutingStage] Effective sequence length scalar was not uploaded before graph capture");
                return false;
            }
            return true;
        }

        bool uploaded = false;
        if (params_.device_id.is_cuda())
        {
#ifdef HAVE_CUDA
            uploaded = cuda::uploadRowSelectParam(
                gpu_effective_seq_len_state_->device_effective_seq_len,
                gpu_effective_seq_len_state_->host_effective_seq_len,
                gpuStream());
#endif
        }
        else if (params_.device_id.is_rocm())
        {
#ifdef HAVE_ROCM
            uploaded = rocm::uploadRowSelectParam(
                gpu_effective_seq_len_state_->device_effective_seq_len,
                gpu_effective_seq_len_state_->host_effective_seq_len,
                gpuStream());
#endif
        }
        gpu_effective_seq_len_state_->device_value_uploaded = uploaded;
        return uploaded;
    }

    void MoERoutingStage::releaseGpuEffectiveSeqLenState()
    {
        if (!gpu_effective_seq_len_state_)
            return;

        if (gpu_effective_seq_len_state_->device.is_cuda())
        {
#ifdef HAVE_CUDA
            cuda::freeRowSelectHostParam(
                gpu_effective_seq_len_state_->device.cuda_ordinal(),
                gpu_effective_seq_len_state_->host_effective_seq_len);
#endif
        }
        else if (gpu_effective_seq_len_state_->device.is_rocm())
        {
#ifdef HAVE_ROCM
            rocm::freeRowSelectHostParam(
                gpu_effective_seq_len_state_->device.rocm_ordinal(),
                gpu_effective_seq_len_state_->host_effective_seq_len);
#endif
        }

        gpu_effective_seq_len_state_.reset();
    }

    bool MoERoutingStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        (void)ctx;
        if (stream)
            setGPUStream(stream);

        if (!hasPrefillReplayParams() || !prefill_replay_params_set_)
            return true;

        if (!ensureGpuEffectiveSeqLenStateInitialized())
            return false;
        const bool uploaded = uploadGpuEffectiveSeqLen();
        if (uploaded && PerfStatsCollector::isEnabled())
        {
            PerfStatsCollector::addCounter(
                "moe",
                "routing_padded_prefill_effective_len_prepare",
                1.0,
                "prefill",
                params_.device_id.toString(),
                PerfStatsCollector::Tags{
                    {"bucket_seq_len", std::to_string(params_.seq_len)},
                    {"effective_seq_len", std::to_string(effectivePrefillSeqLen())},
                    {"layer", std::to_string(params_.layer_idx)}});
        }
        return uploaded;
    }

    bool MoERoutingStage::executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const bool is_gpu = params_.device_id.is_gpu();

        if (seq_len < 1)
            return false;
        if (!params_.input || !params_.gate_weights ||
            !params_.output_indices || !params_.output_weights)
        {
            LOG_ERROR("[MoERoutingStage] Decode-equivalent verifier routing missing tensors");
            return false;
        }
        TensorBase *full_input = params_.input;
        TensorBase *full_indices = params_.output_indices;
        TensorBase *full_output_weights = params_.output_weights;

        if (is_gpu)
        {
            if (!isDecodeEquivalentVerifierPrefillExecutionSupported())
            {
                LOG_ERROR("[MoERoutingStage] Decode-equivalent GPU verifier routing is unsupported "
                          "for seq_len=" << seq_len
                          << " d_model=" << d_model
                          << " num_experts=" << num_experts
                          << " top_k=" << top_k
                          << " device=" << params_.device_id.toString());
                return false;
            }

            if (!full_input->gpu_data_ptr() ||
                !params_.gate_weights->gpu_data_ptr() ||
                !full_indices->gpu_data_ptr() ||
                !full_output_weights->gpu_data_ptr())
            {
                LOG_ERROR("[MoERoutingStage] Decode-equivalent GPU verifier routing requires "
                          "input, gate, indices, and weights to be device-resident before graph replay");
                return false;
            }

            cached_routing_ = MoERoutingResult{};

            if (seq_len == 1 && params_.moe_runtime_table)
            {
                /*
                 * M=1 is the serial decode case.  The backend verifier router
                 * below is row-equivalent to decode for its top-k tensors, but
                 * it deliberately does not publish DeviceMoELayerRuntime state.
                 * Ordinary GPU serial decode does publish that runtime row and
                 * the following expert stage consumes it through
                 * groupedExpertDecodeFromRuntime().  Keep the verifier oracle on
                 * exactly that route for the single-row bucket so the expert
                 * stage sees the same runtime bank, descriptor source, and
                 * accumulation order as the reference serial decode.
                 */
                struct ScopedRuntimeDecodeRoute
                {
                    Params &params;
                    bool force_grouped_verifier_prefill_for_decode;
                    bool force_decode_equivalent_verifier_prefill;
                    int seq_len;

                    ~ScopedRuntimeDecodeRoute()
                    {
                        params.force_grouped_verifier_prefill_for_decode =
                            force_grouped_verifier_prefill_for_decode;
                        params.force_decode_equivalent_verifier_prefill =
                            force_decode_equivalent_verifier_prefill;
                        params.seq_len = seq_len;
                    }
                } restore{
                    params_,
                    params_.force_grouped_verifier_prefill_for_decode,
                    params_.force_decode_equivalent_verifier_prefill,
                    params_.seq_len};

                params_.force_grouped_verifier_prefill_for_decode = false;
                params_.force_decode_equivalent_verifier_prefill = false;
                params_.seq_len = 1;

                if (isDeviceRoutedDecodeGraphCapturable())
                {
                    if (!execute(ctx))
                    {
                        LOG_ERROR("[MoERoutingStage] Decode-equivalent M=1 verifier routing failed "
                                  "through the runtime-table serial decode path for layer "
                                  << params_.layer_idx);
                        return false;
                    }

                    PerfStatsCollector::addCounter(
                        "mtp",
                        "moe_decode_equivalent_verifier_prefill_runs",
                        1.0,
                        "verifier",
                        params_.device_id.toString(),
                        {{"stage", "router"},
                         {"seq_len", "1"},
                         {"layer", std::to_string(params_.layer_idx)}});
                    return true;
                }
            }

            IMoEKernel *kernel = ensureMoEKernel();
            /*
             * GPU all-position verifier rows must use the same route math as
             * serial decode.  ROCm in particular can choose hipBLAS for tiny
             * verifier-prefill routing, which is mathematically valid prefill but
             * not bitwise-equivalent to the M=1 decode router.  Delegate to the
             * backend verifier contract so every runtime-M row uses serial-row
             * math while the backend still performs one grouped publication.
             */
            if (!kernel->routeVerifierRowsDecodeEquivalent(
                    full_input,
                    params_.gate_weights,
                    seq_len,
                    d_model,
                    num_experts,
                    top_k,
                    params_.norm_topk_prob,
                    full_indices,
                    full_output_weights))
            {
                LOG_ERROR("[MoERoutingStage] Decode-equivalent GPU verifier row routing failed "
                          "for layer " << params_.layer_idx);
                return false;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "moe_decode_equivalent_verifier_prefill_runs",
                1.0,
                "verifier",
                params_.device_id.toString(),
                {{"stage", "router"},
                 {"seq_len", std::to_string(seq_len)},
                 {"layer", std::to_string(params_.layer_idx)}});

#ifdef ENABLE_PIPELINE_SNAPSHOTS
            router_logits_.clear();
#endif
            routing_indices_f32_.clear();
            routing_weights_.clear();
            invalidateDumpInfoCache();
            return true;
        }

        /*
         * CPU verifier routing used to rebuild the all-position result by
         * recursively executing this stage one token at a time.  That made the
         * verifier path correct by construction, but it was still a production
         * row-replay path.  The grouped tensor API below is the contract we need
         * the backend to satisfy: every verifier row is routed in one stage
         * invocation, and focused regression tests compare those rows against a
         * diagnostic serial oracle outside production execution.
         */
        IMoEKernel *kernel = ensureMoEKernel();
        cached_routing_ = MoERoutingResult{};
        if (!kernel->routeWithTensors(
                full_input,
                params_.gate_weights,
                seq_len,
                d_model,
                num_experts,
                top_k,
                params_.norm_topk_prob,
                full_indices,
                full_output_weights,
                cached_routing_))
        {
            LOG_ERROR("[MoERoutingStage] Grouped CPU decode-equivalent verifier routing failed "
                      "for layer " << params_.layer_idx);
            return false;
        }

        const size_t expected_topk =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (cached_routing_.expert_indices.size() < expected_topk ||
            cached_routing_.expert_weights.size() < expected_topk)
        {
            LOG_ERROR("[MoERoutingStage] Grouped CPU verifier routing produced incomplete "
                      "top-k results for layer " << params_.layer_idx);
            routing_indices_f32_.clear();
            routing_weights_.clear();
            router_logits_.clear();
            invalidateDumpInfoCache();
            return false;
        }

        if (!cached_routing_.router_logits.empty())
            router_logits_ = std::move(cached_routing_.router_logits);
        else
            router_logits_.clear();
        stashRoutingResults(cached_routing_.expert_indices,
                            cached_routing_.expert_weights,
                            seq_len,
                            top_k);

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_decode_equivalent_verifier_prefill_runs",
            1.0,
            "verifier",
            params_.device_id.toString(),
            {{"stage", "router"},
             {"route", "grouped_tensor"},
             {"seq_len", std::to_string(seq_len)},
             {"layer", std::to_string(params_.layer_idx)}});
        return true;
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

        if (params_.force_decode_equivalent_verifier_prefill)
            return executeDecodeEquivalentVerifierPrefill(ctx);

        // Delegate entirely to the kernel's tensor-aware API.
        // CPU: uses data()/mutable_data() — no device involvement.
        // GPU: routing runs on device, results written D2D to tensors,
        //      host_result populated via D2H for CPU-side expert dispatch.
        //      No intermediate H2D transfers.
        IMoEKernel *kernel = ensureMoEKernel();

        if (isDeviceRoutedDecodeGraphCapturable())
        {
            void *route_stream = gpuStream();
            if (!route_stream)
            {
                LOG_ERROR("[MoERoutingStage] Runtime-table GPU decode routing requires an explicit stream on "
                          << params_.device_id.toString());
                return false;
            }
            params_.moe_runtime_table->recordDecodeHistogramProducerStream(route_stream);

            bool routed = false;
            if (params_.device_rebalance_route_apply)
            {
                if (!bound_workspace_)
                {
                    LOG_ERROR("[MoERoutingStage] Device rebalance route-apply requested without a bound workspace");
                    return false;
                }
                if (params_.device_rebalance_workspace_name.empty() ||
                    params_.device_rebalance_plan_capacity == 0 ||
                    params_.device_rebalance_command_buffer_count == 0 ||
                    !validateDeviceMoERebalanceConfig(params_.device_rebalance_config))
                {
                    LOG_ERROR("[MoERoutingStage] Device rebalance route-apply has an invalid binding"
                              << " workspace='" << params_.device_rebalance_workspace_name
                              << "' plan_capacity=" << params_.device_rebalance_plan_capacity
                              << " command_buffers=" << params_.device_rebalance_command_buffer_count);
                    return false;
                }

                auto requireWorkspaceBuffer =
                    [&](const char *base_name, size_t min_bytes) -> void *
                {
                    const std::string name =
                        MoEDeviceRebalanceStage::workspaceBufferName(
                            base_name,
                            params_.device_rebalance_workspace_name);
                    if (!bound_workspace_->hasBuffer(name) ||
                        bound_workspace_->getBufferSize(name) < min_bytes)
                    {
                        LOG_ERROR("[MoERoutingStage] Device rebalance route-apply missing workspace buffer '"
                                  << name << "' required_bytes=" << min_bytes
                                  << " actual_bytes=" << bound_workspace_->getBufferSize(name));
                        return nullptr;
                    }
                    void *ptr = bound_workspace_->getBuffer(name);
                    if (!ptr)
                    {
                        LOG_ERROR("[MoERoutingStage] Device rebalance route-apply workspace buffer '"
                                  << name << "' resolved to null");
                        return nullptr;
                    }
                    return ptr;
                };

                auto *runtime_layers = params_.moe_runtime_table->deviceLayerState(0);
                auto *plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
                    requireWorkspaceBuffer(
                        MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                        static_cast<size_t>(params_.device_rebalance_command_buffer_count) *
                            static_cast<size_t>(params_.device_rebalance_plan_capacity) *
                            sizeof(DeviceMoERebalancePlanEntry)));
                auto *command_headers = static_cast<DeviceMoERebalanceCommandBufferHeader *>(
                    requireWorkspaceBuffer(
                        MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                        static_cast<size_t>(params_.device_rebalance_command_buffer_count) *
                            sizeof(DeviceMoERebalanceCommandBufferHeader)));
                auto *apply_status = static_cast<DeviceMoERebalanceApplyStatus *>(
                    requireWorkspaceBuffer(
                        MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                        sizeof(DeviceMoERebalanceApplyStatus)));
                auto *controller_state = static_cast<DeviceMoERebalanceGraphControllerState *>(
                    requireWorkspaceBuffer(
                        MoEDeviceRebalanceStage::WS_CONTROLLER_STATE,
                        sizeof(DeviceMoERebalanceGraphControllerState)));
                if (!runtime_layers || !plan_entries || !command_headers ||
                    !apply_status || !controller_state)
                {
                    return false;
                }

                // The current grouped expert decode still consumes the legacy routing
                // tensors, so keep them device-resident while also filling runtime top-k.
                routed = kernel->decodeRouteSelectWithReadyRebalanceApply(
                    runtime_layers,
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
                    /*update_runtime_histogram=*/true,
                    plan_entries,
                    params_.device_rebalance_plan_capacity,
                    command_headers,
                    params_.device_rebalance_local_transfer_slots,
                    params_.device_rebalance_local_transfer_slot_count,
                    params_.device_rebalance_config,
                    apply_status,
                    controller_state,
                    params_.device_rebalance_apply_layer_idx == -2
                        ? params_.layer_idx
                        : params_.device_rebalance_apply_layer_idx,
                    params_.device_rebalance_command_buffer_count);
            }
            else
            {
                // The current grouped expert decode still consumes the legacy routing
                // tensors, so keep them device-resident while also filling runtime top-k.
                routed = kernel->decodeRouteSelect(
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
                    /*update_runtime_histogram=*/true);
            }

            if (!routed)
            {
                LOG_ERROR("[MoERoutingStage] Runtime-table decode routing failed");
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

        if (params_.device_id.is_gpu() && params_.seq_len == 1 &&
            !isDeviceRoutedDecodeGraphCapturable())
        {
            /*
             * Every GPU decode topology, including partial expert owners and
             * depth-scoped MTP sidecars, must publish through a mask-aware
             * device runtime table. routeWithTensors() is the grouped prefill
             * primitive; admitting it here would create an eager-only decode
             * contract outside the captured graph and split ownership of live
             * route state.
             */
            LOG_ERROR("[MoERoutingStage] GPU single-row MoE routing requires "
                      "the runtime-table device path on "
                      << params_.device_id.toString());
            return false;
        }

        const bool padded_prefill_replay =
            params_.device_id.is_gpu() &&
            seq_len > 1 &&
            prefill_replay_params_set_ &&
            effectivePrefillSeqLen() < seq_len;
        const int *device_effective_seq_len = nullptr;
        if (padded_prefill_replay)
        {
            /*
             * Bucketed GPU prefill graphs keep launch dimensions fixed, so
             * MoE routing must read the real prompt length from device memory
             * and produce invalid routes for padded rows. Otherwise replaying
             * a shorter request through a captured larger bucket lets padded
             * hidden rows mutate grouped expert state.
             */
            if (!ensureGpuEffectiveSeqLenStateInitialized() || !uploadGpuEffectiveSeqLen())
                return false;
            device_effective_seq_len = gpu_effective_seq_len_state_->device_effective_seq_len;
            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "moe",
                    "routing_padded_prefill_effective_len_execute",
                    1.0,
                    "prefill",
                    params_.device_id.toString(),
                    PerfStatsCollector::Tags{
                        {"bucket_seq_len", std::to_string(seq_len)},
                        {"effective_seq_len", std::to_string(effectivePrefillSeqLen())},
                        {"layer", std::to_string(params_.layer_idx)}});
            }
        }

        const bool routed = device_effective_seq_len
                                ? kernel->routeWithTensorsEffectiveSeqLen(
                                      params_.input, params_.gate_weights,
                                      seq_len, d_model, num_experts, top_k,
                                      params_.norm_topk_prob,
                                      params_.output_indices, params_.output_weights,
                                      cached_routing_,
                                      device_effective_seq_len)
                                : kernel->routeWithTensors(
                                      params_.input, params_.gate_weights,
                                      seq_len, d_model, num_experts, top_k,
                                      params_.norm_topk_prob,
                                      params_.output_indices, params_.output_weights,
                                      cached_routing_);
        if (!routed)
        {
            LOG_ERROR("[MoERoutingStage] Routing failed");
            return false;
        }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        /*
         * GPU routing always publishes device-authoritative tensors. Snapshot
         * collection drains those tensors only at its explicit observation
         * boundary; live routing never manufactures or adopts a host mirror.
         */
        if (!cached_routing_.router_logits.empty())
            router_logits_ = std::move(cached_routing_.router_logits);
        else
            router_logits_.clear();

        const size_t expected_topk =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (cached_routing_.expert_indices.size() >= expected_topk &&
            cached_routing_.expert_weights.size() >= expected_topk)
        {
            stashRoutingResults(cached_routing_.expert_indices,
                                cached_routing_.expert_weights,
                                seq_len,
                                top_k);
        }
        else
        {
            routing_indices_f32_.clear();
            routing_weights_.clear();
            invalidateDumpInfoCache();
        }
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
        if (params_.force_decode_equivalent_verifier_prefill)
            return isDecodeEquivalentVerifierPrefillGraphCapturable();

        if (params_.force_grouped_verifier_prefill_for_decode)
            return isDeviceRoutedPrefillGraphCapturable();

        return isDeviceRoutedDecodeGraphCapturable() ||
               isDeviceRoutedPrefillGraphCapturable();
    }

    bool MoERoutingStage::supportsWarmupDependentGraphCapture() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (params_.force_decode_equivalent_verifier_prefill)
            return isDecodeEquivalentVerifierPrefillGraphCaptureSupported();

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

    std::string MoERoutingStage::graphCaptureReadinessDebugString() const
    {
        const bool decode_equivalent_supported =
            isDecodeEquivalentVerifierPrefillGraphCaptureSupported();
        const bool decode_equivalent_ready =
            isDecodeEquivalentVerifierPrefillGraphCapturable();
        const bool runtime_decode_ready =
            isDeviceRoutedDecodeGraphCapturable();
        const bool prefill_supported =
            isDeviceRoutedPrefillGraphCaptureSupported();
        const bool prefill_ready =
            isDeviceRoutedPrefillGraphCapturable();

        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer=" << params_.layer_idx
            << " seq_len=" << params_.seq_len
            << " d_model=" << params_.d_model
            << " num_experts=" << params_.num_experts
            << " top_k=" << params_.top_k
            << " force_grouped_verifier="
            << (params_.force_grouped_verifier_prefill_for_decode ? "true" : "false")
            << " force_decode_equivalent="
            << (params_.force_decode_equivalent_verifier_prefill ? "true" : "false")
            << " input=" << (params_.input ? "true" : "false")
            << " gate=" << (params_.gate_weights ? "true" : "false")
            << " indices=" << (params_.output_indices ? "true" : "false")
            << " weights=" << (params_.output_weights ? "true" : "false")
            << " kernel=" << (moe_kernel_ ? "true" : "false")
            << " runtime_table=" << (params_.moe_runtime_table ? "true" : "false")
            << " runtime_layer=" << (moe_runtime_layer_ ? "true" : "false")
            << " runtime_decode_ready="
            << (runtime_decode_ready ? "true" : "false")
            << " grouped_prefill_supported="
            << (prefill_supported ? "true" : "false")
            << " grouped_prefill_ready="
            << (prefill_ready ? "true" : "false")
            << " decode_equivalent_supported="
            << (decode_equivalent_supported ? "true" : "false")
            << " decode_equivalent_ready="
            << (decode_equivalent_ready ? "true" : "false");
        return out.str();
    }

    bool MoERoutingStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        if (params_.force_decode_equivalent_verifier_prefill)
            return isDecodeEquivalentVerifierPrefillGraphCaptureSupported();
        return isDeviceRoutedPrefillGraphCaptureSupported();
    }

    bool MoERoutingStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsLazyPrefillGraphCapturePreflight();
    }

    bool MoERoutingStage::supportsPaddedPrefillRealLengthContract() const
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
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (params_.force_decode_equivalent_verifier_prefill)
            return false;

        // Runtime-table decode routing is capture-safe when the GPU backend
        // keeps top-k routing tensors device-resident. Snapshot builds drain
        // tensor outputs after replay instead of forcing host top-k/logit
        // materialization during capture.
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
        if (params_.force_decode_equivalent_verifier_prefill)
            return false;

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

    bool MoERoutingStage::isDecodeEquivalentVerifierPrefillExecutionSupported() const
    {
        return params_.force_decode_equivalent_verifier_prefill &&
               params_.device_id.is_gpu() &&
               supportsGroupedPrefillExecutionBackend(params_.device_id) &&
               params_.seq_len >= 1 &&
               params_.d_model > 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               params_.top_k <= params_.num_experts &&
               params_.input &&
               params_.gate_weights &&
               params_.output_indices &&
               params_.output_weights;
    }

    bool MoERoutingStage::isDecodeEquivalentVerifierPrefillGraphCaptureSupported() const
    {
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               isDecodeEquivalentVerifierPrefillExecutionSupported();
    }

    bool MoERoutingStage::isDecodeEquivalentVerifierPrefillGraphCapturable() const
    {
        return isDecodeEquivalentVerifierPrefillGraphCaptureSupported() &&
               moe_kernel_ != nullptr;
    }

    bool MoERoutingStage::hasInitializedRuntimeTableIfProvided() const
    {
        if (!params_.moe_runtime_table)
            return false;
        if (!moe_runtime_layer_ || params_.layer_idx < 0)
            return false;
        if (params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx))
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
        if (!params_.device_id.is_cuda() && !params_.device_id.is_rocm())
            return WorkspaceRequirements{};
        WorkspaceRequirements reqs =
            params_.device_id.is_rocm()
                ? MoEWorkspaceBuffers::rocmRouting(
                params_.seq_len,
                params_.d_model,
                params_.num_experts,
                      params_.top_k)
                : MoEWorkspaceBuffers::routing(params_.seq_len, params_.num_experts, params_.top_k);

        if (params_.device_rebalance_route_apply)
        {
            const std::string &workspace_name =
                params_.device_rebalance_workspace_name;
            const size_t command_buffer_count =
                static_cast<size_t>(std::max<uint32_t>(
                    1u,
                    params_.device_rebalance_command_buffer_count));
            const size_t plan_capacity =
                static_cast<size_t>(params_.device_rebalance_plan_capacity);
            reqs.buffers.push_back({
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                    workspace_name),
                command_buffer_count * plan_capacity *
                    sizeof(DeviceMoERebalancePlanEntry),
                256,
                true});
            reqs.buffers.push_back({
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                    workspace_name),
                command_buffer_count *
                    sizeof(DeviceMoERebalanceCommandBufferHeader),
                256,
                true});
            reqs.buffers.push_back({
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                    workspace_name),
                sizeof(DeviceMoERebalanceApplyStatus),
                256,
                true});
            reqs.buffers.push_back({
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_CONTROLLER_STATE,
                    workspace_name),
                sizeof(DeviceMoERebalanceGraphControllerState),
                256,
                true});
        }
        return reqs;
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
        if (gpu_effective_seq_len_state_)
        {
            gpu_effective_seq_len_state_->device_effective_seq_len = nullptr;
            gpu_effective_seq_len_state_->device_value_uploaded = false;
        }
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
