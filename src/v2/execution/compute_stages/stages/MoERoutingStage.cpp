/**
 * @file MoERoutingStage.cpp
 * @brief Implementation of MoE routing stage (softmax top-k)
 */

#include "MoERoutingStage.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>

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
        bool ensureRoutingOutputOnStageDevice(
            TensorBase *tensor,
            DeviceId device,
            void *stream,
            const char *name)
        {
            if (!device.is_gpu())
                return true;

            if (!tensor->ensureOnDevice(device, stream))
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
        if (params_.device_id.is_gpu() && gpuStream() && bound_workspace_)
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

        if (seq_len <= 1)
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

            /*
             * CUDA and ROCm both expose an M=2..4 routeWithTensors() path that
             * is tested against the serial decode router.  Use that batched
             * path here instead of issuing M independent decode-route kernels:
             * it keeps verifier routing graph-capturable and avoids host top-k
             * mirrors in Release builds.  CPU remains on the literal row replay
             * path below until it grows the same proven grouped contract.
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
                LOG_ERROR("[MoERoutingStage] Decode-equivalent GPU verifier batched routing failed "
                          "for layer " << params_.layer_idx);
                return false;
            }

            if (!ensureRoutingOutputOnStageDevice(
                    full_indices, params_.device_id, gpuStream(), "output_indices") ||
                !ensureRoutingOutputOnStageDevice(
                    full_output_weights, params_.device_id, gpuStream(), "output_weights"))
            {
                return false;
            }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
            router_logits_ = std::move(cached_routing_.router_logits);
            stashRoutingResults(cached_routing_.expert_indices,
                                cached_routing_.expert_weights,
                                seq_len,
                                top_k);
#else
            routing_indices_f32_.clear();
            routing_weights_.clear();
            router_logits_.clear();
            invalidateDumpInfoCache();
#endif
            return true;
        }

        FP32Tensor cpu_row_input({1u, static_cast<size_t>(d_model)});
        FP32Tensor cpu_row_indices({1u, static_cast<size_t>(top_k)});
        FP32Tensor cpu_row_weights({1u, static_cast<size_t>(top_k)});
        const float *input_data = full_input->data();
        float *indices_data = full_indices->mutable_data();
        float *weights_data = full_output_weights->mutable_data();
        if (!input_data || !indices_data || !weights_data)
        {
            LOG_ERROR("[MoERoutingStage] Decode-equivalent CPU verifier routing could not access tensors");
            return false;
        }

        struct ScopedRowParams
        {
            Params &params;
            TensorBase *input;
            TensorBase *output_indices;
            TensorBase *output_weights;
            DecodeExpertHistogram *decode_histogram;
            IMoERuntimeTable *moe_runtime_table;
            bool force_grouped_verifier_prefill_for_decode;
            bool force_decode_equivalent_verifier_prefill;
            int seq_len;

            ~ScopedRowParams()
            {
                params.input = input;
                params.output_indices = output_indices;
                params.output_weights = output_weights;
                params.decode_histogram = decode_histogram;
                params.moe_runtime_table = moe_runtime_table;
                params.force_grouped_verifier_prefill_for_decode = force_grouped_verifier_prefill_for_decode;
                params.force_decode_equivalent_verifier_prefill = force_decode_equivalent_verifier_prefill;
                params.seq_len = seq_len;
            }
        } restore{
            params_,
            params_.input,
            params_.output_indices,
            params_.output_weights,
            params_.decode_histogram,
            params_.moe_runtime_table,
            params_.force_grouped_verifier_prefill_for_decode,
            params_.force_decode_equivalent_verifier_prefill,
            params_.seq_len};

        std::vector<float> full_router_logits;
        std::vector<float> full_indices_f32(static_cast<size_t>(seq_len) * static_cast<size_t>(top_k));
        std::vector<float> full_weights(static_cast<size_t>(seq_len) * static_cast<size_t>(top_k));
        if (!is_gpu && num_experts > 0)
            full_router_logits.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(num_experts));

        for (int row = 0; row < seq_len; ++row)
        {
            std::copy_n(input_data + static_cast<size_t>(row) * d_model,
                        d_model,
                        cpu_row_input.mutable_data());
            params_.input = &cpu_row_input;
            params_.output_indices = &cpu_row_indices;
            params_.output_weights = &cpu_row_weights;
            params_.decode_histogram = nullptr;
            params_.moe_runtime_table = nullptr;
            params_.seq_len = 1;
            params_.force_decode_equivalent_verifier_prefill = false;
            params_.force_grouped_verifier_prefill_for_decode = false;

            if (!execute(ctx))
            {
                LOG_ERROR("[MoERoutingStage] Decode-equivalent verifier routing failed at row "
                          << row << " for layer " << params_.layer_idx);
                return false;
            }

            const float *row_indices = routing_indices_f32_.data();
            const float *row_weights = routing_weights_.data();
            if (routing_indices_f32_.size() < static_cast<size_t>(top_k) ||
                routing_weights_.size() < static_cast<size_t>(top_k))
            {
                LOG_ERROR("[MoERoutingStage] Decode-equivalent verifier routing produced incomplete "
                          "snapshot top-k vectors at row " << row);
                return false;
            }
            std::copy_n(row_indices,
                        top_k,
                        full_indices_f32.data() + static_cast<size_t>(row) * top_k);
            std::copy_n(row_weights,
                        top_k,
                        full_weights.data() + static_cast<size_t>(row) * top_k);

            if (!router_logits_.empty())
            {
                if (router_logits_.size() < static_cast<size_t>(num_experts))
                {
                    LOG_ERROR("[MoERoutingStage] Decode-equivalent verifier routing produced incomplete "
                              "router logits at row " << row);
                    return false;
                }
                std::copy_n(router_logits_.data(),
                            num_experts,
                            full_router_logits.data() + static_cast<size_t>(row) * num_experts);
            }

            std::copy_n(cpu_row_indices.data(),
                        top_k,
                        indices_data + static_cast<size_t>(row) * top_k);
            std::copy_n(cpu_row_weights.data(),
                        top_k,
                        weights_data + static_cast<size_t>(row) * top_k);
        }

        routing_indices_f32_ = std::move(full_indices_f32);
        routing_weights_ = std::move(full_weights);
        router_logits_ = std::move(full_router_logits);
        invalidateDumpInfoCache();
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

        if (params_.force_decode_equivalent_verifier_prefill && seq_len > 1)
            return executeDecodeEquivalentVerifierPrefill(ctx);

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

            if (!ensureRoutingOutputOnStageDevice(
                    params_.output_indices, params_.device_id, gpuStream(), "output_indices") ||
                !ensureRoutingOutputOnStageDevice(
                    params_.output_weights, params_.device_id, gpuStream(), "output_weights"))
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

#if !defined(ENABLE_PIPELINE_SNAPSHOTS)
        if (params_.device_id.is_gpu() && params_.seq_len == 1 &&
            !isDeviceRoutedDecodeGraphCapturable())
        {
            if (params_.allow_eager_gpu_single_row_route_for_partial_expert_owner)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[MoERoutingStage] Partial-owner GPU MoE routeWithTensors "
                              "path is eager-only and cannot run during graph capture on "
                              << params_.device_id.toString());
                    return false;
                }
            }
            else
            {
            /*
             * Production GPU decode must not fall back to routeWithTensors()
             * here: that path can materialize top-k routing on the host for
             * decode bookkeeping.  MoE MTP verifier work relies on the
             * runtime-table route being the single owner of live route state,
             * otherwise host/device mirrors can diverge across replay and
             * publication.
             */
            LOG_ERROR("[MoERoutingStage] GPU single-row MoE routing requires "
                      "the runtime-table device path; refusing host-top-k "
                      "fallback on " << params_.device_id.toString());
            return false;
            }
        }
#endif

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

        if (!ensureRoutingOutputOnStageDevice(
                params_.output_indices, params_.device_id, gpuStream(), "output_indices") ||
            !ensureRoutingOutputOnStageDevice(
                params_.output_weights, params_.device_id, gpuStream(), "output_weights"))
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
        if (params_.force_decode_equivalent_verifier_prefill)
            return isDecodeEquivalentVerifierPrefillGraphCapturable();

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

    bool MoERoutingStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        if (params_.force_decode_equivalent_verifier_prefill)
            return isDecodeEquivalentVerifierPrefillGraphCaptureSupported();
        return isDeviceRoutedPrefillGraphCaptureSupported();
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
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        if (params_.force_decode_equivalent_verifier_prefill)
            return false;

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
               params_.seq_len >= 2 &&
               params_.seq_len <= 4 &&
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
#if defined(ENABLE_PIPELINE_SNAPSHOTS)
        return false;
#else
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               isDecodeEquivalentVerifierPrefillExecutionSupported();
#endif
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
        if (params_.device_id.is_rocm())
            return MoEWorkspaceBuffers::rocmRouting(
                params_.seq_len,
                params_.d_model,
                params_.num_experts,
                params_.top_k);
        return MoEWorkspaceBuffers::routing(params_.seq_len, params_.num_experts, params_.top_k);
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
