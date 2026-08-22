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
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

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

    MoERoutingStage::MoERoutingStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.moe_runtime_table && params_.layer_idx >= 0)
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
    }

    void MoERoutingStage::resetSessionState()
    {
        IComputeStage::resetSessionState();
        routing_indices_f32_.clear();
        routing_weights_.clear();
        router_logits_.clear();
        cached_routing_ = MoERoutingResult{};
    }

    void MoERoutingStage::resetSessionStatePreservingCapturedReplay()
    {
        IComputeStage::resetSessionState();
        routing_indices_f32_.clear();
        routing_weights_.clear();
        router_logits_.clear();
        cached_routing_ = MoERoutingResult{};
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

        if (params_.device_id.is_gpu() &&
            params_.routed_pipeline_kernel_owner)
        {
            const auto &publication =
                params_.routed_pipeline_kernel_owner->router_q8_publication;
            if (!moe_kernel_ ||
                !moe_kernel_->bindRouterQ8HiddenPublication(
                    publication,
                    MoERouterQ8PublicationAccess::ProducerAndConsumer))
            {
                LOG_ERROR("[MoERoutingStage] Failed to bind the graph-local "
                          "router Q8 publication on "
                          << params_.device_id.to_string());
                return nullptr;
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

    MoERouteLaunchKind MoERoutingStage::routeLaunchKind() const noexcept
    {
        if (params_.force_decode_equivalent_verifier_prefill)
            return MoERouteLaunchKind::DecodeEquivalentVerifier;
        if (params_.seq_len == 1 &&
            !params_.force_grouped_verifier_prefill_for_decode)
        {
            return MoERouteLaunchKind::RuntimeDecode;
        }
        return MoERouteLaunchKind::GroupedPrefill;
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

    bool MoERoutingStage::publishCPUGroupedRoutingEvidence(
        ExpertHistogramSource source) const
    {
        if (!params_.decode_histogram)
            return true;
        if (source == ExpertHistogramSource::GroupedVerifier)
        {
            LOG_ERROR("[MoERoutingStage] CPU grouped-verifier demand must be "
                      "published by the accepted-state transaction");
            return false;
        }
        if (!params_.device_id.is_cpu())
        {
            LOG_ERROR("[MoERoutingStage] Host grouped-routing evidence is CPU-only; "
                      "GPU evidence must remain device-owned"
                      << " device=" << params_.device_id.toString()
                      << " layer=" << params_.layer_idx);
            return false;
        }
        if (params_.layer_idx < 0)
        {
            LOG_ERROR("[MoERoutingStage] CPU grouped-routing evidence requires a valid layer index");
            return false;
        }

        const int logical_rows =
            params_.host_logical_row_count > 0
                ? params_.host_logical_row_count
                : params_.seq_len == 1
                      ? 1
                      : 0;
        if (logical_rows <= 0 || logical_rows > params_.seq_len)
        {
            LOG_ERROR("[MoERoutingStage] CPU grouped-routing evidence has invalid logical geometry"
                      << " logical_rows=" << logical_rows
                      << " physical_rows=" << params_.seq_len
                      << " layer=" << params_.layer_idx);
            return false;
        }

        const size_t required_routes =
            static_cast<size_t>(logical_rows) *
            static_cast<size_t>(params_.top_k);
        if (cached_routing_.expert_indices.size() < required_routes)
        {
            LOG_ERROR("[MoERoutingStage] CPU grouped-routing evidence is incomplete"
                      << " required_routes=" << required_routes
                      << " available_routes="
                      << cached_routing_.expert_indices.size()
                      << " layer=" << params_.layer_idx);
            return false;
        }

        const ExpertHistogramMergeResult merged =
            params_.decode_histogram->mergeRoutedExpertRows(
                cached_routing_.expert_indices.data(),
                RoutedExpertHistogramMerge{
                    .source = source,
                    .layer_idx = params_.layer_idx,
                    .real_token_count = logical_rows,
                    .bucket_token_count = params_.seq_len,
                    .top_k = params_.top_k,
                    .route_stride = params_.top_k,
                    .count_window_tokens = true,
                });
        if (!merged)
        {
            LOG_ERROR("[MoERoutingStage] CPU grouped-routing evidence publication failed"
                      << " layer=" << params_.layer_idx
                      << " logical_rows=" << logical_rows
                      << " physical_rows=" << params_.seq_len
                      << " reason=" << merged.error);
            return false;
        }

        if (PerfStatsCollector::isDomainEnabled("moe"))
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "cpu_grouped_routing_evidence_rows",
                static_cast<double>(logical_rows),
                source == ExpertHistogramSource::GroupedVerifier
                    ? "verifier"
                    : "prefill",
                params_.device_id.toString(),
                PerfStatsCollector::Tags{
                    {"layer", std::to_string(params_.layer_idx)},
                    {"physical_rows", std::to_string(params_.seq_len)}});
        }
        return true;
    }

    bool MoERoutingStage::validateHostGroupedVerifierHistogramPublication(
        const int32_t *accepted_state_counts,
        int request_count,
        int rows_per_request,
        std::string *error) const
    {
        auto fail = [&](std::string reason) -> bool
        {
            if (error)
                *error = std::move(reason);
            return false;
        };

        if (!requiresHostGroupedVerifierHistogramPublication())
        {
            return fail(
                "router does not own deferred CPU grouped-verifier demand");
        }
        if (!accepted_state_counts)
            return fail("accepted_state_counts must not be null");
        if (params_.layer_idx < 0)
            return fail("routed layer index must be non-negative");
        if (request_count <= 0 || rows_per_request <= 0)
            return fail("request geometry must be positive");
        if (params_.seq_len != request_count * rows_per_request)
        {
            std::ostringstream msg;
            msg << "router row geometry does not match request-major verifier shape"
                << " stage_rows=" << params_.seq_len
                << " request_count=" << request_count
                << " rows_per_request=" << rows_per_request;
            return fail(msg.str());
        }
        if (params_.host_logical_row_count != params_.seq_len)
        {
            std::ostringstream msg;
            msg << "CPU verifier router must retain every physical request row"
                << " logical_rows=" << params_.host_logical_row_count
                << " stage_rows=" << params_.seq_len;
            return fail(msg.str());
        }
        if (params_.top_k <= 0)
            return fail("router top_k must be positive");

        for (int request = 0; request < request_count; ++request)
        {
            const int accepted = accepted_state_counts[request];
            if (accepted < 0 || accepted > rows_per_request)
            {
                std::ostringstream msg;
                msg << "accepted row count is outside its verifier group"
                    << " request=" << request
                    << " accepted=" << accepted
                    << " rows_per_request=" << rows_per_request;
                return fail(msg.str());
            }
        }

        const size_t required_routes =
            static_cast<size_t>(params_.seq_len) *
            static_cast<size_t>(params_.top_k);
        if (cached_routing_.expert_indices.size() < required_routes)
        {
            std::ostringstream msg;
            msg << "retained CPU verifier routes are incomplete"
                << " available=" << cached_routing_.expert_indices.size()
                << " required=" << required_routes;
            return fail(msg.str());
        }
        return true;
    }

    bool MoERoutingStage::publishHostGroupedVerifierHistograms(
        const int32_t *accepted_state_counts,
        int request_count,
        int rows_per_request,
        std::string *error)
    {
        if (!validateHostGroupedVerifierHistogramPublication(
                accepted_state_counts,
                request_count,
                rows_per_request,
                error))
        {
            return false;
        }

        uint64_t accepted_rows = 0;
        for (int request = 0; request < request_count; ++request)
        {
            const int accepted = accepted_state_counts[request];
            if (accepted == 0)
                continue;

            const size_t route_offset =
                static_cast<size_t>(request) *
                static_cast<size_t>(rows_per_request) *
                static_cast<size_t>(params_.top_k);
            const ExpertHistogramMergeResult merged =
                params_.decode_histogram->mergeRoutedExpertRows(
                    cached_routing_.expert_indices.data() + route_offset,
                    RoutedExpertHistogramMerge{
                        .source = ExpertHistogramSource::GroupedVerifier,
                        .layer_idx = params_.layer_idx,
                        .real_token_count = accepted,
                        .bucket_token_count = rows_per_request,
                        .top_k = params_.top_k,
                        .route_stride = params_.top_k,
                        .count_window_tokens = true,
                    });
            if (!merged)
            {
                if (error)
                {
                    std::ostringstream msg;
                    msg << "histogram rejected accepted verifier prefix"
                        << " layer=" << params_.layer_idx
                        << " request=" << request
                        << " accepted=" << accepted
                        << " reason=" << merged.error;
                    *error = msg.str();
                }
                return false;
            }
            accepted_rows += static_cast<uint64_t>(accepted);
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "cpu_committed_grouped_verifier_histogram_rows",
            static_cast<double>(accepted_rows),
            "verifier",
            params_.device_id.toString(),
            {{"layer", std::to_string(params_.layer_idx)},
             {"requests", std::to_string(request_count)},
             {"rows_per_request", std::to_string(rows_per_request)}});
        return true;
    }

    bool MoERoutingStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        (void)ctx;
        if (params_.device_id.is_gpu() && !stream)
        {
            LOG_ERROR("[MoERoutingStage] Graph launch preparation requires the "
                      "exact non-null producer stream");
            return false;
        }
        if (stream)
            setGPUStream(stream);

        /* Async histogram banks are initialized on their maintenance stream.
         * Admit the exact routing stream now, while the graph owner can legally
         * join that uncaptured setup event. execute() will only publish the
         * already-prepared identity from inside the capture interval. */
        if (isRuntimeTableDecodeGraphCapturable() &&
            params_.collect_device_runtime_histogram)
        {
            try
            {
                params_.moe_runtime_table->prepareDecodeHistogramProducerStream(
                    stream);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("[MoERoutingStage] Failed to prepare the runtime "
                          "histogram producer before graph capture"
                          << " device=" << params_.device_id.toString()
                          << " layer=" << params_.layer_idx
                          << " reason=" << error.what());
                return false;
            }
        }

        /*
         * Resolve and bind the backend wrapper before beginCapture(). Creating
         * the wrapper from execute() used to require an eager routing pass; no
         * routing values or model outputs are produced by this preparation.
         */
        IMoEKernel *kernel = nullptr;
        if (params_.device_id.is_gpu())
            kernel = ensureMoEKernel();
        if (params_.device_id.is_gpu() && !kernel)
        {
            LOG_ERROR("[MoERoutingStage] Failed to bind the persistent MoE "
                      "routing kernel before graph capture");
            return false;
        }

        if (params_.device_id.is_gpu())
        {
            const MoERouteLaunchPlan launch_plan{
                .kind = routeLaunchKind(),
                .physical_rows = params_.seq_len,
                .d_model = params_.d_model,
                .num_experts = params_.num_experts,
                .top_k = params_.top_k,
            };
            if (!kernel->prepareRouteLaunch(
                    params_.gate_weights,
                    launch_plan))
            {
                LOG_ERROR("[MoERoutingStage] Failed to prepare persistent "
                          "routing resources before graph capture"
                          << " device=" << params_.device_id.toString()
                          << " layer=" << params_.layer_idx
                          << " kind=" << static_cast<int>(launch_plan.kind)
                          << " rows=" << launch_plan.physical_rows);
                return false;
            }

            /*
             * Snapshot topology is finalized after launch preparation and
             * before beginCapture(). Require the diagnostic descriptor to
             * identify the same persistent route-logit buffer that the kernel
             * just prepared. A missing view is an incomplete graph workspace,
             * not permission to omit raw routing evidence.
             */
            if (!bindRouterLogitsDeviceView())
            {
                LOG_ERROR("[MoERoutingStage] Failed to bind canonical device "
                          "router-logit diagnostics before graph capture"
                          << " device=" << params_.device_id.toString()
                          << " layer=" << params_.layer_idx
                          << " rows=" << params_.seq_len
                          << " experts=" << params_.num_experts);
                return false;
            }
        }

        if (params_.seq_len > 1 && params_.active_row_count_device &&
            PerfStatsCollector::isDomainEnabled("moe"))
        {
            PerfStatsCollector::addCounter(
                "moe",
                "routing_device_row_count_bound",
                1.0,
                "prefill",
                params_.device_id.toString(),
                PerfStatsCollector::Tags{
                    {"physical_rows", std::to_string(params_.seq_len)},
                    {"row_count_source", "device_request_geometry"},
                    {"layer", std::to_string(params_.layer_idx)}});
        }
        return true;
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
                    full_output_weights,
                    params_.active_row_count_device,
                    params_.defer_overlay_grouped_verifier_histogram_publication
                        ? moe_runtime_layer_
                        : nullptr))
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
        /*
         * Acceptance is produced after the verifier graph completes.  Retain
         * request-major top-k rows here; DeviceGraphOrchestrator commits only
         * accepted prefixes as part of the CPU spec-state transaction.
         */
        return true;
    }

    bool MoERoutingStage::enqueueCommittedGroupedVerifierHistograms(
        const int32_t *accepted_state_counts_device,
        const int32_t *publication_ok_flags_device,
        int request_count,
        int rows_per_request,
        void *producer_stream)
    {
        if (!requiresCommittedGroupedVerifierHistogramPublication())
        {
            LOG_ERROR(
                "[MoERoutingStage] Committed grouped-verifier publication was "
                "requested from a router that does not own the overlay ticket ledger"
                << " layer=" << params_.layer_idx);
            return false;
        }
        if (!producer_stream ||
            !accepted_state_counts_device ||
            !publication_ok_flags_device ||
            request_count <= 0 ||
            rows_per_request <= 0 ||
            params_.seq_len != request_count * rows_per_request)
        {
            LOG_ERROR(
                "[MoERoutingStage] Invalid committed overlay-verifier histogram shape"
                << " layer=" << params_.layer_idx
                << " request_count=" << request_count
                << " rows_per_request=" << rows_per_request
                << " stage_rows=" << params_.seq_len
                << " stream=" << producer_stream);
            return false;
        }
        if (!hasCompleteOverlayVerifierLedgerBinding())
        {
            LOG_ERROR(
                "[MoERoutingStage] Committed overlay-verifier publication has "
                "no complete per-layer selected-route ledger"
                << " layer=" << params_.layer_idx);
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
        {
            LOG_ERROR(
                "[MoERoutingStage] Committed overlay-verifier publication "
                "could not resolve its graph-owned kernel"
                << " layer=" << params_.layer_idx);
            return false;
        }

        const MoEKernelLaunchContext launch{
            .stream = producer_stream,
            .workspace = bound_workspace_};
        if (!kernel->commitGroupedVerifierHistograms(
                launch,
                moe_runtime_layer_,
                accepted_state_counts_device,
                publication_ok_flags_device,
                request_count,
                rows_per_request,
                params_.seq_len,
                params_.num_experts,
                params_.top_k))
        {
            LOG_ERROR(
                "[MoERoutingStage] Backend rejected committed overlay-verifier "
                "histogram publication"
                << " layer=" << params_.layer_idx);
            return false;
        }
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

        if (isRuntimeTableDecodeGraphCapturable())
        {
            void *route_stream = gpuStream();
            if (!route_stream)
            {
                LOG_ERROR("[MoERoutingStage] Runtime-table GPU decode routing requires an explicit stream on "
                          << params_.device_id.toString());
                return false;
            }
            if (params_.collect_device_runtime_histogram)
            {
                try
                {
                    params_.moe_runtime_table->recordDecodeHistogramProducerStream(
                        route_stream);
                }
                catch (const std::exception &error)
                {
                    LOG_ERROR("[MoERoutingStage] Runtime histogram producer was not "
                              "prepared for graph execution"
                              << " device=" << params_.device_id.toString()
                              << " layer=" << params_.layer_idx
                              << " reason=" << error.what());
                    return false;
                }
            }

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
                    params_.collect_device_runtime_histogram,
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
                    params_.device_rebalance_command_buffer_count,
                    params_.absolute_position_ids_device,
                    params_.routed_row_execution_policy);
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
                    params_.collect_device_runtime_histogram,
                    params_.absolute_position_ids_device,
                    params_.routed_row_execution_policy);
            }

            if (!routed)
            {
                LOG_ERROR("[MoERoutingStage] Runtime-table decode routing failed");
                return false;
            }

            LOG_TRACE("[MoERoutingStage] Runtime-routed single token to top-"
                      << top_k << " of " << num_experts << " experts");
            if (params_.collect_device_runtime_histogram)
                recordRuntimeHistogramTokenBoundary();
            return true;
        }

        if (isOverlayTicketDecodeGraphCapturable())
        {
            /*
             * The following captured ticket-publish stage copies these exact
             * device tensors into its fixed host-visible ABI.  Placement and
             * histogram ownership live beyond that explicit heterogeneous
             * boundary, so this route neither creates a host mirror nor
             * invents a device runtime table whose expert descriptors would
             * belong to several backends.
             */
            if (!kernel->routeWithTensors(
                    params_.input,
                    params_.gate_weights,
                    seq_len,
                    d_model,
                    num_experts,
                    top_k,
                    params_.norm_topk_prob,
                    params_.output_indices,
                    params_.output_weights,
                    cached_routing_))
            {
                LOG_ERROR("[MoERoutingStage] Fixed-capacity ExpertOverlay "
                          "ticket decode routing failed");
                return false;
            }
            PerfStatsCollector::addCounter(
                "moe_overlay",
                "ticket_decode_route_publications",
                1.0,
                "decode",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"authority", "fixed_capacity_ticket"}});
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
                      "either the runtime-table device path or an explicit "
                      "fixed-capacity ExpertOverlay ticket on "
                      << params_.device_id.toString());
            return false;
        }

        const int32_t *device_active_rows =
            params_.device_id.is_gpu() && seq_len > 1
                ? params_.active_row_count_device
                : nullptr;
        if (device_active_rows &&
            PerfStatsCollector::isDomainEnabled("moe"))
        {
            /*
             * The stage deliberately does not read this scalar on the host.
             * Request admission or verifier preparation publishes it before
             * graph consumption, and the backend kernel masks the physical
             * suffix on every capture and replay, including a full first use.
             */
            PerfStatsCollector::addCounter(
                "moe",
                "routing_device_row_count_execute",
                1.0,
                "prefill",
                params_.device_id.toString(),
                PerfStatsCollector::Tags{
                    {"physical_rows", std::to_string(seq_len)},
                    {"row_count_source", "device_request_geometry"},
                    {"layer", std::to_string(params_.layer_idx)}});
        }

        const bool routed = device_active_rows
                                ? kernel->routeWithTensorsEffectiveSeqLen(
                                      params_.input, params_.gate_weights,
                                      seq_len, d_model, num_experts, top_k,
                                      params_.norm_topk_prob,
                                      params_.output_indices, params_.output_weights,
                                      cached_routing_,
                                      device_active_rows)
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

        // Publish exact host-owned routing evidence. Multi-row CPU execution
        // contributes every logical row at once; GPU evidence remains in its
        // device runtime table until an explicit maintenance boundary.
        if (params_.decode_histogram && params_.device_id.is_cpu() && seq_len > 1)
        {
            if (!publishCPUGroupedRoutingEvidence(
                    ExpertHistogramSource::PrefillChunk))
            {
                return false;
            }
        }
        else if (params_.decode_histogram && params_.layer_idx >= 0 && seq_len == 1)
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

    bool MoERoutingStage::supportsGraphCaptureAfterLaunchPreparation() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (params_.force_decode_equivalent_verifier_prefill)
            return isDecodeEquivalentVerifierPrefillGraphCaptureSupported();

        const bool decode_shape_supported =
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
            params_.output_weights;
        const bool decode_supported =
            decode_shape_supported &&
            ((params_.decode_route_publication ==
                  MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable &&
              params_.moe_runtime_table &&
              params_.layer_idx >= 0) ||
             (params_.decode_route_publication ==
                  MoEDecodeRoutePublicationPolicy::FixedCapacityOverlayTicket &&
              !params_.moe_runtime_table &&
              params_.layer_idx >= 0));

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
            << " row_execution_policy="
            << routedExpertRowExecutionPolicyToString(
                   params_.routed_row_execution_policy)
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
            << " decode_route_publication="
            << (params_.decode_route_publication ==
                        MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable
                    ? "device-runtime-table"
                    : "fixed-capacity-overlay-ticket")
            << " runtime_layer=" << (moe_runtime_layer_ ? "true" : "false")
            << " active_row_count_device="
            << (params_.active_row_count_device ? "true" : "false")
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
        return supportsLazyPrefillGraphCapturePreflight() &&
               params_.active_row_count_device != nullptr;
    }

    bool MoERoutingStage::supportsPaddedPrefillRealLengthContract() const
    {
        return isDeviceRoutedPrefillGraphCaptureSupported() &&
               params_.active_row_count_device != nullptr;
    }

    bool MoERoutingStage::isDeviceRoutedDecodeGraphCapturable() const
    {
        return isRuntimeTableDecodeGraphCapturable() ||
               isOverlayTicketDecodeGraphCapturable();
    }

    bool MoERoutingStage::isRuntimeTableDecodeGraphCapturable() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (params_.force_decode_equivalent_verifier_prefill)
            return false;

        if (params_.decode_route_publication !=
            MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable)
        {
            return false;
        }

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

    bool MoERoutingStage::isOverlayTicketDecodeGraphCaptureSupported() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        return params_.decode_route_publication ==
                   MoEDecodeRoutePublicationPolicy::FixedCapacityOverlayTicket &&
               !params_.force_decode_equivalent_verifier_prefill &&
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
               !params_.moe_runtime_table &&
               params_.layer_idx >= 0;
#endif
    }

    bool MoERoutingStage::isOverlayTicketDecodeGraphCapturable() const
    {
        /*
         * The ticket publisher consumes the ordinary device-authoritative
         * top-k tensors.  Launch preparation must already have resolved the
         * backend wrapper and every persistent router scratch address before
         * this stage can be admitted to a captured segment.
         */
        return isOverlayTicketDecodeGraphCaptureSupported() &&
               moe_kernel_ != nullptr;
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
        // isDeviceRoutedPrefillGraphCapturable() adds prepared-kernel readiness.
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               isDeviceRoutedPrefillExecutionSupported();
    }

    bool MoERoutingStage::isDeviceRoutedPrefillGraphCapturable() const
    {
        // Prefill routing is graph-capturable on supported GPU backends when
        // the full path is device-only and explicit launch preparation has
        // resolved the lazy MoE kernel. routeWithTensors() in non-snapshot
        // Release builds performs no D2H or backend stream synchronization.
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
               params_.output_weights &&
               hasCompleteOverlayVerifierLedgerBinding();
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

    bool MoERoutingStage::hasCompleteOverlayVerifierLedgerBinding() const
    {
        if (!params_.defer_overlay_grouped_verifier_histogram_publication)
            return true;
        if (!params_.device_id.is_gpu() ||
            !params_.force_decode_equivalent_verifier_prefill ||
            !params_.moe_runtime_table ||
            !moe_runtime_layer_ ||
            params_.layer_idx < 0 ||
            params_.seq_len <= 1 ||
            params_.top_k <= 0)
        {
            return false;
        }

        const auto &host_layer =
            params_.moe_runtime_table->hostLayerState(params_.layer_idx);
        const uint64_t required_routes =
            static_cast<uint64_t>(params_.seq_len) *
            static_cast<uint64_t>(params_.top_k);
        return host_layer.deferred_verifier_route_expert_ids != nullptr &&
               host_layer.deferred_verifier_route_participant_ids != nullptr &&
               static_cast<uint64_t>(
                   host_layer.deferred_verifier_route_capacity) >=
                   required_routes;
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

        /*
         * CPU routing may retain host results. GPU routing instead exposes the
         * canonical workspace through a pure-device tensor view so captured
         * diagnostics copy the exact post-softmax router probabilities D2D on
         * the producer stream. The workspace name is historical; every backend
         * publishes the interface-level probability contract after routing.
         */
        if (router_logits_device_view_)
            info.addOutput("router_logits", router_logits_device_view_.get(),
                           static_cast<size_t>(params_.seq_len),
                           static_cast<size_t>(params_.num_experts));
        else if (!router_logits_.empty())
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

    WorkspaceRequirements MoERoutingStage::getWorkspaceRequirements(
        int m,
        int,
        int) const
    {
        if (!params_.device_id.is_cuda() && !params_.device_id.is_rocm())
            return WorkspaceRequirements{};

        /*
         * A stage's tensors describe one concrete graph bucket, whereas `m`
         * describes the stable-address envelope shared by the complete serial
         * graph family.  Neither may weaken the other.  In particular, the
         * server can first capture a 39-row chat request and then admit a
         * 68-row multi-turn request; using only `params_.seq_len` would publish
         * 39 rows of router storage and make the second request impossible.
         */
        const int workspace_rows =
            std::max({1, m, params_.seq_len});
        WorkspaceRequirements reqs =
            params_.device_id.is_rocm()
                ? MoEWorkspaceBuffers::rocmRouting(
                      workspace_rows,
                      params_.d_model,
                      params_.num_experts)
                : MoEWorkspaceBuffers::cudaRouting(
                      workspace_rows,
                      params_.d_model,
                      params_.num_experts);

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
        /*
         * The view must never outlive the workspace address it describes.
         * Drop it before replacing either the workspace or the backend
         * kernel's scratch binding.
         */
        router_logits_device_view_.reset();
        bound_workspace_ = workspace;
        if (moe_kernel_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(moe_kernel_))
                consumer->bindWorkspace(workspace);
        }

        if (!bindRouterLogitsDeviceView() && workspace && params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoERoutingStage] GPU workspace does not expose the "
                      "complete canonical router-logit buffer"
                      << " device=" << params_.device_id.toString()
                      << " layer=" << params_.layer_idx
                      << " rows=" << params_.seq_len
                      << " experts=" << params_.num_experts);
        }
        invalidateDumpInfoCache();
    }

    bool MoERoutingStage::bindRouterLogitsDeviceView()
    {
        if (!params_.device_id.is_gpu() || !bound_workspace_)
            return true;

        if (params_.seq_len <= 0 || params_.num_experts <= 0 ||
            bound_workspace_->device() != params_.device_id)
        {
            router_logits_device_view_.reset();
            return false;
        }

        const size_t rows = static_cast<size_t>(params_.seq_len);
        const size_t experts = static_cast<size_t>(params_.num_experts);
        if (rows > std::numeric_limits<size_t>::max() / experts ||
            rows * experts > std::numeric_limits<size_t>::max() / sizeof(float))
        {
            router_logits_device_view_.reset();
            return false;
        }

        const size_t required_bytes = rows * experts * sizeof(float);
        void *const route_logits =
            bound_workspace_->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS);
        const size_t available_bytes =
            bound_workspace_->getBufferSize(MoEWorkspaceBuffers::ROUTE_LOGITS);
        if (!route_logits || available_bytes < required_bytes)
        {
            router_logits_device_view_.reset();
            return false;
        }

        if (router_logits_device_view_ &&
            router_logits_device_view_->gpu_data_ptr() == route_logits &&
            router_logits_device_view_->shape() ==
                std::vector<size_t>{rows, experts})
        {
            return true;
        }

        router_logits_device_view_ = std::make_unique<GpuTensorView>(
            route_logits,
            rows,
            experts,
            TensorType::FP32,
            params_.device_id);
        invalidateDumpInfoCache();
        return true;
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
