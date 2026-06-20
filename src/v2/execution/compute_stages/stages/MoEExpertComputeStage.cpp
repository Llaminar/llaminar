/**
 * @file MoEExpertComputeStage.cpp
 * @brief Implementation of unified MoE FFN, shared expert, and shared expert gate stages
 */

#include "MoEExpertComputeStage.h"
#include "../ComputeStageUtils.h"
#include "../../../execution/moe/DecodeExpertHistogram.h"
#include "../../../execution/moe/ExpertWeightTransfer.h"
#include "../../../execution/moe/ExpertWeightPayloadProvider.h"
#include "../../../execution/moe/MoEExpertWeightService.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
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
#include "../../../utils/PerfStatsCollector.h"
#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <typeindex>
#include <unordered_set>
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
            DeviceId device, void *stream,
            DeviceWorkspaceManager *workspace)
        {
            // Try fused path first (quantized GEMM engines support this)
            if (down_gemm->multiply_tensor_with_fused_swiglu(
                    gate_tensor, up_tensor, output,
                    m, n, intermediate,
                    1.0f, 0.0f,
                    workspace))
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
                m, n, intermediate,
                true,
                1.0f, 0.0f,
                nullptr,
                nullptr,
                device.toKernelDeviceIndex(),
                workspace);
            if (ok)
                markGpuTensorWritten(output, device, stream);
            return ok;
        }

        void markStandaloneGpuOutputWritten(TensorBase *output, DeviceId device, void *stream)
        {
            markGpuTensorWritten(output, device, stream);
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

        using VerifierKernelModeScopePtr =
            std::unique_ptr<ITensorGemm::VerifierKernelModeScope>;

        void appendVerifierDecodeEquivalentScope(
            ITensorGemm *kernel,
            std::vector<VerifierKernelModeScopePtr> &scopes,
            std::unordered_set<std::type_index> &scoped_backend_types)
        {
            if (!kernel)
                return;
            const std::type_index backend_type(typeid(*kernel));
            if (scoped_backend_types.find(backend_type) != scoped_backend_types.end())
                return;
            if (auto scope = kernel->beginVerifierDecodeEquivalentScope())
            {
                scoped_backend_types.insert(backend_type);
                scopes.emplace_back(std::move(scope));
            }
        }

        /**
         * @brief Enter backend verifier modes for every GEMM used by a stage.
         *
         * Verifier replay can publish rows into live MTP/KV/GDN state, so stage
         * code must not know or guess backend-specific switches.  Instead it
         * asks each GEMM engine for an optional verifier mode scope.  CUDA and
         * ROCm use this to select generated small-M dispatch policies that are
         * reproducible against serial decode; CPU currently returns no scope.
         */
        std::vector<VerifierKernelModeScopePtr> beginVerifierDecodeEquivalentScopes(
            std::initializer_list<ITensorGemm *> kernels)
        {
            std::vector<VerifierKernelModeScopePtr> scopes;
            std::unordered_set<std::type_index> scoped_backend_types;
            scopes.reserve(kernels.size());
            for (ITensorGemm *kernel : kernels)
                appendVerifierDecodeEquivalentScope(kernel, scopes, scoped_backend_types);
            return scopes;
        }

        /**
         * @brief True when GPU kernels can consume explicit routing tensors.
         *
         * Integration builds define ENABLE_PIPELINE_SNAPSHOTS, which disables
         * graph-capture capability helpers so parity can keep richer dumps.
         * The underlying CUDA/ROCm grouped `FromRouting` kernels are still the
         * correct execution path for verifier row replay because the routing
         * row is device-owned and has no reliable host mirror.
         */
        bool supportsDeviceRoutingTensorDecodeExecutionBackend(DeviceId device)
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

        bool shouldUseSharedExpertGroupedDecode(DeviceId device)
        {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
            (void)device;
            return false;
#else
#if defined(HAVE_CUDA)
            if (device.is_cuda())
            {
                /*
                 * CUDA keeps the shared-expert decode shortcut enabled once
                 * its pointer-array metadata is workspace-backed.  The warmup
                 * run uploads stable per-stage pointer arrays into graph-owned
                 * slots; capture then replays kernels that reference those
                 * immutable slots instead of kernel-scoped device allocation pools.
                 */
                return true;
            }
#endif
#if defined(HAVE_ROCM)
            if (device.is_rocm())
                return debugEnv().rocm.shared_expert_grouped_decode;
#endif
            return false;
#endif
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
            if (supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) && params_.seq_len == 1)
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
        runtime_grouped_decode_warmed_ = false;
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
            runtime_grouped_decode_warmed_ = false;
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
        runtime_grouped_decode_warmed_ = false;
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
            if (!output_tensor || !output_tensor->ensureOnDevice(params_.device_id, gpuStream()))
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

        // Fast path for ordinary decode (seq_len=1): eliminates gather/scatter
        // overhead. MTP verifier correction replay explicitly opts into the
        // grouped prefill route so rejected-token replay stays on the same
        // fused graph-captured path as the verifier rows.
        if (params_.seq_len == 1 && !params_.force_grouped_verifier_prefill_for_decode)
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

        if (params_.force_decode_equivalent_verifier_prefill &&
            seq_len > 1)
        {
            return executeDecodeEquivalentVerifierPrefill(ctx);
        }

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
        const bool use_fixed_topology_grouped_prefill =
            is_gpu && canUseFixedTopologyGroupedPrefill();
        if (use_fixed_topology_grouped_prefill)
        {
            /*
             * The grouped prefill kernel owns output initialization because
             * its final scatter-add path accumulates routed expert slots into
             * the dense output. Keep that clear inside the backend pipeline so
             * graph replay captures exactly one initialization node per MoE
             * layer. The generic zero below remains for the older paths where
             * the stage itself owns accumulation setup.
             */
            // Ensure descriptor tables are built (lazy, only first call)
            bool tables_ready = true;
            if (static_cast<int>(all_expert_ids_.size()) != num_experts)
            {
                all_expert_ids_.resize(static_cast<size_t>(num_experts));
                std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
            }

            if (canUseSafeCombinedSharedVerifierComposite())
            {
                tables_ready = ensureGemmEnginesForExperts(all_expert_ids_) &&
                               ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate) &&
                               ensureGroupedDownDescriptorTable(kernel, d_model, intermediate) &&
                               ensureCombinedSharedVerifierResources(kernel, d_model, intermediate);
            }
            else if (grouped_gateup_desc_table_id_ < 0 || grouped_down_desc_table_id_ < 0)
            {
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

            const bool grouped_ok = canUseSafeCombinedSharedVerifierComposite()
                                        ? executeSafeCombinedSharedVerifierComposite(kernel)
                                        : executeFixedTopologyGroupedPrefill(kernel, seq_len);
            if (!grouped_ok)
            {
                LOG_ERROR("[MoEExpertComputeStage] Grouped prefill pipeline failed, layer "
                          << params_.layer_idx);
                return false;
            }

            if (params_.device_id.is_gpu())
                markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
            return true;
        }

        // Zero the output buffer via tensor-aware kernel (works for both CPU and GPU).
        const size_t output_bytes = static_cast<size_t>(seq_len) * d_model * sizeof(float);
        kernel->zeroBuffer(params_.output, output_bytes);

        const bool forced_verifier_decode_replay =
            params_.seq_len == 1 && params_.force_grouped_verifier_prefill_for_decode;
        if (forced_verifier_decode_replay && isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] MTP verifier correction replay requested grouped "
                      "prefill for seq_len=1 inside graph capture, but the fixed-topology grouped path "
                      "is unavailable: "
                      "device=" << params_.device_id.to_string()
                                << ", moe_grouped_prefill=" << debugEnv().rocm.moe_grouped_prefill
                                << ", fullOwnership=" << hasFullLocalExpertOwnership()
                                << ", allEnabled=" << expertMaskAllEnabled()
                                << ", replicas=" << params_.replica_set.num_replicated
                                << ", layer=" << params_.layer_idx);
            return false;
        }

        // ROCm graph capture requires the fixed-topology grouped path. Ordinary
        // stream execution can still run graph-native overlay subsets through the
        // device-grouped gather/scatter path below.
        if (params_.device_id.is_rocm() && params_.seq_len > 1 && isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] ROCm graph-captured prefill (seq_len=" << params_.seq_len
                                                                                      << ") requires fixed-topology grouped prefill but conditions not met: "
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
                markGpuTensorWritten(scratch_batch_.get(), params_.device_id, gpuStream());

                ITensorGemm *gate_gemm = cached_gate_gemm_[expert_id];
                ITensorGemm *up_gemm = cached_up_gemm_[expert_id];
                ITensorGemm *down_gemm = cached_down_gemm_[expert_id];

                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {gate_gemm, scratch_gate_.get(), intermediate, nullptr, "gate"},
                    {up_gemm, scratch_up_.get(), intermediate, nullptr, "up"}};
                if (!gate_gemm->multiply_fused_tensor(
                        scratch_batch_.get(), projections, count, d_model,
                        nullptr, getWorkspace()))
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
                        params_.device_id, gpuStream(), getWorkspace()))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CUDA SwiGLU/down projection failed for expert "
                              << expert_id << " layer " << params_.layer_idx);
                    return false;
                }

                kernel->scatterExpertResults(
                    params_.output, scratch_out_.get(), expert_id, d_model);
                markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
            }

            if (!params_.output_registered_in_arena)
                markStandaloneGpuOutputWritten(params_.output, params_.device_id, gpuStream());

            LOG_TRACE("[MoEExpertComputeStage] GPU prefill: " << seq_len << " tokens, "
                                                              << top_k << " experts per token");
            return true;
        }

        if (forced_verifier_decode_replay)
        {
            /*
             * MTP correction replay rows are part of the verifier publication
             * contract.  Full-ownership graphs use the fixed descriptor-table
             * route above; LocalTP/overlay graphs intentionally use this
             * mask-aware device grouping route.  Do not fall through to the
             * host-routed CPU grouping path when device grouping is unavailable,
             * because that would split verifier state ownership and hide a real
             * graph/runtime capability gap.
             */
            LOG_ERROR("[MoEExpertComputeStage] MTP verifier correction replay could not use "
                      "the required device grouped path for seq_len=1: "
                      "device=" << params_.device_id.to_string()
                                << ", moe_grouped_prefill=" << debugEnv().rocm.moe_grouped_prefill
                                << ", fullOwnership=" << hasFullLocalExpertOwnership()
                                << ", allEnabled=" << expertMaskAllEnabled()
                                << ", replicas=" << params_.replica_set.num_replicated
                                << ", layer=" << params_.layer_idx);
            return false;
        }

        if (params_.device_id.is_rocm() && params_.seq_len > 1)
        {
            LOG_ERROR("[MoEExpertComputeStage] ROCm prefill (seq_len=" << params_.seq_len
                                                                       << ") requires fixed-topology grouped prefill or device expert grouping; "
                                                                       << "prepareExpertGroups failed and CPU fallback is not allowed for ROCm overlay execution"
                                                                       << ", layer=" << params_.layer_idx);
            return false;
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
                if (expert_id < 0 || expert_id >= num_experts)
                {
                    LOG_ERROR("[MoEExpertComputeStage] Invalid routed expert id " << expert_id
                                                                                  << " at token " << t
                                                                                  << " slot " << k
                                                                                  << " for layer " << params_.layer_idx
                                                                                  << " (num_experts=" << num_experts << ")");
                    return false;
                }
                if (!std::isfinite(weight))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Non-finite route weight at token " << t
                                                                                          << " slot " << k
                                                                                          << " for layer " << params_.layer_idx);
                    return false;
                }
                if (weight == 0.0f)
                    continue;
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
            if (is_gpu)
                markGpuTensorWritten(scratch_batch_.get(), params_.device_id, gpuStream());

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
                    num_tokens, d_model,
                    nullptr, getWorkspace()))
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
                    params_.device_id, gpuStream(), getWorkspace()))
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
            if (is_gpu)
                markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
        }

        LOG_TRACE("[MoEExpertComputeStage] Processed " << seq_len << " tokens via GEMM kernels, "
                                                       << top_k << " experts per token");
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

        if (supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.moe_runtime_table && params_.layer_idx >= 0 && !moe_runtime_layer_)
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
        bool runtime_decode_bank_active_for_expert_stage =
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            moe_runtime_layer_ &&
            runtimeTableHasActiveGroupedDecodeBank();
        if (supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            (!moe_runtime_table_initialized_ || !runtime_decode_bank_active_for_expert_stage))
        {
            moe_runtime_table_initialized_ = runtime_decode_bank_active_for_expert_stage ||
                                             initializeMoERuntimeTableForGroupedDecode();
            /*
             * The first decode step after clear_cache() may initialize the
             * placement/runtime bank itself.  Treat that freshly initialized
             * bank as available for the same warmup pass so the fused runtime
             * grouped path is warmed before segmented capture is armed.
             */
            runtime_decode_bank_active_for_expert_stage =
                moe_runtime_table_initialized_ &&
                moe_runtime_layer_ &&
                runtimeTableHasActiveGroupedDecodeBank();
        }

        // Snapshot-enabled builds keep legacy routing tensors authoritative for
        // parity dumps, so MoERoutingStage does not populate the runtime top-k
        // table. Consuming that table here would use stale/zero routing data.
#if defined(ENABLE_PIPELINE_SNAPSHOTS)
        const bool can_try_device_routed_decode = false;
#else
        const bool can_try_device_routed_decode =
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.moe_runtime_table &&
            moe_runtime_layer_ &&
            moe_runtime_table_initialized_ &&
            runtime_decode_bank_active_for_expert_stage &&
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
                /*
                 * CUDA and ROCm both expose a backend-owned fused runtime decode
                 * path. Prefer it over the stage-managed two-step scratch path so
                 * captured decode graphs do not depend on per-slot TensorBase
                 * handoff objects in the hot loop.
                 */
                const bool try_fused_runtime_decode =
                    params_.device_id.is_cuda() || params_.device_id.is_rocm();
                if (try_fused_runtime_decode)
                {
                    device_routed_done = kernel->groupedExpertDecodeFromRuntime(
                        moe_runtime_layer_,
                        input_tensor,
                        grouped_gateup_desc_table_id_,
                        grouped_down_desc_table_id_,
                        top_k,
                        params_.output,
                        d_model,
                        intermediate);

                    if (!device_routed_done && isGraphCaptureActive())
                    {
                        LOG_ERROR("[MoEExpertComputeStage] Fused runtime grouped decode path unavailable during graph capture "
                                  "for layer "
                                  << params_.layer_idx);
                        return false;
                    }
                    if (device_routed_done && !isGraphCaptureActive())
                        runtime_grouped_decode_warmed_ = true;
                }

                if (!device_routed_done)
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
                }
            }

            if (!device_routed_done && isGraphCaptureActive())
            {
                LOG_ERROR("[MoEExpertComputeStage] Device-routed grouped decode path unavailable during graph capture "
                          "for layer "
                          << params_.layer_idx);
                return false;
            }

            if (grouped_tables_ready && !device_routed_done)
            {
                LOG_DEBUG("[MoEExpertComputeStage] Device-routed grouped decode path unavailable for layer "
                          << params_.layer_idx << "; using host-routed fallback");
            }

            if (device_routed_done)
            {
                markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
                return true;
            }

            kernel->zeroBuffer(params_.output, static_cast<size_t>(d_model) * sizeof(float));
        }

        const bool require_device_routing_tensor_decode =
            params_.require_device_routing_tensor_decode;
        const bool can_try_device_routing_tensor_decode =
            is_gpu &&
            supportsDeviceRoutingTensorDecodeExecutionBackend(params_.device_id) &&
            params_.routing_indices &&
            params_.routing_weights &&
            top_k > 0 && top_k <= 16 &&
            params_.replica_set.num_replicated == 0 &&
            hasFullLocalExpertOwnership() &&
            expertMaskAllEnabled();

        if (can_try_device_routing_tensor_decode)
        {
            bool device_routing_done = false;
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

                const bool gateup_done = kernel->groupedExpertGateUpDecodeFromRouting(
                    input_tensor,
                    params_.routing_indices,
                    grouped_gateup_desc_table_id_,
                    top_k,
                    gate_outputs,
                    up_outputs,
                    d_model,
                    intermediate);

                if (gateup_done)
                {
                    device_routing_done = kernel->groupedExpertDownDecodeFromRouting(
                        gate_outputs,
                        up_outputs,
                        params_.routing_indices,
                        params_.routing_weights,
                        grouped_down_desc_table_id_,
                        top_k,
                        params_.output,
                        d_model,
                        intermediate);
                }
            }

            if (device_routing_done)
            {
                markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
                return true;
            }

            if (require_device_routing_tensor_decode || isGraphCaptureActive())
            {
                LOG_ERROR("[MoEExpertComputeStage] Device routing tensor decode failed for layer "
                          << params_.layer_idx
                          << " (tables_ready=" << grouped_tables_ready
                          << ", top_k=" << top_k
                          << ", full_ownership=" << hasFullLocalExpertOwnership()
                          << ", expert_mask_all_enabled=" << expertMaskAllEnabled()
                          << ", replicas=" << params_.replica_set.num_replicated << ")");
                return false;
            }

            kernel->zeroBuffer(params_.output, static_cast<size_t>(d_model) * sizeof(float));
        }
        else if (require_device_routing_tensor_decode)
        {
            LOG_ERROR("[MoEExpertComputeStage] Device routing tensor decode required but unavailable for layer "
                      << params_.layer_idx
                      << " (device=" << params_.device_id.toString()
                      << ", grouped_decode=" << debugEnv().rocm.moe_grouped_decode
                      << ", device_routed_decode=" << debugEnv().rocm.moe_device_routed_decode
                      << ", top_k=" << top_k
                      << ", has_indices=" << (params_.routing_indices != nullptr)
                      << ", has_weights=" << (params_.routing_weights != nullptr)
                      << ", full_ownership=" << hasFullLocalExpertOwnership()
                      << ", expert_mask_all_enabled=" << expertMaskAllEnabled()
                      << ", replicas=" << params_.replica_set.num_replicated << ")");
            return false;
        }

#if !defined(ENABLE_PIPELINE_SNAPSHOTS)
        if (is_gpu && isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU MoE decode entered graph capture without "
                      "a usable device-routed runtime table; refusing host-routed fallback for layer "
                      << params_.layer_idx);
            return false;
        }

        if (is_gpu &&
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.moe_runtime_table &&
            !can_try_device_routed_decode)
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU MoE decode has a runtime table but cannot "
                      "use the device-owned route/expert path; refusing host-routed fallback for layer "
                      << params_.layer_idx
                      << " (initialized=" << moe_runtime_table_initialized_
                      << ", runtime_layer=" << (moe_runtime_layer_ != nullptr)
                      << ", full_ownership=" << hasFullLocalExpertOwnership()
                      << ", expert_mask_all_enabled=" << expertMaskAllEnabled()
                      << ", replicas=" << params_.replica_set.num_replicated << ")");
            return false;
        }
#endif

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
        {
            routing_int_indices[k] = static_cast<int>(routing_idx_data[k]);
            if (routing_int_indices[k] < 0 || routing_int_indices[k] >= num_experts)
            {
                LOG_ERROR("[MoEExpertComputeStage] Invalid routed expert id " << routing_int_indices[k]
                                                                              << " at decode slot " << k
                                                                              << " for layer " << params_.layer_idx
                                                                              << " (num_experts=" << num_experts << ")");
                return false;
            }
            if (!std::isfinite(routing_wt_data[k]))
            {
                LOG_ERROR("[MoEExpertComputeStage] Non-finite decode route weight at slot "
                          << k << " for layer " << params_.layer_idx);
                return false;
            }
        }

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
                    input_tensor, batch_projections_, /*m=*/1, d_model,
                    nullptr, getWorkspace()))
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
                            params_.device_id, gpuStream(), getWorkspace()))
                    {
                        LOG_ERROR("[MoEExpertComputeStage] CUDA decode SwiGLU/down projection failed for expert "
                                  << info.expert_id << " layer " << params_.layer_idx);
                        return false;
                    }

                    // GPU weighted accumulate: output += weight * scratch_out
                    kernel->weightedAddFromTensors(
                        params_.output, scratch_out_.get(), info.weight, d_model);
                    markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
                }
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
                            params_.device_id, gpuStream(), getWorkspace()))
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

    bool MoEExpertComputeStage::executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;

        const bool is_gpu = params_.device_id.is_gpu();

        if (seq_len <= 1)
            return false;
        if (!params_.input || !params_.output ||
            !params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill missing tensors");
            return false;
        }

        const bool has_prepared_expert_state =
            (!params_.prepared_gate_gemm.empty() &&
             params_.prepared_gate_gemm.size() == static_cast<size_t>(num_experts)) ||
            (params_.prepared_store &&
             params_.gate_slab_ref.has_value() &&
             params_.up_slab_ref.has_value() &&
             params_.down_slab_ref.has_value());
        if (params_.expert_gate_views.empty() && !has_prepared_expert_state)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill requires prepared expert engines");
            return false;
        }
        if (is_gpu && isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent MoE verifier prefill is not graph-capturable yet; "
                      "multi-row publication requires a device-resident per-row routing table before capture");
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "moe_decode_equivalent_verifier_prefill_runs",
            1.0,
            "verifier",
            params_.device_id.toString(),
            {{"stage", "routed_expert"},
             {"layer", std::to_string(params_.layer_idx)},
             {"seq_len", std::to_string(params_.seq_len)},
             {"top_k", std::to_string(params_.top_k)}});
        if (top_k > 16)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill top_k="
                      << top_k << " exceeds stack capacity");
            return false;
        }

        const float *routing_idx_data = is_gpu ? nullptr : params_.routing_indices->data();
        const float *routing_wt_data = is_gpu ? nullptr : params_.routing_weights->data();
        if (!is_gpu && (!routing_idx_data || !routing_wt_data))
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill could not access routing tensors");
            return false;
        }

        /*
         * Publication rows are stateful: later KV/GDN/conv state is restored
         * from one selected verifier row.  A grouped M=2..4 MoE helper can be
         * numerically close to decode but still route differently a layer or two
         * later.  To keep one authoritative MoE decode contract, replay each
         * verifier row through the same one-token path used by serial decode,
         * then write the row result back into the original all-position output
         * tensor.  Do not force the grouped verifier-prefill route here: it is
         * a useful performance target, but it is not the serial decode contract
         * until a backend proves row-equivalence for the active codebooks.
        */
        TensorBase *full_input = params_.input;
        TensorBase *full_output = params_.output;
        TensorBase *full_routing_indices = params_.routing_indices;
        TensorBase *full_routing_weights = params_.routing_weights;
        ensureGemmEnginesCached();
        std::vector<VerifierKernelModeScopePtr> verifier_scopes;
        verifier_scopes.reserve(cached_gate_gemm_.size() + cached_up_gemm_.size() + cached_down_gemm_.size());
        std::unordered_set<std::type_index> scoped_backend_types;
        auto append_scopes_for = [&](const std::vector<ITensorGemm *> &kernels)
        {
            for (ITensorGemm *gemm : kernels)
                appendVerifierDecodeEquivalentScope(gemm, verifier_scopes, scoped_backend_types);
        };
        append_scopes_for(cached_gate_gemm_);
        append_scopes_for(cached_up_gemm_);
        append_scopes_for(cached_down_gemm_);
        IMoEKernel *kernel = is_gpu ? ensureMoEKernel() : nullptr;
        if (is_gpu)
        {
            kernel->zeroBuffer(full_output,
                               static_cast<size_t>(seq_len) *
                                   static_cast<size_t>(d_model) *
                                   sizeof(float));
            if (!scratch_batch_ ||
                scratch_batch_->shape() != std::vector<size_t>{1u, static_cast<size_t>(d_model)})
            {
                scratch_batch_ = makeScratchFP32(1, d_model, params_.device_id);
                scratch_capacity_ = std::max(scratch_capacity_, 1);
            }
            if (!verifier_output_row_ ||
                verifier_output_row_->shape() != std::vector<size_t>{1u, static_cast<size_t>(d_model)})
                verifier_output_row_ = makeScratchFP32(1, d_model, params_.device_id);
            if (!verifier_routing_indices_row_ ||
                verifier_routing_indices_row_->shape() != std::vector<size_t>{1u, static_cast<size_t>(top_k)})
                verifier_routing_indices_row_ = makeScratchFP32(1, top_k, params_.device_id);
            if (!verifier_routing_weights_row_ ||
                verifier_routing_weights_row_->shape() != std::vector<size_t>{1u, static_cast<size_t>(top_k)})
                verifier_routing_weights_row_ = makeScratchFP32(1, top_k, params_.device_id);
        }

        FP32Tensor row_input({1u, static_cast<size_t>(d_model)});
        FP32Tensor row_routing_indices({1u, static_cast<size_t>(top_k)});
        FP32Tensor row_routing_weights({1u, static_cast<size_t>(top_k)});
        FP32Tensor row_output({1u, static_cast<size_t>(d_model)});

        const float *input_data = is_gpu ? nullptr : full_input->data();
        float *output_data = is_gpu ? nullptr : full_output->mutable_data();
        if (!is_gpu && (!input_data || !output_data))
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent CPU verifier prefill could not access row tensors");
            return false;
        }

        struct ScopedRowParams
        {
            Params &params;
            TensorBase *input;
            TensorBase *output;
            TensorBase *routing_indices;
            TensorBase *routing_weights;
            IMoERuntimeTable *moe_runtime_table;
            bool force_grouped_verifier_prefill_for_decode;
            bool force_decode_equivalent_verifier_prefill;
            bool require_device_routing_tensor_decode;
            int seq_len;

            ~ScopedRowParams()
            {
                params.input = input;
                params.output = output;
                params.routing_indices = routing_indices;
                params.routing_weights = routing_weights;
                params.moe_runtime_table = moe_runtime_table;
                params.force_grouped_verifier_prefill_for_decode = force_grouped_verifier_prefill_for_decode;
                params.force_decode_equivalent_verifier_prefill = force_decode_equivalent_verifier_prefill;
                params.require_device_routing_tensor_decode = require_device_routing_tensor_decode;
                params.seq_len = seq_len;
            }
        } restore{
            params_,
            params_.input,
            params_.output,
            params_.routing_indices,
            params_.routing_weights,
            params_.moe_runtime_table,
            params_.force_grouped_verifier_prefill_for_decode,
            params_.force_decode_equivalent_verifier_prefill,
            params_.require_device_routing_tensor_decode,
            params_.seq_len};

        for (int row = 0; row < seq_len; ++row)
        {
            const int row_index = row;
            if (is_gpu)
            {
                if (!kernel->copyTokenRowFromTensor(
                        full_input, scratch_batch_.get(), row_index, d_model))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Failed to copy verifier input row "
                              << row << " for layer " << params_.layer_idx);
                    return false;
                }
                /*
                 * Routing is produced by MoERoutingStage as device-owned
                 * all-position tensors.  Gather the row on device and feed it
                 * to routing-aware kernels; do not read params_.routing_* on
                 * host, because those mirrors can be intentionally stale after
                 * D2D route publication.
                 *
                 * Keep using the immutable full_routing_* tensors here. The
                 * scoped one-token replay below temporarily points params_ at
                 * verifier_routing_*_row_; using params_ as the source on the
                 * next loop iteration would copy row 1 from a one-row scratch
                 * tensor and corrupt the verifier route stream.
                 */
                if (!kernel->copyTokenRowFromTensor(
                        full_routing_indices, verifier_routing_indices_row_.get(),
                        row_index, top_k) ||
                    !kernel->copyTokenRowFromTensor(
                        full_routing_weights, verifier_routing_weights_row_.get(),
                        row_index, top_k))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Failed to copy verifier routing row "
                              << row << " for layer " << params_.layer_idx);
                    return false;
                }
                if (debugEnv().runtime_debug.moe_grouped_verifier_snapshot_diagnostic)
                {
                    if (!verifier_routing_indices_row_->ensureOnHost(gpuStream()) ||
                        !verifier_routing_weights_row_->ensureOnHost(gpuStream()))
                    {
                        LOG_ERROR("[MoEExpertComputeStage] Failed to synchronize diagnostic verifier routing row "
                                  << row << " for layer " << params_.layer_idx);
                        return false;
                    }
                    const float *row_indices = verifier_routing_indices_row_->data();
                    const float *row_weights = verifier_routing_weights_row_->data();
                    for (int slot = 0; slot < top_k; ++slot)
                    {
                        const int expert_id = row_indices ? static_cast<int>(row_indices[slot]) : -1;
                        const float weight = row_weights ? row_weights[slot] : std::numeric_limits<float>::quiet_NaN();
                        if (expert_id < 0 || expert_id >= num_experts || !std::isfinite(weight))
                        {
                            LOG_ERROR("[MoEExpertComputeStage] Invalid diagnostic verifier routing row "
                                      << row << " slot " << slot
                                      << " expert=" << expert_id
                                      << " weight=" << weight
                                      << " layer=" << params_.layer_idx
                                      << " num_experts=" << num_experts);
                            return false;
                        }
                    }
                }
                params_.input = scratch_batch_.get();
                params_.output = verifier_output_row_.get();
            }
            else
            {
                std::copy_n(input_data + static_cast<size_t>(row) * d_model,
                            d_model,
                            row_input.mutable_data());
                params_.input = &row_input;
                params_.output = &row_output;
            }

            if (is_gpu)
            {
                /*
                 * Decode-equivalent publication must use the same one-token
                 * stage contract as serial decode.  DeviceMoELayerRuntime
                 * currently stores only one top-k row, so after the separate
                 * routing stage finishes it contains the last verifier row.
                 * Bind explicit row routing tensors here and disable the
                 * runtime table for this scoped replay; a future Phase 9.8
                 * runtime-row bank can replace this with a fully device-owned
                 * economical path once it is proven against this contract.
                 */
                params_.routing_indices = verifier_routing_indices_row_.get();
                params_.routing_weights = verifier_routing_weights_row_.get();
                params_.moe_runtime_table = nullptr;
                params_.seq_len = 1;
                params_.force_decode_equivalent_verifier_prefill = false;
                params_.force_grouped_verifier_prefill_for_decode = false;
                params_.require_device_routing_tensor_decode = true;
            }
            else
            {
                std::copy_n(routing_idx_data + static_cast<size_t>(row) * top_k,
                            top_k,
                            row_routing_indices.mutable_data());
                std::copy_n(routing_wt_data + static_cast<size_t>(row) * top_k,
                            top_k,
                            row_routing_weights.mutable_data());

                params_.routing_indices = &row_routing_indices;
                params_.routing_weights = &row_routing_weights;
                params_.moe_runtime_table = nullptr;
                params_.seq_len = 1;
                params_.force_decode_equivalent_verifier_prefill = false;
                params_.force_grouped_verifier_prefill_for_decode = false;
                params_.require_device_routing_tensor_decode = false;
            }
            const bool row_ok = executeSingleToken(ctx);
            if (!row_ok)
            {
                LOG_ERROR("[MoEExpertComputeStage] Verifier decode-equivalent row "
                          << row << " failed for layer " << params_.layer_idx);
                return false;
            }

            if (is_gpu)
            {
                if (!kernel->writeTokenRowToTensor(
                        full_output, verifier_output_row_.get(), row, d_model))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Failed to write verifier output row "
                              << row << " for layer " << params_.layer_idx);
                    return false;
                }
            }
            else
            {
                std::copy_n(row_output.data(),
                            d_model,
                            output_data + static_cast<size_t>(row) * d_model);
            }
        }

        if (is_gpu)
            markGpuTensorWritten(full_output, params_.device_id, gpuStream());
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
        runtime_grouped_decode_warmed_ = false;
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
        runtime_grouped_decode_warmed_ = false;
        return true;
    }

    bool MoEExpertComputeStage::ensureCombinedSharedVerifierResources(
        IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || !params_.combine_shared_expert_in_verifier ||
            !params_.prepared_store ||
            !params_.prepared_shared_ref_gate ||
            !params_.prepared_shared_ref_up ||
            !params_.prepared_shared_ref_down)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined shared verifier path is missing required prepared state"
                      << " layer=" << params_.layer_idx
                      << " kernel=" << (kernel ? "yes" : "no")
                      << " store=" << (params_.prepared_store ? "yes" : "no")
                      << " gate_ref=" << (params_.prepared_shared_ref_gate ? "yes" : "no")
                      << " up_ref=" << (params_.prepared_shared_ref_up ? "yes" : "no")
                      << " down_ref=" << (params_.prepared_shared_ref_down ? "yes" : "no"));
            return false;
        }

        if (combined_shared_gate_gemm_ &&
            combined_shared_up_gemm_ &&
            combined_shared_down_gemm_ &&
            combined_shared_desc_table_d_model_ == d_model &&
            combined_shared_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        ITensorGemm *shared_gate =
            params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate);
        ITensorGemm *shared_up =
            params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up);
        ITensorGemm *shared_down =
            params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down);
        if (!shared_gate || !shared_up || !shared_down)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined shared verifier path requires prepared shared expert GEMM engines"
                      << " layer=" << params_.layer_idx
                      << " gate=" << (shared_gate ? "yes" : "no")
                      << " up=" << (shared_up ? "yes" : "no")
                      << " down=" << (shared_down ? "yes" : "no"));
            return false;
        }

        shared_gate->setGPUStream(gpuStream());
        shared_up->setGPUStream(gpuStream());
        shared_down->setGPUStream(gpuStream());
        auto bind_if_needed = [&](ITensorGemm *gemm)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                consumer->bindWorkspace(bound_workspace_);
        };
        bind_if_needed(shared_gate);
        bind_if_needed(shared_up);
        bind_if_needed(shared_down);

        DeviceNativeVNNIMatrixDesc gate_desc;
        DeviceNativeVNNIMatrixDesc up_desc;
        DeviceNativeVNNIMatrixDesc down_desc;
        if (!shared_gate->exportNativeVNNIMatrixDesc(gate_desc) ||
            !shared_up->exportNativeVNNIMatrixDesc(up_desc) ||
            !shared_down->exportNativeVNNIMatrixDesc(down_desc) ||
            gate_desc.n != intermediate || gate_desc.k != d_model ||
            up_desc.n != intermediate || up_desc.k != d_model ||
            down_desc.n != d_model || down_desc.k != intermediate)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined safe verifier requires shared expert "
                      "GEMM engines with decode-equivalent native descriptors"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        combined_shared_gate_gemm_ = shared_gate;
        combined_shared_up_gemm_ = shared_up;
        combined_shared_down_gemm_ = shared_down;
        combined_shared_desc_table_d_model_ = d_model;
        combined_shared_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool MoEExpertComputeStage::initializeMoERuntimeTableForGroupedDecode()
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 ||
            params_.num_experts <= 0 || params_.top_k <= 0 ||
            !supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id))
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
        const bool forced_decode_replay =
            params_.force_grouped_verifier_prefill_for_decode && params_.seq_len == 1;
        return supportsGroupedPrefillExecutionBackend(params_.device_id) &&
               (params_.seq_len > 1 || forced_decode_replay) &&
               hasFullLocalExpertOwnership() &&
               expertMaskAllEnabled() &&
               params_.replica_set.num_replicated == 0;
    }

    bool MoEExpertComputeStage::canUseSafeCombinedSharedVerifierComposite() const
    {
        return params_.combine_shared_expert_in_verifier &&
               (params_.device_id.is_cuda() || params_.device_id.is_rocm()) &&
               params_.seq_len > 1 &&
               params_.seq_len <= 4 &&
               params_.d_model > 0 &&
               params_.expert_intermediate > 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               params_.routing_indices &&
               params_.routing_weights &&
               params_.shared_gate_inp &&
               params_.prepared_store &&
               params_.prepared_shared_ref_gate &&
               params_.prepared_shared_ref_up &&
               params_.prepared_shared_ref_down &&
               hasFullLocalExpertOwnership() &&
               expertMaskAllEnabled() &&
               params_.replica_set.num_replicated == 0;
    }

    TensorBase *MoEExpertComputeStage::effectiveSafeCompositeSharedGateInput() const
    {
        if (!params_.shared_gate_inp)
            return nullptr;

        if (params_.shared_gate_inp->native_type() == TensorType::FP32)
            return params_.shared_gate_inp;

        const bool needs_refresh =
            !combined_shared_gate_inp_fp32_ ||
            combined_shared_gate_inp_source_ != params_.shared_gate_inp ||
            combined_shared_gate_inp_fp32_->shape() != params_.shared_gate_inp->shape();
        if (needs_refresh)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[MoEExpertComputeStage] Combined verifier shared-gate "
                          "input was not normalized before graph capture"
                          << " layer=" << params_.layer_idx);
                return nullptr;
            }
            combined_shared_gate_inp_fp32_ =
                std::make_shared<FP32Tensor>(params_.shared_gate_inp->shape());
            combined_shared_gate_inp_source_ = params_.shared_gate_inp;

            /*
             * The backend grouping kernels read the shared-gate vector as `float*`.
             * The standalone SharedExpertGateStage already normalizes non-FP32
             * tensors this way; the combined verifier path must do the same or the
             * fast verifier lane can restore/publish states that do not match the
             * row-by-row decode contract.
             */
            params_.shared_gate_inp->to_fp32(combined_shared_gate_inp_fp32_->mutable_data());
        }

        return combined_shared_gate_inp_fp32_.get();
    }

    bool MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite(IMoEKernel *kernel) const
    {
        if (!kernel || grouped_gateup_desc_table_id_ < 0 ||
            grouped_down_desc_table_id_ < 0 ||
            !combined_shared_gate_gemm_ ||
            !combined_shared_up_gemm_ ||
            !combined_shared_down_gemm_)
        {
            return false;
        }

        TensorBase *shared_gate_inp = effectiveSafeCompositeSharedGateInput();
        if (!shared_gate_inp)
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                      "failed to prepare FP32 shared gate input");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int intermediate = params_.expert_intermediate;

        auto ensure_scratch = [&](std::shared_ptr<FP32Tensor> &slot,
                                  size_t rows,
                                  size_t cols,
                                  const char *name) -> bool
        {
            const std::vector<size_t> expected{rows, cols};
            if (slot && slot->shape() == expected)
                return true;
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                          << name << " scratch was not warmed before graph capture"
                          << " layer=" << params_.layer_idx
                          << " rows=" << rows
                          << " cols=" << cols);
                return false;
            }
            slot = makeScratchFP32(rows, cols, params_.device_id);
            return slot != nullptr;
        };

        if (!ensure_scratch(combined_routed_output_,
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(d_model),
                            "routed_output") ||
            !ensure_scratch(combined_shared_output_,
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(d_model),
                            "shared_output") ||
            !ensure_scratch(combined_shared_gate_scratch_,
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(intermediate),
                            "shared_gate") ||
            !ensure_scratch(combined_shared_up_scratch_,
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(intermediate),
                            "shared_up"))
        {
            return false;
        }

        if (!kernel->prepareExpertGroupsAsync(
                params_.routing_indices,
                params_.routing_weights,
                seq_len,
                params_.num_experts,
                params_.top_k))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] grouping failed");
            return false;
        }

        /*
         * This replaces the rejected "shared expert as one more routed expert"
         * table.  Keep routed and shared math as separate, already-proven branch
         * computations, but let one stage own the lifetime/order so the graph can
         * later overlap them without reintroducing shared backend scratch races.
         */
        if (!kernel->executeGroupedPrefillPipeline(
            params_.input,
            combined_routed_output_.get(),
            grouped_gateup_desc_table_id_,
            grouped_down_desc_table_id_,
            seq_len,
            d_model,
            intermediate,
            params_.num_experts,
            params_.top_k))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] routed grouped verifier failed");
            return false;
        }
        markGpuTensorWritten(combined_routed_output_.get(), params_.device_id, gpuStream());

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {combined_shared_gate_gemm_, combined_shared_gate_scratch_.get(), intermediate, nullptr, "combined_shared_gate"},
            {combined_shared_up_gemm_, combined_shared_up_scratch_.get(), intermediate, nullptr, "combined_shared_up"}};
        {
            auto verifier_scopes = beginVerifierDecodeEquivalentScopes(
                {combined_shared_gate_gemm_, combined_shared_up_gemm_, combined_shared_down_gemm_});
            if (!combined_shared_gate_gemm_->multiply_fused_verifier_rows_decode_equivalent(
                    params_.input,
                    projections,
                    seq_len,
                    d_model,
                    nullptr,
                    getWorkspace()))
            {
                LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] shared gate/up verifier failed");
                return false;
            }
            for (const auto &projection : projections)
                markGpuTensorWritten(projection.output, params_.device_id, gpuStream());

            if (!combined_shared_down_gemm_->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    combined_shared_gate_scratch_.get(),
                    combined_shared_up_scratch_.get(),
                    combined_shared_output_.get(),
                    seq_len,
                    d_model,
                    intermediate,
                    1.0f,
                    0.0f,
                    getWorkspace()))
            {
                LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] shared SwiGLU/down verifier failed");
                return false;
            }
        }
        markGpuTensorWritten(combined_shared_output_.get(), params_.device_id, gpuStream());

        kernel->sharedExpertGateAddFromTensors(
            params_.input,
            shared_gate_inp,
            combined_shared_output_.get(),
            combined_routed_output_.get(),
            params_.output,
            seq_len,
            d_model);
        markGpuTensorWritten(combined_shared_output_.get(), params_.device_id, gpuStream());
        markGpuTensorWritten(params_.output, params_.device_id, gpuStream());

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_combined_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(seq_len),
            "verifier",
            params_.device_id.toString(),
            {{"stage", "routed_plus_shared"},
             {"route", "safe_composite"},
             {"seq_len", std::to_string(seq_len)},
             {"routed_top_k", std::to_string(params_.top_k)},
             {"routed_experts", std::to_string(params_.num_experts)},
             {"layer", std::to_string(params_.layer_idx)}});
        return true;
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
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        return supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
               params_.seq_len == 1 &&
               params_.d_model > 0 &&
               params_.expert_intermediate > 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               params_.top_k <= 16 &&
               params_.top_k <= params_.num_experts &&
               params_.input &&
               params_.output &&
               moe_kernel_ &&
               runtime_grouped_decode_warmed_ &&
               params_.moe_runtime_table &&
               moe_runtime_layer_ &&
               moe_runtime_table_initialized_ &&
               runtimeTableHasActiveGroupedDecodeBank() &&
               params_.replica_set.num_replicated == 0 &&
               hasFullLocalExpertOwnership() &&
               expertMaskAllEnabled();
#endif
    }

    bool MoEExpertComputeStage::supportsFixedTopologyPrefillGraphCapturePreflight() const
    {
        // Cold preflight validates the fixed-topology grouped prefill contract
        // without requiring lazy warmup resources such as the MoE kernel or its
        // grouping scratch. Those are checked by isGraphCapturable() before the
        // actual capture pass begins.
        const bool forced_decode_replay =
            params_.force_grouped_verifier_prefill_for_decode && params_.seq_len == 1;
        if (!supportsGroupedPrefillGraphCaptureBackend(params_.device_id) ||
            (params_.seq_len <= 1 && !forced_decode_replay))
            return false;

        // Require full local expert ownership, no masks, no replicas
        if (!hasFullLocalExpertOwnership() || !expertMaskAllEnabled())
            return false;
        if (params_.replica_set.num_replicated != 0)
            return false;

        if (params_.d_model <= 0 ||
            params_.expert_intermediate <= 0 ||
            params_.num_experts <= 0 ||
            params_.top_k <= 0 ||
            params_.top_k > params_.num_experts ||
            !params_.input ||
            !params_.output ||
            !params_.routing_indices ||
            !params_.routing_weights)
            return false;

        // Must have all prepared GEMM engines ready
        return hasAllPreparedExpertGemmEngines();
    }

    bool MoEExpertComputeStage::isFixedTopologyPrefillGraphCapturable() const
    {
        if (!supportsFixedTopologyPrefillGraphCapturePreflight())
            return false;

        // MoE kernel must exist and have pre-allocated grouping + scratch
        if (!moe_kernel_)
            return false;

        return true;
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
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        if (params_.force_grouped_verifier_prefill_for_decode)
            return isFixedTopologyPrefillGraphCapturable();

        // Device-routed grouped decode path: after warmup, all descriptor tables
        // are built and execution is pure kernel launches reading routing info
        // from the device-resident MoE runtime table.
        if (isDeviceRoutedDecodeGraphCapturable())
            return true;

        // Fixed-topology grouped prefill path
        return isFixedTopologyPrefillGraphCapturable();
#endif
    }

    bool MoEExpertComputeStage::supportsWarmupDependentGraphCapture() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        const bool decode_supported =
            !params_.force_grouped_verifier_prefill_for_decode &&
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.seq_len == 1 &&
            params_.d_model > 0 &&
            params_.expert_intermediate > 0 &&
            params_.num_experts > 0 &&
            params_.top_k > 0 &&
            params_.top_k <= 16 &&
            params_.top_k <= params_.num_experts &&
            params_.input &&
            params_.output &&
            params_.moe_runtime_table &&
            params_.layer_idx >= 0 &&
            params_.replica_set.num_replicated == 0 &&
            hasFullLocalExpertOwnership() &&
            expertMaskAllEnabled();

        return decode_supported || supportsFixedTopologyPrefillGraphCapturePreflight();
#endif
    }

    bool MoEExpertComputeStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsFixedTopologyPrefillGraphCapturePreflight();
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
        {
            /*
             * The combined shared-verifier path writes the final routed+shared
             * row into this stage's output buffer.  Give snapshots the semantic
             * name so parity CSVs do not report a combined row as routed-only
             * expert output.
             */
            info.addOutput(params_.combine_shared_expert_in_verifier
                               ? "combined_output"
                               : "output",
                           params_.output,
                           params_.seq_len,
                           params_.d_model);
        }

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
        WorkspaceRequirements combined;
        const int workspace_experts = params_.num_experts;
        const int workspace_top_k = params_.top_k;
        if (params_.device_id.is_cuda())
        {
            combined.merge(MoEWorkspaceBuffers::expertExecution(
                params_.seq_len,
                params_.d_model,
                params_.expert_intermediate,
                workspace_experts,
                workspace_top_k));
        }
        else if (params_.device_id.is_rocm())
        {
            combined.merge(MoEWorkspaceBuffers::rocmMoE(
                params_.seq_len,
                params_.d_model,
                params_.expert_intermediate,
                workspace_experts,
                workspace_top_k));
        }

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
            {
                combined.merge(consumer->getWorkspaceRequirements(m, n, k));
                addCudaConcurrentDecodeGemvSideStreamWorkspace(
                    combined,
                    params_.device_id,
                    m,
                    static_cast<size_t>(std::max(0, workspace_top_k)) * 2u);
                break;
            }
        }

        if (params_.combine_shared_expert_in_verifier &&
            params_.prepared_store &&
            params_.prepared_shared_ref_gate &&
            params_.prepared_shared_ref_up &&
            params_.prepared_shared_ref_down)
        {
            ITensorGemm *shared_gate =
                params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate);
            ITensorGemm *shared_up =
                params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up);
            ITensorGemm *shared_down =
                params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down);
            const int rows = std::max(1, m > 0 ? m : params_.seq_len);
            const int d_model = params_.d_model > 0 ? params_.d_model : n;
            const int intermediate =
                params_.expert_intermediate > 0 ? params_.expert_intermediate : k;

            if (auto *c = dynamic_cast<IWorkspaceConsumer *>(shared_gate))
                combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
            if (auto *c = dynamic_cast<IWorkspaceConsumer *>(shared_up))
                combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
            if (auto *c = dynamic_cast<IWorkspaceConsumer *>(shared_down))
                combined.merge(c->getWorkspaceRequirements(rows, d_model, intermediate));
            addCudaConcurrentDecodeGemvSideStreamWorkspace(
                combined,
                params_.device_id,
                rows,
                /*projection_count=*/2u);
        }
        return combined;
    }

    void MoEExpertComputeStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        if (workspace != bound_workspace_)
            runtime_grouped_decode_warmed_ = false;

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

        auto bindShared = [workspace](ITensorGemm *gemm)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                consumer->bindWorkspace(workspace);
        };
        if (params_.combine_shared_expert_in_verifier &&
            params_.prepared_store &&
            params_.prepared_shared_ref_gate &&
            params_.prepared_shared_ref_up &&
            params_.prepared_shared_ref_down)
        {
            bindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate));
            bindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up));
            bindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down));
        }

        if (moe_kernel_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(moe_kernel_))
                consumer->bindWorkspace(workspace);
        }

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

        auto unbindShared = [](ITensorGemm *gemm)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                consumer->unbindWorkspace();
        };
        if (params_.combine_shared_expert_in_verifier &&
            params_.prepared_store &&
            params_.prepared_shared_ref_gate &&
            params_.prepared_shared_ref_up &&
            params_.prepared_shared_ref_down)
        {
            unbindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate));
            unbindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up));
            unbindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down));
        }

        if (moe_kernel_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(moe_kernel_))
                consumer->unbindWorkspace();
        }

        bound_workspace_ = nullptr;
        runtime_grouped_decode_warmed_ = false;
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

    bool SharedExpertFFNStage::shouldUseGroupedVerifierPrefillRoute() const
    {
        return params_.device_id.is_gpu() &&
               !shouldUseDecodeEquivalentVerifierPrefill() &&
               params_.force_grouped_verifier_prefill_for_decode &&
               params_.seq_len >= 1 &&
               params_.seq_len <= 4 &&
               supportsGroupedPrefillExecutionBackend(params_.device_id);
    }

    bool SharedExpertFFNStage::usesGroupedVerifierPrefillRouteForTesting() const
    {
        return shouldUseGroupedVerifierPrefillRoute();
    }

    bool SharedExpertFFNStage::shouldUseDecodeEquivalentVerifierPrefill() const
    {
        return (params_.device_id.is_cpu() ||
                params_.device_id.is_cuda() ||
                params_.device_id.is_rocm()) &&
               params_.force_decode_equivalent_verifier_prefill &&
               params_.seq_len > 1 &&
               params_.seq_len <= 4;
    }

    bool SharedExpertFFNStage::usesCPUDecodeEquivalentVerifierPrefillForTesting() const
    {
        return shouldUseDecodeEquivalentVerifierPrefill();
    }

    bool SharedExpertFFNStage::shouldUseGroupedDecodeRoute() const
    {
        return params_.device_id.is_gpu() &&
               params_.seq_len == 1 &&
               !params_.disable_grouped_decode_shortcut &&
               shouldUseSharedExpertGroupedDecode(params_.device_id);
    }

    bool SharedExpertFFNStage::usesGroupedDecodeForTesting() const
    {
        return shouldUseGroupedDecodeRoute();
    }

    bool SharedExpertFFNStage::executeDecodeEquivalentVerifierPrefill(
        IDeviceContext *ctx, IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || !params_.input || !params_.output ||
            !cached_gate_gemm_ || !cached_up_gemm_ || !cached_down_gemm_)
        {
            return false;
        }
        if (params_.device_id.is_gpu() && isGraphCaptureActive())
        {
            LOG_ERROR("[SharedExpertFFNStage] Decode-equivalent shared verifier prefill is not graph-capturable yet; "
                      "multi-row publication requires a device-resident row copy primitive before capture");
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "moe_decode_equivalent_verifier_prefill_runs",
            1.0,
            "verifier",
            params_.device_id.toString(),
            {{"stage", "shared_expert"},
             {"seq_len", std::to_string(params_.seq_len)}});

        if (!scratch_input_row_ ||
            scratch_input_row_->shape() != std::vector<size_t>{1u, static_cast<size_t>(d_model)})
        {
            scratch_input_row_ = makeScratchFP32(1, d_model, params_.device_id);
        }
        if (!scratch_output_row_ ||
            scratch_output_row_->shape() != std::vector<size_t>{1u, static_cast<size_t>(d_model)})
        {
            scratch_output_row_ = makeScratchFP32(1, d_model, params_.device_id);
        }

        const bool is_gpu = params_.device_id.is_gpu();
        TensorBase *full_input = params_.input;
        TensorBase *full_output = params_.output;
        auto verifier_scopes = beginVerifierDecodeEquivalentScopes(
            {cached_gate_gemm_, cached_up_gemm_, cached_down_gemm_});
        const float *input = is_gpu ? nullptr : full_input->data();
        float *output = is_gpu ? nullptr : full_output->mutable_data();
        if (!input || !output)
        {
            if (!is_gpu)
                return false;
        }
        kernel->zeroBuffer(full_output,
                           static_cast<size_t>(params_.seq_len) *
                               static_cast<size_t>(d_model) *
                               sizeof(float));

        for (int row = 0; row < params_.seq_len; ++row)
        {
            if (is_gpu)
            {
                if (!kernel->copyTokenRowFromTensor(
                        full_input, scratch_input_row_.get(), row, d_model))
                {
                    LOG_ERROR("[SharedExpertFFNStage] Failed to copy verifier input row "
                              << row << " for shared expert replay");
                    return false;
                }
            }
            else
            {
                std::copy_n(input + static_cast<size_t>(row) * d_model,
                            d_model,
                            scratch_input_row_->mutable_data());
            }

            /*
             * Re-enter the normal one-token shared-expert decode route instead
             * of open-coding a "close enough" grouped verifier path.  Serial
             * verifier replay is the source of truth for MTP publication.  CUDA
             * uses the scoped guard above to canonicalize M=1 verifier GEMVs to
             * the same small-M reduction contract as grouped verifier rows.
             */
            struct ScopedSingleRowParams
            {
                SharedExpertFFNStage::Params &params;
                TensorBase *input;
                TensorBase *output;
                bool force_grouped_verifier_prefill_for_decode;
                bool force_decode_equivalent_verifier_prefill;
                bool disable_grouped_decode_shortcut;
                int seq_len;

                ~ScopedSingleRowParams()
                {
                    params.input = input;
                    params.output = output;
                    params.force_grouped_verifier_prefill_for_decode = force_grouped_verifier_prefill_for_decode;
                    params.force_decode_equivalent_verifier_prefill = force_decode_equivalent_verifier_prefill;
                    params.disable_grouped_decode_shortcut = disable_grouped_decode_shortcut;
                    params.seq_len = seq_len;
                }
            } row_scope{
                params_,
                params_.input,
                params_.output,
                params_.force_grouped_verifier_prefill_for_decode,
                params_.force_decode_equivalent_verifier_prefill,
                params_.disable_grouped_decode_shortcut,
                params_.seq_len};

            params_.input = scratch_input_row_.get();
            params_.output = scratch_output_row_.get();
            params_.seq_len = 1;
            params_.force_decode_equivalent_verifier_prefill = false;
            params_.force_grouped_verifier_prefill_for_decode = false;
            params_.disable_grouped_decode_shortcut = true;

            if (!execute(ctx))
            {
                LOG_ERROR("[SharedExpertFFNStage] Decode-equivalent shared expert row "
                          << row << " failed through the normal one-token decode route");
                return false;
            }

            if (is_gpu)
            {
                if (!kernel->writeTokenRowToTensor(
                        full_output, scratch_output_row_.get(), row, d_model))
                {
                    LOG_ERROR("[SharedExpertFFNStage] Failed to write verifier output row "
                              << row << " for shared expert replay");
                    return false;
                }
            }
            else
            {
                std::copy_n(scratch_output_row_->data(),
                            d_model,
                            output + static_cast<size_t>(row) * d_model);
            }
        }

        if (is_gpu)
            markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
        return true;
    }

    bool SharedExpertFFNStage::tryGroupedVerifierPrefill(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        (void)kernel;
        if (!shouldUseGroupedVerifierPrefillRoute())
            return false;

        if (params_.device_id.is_gpu() && !gpuStream())
        {
            LOG_ERROR("[SharedExpertFFNStage] Grouped verifier shared expert requires an explicit GPU stream");
            return false;
        }

        /*
         * The shared expert is mathematically a dense FFN, not a routed MoE
         * dispatch.  Running it through IMoEKernel::executeGroupedPrefillPipeline
         * reused expert-group metadata and prefill-tuned kernels that are close
         * but not guaranteed to match serial decode for the verifier rows.  The
         * MTP publication contract is stricter: every verifier row may become
         * live state.  Use the dedicated M=2..4 decode-equivalent GEMV hooks so
         * gate/up input quantization and SwiGLU/down accumulation follow the
         * same contract as normal one-token decode while still batching the rows.
         */
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_.get(), intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_.get(), intermediate, nullptr, "shared_up"}};
        if (!cached_gate_gemm_->multiply_fused_verifier_rows_decode_equivalent(
                params_.input,
                projections,
                params_.seq_len,
                d_model,
                nullptr,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Decode-equivalent grouped shared gate/up failed"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }
        for (const auto &projection : projections)
            markGpuTensorWritten(projection.output, params_.device_id, gpuStream());

        if (!cached_down_gemm_->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                scratch_gate_.get(),
                scratch_up_.get(),
                params_.output,
            params_.seq_len,
            d_model,
            intermediate,
                1.0f,
                0.0f,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Decode-equivalent grouped shared SwiGLU/down failed"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_shared_grouped_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(params_.seq_len),
            "verifier",
            params_.device_id.toString(),
            {{"stage", "shared_expert"},
             {"route", "gemv_many"}});
        return true;
    }

    bool SharedExpertFFNStage::tryGroupedDecode(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        if (!params_.device_id.is_gpu() || params_.seq_len != 1 ||
            !shouldUseSharedExpertGroupedDecode(params_.device_id))
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
        const bool ok = kernel->groupedExpertDownDecodeFromTable(
            gate_tensors,
            up_tensors,
            &expert_id,
            &expert_weight,
            shared_grouped_down_desc_table_id_,
            1,
            params_.output,
            d_model,
            intermediate);
        if (ok)
            grouped_decode_warmed_ = true;
        return ok;
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
            grouped_decode_warmed_ = false;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (shouldUseDecodeEquivalentVerifierPrefill())
        {
            if (!executeDecodeEquivalentVerifierPrefill(ctx, kernel, d_model, intermediate))
            {
                LOG_ERROR("[SharedExpertFFNStage] Decode-equivalent verifier shared expert path failed");
                return false;
            }
            return true;
        }

        if (shouldUseGroupedVerifierPrefillRoute())
        {
            if (!tryGroupedVerifierPrefill(kernel, d_model, intermediate))
            {
                LOG_ERROR("[SharedExpertFFNStage] Verifier grouped shared expert path failed");
                return false;
            }
            markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
            return true;
        }

        const bool grouped_decode_required =
            shouldUseGroupedDecodeRoute();
        if (grouped_decode_required)
        {
            if (!tryGroupedDecode(kernel, d_model, intermediate))
            {
                LOG_ERROR("[SharedExpertFFNStage] Grouped shared expert decode path failed");
                return false;
            }
            markGpuTensorWritten(params_.output, params_.device_id, gpuStream());
            return true;
        }

        // Gate+Up projections via fused multi-projection (quantizes input once)
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_.get(), intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_.get(), intermediate, nullptr, "shared_up"}};
        if (!cached_gate_gemm_->multiply_fused_tensor(
                params_.input, projections,
                seq_len, d_model,
                nullptr, getWorkspace()))
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
                params_.device_id, gpuStream(), getWorkspace()))
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
        auto *kernel = bindStageStream(moe_kernel_);
        if (bound_workspace_)
        {
            // The shared expert can be the first user of the singleton MoE
            // kernel in focused tests and verifier-only paths. Bind explicitly
            // so workspace-owned pointer arrays never depend on sibling stages.
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
                consumer->bindWorkspace(bound_workspace_);
        }
        return kernel;
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
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        if (!supportsGroupedPrefillGraphCaptureBackend(params_.device_id))
            return false;

        /*
         * Forced verifier replay is a small-M grouped-prefill path even when
         * seq_len == 1.  Do not test grouped_decode_warmed_ here: that flag is
         * only written by tryGroupedDecode(), while verifier replay deliberately
         * executes tryGroupedVerifierPrefill() so M=1 correction rows share the
         * same graph-capturable contract as M=2..4 verifier rows.
         */
        if (shouldUseGroupedVerifierPrefillRoute())
            return scratch_seq_len_ >= params_.seq_len && moe_kernel_ != nullptr;

        /*
         * Normal grouped decode bakes device-side pointer arrays into CUDA graph
         * replay.  A cold stage can be warmed by Phase 1, but Phase 2 capture
         * must not start until that warmup has populated arrays for this stage's
         * current scratch buffers and workspace binding.
         */
        if (params_.seq_len == 1)
        {
            if (shouldUseGroupedDecodeRoute())
                return grouped_decode_warmed_ && scratch_seq_len_ >= 1 && moe_kernel_ != nullptr;
            return scratch_seq_len_ >= 1;
        }

        // Prefill capture is safe only after warmup has allocated scratch and
        // resolved the MoE kernel used by fused SwiGLU/down.
        return scratch_seq_len_ >= params_.seq_len && moe_kernel_ != nullptr;
#endif
    }

    bool SharedExpertFFNStage::supportsWarmupDependentGraphCapture() const
    {
        if (supportsPaddedPrefillGraphCapturePreflight())
            return true;

        return shouldUseGroupedDecodeRoute() &&
               params_.d_model > 0 &&
               params_.intermediate > 0 &&
               params_.input &&
               params_.gate_w &&
               params_.up_w &&
               params_.down_w &&
               params_.output;
    }

    bool SharedExpertFFNStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        const bool forced_decode_replay =
            params_.force_grouped_verifier_prefill_for_decode && params_.seq_len == 1;
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               (params_.seq_len > 1 || forced_decode_replay) &&
               params_.d_model > 0 &&
               params_.intermediate > 0 &&
               params_.input &&
               params_.gate_w &&
               params_.up_w &&
               params_.down_w &&
               params_.output;
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

        const int rows = std::max(1, m > 0 ? m : params_.seq_len);
        const int d_model =
            params_.d_model > 0 ? params_.d_model : (n > 0 ? n : 0);
        const int intermediate =
            params_.intermediate > 0 ? params_.intermediate : (k > 0 ? k : 0);

        /*
         * Gate/up and down have transposed logical shapes:
         *
         *   gate/up: [rows, d_model]       x [intermediate, d_model]^T
         *   down:    [rows, intermediate]  x [d_model, intermediate]^T
         *
         * The workspace names are intentionally shared by the GEMM engines and the
         * manager keeps the maximum buffer per name.  Passing the down shape to
         * gate/up under-sized QUANT_A/TEMP_A for M=2/3 verifier serial replay
         * (rows * intermediate < 1 * d_model), which corrupted ROCm M=1 oracle
         * rows.  Request each projection with its true shape so both grouped and
         * row-wise decode-equivalent paths are covered by the declared workspace.
         */
        WorkspaceRequirements combined;
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
            combined.merge(c->getWorkspaceRequirements(rows, d_model, intermediate));

        /*
         * The CUDA grouped verifier path can overlap the shared gate and up
         * projections on separate explicit streams.  The individual GEMM
         * engines only know about their serial stream-0 partial buffer, so the
         * stage must declare the side-stream partial arena once it knows the
         * fused projection fan-out.  This keeps graph capture allocation-free
         * and makes missing workspace fail during planning instead of halfway
         * through verifier replay.
         */
        addCudaConcurrentDecodeGemvSideStreamWorkspace(
            combined,
            params_.device_id,
            rows,
            /*projection_count=*/2);

        /*
         * Grouped shared-expert decode and verifier prefill use IMoEKernel
         * scratch in addition to the three projection GEMM workspaces.  Declare
         * those buffers here so a standalone shared-expert stage cannot rely on
         * a sibling routed-MoE stage to happen to reserve pointer arrays,
         * metadata, and INT8 activation scratch for graph capture.
         */
        const bool may_use_grouped_moe =
            params_.device_id.is_gpu() &&
            (shouldUseGroupedDecodeRoute() || shouldUseGroupedVerifierPrefillRoute());
        if (may_use_grouped_moe)
        {
            const int workspace_seq_len = std::max(1, params_.seq_len);
            if (params_.device_id.is_cuda())
            {
                combined.merge(MoEWorkspaceBuffers::cudaMoE(
                    workspace_seq_len,
                    params_.d_model,
                    params_.intermediate,
                    /*num_experts=*/1,
                    /*top_k=*/1));
            }
            else if (params_.device_id.is_rocm())
            {
                combined.merge(MoEWorkspaceBuffers::rocmMoE(
                    workspace_seq_len,
                    params_.d_model,
                    params_.intermediate,
                    /*num_experts=*/1,
                    /*top_k=*/1));
            }
        }
        return combined;
    }

    void SharedExpertFFNStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        const bool workspace_changed = bound_workspace_ != workspace;
        ensureGemmEnginesCached();

        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_gate_gemm_))
            c->bindWorkspace(workspace);
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_up_gemm_))
            c->bindWorkspace(workspace);
        if (auto *c = dynamic_cast<IWorkspaceConsumer *>(cached_down_gemm_))
            c->bindWorkspace(workspace);

        bound_workspace_ = workspace;
        if (workspace_changed)
            grouped_decode_warmed_ = false;
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
        grouped_decode_warmed_ = false;
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

    TensorBase *SharedExpertGateStage::effectiveGateInput() const
    {
        if (!params_.gate_inp)
            return nullptr;

        if (params_.gate_inp->native_type() == TensorType::FP32)
            return params_.gate_inp;

        const size_t count = params_.gate_inp->numel();
        if (!fp32_gate_inp_ || fp32_gate_source_ != params_.gate_inp ||
            fp32_gate_inp_->shape() != params_.gate_inp->shape())
        {
            fp32_gate_inp_ = std::make_shared<FP32Tensor>(params_.gate_inp->shape());
            params_.gate_inp->to_fp32(fp32_gate_inp_->mutable_data());
            fp32_gate_source_ = params_.gate_inp;
        }
        else if (fp32_gate_inp_->numel() != count)
        {
            fp32_gate_inp_ = std::make_shared<FP32Tensor>(params_.gate_inp->shape());
            params_.gate_inp->to_fp32(fp32_gate_inp_->mutable_data());
            fp32_gate_source_ = params_.gate_inp;
        }

        return fp32_gate_inp_.get();
    }

    bool SharedExpertGateStage::gateInputReadyForGraphCapture() const
    {
        if (!params_.gate_inp)
            return false;

        const TensorBase *gate_input =
            (params_.gate_inp->native_type() == TensorType::FP32)
                ? params_.gate_inp
                : ((fp32_gate_inp_ && fp32_gate_source_ == params_.gate_inp)
                       ? fp32_gate_inp_.get()
                       : nullptr);

        if (!gate_input)
            return false;

        /**
         * Graph replay may not perform a hidden H2D for model weights. Warmup is
         * responsible for materializing the FP32 gate vector and making the
         * selected tensor resident on the exact backend device used by this
         * stage; capture readiness is only true after that device-side handoff.
         */
        return !params_.device_id.is_gpu() || gate_input->is_on_device(params_.device_id);
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

        const bool fused_combine = params_.routed_residual || params_.combined_output;
        if (fused_combine && (!params_.routed_residual || !params_.combined_output))
        {
            LOG_ERROR("[SharedExpertGateStage] Fused combine requires both routed_residual and combined_output");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;

        // Delegate sigmoid gating to device-appropriate MoE kernel.
        // Tensor-aware API handles CPU/GPU dispatch internally.
        TensorBase *gate_inp = effectiveGateInput();
        if (!gate_inp)
        {
            LOG_ERROR("[SharedExpertGateStage] Failed to prepare gate input");
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();

        if (fused_combine)
        {
            kernel->sharedExpertGateAddFromTensors(
                params_.input, gate_inp, params_.shared_output,
                params_.routed_residual, params_.combined_output,
                seq_len, d_model);
            // Fused gate-add materializes both semantic outputs: the gated
            // shared contribution and the final routed+shared combined row.
            markGpuTensorWritten(params_.shared_output, params_.device_id, gpuStream());
            markGpuTensorWritten(params_.combined_output, params_.device_id, gpuStream());
            return true;
        }

        kernel->sharedExpertGateFromTensors(
            params_.input, gate_inp, params_.shared_output,
            seq_len, d_model);
        markGpuTensorWritten(params_.shared_output, params_.device_id, gpuStream());
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
        // Dot product + sigmoid + elementwise multiply, plus one add in fused-combine mode.
        const size_t per_token = static_cast<size_t>(2 * params_.d_model + params_.d_model);
        const size_t fused_add = (params_.routed_residual && params_.combined_output)
                                     ? static_cast<size_t>(params_.d_model)
                                     : 0u;
        return static_cast<size_t>(params_.seq_len) * (per_token + fused_add);
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
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        // Single kernel call (sigmoid gating) — pure kernel launch after warmup.
        // Capturable for both decode (seq_len==1) and prefill (seq_len>1) on supported GPU backends
        // because the stage is just a kernel launch with stable device pointers.
        // MoE kernel must already be cached (from warmup execution).
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               moe_kernel_ != nullptr &&
               gateInputReadyForGraphCapture();
#endif
    }

    bool SharedExpertGateStage::supportsWarmupDependentGraphCapture() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || (!defined(HAVE_ROCM) && !defined(HAVE_CUDA))
        return false;
#else
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               params_.seq_len > 0 &&
               params_.d_model > 0 &&
               params_.input &&
               params_.gate_inp &&
               params_.shared_output &&
               ((!params_.routed_residual && !params_.combined_output) ||
                (params_.routed_residual && params_.combined_output));
#endif
    }

    bool SharedExpertGateStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               params_.seq_len > 1 &&
               params_.d_model > 0 &&
               params_.input &&
               params_.gate_inp &&
               params_.shared_output &&
               ((!params_.routed_residual && !params_.combined_output) ||
                (params_.routed_residual && params_.combined_output));
    }

    StageBufferRequirements SharedExpertGateStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.shared_output)
        {
            if (params_.routed_residual && params_.combined_output)
            {
                reqs.addInput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
                reqs.addOutput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
            }
            else
                reqs.addOutput("shared_output", params_.shared_output->shape(), toBufferTensorType(params_.shared_output->native_type()));
        }
        if (params_.routed_residual)
            reqs.addInput("routed_residual", params_.routed_residual->shape(), toBufferTensorType(params_.routed_residual->native_type()));
        if (params_.combined_output)
            reqs.addOutput("combined_output", params_.combined_output->shape(), toBufferTensorType(params_.combined_output->native_type()));
        return reqs;
    }

    StageBufferContract SharedExpertGateStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        if (params_.routed_residual && params_.combined_output)
        {
            contract.addInOut(params_.output_buffer_id);
            contract.addInput(params_.residual_buffer_id);
            contract.addOutput(params_.combined_output_buffer_id);
        }
        else
        {
            contract.addInOut(params_.output_buffer_id);
        }

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
        {
            if (params_.routed_residual && params_.combined_output)
                info.addOutput("shared_output", params_.shared_output, params_.seq_len, params_.d_model);
            else
                info.addOutput("shared_output", params_.shared_output, params_.seq_len, params_.d_model);
        }
        if (params_.routed_residual)
            info.addInput("routed_residual", params_.routed_residual, params_.seq_len, params_.d_model);
        if (params_.combined_output)
            info.addOutput("combined_output", params_.combined_output, params_.seq_len, params_.d_model);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        return info;
    }

} // namespace llaminar2
