/**
 * @file MoEExpertComputeStageInternal.h
 * @brief Shared internal helpers and trace prototypes for MoEExpertComputeStage shards.
 */
#pragma once
/**
 * @file MoEExpertComputeStage.cpp
 * @brief Implements routed MoE, shared-expert, and shared-gate compute stages.
 *
 * The stages in this file preserve the model's routing arithmetic while
 * selecting the backend-native execution contract.  In particular, CPU
 * NativeQuantised expert projections consume the canonical Q8_1 representation of
 * the hidden rows.  Ordinary CPU graphs receive that representation from the
 * router stage, whereas graph-native heterogeneous ExpertOverlay participants
 * receive FP32 rows through a sparse transport boundary and must explicitly
 * publish the identical Q8_1 rows before either serial or grouped expert
 * execution begins. Weight-service calls retain the graph's declared ownership
 * mode so a departing shared consumer cannot invalidate sibling graph engines.
 * Grouped CPU scratch uses one typed eight-root layout for both its admission
 * BOM and final allocation. Each backing retains its own exact allocator and
 * PMA lease, including the kernel's canonical router publication.
 */

#include "MoEExpertComputeStage.h"
#include "../../moe/ExpertShardSourceSelection.h"
#include "MoEDeviceRebalanceStage.h"
#include "../ComputeStageUtils.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../collective/IGlobalTPContext.h"
#include "../../../execution/moe/CanonicalMoERouteRecord.h"
#include "../../../execution/moe/DecodeExpertHistogram.h"
#include "../../../execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "../../../execution/moe/ExpertWeightTransfer.h"
#include "../../../execution/moe/ExpertWeightPayloadProvider.h"
#include "../../../execution/moe/MoEExpertWeightService.h"
#include "../../../execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "../../../execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/cpu/moe/CPUMoEKernel.h"
#include "../../../kernels/cpu/moe/CPUCanonicalRouteReducer.h"
#include "../../../kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "../../../kernels/cpu/primitives/VectorPrimitives.h"
#include "../../../kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../backends/BackendManager.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/FNV1a.h"
#include <mpi.h>

#ifdef HAVE_CUDA
#include "../../../kernels/cuda/ops/CUDARowSelectKernels.h"
#endif

#ifdef HAVE_ROCM
#include "../../../kernels/rocm/ops/ROCmRowSelectKernels.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <typeindex>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>


namespace llaminar2
{
    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;
    using cpu::native_quantised::CPUNativeQuantisedGemmKernel;

    namespace mec_detail
    {
        uint64_t nextPrefillAssignmentTraceSequence()
;
        bool tracePrefillAssignmentRuntime(
            IBackend *backend,
            DeviceId device,
            void *stream,
            const char *tag,
            DeviceMoELayerRuntime *device_runtime,
            int layer_idx,
            int seq_len,
            int num_experts,
            int top_k)
;
        bool tracePrefillLLEPStatus(
            IBackend *backend,
            DeviceId device,
            void *stream,
            const char *tag,
            const DeviceMoERebalanceStatus *status,
            const DeviceMoERebalanceApplyStatus *apply_status)
;
    } // namespace mec_detail
    using mec_detail::nextPrefillAssignmentTraceSequence;
    using mec_detail::tracePrefillAssignmentRuntime;
    using mec_detail::tracePrefillLLEPStatus;

    namespace
    {
        /**
         * @brief Create host-owned scratch for CPU-only execution.
         *
         * GPU scratch must come from BufferArena or DeviceWorkspaceManager so
         * its address and lifetime are fixed before graph capture. Throwing
         * here turns any legacy per-stage GPU scratch construction into an
         * immediate architecture violation instead of a hidden device allocation.
         */
        std::shared_ptr<FP32Tensor> makeScratchFP32(
            size_t rows, size_t cols, DeviceId device)
        {
            if (device.is_gpu())
            {
                throw std::logic_error(
                    "GPU MoE scratch must be graph/workspace owned before execution");
            }
            return std::make_shared<FP32Tensor>(
                std::vector<size_t>{rows, cols});
        }

        const char *perfBool(bool value)
        {
            return value ? "true" : "false";
        }

        std::string prefillLLEPWorkspaceBufferName(
            const char *base_name,
            const std::string &workspace_name)
        {
            return MoEDeviceRebalanceStage::workspaceBufferName(
                base_name,
                workspace_name.empty()
                    ? std::string("moe_prefill_llep_transfer")
                    : workspace_name);
        }

        /**
         * @brief Build a layer-owned publication-buffer name for prefill LLEP.
         *
         * The large plan and payload arenas deliberately use two rolling lanes
         * because allocating an expert-payload arena per model layer would
         * consume several GiB. Their reuse is ordered by explicit per-layer
         * compute-ready and transfer-done events. Status records are different:
         * they are tiny publication objects consumed by the route-assignment
         * gate immediately after apply. Giving those records layer identity
         * makes it structurally impossible for a later layer using the same
         * bulk lane to replace an earlier layer's completion verdict.
         *
         * @param base_name      Stable workspace-buffer family name.
         * @param workspace_name Bulk transfer-lane identity.
         * @param layer_idx      Model layer that owns the publication record.
         * @return A deterministic workspace name unique to the layer.
         * @throws std::invalid_argument when @p layer_idx is negative.
         */
        std::string prefillLLEPLayerPublicationBufferName(
            const char *base_name,
            const std::string &workspace_name,
            int layer_idx)
        {
            if (layer_idx < 0)
            {
                throw std::invalid_argument(
                    "Prefill LLEP publication buffers require a concrete layer index");
            }
            std::string layer_workspace =
                workspace_name.empty()
                    ? std::string("moe_prefill_llep_transfer")
                    : workspace_name;
            layer_workspace += ":publication_layer=";
            layer_workspace += std::to_string(layer_idx);
            return MoEDeviceRebalanceStage::workspaceBufferName(
                base_name,
                layer_workspace);
        }

        uint32_t boundedU32(size_t value)
        {
            return static_cast<uint32_t>(
                std::min<size_t>(
                    value,
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
        }

        bool tracePrefillLLEPStatusEnabled()
        {
            return !DebugEnv::isFalseyEnv("LLAMINAR_MOE_LLEP_PREFILL_STATUS_TRACE");
        }

        /**
         * @brief Check whether runtime prefill assignment tracing is enabled.
         *
         * The assignment trace is an explicit diagnostic hook for prefix-cache
         * and LLEP parity work.  Normal inference must never pay for these host
         * copies or synchronizations; when the environment flag is enabled, the
         * trace becomes part of the requested diagnostic contract and failures
         * are reported as hard errors.
         */
        bool tracePrefillAssignmentEnabled()
        {
            return !DebugEnv::isFalseyEnv("LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE");
        }

        /**
         * @brief Check the optional layer filter for assignment tracing.
         *
         * Set LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE_LAYER=<layer> to keep the
         * diagnostic focused on the first divergent layer.  When the variable is
         * absent or empty, every layer is traced.
         */
        bool tracePrefillAssignmentLayerMatches(int layer_idx)
        {
            const char *value = DebugEnv::envValue("LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE_LAYER");
            if (!value || value[0] == '\0')
                return true;
            char *end = nullptr;
            const long requested = std::strtol(value, &end, 10);
            return end && *end == '\0' && requested == static_cast<long>(layer_idx);
        }

        /**
         * @brief Check the optional semantic-checkpoint filter for assignment tracing.
         *
         * The full assignment trace intentionally synchronizes the observed GPU
         * stream so copied diagnostics are complete before the host hashes them.
         * That synchronization can hide an ordering race when every checkpoint is
         * enabled at once. Set
         * `LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE_CHECKPOINT=<tag>` to observe one
         * exact checkpoint while leaving all preceding producer/consumer edges
         * asynchronous. Supported tags are the stage tags passed to
         * tracePrefillAssignmentRuntime(), plus `before_group` and
         * `after_pipeline`. An absent or empty value preserves the comprehensive
         * trace used by ordinary diagnostics.
         *
         * @param tag Semantic checkpoint about to be observed.
         * @return true when this checkpoint should execute its diagnostic work.
         */
        bool tracePrefillAssignmentCheckpointMatches(const char *tag)
        {
            const char *value =
                DebugEnv::envValue("LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE_CHECKPOINT");
            if (!value || value[0] == '\0')
                return true;
            return tag && std::strcmp(value, tag) == 0;
        }

        /**
         * @brief Add a byte range to a deterministic FNV-1a diagnostic hash.
         */
        uint64_t updateTraceHash(uint64_t hash, const void *data, size_t bytes)
        {
            constexpr uint64_t kFnvPrime = 1099511628211ULL;
            const auto *raw = static_cast<const uint8_t *>(data);
            for (size_t i = 0; i < bytes; ++i)
            {
                hash ^= static_cast<uint64_t>(raw[i]);
                hash *= kFnvPrime;
            }
            return hash;
        }

        /**
         * @brief Return a deterministic FNV-1a hash for a contiguous vector.
         */
        template <typename T>
        uint64_t hashTraceVector(const std::vector<T> &values)
        {
            constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
            if (values.empty())
                return kFnvOffset;
            return updateTraceHash(
                kFnvOffset,
                values.data(),
                values.size() * sizeof(T));
        }

        /**
         * @brief Hash a vector<bool> without relying on its packed proxy layout.
         */
        uint64_t hashTraceBoolVector(const std::vector<bool> &values)
        {
            constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
            uint64_t hash = kFnvOffset;
            for (bool value : values)
            {
                const uint8_t byte = value ? 1u : 0u;
                hash = updateTraceHash(hash, &byte, sizeof(byte));
            }
            return hash;
        }

        /**
         * @brief Allocate a monotonically increasing diagnostic sequence id.
         *
         * Prefix-restore parity runs build several independent graph instances
         * in one process.  A per-process sequence number makes it possible to
         * line up split, cached, and restored suffix prefill events without
         * relying on pointer values or thread interleaving.
         */


        /**
         * @brief Hash semantic placement fields without including raw pointers.
         *
         * The full/cached parity harness creates independent model instances, so
         * descriptor addresses naturally differ.  This hash intentionally keeps
         * only portable placement and matrix metadata: ownership, residency,
         * local-compute state, quantized matrix shape, and codebook identity.
         */
        uint64_t hashPlacementBankSemantics(
            const DeviceMoELayerRuntime &runtime,
            uint32_t expert_limit)
        {
            constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
            uint64_t hash = kFnvOffset;
            if (runtime.active_bank > 1u)
                return updateTraceHash(hash, &runtime.active_bank, sizeof(runtime.active_bank));

            const auto &bank = runtime.banks[runtime.active_bank];
            const uint32_t count =
                std::min<uint32_t>(
                    std::min<uint32_t>(expert_limit, runtime.expert_count),
                    kDeviceMoEMaxExperts);
            hash = updateTraceHash(hash, &runtime.active_bank, sizeof(runtime.active_bank));
            hash = updateTraceHash(hash, &runtime.active_epoch, sizeof(runtime.active_epoch));
            hash = updateTraceHash(hash, &runtime.participant_id, sizeof(runtime.participant_id));
            hash = updateTraceHash(hash, &runtime.participant_count, sizeof(runtime.participant_count));
            hash = updateTraceHash(hash, &count, sizeof(count));
            for (uint32_t expert = 0; expert < count; ++expert)
            {
                const auto &desc = bank.experts[expert];
                hash = updateTraceHash(hash, &desc.logical_expert_id, sizeof(desc.logical_expert_id));
                hash = updateTraceHash(hash, &desc.owner_participant, sizeof(desc.owner_participant));
                hash = updateTraceHash(hash, &desc.local_slot, sizeof(desc.local_slot));
                hash = updateTraceHash(hash, &desc.flags, sizeof(desc.flags));
                hash = updateTraceHash(hash, &bank.local_compute_mask[expert], sizeof(bank.local_compute_mask[expert]));
                hash = updateTraceHash(hash, &bank.replica_role[expert], sizeof(bank.replica_role[expert]));
                hash = updateTraceHash(hash, &bank.resident_participant_mask[expert], sizeof(bank.resident_participant_mask[expert]));

                const std::array<DeviceNativeQuantisedMatrixDesc, 3> matrices = {
                    desc.gate, desc.up, desc.down};
                for (const auto &matrix : matrices)
                {
                    const uint8_t payload_present = matrix.payload ? 1u : 0u;
                    const uint8_t scales_present = matrix.scales ? 1u : 0u;
                    const uint8_t mins_present = matrix.mins ? 1u : 0u;
                    const uint8_t emins_present = matrix.emins ? 1u : 0u;
                    hash = updateTraceHash(hash, &payload_present, sizeof(payload_present));
                    hash = updateTraceHash(hash, &scales_present, sizeof(scales_present));
                    hash = updateTraceHash(hash, &mins_present, sizeof(mins_present));
                    hash = updateTraceHash(hash, &emins_present, sizeof(emins_present));
                    hash = updateTraceHash(hash, &matrix.n, sizeof(matrix.n));
                    hash = updateTraceHash(hash, &matrix.k, sizeof(matrix.k));
                    hash = updateTraceHash(hash, &matrix.blocks_per_row, sizeof(matrix.blocks_per_row));
                    hash = updateTraceHash(hash, &matrix.codebook_id, sizeof(matrix.codebook_id));
                }
            }
            return hash;
        }

        /**
         * @brief Hash the fixed-topology prepared expert descriptor surface.
         *
         * The fixed prefill path bypasses DeviceMoERuntimeTable grouping and
         * exports descriptors directly from the stage-owned prepared GEMM
         * engines.  Prefix restore bugs can therefore leave the runtime table
         * and expert masks identical while the underlying prepared engines point
         * at different active slots.  This diagnostic hash records both the
         * portable matrix metadata and the device pointer identities so that the
         * parity harness can separate placement drift from residency drift.
         */
        uint64_t hashPreparedExpertDescriptorSurface(
            const std::vector<int> &expert_ids,
            const std::vector<ITensorGemm *> &gate_gemm,
            const std::vector<ITensorGemm *> &up_gemm,
            const std::vector<ITensorGemm *> &down_gemm)
        {
            constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
            uint64_t hash = kFnvOffset;
            auto hash_one = [&](ITensorGemm *gemm)
            {
                const uint8_t present = gemm ? 1u : 0u;
                hash = updateTraceHash(hash, &present, sizeof(present));
                if (!gemm)
                    return;

                DeviceNativeQuantisedMatrixDesc desc{};
                const bool exported = gemm->exportNativeQuantisedMatrixDesc(desc) && desc.valid();
                const uint8_t exported_byte = exported ? 1u : 0u;
                hash = updateTraceHash(hash, &exported_byte, sizeof(exported_byte));
                if (!exported)
                    return;

                const uintptr_t payload = reinterpret_cast<uintptr_t>(desc.payload);
                const uintptr_t scales = reinterpret_cast<uintptr_t>(desc.scales);
                const uintptr_t mins = reinterpret_cast<uintptr_t>(desc.mins);
                const uintptr_t emins = reinterpret_cast<uintptr_t>(desc.emins);
                hash = updateTraceHash(hash, &payload, sizeof(payload));
                hash = updateTraceHash(hash, &scales, sizeof(scales));
                hash = updateTraceHash(hash, &mins, sizeof(mins));
                hash = updateTraceHash(hash, &emins, sizeof(emins));
                hash = updateTraceHash(hash, &desc.n, sizeof(desc.n));
                hash = updateTraceHash(hash, &desc.k, sizeof(desc.k));
                hash = updateTraceHash(hash, &desc.blocks_per_row, sizeof(desc.blocks_per_row));
                hash = updateTraceHash(hash, &desc.codebook_id, sizeof(desc.codebook_id));
            };

            for (int expert_id : expert_ids)
            {
                hash = updateTraceHash(hash, &expert_id, sizeof(expert_id));
                const size_t idx = static_cast<size_t>(std::max(0, expert_id));
                hash_one(expert_id >= 0 && idx < gate_gemm.size() ? gate_gemm[idx] : nullptr);
                hash_one(expert_id >= 0 && idx < up_gemm.size() ? up_gemm[idx] : nullptr);
                hash_one(expert_id >= 0 && idx < down_gemm.size() ? down_gemm[idx] : nullptr);
            }
            return hash;
        }

        /**
         * @brief Copy a GPU runtime buffer to host for an enabled diagnostic.
         */
        bool copyTraceBuffer(
            IBackend *backend,
            DeviceId device,
            void *stream,
            const void *device_ptr,
            void *host_ptr,
            size_t bytes,
            const char *field_name)
        {
            if (bytes == 0)
                return true;
            if (!backend || !stream || !device_ptr || !host_ptr)
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace missing buffer"
                          << " device=" << device.to_string()
                          << " field=" << (field_name ? field_name : "<null>")
                          << " backend=" << static_cast<void *>(backend)
                          << " stream=" << stream
                          << " device_ptr=" << device_ptr
                          << " host_ptr=" << host_ptr);
                return false;
            }
            return backend->deviceToHostOnStream(
                host_ptr,
                device_ptr,
                bytes,
                device.toKernelDeviceIndex(),
                stream);
        }

        /**
         * @brief Trace the semantic runtime state around grouped prefill assignment.
         *
         * This diagnostic is deliberately stage-level instead of CUDA-only: it
         * observes the graph contract after each semantic step
         * (group -> plan -> assign -> regroup -> execute) and works for CUDA and
         * ROCm through the shared backend copy API.  It never runs unless
         * LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE is set; with the flag set, copy
         * or contract failures return false so parity runs fail clearly.
         */




        /**
         * @brief Execute the mandatory fused SwiGLU plus down projection.
         *
         * Every production backend and weight format must implement this grouped
         * operation. Returning false is a hard execution failure; changing the
         * arithmetic to a separate activation and GEMM would violate both the
         * decode-equivalence and economy contracts.
        */
        bool fusedSwigluDown(
            const IComputeStage &producer_stage,
            FP32Tensor *gate_tensor, FP32Tensor *up_tensor, TensorBase *output,
            ITensorGemm *down_gemm,
            int m, int n, int intermediate,
            DeviceWorkspaceManager *workspace)
        {
            const bool ok = down_gemm->multiply_tensor_with_fused_swiglu(
                    gate_tensor, up_tensor, output,
                    m, n, intermediate,
                    1.0f, 0.0f,
                    workspace);
            if (!ok)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage] Mandatory fused SwiGLU/down "
                    "implementation failed on "
                    << producer_stage.device().to_string());
                return false;
            }
            if (producer_stage.device().is_gpu())
                producer_stage.gpuExecution().publish(output);
            return true;
        }

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
         * Snapshot-enabled parity drains tensor outputs after captured replay.
         * The underlying CUDA/ROCm grouped `FromRouting` kernels are the
         * correct execution path whenever an M=1 decode lane binds a
         * device-owned routing tensor that has no reliable host mirror.
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
            {
                /*
                 * ROCm shared-expert decode must use the same grouped table
                 * descriptor family as M=2..4 verifier table-prefill.  Keeping
                 * the old opt-in dense decode path creates two valid-looking
                 * production contracts whose FP32 reductions differ by a few
                 * ulps; those ulps are enough to flip later MoE routing.  The
                 * grouped table route is therefore a hard production
                 * requirement, not a debug-advertised capability.
                 */
                return true;
            }
#endif
            return false;
#endif
        }

        bool supportsSharedExpertGroupedDecodeGraphCaptureBackend(DeviceId device)
        {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
            (void)device;
            return false;
#else
            return shouldUseSharedExpertGroupedDecode(device);
#endif
        }
    } // anonymous namespace
} // namespace llaminar2
