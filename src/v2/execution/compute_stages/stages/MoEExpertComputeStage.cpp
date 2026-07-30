/**
 * @file MoEExpertComputeStage.cpp
 * @brief Implementation of unified MoE FFN, shared expert, and shared expert gate stages
 */

#include "MoEExpertComputeStage.h"
#include "MoEDeviceRebalanceStage.h"
#include "../ComputeStageUtils.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../execution/moe/DecodeExpertHistogram.h"
#include "../../../execution/moe/ExpertWeightTransfer.h"
#include "../../../execution/moe/ExpertWeightPayloadProvider.h"
#include "../../../execution/moe/MoEExpertWeightService.h"
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
#include "../../../kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "../../../kernels/cpu/primitives/VectorPrimitives.h"
#include "../../../kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../backends/BackendManager.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/PerfStatsCollector.h"
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
#include <unordered_set>
#include <vector>

namespace llaminar2
{
    // Alias for fully-qualified KernelFactory access
    using KernelFactory = llaminar::v2::kernels::KernelFactory;
    using cpu::native_vnni::CPUNativeVNNIGemmKernel;

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
        uint64_t nextPrefillAssignmentTraceSequence()
        {
            static std::atomic<uint64_t> sequence{0};
            return sequence.fetch_add(1, std::memory_order_relaxed) + 1u;
        }

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

                const std::array<DeviceNativeVNNIMatrixDesc, 3> matrices = {
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

                DeviceNativeVNNIMatrixDesc desc{};
                const bool exported = gemm->exportNativeVNNIMatrixDesc(desc) && desc.valid();
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
        {
            if (!tracePrefillAssignmentEnabled() ||
                !tracePrefillAssignmentLayerMatches(layer_idx) ||
                !tracePrefillAssignmentCheckpointMatches(tag))
            {
                return true;
            }

            /*
             * Assignment diagnostics are host-readback diagnostics, not graph
             * nodes. Native CUDA/HIP capture must retain a fully device-owned
             * execution path, so the request-boundary diagnostic publication
             * owns any later host inspection. Returning here is not an
             * execution fallback: the production kernels and graph continue
             * unchanged, and PerfStats records that this optional observation
             * point was intentionally deferred.
             */
            if (isGraphCaptureActive())
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_llep_prefill_assignment_trace_deferred",
                    1.0,
                    "moe",
                    device.to_string(),
                    {{"tag", tag ? tag : "<null>"},
                     {"layer", std::to_string(layer_idx)},
                     {"reason", "graph_capture_device_owned"}});
                return true;
            }

            if (!device.is_gpu())
                return true;
            if (!backend || !stream || !device_runtime)
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace requires backend, stream, and runtime"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " backend=" << static_cast<void *>(backend)
                          << " stream=" << stream
                          << " runtime=" << static_cast<void *>(device_runtime));
                return false;
            }

            DeviceMoELayerRuntime runtime{};
            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    device_runtime,
                    &runtime,
                    sizeof(runtime),
                    "runtime"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy runtime"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            const uint64_t route_count =
                static_cast<uint64_t>(std::max(0, seq_len)) *
                static_cast<uint64_t>(std::max(0, top_k));
            if (route_count > static_cast<uint64_t>(runtime.prefill_route_capacity))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace route capacity mismatch"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " route_count=" << route_count
                          << " capacity=" << runtime.prefill_route_capacity);
                return false;
            }

            const uint32_t expert_count =
                std::min<uint32_t>(
                    static_cast<uint32_t>(std::max(0, num_experts)),
                    kDeviceMoEMaxExperts);
            std::vector<int32_t> route_experts(static_cast<size_t>(route_count));
            std::vector<float> route_weights(static_cast<size_t>(route_count));
            std::vector<int32_t> route_participants(static_cast<size_t>(route_count));
            std::vector<int32_t> expert_counts(expert_count);
            std::vector<least_loaded_ep::LeastLoadedExpertAssignmentSpan> spans;
            std::vector<least_loaded_ep::LeastLoadedExpertWeightTransfer> transfers;

            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.route_expert_ids,
                    route_experts.data(),
                    route_experts.size() * sizeof(int32_t),
                    "route_expert_ids") ||
                !copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.route_weights,
                    route_weights.data(),
                    route_weights.size() * sizeof(float),
                    "route_weights") ||
                !copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.route_participant_ids,
                    route_participants.data(),
                    route_participants.size() * sizeof(int32_t),
                    "route_participant_ids") ||
                !copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.expert_counts,
                    expert_counts.data(),
                    expert_counts.size() * sizeof(int32_t),
                    "expert_counts"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy route metadata"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            const uint64_t span_count = runtime.reserved_u64[2];
            const uint64_t span_capacity = runtime.reserved_u64[0];
            if (span_count > span_capacity)
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace span capacity mismatch"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " span_count=" << span_count
                          << " span_capacity=" << span_capacity);
                return false;
            }
            spans.resize(static_cast<size_t>(span_count));
            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.reserved_ptrs[1],
                    spans.data(),
                    spans.size() * sizeof(least_loaded_ep::LeastLoadedExpertAssignmentSpan),
                    "llep_assignment_spans"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy assignment spans"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            const uint64_t transfer_count = runtime.reserved_u64[3];
            const uint64_t transfer_capacity = runtime.reserved_u64[1];
            if (transfer_count > transfer_capacity)
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace transfer capacity mismatch"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx
                          << " transfer_count=" << transfer_count
                          << " transfer_capacity=" << transfer_capacity);
                return false;
            }
            transfers.resize(static_cast<size_t>(transfer_count));
            if (!copyTraceBuffer(
                    backend,
                    device,
                    stream,
                    runtime.reserved_ptrs[2],
                    transfers.data(),
                    transfers.size() * sizeof(least_loaded_ep::LeastLoadedExpertWeightTransfer),
                    "llep_weight_transfers"))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace failed to copy weight transfers"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            if (!backend->synchronizeStream(stream, device.toKernelDeviceIndex()))
            {
                LOG_ERROR("[MoEExpertComputeStage] prefill assignment trace stream synchronization failed"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string()
                          << " layer=" << layer_idx);
                return false;
            }

            std::array<uint64_t, kDeviceMoEMaxParticipants> participant_load{};
            uint64_t invalid_participant_rows = 0;
            for (int32_t participant : route_participants)
            {
                if (participant >= 0 &&
                    static_cast<size_t>(participant) < participant_load.size())
                {
                    ++participant_load[static_cast<size_t>(participant)];
                }
                else
                {
                    ++invalid_participant_rows;
                }
            }

            uint64_t expert_count_sum = 0;
            int32_t max_expert_count = 0;
            uint32_t nonzero_experts = 0;
            for (int32_t count : expert_counts)
            {
                if (count > 0)
                {
                    expert_count_sum += static_cast<uint64_t>(count);
                    max_expert_count = std::max(max_expert_count, count);
                    ++nonzero_experts;
                }
            }

            std::vector<uint32_t> placement_semantics;
            placement_semantics.reserve(expert_count * 4u);
            if (runtime.active_bank <= 1u)
            {
                const auto &bank = runtime.banks[runtime.active_bank];
                for (uint32_t expert = 0; expert < expert_count; ++expert)
                {
                    placement_semantics.push_back(static_cast<uint32_t>(
                        std::max(0, bank.experts[expert].owner_participant)));
                    placement_semantics.push_back(bank.resident_participant_mask[expert]);
                    placement_semantics.push_back(static_cast<uint32_t>(bank.local_compute_mask[expert]));
                    placement_semantics.push_back(static_cast<uint32_t>(bank.replica_role[expert]));
                }
            }

            LOG_INFO("[MoEExpertComputeStage] prefill assignment trace"
                     << " tag=" << (tag ? tag : "<null>")
                     << " device=" << device.to_string()
                     << " layer=" << layer_idx
                     << " seq_len=" << seq_len
                     << " top_k=" << top_k
                     << " route_count=" << route_count
                     << " active_bank=" << runtime.active_bank
                     << " active_epoch=" << runtime.active_epoch
                     << " participant_id=" << runtime.participant_id
                     << " participant_count=" << runtime.participant_count
                     << " expert_count=" << runtime.expert_count
                     << " route_expert_hash=" << hashTraceVector(route_experts)
                     << " route_weight_hash=" << hashTraceVector(route_weights)
                     << " route_participant_hash=" << hashTraceVector(route_participants)
                     << " expert_counts_hash=" << hashTraceVector(expert_counts)
                     << " spans_hash=" << hashTraceVector(spans)
                     << " transfers_hash=" << hashTraceVector(transfers)
                     << " placement_hash=" << hashPlacementBankSemantics(runtime, expert_count)
                     << " placement_owner_mask_hash=" << hashTraceVector(placement_semantics)
                     << " span_count=" << span_count
                     << " span_capacity=" << span_capacity
                     << " transfer_count=" << transfer_count
                     << " transfer_capacity=" << transfer_capacity
                     << " expert_count_sum=" << expert_count_sum
                     << " max_expert_count=" << max_expert_count
                     << " nonzero_experts=" << nonzero_experts
                     << " invalid_participants=" << invalid_participant_rows
                     << " p0=" << participant_load[0]
                     << " p1=" << participant_load[1]
                     << " p2=" << participant_load[2]
                     << " p3=" << participant_load[3]);

            /*
             * A transferred expert is useful only when its published runtime
             * descriptor names the destination participant's stable VRAM slot.
             * Correct payload bytes can otherwise mask a remote-pointer
             * publication bug until decode becomes PCIe-bound. Keep the exact
             * pointer provenance in the opt-in trace so one diagnostic pass can
             * distinguish transfer cost from post-transfer compute cost.
             */
            if (tag &&
                std::strcmp(tag, "after_llep_transfer_apply") == 0 &&
                runtime.active_bank <= 1u)
            {
                const auto &bank = runtime.banks[runtime.active_bank];
                for (const auto &transfer : transfers)
                {
                    if (transfer.expert >= expert_count)
                        continue;
                    const auto &descriptor = bank.experts[transfer.expert];
                    LOG_INFO("[MoEExpertComputeStage] prefill transfer descriptor trace"
                             << " device=" << device.to_string()
                             << " layer=" << layer_idx
                             << " expert=" << transfer.expert
                             << " source_participant=" << transfer.source_participant
                             << " destination_participant=" << transfer.destination_participant
                             << " runtime_participant=" << runtime.participant_id
                             << " owner_participant=" << descriptor.owner_participant
                             << " local_slot=" << descriptor.local_slot
                             << " flags=" << descriptor.flags
                             << " resident_mask="
                             << bank.resident_participant_mask[transfer.expert]
                             << " local_compute="
                             << static_cast<uint32_t>(
                                    bank.local_compute_mask[transfer.expert])
                             << " gate_payload="
                             << static_cast<const void *>(descriptor.gate.payload)
                             << " gate_scales=" << descriptor.gate.scales
                             << " up_payload="
                             << static_cast<const void *>(descriptor.up.payload)
                             << " up_scales=" << descriptor.up.scales
                             << " down_payload="
                             << static_cast<const void *>(descriptor.down.payload)
                             << " down_scales=" << descriptor.down.scales);
                }
            }
            return true;
        }

        bool tracePrefillLLEPStatus(
            IBackend *backend,
            DeviceId device,
            void *stream,
            const char *tag,
            const DeviceMoERebalanceStatus *status,
            const DeviceMoERebalanceApplyStatus *apply_status)
        {
            if (!tracePrefillLLEPStatusEnabled() ||
                !backend ||
                !device.is_gpu() ||
                !stream ||
                !status ||
                !apply_status)
            {
                return true;
            }

            /*
             * Host readback and stream synchronization are intentionally
             * forbidden while CUDA/HIP records a graph. The status buffers are
             * already persistent device-owned graph state and are exported by
             * the request-boundary PerfStats readback after replay. Attempting
             * to inspect them here poisons native stream capture before the
             * first LLEP event edge is recorded.
             */
            if (isGraphCaptureActive())
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_llep_prefill_status_trace_deferred",
                    1.0,
                    "moe",
                    device.to_string(),
                    {{"tag", tag ? tag : "<null>"},
                     {"reason", "graph_capture_device_owned"}});
                return true;
            }

            DeviceMoERebalanceStatus host_status{};
            DeviceMoERebalanceApplyStatus host_apply_status{};
            const int device_ordinal = device.toKernelDeviceIndex();
            const bool copied =
                backend->deviceToHostOnStream(
                    &host_status,
                    status,
                    sizeof(host_status),
                    device_ordinal,
                    stream) &&
                backend->deviceToHostOnStream(
                    &host_apply_status,
                    apply_status,
                    sizeof(host_apply_status),
                    device_ordinal,
                    stream) &&
                backend->synchronizeStream(stream, device_ordinal);
            if (!copied)
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to trace prefill LLEP status"
                          << " tag=" << (tag ? tag : "<null>")
                          << " device=" << device.to_string());
                return false;
            }

            LOG_INFO("[MoEExpertComputeStage] prefill LLEP status"
                     << " tag=" << (tag ? tag : "<null>")
                     << " device=" << device.to_string()
                     << " transfer_status=" << host_status.status_code
                     << " planned=" << host_status.planned_arrivals
                     << " transfer_count=" << host_status.llep_weight_transfer_count
                     << " span_count=" << host_status.llep_assignment_span_count
                     << " candidates=" << host_status.candidate_arrivals_considered
                     << " invalid_runtime=" << host_status.invalid_runtime_layers
                     << " standard_ep=" << host_status.llep_standard_ep_selected
                     << " skipped_balanced=" << host_status.llep_skipped_balanced
                     << " skipped_spread=" << host_status.llep_skipped_insufficient_spread_improvement
                     << " skipped_foreign=" << host_status.llep_skipped_insufficient_foreign_rows
                     << " plan_overflow=" << host_status.plan_overflow
                     << " payload_overflow=" << host_status.payload_bucket_overflow
                     << " payload_requested=" << host_status.payload_bucket_requested_slots
                     << " payload_slots=" << host_status.payload_bucket_slots
                     << " payload_bucket_index=" << host_status.payload_bucket_index
                     << " source_mask=" << host_status.payload_source_participant_mask
                     << " destination_mask=" << host_status.payload_destination_participant_mask
                     << " apply_status=" << host_apply_status.status_code
                     << " seen=" << host_apply_status.plan_entries_seen
                     << " applied=" << host_apply_status.applied_arrivals
                     << " required_local=" << host_apply_status.required_local_arrivals
                     << " ready_local=" << host_apply_status.ready_local_arrivals
                     << " copied=" << host_apply_status.copied_arrivals
                     << " invalid=" << host_apply_status.invalid_plan_entries
                     << " missing_src=" << host_apply_status.missing_source_descriptors
                     << " missing_dst=" << host_apply_status.missing_destination_slots
                     << " mismatch=" << host_apply_status.descriptor_mismatches
                     << " copy_incomplete=" << host_apply_status.copy_incomplete
                     << " changed_layers=" << host_apply_status.changed_layers);
            return true;
        }

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

    // =========================================================================
    // MoEExpertComputeStage — Unified Router + Expert FFN + Combine
    // =========================================================================

    MoEExpertComputeStage::MoEExpertComputeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        /*
         * Runtime placement publication indexes these tables by global expert
         * id even when a participant currently owns no experts. Full model
         * builders normally pre-size the vectors while preparing weights, but
         * placement is a stage-level contract and must not depend on that
         * incidental builder ordering. A fixed-size null table also lets a
         * newly built graph adopt engines from PreparedWeightStore without
         * allocating or reshaping metadata in the token hot path.
         */
        const size_t expert_count =
            static_cast<size_t>(std::max(0, params_.num_experts));
        params_.prepared_gate_gemm.resize(expert_count, nullptr);
        params_.prepared_up_gemm.resize(expert_count, nullptr);
        params_.prepared_down_gemm.resize(expert_count, nullptr);
        invalidateFixedTopologyMaskPublication();

        if (params_.moe_runtime_table && params_.layer_idx >= 0)
        {
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
            /*
             * Construction has no executor-owned stream, so it may inspect an
             * already published bank but must never publish a new one. A cold
             * decode bank is initialized by executeSingleToken() after the
             * executor binds the exact producer stream.
             */
            moe_runtime_table_initialized_ = runtimeTableHasActiveGroupedDecodeBank();
        }
    }

    bool MoEExpertComputeStage::usesGraphStableRuntimeDecodePlacement() const
    {
        return supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
               params_.seq_len == 1 &&
               !params_.force_grouped_verifier_prefill_for_decode &&
               params_.moe_runtime_table != nullptr &&
               params_.layer_idx >= 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0;
    }

    bool MoEExpertComputeStage::usesGraphStableFixedTopologyPrefillPlacement() const
    {
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               params_.seq_len > 1 &&
               !params_.force_grouped_verifier_prefill_for_decode &&
               canUseFixedTopologyGroupedPrefill();
    }

    bool MoEExpertComputeStage::usesGraphStableMoEPlacement() const
    {
        return usesGraphStableRuntimeDecodePlacement() ||
               usesGraphStableFixedTopologyPrefillPlacement();
    }

    bool MoEExpertComputeStage::validatePreparedWeights(std::string *error) const
    {
        auto fail = [error](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };

        if (params_.combine_shared_expert_in_verifier)
        {
            if (!params_.prepared_store ||
                !params_.shared_gate_w ||
                !params_.shared_up_w ||
                !params_.shared_down_w ||
                !params_.shared_gate_inp ||
                !params_.prepared_shared_ref_gate.has_value() ||
                !params_.prepared_shared_ref_up.has_value() ||
                !params_.prepared_shared_ref_down.has_value())
            {
                return fail(
                    "combined shared verifier requires gate/up/down source weights, "
                    "gate input, PreparedWeightStore, and all prepared refs");
            }
            if (!params_.prepared_store->contains(*params_.prepared_shared_ref_gate) ||
                !params_.prepared_store->contains(*params_.prepared_shared_ref_up) ||
                !params_.prepared_store->contains(*params_.prepared_shared_ref_down))
            {
                return fail(
                    "combined shared verifier PreparedWeightStore is missing a "
                    "gate/up/down ref");
            }
        }

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
        if (params_.replica_set.num_replicated > 0 && !params_.expert_mask.empty())
            params_.replica_set.buildPrefillMask(params_.my_socket_id, params_.expert_mask, params_.layer_idx);
        else
            params_.replica_set.prefill_mask.clear();
        grouped_gateup_desc_table_dirty_ = true;
        grouped_down_desc_table_dirty_ = true;
        moe_runtime_table_initialized_ = false;
        invalidateFixedTopologyMaskPublication();
        return refreshGraphStablePlacement(/*preserve_capture_ready=*/true);
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

    std::vector<int> MoEExpertComputeStage::transferExpertsGPUDirectFrom(
        MoEExpertComputeStage &source,
        const std::vector<int> &expert_ids,
        void *source_producer_stream)
    {
        std::vector<int> satisfied_expert_ids;
        if (!source_producer_stream)
        {
            LOG_DEBUG("[MoEExpertComputeStage] GPU-direct transfer requires an explicit source producer stream"
                      << " for layer " << params_.layer_idx);
            return satisfied_expert_ids;
        }

        auto src_ctx = source.buildWeightContext();
        auto dst_ctx = buildWeightContext();
        GpuDirectTransferCompletion completion;
        const bool ok = MoEExpertWeightService::transferExpertsGPUDirect(
            src_ctx,
            dst_ctx,
            expert_ids,
            params_.layer_idx,
            source_producer_stream,
            &satisfied_expert_ids,
            &completion);
        params_.gate_slab_ref = dst_ctx.gate_slab_ref;
        params_.up_slab_ref = dst_ctx.up_slab_ref;
        params_.down_slab_ref = dst_ctx.down_slab_ref;
        if (!satisfied_expert_ids.empty())
        {
            if (completion.valid())
                addPendingGpuDirectTransfer(std::move(completion));
            bindPreparedExpertEnginesForExperts(satisfied_expert_ids);
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            moe_runtime_table_initialized_ = false;
        }
        if (!ok && satisfied_expert_ids.empty())
        {
            LOG_DEBUG("[MoEExpertComputeStage] GPU-direct transfer did not satisfy any requested experts"
                      << " for layer " << params_.layer_idx);
        }
        return satisfied_expert_ids;
    }

    std::vector<int> MoEExpertComputeStage::missingPreparedExpertIds(
        const std::vector<int> &expert_ids) const
    {
        std::vector<int> missing;
        missing.reserve(expert_ids.size());
        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            const size_t idx = static_cast<size_t>(expert_id);
            const bool has_all =
                idx < params_.prepared_gate_gemm.size() &&
                idx < params_.prepared_up_gemm.size() &&
                idx < params_.prepared_down_gemm.size() &&
                params_.prepared_gate_gemm[idx] &&
                params_.prepared_up_gemm[idx] &&
                params_.prepared_down_gemm[idx];
            if (!has_all)
                missing.push_back(expert_id);
        }
        return missing;
    }

    std::vector<int> MoEExpertComputeStage::stageExpertsGPUDirectToTransferSlotsFrom(
        MoEExpertComputeStage &source,
        const std::vector<int> &expert_ids,
        void *source_producer_stream,
        GpuDirectTransferSlotArrivals *staged_arrivals,
        size_t active_arrival_capacity,
        size_t staging_pool_capacity,
        std::vector<std::shared_ptr<GpuExpertTransferStagingPool>> *transfer_staging_pools)
    {
        std::vector<int> satisfied_expert_ids;
        if (staged_arrivals)
            *staged_arrivals = GpuDirectTransferSlotArrivals{};
        if (!source_producer_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU-direct transfer-slot staging requires an explicit source producer stream"
                      << " for layer " << params_.layer_idx);
            return satisfied_expert_ids;
        }
        if (!staged_arrivals)
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU-direct transfer-slot staging requires an output carrier"
                      << " for layer " << params_.layer_idx);
            return satisfied_expert_ids;
        }

        auto src_ctx = source.buildWeightContext();
        auto dst_ctx = buildWeightContext();
        const bool ok = MoEExpertWeightService::stageExpertsGPUDirectToTransferSlots(
            src_ctx,
            dst_ctx,
            expert_ids,
            params_.layer_idx,
            source_producer_stream,
            staged_arrivals,
            &satisfied_expert_ids,
            active_arrival_capacity,
            staging_pool_capacity,
            transfer_staging_pools);
        params_.gate_slab_ref = dst_ctx.gate_slab_ref;
        params_.up_slab_ref = dst_ctx.up_slab_ref;
        params_.down_slab_ref = dst_ctx.down_slab_ref;
        if (!ok && satisfied_expert_ids.empty())
        {
            LOG_DEBUG("[MoEExpertComputeStage] GPU-direct transfer-slot staging did not satisfy requested experts"
                      << " for layer " << params_.layer_idx);
        }
        return satisfied_expert_ids;
    }

    std::vector<int> MoEExpertComputeStage::activateGpuDirectTransferSlotArrivals(
        const GpuDirectTransferSlotArrivals &arrivals,
        void *activation_stream,
        GpuDirectTransferCompletion *completion_out,
        bool retain_pending_completion)
    {
        std::vector<int> activated_expert_ids;
        if (arrivals.empty())
            return activated_expert_ids;
        if (!activation_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-slot activation requires an explicit stream"
                      << " for layer " << params_.layer_idx);
            return activated_expert_ids;
        }

        auto ctx = buildWeightContext();
        GpuDirectStagedExpertArrivals activated_arrivals;
        GpuDirectTransferCompletion completion;
        const bool activated = MoEExpertWeightService::activateGpuDirectTransferSlotArrivals(
            ctx,
            arrivals,
            activation_stream,
            &activated_arrivals,
            &completion);
        params_.gate_slab_ref = ctx.gate_slab_ref;
        params_.up_slab_ref = ctx.up_slab_ref;
        params_.down_slab_ref = ctx.down_slab_ref;
        if (!activated)
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to activate GPU-direct transfer slots"
                      << " for layer " << params_.layer_idx
                      << " on " << params_.device_id.to_string());
            return activated_expert_ids;
        }

        if (completion.valid())
            activated_arrivals.completion = completion;

        if (!MoEExpertWeightService::installActivatedGpuDirectArrivals(ctx, activated_arrivals))
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to publish activated GPU-direct arrivals"
                      << " for layer " << params_.layer_idx
                      << " on " << params_.device_id.to_string());
            return activated_expert_ids;
        }

        params_.gate_slab_ref = ctx.gate_slab_ref;
        params_.up_slab_ref = ctx.up_slab_ref;
        params_.down_slab_ref = ctx.down_slab_ref;
        activated_expert_ids = activated_arrivals.expertIds();
        if (!activated_expert_ids.empty())
        {
            if (completion_out)
                *completion_out = completion;
            if (retain_pending_completion && completion.valid())
                addPendingGpuDirectTransfer(std::move(completion));
            bindPreparedExpertEnginesForExperts(activated_expert_ids);
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            moe_runtime_table_initialized_ = false;
        }
        return activated_expert_ids;
    }

    void MoEExpertComputeStage::bindPreparedExpertEnginesForExperts(
        const std::vector<int> &expert_ids)
    {
        auto bind = [this](ITensorGemm *gemm)
        {
            if (!gemm)
                return;
            bindStageStream(gemm);
            if (bound_workspace_)
            {
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer && !consumer->hasWorkspace())
                    consumer->bindWorkspace(bound_workspace_);
            }
        };

        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            if (expert_id < static_cast<int>(params_.prepared_gate_gemm.size()))
                bind(params_.prepared_gate_gemm[expert_id]);
            if (expert_id < static_cast<int>(params_.prepared_up_gemm.size()))
                bind(params_.prepared_up_gemm[expert_id]);
            if (expert_id < static_cast<int>(params_.prepared_down_gemm.size()))
                bind(params_.prepared_down_gemm[expert_id]);
        }
    }

    void MoEExpertComputeStage::addPendingGpuDirectTransfer(
        GpuDirectTransferCompletion completion)
    {
        if (!completion.valid())
            return;
        const void *event_ptr = completion.ready_event.get();
        for (const auto &pending : pending_gpu_direct_transfers_)
        {
            if (pending.ready_event.get() == event_ptr)
                return;
        }
        pending_gpu_direct_transfers_.push_back(std::move(completion));
    }

    void MoEExpertComputeStage::addPendingGpuDirectTransfersFromStore(
        const std::vector<int> &expert_ids)
    {
        if (!params_.prepared_store)
            return;

        auto adopt_from_slab = [&](const std::optional<ExpertSlabRef> &slab_ref,
                                   int expert_id)
        {
            if (!slab_ref.has_value())
                return;
            auto completion =
                params_.prepared_store->expertGpuDirectCompletion(*slab_ref, expert_id);
            if (completion.has_value())
                addPendingGpuDirectTransfer(std::move(*completion));
        };

        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                continue;
            adopt_from_slab(params_.gate_slab_ref, expert_id);
            adopt_from_slab(params_.up_slab_ref, expert_id);
            adopt_from_slab(params_.down_slab_ref, expert_id);
        }
    }

    size_t MoEExpertComputeStage::releaseRawExpertWeights()
    {
        const auto active_expert = [&](size_t expert)
        {
            if (!params_.expert_mask.empty())
            {
                return expert < params_.expert_mask.size() &&
                       params_.expert_mask[expert];
            }
            const int local_start = std::max(0, params_.local_expert_start);
            const int local_count =
                params_.local_expert_count > 0
                    ? params_.local_expert_count
                    : params_.num_experts;
            return expert >= static_cast<size_t>(local_start) &&
                   expert < static_cast<size_t>(
                                std::min(params_.num_experts,
                                         local_start + local_count));
        };
        const auto table_releases_source =
            [&](const std::vector<ITensorGemm *> &engines)
        {
            if (engines.size() !=
                static_cast<size_t>(std::max(0, params_.num_experts)))
            {
                return false;
            }
            for (size_t expert = 0; expert < engines.size(); ++expert)
            {
                if (!active_expert(expert))
                    continue;
                if (!engines[expert] ||
                    !engines[expert]->canReleaseSourceWeightTensor())
                {
                    return false;
                }
            }
            return true;
        };

        if (!table_releases_source(params_.prepared_gate_gemm) ||
            !table_releases_source(params_.prepared_up_gemm) ||
            !table_releases_source(params_.prepared_down_gemm))
        {
            /*
             * This is not an execution fallback. The prepared engines remain
             * the only compute path; source tensors are retained solely because
             * at least one engine explicitly declares that it still owns a view
             * into those bytes.
             */
            LOG_TRACE("[MoEExpertComputeStage] Retaining raw expert source storage"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string()
                      << " reason=prepared_engine_references_source");
            return 0;
        }

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
        params_.gate_slab_ref = ctx.gate_slab_ref;
        params_.up_slab_ref = ctx.up_slab_ref;
        params_.down_slab_ref = ctx.down_slab_ref;
        if (ok)
        {
            std::vector<int> active_experts;
            active_experts.reserve(new_mask.size());
            for (size_t expert_id = 0; expert_id < new_mask.size(); ++expert_id)
            {
                if (new_mask[expert_id])
                    active_experts.push_back(static_cast<int>(expert_id));
            }
            bindPreparedExpertEnginesForExperts(active_experts);
            addPendingGpuDirectTransfersFromStore(active_experts);
            cached_gate_gemm_ = params_.prepared_gate_gemm;
            cached_up_gemm_ = params_.prepared_up_gemm;
            cached_down_gemm_ = params_.prepared_down_gemm;
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            moe_runtime_table_initialized_ = false;
        }
        return ok;
    }

    void MoEExpertComputeStage::applyExpertMask(const std::vector<bool> &new_mask)
    {
        params_.expert_mask = new_mask;
        if (params_.replica_set.num_replicated > 0 && !params_.expert_mask.empty())
            params_.replica_set.buildPrefillMask(params_.my_socket_id, params_.expert_mask, params_.layer_idx);
        else
            params_.replica_set.prefill_mask.clear();
        cached_gate_gemm_ = params_.prepared_gate_gemm;
        cached_up_gemm_ = params_.prepared_up_gemm;
        cached_down_gemm_ = params_.prepared_down_gemm;
        grouped_gateup_desc_table_dirty_ = true;
        grouped_down_desc_table_dirty_ = true;
        moe_runtime_table_initialized_ = false;
        invalidateFixedTopologyMaskPublication();
        if (!refreshGraphStablePlacement(/*preserve_capture_ready=*/true))
        {
            throw std::runtime_error(
                "MoE expert mask publication failed to refresh graph-stable runtime tables");
        }
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
            params_.down_slab_ref,
            true,
            &params_.gpu_direct_slot_pool};
    }

    IMoEKernel *MoEExpertComputeStage::ensureMoEKernel() const
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

    bool MoEExpertComputeStage::waitForPendingGpuDirectTransfers()
    {
        if (pending_gpu_direct_transfers_.empty())
            return true;

        if (!params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEExpertComputeStage] Pending GPU-direct expert transfers on non-GPU stage"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        void *consumer_stream = gpuStream();
        if (!consumer_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Pending GPU-direct expert transfers require an explicit compute stream"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        IBackend *backend = getBackendFor(params_.device_id);
        if (!backend)
        {
            LOG_ERROR("[MoEExpertComputeStage] No backend available to consume GPU-direct transfer event"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        const int device_ordinal = params_.device_id.is_cuda()
                                       ? params_.device_id.cuda_ordinal()
                                       : params_.device_id.rocm_ordinal();
        int waited_events = 0;
        for (const auto &completion : pending_gpu_direct_transfers_)
        {
            if (!completion.valid())
                continue;
            if (completion.device_id != params_.device_id ||
                completion.device_ordinal != device_ordinal)
            {
                LOG_ERROR("[MoEExpertComputeStage] GPU-direct transfer completion targets "
                          << completion.device_id.to_string()
                          << " ordinal=" << completion.device_ordinal
                          << " but stage is " << params_.device_id.to_string()
                          << " ordinal=" << device_ordinal
                          << " layer=" << params_.layer_idx);
                return false;
            }
            if (!backend->streamWaitEvent(consumer_stream,
                                          completion.ready_event.get(),
                                          device_ordinal))
            {
                LOG_ERROR("[MoEExpertComputeStage] Failed to wait on GPU-direct transfer completion event"
                          << " layer=" << params_.layer_idx
                          << " device=" << params_.device_id.to_string());
                return false;
            }
            ++waited_events;
        }

        if (waited_events > 0)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_consumer_event_waits",
                static_cast<double>(waited_events),
                "rebalance",
                params_.device_id.to_string(),
                {{"layer", std::to_string(params_.layer_idx)}});
        }
        pending_gpu_direct_transfers_.clear();
        return true;
    }

    bool MoEExpertComputeStage::refreshGraphStablePlacement(bool preserve_capture_ready)
    {
        return refreshRuntimeGroupedDecodePlacement(preserve_capture_ready) &&
               refreshFixedTopologyGroupedPrefillPlacement();
    }

    void MoEExpertComputeStage::invalidateFixedTopologyMaskPublication() noexcept
    {
        fixed_topology_mask_publication_state_ =
            usesPublishedFixedTopologyMaskGrouping()
                ? FixedTopologyMaskPublicationState::NeedsPublication
                : FixedTopologyMaskPublicationState::NotRequired;
    }

    bool MoEExpertComputeStage::publishFixedTopologyMaskBeforeCapture(IMoEKernel *kernel)
    {
        if (!usesPublishedFixedTopologyMaskGrouping())
        {
            fixed_topology_mask_publication_state_ =
                FixedTopologyMaskPublicationState::NotRequired;
            return true;
        }

        if (fixed_topology_mask_publication_state_ ==
            FixedTopologyMaskPublicationState::Published)
        {
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Fixed-topology expert mask was not "
                      "published before graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }
        if (!kernel)
        {
            LOG_ERROR("[MoEExpertComputeStage] Fixed-topology mask publication "
                      "requires a backend kernel"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }
        (void)requireGPUStream();

        const std::vector<uint8_t> mask = fixedTopologyPrefillExpertMaskBytes();
        if (!kernel->updateGroupedPrefillExpertMask(mask.data(), params_.num_experts))
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to publish grouped prefill "
                      "expert mask before graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        fixed_topology_mask_publication_state_ =
            FixedTopologyMaskPublicationState::Published;
        return true;
    }

    bool MoEExpertComputeStage::refreshRuntimeGroupedDecodePlacement(bool preserve_capture_ready)
    {
        if (!supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) ||
            params_.seq_len != 1 ||
            !params_.moe_runtime_table ||
            params_.layer_idx < 0 ||
            params_.d_model <= 0 ||
            params_.expert_intermediate <= 0 ||
            params_.num_experts <= 0)
        {
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_warmed_ = false;
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Refusing to refresh MoE runtime placement during graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_warmed_ = false;
            return false;
        }

        const bool was_capture_ready = runtime_grouped_decode_warmed_;
        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
        {
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_warmed_ = false;
            return false;
        }

        all_expert_ids_.clear();
        all_expert_ids_.reserve(static_cast<size_t>(params_.num_experts));
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (expertComputesLocally(expert_id))
                all_expert_ids_.push_back(expert_id);
        }

        if (all_expert_ids_.empty() ||
            !ensureGemmEnginesForExperts(all_expert_ids_) ||
            !ensureGroupedGateUpDescriptorTable(kernel, params_.d_model, params_.expert_intermediate) ||
            !ensureGroupedDownDescriptorTable(kernel, params_.d_model, params_.expert_intermediate) ||
            !initializeMoERuntimeTableForGroupedDecode())
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to refresh graph-stable MoE runtime placement"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            moe_runtime_table_initialized_ = false;
            runtime_grouped_decode_warmed_ = false;
            return false;
        }

        moe_runtime_table_initialized_ = runtimeTableHasActiveGroupedDecodeBank();
        runtime_grouped_decode_warmed_ =
            preserve_capture_ready && was_capture_ready &&
            grouped_gateup_desc_table_id_ >= 0 &&
            grouped_down_desc_table_id_ >= 0 &&
            moe_runtime_table_initialized_;
        return moe_runtime_table_initialized_;
    }

    bool MoEExpertComputeStage::refreshFixedTopologyGroupedPrefillPlacement()
    {
        if (!usesGraphStableFixedTopologyPrefillPlacement())
            return true;

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Refusing to refresh fixed-topology MoE prefill placement during graph capture"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
            return false;

        const std::vector<int> active_expert_ids = fixedTopologyPrefillExpertIds();
        if (active_expert_ids.empty() ||
            !ensureGemmEnginesForExperts(active_expert_ids) ||
            !ensureGroupedGateUpDescriptorTable(kernel, params_.d_model, params_.expert_intermediate) ||
            !ensureGroupedDownDescriptorTable(kernel, params_.d_model, params_.expert_intermediate))
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to refresh graph-stable fixed-topology prefill placement"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        if (usesPublishedFixedTopologyMaskGrouping())
        {
            invalidateFixedTopologyMaskPublication();
            if (!publishFixedTopologyMaskBeforeCapture(kernel))
                return false;
        }

        return true;
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
            const auto output_device =
                output_tensor ? output_tensor->current_device() : std::nullopt;
            if (!output_tensor ||
                !output_tensor->gpu_data_ptr() ||
                output_device != params_.device_id)
            {
                throw std::runtime_error(
                    "MoEExpertComputeStage standalone GPU output must be "
                    "preallocated on the exact stage device; execute() cannot "
                    "allocate graph-visible storage");
            }
        }

        if (!waitForPendingGpuDirectTransfers())
            return false;

        const bool has_complete_expert_views =
            params_.expert_gate_views.size() == static_cast<size_t>(params_.num_experts) &&
            params_.expert_up_views.size() == static_cast<size_t>(params_.num_experts) &&
            params_.expert_down_views.size() == static_cast<size_t>(params_.num_experts);
        if (!raw_weights_released_ &&
            (!params_.gate_exps || !params_.up_exps || !params_.down_exps) &&
            !has_complete_expert_views)
        {
            LOG_ERROR("[MoEExpertComputeStage] Missing both raw expert tensors and complete pre-extracted expert views");
            return false;
        }

        if (!supportsRequestedRoutedAssignmentPolicy())
        {
            LOG_ERROR("[MoEExpertComputeStage] Routed expert assignment policy "
                      << routedExpertAssignmentPolicyToString(params_.routed_assignment_policy)
                      << " is not wired for this execution path. Refusing to execute as StaticOwner.");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;
        const bool is_gpu = params_.device_id.is_gpu();

        if (params_.prefix_runtime_device_rehydration)
        {
            /*
             * This graph is the sole consumer of the transfer list published by
             * portable prefix restore. Execute it before every routing regime,
             * including M=1 decode. The operation reuses the compact LLEP
             * transport lane, fixed workspace, NCCL/RCCL collectives, and event
             * handoffs used by ordinary prefill movement; it performs no host
             * query, allocation, synchronization, or payload round trip.
             */
            if (params_.layer_idx == 0)
            {
                LOG_INFO("[MoEExpertComputeStage] Executing captured "
                         "prefix-runtime device rehydration transaction"
                         << " device=" << params_.device_id.to_string()
                         << " seq_len=" << params_.seq_len);
            }
            IMoEKernel *rehydration_kernel = ensureMoEKernel();
            DeviceMoERebalanceStatus *transfer_status = nullptr;
            DeviceMoERebalanceApplyStatus *apply_status = nullptr;
            if (!params_.prefix_runtime_rehydration_transfer_state ||
                !rehydration_kernel ||
                !executeTransferBackedPrefillLLEPMovement(
                    rehydration_kernel,
                    &transfer_status,
                    &apply_status,
                    params_.prefix_runtime_rehydration_transfer_state.get()) ||
                !transfer_status ||
                !apply_status)
            {
                throw std::runtime_error(
                    "MoE prefix-runtime device rehydration failed before routed expert execution");
            }
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "moe_layer_device_payload_rehydrations",
                1.0,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)}});
        }

        if (params_.force_decode_equivalent_verifier_prefill)
        {
            return executeDecodeEquivalentVerifierPrefill(ctx);
        }

        // Fast path for ordinary decode (seq_len=1): eliminates gather/scatter
        // overhead. MTP verifier rows that need the serial-decode oracle are
        // handled above before this ordinary decode shortcut can bypass them.
        if (params_.seq_len == 1 && !params_.force_grouped_verifier_prefill_for_decode)
        {
            return executeSingleToken(ctx);
        }

        if (!params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null routing_indices or routing_weights");
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
            LOG_ERROR("[MoEExpertComputeStage] Requires pre-extracted expert views or prepared expert slabs. "
                      "Call extractExpertViews()/prepareExpertGemmEngines() at graph build time.");
            return false;
        }

        // Get device-appropriate MoE kernel for gather/scatter
        IMoEKernel *kernel = ensureMoEKernel();

        // Resolve the contiguous expert-ID range owned by this participant.
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
            else if (grouped_gateup_desc_table_id_ < 0 ||
                     grouped_down_desc_table_id_ < 0 ||
                     grouped_gateup_desc_table_dirty_ ||
                     grouped_down_desc_table_dirty_)
            {
                const auto prefill_expert_ids = fixedTopologyPrefillExpertIds();
                tables_ready = ensureGemmEnginesForExperts(prefill_expert_ids) &&
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
                gpuExecution().publish(params_.output);
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
                                << ", fullOwnership=" << hasFullLocalExpertOwnership()
                                << ", allEnabled=" << expertMaskAllEnabled()
                                << ", replicas=" << params_.replica_set.num_replicated
                                << ", layer=" << params_.layer_idx);
            return false;
        }

        /*
         * Every supported GPU MoE mode must enter the same economical grouped
         * pipeline. The removed branch below downloaded counts/offsets and
         * drove one expert at a time from the host whenever capture was not yet
         * active. That made warmup semantically different from capture and
         * allowed stale host routing state to steer production execution.
         */
        if (params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU MoE execution requires the fully device-resident grouped pipeline: "
                      << "seq_len=" << params_.seq_len
                      << ", fullOwnership=" << hasFullLocalExpertOwnership()
                      << ", allEnabled=" << expertMaskAllEnabled()
                      << ", fixedMask=" << hasFixedTopologyPrefillExpertMask()
                      << ", replicas=" << params_.replica_set.num_replicated
                      << ", layer=" << params_.layer_idx);
            return false;
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
                                << ", fullOwnership=" << hasFullLocalExpertOwnership()
                                << ", allEnabled=" << expertMaskAllEnabled()
                                << ", replicas=" << params_.replica_set.num_replicated
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
        // With expert-ID apportionment, process only the local range while still
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
                // With static ownership or a dynamic resident mask, accumulate only local expert rows.
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
                        params_.replica_set.isReplicatedForLayer(params_.layer_idx, expert_id) &&
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
                gpuExecution().publish(scratch_batch_.get());

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
                    gpuExecution().publish(projection.output);
            }

            // Mandatory fused SwiGLU/down grouped implementation.
            if (!fusedSwigluDown(
                    *this,
                    scratch_gate_.get(), scratch_up_.get(), scratch_out_.get(),
                    down_gemm, num_tokens, d_model, intermediate,
                    getWorkspace()))
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
                gpuExecution().publish(params_.output);
        }

        LOG_TRACE("[MoEExpertComputeStage] Processed " << seq_len << " tokens via GEMM kernels, "
                                                       << top_k << " experts per token");
        if (is_gpu && !params_.output_registered_in_arena)
            gpuExecution().publish(params_.output);
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

        // Expert-ID ownership range.
        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

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
            params_.moe_runtime_table &&
            params_.layer_idx >= 0 &&
            (!moe_runtime_table_initialized_ || !runtime_decode_bank_active_for_expert_stage))
        {
            moe_runtime_table_initialized_ = !moe_runtime_table_initialized_
                                                 ? initializeMoERuntimeTableForGroupedDecode()
                                                 : (runtime_decode_bank_active_for_expert_stage ||
                                                    initializeMoERuntimeTableForGroupedDecode());
            /*
             * The first decode step after clear_cache() may initialize the
             * placement/runtime bank itself.  Treat that freshly initialized
             * bank as available for the same warmup pass so the fused runtime
             * grouped path is warmed before cached graph capture is armed.
             */
            runtime_decode_bank_active_for_expert_stage =
                moe_runtime_table_initialized_ &&
                moe_runtime_layer_ &&
                runtimeTableHasActiveGroupedDecodeBank();
        }

        const bool can_try_device_routed_decode =
            supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
            params_.moe_runtime_table &&
            moe_runtime_layer_ &&
            moe_runtime_table_initialized_ &&
            runtime_decode_bank_active_for_expert_stage &&
            top_k > 0 && top_k <= 16;

        if (params_.device_id.is_gpu() &&
            params_.seq_len == 1 &&
            params_.moe_runtime_table)
        {
            PerfStatsCollector::addCounter(
                "moe_runtime_decode",
                can_try_device_routed_decode ? "predicate_ready" : "predicate_skipped",
                1.0,
                "decode",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"backend_supported", perfBool(supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id))},
                 {"runtime_layer", perfBool(moe_runtime_layer_ != nullptr)},
                 {"initialized", perfBool(moe_runtime_table_initialized_)},
                 {"active_bank", perfBool(runtime_decode_bank_active_for_expert_stage)},
                 {"descriptor_source",
                  params_.runtime_decode_uses_mutable_descriptors ? "runtime" : "static_table"},
                 {"top_k", std::to_string(top_k)},
                 {"replicas", std::to_string(params_.replica_set.num_replicated)}});
        }

        if (can_try_device_routed_decode)
        {
            bool device_routed_done = false;
            const MoEDecodeDescriptorSource descriptor_source =
                params_.runtime_decode_uses_mutable_descriptors
                    ? MoEDecodeDescriptorSource::RuntimePlacementTable
                    : MoEDecodeDescriptorSource::StaticDescriptorTable;
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
                all_expert_ids_.clear();
                all_expert_ids_.reserve(static_cast<size_t>(num_experts));
                for (int expert_id = 0; expert_id < num_experts; ++expert_id)
                {
                    if (expertComputesLocally(expert_id))
                        all_expert_ids_.push_back(expert_id);
                }

                const PerfStatsCollector::Tags setup_tags{
                    {"layer", std::to_string(params_.layer_idx)},
                    {"num_experts", std::to_string(num_experts)},
                    {"top_k", std::to_string(top_k)}};
                PerfStatsCollector::addCounter(
                    "moe_runtime_decode", "local_expert_count",
                    static_cast<double>(all_expert_ids_.size()),
                    "decode", params_.device_id.toString(), setup_tags);

                if (all_expert_ids_.empty())
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "no_local_experts",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else if (!ensureGemmEnginesForExperts(all_expert_ids_))
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "ensure_engines_failed",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else if (!ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate))
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "gateup_descriptor_table_failed",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else if (!ensureGroupedDownDescriptorTable(kernel, d_model, intermediate))
                {
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "down_descriptor_table_failed",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
                else
                {
                    grouped_tables_ready = true;
                    PerfStatsCollector::addCounter(
                        "moe_runtime_decode", "descriptor_tables_ready",
                        1.0, "decode", params_.device_id.toString(), setup_tags);
                }
            }

            if (!grouped_tables_ready)
            {
                LOG_ERROR("[MoEExpertComputeStage] GPU runtime grouped decode descriptor tables "
                          "are unavailable for layer "
                          << params_.layer_idx);
                return false;
            }

            /*
             * CUDA and ROCm own their complete runtime decode scratch in the
             * stage-bound persistent workspace. The fused entry point is the one
             * production path: constructing per-slot TensorBase scratch here
             * would reintroduce dynamic allocation and a second decode contract.
             */
            device_routed_done = kernel->groupedExpertDecodeFromRuntime(
                moe_runtime_layer_,
                input_tensor,
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                top_k,
                params_.output,
                d_model,
                intermediate,
                descriptor_source);

            if (!device_routed_done)
            {
                LOG_ERROR("[MoEExpertComputeStage] Mandatory fused runtime grouped decode failed for layer "
                          << params_.layer_idx);
                return false;
            }

            if (!isGraphCaptureActive())
                runtime_grouped_decode_warmed_ = true;
            gpuExecution().publish(params_.output);
            return true;
        }

        const bool require_device_routing_tensor_decode =
            params_.require_device_routing_tensor_decode;
        std::vector<uint8_t> device_routing_expert_mask;
        const uint8_t *device_routing_expert_mask_ptr = nullptr;
        std::vector<int> device_routing_required_expert_ids;
        if (!params_.expert_mask.empty() &&
            params_.expert_mask.size() == static_cast<size_t>(num_experts) &&
            !expertMaskAllEnabled())
        {
            device_routing_expert_mask.resize(static_cast<size_t>(num_experts), 0u);
            device_routing_required_expert_ids.reserve(static_cast<size_t>(num_experts));
            for (int expert_id = 0; expert_id < num_experts; ++expert_id)
            {
                if (!params_.expert_mask[static_cast<size_t>(expert_id)])
                    continue;
                device_routing_expert_mask[static_cast<size_t>(expert_id)] = 1u;
                device_routing_required_expert_ids.push_back(expert_id);
            }
            if (!device_routing_required_expert_ids.empty())
                device_routing_expert_mask_ptr = device_routing_expert_mask.data();
        }

        /*
         * Explicit-routing M=1 decode binds one already-produced routing row as
         * device tensors and asks the backend to select descriptors without
         * reading a stale host mirror.  Dynamic/LLEP overlay masks can remain
         * participant scoped even when this GPU has every expert weight resident
         * locally, so the device route converts masked-off top-k slots to -1 and
         * computes the same participant-local partial result that the host route
         * would have contributed before the MoE allreduce.
         */
        const bool can_try_device_routing_tensor_decode =
            is_gpu &&
            supportsDeviceRoutingTensorDecodeExecutionBackend(params_.device_id) &&
            params_.routing_indices &&
            params_.routing_weights &&
            top_k > 0 && top_k <= 16 &&
            params_.replica_set.num_replicated == 0 &&
            (params_.expert_mask.empty() ||
             params_.expert_mask.size() == static_cast<size_t>(num_experts)) &&
            hasFullLocalExpertOwnership();

        if (can_try_device_routing_tensor_decode)
        {
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

                const std::vector<int> &required_expert_ids =
                    device_routing_expert_mask_ptr ? device_routing_required_expert_ids
                                                   : all_expert_ids_;

                grouped_tables_ready = ensureGemmEnginesForExperts(required_expert_ids) &&
                                       ensureGroupedGateUpDescriptorTable(kernel, d_model, intermediate) &&
                                       ensureGroupedDownDescriptorTable(kernel, d_model, intermediate);
            }

            if (!grouped_tables_ready)
            {
                LOG_ERROR("[MoEExpertComputeStage] Explicit-routing GPU decode "
                          "descriptor tables are unavailable for layer "
                          << params_.layer_idx);
                return false;
            }

            if (!kernel->groupedExpertDecodeFromRouting(
                    input_tensor,
                    params_.routing_indices,
                    params_.routing_weights,
                    grouped_gateup_desc_table_id_,
                    grouped_down_desc_table_id_,
                    top_k,
                    params_.output,
                    d_model,
                    intermediate,
                    device_routing_expert_mask_ptr))
            {
                LOG_ERROR("[MoEExpertComputeStage] Mandatory fused explicit-routing "
                          "GPU decode failed for layer "
                          << params_.layer_idx
                          << " (tables_ready=" << grouped_tables_ready
                          << ", top_k=" << top_k
                          << ", full_ownership=" << hasFullLocalExpertOwnership()
                          << ", expert_mask_all_enabled=" << expertMaskAllEnabled()
                          << ", replicas=" << params_.replica_set.num_replicated << ")");
                return false;
            }

            gpuExecution().publish(params_.output);
            return true;
        }

        if (is_gpu)
        {
            LOG_ERROR("[MoEExpertComputeStage] GPU decode has neither a usable "
                      "runtime-table route nor a usable explicit-routing fused path for layer "
                      << params_.layer_idx
                      << " (device=" << params_.device_id.toString()
                      << ", explicit_required=" << require_device_routing_tensor_decode
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

        /*
         * CPU decode owns reusable host tensors because CPU GEMV APIs consume
         * TensorBase outputs directly. GPU execution has already returned
         * through one of the fused workspace-native contracts above and can
         * never reach these allocations.
         */
        if (static_cast<int>(scratch_gate_batch_.size()) < top_k)
        {
            scratch_gate_batch_.resize(top_k);
            scratch_up_batch_.resize(top_k);
            for (int i = 0; i < top_k; ++i)
            {
                scratch_gate_batch_[i] =
                    makeScratchFP32(1, intermediate, params_.device_id);
                scratch_up_batch_[i] =
                    makeScratchFP32(1, intermediate, params_.device_id);
            }
        }
        if (!scratch_out_)
            scratch_out_ = makeScratchFP32(1, d_model, params_.device_id);

        if (!params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] Null routing_indices or routing_weights (single-token)");
            return false;
        }

        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();

        // ---------------------------------------------------------------
        // Phase 1: Batch all experts' gate+up into ONE fused GEMV call.
        // CPU NativeVNNI consumes the Q8_1 row already published by routing;
        // GPU and floating CPU bundles retain their backend-native activation
        // preparation.  Every route still uses one parallel projection region.
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
        // TODO(gpu-moe-rebalance): after the GPU rebalancing correctness gates
        // are proven, move this replica assignment into the device-side routing
        // path so homogeneous GPU domains do not round-trip token ownership
        // decisions through host-side ExpertReplicaSet logic.
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
                compute_here,
                params_.layer_idx);
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
                          << ", replicated=" << params_.replica_set.isReplicatedForLayer(params_.layer_idx, expert_id)
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

        // Single fused call: router-Q8 reuse on CPU NativeVNNI, then one
        // projection region for every active expert's gate/up pair.
        if (num_active > 0)
        {
            bool projected = false;
            if (!is_gpu)
            {
                bool any_native = false;
                bool all_native = true;
                for (const auto &projection : batch_projections_)
                {
                    const bool native =
                        dynamic_cast<CPUNativeVNNIGemmKernel *>(projection.kernel) != nullptr;
                    any_native = any_native || native;
                    all_native = all_native && native;
                }
                if (any_native != all_native)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU decode cannot mix "
                              "NativeVNNI and floating gate/up projection contracts in layer "
                              << params_.layer_idx);
                    return false;
                }

                if (all_native)
                {
                    auto *cpu_moe_kernel = dynamic_cast<CPUMoEKernel *>(kernel);
                    auto *native_gate = dynamic_cast<CPUNativeVNNIGemmKernel *>(
                        batch_projections_[0].kernel);
                    const float *input_rows = input_tensor->data();
                    const Q8_1Block *published_router_q8 =
                        cpu_moe_kernel && input_rows
                            ? cpu_moe_kernel->publishedRouterQ8Hidden(
                                  input_rows,
                                  /*rows=*/1,
                                  d_model)
                            : nullptr;
                    if (!native_gate || !published_router_q8)
                    {
                        LOG_ERROR("[MoEExpertComputeStage] CPU NativeVNNI decode is "
                                  "missing its router Q8_1 publication for layer "
                                  << params_.layer_idx);
                        return false;
                    }
                    projected =
                        native_gate->multiply_fused_router_q8_hidden_decode_equivalent(
                            published_router_q8,
                            batch_projections_,
                            /*m=*/1,
                            d_model);
                    if (projected)
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "cpu_moe_decode_router_q8_reuse_calls",
                            1.0,
                            "moe",
                            "cpu",
                            {{"top_k", std::to_string(top_k)},
                             {"active_experts", std::to_string(num_active)}});
                    }
                }
                else
                {
                    projected = batch_projections_[0].kernel->multiply_fused_tensor(
                        input_tensor,
                        batch_projections_,
                        /*m=*/1,
                        d_model,
                        nullptr,
                        getWorkspace());
                }
            }
            else
            {
                projected = batch_projections_[0].kernel->multiply_fused_tensor(
                    input_tensor,
                    batch_projections_,
                    /*m=*/1,
                    d_model,
                    nullptr,
                    getWorkspace());
            }

            if (!projected)
            {
                LOG_ERROR("[MoEExpertComputeStage] Decode gate/up batched projection failed for layer "
                          << params_.layer_idx);
                return false;
            }
            if (is_gpu)
            {
                for (const auto &projection : batch_projections_)
                    gpuExecution().publish(projection.output);
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
                            *this,
                            scratch_gate_batch_[info.batch_idx].get(),
                            scratch_up_batch_[info.batch_idx].get(),
                            scratch_out_.get(),
                            down_gemm, /*m=*/1, d_model, intermediate,
                            getWorkspace()))
                    {
                        LOG_ERROR("[MoEExpertComputeStage] CUDA decode SwiGLU/down projection failed for expert "
                                  << info.expert_id << " layer " << params_.layer_idx);
                        return false;
                    }

                    // GPU weighted accumulate: output += weight * scratch_out
                    kernel->weightedAddFromTensors(
                        params_.output, scratch_out_.get(), info.weight, d_model);
                    gpuExecution().publish(params_.output);
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
                            *this,
                            scratch_gate_batch_[info.batch_idx].get(),
                            scratch_up_batch_[info.batch_idx].get(),
                            scratch_out_.get(),
                            down_gemm, /*m=*/1, d_model, intermediate,
                            getWorkspace()))
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
            gpuExecution().publish(params_.output);
        return true;
    }

    bool MoEExpertComputeStage::executeCPUGroupedDecodeEquivalentVerifierPrefill(IDeviceContext *ctx)
    {
        (void)ctx;
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int intermediate = params_.expert_intermediate;

        if (params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier executor called for GPU device "
                      << params_.device_id.to_string());
            return false;
        }
        if (!params_.input || !params_.output ||
            !params_.routing_indices || !params_.routing_weights)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier executor missing tensors");
            return false;
        }

        const float *routing_idx_data = params_.routing_indices->data();
        const float *routing_wt_data = params_.routing_weights->data();
        if (!routing_idx_data || !routing_wt_data)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier executor could not access routing tensors");
            return false;
        }

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
            return false;

        struct VerifierRouteSlot
        {
            int row = 0;
            int route = 0;
            int expert_id = 0;
            float weight = 0.0f;
        };

        std::vector<VerifierRouteSlot> local_slots;
        local_slots.reserve(static_cast<size_t>(seq_len) * static_cast<size_t>(top_k));
        std::vector<int> original_slot_to_local(
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k),
            -1);
        std::vector<std::vector<int>> expert_local_slots(static_cast<size_t>(num_experts));
        std::vector<uint8_t> expert_needed(static_cast<size_t>(num_experts), 0u);

        const int local_start = params_.local_expert_start;
        const int local_count = (params_.local_expert_count < 0)
                                    ? num_experts
                                    : params_.local_expert_count;
        const int local_end = local_start + local_count;

        for (int row = 0; row < seq_len; ++row)
        {
            int routing_int_indices[16] = {};
            bool compute_here[16] = {};
            const float *row_weights =
                routing_wt_data + static_cast<size_t>(row) * static_cast<size_t>(top_k);

            for (int route = 0; route < top_k; ++route)
            {
                const size_t flat_slot =
                    static_cast<size_t>(row) * static_cast<size_t>(top_k) +
                    static_cast<size_t>(route);
                const int expert_id = static_cast<int>(routing_idx_data[flat_slot]);
                routing_int_indices[route] = expert_id;
                if (expert_id < 0 || expert_id >= num_experts)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier invalid expert id "
                              << expert_id << " row=" << row
                              << " route=" << route
                              << " layer=" << params_.layer_idx
                              << " num_experts=" << num_experts);
                    return false;
                }
                if (!std::isfinite(row_weights[route]))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier non-finite route weight"
                              << " row=" << row
                              << " route=" << route
                              << " layer=" << params_.layer_idx);
                    return false;
                }
            }

            if (params_.replica_set.num_replicated > 0)
            {
                params_.replica_set.assignForToken(
                    routing_int_indices,
                    row_weights,
                    top_k,
                    params_.my_socket_id,
                    params_.expert_mask,
                    compute_here,
                    params_.layer_idx);
            }
            else
            {
                if (!params_.expert_mask.empty() &&
                    params_.expert_mask.size() != static_cast<size_t>(num_experts))
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier expert mask size "
                              << params_.expert_mask.size()
                              << " does not match num_experts=" << num_experts
                              << " layer=" << params_.layer_idx);
                    return false;
                }
                for (int route = 0; route < top_k; ++route)
                {
                    const int expert_id = routing_int_indices[route];
                    compute_here[route] = params_.expert_mask.empty()
                                              ? (expert_id >= local_start && expert_id < local_end)
                                              : params_.expert_mask[static_cast<size_t>(expert_id)];
                }
            }

            for (int route = 0; route < top_k; ++route)
            {
                if (!compute_here[route])
                    continue;

                const int expert_id = routing_int_indices[route];
                const int local_slot = static_cast<int>(local_slots.size());
                const size_t flat_slot =
                    static_cast<size_t>(row) * static_cast<size_t>(top_k) +
                    static_cast<size_t>(route);
                original_slot_to_local[flat_slot] = local_slot;
                expert_local_slots[static_cast<size_t>(expert_id)].push_back(local_slot);
                expert_needed[static_cast<size_t>(expert_id)] = 1u;
                local_slots.push_back({row, route, expert_id, row_weights[route]});
            }
        }

        std::vector<int> active_experts;
        active_experts.reserve(static_cast<size_t>(num_experts));
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            if (expert_needed[static_cast<size_t>(expert_id)] != 0u)
                active_experts.push_back(expert_id);
        }
        if (!ensureGemmEnginesForExperts(active_experts))
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier missing prepared GEMM engines");
            return false;
        }

        const size_t output_bytes =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model) * sizeof(float);
        kernel->zeroBuffer(params_.output, output_bytes);
        if (local_slots.empty())
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "moe_routed_grouped_decode_equivalent_verifier_prefill_rows",
                static_cast<double>(seq_len),
                "verifier",
                params_.device_id.toString(),
                {{"stage", "routed_expert"},
                 {"route", "cpu_expert_slot_grouped"},
                 {"local_route_slots", "0"},
                 {"active_experts", "0"}});
            return true;
        }

        /*
         * NativeVNNI gate/up projections all consume the same Q8_1 activation
         * contract.  Require the immediately preceding CPU router publication
         * once for the complete verifier batch, then gather blocks from it for
         * each expert chunk.  Floating-point expert bundles retain their own
         * grouped implementation; mixing the two contracts within one routed
         * layer is rejected because it would make publication semantics depend
         * on which expert happened to win top-k.
         */
        bool any_native_gateup = false;
        bool all_native_gateup = true;
        for (int expert_id : active_experts)
        {
            const bool native_gate =
                dynamic_cast<CPUNativeVNNIGemmKernel *>(
                    cached_gate_gemm_[static_cast<size_t>(expert_id)]) != nullptr;
            const bool native_up =
                dynamic_cast<CPUNativeVNNIGemmKernel *>(
                    cached_up_gemm_[static_cast<size_t>(expert_id)]) != nullptr;
            if (native_gate != native_up)
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier found mixed "
                          "gate/up kernel contracts for expert "
                          << expert_id << " layer=" << params_.layer_idx);
                return false;
            }
            any_native_gateup = any_native_gateup || native_gate;
            all_native_gateup = all_native_gateup && native_gate;
        }
        if (any_native_gateup != all_native_gateup)
        {
            LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier cannot mix "
                      "NativeVNNI and floating expert gate/up contracts in layer "
                      << params_.layer_idx);
            return false;
        }

        const Q8_1Block *published_router_q8 = nullptr;
        const int router_q8_blocks_per_row =
            (d_model + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        if (all_native_gateup)
        {
            auto *cpu_moe_kernel = dynamic_cast<CPUMoEKernel *>(kernel);
            const float *input_rows = params_.input->data();
            if (!cpu_moe_kernel || !input_rows)
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU NativeVNNI grouped verifier "
                          "requires the production CPUMoEKernel router");
                return false;
            }
            published_router_q8 = cpu_moe_kernel->publishedRouterQ8Hidden(
                input_rows,
                seq_len,
                d_model);
            if (!published_router_q8)
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU NativeVNNI grouped verifier "
                          "is missing the matching router Q8_1 publication for layer "
                          << params_.layer_idx << " seq_len=" << seq_len
                          << " d_model=" << d_model);
                return false;
            }
        }

        std::vector<float> route_slot_outputs(
            local_slots.size() * static_cast<size_t>(d_model),
            0.0f);

        auto scratch_has_shape = [](const std::shared_ptr<FP32Tensor> &tensor,
                                    int min_rows,
                                    int cols) -> bool
        {
            if (!tensor)
                return false;
            const auto shape = tensor->shape();
            return shape.size() == 2u &&
                   shape[0] >= static_cast<size_t>(min_rows) &&
                   shape[1] == static_cast<size_t>(cols);
        };

        auto ensure_cpu_scratch = [&](int rows) -> bool
        {
            if (!all_native_gateup && !scratch_has_shape(scratch_batch_, rows, d_model))
                scratch_batch_ = makeScratchFP32(rows, d_model, params_.device_id);
            if (!scratch_has_shape(scratch_gate_, rows, intermediate))
                scratch_gate_ = makeScratchFP32(rows, intermediate, params_.device_id);
            if (!scratch_has_shape(scratch_up_, rows, intermediate))
                scratch_up_ = makeScratchFP32(rows, intermediate, params_.device_id);
            if (!scratch_has_shape(scratch_out_, rows, d_model))
                scratch_out_ = makeScratchFP32(rows, d_model, params_.device_id);
            scratch_capacity_ = std::max(scratch_capacity_, rows);
            return (all_native_gateup || scratch_batch_) &&
                   scratch_gate_ && scratch_up_ && scratch_out_;
        };

        std::vector<Q8_1Block> gathered_router_q8;
        if (all_native_gateup)
        {
            gathered_router_q8.resize(
                static_cast<size_t>(4) *
                static_cast<size_t>(router_q8_blocks_per_row));
        }

        for (int expert_id : active_experts)
        {
            const auto &slots_for_expert =
                expert_local_slots[static_cast<size_t>(expert_id)];
            ITensorGemm *gate_gemm = cached_gate_gemm_[static_cast<size_t>(expert_id)];
            ITensorGemm *up_gemm = cached_up_gemm_[static_cast<size_t>(expert_id)];
            ITensorGemm *down_gemm = cached_down_gemm_[static_cast<size_t>(expert_id)];
            if (!gate_gemm || !up_gemm || !down_gemm)
            {
                LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier found null GEMM for expert "
                          << expert_id << " layer=" << params_.layer_idx);
                return false;
            }

            for (size_t chunk_begin = 0; chunk_begin < slots_for_expert.size(); chunk_begin += 4u)
            {
                const int chunk_rows = static_cast<int>(
                    std::min<size_t>(4u, slots_for_expert.size() - chunk_begin));
                if (!ensure_cpu_scratch(chunk_rows))
                    return false;

                std::array<int, 4> token_indices = {};
                for (int i = 0; i < chunk_rows; ++i)
                {
                    const int local_slot =
                        slots_for_expert[chunk_begin + static_cast<size_t>(i)];
                    token_indices[static_cast<size_t>(i)] =
                        local_slots[static_cast<size_t>(local_slot)].row;
                }

                if (all_native_gateup)
                {
                    for (int i = 0; i < chunk_rows; ++i)
                    {
                        const Q8_1Block *source_row =
                            published_router_q8 +
                            static_cast<size_t>(token_indices[static_cast<size_t>(i)]) *
                                static_cast<size_t>(router_q8_blocks_per_row);
                        Q8_1Block *destination_row =
                            gathered_router_q8.data() +
                            static_cast<size_t>(i) *
                                static_cast<size_t>(router_q8_blocks_per_row);
                        std::copy_n(
                            source_row,
                            router_q8_blocks_per_row,
                            destination_row);
                    }
                }
                else
                {
                    kernel->gatherTokenBatchFromTensors(
                        params_.input,
                        scratch_batch_.get(),
                        token_indices.data(),
                        chunk_rows,
                        d_model);
                }

                batch_projections_.clear();
                batch_projections_.push_back(
                    {gate_gemm, scratch_gate_.get(), intermediate, nullptr, "gate"});
                batch_projections_.push_back(
                    {up_gemm, scratch_up_.get(), intermediate, nullptr, "up"});

                bool projected = false;
                if (all_native_gateup)
                {
                    auto *native_gate =
                        dynamic_cast<CPUNativeVNNIGemmKernel *>(gate_gemm);
                    projected = native_gate &&
                                native_gate->multiply_fused_router_q8_hidden_decode_equivalent(
                                    gathered_router_q8.data(),
                                    batch_projections_,
                                    chunk_rows,
                                    d_model);
                }
                else if (chunk_rows > 1)
                {
                    projected = gate_gemm->multiply_fused_verifier_rows_decode_equivalent(
                        scratch_batch_.get(),
                        batch_projections_,
                        chunk_rows,
                        d_model,
                        nullptr,
                        getWorkspace());
                }
                else
                {
                    projected = gate_gemm->multiply_fused_tensor(
                        scratch_batch_.get(),
                        batch_projections_,
                        chunk_rows,
                        d_model,
                        nullptr,
                        getWorkspace());
                }
                if (!projected)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier gate/up projection failed"
                              << " expert=" << expert_id
                              << " rows=" << chunk_rows
                              << " layer=" << params_.layer_idx);
                    return false;
                }

                const bool down_ok =
                    chunk_rows > 1
                        ? down_gemm->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                              scratch_gate_.get(),
                              scratch_up_.get(),
                              scratch_out_.get(),
                              chunk_rows,
                              d_model,
                              intermediate,
                              1.0f,
                              0.0f,
                              getWorkspace())
                        : fusedSwigluDown(
                              *this,
                              scratch_gate_.get(),
                              scratch_up_.get(),
                              scratch_out_.get(),
                              down_gemm,
                              chunk_rows,
                              d_model,
                              intermediate,
                              getWorkspace());
                if (!down_ok)
                {
                    LOG_ERROR("[MoEExpertComputeStage] CPU grouped verifier SwiGLU/down projection failed"
                              << " expert=" << expert_id
                              << " rows=" << chunk_rows
                              << " layer=" << params_.layer_idx);
                    return false;
                }

                const float *down_rows = scratch_out_->data();
                if (!down_rows)
                    return false;
                for (int i = 0; i < chunk_rows; ++i)
                {
                    const int local_slot =
                        slots_for_expert[chunk_begin + static_cast<size_t>(i)];
                    std::copy_n(
                        down_rows + static_cast<size_t>(i) * static_cast<size_t>(d_model),
                        d_model,
                        route_slot_outputs.data() +
                            static_cast<size_t>(local_slot) *
                                static_cast<size_t>(d_model));
                }
            }
        }

        float *output = params_.output->mutable_data();
        if (!output)
            return false;
        for (int row = 0; row < seq_len; ++row)
        {
            float *row_output =
                output + static_cast<size_t>(row) * static_cast<size_t>(d_model);
            for (int route = 0; route < top_k; ++route)
            {
                const size_t flat_slot =
                    static_cast<size_t>(row) * static_cast<size_t>(top_k) +
                    static_cast<size_t>(route);
                const int local_slot = original_slot_to_local[flat_slot];
                if (local_slot < 0)
                    continue;
                const auto &slot = local_slots[static_cast<size_t>(local_slot)];
                primitives::vec_axpy(
                    row_output,
                    route_slot_outputs.data() +
                        static_cast<size_t>(local_slot) *
                            static_cast<size_t>(d_model),
                    slot.weight,
                    d_model);
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_routed_grouped_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(seq_len),
            "verifier",
            params_.device_id.toString(),
            {{"stage", "routed_expert"},
             {"route", "cpu_expert_slot_grouped"},
             {"local_route_slots", std::to_string(local_slots.size())},
             {"active_experts", std::to_string(active_experts.size())},
             {"seq_len", std::to_string(seq_len)},
             {"top_k", std::to_string(top_k)}});
        if (all_native_gateup)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_moe_grouped_verifier_router_q8_reuse_calls",
                1.0,
                "moe",
                "cpu",
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"active_experts", std::to_string(active_experts.size())},
                 {"route", "cpu_expert_slot_grouped"}});
        }
        return true;
    }

    bool MoEExpertComputeStage::executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx)
    {
        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;

        const bool is_gpu = params_.device_id.is_gpu();

        if (seq_len < 1)
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
        PerfStatsCollector::addCounter(
            "mtp",
            "moe_decode_equivalent_verifier_prefill_runs",
            1.0,
            "verifier",
            params_.device_id.toString(),
            {{"stage", "routed_expert"},
             {"route", seq_len == 1
                           ? (is_gpu ? "grouped_prefill"
                                     : "single_row_decode")
                           : (is_gpu ? "grouped_prefill"
                                     : "cpu_expert_slot_grouped")},
             {"layer", std::to_string(params_.layer_idx)},
             {"seq_len", std::to_string(params_.seq_len)},
             {"top_k", std::to_string(params_.top_k)}});
        if (top_k > 16)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent verifier prefill top_k="
                      << top_k << " exceeds stack capacity");
            return false;
        }

        if (seq_len == 1)
        {
            struct ScopedSingleVerifierRow
            {
                Params &params;
                bool force_grouped_verifier_prefill_for_decode;
                bool force_decode_equivalent_verifier_prefill;
                bool require_device_routing_tensor_decode;
                int seq_len;

                ~ScopedSingleVerifierRow()
                {
                    params.force_grouped_verifier_prefill_for_decode =
                        force_grouped_verifier_prefill_for_decode;
                    params.force_decode_equivalent_verifier_prefill =
                        force_decode_equivalent_verifier_prefill;
                    params.require_device_routing_tensor_decode =
                        require_device_routing_tensor_decode;
                    params.seq_len = seq_len;
                }
            } restore{
                params_,
                params_.force_grouped_verifier_prefill_for_decode,
                params_.force_decode_equivalent_verifier_prefill,
                params_.require_device_routing_tensor_decode,
                params_.seq_len};

            /*
             * M=1 is the degenerate verifier bucket: there is no grouping economy
             * to harvest, so the production contract is the backend's exact
             * one-row decode implementation.  GPU participants still stay
             * device-resident by requiring the explicit routing tensor path; a
             * missing device route is a hard failure rather than a host replay.
             */
            params_.seq_len = 1;
            params_.force_decode_equivalent_verifier_prefill = false;
            params_.force_grouped_verifier_prefill_for_decode = false;
            params_.require_device_routing_tensor_decode = is_gpu;

            return executeSingleToken(ctx);
        }

        if (!is_gpu)
            return executeCPUGroupedDecodeEquivalentVerifierPrefill(ctx);

        IMoEKernel *kernel = ensureMoEKernel();
        if (!kernel)
            return false;
        if (!canUseFixedTopologyGroupedPrefill())
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent GPU verifier M="
                      << seq_len
                      << " requires the grouped fixed-topology prefill path; "
                      << "row replay is not a production fallback"
                      << " device=" << params_.device_id.to_string()
                      << " layer=" << params_.layer_idx
                      << " fullOwnership=" << hasFullLocalExpertOwnership()
                      << " allEnabled=" << expertMaskAllEnabled()
                      << " replicas=" << params_.replica_set.num_replicated);
            return false;
        }

        if (static_cast<int>(all_expert_ids_.size()) != num_experts)
        {
            all_expert_ids_.resize(static_cast<size_t>(num_experts));
            std::iota(all_expert_ids_.begin(), all_expert_ids_.end(), 0);
        }

        const auto prefill_expert_ids = fixedTopologyPrefillExpertIds();
        const bool tables_ready =
            ensureGemmEnginesForExperts(prefill_expert_ids) &&
            ensureGroupedGateUpDescriptorTable(kernel, d_model, params_.expert_intermediate) &&
            ensureGroupedDownDescriptorTable(kernel, d_model, params_.expert_intermediate);
        if (!tables_ready)
        {
            LOG_ERROR("[MoEExpertComputeStage] Decode-equivalent GPU verifier failed to prepare grouped tables"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        const bool grouped_ok = executeFixedTopologyGroupedPrefill(kernel, seq_len);
        if (grouped_ok)
            gpuExecution().publish(params_.output);
        return grouped_ok;
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

        addPendingGpuDirectTransfersFromStore(expert_ids);
        if (!waitForPendingGpuDirectTransfers())
            return false;

        auto bind_if_needed = [this](ITensorGemm *gemm)
        {
            if (!gemm)
                return;
            /*
             * Workspace planning resolves prepared engines before the executor
             * binds a graph stream. Stream propagation is therefore conditional
             * here; execute() performs the mandatory propagation immediately
             * before any kernel launch. Calling gpuStream() merely to test for a
             * binding would turn harmless planning into an implicit-stream
             * failure and blur the scheduler/compute ownership boundary.
             */
            if (!params_.device_id.is_gpu() || hasGPUStream())
                bindStageStream(gemm);
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
            grouped_gateup_desc_table_intermediate_ == intermediate &&
            !grouped_gateup_desc_table_dirty_)
        {
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Grouped gate/up descriptor table is dirty during graph capture"
                      << " layer=" << params_.layer_idx);
            return false;
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

        int table_id = grouped_gateup_desc_table_id_;
        if (table_id >= 0 &&
            grouped_gateup_desc_table_num_experts_ == params_.num_experts &&
            grouped_gateup_desc_table_d_model_ == d_model &&
            grouped_gateup_desc_table_intermediate_ == intermediate)
        {
            if (!kernel->updateGroupedExpertGateUpDescriptorTables(
                    table_id, gate_descs.data(), up_descs.data(),
                    params_.num_experts, d_model, intermediate))
            {
                return false;
            }
            grouped_gateup_desc_table_dirty_ = false;
            return true;
        }

        table_id = kernel->uploadGroupedExpertGateUpDescriptorTables(
            gate_descs.data(), up_descs.data(), params_.num_experts, d_model, intermediate);
        if (table_id < 0)
            return false;

        grouped_gateup_desc_table_id_ = table_id;
        grouped_gateup_desc_table_num_experts_ = params_.num_experts;
        grouped_gateup_desc_table_d_model_ = d_model;
        grouped_gateup_desc_table_intermediate_ = intermediate;
        grouped_gateup_desc_table_dirty_ = false;
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
            grouped_down_desc_table_intermediate_ == intermediate &&
            !grouped_down_desc_table_dirty_)
        {
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Grouped down descriptor table is dirty during graph capture"
                      << " layer=" << params_.layer_idx);
            return false;
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

        int table_id = grouped_down_desc_table_id_;
        if (table_id >= 0 &&
            grouped_down_desc_table_num_experts_ == params_.num_experts &&
            grouped_down_desc_table_d_model_ == d_model &&
            grouped_down_desc_table_intermediate_ == intermediate)
        {
            if (!kernel->updateGroupedExpertDownDescriptorTable(
                    table_id, down_descs.data(),
                    params_.num_experts, d_model, intermediate))
            {
                return false;
            }
            grouped_down_desc_table_dirty_ = false;
            return true;
        }

        table_id = kernel->uploadGroupedExpertDownDescriptorTable(
            down_descs.data(), params_.num_experts, d_model, intermediate);
        if (table_id < 0)
            return false;

        grouped_down_desc_table_id_ = table_id;
        grouped_down_desc_table_num_experts_ = params_.num_experts;
        grouped_down_desc_table_d_model_ = d_model;
        grouped_down_desc_table_intermediate_ = intermediate;
        grouped_down_desc_table_dirty_ = false;
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
            combined_shared_desc_table_intermediate_ == intermediate &&
            combined_shared_gateup_desc_table_id_ >= 0 &&
            combined_shared_gateup_desc_table_d_model_ == d_model &&
            combined_shared_gateup_desc_table_intermediate_ == intermediate &&
            combined_shared_down_desc_table_id_ >= 0 &&
            combined_shared_down_desc_table_d_model_ == d_model &&
            combined_shared_down_desc_table_intermediate_ == intermediate)
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

        bindStageStream(shared_gate);
        bindStageStream(shared_up);
        bindStageStream(shared_down);
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

        const int gateup_table_id = kernel->uploadGroupedExpertGateUpDescriptorTables(
            &gate_desc, &up_desc, /*num_experts=*/1, d_model, intermediate);
        if (gateup_table_id < 0)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined safe verifier failed to upload "
                      "shared expert grouped gate/up descriptor table"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        const int down_table_id = kernel->uploadGroupedExpertDownDescriptorTable(
            &down_desc, /*num_experts=*/1, d_model, intermediate);
        if (down_table_id < 0)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined safe verifier failed to upload "
                      "shared expert grouped down descriptor table"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        combined_shared_gate_gemm_ = shared_gate;
        combined_shared_up_gemm_ = shared_up;
        combined_shared_down_gemm_ = shared_down;
        combined_shared_desc_table_d_model_ = d_model;
        combined_shared_desc_table_intermediate_ = intermediate;
        combined_shared_gateup_desc_table_id_ = gateup_table_id;
        combined_shared_gateup_desc_table_d_model_ = d_model;
        combined_shared_gateup_desc_table_intermediate_ = intermediate;
        combined_shared_down_desc_table_id_ = down_table_id;
        combined_shared_down_desc_table_d_model_ = d_model;
        combined_shared_down_desc_table_intermediate_ = intermediate;
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

            const bool publication_required =
                params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx);
            const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            if (!publication_required &&
                state.active_epoch == std::numeric_limits<uint32_t>::max())
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                          << params_.layer_idx << ": epoch counter exhausted");
                return false;
            }

            const bool has_replicas = params_.replica_set.num_replicated > 0;
            if (params_.runtime_decode_uses_mutable_descriptors ||
                params_.runtime_decode_has_explicit_owner_metadata)
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot synthesize MoE runtime decode bank for layer "
                          << params_.layer_idx
                          << ": explicit owner/resident metadata decode requires a "
                          << "graph-initialized placement table");
                return false;
            }
            if (has_replicas &&
                (params_.replica_set.is_replicated.size() != static_cast<size_t>(params_.num_experts) ||
                 params_.replica_set.owner_socket.size() != static_cast<size_t>(params_.num_experts)))
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                          << params_.layer_idx << ": replica metadata does not match num_experts="
                          << params_.num_experts);
                return false;
            }

            const int participant_count = expectedGroupedDecodeParticipantCount();
            if (params_.my_socket_id < 0 ||
                params_.my_socket_id >= participant_count ||
                participant_count > static_cast<int>(kDeviceMoEMaxParticipants))
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                          << params_.layer_idx << ": invalid participant metadata id="
                          << params_.my_socket_id << " count=" << participant_count);
                return false;
            }
            if (!has_replicas &&
                !expertMaskAllEnabled() &&
                participant_count > 1)
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot synthesize masked multi-participant "
                          << "MoE runtime decode bank for layer " << params_.layer_idx
                          << ": explicit owner metadata is required");
                return false;
            }

            all_expert_ids_.clear();
            all_expert_ids_.reserve(static_cast<size_t>(params_.num_experts));
            for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
            {
                if (expertComputesLocally(expert_id))
                    all_expert_ids_.push_back(expert_id);
            }
            if (all_expert_ids_.empty())
                return false;
            if (!ensureGemmEnginesForExperts(all_expert_ids_))
                return false;

            MoEPlacementUpdate update;
            update.epoch = publication_required ? 1u : state.active_epoch + 1u;
            update.expert_count = static_cast<uint32_t>(params_.num_experts);
            update.participant_id = static_cast<uint32_t>(params_.my_socket_id);
            update.participant_count = static_cast<uint32_t>(participant_count);
            update.experts.resize(static_cast<size_t>(params_.num_experts));
            update.local_compute_mask.assign(static_cast<size_t>(params_.num_experts), 0u);
            update.replica_role.assign(static_cast<size_t>(params_.num_experts),
                                       static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(static_cast<size_t>(params_.num_experts), 0u);

            for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
            {
                const bool computes_locally = expertComputesLocally(expert_id);
                const bool replicated =
                    has_replicas && params_.replica_set.isReplicatedForLayer(params_.layer_idx, expert_id);
                int owner_participant = params_.my_socket_id;
                if (has_replicas)
                    owner_participant = params_.replica_set.owner_socket[static_cast<size_t>(expert_id)];

                uint32_t resident_mask = 0u;
                if (owner_participant >= 0 && owner_participant < participant_count)
                    resident_mask |= (1u << static_cast<uint32_t>(owner_participant));
                if (replicated)
                {
                    for (int participant = 0; participant < participant_count; ++participant)
                    {
                        if (params_.replica_set.hasReplicaOnParticipant(
                                params_.layer_idx, expert_id, participant))
                        {
                            resident_mask |= (1u << static_cast<uint32_t>(participant));
                        }
                    }
                }
                if (computes_locally)
                    resident_mask |= (1u << static_cast<uint32_t>(params_.my_socket_id));
                update.resident_participant_mask[static_cast<size_t>(expert_id)] = resident_mask;

                DeviceMoEExpertDescriptor desc;
                desc.logical_expert_id = expert_id;
                desc.owner_participant = owner_participant;
                desc.local_slot = computes_locally ? expert_id : -1;
                if (replicated)
                    desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
                update.experts[static_cast<size_t>(expert_id)] = desc;

                if (!expertComputesLocally(expert_id))
                    continue;

                if (!cached_gate_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(desc.gate) ||
                    !cached_up_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(desc.up) ||
                    !cached_down_gemm_[static_cast<size_t>(expert_id)]->exportNativeVNNIMatrixDesc(desc.down))
                {
                    LOG_DEBUG("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                              << params_.layer_idx << " expert " << expert_id
                              << ": prepared GEMM engines did not export native-VNNI descriptors");
                    return false;
                }

                DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                             DeviceMoEExpertFlags::Resident |
                                             DeviceMoEExpertFlags::LocalCompute;
                if (!has_replicas || owner_participant == params_.my_socket_id)
                    flags |= DeviceMoEExpertFlags::PreferredOwner;
                if (replicated)
                    flags |= DeviceMoEExpertFlags::Replicated;

                desc.local_slot = expert_id;
                desc.flags = toMoEExpertFlags(flags);
                update.experts[static_cast<size_t>(expert_id)] = desc;
                update.local_compute_mask[static_cast<size_t>(expert_id)] = 1u;
                if (replicated)
                {
                    update.replica_role[static_cast<size_t>(expert_id)] =
                        static_cast<uint8_t>(owner_participant == params_.my_socket_id
                                                 ? DeviceMoEReplicaRole::Primary
                                                 : DeviceMoEReplicaRole::Replica);
                }
                else
                {
                    update.replica_role[static_cast<size_t>(expert_id)] =
                        static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
                }
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

    int MoEExpertComputeStage::expectedGroupedDecodeParticipantCount() const
    {
        int participant_count = params_.participant_count;
        const bool has_replicas = params_.replica_set.num_replicated > 0;
        if (participant_count <= 0 && has_replicas)
            participant_count = params_.replica_set.num_sockets;
        if (participant_count <= 0 && has_replicas)
        {
            for (int owner : params_.replica_set.owner_socket)
                participant_count = std::max(participant_count, owner + 1);
        }
        if (participant_count <= 0)
            participant_count = std::max(1, params_.my_socket_id + 1);
        return participant_count;
    }

    bool MoEExpertComputeStage::initializeMoERuntimeTableForGroupedPrefill()
    {
        moe_prefill_runtime_grouping_available_ = false;
        const bool static_owner_runtime_grouping =
            params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner &&
            ((hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
             hasFixedTopologyPrefillExpertMask());
        const bool least_loaded_runtime_grouping =
            params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident &&
            hasFixedTopologyPrefillExpertMask();
        if (!params_.use_runtime_prefill_grouping ||
            !params_.moe_runtime_table ||
            params_.layer_idx < 0 ||
            params_.seq_len <= 1 ||
            params_.num_experts <= 0 ||
            params_.top_k <= 0 ||
            !supportsGroupedPrefillExecutionBackend(params_.device_id) ||
            (!static_owner_runtime_grouping && !least_loaded_runtime_grouping))
        {
            return false;
        }

        auto *runtime_table = dynamic_cast<DeviceMoERuntimeTable *>(params_.moe_runtime_table);
        if (!runtime_table)
        {
            LOG_ERROR("[MoEExpertComputeStage] Runtime prefill grouping requires DeviceMoERuntimeTable"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        if (!runtime_table->hasPrefillRouteScratchCapacity(params_.layer_idx, params_.seq_len))
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[MoEExpertComputeStage] Runtime prefill scratch was not warmed before graph capture"
                          << " layer=" << params_.layer_idx
                          << " seq_len=" << params_.seq_len);
                return false;
            }
            runtime_table->ensurePrefillRouteScratchCapacity(params_.seq_len, gpuStream());
        }

        if (!runtime_table->hasPrefillRouteScratchCapacity(params_.layer_idx, params_.seq_len))
        {
            LOG_ERROR("[MoEExpertComputeStage] Runtime prefill scratch capacity check failed"
                      << " layer=" << params_.layer_idx
                      << " seq_len=" << params_.seq_len);
            return false;
        }

        moe_runtime_layer_ = runtime_table->deviceLayerState(params_.layer_idx);
        const auto &state = runtime_table->hostLayerState(params_.layer_idx);
        moe_prefill_runtime_grouping_available_ =
            moe_runtime_layer_ &&
            state.expert_count == static_cast<uint32_t>(params_.num_experts) &&
            state.top_k == static_cast<uint32_t>(params_.top_k) &&
            state.prefill_token_capacity >= static_cast<uint32_t>(params_.seq_len) &&
            state.prefill_route_capacity >= static_cast<uint32_t>(params_.seq_len * params_.top_k) &&
            state.route_expert_ids &&
            state.route_weights &&
            state.route_participant_ids &&
            state.expert_counts &&
            state.expert_offsets &&
            state.grouped_token_ids &&
            state.grouped_route_weights;
        return moe_prefill_runtime_grouping_available_;
    }

    bool MoEExpertComputeStage::initializeFixedTopologyGroupedPrefill()
    {
        return false;
    }

    bool MoEExpertComputeStage::runtimeTableHasActiveGroupedDecodeBank() const
    {
        if (!params_.moe_runtime_table ||
            params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx))
        {
            return false;
        }

        const DeviceMoEPlacementBank *bank = activeRuntimePlacementBank();
        if (!bank || !moe_runtime_layer_)
            return false;

        const bool mutable_runtime_descriptors =
            params_.runtime_decode_uses_mutable_descriptors;
        const bool runtime_owner_metadata =
            mutable_runtime_descriptors ||
            params_.runtime_decode_has_explicit_owner_metadata;
        const bool has_replicas = params_.replica_set.num_replicated > 0;
        int participant_count = 0;
        try
        {
            const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            if (state.participant_id != static_cast<uint32_t>(params_.my_socket_id))
                return false;
            if (state.expert_count != static_cast<uint32_t>(params_.num_experts) ||
                state.top_k != static_cast<uint32_t>(params_.top_k))
            {
                return false;
            }
            if (has_replicas &&
                (params_.replica_set.is_replicated.size() != static_cast<size_t>(params_.num_experts) ||
                 params_.replica_set.owner_socket.size() != static_cast<size_t>(params_.num_experts)))
            {
                return false;
            }
            participant_count = expectedGroupedDecodeParticipantCount();
            if (params_.my_socket_id < 0 ||
                params_.my_socket_id >= participant_count ||
                participant_count <= 0 ||
                participant_count > static_cast<int>(kDeviceMoEMaxParticipants) ||
                state.participant_count != static_cast<uint32_t>(participant_count))
            {
                return false;
            }
        }
        catch (...)
        {
            return false;
        }

        const uint32_t valid_mask =
            participant_count >= static_cast<int>(kDeviceMoEMaxParticipants)
                ? ((1u << kDeviceMoEMaxParticipants) - 1u)
                : ((1u << static_cast<uint32_t>(participant_count)) - 1u);
        const uint32_t local_participant_bit =
            1u << static_cast<uint32_t>(params_.my_socket_id);

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            const uint32_t raw_resident_mask =
                bank->resident_participant_mask[static_cast<size_t>(expert_id)];
            const auto &expert = bank->experts[static_cast<size_t>(expert_id)];
            int owner_participant = expert.owner_participant;
            bool replicated = false;
            uint32_t effective_resident_mask = raw_resident_mask & valid_mask;

            if (runtime_owner_metadata)
            {
                if (owner_participant >= 0 && owner_participant < participant_count)
                    effective_resident_mask |=
                        1u << static_cast<uint32_t>(owner_participant);
                if ((raw_resident_mask & ~valid_mask) != 0u ||
                    effective_resident_mask == 0u ||
                    expert.logical_expert_id != expert_id ||
                    owner_participant < -1 ||
                    owner_participant >= participant_count)
                {
                    return false;
                }
                replicated =
                    (effective_resident_mask & (effective_resident_mask - 1u)) != 0u;
            }
            else
            {
                owner_participant = params_.my_socket_id;
                if (has_replicas)
                    owner_participant = params_.replica_set.owner_socket[static_cast<size_t>(expert_id)];
                replicated =
                    has_replicas &&
                    params_.replica_set.isReplicatedForLayer(params_.layer_idx, expert_id);
            }

            const bool expected_local =
                runtime_owner_metadata
                    ? ((effective_resident_mask & local_participant_bit) != 0u)
                    : expertComputesLocally(expert_id);
            if (bank->local_compute_mask[static_cast<size_t>(expert_id)] !=
                (expected_local ? 1u : 0u))
            {
                return false;
            }

            uint32_t expected_resident_mask = 0u;
            if (owner_participant >= 0 && owner_participant < participant_count)
                expected_resident_mask |= (1u << static_cast<uint32_t>(owner_participant));
            if (!mutable_runtime_descriptors && replicated)
            {
                for (int participant = 0; participant < participant_count; ++participant)
                {
                    if (params_.replica_set.hasReplicaOnParticipant(
                            params_.layer_idx,
                            expert_id,
                            participant))
                    {
                        expected_resident_mask |= (1u << static_cast<uint32_t>(participant));
                    }
                }
            }
            if (expected_local)
                expected_resident_mask |= (1u << static_cast<uint32_t>(params_.my_socket_id));
            if (!runtime_owner_metadata && raw_resident_mask != expected_resident_mask)
            {
                return false;
            }

            if (expert.logical_expert_id != expert_id)
                return false;
            if (!runtime_owner_metadata &&
                expert.owner_participant != owner_participant)
            {
                return false;
            }

            if (!mutable_runtime_descriptors)
            {
                if (replicated != hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Replicated))
                    return false;
                const auto expected_role =
                    replicated
                        ? (owner_participant == params_.my_socket_id
                               ? DeviceMoEReplicaRole::Primary
                               : DeviceMoEReplicaRole::Replica)
                        : (expected_local ? DeviceMoEReplicaRole::Primary : DeviceMoEReplicaRole::None);
                if (bank->replica_role[static_cast<size_t>(expert_id)] !=
                    static_cast<uint8_t>(expected_role))
                {
                    return false;
                }
            }

            if (!expected_local)
                continue;

            if (expert.local_slot < 0 ||
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

    bool MoEExpertComputeStage::supportsRequestedRoutedAssignmentPolicy() const
    {
        if (params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner)
            return true;

        if (params_.routed_assignment_policy != RoutedExpertAssignmentPolicy::LeastLoadedResident)
            return false;

        const bool supports_llep_prefill =
            params_.seq_len > 1 &&
            params_.use_runtime_prefill_grouping &&
            params_.moe_runtime_table &&
            params_.prefill_llep_tp_ctx &&
            supportsGroupedPrefillExecutionBackend(params_.device_id);
        if (supports_llep_prefill)
            return true;

        return params_.seq_len == 1 &&
               !params_.force_grouped_verifier_prefill_for_decode &&
               params_.moe_runtime_table &&
               params_.layer_idx >= 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id);
    }

    bool MoEExpertComputeStage::canUseRuntimePrefillGrouping() const
    {
        if (!params_.use_runtime_prefill_grouping ||
            !moe_prefill_runtime_grouping_available_ ||
            !params_.moe_runtime_table ||
            !moe_runtime_layer_ ||
            params_.seq_len <= 1)
        {
            return false;
        }
        /*
         * Prefix restore and MTP transaction reset mutate the contents behind
         * this stable runtime-table pointer.  A cached stage-local warmup bit
         * therefore cannot prove that the current table still owns every
         * persistent grouping buffer.  Revalidate the authoritative table on
         * every execution boundary; this is a host metadata check only and
         * performs no allocation, copy, or synchronization in the hot path.
         */
        const auto *runtime_table =
            dynamic_cast<const DeviceMoERuntimeTable *>(params_.moe_runtime_table);
        if (!runtime_table ||
            !runtime_table->hasPrefillRouteScratchCapacity(
                params_.layer_idx,
                params_.seq_len))
        {
            return false;
        }
        if (params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner)
            return (hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
                   hasFixedTopologyPrefillExpertMask();
        if (params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
            return hasFixedTopologyPrefillExpertMask();
        return false;
    }

    bool MoEExpertComputeStage::canUseFixedTopologyGroupedPrefill() const
    {
        const bool forced_decode_replay =
            params_.force_grouped_verifier_prefill_for_decode && params_.seq_len == 1;
        return supportsGroupedPrefillExecutionBackend(params_.device_id) &&
               (params_.seq_len > 1 || forced_decode_replay) &&
               (!forced_decode_replay || params_.replica_set.num_replicated == 0) &&
               ((hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
                hasFixedTopologyPrefillExpertMask());
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
            combined_shared_gateup_desc_table_id_ < 0 ||
            combined_shared_down_desc_table_id_ < 0 ||
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
                            "shared_output"))
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
        gpuExecution().publish(combined_routed_output_.get());

        /*
         * The standalone shared-expert verifier path targets serial decode's
         * grouped table-decode family by using the sibling grouped table-prefill
         * kernels for M=2..4.  The composite path must use that same production
         * route; otherwise it silently reintroduces the dense GEMM verifier hooks
         * that can differ by a single FP32 ulp in layer 0 and later flip MoE
         * routing decisions.  This is still a grouped implementation: one device
         * grouping setup and one grouped prefill pipeline cover all verifier rows.
         */
        if (!kernel->prepareSharedExpertPrefillGroup(seq_len))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                      "shared grouped verifier setup failed");
            return false;
        }
        if (!kernel->executeGroupedPrefillPipeline(
                params_.input,
                combined_shared_output_.get(),
                combined_shared_gateup_desc_table_id_,
                combined_shared_down_desc_table_id_,
                seq_len,
                d_model,
                intermediate,
                /*num_experts=*/1,
                /*top_k=*/1))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                      "shared grouped verifier failed");
            return false;
        }
        gpuExecution().publish(combined_shared_output_.get());

        kernel->sharedExpertGateAddFromTensors(
            params_.input,
            shared_gate_inp,
            combined_shared_output_.get(),
            combined_routed_output_.get(),
            params_.output,
            seq_len,
            d_model);
        gpuExecution().publish(combined_shared_output_.get());
        if (params_.device_id.is_gpu())
            gpuExecution().publish(params_.output);

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_combined_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(seq_len),
            "verifier",
            params_.device_id.toString(),
            {{"stage", "routed_plus_shared"},
             {"route", "safe_composite_grouped_table_prefill"},
             {"seq_len", std::to_string(seq_len)},
             {"routed_top_k", std::to_string(params_.top_k)},
             {"routed_experts", std::to_string(params_.num_experts)},
             {"layer", std::to_string(params_.layer_idx)}});
        return true;
    }

    bool MoEExpertComputeStage::
        requestsTransferBackedCurrentBatchPrefillLLEP() const noexcept
    {
        return params_.prefill_llep_assignment_mode ==
               PrefillLLEPAssignmentMode::TransferBackedCurrentBatch;
    }

    bool MoEExpertComputeStage::hasValidCompactLLEPTransferBinding() const
    {
        if (params_.routed_assignment_policy != RoutedExpertAssignmentPolicy::LeastLoadedResident ||
            !params_.prefill_llep_tp_ctx ||
            !params_.moe_runtime_table ||
            !params_.prefill_llep_transfer_slots ||
            params_.prefill_llep_transfer_slot_count == 0 ||
            params_.prefill_llep_payload_slot_bytes == 0 ||
            params_.prefill_llep_payload_slot_capacity == 0 ||
            !params_.prefill_llep_transfer_state ||
            params_.layer_idx < 0)
        {
            return false;
        }

        if (params_.prefill_llep_transfer_mode !=
            DeviceMoERebalanceTransferMode::CompactTransferSlots)
        {
            return false;
        }
        if (!validateDeviceMoERebalanceConfig(params_.prefill_llep_rebalance_config))
            return false;
        if (params_.prefill_llep_tp_ctx->degree() <= 1 ||
            params_.prefill_llep_tp_ctx->degree() !=
                static_cast<int>(params_.prefill_llep_rebalance_config.participant_count))
        {
            return false;
        }
        return params_.prefill_llep_rebalance_config.participant_id <
               params_.prefill_llep_rebalance_config.participant_count;
    }

    bool MoEExpertComputeStage::hasTransferBackedPrefillLLEP() const
    {
        return requestsTransferBackedCurrentBatchPrefillLLEP() &&
               params_.seq_len > 1 &&
               hasValidCompactLLEPTransferBinding();
    }

    bool MoEExpertComputeStage::executeTransferBackedPrefillLLEPMovement(
        IMoEKernel *kernel,
        DeviceMoERebalanceStatus **transfer_status_out,
        DeviceMoERebalanceApplyStatus **apply_status_out,
        DeviceMoERebalanceTransferState *transfer_state_override) const
    {
        if (!kernel)
            return false;
        if (transfer_status_out)
            *transfer_status_out = nullptr;
        if (apply_status_out)
            *apply_status_out = nullptr;
        if (!hasValidCompactLLEPTransferBinding())
        {
            throw std::logic_error(
                "MoE compact LLEP payload movement requires a valid graph-owned "
                "LocalTP transport binding");
        }
        if (!bound_workspace_)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill requires bound workspace buffers");
            return false;
        }

        void *compute_stream = gpuStream();
        if (!compute_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill requires an explicit compute stream");
            return false;
        }

        auto *runtime_layers = params_.moe_runtime_table->deviceLayerState(0);
        if (!runtime_layers || !moe_runtime_layer_)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill requires device runtime layers");
            return false;
        }

        const auto &config = params_.prefill_llep_rebalance_config;
        if (tracePrefillLLEPStatusEnabled())
        {
            LOG_INFO("[MoEExpertComputeStage] prefill LLEP transfer config"
                     << " device=" << params_.device_id.to_string()
                     << " layer=" << params_.layer_idx
                     << " participant_id=" << config.participant_id
                     << " participant_count=" << config.participant_count
                     << " root_participant=" << config.root_participant);
        }
        const uint32_t captured_payload_slots =
            std::min<uint32_t>(
                params_.prefill_llep_payload_slot_capacity,
                params_.prefill_llep_transfer_slot_count);
        const uint64_t configured_plan_capacity =
            deviceMoERebalanceCommandPlanCapacity(
                config,
                params_.prefill_llep_transfer_mode);
        const uint32_t plan_capacity =
            std::max<uint32_t>(
                1u,
                boundedU32(deviceMoEPrefillLLEPMergedPlanCapacity(
                    config,
                    configured_plan_capacity,
                    captured_payload_slots)));
        const uint32_t payload_slot_count =
            std::min<uint32_t>(captured_payload_slots, plan_capacity);
        if (payload_slot_count == 0)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill has zero payload slots");
            return false;
        }

        const std::string &workspace_name = params_.prefill_llep_workspace_name;
        auto *plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                workspace_name)));
        auto *plan_count = static_cast<uint32_t *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT,
                workspace_name)));
        auto *command_header = static_cast<DeviceMoERebalanceCommandBufferHeader *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                workspace_name)));
        auto *gathered_plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN,
                workspace_name)));
        auto *gathered_command_headers = static_cast<DeviceMoERebalanceCommandBufferHeader *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER,
                workspace_name)));
        auto *status = static_cast<DeviceMoERebalanceStatus *>(
            bound_workspace_->getBuffer(prefillLLEPLayerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_STATUS,
                workspace_name,
                params_.layer_idx)));
        auto *apply_status = static_cast<DeviceMoERebalanceApplyStatus *>(
            bound_workspace_->getBuffer(prefillLLEPLayerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                workspace_name,
                params_.layer_idx)));
        auto *local_source_descriptors = static_cast<DeviceMoEExpertDirectoryEntry *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_LOCAL_SOURCE_DESCRIPTORS,
                workspace_name)));
        auto *local_payload = static_cast<uint8_t *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                workspace_name)));
        auto *gathered_payload = static_cast<uint8_t *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                workspace_name)));

        if (!plan_entries || !plan_count || !command_header ||
            !gathered_plan_entries || !gathered_command_headers ||
            !status || !apply_status || !local_source_descriptors ||
            !local_payload || !gathered_payload)
        {
            LOG_ERROR("[MoEExpertComputeStage] Missing transfer-backed LLEP prefill workspace buffers"
                      << " plan_entries=" << static_cast<void *>(plan_entries)
                      << " plan_count=" << static_cast<void *>(plan_count)
                      << " command_header=" << static_cast<void *>(command_header)
                      << " gathered_plan_entries=" << static_cast<void *>(gathered_plan_entries)
                      << " gathered_command_headers=" << static_cast<void *>(gathered_command_headers)
                      << " status=" << static_cast<void *>(status)
                      << " apply_status=" << static_cast<void *>(apply_status)
                      << " local_source_descriptors=" << static_cast<void *>(local_source_descriptors)
                      << " local_payload=" << static_cast<void *>(local_payload)
                      << " gathered_payload=" << static_cast<void *>(gathered_payload));
            return false;
        }
        if (transfer_status_out)
            *transfer_status_out = status;
        if (apply_status_out)
            *apply_status_out = apply_status;

        auto *transfer_state =
            transfer_state_override
                ? transfer_state_override
                : params_.prefill_llep_transfer_state.get();
        if (!transfer_state ||
            !transfer_state->ensure(params_.device_id, workspace_name) ||
            !transfer_state->transferStream() ||
            !transfer_state->computeReadyEvent() ||
            !transfer_state->transferDoneEvent())
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill could not initialize transfer stream state");
            return false;
        }

        IWorkerGPUContext *gpu_ctx = nullptr;
        try
        {
            gpu_ctx = &GPUDeviceContextPool::instance().getContext(params_.device_id);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill could not resolve GPU context for "
                      << params_.device_id.to_string() << ": " << e.what());
            return false;
        }
        if (!gpu_ctx)
            return false;

        const MoEKernelLaunchContext compute_launch{
            .stream = compute_stream,
            .workspace = bound_workspace_,
        };
        if (!kernel->materializePrefillLeastLoadedTransferCommands(
                compute_launch,
                moe_runtime_layer_,
                plan_entries,
                plan_count,
                plan_capacity,
                command_header,
                status,
                config,
                payload_slot_count,
                static_cast<uint32_t>(params_.layer_idx),
                1))
        {
                LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill command materialization failed");
                return false;
        }
        IBackend *backend = getBackendFor(params_.device_id);
        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                compute_stream,
                "after_materialize",
                status,
                apply_status))
        {
            return false;
        }

        void *transfer_stream = transfer_state->transferStream();
        const MoEKernelLaunchContext transfer_launch{
            .stream = transfer_stream,
            .workspace = bound_workspace_,
        };
        if (!gpu_ctx->recordEventChecked(transfer_state->computeReadyEvent(), compute_stream) ||
            !gpu_ctx->waitEventChecked(transfer_state->computeReadyEvent(), transfer_stream))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill failed to queue compute-to-transfer dependency");
            return false;
        }

        static_assert((sizeof(DeviceMoERebalancePlanEntry) % sizeof(int32_t)) == 0);
        static_assert((sizeof(DeviceMoERebalanceCommandBufferHeader) % sizeof(int32_t)) == 0);
        const size_t plan_int32_words =
            (static_cast<size_t>(plan_capacity) * sizeof(DeviceMoERebalancePlanEntry)) /
            sizeof(int32_t);
        const size_t header_int32_words =
            sizeof(DeviceMoERebalanceCommandBufferHeader) / sizeof(int32_t);
        if (!params_.prefill_llep_tp_ctx->allgatherRawOnStream(
                plan_entries,
                gathered_plan_entries,
                plan_int32_words,
                CollectiveDataType::INT32,
                static_cast<int>(config.participant_id),
                transfer_stream,
                workspace_name.empty()
                    ? std::string("moe_prefill_llep_transfer_plan")
                    : workspace_name + "_plan"))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill plan allgather failed");
            return false;
        }
        if (!params_.prefill_llep_tp_ctx->allgatherRawOnStream(
                command_header,
                gathered_command_headers,
                header_int32_words,
                CollectiveDataType::INT32,
                static_cast<int>(config.participant_id),
                transfer_stream,
                workspace_name.empty()
                    ? std::string("moe_prefill_llep_transfer_header")
                    : workspace_name + "_header"))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill command-header allgather failed");
            return false;
        }
        if (!kernel->projectPrefillLeastLoadedDomainCommands(
                transfer_launch,
                gathered_plan_entries,
                gathered_command_headers,
                plan_capacity,
                plan_entries,
                plan_count,
                command_header,
                config,
                status,
                payload_slot_count,
                runtime_layers,
                params_.prefill_llep_transfer_slots,
                params_.prefill_llep_transfer_slot_count,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill domain command projection failed");
            return false;
        }
        if (!kernel->packDeviceRebalanceSourceDescriptors(
                transfer_launch,
                runtime_layers,
                plan_entries,
                command_header,
                plan_capacity,
                local_source_descriptors,
                config,
                nullptr,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill source descriptor pack failed");
            return false;
        }
        if (!kernel->packDeviceRebalanceCompactPayloads(
                transfer_launch,
                plan_entries,
                command_header,
                plan_capacity,
                local_source_descriptors,
                local_payload,
                payload_slot_count,
                params_.prefill_llep_payload_slot_bytes,
                config,
                apply_status,
                nullptr,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill payload pack failed");
            return false;
        }

        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                transfer_stream,
                "after_pack",
                status,
                apply_status))
        {
            return false;
        }

        const size_t local_payload_bytes =
            static_cast<size_t>(payload_slot_count) *
            static_cast<size_t>(params_.prefill_llep_payload_slot_bytes);
        if (!params_.prefill_llep_tp_ctx->allgatherRawOnStream(
                local_payload,
                gathered_payload,
                local_payload_bytes,
                CollectiveDataType::INT8,
                static_cast<int>(config.participant_id),
                transfer_stream,
                workspace_name.empty()
                    ? std::string("moe_prefill_llep_transfer_payload")
                    : workspace_name + "_payload"))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill payload allgather failed");
            return false;
        }

        if (!kernel->unpackDeviceRebalanceCollectivePayloads(
                transfer_launch,
                plan_entries,
                plan_count,
                plan_capacity,
                command_header,
                gathered_payload,
                payload_slot_count,
                params_.prefill_llep_payload_slot_bytes,
                params_.prefill_llep_transfer_slots,
                params_.prefill_llep_transfer_slot_count,
                config,
                apply_status,
                nullptr,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill payload unpack failed");
            return false;
        }

        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                transfer_stream,
                "after_unpack",
                status,
                apply_status))
        {
            return false;
        }

        if (!gpu_ctx->recordEventChecked(transfer_state->transferDoneEvent(), transfer_stream) ||
            !gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), compute_stream))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill failed to queue transfer-to-compute dependency");
            return false;
        }

        if (!kernel->applyDeviceRebalanceArrivals(
                compute_launch,
                runtime_layers,
                plan_entries,
                plan_count,
                plan_capacity,
                params_.prefill_llep_transfer_slots,
                params_.prefill_llep_transfer_slot_count,
                config,
                apply_status,
                command_header,
                params_.layer_idx))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill arrival apply failed");
            return false;
        }

        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                compute_stream,
                "after_apply",
                status,
                apply_status))
        {
            return false;
        }

        return true;
    }

    bool MoEExpertComputeStage::executeFixedTopologyGroupedPrefill(IMoEKernel *kernel, int max_tokens)
    {
        (void)max_tokens;
        if (!kernel)
            return false;

        const int seq_len = params_.seq_len;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int d_model = params_.d_model;
        const int intermediate = params_.expert_intermediate;
        const MoEKernelLaunchContext compute_launch{
            .stream = gpuStream(),
            .workspace = bound_workspace_,
        };
        if (!compute_launch.hasExplicitStream())
        {
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      "an explicit stage-owned compute stream is required");
            return false;
        }
        if (params_.use_runtime_prefill_grouping && !canUseRuntimePrefillGrouping())
        {
            auto *self = const_cast<MoEExpertComputeStage *>(this);
            if (!self->initializeMoERuntimeTableForGroupedPrefill())
            {
                LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                          "runtime prefill grouping requested but runtime scratch is unavailable");
                return false;
            }
        }
        const bool runtime_grouping = canUseRuntimePrefillGrouping();
        uint64_t fixed_topology_trace_sequence = 0;
        IBackend *assignment_trace_backend = nullptr;
        void *assignment_trace_stream = nullptr;
        auto trace_runtime_assignment = [&](const char *tag) -> bool {
            if (!tracePrefillAssignmentEnabled())
                return true;
            if (!params_.device_id.is_gpu())
                return true;
            if (!assignment_trace_backend)
                assignment_trace_backend = getBackendFor(params_.device_id);
            if (!assignment_trace_stream)
                assignment_trace_stream = gpuStream();
            return tracePrefillAssignmentRuntime(
                assignment_trace_backend,
                params_.device_id,
                assignment_trace_stream,
                tag,
                moe_runtime_layer_,
                params_.layer_idx,
                seq_len,
                num_experts,
                top_k);
        };

        // Async grouping (no D2H, no sync). Masked LocalTP overlays exclude
        // non-local experts from this participant's grouping scratch while
        // preserving the original routing tensors for rebalance histograms.
        const bool masked_grouping = usesPublishedFixedTopologyMaskGrouping();
        if (tracePrefillAssignmentEnabled() &&
            tracePrefillAssignmentLayerMatches(params_.layer_idx) &&
            tracePrefillAssignmentCheckpointMatches("before_group"))
        {
            fixed_topology_trace_sequence = nextPrefillAssignmentTraceSequence();
            const std::vector<int> fixed_expert_ids = fixedTopologyPrefillExpertIds();
            const std::vector<uint8_t> fixed_mask = fixedTopologyPrefillExpertMaskBytes();
            uint64_t runtime_placement_hash = 0;
            uint32_t runtime_active_bank = 0;
            uint32_t runtime_active_epoch = 0;
            uint32_t runtime_participant_id = 0;
            uint32_t runtime_participant_count = 0;
            bool runtime_state_available = false;
            if (params_.moe_runtime_table && params_.layer_idx >= 0)
            {
                try
                {
                    const auto &runtime_state =
                        params_.moe_runtime_table->hostLayerState(params_.layer_idx);
                    runtime_placement_hash =
                        hashPlacementBankSemantics(runtime_state, static_cast<uint32_t>(num_experts));
                    runtime_active_bank = runtime_state.active_bank;
                    runtime_active_epoch = runtime_state.active_epoch;
                    runtime_participant_id = runtime_state.participant_id;
                    runtime_participant_count = runtime_state.participant_count;
                    runtime_state_available = true;
                }
                catch (const std::exception &ex)
                {
                    LOG_ERROR("[MoEExpertComputeStage] fixed topology prefill assignment trace "
                              "failed to observe host runtime table"
                              << " device=" << params_.device_id.to_string()
                              << " layer=" << params_.layer_idx
                              << " error=" << ex.what());
                    return false;
                }
            }

            const auto enabled_mask_entries =
                static_cast<uint32_t>(std::count(fixed_mask.begin(), fixed_mask.end(), uint8_t{1}));
            const uint64_t prepared_desc_surface_hash =
                hashPreparedExpertDescriptorSurface(
                    fixed_expert_ids,
                    cached_gate_gemm_,
                    cached_up_gemm_,
                    cached_down_gemm_);
            LOG_INFO("[MoEExpertComputeStage] fixed topology prefill assignment trace"
                     << " seq=" << fixed_topology_trace_sequence
                     << " tag=before_group"
                     << " device=" << params_.device_id.to_string()
                     << " layer=" << params_.layer_idx
                     << " seq_len=" << seq_len
                     << " top_k=" << top_k
                     << " assignment_policy="
                     << routedExpertAssignmentPolicyToString(params_.routed_assignment_policy)
                     << " runtime_grouping=" << perfBool(runtime_grouping)
                     << " masked_grouping=" << perfBool(masked_grouping)
                     << " requested_runtime_grouping=" << perfBool(params_.use_runtime_prefill_grouping)
                     << " has_full_local_ownership=" << perfBool(hasFullLocalExpertOwnership())
                     << " expert_mask_all_enabled=" << perfBool(expertMaskAllEnabled())
                     << " participant_id=" << params_.my_socket_id
                     << " participant_count=" << params_.participant_count
                     << " local_start=" << params_.local_expert_start
                     << " local_count=" << params_.local_expert_count
                     << " fixed_expert_count=" << fixed_expert_ids.size()
                     << " enabled_mask_entries=" << enabled_mask_entries
                     << " fixed_expert_ids_hash=" << hashTraceVector(fixed_expert_ids)
                     << " fixed_mask_hash=" << hashTraceVector(fixed_mask)
                     << " prepared_desc_surface_hash=" << prepared_desc_surface_hash
                     << " expert_mask_hash=" << hashTraceBoolVector(params_.expert_mask)
                     << " replica_prefill_mask_hash="
                     << hashTraceBoolVector(params_.replica_set.prefill_mask)
                     << " replica_count=" << params_.replica_set.num_replicated
                     << " runtime_state_available=" << perfBool(runtime_state_available)
                     << " runtime_active_bank=" << runtime_active_bank
                     << " runtime_active_epoch=" << runtime_active_epoch
                     << " runtime_participant_id=" << runtime_participant_id
                     << " runtime_participant_count=" << runtime_participant_count
                     << " runtime_placement_hash=" << runtime_placement_hash
                     << " gateup_desc_table=" << grouped_gateup_desc_table_id_
                     << " down_desc_table=" << grouped_down_desc_table_id_);
        }
        bool groups_prepared = false;
        if (runtime_grouping)
        {
            const bool filter_runtime_grouping_to_local_experts =
                params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner;
            const bool publish_grouped_verifier_histogram =
                shouldPublishGroupedVerifierHistograms();
            const MoEGroupedHistogramUpdate initial_histogram_update =
                !publish_grouped_verifier_histogram
                    ? MoEGroupedHistogramUpdate::None
                    : (filter_runtime_grouping_to_local_experts
                           ? MoEGroupedHistogramUpdate::SelectedAndLocallyAssignedRoutes
                           : MoEGroupedHistogramUpdate::SelectedRoutes);
            groups_prepared = kernel->groupPrefillRoutes(
                moe_runtime_layer_,
                params_.routing_indices,
                params_.routing_weights,
                seq_len,
                seq_len,
                num_experts,
                top_k,
                filter_runtime_grouping_to_local_experts,
                initial_histogram_update);
            if (groups_prepared &&
                !trace_runtime_assignment("after_group"))
            {
                return false;
            }
            if (groups_prepared &&
                params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
            {
                const auto &runtime_state =
                    params_.moe_runtime_table->hostLayerState(params_.layer_idx);
                const auto &moe_env = debugEnv().moe_rebalance;
                const uint64_t routed_rows =
                    static_cast<uint64_t>(std::max(0, seq_len)) *
                    static_cast<uint64_t>(std::max(0, top_k));
                const uint64_t min_routed_rows =
                    moe_env.llep_prefill_min_routed_rows;
                if (!params_.force_grouped_verifier_prefill_for_decode &&
                    min_routed_rows > 0ULL &&
                    routed_rows < min_routed_rows)
                {
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "device_rebalance_llep_prefill_policy_skips",
                        1.0,
                        "prefill",
                        params_.device_id.toString(),
                        {{"stage", "moe_expert_grouped_prefill"},
                         {"reason", "insufficient_routed_rows"},
                         {"layer", std::to_string(params_.layer_idx)},
                         {"seq_len", std::to_string(seq_len)},
                         {"top_k", std::to_string(top_k)},
                         {"routed_rows", std::to_string(routed_rows)},
                         {"min_routed_rows", std::to_string(min_routed_rows)}});
                    return groups_prepared;
                }

                if (requestsTransferBackedCurrentBatchPrefillLLEP())
                {
                    if (!hasTransferBackedPrefillLLEP())
                    {
                        throw std::logic_error(
                            "TransferBackedCurrentBatch LLEP assignment requires "
                            "a valid compact graph-owned transport binding");
                    }

                    least_loaded_ep::LeastLoadedExpertAssignmentConfig llep_config;
                    llep_config.expert_count =
                        static_cast<uint32_t>(num_experts);
                    llep_config.participant_count =
                        runtime_state.participant_count > 0u
                            ? runtime_state.participant_count
                            : static_cast<uint32_t>(
                                  std::max(1, params_.participant_count));
                    /*
                     * Long-prefill LLEP uses the same policy knobs as captured
                     * decode maintenance. The transfer directory is the
                     * physical working set, so both new arrivals and the full
                     * non-owner assignment are bounded by its real capacity.
                     */
                    llep_config.alpha_numerator =
                        std::max<uint32_t>(
                            1u,
                            params_.prefill_llep_rebalance_config
                                .llep_alpha_numerator);
                    llep_config.alpha_denominator =
                        std::max<uint32_t>(
                            1u,
                            params_.prefill_llep_rebalance_config
                                .llep_alpha_denominator);
                    llep_config.lambda_numerator =
                        std::max<uint32_t>(
                            1u,
                            params_.prefill_llep_rebalance_config
                                .llep_lambda_numerator);
                    llep_config.lambda_denominator =
                        std::max<uint32_t>(
                            1u,
                            params_.prefill_llep_rebalance_config
                                .llep_lambda_denominator);
                    llep_config.enable_balanced_skip =
                        params_.prefill_llep_rebalance_config
                            .llep_enable_balanced_skip != 0u;
                    llep_config.min_spread_improvement =
                        static_cast<uint64_t>(
                            std::max(
                                0,
                                moe_env
                                    .device_rebalance_min_load_spread_improvement));
                    llep_config.min_spread_improvement_divisor =
                        static_cast<uint32_t>(
                            std::max(
                                0,
                                moe_env
                                    .device_rebalance_min_load_spread_improvement_divisor));
                    llep_config.min_spread_improvement_per_transfer =
                        static_cast<uint64_t>(
                            std::max(
                                0,
                                moe_env
                                    .device_rebalance_min_wave_spread_improvement_per_payload_slot));
                    llep_config.min_foreign_rows_per_transfer =
                        static_cast<uint64_t>(
                            std::max(
                                0,
                                moe_env
                                    .device_rebalance_min_foreign_rows_per_transfer));
                    const uint32_t physical_transfer_slot_capacity =
                        std::min<uint32_t>(
                            params_.prefill_llep_payload_slot_capacity,
                            params_.prefill_llep_transfer_slot_count);
                    llep_config.max_weight_transfers =
                        physical_transfer_slot_capacity;
                    llep_config.max_non_owner_experts_per_participant =
                        physical_transfer_slot_capacity;

                    groups_prepared =
                        kernel->planPrefillRoutesLeastLoadedCurrentBatch(
                            compute_launch,
                            moe_runtime_layer_,
                            seq_len,
                            seq_len,
                            num_experts,
                            top_k,
                            llep_config);
                    if (!groups_prepared)
                    {
                        LOG_ERROR(
                            "[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                            "planPrefillRoutesLeastLoadedCurrentBatch failed");
                        return false;
                    }
                    if (!trace_runtime_assignment("after_llep_plan"))
                        return false;

                    DeviceMoERebalanceStatus *transfer_status = nullptr;
                    DeviceMoERebalanceApplyStatus *apply_status = nullptr;
                    if (!executeTransferBackedPrefillLLEPMovement(kernel, &transfer_status, &apply_status))
                    {
                        LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                                  "transfer-backed LLEP movement failed");
                        return false;
                    }
                    if (!transfer_status || !apply_status)
                    {
                        LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                                  "transfer-backed LLEP movement did not publish transfer/apply status buffers");
                        return false;
                    }
                    /*
                     * This checkpoint observes the exact device-owned handoff
                     * between transfer publication and route assignment.  It
                     * remains completely absent from production execution
                     * unless LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE is enabled.
                     * In particular, it gives diagnostics a coherent view of
                     * the newly active placement bank without adding a host
                     * mirror, stream synchronization, or D2H read to the hot
                     * path.
                     */
                    if (!trace_runtime_assignment("after_llep_transfer_apply"))
                        return false;
                    groups_prepared =
                        kernel->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                            compute_launch,
                            moe_runtime_layer_,
                            seq_len,
                            seq_len,
                            num_experts,
                            top_k,
                            transfer_status,
                            apply_status);
                }
                else
                {
                    /*
                     * Resident-only is a first-class assignment policy, not a
                     * failed transfer plan. In particular, MTP verifier rows
                     * enter this branch even when the graph also owns compact
                     * transport resources for a preceding prefix-rehydration
                     * transaction. The backend planner considers only the
                     * active bank's resident masks, making a missing-payload
                     * destination structurally unrepresentable.
                     */
                    groups_prepared =
                        kernel->assignPrefillRoutesLeastLoadedResident(
                            compute_launch,
                            moe_runtime_layer_,
                            seq_len,
                            seq_len,
                            num_experts,
                            top_k);
                    if (groups_prepared)
                    {
                        PerfStatsCollector::addCounter(
                            "moe_rebalance",
                            "device_rebalance_llep_resident_assignment_calls",
                            1.0,
                            params_.force_grouped_verifier_prefill_for_decode
                                ? "verifier"
                                : "prefill",
                            params_.device_id.toString(),
                            {{"stage", "moe_expert_grouped_prefill"},
                             {"assignment", "resident_only"},
                             {"current_batch_transport", "none"},
                             {"layer", std::to_string(params_.layer_idx)},
                             {"seq_len", std::to_string(seq_len)},
                             {"top_k", std::to_string(top_k)}});
                    }
                }
                if (!groups_prepared)
                {
                    LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                              "least-loaded current-batch route assignment failed");
                    return false;
                }
                if (!trace_runtime_assignment("after_llep_assign"))
                    return false;
                groups_prepared = kernel->regroupPrefillRoutesFromRuntimeAssignments(
                    moe_runtime_layer_,
                    seq_len,
                    seq_len,
                    num_experts,
                    top_k,
                    publish_grouped_verifier_histogram
                        ? MoEGroupedHistogramUpdate::LocallyAssignedRoutes
                        : MoEGroupedHistogramUpdate::None);
                if (groups_prepared &&
                    !trace_runtime_assignment("after_llep_regroup"))
                {
                    return false;
                }
            }
        }
        else if (masked_grouping)
        {
            if (!publishFixedTopologyMaskBeforeCapture(kernel))
                return false;
            groups_prepared = kernel->prepareExpertGroupsAsyncUsingPublishedMask(
                params_.routing_indices, params_.routing_weights,
                seq_len, num_experts, top_k);
        }
        else
        {
            groups_prepared = kernel->prepareExpertGroupsAsync(
                params_.routing_indices, params_.routing_weights,
                seq_len, num_experts, top_k);
        }

        if (!groups_prepared)
        {
            const char *grouping_name = runtime_grouping
                                            ? "groupPrefillRoutes"
                                            : (masked_grouping ? "prepareExpertGroupsAsyncUsingPublishedMask"
                                                               : "prepareExpertGroupsAsync");
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      << grouping_name << " failed");
            return false;
        }

        // Execute the full grouped pipeline (5 kernel launches, zero sync)
        bool pipeline_ok = false;
        if (runtime_grouping)
        {
            if (!trace_runtime_assignment("before_runtime_pipeline"))
                return false;
            const auto &runtime_state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            pipeline_ok = kernel->executeGroupedPrefillPipelineFromRuntime(
                moe_runtime_layer_,
                runtime_state,
                params_.input,
                params_.output,
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k);
        }
        else
        {
            pipeline_ok = kernel->executeGroupedPrefillPipeline(
                params_.input, params_.output,
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                seq_len, d_model, intermediate,
                num_experts, top_k);
        }

        if (!pipeline_ok)
        {
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      "grouped prefill pipeline failed");
            return false;
        }

        /*
         * `after_pipeline` is the only checkpoint that can attribute an
         * asynchronous grouped GEMM fault to the layer that launched it.
         * The diagnostic is entirely absent unless assignment tracing is
         * enabled, and tracePrefillAssignmentRuntime() defers it during native
         * graph capture so production capture topology remains unchanged.
         */
        if (!trace_runtime_assignment("after_pipeline"))
            return false;

        if (fixed_topology_trace_sequence != 0 &&
            tracePrefillAssignmentCheckpointMatches("after_pipeline"))
        {
            LOG_INFO("[MoEExpertComputeStage] fixed topology prefill assignment trace"
                     << " seq=" << fixed_topology_trace_sequence
                     << " tag=after_pipeline"
                     << " device=" << params_.device_id.to_string()
                     << " layer=" << params_.layer_idx
                     << " seq_len=" << seq_len
                     << " runtime_grouping=" << perfBool(runtime_grouping)
                     << " masked_grouping=" << perfBool(masked_grouping)
                     << " gateup_desc_table=" << grouped_gateup_desc_table_id_
                     << " down_desc_table=" << grouped_down_desc_table_id_);
        }

        return true;
    }

    bool MoEExpertComputeStage::isDeviceRoutedDecodeGraphCapturable() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
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
               runtimeTableHasActiveGroupedDecodeBank();
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

        // Require either full local expert ownership or a fixed local expert
        // mask whose non-local routes can be dropped before grouping.
        if (!((hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
              hasFixedTopologyPrefillExpertMask()))
            return false;
        if (forced_decode_replay && params_.replica_set.num_replicated != 0)
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

        // Must have prepared GEMM engines for every expert this participant
        // can actually compute. Descriptor tables tolerate holes for masked
        // experts because masked grouping never schedules those rows.
        return hasPreparedExpertGemmEnginesForExperts(fixedTopologyPrefillExpertIds());
    }

    bool MoEExpertComputeStage::isFixedTopologyPrefillGraphCapturable() const
    {
        if (!supportsFixedTopologyPrefillGraphCapturePreflight())
            return false;

        // MoE kernel must exist and have pre-allocated grouping + scratch
        if (!moe_kernel_)
            return false;
        if (params_.use_runtime_prefill_grouping && !canUseRuntimePrefillGrouping())
            return false;
        if (usesPublishedFixedTopologyMaskGrouping() &&
            fixed_topology_mask_publication_state_ !=
                FixedTopologyMaskPublicationState::Published)
        {
            return false;
        }

        return true;
    }

    bool MoEExpertComputeStage::expertComputesLocally(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= params_.num_experts)
            return false;

        if (!params_.expert_mask.empty())
        {
            return params_.expert_mask.size() == static_cast<size_t>(params_.num_experts) &&
                   params_.expert_mask[static_cast<size_t>(expert_id)];
        }

        const int local_count =
            params_.local_expert_count < 0 ? params_.num_experts : params_.local_expert_count;
        return expert_id >= params_.local_expert_start &&
               expert_id < params_.local_expert_start + local_count;
    }

    const std::vector<bool> *MoEExpertComputeStage::fixedTopologyPrefillMask() const
    {
        if (params_.num_experts <= 0)
            return nullptr;
        if (params_.replica_set.num_replicated > 0 &&
            params_.replica_set.prefill_mask.size() == static_cast<size_t>(params_.num_experts))
        {
            return &params_.replica_set.prefill_mask;
        }
        if (params_.expert_mask.size() == static_cast<size_t>(params_.num_experts))
            return &params_.expert_mask;
        return nullptr;
    }

    bool MoEExpertComputeStage::hasFixedTopologyPrefillExpertMask() const
    {
        const std::vector<bool> *mask = fixedTopologyPrefillMask();
        if (mask)
        {
            return std::any_of(
                mask->begin(),
                mask->end(),
                [](bool enabled)
                {
                    return enabled;
                });
        }

        /*
         * Static contiguous ownership is already a complete fixed topology.
         * Synthesize its device mask instead of requiring graph construction to
         * duplicate the same range into expert_mask.
         */
        const int local_count =
            params_.local_expert_count < 0
                ? params_.num_experts
                : params_.local_expert_count;
        return params_.local_expert_start >= 0 &&
               local_count > 0 &&
               params_.local_expert_start + local_count <= params_.num_experts &&
               !(params_.local_expert_start == 0 &&
                 local_count == params_.num_experts);
    }

    bool MoEExpertComputeStage::usesMaskedFixedTopologyPrefill() const
    {
        return hasFixedTopologyPrefillExpertMask() &&
               !(hasFullLocalExpertOwnership() && expertMaskAllEnabled());
    }

    bool MoEExpertComputeStage::usesPublishedFixedTopologyMaskGrouping() const
    {
        /*
         * `use_runtime_prefill_grouping` is a required route, not a preference:
         * executeFixedTopologyGroupedPrefill() hard-fails if its persistent
         * runtime table or scratch is unavailable.  Therefore it is both safe
         * and necessary to choose the placement source from the requested
         * policy rather than from transient warmup readiness.  Falling through
         * to the fixed mask when runtime initialization fails would hide a
         * broken device-resident LLEP contract.
         */
        return !params_.use_runtime_prefill_grouping &&
               usesMaskedFixedTopologyPrefill();
    }

    std::vector<int> MoEExpertComputeStage::fixedTopologyPrefillExpertIds() const
    {
        std::vector<int> expert_ids;
        if (const std::vector<bool> *mask = fixedTopologyPrefillMask())
        {
            expert_ids.reserve(static_cast<size_t>(params_.num_experts));
            for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
            {
                if ((*mask)[static_cast<size_t>(expert_id)])
                    expert_ids.push_back(expert_id);
            }
            return expert_ids;
        }

        const int local_count =
            params_.local_expert_count < 0
                ? params_.num_experts
                : params_.local_expert_count;
        const int first =
            local_count == params_.num_experts ? 0 : params_.local_expert_start;
        expert_ids.resize(static_cast<size_t>(local_count));
        std::iota(expert_ids.begin(), expert_ids.end(), first);
        return expert_ids;
    }

    std::vector<uint8_t> MoEExpertComputeStage::fixedTopologyPrefillExpertMaskBytes() const
    {
        std::vector<uint8_t> mask(static_cast<size_t>(params_.num_experts), 0u);
        const std::vector<bool> *prefill_mask = fixedTopologyPrefillMask();
        if (!prefill_mask)
        {
            const int local_count =
                params_.local_expert_count < 0
                    ? params_.num_experts
                    : params_.local_expert_count;
            const int first =
                local_count == params_.num_experts ? 0 : params_.local_expert_start;
            const int last = std::min(params_.num_experts, first + local_count);
            for (int expert_id = std::max(0, first);
                 expert_id < last;
                 ++expert_id)
            {
                mask[static_cast<size_t>(expert_id)] = 1u;
            }
            return mask;
        }

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            mask[static_cast<size_t>(expert_id)] =
                (*prefill_mask)[static_cast<size_t>(expert_id)] ? 1u : 0u;
        }
        return mask;
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

    bool MoEExpertComputeStage::hasPreparedExpertGemmEnginesForExperts(
        const std::vector<int> &expert_ids) const
    {
        if (expert_ids.empty())
            return false;
        if (params_.prepared_gate_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_up_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_down_gemm.size() != static_cast<size_t>(params_.num_experts))
        {
            return false;
        }

        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                return false;
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
        if (params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx))
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

    bool MoEExpertComputeStage::isCollectiveStage() const
    {
        /*
         * Report the graph-build policy, not whether every pointer in the
         * binding happens to validate. An invalid transport request must remain
         * a collective-stage construction error; classifying it as ordinary
         * compute would permit segmented or unordered execution before the
         * invariant is diagnosed.
         */
        return requestsTransferBackedCurrentBatchPrefillLLEP() ||
               params_.prefix_runtime_device_rehydration;
    }

    bool MoEExpertComputeStage::isGraphCapturable() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
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

    std::string MoEExpertComputeStage::graphCaptureReadinessDebugString() const
    {
        const char *mask_publication = "not_required";
        switch (fixed_topology_mask_publication_state_)
        {
        case FixedTopologyMaskPublicationState::NotRequired:
            break;
        case FixedTopologyMaskPublicationState::NeedsPublication:
            mask_publication = "needs_publication";
            break;
        case FixedTopologyMaskPublicationState::Published:
            mask_publication = "published";
            break;
        }

        const bool runtime_prefill_requested =
            params_.use_runtime_prefill_grouping;
        const bool runtime_prefill_ready =
            canUseRuntimePrefillGrouping();
        const bool fixed_preflight =
            supportsFixedTopologyPrefillGraphCapturePreflight();
        const bool runtime_decode_ready =
            isDeviceRoutedDecodeGraphCapturable();
        const std::vector<int> fixed_experts =
            fixedTopologyPrefillExpertIds();
        const bool fixed_engines_ready =
            hasPreparedExpertGemmEnginesForExperts(fixed_experts);

        std::ostringstream out;
        out << "device=" << params_.device_id.to_string()
            << " layer=" << params_.layer_idx
            << " seq_len=" << params_.seq_len
            << " forced_grouped_verifier="
            << perfBool(params_.force_grouped_verifier_prefill_for_decode)
            << " route="
            << (runtime_prefill_requested
                    ? "runtime_table_prefill"
                    : (params_.seq_len == 1 &&
                               !params_.force_grouped_verifier_prefill_for_decode
                           ? "runtime_table_decode"
                           : "fixed_topology_prefill"))
            << " backend_grouped="
            << perfBool(supportsGroupedPrefillGraphCaptureBackend(params_.device_id))
            << " kernel=" << perfBool(moe_kernel_ != nullptr)
            << " fixed_preflight=" << perfBool(fixed_preflight)
            << " fixed_engines=" << perfBool(fixed_engines_ready)
            << " fixed_mask_consumed="
            << perfBool(usesPublishedFixedTopologyMaskGrouping())
            << " fixed_mask_publication=" << mask_publication
            << " runtime_prefill_requested="
            << perfBool(runtime_prefill_requested)
            << " runtime_prefill_ready=" << perfBool(runtime_prefill_ready)
            << " runtime_layer=" << perfBool(moe_runtime_layer_ != nullptr)
            << " runtime_table=" << perfBool(params_.moe_runtime_table != nullptr)
            << " runtime_decode_initialized="
            << perfBool(moe_runtime_table_initialized_)
            << " runtime_decode_warmed="
            << perfBool(runtime_grouped_decode_warmed_)
            << " runtime_decode_ready=" << perfBool(runtime_decode_ready)
            << " gateup_desc=" << grouped_gateup_desc_table_id_
            << " down_desc=" << grouped_down_desc_table_id_;
        return out.str();
    }

    bool MoEExpertComputeStage::supportsWarmupDependentGraphCapture() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        bool decode_supported = false;
        decode_supported =
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
            params_.layer_idx >= 0;

        return decode_supported || supportsFixedTopologyPrefillGraphCapturePreflight();
#endif
    }

    bool MoEExpertComputeStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        return supportsFixedTopologyPrefillGraphCapturePreflight();
    }

    bool MoEExpertComputeStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsLazyPrefillGraphCapturePreflight();
    }

    bool MoEExpertComputeStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Graph launch preparation requires an explicit stream"
                      << " device=" << params_.device_id.to_string()
                      << " layer=" << params_.layer_idx);
            return false;
        }
        setGPUStream(stream);
        const bool prepare_current_batch =
            requestsTransferBackedCurrentBatchPrefillLLEP();
        const bool prepare_prefix_rehydration =
            params_.prefix_runtime_device_rehydration;
        if (!prepare_current_batch && !prepare_prefix_rehydration)
            return true;
        if (!hasValidCompactLLEPTransferBinding())
        {
            throw std::logic_error(
                "MoE graph launch requested compact LLEP transport without a "
                "valid graph-owned LocalTP binding");
        }
        if (prepare_current_batch &&
            !params_.prefill_llep_transfer_state->prepareForCapture(
                params_.device_id,
                params_.prefill_llep_workspace_name,
                stream))
        {
            throw std::runtime_error(
                "Transfer-backed current-batch LLEP graph preparation could "
                "not publish its rolling-lane event ordering");
        }
        if (prepare_prefix_rehydration &&
            (!params_.prefix_runtime_rehydration_transfer_state ||
             !params_.prefix_runtime_rehydration_transfer_state->
                 prepareForCapture(
                     params_.device_id,
                     params_.prefill_llep_workspace_name,
                     stream)))
        {
            throw std::runtime_error(
                "Prefix-runtime rehydration graph preparation could not "
                "establish distinct event ownership");
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_llep_prefill_precapture_lane_event_fence",
            1.0,
            "prefill",
            params_.device_id.to_string(),
            {{"stage", "moe_expert_grouped_prefill"},
             {"layer", std::to_string(params_.layer_idx)},
             {"current_batch",
              prepare_current_batch ? "transfer_backed" : "resident_only"},
             {"prefix_rehydration",
              prepare_prefix_rehydration ? "enabled" : "disabled"},
             {"workspace", params_.prefill_llep_workspace_name}});
        return true;
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
        contract.inputs.reserve(3);
        contract.outputs.reserve(1);

        contract.addInput(params_.input_buffer_id);
        contract.addInput(params_.routing_indices_buffer_id);
        contract.addInput(params_.routing_weights_buffer_id);
        if (params_.output_registered_in_arena)
            contract.addOutput(params_.output_buffer_id);

        if (params_.combine_shared_expert_in_verifier)
        {
            if (params_.shared_gate_w)
            {
                contract.addPreparedWeight(
                    params_.shared_gate_w,
                    params_.prepared_store,
                    params_.prepared_shared_ref_gate.value_or(PreparedWeightRef{}));
            }
            if (params_.shared_up_w)
            {
                contract.addPreparedWeight(
                    params_.shared_up_w,
                    params_.prepared_store,
                    params_.prepared_shared_ref_up.value_or(PreparedWeightRef{}));
            }
            if (params_.shared_down_w)
            {
                contract.addPreparedWeight(
                    params_.shared_down_w,
                    params_.prepared_store,
                    params_.prepared_shared_ref_down.value_or(PreparedWeightRef{}));
            }
            if (params_.shared_gate_inp)
                contract.addWeight(params_.shared_gate_inp);
        }

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
        info.addScalarInt("participant_id", params_.my_socket_id);
        info.addScalarInt("participant_count", params_.participant_count);
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
            combined.merge(MoEWorkspaceBuffers::cudaMoE(
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
        const bool needs_llep_transport_workspace =
            requestsTransferBackedCurrentBatchPrefillLLEP() ||
            params_.prefix_runtime_device_rehydration;
        if (needs_llep_transport_workspace &&
            !hasValidCompactLLEPTransferBinding())
        {
            throw std::logic_error(
                "MoE compact LLEP transport workspace requested without a "
                "valid graph-owned LocalTP binding");
        }
        if (needs_llep_transport_workspace)
        {
            const auto &config = params_.prefill_llep_rebalance_config;
            const uint32_t captured_payload_slots =
                std::min<uint32_t>(
                    params_.prefill_llep_payload_slot_capacity,
                    params_.prefill_llep_transfer_slot_count);
            const uint32_t plan_capacity =
                std::max<uint32_t>(
                    1u,
                    boundedU32(deviceMoEPrefillLLEPMergedPlanCapacity(
                        config,
                        deviceMoERebalanceCommandPlanCapacity(
                            config,
                            params_.prefill_llep_transfer_mode),
                        captured_payload_slots)));
            const uint32_t payload_slot_count =
                std::min<uint32_t>(captured_payload_slots, plan_capacity);
            const size_t local_payload_bytes =
                static_cast<size_t>(payload_slot_count) *
                static_cast<size_t>(params_.prefill_llep_payload_slot_bytes);
            const size_t participant_count =
                static_cast<size_t>(std::max<uint32_t>(1u, config.participant_count));
            const std::string &workspace_name = params_.prefill_llep_workspace_name;

            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_TRANSFER_PLAN, workspace_name),
                static_cast<size_t>(plan_capacity) * sizeof(DeviceMoERebalancePlanEntry),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT, workspace_name),
                sizeof(uint32_t),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_COMMAND_HEADER, workspace_name),
                sizeof(DeviceMoERebalanceCommandBufferHeader),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN, workspace_name),
                participant_count * static_cast<size_t>(plan_capacity) *
                    sizeof(DeviceMoERebalancePlanEntry),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER, workspace_name),
                participant_count * sizeof(DeviceMoERebalanceCommandBufferHeader),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPLayerPublicationBufferName(
                    MoEDeviceRebalanceStage::WS_STATUS,
                    workspace_name,
                    params_.layer_idx),
                sizeof(DeviceMoERebalanceStatus),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPLayerPublicationBufferName(
                    MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                    workspace_name,
                    params_.layer_idx),
                sizeof(DeviceMoERebalanceApplyStatus),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_LOCAL_SOURCE_DESCRIPTORS, workspace_name),
                participant_count * static_cast<size_t>(plan_capacity) *
                    sizeof(DeviceMoEExpertDirectoryEntry),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD, workspace_name),
                local_payload_bytes,
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD, workspace_name),
                participant_count * local_payload_bytes,
                256,
                true});
        }
        return combined;
    }

    void MoEExpertComputeStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        if (workspace != bound_workspace_)
        {
            runtime_grouped_decode_warmed_ = false;
            invalidateFixedTopologyMaskPublication();
        }

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
        invalidateFixedTopologyMaskPublication();
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
        scratch_gate_ = dynamic_cast<FP32Tensor *>(params_.gate_scratch);
        scratch_up_ = dynamic_cast<FP32Tensor *>(params_.up_scratch);

        /*
         * Standalone CPU stages have no BufferArena executor. Materialize their
         * host scratch once at construction. GPU stages are intentionally not
         * given an analogous path: their stable device buffers are a mandatory
         * graph-construction contract.
         */
        if (params_.device_id.is_cpu() &&
            params_.seq_len > 0 &&
            params_.intermediate > 0)
        {
            if (!scratch_gate_)
            {
                owned_cpu_scratch_gate_ = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{
                        static_cast<size_t>(params_.seq_len),
                        static_cast<size_t>(params_.intermediate)});
                scratch_gate_ = owned_cpu_scratch_gate_.get();
            }
            if (!scratch_up_)
            {
                owned_cpu_scratch_up_ = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{
                        static_cast<size_t>(params_.seq_len),
                        static_cast<size_t>(params_.intermediate)});
                scratch_up_ = owned_cpu_scratch_up_.get();
            }
        }

        if (scratch_gate_ && scratch_up_ &&
            scratch_gate_->shape().size() >= 2 &&
            scratch_up_->shape().size() >= 2)
        {
            scratch_seq_len_ = static_cast<int>(std::min(
                scratch_gate_->shape()[0],
                scratch_up_->shape()[0]));
        }
    }

    bool SharedExpertFFNStage::validatePlannedScratch(
        int rows,
        int intermediate) const
    {
        if (!scratch_gate_ || !scratch_up_)
        {
            LOG_ERROR(
                "[SharedExpertFFNStage] Missing graph-owned FP32 gate/up scratch; "
                "GPU execution cannot allocate scratch dynamically");
            return false;
        }

        const auto has_capacity = [rows, intermediate](
                                      const FP32Tensor *tensor)
        {
            const auto &shape = tensor->shape();
            return shape.size() >= 2 &&
                   shape[0] >= static_cast<size_t>(rows) &&
                   shape[1] >= static_cast<size_t>(intermediate);
        };
        if (!has_capacity(scratch_gate_) || !has_capacity(scratch_up_))
        {
            LOG_ERROR(
                "[SharedExpertFFNStage] Graph-owned gate/up scratch is undersized: "
                "required_rows="
                << rows
                << " required_intermediate=" << intermediate
                << " gate_shape="
                << (scratch_gate_->shape().size() >= 2
                        ? std::to_string(scratch_gate_->shape()[0]) + "x" +
                              std::to_string(scratch_gate_->shape()[1])
                        : "invalid")
                << " up_shape="
                << (scratch_up_->shape().size() >= 2
                        ? std::to_string(scratch_up_->shape()[0]) + "x" +
                              std::to_string(scratch_up_->shape()[1])
                        : "invalid"));
            return false;
        }

        if (params_.device_id.is_gpu() &&
            (!scratch_gate_->gpu_data_ptr() ||
             !scratch_up_->gpu_data_ptr()))
        {
            LOG_ERROR(
                "[SharedExpertFFNStage] Graph executor did not establish "
                "device storage for arena-owned gate/up scratch");
            return false;
        }
        return true;
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
            /*
             * Workspace planning resolves prepared engines before the graph
             * scheduler binds a GPU stream. Propagate a stream only when one
             * already exists; execute() performs the mandatory propagation
             * immediately before launching any kernel.
             */
            if (!params_.device_id.is_gpu() || hasGPUStream())
                bindStageStream(gemm);
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
               /*
                * The verifier-row kernels own the complete small-M contract.
                * M=1 still represents an MTP verifier bucket whose counters and
                * publication math must stay aligned with M=2..4 instead of
                * silently taking the ordinary decode shortcut.
                */
               params_.seq_len >= 1 &&
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
               params_.seq_len >= 1;
    }

    bool SharedExpertFFNStage::usesDecodeEquivalentVerifierPrefillForTesting() const
    {
        return shouldUseDecodeEquivalentVerifierPrefill();
    }

    bool SharedExpertFFNStage::usesCPUDecodeEquivalentVerifierPrefillForTesting() const
    {
        return usesDecodeEquivalentVerifierPrefillForTesting();
    }

    bool SharedExpertFFNStage::shouldUseGroupedDecodeRoute() const
    {
        return params_.device_id.is_gpu() &&
               params_.seq_len == 1 &&
               !shouldUseDecodeEquivalentVerifierPrefill() &&
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
             {"route", params_.seq_len == 1
                           ? "single_row_decode"
                           : ((params_.device_id.is_cuda() ||
                               params_.device_id.is_rocm())
                                  ? "gemm_grouped_verifier_hooks"
                                  : "cpu_grouped_verifier_hooks")},
             {"seq_len", std::to_string(params_.seq_len)}});

        if (params_.seq_len == 1)
        {
            /*
             * M=1 has no grouped dimension to exploit. Execute the ordinary
             * production one-row GEMM decode contract once.  Verifier
             * publication uses this row as the byte-for-byte oracle for grouped
             * M=2..4 rows, so the ordinary shared-expert grouped-table shortcut
             * is deliberately suppressed here even when normal decode would use
             * it.  Otherwise the grouped side and the serial witness can choose
             * different reduction kernels for the same TP-sharded shared expert
             * projection, which creates tiny layer-0 drift that later flips MoE
             * routes.
             */
            struct ScopedRuntimeDecodeSharedExpert
            {
                SharedExpertFFNStage::Params &params;
                bool force_grouped_verifier_prefill_for_decode;
                bool force_decode_equivalent_verifier_prefill;
                bool disable_grouped_decode_shortcut;
                int seq_len;

                ~ScopedRuntimeDecodeSharedExpert()
                {
                    params.force_grouped_verifier_prefill_for_decode =
                        force_grouped_verifier_prefill_for_decode;
                    params.force_decode_equivalent_verifier_prefill =
                        force_decode_equivalent_verifier_prefill;
                    params.disable_grouped_decode_shortcut =
                        disable_grouped_decode_shortcut;
                    params.seq_len = seq_len;
                }
            } restore{
                params_,
                params_.force_grouped_verifier_prefill_for_decode,
                params_.force_decode_equivalent_verifier_prefill,
                params_.disable_grouped_decode_shortcut,
                params_.seq_len};

            params_.seq_len = 1;
            params_.force_decode_equivalent_verifier_prefill = false;
            params_.force_grouped_verifier_prefill_for_decode = false;
            params_.disable_grouped_decode_shortcut = true;
            return execute(ctx);
        }

        if (!validatePlannedScratch(params_.seq_len, intermediate))
            return false;

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_, intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_, intermediate, nullptr, "shared_up"}};
        auto verifier_scopes = beginVerifierDecodeEquivalentScopes(
            {cached_gate_gemm_, cached_up_gemm_, cached_down_gemm_});
        if (!cached_gate_gemm_->multiply_fused_verifier_rows_decode_equivalent(
                params_.input,
                projections,
                params_.seq_len,
                d_model,
                nullptr,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Grouped verifier shared gate/up projection failed"
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }
        if (params_.device_id.is_gpu())
        {
            const StageGPUExecution execution = gpuExecution();
            for (const auto &projection : projections)
                execution.publish(projection.output);
        }

        if (!cached_down_gemm_->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                scratch_gate_,
                scratch_up_,
                params_.output,
                params_.seq_len,
                d_model,
                intermediate,
                1.0f,
                0.0f,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Grouped verifier shared SwiGLU/down projection failed"
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }
        if (params_.device_id.is_gpu())
            gpuExecution().publish(params_.output);

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_shared_grouped_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(params_.seq_len),
            "verifier",
             params_.device_id.toString(),
            {{"stage", "shared_expert"},
             {"route", (params_.device_id.is_cuda() ||
                         params_.device_id.is_rocm())
                           ? "gemm_grouped_verifier_hooks"
                           : "cpu_grouped_verifier_hooks"}});
        return true;
    }

    bool SharedExpertFFNStage::tryGroupedVerifierPrefill(
        IMoEKernel *kernel, int d_model, int intermediate) const
    {
        (void)kernel;
        if (!shouldUseGroupedVerifierPrefillRoute())
            return false;

        if (params_.device_id.is_gpu())
            (void)requireGPUStream();

        /*
         * The serial CUDA shared-expert decode route uses the MoE grouped table
         * kernels even though there is only one always-active shared expert.
         * Verifier rows therefore need to be grouped through the same table
         * descriptor family, reduction order, and split-K partial layout rather
         * than through the generic dense GEMM verifier hooks.  This keeps the
         * implementation genuinely grouped for M=2..4 while making its math
         * target the exact path that ordinary one-token decode exercises.
         */
        if (!ensureSharedGroupedGateUpDescriptorTable(kernel, d_model, intermediate) ||
            !ensureSharedGroupedDownDescriptorTable(kernel, d_model, intermediate))
        {
            LOG_ERROR("[SharedExpertFFNStage] Failed to prepare shared expert grouped descriptor tables"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.seq_len
                      << " d_model=" << d_model
                      << " intermediate=" << intermediate);
            return false;
        }
        if (!kernel->prepareSharedExpertPrefillGroup(params_.seq_len))
        {
            LOG_ERROR("[SharedExpertFFNStage] Failed to prepare shared expert verifier group"
                      << " device=" << params_.device_id.to_string()
                      << " m=" << params_.seq_len);
            return false;
        }
        if (!kernel->executeGroupedPrefillPipeline(
                params_.input,
                params_.output,
                shared_grouped_gateup_desc_table_id_,
                shared_grouped_down_desc_table_id_,
                params_.seq_len,
                d_model,
                intermediate,
                /*num_experts=*/1,
                /*top_k=*/1))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared expert grouped verifier pipeline failed"
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
             {"route", "grouped_table_prefill"}});
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
        ITensor *gate_outputs[1] = {scratch_gate_};
        ITensor *up_outputs[1] = {scratch_up_};
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

        ITensor *gate_tensors[1] = {scratch_gate_};
        ITensor *up_tensors[1] = {scratch_up_};
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
        bindStageStream(cached_gate_gemm_);
        bindStageStream(cached_up_gemm_);
        bindStageStream(cached_down_gemm_);

        if (!validatePlannedScratch(seq_len, intermediate))
            return false;

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
            gpuExecution().publish(params_.output);
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
            gpuExecution().publish(params_.output);
            return true;
        }

        // Gate+Up projections via fused multi-projection (quantizes input once)
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {cached_gate_gemm_, scratch_gate_, intermediate, nullptr, "shared_gate"},
            {cached_up_gemm_, scratch_up_, intermediate, nullptr, "shared_up"}};
        if (!cached_gate_gemm_->multiply_fused_tensor(
                params_.input, projections,
                seq_len, d_model,
                nullptr, getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared gate/up projection failed");
            return false;
        }
        if (params_.device_id.is_gpu())
        {
            const StageGPUExecution execution = gpuExecution();
            for (const auto &projection : projections)
                execution.publish(projection.output);
        }

        // Mandatory fused SwiGLU/down grouped implementation.
        if (!fusedSwigluDown(
                *this,
                scratch_gate_, scratch_up_, params_.output,
                cached_down_gemm_, seq_len, d_model, intermediate,
                getWorkspace()))
        {
            LOG_ERROR("[SharedExpertFFNStage] Shared SwiGLU/down projection failed");
            return false;
        }

        return true;
    }

    IMoEKernel *SharedExpertFFNStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
        {
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        auto *kernel = bindStageStream(moe_kernel_);
        if (bound_workspace_)
        {
            // The shared expert can be the first MoE stage in focused tests
            // and verifier-only paths. Bind its private workspace explicitly
            // before any pointer array or descriptor table is materialized.
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
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (!params_.device_id.is_gpu())
            return false;

        /*
         * Forced verifier replay is a small-M grouped-prefill path for M=1..4.
         * Do not test grouped_decode_warmed_ here: that flag is only written by
         * tryGroupedDecode(), while verifier replay deliberately executes
         * tryGroupedVerifierPrefill() and owns its own warmup contract.
         */
        if (shouldUseGroupedVerifierPrefillRoute())
        {
            if (!supportsGroupedPrefillGraphCaptureBackend(params_.device_id))
                return false;
            return scratch_seq_len_ >= params_.seq_len && moe_kernel_ != nullptr;
        }

        /*
         * Normal grouped decode bakes device-side pointer arrays into CUDA graph
         * replay.  A cold stage can be warmed by Phase 1, but Phase 2 capture
         * must not start until that warmup has populated arrays for this stage's
         * current scratch buffers and workspace binding.
         */
        if (params_.seq_len == 1)
        {
            if (shouldUseGroupedDecodeRoute())
            {
                if (!supportsSharedExpertGroupedDecodeGraphCaptureBackend(params_.device_id))
                    return false;
                return grouped_decode_warmed_ && scratch_seq_len_ >= 1 && moe_kernel_ != nullptr;
            }
            return scratch_seq_len_ >= 1;
        }

        // Prefill capture is safe only after warmup has allocated scratch and
        // resolved the MoE kernel used by fused SwiGLU/down.
        if (!supportsGroupedPrefillGraphCaptureBackend(params_.device_id))
            return false;
        return scratch_seq_len_ >= params_.seq_len && moe_kernel_ != nullptr;
#endif
    }

    bool SharedExpertFFNStage::supportsWarmupDependentGraphCapture() const
    {
        if (supportsPaddedPrefillGraphCapturePreflight())
            return true;

        return shouldUseGroupedDecodeRoute() &&
               supportsSharedExpertGroupedDecodeGraphCaptureBackend(params_.device_id) &&
               params_.d_model > 0 &&
               params_.intermediate > 0 &&
               params_.input &&
               params_.gate_w &&
               params_.up_w &&
               params_.down_w &&
               params_.output;
    }

    std::string SharedExpertFFNStage::graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        const bool prefill_graph_backend =
            supportsGroupedPrefillGraphCaptureBackend(params_.device_id);
        const bool decode_graph_backend =
            supportsSharedExpertGroupedDecodeGraphCaptureBackend(params_.device_id);
        const bool verifier_route = shouldUseGroupedVerifierPrefillRoute();
        const bool decode_equivalent_route = shouldUseDecodeEquivalentVerifierPrefill();
        const bool grouped_decode_route = shouldUseGroupedDecodeRoute();

        out << "device=" << params_.device_id.toString()
            << " seq_len=" << params_.seq_len
            << " d_model=" << params_.d_model
            << " intermediate=" << params_.intermediate
            << " prefill_graph_backend=" << perfBool(prefill_graph_backend)
            << " decode_graph_backend=" << perfBool(decode_graph_backend)
            << " verifier_route=" << perfBool(verifier_route)
            << " decode_equivalent_route=" << perfBool(decode_equivalent_route)
            << " grouped_decode_route=" << perfBool(grouped_decode_route)
            << " scratch_seq_len=" << scratch_seq_len_
            << " scratch_ready=" << perfBool(scratch_seq_len_ >= std::max(1, params_.seq_len))
            << " moe_kernel=" << perfBool(moe_kernel_ != nullptr)
            << " grouped_decode_warmed=" << perfBool(grouped_decode_warmed_)
            << " workspace_bound=" << perfBool(bound_workspace_ != nullptr)
            << " gate_gemm=" << perfBool(cached_gate_gemm_ != nullptr)
            << " up_gemm=" << perfBool(cached_up_gemm_ != nullptr)
            << " down_gemm=" << perfBool(cached_down_gemm_ != nullptr)
            << " input=" << perfBool(params_.input != nullptr)
            << " gate_w=" << perfBool(params_.gate_w != nullptr)
            << " up_w=" << perfBool(params_.up_w != nullptr)
            << " down_w=" << perfBool(params_.down_w != nullptr)
            << " output=" << perfBool(params_.output != nullptr);
        return out.str();
    }

    bool SharedExpertFFNStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
               params_.seq_len > 1 &&
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
        if (params_.force_grouped_verifier_prefill_for_decode)
        {
            return supportsGroupedPrefillGraphCaptureBackend(params_.device_id) &&
                   params_.seq_len >= 1 &&
                   params_.d_model > 0 &&
                   params_.intermediate > 0 &&
                   params_.input &&
                   params_.gate_w &&
                   params_.up_w &&
                   params_.down_w &&
                   params_.output;
        }
        return supportsLazyPrefillGraphCapturePreflight();
    }

    StageBufferRequirements SharedExpertFFNStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        if (params_.gate_scratch)
            reqs.addScratch(
                "gate_scratch",
                params_.gate_scratch->shape(),
                toBufferTensorType(params_.gate_scratch->native_type()));
        if (params_.up_scratch)
            reqs.addScratch(
                "up_scratch",
                params_.up_scratch->shape(),
                toBufferTensorType(params_.up_scratch->native_type()));
        return reqs;
    }

    StageBufferContract SharedExpertFFNStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();

        contract.addInput(params_.input_buffer_id);
        contract.addOutput(params_.output_buffer_id);
        if (params_.gate_scratch)
            contract.addOutput(params_.gate_scratch_buffer_id);
        if (params_.up_scratch)
            contract.addOutput(params_.up_scratch_buffer_id);

        // Shared-expert GEMMs consume store-owned prepared representations.
        if (params_.gate_w)
            contract.addPreparedWeight(
                params_.gate_w,
                params_.prepared_store,
                params_.prepared_ref_gate.value_or(PreparedWeightRef{}));
        if (params_.up_w)
            contract.addPreparedWeight(
                params_.up_w,
                params_.prepared_store,
                params_.prepared_ref_up.value_or(PreparedWeightRef{}));
        if (params_.down_w)
            contract.addPreparedWeight(
                params_.down_w,
                params_.prepared_store,
                params_.prepared_ref_down.value_or(PreparedWeightRef{}));

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

    struct SharedExpertGateStage::GpuEffectiveSeqLenState
    {
        DeviceId device = DeviceId::invalid();   ///< Device that owns device_effective_seq_len.
        int *host_effective_seq_len = nullptr;   ///< Pinned host scalar uploaded before capture/replay.
        int *device_effective_seq_len = nullptr; ///< Workspace scalar read by shared-gate kernels.
        bool device_value_uploaded = false;      ///< True once device scalar matches host_effective_seq_len.
    };

    SharedExpertGateStage::~SharedExpertGateStage()
    {
        releaseGpuEffectiveSeqLenState();
    }

    void SharedExpertGateStage::resetSessionState()
    {
        IComputeStage::resetSessionState();
        prefill_effective_seq_len_ = 0;
        prefill_replay_params_set_ = false;
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
    }

    void SharedExpertGateStage::resetSessionStatePreservingCapturedReplay()
    {
        IComputeStage::resetSessionState();
        prefill_effective_seq_len_ = 0;
        prefill_replay_params_set_ = false;
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
    }

    void SharedExpertGateStage::resetSessionStatePreservingLazyInitialization()
    {
        resetSessionStatePreservingCapturedReplay();
    }

    void SharedExpertGateStage::invalidateKernelDynamicState()
    {
        if (!owned_moe_kernel_)
            return;

        owned_moe_kernel_->resetDynamicState();
        owned_moe_kernel_->clearGPUStreamBinding();
    }

    int SharedExpertGateStage::effectivePrefillSeqLen() const
    {
        if (!prefill_replay_params_set_ || prefill_effective_seq_len_ <= 0)
            return params_.seq_len;
        return std::clamp(prefill_effective_seq_len_, 1, std::max(1, params_.seq_len));
    }

    void SharedExpertGateStage::updatePrefillReplayParams(const PrefillReplayParams &replay)
    {
        prefill_replay_params_set_ = true;
        const int real_seq_len = replay.real_seq_len > 0 ? replay.real_seq_len : params_.seq_len;
        prefill_effective_seq_len_ = std::clamp(real_seq_len, 1, std::max(1, params_.seq_len));
        refreshPinnedEffectiveSeqLen();
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
        if (params_.device_id.is_gpu() && hasGPUStream() && bound_workspace_)
            (void)(ensureGpuEffectiveSeqLenStateInitialized() && uploadGpuEffectiveSeqLen());
    }

    void SharedExpertGateStage::refreshPinnedEffectiveSeqLen()
    {
        if (gpu_effective_seq_len_state_ && gpu_effective_seq_len_state_->host_effective_seq_len)
            *gpu_effective_seq_len_state_->host_effective_seq_len = effectivePrefillSeqLen();
    }

    bool SharedExpertGateStage::ensureGpuEffectiveSeqLenStateInitialized()
    {
        if (!bound_workspace_ ||
            !bound_workspace_->hasBuffer(MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN) ||
            bound_workspace_->getBufferSize(MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN) < sizeof(int))
        {
            LOG_ERROR("[SharedExpertGateStage] Missing graph workspace buffer '"
                      << MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN
                      << "' for padded shared-expert gate replay on "
                      << params_.device_id.toString());
            return false;
        }

        auto *device_effective_seq_len = static_cast<int *>(
            bound_workspace_->getBuffer(MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN));
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[SharedExpertGateStage] Graph workspace buffer '"
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
            LOG_ERROR("[SharedExpertGateStage] Failed to allocate pinned effective-length scalar for "
                      << params_.device_id.toString());
            return false;
        }

        gpu_effective_seq_len_state_ = std::move(state);
        refreshPinnedEffectiveSeqLen();
        return true;
    }

    bool SharedExpertGateStage::uploadGpuEffectiveSeqLen()
    {
        if (!gpu_effective_seq_len_state_)
            return false;
        refreshPinnedEffectiveSeqLen();

        if (isGraphCaptureActive())
        {
            if (!gpu_effective_seq_len_state_->device_value_uploaded)
            {
                LOG_ERROR("[SharedExpertGateStage] Effective sequence length scalar was not uploaded before graph capture");
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

    void SharedExpertGateStage::releaseGpuEffectiveSeqLenState()
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

    bool SharedExpertGateStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
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
                "shared_gate_padded_prefill_effective_len_prepare",
                1.0,
                "prefill",
                params_.device_id.toString(),
                PerfStatsCollector::Tags{
                    {"bucket_seq_len", std::to_string(params_.seq_len)},
                    {"effective_seq_len", std::to_string(effectivePrefillSeqLen())}});
        }
        return uploaded;
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
            if (hasPrefillReplayParams() && prefill_replay_params_set_)
            {
                if (!ensureGpuEffectiveSeqLenStateInitialized() || !uploadGpuEffectiveSeqLen())
                    return false;
                if (!kernel->sharedExpertGateAddFromTensorsEffectiveSeqLen(
                        params_.input, gate_inp, params_.shared_output,
                        params_.routed_residual, params_.combined_output,
                        seq_len, d_model,
                        gpu_effective_seq_len_state_->device_effective_seq_len))
                {
                    LOG_ERROR("[SharedExpertGateStage] Effective-length fused shared gate-add failed");
                    return false;
                }
            }
            else
            {
                kernel->sharedExpertGateAddFromTensors(
                    params_.input, gate_inp, params_.shared_output,
                    params_.routed_residual, params_.combined_output,
                    seq_len, d_model);
            }
            // Fused gate-add materializes both semantic outputs: the gated
            // shared contribution and the final routed+shared combined row.
            if (params_.device_id.is_gpu())
            {
                const StageGPUExecution execution = gpuExecution();
                execution.publish(params_.shared_output);
                execution.publish(params_.combined_output);
            }
            return true;
        }

        if (hasPrefillReplayParams() && prefill_replay_params_set_)
        {
            if (!ensureGpuEffectiveSeqLenStateInitialized() || !uploadGpuEffectiveSeqLen())
                return false;
            if (!kernel->sharedExpertGateFromTensorsEffectiveSeqLen(
                    params_.input, gate_inp, params_.shared_output,
                    seq_len, d_model,
                    gpu_effective_seq_len_state_->device_effective_seq_len))
            {
                LOG_ERROR("[SharedExpertGateStage] Effective-length shared gate failed");
                return false;
            }
        }
        else
        {
            kernel->sharedExpertGateFromTensors(
                params_.input, gate_inp, params_.shared_output,
                seq_len, d_model);
        }
        if (params_.device_id.is_gpu())
            gpuExecution().publish(params_.shared_output);
        if (params_.device_id.is_gpu() && params_.shared_output->needsUpload())
        {
            throw std::runtime_error(
                "SharedExpertGateStage GPU kernel returned with a host-authoritative "
                "output; host repair/upload is forbidden");
        }

        return true;
    }

    IMoEKernel *SharedExpertGateStage::ensureMoEKernel() const
    {
        if (!moe_kernel_)
        {
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
            moe_kernel_ = owned_moe_kernel_.get();
        }
        auto *kernel = bindStageStream(moe_kernel_);
        if (bound_workspace_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
                consumer->bindWorkspace(bound_workspace_);
        }
        return kernel;
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
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
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
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
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

    bool SharedExpertGateStage::supportsLazyPrefillGraphCapturePreflight() const
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

    bool SharedExpertGateStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsLazyPrefillGraphCapturePreflight();
    }

    bool SharedExpertGateStage::supportsPaddedPrefillRealLengthContract() const
    {
        return supportsPaddedPrefillGraphCapturePreflight();
    }

    WorkspaceRequirements SharedExpertGateStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements reqs;
        if (hasPrefillReplayParams())
            MoEWorkspaceBuffers::add(reqs, MoEWorkspaceBuffers::PREFILL_EFFECTIVE_SEQ_LEN, sizeof(int));
        return reqs;
    }

    void SharedExpertGateStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        if (gpu_effective_seq_len_state_)
            gpu_effective_seq_len_state_->device_value_uploaded = false;
    }

    void SharedExpertGateStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
        if (gpu_effective_seq_len_state_)
        {
            gpu_effective_seq_len_state_->device_effective_seq_len = nullptr;
            gpu_effective_seq_len_state_->device_value_uploaded = false;
        }
    }

    bool SharedExpertGateStage::hasWorkspace() const
    {
        return bound_workspace_ != nullptr;
    }

    DeviceWorkspaceManager *SharedExpertGateStage::getWorkspace() const
    {
        return bound_workspace_;
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
