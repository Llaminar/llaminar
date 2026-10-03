/**
 * @file MoEProjectionPipeline.cpp
 * @brief Shared native-domain lowering and local phases for projection-distributed MoE.
 *
 * Preparation creates only metadata over already prepared weights and binds
 * the ordinary admitted MoE workspace. Captured execution reads placement on
 * device and exchanges the existing activation representation unchanged. All
 * collectives remain visible nodes. The counted exchange retains arena owners
 * and the domain's pre-admitted channels, without allocating per-layer payloads.
 * The only final arithmetic is the same
 * full-K down dot and ordered top-k fold used by serial decode.
 * Diagnostics borrow each down stage's existing per-route column bank at its
 * producer boundary; they never add a full-route bank or another collective.
 */
#include "MoEProjectionPipeline.h"
#include "MoEOverlayPinnedRouteEvidenceViews.h"
#include "NativeAllGatherStage.h"
#include "TPColumnReduceScatterStage.h"
#include "TPLocalReduceOverlap.h"

#include "collective/ILocalTPContext.h"
#include "collective/DeviceCountedAllGather.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/ColumnShardAssemblyKernels.h"
#include "kernels/common/SharedExpertColumnGateKernels.h"
#include "models/GraphTypes.h"
#include "tensors/ITensor.h"
#include "tensors/TensorClasses.h"
#include "utils/FNV1a.h"
#include "utils/PerfStatsCollector.h"

#include <array>
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaminar2
{
void MoEProjectionPipelineParams::bindArena(const ActivationBuffers &buffers)
{
    const auto extension = [&](BufferId id) {
        const auto resolved = buffers.idFor(id);
        return MoEProjectionTensorBinding{buffers.get(resolved), resolved};
    };
    hidden = {buffers.normalized, buffers.idFor(BufferId::NORMALIZED)};
    routing_indices = extension(BufferId::MOE_EXPERT_INDICES);
    routing_weights = extension(BufferId::MOE_EXPERT_WEIGHTS);
    local_packet = extension(BufferId::MOE_PROJECTION_LOCAL_PACKET);
    gathered_packets = extension(BufferId::MOE_PROJECTION_GATHERED_PACKETS);
    local_route_columns = extension(BufferId::MOE_PROJECTION_ROUTE_COLUMNS);
    local_columns = extension(BufferId::MOE_PROJECTION_LOCAL_COLUMNS);
    gathered_columns = extension(BufferId::MOE_PROJECTION_GATHERED_COLUMNS);
    output = extension(BufferId::MOE_COMBINED_OUTPUT);
}

namespace
{
    /** @brief Local work on either side of the two explicit native collectives. */
    enum class Phase { GateUpPack, ImportDown, FoldRoutes, Assemble };

    /**
     * @brief Graph-local immutable binding and its one prepared metadata publication.
     *
     * This is not another runtime authority: only descriptor IDs and borrowed
     * scratch addresses are prepared here. Device placement, histograms and
     * route assignments remain in DeviceMoERuntimeTable. Stages share the same
     * existing routed kernel so the two groupings reuse its scratch sequentially.
     */
    class Binding final
    {
    public:
        /** @brief Validate the entire declaration before the builder mutates a graph. */
        explicit Binding(MoEProjectionPipelineParams declaration) : params(std::move(declaration))
        {
            if (!params.fixed_down || !params.runtime || !params.tp || !params.kernel_owner ||
                params.rows <= 0 || params.top_k <= 0 || params.prefix.empty() || params.router_node.empty() ||
                params.rows > std::numeric_limits<int>::max() / params.top_k ||
                !validMoEGroupedPlanDemand(params.demand))
                throw std::invalid_argument("Projection graph requires complete immutable execution bindings");
            const auto &ownership = params.fixed_down->ownership();
            const auto geometry = ownership.geometry();
            if (params.top_k > geometry.experts ||
                params.initial_gate_up.size() != static_cast<std::size_t>(geometry.experts) ||
                params.runtime->deviceId() != device() ||
                ownership.participants() != params.tp->degree() || ownership.participants() < 2)
                throw std::invalid_argument("Projection graph source/domain geometry disagrees");
            const auto *fixed = params.runtime->fixedDownProjectionBank(params.fixed_down->layer());
            if (!fixed || !fixed->sameIdentity(*params.fixed_down))
                throw std::logic_error("Projection graph cannot substitute its runtime's fixed down authority");
            telemetry = params.runtime->deviceOverlayServiceTelemetryBinding(params.fixed_down->layer());
            if (!telemetry.empty() && (!telemetry.valid() || params.service_phase == MoEOverlayServicePhaseHint::Auto))
                throw std::invalid_argument("Projection service measurement requires a complete binding and explicit graph phase");
            const auto &runtime = params.runtime->hostLayerState(params.fixed_down->layer());
            if (runtime.participant_id != static_cast<std::uint32_t>(ownership.participant()) ||
                params.fixed_down->participantId() != ownership.participant() ||
                runtime.participant_count != static_cast<std::uint32_t>(ownership.participants()) ||
                runtime.expert_count != static_cast<std::uint32_t>(geometry.experts) ||
                runtime.top_k != static_cast<std::uint32_t>(params.top_k) ||
                runtime.prefill_token_capacity < static_cast<std::uint32_t>(params.rows) ||
                runtime.prefill_route_capacity < static_cast<std::uint32_t>(params.rows * params.top_k) ||
                !runtime.route_participant_ids)
                throw std::logic_error("Native projection graph requires exact domain-local runtime coordinates and capacity");
            // The fixed bank authenticates executed precision, independently
            // of whether this participant currently owns any routed gate/up.
            fixed_table = params.fixed_down->exportDescriptorTable();
            family = std::visit([](const auto &table) {
                if constexpr (std::is_same_v<std::decay_t<decltype(table)>,
                              MoEOverlayFixedDownProjectionBank::FloatingDescriptorTable>)
                    return table.format;
                else return DeviceMoEWeightFormat::NativeVNNI;
            }, fixed_table);
            layout = {deviceMoEWeightFormatIsFloating(family) ? MoEGroupedIntermediateEncoding::FP32
                        : MoEGroupedIntermediateEncoding::BlockQ8FP32Scales,
                static_cast<std::uint32_t>(geometry.intermediate_columns),
                static_cast<std::uint32_t>(params.rows * params.top_k),
                static_cast<std::uint32_t>(ownership.participants())};
            if (!layout.valid()) throw std::invalid_argument("Projection packet geometry is not representable");
            const auto row_bytes = std::size_t(params.rows) * geometry.model_columns * sizeof(float);
            requireBuffer(params.hidden, row_bytes, true);
            requireBuffer(params.routing_indices, std::size_t(layout.route_capacity) * sizeof(float), true);
            requireBuffer(params.routing_weights, std::size_t(layout.route_capacity) * sizeof(float), true);
            if (params.counted_exchange)
            {
                if (!layout.compactValid() || params.counted_exchange->devices().size() != layout.participants ||
                    params.counted_exchange->capacityBytes() < layout.compactCapacityBytes())
                    throw std::invalid_argument("Projection compact fabric disagrees with the admitted layout");
                for (std::size_t peer = 0; peer < layout.participants; ++peer)
                    if (params.counted_exchange->devices()[peer] != params.tp->devices()[peer].toLocalDeviceId())
                        throw std::invalid_argument("Projection compact fabric has foreign communicator membership");
            }
            const auto packet_bytes = params.counted_exchange ? layout.compactCapacityBytes() : layout.packetBytes();
            requireBuffer(params.local_packet, packet_bytes, false);
            requireBuffer(params.gathered_packets, packet_bytes *
                (layout.participants - (params.counted_exchange ? 1u : 0u)), false);
            requireBuffer(params.local_route_columns,
                std::size_t(layout.route_capacity) * ownership.projection(WeightRole::MoEExpertDown).rows * sizeof(float), true);
            requireBuffer(params.local_columns, row_bytes / layout.participants, true);
            requireBuffer(params.gathered_columns, row_bytes, true);
            requireBuffer(params.output, row_bytes, true);
            const std::array bindings{params.hidden, params.routing_indices, params.routing_weights,
                params.local_packet, params.gathered_packets, params.local_route_columns,
                params.local_columns, params.gathered_columns, params.output};
            for (std::size_t i = 0; i < bindings.size(); ++i)
                for (std::size_t j = 0; j < i; ++j)
                    if (bindings[i].tensor == bindings[j].tensor || bindings[i].id == bindings[j].id)
                        throw std::invalid_argument("Projection graph requires distinct logical arena bindings");
            // A production overlay acquires a request-pinned placement bank.
            // Expose its existing storage at the final assembly boundary, not
            // after a later layer has reused the route scratch. Component-only
            // runtime tables without an overlay ticket have no such evidence.
            // Projection execution always publishes grouped slots, even M=1;
            // borrowing serial-decode top-k here would observe stale weights.
            if (params.runtime->usesOverlayEpochTicket())
                route_evidence = std::make_unique<MoEOverlayPinnedRouteEvidenceViews>(
                    MoEOverlayPinnedRouteEvidenceViews::Params{
                        .device = device(),
                        .domain_route_assignment = {runtime.route_participant_ids, runtime.prefill_route_capacity},
                        .runtime_route_weights = bindMoERuntimeRouteWeights(
                            params.runtime->deviceLayerState(params.fixed_down->layer()), runtime,
                            MoERuntimeRouteWeightProjection::GroupedRouteSlots),
                        .overlay_route_placement = params.runtime->overlayRoutePlacementBinding(params.fixed_down->layer()),
                        .physical_rows = static_cast<std::uint32_t>(params.rows),
                        .top_k = static_cast<std::uint32_t>(params.top_k)});
        }

        /** @return The fixed bank's authenticated physical endpoint. */
        DeviceId device() const noexcept { return params.fixed_down->device(); }

        /** @brief Bind metadata once before capture; reject workspace replacement. */
        void prepare(DeviceWorkspaceManager *workspace, void *stream)
        {
            if (!workspace || !stream || isGraphCaptureActive())
                throw std::logic_error("Projection preparation requires admitted workspace outside graph capture");
            if (prepared)
            {
                if (prepared->workspace != workspace || prepared->workspace_id != workspace->id())
                    throw std::logic_error("Projection graph workspace identity changed after preparation");
                if (prepared->stream != stream)
                    throw std::logic_error("Projection graph descriptor publication belongs to another capture stream");
                return;
            }
            auto &kernel = params.kernel_owner->kernel;
            if (!kernel) kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device());
            if (!kernel) throw std::runtime_error("Projection graph has no backend kernel");
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            if (!consumer) throw std::logic_error("Projection kernel has no admitted workspace contract");
            consumer->bindWorkspace(workspace);
            const auto geometry = params.fixed_down->ownership().geometry();
            std::vector<DeviceNativeVNNIMatrixDesc> gates(geometry.experts), ups(geometry.experts);
            std::vector<DeviceMoEFloatingMatrixDesc> fp_gates(geometry.experts), fp_ups(geometry.experts);
            for (int expert = 0; expert < geometry.experts; ++expert)
            {
                const auto &payload = params.initial_gate_up[expert];
                if (payload.empty()) continue;
                DeviceMoEExpertDescriptor descriptor{};
                if (!payload.readyFor(DeviceMoEProjectionSet::GateUp) ||
                    !exportDeviceMoEPreparedPayload(payload, descriptor) || descriptor.weight_format != family)
                    throw std::invalid_argument("Projection graph initial payload has a foreign projection/precision contract");
                gates[expert] = descriptor.gate; ups[expert] = descriptor.up;
                fp_gates[expert] = descriptor.floating_gate; fp_ups[expert] = descriptor.floating_up;
            }
            const int gate_table = deviceMoEWeightFormatIsFloating(family)
                ? kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(fp_gates.data(), fp_ups.data(), family,
                    geometry.experts, geometry.model_columns, geometry.intermediate_columns)
                : kernel->uploadGroupedExpertGateUpDescriptorTables(gates.data(), ups.data(), geometry.experts,
                    geometry.model_columns, geometry.intermediate_columns, MoEDecodeDescriptorSource::RuntimePlacementTable);
            const int down_table = std::visit([&](const auto &table) {
                const auto n = params.fixed_down->ownership().projection(WeightRole::MoEExpertDown).rows;
                if constexpr (std::is_same_v<std::decay_t<decltype(table)>,
                              MoEOverlayFixedDownProjectionBank::FloatingDescriptorTable>)
                    return kernel->uploadGroupedExpertFloatingDownDescriptorTable(table.experts.data(), family,
                        geometry.experts, n, geometry.intermediate_columns);
                else return kernel->uploadGroupedExpertDownDescriptorTable(table.experts.data(), geometry.experts,
                    n, geometry.intermediate_columns);
            }, fixed_table);
            if (gate_table < 0 || down_table < 0)
                throw std::runtime_error("Projection descriptor preparation rejected its exact source family");
            if (params.counted_exchange)
            {
                const auto retain = [](ITensor *tensor) -> std::shared_ptr<TensorBase> {
                    auto *owner = dynamic_cast<TensorBase *>(tensor);
                    auto retained = owner ? owner->weak_from_this().lock() : nullptr;
                    if (!retained) throw std::invalid_argument("Projection packet requires retained arena ownership through graph retirement");
                    return retained;
                };
                const std::string identity = params.prefix + "/" + std::to_string(params.rows) + "/" +
                    std::to_string(layout.columns) + "/" + std::to_string(static_cast<int>(layout.encoding));
                compact = params.counted_exchange->bind(params.fixed_down->ownership().participant(),
                    {fnv1a64(identity.data(), identity.size()), layout.compactCapacityBytes()},
                    retain(params.local_packet.tensor), retain(params.gathered_packets.tensor));
            }
            // Publication is atomic on the host setup path. Execute sees either
            // the complete immutable metadata identity or a hard failure.
            prepared = Prepared{workspace, workspace->id(), stream, gate_table, down_table};
        }

        /**
         * @brief Invalidate descriptor handles after the executor retires native graphs.
         *
         * All local phases share this publication. Only the first hook
         * resets the kernel; subsequent phase hooks see an already cold binding.
         * Ordinary request reset does not call this or discard captured metadata.
         */
        void invalidate()
        {
            if (!prepared) return;
            params.kernel_owner->kernel->resetDynamicState();
            params.kernel_owner->kernel->clearGPUStreamBinding();
            compact.reset();
            prepared.reset();
        }

        /** @brief Enqueue local math only; collectives are separate graph nodes. */
        bool execute(Phase phase, void *stream)
        {
            if (!prepared || !stream || !params.kernel_owner->kernel)
                throw std::logic_error("Projection graph executed before complete preparation");
            if (prepared->stream != stream)
                throw std::logic_error("Projection graph execution changed its prepared metadata stream");
            auto &kernel = *params.kernel_owner->kernel;
            kernel.setGPUStream(stream);
            const auto &ownership = params.fixed_down->ownership();
            const auto geometry = ownership.geometry();
            const auto &runtime = params.runtime->hostLayerState(params.fixed_down->layer());
            if (phase == Phase::GateUpPack)
            {
                auto *device_runtime = params.runtime->deviceLayerState(params.fixed_down->layer());
                const MoEKernelLaunchContext launch{.stream = stream, .workspace = prepared->workspace};
                if (!telemetry.empty() && !kernel.beginMoEOverlayServiceTelemetry(launch, telemetry.sample)) return false;
                if (!kernel.publishCompleteGroupedPrefillPlanFromRouter(device_runtime,
                        params.routing_indices.tensor, params.routing_weights.tensor, params.rows, params.rows,
                        geometry.experts, params.top_k, prepared->gate, prepared->down, true,
                        params.demand, DeviceMoEProjectionSet::GateUp) ||
                    !kernel.executeGroupedPrefillProjectionFromPublishedRuntimePlan(device_runtime, runtime,
                        params.hidden.tensor, nullptr, prepared->gate, prepared->down, params.rows,
                        geometry.model_columns, geometry.intermediate_columns, geometry.experts, params.top_k,
                        MoEPrefillProjectionExecution::gateUp(geometry.model_columns))) return false;
                // Only these owner-local dots move. Fixed down columns and
                // native exchanges are invariant under gate/up reassignment;
                // charging them would exaggerate the economic payoff.
                if (!telemetry.empty())
                {
                    if (!kernel.finishMoEOverlayServiceTelemetry(launch, telemetry.runtime_layer,
                        telemetry.layer_telemetry, telemetry.sample, geometry.experts, params.service_phase, nullptr)) return false;
                    PerfStatsCollector::addCounter("moe_overlay_residency", "device_service_marker_pairs_enqueued", 1.0,
                        "graph_capture", device().toString(),
                        {{"layer", std::to_string(params.fixed_down->layer())}, {"projection", "gate_up"}});
                }
                if (compact)
                    return kernel.exportCompactGroupedPrefillIntermediates(layout, runtime, params.local_packet.tensor,
                        compact->extent(ownership.participant()));
                return kernel.exportGroupedPrefillIntermediates(layout, runtime.route_participant_ids,
                        params.local_packet.tensor, ownership.participant());
            }
            if (phase == Phase::ImportDown)
            {
                const auto down = ownership.projection(WeightRole::MoEExpertDown);
                // Pack has already consumed the owner-local inverse map. This
                // regroup changes only private compute scratch, not runtime
                // route assignments or the weights used by another participant.
                if (!kernel.prepareExpertGroupsAsync(params.routing_indices.tensor, params.routing_weights.tensor,
                        params.rows, geometry.experts, params.top_k)) return false;
                if (compact)
                {
                    for (std::uint32_t source = 0; source < layout.participants; ++source)
                    {
                        const auto own = static_cast<std::uint32_t>(ownership.participant());
                        // All peer packets share one arena publication. Keep its
                        // owner as the capture-ledger input and select a checked
                        // byte offset; tensor views would add unrelated coherence.
                        const auto offset = source == own ? 0 :
                            (source < own ? source : source - 1) * layout.compactCapacityBytes();
                        auto *packet = source == own ? params.local_packet.tensor : params.gathered_packets.tensor;
                        if (!kernel.importCompactGroupedPrefillIntermediates(layout, runtime.route_participant_ids,
                            packet, offset, compact->extent(source), source)) return false;
                    }
                }
                else if (!kernel.importGroupedPrefillIntermediates(layout, runtime.route_participant_ids,
                    params.gathered_packets.tensor)) return false;
                return kernel.executeGroupedPrefillProjection(nullptr, nullptr,
                        prepared->gate, prepared->down, params.rows, geometry.model_columns, geometry.intermediate_columns,
                        geometry.experts, params.top_k,
                        MoEPrefillProjectionExecution::down(geometry.model_columns, down.first_row, down.rows),
                        params.local_route_columns.tensor);
            }
            if (phase == Phase::FoldRoutes)
                return kernel.reduceCanonicalRouteContributions(params.local_route_columns.tensor,
                    params.local_columns.tensor, params.rows, params.top_k,
                    ownership.projection(WeightRole::MoEExpertDown).rows);
            return assembleColumnShardsFP32(device(),
                static_cast<const float *>(params.gathered_columns.tensor->gpu_data_ptr()),
                static_cast<float *>(params.output.tensor->gpu_data_ptr()),
                params.rows, ownership.participants(), geometry.model_columns / ownership.participants(), stream);
        }

        /** @brief Fully installed descriptor identity; no placement or request state. */
        struct Prepared
        {
            DeviceWorkspaceManager *workspace;
            std::uint64_t workspace_id;
            void *stream;
            int gate;
            int down;
        };
        const MoEProjectionPipelineParams params;
        MoEGroupedIntermediateLayout layout;
        DeviceMoEWeightFormat family;
        MoEOverlayFixedDownProjectionBank::DescriptorTable fixed_table;
        std::optional<Prepared> prepared;
        std::optional<DeviceCountedAllGatherBinding> compact; ///< Immutable transport binding, not live host counts.
        DeviceMoEOverlayServiceTelemetryBinding telemetry; ///< Borrowed table-owned economy observations.
        std::unique_ptr<MoEOverlayPinnedRouteEvidenceViews> route_evidence; ///< Views only; runtime owns every byte.

    private:
        /** @brief Validate extents against an admitted tensor, without capacity accounting. */
        static void requireBuffer(MoEProjectionTensorBinding binding, std::size_t bytes, bool fp32)
        {
            if (!binding.tensor || binding.tensor->size_bytes() < bytes ||
                (fp32 && binding.tensor->native_type() != TensorType::FP32))
                throw std::invalid_argument("Projection graph arena binding " + std::string(bufferIdName(binding.id)) +
                    " requires " + std::to_string(bytes) + " bytes" + (fp32 ? " of FP32" : "") +
                    "; actual bytes=" + std::to_string(binding.tensor ? binding.tensor->size_bytes() : 0) +
                    " type=" + (binding.tensor ? std::to_string(static_cast<int>(binding.tensor->native_type())) : "missing"));
        }
    };

    /** @brief Thin participant-local phase; it owns no transport or private scratch bank. */
    class ProjectionStage final : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        /** @brief Bind one immutable local phase to the shared graph-local metadata. */
        ProjectionStage(std::shared_ptr<Binding> binding, Phase phase, std::string name,
                        MoEProjectionTensorBinding combined_output = {})
            : IComputeStage(binding->device()), binding_(std::move(binding)), phase_(phase),
              name_(std::move(name)), combined_output_(combined_output)
        {
            if (combined_output_.tensor && phase_ != Phase::Assemble)
                throw std::invalid_argument("Only projection assembly can publish a full combined row");
        }
        /** @return Setup-only shared declaration; no second owner or copied runtime state. */
        const std::shared_ptr<Binding> &binding() const noexcept { return binding_; }
        /** @return Immutable local phase used to authenticate graph rewrites before capture. */
        Phase phase() const noexcept { return phase_; }
        /** @return Whether final assembly already consumes combined columns. */
        bool publishesCombinedColumns() const noexcept { return combined_output_.tensor != nullptr; }
        /** @return Dedicated local-compute classification, not a collective. */
        ComputeStageType type() const override { return ComputeStageType::MOE_PROJECTION_PHASE; }
        /** @return The frozen model/layer/phase identity. */
        std::string name() const override { return name_; }
        /** @return Only the exact compiled GPU backend is supported. */
        bool supportsBackend(ComputeBackendType backend) const override
        {
            return (binding_->device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
                   (binding_->device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
        }
        /** @return True only after the single metadata publication completed. */
        bool isGraphCapturable() const override { return workspace_ && binding_->prepared.has_value(); }
        /** @return Preparation installs persistent descriptors without an eager forward. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        /** @return Router invalid slots exclude padded rows on every phase. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
        /** @return No eager inference is needed to establish the capture contract. */
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        /** @return Preparation runs before capture, never during replay. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }
        /** @brief Prepare shared descriptor metadata on the exact graph stream. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            if (!ctx || ctx->deviceId() != binding_->device() || !stream) return false;
            binding_->prepare(workspace_, stream);
            // Publish the stage binding only after shared metadata accepts it;
            // a rejected preparation must preserve the previous valid stream.
            setGPUStream(stream);
            return true;
        }
        /** @brief Participate in the executor's existing kernel-metadata reset boundary. */
        void invalidateKernelDynamicState() override { binding_->invalidate(); }
        /** @brief Execute this local phase with arena-visible input/output publication. */
        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != binding_->device() || !workspace_) return false;
            const auto execution = gpuExecution();
            for (const auto &input : inputs()) execution.requirePreparedInput(input.tensor);
            execution.requirePreparedOutput(output().tensor);
            if (combined_output_.tensor)
            {
                const auto &p = binding_->params;
                const auto &owner = p.fixed_down->ownership();
                if (!assembleColumnShardsFP32(binding_->device(),
                    static_cast<const float *>(p.gathered_columns.tensor->gpu_data_ptr()),
                    static_cast<float *>(combined_output_.tensor->gpu_data_ptr()), p.rows,
                    owner.participants(), owner.geometry().model_columns / owner.participants(),
                    execution.nativeStream())) return false;
            }
            else if (!binding_->execute(phase_, execution.nativeStream())) return false;
            execution.publish(output().tensor);
            return true;
        }
        /** @return This phase's logical arena dependency contract. */
        StageBufferContract bufferContract() const override
        {
            auto contract = StageBufferContract::build();
            for (const auto &input : inputs()) contract.addInput(input.id);
            return contract.addOutput(output().id);
        }
        /** @return Exact external tensor geometries; workspace is declared separately. */
        StageBufferRequirements getBufferRequirements() const override
        {
            StageBufferRequirements requirements;
            std::size_t index = 0;
            for (const auto &input : inputs())
                requirements.addInput("projection_input_" + std::to_string(index++), input.tensor->shape(),
                    toBufferTensorType(input.tensor->native_type()));
            requirements.addOutput("projection_output", output().tensor->shape(), toBufferTensorType(output().tensor->native_type()));
            return requirements;
        }
        /** @return The ordinary kernel workspace; phases merge/reuse identical names. */
        WorkspaceRequirements getWorkspaceRequirements(int m, int = 0, int = 0) const override
        {
            const auto &p = binding_->params;
            const auto g = p.fixed_down->ownership().geometry();
            const int rows = std::max(m, p.rows);
            return binding_->device().is_cuda() ? MoEWorkspaceBuffers::cudaMoE(rows, g.model_columns, g.intermediate_columns, g.experts, p.top_k)
                : MoEWorkspaceBuffers::rocmMoE(rows, g.model_columns, g.intermediate_columns, g.experts, p.top_k);
        }
        /** @brief Borrow the executor's sole admitted workspace. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        /** @brief Retire only this stage's borrow; graph retirement owns the shared metadata. */
        void unbindWorkspace() override { workspace_ = nullptr; }
        /** @return Whether this stage has a live workspace binding. */
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        /** @return Exact executor-owned workspace. */
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
        /** @return Outputs are prepared before recording; execute publishes exact-stream writes. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @return Phase outputs and zero-copy descriptors of the acquired route epoch. */
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addScalarInt("projection_phase", static_cast<int>(phase_));
            if (phase_ == Phase::ImportDown)
            {
                // Down writes weighted addends in original (token, top-k)
                // order. Expose that storage before the fold consumes it,
                // with one packed route per row and only our output columns.
                // Snapshot aggregation joins columns, never sums participants.
                const auto &p = binding_->params;
                info.addOutput("canonical_route_contributions", output().tensor,
                    static_cast<std::size_t>(p.rows) * p.top_k,
                    p.fixed_down->ownership().projection(WeightRole::MoEExpertDown).rows);
            }
            else if (phase_ == Phase::Assemble)
            {
                // Every participant now owns the complete row, not a partial
                // expert sum. SnapshotCapture preserves that publication type
                // so diagnostic TP aggregation cannot multiply it by degree.
                info.addOutput(combined_output_.tensor ? "combined_output" : "output", output().tensor, binding_->params.rows,
                    binding_->params.fixed_down->ownership().geometry().model_columns);
                if (binding_->route_evidence) binding_->route_evidence->appendOutputs(info);
            }
            else
                info.addOutput("projection_output", output().tensor, output().tensor->rows(), output().tensor->cols());
            return info;
        }

    private:
        /** @return Immutable phase inputs; fixed arrays avoid per-launch allocation. */
        std::span<const MoEProjectionTensorBinding> inputs() const
        {
            // Params members are not an array. Keep explicit immutable arrays,
            // built once with the stage, instead of relying on object layout.
            switch (phase_)
            {
            case Phase::GateUpPack: return gate_inputs_;
            case Phase::ImportDown: return down_inputs_;
            case Phase::FoldRoutes: return fold_inputs_;
            case Phase::Assemble: return column_inputs_;
            }
            throw std::logic_error("Unknown projection phase");
        }
        /** @return The sole tensor publication owned by this phase. */
        MoEProjectionTensorBinding output() const
        {
            const auto &p = binding_->params;
            switch (phase_)
            {
            case Phase::GateUpPack: return p.local_packet;
            case Phase::ImportDown: return p.local_route_columns;
            case Phase::FoldRoutes: return p.local_columns;
            case Phase::Assemble: return combined_output_.tensor ? combined_output_ : p.output;
            }
            throw std::logic_error("Unknown projection phase");
        }
        std::shared_ptr<Binding> binding_;
        const Phase phase_;
        const std::string name_;
        const MoEProjectionTensorBinding combined_output_; ///< Optional final combined-row publication.
        const std::array<MoEProjectionTensorBinding, 3> gate_inputs_{
            binding_->params.hidden, binding_->params.routing_indices, binding_->params.routing_weights};
        const std::array<MoEProjectionTensorBinding, 4> down_inputs_{
            binding_->params.gathered_packets, binding_->params.local_packet,
            binding_->params.routing_indices, binding_->params.routing_weights};
        const std::array<MoEProjectionTensorBinding, 1> fold_inputs_{binding_->params.local_route_columns};
        const std::array<MoEProjectionTensorBinding, 1> column_inputs_{binding_->params.gathered_columns};
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

    /** @brief Explicit counted collective; all model arithmetic remains in local phases. */
    class CompactExchangeStage final : public CapturedAllGatherStage
    {
    public:
        /** @brief Borrow one frozen participant's model-local binding. */
        explicit CompactExchangeStage(std::shared_ptr<Binding> binding)
            : CapturedAllGatherStage(binding->device()), binding_(std::move(binding)) {}
        /** @return Capturable collective classification, never an ordinary compute leaf. */
        ComputeStageType type() const override { return ComputeStageType::DEVICE_COUNTED_ALLGATHER; }
        /** @return Stable semantic identity retained by downstream dependency edges. */
        std::string name() const override { return binding_->params.prefix + "intermediate_allgather"; }
        /** @return True so graph scheduling retains the participant rendezvous. */
        bool requiresAllreduce() const override { return true; }
        /** @return The exact compiled backend of the frozen participant. */
        bool supportsBackend(ComputeBackendType backend) const override
        { return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
                 (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM); }
        /** @return True only after the local producer has prepared the shared binding. */
        bool isGraphCapturable() const override { return binding_->compact.has_value(); }
        /** @return Setup is completed by the preceding gate/up stage before recording. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        /** @return Cold exact buckets may prepare the already-admitted fabric without executing inference. */
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        /** @return The device extent, not a padded bucket, determines wire traffic. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
        /** @brief Enqueue all sends/receives on the exact consumer stream. */
        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != device() || !binding_->compact) return false;
            const auto execution = gpuExecution();
            validateEnqueue(execution);
            binding_->compact->enqueue(execution.nativeStream());
            execution.publish(binding_->params.gathered_packets.tensor);
            return true;
        }
        /** @return Canonical packet owners, never offset-view coherence identities. */
        CapturedAllGatherBuffers exchangeBuffers() const override
        { return {binding_->params.local_packet.tensor, binding_->params.gathered_packets.tensor,
            binding_->params.local_packet.id, binding_->params.gathered_packets.id}; }
        /** @brief Authenticate the prepared packet banks on the canonical producer stream. */
        void validateEnqueue(const StageGPUExecution &execution) const override
        {
            if (execution.device() != device() || !binding_->compact)
                throw std::logic_error("Counted allgather requires its prepared exact-device binding");
            execution.requirePreparedInput(binding_->params.local_packet.tensor);
            execution.requirePreparedOutput(binding_->params.gathered_packets.tensor);
        }
        /** @brief Borrow the existing fork proof; only the paired join publishes received bytes. */
        bool enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const override
        {
            if (!binding_->compact) throw std::logic_error("Counted allgather binding is unprepared");
            binding_->compact->enqueueAcquiredInput(input);
            return true;
        }
        /** @return Exact arena dependencies; extent lifetime is retained by the frozen fabric. */
        StageBufferContract bufferContract() const override
        { return StageBufferContract::build().addInput(binding_->params.local_packet.id)
            .addOutput(binding_->params.gathered_packets.id); }
        /** @return Setup-only destination preparation; replay cannot allocate or rebind it. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @return Physical capacity is metadata, not a claim of bytes transferred. */
        StageDumpInfo buildDumpInfoImpl() const override
        { StageDumpInfo info; info.addScalarInt("device_counted", 1); return info; }
    private:
        std::shared_ptr<Binding> binding_;
    };

    /** @brief Local epilogue over already reduced columns; all transport remains separate. */
    class SharedColumnGateStage final : public IComputeStage
    {
    public:
        /** @brief Borrow the frozen pipeline and its complete shared-gate declaration. */
        SharedColumnGateStage(std::shared_ptr<Binding> binding, MoEProjectionSharedColumns shared)
            : IComputeStage(binding->device()), binding_(std::move(binding)), shared_(std::move(shared)) {}
        /** @return Shared-gate arithmetic classification, never an implicit collective. */
        ComputeStageType type() const override { return ComputeStageType::MOE_SHARED_EXPERT_GATE; }
        /** @return Stable per-layer identity for column-partitioned checkpoints. */
        std::string name() const override { return binding_->params.prefix + "shared_columns"; }
        /** @return Only this native GPU participant's compiled backend. */
        bool supportsBackend(ComputeBackendType backend) const override
        {
            return (binding_->device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
                (binding_->device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
        }
        /** @return One allocation-free launch over prepared arena/weight storage. */
        bool isGraphCapturable() const override { return true; }
        /** @return Captured row masking is read directly from the declared device scalar. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return shared_.active_rows != nullptr; }
        /** @return The same device scalar masks both shared and combined output tails. */
        bool supportsPaddedPrefillRealLengthContract() const override { return shared_.active_rows != nullptr; }
        /** @return A saturated sigmoid or masked row may legitimately be all zero. */
        bool allowsZeroOutput() const override { return true; }
        /** @brief Execute the full-width gate dot and only the owned output columns. */
        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != binding_->device()) return false;
            const auto execution = gpuExecution();
            const auto &p = binding_->params;
            for (auto *tensor : std::array<ITensor *, 4>{p.hidden.tensor, shared_.gate,
                    shared_.reduction.tensor, p.local_columns.tensor})
                execution.requirePreparedInput(tensor);
            execution.requirePreparedOutput(shared_.reduction.tensor);
            execution.requirePreparedOutput(p.local_route_columns.tensor);
            const auto &owner = p.fixed_down->ownership();
            if (!gateSharedExpertColumnsFP32(binding_->device(), {
                .input = static_cast<const float *>(p.hidden.tensor->gpu_data_ptr()),
                .gate = static_cast<const float *>(shared_.gate->gpu_data_ptr()),
                .shared = static_cast<float *>(shared_.reduction.tensor->gpu_data_ptr()),
                .routed = static_cast<const float *>(p.local_columns.tensor->gpu_data_ptr()),
                .combined = static_cast<float *>(p.local_route_columns.tensor->gpu_data_ptr()),
                .rows = p.rows, .model_columns = owner.geometry().model_columns,
                .local_columns = owner.projection(WeightRole::MoEExpertDown).rows,
                .active_rows = shared_.active_rows}, execution.nativeStream())) return false;
            execution.publish(shared_.reduction.tensor);
            execution.publish(p.local_route_columns.tensor);
            return true;
        }
        /** @return Exact borrowed inputs and both writes, including prepared gate weight. */
        StageBufferContract bufferContract() const override
        {
            const auto &p = binding_->params;
            return StageBufferContract::build().addInput(p.hidden.id).addWeight(shared_.gate)
                .addInput(p.local_columns.id).addInOut(*shared_.reduction.tensor_buffer_id)
                .addOutput(p.local_route_columns.id);
        }
        /** @return Existing full-bank tensor requirements; local prefixes do not resize storage. */
        StageBufferRequirements getBufferRequirements() const override
        {
            StageBufferRequirements requirements;
            const auto &p = binding_->params;
            requirements.addInput("input", p.hidden.tensor->shape(), BufferTensorType::FP32);
            requirements.addInput("routed_columns", p.local_columns.tensor->shape(), BufferTensorType::FP32);
            requirements.addInput("shared_columns", shared_.reduction.tensor->shape(), BufferTensorType::FP32);
            requirements.addOutput("shared_columns", shared_.reduction.tensor->shape(), BufferTensorType::FP32);
            requirements.addOutput("combined_columns", p.local_route_columns.tensor->shape(), BufferTensorType::FP32);
            return requirements;
        }
        /** @return No new allocation; only the executor's pre-bound output preparation. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @return Routed and gated shared columns before their storage is reused. */
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            const auto &p = binding_->params;
            const auto columns = p.fixed_down->ownership().projection(WeightRole::MoEExpertDown).rows;
            info.addOutput("routed_output", p.local_columns.tensor, p.rows, columns);
            info.addOutput("shared_output", shared_.reduction.tensor, p.rows, columns);
            return info;
        }
    private:
        const std::shared_ptr<Binding> binding_;
        const MoEProjectionSharedColumns shared_;
    };
} // namespace

std::string appendMoEProjectionPipeline(ComputeGraph &graph, MoEProjectionPipelineParams params)
{
    auto binding = std::make_shared<Binding>(std::move(params));
    const auto &p = binding->params;
    const auto &ownership = p.fixed_down->ownership();
    const auto column_bytes = std::size_t(p.rows) * ownership.projection(WeightRole::MoEExpertDown).rows * sizeof(float);
    const std::array names{p.prefix + "gate_up_pack", p.prefix + "intermediate_allgather",
        p.prefix + "import_down", p.prefix + "fold_routes", p.prefix + "column_allgather", p.prefix + "assemble"};
    if (!graph.getNode(p.router_node))
        throw std::invalid_argument("Projection pipeline has no graph-owned router producer");
    for (const auto &name : names)
        if (graph.getNode(name)) throw std::invalid_argument("Projection pipeline node identity already exists");
    const auto gather = [&](MoEProjectionTensorBinding input, MoEProjectionTensorBinding output,
                            std::size_t bytes, const std::string &name) {
        NativeAllGatherStage::Params params;
        params.device_id = binding->device(); params.tp_ctx = p.tp;
        params.local_input = input.tensor; params.rank_major_output = output.tensor;
        params.bytes_per_participant = bytes; params.participant = ownership.participant();
        params.stage_name = name; params.input_buffer_id = input.id; params.output_buffer_id = output.id;
        // Only the finished column banks are dense equal-prefix rows. Routed
        // intermediate packets have independent ownership/count semantics.
        if (input.id == p.local_columns.id && p.live_rows)
            params.live_rows = NativeCollectiveRows(*p.live_rows,
                ownership.projection(WeightRole::MoEExpertDown).rows * sizeof(float));
        return std::make_unique<NativeAllGatherStage>(std::move(params));
    };
    // Native membership and every tensor contract must validate before adding
    // any graph node. Neither constructor performs backend work or allocates VRAM.
    std::array<std::unique_ptr<IComputeStage>, 6> stages{
        std::make_unique<ProjectionStage>(binding, Phase::GateUpPack, names[0]),
        p.counted_exchange ? std::unique_ptr<IComputeStage>(std::make_unique<CompactExchangeStage>(binding))
            : gather(p.local_packet, p.gathered_packets, binding->layout.packetBytes(), names[1]),
        std::make_unique<ProjectionStage>(binding, Phase::ImportDown, names[2]),
        std::make_unique<ProjectionStage>(binding, Phase::FoldRoutes, names[3]),
        gather(p.local_columns, p.gathered_columns, column_bytes, names[4]),
        std::make_unique<ProjectionStage>(binding, Phase::Assemble, names[5])};
    for (std::size_t i = 0; i < stages.size(); ++i)
    {
        graph.addNode(names[i], std::move(stages[i]), binding->device());
        graph.addDependency(names[i], i ? names[i - 1] : p.router_node);
    }
    return names.back();
}

void overlapMoEProjectionSharedAllreduce(ComputeGraph &graph,
    const std::string &prefix, const std::string &sum_name)
{
    const auto *down_node = graph.getNode(prefix + "import_down");
    const auto *down = down_node ? dynamic_cast<const ProjectionStage *>(down_node->stage.get()) : nullptr;
    const auto *sum_node = graph.getNode(sum_name);
    const auto *sum = sum_node ? dynamic_cast<const TPAllreduceStage *>(sum_node->stage.get()) : nullptr;
    if (!down || down->phase() != Phase::ImportDown || !sum ||
        down->binding()->params.prefix != prefix ||
        sum->params().tp_ctx != down->binding()->params.tp ||
        sum->device() != down->device() ||
        !graph.getNode(prefix + "intermediate_allgather") ||
        !graph.getNode(prefix + "fold_routes"))
        throw std::invalid_argument("Shared full-row overlap requires its exact projection domain and native sum");

    // The common builder rejects aliases and cycles before replacing the sum.
    // Reuse its exact allreduce; changing to reduce-scatter would also change
    // the serial verifier's arithmetic contract, not merely its schedule.
    overlapTPLocalAllreduce(graph, sum_name, prefix + "import_down");
    graph.addDependency(sum_name + "_submit", prefix + "intermediate_allgather");
    graph.addDependency(prefix + "fold_routes", sum_name);
}

std::string finalizeMoEProjectionSharedColumns(ComputeGraph &graph,
    const std::string &prefix, MoEProjectionSharedColumns shared)
{
    auto *assembly_node = graph.getNode(prefix + "assemble");
    const auto *assembly = assembly_node ? dynamic_cast<const ProjectionStage *>(assembly_node->stage.get()) : nullptr;
    auto *gather_node = graph.getNode(prefix + "column_allgather");
    const auto *gather = gather_node ? dynamic_cast<const NativeAllGatherStage *>(gather_node->stage.get()) : nullptr;
    const auto *producer = graph.getNode(shared.producer);
    if (!assembly || assembly->phase() != Phase::Assemble || assembly->publishesCombinedColumns() ||
        !gather || !producer || !producer->stage || !shared.gate || !shared.combined_output.tensor)
        throw std::invalid_argument("Shared-column finalization requires an unmodified projection DAG and complete shared producer");
    const auto binding = assembly->binding();
    const auto &p = binding->params;
    const auto &owner = p.fixed_down->ownership();
    const auto &sum = shared.reduction;
    const auto columns = owner.geometry().model_columns;
    const auto full_bytes = std::size_t(p.rows) * columns * sizeof(float);
    if (p.prefix != prefix || sum.device_id != binding->device() || sum.tp_ctx != p.tp ||
        sum.count != full_bytes / sizeof(float) || !sum.tensor || !sum.tensor_buffer_id ||
        sum.arithmetic_policy != TPAllreduceArithmeticPolicy::NativeCollective ||
        !sum.sidebands.empty() || !sum.sideband_workspace_bindings.empty() ||
        producer->device != binding->device() || shared.gate->native_type() != TensorType::FP32 ||
        shared.gate->size_bytes() < std::size_t(columns) * sizeof(float) ||
        shared.combined_output.tensor->native_type() != TensorType::FP32 ||
        shared.combined_output.tensor->size_bytes() < full_bytes ||
        gather->params().local_input != p.local_columns.tensor ||
        gather->params().rank_major_output != p.gathered_columns.tensor ||
        gather_node->dependencies != std::vector<std::string>{prefix + "fold_routes"} ||
        assembly_node->dependencies != std::vector<std::string>{prefix + "column_allgather"})
        throw std::invalid_argument("Shared-column finalization disagrees with native policy, bank geometry or graph ownership");
    for (const auto &b : {p.hidden, p.local_columns, p.local_route_columns, p.gathered_columns,
                          p.local_packet, p.gathered_packets, p.routing_indices, p.routing_weights})
        if (b.tensor == sum.tensor || b.id == *sum.tensor_buffer_id ||
            b.tensor == shared.combined_output.tensor || b.id == shared.combined_output.id)
            throw std::invalid_argument("Shared-column finalization aliases a live pipeline operand");
    if (sum.tensor == shared.combined_output.tensor || *sum.tensor_buffer_id == shared.combined_output.id)
        throw std::invalid_argument("Shared and final publication require disjoint banks");

    const auto scatter_name = prefix + "shared_reduce_scatter";
    const auto gate_name = prefix + "shared_columns";
    if (graph.getNode(scatter_name) || graph.getNode(gate_name))
        throw std::invalid_argument("Shared-column finalization was already declared");
    TPColumnReduceScatterStage::Params scatter;
    scatter.device_id = binding->device(); scatter.tp_ctx = p.tp;
    scatter.tensor = sum.tensor; scatter.packing = p.gathered_columns.tensor;
    scatter.rows = p.rows; scatter.model_columns = columns;
    scatter.live_rows = p.live_rows;
    scatter.participant = owner.participant(); scatter.precision = sum.precision;
    scatter.stage_name = scatter_name; scatter.tensor_buffer_id = sum.tensor_buffer_id;
    scatter.packing_buffer_id = p.gathered_columns.id;
    auto scatter_stage = std::make_unique<TPColumnReduceScatterStage>(scatter);
    auto combined_gather = gather->params();
    // The route fold has consumed these weighted addends. Its dead storage is
    // now the compact combined-column send bank; no third allocation is needed.
    combined_gather.local_input = p.local_route_columns.tensor;
    combined_gather.input_buffer_id = p.local_route_columns.id;
    auto combined_gather_stage = std::make_unique<NativeAllGatherStage>(combined_gather);
    auto combined_assembly = std::make_unique<ProjectionStage>(binding, Phase::Assemble,
        prefix + "assemble", shared.combined_output);
    auto gate_stage = std::make_unique<SharedColumnGateStage>(binding, shared);

    graph.addNode(scatter_name, std::move(scatter_stage), binding->device());
    graph.addDependency(scatter_name, shared.producer);
    overlapTPLocalColumnReduceScatter(graph, scatter_name, prefix + "import_down");
    // Native communicators require one common submission order even when the
    // shared producer and gate/up path are independent graph branches.
    graph.addDependency(scatter_name + "_submit", prefix + "intermediate_allgather");
    graph.addNode(gate_name, std::move(gate_stage), binding->device());
    graph.addDependency(gate_name, scatter_name);
    graph.addDependency(gate_name, prefix + "fold_routes");
    // All participants submit intermediate gather -> shared scatter -> column
    // gather. Only local down compute occupies the scatter's event window.
    gather_node->stage = std::move(combined_gather_stage);
    graph.addDependency(prefix + "column_allgather", gate_name);
    assembly_node->stage = std::move(combined_assembly);
    return prefix + "assemble";
}
} // namespace llaminar2
