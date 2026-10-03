/**
 * @file MoERoutingPipeline.cpp
 * @brief Captured local router -> exact-count exchange -> complete route publication.
 *
 * The domain fabric alone owns transfer epochs/count storage. Existing arena
 * packet owners are borrowed until publication, then projection gate/up may
 * overwrite them. Model files declare this transaction rather than wiring
 * native streams or packet state. Diagnostic probabilities remain row-owned;
 * only top-k IDs and FP32 weights are communicated during inference.
 */
#include "MoERoutingPipeline.h"
#include "CapturedAllGatherStage.h"
#include "collective/DeviceCountedAllGather.h"
#include "collective/ILocalTPContext.h"
#include "models/GraphTypes.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "tensors/TensorClasses.h"
#include "utils/FNV1a.h"
#include <array>
#include <optional>

namespace llaminar2
{
MoERoutingDistribution MoERoutingDistribution::fromDomain(
    ILocalTPContext &domain, DeviceId device, const ActivationBuffers &buffers)
{
    auto fabric = domain.deviceCountedAllGather();
    if (!fabric) return {};
    const auto &members = fabric->devices();
    if (members.size() != domain.devices().size())
        throw std::invalid_argument("Router fabric has foreign TP membership");
    int participant = -1;
    for (std::size_t index = 0; index < members.size(); ++index)
    {
        if (members[index] != domain.devices()[index].toLocalDeviceId())
            throw std::invalid_argument("Router fabric changed the TP participant ordering");
        if (members[index] == device) participant = static_cast<int>(index);
    }
    if (participant < 0) throw std::invalid_argument("Router is not a member of its declared domain");
    const auto binding = [&](BufferId id) {
        const auto resolved = buffers.idFor(id);
        return MoEProjectionTensorBinding{buffers.get(resolved), resolved};
    };
    return {std::move(fabric), participant, binding(BufferId::MOE_PROJECTION_LOCAL_PACKET),
            binding(BufferId::MOE_PROJECTION_GATHERED_PACKETS)};
}

namespace
{
    /** @brief Frozen graph binding; prepared state contains no host copy of live GPU values. */
    struct RoutingBinding
    {
        MoERoutingStage::Params route;
        MoERoutingDistribution distribution;
        MoERouterRowPacketLayout layout;
        std::string node;
        std::optional<DeviceCountedAllGatherBinding> exchange;

        /** @brief Bind canonical arena owners once, after materialization and before capture. */
        void prepare()
        {
            if (exchange) return;
            const auto retain = [](ITensor *tensor) -> std::shared_ptr<TensorBase> {
                auto *base = dynamic_cast<TensorBase *>(tensor);
                auto owner = base ? base->weak_from_this().lock() : nullptr;
                if (!owner) throw std::logic_error("Owned routing requires canonical retained packet owners");
                return owner;
            };
            const auto identity = node + "/owned/" + std::to_string(layout.capacity) + "/" +
                std::to_string(layout.top_k) + "/" + std::to_string(layout.partition.participants());
            exchange = distribution.fabric->bind(layout.partition.participant(),
                {fnv1a64(identity.data(), identity.size()), layout.packetBytes()},
                retain(distribution.local_packet.tensor), retain(distribution.received_packets.tensor));
        }
    };

    /** @brief Visible graph collective; it performs no model math or ownership decisions. */
    class RoutingExchangeStage final : public CapturedAllGatherStage
    {
    public:
        /** @brief Borrow the frozen domain and packet lifetimes. */
        explicit RoutingExchangeStage(std::shared_ptr<RoutingBinding> binding)
            : CapturedAllGatherStage(binding->route.device_id), binding_(std::move(binding)) {}
        /** @return Explicit captured byte-exchange classification. */
        ComputeStageType type() const override { return ComputeStageType::DEVICE_COUNTED_ALLGATHER; }
        /** @return Stable collective identity distinct from the complete-route frontier. */
        std::string name() const override { return binding_->node + "_exchange"; }
        /** @return Membership rendezvous is required even for an empty local owner. */
        bool requiresAllreduce() const override { return true; }
        /** @return Whether exact frozen bindings are prepared. */
        bool isGraphCapturable() const override { return binding_->exchange.has_value(); }
        /** @return Cold setup binds owners without inference or allocation. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        /** @return Exact buckets support the same capture-only setup. */
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        /** @return Counts are authored on device for partial/empty replay. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
        /** @return No payload-capacity padding is communicated. */
        bool supportsPaddedPrefillRealLengthContract() const override { return true; }
        /** @return Binding-only setup, never an eager trial. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        { return GraphLaunchPreparationPolicy::CaptureOnly; }
        /** @brief Bind after the arena has materialized its canonical owners. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            if (!ctx || ctx->deviceId() != device() || !stream) return false;
            binding_->prepare();
            return true;
        }
        /** @return Only this participant's native GPU backend. */
        bool supportsBackend(ComputeBackendType backend) const override
        { return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
                 (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM); }
        /** @return Both banks are visible to arena lifetime analysis. */
        StageBufferContract bufferContract() const override
        { return StageBufferContract::build().addInput(binding_->distribution.local_packet.id)
            .addOutput(binding_->distribution.received_packets.id); }
        /** @return Output materialization belongs to setup; execution publishes exact bytes. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @return Capacity is a diagnostic bound, never a claimed communication extent. */
        StageDumpInfo buildDumpInfoImpl() const override
        { StageDumpInfo info; info.addScalarInt("device_counted", 1); return info; }
        /** @return Canonical storage identities for any legal graph fork/join. */
        CapturedAllGatherBuffers exchangeBuffers() const override
        {
            const auto &d = binding_->distribution;
            return {d.local_packet.tensor, d.received_packets.tensor, d.local_packet.id, d.received_packets.id};
        }
        /** @brief Authenticate the prepared exact-device event frontier. */
        void validateEnqueue(const StageGPUExecution &execution) const override
        {
            if (execution.device() != device() || !binding_->exchange)
                throw std::logic_error("Owned router exchange has no prepared participant binding");
            execution.requirePreparedInput(binding_->distribution.local_packet.tensor);
            execution.requirePreparedOutput(binding_->distribution.received_packets.tensor);
        }
        /** @brief Enqueue on an acquired fork; its paired join alone publishes the result. */
        bool enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const override
        {
            if (!binding_->exchange) throw std::logic_error("Unprepared owned router exchange");
            binding_->exchange->enqueueAcquiredInput(input);
            return true;
        }
        /** @brief Enqueue all sends before receives, preserving the domain's existing protocol. */
        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != device()) return false;
            const auto execution = gpuExecution();
            validateEnqueue(execution);
            binding_->exchange->enqueue(execution.nativeStream());
            execution.publish(binding_->distribution.received_packets.tensor);
            return true;
        }
    private:
        std::shared_ptr<RoutingBinding> binding_;
    };

    /** @brief Local lossless assembly; downstream stages see exactly the old complete arrays. */
    class RoutingPublicationStage final : public IComputeStage
    {
    public:
        /** @brief Retain the same prepared kernel and domain lifetimes as the local producer. */
        explicit RoutingPublicationStage(std::shared_ptr<RoutingBinding> binding)
            : IComputeStage(binding->route.device_id), binding_(std::move(binding)) {}
        /** @return This is local routing math, not an implicit collective. */
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        /** @return The existing semantic complete-router frontier. */
        std::string name() const override { return binding_->node; }
        /** @return The operation needs only immutable addresses and one kernel launch. */
        bool isGraphCapturable() const override { return true; }
        /** @return Cold exact buckets need no eager work. */
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        /** @return Source counts and suffix masks remain device-owned. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
        /** @return Inactive original rows receive the established -1/zero sentinel. */
        bool supportsPaddedPrefillRealLengthContract() const override { return true; }
        /** @return Empty live prefixes are valid collective transactions. */
        bool allowsZeroOutput() const override { return true; }
        /** @return Only the exact native backend can consume the packets. */
        bool supportsBackend(ComputeBackendType backend) const override
        { return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
                 (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM); }
        /** @return The last reads of packet storage, before projection may overwrite it. */
        StageBufferContract bufferContract() const override
        {
            const auto &r = binding_->route; const auto &d = binding_->distribution;
            return StageBufferContract::build().addInput(d.local_packet.id).addInput(d.received_packets.id)
                .addOutput(r.output_indices_buffer_id).addOutput(r.output_weights_buffer_id);
        }
        /** @return Complete-route outputs are admitted before recording. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @brief Authenticate extents and copy each selection without arithmetic. */
        bool execute(IDeviceContext *ctx) override
        {
            const auto &r = binding_->route; const auto &d = binding_->distribution;
            if (!ctx || ctx->deviceId() != device() || !binding_->exchange ||
                !r.routed_pipeline_kernel_owner || !r.routed_pipeline_kernel_owner->kernel) return false;
            const auto execution = gpuExecution();
            execution.requirePreparedInput(d.local_packet.tensor);
            execution.requirePreparedInput(d.received_packets.tensor);
            execution.requirePreparedOutput(r.output_indices);
            execution.requirePreparedOutput(r.output_weights);
            auto &kernel = r.routed_pipeline_kernel_owner->kernel;
            kernel->setGPUStream(execution.nativeStream());
            if (!kernel->publishOwnedRouteRows({
                    .layout = binding_->layout, .live_rows = r.active_row_count_device,
                    .local = static_cast<const MoERouterSelectedRoute *>(d.local_packet.tensor->gpu_data_ptr()),
                    .peers = d.received_packets.tensor->gpu_data_ptr(),
                    .acquired_bytes = binding_->exchange->extent(0),
                    .indices = static_cast<float *>(r.output_indices->gpu_data_ptr()),
                    .weights = static_cast<float *>(r.output_weights->gpu_data_ptr())})) return false;
            execution.publish(r.output_indices); execution.publish(r.output_weights);
            return true;
        }
        /** @return Complete selected-route checkpoints; probabilities belong to the first stage. */
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info; const auto &r = binding_->route;
            info.addOutput("output_indices_tensor", r.output_indices, r.seq_len, r.top_k);
            info.addOutput("output_weights_tensor", r.output_weights, r.seq_len, r.top_k);
            return info;
        }
    private:
        std::shared_ptr<RoutingBinding> binding_;
    };
}

void appendMoERoutingPipeline(ComputeGraph &graph, MoERoutingStage::Params route,
    std::string node, const std::string &input_node, MoERoutingDistribution distribution)
{
    if (node.empty() || graph.getNode(node) || !graph.getNode(input_node))
        throw std::invalid_argument("Router graph frontier requires a unique node and existing input");
    if (!distribution.fabric)
    {
        graph.addNode(node, std::make_unique<MoERoutingStage>(route), route.device_id);
        graph.addDependency(node, input_node);
        return;
    }
    const auto &members = distribution.fabric->devices();
    if (!route.routed_pipeline_kernel_owner || members.size() < 2 ||
        distribution.participant < 0 || std::size_t(distribution.participant) >= members.size() ||
        members[distribution.participant] != route.device_id ||
        !route.active_row_count_device || route.force_decode_equivalent_verifier_prefill)
        throw std::invalid_argument("Distributed router declaration disagrees with its prefill domain");
    const auto partition = DeviceRowPartition::balanced(distribution.participant, static_cast<int>(members.size()));
    if (route.seq_len < 512)
    {
        // Replication is the measured economical small-batch compute policy.
        // Diagnostics still assign each row to one producer, so mixed-size
        // prompt chunks preserve a single explicit assembly contract.
        route.probability_snapshot_partition = partition;
        graph.addNode(node, std::make_unique<MoERoutingStage>(route), route.device_id);
        graph.addDependency(node, input_node);
        return;
    }
    const MoERouterRowPacketLayout layout{
        partition,
        route.seq_len, route.top_k};
    const auto bytes = layout.packetBytes();
    if (distribution.fabric->capacityBytes() < bytes ||
        !distribution.local_packet.tensor || !distribution.received_packets.tensor ||
        distribution.local_packet.tensor->size_bytes() < bytes ||
        distribution.received_packets.tensor->size_bytes() < bytes * (members.size() - 1))
        throw std::invalid_argument("Distributed router exceeds its admitted packet BOM");
    const std::array<MoEProjectionTensorBinding, 5> buffers{distribution.local_packet, distribution.received_packets,
        MoEProjectionTensorBinding{route.input, route.input_buffer_id},
        MoEProjectionTensorBinding{route.output_indices, route.output_indices_buffer_id},
        MoEProjectionTensorBinding{route.output_weights, route.output_weights_buffer_id}};
    for (std::size_t i = 0; i < buffers.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (buffers[i].id == buffers[j].id || buffers[i].tensor == buffers[j].tensor)
                throw std::invalid_argument("Router packet aliases a live input or output");
    auto binding = std::make_shared<RoutingBinding>(RoutingBinding{
        route, std::move(distribution), layout, node, {}});
    route.owned_rows = MoERouterOwnedStageOutput{layout,
        dynamic_cast<TensorBase *>(binding->distribution.local_packet.tensor),
        binding->distribution.local_packet.id,
        binding->distribution.fabric->producerExtentAddress(layout.partition.participant())};
    std::array<std::unique_ptr<IComputeStage>, 3> stages{
        std::make_unique<MoERoutingStage>(std::move(route)),
        std::make_unique<RoutingExchangeStage>(binding), std::make_unique<RoutingPublicationStage>(binding)};
    const std::array names{node + "_owned", node + "_exchange", node};
    for (std::size_t i = 0; i < stages.size(); ++i)
    {
        graph.addNode(names[i], std::move(stages[i]), binding->route.device_id);
        graph.addDependency(names[i], i ? names[i - 1] : input_node);
    }
}
}
