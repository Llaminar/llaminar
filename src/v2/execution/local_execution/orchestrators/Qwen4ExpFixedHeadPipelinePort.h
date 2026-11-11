/**
 * @file Qwen4ExpFixedHeadPipelinePort.h
 * @brief Fixed native QFN range service on original captured pipeline participants.
 *
 * The actual PLE stage owns original query and history. Every other stage
 * supplies its admitted immutable native range inside its own forward DAG.
 * The pipeline's original native context carries only addressed IDs and rows.
 */
#pragma once
#include "PipelineNativeByteChannel.h"
#include "loaders/ImmutableFixedHeadPipelineProjection.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "kernels/qwen4exp/ImmutableFixedHeadLookupKernels.h"
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace llaminar2 {
class BufferArena;
class ComputeGraph;
class DeviceCountedAllGather;
class IModelContext;
class Qwen4ExpAuxiliaryState;
class Qwen4ExpReplicatedPLEQuery;
class ResidentImmutableRowTable;
class ResolvedImmutableRowTablePlacement;
struct ForwardInput;
struct RankInventory;

/** @brief One actual native pipeline participant's fixed-range roles and immutable roots. */
class Qwen4ExpFixedHeadPipelinePort final : public std::enable_shared_from_this<Qwen4ExpFixedHeadPipelinePort> {
public:
    /** @brief Exact already admitted original participant, in complete native context order. */
    struct Participant {
        DeviceId device;
        int stage;
        int stage_member;
        BufferArena *arena;
        const IModelContext *model;
        ILocalTPContext *stage_context;
        void *setup_stream;
    };
    /** @brief Materialize original role/edge resources once before forward declaration and capture. */
    static std::vector<std::shared_ptr<const Qwen4ExpFixedHeadPipelinePort>> prepare(
        std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement,
        std::span<const Participant> members, ILocalTPContext &pipeline,
        const RankInventory &inventory, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
        const qwen4exp::NgramHashPolicy &policy);
    /** @return Whether this actual participant owns main PLE computation in its existing stage. */
    bool consumesPLE() const noexcept { return owner_.stage_index == projection_->consumerStage(); }
    /** @return Whether a storage provider must follow its own outgoing activation frontier. */
    bool suppliesAfterActivation() const noexcept { return owner_.stage_index < projection_->consumerStage(); }
    /** @brief Authenticate actual consumer model, device and arena, and return its native coordinate. */
    int requireConsumer(const IModelContext &model, const BufferArena &arena, DeviceId device) const;
    /** @brief Authenticate the DGO-issued query and actual PLE history against this exact port. */
    void requireQuery(const Qwen4ExpReplicatedPLEQuery &query, const ForwardInput &input,
        DeviceRowRange rows, const Qwen4ExpAuxiliaryState &history) const;
    /** @brief Append this native consumer's lookup fragment to the original declarative DAG. */
    void appendConsumer(ComputeGraph &fragment, const std::string &prefix, const std::string &terminal,
        const Qwen4ExpReplicatedPLEQuery &query, std::optional<Qwen4ExpIntBinding> count,
        Qwen4ExpIntBinding ids, Qwen4ExpFloatBinding output) const;
    /** @brief Append the storage participant's service and return its exact response terminal. */
    std::string appendStorage(ComputeGraph &graph, const ForwardInput &input,
        std::span<const std::string> dependencies) const;

private:
    /** @brief One real consumer/range pair, with two directions and exact arena subspans. */
    struct Edge {
        ImmutableFixedHeadPipelineOwner storage;
        ImmutableFixedHeadPipelineEdgeScratch scratch;
        std::shared_ptr<PipelineNativeByteChannel> ids;
        std::shared_ptr<PipelineNativeByteChannel> rows;
    };
    /** @brief Retain the exact original immutable range and kernel binding. */
    Qwen4ExpFixedHeadPipelinePort(std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement,
        Participant participant, int native_member, const qwen4exp::NgramHashPolicy &policy);
    /** @return Original participant-local arena tensor for this graph role. */
    std::shared_ptr<TensorBase> bank(BufferId role) const;
    /** @return Capture-frozen direction/shape identity, excluding live counts. */
    CapturedTransferMessage wireMessage(const Edge &edge, bool response, int rows) const;

    std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement_;
    const ImmutableFixedHeadPipelineProjection *projection_;
    const Participant participant_;
    const int native_member_;
    const ImmutableFixedHeadPipelineOwner owner_;
    std::shared_ptr<const ResidentImmutableRowTable> table_;
    std::unique_ptr<qwen4exp::IImmutableFixedHeadPipelineKernels> kernels_;
    std::shared_ptr<DeviceCountedAllGather> variable_gather_;
    std::shared_ptr<TensorBase> scratch_;
    std::vector<Edge> edges_;
};
}
