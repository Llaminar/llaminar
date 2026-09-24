/**
 * @file PipelineActivationExchange.h
 * @brief Explicit activation collective for native pipeline peers and TP domains.
 *
 * Native peers retain NCCL/RCCL point-to-point transport. At a declared
 * cross-backend boundary only the source/destination domain leaders touch the
 * TransferEngine channel; destination TP members use their native broadcast.
 * Every graph is still participant-local. No host relay, foreign tensor binding
 * or nested multi-device capture is hidden inside this collective stage.
 */
#pragma once

#include "execution/compute_stages/IComputeStage.h"
#include "PipelineNativeDomain.h"
#include "collective/ILocalTPContext.h"
#include "transfer/CapturedTransferChannel.h"
#include <optional>
#include <variant>

namespace llaminar2
{
/** @brief One exact physical activation exchange, frozen before graph capture. */
class PipelineActivationExchange final : public IComputeStage
{
public:
    /** @brief Existing homogeneous pipeline point-to-point membership. */
    struct NativePeer
    {
        ILocalTPContext &context;
        int participant;
        CollectiveP2POpKind direction;
    };
    /**
     * @brief Cross-backend channel endpoint and optional native TP domain.
     *
     * The source contributes a stage only on domain member zero. Every
     * destination member contributes a receive stage: member zero receives
     * before broadcasting, and all others receive that native broadcast.
     * A null context means exactly one GPU, never a host collective substitute.
     */
    struct CapturedDomain
    {
        ILocalTPContext *context = nullptr;
        int participant = 0;
        std::shared_ptr<CapturedTransferChannel> channel;
        CapturedTransferMessage message;
        CapturedTransferEndpoint endpoint = CapturedTransferEndpoint::Producer;
    };

    /** @brief Bind an existing native pipeline edge without changing its wire protocol.
     * @param device Exact participant-local GPU.
     * @param hidden Persistent FP32 arena bank, owned beyond this stage.
     * @param elements Physical row extent, never the arena's full capacity.
     * @param peer Frozen native communicator and adjacent peer direction. */
    PipelineActivationExchange(DeviceId device, TensorBase &hidden, size_t elements, NativePeer peer);
    /** @brief Bind a declared heterogeneous domain boundary and retain its storage.
     * @param device Exact participant-local GPU.
     * @param hidden Persistent local FP32 arena bank; no foreign-device view.
     * @param domain Exact endpoint, message, and destination native TP membership.
     * @param transfer Canonical transfer authority (injectable in device-free tests).
     * @throws std::invalid_argument for a missing/non-native/misowned boundary. */
    PipelineActivationExchange(DeviceId device, std::shared_ptr<TensorBase> hidden,
        CapturedDomain domain, TransferEngine &transfer = TransferEngine::instance());

    /** @brief Record exact transport then native broadcast on one non-null stream.
     * @param context The participant's own execution context.
     * @return Whether native transport accepted every required operation. */
    bool execute(IDeviceContext *context) override;
    /** @return Explicit collective classification for executor capture policy. */
    ComputeStageType type() const override { return ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE; }
    /** @return Stable direction-qualified semantic identity. */
    std::string name() const override;
    /** @return Whether this backend is the exact frozen GPU family. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @return All operations are native graph nodes; no host work on replay. */
    bool isGraphCapturable() const override { return true; }
    /** @return Exact physical payload, excluding unused arena capacity. */
    size_t estimatedMemoryBytes() const override { return elements_ * sizeof(float); }
    /** @return Input/output direction for the persistent local bank. */
    StageBufferRequirements getBufferRequirements() const override;
    /** @return The canonical hidden-state bank, never a second workspace. */
    StageBufferContract bufferContract() const override;
    /** @return Pointer/shape diagnostics without reading any live device state. */
    StageDumpInfo buildDumpInfoImpl() const override;

private:
    /** @brief Captured domain state retains its local owner and leader-only binding. */
    struct DomainBinding
    {
        CapturedDomain domain;
        std::shared_ptr<TensorBase> owner;
        std::optional<CapturedTransferBinding> endpoint;
        TransferEngine *transfer;
    };
    /** @brief Reject membership drift before any collective or channel submission. */
    void validateMembership() const;
    /** @return Whether this stage publishes rather than receives activations. */
    bool sends() const noexcept;

    TensorBase &hidden_;
    size_t elements_;
    std::variant<NativePeer, DomainBinding> transport_;
    /** Frozen capture identity only; the context remains the membership authority. */
    PipelineNativeDomain membership_;
};
}
