/**
 * @file PipelineForwardGraphEdges.h
 * @brief Immutable native collective edges around participant-local forward graphs.
 *
 * The rank freezes communicator membership before graph construction. Each
 * stage receives into and sends from its own arena allocation. The existing
 * forward role distinguishes captured prefill, scalar MTP condition and grouped
 * verification; no prefill materializer is invented for a verifier. No tensor is
 * moved to another device or rebound after capture. The owner outlives every
 * graph borrowing it, and its address is part of forward-cache identity.
 */
#pragma once
#include "backends/DeviceId.h"
#include "execution/compute_stages/stages/MTPVerifierPreparationStage.h"
#include "execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.h"
#include <optional>
#include <memory>
#include <string>

namespace llaminar2
{
class ILocalTPContext;
class TensorBase;
class ComputeGraph;
class DeviceGraphExecutor;
class IGPUGraphCapture;
class IWorkerGPUContext;
struct PipelinePublicationTransport;
struct ForwardInput;

/** @brief Frozen, allocation-free topology contributor; never a request ledger. */
class PipelineForwardGraphEdges final
{
public:
    /** @brief Four independent committed-state banks; geometry and outcomes are not wire identity. */
    struct PublicationBanks
    {
        int32_t *restore_rows = nullptr;
        int32_t *cached_tokens = nullptr;
        int32_t *accepted_counts = nullptr;
        int32_t *healthy = nullptr;
        /** @return Whether every field has distinct persistent storage. */
        [[nodiscard]] bool valid() const noexcept;
        /** @return The wire addresses contributed by the local publication authority. */
        static PublicationBanks from(const MTPSpeculativeStatePublicationStage::Params &params) noexcept;
        /** @return Whether every independently owned pointer identity matches exactly. */
        bool operator==(const PublicationBanks &) const = default;
    };
    /** @brief Follower-local rollback destination admitted by the existing state owner. */
    struct FollowerState
    {
        IBackend *backend = nullptr;
        MTPVerifierPreparationStage::MainKVCheckpointBinding checkpoint;
        /** @return Whether the follower can checkpoint without borrowing tail state. */
        [[nodiscard]] bool valid() const noexcept { return backend && checkpoint.valid(); }
    };
    /** @brief Validate an exact GPU participant and its persistent FP32 bank.
     * @param context Rank-owned native communicator, alive beyond all graphs.
     * @param participant Ordered stage index, not physical GPU ordinal.
     * @param hidden This stage's arena-owned incoming/outgoing hidden bank.
     * @param width Model activation width in FP32 elements.
     * @param follower Optional main-KV rollback binding; required for follower verification.
     * @throws std::invalid_argument for incomplete or non-native ownership. */
    PipelineForwardGraphEdges(ILocalTPContext &context, int participant, TensorBase &hidden, int width,
        std::optional<FollowerState> follower = std::nullopt);
    /** @brief Forget topology references; native recordings remain participant-owned leases. */
    ~PipelineForwardGraphEdges();
    /** @brief Record the fixed acceptance exchange once, in a symmetric serving-setup wave.
     * @param banks Durable local metadata, unchanged for the frozen arena lifetime.
     * @param executor Existing participant executor, not an alternative capture implementation.
     * @param context Exact local execution context.
     * @param gpu Exact native resource owner.
     * @return The participant-owned capture lease, or empty on failed preparation.
     * The DGO retires this lease after its parents and before releasing its arena;
     * the longer-lived rank topology keeps only a weak reference.
     * Repeated identical preparation reuses it; rebinding is a fatal caller error. */
    std::shared_ptr<PipelinePublicationTransport> materializePublicationTransport(PublicationBanks banks, DeviceGraphExecutor &executor,
        IDeviceContext &context, IWorkerGPUContext &gpu);
    /** @return The immutable native exchange for these exact addresses, or null before setup.
     * Changing the local commit limit or verifier width never invalidates this transport. */
    [[nodiscard]] const IGPUGraphCapture *publicationTransport(PublicationBanks banks) const;
    /** @brief Reject a local publication that would violate frozen pipeline state ownership.
     * @param params The full authority-owned local commit identity.
     * @throws std::invalid_argument for a foreign role, checkpoint, device or metadata bank. */
    void validatePublication(const MTPSpeculativeStatePublicationStage::Params &params) const;
    /** @brief Select the forward invocations whose native activation edges live in this graph.
     * @param input Immutable invocation policy; row count alone never selects a role.
     * @return True for a captured prefill chunk or scalar/grouped MTP main forward.
     * Ordinary decode retains the activation edges in its complete generation
     * parent, so this contributor must not insert those edges a second time. */
    [[nodiscard]] static bool encloses(const ForwardInput &input) noexcept;
    /** @brief Enclose one local prefill, condition or verifier DAG in native collectives.
     * @param graph Participant-local model DAG before executor preparation.
     * @param input Exact physical row geometry and local buffer bindings.
     * @throws std::invalid_argument if geometry, semantic role or ownership changed.
     * Validation precedes graph mutation. A malformed claimed role fails rather
     * than silently omitting its required transport. */
    void append(ComputeGraph &graph, const ForwardInput &input) const;
    /** @brief Validate a follower's captured receive before the forward prelude.
     * @param input Exact invocation binding the local collective destination.
     * @throws std::invalid_argument for terminal ownership or malformed rows.
     * This performs no enqueue or graph mutation. The same immutable geometry
     * validation is used when constructing the graph, so replay admission cannot
     * quietly accept a role that its retained topology does not implement. */
    void validateFollowerMTPInput(const ForwardInput &input) const;
    /** @brief Compose one complete native publication parent from two retained recordings.
     * @param destination Uninstantiated participant-local parent owned by the executor.
     * @param local Existing captured KV/recurrent/histogram publication, without transport.
     * @param params The complete identity of that local publication.
     * @return Whether tail commit-before-exchange or follower exchange-before-commit was built.
     * This clones native nodes only. It never enters collective capture, launches
     * children separately, or derives another participant's acceptance decision. */
    bool composePublication(IGPUGraphCapture &destination, const IGPUGraphCapture &local,
        const MTPSpeculativeStatePublicationStage::Params &params) const;
    /** @brief Join a setup-only native capture boundary on the exact stream.
     * @param boundary Shared compiler phase name, identical across participants.
     * @param stream Explicit participant capture stream.
     * @return Whether every participant entered the same bounded setup phase. */
    bool captureBoundary(const std::string &boundary, void *stream) const;
    /** @return The one rank-owned transport authority used by these graph edges. */
    ILocalTPContext &context() const noexcept { return context_; }
private:
    /** @brief Reject inconsistent semantic roles, local banks and checkpoint ownership.
     * @param input Immutable physical invocation, never observed device contents.
     * @throws std::invalid_argument before enqueue or graph mutation. */
    void validateInput(const ForwardInput &input) const;
    ILocalTPContext &context_;
    const int participant_;
    TensorBase &hidden_;
    const int width_;
    const DeviceId device_;
    const std::optional<FollowerState> follower_;
    /** Write-once wire identity; an expired participant lease cannot be resurrected. */
    std::optional<PublicationBanks> publication_banks_;
    std::weak_ptr<PipelinePublicationTransport> publication_transport_;
};
}
