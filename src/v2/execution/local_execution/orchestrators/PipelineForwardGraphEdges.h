/**
 * @file PipelineForwardGraphEdges.h
 * @brief Immutable native/domain collective edges around participant-local forwards.
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
#include "PipelineMetadataExchange.h"
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
class BufferArena;
struct PipelinePublicationTransport;
struct ForwardInput;

/** @brief Frozen, allocation-free topology contributor; never a request ledger. */
class PipelineForwardGraphEdges final
{
public:
    /** @brief Complete publication topology, derived from frozen domain ownership. */
    enum class PublicationOrder { LocalCommit, CommitThenExchange, ExchangeThenCommit };
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
    /** @brief Frozen adjacent cross-backend edges plus this domain's native TP membership. */
    struct CapturedDomain
    {
        size_t stage_index = 0;
        size_t stage_count = 0;
        ILocalTPContext *context = nullptr;
        int participant = 0;
        std::shared_ptr<CapturedTransferChannel> activation_in;
        std::shared_ptr<CapturedTransferChannel> activation_out;
        std::shared_ptr<CapturedTransferChannel> metadata_in;
        std::shared_ptr<CapturedTransferChannel> metadata_out;
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
    /** @brief Bind a real heterogeneous domain to its own arena and retained channels.
     * @param device Exact local member; domain index is independent of its GPU ordinal.
     * @param domain Frozen native membership and directed adjacent edges.
     * @param arena Original participant-local allocation owner, alive beyond its graphs.
     * @param width Full activation row width; TP domains exchange replicated hidden rows.
     * @param follower Local rollback authority, required only for nonterminal MTP stages.
     * @param transfer Canonical transfer authority; injectable for device-free proofs. */
    PipelineForwardGraphEdges(DeviceId device, CapturedDomain domain, BufferArena &arena, int width,
        std::optional<FollowerState> follower = std::nullopt,
        TransferEngine &transfer = TransferEngine::instance());
    /** @brief Forget topology references; native recordings remain participant-owned leases. */
    ~PipelineForwardGraphEdges();
    /** @brief Prepare the exact acceptance topology once, before request admission.
     * @param banks Durable local metadata, unchanged for the frozen arena lifetime.
     * @param executor Existing participant executor, not an alternative capture implementation.
     * @param context Exact local execution context.
     * @param gpu Exact native resource owner.
     * @param retained_banks Original storage leases for captured-domain metadata.
     * @return The participant-owned topology/storage lease, or empty on failure.
     * Local-only terminal siblings retain their banks without recording wire work.
     * The DGO retires this lease after its parents and before releasing its arena;
     * the longer-lived rank topology keeps only a weak reference.
     * Repeated identical preparation reuses it; rebinding is a fatal caller error. */
    std::shared_ptr<PipelinePublicationTransport> materializePublicationTransport(PublicationBanks banks, DeviceGraphExecutor &executor,
        IDeviceContext &context, IWorkerGPUContext &gpu,
        std::vector<PipelineMetadataBank> retained_banks = {});
    /** @return The prepared publication order, or empty for missing/stale setup.
     * Local-only publication is a first-class topology, not an empty/missing
     * transport. Changing commit limits does not change the frozen wire banks. */
    [[nodiscard]] std::optional<PublicationOrder> preparedPublication(PublicationBanks banks) const;
    /** @brief Reject a local publication that would violate frozen pipeline state ownership.
     * @param params The full authority-owned local commit identity.
     * @throws std::invalid_argument for a foreign role, checkpoint, device or metadata bank. */
    void validatePublication(const MTPSpeculativeStatePublicationStage::Params &params) const;
    /** @brief Select the forward invocations whose native activation edges live in this graph.
     * @param input Immutable invocation policy; row count alone never selects a role.
     * @return True for prefill/MTP, or ordinary decode at a heterogeneous boundary.
     * Homogeneous ordinary decode retains the edges in its complete generation
     * parent. A heterogeneous domain instead encloses its complete local forward
     * here; its transaction composer must not add a second set of edges. */
    [[nodiscard]] bool encloses(const ForwardInput &input) const noexcept;
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
    /** @brief Compose the local commit and its topology-required transport recordings.
     * @param destination Uninstantiated participant-local parent owned by the executor.
     * @param local Existing captured KV/recurrent/histogram publication, without transport.
     * @param params The complete identity of that local publication.
     * @return Whether the exact publication was built. Terminal TP siblings
     * commit locally; the leader alone publishes their domain's outgoing word banks.
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
    ILocalTPContext *context() const noexcept { return context_; }
    /** @return A compiled inter-vendor boundary, not a backend capability advertisement. */
    bool hasHeterogeneousBoundary() const noexcept { return domain_.has_value(); }
private:
    /** @brief Reject inconsistent semantic roles, local banks and checkpoint ownership.
     * @param input Immutable physical invocation, never observed device contents.
     * @throws std::invalid_argument before enqueue or graph mutation. */
    void validateInput(const ForwardInput &input) const;
    /** @return Whether this participant owns the terminal layer domain. */
    bool terminal() const noexcept;
    /** @return Whether this domain consumes an earlier layer's activation. */
    bool receivesActivation() const noexcept;
    /** @return Whether this participant publishes its domain's activation to the next one. */
    bool sendsActivation() const noexcept;
    /** @return The one publication order required by this participant.
     * Terminal TP siblings own their local commit but no incoming metadata or
     * outgoing channel. Their publication must not manufacture an empty GPU
     * transport executable or a redundant native collective. */
    PublicationOrder publicationOrder() const noexcept;
    /** @return The retained native wire recording, or null for missing/no wire work. */
    const IGPUGraphCapture *publicationTransport(PublicationBanks banks) const;
    /** @brief Bind exact row banks to the declared metadata family without copying them. */
    std::unique_ptr<PipelineMetadataExchange> metadataRows(const ForwardInput &input) const;
    /** @brief Retain a checked subregion from one canonical arena tensor. */
    PipelineMetadataBank arenaBank(BufferId id, const void *pointer, size_t elements) const;
    ILocalTPContext *context_;
    const int participant_;
    TensorBase &hidden_;
    const int width_;
    const DeviceId device_;
    const PipelineNativeDomain membership_;
    const std::optional<FollowerState> follower_;
    const std::optional<CapturedDomain> domain_;
    BufferArena *arena_ = nullptr; ///< Borrowed only during graph construction, never a second allocator.
    TransferEngine *transfer_ = &TransferEngine::instance(); ///< Same public authority as channel admission.
    /** Write-once wire identity; an expired participant lease cannot be resurrected. */
    std::optional<PublicationBanks> publication_banks_;
    std::weak_ptr<PipelinePublicationTransport> publication_transport_;
};
}
