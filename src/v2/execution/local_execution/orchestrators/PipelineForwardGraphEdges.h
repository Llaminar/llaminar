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
#include "PipelineMaintenanceAdmission.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "backends/DeviceId.h"
#include "execution/compute_stages/stages/MTPVerifierPreparationStage.h"
#include "execution/compute_stages/stages/MTPSpeculativeStatePublicationStage.h"
#include "PipelineMetadataExchange.h"
#include <optional>
#include <memory>
#include <string>
#include <string_view>
#include <cstdint>

namespace llaminar2
{
class ILocalTPContext;
class TensorBase;
class ComputeGraph;
class DeviceGraphExecutor;
class IGPUGraphCapture;
class IWorkerGPUContext;
class BufferArena;
class PipelineActivationFanout;
class Qwen4ExpFixedHeadPipelinePort;
struct PipelinePublicationTransport;
struct ForwardInput;
enum class ForwardExecutionRole : uint8_t;
namespace test { struct PipelinePublicationCaptureReplayAccess; }

/** @brief Frozen, allocation-free topology contributor; never a request ledger. */
class PipelineForwardGraphEdges final
{
public:
    /** @brief Complete publication topology, derived from frozen domain ownership. */
    enum class PublicationOrder { LocalCommit, CommitThenExchange, ExchangeThenCommit };
    /** @brief Four committed-state banks and optional exact tail-authored cadence evidence fields. */
    struct PublicationBanks
    {
        int32_t *restore_rows = nullptr;
        int32_t *cached_tokens = nullptr;
        int32_t *accepted_counts = nullptr;
        int32_t *healthy = nullptr;
        std::array<int32_t *, 3> committed_round_evidence{}; ///< Exact original health/output/leading compact fields.
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
        std::shared_ptr<PipelineActivationFanout> activation_fanout; ///< This receiving TP member's actual metadata owner.
        ILocalTPContext *native_context = nullptr; ///< Original homogeneous pipeline membership, distinct from stage TP.
        int native_participant = -1;
        bool carries_moe_commit_evidence = false; ///< All relays retain only the original tail's needed compact fields.
        std::shared_ptr<PipelineMaintenanceAdmission> maintenance_admission; ///< Original complete native budget recording owner.
        std::shared_ptr<PipelineNativeByteChannel> native_activation_in, native_activation_out;
        std::shared_ptr<PipelineNativeByteChannel> native_metadata_in, native_metadata_out;
        std::shared_ptr<const Qwen4ExpFixedHeadPipelinePort> fixed_head_port;

    };
    /**
     * @brief Ordinary forward-only rank-stage ports; no reverse generation/MTP authority.
     *
     * Indices are original global stage coordinates, independent of this rank's
     * local participant list. Existing native TP membership remains local to
     * the stage; the stage leader alone owns the original inter-rank endpoint.
     */
    struct RankBoundary
    {
        size_t stage_index = 0;
        size_t stage_count = 0;
        ILocalTPContext *context = nullptr;
        int participant = 0;
        std::shared_ptr<CapturedTransferChannel> activation_in;
        std::shared_ptr<CapturedTransferChannel> activation_out;
        std::shared_ptr<PipelineActivationFanout> activation_fanout;
    };
    /** @brief Install ordinary forward-only ports before any serving capture.
     * @param device Exact original local GPU.
     * @param boundary Complete original global stage/endpoint roles.
     * @param arena The same local participant's persistent hidden/count owners.
     * @param width FP32 residual row width, never a TP shard width.
     * @param transfer Canonical movement/publication authority.
     * @throws std::invalid_argument For missing, foreign or incomplete ports. */
    PipelineForwardGraphEdges(DeviceId device, RankBoundary boundary, BufferArena &arena, int width,
        TransferEngine &transfer = TransferEngine::instance());

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
     * @param original_setup_stream Original retained participant stream for a passive native boundary.
     * @param retained_banks Original storage leases for captured-domain metadata.
     * @return The participant-owned topology/storage lease, or empty on failure.
     * Local-only terminal siblings retain their banks without recording wire work.
     * The DGO retires this lease after its parents and before releasing its arena;
     * the longer-lived rank topology keeps only a weak reference.
     * Repeated identical preparation reuses it; rebinding is a fatal caller error. */
    std::shared_ptr<PipelinePublicationTransport> materializePublicationTransport(PublicationBanks banks, DeviceGraphExecutor &executor,
        IDeviceContext &context, IWorkerGPUContext &gpu, void *original_setup_stream,
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
     * @return True for prefill/MTP and complete ordinary scalar invocations.
     * @throws std::invalid_argument for an unknown ownership policy or a parent
     *         claim outside exact setup-only native ordinary decode.
     * A resident native generation child explicitly delegates activation edges
     * to its complete parent. Request-admitted history forwards enclose their
     * own edges; heterogeneous domains always retain their existing enclosure. */
    [[nodiscard]] bool encloses(const ForwardInput &input) const;
    /** @brief Enclose one local prefill, condition or verifier DAG in native collectives.
     * @param graph Participant-local model DAG before executor preparation.
     * @param input Exact physical row geometry and local buffer bindings.
     * @throws std::invalid_argument if geometry, semantic role or ownership changed.
     * Validation precedes graph mutation. A malformed claimed role fails rather
     * than silently omitting its required transport. */
    void append(ComputeGraph &graph, const ForwardInput &input) const;
    /** @return Original fixed-head port only on the actual PLE consumer stage. */
    std::shared_ptr<const Qwen4ExpFixedHeadPipelinePort> fixedHeadPLEConsumer() const noexcept;
    /** @brief Validate a follower's captured receive before the forward prelude.
     * @param input Exact invocation binding the local collective destination.
     * @throws std::invalid_argument for terminal ownership or malformed rows.
     * This performs no enqueue or graph mutation. The same immutable geometry
     * validation is used when constructing the graph, so replay admission cannot
     * quietly accept a role that its retained topology does not implement. */
    void validateFollowerMTPInput(const ForwardInput &input) const;
    /** @brief Bind the exact retained metadata producer and its publication child as one closed composition.
     * Only an incoming committed-round transport may authenticate META as a prior-child input.
     * @param graph Original publication graph. @param stage Its exact local publication stage.
     * @param evidence Same arena-owned compact metadata tensor consumed by that stage.
     * @param memory Original source PMA, never an independently recreated authority.
     * @param workspace_generation Original consumer workspace identity.
     * @param worker Original prepared GPU worker. @param stream Exact local parent stream.
     * @return Composer retaining the actual wire source until the final native parent retires.
     * @throws std::exception For absent, stale, foreign or differently ordered producers.
     */
    DeviceGraphExecutor::RetainedParentCompositionHook publicationComposer(ComputeGraph &graph,
        MTPSpeculativeStatePublicationStage *stage, std::shared_ptr<TensorBase> evidence,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory, uint64_t workspace_generation,
        IWorkerGPUContext &worker, void *stream) const;
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
    /** @brief Authenticate the exact full-generation port without exposing private pointers.
     * @param global_stage Original global stage index, not local participant index.
     * @param global_count Original complete stage count.
     * @param device Exact local native participant.
     * @param context Original local TP context, null only for a degree-one stage.
     * @param participant Exact ordered local TP member.
     * @return True only for a complete CapturedDomain; ordinary RankBoundary ports cannot certify generation. */
    bool hasOriginalGenerationBoundary(size_t global_stage, size_t global_count, DeviceId device,
        ILocalTPContext *context, int participant) const noexcept
    {
        return domain_ && !rank_forward_only_ &&
            (!context_ || domain_->stage_index == 0 || domain_->activation_fanout || domain_->native_activation_in) &&
            domain_->stage_index == global_stage &&
            domain_->stage_count == global_count && device_ == device &&
            context_ == context && participant_ == participant;
    }
    /** @return A compiled inter-vendor boundary, not a backend capability advertisement. */
    bool hasHeterogeneousBoundary() const noexcept { return domain_ && !domain_->native_context; }
    /** @return Any explicitly typed domain boundary, including original homogeneous native edges. */
    bool hasDomainBoundary() const noexcept { return domain_.has_value(); }
    /** @return Whether this original domain relays exact MTP committed-round evidence to maintenance consumers. */
    bool carriesMoECommitEvidence() const noexcept { return domain_ && domain_->carries_moe_commit_evidence; }
    /** @return Whether this original follower must precede its grouped verifier with native budget admission. */
    bool requiresMaintenanceAdmission(ForwardExecutionRole role) const noexcept;
    /** @return Original budget source recording owner, retained through every compiled parent. */
    std::shared_ptr<PipelineMaintenanceAdmission> maintenanceAdmission() const noexcept {
        return domain_ ? domain_->maintenance_admission : nullptr;
    }
    /** @return Original flattened native member coordinate for capture composition. */
    std::size_t maintenanceParticipant() const {
        if (!domain_ || !domain_->maintenance_admission || domain_->native_participant < 0)
            throw std::logic_error("Pipeline maintenance composition has no original member");
        return std::size_t(domain_->native_participant);
    }
private:
    friend struct test::PipelinePublicationCaptureReplayAccess;
    /** @brief Resolve validated domain roles before building a wire capture. */
    static PublicationOrder domainPublicationOrder(const CapturedDomain &domain) noexcept;
    /** @brief Join one original cohort's cold begin/end phases around its active wire unit.
     * The executor invokes active boundaries at its actual native begin/end. A
     * passive member uses its existing setup stream and never creates a graph.
     */
    class NativePublicationCaptureWave final
    {
    public:
        /** @brief Bind the full original native context and its flattened member coordinate. */
        NativePublicationCaptureWave(ILocalTPContext &context, int participant);
        NativePublicationCaptureWave(const NativePublicationCaptureWave &) = delete;
        NativePublicationCaptureWave &operator=(const NativePublicationCaptureWave &) = delete;
        /** @brief Authenticate and join the executor's exact cold publication boundary. */
        bool activeBoundary(std::string_view boundary, void *stream);
        /** @brief Join both original phases on the actual retained passive setup stream. */
        bool joinPassive(void *stream);
        /** @return Whether both boundaries completed successfully in order. */
        bool complete() const noexcept { return state_ == State::Complete; }
    private:
        enum class State { Begin, End, Complete, Failed };
        /** @brief Advance one expected phase on the same exact explicit stream. */
        bool arrive(State phase, void *stream);
        ILocalTPContext &context_;
        int participant_;
        void *stream_ = nullptr;
        State state_ = State::Begin;
    };
    /** @brief Keep ordinary forward ports distinct from complete generation metadata ownership. */
    enum class BoundaryCoverage { ForwardOnly, CompleteGeneration };
    /** @brief Shared cold validation; only typed public port constructors select coverage. */
    PipelineForwardGraphEdges(DeviceId device, CapturedDomain domain, BufferArena &arena, int width,
        std::optional<FollowerState> follower, TransferEngine &transfer, BoundaryCoverage coverage);
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
    const bool rank_forward_only_ = false; ///< Forward-only ports never certify generation or MTP.
    BufferArena *arena_ = nullptr; ///< Borrowed only during graph construction, never a second allocator.
    TransferEngine *transfer_ = &TransferEngine::instance(); ///< Same public authority as channel admission.
    /** Write-once wire identity; an expired participant lease cannot be resurrected. */
    std::optional<PublicationBanks> publication_banks_;
    std::weak_ptr<PipelinePublicationTransport> publication_transport_;
};
}
