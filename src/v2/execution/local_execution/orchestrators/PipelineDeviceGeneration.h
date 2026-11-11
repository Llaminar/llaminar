/**
 * @file PipelineDeviceGeneration.h
 * @brief Rank-local composition of participant-owned pipeline generation graphs.
 *
 * The vocabulary stage alone admits a response budget and runs a sampler.
 * Earlier stages receive only a continuation command and token; their layer
 * caches remain independent owners. Native collectives connect homogeneous
 * participants. Cross-vendor domain leaders use captured TransferEngine channels,
 * while the terminal domain's authenticated ticket selects complete local
 * transactions. Neither case creates a nested multi-device graph or host sampler.
 */
#pragma once
#include "IInferenceRunner.h"
#include "PipelineForwardGraphEdges.h"
#include "execution/mtp/MTPMainForwardPolicy.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include <functional>
#include <array>
#include <memory>
#include <span>
#include <vector>

namespace llaminar2
{
class DeviceGraphOrchestrator;
class IBackend;
class LocalTPContext;
class ILocalTPContext;
class CapturedTransferChannel;
class TPWorkerPool;
class PipelineForwardGraphEdges;
class PhysicalMemoryAuthority;
struct ClusterInventory;
struct RankStageAction;
struct GlobalPPTopology;
struct GlobalPPRankPlan;

/** @brief Complete captured pipeline lifecycle, borrowing already-admitted stages. */
class PipelineDeviceGeneration final : public IMoEOverlayInferenceTransactionExecutor
{
public:
    /** @brief Borrow one runner and its exact action in the original global rank plan. */
    struct RankStageBinding
    {
        IInferenceRunner *runner = nullptr; ///< Global registry retains the original runner.
        const RankStageAction *action = nullptr; ///< Exact address returned by original executeStages().
    };
    /**
     * @brief Retain only this real rank's stages from the full original global topology.
     * @param topology Canonical inventory-bound complete layer/rank topology.
     * @param rank_plan Original local plan retained by the enclosing Global owner.
     * @param stages Exact local EXECUTE action pointers and their original runners, in plan order.
     * @param edges Complete installed port owners in local TP order; moved only after all validation and setup succeed.
     * @throws std::exception For incomplete membership, role, capture, PMA or edge identity.
     *
     * No remote DGO is represented locally. The tail may be absent; APIs needing
     * a local sampler reject that absence rather than substituting a local last
     * stage. Construction itself admits no request and advertises no MTP support.
     */
    PipelineDeviceGeneration(const GlobalPPTopology &topology, const GlobalPPRankPlan &rank_plan,
        std::span<const RankStageBinding> stages,
        std::vector<std::unique_ptr<PipelineForwardGraphEdges>> &edges);
    /** @brief Borrowed original stage endpoint for cold rank transport composition. */
    struct RankStageEndpoint
    {
        DeviceId leader = DeviceId::invalid(); ///< Original domain leader, never device zero by assumption.
        int row_capacity = 0; ///< Actual planner-admitted activation rows in its original arena.
        std::shared_ptr<PhysicalMemoryAuthority> memory; ///< Same sole rank allocator as every local TP member.
        void *setup_stream = nullptr; ///< Exact original generation/setup stream, never default.
        TensorBase *hidden = nullptr; ///< Stable arena-owned FP32 bank, borrowed through original runner lifetime.
        int verifier_rows = 0; ///< Original retained target-query rows, one for ordinary-only storage.
        int residual_width = 0; ///< Actual full residual row width; HC streams are not embedding-width rows.
    };
    /**
     * @brief Inspect one real local stage before a global transport can bind it.
     * @param runner Original single-device/local-TP stage, retained by GlobalOrchestrator.
     * @param action Canonical EXECUTE action with exact ordered devices and layer roles.
     * @return Complete original leader/arena/PMA projection, without allocation or native calls.
     * @throws std::invalid_argument For absent/foreign geometry or TP membership.
     * @throws std::logic_error For a missing original arena/PMA/stream or a late edge installation.
     */
    static RankStageEndpoint inspectOriginalRankStage(IInferenceRunner &runner, const RankStageAction &action);
    /**
     * @brief Freeze ordinary global-stage activation edges on its original local members.
     * @param runner Original stage runner retained beyond all returned edge owners.
     * @param action Original EXECUTE action authenticated by the cold projection.
     * @param boundary Global coordinates and channel owners, with neutral local TP fields.
     * @return One edge owner per actual local participant, in original device order.
     * @throws std::invalid_argument For changed global roles or caller-supplied TP identity.
     * @throws std::logic_error For foreign/late ownership or a failed original bank allocation.
     *
     * The caller retains these owners until after its stage runners retire. Every
     * edge is constructed before any DGO pointer is published. Partial allocation
     * failure remains with the original arena/PMA; it never installs a dangling
     * edge. This activation-only operation grants no generation/MTP authority.
     */
    static std::vector<std::unique_ptr<PipelineForwardGraphEdges>> installOriginalRankStageForwardEdges(
        IInferenceRunner &runner, const RankStageAction &action,
        const PipelineForwardGraphEdges::RankBoundary &boundary);
    /**
     * @brief Freeze complete original activation/reverse-metadata ports before cohort capture.
     * @param runner Original single-device or local-TP stage.
     * @param action Exact canonical EXECUTE action for its global stage.
     * @param boundary Global roles and both original directed channel pairs, with neutral TP fields.
     * @return Complete per-participant owners, retained until after every stage graph retires.
     * @throws std::exception For late/foreign ownership or absent original follower checkpoint.
     *
     * This is the same full CapturedDomain port used by local heterogeneous
     * pipelines. It changes neither wire layouts nor sampler ownership. Merely
     * installing ports does not advertise or admit a generation lifecycle.
     */
    static std::vector<std::unique_ptr<PipelineForwardGraphEdges>> installOriginalRankStageGenerationEdges(
        IInferenceRunner &runner, const RankStageAction &action,
        const PipelineForwardGraphEdges::CapturedDomain &boundary);
    /** @return Whether every declared layer domain is GPU-resident.
     * This is topology classification only; construction still validates every
     * role and backend capability, with no alternative on admission failure. */
    static bool isGPUStageSet(std::span<const std::unique_ptr<IInferenceRunner>> stages);
    /** @brief Freeze ordered domains and their exact native/cross-vendor edges.
     * @param stages Stable participant runners, in layer order, tail last.
     * @param original_mpi Actual admitted parent context, retaining canonical discovery/execution rank membership.
     * @throws std::invalid_argument for incomplete or unsupported composition. */
    explicit PipelineDeviceGeneration(std::span<const std::unique_ptr<IInferenceRunner>> stages,
        std::shared_ptr<IMPIContext> original_mpi);
    /** @brief Join accepted one-shot work before releasing its borrowed native recordings. */
    ~PipelineDeviceGeneration();
    /** @brief Freeze native prefill edges and prepare all stage graph families.
     * @param plan Canonical bucket family shared by every participant.
     * @return Whether all stage families materialized before request admission. */
    bool prepareServing(const ServingGraphFamilyMaterializationPlan &plan);
    /** @return Whether the real global vocabulary domain is owned in this process. */
    bool hasLocalTail() const noexcept { return terminalIndex() < stages_.size(); }
    /** @return Whether native prefill edge ownership has been frozen. */
    bool prefillPrepared() const noexcept { return forward_edges_.size() == stages_.size(); }
    /** @return Whether the installed terminal and every local follower own publication. */
    bool supportsMTPPublication() const;
    /** @brief Admit request inputs once per stage and submit captured chunk DAGs.
     * @param tokens Immutable unprocessed token span (the suffix after a prefix hit).
     * @param count Rows in that span, excluding restored tokens and bucket padding.
     * @param policy Root-owned chunk geometry, never re-sliced by a child.
     * @param pad_token Padding value bound by the serving family.
     * @param allow_padding Whether the caller admits padded native execution.
     * @return Whether every stage submitted its complete schedule. */
    bool prefill(const int *tokens, int count, const PrefillChunkSchedulerPolicy &policy,
        int pad_token, bool allow_padding);
    /** @brief Admit one ordinary input token through all captured pipeline domains.
     * @param token_id Request input consumed with scalar decode arithmetic.
     * @return Whether every domain submitted its retained input transaction.
     * This handles an initial or restored-prefix input; generation remains
     * owned by the admitted device controller. */
    bool forwardDecodeInput(int32_t token_id);
    /** @brief Replay one restored-prefix bridge through every captured stage edge.
     * @param request The exact restored history and single new request token.
     * @return Whether all domains submitted their participant-local bridge.
     * Each domain owns its TP fanout; the pipeline admits all domains together
     * so activation transfers have both endpoints without host staging. */
    bool restoredPrefixMTPDecodeBridge(const RestoredPrefixMTPDecodeBridgeRequest &request);
    /** @brief Submit an initial scalar condition on every main-model participant.
     * @param purpose Immutable commit ownership shared with serving-time capture.
     * @param terminal Exact tail operation which owns/publishes the input mailbox.
     * @return Whether all participants submitted their native captured forward.
     * This is transaction admission, not a host per-token generation loop. */
    bool forwardMTPCondition(MTPConditionForwardPurpose purpose,
        const std::function<bool(IInferenceRunner &)> &terminal);
    /** @brief Submit the initial grouped verifier with one terminal input authority.
     * @param logical_rows Tail transaction's positive logical verifier extent.
     * @param terminal Exact tail verifier operation; followers receive its resident rows.
     * @return Whether all stages submitted the same retained physical geometry. */
    bool forwardMTPVerifier(int logical_rows, const std::function<bool(IInferenceRunner &)> &terminal);
    /**
     * @brief Submit a ticket-named exact verifier on this original rank's nonterminal stages.
     * @param physical_rows Exact retained width authenticated by the original ticket protocol.
     * @return True after every original local verifier was enqueued.
     * This does not request or fabricate a terminal outcome on this process.
     */
    bool forwardOriginalRankFollowerVerifier(int physical_rows);
    /**
     * @brief Receive and publish terminal-authored acceptance through original local banks.
     * @param physical_rows The same exact verifier width submitted on these stages.
     * @param error Receives original participant/publication failure detail.
     * @return True only after each follower's original publication was submitted.
     * No remote device outcome handle crosses this interface.
     */
    bool publishOriginalRankFollowerState(int physical_rows, std::string *error);
    /** @brief Publish one tail outcome into each participant's own KV/recurrent state.
     * @param request Tail-owned immutable outcome handle and publication geometry.
     * @param error Receives the first participant's publication diagnostic.
     * @return Whether every captured commit was submitted, without host outcome reads. */
    bool publishMTPOutcome(const DeviceSpeculativePublicationRequest &request, std::string *error);
    /** @brief Consume one original tail mailbox through a complete no-sampling PP transaction.
     * @param logical_state Current immutable tail-owned request-zero mailbox.
     * @param request_index Only the scalar request-zero lane is admitted.
     * @return Whether every original participant committed exactly one model row.
     * Native PP uses its command collective; captured domains reuse their
     * existing scalar metadata/activation bodies and mirrored tail publishers.
     * Both paths retire every physical model row without a sampler or response. */
    bool advanceOrdinaryCondition(const DeviceResidentLogicalSequenceStateHandle &logical_state, int request_index);
    /**
     * @brief Admit the original command on a rank without the vocabulary owner.
     * @param request Complete immutable request policy, including original prefix provenance.
     * @param sampling The same frozen sampler topology selected by the real tail.
     * @param error Receives the first rejected ownership/admission diagnostic.
     * @return True only after local admission and ordinary retained-body setup complete.
     * No sampler, response ledger, remote pointer or mutable model state is installed.
     */
    bool beginOriginalRankFollowerCommand(const DeviceGenerationAdmissionRequest &request,
        DeviceGenerationSamplingMode sampling, std::string *error);
    /**
     * @brief Execute exactly one original protocol-authenticated local transaction.
     * @param ticket Immutable source ticket naming a retained condition/ordinary/verifier body.
     * @param error Receives submission, identity or actual event-completion failure.
     * @return False: GPU callers must use the typed asynchronous submission API.
     * The source placement epoch is not a local placement/workspace generation.
     */
    bool executeMoEOverlayInferenceTransaction(const MoEOverlayInferenceTransactionTicket &ticket,
        std::string *error = nullptr) override;
    /** @brief Submit native work and return its retained poll-only completion lease.
     * @param ticket Exact authenticated retained graph and live geometry.
     * @param error Receives submission or native identity failures.
     * @return Immediate metadata completion or a pending native terminal lease. */
    MoEOverlayInferenceAsyncSubmission submitMoEOverlayInferenceTransaction(
        const MoEOverlayInferenceTransactionTicket &ticket, std::string *error = nullptr) override;
    /** @brief One original event's host submission ownership; allocated once with the cohort. */
    struct RankTransactionTerminal
    {
        void *stream = nullptr; ///< Exact submitted parent stream, never inferred at acknowledgment.
        uint64_t workspace = 0; ///< Original local pointer/workspace identity.
        uint64_t session = 0; ///< Original local request/reset identity.
        MoEOverlayInferenceGraphRole role = MoEOverlayInferenceGraphRole::None; ///< Authenticated semantic body.
        bool recorded = false; ///< True only after original native event publication.
    };
    /** @brief Record ACK completion after an original submitted producer, without consuming model state.
     * @param backend Original device backend; a foreign backend is rejected.
     * @param device Exact original physical participant owning event and producer.
     * @param event Persistent native completion slot allocated at setup.
     * @param receipt Exact stream, graph role and lifecycle recorded by the submitted body.
     * @param workspace Current original participant workspace generation.
     * @param session Current original participant request generation.
     * @param error Receives the precise receipt or backend recording failure.
     * @return True after recording only the native event on its original producer stream.
     * An ACK observes ordering. It never borrows live model state, consumes
     * accepted publication, records another producer, or exports a host result.
     */
    static bool recordOriginalRankProducerCompletion(IBackend &backend, DeviceId device, void *event,
        const RankTransactionTerminal &receipt, uint64_t workspace, uint64_t session, std::string *error = nullptr);

    /** @brief Allocate bounded native completion slots during cold serving setup.
     * @param slot_count Exact original protocol ring capacity.
     * @throws std::logic_error If the owner is not cold or already initialized. */
    void prepareOriginalRankCompletionSlots(std::size_t slot_count);
    /** @brief Join retained local completion leases only at the outer command terminal.
     * @param error Receives backend, identity or deadline failure.
     * @return True after every retained source lease has real native completion. */
    bool retireOriginalRankCompletions(std::string *error);
    /**
     * @brief Validate final local KV and retire a successful remote-tail command.
     * @param source_terminal_commit_count Actual tail terminal publication count, never ticket cadence.
     * @param expected_source_cached_tokens Actual tail terminal KV observation, validation-only.
     * @param error Receives local terminal or independent-KV validation failure.
     * @return True after original local storage/readiness retirement and counter publication.
     */
    bool finishOriginalRankFollowerCommand(int source_terminal_commit_count,
        int expected_source_cached_tokens, std::string *error);
    /** @return Completed actual tail KV observation; absent before a successful local finish. */
    std::optional<int> originalTerminalCachedTokens() const noexcept { return terminal_cached_tokens_; }
    /** @brief Bind the sole source publication edges around complete hosted transactions.
     * @param publish Publish every original remote lane before any source participant submits.
     * @param retire Retire the previous remote transaction after authenticated source completion.
     * @throws std::invalid_argument For non-source ownership or absent callbacks.
     */
    void bindOriginalRankTransactionCallbacks(
        std::function<bool(MoEOverlayInferenceGraphRole, int, int)> publish,
        std::function<bool(uint64_t)> retire);
    /** @return Existing source request/reset epoch; never an independent service generation. */
    uint64_t originalRankRequestGeneration() const;
    /** @brief Project command ordering from the real original terminal source owner.
     * @param command_id Source OR's monotonic command identifier.
     * @return Existing terminal request epoch and its explicit placement authority.
     * Independent participants retain their own native capture/session/workspace identity.
     * @throws std::logic_error For missing source/graph ownership or invalid command provenance.
     */
    MoEOverlayInferenceCommandIdentity originalRankCommandIdentity(uint64_t command_id) const;
    /** @brief Record original condition/verifier/publication native producers without waiting.
     * @param role Exact original semantic body whose producer streams must be retained.
     * @param error Receives stream, workspace or native publication failures.
     * @return Whether the command now retains every exact native completion lease.
     */
    bool completeOriginalRankSourceTransaction(MoEOverlayInferenceGraphRole role, std::string *error);
    /** @brief Admit the tail's policy and each follower's local event handoff. */
    bool begin(const DeviceGenerationAdmissionRequest &request);
    /** @brief Compile or reuse all complete parents before submitting any work. */
    /** Authenticate an already retained hosted family for the tail-owned ContinueResponse admission. */
    bool prepareContinuation(DeviceGenerationSamplingMode sampling);
    bool materialize(int requests, int depth, DeviceGenerationLoopTopology topology,
        DeviceGenerationSamplingMode sampling);
    /** @brief Enqueue every parent before any terminal observation. */
    bool launch();
    /** @brief Return only the tail response and validate independent stage KV positions. */
    bool finish(DeviceGenerationTerminalResult *result);
    /** @return Whether this owner has an admitted, not-yet-retired request. */
    bool active() const noexcept { return phase_ != Phase::Idle; }

private:
    /** @brief Enclosing owner orders parent retirement before its original PP stages. */
    friend class RankOrchestrator;
    friend class GlobalOrchestrator;
    /**
     * @brief Retire one-shot parents and setup workers while borrowed DGO owners remain alive.
     * @details Idempotent exclusive teardown. Every accepted outstanding parent
     * joins its original exact terminal event before its executable is destroyed.
     * Without a derived maintenance family, reverse member destruction retains
     * the established collective/edge order. With that family, all stage graph
     * borrowers retire together before its controller workspace leases and
     * native scalar roots are released; no stage may publish reusable backing
     * while a source from another original stage still retains it.
     * Only Rank destruction and this owner's destructor may call this boundary.
     */
    void retireForwardOnlyBeforeStageRelease() noexcept;
    /** @brief Frozen layer-domain role; its runner remains the TP authority. */
    struct Domain
    {
        IInferenceRunner *runner;
        ILocalTPContext *context;
        size_t first;
        size_t count;
        size_t global_stage; ///< Original layer-domain index, independent of rank/local position.
        bool head; ///< This domain owns the model's embedding boundary.
        bool tail; ///< This domain owns the only vocabulary/sampler boundary.
    };
    /** @brief Two opposite directions, shared by every retained graph family. */
    struct Boundary
    {
        std::shared_ptr<CapturedTransferChannel> activation;
        std::shared_ptr<CapturedTransferChannel> metadata;
        std::shared_ptr<PipelineNativeByteChannel> native_activation, native_metadata;
    };
    /** @brief Exact installed composition, not a runtime failure alternative. */
    enum class Transport { NativePipeline, NativeDomains, CapturedDomains };
    bool follower_moe_maintenance_ = false; ///< Immutable active upstream native controller demand.
    bool follower_moe_commit_evidence_ = false; ///< Immutable topology demand for exact tail-authored compact fields.
    /** @brief Retire each original HIP follower cadence transaction before its next generation body. */
    bool advanceHostedFollowerMaintenance();
    /** @brief Submit one complete original follower parent and publish its actual captured epoch release. */
    bool submitHostedFollowerCapture(size_t participant, const IGPUGraphCapture &capture);
    /** @return This participant's complete retained CUDA cadence transaction, or null for inactive placement. */
    const IGPUGraphCapture *nativeStageMaintenance(size_t participant) const;
    /** @return Frozen domain containing one flattened participant. */
    const Domain &domainFor(size_t participant) const;
    /** @return First participant of the terminal domain (not the last physical GPU). */
    size_t terminalIndex() const noexcept;
    /** @return The actual local terminal owner; rejects a remote tail rather than substituting a local stage. */
    const Domain &tailDomain() const;
    /** @return Original immutable request policy; never a remote response/control pointer. */
    const std::optional<DeviceGenerationAdmissionRequest> &originalAdmission() const;

    /** @return Whether a participant belongs to the one terminal authority. */
    bool terminalParticipant(size_t index) const noexcept
    { return index < stages_.size() && index >= terminalIndex(); }
    /**
     * @brief Order original terminal events after each actual participant's submitted work.
     * @return True only after all recorded exact events complete under the collective deadline.
     * Uses the existing admitted terminal event and non-consuming producer dependencies;
     * no stream/device synchronization, mutable device readback or event allocation occurs.
     */
    bool joinOriginalRankTransaction(std::string *error);
    /** @brief Authenticate and enqueue one exact remote transaction without host completion waits. */
    bool enqueueOriginalRankTransaction(const MoEOverlayInferenceTransactionTicket &ticket, std::string *error);
    /** @brief Observe only completed original local KV terminals for a command with no response ledger.
     * @param error Receives the first native terminal or independent-local-KV discrepancy.
     * @return Actual common completed local KV count, never execution input. */
    std::optional<int> observeOriginalRankKVTerminal(std::string *error);
    /** Immutable policy admission receiver, invoked only by an authenticated Admission ticket. */
    std::function<bool(std::string *)> rank_receive_admission_;
    /** @brief Submit one exact already-materialized ordinary or verifier follower body. */
    bool submitOriginalRankFollowerBody(int physical_rows, std::string *error);
    /** @brief Abort every installed native communicator after a fatal partial submission. */
    void abortParticipants();
    /** @brief Materialize admitted channels, before any graph borrows their identities. */
    void prepareDomainBoundaries();
    /** @brief Submit complete ordinary follower transactions around terminal-domain tickets. */
    bool launchHostedDomainTransactions();
    /** @brief Host submission lifecycle only; no mirrored device decision state. */
    enum class Phase { Idle, Admitted, Materialized, Submitted, Failed };
    /**
     * @brief Admit one original nonterminal participant without tail-owned response state.
     * @param index Exact flattened local participant, already role-validated by begin().
     * @param request The same immutable request declaration admitted by the tail.
     * @return True only after original reset/readiness, storage and admission publication succeed.
     *
     * This is the shared follower body for cohort lifecycle factoring. It never
     * initializes a sampler, reads a remote device pointer or resets a received
     * command. The enclosing owner retains the absorbing partial-failure phase.
     */
    bool admitOriginalFollower(size_t index, const DeviceGenerationAdmissionRequest &request);
    /**
     * @brief Reuse the original accepted-state publication body for one local follower.
     * @param index Exact local participant with an already-submitted verifier.
     * @param physical_rows Its exact original retained grouped width.
     * @param error Original participant diagnostic destination.
     * @return True after checkpoint/publication/readiness submission succeeds.
     */
    bool publishOriginalFollowerMTPState(size_t index, int physical_rows, std::string *error);
    /** @brief Persistent worker fanout for setup, never a per-token dispatch. */
    bool prepareParticipants(DeviceGenerationSamplingMode sampling);
    /** @brief Submit one request input through every domain's existing TP authority.
     * @param forward Exact ordinary or restored-MTP scalar entry point.
     * @return Whether all domains submitted their captured activation edges.
     * Upstream domains retire their unused logits handoffs before reuse. */
    bool forwardMainInput(const std::function<bool(IInferenceRunner &)> &forward);
    /** @brief Fan out one typed main-forward transaction without duplicating tail state.
     * @param policy Frozen local condition/verifier topology.
     * @param terminal Sole terminal input/outcome owner.
     * @return False with absorbing failure state if any participant fails. */
    bool forwardMTP(const MTPMainForwardPolicy &policy,
        const std::function<bool(IInferenceRunner &)> &terminal);
    /** @brief Record the exact native command and adjacent activation edges. */
    bool composeParticipants();
    /** @brief Borrow the tail policy and the followers' exact verifier/commit captures.
     * @param depth Fixed depth or dynamic policy's maximum captured depth.
     * @param topology Must match the tail's admitted device controller.
     * @param sampling Tail's greedy/stochastic outcome policy.
     * @return Whether every participant has a complete current transaction. */
    bool prepareSpeculativeParticipants(int depth, DeviceGenerationLoopTopology topology,
        DeviceGenerationSamplingMode sampling);
    /** @brief Compose complete follower transactions and native continuation commands.
     * The tail retains the sole depth selector; followers execute a fixed physical
     * verifier envelope whose logical row count arrives through captured edges. */
    bool composeSpeculativeParticipants();
    /** @brief Submit HIP follower transactions only from authenticated tail tickets. */
    bool launchHostedSpeculativeTransactions();
    /** @brief Submit complete HIP transactions selected solely by tail tickets.
     * Every follower is submitted before the next ticket is observed. The host
     * neither samples tokens nor inspects mutable model/cache state. */
    bool launchHostedTransactions();
    /** @brief One separately priced complete parent per original physical participant. */
    struct ForwardOnlyFamily;
    /** @brief Freeze source recordings, native edges and no-sampling publication at setup. */
    bool prepareForwardOnlyFamily();
    /**
     * @brief Retain one sampler-free transaction per captured-domain participant/program.
     * @return Whether every exact local model/transport identity and terminal publisher is retained.
     * Existing resident forwards already own scalar metadata and activation
     * transfers. Only terminal TP participants publish complete frontiers;
     * upstream participants retain their partial-row writer/event ownership.
     * No tensor payload, transport channel or sampler recording is created.
     */
    bool prepareCapturedDomainForwardOnlyFamily();
    /** @brief Authenticate all original identities before request admission or submission. */
    bool forwardOnlyFamilyCurrent() const;
    /** @brief Observe only fixed terminal control/KV metadata and retire the actual admission. */
    bool finishForwardOnly();
    /** @brief Launch the admitted sampler-free parent on every original local participant.
     * @return True only after exact local native terminal observation and admission retirement. */
    bool submitForwardOnly();
    /** One complete forced transaction per program; all borrow the same original state banks. */
    std::array<std::unique_ptr<ForwardOnlyFamily>, kSparseQueryExecutionProgramCount> forward_only_families_;
    /** @return Family proved for this exact forced/outer-response transaction, never budget0 inference. */
    std::unique_ptr<ForwardOnlyFamily> &forwardOnlyFamily();
    /** @return Same retained family for immutable identity/lifetime checks. */
    const std::unique_ptr<ForwardOnlyFamily> &forwardOnlyFamily() const;
    std::vector<DeviceGraphOrchestrator *> stages_;
    std::vector<Domain> domains_;
    size_t global_stage_count_ = 0; ///< Complete original topology, never the local stage-vector length.
    const GlobalPPTopology *rank_topology_ = nullptr; ///< Borrowed original declaration, retained by Global beyond this owner.
    const GlobalPPRankPlan *rank_plan_ = nullptr; ///< Same exact rank namespace; no synthetic remote participant.
    std::optional<DeviceGenerationAdmissionRequest> rank_admission_; ///< Immutable original request policy only.
    std::function<bool(MoEOverlayInferenceGraphRole, int, int)> rank_publish_transaction_; ///< Source-only pre-submission protocol edge.
    std::function<bool(uint64_t)> rank_retire_transaction_; ///< Source-only authenticated completion edge.
    std::optional<DeviceGenerationSamplingMode> rank_sampling_; ///< Same original immutable command capture policy.
    bool rank_initial_verifier_complete_ = false; ///< Original command lifecycle only; no verifier/model state mirror.
    std::optional<int> terminal_cached_tokens_; ///< Completed terminal observation only; never execution input.
    std::vector<RankTransactionTerminal> rank_transaction_terminals_;
    struct RankTransactionCompletion;
    std::vector<std::shared_ptr<RankTransactionCompletion>> rank_completion_slots_;
    std::vector<std::shared_ptr<IMoEOverlayInferenceCompletionFence>> rank_source_completions_;
    /** @brief Record exact producer events and transfer their immutable ownership without waiting. */
    std::shared_ptr<IMoEOverlayInferenceCompletionFence> recordOriginalRankTransaction(std::string *error);

    std::vector<Boundary> boundaries_;
    Transport transport_ = Transport::NativePipeline;
    std::unique_ptr<LocalTPContext> collective_;
    std::shared_ptr<const ClusterInventory> pipeline_inventory_; ///< Canonical selected execution inventory used by native domains.
    bool fixed_head_pipeline_ = false; ///< The actual admitted placement requires stage-relative native service.
    std::shared_ptr<PipelineMaintenanceAdmission> maintenance_admission_; ///< Definitions retire before the original native context.
    std::unique_ptr<TPWorkerPool> workers_;
    /** Frozen edge owners outlive the stages' borrowed forward-cache identities. */
    std::vector<std::unique_ptr<PipelineForwardGraphEdges>> forward_edges_;
    /** Immutable topology capability, not a retry/fallback selection. */
    DeviceGenerationExecutionPolicy execution_policy_ = DeviceGenerationExecutionPolicy::Unsupported;
    Phase phase_ = Phase::Idle;
};
}
