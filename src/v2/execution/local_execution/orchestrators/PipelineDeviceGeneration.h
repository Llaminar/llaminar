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
#include "execution/mtp/MTPMainForwardPolicy.h"
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace llaminar2
{
class DeviceGraphOrchestrator;
class LocalTPContext;
class ILocalTPContext;
class CapturedTransferChannel;
class TPWorkerPool;
class PipelineForwardGraphEdges;

/** @brief Complete captured pipeline lifecycle, borrowing already-admitted stages. */
class PipelineDeviceGeneration final
{
public:
    /** @return Whether every declared layer domain is GPU-resident.
     * This is topology classification only; construction still validates every
     * role and backend capability, with no alternative on admission failure. */
    static bool isGPUStageSet(std::span<const std::unique_ptr<IInferenceRunner>> stages);
    /** @brief Freeze ordered domains and their exact native/cross-vendor edges.
     * @param stages Stable participant runners, in layer order, tail last.
     * @throws std::invalid_argument for incomplete or unsupported composition. */
    explicit PipelineDeviceGeneration(std::span<const std::unique_ptr<IInferenceRunner>> stages);
    /** @brief Join workers; stage graphs must already be retired by their owners. */
    ~PipelineDeviceGeneration();
    /** @brief Freeze native prefill edges and prepare all stage graph families.
     * @param plan Canonical bucket family shared by every participant.
     * @return Whether all stage families materialized before request admission. */
    bool prepareServing(const ServingGraphFamilyMaterializationPlan &plan);
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
    /** @brief Publish one tail outcome into each participant's own KV/recurrent state.
     * @param request Tail-owned immutable outcome handle and publication geometry.
     * @param error Receives the first participant's publication diagnostic.
     * @return Whether every captured commit was submitted, without host outcome reads. */
    bool publishMTPOutcome(const DeviceSpeculativePublicationRequest &request, std::string *error);
    /** @brief Admit the tail's policy and each follower's local event handoff. */
    bool begin(const DeviceGenerationAdmissionRequest &request);
    /** @brief Compile or reuse all complete parents before submitting any work. */
    bool materialize(int requests, int depth, DeviceGenerationLoopTopology topology,
        DeviceGenerationSamplingMode sampling);
    /** @brief Enqueue every parent before any terminal observation. */
    bool launch();
    /** @brief Return only the tail response and validate independent stage KV positions. */
    bool finish(DeviceGenerationTerminalResult *result);
    /** @return Whether this owner has an admitted, not-yet-retired request. */
    bool active() const noexcept { return phase_ != Phase::Idle; }

private:
    /** @brief Frozen layer-domain role; its runner remains the TP authority. */
    struct Domain
    {
        IInferenceRunner *runner;
        ILocalTPContext *context;
        size_t first;
        size_t count;
    };
    /** @brief Two opposite directions, shared by every retained graph family. */
    struct Boundary
    {
        std::shared_ptr<CapturedTransferChannel> activation;
        std::shared_ptr<CapturedTransferChannel> metadata;
    };
    /** @brief Exact installed composition, not a runtime failure alternative. */
    enum class Transport { NativePipeline, CapturedDomains };
    /** @return Frozen domain containing one flattened participant. */
    const Domain &domainFor(size_t participant) const;
    /** @return First participant of the terminal domain (not the last physical GPU). */
    size_t terminalIndex() const noexcept { return domains_.back().first; }
    /** @return Whether a participant belongs to the one terminal authority. */
    bool terminalParticipant(size_t index) const noexcept { return index >= terminalIndex(); }
    /** @brief Abort every installed native communicator after a fatal partial submission. */
    void abortParticipants();
    /** @brief Materialize admitted channels, before any graph borrows their identities. */
    void prepareDomainBoundaries();
    /** @brief Submit complete ordinary follower transactions around terminal-domain tickets. */
    bool launchHostedDomainTransactions();
    /** @brief Host submission lifecycle only; no mirrored device decision state. */
    enum class Phase { Idle, Admitted, Materialized, Submitted, Failed };
    /** @brief Persistent worker fanout for setup, never a per-token dispatch. */
    bool prepareParticipants(DeviceGenerationSamplingMode sampling);
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
    std::vector<DeviceGraphOrchestrator *> stages_;
    std::vector<Domain> domains_;
    std::vector<Boundary> boundaries_;
    Transport transport_ = Transport::NativePipeline;
    std::unique_ptr<LocalTPContext> collective_;
    std::unique_ptr<TPWorkerPool> workers_;
    /** Frozen edge owners outlive the stages' borrowed forward-cache identities. */
    std::vector<std::unique_ptr<PipelineForwardGraphEdges>> forward_edges_;
    /** Immutable topology capability, not a retry/fallback selection. */
    DeviceGenerationExecutionPolicy execution_policy_ = DeviceGenerationExecutionPolicy::Unsupported;
    Phase phase_ = Phase::Idle;
};
}
