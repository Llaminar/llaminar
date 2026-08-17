/**
 * @file MoEOverlayInferenceTransactionService.h
 * @brief Fixed-slot MPI control channel and remote ExpertOverlay follower.
 *
 * A heterogeneous continuation graph cannot launch a retained graph on another
 * process or GPU backend directly. The continuation rank therefore publishes
 * one immutable @ref MoEOverlayInferenceTransactionTicket immediately before
 * the first sparse dispatch of that graph transaction. The remote rank consumes
 * the ticket, executes exactly the named retained graph, and then waits for the
 * next ticket. Compact activations and expert outputs continue to travel through
 * @ref MoEOverlayMPIRankBatchTransport; this control channel never carries model
 * state.
 *
 * Both directions use setup-owned fixed storage. Publishing a ticket retains its
 * bytes in a ring until `MPI_Test` reports completion, while receiving actively
 * progresses only the exact request whose bytes are required to choose a graph.
 * No blocking MPI wait or hot-path allocation is permitted.
 */

#pragma once

#include "MoEOverlayInferenceTransaction.h"

#include <mpi.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;

    /** @brief Result of receiving one authenticated fixed-size control ticket. */
    struct MoEOverlayInferenceTransactionReceiveResult
    {
        bool ok = false; ///< True only when exactly one complete ticket arrived.
        MoEOverlayInferenceTransactionTicket ticket{}; ///< Received immutable value.
        std::string error; ///< Stable failure diagnostic; empty on success.
    };

    /**
     * @brief One direct root-to-follower MPI control lane.
     *
     * A source instance publishes and a target instance receives. The class is
     * intentionally rank-pair scoped: multi-rank overlays create one lane for
     * each follower rank, so a slow endpoint cannot make unrelated ranks enter
     * a control collective merely to learn that no work targets them yet.
     */
    class MoEOverlayMPIInferenceTransactionChannel final
    {
    public:
        /** @brief Immutable communicator, rank pair, and fixed send-ring size. */
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_ctx; ///< Exact control communicator.
            int source_world_rank = -1; ///< Continuation authority publishing tickets.
            int target_world_rank = -1; ///< Remote sparse-graph follower.
            std::size_t send_slot_count = 4; ///< Stable ticket slots retained by source.
        };

        /**
         * @brief Bind one rank-pair control lane and allocate its fixed slots.
         * @throws std::invalid_argument for an invalid context, rank, or capacity.
         */
        explicit MoEOverlayMPIInferenceTransactionChannel(Config config);

        /** @brief Drain source-owned sends before their fixed storage is released. */
        ~MoEOverlayMPIInferenceTransactionChannel();

        MoEOverlayMPIInferenceTransactionChannel(
            const MoEOverlayMPIInferenceTransactionChannel &) = delete;
        MoEOverlayMPIInferenceTransactionChannel &operator=(
            const MoEOverlayMPIInferenceTransactionChannel &) = delete;
        MoEOverlayMPIInferenceTransactionChannel(
            MoEOverlayMPIInferenceTransactionChannel &&) = delete;
        MoEOverlayMPIInferenceTransactionChannel &operator=(
            MoEOverlayMPIInferenceTransactionChannel &&) = delete;

        /**
         * @brief Publish one ticket from a stable fixed slot without waiting.
         *
         * Completion means MPI owns an immutable transport slot. The caller may
         * immediately release or mutate its original ticket value.
         */
        bool publish(
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error = nullptr);

        /**
         * @brief Receive exactly one ticket required by the follower scheduler.
         *
         * This is a true control dependency: the target cannot choose a retained
         * graph until the ticket arrives. The implementation progresses the
         * non-blocking receive with `MPI_Test` under the standard timeout.
         */
        [[nodiscard]] MoEOverlayInferenceTransactionReceiveResult receive();

        /** @return Continuation authority fixed at construction. */
        [[nodiscard]] int sourceWorldRank() const noexcept
        {
            return config_.source_world_rank;
        }

        /** @return Remote follower rank fixed at construction. */
        [[nodiscard]] int targetWorldRank() const noexcept
        {
            return config_.target_world_rank;
        }

        /** @return MPI world rank of this process on the bound communicator. */
        [[nodiscard]] int localWorldRank() const noexcept;

        /** @return Number of source ticket sends not yet retired by MPI. */
        [[nodiscard]] std::size_t inFlightSendCount() const noexcept;

    private:
        /** @brief One setup-owned immutable ticket and its exact MPI request. */
        struct SendSlot
        {
            MoEOverlayInferenceTransactionTicket ticket{};
            MPI_Request request = MPI_REQUEST_NULL;
            bool in_flight = false;
        };

        /** @brief Retire sends completed by one non-blocking progress pass. */
        void progressSendSlots() const;

        /** @brief Return one reusable send slot, progressing bounded backpressure. */
        SendSlot *acquireSendSlot(std::string *error);

        /** @brief Progress one exact request until completion or timeout. */
        bool progressRequestToCompletion(
            MPI_Request *request,
            MPI_Status *status,
            const char *operation,
            std::string *error) const;

        /** @brief Teardown-only bounded drain of every source send. */
        void drainNoexcept() noexcept;

        Config config_;
        mutable std::vector<SendSlot> send_slots_;
        std::size_t next_send_slot_ = 0;
        MoEOverlayInferenceTransactionTicket receive_ticket_{};
    };

    /**
     * @brief Retained graph authority used by the remote transaction follower.
     *
     * Implementations own graph selection, device streams, prepared weights, and
     * sparse data-plane execution. They must not sample, mutate continuation KV,
     * or manufacture token state merely because a control ticket arrived.
     */
    class IMoEOverlayInferenceTransactionExecutor
    {
    public:
        virtual ~IMoEOverlayInferenceTransactionExecutor() = default;

        /**
         * @brief Execute exactly the graph family and geometry named by a ticket.
         * @param ticket Already-authenticated execution ticket.
         * @param error Optional failure diagnostic.
         * @return True after graph work and its return handoff complete.
         */
        virtual bool executeMoEOverlayInferenceTransaction(
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error = nullptr) = 0;
    };

    /** @brief Role and geometry supplied at one root graph-launch boundary. */
    struct MoEOverlayInferenceExecutionDescriptor
    {
        MoEOverlayInferenceGraphRole graph_role =
            MoEOverlayInferenceGraphRole::None; ///< Exact retained graph family.
        std::uint64_t logical_step_id = 0; ///< Sparse data-plane operation id.
        std::uint64_t placement_epoch = 0; ///< Immutable residency bank epoch.
        int request_count = 0; ///< Logical requests represented by this graph.
        int logical_rows_per_request = 0; ///< Live rows per logical request.
        int physical_rows_per_request = 0; ///< Captured/admitted rows per request.
        int draft_depth = -1; ///< Admitted speculative width, when applicable.
        int sidecar_depth = -1; ///< Speculative ordinal, not learned graph depth.

        bool operator==(
            const MoEOverlayInferenceExecutionDescriptor &) const = default;
    };

    /** @brief Source-owned live transaction retained until data return completes. */
    struct MoEOverlayPublishedInferenceTransaction
    {
        bool ok = false; ///< True only after the control ticket was submitted.
        std::size_t slot_index =
            MoEOverlayInferenceAdmission::kNoSlot; ///< Source lifecycle slot.
        MoEOverlayInferenceTransactionTicket ticket{}; ///< Exact published value.
        std::string error; ///< Stable failure diagnostic.
    };

    /**
     * @brief Continuation-side authority for one direct follower lane.
     *
     * A caller opens one outer command, publishes a ticket immediately before
     * launching its matching continuation graph, then retires the returned
     * handle only after that graph has consumed the remote return. This keeps
     * the source slot live for the same interval as the remote data-plane work
     * and makes an early Complete ticket structurally impossible.
     */
    class IMoEOverlayInferenceTransactionPublisher
    {
    public:
        virtual ~IMoEOverlayInferenceTransactionPublisher() = default;

        /** @brief Open one root-authoritative outer inference command. */
        virtual bool beginCommand(
            const MoEOverlayInferenceCommandIdentity &command,
            std::string *error = nullptr) = 0;
        /** @brief Publish one exact retained-graph execution ticket. */
        [[nodiscard]] virtual MoEOverlayPublishedInferenceTransaction publish(
            const MoEOverlayInferenceExecutionDescriptor &descriptor) = 0;
        /** @brief Retire one source slot after its data-plane return is complete. */
        virtual bool retire(
            const MoEOverlayPublishedInferenceTransaction &transaction,
            std::string *error = nullptr) = 0;
        /** @brief Publish the successful terminal ticket. */
        virtual bool complete(
            std::uint64_t placement_epoch,
            std::string *error = nullptr) = 0;
        /** @brief Publish a fatal terminal ticket. */
        virtual bool abort(
            std::uint64_t placement_epoch,
            int error_code,
            std::string *error = nullptr) = 0;
        /** @return Immutable source/target topology for this publisher lane. */
        [[nodiscard]] virtual const MoEOverlayInferenceTopologyIdentity &
        topologyIdentity() const noexcept = 0;
    };

    class MoEOverlayInferenceTransactionPublisher final
        : public IMoEOverlayInferenceTransactionPublisher
    {
    public:
        /** @brief Fixed control lane and the identical source-side protocol. */
        struct Config
        {
            std::shared_ptr<MoEOverlayMPIInferenceTransactionChannel> channel;
            MoEOverlayInferenceTransactionProtocol::Config protocol;
        };

        /**
         * @brief Bind a continuation-owned publisher.
         * @throws std::invalid_argument for a missing lane or rank mismatch.
         */
        explicit MoEOverlayInferenceTransactionPublisher(Config config);

        /** @brief Open the next monotonic inference command. */
        bool beginCommand(
            const MoEOverlayInferenceCommandIdentity &command,
            std::string *error = nullptr) override;

        /**
         * @brief Publish one execution ticket and retain its source slot.
         * @return Handle that must be passed exactly once to @ref retire.
         */
        [[nodiscard]] MoEOverlayPublishedInferenceTransaction publish(
            const MoEOverlayInferenceExecutionDescriptor &descriptor) override;

        /** @brief Retire a graph transaction after its sparse return completed. */
        bool retire(
            const MoEOverlayPublishedInferenceTransaction &transaction,
            std::string *error = nullptr) override;

        /** @brief Publish the successful terminal after every handle retired. */
        bool complete(
            std::uint64_t placement_epoch,
            std::string *error = nullptr) override;

        /** @brief Publish a process-fatal terminal for the active command. */
        bool abort(
            std::uint64_t placement_epoch,
            int error_code,
            std::string *error = nullptr) override;

        /** @inheritdoc IMoEOverlayInferenceTransactionPublisher */
        [[nodiscard]] const MoEOverlayInferenceTopologyIdentity &
        topologyIdentity() const noexcept override
        {
            return protocol_.topologyIdentity();
        }

        /** @return Source protocol state for orchestration diagnostics. */
        [[nodiscard]] MoEOverlayInferenceProtocolState state() const noexcept
        {
            return protocol_.state();
        }

        /** @return Ordinal that the next execution or terminal must use. */
        [[nodiscard]] std::uint64_t nextTransactionOrdinal() const noexcept
        {
            return protocol_.nextTransactionOrdinal();
        }

    private:
        /** @brief Prove the source process owns this channel and topology. */
        void validateConstruction() const;
        /** @brief Publish one terminal action through the shared validator. */
        bool publishTerminal(
            MoEOverlayInferenceTransactionAction action,
            std::uint64_t placement_epoch,
            int error_code,
            std::string *error);

        std::shared_ptr<MoEOverlayMPIInferenceTransactionChannel> channel_;
        MoEOverlayInferenceTransactionProtocol protocol_;
        MoEOverlayInferenceCommandIdentity active_command_{};
    };

    /** @brief Participant-local lease for one rank-wide published graph ticket. */
    struct MoEOverlayInferenceParticipantGraphBinding
    {
        bool ok = false; ///< False only for a coordinator/protocol defect.
        bool active = false; ///< False when no outer overlay command is open.
        bool owns_ticket_authority = false; ///< This participant owns the one remote publication edge.
        std::uint64_t group_id = 0; ///< Rank-wide graph invocation identity.
        int participant_index = -1; ///< Exact continuation participant owner.
        std::uint64_t request_generation = 0; ///< Wire generation to stamp.
        MoEOverlayInferenceExecutionDescriptor descriptor{}; ///< Effective geometry.
        std::string error; ///< Stable failure diagnostic.
    };

    /**
     * @brief Publish one control ticket for a symmetric continuation rank.
     *
     * LocalTP participants enter the same graph invocation independently. Entry
     * reserves one immutable descriptor and operation id but does not publish it.
     * The planner-resolved continuation root reaching its exact executable
     * launch boundary arms the group and publishes one ticket to every remote
     * rank; sibling participants authenticate that same armed descriptor. This
     * prevents a follower from
     * starting its bounded activation epoch while a continuation parent is still
     * undergoing cold native capture. Source protocol slots remain live until the
     * caller presents the exact completion boundary for that graph sequence. All
     * vectors and participant masks are allocated during construction.
     */
    class MoEOverlayInferenceTransactionCoordinator final
    {
    public:
        /** @brief Setup-owned publishers and fixed command capacities. */
        struct Config
        {
            std::vector<
                std::shared_ptr<IMoEOverlayInferenceTransactionPublisher>>
                publishers; ///< One direct lane per remote follower rank.
            int continuation_participant_count = 0; ///< Symmetric local graph count.
            int ticket_authority_participant_index = -1; ///< Planner-resolved LocalTP child owning the remote packet parent.
            std::size_t max_transactions_per_command = 0; ///< Fixed retained ring.
            int max_mtp_draft_depth = 0; ///< Maximum admitted speculative width.
        };

        /** @brief Validate topology and allocate every command-local slot. */
        explicit MoEOverlayInferenceTransactionCoordinator(Config config);

        /** @brief Open one monotonic command on every target publisher. */
        bool beginCommand(
            const MoEOverlayInferenceCommandIdentity &command,
            std::string *error = nullptr);

        /**
         * @brief Open one bounded graph sequence inside the outer command.
         *
         * Zero names an ordinary serial transaction. A positive value names one
         * complete speculative transaction containing every recurrent sidecar
         * followed by one grouped verifier. Device-resident generation may open
         * many such sequences under one decode command, including different
         * dynamic depths, but only after the previous sequence's sparse return
         * has been proven complete and retired through
         * @ref retireCompletedGraphSequence.
         *
         * @param draft_depth Controller-selected speculative width, or zero.
         * @param error Optional stable failure diagnostic.
         * @return True when the next graph may enter this sequence.
         */
        bool beginGraphSequence(
            int draft_depth,
            std::string *error = nullptr);

        /**
         * @brief Admit the current participant into a serial prefill sequence.
         *
         * Every continuation participant calls this immediately before its
         * matching main-prefill graph scope. The first participant opens the
         * depth-zero sequence. When a bounded prefill schedule advances to its
         * next chunk, the first entrant retires the fully submitted prior
         * sequence and opens the next one; symmetric siblings then join that
         * same sequence. Decode and MTP callers remain explicitly controlled by
         * @ref beginGraphSequence and cannot use this prefill-only transition.
         *
         * An ahead participant whose previous graph has already been submitted
         * waits here until every sibling has submitted that same graph. This is
         * a rank-local host submission barrier at an explicitly heterogeneous
         * graph boundary; it never synchronizes a GPU stream or copies model
         * state to the host.
         *
         * @param participant_index Stable continuation participant ordinal.
         * @param error Optional stable failure diagnostic.
         * @return True when one current MainPrefill graph group may enter.
         */
        bool admitSerialPrefillGraph(
            int participant_index,
            std::string *error = nullptr);

        /**
         * @brief Enter one participant into the current symmetric graph group.
         *
         * The first entrant allocates a command-local operation id and reserves
         * every target handle. Publication is deferred to the designated ticket
         * authority's @ref armParticipantGraph call so cold capture cannot
         * consume a follower epoch. No participant may enter the next graph
         * until all participants finish the current one.
         */
        [[nodiscard]] MoEOverlayInferenceParticipantGraphBinding
        beginParticipantGraph(
            MoEOverlayInferenceExecutionDescriptor descriptor,
            int participant_index);

        /**
         * @brief Publish the reserved graph ticket at an executable launch edge.
         *
         * Only the planner-resolved ticket-authority binding may publish every
         * target lane. Repeated calls from that same binding are idempotent;
         * sibling and stale bindings are rejected. The authority invokes it
         * immediately before its GPU executable launch, after capture and
         * instantiation have completed.
         *
         * @param binding Exact binding returned by @ref beginParticipantGraph.
         * @param error Optional stable failure diagnostic.
         * @return True when the group is armed or was already armed by a sibling.
         */
        bool armParticipantGraph(
            const MoEOverlayInferenceParticipantGraphBinding &binding,
            std::string *error = nullptr);

        /** @brief Finish one participant; the last finisher seals the group. */
        bool finishParticipantGraph(
            const MoEOverlayInferenceParticipantGraphBinding &binding,
            bool execution_succeeded,
            std::string *error = nullptr);

        /**
         * @brief Recycle the current fixed slots after an exact return fence.
         *
         * The caller may invoke this only after the submitted continuation graph
         * has consumed every remote sparse return. For hosted retained generation,
         * observing the next authenticated device dispatch ticket is that fence;
         * for the terminal sequence, terminal-result publication is the fence.
         * No MPI wait is introduced here: each publisher merely advances its
         * already-completed protocol slots back to Available.
         *
         * @param error Optional stable failure diagnostic.
         * @return True when all sequence slots were retired and may be reused.
         */
        bool retireCompletedGraphSequence(std::string *error = nullptr);

        /** @brief Retire all graph slots and publish Complete to every target. */
        bool completeCommand(
            std::uint64_t placement_epoch,
            std::string *error = nullptr);

        /** @brief Publish a fatal command terminal to every target. */
        bool abortCommand(
            std::uint64_t placement_epoch,
            int error_code,
            std::string *error = nullptr) noexcept;

        /** @return Number of symmetric continuation participants. */
        [[nodiscard]] int continuationParticipantCount() const noexcept
        {
            return config_.continuation_participant_count;
        }

        /** @return Current coordinator lifecycle. */
        [[nodiscard]] MoEOverlayInferenceProtocolState state() const noexcept;

        /** @return Active command placement epoch including graph advances. */
        [[nodiscard]] std::uint64_t currentPlacementEpoch() const noexcept;

        /** @return Controller-selected MTP width, or -1 before declaration. */
        [[nodiscard]] int activeMTPDraftDepth() const noexcept;

    private:
        /** @brief One preallocated command slot containing all target handles. */
        struct RetainedTransaction
        {
            bool used = false;
            bool armed = false; ///< Every target ticket has been published.
            std::uint64_t group_id = 0;
            MoEOverlayInferenceExecutionDescriptor descriptor{};
            std::vector<MoEOverlayPublishedInferenceTransaction>
                target_transactions;
        };

        /** @brief Transition to Failed while preserving the first diagnostic. */
        bool failLocked(std::string message, std::string *error);
        /** @brief Return whether every local participant has the supplied bit. */
        bool allParticipantsMarked(const std::vector<std::uint8_t> &marks) const;
        /** @brief Validate and retire one completed sequence while mutex_ is held. */
        bool retireCompletedGraphSequenceLocked(std::string *error);
        /** @brief Open one graph sequence while mutex_ is already held. */
        bool beginGraphSequenceLocked(int draft_depth, std::string *error);
        /** @brief Clear fixed slots and sequence-local role counters. */
        void resetGraphSequenceLocked() noexcept;

        Config config_;
        mutable std::mutex mutex_;
        /** Wakes an ahead prefill submitter after the last sibling seals a group. */
        std::condition_variable graph_group_completion_cv_;
        std::vector<RetainedTransaction> retained_transactions_;
        std::vector<std::uint8_t> entered_participants_;
        std::vector<std::uint8_t> finished_participants_;
        MoEOverlayInferenceProtocolState state_ =
            MoEOverlayInferenceProtocolState::Idle;
        MoEOverlayInferenceCommandIdentity active_command_{};
        std::size_t transaction_count_ = 0;
        std::size_t total_transaction_count_ = 0;
        std::size_t graph_sequence_count_ = 0;
        bool graph_group_active_ = false;
        bool graph_execution_failed_ = false;
        bool mtp_depth_declared_ = false;
        int active_mtp_draft_depth_ = -1;
        int next_sidecar_ordinal_ = 0;
        int sequence_main_graph_count_ = 0;
        int sequence_sidecar_graph_count_ = 0;
        int sequence_verifier_graph_count_ = 0;
        std::uint64_t next_group_id_ = 1;
        std::uint64_t next_logical_step_id_ = 1;
        std::uint64_t current_placement_epoch_ = 0;
        std::string failure_;
    };

    /**
     * @brief RAII participant entry around one real graph submission.
     *
     * Destruction reports a failed graph when a throwing or early-return path
     * did not explicitly call @ref finish. This keeps the coordinator from
     * accepting a later graph after one local participant abandoned its slot.
     */
    class MoEOverlayInferenceParticipantGraphScope final
    {
    public:
        /** @brief Enter the coordinator, or create an inert successful scope. */
        MoEOverlayInferenceParticipantGraphScope(
            std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                coordinator,
            MoEOverlayInferenceExecutionDescriptor descriptor,
            int participant_index);
        /** @brief Fail an unfinished active graph without throwing. */
        ~MoEOverlayInferenceParticipantGraphScope() noexcept;

        MoEOverlayInferenceParticipantGraphScope(
            const MoEOverlayInferenceParticipantGraphScope &) = delete;
        MoEOverlayInferenceParticipantGraphScope &operator=(
            const MoEOverlayInferenceParticipantGraphScope &) = delete;

        /** @return Whether coordinator entry succeeded. */
        [[nodiscard]] bool ready() const noexcept { return binding_.ok; }
        /** @return Whether a live outer command owns this graph. */
        [[nodiscard]] bool active() const noexcept { return binding_.active; }
        /** @return Whether this participant owns the one remote ticket edge. */
        [[nodiscard]] bool ownsTicketAuthority() const noexcept
        {
            return binding_.active && binding_.owns_ticket_authority;
        }
        /** @return Effective generation, operation id, role, and geometry. */
        [[nodiscard]] const MoEOverlayInferenceParticipantGraphBinding &
        binding() const noexcept
        {
            return binding_;
        }
        /** @return Entry failure diagnostic, if any. */
        [[nodiscard]] const std::string &error() const noexcept
        {
            return binding_.error;
        }

        /**
         * @brief Arm this graph at its exact executable launch boundary.
         *
         * Inert scopes succeed without side effects. The designated authority
         * provides the non-null stream about to submit its executable; the stream
         * is validated here so ticket publication cannot drift back to graph
         * entry. Sibling scopes must not call this method. Repeated authority
         * calls are idempotent.
         *
         * @param execution_stream Exact non-null GPU executable stream.
         * @param error Optional stable failure diagnostic.
         * @return True once the remote transaction ticket is published.
         */
        bool armForExecutableLaunch(
            void *execution_stream,
            std::string *error = nullptr);

        /**
         * @brief Arm an explicitly synchronous CPU continuation transaction.
         *
         * CPU execution has no device stream or native graph launch. Its typed
         * heterogeneous boundary is the call into the host-owned compute graph,
         * so CPU callers invoke this method immediately before that call. GPU
         * callers must use @ref armForExecutableLaunch instead.
         */
        bool armForHostExecution(std::string *error = nullptr);

        /** @brief Publish this participant's exact success/failure once. */
        bool finish(bool execution_succeeded, std::string *error = nullptr);

    private:
        std::shared_ptr<MoEOverlayInferenceTransactionCoordinator> coordinator_;
        MoEOverlayInferenceParticipantGraphBinding binding_{};
        bool armed_ = false;
        bool finished_ = false;
    };

    /** @brief Terminal result of one outer follower command. */
    struct MoEOverlayInferenceFollowerCommandResult
    {
        bool ok = false; ///< True only for a successful Complete terminal.
        bool aborted = false; ///< True when root published an Abort terminal.
        int error_code = 0; ///< Positive root error code for an abort.
        std::size_t executed_transactions = 0; ///< Retained graphs completed.
        std::string error; ///< Protocol, transport, or executor diagnostic.
    };

    /**
     * @brief Authenticate and execute one root-published transaction sequence.
     *
     * The follower has one authority for slot lifecycle. Each execution ticket
     * transitions `Accepted -> Submitted -> ReturnReady -> Available` around the
     * exact retained graph invocation. Complete is accepted only after every
     * slot has retired; stale or divergent control state fails before device work.
     */
    class MoEOverlayInferenceTransactionFollower final
    {
    public:
        /** @brief Fixed protocol capacities and process-owned collaborators. */
        struct Config
        {
            std::shared_ptr<MoEOverlayMPIInferenceTransactionChannel> channel;
            IMoEOverlayInferenceTransactionExecutor *executor = nullptr;
            MoEOverlayInferenceTransactionProtocol::Config protocol;
        };

        /**
         * @brief Bind one fixed follower.
         * @throws std::invalid_argument for a missing channel/executor or rank mismatch.
         */
        explicit MoEOverlayInferenceTransactionFollower(Config config);

        /**
         * @brief Receive and execute tickets until Complete or Abort arrives.
         *
         * The method performs no token, sampler, logits, or KV coordination. It
         * is intended to run inside the remote rank's coordinated worker command.
         */
        [[nodiscard]] MoEOverlayInferenceFollowerCommandResult runOneCommand();

        /** @return Protocol state for diagnostics and adversarial tests. */
        [[nodiscard]] MoEOverlayInferenceProtocolState state() const noexcept
        {
            return protocol_.state();
        }

    private:
        /** @brief Prove channel, protocol, and local follower ownership agree. */
        void validateConstruction() const;

        std::shared_ptr<MoEOverlayMPIInferenceTransactionChannel> channel_;
        IMoEOverlayInferenceTransactionExecutor *executor_ = nullptr;
        MoEOverlayInferenceTransactionProtocol protocol_;
    };

} // namespace llaminar2
