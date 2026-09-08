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

#include "MoEOverlayInferenceInterferenceProbe.h"
#include "MoEOverlayInferenceTransaction.h"
#include "MoEOverlayResidencyAuthority.h"

#include <mpi.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;

    /**
     * @brief Non-blocking sink for one fully retired prefill transaction.
     *
     * The count comes from the authenticated ticket's real rows, never its
     * padded graph bucket. Production binds this to the process-local device
     * controller's coalescing counter. The sink must not launch policy, perform
     * MPI, allocate, or wait; returning false makes the command terminal.
     */
    using MoEOverlayRetiredPrefillProgressSink =
        std::function<bool(std::uint64_t completed_tokens,
                           std::string *error)>;

    /**
     * @brief Non-blocking sink for device-authenticated decode progress.
     *
     * Hosted HIP generation publishes a cumulative committed-token count in
     * its immutable scheduler ticket.  The transaction coordinator converts
     * consecutive snapshots into an exactly-once positive delta only after
     * the matching sparse graph sequence retires.  Production binds this sink
     * to host-authoritative background residency maintenance; it must not run
     * policy, perform MPI, allocate, or wait.
     */
    using MoEOverlayRetiredDecodeProgressSink =
        std::function<bool(std::uint64_t completed_tokens,
                           std::string *error)>;

    /** @brief Exact terminal owned by one continuation graph participant. */
    enum class MoEOverlayInferenceCompletionBoundaryKind : std::uint8_t
    {
        Unspecified = 0,  ///< Invalid setup state; callers must choose.
        HostSynchronous, ///< CPU graph calls return only after execution.
        DeviceEvent,     ///< GPU graph completion is proven by an exact event.
    };

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
        /**
         * Complete logical prefill schedule containing this graph ticket.
         *
         * The coordinator fills this field from its schedule declaration; an
         * ordinary graph caller must leave it invalid.  Carrying it on every
         * authenticated segment lets a remote follower time the same complete
         * interval without owning tokens or recreating root scheduling policy.
         */
        MoEOverlayInferenceWorkloadIdentity prefill_schedule_workload{};
        /**
         * Coordinator-owned cumulative decode frontier for follower cadence.
         * Ordinary graph callers leave this zero; the rank-wide transaction
         * coordinator overwrites it from the authenticated hosted ticket before
         * comparing symmetric descriptors or publishing remote work.
         */
        std::uint64_t retired_decode_progress_tokens = 0u;

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
            std::uint64_t retired_decode_progress_tokens,
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
            std::uint64_t retired_decode_progress_tokens,
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
            std::uint64_t retired_decode_progress_tokens,
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
        std::uint64_t sequence_id = 0; ///< Rank-local immutable sequence identity.
        int sequence_graph_ordinal = -1; ///< Zero-based role ordinal in sequence.
        int sequence_graph_count = 0; ///< Complete admitted sequence cardinality.
        MoEOverlayInferenceExecutionDescriptor descriptor{}; ///< Effective geometry.
        std::string error; ///< Stable failure diagnostic.

        /** @return Whether this graph acquires the sequence residency lease. */
        [[nodiscard]] bool beginsSequence() const noexcept
        {
            return active && sequence_id != 0 &&
                   sequence_graph_ordinal == 0 &&
                   sequence_graph_count > 0;
        }

        /** @return Whether this graph publishes the sequence terminal. */
        [[nodiscard]] bool endsSequence() const noexcept
        {
            return active && sequence_id != 0 &&
                   sequence_graph_count > 0 &&
                   sequence_graph_ordinal == sequence_graph_count - 1;
        }
    };

    /** @brief Role owned by one process-local segment of a global transaction. */
    enum class MoEOverlayInferenceCoordinatorSegmentRole : std::uint8_t
    {
        Continuation = 0, ///< Dense continuation graph group and ticket authority.
        ExpertFollower,  ///< Remote retained sparse-expert transaction follower.
    };

    /**
     * @brief Sealed serving-family representation owned by one rank segment.
     *
     * There is deliberately no unresolved value. A segment may enter the
     * coordinator plan only after the rank-synchronized serving-family phase
     * has either materialized every native executable or certified every eager
     * host endpoint. This keeps setup capability discovery out of inference.
     */
    enum class MoEOverlayInferenceSegmentMaterializationKind : std::uint8_t
    {
        NativeDeviceExecutable = 0, ///< CUDA/HIP executable family is resident.
        EagerHostGraph,             ///< CPU declarative endpoints are certified.
    };

    /**
     * @brief Immutable global segment plan for one heterogeneous graph family.
     *
     * A transaction ticket selects one complete process-local retained graph,
     * not an individual MoE layer. Consequently the plan contains one
     * continuation-rank segment and one segment per remote expert-follower
     * rank. `local_participant_count` records the native/CPU endpoints composed
     * by that process-local transaction without pretending they are independent
     * cross-rank ticket boundaries.
     *
     * The plan can be created only through
     * @ref sealAfterSynchronizedMaterialization. The caller must have completed
     * the rank-synchronized serving-family phase first; construction then binds
     * that proof to the exact graph-family generation consumed by every ticket
     * publisher. Runtime PerfStats merely mirrors this production authority.
     */
    class MoEOverlayInferenceCoordinatorGraphPlan final
    {
    public:
        /** @brief One process-local graph transaction in the global schedule. */
        struct Segment
        {
            MoEOverlayInferenceCoordinatorSegmentRole role =
                MoEOverlayInferenceCoordinatorSegmentRole::Continuation;
            MoEOverlayInferenceSegmentMaterializationKind materialization =
                MoEOverlayInferenceSegmentMaterializationKind::
                    NativeDeviceExecutable;
            int world_rank = -1; ///< Exact MPI rank executing this segment.
            std::size_t local_participant_count = 0; ///< Endpoints composed locally.

            bool operator==(const Segment &) const = default;
        };

        /**
         * @brief Seal a complete deterministic plan after graph-family consensus.
         * @param graph_family_generation Exact generation embedded in tickets.
         * @param segments One continuation plus every remote follower rank.
         * @return Valid immutable plan in deterministic role/rank order.
         * @throws std::invalid_argument when the plan is incomplete or aliases a
         *         rank, continuation authority, or follower boundary.
         */
        static MoEOverlayInferenceCoordinatorGraphPlan
        sealAfterSynchronizedMaterialization(
            std::uint64_t graph_family_generation,
            std::vector<Segment> segments);

        /** @return Whether this value contains one complete global plan. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Exact graph-family generation represented by the plan. */
        [[nodiscard]] std::uint64_t graphFamilyGeneration() const noexcept
        {
            return graph_family_generation_;
        }

        /** @return Number of process-local segments in one global replay. */
        [[nodiscard]] std::size_t segmentCount() const noexcept
        {
            return segments_.size();
        }

        /** @return Number of remote expert-follower rank segments. */
        [[nodiscard]] std::size_t followerSegmentCount() const noexcept;

        /** @return Number of segments backed by native device executables. */
        [[nodiscard]] std::size_t nativeSegmentCount() const noexcept;

        /** @return Number of segments backed by certified eager host graphs. */
        [[nodiscard]] std::size_t eagerHostSegmentCount() const noexcept;

        /** @return Total local endpoints composed by all native segments. */
        [[nodiscard]] std::size_t nativeParticipantCount() const noexcept;

        /** @return Exact continuation authority rank, or -1 when invalid. */
        [[nodiscard]] int continuationWorldRank() const noexcept;

        /**
         * @brief Find the unique segment owned by @p world_rank.
         * @return Stable plan-owned pointer, or null when the rank is absent.
         */
        [[nodiscard]] const Segment *segmentForWorldRank(
            int world_rank) const noexcept;

        /** @return Deterministically ordered immutable segment inventory. */
        [[nodiscard]] const std::vector<Segment> &segments() const noexcept
        {
            return segments_;
        }

    private:
        std::uint64_t graph_family_generation_ = 0;
        std::vector<Segment> segments_;
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
            MoEOverlayInferenceCoordinatorGraphPlan
                graph_plan; ///< Sealed complete heterogeneous transaction plan.
            int continuation_participant_count = 0; ///< Symmetric local graph count.
            int ticket_authority_participant_index = -1; ///< Planner-resolved LocalTP child owning the remote packet parent.
            std::vector<MoEOverlayInferenceCompletionBoundaryKind>
                participant_completion_boundaries; ///< Exact CPU/GPU terminal for every continuation child.
            std::size_t max_transactions_per_command = 0; ///< Fixed retained ring.
            int max_mtp_draft_depth = 0; ///< Maximum admitted speculative width.
            /** Optional process-local device-controller wake sideband. */
            MoEOverlayRetiredPrefillProgressSink
                retired_prefill_progress_sink;
            /** Optional host-maintenance wake for retired hosted MTP sequences. */
            MoEOverlayRetiredDecodeProgressSink
                retired_decode_progress_sink;
            /**
             * Host RCU authority whose current epoch is pinned per sequence.
             * Device-resident homogeneous controllers leave this empty because
             * their captured epoch arena owns the equivalent lease entirely on
             * device.
             */
            std::shared_ptr<MoEOverlayResidencyAuthority>
                residency_authority;
        };

        /** @brief Validate topology and allocate every command-local slot. */
        explicit MoEOverlayInferenceTransactionCoordinator(Config config);

        /**
         * @brief Bind the one-shot calibration probe before command admission.
         *
         * The coordinator owns the only rank-wide view of a heterogeneous
         * prefill graph group.  Binding here lets it time each real retained
         * chunk from first participant entry through last participant finish,
         * while the ordinary RankOrchestrator scope remains responsible for
         * non-segmented phases. Rebinding the same owner is idempotent; changing
         * it or binding after a command starts is rejected.
         *
         * @param probe Model-lifetime lock-free timing authority.
         * @param error Optional stable validation diagnostic.
         * @return True when future prefill groups own the probe boundary.
         */
        bool bindPrefillInterferenceProbe(
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe,
            std::string *error = nullptr);

        /**
         * @brief Declare one complete segmented-prefill calibration interval.
         *
         * This call precedes the first graph of a caller-visible chunk schedule.
         * It claims the probe once for the aggregate workload, stamps the same
         * identity into every remote execution ticket, and defers completion to
         * the final continuation graph's exact participant-event fence.  A
         * declaration is retained even when no probe request is armed, which
         * prevents a request published mid-schedule from timing a misleading
         * tail segment.
         *
         * @param workload Exact complete bucket/chunk schedule identity.
         * @param error Optional stable lifecycle diagnostic.
         * @return True when this schedule exclusively owns the next declared
         *         number of main-prefill transactions.
         */
        bool declarePrefillInterferenceSchedule(
            MoEOverlayInferenceWorkloadIdentity workload,
            std::string *error = nullptr);

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
         * consume a follower epoch. An ahead participant waits, with the
         * standard bounded collective timeout, for every sibling to submit the
         * current graph before entering the next one. This wait protects only
         * rank-local host metadata; device streams continue asynchronously.
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
         * @brief Bind an active prefill probe to its exact GPU terminal event.
         *
         * Every GPU participant calls this from the forward engine's
         * post-launch hook on its exact producer stream. If no probe claimed
         * this graph, the call is an allocation-free no-op and does not record
         * the reusable event. The coordinator publishes one aggregate probe
         * fence only after every continuation participant has supplied its
         * configured CPU-call or GPU-event terminal.
         *
         * @param logical_step_id Active graph descriptor operation id.
         * @param participant_index Exact continuation participant caller.
         * @param event Setup-owned participant-local persistent backend event.
         * @param producer_stream Exact non-null graph terminal stream.
         * @param error Optional precise lifecycle/backend diagnostic.
         * @return True when no sample was active or device completion owns it.
         */
        bool deferPrefillInterferenceCompletionAtDeviceTerminal(
            std::uint64_t logical_step_id,
            int participant_index,
            std::shared_ptr<IMoEOverlayInferenceCompletionEvent> event,
            void *producer_stream,
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

        /**
         * @brief Idempotently advance one hosted device-generation sequence.
         *
         * Every symmetric continuation participant presents the same
         * authenticated dispatch-ticket transaction id before submitting its
         * retained branch. The first caller retires the completed sparse graph
         * sequence and, for a non-terminal ticket, opens the controller-selected
         * next depth. Later callers carrying that exact transition observe the
         * already-published result. Participant order is therefore irrelevant:
         * no child is elected as a host-side lifecycle authority.
         *
         * A different decision for an already observed id, or a stale id, is a
         * fatal protocol disagreement. The transition never waits for device
         * work; the authenticated ticket is itself the proof that every sparse
         * return and controller publication in the prior sequence completed.
         *
         * @param transaction_id Positive monotonically increasing ticket id.
         * @param next_draft_depth Controller-selected next depth, or nullopt
         *        when the ticket closes the outer generation command.
         * @param committed_output_tokens Cumulative logical response count in
         *        the same authenticated ticket.
         * @param error Optional stable lifecycle diagnostic.
         * @return True when this exact transition was applied or had already
         *         been applied by a symmetric participant.
         */
        bool advanceHostedGraphSequence(
            std::uint64_t transaction_id,
            std::optional<int> next_draft_depth,
            std::uint64_t committed_output_tokens,
            std::string *error = nullptr);

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

        /** @return Planner-selected participant owning remote ticket publication. */
        [[nodiscard]] int ticketAuthorityParticipantIndex() const noexcept
        {
            return config_.ticket_authority_participant_index;
        }

        /**
         * @brief Return the exact terminal required from one participant.
         * @param participant_index Stable continuation participant ordinal.
         * @return Planner-resolved CPU-call or GPU-event boundary.
         */
        [[nodiscard]] MoEOverlayInferenceCompletionBoundaryKind
        participantCompletionBoundary(int participant_index) const noexcept
        {
            if (participant_index < 0 ||
                static_cast<std::size_t>(participant_index) >=
                    config_.participant_completion_boundaries.size())
            {
                return MoEOverlayInferenceCompletionBoundaryKind::Unspecified;
            }
            return config_.participant_completion_boundaries[
                static_cast<std::size_t>(participant_index)];
        }

        /** @return Current coordinator lifecycle. */
        [[nodiscard]] MoEOverlayInferenceProtocolState state() const noexcept;

        /** @return Active command placement epoch including graph advances. */
        [[nodiscard]] std::uint64_t currentPlacementEpoch() const noexcept;

        /** @return Controller-selected MTP width, or -1 before declaration. */
        [[nodiscard]] int activeMTPDraftDepth() const noexcept;

        /** @return Immutable global graph plan governing every command. */
        [[nodiscard]] const MoEOverlayInferenceCoordinatorGraphPlan &
        graphPlan() const noexcept
        {
            return config_.graph_plan;
        }

    private:
        /** @brief Preallocated poll-only fence aggregating participant events. */
        class CompletionFenceSet;

        /** @brief Complete lifecycle of one serial or speculative sequence. */
        enum class ExecutionSequenceState : std::uint8_t
        {
            Idle = 0,     ///< No residency-owning execution sequence exists.
            Open,         ///< The next graph role may be admitted.
            GraphInFlight,///< One symmetric graph group owns the sequence.
            Releasing,    ///< Every graph is terminal and slots are retiring.
            Failed,       ///< An invalid transition made the sequence unusable.
        };

        /**
         * @brief Immutable shape and single cursor for one execution sequence.
         *
         * Depth zero admits exactly one main prefill or decode graph. Positive
         * depth admits exactly `depth` MTP draft graphs followed by one grouped
         * verifier. The first graph pins the placement epoch for the sequence;
         * every later graph must name that same epoch. This replaces independent
         * depth, sidecar ordinal, and per-role counters whose combinations could
         * describe impossible partial sequences.
         */
        struct ExecutionSequencePlan
        {
            ExecutionSequenceState state = ExecutionSequenceState::Idle;
            int draft_depth = -1;
            int next_graph_ordinal = 0;
            std::uint64_t placement_epoch = 0;
            std::uint64_t sequence_id = 0;
            /** Host RCU lifetime matching @ref placement_epoch. */
            std::optional<MoEOverlayResidencyAuthority::TicketLease>
                placement_epoch_lease;

            /** @return Whether a sequence currently owns lifecycle state. */
            [[nodiscard]] bool active() const noexcept
            {
                return state != ExecutionSequenceState::Idle;
            }

            /** @return Whether a participant graph currently owns the cursor. */
            [[nodiscard]] bool graphInFlight() const noexcept
            {
                return state == ExecutionSequenceState::GraphInFlight;
            }

            /** @return Exact number of graph groups required by this shape. */
            [[nodiscard]] int expectedGraphCount() const noexcept
            {
                return draft_depth == 0 ? 1 : draft_depth + 1;
            }

            /** @brief Reset to the only reusable state. */
            void reset() noexcept
            {
                state = ExecutionSequenceState::Idle;
                draft_depth = -1;
                next_graph_ordinal = 0;
                placement_epoch = 0;
                sequence_id = 0;
                placement_epoch_lease.reset();
            }
        };

        /** @brief Typed lifecycle of one setup-owned graph-group slot. */
        enum class GraphGroupSlotState : std::uint8_t
        {
            Available = 0, ///< Slot contains no live descriptor or handles.
            Admitting,     ///< Participants may enter; ticket is not armed.
            Armed,         ///< Remote tickets are published; terminals may arrive.
            AdmittingFailed, ///< A participant failed before ticket publication.
            ArmedFailed,   ///< A participant failed after ticket publication.
            Terminal,      ///< Every participant published its submission terminal.
            Failed,        ///< Every participant retired a failed execution.
        };

        /** @brief One preallocated command slot containing all target handles. */
        struct GraphGroupSlot
        {
            GraphGroupSlotState state = GraphGroupSlotState::Available;
            std::uint64_t group_id = 0;
            std::uint64_t sequence_id = 0;
            int sequence_graph_ordinal = -1;
            int sequence_graph_count = 0;
            MoEOverlayInferenceExecutionDescriptor descriptor{};
            /** One entry mark per symmetric continuation participant. */
            std::vector<std::uint8_t> entered_participants;
            /** One submission-terminal mark per symmetric participant. */
            std::vector<std::uint8_t> terminal_participants;
            /** Rank-wide interval claimed only by a matching prefill group. */
            MoEOverlayInterferenceProbeTicket interference_ticket{};
            /** True after every local terminal owns the probe completion. */
            bool interference_completion_published = false;
            /** Setup-sized participant-local GPU events; null for CPU children. */
            std::vector<std::shared_ptr<IMoEOverlayInferenceCompletionEvent>>
                participant_completion_events;
            /** One mark per exact CPU-call or GPU-event terminal. */
            std::vector<std::uint8_t> participant_completion_recorded;
            std::vector<MoEOverlayPublishedInferenceTransaction>
                target_transactions;

            /** @return Whether this slot is the currently admitted group. */
            [[nodiscard]] bool inFlight() const noexcept
            {
                return state == GraphGroupSlotState::Admitting ||
                       state == GraphGroupSlotState::Armed ||
                       state == GraphGroupSlotState::AdmittingFailed ||
                       state == GraphGroupSlotState::ArmedFailed;
            }

            /** @return Whether remote execution tickets were published. */
            [[nodiscard]] bool armed() const noexcept
            {
                return state == GraphGroupSlotState::Armed ||
                       state == GraphGroupSlotState::ArmedFailed ||
                       state == GraphGroupSlotState::Terminal;
            }

            /** @return Whether any participant reported execution failure. */
            [[nodiscard]] bool failed() const noexcept
            {
                return state == GraphGroupSlotState::AdmittingFailed ||
                       state == GraphGroupSlotState::ArmedFailed ||
                       state == GraphGroupSlotState::Failed;
            }
        };

        /** @brief Transition to Failed while preserving the first diagnostic. */
        bool failLocked(std::string message, std::string *error);
        /** @brief Return whether every local participant has the supplied bit. */
        bool allParticipantsMarked(const std::vector<std::uint8_t> &marks) const;
        /** @brief Validate and retire one completed sequence while mutex_ is held. */
        bool retireCompletedGraphSequenceLocked(std::string *error);
        /** @brief Open one graph sequence while mutex_ is already held. */
        bool beginGraphSequenceLocked(int draft_depth, std::string *error);
        /** @brief Publish the probe terminal once every local boundary exists. */
        bool tryPublishPrefillInterferenceCompletionLocked(
            GraphGroupSlot &transaction,
            std::string *error);
        /** @brief Clear fixed slots and sequence-local role counters. */
        void resetGraphSequenceLocked() noexcept;

        Config config_;
        /** Optional model-lifetime probe; policy/calibration remain host-owned. */
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe>
            prefill_interference_probe_;
        mutable std::mutex mutex_;
        /** Wakes an ahead prefill submitter after the last sibling seals a group. */
        std::condition_variable graph_group_completion_cv_;
        std::vector<GraphGroupSlot> graph_group_slots_;
        /** One reusable aggregate because the probe admits one sample at a time. */
        std::shared_ptr<CompletionFenceSet> prefill_completion_fence_set_;
        MoEOverlayInferenceProtocolState state_ =
            MoEOverlayInferenceProtocolState::Idle;
        MoEOverlayInferenceCommandIdentity active_command_{};
        std::size_t transaction_count_ = 0;
        std::size_t total_transaction_count_ = 0;
        std::size_t graph_sequence_count_ = 0;
        /** Last authenticated hosted transition accepted in this command. */
        std::uint64_t last_hosted_sequence_transition_id_ = 0;
        /** Decision paired with @ref last_hosted_sequence_transition_id_. */
        std::optional<int> last_hosted_sequence_next_draft_depth_;
        /** Cumulative token count paired with the last hosted transition. */
        std::uint64_t last_hosted_sequence_committed_output_tokens_ = 0u;
        ExecutionSequencePlan execution_sequence_{};
        std::uint64_t next_group_id_ = 1;
        std::uint64_t next_sequence_id_ = 1;
        std::uint64_t next_logical_step_id_ = 1;
        std::uint64_t current_placement_epoch_ = 0;
        /** Current aggregate prefill schedule, if explicitly declared. */
        std::optional<MoEOverlayInferenceWorkloadIdentity>
            declared_prefill_schedule_;
        /** Absolute one-based transaction ordinals covered by the declaration. */
        std::size_t declared_prefill_schedule_begin_ordinal_ = 0;
        std::size_t declared_prefill_schedule_end_ordinal_ = 0;
        /** One probe ticket transferred to the final graph's event fence. */
        MoEOverlayInterferenceProbeTicket declared_prefill_probe_ticket_{};
        bool declared_prefill_probe_released_ = false;
        bool declared_prefill_terminal_published_ = false;
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
        /** Cumulative device-authenticated decode progress on Complete. */
        std::uint64_t retired_decode_progress_tokens = 0u;
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
            /** Sole timing owner for complete follower-side transactions. */
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe>
                interference_probe;
            /** Optional process-local device-controller wake sideband. */
            MoEOverlayRetiredPrefillProgressSink
                retired_prefill_progress_sink;
            /** Optional hosted-maintenance wake from the source frontier. */
            MoEOverlayRetiredDecodeProgressSink
                retired_decode_progress_sink;
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
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe>
            interference_probe_;
        MoEOverlayRetiredPrefillProgressSink
            retired_prefill_progress_sink_;
        MoEOverlayRetiredDecodeProgressSink
            retired_decode_progress_sink_;
    };

} // namespace llaminar2
