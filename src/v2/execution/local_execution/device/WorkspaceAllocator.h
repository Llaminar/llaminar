/**
 * @file WorkspaceAllocator.h
 * @brief Standalone workspace allocation for compute graphs
 *
 * Extracted from DeviceGraphBufferManager to decouple workspace management
 * from buffer lifecycle management. Provides model-aware GPU/CPU workspace
 * allocation with per-device budget enforcement. Graph-stage workspace binding
 * is GPU-only; CPU scratch is owned by CPU kernels or higher-level CPU memory
 * managers rather than DeviceWorkspaceManager.
 *
 * @author David Sanftenberg
 * @date March 2026
 */

#pragma once

#include "DeviceWorkspaceManager.h"
#include "WorkspaceDescriptor.h"
#include "../../../backends/DeviceId.h"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;
    class IComputeStage;
    class IWorkspaceConsumer;
    class IBackend;

    /**
     * @brief Declares the physical lifetime relationship between executable graphs.
     *
     * GPU graph executables and CPU stage objects both retain raw workspace
     * addresses. The policy therefore describes both the sizing surface that
     * must be known before capture/materialization and whether another graph may
     * use the same physical bytes.
     *
     * A serial device family includes ordinary prefill/decode, MTP sidecars,
     * grouped verification, accepted-state publication, and decode catch-up on
     * one device. Producer events order every transition between those graphs,
     * so their graph-local layouts may alias one primary allocation. A
     * participant whose state survives that transition instead declares
     * @ref WorkspaceGraphParticipantLifetime::PersistentAcrossParticipants;
     * its buffers remain disjoint while graph-local siblings still alias.
     */
    enum class WorkspaceGraphFamilyPolicy : uint8_t
    {
        /**
         * @brief Storage may overlap other serial participants; size this graph
         *        from its exact declared rows plus decode/compact regimes.
         */
        SerialDeviceFamilyExactParticipant,

        /**
         * @brief Storage may overlap other serial participants; additionally
         *        size row-scaled main-forward scratch for the largest bucket.
         */
        SerialDeviceFamilyLargestParticipant,

        /**
         * @brief Storage can remain live while another graph executes.
         *
         * Late requirements in this class receive append-only storage. This is
         * intentionally exceptional and must be selected explicitly.
         */
        ExclusiveLifetime,
    };

    /**
     * @brief Mathematical execution role of one materialized family graph.
     *
     * Row count cannot identify graph semantics: an M=16 prompt is prefill,
     * while an M=16 speculative continuation is grouped decode. Workspace
     * descriptors use this role to retain prefill-only or compact-only buffers
     * without inferring policy from tensor geometry.
     */
    enum class WorkspaceGraphParticipantRole : uint8_t
    {
        Prefill,                 ///< Prompt rows and prefill-only workspace.
        Decode,                  ///< Ordinary one-row autoregressive decode.
        GroupedVerifier,         ///< Compact serial-row-equivalent MTP verification.
        MTPCondition,            ///< One device-resident main-model row per active request.
        MoERebalanceMaintenance, ///< Device-owned asynchronous MoE maintenance transaction.
    };

    /**
     * @brief Physical lifetime of one graph participant's workspace contents.
     *
     * Execution order alone does not prove that a completed graph's storage is
     * dead. Device MoE maintenance, for example, publishes status and controller
     * records that remain device-owned until a request epilogue exports them.
     * This closed enum makes that retained lifetime part of family declaration
     * instead of relying on buffer names or delayed host behavior.
     */
    enum class WorkspaceGraphParticipantLifetime : uint8_t
    {
        /**
         * @brief Every workspace value is dead after the participant handoff.
         */
        SerialGraphLocal,

        /**
         * @brief Workspace contents remain live while other family graphs run.
         */
        PersistentAcrossParticipants,
    };

    /**
     * @brief Typed non-owning declaration of one exact graph-family member.
     */
    struct WorkspaceGraphParticipant
    {
        const ComputeGraph *graph = nullptr; ///< Materialized production topology.
        WorkspaceGraphParticipantRole role =
            WorkspaceGraphParticipantRole::Decode; ///< Explicit execution semantics.
        WorkspaceGraphParticipantLifetime lifetime =
            WorkspaceGraphParticipantLifetime::
                SerialGraphLocal; ///< Whether this participant may physically alias siblings.
    };

    /**
     * @brief Configuration for workspace memory budget calculation
     *
     * Controls how WorkspaceAllocator computes workspace budgets for GPU and CPU
     * devices. The budget is calculated as:
     *   budget = min(max(available * fraction - headroom, min_budget), max_budget)
     */
    struct WorkspaceBudgetConfig
    {
        float gpu_fraction = 0.8f;                     ///< Fraction of free GPU memory to use (0.0-1.0)
        float cpu_fraction = 0.3f;                     ///< Fraction of free CPU memory to use (conservative)
        size_t min_budget = 64 * 1024 * 1024;          ///< Minimum budget (64MB)
        size_t max_budget = 4ULL * 1024 * 1024 * 1024; ///< Maximum budget (4GB)
        size_t headroom = 128 * 1024 * 1024;           ///< Reserved headroom (128MB)
    };

    /**
     * @brief Model-aware sizing hints for workspace consumers
     *
     * Used to derive per-stage workspace dimensions from model metadata.
     */
    struct WorkspaceSizingHints
    {
        int max_seq_len = 4096;
        /**
         * @brief Largest row count that any serial forward graph may execute.
         *
         * This field is consulted only when @ref graph_family_policy is
         * `SerialDeviceFamilyLargestParticipant`. It is separate from
         * `max_seq_len`, which remains the exact shape of the graph currently
         * being bound.
         */
        int serial_family_max_rows = 0;
        /**
         * @brief Largest compact decode-equivalent row group in this family.
         *
         * Workspace demand is not monotonic in M: a large prefill may select a
         * GEMM that needs no K-partition partials while M=2..16 selects grouped
         * GEMV and needs a larger reduction bank. Querying this explicit regime
         * before the first capture keeps common names address-stable for every
         * configured MTP depth.
         */
        int serial_family_max_compact_rows = 0;
        /**
         * @brief Largest output width owned by a terminal projection in this family.
         *
         * A phase-split LocalTP family can capture column-parallel prefill
         * first and bind a replicated full-vocabulary LM head for decode or
         * grouped MTP verification later. Those graph participants share
         * stable workspace names even though their output widths differ.
         * Advertising the family envelope before the first capture lets the
         * allocator publish one address with enough capacity for every
         * participant; changing that address after capture remains forbidden.
         *
         * Zero preserves the consumer's own prepared width. A positive value
         * is an envelope, not an unconditional replacement: callers that
         * already declare a wider projection keep that wider value.
         */
        int serial_family_max_terminal_projection_columns = 0;

        /**
         * @brief Resolve a participant's terminal projection width.
         *
         * This pure helper makes the width-envelope policy independently
         * testable without constructing a GPU graph or allocating device
         * memory. The returned zero retains the existing
         * `IWorkspaceConsumer` convention that the prepared kernel supplies
         * its own N dimension.
         *
         * @param participant_columns Width explicitly declared by the current
         *        graph participant, or zero to use the prepared kernel width.
         * @return The widest declared participant/family projection width.
         */
        [[nodiscard]] constexpr int resolveTerminalProjectionColumns(
            int participant_columns) const noexcept
        {
            return serial_family_max_terminal_projection_columns >
                           participant_columns
                       ? serial_family_max_terminal_projection_columns
                       : participant_columns;
        }

        int n_heads = 0;
        int head_dim = 0;
        int d_model = 0;
        int batch_size = 1;
        int vocab_size = 0;
        WorkspaceGraphFamilyPolicy graph_family_policy =
            WorkspaceGraphFamilyPolicy::ExclusiveLifetime;
    };

    /**
     * @brief Defines how an explicit workspace consumer interprets its M value.
     *
     * Most kernel workspaces use M as graph token rows and must be queried for
     * each serial participant. Control-plane workspaces can use M for another
     * dimension, such as request count; substituting prompt or verifier rows in
     * that case silently changes the declared data structure.
     */
    enum class WorkspaceConsumerShapePolicy : uint8_t
    {
        /**
         * @brief Replace M with each prefill/decode/verifier participant's rows.
         */
        GraphParticipantRows,

        /**
         * @brief Preserve the request's explicit M, N, and K for every participant.
         */
        FixedDeclaredShape,
    };

    /**
     * @brief Explicit workspace request for non-graph consumers
     */
    struct WorkspaceConsumerRequest
    {
        IWorkspaceConsumer *consumer = nullptr;
        DeviceId device;
        int m = 4096;
        int n = 0;
        int k = 0;
        WorkspaceConsumerShapePolicy shape_policy =
            WorkspaceConsumerShapePolicy::GraphParticipantRows;
    };

    /**
     * @brief Standalone workspace allocator for compute graphs
     *
     * Manages per-device workspace allocation with model-aware budget calculation.
     * Each device gets a single DeviceWorkspaceManager with a computed budget.
     *
     * ## Usage
     *
     * ```cpp
     * WorkspaceAllocator allocator;
     *
     * WorkspaceSizingHints hints;
     * hints.max_seq_len = 4096;
     * hints.n_heads = 32;
     * // ...
     *
     * allocator.allocateForGraph(graph, hints);
     * ```
     *
     * ## Thread Safety
     *
     * NOT thread-safe. Should be used from a single thread.
     */
    class WorkspaceAllocator
    {
    public:
        WorkspaceAllocator() = default;
        ~WorkspaceAllocator() = default;

        // Non-copyable
        WorkspaceAllocator(const WorkspaceAllocator &) = delete;
        WorkspaceAllocator &operator=(const WorkspaceAllocator &) = delete;

        // Movable
        WorkspaceAllocator(WorkspaceAllocator &&) = default;
        WorkspaceAllocator &operator=(WorkspaceAllocator &&) = default;

        // =====================================================================
        // Memory Query
        // =====================================================================

        /**
         * @brief Query available memory for a device
         *
         * Uses the appropriate backend (CPU, CUDA, ROCm) to query free memory.
         *
         * @param device Target device
         * @return Available bytes
         */
        size_t queryAvailableMemory(DeviceId device);

        /**
         * @brief Compute workspace budget for a device
         *
         * Applies budget configuration to compute an appropriate workspace size.
         *
         * @param device Target device
         * @param config Budget configuration
         * @return Computed budget in bytes
         */
        size_t computeWorkspaceBudget(DeviceId device,
                                      const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        // =====================================================================
        // Allocation
        // =====================================================================

        /**
         * @brief Allocate workspace for all CPU/GPU consumers in a graph
         *
         * Scans graph stages for IWorkspaceConsumer implementations, derives
         * per-stage dimension hints, allocates per-device workspace, and binds
         * all consumers. Optional extra consumers (e.g., KV cache) can be provided.
         *
         * @param graph The compute graph
         * @param hints Model-aware sizing hints
         * @param extra_consumers Additional workspace consumers outside the graph
         * @param config Budget configuration
         * @return true if all allocations succeeded
         */
        bool allocateForGraph(
            const ComputeGraph &graph,
            const WorkspaceSizingHints &hints,
            const std::vector<WorkspaceConsumerRequest> &extra_consumers = {},
            const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        /**
         * @brief Allocate one stable workspace for a complete serial graph family.
         *
         * GPU graph executables retain the raw addresses returned by
         * DeviceWorkspaceManager.  Consequently every graph that can run in the
         * same event-ordered request lifetime must participate in the first
         * layout decision. This method treats @p primary_graph according to its
         * explicit mathematical role, including any configured family envelope,
         * and treats every graph in
         * @p exact_serial_participants as a distinct exact-shape participant
         * with an explicit mathematical execution role.
         *
         * Distinct participants may reuse physical bytes because their producer
         * and consumer events serialize execution.  Shared workspace names
         * nevertheless receive one family-wide capacity and address, so a later
         * graph capture cannot observe a smaller allocation than it requires.
         * All participant consumers are bound only after the complete layout has
         * been allocated successfully.
         *
         * @param primary_graph Main forward graph whose sizing policy is
         *        described by @p hints.
         * @param primary_role Mathematical role of the primary graph. This may
         *        not be inferred from M because prompt and verifier M overlap.
         * @param exact_serial_participants Additional already-materialized
         *        graph topologies, each queried at its own declared stage shape.
         * @param hints Model and graph-family sizing policy.
         * @param extra_consumers Non-graph consumers that share the primary
         *        participant lifetime.
         * @param config Workspace budget policy.
         * @return true when one stable allocation covers and binds every member.
         */
        bool allocateForGraphFamily(
            const ComputeGraph &primary_graph,
            WorkspaceGraphParticipantRole primary_role,
            const std::vector<WorkspaceGraphParticipant> &
                exact_serial_participants,
            const WorkspaceSizingHints &hints,
            const std::vector<WorkspaceConsumerRequest> &extra_consumers = {},
            const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        /**
         * @brief Allocate workspace for a flat list of stages
         *
         * @param stages Stages to scan for IWorkspaceConsumer
         * @param config Budget configuration
         * @return true if all allocations succeeded
         */
        bool allocateForStages(const std::vector<IComputeStage *> &stages,
                               const WorkspaceBudgetConfig &config = WorkspaceBudgetConfig{});

        /**
         * @brief Release all workspace allocations
         */
        void releaseAll();

        // =====================================================================
        // Access
        // =====================================================================

        /**
         * @brief Get workspace manager for a device
         * @param device Target device
         * @return Workspace manager (nullptr if not allocated)
         */
        DeviceWorkspaceManager *getDeviceWorkspace(DeviceId device);

        /**
         * @brief Return the current workspace address epoch for a device.
         *
         * The epoch changes whenever the device workspace manager is replaced
         * or released. Cached graphs use this to detect that captured GPU graph
         * replay state contains stale raw workspace pointers.
         *
         * @param device Target device.
         * @return Monotonic generation for the device, or 0 if it has never had workspace.
         */
        uint64_t deviceGeneration(DeviceId device) const;

        // =====================================================================
        // Metrics
        // =====================================================================

        /**
         * @brief Total workspace allocated across all devices
         */
        size_t totalAllocated() const;

        /**
         * @brief Workspace allocated for a specific device
         */
        size_t deviceAllocated(DeviceId device) const;

        /**
         * @brief Number of devices with workspace allocated
         */
        size_t deviceCount() const { return device_workspaces_.size(); }

    private:
        /// Per-device workspace managers
        std::unordered_map<DeviceId, std::unique_ptr<DeviceWorkspaceManager>> device_workspaces_;

        /// Per-device workspace budgets (for metrics)
        std::unordered_map<DeviceId, size_t> device_workspace_budgets_;

        /// Per-device workspace address epochs for cached-graph invalidation.
        std::unordered_map<DeviceId, uint64_t> device_workspace_generations_;

        /// Monotonic source for device workspace generations.
        uint64_t next_workspace_generation_ = 1;

        /**
         * @brief Advance the generation for a device after workspace addresses change.
         */
        void bumpDeviceGeneration(DeviceId device);

        /**
         * @brief Compute model-aware minimum budget floor
         */
        size_t computeModelAwareBudgetFloor(const WorkspaceSizingHints &hints) const;
    };

} // namespace llaminar2
