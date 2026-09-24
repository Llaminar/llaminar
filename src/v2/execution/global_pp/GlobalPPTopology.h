/**
 * @file GlobalPPTopology.h
 * @brief Topology description for Global Pipeline Parallelism
 *
 * Describes a per-machine pipeline topology that composes multiple MPI ranks'
 * resources into a single inference pipeline. Each stage in the pipeline is
 * either:
 * - A single-rank stage (one MPI rank executes, optionally with LocalTP/LocalPP)
 * - A global-TP stage (multiple MPI ranks execute in tensor parallel)
 *
 * The topology is created from CLI/YAML configuration and distributed to all
 * ranks. Each rank derives its own GlobalPPRankPlan from the topology.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../../backends/GlobalDeviceAddress.h"
#include "../../config/CollectiveBackendType.h"
#include "../mpi_orchestration/DeviceInventory.h"
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // Enums
    // =========================================================================

    /**
     * @brief Parallelism mode for devices within a single-rank Global PP stage
     */
    enum class InnerParallelism
    {
        SINGLE_DEVICE, ///< One device, no inner parallelism
        LOCAL_TP,      ///< Tensor parallel across local devices
        LOCAL_PP,      ///< Pipeline parallel across local devices
    };

    /**
     * @brief Convert InnerParallelism to string
     */
    inline const char *innerParallelismName(InnerParallelism mode)
    {
        switch (mode)
        {
        case InnerParallelism::SINGLE_DEVICE:
            return "SINGLE_DEVICE";
        case InnerParallelism::LOCAL_TP:
            return "LOCAL_TP";
        case InnerParallelism::LOCAL_PP:
            return "LOCAL_PP";
        default:
            return "UNKNOWN";
        }
    }

    // =========================================================================
    // Physical connection projection
    // =========================================================================

    /** @return Diagnostic label for the canonical physical connection kind. */
    inline const char *pipelineConnectionName(RankConnectionLocality locality)
    {
        switch (locality)
        {
        case RankConnectionLocality::SameRank: return "SAME_RANK";
        case RankConnectionLocality::SameNode: return "SAME_NODE";
        case RankConnectionLocality::CrossNode: return "CROSS_NODE";
        }
        throw std::invalid_argument("Invalid pipeline physical connection kind");
    }

    // =========================================================================
    // GlobalPPStageSpec
    // =========================================================================

    /**
     * @brief Specification for one stage in the Global PP pipeline
     *
     * A stage is a contiguous range of transformer layers assigned to specific
     * hardware. Stages execute sequentially in the pipeline, with activations
     * transferred between them via MPI send/recv.
     */
    struct GlobalPPStageSpec
    {
        int stage_id = -1; ///< 0-indexed stage number in the pipeline
        std::string domain_name; ///< Optional named execution domain for this stage

        // =====================================================================
        // Layer assignment
        // =====================================================================
        int first_layer = -1;        ///< First layer index (inclusive, 0-based)
        int last_layer = -1;         ///< Last layer index (inclusive, 0-based)
        bool has_embedding = false;  ///< This stage includes the embedding layer
        bool has_lm_head = false;    ///< This stage includes the LM head

        // =====================================================================
        // Ownership
        // =====================================================================
        bool is_global_tp = false;   ///< Multi-rank stage (all participating ranks execute)
        int owning_rank = -1;        ///< For single-rank stages: which MPI rank owns this stage

        // =====================================================================
        // Inner parallelism (single-rank stages only)
        // =====================================================================
        InnerParallelism inner_mode = InnerParallelism::SINGLE_DEVICE;
        std::vector<GlobalDeviceAddress> devices; ///< Devices for this stage
        std::vector<float> tp_weights;            ///< Proportional weights (heterogeneous LocalTP)
        CollectiveBackendType backend = CollectiveBackendType::AUTO; ///< Domain-preferred collective backend

        // =====================================================================
        // Global TP config (multi-rank stages only)
        // =====================================================================
        std::vector<int> participating_ranks;     ///< Which MPI ranks participate (sorted)
        GlobalDeviceAddress per_rank_device;       ///< Device to use on each rank (e.g., cpu:0)
        std::vector<GlobalDeviceAddress> per_rank_devices; ///< Optional device per participating_ranks entry

        // =====================================================================
        // Helpers
        // =====================================================================

        /** @brief Number of layers in this stage */
        int layerCount() const
        {
            return (first_layer >= 0 && last_layer >= first_layer) ? (last_layer - first_layer + 1) : 0;
        }

        /** @brief Check if a given layer belongs to this stage */
        bool hasLayer(int layer) const
        {
            return layer >= first_layer && layer <= last_layer;
        }

        /** @brief Check if a given rank participates in this stage */
        bool rankParticipates(int rank) const
        {
            if (is_global_tp)
            {
                for (int r : participating_ranks)
                {
                    if (r == rank) return true;
                }
                return false;
            }
            return rank == owning_rank;
        }

        /** @brief Validate this stage spec; returns error messages */
        std::vector<std::string> validate() const;
    };

    /**
     * @brief Kind of transfer between adjacent Global PP stages
     */
    enum class GlobalPPTransferKind
    {
        MPI,           ///< Activation moves between MPI ranks
        LOCAL_HANDOFF, ///< Activation is handed off between local stage/domain runners
    };

    /** @brief Convert GlobalPPTransferKind to string */
    inline const char *globalPPTransferKindName(GlobalPPTransferKind kind)
    {
        switch (kind)
        {
        case GlobalPPTransferKind::MPI:
            return "MPI";
        case GlobalPPTransferKind::LOCAL_HANDOFF:
            return "LOCAL_HANDOFF";
        default:
            return "UNKNOWN";
        }
    }

    // =========================================================================
    // GlobalPPTransfer
    // =========================================================================

    /**
     * @brief Describes a transfer of activations between two adjacent Global PP stages
     *
     * Derived automatically from the stage specifications. MPI transfers have
     * distinct sender/receiver ranks; local handoffs explicitly represent a
     * same-rank activation handoff between adjacent stage/domain runners.
     */
    struct GlobalPPTransfer
    {
        GlobalPPTransferKind kind = GlobalPPTransferKind::MPI; ///< Transfer mechanism
        int from_stage = -1;     ///< Source stage ID
        int to_stage = -1;       ///< Destination stage ID
        int sender_rank = -1;    ///< MPI rank that sends, or local handoff rank
        int receiver_rank = -1;  ///< MPI rank that receives, or local handoff rank
        int mpi_tag = 0;         ///< Unique MPI tag for this transfer
        /** Inventory-authored membership; absent only in a structural, unbound plan. */
        std::optional<RankConnectionTopology> connection;

        /**
         * @return Inventory-owned physical identity for these exact endpoints.
         * @throws std::logic_error if this is an unbound or subsequently edited edge.
         *
         * Transport preparation must use this checked view, never a hostname,
         * rank-number heuristic or a missing-observation default.
         */
        [[nodiscard]] const RankConnectionTopology &physicalConnection() const
        {
            if (!connection || connection->sourceRank() != sender_rank ||
                connection->destinationRank() != receiver_rank ||
                (kind != GlobalPPTransferKind::MPI && kind != GlobalPPTransferKind::LOCAL_HANDOFF) ||
                ((kind == GlobalPPTransferKind::LOCAL_HANDOFF) != (sender_rank == receiver_rank)))
                throw std::logic_error("Pipeline transfer has no matching physical inventory binding");
            return *connection;
        }

        /** @brief Check if this transfer is a legacy MPI no-op */
        bool isNoop() const { return kind == GlobalPPTransferKind::MPI && sender_rank == receiver_rank; }

        /** @brief Check if this transfer is a local handoff */
        bool isLocalHandoff() const { return kind == GlobalPPTransferKind::LOCAL_HANDOFF; }
    };

    // =========================================================================
    // GlobalPPTopology
    // =========================================================================

    /**
     * @brief Full pipeline topology in one local or cross-host MPI rank namespace
     *
     * Created from CLI/YAML configuration and broadcast to all ranks.
     * Each rank uses this to derive its own GlobalPPRankPlan.
     *
     * Invariants:
     * - Stages are sorted by stage_id (0, 1, 2, ...)
     * - Stage layer ranges cover all model layers exactly once
     * - Stage 0 has_embedding = true; last stage has_lm_head = true
     * - Global TP stages have ≥2 participating ranks
     * - Single-rank stages have a valid owning_rank
     * - Transfers are derived deterministically from stages
     */
    struct GlobalPPTopology
    {
        std::vector<GlobalPPStageSpec> stages;      ///< Pipeline stages (ordered)
        std::vector<GlobalPPTransfer> transfers;     ///< Inter-stage transfers (derived)
        int total_layers = 0;                        ///< Total transformer layers in model
        int world_size = 0;                          ///< Number of MPI ranks

        // =====================================================================
        // Factory
        // =====================================================================

        /**
         * @brief Build topology from stage specs and derive transfers
         *
         * Sets up structural transfer records between adjacent stages. This
         * geometry-only overload does not bind physical membership; callers
         * inspect validate() for structural errors. Production construction
         * uses the inventory overload below to validate and bind together.
         *
         * @param specs Stage specifications (will be sorted by stage_id)
         * @param total_layers Total number of transformer layers in the model
         * @param world_size Number of MPI ranks
         * @return Constructed topology
         */
        static GlobalPPTopology build(std::vector<GlobalPPStageSpec> specs,
                                      int total_layers, int world_size);

        /**
         * @brief Bind structural topology to the canonical discovery/execution inventory.
         *
         * Each edge retains an immutable membership projection. The topology
         * neither owns another inventory nor independently groups hostnames.
         * @param specs Complete model-stage declarations.
         * @param total_layers Number of main-model layers covered by the stages.
         * @param inventory Exact communicator namespace used by every stage.
         * @return Topology with authenticated physical identity on every edge.
         * @throws std::invalid_argument for incomplete or malformed rank membership.
         */
        static GlobalPPTopology build(std::vector<GlobalPPStageSpec> specs,
                                      int total_layers, const ClusterInventory &inventory);

        // =====================================================================
        // Queries
        // =====================================================================

        /** @brief Get stages that a given rank participates in */
        std::vector<int> stagesForRank(int rank) const;

        /** @brief Get the stage that contains a given layer */
        const GlobalPPStageSpec *stageForLayer(int layer) const;

        /** @brief Check if a rank participates in a given stage */
        bool rankParticipatesInStage(int rank, int stage_id) const;

        /** @brief Get transfer record between two adjacent stages (nullptr if not found) */
        const GlobalPPTransfer *transferBetween(int from_stage, int to_stage) const;

        /** @brief Number of stages */
        int numStages() const { return static_cast<int>(stages.size()); }

        // =====================================================================
        // Validation
        // =====================================================================

        /**
         * @brief Validate the topology for consistency
         * @return List of error messages (empty = valid)
         */
        std::vector<std::string> validate() const;

        /**
         * @brief Human-readable summary for logging
         */
        std::string toString() const;

        /**
         * @brief Formatted ASCII table for --show-topology output
         *
         * Produces a libfort-formatted table showing PP stages, layer ranges,
         * rank assignments, devices, and transfer paths.
         */
        std::string toTable() const;
    };

} // namespace llaminar2
