/**
 * @file ModelParityDefinition.h
 * @brief Typed model/topology definitions and canonical parity-matrix expansion.
 *
 * Every production model-parity campaign is described by one real model, one
 * topology, and a set of standard capability profiles.  The framework expands
 * those declarations over activation precision, KV-cache precision, MTP, and
 * (when the topology is an ExpertOverlay) physical placement and durable
 * movement. Prefix restore is deliberately not a matrix axis: every generated
 * production case must prove a fresh fill, a full restore, and a partial
 * restore through one retained runner. Generated test names are output only;
 * runtime policy is projected directly from the typed case and is never
 * recovered by parsing a GoogleTest name.
 */

#pragma once

#include "ParityTestBase.h"

#include "backends/GlobalDeviceAddress.h"
#include "config/ActivationPrecisionPolicy.h"
#include "execution/config/RoutedExpertPolicy.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llaminar2::test::parity
{
    /** Deepest fixed/adaptive MTP draft policy in the canonical matrix. */
    inline constexpr int kModelParityRequiredMaximumMTPDepth = 15;

    /** MTP execution policy selected by one generated parity case. */
    enum class ModelParityMTP : std::uint8_t
    {
        Off,
        Depth1,
        Depth2,
        Depth3,
        Depth15,
        DynamicDepth,
    };

    /**
     * @brief Phase-aware route evidence retained by one ExpertOverlay cell.
     *
     * Main-model route ledgers execute in ordinary prefill and decode. MTP
     * sidecars execute only inside speculative transactions, but their graph
     * outputs still have to be selected before the serving graphs are sealed.
     * Keeping these inventories distinct prevents setup selection from being
     * confused with a claim that every request phase executes every graph.
     */
    struct ModelParityExpertOverlayRouteSnapshotInventory
    {
        /** Route ledgers that every ordinary main-model forward must publish. */
        std::vector<std::string> main_model;
        /** Route ledgers selected at setup and proved by live MTP transactions. */
        std::vector<std::string> mtp_sidecar;
    };

    /**
     * @brief Build the complete phase-aware pinned-route inventory for a case.
     *
     * ExpertOverlay movement proofs need the global placement banks and the
     * invocation-local route projection that one ordered reducer consumed.
     * GPU snapshot manifests filter individual outputs, not merely stages, so
     * naming `MOE_EXPERT_OUTPUT` does not implicitly retain these companion
     * values. MTP sidecars reuse an `MTP0_` graph inside context-qualified
     * transaction banks; those keys belong to setup capture and the explicit
     * speculative-transaction proof, not ordinary prefill validation.
     *
     * @param main_layer_count Number of ordinary transformer layers.
     * @param mtp Generated MTP policy for the exact matrix cell.
     * @return Stable main-model and MTP-sidecar semantic key inventories.
     */
    inline ModelParityExpertOverlayRouteSnapshotInventory
    modelParityExpertOverlayRouteSnapshotInventory(
        int main_layer_count,
        ModelParityMTP mtp)
    {
        if (main_layer_count <= 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay pinned-route inventory requires a positive main-layer count");
        }

        static constexpr std::array<std::string_view, 8> kRouteSuffixes{
            "_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
            "_MOE_RUNTIME_ROUTE_WEIGHTS",
            "_MOE_ROUTE_CONTRIBUTIONS",
            "_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
            "_MOE_OVERLAY_ROUTE_BANK0_EPOCH",
            "_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
            "_MOE_OVERLAY_ROUTE_BANK1_EPOCH",
            "_MOE_OVERLAY_ROUTE_SELECTED_BANK",
        };

        ModelParityExpertOverlayRouteSnapshotInventory inventory;
        inventory.main_model.reserve(
            static_cast<std::size_t>(main_layer_count) *
            kRouteSuffixes.size());
        const auto append_prefix = [&](std::vector<std::string> &keys,
                                       const std::string &prefix)
        {
            for (const std::string_view suffix : kRouteSuffixes)
                keys.push_back(prefix + std::string(suffix));
        };
        for (int layer = 0; layer < main_layer_count; ++layer)
        {
            append_prefix(
                inventory.main_model,
                "layer" + std::to_string(layer));
        }
        if (mtp != ModelParityMTP::Off)
        {
            inventory.mtp_sidecar.reserve(kRouteSuffixes.size());
            append_prefix(inventory.mtp_sidecar, "MTP0");
        }
        return inventory;
    }

    /** Geometry policy for the non-optional prefix restore proof. */
    enum class ModelParityPrefixRestoreGeometry : std::uint8_t
    {
        /**
         * Store the authenticated prompt as one terminal block, then append
         * one reference token to prove a real partial restore. The fixture
         * resolves the block width only after token identity is authenticated.
         */
        AuthenticatedPromptBlock,
    };

    /** Durable ExpertOverlay residency policy selected by one generated case. */
    enum class ModelParityExpertMovement : std::uint8_t
    {
        Static,
        Dynamic,
    };

    /** Exact orchestration shape expressed by a general parity topology. */
    enum class ModelParityTopologyKind : std::uint8_t
    {
        SingleDevice,
        RankLocalTensorParallel,
        RankLocalPipelineParallel,
        NodeTensorParallel,
        NodePipelineParallel,
        GlobalTensorParallel,
        RankLocalMultiDomain,
        NodeMultiDomain,
        GlobalMultiDomain,
    };

    /** Whether a standard optional axis is disabled or fully expanded. */
    enum class ModelParityAxisProfile : std::uint8_t
    {
        Disabled,
        Standard,
    };

    /** Production prefill graph schedule exercised by one generated cell. */
    enum class ModelParityPrefillGraphMode : std::uint8_t
    {
        /** Use the ordinary production bucket selection for the prompt. */
        Standard,
        /** Replay the prompt through one fixed captured segment bucket. */
        SegmentedCaptured,
    };

    /**
     * @brief Typed prefill graph profile crossed by the central expander.
     *
     * A segmented profile names the exact physical row bucket retained by the
     * production graph cache.  Keeping this value in the generated case makes
     * graph scheduling policy independent of GoogleTest names and lets focused
     * heterogeneous graph proofs use the same declarative machinery as every
     * ordinary production-parity cell.
     */
    struct ModelParityPrefillGraphPolicy
    {
        ModelParityPrefillGraphMode mode =
            ModelParityPrefillGraphMode::Standard;
        int captured_rows = 0;

        /** @return Stable optional name fragment for generated diagnostics. */
        [[nodiscard]] std::string testNameFragment() const
        {
            return mode == ModelParityPrefillGraphMode::Standard
                       ? std::string{}
                       : "SegmentedPrefillRows" +
                             std::to_string(captured_rows);
        }

        /** @return Whether this profile requires fixed segmented replay. */
        [[nodiscard]] constexpr bool isSegmentedCaptured() const noexcept
        {
            return mode ==
                   ModelParityPrefillGraphMode::SegmentedCaptured;
        }

        friend bool operator==(
            const ModelParityPrefillGraphPolicy &,
            const ModelParityPrefillGraphPolicy &) = default;
    };

    /** Request-time movement evidence required from PerfStats. */
    enum class ModelParityMovementEvidence : std::uint8_t
    {
        NotApplicable,
        NoMovement,
        MovementRequired,
    };

    /**
     * End-to-end economy evidence owned by a generated Dynamic cell.
     *
     * Every Dynamic cell must certify production economics and publish real
     * movement.  A definition may additionally select an explicit owner-order
     * representative for the costly matched before/after speed proof. MTP
     * depth, alternate prefill schedules, and additional precision pairs remain
     * independent numerical axes and do not repeat that identical benchmark.
     */
    enum class ModelParityDynamicEvidence : std::uint8_t
    {
        NotApplicable,
        EconomicMovement,
        EconomicMovementAndObservedSpeedup,
    };

    /**
     * @brief Owner-order cohort selected for one matched Dynamic speed proof.
     *
     * Physical movement remains mandatory in every Dynamic matrix cell.  This
     * policy only assigns the substantially more expensive before/after timing
     * cohort.  Keeping the selection in the model definition prevents the
     * expander from multiplying one economy benchmark across every topology,
     * precision pair, and owner order merely because a CPU participant exists.
     */
    enum class ModelParityDynamicSpeedupWitness : std::uint8_t
    {
        Disabled,
        Ordinal,
        Random,
        BothOwnerOrders,
    };

    /**
     * @brief Return whether an owner order owns the definition's speed witness.
     *
     * @param witness Definition-selected matched timing cohort.
     * @param owner_order Physical owner placement of the generated cell.
     * @return True exactly for owner orders selected by @p witness.
     */
    [[nodiscard]] constexpr bool selectsDynamicSpeedupWitness(
        ModelParityDynamicSpeedupWitness witness,
        RoutedExpertOwnerOrder owner_order) noexcept
    {
        switch (witness)
        {
        case ModelParityDynamicSpeedupWitness::Disabled:
            return false;
        case ModelParityDynamicSpeedupWitness::Ordinal:
            return owner_order == RoutedExpertOwnerOrder::Ordinal;
        case ModelParityDynamicSpeedupWitness::Random:
            return owner_order == RoutedExpertOwnerOrder::Random;
        case ModelParityDynamicSpeedupWitness::BothOwnerOrders:
            return true;
        }
        return false;
    }

    /**
     * @brief Runtime policies selected by the Dynamic evidence a cell owns.
     *
     * Movement-only cells prove one real economically admitted publication and
     * then preserve that epoch for numerical parity.  The designated speed
     * witness may deliberately use a wider, longer observation horizon to
     * measure convergence.  Keeping both policies in the canonical definition
     * prevents fixtures from silently rewriting controller geometry after
     * capacity admission.
     */
    struct ModelParityDynamicRuntimePolicies
    {
        MoERebalanceRuntimeConfig economic_movement;
        MoERebalanceRuntimeConfig economic_movement_and_observed_speedup;

        /**
         * @brief Select the exact policy owned by a generated evidence cell.
         *
         * Static and non-overlay cells use the movement policy as their stable
         * configuration identity; @ref ModelParityCase::applyRuntimePolicy
         * disables its controller mode before execution.
         *
         * @param evidence Evidence assigned by the canonical matrix expander.
         * @return Definition-owned immutable runtime policy.
         */
        [[nodiscard]] const MoERebalanceRuntimeConfig &policyFor(
            ModelParityDynamicEvidence evidence) const noexcept
        {
            return evidence ==
                           ModelParityDynamicEvidence::
                               EconomicMovementAndObservedSpeedup
                       ? economic_movement_and_observed_speedup
                       : economic_movement;
        }
    };

    /** MTP execution evidence required from PerfStats and numerical artifacts. */
    enum class ModelParityMTPEvidence : std::uint8_t
    {
        Disabled,
        FixedDepth,
        DynamicDepth,
    };

    /**
     * @brief Immutable real-model and authenticated reference identity.
     *
     * `reference_directory` contains the CPU/PyTorch-Hugging Face checkpoint
     * pack authenticated against `model_path`.  A positive maximum MTP depth
     * declares that the reference and model can certify recursive predictors;
     * the standard MTP profile requires capacity through depth 15.
     */
    struct ModelParityModelDefinition
    {
        std::string test_id;             ///< Stable GoogleTest-safe model identifier.
        std::string model_path;          ///< Default real GGUF path or first split.
        std::string reference_directory; ///< Authenticated reference pack.
        std::string prompt;              ///< Exact reference prompt, if overridden.
        std::vector<int> token_ids;       ///< Exact reference tokens, if pre-resolved.
        int decode_steps = 0;             ///< Zero retains the model fixture default.
        int max_seq_len = 4096;           ///< Production context admission for the case.
        /**
         * Transformer blocks in the authenticated GGUF.
         *
         * This is required for an explicitly staged pipeline topology and for
         * any ExpertOverlay whose immutable snapshot graph names per-layer
         * route publications before model loading. The production runner or
         * fixture validates the declaration against the loaded model, so a
         * stale value fails during setup rather than silently constructing a
         * different graph identity.
         */
        int transformer_layers = 0;
        /**
         * Query-attention heads in the authenticated model.
         *
         * Cross-rank TP currently owns one uniform physical shard per MPI
         * participant, so its admissible degree is a model property and must
         * be checked before a campaign is registered.
         */
        int attention_heads = 0;
        /** KV-attention heads used by the cross-rank TP admissibility proof. */
        int kv_heads = 0;
        int maximum_mtp_draft_depth = 0;  ///< Zero means the model has no MTP lane.
        /**
         * Exact recursive checkpoints materialized by the optimized graph.
         *
         * An empty list means every authenticated reference checkpoint is
         * observable and therefore mandatory.  A non-empty list is the typed
         * production cut for fused models: the campaign compares every named
         * value and must not create an eager/test-only intermediate merely
         * because the Hugging Face oracle can expose it.
         */
        std::vector<std::string> mtp_checkpoint_surface;
    };

    /**
     * @brief One physical participant and the MPI rank that owns it.
     *
     * Rank ownership is part of topology identity.  It cannot be inferred from
     * declaration order because one process may own several accelerators while
     * another owns one CPU NUMA participant.  Keeping the relationship in one
     * value also prevents model fixtures from independently rebuilding a rank
     * map from backend counts.
     */
    struct ModelParityParticipant
    {
        GlobalDeviceAddress address;
        /**
         * Fixed MPI owner, or no value when production inventory binding owns
         * rank selection. An unresolved owner is valid only for a cross-rank
         * topology; rank-local definitions must name rank zero exactly.
         */
        std::optional<int> world_rank = 0;

        friend bool operator==(
            const ModelParityParticipant &,
            const ModelParityParticipant &) = default;
    };

    /**
     * @brief General typed topology shared by dense, MoE, and overlay parity.
     *
     * Participant addresses preserve accelerator ordinals and optional NUMA
     * intent.  `kind`, collectives, MPI size, and pipeline grouping replace the
     * parallel boolean/string tables historically repeated in each model file.
     * An ExpertOverlay topology additionally owns one immutable declarative
     * placement-plan blueprint.  Each generated case copies that blueprint
     * before installing placement and movement policy.
     */
    struct ModelParityTopologyDefinition
    {
        std::string test_id; ///< Stable backend/topology identity.
        ModelParityTopologyKind kind =
            ModelParityTopologyKind::SingleDevice;
        std::vector<ModelParityParticipant> participants;
        Collective collective = Collective::None;
        int mpi_ranks = 1;
        std::vector<int> pipeline_stage_sizes;
        std::vector<float> pipeline_weights;
        Collective tensor_parallel_collective = Collective::None;
        std::shared_ptr<const MoERoutedExpertPlacementPlan>
            expert_overlay_plan; ///< Null for a non-overlay topology.

        /** @return Whether this topology enables ExpertOverlay matrix axes. */
        [[nodiscard]] bool isExpertOverlay() const noexcept
        {
            return expert_overlay_plan != nullptr;
        }
    };

    /** Precision axes crossed for one model/topology definition. */
    struct ModelParityPrecisionThresholdOverride
    {
        ActivationPrecision activation = ActivationPrecision::FP32;
        KVCachePrecision kv_cache = KVCachePrecision::FP16;
        BackendThresholds thresholds;
    };

    /** Precision axes crossed for one model/topology definition. */
    struct ModelParityPrecisionMatrix
    {
        std::vector<ActivationPrecision> activation = {
            ActivationPrecision::FP32,
        };
        std::vector<KVCachePrecision> kv_cache = {
            KVCachePrecision::FP16,
        };
        /** Exact pair-specific numerical contracts; unspecified pairs use the definition default. */
        std::vector<ModelParityPrecisionThresholdOverride>
            threshold_overrides;
    };

    /** One MTP-policy-specific override of the recursive logit KL budget. */
    struct ModelParityMTPKLThresholdOverride
    {
        ModelParityMTP policy = ModelParityMTP::Off;
        float maximum_kl_divergence = 0.0f;
    };

    /** One deep-MTP-policy-specific override of the recursive cosine floor. */
    struct ModelParityMTPAggregateCosineThresholdOverride
    {
        ModelParityMTP policy = ModelParityMTP::Off;
        float minimum_cosine_similarity = 0.0f;
    };

    /** Optional standard feature axes for one model/topology definition. */
    struct ModelParityFeatureMatrix
    {
        ModelParityAxisProfile mtp = ModelParityAxisProfile::Disabled;
        /**
         * Matched Dynamic timing cohort for this exact model/topology.
         *
         * Disabled is deliberate: ordinary definitions certify movement and
         * numerical parity without silently becoming performance benchmarks.
         * A selected witness is emitted only for the first activation/KV pair,
         * ordinary prefill schedule, and MTP-off policy.
         */
        ModelParityDynamicSpeedupWitness dynamic_speedup_witness =
            ModelParityDynamicSpeedupWitness::Disabled;
        /**
         * Prefill graph schedules crossed with precision and execution policy.
         * One ordinary profile preserves the historical matrix by default.
         */
        std::vector<ModelParityPrefillGraphPolicy> prefill_graph = {
            ModelParityPrefillGraphPolicy{},
        };
        /**
         * Policy-local recursive KL budgets applied after precision overrides.
         *
         * Keeping this keyed by the typed MTP policy prevents a deep-recursion
         * tolerance from weakening policies that cannot reach the authorized
         * recurrence. Adaptive depth must be named explicitly when its
         * admitted ceiling reaches the same recurrence as a fixed-depth cell.
         */
        std::vector<ModelParityMTPKLThresholdOverride>
            mtp_kl_threshold_overrides;
        /**
         * Policy-local recursive aggregate floors for depth-15-capable cells.
         *
         * Individual checkpoint, routing, KL, ordinary decode, and prefill
         * contracts remain unchanged. Fixed shallow policies are deliberately
         * ineligible, making a deep quantized-recurrence allowance explicit.
         */
        std::vector<ModelParityMTPAggregateCosineThresholdOverride>
            mtp_recursive_aggregate_cosine_threshold_overrides;
    };

    /** HTTP reasoning coverage is declared, never inferred from a GGUF filename. */
    enum class ModelParityE2EThinkingModes
    {
        NonThinkingOnly,
        ThinkingAndNonThinking,
    };

    /** @return Stable harness spelling for one validated reasoning-coverage policy. */
    inline const char *modelParityE2EThinkingModesName(ModelParityE2EThinkingModes modes)
    {
        switch (modes)
        {
        case ModelParityE2EThinkingModes::NonThinkingOnly: return "non-thinking";
        case ModelParityE2EThinkingModes::ThinkingAndNonThinking: return "both";
        }
        throw std::invalid_argument("invalid E2E thinking coverage policy");
    }

    /** Full HTTP needle/long-context workload, independent of numerical prompt size. */
    struct ModelParityE2EProfile
    {
        int context_length = 8192;
        int minimum_prompt_tokens = 4096;
        int generation_tokens = 2048;
        int request_timeout_seconds = 600;
        /** Startup readiness budget; independent of the request and exact-cell watchdogs. */
        int readiness_timeout_seconds = 60;
        /** Test both reasoning modes by default, including renamed fine-tunes. */
        ModelParityE2EThinkingModes thinking_modes = ModelParityE2EThinkingModes::ThinkingAndNonThinking;
        friend bool operator==(const ModelParityE2EProfile &,
                               const ModelParityE2EProfile &) = default;
    };

    /**
     * @brief Opt one existing configuration into full HTTP certification.
     *
     * The containing definition owns model and topology. All remaining axes
     * are exact, typed matches: tags neither multiply nor prune parity cells.
     * An absent owner/movement pair selects non-overlay cells only. Expansion
     * rejects a tag that matches no cell or overlaps another tag.
     */
    struct ModelParityE2ESelection
    {
        ActivationPrecision activation = ActivationPrecision::FP32;
        KVCachePrecision kv_cache = KVCachePrecision::FP16;
        ModelParityMTP mtp = ModelParityMTP::Off;
        std::optional<RoutedExpertOwnerOrder> owner_order;
        std::optional<ModelParityExpertMovement> movement;
        ModelParityPrefillGraphPolicy prefill_graph;
        ModelParityE2EProfile profile;
    };

    /**
     * @brief Complete suite-owned declaration for one model and one topology.
     *
     * Backend-specific numerical tolerances belong to the pair, while all
     * execution policy dimensions are generated by @ref expandModelParityDefinition.
     */
    struct ModelParityDefinition
    {
        ModelParityModelDefinition model;
        ModelParityTopologyDefinition topology;
        BackendThresholds thresholds;
        ModelParityPrecisionMatrix precisions;
        ModelParityFeatureMatrix features;
        MoEHotExpertCacheConfig moe_hot_expert_cache;
        RoutedExpertPrefillRuntimeConfig moe_routed_prefill;
        ModelParityDynamicRuntimePolicies dynamic_rebalance;
        ParityGraphSnapshotPolicy graph_snapshot_policy;
        std::optional<ParityCollectiveEvidenceSource>
            collective_evidence_source;
        std::string tp_allreduce_precision_override;
        std::vector<ModelParityE2ESelection> e2e_certifiable;
    };

    /** Placement and durable movement axes present only on ExpertOverlay. */
    struct ModelParityExpertOverlayPolicy
    {
        RoutedExpertOwnerOrder owner_order =
            RoutedExpertOwnerOrder::Ordinal;
        ModelParityExpertMovement movement =
            ModelParityExpertMovement::Static;

        /** @return Stable GoogleTest-safe name fragment. */
        [[nodiscard]] std::string testName() const
        {
            const char *movement_name =
                movement == ModelParityExpertMovement::Static
                    ? "Static"
                    : "Dynamic";
            const char *owner_name =
                owner_order == RoutedExpertOwnerOrder::Ordinal
                    ? "Ordinal"
                    : "Random";
            return std::string(movement_name) + "_" + owner_name;
        }

        /** @return Request-local PerfStats contract for this movement policy. */
        [[nodiscard]] constexpr ModelParityMovementEvidence
        movementEvidence() const noexcept
        {
            return movement == ModelParityExpertMovement::Static
                       ? ModelParityMovementEvidence::NoMovement
                       : ModelParityMovementEvidence::MovementRequired;
        }

        friend constexpr bool operator==(
            const ModelParityExpertOverlayPolicy &,
            const ModelParityExpertOverlayPolicy &) = default;
    };

    /**
     * @brief One fully specified case emitted by the canonical generator.
     *
     * Every selectable field is explicit even when disabled. Prefix restore is
     * instead a non-optional production-campaign invariant, so every case owns
     * the same typed proof geometry and fixtures never branch on a prefix mode.
     */
    struct ModelParityCase
    {
        ModelParityModelDefinition model;
        ModelParityTopologyDefinition topology;
        BackendThresholds thresholds;
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
        KVCachePrecision kv_cache_precision = KVCachePrecision::FP16;
        ModelParityMTP mtp = ModelParityMTP::Off;
        /** Explicit recursive aggregate floor; absent retains the global 0.99. */
        std::optional<float> mtp_recursive_aggregate_cosine_floor;
        /**
         * Setup-time MTP envelope shared by every cell in this definition.
         *
         * Zero means the definition has no MTP axis. A positive value is kept
         * even by the MTPOff execution cell so one process-resident model and
         * ExpertOverlay placement authority can serve the entire generated
         * matrix without changing its physical capacity solution.
         */
        int retained_mtp_draft_capacity = 0;
        std::optional<ModelParityExpertOverlayPolicy> expert_overlay;
        ModelParityDynamicEvidence dynamic_evidence =
            ModelParityDynamicEvidence::NotApplicable;
        ModelParityPrefillGraphPolicy prefill_graph;
        ModelParityPrefixRestoreGeometry prefix_restore_geometry =
            ModelParityPrefixRestoreGeometry::AuthenticatedPromptBlock;
        MoEHotExpertCacheConfig moe_hot_expert_cache;
        RoutedExpertPrefillRuntimeConfig moe_routed_prefill;
        MoERebalanceRuntimeConfig dynamic_rebalance;
        ParityGraphSnapshotPolicy graph_snapshot_policy;
        std::optional<ParityCollectiveEvidenceSource>
            collective_evidence_source;
        std::string tp_allreduce_precision_override;

        /** Eligibility is not a certificate: the HTTP runner must prove it. */
        std::optional<ModelParityE2EProfile> e2e_certification;

        /** @return Whether this cell executes an MTP generation policy. */
        [[nodiscard]] constexpr bool mtpEnabled() const noexcept
        {
            return mtp != ModelParityMTP::Off;
        }

        /**
         * @brief Whether this cell owns the TP>2 canonical MTP reduction proof.
         *
         * Canonical rank-order reduction exists to make a grouped verifier row
         * byte-identical to its serial decode row when a homogeneous GPU
         * continuation domain has more than two participants.  A TP1/TP2,
         * CPU, heterogeneous-continuation, or MTP-off cell has no such graph
         * value and must not require its diagnostic snapshots.
         *
         * @return True exactly when the typed topology can execute that proof.
         * @throws std::logic_error when an MTP ExpertOverlay case names no
         *         continuation domain in its immutable placement plan.
         */
        [[nodiscard]] bool requiresCanonicalTPAllreduceMTPDiagnostics() const
        {
            if (!mtpEnabled() || !expert_overlay ||
                !topology.expert_overlay_plan)
            {
                return false;
            }

            const auto &plan = *topology.expert_overlay_plan;
            const auto continuation = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &domain)
                { return domain.name == plan.continuation_domain; });
            if (continuation == plan.domains.end())
            {
                throw std::logic_error(
                    "MTP ExpertOverlay parity topology has no continuation domain: " +
                    topology.test_id);
            }
            if (continuation->participants.size() <= 2u)
                return false;

            const GlobalDeviceAddress &first =
                continuation->participants.front();
            if (!first.isGPU())
                return false;
            return std::all_of(
                continuation->participants.begin(),
                continuation->participants.end(),
                [&](const GlobalDeviceAddress &participant)
                {
                    return (first.isCUDA() && participant.isCUDA()) ||
                           (first.isROCm() && participant.isROCm());
                });
        }

        /** @return Whether this cell uses the adaptive device depth controller. */
        [[nodiscard]] constexpr bool usesDynamicMTPDepth() const noexcept
        {
            return mtp == ModelParityMTP::DynamicDepth;
        }

        /** @return Exact selected MTP depth, or the dynamic policy ceiling. */
        [[nodiscard]] constexpr int requestedMTPDraftDepth() const noexcept
        {
            switch (mtp)
            {
            case ModelParityMTP::Off:
                return 0;
            case ModelParityMTP::Depth1:
                return 1;
            case ModelParityMTP::Depth2:
                return 2;
            case ModelParityMTP::Depth3:
                return 3;
            case ModelParityMTP::Depth15:
            case ModelParityMTP::DynamicDepth:
                return kModelParityRequiredMaximumMTPDepth;
            }
            return 0;
        }

        /** @return Required MTP path/checkpoint evidence for this case. */
        [[nodiscard]] constexpr ModelParityMTPEvidence mtpEvidence() const noexcept
        {
            if (!mtpEnabled())
                return ModelParityMTPEvidence::Disabled;
            return usesDynamicMTPDepth()
                       ? ModelParityMTPEvidence::DynamicDepth
                       : ModelParityMTPEvidence::FixedDepth;
        }

        /** @return Required request-local movement evidence for this case. */
        [[nodiscard]] constexpr ModelParityMovementEvidence
        movementEvidence() const noexcept
        {
            return expert_overlay
                       ? expert_overlay->movementEvidence()
                       : ModelParityMovementEvidence::NotApplicable;
        }

        /** @return Whether this cell must prove zero request-time movement. */
        [[nodiscard]] constexpr bool requiresNoExpertMovement() const noexcept
        {
            return movementEvidence() ==
                   ModelParityMovementEvidence::NoMovement;
        }

        /** @return Whether this cell must prove copied and applied expert bytes. */
        [[nodiscard]] constexpr bool requiresPhysicalExpertMovement() const noexcept
        {
            return movementEvidence() ==
                   ModelParityMovementEvidence::MovementRequired;
        }

        /** @return Whether this cell owns the matched convergence speed gate. */
        [[nodiscard]] constexpr bool
        requiresObservedConvergenceSpeedup() const noexcept
        {
            return dynamic_evidence ==
                   ModelParityDynamicEvidence::
                       EconomicMovementAndObservedSpeedup;
        }

        /** @return Stable GoogleTest-safe full matrix identity. */
        [[nodiscard]] std::string testName() const
        {
            const std::string prefill_name =
                prefill_graph.testNameFragment();
            return model.test_id + "_" + topology.test_id + "_" +
                   (expert_overlay
                        ? expert_overlay->testName() + "_"
                        : std::string{}) +
                   (prefill_name.empty()
                        ? std::string{}
                        : prefill_name + "_") +
                   activationName() + "_" + kvCacheName() + "_" +
                   mtpName();
        }

        /**
         * @brief Convert common fields to the existing parity fixture adapter.
         *
         * This method supports incremental migration of the current model
         * fixtures. New policy-aware bodies should retain `ModelParityCase` as
         * their GoogleTest parameter and use @ref applyRuntimePolicy for MTP
         * and the mandatory prefix lifecycle rather than extending legacy
         * `TestConfig`. Setup capacity remains distinct from the active MTP
         * evidence expected by the legacy adapter.
         */
        [[nodiscard]] TestConfig toTestConfig() const
        {
            TestConfig config;
            config.name = testName();
            config.parallelism = legacyParallelism();
            config.collective = topology.collective;
            config.thresholds = thresholds;
            config.mpi_ranks = topology.mpi_ranks;
            config.model_path = model.model_path;
            config.snapshot_dir = model.reference_directory;
            config.prompt = model.prompt;
            config.token_ids = model.token_ids;
            config.pp_stage_sizes = topology.pipeline_stage_sizes;
            config.pp_weights = topology.pipeline_weights;
            config.tp_collective = topology.tensor_parallel_collective;
            config.activation_precision = activation_precision;
            config.kv_cache_precision = kv_cache_precision;
            config.decode_steps = model.decode_steps;
            config.moe_hot_expert_cache = moe_hot_expert_cache;
            config.moe_routed_prefill = moe_routed_prefill;
            config.moe_rebalance = dynamic_rebalance;
            config.moe_rebalance.mode =
                movementEvidence() ==
                        ModelParityMovementEvidence::MovementRequired
                    ? MoERebalanceRuntimeMode::Dynamic
                    : MoERebalanceRuntimeMode::Off;
            config.graph_snapshot_policy = graph_snapshot_policy;
            config.collective_evidence_source = collective_evidence_source;
            config.tp_allreduce_precision_override =
                tp_allreduce_precision_override;

            for (const auto &participant : topology.participants)
            {
                if (participant.address.isCUDA())
                    config.devices.push_back(ParityDeviceType::CUDA);
                else if (participant.address.isROCm())
                    config.devices.push_back(ParityDeviceType::ROCm);
                else
                    config.devices.push_back(ParityDeviceType::CPU);
            }

            if (expert_overlay)
            {
                config.routed_expert_owner_order =
                    expert_overlay->owner_order;
                config.moe_routed_expert_plan =
                    std::make_shared<MoERoutedExpertPlacementPlan>(
                        *topology.expert_overlay_plan);
                config.moe_routed_expert_plan->owner_order =
                    expert_overlay->owner_order;
                config.moe_routed_expert_plan->residency_policy =
                    expert_overlay->movement ==
                            ModelParityExpertMovement::Dynamic
                        ? RoutedExpertResidencyPolicy::RoutedTierRebalanced
                        : RoutedExpertResidencyPolicy::StaticById;
                config.moe_movement_expectation =
                    requiresPhysicalExpertMovement()
                        ? ParityMoEMovementExpectation::PhysicalMovement
                        : ParityMoEMovementExpectation::NoMovement;
                if (requiresPhysicalExpertMovement())
                {
                    /*
                     * Wake the production authority at ordinary request
                     * boundaries. The physical movement assertion consumes
                     * completed PerfStats at the result boundary; it does not
                     * force a stream/device synchronization into inference.
                     */
                    config.moe_rebalance_exercise = {
                        .enabled = true,
                        .require_production_overlay_authority = true,
                        .request_after_prefill = true,
                        .request_every_decode_steps = 1,
                        .min_decode_steps = std::min(2, model.decode_steps),
                        .require_movement_epoch_advance = false,
                        .min_movement_epoch_delta = 1,
                    };
                }
            }
            config.mtp_expectation =
                !mtpEnabled()
                    ? ParityMTPExpectation::Disabled
                    : usesDynamicMTPDepth()
                          ? ParityMTPExpectation::DynamicDepth
                          : ParityMTPExpectation::FixedDepth;
            config.mtp_expected_draft_depth = requestedMTPDraftDepth();
            config.mtp_expected_graph_capacity =
                retained_mtp_draft_capacity;
            config.mtp_recursive_aggregate_cosine_floor =
                mtp_recursive_aggregate_cosine_floor;
            return config;
        }

        /**
         * @brief Install generated precision, mandatory prefix, MTP, and overlay policy.
         *
         * Call this after model/topology-specific defaults have been installed.
         * Economic Dynamic scalars are retained; only the matrix-owned mode is
         * replaced. Every cell in an MTP-capable definition, including the
         * execution-off control, shares the model's depth-15 setup envelope
         * while selecting its independent logical depth.
         *
         * @param config Production configuration used by the live runner.
         */
        void applyRuntimePolicy(OrchestrationConfig &config) const
        {
            config.activation_precision = activationConfigValue();
            config.kv_cache_precision = kvCacheConfigValue();

            // Prefix restore is a production invariant, not a generated axis.
            // The parity fixture resolves authenticated-prompt block geometry
            // after tokenization; this typed projection only enables the real
            // bounded production tiers and terminal-state payload.
            config.prefix_cache.enabled = true;
            config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
            config.prefix_cache.terminal_state =
                PrefixCacheTerminalStateMode::Auto;

            config.mtp.enabled = mtpEnabled();
            config.mtp.draft_tokens =
                std::max(1, requestedMTPDraftDepth());
            config.mtp.graph_capacity_draft_tokens =
                retained_mtp_draft_capacity;
            config.mtp.verify_mode = MTPVerifyMode::Greedy;
            config.mtp.depth_policy = MTPDepthPolicyConfig{};
            if (usesDynamicMTPDepth())
            {
                config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
                config.mtp.depth_policy.min_depth = 1;
                config.mtp.depth_policy.max_depth =
                    model.maximum_mtp_draft_depth;
                config.mtp.depth_policy.initial_depth =
                    model.maximum_mtp_draft_depth;
                // A short parity request must prove controller participation.
                config.mtp.depth_policy.window_size = 1;
                config.mtp.depth_policy.min_samples = 1;
                config.mtp.depth_policy.cooldown_steps = 0;
                config.mtp.depth_policy.promote_consecutive_windows = 1;
            }

            if (expert_overlay)
            {
                config.routed_expert_owner_order =
                    expert_overlay->owner_order;
                // The generated case owns both the movement mode and its
                // economics.  Copy the complete policy before specializing
                // the mode so fixture defaults cannot alter one matrix cell.
                config.moe_rebalance = dynamic_rebalance;
                config.moe_rebalance.mode =
                    expert_overlay->movement ==
                            ModelParityExpertMovement::Dynamic
                        ? MoERebalanceRuntimeMode::Dynamic
                        : MoERebalanceRuntimeMode::Off;
                config.moe_routed_expert_plan =
                    std::make_shared<MoERoutedExpertPlacementPlan>(
                        *topology.expert_overlay_plan);
                config.moe_routed_expert_plan->owner_order =
                    expert_overlay->owner_order;
                config.moe_routed_expert_plan->residency_policy =
                    expert_overlay->movement ==
                            ModelParityExpertMovement::Dynamic
                        ? RoutedExpertResidencyPolicy::RoutedTierRebalanced
                        : RoutedExpertResidencyPolicy::StaticById;
            }
        }

        /**
         * @brief Build the complete server-facing topology and runtime config.
         *
         * This is the sole projection from a generated parity case into the
         * production orchestration surface.  It deliberately emits the same
         * `OrchestrationConfig` consumed by the HTTP/application runner instead
         * of compiling a test-owned runner tree. Pipeline participant groups
         * become named execution domains and exact inclusive layer ranges;
         * TP and cross-rank ownership become their corresponding typed config
         * fields. ExpertOverlay remains the sole authority for multi-domain
         * routed execution and therefore does not synthesize a competing flat
         * TP map.
         *
         * @param resolved_model_path GGUF path after campaign tmpfs staging.
         * @param current_rank MPI rank constructing its process-local runner.
         * @return Complete production runner configuration for this case.
         * @throws std::logic_error if the already-validated case cannot be
         *         represented without inventing topology information.
         */
        [[nodiscard]] OrchestrationConfig makeOrchestrationConfig(
            const std::string &resolved_model_path,
            int current_rank) const
        {
            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = resolved_model_path;
            config.max_seq_len = model.max_seq_len;
            config.batch_size = 1;
            config.device_mode = DeviceAssignmentMode::AUTO;
            config.tp_degree = 1;
            config.pp_degree = 1;
            config.default_backend = toCollectiveBackend(topology.collective);
            config.tp_allreduce_precision_override =
                tp_allreduce_precision_override;
            config.moe_hot_expert_cache = moe_hot_expert_cache;
            config.moe_routed_prefill = moe_routed_prefill;

            const auto participantsForRank = [&](int rank)
            {
                std::vector<GlobalDeviceAddress> addresses;
                for (const auto &participant : topology.participants)
                {
                    if (participant.world_rank &&
                        *participant.world_rank == rank)
                    {
                        addresses.push_back(participant.address);
                    }
                }
                return addresses;
            };

            const auto installExplicitRankDevices = [&]
            {
                config.device_mode = DeviceAssignmentMode::EXPLICIT;
                std::vector<bool> installed(
                    static_cast<std::size_t>(topology.mpi_ranks), false);
                for (const auto &participant : topology.participants)
                {
                    if (!participant.world_rank)
                    {
                        throw std::logic_error(
                            "ordinary cross-rank parity topology cannot defer primary-device ownership to inventory binding");
                    }
                    const int rank = *participant.world_rank;
                    if (installed.at(static_cast<std::size_t>(rank)))
                    {
                        throw std::logic_error(
                            "flat TP/PP parity topology requires exactly one primary participant per MPI rank");
                    }
                    installed[static_cast<std::size_t>(rank)] = true;
                    config.device_map.emplace_back(rank, participant.address);
                    config.device_map_numa_explicit.emplace_back(
                        rank,
                        participant.address.hasValidNuma());
                }
            };

            const auto stageSizes = [&]
            {
                if (!topology.pipeline_stage_sizes.empty())
                    return topology.pipeline_stage_sizes;
                return std::vector<int>(topology.participants.size(), 1);
            };

            const auto stageLayerCounts = [&](std::size_t stage_count)
            {
                if (model.transformer_layers <
                    static_cast<int>(stage_count))
                {
                    throw std::logic_error(
                        "pipeline parity requires an authenticated transformer layer count at least as large as its stage count");
                }
                std::vector<int> counts(stage_count, 0);
                if (topology.pipeline_weights.empty())
                {
                    const int base = model.transformer_layers /
                                     static_cast<int>(stage_count);
                    const int extra = model.transformer_layers %
                                      static_cast<int>(stage_count);
                    for (std::size_t stage = 0; stage < stage_count; ++stage)
                    {
                        counts[stage] = base +
                                        (static_cast<int>(stage) < extra ? 1 : 0);
                    }
                    return counts;
                }

                const float total_weight = std::accumulate(
                    topology.pipeline_weights.begin(),
                    topology.pipeline_weights.end(), 0.0f);
                if (!(total_weight > 0.0f))
                    throw std::logic_error("pipeline parity weights must sum to a positive value");
                int assigned = 0;
                for (std::size_t stage = 0; stage < stage_count; ++stage)
                {
                    if (stage + 1u == stage_count)
                    {
                        counts[stage] = model.transformer_layers - assigned;
                        break;
                    }
                    const float fraction =
                        topology.pipeline_weights[stage] / total_weight;
                    int count = std::max(
                        1,
                        static_cast<int>(
                            fraction * model.transformer_layers + 0.5f));
                    const int remaining_stages =
                        static_cast<int>(stage_count - stage - 1u);
                    count = std::min(
                        count,
                        model.transformer_layers - assigned - remaining_stages);
                    counts[stage] = count;
                    assigned += count;
                }
                return counts;
            };

            const auto installPipeline = [&](bool cross_rank)
            {
                const auto sizes = stageSizes();
                const auto layer_counts = stageLayerCounts(sizes.size());
                config.pp_degree = static_cast<int>(sizes.size());
                config.pp_split = PPSplitMode::MANUAL;

                std::size_t participant_offset = 0u;
                int layer_offset = 0;
                for (std::size_t stage = 0; stage < sizes.size(); ++stage)
                {
                    DomainDefinition domain;
                    domain.name = "parity_stage_" + std::to_string(stage);
                    const int stage_size = sizes[stage];
                    std::vector<int> rank_map;
                    for (int index = 0; index < stage_size; ++index)
                    {
                        const auto &participant = topology.participants.at(
                            participant_offset++);
                        if (!participant.world_rank)
                        {
                            throw std::logic_error(
                                "pipeline parity participant requires explicit MPI ownership");
                        }
                        domain.devices.push_back(participant.address);
                        rank_map.push_back(*participant.world_rank);
                    }

                    std::vector<int> unique_ranks = rank_map;
                    std::sort(unique_ranks.begin(), unique_ranks.end());
                    unique_ranks.erase(
                        std::unique(unique_ranks.begin(), unique_ranks.end()),
                        unique_ranks.end());
                    if (unique_ranks.size() == 1u)
                    {
                        domain.scope = TPScope::RANK_LOCAL;
                        domain.owner_rank = unique_ranks.front();
                    }
                    else
                    {
                        if (!cross_rank)
                        {
                            throw std::logic_error(
                                "rank-local pipeline stage cannot span MPI ranks");
                        }
                        domain.scope = TPScope::NODE_LOCAL;
                        domain.explicit_ranks = std::move(rank_map);
                    }
                    if (domain.devices.size() > 1u)
                    {
                        domain.backend = toCollectiveBackend(
                            topology.tensor_parallel_collective);
                        domain.weights.assign(
                            domain.devices.size(),
                            1.0f / static_cast<float>(domain.devices.size()));
                    }
                    config.domain_definitions.push_back(std::move(domain));
                    config.pp_stage_definitions.push_back(PPStageDefinition{
                        .stage_id = static_cast<int>(stage),
                        .domain_name = "parity_stage_" + std::to_string(stage),
                        .first_layer = layer_offset,
                        .last_layer = layer_offset + layer_counts[stage] - 1,
                    });
                    layer_offset += layer_counts[stage];
                }
            };

            /*
             * ExpertOverlay is the sole topology authority whenever a plan is
             * present. The kind still describes the physical orchestration
             * shape to validation and legacy fixture hooks, but it must not
             * synthesize a competing flat TP/PP map beside the named domains.
             */
            if (topology.isExpertOverlay())
            {
                applyRuntimePolicy(config);
                return config;
            }

            switch (topology.kind)
            {
            case ModelParityTopologyKind::SingleDevice:
                config.device_for_this_rank = topology.participants.front().address;
                break;
            case ModelParityTopologyKind::RankLocalTensorParallel:
                config.tp_scope = TPScope::RANK_LOCAL;
                config.tp_devices = participantsForRank(current_rank);
                config.tp_degree = static_cast<int>(config.tp_devices.size());
                break;
            case ModelParityTopologyKind::RankLocalPipelineParallel:
                installPipeline(false);
                break;
            case ModelParityTopologyKind::NodeTensorParallel:
                installExplicitRankDevices();
                config.tp_scope = TPScope::NODE_LOCAL;
                config.tp_degree = topology.mpi_ranks;
                break;
            case ModelParityTopologyKind::GlobalTensorParallel:
                installExplicitRankDevices();
                config.tp_scope = TPScope::GLOBAL;
                config.tp_degree = topology.mpi_ranks;
                break;
            case ModelParityTopologyKind::NodePipelineParallel:
                installPipeline(true);
                break;
            case ModelParityTopologyKind::RankLocalMultiDomain:
            case ModelParityTopologyKind::NodeMultiDomain:
            case ModelParityTopologyKind::GlobalMultiDomain:
                if (!topology.isExpertOverlay())
                {
                    throw std::logic_error(
                        "multi-domain parity currently requires one ExpertOverlay placement authority");
                }
                break;
            }

            applyRuntimePolicy(config);
            return config;
        }

    private:
        /** @return Legacy fixture parallelism equivalent to the typed topology. */
        [[nodiscard]] Parallelism legacyParallelism() const
        {
            switch (topology.kind)
            {
            case ModelParityTopologyKind::SingleDevice:
                return Parallelism::None;
            case ModelParityTopologyKind::RankLocalTensorParallel:
                return Parallelism::LocalTP;
            case ModelParityTopologyKind::RankLocalPipelineParallel:
                return Parallelism::LocalPP;
            case ModelParityTopologyKind::NodeTensorParallel:
                return Parallelism::NodeTP;
            case ModelParityTopologyKind::NodePipelineParallel:
                return Parallelism::NodeLocalPP;
            case ModelParityTopologyKind::GlobalTensorParallel:
                return Parallelism::GlobalTP;
            case ModelParityTopologyKind::RankLocalMultiDomain:
            case ModelParityTopologyKind::NodeMultiDomain:
            case ModelParityTopologyKind::GlobalMultiDomain:
                // The production runner consumes the named-domain/overlay
                // declaration directly. There is no equivalent legacy flat
                // TP/PP mode, so the incremental adapter must not invent one.
                return Parallelism::None;
            }
            throw std::logic_error("unknown model parity topology kind");
        }

        /** @return Stable activation-precision name fragment. */
        [[nodiscard]] std::string activationName() const
        {
            return "Act" + std::string(
                               activationPrecisionToString(
                                   activation_precision));
        }

        /** @return Stable KV-cache precision name fragment. */
        [[nodiscard]] std::string kvCacheName() const
        {
            switch (kv_cache_precision)
            {
            case KVCachePrecision::AUTO:
                return "KVAUTO";
            case KVCachePrecision::FP32:
                return "KVFP32";
            case KVCachePrecision::FP16:
                return "KVFP16";
            case KVCachePrecision::Q8_1:
                return "KVQ8_1";
            case KVCachePrecision::Q16_1:
                return "KVQ16_1";
            case KVCachePrecision::TQ4:
                return "KVTQ4";
            case KVCachePrecision::TQ:
                return "KVTQ";
            }
            return "KVUnknown";
        }

        /** @return Stable MTP-policy name fragment. */
        [[nodiscard]] std::string mtpName() const
        {
            switch (mtp)
            {
            case ModelParityMTP::Off:
                return "MTPOff";
            case ModelParityMTP::Depth1:
                return "MTPDepth1";
            case ModelParityMTP::Depth2:
                return "MTPDepth2";
            case ModelParityMTP::Depth3:
                return "MTPDepth3";
            case ModelParityMTP::Depth15:
                return "MTPDepth15";
            case ModelParityMTP::DynamicDepth:
                return "MTPDynamicDepth";
            }
            return "MTPUnknown";
        }

        /** @return Production activation precision spelling. */
        [[nodiscard]] std::string activationConfigValue() const
        {
            std::string value = activationPrecisionToString(
                activation_precision);
            std::transform(
                value.begin(), value.end(), value.begin(),
                [](unsigned char character)
                { return static_cast<char>(std::tolower(character)); });
            return value;
        }

        /** @return Production KV-cache precision spelling. */
        [[nodiscard]] std::string kvCacheConfigValue() const
        {
            switch (kv_cache_precision)
            {
            case KVCachePrecision::AUTO:
                return "auto";
            case KVCachePrecision::FP32:
                return "fp32";
            case KVCachePrecision::FP16:
                return "fp16";
            case KVCachePrecision::Q8_1:
                return "q8_1";
            case KVCachePrecision::Q16_1:
                return "q16_1";
            case KVCachePrecision::TQ4:
                return "tq4";
            case KVCachePrecision::TQ:
                return "tq";
            }
            throw std::logic_error("unknown KV-cache precision");
        }
    };

    /**
     * @brief Reusable GoogleTest parameter adapter for config-driven fixtures.
     *
     * The generated case remains the GoogleTest parameter and policy authority.
     * `TestConfig` is materialized only as a compatibility view for the current
     * CRTP parity bases; new fixtures should read lifecycle axes directly from
     * @ref modelParityCase instead of adding fields to that legacy adapter.
     */
    class ModelParityCaseParameter
        : public ::testing::WithParamInterface<ModelParityCase>
    {
    public:
        /** @return The complete typed case owned by this fixture instance. */
        [[nodiscard]] const ModelParityCase &modelParityCase() const
        {
            return GetParam();
        }

        /** @return Stable compatibility configuration for the current case. */
        [[nodiscard]] const TestConfig &getTestConfig() const
        {
            const std::string case_name = GetParam().testName();
            if (!test_config_ || test_config_->name != case_name)
                test_config_ = GetParam().toTestConfig();
            return *test_config_;
        }

    private:
        mutable std::optional<TestConfig> test_config_;
    };

    /** Every MTP policy required by the canonical capable-model profile. */
    inline constexpr std::array<ModelParityMTP, 6>
        kCanonicalModelParityMTPPolicies = {
            ModelParityMTP::Off,
            ModelParityMTP::Depth1,
            ModelParityMTP::Depth2,
            ModelParityMTP::Depth3,
            ModelParityMTP::Depth15,
            ModelParityMTP::DynamicDepth,
        };

    /** ExpertOverlay owner/movement product required for every overlay topology. */
    inline constexpr std::array<ModelParityExpertOverlayPolicy, 4>
        kCanonicalModelParityExpertOverlayPolicies = {
            ModelParityExpertOverlayPolicy{
                .owner_order = RoutedExpertOwnerOrder::Ordinal,
                .movement = ModelParityExpertMovement::Static,
            },
            ModelParityExpertOverlayPolicy{
                .owner_order = RoutedExpertOwnerOrder::Ordinal,
                .movement = ModelParityExpertMovement::Dynamic,
            },
            ModelParityExpertOverlayPolicy{
                .owner_order = RoutedExpertOwnerOrder::Random,
                .movement = ModelParityExpertMovement::Static,
            },
            ModelParityExpertOverlayPolicy{
                .owner_order = RoutedExpertOwnerOrder::Random,
                .movement = ModelParityExpertMovement::Dynamic,
            },
        };

    /**
     * @brief Validate and expand one definition over every enabled axis.
     *
     * Non-applicable axes are represented by their one explicit disabled
     * value. ExpertOverlay is inferred from the topology blueprint and always
     * expands its four placement/movement points; it cannot be selectively
     * weakened by a model file. MTP is an all-or-nothing standard profile;
     * prefix restore is mandatory within every emitted case and never
     * multiplies the matrix. Precision axes remain explicit because support is
     * a model/kernel fact, but their cross product is generated centrally.
     *
     * @param definition Real model, typed topology, thresholds, and profiles.
     * @return Unique ordered canonical cases.
     * @throws std::invalid_argument for an incomplete or contradictory spec.
     */
    [[nodiscard]] inline std::vector<ModelParityCase>
    expandModelParityDefinition(const ModelParityDefinition &definition)
    {
        const auto &model = definition.model;
        const auto &topology = definition.topology;
        const auto is_test_identifier = [](const std::string &value)
        {
            return !value.empty() &&
                   std::all_of(
                       value.begin(), value.end(),
                       [](unsigned char character)
                       {
                           return std::isalnum(character) != 0 ||
                                  character == '_';
                       });
        };
        if (model.test_id.empty() || model.model_path.empty() ||
            model.reference_directory.empty())
        {
            throw std::invalid_argument(
                "model parity requires non-empty model id, GGUF path, and reference directory");
        }
        if (!is_test_identifier(model.test_id) ||
            !is_test_identifier(topology.test_id))
        {
            throw std::invalid_argument(
                "model parity model/topology ids must contain only letters, digits, and underscores");
        }
        if (topology.test_id.empty() || topology.participants.empty() ||
            topology.mpi_ranks <= 0)
        {
            throw std::invalid_argument(
                "model parity requires a named topology, participants, and positive MPI size");
        }
        if (definition.precisions.activation.empty() ||
            definition.precisions.kv_cache.empty() ||
            definition.features.prefill_graph.empty())
        {
            throw std::invalid_argument(
                "model parity precision and prefill graph axes must not be empty");
        }
        for (const auto activation : definition.precisions.activation)
            requireImplementedActivationPrecision(
                activationPrecisionToString(activation));
        if (std::set<ActivationPrecision>(
                definition.precisions.activation.begin(),
                definition.precisions.activation.end())
                    .size() != definition.precisions.activation.size() ||
            std::set<KVCachePrecision>(
                definition.precisions.kv_cache.begin(),
                definition.precisions.kv_cache.end())
                    .size() != definition.precisions.kv_cache.size())
        {
            throw std::invalid_argument(
                "model parity precision axes must contain unique values");
        }
        for (std::size_t index = 0;
             index < definition.precisions.threshold_overrides.size();
             ++index)
        {
            const auto &override =
                definition.precisions.threshold_overrides[index];
            if (std::find(
                    definition.precisions.activation.begin(),
                    definition.precisions.activation.end(),
                    override.activation) ==
                    definition.precisions.activation.end() ||
                std::find(
                    definition.precisions.kv_cache.begin(),
                    definition.precisions.kv_cache.end(),
                    override.kv_cache) ==
                    definition.precisions.kv_cache.end())
            {
                throw std::invalid_argument(
                    "model parity precision threshold override names a pair outside the declared axes");
            }
            const auto duplicate = std::find_if(
                definition.precisions.threshold_overrides.begin(),
                definition.precisions.threshold_overrides.begin() +
                    static_cast<std::ptrdiff_t>(index),
                [&](const ModelParityPrecisionThresholdOverride &candidate)
                {
                    return candidate.activation == override.activation &&
                           candidate.kv_cache == override.kv_cache;
                });
            if (duplicate !=
                definition.precisions.threshold_overrides.begin() +
                    static_cast<std::ptrdiff_t>(index))
            {
                throw std::invalid_argument(
                    "model parity precision threshold overrides must be unique by activation/KV pair");
            }
        }
        for (std::size_t index = 0;
             index < definition.features.prefill_graph.size(); ++index)
        {
            const auto &profile =
                definition.features.prefill_graph[index];
            const bool valid_standard =
                profile.mode == ModelParityPrefillGraphMode::Standard &&
                profile.captured_rows == 0;
            const bool valid_segmented =
                profile.mode ==
                    ModelParityPrefillGraphMode::SegmentedCaptured &&
                profile.captured_rows > 0;
            if (!valid_standard && !valid_segmented)
            {
                throw std::invalid_argument(
                    "model parity prefill graph profile requires zero rows for Standard or positive rows for SegmentedCaptured");
            }
            const auto duplicate = std::find(
                definition.features.prefill_graph.begin(),
                definition.features.prefill_graph.begin() +
                    static_cast<std::ptrdiff_t>(index),
                profile);
            if (duplicate !=
                definition.features.prefill_graph.begin() +
                    static_cast<std::ptrdiff_t>(index))
            {
                throw std::invalid_argument(
                    "model parity prefill graph profiles must be unique");
            }
        }
        if (definition.features.mtp == ModelParityAxisProfile::Standard &&
            model.maximum_mtp_draft_depth <
                kModelParityRequiredMaximumMTPDepth)
        {
            throw std::invalid_argument(
                "standard model parity MTP profile requires depth-15 model/reference capacity");
        }
        if (!definition.features.mtp_kl_threshold_overrides.empty() &&
            definition.features.mtp != ModelParityAxisProfile::Standard)
        {
            throw std::invalid_argument(
                "model parity MTP KL overrides require the standard MTP axis");
        }
        for (std::size_t index = 0;
             index < definition.features.mtp_kl_threshold_overrides.size();
             ++index)
        {
            const auto &override =
                definition.features.mtp_kl_threshold_overrides[index];
            if (override.policy == ModelParityMTP::Off ||
                !std::isfinite(override.maximum_kl_divergence) ||
                !(override.maximum_kl_divergence > 0.0f))
            {
                throw std::invalid_argument(
                    "model parity MTP KL override requires an enabled policy and finite positive budget");
            }
            const auto duplicate = std::find_if(
                definition.features.mtp_kl_threshold_overrides.begin(),
                definition.features.mtp_kl_threshold_overrides.begin() +
                    static_cast<std::ptrdiff_t>(index),
                [&](const ModelParityMTPKLThresholdOverride &candidate)
                { return candidate.policy == override.policy; });
            if (duplicate !=
                definition.features.mtp_kl_threshold_overrides.begin() +
                    static_cast<std::ptrdiff_t>(index))
            {
                throw std::invalid_argument(
                    "model parity MTP KL overrides must be unique by policy");
            }
        }
        if (!definition.features
                 .mtp_recursive_aggregate_cosine_threshold_overrides.empty() &&
            definition.features.mtp != ModelParityAxisProfile::Standard)
        {
            throw std::invalid_argument(
                "model parity recursive MTP cosine overrides require the standard MTP axis");
        }
        for (std::size_t index = 0;
             index < definition.features
                         .mtp_recursive_aggregate_cosine_threshold_overrides
                         .size();
             ++index)
        {
            const auto &override = definition.features
                                       .mtp_recursive_aggregate_cosine_threshold_overrides[index];
            if ((override.policy != ModelParityMTP::Depth15 &&
                 override.policy != ModelParityMTP::DynamicDepth) ||
                !std::isfinite(override.minimum_cosine_similarity) ||
                !(override.minimum_cosine_similarity > 0.0f) ||
                override.minimum_cosine_similarity > 1.0f)
            {
                throw std::invalid_argument(
                    "model parity recursive MTP cosine override requires depth-15/flexible-depth policy and a finite floor in (0, 1]");
            }
            const auto duplicate = std::find_if(
                definition.features
                    .mtp_recursive_aggregate_cosine_threshold_overrides.begin(),
                definition.features
                        .mtp_recursive_aggregate_cosine_threshold_overrides.begin() +
                    static_cast<std::ptrdiff_t>(index),
                [&](const ModelParityMTPAggregateCosineThresholdOverride &candidate)
                { return candidate.policy == override.policy; });
            if (duplicate !=
                definition.features
                        .mtp_recursive_aggregate_cosine_threshold_overrides.begin() +
                    static_cast<std::ptrdiff_t>(index))
            {
                throw std::invalid_argument(
                    "model parity recursive MTP cosine overrides must be unique by policy");
            }
        }
        if (topology.isExpertOverlay() &&
            !topology.expert_overlay_plan->enabled)
        {
            throw std::invalid_argument(
                "ExpertOverlay parity topology supplied a disabled placement plan");
        }
        if (topology.isExpertOverlay() && model.transformer_layers <= 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay parity requires an authenticated transformer layer count for pre-load graph identity");
        }
        if (!topology.isExpertOverlay() &&
            definition.features.dynamic_speedup_witness !=
                ModelParityDynamicSpeedupWitness::Disabled)
        {
            throw std::invalid_argument(
                "Dynamic speedup witness requires an ExpertOverlay topology");
        }
        if (topology.isExpertOverlay())
        {
            const auto validate_dynamic_policy =
                [&](const MoERebalanceRuntimeConfig &policy,
                    const char *evidence_name)
            {
                if (policy.migration_transfer_slots == 0u ||
                    policy.resolvedMigrationExecutionStreams() == 0u ||
                    policy.resolvedMigrationExecutionStreams() >
                        policy.migration_transfer_slots ||
                    policy.resolvedMigrationCyclesPerWave() == 0u ||
                    policy.resolvedMigrationCyclesPerWave() >
                        policy.migration_transfer_slots)
                {
                    throw std::invalid_argument(
                        std::string("ExpertOverlay ") + evidence_name +
                        " policy requires positive stream/cycle geometry within its retained migration slots");
                }
                if (policy.device_maintenance_slack_tokens < 0 ||
                    policy.device_min_maintenance_period_tokens <= 0 ||
                    policy.device_initial_maintenance_period_tokens <= 0)
                {
                    /*
                     * Dynamic cells promise physical movement, so their
                     * device-owned decision clock is part of typed identity.
                     * An environment-owned cadence would make one registered
                     * case execute a different lifecycle on another host.
                     */
                    throw std::invalid_argument(
                        std::string("ExpertOverlay ") + evidence_name +
                        " policy requires an explicit non-negative maintenance slack and positive initial/recurring device cadence");
                }
                if (definition.features.mtp ==
                            ModelParityAxisProfile::Standard &&
                    policy.device_min_maintenance_period_tokens <
                        model.maximum_mtp_draft_depth + 1)
                {
                    /*
                     * The initial cadence may force an early publication.
                     * Recurring maintenance must still admit one complete
                     * maximum-width predictor-plus-verifier transaction.
                     */
                    throw std::invalid_argument(
                        std::string("ExpertOverlay standard MTP ") +
                        evidence_name +
                        " policy requires a recurring maintenance cadence at least as wide as the maximum predictor-plus-verifier transaction");
                }
            };
            validate_dynamic_policy(
                definition.dynamic_rebalance.economic_movement,
                "economic-movement");
            validate_dynamic_policy(
                definition.dynamic_rebalance
                    .economic_movement_and_observed_speedup,
                "observed-speedup");
        }
        if (topology.kind == ModelParityTopologyKind::SingleDevice &&
            topology.participants.size() != 1u)
        {
            throw std::invalid_argument(
                "single-device parity topology must declare exactly one participant");
        }
        const bool cross_rank =
            topology.kind == ModelParityTopologyKind::NodeTensorParallel ||
            topology.kind == ModelParityTopologyKind::NodePipelineParallel ||
            topology.kind == ModelParityTopologyKind::GlobalTensorParallel ||
            topology.kind == ModelParityTopologyKind::NodeMultiDomain ||
            topology.kind == ModelParityTopologyKind::GlobalMultiDomain;

        const auto require_uniform_cross_rank_head_shards =
            [&](int degree, const std::string &identity)
        {
            if (degree <= 1)
                return;
            if (model.attention_heads <= 0 || model.kv_heads <= 0)
            {
                throw std::invalid_argument(
                    identity +
                    " requires authenticated query/KV head counts");
            }
            if (model.attention_heads % degree != 0 ||
                model.kv_heads % degree != 0)
            {
                throw std::invalid_argument(
                    identity + " degree " + std::to_string(degree) +
                    " cannot uniformly shard model heads (query=" +
                    std::to_string(model.attention_heads) + ", KV=" +
                    std::to_string(model.kv_heads) + ")");
            }
        };

        if (topology.kind == ModelParityTopologyKind::NodeTensorParallel ||
            topology.kind == ModelParityTopologyKind::GlobalTensorParallel)
        {
            require_uniform_cross_rank_head_shards(
                topology.mpi_ranks, "cross-rank TP parity topology");
        }
        else if (topology.kind ==
                 ModelParityTopologyKind::NodePipelineParallel)
        {
            const auto sizes = topology.pipeline_stage_sizes.empty()
                                   ? std::vector<int>(
                                         topology.participants.size(), 1)
                                   : topology.pipeline_stage_sizes;
            std::size_t participant_offset = 0;
            for (std::size_t stage = 0; stage < sizes.size(); ++stage)
            {
                std::set<int> ranks;
                for (int index = 0; index < sizes[stage]; ++index)
                {
                    const auto &participant = topology.participants.at(
                        participant_offset++);
                    if (participant.world_rank)
                        ranks.insert(*participant.world_rank);
                }
                require_uniform_cross_rank_head_shards(
                    static_cast<int>(ranks.size()),
                    "cross-rank pipeline stage " + std::to_string(stage));
            }
        }
        if (cross_rank && topology.mpi_ranks < 2)
        {
            throw std::invalid_argument(
                "cross-rank parity topology requires at least two MPI ranks");
        }
        if (!cross_rank && topology.mpi_ranks != 1)
        {
            throw std::invalid_argument(
                "rank-local parity topology requires exactly one MPI rank");
        }

        std::vector<bool> rank_has_participant(
            static_cast<std::size_t>(topology.mpi_ranks), false);
        bool has_inventory_bound_participant = false;
        for (std::size_t participant_index = 0;
             participant_index < topology.participants.size();
             ++participant_index)
        {
            const auto &participant =
                topology.participants[participant_index];
            const auto duplicate = std::find_if(
                topology.participants.begin(),
                topology.participants.begin() +
                    static_cast<std::ptrdiff_t>(participant_index),
                [&](const ModelParityParticipant &candidate)
                { return candidate.address == participant.address; });
            if (duplicate != topology.participants.begin() +
                                 static_cast<std::ptrdiff_t>(participant_index))
            {
                throw std::invalid_argument(
                    "model parity topology cannot name one physical participant twice");
            }
            if (!participant.world_rank)
            {
                if (!cross_rank)
                {
                    throw std::invalid_argument(
                        "rank-local parity participant cannot defer MPI ownership to inventory binding");
                }
                has_inventory_bound_participant = true;
                continue;
            }
            if (*participant.world_rank < 0 ||
                *participant.world_rank >= topology.mpi_ranks)
            {
                throw std::invalid_argument(
                    "model parity participant names an out-of-range MPI rank");
            }
            rank_has_participant[static_cast<std::size_t>(
                *participant.world_rank)] = true;
        }
        if (cross_rank && !has_inventory_bound_participant &&
            std::find(
                rank_has_participant.begin(),
                rank_has_participant.end(), false) !=
                rank_has_participant.end())
        {
            throw std::invalid_argument(
                "cross-rank parity topology must assign a participant to every MPI rank");
        }

        const bool pipeline =
            topology.kind ==
                ModelParityTopologyKind::RankLocalPipelineParallel ||
            topology.kind == ModelParityTopologyKind::NodePipelineParallel;
        const std::size_t pipeline_stage_count =
            topology.pipeline_stage_sizes.empty()
                ? topology.participants.size()
                : topology.pipeline_stage_sizes.size();
        if (pipeline &&
            model.transformer_layers <
                static_cast<int>(pipeline_stage_count))
        {
            throw std::invalid_argument(
                "pipeline parity requires an authenticated transformer layer count at least as large as its stage count");
        }
        if (!topology.pipeline_stage_sizes.empty())
        {
            if (!pipeline ||
                std::any_of(
                    topology.pipeline_stage_sizes.begin(),
                    topology.pipeline_stage_sizes.end(),
                    [](int size) { return size <= 0; }))
            {
                throw std::invalid_argument(
                    "pipeline stage sizes require a pipeline topology and positive groups");
            }
            const auto grouped_participants = std::accumulate(
                topology.pipeline_stage_sizes.begin(),
                topology.pipeline_stage_sizes.end(), std::size_t{0},
                [](std::size_t total, int size)
                { return total + static_cast<std::size_t>(size); });
            if (grouped_participants != topology.participants.size())
            {
                throw std::invalid_argument(
                    "pipeline stage sizes must cover every topology participant exactly once");
            }
        }
        if (!topology.pipeline_weights.empty() &&
            topology.pipeline_weights.size() !=
                (topology.pipeline_stage_sizes.empty()
                     ? topology.participants.size()
                     : topology.pipeline_stage_sizes.size()))
        {
            throw std::invalid_argument(
                "pipeline weights must match the explicit or implicit stage count");
        }
        if (std::any_of(
                topology.pipeline_weights.begin(),
                topology.pipeline_weights.end(),
                [](float weight)
                {
                    return !std::isfinite(weight) || !(weight > 0.0f);
                }))
        {
            throw std::invalid_argument(
                "pipeline weights must be finite positive values");
        }

        const std::vector<ModelParityMTP> mtp_policies =
            definition.features.mtp == ModelParityAxisProfile::Standard
                ? std::vector<ModelParityMTP>(
                      kCanonicalModelParityMTPPolicies.begin(),
                      kCanonicalModelParityMTPPolicies.end())
                : std::vector<ModelParityMTP>{ModelParityMTP::Off};
        const std::vector<std::optional<ModelParityExpertOverlayPolicy>>
            overlay_policies = topology.isExpertOverlay()
                                   ? std::vector<std::optional<
                                         ModelParityExpertOverlayPolicy>>(
                                         kCanonicalModelParityExpertOverlayPolicies.begin(),
                                         kCanonicalModelParityExpertOverlayPolicies.end())
                                   : std::vector<std::optional<
                                         ModelParityExpertOverlayPolicy>>{
                                         std::nullopt};
        std::vector<ModelParityCase> cases;
        cases.reserve(
            definition.precisions.activation.size() *
            definition.precisions.kv_cache.size() * mtp_policies.size() *
            overlay_policies.size() *
            definition.features.prefill_graph.size());
        std::set<std::string> names;
        for (const auto activation : definition.precisions.activation)
        {
            for (const auto kv_cache : definition.precisions.kv_cache)
            {
                for (const auto &overlay : overlay_policies)
                {
                    for (const auto &prefill_graph :
                         definition.features.prefill_graph)
                    {
                        for (const auto mtp : mtp_policies)
                        {
                            BackendThresholds thresholds =
                                definition.thresholds;
                            const auto precision_override = std::find_if(
                                definition.precisions.threshold_overrides.begin(),
                                definition.precisions.threshold_overrides.end(),
                                [&](const ModelParityPrecisionThresholdOverride &candidate)
                                {
                                    return candidate.activation == activation &&
                                           candidate.kv_cache == kv_cache;
                                });
                            if (precision_override !=
                                definition.precisions.threshold_overrides.end())
                            {
                                thresholds = precision_override->thresholds;
                            }
                            const auto mtp_kl_override = std::find_if(
                                definition.features.mtp_kl_threshold_overrides.begin(),
                                definition.features.mtp_kl_threshold_overrides.end(),
                                [&](const ModelParityMTPKLThresholdOverride &candidate)
                                { return candidate.policy == mtp; });
                            if (mtp_kl_override !=
                                definition.features.mtp_kl_threshold_overrides.end())
                            {
                                // Alter only recursive-logit KL; every other
                                // precision-specific numerical contract survives.
                                thresholds.mtp_kl_threshold =
                                    mtp_kl_override->maximum_kl_divergence;
                            }
                            const auto mtp_cosine_override = std::find_if(
                                definition.features
                                    .mtp_recursive_aggregate_cosine_threshold_overrides.begin(),
                                definition.features
                                    .mtp_recursive_aggregate_cosine_threshold_overrides.end(),
                                [&](const ModelParityMTPAggregateCosineThresholdOverride &candidate)
                                { return candidate.policy == mtp; });
                            const std::optional<float>
                                mtp_recursive_aggregate_cosine_floor =
                                    mtp_cosine_override !=
                                            definition.features
                                                .mtp_recursive_aggregate_cosine_threshold_overrides.end()
                                        ? std::optional<float>{
                                              mtp_cosine_override
                                                  ->minimum_cosine_similarity}
                                        : std::nullopt;
                            const ModelParityDynamicEvidence dynamic_evidence =
                                !overlay ||
                                        overlay->movement !=
                                            ModelParityExpertMovement::Dynamic
                                    ? ModelParityDynamicEvidence::NotApplicable
                                    : activation ==
                                              definition.precisions.activation.front() &&
                                              kv_cache ==
                                                  definition.precisions.kv_cache.front() &&
                                              selectsDynamicSpeedupWitness(
                                                  definition.features
                                                      .dynamic_speedup_witness,
                                                  overlay->owner_order) &&
                                              mtp == ModelParityMTP::Off &&
                                              prefill_graph ==
                                                  definition.features
                                                      .prefill_graph.front()
                                          ? ModelParityDynamicEvidence::
                                                EconomicMovementAndObservedSpeedup
                                          : ModelParityDynamicEvidence::
                                                EconomicMovement;
                            ModelParityCase test_case{
                                .model = model,
                                .topology = topology,
                                .thresholds = std::move(thresholds),
                                .activation_precision = activation,
                                .kv_cache_precision = kv_cache,
                                .mtp = mtp,
                                .mtp_recursive_aggregate_cosine_floor =
                                    mtp_recursive_aggregate_cosine_floor,
                                .retained_mtp_draft_capacity =
                                    definition.features.mtp ==
                                            ModelParityAxisProfile::Standard
                                        ? model.maximum_mtp_draft_depth
                                        : 0,
                                .expert_overlay = overlay,
                                .dynamic_evidence = dynamic_evidence,
                                .prefill_graph = prefill_graph,
                                .prefix_restore_geometry =
                                    ModelParityPrefixRestoreGeometry::
                                        AuthenticatedPromptBlock,
                                .moe_hot_expert_cache =
                                    definition.moe_hot_expert_cache,
                                .moe_routed_prefill =
                                    definition.moe_routed_prefill,
                                .dynamic_rebalance =
                                    definition.dynamic_rebalance.policyFor(
                                        dynamic_evidence),
                                .graph_snapshot_policy =
                                    definition.graph_snapshot_policy,
                                .collective_evidence_source =
                                    definition.collective_evidence_source,
                                .tp_allreduce_precision_override =
                                    definition.tp_allreduce_precision_override,
                            };
                            if (!names.insert(test_case.testName()).second)
                            {
                                throw std::invalid_argument(
                                    "model parity definition generated duplicate case '" +
                                    test_case.testName() + "'");
                            }
                            cases.push_back(std::move(test_case));
                        }
                    }
                }
            }
        }
        for (const auto &selection : definition.e2e_certifiable)
        {
            const auto &profile = selection.profile;
            (void)modelParityE2EThinkingModesName(profile.thinking_modes);
            if (selection.owner_order.has_value() != selection.movement.has_value() ||
                profile.context_length <= 0 || profile.minimum_prompt_tokens <= 0 ||
                profile.generation_tokens <= 0 || profile.request_timeout_seconds <= 0 ||
                profile.readiness_timeout_seconds <= 0 ||
                static_cast<std::int64_t>(profile.minimum_prompt_tokens) + profile.generation_tokens >=
                    profile.context_length)
                throw std::invalid_argument("invalid E2E certification selection/profile");
            std::size_t matched = 0;
            for (auto &test_case : cases)
            {
                if (test_case.activation_precision != selection.activation ||
                    test_case.kv_cache_precision != selection.kv_cache ||
                    test_case.mtp != selection.mtp ||
                    test_case.prefill_graph != selection.prefill_graph ||
                    test_case.expert_overlay.has_value() != selection.owner_order.has_value())
                    continue;
                if (test_case.expert_overlay &&
                    (test_case.expert_overlay->owner_order != *selection.owner_order ||
                     test_case.expert_overlay->movement != *selection.movement))
                    continue;
                if (test_case.e2e_certification)
                    throw std::invalid_argument("overlapping E2E certification selections");
                test_case.e2e_certification = profile;
                ++matched;
            }
            if (matched != 1)
                throw std::invalid_argument("E2E certification selection must match exactly one parity cell");
        }
        return cases;
    }

} // namespace llaminar2::test::parity

// Kept separate so the matrix definition and its HTTP transport projection
// have independently reviewable ownership boundaries.
#include "ModelParityE2EExport.h"
