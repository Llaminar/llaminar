/**
 * @file ParityTestBase.h
 * @brief Base class and utilities for PyTorch parity tests
 *
 * Provides standardized infrastructure for comparing Llaminar inference
 * against PyTorch ground truth. All parity tests should inherit from
 * ParityTestBase to ensure consistent:
 *
 * - Metric calculations (cosine similarity, KL divergence, Top-K overlap)
 * - Table visualization of layer-by-layer results
 * - Pass/fail assertions with configurable thresholds
 * - Snapshot loading and regeneration
 *
 * MPI Support:
 *   For tensor-parallel tests, set mpi_ctx_ in SetUp() and override
 *   getDeviceForRank() instead of getDevice(). The base class handles:
 *   - Printing only on rank 0
 *   - Snapshot regeneration only on rank 0 (with barrier)
 *   - MPI barriers at key synchronization points
 *
 * Usage (single-rank):
 *   class Test__MyCUDAParity : public ParityTestBase {
 *   protected:
 *       void SetUp() override {
 *           config_.cosine_threshold = 0.99f;
 *           config_.early_layers_count = 6;
 *           ParityTestBase::SetUp();  // Regenerates snapshots
 *       }
 *
 *       DeviceId getDevice() override { return gpu_device_; }
 *       std::string getBackendName() override { return "CUDA"; }
 *   };
 *
 * Usage (tensor-parallel MPI):
 *   class Test__MyTPParity : public ParityTestBase {
 *   protected:
 *       void SetUp() override {
 *           mpi_ctx_ = std::make_shared<MPIContext>();
 *           config_.cosine_threshold = 0.94f;  // Relaxed for TP
 *           ParityTestBase::SetUp();
 *       }
 *       DeviceId getDeviceForRank() override {
 *           return (mpi_ctx_->rank() == 0) ? DeviceId::cuda(0) : DeviceId::rocm(0);
 *       }
 *       std::string getBackendName() override { return "TensorParallel"; }
 *   };
 *
 * @author David Sanftenberg
 * @date 2026-01-11
 */

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <charconv>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <optional>

// libfort for formatted table output
#include "fort.hpp"

#include "loaders/ModelContext.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/debug/TPSnapshot.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/mtp/MTPStateTransaction.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "utils/MTPParitySnapshotContext.h"
#include "integration/parity/ParityCellLifecycle.h"
#include "integration/parity/ProductionParityModelPath.h"

// Pipeline parallelism support
#include "config/PipelineConfig.h"
#include "collective/BackendRouter.h"
#include "collective/BackendRouter.h"

// Modern orchestration runner support (for incremental migration)
#include "utils/TestOrchestrationHelper.h"
#include "../../utils/ParityGDNHeadPermutation.h"
#include "../../utils/MoERoutingBoundary.h"
#include "../../utils/ParityNumericalAggregation.h"
#include "../../utils/ParitySnapshotSelection.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "kernels/KernelFactory.h"
#include "backends/BackendManager.h"
#include "backends/GlobalDeviceAddress.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Sha256.h"
#include "utils/ProductionParityEvidence.h"
#include "utils/ReferenceGenerationLease.h"
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/ops/ROCmEmbeddingKernelT.h"
#endif
#include "utils/Logger.h"
#include "utils/MPIContext.h"
#include "utils/NUMATopology.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"

// NumPy .npy file loading
#include <cnpy.h>

// MPI for tensor-parallel tests
#include <mpi.h>

// For snapshot cache mutex
#include <mutex>

// For CSV results directory creation
#include <filesystem>
#include <cstdlib>

namespace llaminar2::test::parity
{
    /** One-token blocks make a strict partial hit possible for every prompt. */
    inline constexpr int kProductionParityPrefixRestoreProofBlockSize = 1;


    // =============================================================================
    // Configuration
    // =============================================================================

    enum class ParityForwardPhase
    {
        Prefill,
        Decode,
    };

    /**
     * @brief Evidence contract for the prefill that precedes decode parity.
     *
     * A fresh prefill must execute the model graph and publish every requested
     * checkpoint. A complete prefix restore deliberately performs no model
     * compute, so its proof is the restored byte state plus the ordinary
     * checkpoint-by-checkpoint comparisons produced by subsequent decode.
     */
    enum class ParityDecodePrefillMode
    {
        FreshCompute,
        CompletePrefixRestore,
    };

    /** @brief Ordered prefix-cache lifecycle phases proved by every campaign. */
    enum class PrefixRestoreProofPhase
    {
        FreshSeed,
        CompleteHit,
        PartialHit,
    };

    /**
     * @brief Rank participation contract for additive reference generation.
     *
     * Symmetric parity runners execute test control flow on every rank and may
     * collectively publish one generated reference. A server-shaped MPI
     * runner instead keeps peer ranks inside the production command loop; in
     * that regime only the artifact authority may touch the reference lease or
     * test-only coordination primitives.
     */
    enum class ParityReferenceGenerationCoordination : std::uint8_t
    {
        SymmetricRanks,
        ArtifactAuthorityOnly,
    };

    /** @return Stable CSV spelling for a prefix lifecycle proof phase. */
    inline const char *prefixRestoreProofPhaseName(
        PrefixRestoreProofPhase phase)
    {
        switch (phase)
        {
        case PrefixRestoreProofPhase::FreshSeed:
            return "fresh_seed";
        case PrefixRestoreProofPhase::CompleteHit:
            return "complete_hit";
        case PrefixRestoreProofPhase::PartialHit:
            return "partial_hit";
        }
        return "unknown";
    }

    /** @return Stable CSV spelling for a terminal-payload comparison policy. */
    inline const char *mtpTerminalPayloadComparisonPolicyName(
        MTPTerminalPayloadComparisonPolicy policy)
    {
        switch (policy)
        {
        case MTPTerminalPayloadComparisonPolicy::ExactBytes:
            return "exact_bytes";
        case MTPTerminalPayloadComparisonPolicy::
            ExactUnlessMoEPlacementChanged:
            return "exact_unless_moe_placement_changed";
        }
        return "unknown";
    }

    /** @return Stable CSV spelling for a main-KV comparison policy. */
    inline const char *mtpMainKVPayloadComparisonPolicyName(
        MTPMainKVPayloadComparisonPolicy policy)
    {
        switch (policy)
        {
        case MTPMainKVPayloadComparisonPolicy::ExactBytes:
            return "exact_bytes";
        case MTPMainKVPayloadComparisonPolicy::
            ExactPrefixNumericalSuffixAfterMoEPlacementChange:
            return "exact_prefix_numerical_suffix_after_moe_placement_change";
        }
        return "unknown";
    }

    inline const char *parityForwardPhaseName(ParityForwardPhase phase)
    {
        switch (phase)
        {
        case ParityForwardPhase::Prefill:
            return "prefill";
        case ParityForwardPhase::Decode:
            return "decode";
        }
        return "unknown";
    }

    /** Diagnostic publication inventory captured into production GPU graphs. */
    enum class ParitySnapshotCaptureInventory : std::uint8_t
    {
        /** Capture exactly the checkpoints authenticated by the reference pack. */
        AuthenticatedReferenceCheckpoints = 0,
        /** Capture every stage output, including values absent from the oracle. */
        EveryPublishedOutput,
    };

    /**
     * @brief Graph/snapshot execution contract for parity forward calls.
     *
     * This is intentionally model-agnostic. GPU parity tests can opt into this
     * contract to exercise the production graph path while still collecting
     * snapshots through graph-stable copy storage. The base forwards through a
     * small polymorphic API so future parity suites can reuse the same machinery
     * instead of hand-rolling stage execution or snapshot publication.
     */
    struct ParityGraphSnapshotPolicy
    {
        ParitySnapshotCaptureInventory capture_inventory =
            ParitySnapshotCaptureInventory::
                AuthenticatedReferenceCheckpoints;
        bool enabled = false;
        bool require_graph_execution_on_gpu = false;
        bool require_snapshot_publication = true;
        bool require_prefill_graph_capture_on_gpu = false;
        bool retry_prefill_after_warmup_for_capture = false;
        std::vector<std::string> prefill_snapshot_capture_filter;
        std::vector<std::string> decode_snapshot_capture_filter;
        std::vector<std::string> required_prefill_snapshot_keys;
        std::vector<std::string> required_decode_snapshot_keys;

        const std::vector<std::string> &snapshotCaptureFilter(
            ParityForwardPhase phase) const
        {
            return phase == ParityForwardPhase::Prefill
                       ? prefill_snapshot_capture_filter
                       : decode_snapshot_capture_filter;
        }

        const std::vector<std::string> &requiredSnapshotKeys(
            ParityForwardPhase phase) const
        {
            return phase == ParityForwardPhase::Prefill
                       ? required_prefill_snapshot_keys
                       : required_decode_snapshot_keys;
        }
    };

    /** @brief Resolve the shared soft wall-time target reported by matrix cells. */
    inline double productionParityTargetSecondsFromEnvironment()
    {
        constexpr double default_target_seconds = 4500.0;
        const char *raw =
            std::getenv("LLAMINAR_PRODUCTION_PARITY_TARGET_SECONDS");
        if (!raw || !*raw)
            return default_target_seconds;

        char *end = nullptr;
        const double parsed = std::strtod(raw, &end);
        if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0)
        {
            ADD_FAILURE()
                << "LLAMINAR_PRODUCTION_PARITY_TARGET_SECONDS must be a "
                   "finite positive number, got '"
                << raw << "'";
            return default_target_seconds;
        }
        return parsed;
    }

    /**
     * @brief Resolve the aggregate runner's trusted model-digest cache.
     *
     * The directory is deliberately optional for focused diagnostics outside
     * the process-campaign lifecycle. The aggregate runner creates a private,
     * run-scoped directory and sibling campaign processes coalesce only
     * identical device/inode/size/mtime/ctime keys. Registered process
     * campaigns themselves cannot bypass the aggregate runner's tmpfs staging.
     */
    inline std::optional<std::filesystem::path>
    productionParityDigestCacheFromEnvironment()
    {
        const char *raw =
            std::getenv("LLAMINAR_PRODUCTION_PARITY_DIGEST_CACHE");
        if (!raw || !*raw)
            return std::nullopt;
        return std::filesystem::path(raw);
    }

    /** @brief Return whether request-local evidence contains a graph counter. */
    inline bool parityForwardGraphHasCounter(
        const std::vector<PerfStatRecord> &records,
        const std::string &name)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&name](const PerfStatRecord &record)
            {
                return record.kind == PerfStatRecord::Kind::Counter &&
                       record.domain == "forward_graph" &&
                       record.name == name &&
                       record.value > 0.0;
            });
    }

    /** @brief Return whether decode captured or replayed a native graph phase. */
    inline bool parityForwardGraphHasDecodePhase(
        const std::vector<PerfStatRecord> &records,
        const std::string &capture_phase)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&capture_phase](const PerfStatRecord &record)
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != "forward_graph" ||
                    record.name != "decode_graph_phase" ||
                    record.phase != "decode" ||
                    record.value <= 0.0)
                {
                    return false;
                }
                const auto it = record.tags.find("phase");
                return it != record.tags.end() &&
                       it->second == capture_phase;
            });
    }

    /** @brief Return whether one exact forward counter ran in a named phase. */
    inline bool parityForwardGraphHasCounterPhase(
        const std::vector<PerfStatRecord> &records,
        const std::string &name,
        const std::string &phase)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&name, &phase](const PerfStatRecord &record)
            {
                return record.kind == PerfStatRecord::Kind::Counter &&
                       record.domain == "forward_graph" &&
                       record.name == name && record.phase == phase &&
                       record.value > 0.0;
            });
    }

    /** @brief Materialize the canonical graph/economy CSV record. */
    inline ProductionParityEvidence collectProductionParityEvidence(
        const std::vector<PerfStatRecord> &records,
        bool graph_execution,
        ProductionParityExecutionTopology execution_topology,
        ProductionParityGraphContract graph_contract,
        bool model_context_reused,
        double elapsed_seconds)
    {
        ProductionParityEvidence evidence;
        evidence.graph_execution = graph_execution;
        evidence.execution_topology = execution_topology;
        evidence.graph_contract = graph_contract;
        evidence.forward_full_graph_capture = parityForwardGraphHasCounter(
            records, "full_graph_capture_executable_nodes");
        evidence.forward_full_graph_replay = parityForwardGraphHasCounter(
            records, "full_graph_replay_calls");
        const ProductionDeviceGenerationEvidence generation_evidence =
            collectProductionDeviceGenerationEvidence(records);
        evidence.device_generation_controller =
            generation_evidence.controller_observed;
        evidence.generation_execution_policy = generation_evidence.policy;
        evidence.native_generation_parent =
            generation_evidence.hasNativeParent();
        evidence.hosted_ticket_boundary_certified =
            generation_evidence.hosted_ticket_boundary_certified;
        evidence.generation_loop_certified =
            generation_evidence.hasCertifiedGenerationLoop();
        evidence.generation_certification_detail =
            generation_evidence.certification_detail;
        // A captured forward nested inside a host-scheduled MTP transaction is
        // useful but is not one complete production generation graph. Preserve
        // the inner-forward fields for diagnosis and make the established
        // full-graph fields describe the complete request as their name claims.
        const bool complete_generation_parent =
            !evidence.device_generation_controller ||
            evidence.native_generation_parent;
        evidence.full_graph_capture =
            evidence.forward_full_graph_capture && complete_generation_parent;
        evidence.full_graph_replay =
            evidence.forward_full_graph_replay && complete_generation_parent;
        evidence.segmented_plan = parityForwardGraphHasCounter(
            records, "segmented_plan_segments");
        const bool retained_parent_capture =
            parityForwardGraphHasCounter(
                records, "retained_parent_transaction_zero_launches") ||
            parityForwardGraphHasCounter(
                records, "retained_parent_transaction_captures");
        const bool retained_parent_replay =
            parityForwardGraphHasCounter(records, "retained_parent_replays") ||
            parityForwardGraphHasCounter(
                records, "retained_parent_transaction_replays");
        evidence.decode_graph_capture =
            parityForwardGraphHasDecodePhase(records, "capture") ||
            parityForwardGraphHasCounterPhase(
                records,
                "retained_parent_transaction_captures",
                "decode");
        evidence.decode_graph_replay =
            parityForwardGraphHasDecodePhase(records, "replay") ||
            parityForwardGraphHasCounterPhase(
                records,
                "retained_parent_transaction_replays",
                "decode");
        evidence.segmented_capture =
            evidence.segmented_plan &&
            (parityForwardGraphHasCounter(
                 records, "segmented_graph_capture_executable_nodes") ||
             parityForwardGraphHasCounter(
                 records, "segmented_graph_capture_segments") ||
             retained_parent_capture);
        evidence.segmented_replay =
            evidence.segmented_plan &&
            (parityForwardGraphHasCounter(
                 records, "segmented_replay_segments") ||
             retained_parent_replay);
        evidence.model_context_reused = model_context_reused;
        evidence.elapsed_seconds = elapsed_seconds;
        evidence.target_seconds =
            productionParityTargetSecondsFromEnvironment();
        return evidence;
    }

    /**
     * @brief Classify a fully qualified production participant inventory.
     *
     * This overload keeps model-family fixtures from independently rebuilding
     * the homogeneous/heterogeneous decision and accidentally omitting CPU or
     * one GPU vendor.
     */
    inline ProductionParityExecutionTopology
    classifyProductionParityExecutionTopology(
        const std::vector<GlobalDeviceAddress> &devices)
    {
        const std::size_t cpu_count = static_cast<std::size_t>(std::count_if(
            devices.begin(),
            devices.end(),
            [](const GlobalDeviceAddress &device)
            {
                return device.isCPU();
            }));
        const std::size_t cuda_count = static_cast<std::size_t>(std::count_if(
            devices.begin(),
            devices.end(),
            [](const GlobalDeviceAddress &device)
            {
                return device.isCUDA();
            }));
        const std::size_t rocm_count = static_cast<std::size_t>(std::count_if(
            devices.begin(),
            devices.end(),
            [](const GlobalDeviceAddress &device)
            {
                return device.isROCm();
            }));
        return classifyProductionParityExecutionTopology(
            cpu_count,
            cuda_count,
            rocm_count);
    }

    /** Request-lifetime physical expert-movement contract for one parity cell. */
    enum class ParityMoEMovementExpectation : std::uint8_t
    {
        NotApplicable,
        NoMovement,
        PhysicalMovement,
    };

    /** Request-lifetime MTP policy contract for one generated parity cell. */
    enum class ParityMTPExpectation : std::uint8_t
    {
        Disabled,
        FixedDepth,
        DynamicDepth,
    };

    /**
     * @brief Configuration for parity test thresholds
     *
     * Different backends (CPU, CUDA, ROCm) may need different thresholds
     * due to varying quantization schemes and numerical precision.
     */
    struct ParityConfig
    {
        // Model and test setup — MUST be set by subclass (no defaults)
        std::string model_path;
        std::string snapshot_dir;
        std::string prompt;
        std::vector<int> token_ids;
        int decode_steps = 5;

        // Layer-by-layer thresholds
        float cosine_threshold = 0.99f;  ///< Minimum avg cosine similarity for layer pass
        bool use_avg_cosine = true;      ///< Use avg (true) or min (false) cosine for pass criteria
        int early_layers_count = 6;      ///< Number of early layers to enforce threshold on
        int min_early_layers_passed = 6; ///< Minimum early layers that must pass

        // LM_HEAD thresholds
        float kl_threshold = 0.15f;      ///< Maximum KL divergence for logits
        /**
         * @brief Optional KL budget for recursively conditioned MTP logits.
         *
         * An unset value deliberately inherits @ref kl_threshold so adding MTP
         * to a model cannot silently weaken its existing logit contract.  A
         * model may set this only when its typed parity definition separately
         * certifies the bounded numerical accumulation of its recursive
         * predictor.
         */
        std::optional<float> mtp_kl_threshold;
        float min_top1_accuracy = 60.0f; ///< Minimum Top-1 accuracy percentage
        float min_top5_accuracy = 80.0f; ///< Minimum Top-5 accuracy percentage
        int pytorch_top1_in_topk = 3;    ///< PyTorch's top-1 must be in llaminar's top-K (0=disabled)

        // Decode thresholds (for incremental decode tests)
        float decode_cosine_threshold = 0.99f;
        float min_decode_pass_rate = 0.8f; ///< Minimum fraction of decode steps that must pass

        /// Stages to exclude from per-layer parity comparison.
        /// Used for GLOBAL scope TP where column-parallel stages (Q/K/V projections)
        /// produce partial outputs that can't be directly compared to full PyTorch outputs.
        std::vector<std::string> excluded_stages;

        /// Stages whose semantic values are completed by a production
        /// collective. The evidence-source policy below selects an explicit
        /// `_ALLREDUCED` graph snapshot or legacy harness reconstruction.
        /// Stages listed here should not also be excluded.
        std::vector<std::string> allreduce_stages;

        /**
         * @brief Authority used for completed collective checkpoint evidence.
         *
         * Coordinated production runners and LocalTP publish explicit
         * `_ALLREDUCED` graph slots. Legacy cross-rank fixtures may instead
         * expose rank partials for evidence-only MPI reconstruction.
         */
        ParityCollectiveEvidenceSource collective_evidence_source =
            ParityCollectiveEvidenceSource::CrossRankPartials;

        /**
         * @brief Whether MPI ranks own disjoint pipeline layer ranges.
         *
         * Cross-rank PP retains rank-local layer and stage diagnostics while
         * taking embedding evidence from the head and logit evidence from the
         * tail. This explicit bit prevents generic multi-rank TP campaigns from
         * being mistaken for PP during artifact aggregation.
         */
        bool uses_cross_rank_pipeline = false;

        /**
         * @brief Declarative MoE rebalance/migration exercise policy.
         *
         * Parity tests that need to cover dynamic expert movement can opt in
         * here without hand-writing maintenance calls in each fixture. For
         * The base invokes the production ExpertOverlay boundary hook at the
         * declared cadence; that hook only wakes the asynchronous authority.
         */
        struct MoERebalanceExercise
        {
            bool enabled = false;
            bool require_production_overlay_authority = false;
            bool request_after_prefill = false;
            int request_every_decode_steps = 0;
            int min_decode_steps = 0;
            bool require_movement_epoch_advance = false;
            uint64_t min_movement_epoch_delta = 1;
        } moe_rebalance_exercise;

        /** Physical movement outcome that PerfStats must prove for this cell. */
        ParityMoEMovementExpectation moe_movement_expectation =
            ParityMoEMovementExpectation::NotApplicable;

        /** MTP controller mode and active depth that runtime probes must prove. */
        ParityMTPExpectation mtp_expectation =
            ParityMTPExpectation::Disabled;
        int mtp_expected_draft_depth = 0;
        /** Setup-time draft capacity retained even when execution is disabled. */
        int mtp_expected_graph_capacity = 0;

        /// Optional execution contract used by GPU parity suites that must run
        /// production graph replay/capture and collect snapshots from that path.
        ParityGraphSnapshotPolicy graph_snapshot_policy;
    };

    /**
     * @brief Read the comparison-time GDN layout directly from GGUF metadata.
     */
    inline ParityGDNHeadConfig parityGDNHeadConfigFromModel(
        const ModelContext *model_ctx)
    {
        if (!model_ctx)
            return {};

        const auto &arch = model_ctx->architecture();
        const auto &meta = model_ctx->model().metadata;
        const auto get_meta_int = [&](const std::string &suffix) -> int
        {
            const auto it = meta.find(arch + "." + suffix);
            if (it == meta.end())
                return 0;
            const auto &value = it->second;
            if (value.type == GGUFValueType::UINT32)
                return static_cast<int>(value.asUInt32());
            if (value.type == GGUFValueType::UINT64)
                return static_cast<int>(value.asUInt64());
            return 0;
        };

        return ParityGDNHeadConfig{
            .n_k_heads = get_meta_int("ssm.group_count"),
            .n_v_heads = get_meta_int("ssm.time_step_rank"),
            .d_state = get_meta_int("ssm.state_size"),
            .is_moe = get_meta_int("expert_count") > 0,
        };
    }

    // =============================================================================
    // Result Structures
    // =============================================================================

    /**
     * @brief Distribution statistics for a single tensor
     */
    struct TensorDistributionStats
    {
        float min = 0.0f;
        float max = 0.0f;
        float mean = 0.0f;
        float stddev = 0.0f;
        float kurtosis = 0.0f;
        float skewness = 0.0f; ///< Asymmetry; symmetric quant schemes assume ~0
        float p95 = 0.0f;
        float p99 = 0.0f;
        float outlier_fraction = 0.0f; ///< Fraction of |x| > 6σ (SmoothQuant indicator)
        float dynamic_range = 0.0f;    ///< log2(max|x| / median|x|) — bits needed
        float sparsity = 0.0f;         ///< Fraction of |x| < 1e-6
        float zero_fraction = 0.0f;    ///< Fraction of x == 0 exactly
        size_t nan_count = 0;          ///< Number of NaN values
        size_t inf_count = 0;          ///< Number of ±Inf values
        size_t element_count = 0;
    };

    /**
     * @brief Result of comparing a single tensor/stage
     */
    struct StageComparisonResult
    {
        std::string stage_name;
        bool passed = false;
        float cosine_similarity = 0.0f;
        float cosine_drop = 0.0f; ///< Drop from previous stage (positive = error introduced)
        float rel_l2_norm = 0.0f;
        float max_abs_diff = 0.0f;
        float kl_divergence = 0.0f;
        float snr_db = 0.0f;        ///< Signal-to-noise ratio in dB: 10·log10(‖signal‖²/‖error‖²)
        float rmse = 0.0f;          ///< Root mean squared error
        float error_entropy = 0.0f; ///< Shannon entropy of error histogram (bits)
        size_t total_elements = 0;
        TensorDistributionStats llaminar_stats;
        TensorDistributionStats pytorch_stats;

        // --- MoE routing-specific metrics (NaN for non-routing stages) ---
        bool is_routing_stage = false;                                      ///< True for MOE_ROUTING_INDICES / MOE_ROUTING_WEIGHTS
        float routing_overlap = std::numeric_limits<float>::quiet_NaN();    ///< Set overlap (Jaccard) for indices, sparse-vector cosine for weights
        float routing_top1_match = std::numeric_limits<float>::quiet_NaN(); ///< Fraction of tokens where top-1 expert matches (indices only)
        float routing_weight_l1 = std::numeric_limits<float>::quiet_NaN();  ///< Mean L1 distance of sparse weight vectors (weights only)
    };

    /**
     * @brief Aggregated statistics for a single layer
     */
    struct LayerStats
    {
        int layer_idx = 0;
        float avg_cosine_sim = 0.0f;
        float min_cosine_sim = 1.0f;
        std::string worst_stage;
        float max_cosine_drop = 0.0f; ///< Largest single-stage cosine drop (error introduced)
        std::string max_drop_stage;   ///< Stage that introduced the most error
        int stages_compared = 0;
        bool passed = false;
        float max_kurtosis = 0.0f;                        ///< Highest excess kurtosis across stages
        std::string max_kurtosis_stage;                   ///< Stage with highest kurtosis
        std::vector<StageComparisonResult> stage_results; ///< Per-stage detailed results
    };

    /**
     * @brief Summary of parity test results
     */
    struct ParityTestSummary
    {
        // Embedding
        float embedding_cosine = 0.0f;
        bool embedding_passed = false;

        // Per-layer stats
        std::vector<LayerStats> layer_stats;

        // LM_HEAD
        float lm_head_cosine = 0.0f;
        float lm_head_kl = 0.0f;
        float lm_head_top1 = 0.0f;
        float lm_head_top5 = 0.0f;
        bool lm_head_pytorch_top1_in_top3 = false; ///< PyTorch's top-1 is in llaminar's top-3
        bool lm_head_passed = false;

        // Overall
        int early_layers_passed = 0;
        int total_layers_passed = 0;
        bool overall_passed = false;
    };

    /**
     * @brief Statistics for a single decode step
     */
    struct DecodeStepStats
    {
        int step_idx = 0;
        bool has_logit_data = false; ///< This PP rank owns the LM-head result for the step
        float cosine_similarity = 0.0f;
        float kl_divergence = 0.0f;
        float top1_overlap = 0.0f;
        float top5_overlap = 0.0f;
        int llaminar_token = -1;
        int pytorch_token = -1;
        bool token_match = false;
        bool top5_match = false; ///< True if PyTorch top-1 appears in Llaminar top-5
        bool top3_match = false; ///< True if PyTorch top-1 appears in Llaminar top-3
        bool passed = false;
        std::vector<LayerStats> layer_stats; ///< Per-layer cosine similarity for this decode step
    };

    /**
     * @brief Summary of incremental decode parity results
     */
    struct DecodeParitySummary
    {
        std::vector<DecodeStepStats> step_stats;
        int steps_passed = 0;
        int steps_total = 0;
        int top1_matches = 0;
        int top3_matches = 0;
        int top5_matches = 0;
        float avg_cosine = 0.0f;
        float avg_kl = 0.0f;
        float top1_accuracy = 0.0f;
        float top3_accuracy = 0.0f;
        float top5_accuracy = 0.0f;
        bool overall_passed = false;
    };

    /**
     * @brief One authenticated serial-decode state boundary.
     *
     * A logical position alone cannot identify a decode result: a stale input
     * mailbox can carry the same integer beside a newer hidden row, and grouped
     * runners may publish several typed transitions around one token. Pairing
     * the reference step, committed token, expected position, and deep runtime
     * state makes that invalid combination observable and gives prefix replay
     * an unambiguous oracle.
     */
    struct ProductionParityDecodeBoundary
    {
        size_t reference_step = 0;
        int32_t committed_token = -1;
        /**
         * CPU serial execution returns a selected token before its main-model
         * row is forwarded.  The parity driver submits the authenticated
         * successor to forward @ref committed_token and thereby leaves that
         * successor as the production runner's pending condition.  Retaining
         * its identity makes the boundary complete: prefix replay can stage
         * the same pending condition before demanding byte-state equivalence.
         */
        std::optional<int32_t> pending_condition_token;
        int expected_current_position = 0;
        PrefixRuntimeStateSnapshot runtime_state;
    };

    /**
     * @brief One real production MTP transaction observed by the campaign.
     *
     * Serial teacher forcing remains the numerical oracle for ordinary main
     * decode rows, but it cannot prove that predictor sampling or grouped
     * verification ran. This boundary pairs the controller decision, response
     * clipping geometry, emitted token prefix, monotonic counter deltas, and
     * before/after request state for one actual `decodeStep()` call.
     */
    struct ProductionParityMTPTransactionBoundary
    {
        int requested_draft_depth = 0;
        int response_limited_draft_depth = 0;
        /** Draft depth of the externally materialized transaction snapshots. */
        int snapshot_execution_draft_depth = 0;
        int reference_step = 0;
        int condition_position = 0;
        std::vector<int32_t> emitted_tokens;
        std::vector<int32_t> serial_oracle_tokens;
        bool serial_token_exact = false;
        uint64_t attempted_draft_tokens = 0;
        uint64_t verifier_transactions = 0;
        /** Whether a separate full-budget adaptive-policy transaction ran. */
        bool dynamic_policy_witness_executed = false;
        /** Exact serial-token proof for the adaptive-policy transaction. */
        bool dynamic_policy_witness_serial_token_exact = false;
        /** Controller windows evaluated by the adaptive-policy transaction. */
        uint64_t dynamic_policy_witness_window_delta = 0;
        /** Predictor rows attempted by the adaptive-policy transaction. */
        uint64_t dynamic_policy_witness_attempted_draft_tokens = 0;
        /** Grouped verifiers run by the adaptive-policy transaction. */
        uint64_t dynamic_policy_witness_verifier_transactions = 0;
        /** Production tokens emitted by the adaptive-policy transaction. */
        std::vector<int32_t> dynamic_policy_witness_emitted_tokens;
        /** Matching serial-oracle suffix for the adaptive-policy transaction. */
        std::vector<int32_t> dynamic_policy_witness_serial_oracle_tokens;
        PrefixRuntimeStateSnapshot before;
        PrefixRuntimeStateSnapshot after;
    };

    /**
     * @brief Machine-readable proof for one mandatory prefix lifecycle phase.
     *
     * The request fields prove which production cache path ran. State evidence
     * proves byte-equivalent restored execution state, while checkpoint
     * evidence ties the resulting compute boundary back to Hugging Face.
     */
    struct PrefixRestoreEvidence
    {
        /**
         * @brief Exact terminal payload identity at one runtime boundary.
         *
         * GPU payloads are materialized only by the parity campaign's explicit
         * diagnostic probe.  Keeping the byte counts beside the digests makes
         * an unavailable payload distinguishable from an empty or differently
         * sharded payload in the canonical CSV evidence.
         */
        struct TerminalPayloadIdentity
        {
            bool hidden_hash_available = false;
            size_t hidden_bytes = 0;
            uint64_t hidden_hash = 0;
            bool logits_hash_available = false;
            size_t logits_bytes = 0;
            uint64_t logits_hash = 0;

            /**
             * @brief Project a deep runtime snapshot onto its terminal bytes.
             * @param snapshot State observed at an authenticated result boundary.
             * @return Exact availability, geometry, and digest evidence.
             */
            static TerminalPayloadIdentity fromSnapshot(
                const PrefixRuntimeStateSnapshot &snapshot)
            {
                return {
                    .hidden_hash_available =
                        snapshot.terminal_hidden_hash_available,
                    .hidden_bytes = snapshot.terminal_hidden_bytes,
                    .hidden_hash = snapshot.terminal_hidden_hash,
                    .logits_hash_available =
                        snapshot.terminal_logits_hash_available,
                    .logits_bytes = snapshot.terminal_logits_bytes,
                    .logits_hash = snapshot.terminal_logits_hash,
                };
            }
        };

        PrefixRestoreProofPhase phase = PrefixRestoreProofPhase::FreshSeed;
        bool cache_config_enabled = false;
        bool cache_ready = false;
        bool cache_bypassed = false;
        std::string cache_bypass_reason;
        PrefixCacheRequestSummary request;
        int current_position = 0;
        bool state_compared = false;
        bool state_equivalent = false;
        std::string state_detail;
        uint64_t observed_moe_movement_epoch = 0;
        uint64_t oracle_moe_movement_epoch = 0;
        MTPMainKVPayloadComparisonPolicy main_kv_policy =
            MTPMainKVPayloadComparisonPolicy::ExactBytes;
        MTPStateValidationResult::MainKVNumericalEvidence main_kv_numerical;
        MTPTerminalPayloadComparisonPolicy terminal_hidden_policy =
            MTPTerminalPayloadComparisonPolicy::ExactBytes;
        MTPStateValidationResult::TerminalPayloadNumericalEvidence
            terminal_hidden_numerical;
        MTPTerminalPayloadComparisonPolicy terminal_logits_policy =
            MTPTerminalPayloadComparisonPolicy::ExactBytes;
        MTPStateValidationResult::TerminalPayloadNumericalEvidence
            terminal_logits_numerical;
        TerminalPayloadIdentity observed_payload;
        TerminalPayloadIdentity oracle_payload;
        bool checkpoint_compared = false;
        float checkpoint_cosine = 0.0f;
        bool checkpoint_passed = false;
        bool passed = false;
    };

    /**
     * @brief Single authority for the numerical CSV contract of parity tests.
     *
     * Production-path campaigns and focused diagnostics must emit the same
     * machine-readable artifacts.  Keeping the serialization independent of a
     * GTest fixture lets MTP, prefix-cache, and expert-placement runners retain
     * their production orchestration while publishing the exact schemas used
     * by the classic PyTorch parity suites.
     */
    class ParityCSVArtifactWriter final
    {
    public:
        /**
         * @brief Return the canonical per-test results directory.
         *
         * The identity intentionally matches ParityTestBase exactly so a
         * production MTP helper and the classic fixture cannot split one test's
         * artifacts across different directories.
         */
        static std::filesystem::path resultsDir()
        {
            std::string test_name = "unknown";
            const auto *info =
                ::testing::UnitTest::GetInstance()->current_test_info();
            if (info)
            {
                test_name = std::string(
                                info->test_suite_name()
                                    ? info->test_suite_name()
                                    : "") +
                            "/" +
                            std::string(info->name() ? info->name() : "");
            }
            for (char &character : test_name)
            {
                if (character == '/' || character == '\\' ||
                    character == ':' || character == '*' ||
                    character == '?' || character == '"' ||
                    character == '<' || character == '>' ||
                    character == '|')
                {
                    character = '_';
                }
            }

            // Aggregate campaigns own a fresh, report-addressable root. This
            // makes stale files unrepresentable while focused/manual tests
            // retain the familiar source-tree revision namespace below.
            if (const char *root = std::getenv(
                    "LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT"))
            {
                const std::filesystem::path explicit_root(root);
                if (explicit_root.empty() || !explicit_root.is_absolute())
                {
                    throw std::runtime_error(
                        "LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT must be an "
                        "absolute path");
                }
                return explicit_root / test_name;
            }

#ifdef LLAMINAR_PARITY_SOURCE_DIR
            const std::filesystem::path parity_dir(LLAMINAR_PARITY_SOURCE_DIR);
#else
            const std::filesystem::path parity_dir =
                std::filesystem::absolute(std::filesystem::path(__FILE__))
                    .parent_path();
#endif
            std::string hash = "unknown";
            if (FILE *pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r"))
            {
                char buffer[64];
                if (fgets(buffer, sizeof(buffer), pipe))
                {
                    hash = buffer;
                    while (!hash.empty() &&
                           (hash.back() == '\n' || hash.back() == '\r'))
                    {
                        hash.pop_back();
                    }
                }
                pclose(pipe);
            }
            return parity_dir / "results" / hash / test_name;
        }

        /**
         * @brief Write prefill layer, stage, and aggregate artifacts.
         * @return true when every required file was written successfully.
         */
        static bool writePrefill(
            const std::filesystem::path &dir,
            const std::string &backend,
            const ParityTestSummary &summary)
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
                return false;

            bool ok = true;
            {
                std::ofstream out(dir / "prefill_layers.csv", std::ios::trunc);
                ok = out.is_open() && ok;
                if (out.is_open())
                {
                    out << "backend,layer,avg_cosine,min_cosine,worst_stage,"
                           "max_cosine_drop,max_drop_stage,stages_compared,"
                           "max_kurtosis,max_kurtosis_stage,passed\n";
                    if (summary.embedding_passed ||
                        summary.embedding_cosine != 0.0f)
                    {
                        out << csvEscape(backend) << ",EMBEDDING,"
                            << summary.embedding_cosine << ",,"
                            << ",,,,,,"
                            << boolText(summary.embedding_passed) << '\n';
                    }
                    for (const auto &layer : summary.layer_stats)
                    {
                        if (layer.stages_compared == 0 &&
                            layer.stage_results.empty())
                        {
                            continue;
                        }
                        out << csvEscape(backend) << ','
                            << layer.layer_idx << ','
                            << layer.avg_cosine_sim << ','
                            << layer.min_cosine_sim << ','
                            << csvEscape(layer.worst_stage) << ','
                            << layer.max_cosine_drop << ','
                            << csvEscape(layer.max_drop_stage) << ','
                            << layer.stages_compared << ','
                            << layer.max_kurtosis << ','
                            << csvEscape(layer.max_kurtosis_stage) << ','
                            << boolText(layer.passed) << '\n';
                    }
                    ok = out.good() && ok;
                }
            }
            {
                std::ofstream out(dir / "prefill_summary.csv", std::ios::trunc);
                ok = out.is_open() && ok;
                if (out.is_open())
                {
                    out << "backend,lm_head_cosine,lm_head_kl,lm_head_top1,"
                           "lm_head_top5,lm_head_pytorch_top1_in_topk,"
                           "early_layers_passed,total_layers_passed,"
                           "overall_passed\n";
                    out << csvEscape(backend) << ','
                        << summary.lm_head_cosine << ','
                        << summary.lm_head_kl << ','
                        << summary.lm_head_top1 << ','
                        << summary.lm_head_top5 << ','
                        << boolText(summary.lm_head_pytorch_top1_in_top3) << ','
                        << summary.early_layers_passed << ','
                        << summary.total_layers_passed << ','
                        << boolText(summary.overall_passed) << '\n';
                    ok = out.good() && ok;
                }
            }
            ok = writeStages(
                     dir / "prefill_stages.csv",
                     backend,
                     summary.layer_stats,
                     std::nullopt) &&
                 ok;
            return ok;
        }

        /**
         * @brief Write incremental-decode step, layer, and stage artifacts.
         * @return true when every required file was written successfully.
         */
        static bool writeDecode(
            const std::filesystem::path &dir,
            const std::string &backend,
            const DecodeParitySummary &summary)
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
                return false;

            bool ok = true;
            {
                std::ofstream out(dir / "decode_steps.csv", std::ios::trunc);
                ok = out.is_open() && ok;
                if (out.is_open())
                {
                    out << "backend,step,cosine,kl_divergence,top1_overlap,"
                           "top5_overlap,llaminar_token,pytorch_token,"
                           "token_match,top3_match,top5_match,passed\n";
                    for (const auto &step : summary.step_stats)
                    {
                        if (!step.has_logit_data)
                            continue;
                        out << csvEscape(backend) << ','
                            << step.step_idx << ','
                            << step.cosine_similarity << ','
                            << step.kl_divergence << ','
                            << step.top1_overlap << ','
                            << step.top5_overlap << ','
                            << step.llaminar_token << ','
                            << step.pytorch_token << ','
                            << boolText(step.token_match) << ','
                            << boolText(step.top3_match) << ','
                            << boolText(step.top5_match) << ','
                            << boolText(step.passed) << '\n';
                    }
                    ok = out.good() && ok;
                }
            }

            {
                std::ofstream out(dir / "decode_layers.csv", std::ios::trunc);
                ok = out.is_open() && ok;
                if (out.is_open())
                {
                    out << "backend,step,layer,avg_cosine,min_cosine,worst_stage,"
                           "max_cosine_drop,max_drop_stage,stages_compared,passed\n";
                    for (const auto &step : summary.step_stats)
                    {
                        for (const auto &layer : step.layer_stats)
                        {
                            if (layer.stages_compared == 0)
                                continue;
                            out << csvEscape(backend) << ','
                                << step.step_idx << ','
                                << layer.layer_idx << ','
                                << layer.avg_cosine_sim << ','
                                << layer.min_cosine_sim << ','
                                << csvEscape(layer.worst_stage) << ','
                                << layer.max_cosine_drop << ','
                                << csvEscape(layer.max_drop_stage) << ','
                                << layer.stages_compared << ','
                                << boolText(layer.passed) << '\n';
                        }
                    }
                    ok = out.good() && ok;
                }
            }

            if (summary.step_stats.empty())
            {
                ok = writeStages(
                         dir / "decode_stages.csv",
                         backend,
                         {},
                         0) &&
                     ok;
            }
            for (const auto &step : summary.step_stats)
            {
                ok = writeStages(
                         dir / "decode_stages.csv",
                         backend,
                         step.layer_stats,
                         step.step_idx,
                         step.step_idx != summary.step_stats.front().step_idx) &&
                     ok;
            }
            return ok;
        }

        /**
         * @brief Write exact serial-versus-grouped MTP transaction evidence.
         *
         * Hugging Face checkpoint metrics remain in `decode_stages.csv`. This
         * companion artifact records the independent byte-exact token contract
         * against the free-running serial Llaminar oracle plus the draft and
         * verifier counter deltas that prove grouped execution actually ran.
         *
         * @return True when the complete transaction table was written.
         */
        static bool writeMTPTransactions(
            const std::filesystem::path &dir,
            const std::string &backend,
            const std::vector<ProductionParityMTPTransactionBoundary> &rows)
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
                return false;

            std::ofstream out(dir / "mtp_transactions.csv", std::ios::trunc);
            if (!out.is_open())
                return false;
            out << "backend,requested_draft_depth,response_limited_draft_depth,"
                   "snapshot_execution_draft_depth,last_transaction_draft_depth,"
                   "reference_step,condition_position,emitted_token_count,"
                   "emitted_tokens,serial_oracle_tokens,serial_token_exact,"
                   "attempted_draft_tokens,verifier_transactions,"
                   "verifier_identity_transaction_count,verifier_identity_depth,"
                   "production_verifier_draft_tokens,"
                   "before_position,after_position,before_draft_steps,"
                   "after_draft_steps,before_verifier_runs,after_verifier_runs,"
                   "dynamic_policy_witness_executed,"
                   "dynamic_policy_witness_serial_token_exact,"
                   "dynamic_policy_witness_window_delta,"
                   "dynamic_policy_witness_attempted_draft_tokens,"
                   "dynamic_policy_witness_verifier_transactions,"
                   "dynamic_policy_witness_emitted_tokens,"
                   "dynamic_policy_witness_serial_oracle_tokens\n";
            for (const auto &row : rows)
            {
                out << csvEscape(backend) << ','
                    << row.requested_draft_depth << ','
                    << row.response_limited_draft_depth << ','
                    << row.snapshot_execution_draft_depth << ','
                    << row.after.mtp_last_transaction_draft_depth << ','
                    << row.reference_step << ','
                    << row.condition_position << ','
                    << row.emitted_tokens.size() << ','
                    << csvEscape(tokenList(row.emitted_tokens)) << ','
                    << csvEscape(tokenList(row.serial_oracle_tokens)) << ','
                    << boolText(row.serial_token_exact) << ','
                    << row.attempted_draft_tokens << ','
                    << row.verifier_transactions << ','
                    << row.after.mtp_observed_verifier_transaction_count << ','
                    << row.after.mtp_observed_verifier_draft_depth << ','
                    << csvEscape(tokenList(
                           row.after.mtp_observed_verifier_draft_tokens))
                    << ','
                    << row.before.current_position << ','
                    << row.after.current_position << ','
                    << row.before.mtp_draft_steps << ','
                    << row.after.mtp_draft_steps << ','
                    << row.before.mtp_verifier_runs << ','
                    << row.after.mtp_verifier_runs << ','
                    << boolText(row.dynamic_policy_witness_executed) << ','
                    << boolText(
                           row.dynamic_policy_witness_serial_token_exact)
                    << ','
                    << row.dynamic_policy_witness_window_delta << ','
                    << row.dynamic_policy_witness_attempted_draft_tokens << ','
                    << row.dynamic_policy_witness_verifier_transactions << ','
                    << csvEscape(tokenList(
                           row.dynamic_policy_witness_emitted_tokens))
                    << ','
                    << csvEscape(tokenList(
                           row.dynamic_policy_witness_serial_oracle_tokens))
                    << '\n';
            }
            return out.good();
        }

        /**
         * @brief Write fresh, complete-hit, and partial-hit prefix evidence.
         * @return true when the canonical artifact was written successfully.
         */
        static bool writePrefixRestore(
            const std::filesystem::path &dir,
            const std::string &backend,
            const std::vector<PrefixRestoreEvidence> &evidence)
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
                return false;

            std::ofstream out(dir / "prefix_restore.csv", std::ios::trunc);
            if (!out.is_open())
                return false;
            out << "backend,phase,cache_config_enabled,cache_ready,"
                   "cache_bypassed,cache_bypass_reason,request_enabled,"
                   "request_bypassed,request_bypass_reason,hit,partial_hit,"
                   "requested_tokens,matched_tokens,matched_blocks,"
                   "terminal_logits_restored,terminal_hidden_restored,"
                   "mtp_state_restored,hybrid_state_restored,storage_tier,"
                   "current_position,state_compared,state_equivalent,"
                   "state_detail,observed_moe_movement_epoch,"
                   "oracle_moe_movement_epoch,main_kv_policy,"
                   "main_kv_numerically_compared,"
                   "main_kv_exact_prefix_segments,"
                   "main_kv_numerical_suffix_payloads,"
                   "main_kv_numerical_elements,"
                   "main_kv_minimum_cosine,"
                   "main_kv_maximum_relative_l2,"
                   "main_kv_maximum_abs,"
                   "main_kv_numerically_passed,terminal_hidden_policy,"
                   "terminal_hidden_numerically_compared,"
                   "terminal_hidden_numerical_elements,"
                   "terminal_hidden_numerical_cosine,"
                   "terminal_hidden_numerical_rel_l2,"
                   "terminal_hidden_numerical_max_abs,"
                   "terminal_hidden_numerical_passed,"
                   "terminal_logits_policy,"
                   "terminal_logits_numerically_compared,"
                   "terminal_logits_numerical_elements,"
                   "terminal_logits_numerical_cosine,"
                   "terminal_logits_numerical_rel_l2,"
                   "terminal_logits_numerical_max_abs,"
                   "terminal_logits_numerical_passed,"
                   "observed_terminal_hidden_hash_available,"
                   "observed_terminal_hidden_bytes,observed_terminal_hidden_hash,"
                   "observed_terminal_logits_hash_available,"
                   "observed_terminal_logits_bytes,observed_terminal_logits_hash,"
                   "oracle_terminal_hidden_hash_available,"
                   "oracle_terminal_hidden_bytes,oracle_terminal_hidden_hash,"
                   "oracle_terminal_logits_hash_available,"
                   "oracle_terminal_logits_bytes,oracle_terminal_logits_hash,"
                   "checkpoint_compared,checkpoint_cosine,"
                   "checkpoint_passed,passed\n";
            for (const auto &row : evidence)
            {
                out << csvEscape(backend) << ','
                    << prefixRestoreProofPhaseName(row.phase) << ','
                    << boolText(row.cache_config_enabled) << ','
                    << boolText(row.cache_ready) << ','
                    << boolText(row.cache_bypassed) << ','
                    << csvEscape(row.cache_bypass_reason) << ','
                    << boolText(row.request.enabled) << ','
                    << boolText(row.request.bypassed) << ','
                    << csvEscape(row.request.bypass_reason) << ','
                    << boolText(row.request.hit) << ','
                    << boolText(row.request.partial_hit) << ','
                    << row.request.requested_tokens << ','
                    << row.request.matched_tokens << ','
                    << row.request.matched_blocks << ','
                    << boolText(row.request.terminal_logits_restored) << ','
                    << boolText(row.request.terminal_hidden_restored) << ','
                    << boolText(row.request.mtp_state_restored) << ','
                    << boolText(row.request.hybrid_state_restored) << ','
                    << csvEscape(row.request.storage_tier) << ','
                    << row.current_position << ','
                    << boolText(row.state_compared) << ','
                    << boolText(row.state_equivalent) << ','
                    << csvEscape(row.state_detail) << ','
                    << row.observed_moe_movement_epoch << ','
                    << row.oracle_moe_movement_epoch << ','
                    << mtpMainKVPayloadComparisonPolicyName(
                           row.main_kv_policy)
                    << ','
                    << boolText(row.main_kv_numerical.compared) << ','
                    << row.main_kv_numerical.exact_prefix_segments << ','
                    << row.main_kv_numerical.numerical_suffix_payloads << ','
                    << row.main_kv_numerical.elements << ','
                    << row.main_kv_numerical.minimum_cosine << ','
                    << row.main_kv_numerical.maximum_relative_l2 << ','
                    << row.main_kv_numerical.maximum_abs << ','
                    << boolText(row.main_kv_numerical.passed) << ','
                    << mtpTerminalPayloadComparisonPolicyName(
                           row.terminal_hidden_policy)
                    << ','
                    << boolText(row.terminal_hidden_numerical.compared)
                    << ','
                    << row.terminal_hidden_numerical.elements << ','
                    << row.terminal_hidden_numerical.cosine << ','
                    << row.terminal_hidden_numerical.relative_l2 << ','
                    << row.terminal_hidden_numerical.max_abs << ','
                    << boolText(row.terminal_hidden_numerical.passed) << ','
                    << mtpTerminalPayloadComparisonPolicyName(
                           row.terminal_logits_policy)
                    << ','
                    << boolText(row.terminal_logits_numerical.compared)
                    << ','
                    << row.terminal_logits_numerical.elements << ','
                    << row.terminal_logits_numerical.cosine << ','
                    << row.terminal_logits_numerical.relative_l2 << ','
                    << row.terminal_logits_numerical.max_abs << ','
                    << boolText(row.terminal_logits_numerical.passed) << ','
                    << boolText(
                           row.observed_payload.hidden_hash_available)
                    << ','
                    << row.observed_payload.hidden_bytes << ','
                    << row.observed_payload.hidden_hash << ','
                    << boolText(
                           row.observed_payload.logits_hash_available)
                    << ','
                    << row.observed_payload.logits_bytes << ','
                    << row.observed_payload.logits_hash << ','
                    << boolText(row.oracle_payload.hidden_hash_available)
                    << ','
                    << row.oracle_payload.hidden_bytes << ','
                    << row.oracle_payload.hidden_hash << ','
                    << boolText(row.oracle_payload.logits_hash_available)
                    << ','
                    << row.oracle_payload.logits_bytes << ','
                    << row.oracle_payload.logits_hash << ','
                    << boolText(row.checkpoint_compared) << ','
                    << row.checkpoint_cosine << ','
                    << boolText(row.checkpoint_passed) << ','
                    << boolText(row.passed) << '\n';
            }
            return out.good();
        }

        /**
         * @brief Publish one canonical prefill artifact set from PP rank fragments.
         *
         * The aggregate carries head-owned embedding/early-layer counts and
         * tail-owned LM-head metrics. Layer and stage rows remain rank-local
         * until this diagnostic-only post-inference merge so production PP
         * execution itself is unchanged.
         */
        static bool mergePipelinePrefill(
            const std::filesystem::path &dir,
            const std::string &backend,
            const ParityTestSummary &aggregate,
            const std::vector<std::filesystem::path> &rank_dirs)
        {
            bool ok = writePrefill(dir, backend, aggregate);
            ok = mergeCsvFragments(
                     dir / "prefill_layers.csv",
                     rank_dirs,
                     "prefill_layers.csv") &&
                 ok;
            ok = mergeCsvFragments(
                     dir / "prefill_stages.csv",
                     rank_dirs,
                     "prefill_stages.csv") &&
                 ok;
            return ok;
        }

        /**
         * @brief Publish canonical decode artifacts from disjoint PP ranks.
         *
         * Only the tail fragment contributes decode-step logits. Every rank
         * contributes the layer/stage rows for its owned pipeline interval.
         */
        static bool mergePipelineDecode(
            const std::filesystem::path &dir,
            const std::vector<std::filesystem::path> &rank_dirs)
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
                return false;

            bool ok = mergeCsvFragments(
                dir / "decode_steps.csv", rank_dirs, "decode_steps.csv");
            ok = mergeCsvFragments(
                     dir / "decode_layers.csv",
                     rank_dirs,
                     "decode_layers.csv") &&
                 ok;
            ok = mergeCsvFragments(
                     dir / "decode_stages.csv",
                     rank_dirs,
                     "decode_stages.csv") &&
                 ok;
            return ok;
        }

        /**
         * @brief Write the graph-path and economy proof for one matrix cell.
         */
        static bool writeProductionPath(
            const std::filesystem::path &dir,
            const std::string &backend,
            const std::string &device,
            const ProductionParityEvidence &evidence)
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
                return false;

            std::ofstream out(dir / "production_path.csv", std::ios::trunc);
            if (!out.is_open())
                return false;
            out << "backend,device,execution_path,execution_topology,"
                   "graph_contract,"
                   "forward_full_graph_capture,forward_full_graph_replay,"
                   "full_graph_capture,full_graph_replay,decode_graph_capture,"
                   "decode_graph_replay,device_generation_controller,"
                   "generation_execution_policy,native_generation_parent,"
                   "hosted_ticket_boundary_certified,"
                   "generation_loop_certified,"
                   "generation_certification_detail,"
                   "segmented_plan,segmented_capture,"
                   "segmented_replay,model_context_reused,elapsed_seconds,"
                   // Here "budget" is the measured performance target; it is
                   // not a process or test timeout.
                   "budget_seconds,within_budget\n";
            out << csvEscape(backend) << ','
                << csvEscape(device) << ','
                << (evidence.graph_execution ? "graph" : "non_graph") << ','
                << productionParityExecutionTopologyName(
                       evidence.execution_topology)
                << ','
                << productionParityGraphContractName(evidence.graph_contract)
                << ','
                << boolText(evidence.forward_full_graph_capture) << ','
                << boolText(evidence.forward_full_graph_replay) << ','
                << boolText(evidence.full_graph_capture) << ','
                << boolText(evidence.full_graph_replay) << ','
                << boolText(evidence.decode_graph_capture) << ','
                << boolText(evidence.decode_graph_replay) << ','
                << boolText(evidence.device_generation_controller) << ','
                << productionDeviceGenerationPolicyName(
                       evidence.generation_execution_policy)
                << ','
                << boolText(evidence.native_generation_parent) << ','
                << boolText(evidence.hosted_ticket_boundary_certified) << ','
                << boolText(evidence.generation_loop_certified) << ','
                << csvEscape(evidence.generation_certification_detail) << ','
                << boolText(evidence.segmented_plan) << ','
                << boolText(evidence.segmented_capture) << ','
                << boolText(evidence.segmented_replay) << ','
                << boolText(evidence.model_context_reused) << ','
                << evidence.elapsed_seconds << ','
                << evidence.target_seconds << ','
                << boolText(evidence.elapsed_seconds <= evidence.target_seconds)
                << '\n';
            return out.good();
        }

    private:
        static const char *boolText(bool value)
        {
            return value ? "true" : "false";
        }

        static std::string csvEscape(const std::string &value)
        {
            if (value.find_first_of(",\"\n\r") == std::string::npos)
                return value;
            std::string escaped = "\"";
            for (const char character : value)
            {
                if (character == '"')
                    escaped.push_back('"');
                escaped.push_back(character);
            }
            escaped.push_back('"');
            return escaped;
        }

        /** @brief Serialize a token row without introducing CSV delimiters. */
        static std::string tokenList(const std::vector<int32_t> &tokens)
        {
            std::ostringstream out;
            for (size_t index = 0; index < tokens.size(); ++index)
            {
                if (index != 0)
                    out << ';';
                out << tokens[index];
            }
            return out.str();
        }

        /**
         * @brief Concatenate rank-local CSV bodies after verifying one schema.
         */
        static bool mergeCsvFragments(
            const std::filesystem::path &destination,
            const std::vector<std::filesystem::path> &rank_dirs,
            const std::filesystem::path &filename)
        {
            std::ofstream output(destination, std::ios::trunc);
            if (!output.is_open())
                return false;

            std::string canonical_header;
            for (const auto &rank_dir : rank_dirs)
            {
                std::ifstream input(rank_dir / filename);
                if (!input.is_open())
                    return false;

                std::string header;
                if (!std::getline(input, header) || header.empty())
                    return false;
                if (canonical_header.empty())
                {
                    canonical_header = header;
                    output << canonical_header << '\n';
                }
                else if (header != canonical_header)
                {
                    return false;
                }

                std::string row;
                while (std::getline(input, row))
                {
                    if (!row.empty())
                        output << row << '\n';
                }
            }
            return !canonical_header.empty() && output.good();
        }

        static void writeDistributionHeader(
            std::ofstream &out,
            const std::string &prefix)
        {
            out << prefix << "min," << prefix << "max,"
                << prefix << "mean," << prefix << "stddev,"
                << prefix << "kurtosis," << prefix << "skewness,"
                << prefix << "p95," << prefix << "p99,"
                << prefix << "outlier_frac," << prefix << "dynamic_range,"
                << prefix << "sparsity," << prefix << "zero_frac,"
                << prefix << "nan_count," << prefix << "inf_count,"
                << prefix << "elements";
        }

        static void writeDistribution(
            std::ofstream &out,
            const TensorDistributionStats &stats)
        {
            out << stats.min << ',' << stats.max << ','
                << stats.mean << ',' << stats.stddev << ','
                << stats.kurtosis << ',' << stats.skewness << ','
                << stats.p95 << ',' << stats.p99 << ','
                << stats.outlier_fraction << ',' << stats.dynamic_range << ','
                << stats.sparsity << ',' << stats.zero_fraction << ','
                << stats.nan_count << ',' << stats.inf_count << ','
                << stats.element_count;
        }

        static bool writeStages(
            const std::filesystem::path &path,
            const std::string &backend,
            const std::vector<LayerStats> &layers,
            std::optional<int> decode_step,
            bool append = false)
        {
            std::ofstream out(
                path,
                append ? (std::ios::out | std::ios::app)
                       : (std::ios::out | std::ios::trunc));
            if (!out.is_open())
                return false;
            if (!append)
            {
                out << "backend,";
                if (decode_step)
                    out << "step,";
                out << "layer,stage,cosine,cosine_drop,rel_l2,max_abs_diff,"
                       "snr_db,rmse,error_entropy,is_routing,routing_overlap,"
                       "routing_top1_match,routing_weight_l1,";
                writeDistributionHeader(out, "llaminar_");
                out << ',';
                writeDistributionHeader(out, "pytorch_");
                out << '\n';
            }
            for (const auto &layer : layers)
            {
                for (const auto &stage : layer.stage_results)
                {
                    out << csvEscape(backend) << ',';
                    if (decode_step)
                        out << *decode_step << ',';
                    out << layer.layer_idx << ','
                        << csvEscape(stage.stage_name) << ','
                        << (stage.is_routing_stage ? "" : std::to_string(stage.cosine_similarity)) << ','
                        << (stage.is_routing_stage ? "" : std::to_string(stage.cosine_drop)) << ','
                        << (stage.is_routing_stage ? "" : std::to_string(stage.rel_l2_norm)) << ','
                        << stage.max_abs_diff << ','
                        << (stage.is_routing_stage ? "" : std::to_string(stage.snr_db)) << ','
                        << (stage.is_routing_stage ? "" : std::to_string(stage.rmse)) << ','
                        << (stage.is_routing_stage ? "" : std::to_string(stage.error_entropy)) << ','
                        << (stage.is_routing_stage ? "1" : "0") << ','
                        << (stage.is_routing_stage ? std::to_string(stage.routing_overlap) : "") << ','
                        << (std::isnan(stage.routing_top1_match) ? "" : std::to_string(stage.routing_top1_match)) << ','
                        << (std::isnan(stage.routing_weight_l1) ? "" : std::to_string(stage.routing_weight_l1)) << ',';
                    writeDistribution(out, stage.llaminar_stats);
                    out << ',';
                    writeDistribution(out, stage.pytorch_stats);
                    out << '\n';
                }
            }
            return out.good();
        }
    };

    // =============================================================================
    // TP-Aware Result Structures
    // =============================================================================

    /**
     * @brief Per-device comparison result for tensor-parallel parity
     */
    struct TPDeviceComparisonResult
    {
        std::string device_id;          ///< Device identifier (e.g., "rank0_cuda0")
        int device_index = 0;           ///< Index in TP group
        float cosine_similarity = 0.0f; ///< Cosine vs corresponding PyTorch slice
        size_t slice_start = 0;         ///< Start column in PyTorch reference
        size_t slice_size = 0;          ///< Number of elements compared
        bool passed = false;
    };

    /**
     * @brief TP-aware comparison result for a single stage
     */
    struct TPStageComparisonResult
    {
        std::string stage_name;
        SnapshotShardingMode sharding_mode = SnapshotShardingMode::UNKNOWN;

        // Per-device comparisons (for column-parallel stages)
        std::vector<TPDeviceComparisonResult> device_results;

        // Combined result (concatenated partial outputs vs full PyTorch)
        StageComparisonResult combined_result;
        float combined_cosine = 0.0f;
        size_t combined_elements = 0;
        size_t reference_elements = 0;
        bool combined_passed = false;

        // Overall for this stage
        bool passed = false;
    };

    /**
     * @brief TP-aware layer statistics
     */
    struct TPLayerStats
    {
        int layer_idx = 0;
        int tp_degree = 1;

        // Per-stage results
        std::vector<TPStageComparisonResult> stage_results;

        // Aggregated metrics
        float avg_combined_cosine = 0.0f;
        float min_combined_cosine = 1.0f;
        std::string worst_stage;
        int stages_compared = 0;
        bool passed = false;
    };

    /**
     * @brief TP-aware parity test summary
     */
    struct TPParityTestSummary
    {
        int tp_degree = 1;
        std::vector<std::string> device_names; ///< Device IDs in TP group

        // Embedding
        TPStageComparisonResult embedding_result;

        // Per-layer stats
        std::vector<TPLayerStats> layer_stats;

        // LM_HEAD (always gathered, so single combined result)
        float lm_head_cosine = 0.0f;
        float lm_head_kl = 0.0f;
        float lm_head_top1 = 0.0f;
        float lm_head_top5 = 0.0f;
        bool lm_head_pytorch_top1_in_top3 = false;
        bool lm_head_passed = false;

        // Overall
        int early_layers_passed = 0;
        int total_layers_passed = 0;
        bool overall_passed = false;
    };

    // =============================================================================
    // Device and Parallelism Configuration (Model-Agnostic)
    // =============================================================================

    /**
     * @brief Device type for parity tests
     * Note: Named ParityDeviceType to avoid collision with llaminar2::DeviceType
     */
    enum class ParityDeviceType
    {
        CPU,
        CUDA,
        ROCm
    };

    /**
     * @brief Parallelism strategy for multi-device tests
     */
    enum class Parallelism
    {
        None,        ///< Single device, no parallelism
        LocalTP,     ///< Local Tensor Parallelism (multi-device, single process)
        LocalPP,     ///< Local Pipeline Parallelism (multi-device, single process)
        NodeTP, ///< Node-Local Tensor Parallelism (multi-rank MPI, same node)
        NodeLocalPP, ///< Node-Local Pipeline Parallelism (multi-rank MPI, same node)
        GlobalTP,    ///< Global Tensor Parallelism (multi-rank MPI, cross-node)
    };

    /**
     * @brief Collective backend for parallel tests
     */
    enum class Collective
    {
        None,          ///< No collective needed (single device)
        HOST,          ///< Host-based collective (staging through CPU)
        NCCL,          ///< NVIDIA NCCL (CUDA-CUDA)
        RCCL,          ///< AMD RCCL (ROCm-ROCm)
        HETEROGENEOUS, ///< Cross-vendor heterogeneous (CUDA-ROCm)
        MPI,           ///< MPI backend (for global TP, cross-rank)
    };

    // =============================================================================
    // Device Configuration Utilities
    // =============================================================================

    inline DeviceId toDeviceId(ParityDeviceType type, int index = 0)
    {
        switch (type)
        {
        case ParityDeviceType::CPU:
            return DeviceId::cpu();
        case ParityDeviceType::CUDA:
            return DeviceId::cuda(index);
        case ParityDeviceType::ROCm:
            return DeviceId::rocm(index);
        }
        return DeviceId::cpu();
    }

    inline GlobalDeviceAddress toGlobalAddress(ParityDeviceType type, int index = 0)
    {
        switch (type)
        {
        case ParityDeviceType::CPU:
            return GlobalDeviceAddress::cpu();
        case ParityDeviceType::CUDA:
            return GlobalDeviceAddress::cuda(index);
        case ParityDeviceType::ROCm:
            return GlobalDeviceAddress::rocm(index);
        }
        return GlobalDeviceAddress::cpu();
    }

    inline CollectiveBackendType toCollectiveBackend(Collective c)
    {
        switch (c)
        {
        case Collective::None:
            return CollectiveBackendType::HOST;
        case Collective::HOST:
            return CollectiveBackendType::HOST;
        case Collective::NCCL:
            return CollectiveBackendType::NCCL;
        case Collective::RCCL:
            return CollectiveBackendType::RCCL;
        case Collective::HETEROGENEOUS:
            return CollectiveBackendType::HETEROGENEOUS;
        case Collective::MPI:
            return CollectiveBackendType::MPI;
        }
        return CollectiveBackendType::HOST;
    }

    inline std::string deviceTypeName(ParityDeviceType type)
    {
        switch (type)
        {
        case ParityDeviceType::CPU:
            return "CPU";
        case ParityDeviceType::CUDA:
            return "CUDA";
        case ParityDeviceType::ROCm:
            return "ROCm";
        }
        return "Unknown";
    }

    inline std::string parallelismName(Parallelism p)
    {
        switch (p)
        {
        case Parallelism::None:
            return "None";
        case Parallelism::LocalTP:
            return "LocalTP";
        case Parallelism::LocalPP:
            return "LocalPP";
        case Parallelism::NodeTP:
            return "NodeTP";
        case Parallelism::NodeLocalPP:
            return "NodeLocalPP";
        case Parallelism::GlobalTP:
            return "GlobalTP";
        }
        return "Unknown";
    }

    inline std::string collectiveName(Collective c)
    {
        switch (c)
        {
        case Collective::None:
            return "None";
        case Collective::HOST:
            return "Host";
        case Collective::NCCL:
            return "NCCL";
        case Collective::RCCL:
            return "RCCL";
        case Collective::HETEROGENEOUS:
            return "HETEROGENEOUS";
        case Collective::MPI:
            return "MPI";
        }
        return "Unknown";
    }

    // =============================================================================
    // Hardware Detection Utilities
    // =============================================================================

    inline bool isMpiInitialized()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        return initialized != 0;
    }

    inline int getCudaDeviceCount()
    {
#ifdef HAVE_CUDA
        if (auto *backend = getCUDABackend())
            return backend->deviceCount();
#endif
        return 0;
    }

    inline int getRocmDeviceCount()
    {
#ifdef HAVE_ROCM
        if (auto *backend = getROCmBackend())
            return backend->deviceCount();
#endif
        return 0;
    }

    // =============================================================================
    // Backend Thresholds for Parity Tests
    // =============================================================================

    /**
     * @brief Backend-specific parity thresholds
     *
     * Different backends may have different numerical precision characteristics.
     * These thresholds define what constitutes "passing" parity for each backend.
     */
    struct BackendThresholds
    {
        float cosine_threshold = 0.999f;                ///< Min avg cosine for layer pass
        float decode_cosine_threshold = 0.99f;          ///< Threshold for decode parity
        int early_layers_count = 4;                     ///< Number of early layers to check
        int min_early_layers_passed = 4;                ///< Min early layers that must pass
        float kl_threshold = 0.05f;                     ///< Max KL divergence for logits
        /**
         * @brief Optional MTP-only KL budget; unset preserves @ref kl_threshold.
         */
        std::optional<float> mtp_kl_threshold;
        std::vector<std::string> excluded_stages = {};  ///< Stages to exclude from parity comparison
        std::vector<std::string> allreduce_stages = {}; ///< Stages to allreduce across MPI ranks before comparison
        float min_top1_accuracy = 80.0f;                ///< Min Top-1 accuracy %
        float min_top5_accuracy = 80.0f;                ///< Min Top-5 accuracy %
        float min_decode_pass_rate = 0.8f;              ///< Min fraction of decode steps passing
        int pytorch_top1_in_topk = 3;                   ///< PyTorch's top-1 must be in llaminar's top-K (0=disabled)
    };

    // =============================================================================
    // Test Configuration (Model-Agnostic)
    // =============================================================================

    /**
     * @brief Complete declarative test configuration
     *
     * The devices vector is the single source of truth for device configuration.
     * Use the helper methods device_count() and primary_device() to derive values.
     */
    struct TestConfig
    {
        std::string name;                      ///< Human-readable test name
        std::vector<ParityDeviceType> devices; ///< Device list (heterogeneous supported)
        Parallelism parallelism;               ///< Parallelism strategy

        /// Collective backend for TP modes (LocalTP, GlobalTP, NodeTP).
        /// PP modes should leave this as None — PP transfers are auto-selected
        /// by LocalPPContext based on device vendor types. For hybrid PP+TP,
        /// use tp_collective instead.
        Collective collective = Collective::None;

        BackendThresholds thresholds; ///< Parity thresholds
        std::string skip_reason;      ///< If non-empty, test will skip with this message
        int mpi_ranks = 1;            ///< Required MPI ranks for GlobalTP tests

        /// Model path override. If non-empty, overrides ParityConfig::model_path.
        /// Use this to test different quantization formats (e.g., Q4_0 vs Q8_0)
        /// which exercise different GEMM code paths (native-VNNI vs INT8-VNNI).
        std::string model_path;

        /// Snapshot directory override. If non-empty, overrides ParityConfig::snapshot_dir.
        /// Must be unique per model to avoid snapshot collisions between quant formats.
        std::string snapshot_dir;

        /// Prompt override. If non-empty, overrides ParityConfig::prompt for
        /// PyTorch snapshot generation.
        std::string prompt;

        /// Token ID override. If non-empty, overrides ParityConfig::token_ids
        /// for Llaminar execution. This is useful for chat-template prompts
        /// where the rendered text contains special tokens.
        std::vector<int> token_ids;

        /// PP stage device sizes for hybrid PP+TP configurations
        /// Example: {2, 1} means stage 0 has 2 devices (TP domain), stage 1 has 1 device
        /// Empty means one device per stage (pure PP) or all devices in one domain (pure TP)
        std::vector<int> pp_stage_sizes;

        /// Proportional layer split weights for PP stages.
        /// Example: {0.31, 0.69} gives stage 0 ~31% of layers and stage 1 ~69%.
        /// Empty means equal split. Must match num_pp_stages() if set.
        std::vector<float> pp_weights;

        /// TP backend for stages that are TP domains (only used when pp_stage_sizes has entries > 1)
        Collective tp_collective = Collective::None;

        /// Runner precision controls (defaults preserve current parity behavior)
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
        KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

        /// Decode steps override. 0 means use ParityConfig default (5).
        int decode_steps = 0;

        /// Optional MoE rebalance/migration exercise policy for parity bodies
        /// that use runPrefillParity()/runDecodeParity().
        ParityConfig::MoERebalanceExercise moe_rebalance_exercise;

        /** Request-lifetime movement evidence generated by the typed matrix. */
        ParityMoEMovementExpectation moe_movement_expectation =
            ParityMoEMovementExpectation::NotApplicable;

        /** Request-lifetime MTP evidence generated by the typed matrix. */
        ParityMTPExpectation mtp_expectation =
            ParityMTPExpectation::Disabled;
        int mtp_expected_draft_depth = 0;
        /** Setup-time draft capacity retained even when execution is disabled. */
        int mtp_expected_graph_capacity = 0;

        /// Physical and semantic MoE policy consumed by production runner setup.
        RoutedExpertComputePolicy routed_expert_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        RoutedExpertOwnerOrder routed_expert_owner_order =
            RoutedExpertOwnerOrder::Ordinal;
        MoEHotExpertCacheConfig moe_hot_expert_cache;
        RoutedExpertPrefillRuntimeConfig moe_routed_prefill;
        // A parity cell enters durable Dynamic maintenance only by declaring
        // it. The production runtime default is intentionally dynamic, but
        // inheriting that default here would route dense/static test cells
        // through a MoE-only orchestration path and make matrix behavior depend
        // on which cells happened to run earlier in the process campaign.
        MoERebalanceRuntimeConfig moe_rebalance{
            .mode = MoERebalanceRuntimeMode::Off,
        };
        std::shared_ptr<MoERoutedExpertPlacementPlan> moe_routed_expert_plan;

        /// Optional model-schema TP transport override used by hybrid GDN/MoE.
        std::string tp_allreduce_precision_override;

        /// Optional graph execution/snapshot contract for parity bodies that
        /// must exercise production graph capture/replay while collecting
        /// snapshots from that path.
        ParityGraphSnapshotPolicy graph_snapshot_policy;

        /**
         * Optional evidence authority override for stages completed by a
         * production collective. LocalTP derives post-collective evidence by
         * default; coordinated NodeTP production campaigns declare the same
         * authority explicitly so no test-only MPI operation can interleave
         * with their worker protocol.
         */
        std::optional<ParityCollectiveEvidenceSource>
            collective_evidence_source;

        // Derived accessors
        size_t device_count() const { return devices.size(); }
        ParityDeviceType primary_device() const { return devices.empty() ? ParityDeviceType::CPU : devices[0]; }
        bool is_local_tp() const { return parallelism == Parallelism::LocalTP; }
        bool is_local_pp() const { return parallelism == Parallelism::LocalPP; }
        bool is_node_tp() const { return parallelism == Parallelism::NodeTP; }
        bool is_node_local_pp() const { return parallelism == Parallelism::NodeLocalPP; }
        bool is_global_tp() const { return parallelism == Parallelism::GlobalTP; }
        /// Returns true for any cross-rank TP (NodeLocal or Global)
        bool is_cross_rank_tp() const { return is_node_tp() || is_global_tp(); }
        /// Returns true for any cross-rank PP
        bool is_cross_rank_pp() const { return is_node_local_pp(); }
        /// Returns true for any cross-rank parallelism (TP or PP)
        bool is_cross_rank() const { return is_cross_rank_tp() || is_cross_rank_pp(); }
        bool is_single_device() const { return parallelism == Parallelism::None && devices.size() == 1; }
        bool should_skip() const { return !skip_reason.empty(); }

        /// Check if this is a hybrid PP+TP configuration (PP with TP domains inside stages)
        bool is_hybrid_pp_tp() const
        {
            if (!is_local_pp() || pp_stage_sizes.empty())
                return false;
            // Hybrid if any stage has more than 1 device
            for (int size : pp_stage_sizes)
            {
                if (size > 1)
                    return true;
            }
            return false;
        }

        /// Get number of PP stages (uses pp_stage_sizes if set, otherwise device_count)
        size_t num_pp_stages() const
        {
            if (!pp_stage_sizes.empty())
                return pp_stage_sizes.size();
            return device_count();
        }
    };

    /**
     * @brief Check if a TestConfig can run on current hardware
     * @return std::nullopt if available, or a skip reason string
     */
    inline std::optional<std::string> checkHardwareAvailability(const TestConfig &cfg)
    {
        // Check explicit skip reason first
        if (cfg.should_skip())
            return cfg.skip_reason;

        // Check MPI initialization for LocalTP/LocalPP/NodeTP/NodeLocalPP/GlobalTP tests
        if (cfg.is_local_tp() || cfg.is_local_pp() || cfg.is_cross_rank())
        {
            if (!isMpiInitialized())
                return "Test requires MPI (run with mpirun)";
        }

        // Check MPI world size for cross-rank tests
        if (cfg.is_cross_rank())
        {
            int world_size = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            if (world_size < cfg.mpi_ranks)
            {
                return "Cross-rank test requires " + std::to_string(cfg.mpi_ranks) +
                       " MPI ranks (got " + std::to_string(world_size) + ")";
            }
        }

        int required_cuda = 0, required_rocm = 0;
        for (auto dt : cfg.devices)
        {
            if (dt == ParityDeviceType::CUDA)
                required_cuda++;
            if (dt == ParityDeviceType::ROCm)
                required_rocm++;
        }

        /*
         * A production campaign's typed backend signature is also its physical
         * resource claim.  Constructing the other vendor's backend merely to
         * discover an irrelevant count initializes that driver and defeats the
         * aggregate scheduler's CUDA/ROCm isolation.  In particular, concurrent
         * homogeneous campaigns can then register/free pinned host allocations
         * through both runtimes.  Probe only a backend the exact cell requires;
         * a true hybrid cell still probes both and therefore retains production
         * topology discovery.
         */
        const int cuda_count =
            required_cuda > 0 ? getCudaDeviceCount() : 0;
        const int rocm_count =
            required_rocm > 0 ? getRocmDeviceCount() : 0;

        if (required_cuda > cuda_count)
            return "Need " + std::to_string(required_cuda) + " CUDA devices, found " + std::to_string(cuda_count);
        if (required_rocm > rocm_count)
            return "Need " + std::to_string(required_rocm) + " ROCm devices, found " + std::to_string(rocm_count);

        return std::nullopt; // Hardware is available
    }

    // =============================================================================
    // Metric Computation Functions
    // =============================================================================

    /**
     * @brief Compute cosine similarity between two vectors
     *
     * Cosine similarity measures directional alignment, ignoring magnitude.
     * Preferred for embedding comparisons because quantization noise affects
     * magnitude but preserves direction.
     *
     * @return Value in [-1, 1], where 1 = identical direction
     */
    /**
     * @brief Compute excess kurtosis of a float array.
     *
     * Excess kurtosis = E[(x-μ)^4] / σ^4 - 3.
     * Normal distribution has excess kurtosis = 0.
     * High kurtosis (>10) indicates heavy-tailed outliers that destroy
     * int8 quantization accuracy.
     */
    inline float computeKurtosis(const float *data, size_t size)
    {
        if (size < 4)
            return 0.0f;

        double sum = 0.0, sum2 = 0.0;
        for (size_t i = 0; i < size; ++i)
        {
            double v = static_cast<double>(data[i]);
            sum += v;
            sum2 += v * v;
        }
        double mean = sum / static_cast<double>(size);
        double var = sum2 / static_cast<double>(size) - mean * mean;
        if (var < 1e-30)
            return 0.0f;

        double sum4 = 0.0;
        for (size_t i = 0; i < size; ++i)
        {
            double d = static_cast<double>(data[i]) - mean;
            double d2 = d * d;
            sum4 += d2 * d2;
        }
        double m4 = sum4 / static_cast<double>(size);
        return static_cast<float>(m4 / (var * var) - 3.0);
    }

    /**
     * @brief Compute distribution statistics for a float tensor
     *
     * Computes min, max, mean, stddev, kurtosis, and percentiles (p95, p99).
     * Uses OpenMP for large tensors.
     */
    inline TensorDistributionStats computeDistributionStats(const float *data, size_t size)
    {
        TensorDistributionStats stats;
        stats.element_count = size;
        if (size == 0)
            return stats;

        // Pass 1: min, max, sum, sum2, nan/inf/zero/sparsity counts
        double sum = 0.0, sum2 = 0.0;
        float vmin = std::numeric_limits<float>::max();
        float vmax = std::numeric_limits<float>::lowest();
        size_t nan_count = 0, inf_count = 0, zero_count = 0, sparse_count = 0;
        constexpr float sparsity_eps = 1e-6f;

#pragma omp parallel for reduction(+ : sum, sum2, nan_count, inf_count, zero_count, sparse_count) \
    reduction(min : vmin) reduction(max : vmax) schedule(static) if (size > 8192)
        for (size_t i = 0; i < size; ++i)
        {
            float v = data[i];
            if (std::isnan(v))
            {
                nan_count++;
                continue;
            }
            if (std::isinf(v))
            {
                inf_count++;
                continue;
            }
            if (v < vmin)
                vmin = v;
            if (v > vmax)
                vmax = v;
            if (v == 0.0f)
                zero_count++;
            if (std::abs(v) < sparsity_eps)
                sparse_count++;
            double d = static_cast<double>(v);
            sum += d;
            sum2 += d * d;
        }
        stats.nan_count = nan_count;
        stats.inf_count = inf_count;
        stats.zero_fraction = static_cast<float>(zero_count) / static_cast<float>(size);
        stats.sparsity = static_cast<float>(sparse_count) / static_cast<float>(size);

        size_t finite_count = size - nan_count - inf_count;
        if (finite_count == 0)
            return stats;

        stats.min = vmin;
        stats.max = vmax;
        double mean = sum / static_cast<double>(finite_count);
        stats.mean = static_cast<float>(mean);
        double var = sum2 / static_cast<double>(finite_count) - mean * mean;
        double sd = std::sqrt(std::max(var, 0.0));
        stats.stddev = static_cast<float>(sd);

        // Pass 2: kurtosis, skewness, outliers
        if (var > 1e-30)
        {
            double sum3 = 0.0, sum4 = 0.0;
            size_t outlier_count = 0;
            double outlier_threshold = 6.0 * sd;
#pragma omp parallel for reduction(+ : sum3, sum4, outlier_count) schedule(static) if (size > 8192)
            for (size_t i = 0; i < size; ++i)
            {
                float v = data[i];
                if (std::isnan(v) || std::isinf(v))
                    continue;
                double d = static_cast<double>(v) - mean;
                double d2 = d * d;
                sum3 += d2 * d;
                sum4 += d2 * d2;
                if (std::abs(d) > outlier_threshold)
                    outlier_count++;
            }
            double n = static_cast<double>(finite_count);
            stats.kurtosis = static_cast<float>(sum4 / (n * var * var) - 3.0);
            stats.skewness = static_cast<float>(sum3 / (n * sd * sd * sd));
            stats.outlier_fraction = static_cast<float>(outlier_count) / static_cast<float>(finite_count);
        }

        // Percentiles via approximate approach: sample if very large
        if (finite_count >= 4)
        {
            // For large tensors, subsample to cap percentile cost
            constexpr size_t MAX_PERCENTILE_SAMPLES = 100000;
            std::vector<float> abs_vals;
            if (finite_count <= MAX_PERCENTILE_SAMPLES)
            {
                abs_vals.reserve(finite_count);
                for (size_t i = 0; i < size; ++i)
                {
                    float v = data[i];
                    if (!std::isnan(v) && !std::isinf(v))
                        abs_vals.push_back(std::abs(v));
                }
            }
            else
            {
                // Deterministic stride-based sampling
                abs_vals.reserve(MAX_PERCENTILE_SAMPLES);
                size_t stride = size / MAX_PERCENTILE_SAMPLES;
                for (size_t i = 0; i < size && abs_vals.size() < MAX_PERCENTILE_SAMPLES; i += stride)
                {
                    float v = data[i];
                    if (!std::isnan(v) && !std::isinf(v))
                        abs_vals.push_back(std::abs(v));
                }
            }

            size_t n = abs_vals.size();
            if (n >= 4)
            {
                size_t p50_idx = n / 2;
                size_t p95_idx = static_cast<size_t>(0.95 * (n - 1));
                size_t p99_idx = static_cast<size_t>(0.99 * (n - 1));

                std::nth_element(abs_vals.begin(), abs_vals.begin() + p99_idx, abs_vals.end());
                stats.p99 = abs_vals[p99_idx];

                std::nth_element(abs_vals.begin(), abs_vals.begin() + p95_idx, abs_vals.begin() + p99_idx);
                stats.p95 = abs_vals[p95_idx];

                std::nth_element(abs_vals.begin(), abs_vals.begin() + p50_idx, abs_vals.begin() + p95_idx);
                float median_abs = abs_vals[p50_idx];

                float max_abs = std::max(std::abs(vmin), std::abs(vmax));
                if (median_abs > 1e-10f && max_abs > 1e-10f)
                    stats.dynamic_range = std::log2(max_abs / median_abs);
            }
        }

        return stats;
    }

    /**
     * @brief Compare one live checkpoint with its real-weight reference tensor.
     *
     * This is the numerical authority shared by the classic fixture and the
     * production MTP/MoE artifact recorders.  It deliberately retains all
     * diagnostics used by the established CSV contract rather than reducing a
     * checkpoint to a token match or a single cosine value.
     */
    inline StageComparisonResult compareParityTensorData(
        const float *actual,
        const float *expected,
        size_t size,
        const std::string &stage_name,
        float cosine_threshold)
    {
        StageComparisonResult result;
        result.stage_name = stage_name;
        result.total_elements = size;
        if (!actual || !expected || size == 0)
            return result;

        double sum_sq_diff = 0.0;
        double sum_sq_expected = 0.0;
        double dot_product = 0.0;
        double norm_actual_sq = 0.0;
        double norm_expected_sq = 0.0;
        float max_abs_diff = 0.0f;
#pragma omp parallel for reduction(+ : sum_sq_diff, sum_sq_expected, dot_product, norm_actual_sq, norm_expected_sq) \
    reduction(max : max_abs_diff) schedule(static) if (size > 8192)
        for (size_t index = 0; index < size; ++index)
        {
            const double actual_value = static_cast<double>(actual[index]);
            const double expected_value = static_cast<double>(expected[index]);
            const double difference = actual_value - expected_value;
            sum_sq_diff += difference * difference;
            sum_sq_expected += expected_value * expected_value;
            dot_product += actual_value * expected_value;
            norm_actual_sq += actual_value * actual_value;
            norm_expected_sq += expected_value * expected_value;
            max_abs_diff = std::max(
                max_abs_diff,
                std::abs(actual[index] - expected[index]));
        }

        result.max_abs_diff = max_abs_diff;
        if (sum_sq_expected > 1.0e-10)
        {
            result.rel_l2_norm = static_cast<float>(
                std::sqrt(sum_sq_diff / sum_sq_expected));
        }
        const double norm_product =
            std::sqrt(norm_actual_sq) * std::sqrt(norm_expected_sq);
        if (norm_product > kParityCosineNormProductResolution)
        {
            result.cosine_similarity = static_cast<float>(
                dot_product / norm_product);
        }
        result.rmse = static_cast<float>(
            std::sqrt(sum_sq_diff / static_cast<double>(size)));
        if (sum_sq_diff > 1.0e-30)
        {
            result.snr_db = static_cast<float>(
                10.0 * std::log10(sum_sq_expected / sum_sq_diff));
        }
        else if (sum_sq_expected > 1.0e-30)
        {
            result.snr_db = 100.0f;
        }

        constexpr int error_bins = 64;
        std::array<size_t, error_bins> histogram{};
        if (max_abs_diff > 1.0e-10f)
        {
            const float scale =
                static_cast<float>(error_bins - 1) / max_abs_diff;
            for (size_t index = 0; index < size; ++index)
            {
                const float error = std::abs(actual[index] - expected[index]);
                const int bin = std::min(
                    static_cast<int>(error * scale), error_bins - 1);
                ++histogram[static_cast<size_t>(bin)];
            }
            double entropy = 0.0;
            for (const size_t count : histogram)
            {
                if (count == 0)
                    continue;
                const double probability =
                    static_cast<double>(count) / static_cast<double>(size);
                entropy -= probability * std::log2(probability);
            }
            result.error_entropy = static_cast<float>(entropy);
        }

        result.llaminar_stats = computeDistributionStats(actual, size);
        result.pytorch_stats = computeDistributionStats(expected, size);
        result.passed = result.cosine_similarity >= cosine_threshold &&
                        result.llaminar_stats.nan_count == 0 &&
                        result.llaminar_stats.inf_count == 0;
        return result;
    }

    inline float computeCosineSimilarity(const float *a, const float *b, size_t size)
    {
        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

#pragma omp parallel for reduction(+ : dot_product, norm_a, norm_b) schedule(static) if (size > 8192)
        for (size_t i = 0; i < size; ++i)
        {
            dot_product += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        double denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denominator <= kParityCosineNormProductResolution)
        {
            return 0.0f;
        }

        return static_cast<float>(dot_product / denominator);
    }

    /**
     * @brief Compute KL divergence between probability distributions
     *
     * KL(P || Q) measures how P diverges from Q.
     * First applies softmax to convert logits to probabilities.
     *
     * @param actual_logits Llaminar logits (unnormalized)
     * @param expected_logits PyTorch logits (unnormalized)
     * @param size Total elements (seq_len * vocab_size)
     * @param vocab_size Vocabulary size for per-position softmax
     * @return Average KL divergence per position (in nats)
     */
    inline float computeKLDivergence(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size)
    {
        size_t seq_len = size / vocab_size;
        double total_kl = 0.0;

#pragma omp parallel for reduction(+ : total_kl) schedule(static) if (seq_len > 1)
        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            // Find max for numerical stability (log-sum-exp trick)
            float max_actual = actual_row[0];
            float max_expected = expected_row[0];
            for (size_t i = 1; i < vocab_size; ++i)
            {
                max_actual = std::max(max_actual, actual_row[i]);
                max_expected = std::max(max_expected, expected_row[i]);
            }

            // Compute softmax denominators
            double sum_exp_actual = 0.0;
            double sum_exp_expected = 0.0;
            for (size_t i = 0; i < vocab_size; ++i)
            {
                sum_exp_actual += std::exp(static_cast<double>(actual_row[i] - max_actual));
                sum_exp_expected += std::exp(static_cast<double>(expected_row[i] - max_expected));
            }
            double log_sum_actual = max_actual + std::log(sum_exp_actual);
            double log_sum_expected = max_expected + std::log(sum_exp_expected);

            // KL divergence: KL(expected || actual)
            double pos_kl = 0.0;
            for (size_t i = 0; i < vocab_size; ++i)
            {
                double log_p = expected_row[i] - log_sum_expected;
                double log_q = actual_row[i] - log_sum_actual;
                double p = std::exp(log_p);

                if (p > 1e-10)
                {
                    pos_kl += p * (log_p - log_q);
                }
            }
            total_kl += pos_kl;
        }

        return static_cast<float>(total_kl / seq_len);
    }

    /**
     * @brief Compute Top-K overlap between two sets of logits
     *
     * Checks if the top K tokens predicted by both models overlap.
     * This is a "smoke test" for decision quality.
     *
     * @return Overlap percentage in [0, 1]
     */
    inline float computeTopKOverlap(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size,
        int k)
    {
        size_t seq_len = size / vocab_size;
        double total_overlap = 0.0;

#pragma omp parallel for reduction(+ : total_overlap) schedule(static) if (seq_len > 1)
        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            // Use a min-heap of size K instead of allocating vocab_size pairs
            auto get_top_k = [&](const float *logits)
            {
                // Min-heap: smallest of top-K on top, so we can eject it when we find larger
                std::vector<std::pair<float, int>> heap;
                heap.reserve(k + 1);
                for (size_t i = 0; i < vocab_size; ++i)
                {
                    if (static_cast<int>(heap.size()) < k)
                    {
                        heap.push_back({logits[i], static_cast<int>(i)});
                        if (static_cast<int>(heap.size()) == k)
                            std::make_heap(heap.begin(), heap.end(),
                                           [](const auto &a, const auto &b)
                                           { return a.first > b.first; });
                    }
                    else if (logits[i] > heap.front().first)
                    {
                        std::pop_heap(heap.begin(), heap.end(),
                                      [](const auto &a, const auto &b)
                                      { return a.first > b.first; });
                        heap.back() = {logits[i], static_cast<int>(i)};
                        std::push_heap(heap.begin(), heap.end(),
                                       [](const auto &a, const auto &b)
                                       { return a.first > b.first; });
                    }
                }
                std::vector<int> indices(heap.size());
                for (size_t i = 0; i < heap.size(); ++i)
                    indices[i] = heap[i].second;
                std::sort(indices.begin(), indices.end());
                return indices;
            };

            auto actual_topk = get_top_k(actual_row);
            auto expected_topk = get_top_k(expected_row);

            std::vector<int> intersection;
            std::set_intersection(
                actual_topk.begin(), actual_topk.end(),
                expected_topk.begin(), expected_topk.end(),
                std::back_inserter(intersection));

            total_overlap += static_cast<double>(intersection.size()) / k;
        }

        return static_cast<float>(total_overlap / seq_len);
    }

    /**
     * @brief Check if PyTorch's top-1 token appears in llaminar's top-K
     *
     * For a single logit row: finds the argmax token in expected_logits (PyTorch)
     * and checks if that token is among the top-K tokens in actual_logits (llaminar).
     *
     * For multiple rows (seq_len > 1): returns the fraction of positions where
     * the check passes.
     *
     * This is a clearer gate than Top-K overlap: it answers "does llaminar consider
     * the correct answer to be a plausible choice?"
     *
     * @param actual_logits Llaminar logits
     * @param expected_logits PyTorch reference logits
     * @param size Total float count (seq_len * vocab_size)
     * @param vocab_size Vocabulary size
     * @param k Top-K to search in llaminar (e.g., 3)
     * @return Fraction of positions where PyTorch's top-1 is in llaminar's top-K [0, 1]
     */
    inline float pytorchTop1InLlaminarTopK(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size,
        int k)
    {
        size_t seq_len = size / vocab_size;
        if (seq_len == 0 || vocab_size == 0)
            return 0.0f;

        int hits = 0;
#pragma omp parallel for reduction(+ : hits) schedule(static) if (seq_len > 1)
        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            // Find PyTorch's argmax
            int pytorch_argmax = 0;
            float pytorch_max = expected_row[0];
            for (size_t i = 1; i < vocab_size; ++i)
            {
                if (expected_row[i] > pytorch_max)
                {
                    pytorch_max = expected_row[i];
                    pytorch_argmax = static_cast<int>(i);
                }
            }

            // Find llaminar's top-K tokens using a min-heap of size K
            std::vector<std::pair<float, int>> heap;
            heap.reserve(k + 1);
            for (size_t i = 0; i < vocab_size; ++i)
            {
                if (static_cast<int>(heap.size()) < k)
                {
                    heap.push_back({actual_row[i], static_cast<int>(i)});
                    if (static_cast<int>(heap.size()) == k)
                        std::make_heap(heap.begin(), heap.end(),
                                       [](const auto &a, const auto &b)
                                       { return a.first > b.first; });
                }
                else if (actual_row[i] > heap.front().first)
                {
                    std::pop_heap(heap.begin(), heap.end(),
                                  [](const auto &a, const auto &b)
                                  { return a.first > b.first; });
                    heap.back() = {actual_row[i], static_cast<int>(i)};
                    std::push_heap(heap.begin(), heap.end(),
                                   [](const auto &a, const auto &b)
                                   { return a.first > b.first; });
                }
            }

            // Check if PyTorch's argmax is in llaminar's top-K
            for (size_t i = 0; i < heap.size(); ++i)
            {
                if (heap[i].second == pytorch_argmax)
                {
                    hits++;
                    break;
                }
            }
        }

        return static_cast<float>(hits) / seq_len;
    }

    // =============================================================================
    // Table Rendering
    // =============================================================================

    /**
     * @brief Render a formatted parity results table to stdout using libfort
     *
     * Produces a consistent Unicode box-drawing table showing:
     * - Per-layer cosine similarity (avg and min)
     * - Worst stage per layer
     * - Pass/fail status with checkmarks
     * - LM_HEAD KL divergence and Top-K accuracy
     *
     * Uses libfort for clean, automatic column sizing and Unicode borders.
     */
    inline void renderParityTable(
        const ParityTestSummary &summary,
        const ParityConfig &config,
        const std::string &backend_name)
    {
        // Helper: format float to string with precision
        auto fmt_f6 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << v;
            return ss.str();
        };

        auto fmt_f4 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            return ss.str();
        };

        auto fmt_f1 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << v;
            return ss.str();
        };

        // Helper: status icon
        auto status_str = [](bool passed) -> std::string
        {
            return passed ? "✓" : "✗";
        };

        std::cout << "\n";

        // =========================================================================
        // Title table
        // =========================================================================
        {
            fort::utf8_table title_table;
            title_table.set_border_style(FT_DOUBLE2_STYLE);

            std::ostringstream title_ss;
            title_ss << backend_name << " vs PyTorch LAYER-BY-LAYER PARITY";

            std::ostringstream subtitle_ss;
            subtitle_ss << "Threshold: " << (config.use_avg_cosine ? "avg" : "min")
                        << " cosine similarity >= " << std::fixed << std::setprecision(3)
                        << config.cosine_threshold;

            title_table << title_ss.str() << fort::endr;
            title_table << subtitle_ss.str() << fort::endr;

            title_table[0][0].set_cell_text_align(fort::text_align::center);
            title_table[1][0].set_cell_text_align(fort::text_align::center);
            title_table.row(0).set_cell_row_type(fort::row_type::header);

            std::cout << title_table.to_string();
        }

        // =========================================================================
        // Main parity table
        // =========================================================================
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Header row
        table << fort::header
              << "Layer" << "Avg Cosine" << "Min Cosine" << "Worst Stage"
              << "Max Drop" << "Drop Stage" << "Kurtosis" << "OK"
              << fort::endr;

        // Set column alignments
        table.column(0).set_cell_text_align(fort::text_align::center);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::left);
        table.column(4).set_cell_text_align(fort::text_align::right);
        table.column(5).set_cell_text_align(fort::text_align::left);
        table.column(6).set_cell_text_align(fort::text_align::right);
        table.column(7).set_cell_text_align(fort::text_align::center);

        // Embedding row. PP non-head ranks do not manufacture a zero row.
        if (summary.embedding_passed || summary.embedding_cosine != 0.0f)
        {
            table << "EMBEDDING"
                  << fmt_f6(summary.embedding_cosine)
                  << fmt_f6(summary.embedding_cosine)
                  << "-"
                  << "-"
                  << "-"
                  << "-"
                  << status_str(summary.embedding_passed)
                  << fort::endr;
        }

        // Per-layer rows
        for (const auto &stats : summary.layer_stats)
        {
            if (stats.stages_compared == 0 && stats.stage_results.empty())
                continue;
            std::ostringstream layer_ss;
            layer_ss << "Layer " << stats.layer_idx;

            std::string kurtosis_str;
            if (stats.max_kurtosis > 0.0f)
            {
                std::ostringstream ks;
                ks << std::fixed << std::setprecision(0) << stats.max_kurtosis;
                kurtosis_str = ks.str();
            }
            else
            {
                kurtosis_str = "-";
            }

            std::string drop_str = (stats.max_cosine_drop > 0.001f)
                                       ? fmt_f6(stats.max_cosine_drop)
                                       : "-";
            std::string drop_stage = stats.max_drop_stage.empty() ? "-" : stats.max_drop_stage;

            table << layer_ss.str()
                  << fmt_f6(stats.avg_cosine_sim)
                  << fmt_f6(stats.min_cosine_sim)
                  << stats.worst_stage
                  << drop_str
                  << drop_stage
                  << kurtosis_str
                  << status_str(stats.passed)
                  << fort::endr;
        }

        const bool has_lm_head_data =
            summary.lm_head_passed || summary.lm_head_cosine != 0.0f ||
            summary.lm_head_kl != 0.0f || summary.lm_head_top1 != 0.0f ||
            summary.lm_head_top5 != 0.0f;
        if (has_lm_head_data)
        {
            // Separator before LM_HEAD
            table << fort::separator;

            // LM_HEAD row with extra info
            std::ostringstream lm_info;
            lm_info << "KL=" << fmt_f4(summary.lm_head_kl)
                    << " Top1=" << fmt_f1(summary.lm_head_top1 * 100.0f) << "%";

            table << "LM_HEAD"
                  << fmt_f6(summary.lm_head_cosine)
                  << fmt_f6(summary.lm_head_cosine)
                  << lm_info.str()
                  << "-"
                  << "-"
                  << "-"
                  << status_str(summary.lm_head_passed)
                  << fort::endr;
        }

        std::cout << table.to_string();

        // =========================================================================
        // Summary footer
        // =========================================================================
        if (has_lm_head_data)
        {
            std::cout << "\nLM_HEAD Top-5: " << std::fixed
                      << std::setprecision(1)
                      << (summary.lm_head_top5 * 100.0f) << "%\n";
        }
        std::cout << "Early layers passed: " << summary.early_layers_passed
                  << "/" << config.early_layers_count << "\n";
        if (has_lm_head_data)
        {
            std::cout << "LM_HEAD KL divergence: " << std::fixed
                      << std::setprecision(4) << summary.lm_head_kl
                      << " (threshold: " << config.kl_threshold << ")\n";
        }
    }

    /**
     * @brief Render a TP-aware parity results table to stdout using libfort
     *
     * For tensor-parallel tests, shows:
     * - Per-device cosine similarity against PyTorch slices
     * - Combined (concatenated) result vs full PyTorch
     * - Sharding mode for each stage
     *
     * Supports an arbitrary number of TP devices with dynamic column sizing.
     * Uses libfort for clean, Unicode box-drawing table formatting.
     */
    inline void renderTPParityTable(
        const TPParityTestSummary &summary,
        const ParityConfig &config,
        const std::string &backend_name)
    {
        const size_t num_devices = summary.device_names.size();

        // Helper: sharding mode to string
        auto mode_str = [](SnapshotShardingMode mode) -> std::string
        {
            switch (mode)
            {
            case SnapshotShardingMode::REPLICATED:
                return "R";
            case SnapshotShardingMode::COLUMN_PARALLEL:
                return "C";
            case SnapshotShardingMode::PACKED_COLUMN_PARALLEL:
                return "P";
            case SnapshotShardingMode::ROW_PARALLEL:
                return "W";
            case SnapshotShardingMode::ROOT_ONLY:
                return "O";
            case SnapshotShardingMode::GATHERED:
                return "G";
            default:
                return "?";
            }
        };

        // Helper: status icon
        auto status_str = [](bool passed) -> std::string
        {
            return passed ? "✓" : "✗";
        };

        // Helper: format float to string
        auto fmt_f6 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << v;
            return ss.str();
        };

        // =========================================================================
        // Title table (separate table for clean title block)
        // =========================================================================
        std::cout << "\n";
        {
            fort::utf8_table title_table;
            title_table.set_border_style(FT_DOUBLE2_STYLE);

            std::ostringstream title_ss;
            title_ss << backend_name << " vs PyTorch TP-AWARE PARITY";

            std::ostringstream subtitle_ss;
            subtitle_ss << summary.tp_degree << "-way LOCAL TP, Threshold: cosine >= "
                        << std::fixed << std::setprecision(3) << config.cosine_threshold;

            title_table << title_ss.str() << fort::endr;
            title_table << subtitle_ss.str() << fort::endr;

            title_table[0][0].set_cell_text_align(fort::text_align::center);
            title_table[1][0].set_cell_text_align(fort::text_align::center);
            title_table.row(0).set_cell_row_type(fort::row_type::header);

            std::cout << title_table.to_string();
        }

        // =========================================================================
        // Main parity table
        // =========================================================================
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Build header row: Stage | Dev0 | Dev1 | ... | Combined | Mode | Status
        table << fort::header << "Stage";
        for (size_t i = 0; i < num_devices; ++i)
        {
            table << summary.device_names[i];
        }
        table << "Combined" << "Mode" << "OK" << fort::endr;

        // Set center alignment for all columns
        for (size_t col = 0; col < num_devices + 4; ++col)
        {
            table.column(col).set_cell_text_align(fort::text_align::center);
        }

        // Embedding row
        {
            const auto &emb = summary.embedding_result;
            table << "EMBEDDING";
            for (size_t i = 0; i < num_devices; ++i)
            {
                if (i < emb.device_results.size())
                {
                    table << fmt_f6(emb.device_results[i].cosine_similarity);
                }
                else
                {
                    table << "-";
                }
            }
            table << fmt_f6(emb.combined_cosine)
                  << mode_str(emb.sharding_mode)
                  << status_str(emb.passed)
                  << fort::endr;
        }

        // Per-layer rows
        for (const auto &layer : summary.layer_stats)
        {
            // Layer header row
            std::ostringstream layer_ss;
            layer_ss << "Layer " << layer.layer_idx;

            table << fort::separator;
            table << layer_ss.str();
            for (size_t i = 0; i < num_devices; ++i)
            {
                table << "";
            }
            table << fmt_f6(layer.avg_combined_cosine)
                  << ""
                  << status_str(layer.passed)
                  << fort::endr;

            // Per-stage details (limit to 6 for readability)
            int stage_count = 0;
            for (const auto &stage : layer.stage_results)
            {
                if (stage_count++ >= 6)
                    break;

                // Truncate stage name if needed
                std::string stage_name = stage.stage_name;
                if (stage_name.size() > 14)
                {
                    stage_name = stage_name.substr(0, 12) + "..";
                }

                table << stage_name;
                for (size_t i = 0; i < num_devices; ++i)
                {
                    if (i < stage.device_results.size())
                    {
                        table << fmt_f6(stage.device_results[i].cosine_similarity);
                    }
                    else
                    {
                        table << "-";
                    }
                }
                table << fmt_f6(stage.combined_cosine)
                      << mode_str(stage.sharding_mode)
                      << status_str(stage.passed)
                      << fort::endr;
            }
        }

        // LM_HEAD row
        table << fort::separator;
        table << "LM_HEAD";
        for (size_t i = 0; i < num_devices; ++i)
        {
            table << "[gathered]";
        }
        table << fmt_f6(summary.lm_head_cosine)
              << "G"
              << status_str(summary.lm_head_passed)
              << fort::endr;

        std::cout << table.to_string();

        // =========================================================================
        // Summary footer
        // =========================================================================
        std::cout << "\nSharding modes: R=Replicated, C=Column-parallel, "
                     "P=Packed-column, W=roW-parallel, O=Root-only, G=Gathered\n";
        std::cout << std::fixed << std::setprecision(4)
                  << "LM_HEAD: KL=" << summary.lm_head_kl
                  << std::setprecision(1)
                  << " Top-1=" << (summary.lm_head_top1 * 100.0f) << "%"
                  << " Top-5=" << (summary.lm_head_top5 * 100.0f) << "%\n";
        std::cout << "Early layers passed: " << summary.early_layers_passed
                  << "/" << config.early_layers_count << "\n";
    }

    // =============================================================================
    // Base Test Class
    // =============================================================================

    /**
     * @brief Base class for PyTorch parity tests
     *
     * Provides common infrastructure for comparing Llaminar backends against
     * PyTorch ground truth. Subclasses must implement:
     * - getDevice() OR getDeviceForRank() - Return the DeviceId to use for inference
     * - getBackendName() - Return a display name (e.g., "CUDA", "CPU", "ROCm")
     *
     * For MPI/tensor-parallel tests:
     * - Set mpi_ctx_ in SetUp() before calling ParityTestBase::SetUp()
     * - Override getDeviceForRank() instead of getDevice()
     *
     * Optional overrides:
     * - setupDeviceSpecific() - Device-specific initialization (e.g., CUDA checks)
     */
    class ParityTestBase : public ::testing::Test
    {
    private:
        // Static set of snapshot generations completed this run.
        // This prevents regenerating snapshots for every test case in a suite.
        // Key: snapshot_dir|model_path
        static inline std::set<std::string> s_generated_snapshots_;
        static inline std::mutex s_snapshot_mutex_;

        static std::string sanitizeSnapshotToken(const std::string &input)
        {
            std::string token;
            token.reserve(input.size());

            bool last_was_underscore = false;
            for (unsigned char ch : input)
            {
                if (std::isalnum(ch))
                {
                    token.push_back(static_cast<char>(std::tolower(ch)));
                    last_was_underscore = false;
                }
                else if (!last_was_underscore)
                {
                    token.push_back('_');
                    last_was_underscore = true;
                }
            }

            while (!token.empty() && token.front() == '_')
                token.erase(token.begin());
            while (!token.empty() && token.back() == '_')
                token.pop_back();

            return token.empty() ? "model" : token;
        }

        static std::string inferSnapshotDirFromModelPath(const std::string &model_path)
        {
            std::string filename = model_path;
            const size_t slash = filename.find_last_of("/\\");
            if (slash != std::string::npos)
                filename = filename.substr(slash + 1);

            const size_t dot = filename.rfind('.');
            if (dot != std::string::npos)
                filename = filename.substr(0, dot);

            return "pytorch_" + sanitizeSnapshotToken(filename) + "_snapshots";
        }

        void resolveSnapshotDirIfNeeded()
        {
            if (config_.snapshot_dir.empty())
            {
                config_.snapshot_dir = inferSnapshotDirFromModelPath(config_.model_path);
            }
        }

        struct SavedEnvValue
        {
            std::string name;
            std::optional<std::string> previous;
        };

        struct ParityProfileRecord
        {
            size_t count = 0;
            double total_ms = 0.0;
            double max_ms = 0.0;
        };

        std::string snapshotCacheKey() const
        {
            // Prompt and decode depth are part of the reference-pack identity.
            // Without them, a parameterized process can incorrectly bless the
            // first pack for a later case that reuses the same directory.
            return config_.snapshot_dir + "|" + config_.model_path +
                   "|prompt=" + config_.prompt +
                   "|decode_steps=" + std::to_string(config_.decode_steps);
        }

        void setParityEnvOverride(const char *name, const std::string &value)
        {
            const auto already_saved = std::find_if(
                parity_env_overrides_.begin(),
                parity_env_overrides_.end(),
                [name](const SavedEnvValue &entry)
                {
                    return entry.name == name;
                });
            if (already_saved == parity_env_overrides_.end())
            {
                const char *old_value = std::getenv(name);
                parity_env_overrides_.push_back(
                    SavedEnvValue{
                        name,
                        old_value ? std::optional<std::string>(old_value)
                                  : std::nullopt});
            }
            setenv(name, value.c_str(), 1);
        }

        void restoreParityEnvOverrides()
        {
            for (auto it = parity_env_overrides_.rbegin();
                 it != parity_env_overrides_.rend();
                 ++it)
            {
                if (it->previous)
                    setenv(it->name.c_str(), it->previous->c_str(), 1);
                else
                    unsetenv(it->name.c_str());
            }
            parity_env_overrides_.clear();
            mutableDebugEnv().reload();
        }

    protected:
        class ParityProfileScope
        {
        public:
            ParityProfileScope(ParityTestBase *owner, std::string phase)
                : owner_(owner),
                  phase_(std::move(phase)),
                  enabled_(owner_ && owner_->parityProfileEnabled()),
                  start_(std::chrono::steady_clock::now())
            {
            }

            ParityProfileScope(const ParityProfileScope &) = delete;
            ParityProfileScope &operator=(const ParityProfileScope &) = delete;

            ParityProfileScope(ParityProfileScope &&other) noexcept
                : owner_(other.owner_),
                  phase_(std::move(other.phase_)),
                  enabled_(other.enabled_),
                  start_(other.start_)
            {
                other.enabled_ = false;
                other.owner_ = nullptr;
            }

            ~ParityProfileScope()
            {
                if (!enabled_ || !owner_)
                    return;
                const auto elapsed = std::chrono::steady_clock::now() - start_;
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(elapsed).count();
                owner_->recordParityProfile(phase_, elapsed_ms);
            }

        private:
            ParityTestBase *owner_ = nullptr;
            std::string phase_;
            bool enabled_ = false;
            std::chrono::steady_clock::time_point start_;
        };

        bool parityProfileEnabled() const
        {
            return isRank0() && DebugEnv::isTruthyEnv("LLAMINAR_PARITY_PROFILE");
        }

        ParityProfileScope profileParityScope(std::string phase)
        {
            return ParityProfileScope(this, std::move(phase));
        }

        void recordParityProfile(const std::string &phase, double elapsed_ms)
        {
            auto &record = parity_profile_records_[phase];
            record.count += 1;
            record.total_ms += elapsed_ms;
            record.max_ms = std::max(record.max_ms, elapsed_ms);

            if (DebugEnv::isTruthyEnv("LLAMINAR_PARITY_PROFILE_VERBOSE"))
            {
                std::cerr << "[parity-profile] case=" << cfg().name
                          << " backend=" << getBackendName()
                          << " phase=" << phase
                          << " elapsed_ms=" << std::fixed << std::setprecision(3)
                          << elapsed_ms << '\n';
            }
        }

        void printParityProfileSummary()
        {
            if (!parityProfileEnabled() || parity_profile_records_.empty())
                return;

            std::vector<std::pair<std::string, ParityProfileRecord>> rows(
                parity_profile_records_.begin(),
                parity_profile_records_.end());
            std::sort(
                rows.begin(),
                rows.end(),
                [](const auto &lhs, const auto &rhs)
                {
                    if (lhs.second.total_ms != rhs.second.total_ms)
                        return lhs.second.total_ms > rhs.second.total_ms;
                    return lhs.first < rhs.first;
                });

            for (const auto &[phase, record] : rows)
            {
                const double avg_ms = record.count == 0
                                          ? 0.0
                                          : record.total_ms / static_cast<double>(record.count);
                std::cerr << "[parity-profile-summary] case=" << cfg().name
                          << " backend=" << getBackendName()
                          << " phase=" << phase
                          << " count=" << record.count
                          << " total_ms=" << std::fixed << std::setprecision(3)
                          << record.total_ms
                          << " avg_ms=" << avg_ms
                          << " max_ms=" << record.max_ms << '\n';
            }
        }

        void setScopedParityEnvOverride(const char *name, const std::string &value)
        {
            setParityEnvOverride(name, value);
            mutableDebugEnv().reload();
        }

        /**
         * @brief Capture production parity at the authenticated prompt geometry.
         *
         * The serving default pads small prompts to a 256-row graph bucket for
         * throughput reuse. Stage-by-stage parity materializes every published
         * row, so comparing the oracle's nine real rows plus 247 known padding
         * rows multiplies diagnostic transfer and CSV work without adding a
         * mathematical input. Production campaigns instead declare one exact
         * graph bucket from authenticated `token_ids`, plus the one-row bucket
         * required by the mandatory partial-prefix suffix. Both graphs are
         * fully captured/replayed and their geometry remains part of cache
         * identity; this method never permits eager execution or recapture.
         */
        void configureExactProductionParityPrefillGraphBucket()
        {
            if (!productionParityCampaignEnabled())
                return;
            ASSERT_FALSE(config_.token_ids.empty())
                << "Production parity requires authenticated prompt tokens "
                   "before graph-bucket configuration";
            if (config_.token_ids.empty())
                return;

            const std::string exact_bucket =
                std::to_string(config_.token_ids.size());
            setScopedParityEnvOverride(
                "LLAMINAR_PREFILL_GRAPH_MIN_SEQ",
                "1");
            setScopedParityEnvOverride(
                "LLAMINAR_PREFILL_GRAPH_BUCKETS",
                "1");
            setScopedParityEnvOverride(
                "LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES",
                config_.token_ids.size() == 1u
                    ? exact_bucket
                    : "1," + exact_bucket);
        }

        /**
         * @brief Deliberately force a bounded captured-prefill campaign shape.
         *
         * Ordinary production parity uses the authenticated prompt length as
         * one exact bucket because that is the cheapest complete checkpoint
         * comparison. A named segmented-prefill campaign must prove the other
         * production contract: the same real request is divided into several
         * fixed captured buckets and its live rows are joined back into the
         * reference-sized checkpoints. This helper is explicit so no normal
         * parity cell can accidentally become segmented through ambient shell
         * configuration.
         *
         * @param bucket_rows Positive fixed physical bucket width, smaller than
         *        the authenticated prompt for the caller's segmented test.
         */
        void configureSegmentedProductionParityPrefillGraphBucket(
            int bucket_rows)
        {
            ASSERT_TRUE(productionParityCampaignEnabled())
                << "Segmented parity buckets are only valid in a production "
                   "campaign";
            ASSERT_GT(bucket_rows, 0)
                << "Segmented production parity requires a positive bucket";
            ASSERT_FALSE(config_.token_ids.empty())
                << "Segmented production parity requires authenticated prompt "
                   "tokens before graph-bucket configuration";
            if (!productionParityCampaignEnabled() || bucket_rows <= 0 ||
                config_.token_ids.empty())
            {
                return;
            }

            setScopedParityEnvOverride(
                "LLAMINAR_PREFILL_GRAPH_MIN_SEQ",
                std::to_string(bucket_rows));
            setScopedParityEnvOverride(
                "LLAMINAR_PREFILL_GRAPH_BUCKETS",
                "1");
            setScopedParityEnvOverride(
                "LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES",
                std::to_string(bucket_rows));
        }

        /**
         * @brief Required snapshot version for compatibility.
         *
         * Bump this when the snapshot format or V-head reversal semantics change.
         * Snapshots with a lower version will be automatically regenerated.
         *   v1: original format (V-head reversal applied to all models)
         *   v2: MoE-only V-head reversal (dense models skip reversal)
         *   v3: Qwen3.5 prefill GDN conv and Q/K norm snapshots match C++ layout
         */
        static constexpr int kRequiredSnapshotVersion = 3;

        static constexpr int kRequiredReferenceIdentityVersion = 1;

        struct ReferenceSnapshotValidation
        {
            bool usable = false;
            std::string reason;
        };

        /**
         * @brief Validate model-family checkpoint semantics beyond provenance.
         *
         * The common identity authenticates model bytes and inference inputs,
         * but a model-specific snapshot key can still change meaning without a
         * file-format change. Specialized parity bases override this hook and
         * require their own semantic schema marker, preventing an authenticated
         * but semantically stale reference pack from entering comparison.
         */
        virtual ReferenceSnapshotValidation
        validateModelSpecificReferenceSnapshotMetadata(
            const std::filesystem::path &metadata_path) const
        {
            (void)metadata_path;
            return {true, {}};
        }

        /**
         * @brief Read snapshot_version from metadata.txt
         * @return version number, or 0 if not found (pre-versioning snapshots)
         */
        static int readSnapshotVersion(const std::filesystem::path &metadata_path)
        {
            std::ifstream f(metadata_path);
            if (!f.is_open())
                return 0;
            std::string line;
            while (std::getline(f, line))
            {
                if (line.rfind("snapshot_version:", 0) == 0)
                {
                    try
                    {
                        return std::stoi(line.substr(17));
                    }
                    catch (...)
                    {
                        return 0;
                    }
                }
            }
            return 0; // Pre-versioning snapshot (no version line)
        }

        /**
         * @brief Read and trim one scalar from line-oriented snapshot metadata.
         */
        static std::optional<std::string> readSnapshotMetadataValue(
            const std::filesystem::path &metadata_path,
            const std::string &key)
        {
            std::ifstream file(metadata_path);
            if (!file.is_open())
                return std::nullopt;

            const std::string prefix = key + ":";
            std::string line;
            while (std::getline(file, line))
            {
                if (line.rfind(prefix, 0) != 0)
                    continue;
                std::string value = line.substr(prefix.size());
                const auto first = value.find_first_not_of(" \t\r\n");
                if (first == std::string::npos)
                    return std::string{};
                const auto last = value.find_last_not_of(" \t\r\n");
                return value.substr(first, last - first + 1);
            }
            return std::nullopt;
        }

        /**
         * @brief Parse a strict comma-separated token field and canonical form.
         */
        static std::optional<std::pair<std::vector<int>, std::string>>
        readSnapshotTokenIdentity(
            const std::filesystem::path &metadata_path,
            const std::string &key)
        {
            const auto raw = readSnapshotMetadataValue(metadata_path, key);
            if (!raw || raw->empty())
                return std::nullopt;

            std::vector<int> tokens;
            std::stringstream input(*raw);
            std::string item;
            while (std::getline(input, item, ','))
            {
                const auto first = item.find_first_not_of(" \t");
                const auto last = item.find_last_not_of(" \t");
                if (first == std::string::npos || last == std::string::npos)
                    return std::nullopt;
                const std::string trimmed = item.substr(first, last - first + 1);
                size_t parsed = 0;
                try
                {
                    const int token = std::stoi(trimmed, &parsed);
                    if (parsed != trimmed.size() || token < 0)
                        return std::nullopt;
                    tokens.push_back(token);
                }
                catch (...)
                {
                    return std::nullopt;
                }
            }
            if (tokens.empty())
                return std::nullopt;

            std::ostringstream canonical;
            for (size_t index = 0; index < tokens.size(); ++index)
            {
                if (index != 0)
                    canonical << ',';
                canonical << tokens[index];
            }
            return std::make_pair(std::move(tokens), canonical.str());
        }

        static bool isSha256Hex(const std::string &value)
        {
            return value.size() == 64 &&
                   std::all_of(
                       value.begin(),
                       value.end(),
                       [](unsigned char character)
                       {
                           return (character >= '0' && character <= '9') ||
                                  (character >= 'a' && character <= 'f');
                       });
        }

        /**
         * @brief Authenticate a reusable CPU-reference pack for this exact case.
         *
         * Focused diagnostic tests retain version-only compatibility. Production
         * campaigns additionally prove that the reference came from the exact
         * GGUF bytes, exact prompt bytes, CPU/FP32 PyTorch execution, tokenizer
         * output, and sufficient incremental-decode depth.
         */
        ReferenceSnapshotValidation validateReferenceSnapshotMetadata(
            const std::filesystem::path &metadata_path,
            bool require_content_identity,
            bool require_model_specific_extensions = true) const
        {
            const int snapshot_version = readSnapshotVersion(metadata_path);
            if (snapshot_version < kRequiredSnapshotVersion)
            {
                return {
                    false,
                    "snapshot_version " + std::to_string(snapshot_version) +
                        " is older than required v" +
                        std::to_string(kRequiredSnapshotVersion)};
            }
            if (require_model_specific_extensions)
            {
                const auto model_specific_validation =
                    validateModelSpecificReferenceSnapshotMetadata(metadata_path);
                if (!model_specific_validation.usable)
                    return model_specific_validation;
            }
            if (!require_content_identity)
                return {true, {}};

            const auto expect_scalar = [&metadata_path](
                                           const std::string &key,
                                           const std::string &expected)
                -> std::optional<std::string>
            {
                const auto actual = readSnapshotMetadataValue(metadata_path, key);
                if (!actual)
                    return "missing " + key;
                if (*actual != expected)
                    return key + " is '" + *actual + "', expected '" + expected + "'";
                return std::nullopt;
            };

            if (const auto error = expect_scalar(
                    "reference_identity_version",
                    std::to_string(kRequiredReferenceIdentityVersion)))
                return {false, *error};
            if (const auto error = expect_scalar("reference_engine", "pytorch"))
                return {false, *error};
            if (const auto error = expect_scalar("reference_device", "cpu"))
                return {false, *error};
            if (const auto error = expect_scalar("reference_dtype", "float32"))
                return {false, *error};

            const auto decode_depth = readSnapshotMetadataValue(
                metadata_path, "decode_steps");
            if (!decode_depth)
                return {false, "missing decode_steps"};
            try
            {
                size_t parsed = 0;
                const int available = std::stoi(*decode_depth, &parsed);
                if (parsed != decode_depth->size() ||
                    available < config_.decode_steps)
                {
                    return {
                        false,
                        "decode_steps does not cover requested depth " +
                            std::to_string(config_.decode_steps)};
                }
            }
            catch (...)
            {
                return {false, "decode_steps is not an integer"};
            }

            const auto token_identity = readSnapshotTokenIdentity(
                metadata_path, "token_ids");
            if (!token_identity)
                return {false, "token_ids is missing or malformed"};
            const auto token_digest = readSnapshotMetadataValue(
                metadata_path, "token_ids_sha256");
            std::string digest_error;
            const auto expected_token_digest = sha256BytesHex(
                token_identity->second, &digest_error);
            if (!token_digest || !isSha256Hex(*token_digest) ||
                !expected_token_digest || *token_digest != *expected_token_digest)
            {
                return {
                    false,
                    "token_ids_sha256 does not authenticate token_ids" +
                        (digest_error.empty() ? std::string{} : ": " + digest_error)};
            }

            const auto prompt_digest = readSnapshotMetadataValue(
                metadata_path, "prompt_sha256");
            const auto expected_prompt_digest = sha256BytesHex(
                config_.prompt, &digest_error);
            if (!prompt_digest || !isSha256Hex(*prompt_digest) ||
                !expected_prompt_digest || *prompt_digest != *expected_prompt_digest)
            {
                return {
                    false,
                    "prompt_sha256 does not match the configured prompt" +
                        (digest_error.empty() ? std::string{} : ": " + digest_error)};
            }

            const auto model_digest = readSnapshotMetadataValue(
                metadata_path, "model_sha256");
            const auto digest_cache =
                productionParityDigestCacheFromEnvironment();
            const auto expected_model_digest = digest_cache
                                                   ? sha256FileHexShared(
                                                         config_.model_path,
                                                         *digest_cache,
                                                         &digest_error)
                                                   : sha256FileHex(
                                                         config_.model_path,
                                                         &digest_error);
            if (!model_digest || !isSha256Hex(*model_digest) ||
                !expected_model_digest || *model_digest != *expected_model_digest)
            {
                return {
                    false,
                    "model_sha256 does not match the live GGUF bytes" +
                        (digest_error.empty() ? std::string{} : ": " + digest_error)};
            }

            const auto snapshot_dir = metadata_path.parent_path();
            if (productionParityRequiresPrefillSnapshots() &&
                (!std::filesystem::is_regular_file(snapshot_dir / "EMBEDDING.npy") ||
                 !std::filesystem::is_regular_file(snapshot_dir / "LM_HEAD.npy")))
            {
                return {false, "reference pack is missing prefill boundary snapshots"};
            }
            if (config_.decode_steps > 0)
            {
                const auto decode_tokens = readSnapshotTokenIdentity(
                    metadata_path, "decode_tokens");
                if (!decode_tokens ||
                    decode_tokens->first.size() <
                        static_cast<size_t>(config_.decode_steps))
                {
                    return {false, "decode_tokens does not cover requested depth"};
                }
                const auto last_decode =
                    snapshot_dir /
                    ("decode_step" + std::to_string(config_.decode_steps - 1) +
                     "_LM_HEAD.npy");
                if (!std::filesystem::is_regular_file(last_decode))
                    return {false, "reference pack is missing the final decode LM_HEAD"};
            }

            return {true, {}};
        }

        const std::chrono::steady_clock::time_point parity_fixture_started_at_ =
            std::chrono::steady_clock::now();
        bool production_parity_campaign_active_ = false;
        bool production_parity_model_context_reused_ = false;
        std::optional<PrefixRuntimeStateSnapshot>
            production_parity_decode_prefill_state_;
        std::vector<ProductionParityDecodeBoundary>
            production_parity_decode_boundaries_;
        std::vector<ProductionParityMTPTransactionBoundary>
            production_parity_mtp_transaction_boundaries_;
        std::vector<PrefixRestoreEvidence>
            production_parity_prefix_restore_evidence_;
        ParityConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::unique_ptr<IInferenceRunner> runner_;
        IInferenceRunner *borrowed_runner_ = nullptr;
        std::unordered_map<std::string, std::vector<float>> pytorch_snapshots_;
        mutable std::unordered_map<std::string, std::vector<float>> active_snapshot_combined_cache_;
        mutable std::string parity_reference_capture_filter_cache_dir_;
        mutable std::vector<std::string>
            parity_reference_capture_filter_cache_;
        std::unordered_map<std::string, ParityProfileRecord> parity_profile_records_;
        std::vector<SavedEnvValue> parity_env_overrides_;

        /** Sole owner of the fixture's isolated test-control protocol. */
        ParityCellLifecycle parity_cell_lifecycle_;

        // Stable production MPI context (optional, null for single-rank).
        std::shared_ptr<IMPIContext> mpi_ctx_;

        // Modern orchestration runner (for incremental migration from IInferenceRunner)
        // Tests can use either runner_ OR orch_runner_, but not both simultaneously.
        // Use setupOrchestrationRunner() to initialize orch_runner_, or
        // setupPipeline() to initialize runner_ (legacy path).
        std::unique_ptr<IOrchestrationRunner> orch_runner_;

        /** @brief Observed serving contract for forced-token state advancement. */
        enum class ParityForcedTokenCommitMode : std::uint8_t
        {
            Unknown = 0,
            Deferred = 1,
            Immediate = 2,
        };

        /**
         * @brief Authenticated teacher-forced production decode trajectory.
         *
         * Quantized logits may legitimately place the reference argmax inside
         * the accepted top-k without making it local top-1. An autonomous
         * decode loop would then leave the Hugging Face trajectory and compare
         * unrelated later checkpoints. Production parity instead uses the
         * serving `forceDecodeToken()` protocol that also powers constrained
         * continuations. Its typed result says whether a token remains pending
         * or was committed immediately by a device-resident transaction, so
         * the harness never guesses and never forwards one reference row twice.
         */
        std::vector<int32_t> orchestration_parity_decode_trajectory_;
        size_t orchestration_parity_decode_trajectory_index_ = 0;
        ParityForcedTokenCommitMode orchestration_parity_force_mode_ =
            ParityForcedTokenCommitMode::Unknown;

        IInferenceRunner *activeLegacyRunner() const
        {
            return runner_ ? runner_.get() : borrowed_runner_;
        }

        /**
         * @brief Return the active runner's read-only concrete model metadata.
         *
         * Parity layout projection consumes GGUF metadata but must never create
         * a second model context beside OrchestrationRunner's weight authority.
         */
        const ModelContext *activeModelContextForDiagnostics() const
        {
            if (model_ctx_)
                return model_ctx_.get();
            if (!orch_runner_)
                return nullptr;
            return dynamic_cast<const ModelContext *>(
                orch_runner_->modelContextForDiagnostics());
        }

        int parityLayerCount() const
        {
            const ModelContext *active_model =
                activeModelContextForDiagnostics();
            if (!active_model)
            {
                return orch_runner_
                           ? orch_runner_->executionPlan().layerCount()
                           : 0;
            }

            const int local_layers = active_model->blockCount();
            const int total_layers = active_model->totalBlockCount();
            if (local_layers != total_layers)
                return local_layers;

            return mainLayerCountExcludingMTP(
                active_model->concreteLoader(),
                active_model->architecture(),
                local_layers);
        }

        // Optional TestConfig for declarative test configuration (LocalPP, LocalTP, etc.)
        // Subclasses override cfg() to return their test configuration.
        // Default returns a single-device CPU config for backward compatibility.
        std::optional<TestConfig> test_config_;

        // =============================================================================
        // Test Configuration
        // =============================================================================

        /**
         * @brief Get the test configuration (override in subclasses)
         * @return Reference to TestConfig for this test
         *
         * Subclasses with declarative config should override this to return
         * their TestConfig. Default returns a single-device CPU configuration.
         */
        virtual const TestConfig &cfg() const
        {
            static const TestConfig default_config = {
                .name = "SingleDevice",
                .devices = {ParityDeviceType::CPU},
                .parallelism = Parallelism::None,
                .collective = Collective::None,
                .thresholds = {},
                .skip_reason = "",
                .mpi_ranks = 1};
            if (test_config_.has_value())
                return test_config_.value();
            return default_config;
        }

        // =============================================================================
        // MPI Helper Methods
        // =============================================================================

    private:
        /**
         * @brief Enter the one base-owned MPI lifecycle for this fixture.
         *
         * The derived fixture may install `mpi_ctx_` first when production
         * execution spans ranks.  Single-rank runners intentionally keep that
         * production pointer null, but their test lifecycle still duplicates
         * `MPI_COMM_WORLD` (of size one).  Production runners never receive the
         * duplicated test channels.
         *
         * @throws std::logic_error for re-entry or an unavailable MPI runtime.
         * @throws std::runtime_error when MPI cannot create the isolated scope.
         */
        void enterParityCellLifecycle()
        {
            int initialized = 0;
            int finalized = 0;
            if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
                MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
            {
                throw std::logic_error(
                    "Parity cell lifecycle requires an active MPI runtime");
            }

            MPI_Comm source = MPI_COMM_WORLD;
            if (mpi_ctx_)
            {
                source = mpi_ctx_->communicator();
                if (source == MPI_COMM_NULL)
                {
                    throw std::logic_error(
                        "Parity production context has no live communicator");
                }
            }

            parity_cell_lifecycle_.enter(source);
        }

        /**
         * @brief Retire both test-only channels after the final rendezvous.
         *
         * The base fixture is the sole lifecycle authority.  Retiring while a
         * production runner is live would free communicator handles beneath
         * active state and is therefore a fatal transition error.
         */
        void retireParityCellLifecycle()
        {
            if (runner_ || orch_runner_ || borrowed_runner_)
            {
                throw std::logic_error(
                    "Parity cell lifecycle cannot retire while execution state is live");
            }
            parity_cell_lifecycle_.retire();
        }

        /**
         * @brief Rendezvous on the teardown-only lifecycle channel.
         *
         * No setup, evidence, production, or CSV collective may use this
         * channel. Consequently its entry and exit barriers have one exact
         * sequence and cannot be satisfied by a different logical phase.
         *
         * @throws std::logic_error when no typed lifecycle channel is active.
         */
        void parityLifecycleBarrier()
        {
            parity_cell_lifecycle_.teardownBarrier();
        }

    protected:

        /**
         * @return Test-control context when a cell scope is active, otherwise
         *         the stable production context for legacy parity fixtures.
         */
        const std::shared_ptr<IMPIContext> &parityCoordinationMPIContext() const
        {
            return parity_cell_lifecycle_.active()
                       ? parity_cell_lifecycle_.controlContext()
                       : mpi_ctx_;
        }

        /**
         * @return Exact communicator for test barriers and evidence exchange.
         * @throws std::logic_error when the fixture has no MPI coordination.
         */
        MPI_Comm parityCoordinationCommunicator() const
        {
            const auto &context = parityCoordinationMPIContext();
            if (!context || context->communicator() == MPI_COMM_NULL)
            {
                throw std::logic_error(
                    "Parity MPI coordination has no live communicator");
            }
            return context->communicator();
        }

        /**
         * @brief Return the rank that owns comparison and CSV artifact output.
         *
         * Most parity cells use rank zero. A rank-agnostic heterogeneous graph
         * may bind its dense continuation and snapshot store to another rank;
         * that fixture overrides this after its production runner resolves the
         * topology. Setup-time reference preparation still defaults to zero
         * before the runner exists.
         */
        virtual int parityArtifactAuthorityRank() const
        {
            return 0;
        }

        /**
         * @brief Declare which ranks share reference-generation control flow.
         * @return Symmetric participation unless a production command loop
         *         owns peer ranks.
         */
        virtual ParityReferenceGenerationCoordination
        parityReferenceGenerationCoordination() const
        {
            return ParityReferenceGenerationCoordination::SymmetricRanks;
        }

        /**
         * @brief Check whether this rank owns parity artifacts and assertions.
         * @return true for the configured artifact authority or single-rank mode.
         */
        bool isRank0() const
        {
            return !mpi_ctx_ ||
                   mpi_ctx_->rank() == parityArtifactAuthorityRank();
        }

        /**
         * @brief Get MPI rank (0 if no MPI context)
         */
        int mpiRank() const
        {
            return mpi_ctx_ ? mpi_ctx_->rank() : 0;
        }

        /**
         * @brief Get MPI world size (1 if no MPI context)
         */
        int mpiWorldSize() const
        {
            return mpi_ctx_ ? mpi_ctx_->world_size() : 1;
        }

        /**
         * @brief Execute MPI barrier if MPI context exists
         */
        void mpiBarrier()
        {
            const auto &context = parityCoordinationMPIContext();
            if (context)
            {
                context->barrier();
            }
        }

        // =============================================================================
        // Device Selection (override one of these)
        // =============================================================================

        /**
         * @brief Get the device to use for inference (single-rank tests)
         * @return DeviceId (e.g., DeviceId::cpu(), DeviceId::cuda(0))
         *
         * Override this for single-rank tests. For MPI tests, override
         * getDeviceForRank() instead.
         */
        virtual DeviceId getDevice()
        {
            // Default implementation calls getDeviceForRank() for MPI compatibility
            return getDeviceForRank();
        }

        /**
         * @brief Get the device for this MPI rank (tensor-parallel tests)
         * @return DeviceId for the current rank
         *
         * Override this for MPI tests to return different devices per rank.
         * Default implementation returns CPU.
         */
        virtual DeviceId getDeviceForRank()
        {
            return DeviceId::cpu();
        }

        /**
         * @brief Get the backend name for display
         * @return Name string (e.g., "CUDA", "CPU", "ROCm")
         */
        virtual std::string getBackendName() = 0;

        // =============================================================================
        // Weight Distribution (override for tensor parallelism)
        // =============================================================================

        /**
         * @brief Get the weight distribution strategy
         * @return WeightDistributionStrategy (REPLICATED for single-rank, SHARDED for TP)
         *
         * Override this for tensor-parallel tests to enable weight sharding.
         */
        virtual WeightDistributionStrategy getWeightStrategy()
        {
            return WeightDistributionStrategy::REPLICATED;
        }

        /**
         * @brief Configure model after loading (optional hook)
         * @param model_ctx The loaded model context
         *
         * Override this for tensor-parallel tests to configure weight sharding schema.
         * Called after ModelContext::create() but before createInferenceRunner().
         */
        virtual void configureModel(std::shared_ptr<ModelContext> model_ctx)
        {
            // Default: no additional configuration
        }

        /**
         * @brief Device-specific setup (optional)
         *
         * Override to add device availability checks, GPU initialization, etc.
         * Call GTEST_SKIP() if device is not available.
         */
        virtual void setupDeviceSpecific() {}

        virtual bool preserveParityPipelineCachesBetweenTests() const
        {
            return false;
        }

        /**
         * @brief Load and configure the authoritative model context for a runner.
         *
         * Campaign fixtures may override this boundary to retain one immutable
         * real-weight context while rebuilding precision-specific runners.  The
         * returned context remains the sole owner of its WeightManager and
         * PreparedWeightStore; callers must not construct a host mirror.
         */
        virtual std::shared_ptr<ModelContext> acquireParityModelContext(
            WeightDistributionStrategy strategy)
        {
            auto context = ModelContext::create(
                config_.model_path,
                mpi_ctx_,
                nullptr,
                nullptr,
                strategy);
            if (context)
                configureModel(context);
            return context;
        }

        void borrowParityPipeline(
            std::shared_ptr<ModelContext> model_ctx,
            IInferenceRunner *runner)
        {
            runner_.reset();
            orch_runner_.reset();
            model_ctx_ = std::move(model_ctx);
            borrowed_runner_ = runner;
            if (borrowed_runner_)
                borrowed_runner_->enableSnapshotCapture();
        }

        void releaseBorrowedParityPipeline()
        {
            borrowed_runner_ = nullptr;
            model_ctx_.reset();
        }

        /**
         * @brief Quote one argument for an exact `bash -c` command boundary.
         *
         * Prompts may contain newlines, quotes, dollar signs, and shell syntax.
         * Snapshot generation must receive those bytes unchanged and must never
         * interpret a model path or prompt as part of the command program.
         */
        static std::string parityShellQuote(const std::string &value)
        {
            std::string quoted = "'";
            for (const char character : value)
            {
                if (character == '\'')
                    quoted += "'\"'\"'";
                else
                    quoted.push_back(character);
            }
            quoted.push_back('\'');
            return quoted;
        }

        void SetUp() override
        {
            // Model parity is a production-path canary. Do not let an inherited
            // shell env or a previous test route it through deterministic GEMM.
            setenv("LLAMINAR_DETERMINISTIC", "0", 1);
            mutableDebugEnv().reload();
            auto parity_profile_scope = profileParityScope("set_up.total");

            /*
             * Every parity topology, including a one-rank CPU/GPU case, owns
             * the same isolated evidence and teardown channels.  Starting the
             * lifecycle here prevents individual model/topology fixtures from
             * partially adopting the protocol.
             */
            enterParityCellLifecycle();

            // Resolve only after the derived fixture has selected its exact
            // model. The RAM copy remains immutable for the complete aggregate
            // run and is re-authenticated against reference metadata below.
            if (!config_.model_path.empty())
            {
                try
                {
                    config_.model_path = productionParityResolvedModelPath(
                        config_.model_path);
                }
                catch (const std::exception &error)
                {
                    FAIL() << error.what();
                }
            }

            /*
             * Parity binaries construct production runners directly and therefore
             * do not pass through RuntimeInitPhase, which normally publishes the
             * process-wide CPU allocation domain before graph preparation.  A
             * cross-rank CPU participant owns one exact NUMA domain: packed expert
             * arrivals and every persistent CPU workspace must be allocated there.
             * Derive that identity from the process affinity established by the
             * registered MPI launch, exactly as RuntimeInitPhase does.  An
             * aggregate backend is valid only for a genuinely single-rank test.
             */
            if (!llaminar2::hasCPUBackend())
            {
                int backend_numa_node = -1;
                const bool cross_rank_cpu =
                    mpi_ctx_ && mpi_ctx_->world_size() > 1 &&
                    getDeviceForRank().is_cpu();
                if (cross_rank_cpu)
                {
                    const auto numa =
                        llaminar2::NUMATopology::detectLocalNUMANode();
                    ASSERT_TRUE(numa.detection_succeeded)
                        << "Cross-rank CPU production parity requires exact "
                           "rank-to-NUMA affinity before backend initialization";
                    ASSERT_GE(numa.local_numa_node, 0)
                        << "Cross-rank CPU production parity resolved an invalid "
                           "NUMA node";
                    backend_numa_node = numa.local_numa_node;
                }
                llaminar2::initCPUBackend(backend_numa_node);
            }

            /*
             * A commit/test-name directory is reused across invocations. Remove
             * only this cell's generated directory before opening its log so an
             * early setup failure cannot leave CSV evidence from a prior run.
             * Rank zero owns deletion; every other rank waits until that exact
             * artifact authority has been recreated.
             */
            int32_t results_dir_ready = 1;
            if (isRank0())
            {
                const auto dir = getResultsDir();
                std::error_code ec;
                std::filesystem::remove_all(dir, ec);
                if (!ec)
                    std::filesystem::create_directories(dir, ec);
                results_dir_ready = ec ? 0 : 1;
            }
            if (mpi_ctx_ && mpiWorldSize() > 1)
                parityCoordinationMPIContext()->broadcast_int32(
                    &results_dir_ready, 1, 0);
            mpiBarrier();
            ASSERT_EQ(results_dir_ready, 1)
                << "Cannot prepare a fresh parity artifact directory for "
                << getTestName();

            /*
             * Every rank owns a separate diagnostic stream. The artifact
             * authority retains the canonical `test_log.txt` progress marker
             * consumed by the campaign watchdog; peers publish rank-qualified
             * logs so a local setup failure is visible even when another rank
             * is blocked in a production initialization consensus.
             */
            {
                auto dir = ensureResultsDir();
                const auto log_path =
                    dir /
                    (isRank0()
                         ? std::string("test_log.txt")
                         : "rank_" + std::to_string(mpiRank()) +
                               "_test_log.txt");
                if (Logger::getInstance().setLogFile(log_path.string()))
                {
                    LOG_INFO("[Parity] Log file: " << log_path.string());
                }
            }

            // CRITICAL: Clear kernel caches at test start for clean state.
            // TearDown() also clears, but this guards against incomplete teardown
            // from a prior test (crash, skip, or assertion failure) leaving stale
            // GEMM engines or prepared-weight handles.
            if (!preserveParityPipelineCachesBetweenTests())
                llaminar::v2::kernels::KernelFactory::clearCache();

            // Device-specific setup first (may skip)
            setupDeviceSpecific();

            // Skip cleanly if the model GGUF isn't staged on this runner.
            // Missing models otherwise produce confusing snapshot-regeneration
            // failures inside Python rather than a clear "model not found".
            if (!config_.model_path.empty() && !std::filesystem::exists(config_.model_path))
            {
                if (productionParityCampaignEnabled())
                {
                    FAIL() << "Production parity model file not found: "
                           << config_.model_path
                           << " (the real-weight campaign cannot certify a skipped cell)";
                }
                GTEST_SKIP() << "Model file not found: " << config_.model_path
                             << " (populate MODELS_DIR on the runner)";
            }

            resolveSnapshotDirIfNeeded();
            // Regenerate snapshots only on rank 0 to avoid race conditions
            // and redundant work. All ranks wait at barrier before proceeding.
            // OPTIMIZATION: Skip regeneration if snapshots already exist on disk
            // (metadata.txt is the marker file written by all Python generators)
            // AND the snapshot version matches the expected version.
            // This is critical for MPI_PROCS>1 tests where popen()/fork() inside
            // an MPI-managed process can crash the HNP event loop.
            int32_t snapshots_ready = 1;
            std::string snapshot_failure_reason;
            {
                auto scope = profileParityScope("set_up.snapshot_ready");
                if (isRank0())
                {
                    bool need_regen = false;
                    {
                        std::lock_guard<std::mutex> lock(s_snapshot_mutex_);
                        need_regen = (s_generated_snapshots_.find(snapshotCacheKey()) == s_generated_snapshots_.end());
                    }

                    // Production cells revalidate on every fixture. SHA-256 file
                    // hashing is cached by canonical path/size/mtime in-process,
                    // so this catches a changed pack without rescanning weights.
                    if (need_regen || productionParityCampaignEnabled())
                    {
                        // Check disk first — snapshots may exist from a prior test run
                        auto metadata_path = std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
                        if (std::filesystem::exists(metadata_path))
                        {
                            const auto validation = validateReferenceSnapshotMetadata(
                                metadata_path,
                                productionParityCampaignEnabled());
                            if (validation.usable)
                            {
                                LOG_INFO("[" << getBackendName()
                                             << " Parity] Found reusable v"
                                             << readSnapshotVersion(metadata_path)
                                             << " reference pack on disk: "
                                             << config_.snapshot_dir);
                                need_regen = false;
                            }
                            else
                            {
                                LOG_WARN("[" << getBackendName()
                                             << " Parity] Reference pack is stale or unauthenticated ("
                                             << validation.reason << ") — regenerating: "
                                             << config_.snapshot_dir);
                                need_regen = true;
                            }
                        }
                        else
                        {
                            need_regen = true;
                        }
                    }

                    std::unique_ptr<ReferenceGenerationLease>
                        generation_lease;
                    if (need_regen)
                    {
                        /*
                         * The backend scheduler can run CPU, CUDA, and ROCm
                         * cells together, but all three generate references in
                         * host RAM. Serialize only this heavyweight writer
                         * phase, then validate again: a peer may have completed
                         * the same immutable pack while this process waited.
                         */
                        generation_lease =
                            std::make_unique<ReferenceGenerationLease>();
                        const auto metadata_path =
                            std::filesystem::path(config_.snapshot_dir) /
                            "metadata.txt";
                        if (std::filesystem::exists(metadata_path))
                        {
                            const auto validation =
                                validateReferenceSnapshotMetadata(
                                    metadata_path,
                                    productionParityCampaignEnabled());
                            if (validation.usable)
                            {
                                LOG_INFO("[" << getBackendName()
                                             << " Parity] Reusing reference pack "
                                                "published while waiting for the "
                                                "generation lease: "
                                             << config_.snapshot_dir);
                                need_regen = false;
                            }
                        }
                    }

                    if (need_regen)
                    {
                        if (!regeneratePyTorchSnapshots())
                        {
                            snapshots_ready = 0;
                            snapshot_failure_reason =
                                "PyTorch snapshot generation failed";
                        }
                        else
                        {
                            const auto metadata_path =
                                std::filesystem::path(config_.snapshot_dir) /
                                "metadata.txt";
                            const auto validation = validateReferenceSnapshotMetadata(
                                metadata_path,
                                productionParityCampaignEnabled());
                            if (!validation.usable)
                            {
                                snapshots_ready = 0;
                                snapshot_failure_reason =
                                    "generated reference pack failed validation: " +
                                    validation.reason;
                            }
                        }
                    }

                    if (snapshots_ready != 0)
                    {
                        // Mark only a completely generated/validated pack. Later
                        // production fixtures still revalidate its content identity.
                        std::lock_guard<std::mutex> lock(s_snapshot_mutex_);
                        s_generated_snapshots_.insert(snapshotCacheKey());
                        if (!need_regen)
                        {
                            LOG_DEBUG("[" << getBackendName()
                                           << " Parity] Reusing cached snapshots from: "
                                           << config_.snapshot_dir);
                        }
                    }
                }

                if (mpi_ctx_ && mpiWorldSize() > 1)
                    parityCoordinationMPIContext()->broadcast_int32(
                        &snapshots_ready, 1, 0);
            }
            {
                auto scope = profileParityScope("set_up.snapshot_barrier");
                mpiBarrier(); // All ranks wait for snapshots to be ready
            }
            ASSERT_EQ(snapshots_ready, 1)
                << (snapshot_failure_reason.empty()
                        ? "rank 0 failed to prepare the PyTorch reference pack"
                        : snapshot_failure_reason);
        }

        void TearDown() override
        {
            /*
             * GoogleTest invokes TearDown after a derived SetUp skips or fails
             * before calling this base.  That is the only valid no-lifecycle
             * state; once base setup has entered, every teardown transition is
             * strict and must retire exactly once.
             */
            if (parity_cell_lifecycle_.state() ==
                ParityCellLifecycleState::NotEntered)
            {
                return;
            }
            if (!parity_cell_lifecycle_.active())
            {
                throw std::logic_error(
                    "Parity teardown entered from an invalid lifecycle state");
            }

            // Barrier before teardown to ensure all ranks are done
            {
                auto scope = profileParityScope("tear_down.initial_barrier");
                parityLifecycleBarrier();
            }

            // Ensure all GPU work that may reference graph stages or cached
            // prepared weights has completed before any owner is destroyed.
            {
                auto scope = profileParityScope("tear_down.pre_destroy_gpu_sync");
#ifdef HAVE_CUDA
                if (auto *cuda_backend = llaminar2::getCUDABackend())
                {
                    for (int d = 0; d < cuda_backend->deviceCount(); ++d)
                    {
                        cuda_backend->synchronize(d);
                    }
                    cudaGetLastError();
                }
#endif
#ifdef HAVE_ROCM
                if (auto *rocm_backend = llaminar2::getROCmBackend())
                {
                    for (int d = 0; d < rocm_backend->deviceCount(); ++d)
                    {
                        rocm_backend->synchronize(d);
                    }
                }
#endif
            }

            const bool preserve_campaign_caches =
                preserveParityPipelineCachesBetweenTests();
            if (preserve_campaign_caches && borrowed_runner_)
            {
                auto scope = profileParityScope("tear_down.reset_campaign_pipeline");
                /*
                 * Only a borrowed runner survives this fixture and therefore
                 * needs a request reset. Owned legacy and orchestration runners
                 * are destroyed immediately below; resetting an orchestration
                 * runner whose coordinated worker loop already consumed its
                 * terminal SHUTDOWN would publish an unmatched MPI command.
                 */
                activeClearSnapshots();
                activeClearCache();
            }

            // Destroy graph/stage owners before touching process-wide caches.
            // A borrowed runner stays owned by its exact-identity pipeline cache;
            // an ordinary campaign runner is retired between precision cells.
            {
                auto scope = profileParityScope("tear_down.destroy_runner");
                runner_.reset();
                orch_runner_.reset();
                if (borrowed_runner_)
                    releaseBorrowedParityPipeline();
            }

            if (!preserve_campaign_caches)
            {
                // CRITICAL: Clear kernel cache BEFORE destroying model context!
                // KernelFactory::clearCache() accesses tensor->cache_ (CPU packed weights)
                // to free resources. If we destroy the tensors first (via model_ctx_.reset()),
                // clearCache() would be accessing freed memory (use-after-free).
                {
                    auto scope = profileParityScope("tear_down.clear_kernel_cache");
                    llaminar::v2::kernels::KernelFactory::clearCache();
                }

            }

            {
                auto scope = profileParityScope("tear_down.clear_model_and_snapshots");
                model_ctx_.reset();
                pytorch_snapshots_.clear();
            }

            // CRITICAL: Synchronize and clear error state on all GPU devices!
            // After heterogeneous tests (CUDA+ROCm), the HIP runtime can be left
            // in a bad state that causes subsequent ROCm-only tests to fail with
            // "invalid argument" on kernel launch. Synchronizing each backend
            // cleans up any lingering issues.
#ifdef HAVE_CUDA
            {
                auto scope = profileParityScope("tear_down.final_cuda_sync");
                if (auto *cuda_backend = llaminar2::getCUDABackend())
                {
                    // Synchronize ALL CUDA devices, not just device 0. LocalTP
                    // parity cases keep borrowed multi-GPU runners alive across
                    // tests, and clear_cache() may enqueue request-reset work on
                    // every participant stream.
                    for (int d = 0; d < cuda_backend->deviceCount(); ++d)
                    {
                        cuda_backend->synchronize(d);
                    }
                    // Clear CUDA sticky error state to prevent async kernel
                    // errors from one test case propagating to the next test's
                    // first CUDA API call.
                    cudaGetLastError();
                }
            }
#endif
#ifdef HAVE_ROCM
            {
                auto scope = profileParityScope("tear_down.final_rocm_sync");
                if (auto *rocm_backend = llaminar2::getROCmBackend())
                {
                    // Synchronize ALL ROCm devices, not just device 0.
                    // TP configs use multiple ROCm GPUs; leaving device 1+
                    // unsynchronized causes ROCm runtime corruption (null pointer
                    // in memobj map) when the next test config allocates memory.
                    for (int d = 0; d < rocm_backend->deviceCount(); ++d)
                    {
                        rocm_backend->synchronize(d);
                    }
                }
            }
#endif

            // Close log file at end of test
            printParityProfileSummary();
            Logger::getInstance().closeLogFile();
            restoreParityEnvOverrides();

            /*
             * The initial barrier protects objects owned by the test that is
             * ending; it does not protect the communicator from a fast rank
             * entering the next process-resident campaign cell while another
             * rank is still synchronizing devices, clearing caches, or
             * printing evidence. End teardown with a second rendezvous so the
             * next SetUp's inventory broadcasts cannot alias a prior cell's
             * teardown lifecycle.
             */
            {
                auto scope = profileParityScope("tear_down.final_barrier");
                parityLifecycleBarrier();
            }
            retireParityCellLifecycle();
        }

        /**
         * @brief Regenerate PyTorch snapshots from the GGUF model
         *
         * Override in derived classes to use architecture-specific generators
         * (e.g., Qwen3.5 requires a dedicated generator for GDN layers).
         */
        virtual bool regeneratePyTorchSnapshots()
        {
            LOG_INFO("[" << getBackendName() << " Parity] Regenerating PyTorch snapshots from GGUF: " << config_.model_path);

            std::ostringstream script;
            // Source devcontainer venv if present, else fall back to system
            // python3 (which is what the CI builder image uses, where Python
            // deps were installed via `pip install --break-system-packages`).
            //
            // IMPORTANT: CTest sets OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for the
            // Llaminar test process (intentional: mpirun -np 1 handles affinity itself).
            // Those env vars are inherited by this python3 subprocess, which would pin
            // PyTorch's CPU forward pass to a single thread — catastrophic for large
            // models. We unset them and let PyTorch/OpenMP/MKL use all available cores.
            script << "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
                   << "if [ -f /workspaces/llaminar/.venv/bin/activate ]; then "
                   << "source /workspaces/llaminar/.venv/bin/activate; fi; "
                   << "python3 python/reference/generate_qwen_pipeline_snapshots.py"
                   << " --model " << parityShellQuote(config_.model_path)
                   << " --prompt " << parityShellQuote(config_.prompt)
                   << " --output " << parityShellQuote(config_.snapshot_dir)
                   << " --decode-steps " << config_.decode_steps;
            const std::string command =
                "bash -c " + parityShellQuote(script.str()) + " 2>&1";

            FILE *pipe = popen(command.c_str(), "r");
            if (!pipe)
            {
                LOG_ERROR("[Parity] Failed to execute snapshot generator");
                return false;
            }

            char buffer[256];
            std::string output;
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            {
                output += buffer;
            }

            int exit_code = pclose(pipe);
            if (exit_code != 0)
            {
                LOG_ERROR("[Parity] Snapshot generation failed:\n"
                          << output);
                return false;
            }

            LOG_INFO("[Parity] Snapshots regenerated successfully");
            return true;
        }

        /**
         * @brief Generate one additive Hugging Face MTP branch reference.
         *
         * A recursively sampled production predictor may leave the canonical
         * Hugging Face greedy branch while remaining numerically correct. The
         * resulting checkpoint is comparable only after replaying the sidecar
         * with the exact committed proposal-token identity. Architectures that
         * support MTP override this boundary with their sidecar-only generator;
         * the default fails closed so an unrelated canonical tensor can never
         * masquerade as the oracle.
         *
         * @param reference_step Main-model decode step owning the transaction.
         * @param condition_tokens Exact tokens consumed by recursive MTP1..N.
         * @return True only after every branch-qualified checkpoint is durable.
         */
        virtual bool regeneratePyTorchMTPBranchSnapshots(
            int reference_step,
            const std::vector<int32_t> &condition_tokens)
        {
            (void)reference_step;
            (void)condition_tokens;
            LOG_ERROR(
                "[Parity] This model reference generator does not implement "
                "forced-branch MTP snapshots");
            return false;
        }

        /**
         * @brief Load PyTorch snapshot from .npy file
         */
        std::vector<float> loadPyTorchSnapshot(const std::string &name)
        {
            if (pytorch_snapshots_.find(name) != pytorch_snapshots_.end())
            {
                auto scope = profileParityScope("pytorch_snapshot.cache_copy");
                return pytorch_snapshots_[name];
            }

            auto scope = profileParityScope("pytorch_snapshot.load_npy");
            std::string npy_path = config_.snapshot_dir + "/" + name + ".npy";

            try
            {
                cnpy::NpyArray arr = cnpy::npy_load(npy_path);

                std::vector<float> data;
                if (arr.word_size == sizeof(float))
                {
                    float *data_ptr = arr.data<float>();
                    data.assign(data_ptr, data_ptr + arr.num_vals);
                }
                else if (arr.word_size == sizeof(double))
                {
                    double *data_ptr = arr.data<double>();
                    data.resize(arr.num_vals);
                    for (size_t i = 0; i < arr.num_vals; ++i)
                    {
                        data[i] = static_cast<float>(data_ptr[i]);
                    }
                }
                else
                {
                    LOG_ERROR("[Parity] Unsupported data type in snapshot '" << name << "'");
                    return {};
                }

                pytorch_snapshots_[name] = data;
                return data;
            }
            catch (const std::exception &e)
            {
                /*
                 * Stage snapshots are sparse by design: callers probe a common
                 * stage list across dense, MoE, attention, and GDN layers.
                 * Required snapshots are asserted by the caller after this
                 * method returns empty, so logging every expected miss as an
                 * error hides the actual parity failure in thousands of lines.
                 */
                LOG_DEBUG("[Parity] Snapshot unavailable '" << name << "': " << e.what());
                return {};
            }
        }

        // =================================================================
        // GDN V-head permutation for parity comparison
        // =================================================================
        //
        // V-head ordering context:
        //
        // GGUF stores V-heads in tiled order for efficient ggml broadcast.
        // For **dense** Qwen3.5 models, both Llaminar and PyTorch use
        // GGUF tiled V-head order (the Python GGUF loader skips reversal
        // for dense models). No comparison-time permutation is needed.
        //
        // For **MoE** Qwen3.5 models, the Python GGUF loader reverses
        // V-head tiling to HF grouped order (the MoE Llaminar GDN
        // implementation expects grouped V-head order). Comparison-time
        // permutation maps Llaminar's output (grouped) to PyTorch's
        // output (also grouped after reversal) — which are already
        // aligned. However, the QKV_PROJECTION snapshot captures the
        // raw projection output before GDN processes it, so V-heads
        // appear in different order between Llaminar (tiled) and
        // PyTorch (grouped after weight reversal). The permutation
        // fixes this for comparison.
        //

        /**
         * @brief GDN head configuration with MoE detection for V-head permutation
         *
         * V-head permutation is only needed for MoE models where the Python
         * GGUF loader applies V-head reversal (tiled→grouped). Dense models
         * skip reversal, so both sides use tiled order and no permutation
         * is needed at comparison time.
         */
        ParityGDNHeadConfig getGDNHeadConfig() const
        {
            return parityGDNHeadConfigFromModel(
                activeModelContextForDiagnostics());
        }

        /**
         * @brief Read MoE configuration from GGUF metadata
         */
        struct MoEConfig
        {
            int num_experts = 0;
            int top_k = 0;
        };

        MoEConfig getMoEConfig() const
        {
            const ModelContext *model_ctx =
                activeModelContextForDiagnostics();
            if (!model_ctx)
                return {};

            const auto &arch = model_ctx->architecture();
            const auto &meta = model_ctx->model().metadata;

            auto getMetaInt = [&](const std::string &suffix) -> int
            {
                auto it = meta.find(arch + "." + suffix);
                if (it == meta.end())
                    return 0;
                const auto &val = it->second;
                if (val.type == GGUFValueType::UINT32)
                    return static_cast<int>(val.asUInt32());
                if (val.type == GGUFValueType::UINT64)
                    return static_cast<int>(val.asUInt64());
                return 0;
            };

            MoEConfig cfg;
            cfg.num_experts = getMetaInt("expert_count");
            cfg.top_k = getMetaInt("expert_used_count");
            return cfg;
        }

        /**
         * @brief Apply GDN V-head permutation to Llaminar data for comparison.
         *
         * For stages that contain V-head-ordered data (Z projection, delta rule,
         * norm gate), permutes from Llaminar's ratio-grouped order to PyTorch's
         * interleaved order. For QKV_PROJECTION, only the V portion is permuted.
         *
         * @return Permuted copy if permutation was applied, empty vector otherwise.
         *         When non-empty, use permuted.data() instead of llaminar_data.
         */
        std::vector<float> applyGDNHeadPermutation(
            const float *llaminar_data,
            size_t size,
            const std::string &stage,
            const ParityGDNHeadConfig &gdn) const
        {
            return applyParityGDNHeadPermutation(
                llaminar_data,
                size,
                stage,
                gdn);
        }

        /**
         * @brief Compare tensors and compute metrics
         */
        StageComparisonResult compareTensors(
            const float *actual,
            const std::vector<float> &expected,
            size_t size,
            const std::string &stage_name = "")
        {
            if (expected.empty() || expected.size() != size)
                return StageComparisonResult{.stage_name = stage_name};
            return compareParityTensorData(
                actual,
                expected.data(),
                size,
                stage_name,
                config_.cosine_threshold);
        }

        /**
         * @brief Compare MoE routing indices (expert selection).
         *
         * For each token, computes the set overlap between selected expert IDs
         * (Jaccard-like: |intersection| / top_k). Stores mean overlap in
         * cosine_similarity for consistent table rendering.
         *
         * @param actual    Llaminar routing indices [seq_len * top_k] (int cast to float)
         * @param expected  PyTorch routing indices [seq_len * top_k] (int cast to float)
         * @param size      Total elements (seq_len * top_k)
         * @param top_k     Number of experts selected per token
         */
        StageComparisonResult compareRoutingIndices(
            const float *actual,
            const std::vector<float> &expected,
            size_t size,
            int top_k,
            const std::string &stage_name = "MOE_ROUTING_INDICES")
        {
            StageComparisonResult result;
            result.stage_name = stage_name;
            result.total_elements = size;

            if (expected.empty() || expected.size() != size || top_k <= 0)
                return result;

            const size_t seq_len = size / static_cast<size_t>(top_k);
            double total_overlap = 0.0;
            double total_rank_corr = 0.0;
            int position_0_match = 0; // How often the top-1 expert matches

            for (size_t t = 0; t < seq_len; ++t)
            {
                const float *ll_row = actual + t * top_k;
                const float *pt_row = expected.data() + t * top_k;

                // Build sets for intersection
                std::set<int> ll_set, pt_set;
                for (int k = 0; k < top_k; ++k)
                {
                    ll_set.insert(static_cast<int>(ll_row[k]));
                    pt_set.insert(static_cast<int>(pt_row[k]));
                }

                // Set intersection size
                int overlap = 0;
                for (int id : ll_set)
                    if (pt_set.count(id))
                        overlap++;

                total_overlap += static_cast<double>(overlap) / top_k;

                // Top-1 match (most-weighted expert)
                if (static_cast<int>(ll_row[0]) == static_cast<int>(pt_row[0]))
                    position_0_match++;
            }

            result.is_routing_stage = true;
            result.routing_overlap = static_cast<float>(total_overlap / seq_len);
            result.routing_top1_match = static_cast<float>(position_0_match) / static_cast<float>(seq_len);
            result.cosine_similarity = result.routing_overlap;   // Keep for backward compat (layer log)
            result.max_abs_diff = 1.0f - result.routing_overlap; // "distance" from perfect
            result.passed = (result.routing_overlap >= config_.cosine_threshold);
            return result;
        }

        /**
         * @brief Compare MoE routing weights (expert contributions).
         *
         * For each token, creates a sparse [num_experts]-dim vector where
         * vec[expert_id] = routing_weight, then computes cosine similarity
         * between the two sparse vectors. This naturally handles different
         * expert orderings and partial set overlaps.
         *
         * @param actual_weights   Llaminar routing weights [seq_len * top_k]
         * @param expected_weights PyTorch routing weights [seq_len * top_k]
         * @param actual_indices   Llaminar routing indices [seq_len * top_k]
         * @param expected_indices PyTorch routing indices [seq_len * top_k]
         * @param size             Elements in weight arrays (seq_len * top_k)
         * @param top_k            Experts per token
         * @param num_experts      Total expert count (for sparse vector dim)
         */
        StageComparisonResult compareRoutingWeights(
            const float *actual_weights,
            const std::vector<float> &expected_weights,
            const float *actual_indices,
            const std::vector<float> &expected_indices,
            size_t size,
            int top_k,
            int num_experts,
            const std::string &stage_name = "MOE_ROUTING_WEIGHTS")
        {
            StageComparisonResult result;
            result.stage_name = stage_name;
            result.total_elements = size;

            if (expected_weights.empty() || expected_weights.size() != size || top_k <= 0)
                return result;

            const size_t seq_len = size / static_cast<size_t>(top_k);
            double total_cosine = 0.0;
            double total_l1 = 0.0;
            float max_weight_diff = 0.0f;

            for (size_t t = 0; t < seq_len; ++t)
            {
                // Build sparse weight vectors indexed by expert ID
                std::vector<float> ll_sparse(num_experts, 0.0f);
                std::vector<float> pt_sparse(num_experts, 0.0f);

                for (int k = 0; k < top_k; ++k)
                {
                    int ll_id = static_cast<int>(actual_indices[t * top_k + k]);
                    int pt_id = static_cast<int>(expected_indices[t * top_k + k]);
                    if (ll_id >= 0 && ll_id < num_experts)
                        ll_sparse[ll_id] = actual_weights[t * top_k + k];
                    if (pt_id >= 0 && pt_id < num_experts)
                        pt_sparse[pt_id] = expected_weights[t * top_k + k];
                }

                // Cosine similarity of sparse weight vectors
                double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
                double l1 = 0.0;
                for (int e = 0; e < num_experts; ++e)
                {
                    dot += ll_sparse[e] * pt_sparse[e];
                    norm_a += ll_sparse[e] * ll_sparse[e];
                    norm_b += pt_sparse[e] * pt_sparse[e];
                    float diff = std::abs(ll_sparse[e] - pt_sparse[e]);
                    l1 += diff;
                    if (diff > max_weight_diff)
                        max_weight_diff = diff;
                }

                double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
                total_cosine += (denom > 1e-30) ? (dot / denom) : 0.0;
                total_l1 += l1;
            }

            result.is_routing_stage = true;
            result.routing_overlap = static_cast<float>(total_cosine / seq_len); // sparse-vector cosine
            result.routing_weight_l1 = static_cast<float>(total_l1 / seq_len);
            result.cosine_similarity = result.routing_overlap; // Keep for backward compat (layer log)
            result.max_abs_diff = max_weight_diff;
            result.passed = (result.routing_overlap >= config_.cosine_threshold);
            return result;
        }

        /**
         * @brief Check if a stage name is a MoE routing stage needing special comparison
         */
        static bool isRoutingStage(const std::string &stage)
        {
            return stage == "MOE_ROUTING_INDICES" || stage == "MOE_ROUTING_WEIGHTS";
        }

        /**
         * @brief Compare TP snapshot against PyTorch reference with sharding awareness
         *
         * For column-parallel stages:
         * 1. Compare each device's partial output against corresponding PyTorch slice
         * 2. Concatenate all device outputs and compare against full PyTorch
         *
         * For row-parallel/replicated stages:
         * 1. Verify all devices have consistent data
         * 2. Compare combined output against full PyTorch
         *
         * @param tp_snapshot The TPSnapshot from RankOrchestrator::getTPSnapshot()
         * @param pytorch_data Full PyTorch reference data
         * @param pytorch_rows Number of rows in PyTorch data (seq_len)
         * @param pytorch_cols Number of columns in PyTorch data (feature_dim)
         * @return TPStageComparisonResult with per-device and combined metrics
         */
        TPStageComparisonResult compareTPSnapshot(
            TPSnapshot &tp_snapshot,
            const std::vector<float> &pytorch_data,
            size_t pytorch_rows,
            size_t pytorch_cols)
        {
            TPStageComparisonResult result;

            // Debug: print first few values for layer0 attention to diagnose
            bool debug_print = (tp_snapshot.key.find("layer0_ATTENTION_CONTEXT") != std::string::npos);
            result.stage_name = tp_snapshot.key;
            result.sharding_mode = tp_snapshot.mode;
            result.reference_elements = pytorch_data.size();

            if (pytorch_data.empty())
            {
                LOG_WARN("[TP Parity] No PyTorch data for stage " << tp_snapshot.key);
                return result;
            }

            const int tp_degree = static_cast<int>(tp_snapshot.device_data.size());
            if (tp_degree == 0)
            {
                LOG_WARN("[TP Parity] No device data for stage " << tp_snapshot.key);
                return result;
            }

            LOG_DEBUG("[TP Parity] Comparing stage " << tp_snapshot.key
                                                     << " mode=" << shardingModeToString(tp_snapshot.mode)
                                                     << " tp_degree=" << tp_degree
                                                     << " pytorch_size=" << pytorch_data.size()
                                                     << " (" << pytorch_rows << "x" << pytorch_cols << ")");

            // Per-device comparison for column-parallel stages
            if (tp_snapshot.mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                std::vector<size_t> participant_widths;
                participant_widths.reserve(tp_snapshot.device_data.size());
                for (const auto &device_data : tp_snapshot.device_data)
                    participant_widths.push_back(device_data.cols);

                std::vector<ParityTPColumnSlice> slices;
                try
                {
                    slices = makeParityTPColumnSlices(
                        participant_widths, pytorch_cols);
                }
                catch (const std::invalid_argument &error)
                {
                    ADD_FAILURE()
                        << "Invalid column-parallel TP checkpoint geometry for '"
                        << tp_snapshot.key << "': " << error.what();
                    return result;
                }

                for (int dev_idx = 0; dev_idx < tp_degree; ++dev_idx)
                {
                    const auto &dev_data = tp_snapshot.device_data[dev_idx];

                    // Production partitions attention at whole-head boundaries,
                    // so a non-divisible head count produces heterogeneous
                    // shard widths (for example 14 heads across four devices
                    // becomes 4/4/4/2). Derive each authenticated reference
                    // interval from the live typed snapshot geometry instead
                    // of inventing an equal-width test-only partition.
                    const auto &slice = slices[static_cast<size_t>(dev_idx)];
                    const size_t slice_start = slice.start_column;
                    const size_t slice_cols = slice.column_count;

                    // Extract PyTorch slice
                    std::vector<float> pytorch_slice = extractColumnSlice(
                        pytorch_data.data(), pytorch_rows, pytorch_cols, slice_start, slice_cols);

                    // Debug: print actual values for diagnosis - VERBOSE for ATTENTION_CONTEXT and Q_ROPE stages
                    bool debug_verbose = (tp_snapshot.key.find("ATTENTION_CONTEXT") != std::string::npos) ||
                                         (tp_snapshot.key.find("Q_ROPE") != std::string::npos && tp_snapshot.key.find("layer0") != std::string::npos);
                    if (debug_verbose)
                    {
                        LOG_INFO("[TP Debug] " << tp_snapshot.key << " device " << dev_idx
                                               << " pytorch_rows=" << pytorch_rows
                                               << " pytorch_cols=" << pytorch_cols
                                               << " slice_start=" << slice_start
                                               << " slice_cols=" << slice_cols
                                               << " pytorch_slice.size=" << pytorch_slice.size()
                                               << " dev_data.size=" << dev_data.data.size()
                                               << " dev_data.cols=" << dev_data.cols);

                        // Print first 8 values from each
                        std::stringstream ss_pt, ss_ll;
                        for (size_t i = 0; i < std::min(size_t(8), pytorch_slice.size()); ++i)
                            ss_pt << std::setprecision(6) << pytorch_slice[i] << ", ";
                        for (size_t i = 0; i < std::min(size_t(8), dev_data.data.size()); ++i)
                            ss_ll << std::setprecision(6) << dev_data.data[i] << ", ";
                        LOG_INFO("[TP Debug]   Row0 PyTorch: " << ss_pt.str());
                        LOG_INFO("[TP Debug]   Row0 Llaminar: " << ss_ll.str());

                        // Row 1 values (at offset slice_cols for Llaminar, slice_cols for pytorch_slice)
                        size_t row_stride_ll = dev_data.cols > 0 ? dev_data.cols : slice_cols;
                        size_t row_stride_pt = slice_cols;
                        if (pytorch_slice.size() > row_stride_pt + 8 && dev_data.data.size() > row_stride_ll + 8)
                        {
                            std::stringstream ss3, ss4;
                            for (size_t i = 0; i < 8; ++i)
                                ss3 << std::setprecision(6) << pytorch_slice[row_stride_pt + i] << ", ";
                            for (size_t i = 0; i < 8; ++i)
                                ss4 << std::setprecision(6) << dev_data.data[row_stride_ll + i] << ", ";
                            LOG_INFO("[TP Debug]   Row1 PyTorch (stride=" << row_stride_pt << "): " << ss3.str());
                            LOG_INFO("[TP Debug]   Row1 Llaminar (stride=" << row_stride_ll << "): " << ss4.str());
                        }
                    }

                    // Compare device data against slice
                    size_t compare_size = std::min(dev_data.data.size(), pytorch_slice.size());
                    EXPECT_EQ(dev_data.data.size(), pytorch_slice.size())
                        << "TP checkpoint '" << tp_snapshot.key << "' device "
                        << dev_idx << " must exactly match its authenticated "
                        << "PyTorch shard shape";
                    float cosine = computeCosineSimilarity(
                        dev_data.data.data(), pytorch_slice.data(), compare_size);

                    // Per-row cosine diagnostic for ATTENTION_CONTEXT
                    if (debug_verbose && slice_cols > 0)
                    {
                        size_t num_rows = compare_size / slice_cols;
                        float max_abs_diff = 0.0f;
                        size_t max_diff_idx = 0;
                        std::stringstream row_cosines;
                        for (size_t r = 0; r < num_rows && r < 9; ++r)
                        {
                            float rc = computeCosineSimilarity(
                                dev_data.data.data() + r * slice_cols,
                                pytorch_slice.data() + r * slice_cols,
                                slice_cols);
                            row_cosines << "row" << r << "=" << std::fixed << std::setprecision(6) << rc << " ";
                            // Find max absolute difference in this row
                            for (size_t c = 0; c < slice_cols; ++c)
                            {
                                float diff = std::abs(dev_data.data[r * slice_cols + c] - pytorch_slice[r * slice_cols + c]);
                                if (diff > max_abs_diff)
                                {
                                    max_abs_diff = diff;
                                    max_diff_idx = r * slice_cols + c;
                                }
                            }
                        }
                        LOG_INFO("[TP Diag] " << tp_snapshot.key << " dev" << dev_idx
                                              << " per-row cosine: " << row_cosines.str());
                        LOG_INFO("[TP Diag] " << tp_snapshot.key << " dev" << dev_idx
                                              << " max_abs_diff=" << max_abs_diff
                                              << " at idx=" << max_diff_idx
                                              << " (row=" << max_diff_idx / slice_cols
                                              << " col=" << max_diff_idx % slice_cols << ")"
                                              << " llaminar=" << dev_data.data[max_diff_idx]
                                              << " pytorch=" << pytorch_slice[max_diff_idx]);
                    }

                    TPDeviceComparisonResult dev_result;
                    dev_result.device_id = dev_data.device_id.toString();
                    dev_result.device_index = dev_idx;
                    dev_result.cosine_similarity = cosine;
                    dev_result.slice_start = slice_start;
                    dev_result.slice_size = compare_size;
                    dev_result.passed = (cosine >= config_.cosine_threshold);

                    LOG_DEBUG("[TP Parity] Device " << dev_idx << " (" << dev_result.device_id << ")"
                                                    << " slice=[" << slice_start << "," << slice_start + slice_cols << ")"
                                                    << " cosine=" << cosine
                                                    << " device_size=" << dev_data.data.size()
                                                    << " slice_size=" << pytorch_slice.size());

                    result.device_results.push_back(std::move(dev_result));
                }
            }
            else if (tp_snapshot.mode ==
                     SnapshotShardingMode::PACKED_COLUMN_PARALLEL)
            {
                /*
                 * A participant's packed row contains several disjoint
                 * reference intervals, so comparing it with one contiguous
                 * PyTorch slice would be mathematically false. The combined
                 * comparison below still covers every partitioned value, while
                 * TPSnapshot::computeCombined() separately verifies every
                 * replicated group before publishing the full row.
                 */
                LOG_DEBUG("[TP Parity] Packed checkpoint "
                          << tp_snapshot.key
                          << " is certified through typed group reassembly");
            }
            else
            {
                // For replicated/row-parallel, each device should match full PyTorch
                for (int dev_idx = 0; dev_idx < tp_degree; ++dev_idx)
                {
                    const auto &dev_data = tp_snapshot.device_data[dev_idx];

                    size_t compare_size = std::min(dev_data.data.size(), pytorch_data.size());
                    EXPECT_EQ(dev_data.data.size(), pytorch_data.size())
                        << "TP checkpoint '" << tp_snapshot.key << "' device "
                        << dev_idx << " must exactly match its authenticated "
                        << "replicated PyTorch shape";
                    float cosine = computeCosineSimilarity(
                        dev_data.data.data(), pytorch_data.data(), compare_size);

                    TPDeviceComparisonResult dev_result;
                    dev_result.device_id = dev_data.device_id.toString();
                    dev_result.device_index = dev_idx;
                    dev_result.cosine_similarity = cosine;
                    dev_result.slice_start = 0;
                    dev_result.slice_size = compare_size;
                    dev_result.passed = (cosine >= config_.cosine_threshold);

                    result.device_results.push_back(std::move(dev_result));
                }
            }

            // Compute combined result
            size_t combined_size = 0;
            const float *combined_ptr = tp_snapshot.getCombinedData(combined_size);

            if (combined_ptr && combined_size > 0)
            {
                size_t compare_size = std::min(combined_size, pytorch_data.size());
                EXPECT_EQ(combined_size, pytorch_data.size())
                    << "Combined TP checkpoint '" << tp_snapshot.key
                    << "' must exactly match the authenticated PyTorch shape";
                result.combined_result = compareParityTensorData(
                    combined_ptr,
                    pytorch_data.data(),
                    compare_size,
                    tp_snapshot.key,
                    config_.cosine_threshold);
                result.combined_cosine =
                    result.combined_result.cosine_similarity;
                result.combined_elements = compare_size;
                result.combined_passed =
                    combined_size == pytorch_data.size() &&
                    result.combined_result.passed;
                result.combined_result.passed = result.combined_passed;

                LOG_DEBUG("[TP Parity] Combined: size=" << combined_size
                                                        << " pytorch_size=" << pytorch_data.size()
                                                        << " cosine=" << result.combined_cosine);
            }
            else
            {
                LOG_WARN("[TP Parity] Failed to compute combined data for " << tp_snapshot.key);
            }

            // Overall pass: combined must pass
            result.passed = result.combined_passed;

            return result;
        }

        /**
         * @brief Setup the inference pipeline
         *
         * MPI-aware: Uses getDevice() which calls getDeviceForRank() for per-rank device selection.
         * Passes mpi_ctx_ to createInferenceRunner (nullptr for single-rank tests).
         * Uses getWeightStrategy() for weight distribution (REPLICATED or SHARDED).
         * Calls configureModel() hook for tensor-parallel schema configuration.
         */
        bool setupPipeline()
        {
            DeviceManager::instance().initialize(-1);

            // Load model with MPI context and weight strategy
            model_ctx_ = acquireParityModelContext(getWeightStrategy());
            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

            InferenceRunnerConfig inf_config;
            inf_config.max_seq_len = 4096;
            inf_config.batch_size = 1;
            inf_config.force_graph = true;
            inf_config.activation_precision = cfg().activation_precision;
            inf_config.kv_cache_precision = cfg().kv_cache_precision;

            // Enable mapped memory for GPU devices to avoid slow D2H syncs during snapshot capture
            // This works for both CUDA and ROCm - mapped memory enables zero-copy host access
            DeviceId device = getDevice();
            if (device.is_gpu())
            {
                inf_config.use_mapped_memory = true;
                if (isRank0())
                {
                    LOG_INFO("[" << getBackendName() << " Parity] Enabling mapped memory for GPU snapshot capture");
                }
            }

            // Pass mpi_ctx_ to enable tensor parallelism (nullptr for single-rank tests)
            runner_ = createInferenceRunner(model_ctx_, mpi_ctx_, device, inf_config);
            if (!runner_)
            {
                LOG_ERROR("[Parity] Failed to create inference runner");
                return false;
            }

            runner_->enableSnapshotCapture();
            if (isRank0())
            {
                LOG_INFO("[" << getBackendName() << " Parity] Inference runner created"
                             << (mpi_ctx_ ? " (MPI world_size=" + std::to_string(mpiWorldSize()) + ")" : ""));
            }
            return true;
        }

        /**
         * @brief Setup pipeline for LocalPP (Pipeline Parallelism) tests
         *
         * Creates a pipeline parallel configuration where layers are split across
         * multiple devices. Uses RankOrchestrator with PP mode which:
         * - Creates per-stage DeviceGraphOrchestrator instances
         * - Handles sequential forward execution through stages
         * - Manages activation transfer via LocalPPContext
         *
         * This is model-agnostic: any model that reports blockCount() can be used.
         * Uses cfg() to get device list, collective backend, etc.
         *
         * @return true if setup succeeded, false on error
         */
        bool setupLocalPPPipeline()
        {
            DeviceManager::instance().initialize(-1);

            // Initialize GlobalBackendRouter for activation transfers between PP stages
            GlobalBackendRouter::initForTests();

            // Load model with REPLICATED strategy (each stage needs full model access
            // for weight lookup - layer partitioning happens at runtime)
            model_ctx_ = acquireParityModelContext(
                WeightDistributionStrategy::REPLICATED);

            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

            // Get model parameters (model-agnostic)
            int n_layers = parityLayerCount();

            // Build GlobalDeviceAddress list from TestConfig
            std::vector<GlobalDeviceAddress> all_device_addresses;
            int cuda_idx = 0, rocm_idx = 0;
            for (auto dt : cfg().devices)
            {
                switch (dt)
                {
                case ParityDeviceType::CPU:
                    all_device_addresses.push_back(GlobalDeviceAddress::cpu());
                    break;
                case ParityDeviceType::CUDA:
                    all_device_addresses.push_back(GlobalDeviceAddress::cuda(cuda_idx++));
                    break;
                case ParityDeviceType::ROCm:
                    all_device_addresses.push_back(GlobalDeviceAddress::rocm(rocm_idx++));
                    break;
                }
            }

            // Determine number of PP stages:
            // - If pp_stage_sizes is set, use its length (supports hybrid PP+TP)
            // - Otherwise, assume 1 device per stage (pure PP)
            int num_stages = static_cast<int>(cfg().num_pp_stages());
            if (num_stages < 1)
                num_stages = 1;

            // Calculate layer boundaries (equal split)
            int layers_per_stage = n_layers / num_stages;

            if (cfg().is_hybrid_pp_tp())
            {
                LOG_INFO("[Parity] Hybrid PP+TP configuration: " << num_stages << " PP stages, "
                                                                 << n_layers << " layers");
                for (size_t i = 0; i < cfg().pp_stage_sizes.size(); ++i)
                {
                    LOG_INFO("[Parity]   Stage " << i << " has " << cfg().pp_stage_sizes[i]
                                                 << " device(s)" << (cfg().pp_stage_sizes[i] > 1 ? " (TP domain)" : ""));
                }
            }
            else
            {
                LOG_INFO("[Parity] LocalPP configuration: " << num_stages << " stages, "
                                                            << n_layers << " layers");
            }

            // Create RankOrchestrator::Config with PP mode (or TP_PP for hybrid)
            RankOrchestrator::Config mdo_config;
            mdo_config.mode = cfg().is_hybrid_pp_tp()
                                  ? RankOrchestrator::ParallelismMode::TP_PP
                                  : RankOrchestrator::ParallelismMode::PP;
            mdo_config.max_seq_len = 4096;
            mdo_config.batch_size = 1;
            mdo_config.activation_precision = ActivationPrecision::FP32;

            // Match single-device parity setup: GPU snapshot capture uses mapped
            // memory to avoid stage-by-stage D2H races on reused activation buffers.
            for (auto dt : cfg().devices)
            {
                if (dt == ParityDeviceType::CUDA || dt == ParityDeviceType::ROCm)
                {
                    mdo_config.use_mapped_memory = true;
                    break;
                }
            }

            // Create PP stage configurations
            // Track which device index we're at in the flat device list
            size_t device_offset = 0;

            for (int s = 0; s < num_stages; ++s)
            {
                RankOrchestrator::PPStageConfig stage_config;
                stage_config.first_layer = s * layers_per_stage;
                stage_config.last_layer = (s == num_stages - 1) ? n_layers : (s + 1) * layers_per_stage;
                stage_config.has_embedding = (s == 0);
                stage_config.has_lm_head = (s == num_stages - 1);

                // Determine how many devices this stage gets
                int stage_device_count = 1;
                if (!cfg().pp_stage_sizes.empty() && s < static_cast<int>(cfg().pp_stage_sizes.size()))
                {
                    stage_device_count = cfg().pp_stage_sizes[s];
                }

                // Add devices for this stage
                for (int d = 0; d < stage_device_count && device_offset < all_device_addresses.size(); ++d)
                {
                    stage_config.stage_devices.push_back(all_device_addresses[device_offset++]);
                }

                // If this is a TP domain (multiple devices), set TP backend
                if (stage_config.isTPDomain())
                {
                    // Use tp_collective for intra-stage TP backend
                    stage_config.tp_backend = toCollectiveBackend(cfg().tp_collective);

                    // Equal TP weights for this domain
                    for (int d = 0; d < stage_device_count; ++d)
                    {
                        stage_config.tp_weights.push_back(1.0f / static_cast<float>(stage_device_count));
                    }
                }

                // Log stage configuration
                std::ostringstream devices_str;
                for (size_t i = 0; i < stage_config.stage_devices.size(); ++i)
                {
                    if (i > 0)
                        devices_str << ", ";
                    devices_str << stage_config.stage_devices[i].toString();
                }
                LOG_INFO("[Parity]   Stage " << s << ": layers ["
                                             << stage_config.first_layer << ", " << stage_config.last_layer << ") on "
                                             << (stage_config.isTPDomain() ? "TP(" : "")
                                             << devices_str.str()
                                             << (stage_config.isTPDomain() ? ")" : ""));

                mdo_config.pp_stages.push_back(stage_config);
            }

            // Validate PP config
            if (!mdo_config.validate())
            {
                LOG_ERROR("[Parity] Invalid RankOrchestrator PP config");
                return false;
            }

            // Create RankOrchestrator with PP mode
            auto multi_orch = std::make_unique<RankOrchestrator>(model_ctx_, mdo_config);

            if (!multi_orch)
            {
                LOG_ERROR("[Parity] Failed to create RankOrchestrator for PP");
                return false;
            }

            // Enable snapshot capture for parity testing
            multi_orch->enableSnapshotCapture();

            LOG_INFO("[Parity] RankOrchestrator " << (cfg().is_hybrid_pp_tp() ? "TP_PP" : "PP")
                                                  << " created with " << num_stages << " stages");

            // Transfer ownership to base class runner_
            runner_ = std::move(multi_orch);

            return true;
        }

        // =========================================================================
        // Modern Orchestration Runner Support (for incremental migration)
        // =========================================================================

        /**
         * @brief Diagnostic graph policy applied after production runner setup.
         *
         * Most parity fixtures need checkpoint publication immediately. A
         * production-path performance phase can instead initialize with the
         * ordinary graph topology and enable diagnostic nodes only when it
         * reaches the later numerical-comparison phase. Encoding that choice
         * as a scoped policy keeps callers from relying on a positional bool.
         */
        enum class ParitySnapshotSetupMode
        {
            Enabled,
            Disabled,
        };

        /**
         * @brief Setup an OrchestrationRunner for parity testing
         *
         * This is the modern alternative to setupPipeline(). Creates an
         * OrchestrationRunner using the TestOrchestrationHelper utilities.
         *
         * Unlike setupPipeline(), this method:
         * - Uses the OrchestrationConfig/OrchestrationRunner pattern
         * - Has a simpler, more declarative API
         * - Supports all orchestration features (PP, LOCAL TP, etc.)
         *
         * @param orch_config Pre-configured OrchestrationConfig
         * @param preloaded_model_context Optional retained immutable model and
         *        prepared-weight authority; the production runner validates it
         *        against the newly built plan before graph construction
         * @param snapshot_mode Whether diagnostic checkpoint nodes are enabled
         *        immediately after initialization.
         * @return true on success, false on failure
         *
         * @note Call this OR setupPipeline(), not both. The active runner is
         *       whichever was last set up successfully.
         */
        bool setupOrchestrationRunner(
            const OrchestrationConfig &orch_config,
            std::shared_ptr<ModelContext> preloaded_model_context = nullptr,
            ParitySnapshotSetupMode snapshot_mode =
                ParitySnapshotSetupMode::Enabled)
        {
            return setupOrchestrationRunnerImpl(
                orch_config,
                std::move(preloaded_model_context),
                std::nullopt,
                snapshot_mode);
        }

        /**
         * @brief Setup from a production-certified prepared-weight authority.
         *
         * Unlike the raw ModelContext overload, this path lets the production
         * runner credit and adopt an existing PreparedWeightStore only after
         * its newly resolved rank plan and routed-weight topology match the
         * prior runner's exact certificate.
         *
         * @param orch_config Complete fresh runner configuration.
         * @param reuse_contract Model-owned weights plus production certificate.
         * @param snapshot_mode Initial diagnostic checkpoint policy.
         * @return True after the fresh production runner initializes.
         */
        bool setupOrchestrationRunner(
            const OrchestrationConfig &orch_config,
            ModelContextReuseContract reuse_contract,
            ParitySnapshotSetupMode snapshot_mode =
                ParitySnapshotSetupMode::Enabled)
        {
            return setupOrchestrationRunnerImpl(
                orch_config,
                nullptr,
                std::move(reuse_contract),
                snapshot_mode);
        }

        /**
         * @brief Install exactly one fresh production runner implementation.
         *
         * The two public overloads differ only at the model-authority boundary;
         * all teardown, initialization, and snapshot lifecycle remains one
         * implementation so campaign reuse cannot acquire special graph rules.
         */
        bool setupOrchestrationRunnerImpl(
            const OrchestrationConfig &orch_config,
            std::shared_ptr<ModelContext> preloaded_model_context,
            std::optional<ModelContextReuseContract> reuse_contract,
            ParitySnapshotSetupMode snapshot_mode)
        {
            // Install one runner/model authority.  A stale legacy context must
            // never shadow the model owned by the modern production runner.
            runner_.reset();
            borrowed_runner_ = nullptr;
            model_ctx_.reset();
            orch_runner_.reset();
            orchestration_parity_decode_trajectory_.clear();
            orchestration_parity_decode_trajectory_index_ = 0;
            orchestration_parity_force_mode_ =
                ParityForcedTokenCommitMode::Unknown;

            // Create runner via TestOrchestrationHelper
            if (reuse_contract)
            {
                orch_runner_ = test::TestOrchestrationHelper::create(
                    orch_config,
                    std::move(*reuse_contract));
            }
            else if (preloaded_model_context)
            {
                orch_runner_ = test::TestOrchestrationHelper::create(
                    orch_config,
                    std::move(preloaded_model_context));
            }
            else
            {
                orch_runner_ = test::TestOrchestrationHelper::create(orch_config);
            }
            if (!orch_runner_)
            {
                LOG_ERROR("[Parity] Failed to create OrchestrationRunner");
                return false;
            }

            if (snapshot_mode == ParitySnapshotSetupMode::Enabled)
            {
                /*
                 * Snapshot D2D nodes belong to native graph identity. Install
                 * one stable prefill/decode union before initialize() so the
                 * serving-family setup phase captures the exact production
                 * diagnostic topology. The typed inventory resolves an empty
                 * user list to the authenticated reference checkpoint set;
                 * only an explicit EveryPublishedOutput policy selects all.
                 */
                orch_runner_->setSnapshotCaptureFilter(
                    paritySnapshotSetupCaptureFilter());
                orch_runner_->enableSnapshotCapture();
            }

            // Initialize only after the complete diagnostic topology is known.
            if (!orch_runner_->initialize())
            {
                LOG_ERROR("[Parity] Failed to initialize OrchestrationRunner: " << orch_runner_->lastError());
                orch_runner_.reset();
                return false;
            }

            if (isRank0())
            {
                LOG_INFO("[" << getBackendName() << " Parity] OrchestrationRunner created and initialized");
            }
            return true;
        }

        /**
         * @brief Setup a simple single-device OrchestrationRunner
         *
         * Convenience wrapper that creates a simple OrchestrationConfig
         * and calls setupOrchestrationRunner().
         *
         * @return true on success, false on failure
         */
        bool setupSimpleOrchestrationRunner()
        {
            OrchestrationConfig config;
            config.model_path = config_.model_path;
            config.device_mode = DeviceAssignmentMode::EXPLICIT;
            config.device_for_this_rank = GlobalDeviceAddress::fromLocalDeviceId(getDevice());
            config.max_seq_len = 4096;
            // No PP, no TP - simple single-device execution
            return setupOrchestrationRunner(config);
        }

        /**
         * @brief Run forward pass with whichever runner is active
         *
         * This helper allows tests to work with either the legacy IInferenceRunner
         * or the modern IOrchestrationRunner without changing test logic.
         *
         * @param tokens Input token IDs
         * @return true on success, false on failure
         */
        bool runForward(const std::vector<int32_t> &tokens)
        {
            if (orch_runner_)
            {
                // Modern path: use prefill
                if (!orch_runner_->prefill(tokens))
                {
                    LOG_ERROR("[Parity] OrchestrationRunner prefill failed: " << orch_runner_->lastError());
                    return false;
                }
                // For single-step forward (prefill only), no decode needed
                return true;
            }
            else if (auto *runner = activeLegacyRunner())
            {
                // Legacy path: use forward()
                return runner->forward(tokens.data(), tokens.size());
            }
            LOG_ERROR("[Parity] No runner available - call setupPipeline() or setupOrchestrationRunner() first");
            return false;
        }

        /**
         * @brief Get logits from whichever runner is active
         *
         * @return Pointer to logits, or nullptr if unavailable
         */
        const float *getActiveLogits() const
        {
            if (orch_runner_)
            {
                return orch_runner_->lastLogits();
            }
            else if (auto *runner = activeLegacyRunner())
            {
                return runner->logits();
            }
            return nullptr;
        }

        /**
         * @brief Get vocabulary size from whichever runner is active
         *
         * @return Vocabulary size, or 0 if unavailable
         */
        int getActiveVocabSize() const
        {
            if (orch_runner_)
            {
                return orch_runner_->vocabSize();
            }
            else if (auto *runner = activeLegacyRunner())
            {
                return static_cast<int>(runner->vocab_size());
            }
            return 0;
        }

        /**
         * @brief Get snapshot from whichever runner is active
         *
         * @param key Snapshot identifier
         * @param out_size Output parameter for snapshot size
         * @return Pointer to snapshot data, or nullptr if not found
         */
        const float *activeSnapshot(const std::string &key, size_t &out_size) const
        {
            if (orch_runner_)
            {
                return orch_runner_->getSnapshot(key, out_size);
            }
            else if (auto *runner = activeLegacyRunner())
            {
                if (const auto *rank_orchestrator =
                        dynamic_cast<const RankOrchestrator *>(runner))
                {
                    TPSnapshot tp_snapshot = rank_orchestrator->getTPSnapshot(key);
                    size_t combined_size = 0;
                    const float *combined = tp_snapshot.getCombinedData(combined_size);
                    if (combined && combined_size > 0)
                    {
                        auto &cache = active_snapshot_combined_cache_[key];
                        cache.assign(combined, combined + combined_size);
                        out_size = cache.size();
                        return cache.data();
                    }
                }
                return runner->getSnapshot(key, out_size);
            }
            out_size = 0;
            return nullptr;
        }

        /**
         * @brief Get snapshot keys from whichever runner is active
         *
         * @return Vector of snapshot keys, empty if no runner
         */
        std::vector<std::string> activeSnapshotKeys() const
        {
            if (orch_runner_)
            {
                return orch_runner_->getSnapshotKeys();
            }
            else if (auto *runner = activeLegacyRunner())
            {
                return runner->getSnapshotKeys();
            }
            return {};
        }

        static std::string trimParityEnvToken(const std::string &input)
        {
            size_t begin = 0;
            while (begin < input.size() &&
                   std::isspace(static_cast<unsigned char>(input[begin])))
            {
                ++begin;
            }

            size_t end = input.size();
            while (end > begin &&
                   std::isspace(static_cast<unsigned char>(input[end - 1])))
            {
                --end;
            }

            return input.substr(begin, end - begin);
        }

        static std::vector<std::string> readParityEnvList(const char *name)
        {
            std::vector<std::string> values;
            const char *raw = std::getenv(name);
            if (!raw || raw[0] == '\0')
                return values;

            std::stringstream ss(raw);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                token = trimParityEnvToken(token);
                if (!token.empty())
                    values.push_back(token);
            }
            return values;
        }

        static std::set<int> readParityEnvStepSet(const char *name)
        {
            std::set<int> steps;
            for (const std::string &token : readParityEnvList(name))
            {
                const size_t dash = token.find('-', token.front() == '-' ? 1 : 0);
                if (dash != std::string::npos)
                {
                    const int begin = std::stoi(token.substr(0, dash));
                    const int end = std::stoi(token.substr(dash + 1));
                    const int lo = std::min(begin, end);
                    const int hi = std::max(begin, end);
                    for (int value = lo; value <= hi; ++value)
                        steps.insert(value);
                    continue;
                }
                steps.insert(std::stoi(token));
            }
            return steps;
        }

        static size_t readParityEnvSize(const char *name, size_t default_value)
        {
            const char *raw = std::getenv(name);
            if (!raw || raw[0] == '\0')
                return default_value;

            char *end = nullptr;
            const unsigned long long value = std::strtoull(raw, &end, 10);
            if (end == raw)
                return default_value;
            return static_cast<size_t>(value);
        }

        static bool paritySnapshotKeyMatches(
            const std::string &key,
            const std::vector<std::string> &filters)
        {
            if (filters.empty())
                return false;

            return std::any_of(
                filters.begin(),
                filters.end(),
                [&key](const std::string &filter)
                {
                    return key.find(filter) != std::string::npos;
                });
        }

        static std::string csvEscape(const std::string &value)
        {
            const bool needs_quotes =
                value.find_first_of(",\"\n\r") != std::string::npos;
            if (!needs_quotes)
                return value;

            std::string escaped;
            escaped.reserve(value.size() + 2);
            escaped.push_back('"');
            for (char c : value)
            {
                if (c == '"')
                    escaped.push_back('"');
                escaped.push_back(c);
            }
            escaped.push_back('"');
            return escaped;
        }

        void exportActiveSnapshotsForParityDiagnostics(
            ParityForwardPhase phase,
            int decode_step)
        {
            if (!isRank0())
                return;

            const auto filters =
                readParityEnvList("LLAMINAR_PARITY_EXPORT_SNAPSHOT_KEYS");
            if (filters.empty())
                return;

            if (phase == ParityForwardPhase::Decode)
            {
                const auto allowed_steps =
                    readParityEnvStepSet("LLAMINAR_PARITY_EXPORT_SNAPSHOT_STEPS");
                if (!allowed_steps.empty() &&
                    allowed_steps.count(decode_step) == 0)
                {
                    return;
                }
            }

            const size_t max_elements =
                readParityEnvSize("LLAMINAR_PARITY_EXPORT_SNAPSHOT_MAX_ELEMENTS", 0);

            auto out_dir = ensureResultsDir() / "llaminar_snapshots";
            std::error_code ec;
            std::filesystem::create_directories(out_dir, ec);
            if (ec)
            {
                LOG_WARN("[Parity Snapshot Export] Failed to create " << out_dir
                                                                      << ": " << ec.message());
                return;
            }

            auto manifest_path = out_dir / "manifest.csv";
            ec.clear();
            const bool manifest_exists = std::filesystem::exists(manifest_path, ec);
            ec.clear();
            const bool write_header =
                !manifest_exists ||
                std::filesystem::file_size(manifest_path, ec) == 0;
            std::ofstream manifest(manifest_path, std::ios::app);
            if (!manifest.is_open())
            {
                LOG_WARN("[Parity Snapshot Export] Cannot write " << manifest_path);
                return;
            }
            if (write_header)
            {
                manifest << "backend,phase,step,key,elements,exported_elements,file\n";
            }

            const std::string phase_name = parityForwardPhaseName(phase);
            const std::string step_name =
                (phase == ParityForwardPhase::Decode)
                    ? ("step" + std::to_string(decode_step))
                    : "prefill";

            for (const auto &key : activeSnapshotKeys())
            {
                if (!paritySnapshotKeyMatches(key, filters))
                    continue;

                size_t elements = 0;
                const float *data = activeSnapshot(key, elements);
                if (!data || elements == 0)
                    continue;

                const size_t exported_elements =
                    (max_elements > 0) ? std::min(elements, max_elements) : elements;
                const std::string safe_key = sanitizeSnapshotToken(key);
                auto file_path = out_dir /
                                 (phase_name + "_" + step_name + "_" + safe_key + ".f32");

                std::ofstream out(file_path, std::ios::binary);
                if (!out.is_open())
                {
                    LOG_WARN("[Parity Snapshot Export] Cannot write " << file_path);
                    continue;
                }

                out.write(
                    reinterpret_cast<const char *>(data),
                    static_cast<std::streamsize>(exported_elements * sizeof(float)));
                if (!out.good())
                {
                    LOG_WARN("[Parity Snapshot Export] Failed while writing " << file_path);
                    continue;
                }

                manifest << csvEscape(getBackendName()) << ","
                         << phase_name << ","
                         << decode_step << ","
                         << csvEscape(key) << ","
                         << elements << ","
                         << exported_elements << ","
                         << csvEscape(file_path.string()) << "\n";
            }
        }

        /**
         * @brief Clear KV cache on whichever runner is active
         */
        void activeClearCache()
        {
            orchestration_parity_decode_trajectory_.clear();
            orchestration_parity_decode_trajectory_index_ = 0;
            orchestration_parity_force_mode_ =
                ParityForcedTokenCommitMode::Unknown;
            if (orch_runner_)
            {
                orch_runner_->clearCache();
            }
            else if (auto *runner = activeLegacyRunner())
            {
                runner->clear_cache();
            }
        }

        /**
         * @brief Clear snapshots on whichever runner is active
         */
        void activeClearSnapshots()
        {
            active_snapshot_combined_cache_.clear();
            if (orch_runner_)
            {
                orch_runner_->clearSnapshots();
            }
            else if (auto *runner = activeLegacyRunner())
            {
                runner->clearSnapshots();
            }
        }

        void activeSetSnapshotCaptureFilter(const std::vector<std::string> &keys)
        {
            if (orch_runner_)
            {
                orch_runner_->setSnapshotCaptureFilter(keys);
            }
            else if (auto *runner = activeLegacyRunner())
            {
                runner->setSnapshotCaptureFilter(keys);
            }
        }

        PrefixRuntimeStateSnapshot activePrefixStateProbe() const
        {
            if (orch_runner_)
                return orch_runner_->prefixStateProbe();
            if (auto *runner = activeLegacyRunner())
                return runner->prefixStateProbe();
            return {};
        }

        DeviceId activePrimaryDevice() const
        {
            if (orch_runner_)
                return orch_runner_->primaryDeviceId();
            if (auto *runner = activeLegacyRunner())
                return runner->primaryDeviceId();
            return DeviceId::invalid();
        }

        ExecutionPath activeExecutionPath() const
        {
            if (auto *runner = activeLegacyRunner())
                return runner->executionPath();
            return ExecutionPath::GRAPH;
        }

        /**
         * @brief Return whether this test is the fused live-production gate.
         *
         * The environment bit is attached only to `ProductionParity` CTest
         * registrations.  Legacy focused tests retain their existing isolation
         * and diagnostics while the campaign gate enforces graph contracts and
         * reports timing in addition to the same numerical comparisons.
         */
        bool productionParityCampaignEnabled() const
        {
            return production_parity_campaign_active_ ||
                   DebugEnv::isTruthyEnv("LLAMINAR_PRODUCTION_PARITY");
        }

        /**
         * @brief Resolve the soft whole-matrix wall-time target for reporting.
         *
         * Invalid, non-finite, and non-positive overrides are fatal test
         * configuration errors rather than reasons to omit economy evidence.
         */
        double productionParityTargetSeconds() const
        {
            return productionParityTargetSecondsFromEnvironment();
        }

        /**
         * @brief Resolve the graph contract from the complete participant set.
         *
         * The typed result prevents accelerator-bearing heterogeneous cells
         * from falling through the homogeneous-GPU branch and being certified
         * by declarative graph entry alone.
         */
        ProductionParityExecutionTopology
        productionParityExecutionTopology() const
        {
            std::size_t cpu_count = 0;
            std::size_t cuda_count = 0;
            std::size_t rocm_count = 0;
            for (const ParityDeviceType device : cfg().devices)
            {
                switch (device)
                {
                case ParityDeviceType::CPU:
                    ++cpu_count;
                    break;
                case ParityDeviceType::CUDA:
                    ++cuda_count;
                    break;
                case ParityDeviceType::ROCm:
                    ++rocm_count;
                    break;
                }
            }
            return classifyProductionParityExecutionTopology(
                cpu_count,
                cuda_count,
                rocm_count);
        }

        /**
         * @brief Resolve this rank's graph proof after topology binding.
         *
         * `isRank0()` names the inventory-resolved continuation/artifact
         * authority, not a hard-coded MPI rank. Moving accelerators between
         * sockets therefore moves the segmented-coordinator obligation with
         * the production authority.
         */
        ProductionParityGraphContract productionParityGraphContract() const
        {
            return resolveProductionParityGraphContract(
                productionParityExecutionTopology(),
                isRank0(),
                activePrimaryDevice().is_gpu());
        }

        static bool productionParityHasCounter(
            const std::vector<PerfStatRecord> &records,
            const std::string &name)
        {
            return parityForwardGraphHasCounter(records, name);
        }

        static bool productionParityHasDecodeGraphPhase(
            const std::vector<PerfStatRecord> &records,
            const std::string &capture_phase)
        {
            return parityForwardGraphHasDecodePhase(records, capture_phase);
        }

        /**
         * @brief Reset request-scoped structured evidence before campaign replay.
         */
        void beginProductionParityEvidence()
        {
            production_parity_campaign_active_ = true;
            production_parity_decode_prefill_state_.reset();
            production_parity_decode_boundaries_.clear();
            production_parity_mtp_transaction_boundaries_.clear();
            production_parity_prefix_restore_evidence_.clear();

            /*
             * Prefix replay is a byte-state contract, not merely a cache-hit
             * counter. These diagnostic-only probes hash exact KV and terminal
             * payloads and materialize device logical positions at the explicit
             * parity result boundary. GDN device payloads remain outside the
             * universal probe because copying complete recurrent banks in every
             * matrix cell would dominate campaign runtime; GDN-capable suites
             * retain their focused device-state proof.
             */
            setScopedParityEnvOverride(
                "LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1");
            if (config_.moe_movement_expectation ==
                ParityMoEMovementExpectation::PhysicalMovement)
            {
                /*
                 * A dynamic placement cell can recompute its one uncached row
                 * on a different device type.  Retain only that bounded row;
                 * the leading digest remains the exact cached-prefix proof.
                 * Static and non-overlay cells pay no extra segment export.
                 */
                setScopedParityEnvOverride(
                    "LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS", "1");
                setScopedParityEnvOverride(
                    "LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT",
                    std::to_string(config_.token_ids.size()));
                setScopedParityEnvOverride(
                    "LLAMINAR_PREFIX_PROBE_KV_SEGMENTS",
                    "recomputed_suffix=" +
                        std::to_string(config_.token_ids.size()) + ":1");
                setScopedParityEnvOverride(
                    "LLAMINAR_PREFIX_PROBE_CAPTURE_KV_SEGMENT_PAYLOADS", "1");
            }
            setScopedParityEnvOverride(
                "LLAMINAR_PREFIX_PROBE_HASH_TERMINAL_STATE", "1");
            setScopedParityEnvOverride(
                "LLAMINAR_PREFIX_PROBE_CAPTURE_TERMINAL_HIDDEN_VALUES",
                "1");
            setScopedParityEnvOverride(
                "LLAMINAR_PREFIX_PROBE_CAPTURE_TERMINAL_LOGITS_VALUES",
                "1");
            setScopedParityEnvOverride(
                "LLAMINAR_PREFIX_PROBE_CAPTURE_DEVICE_LOGICAL_STATE", "1");
            PerfStatsCollector::reset();
        }

        /** @return Deep state captured immediately after decode's prefix prefill. */
        const std::optional<PrefixRuntimeStateSnapshot> &
        productionParityDecodePrefillState() const noexcept
        {
            return production_parity_decode_prefill_state_;
        }

        /** @return Typed state boundary for every authenticated decode row. */
        const std::vector<ProductionParityDecodeBoundary> &
        productionParityDecodeBoundaries() const noexcept
        {
            return production_parity_decode_boundaries_;
        }

        /** @return Every actual predictor/verifier transaction in this cell. */
        const std::vector<ProductionParityMTPTransactionBoundary> &
        productionParityMTPTransactionBoundaries() const noexcept
        {
            return production_parity_mtp_transaction_boundaries_;
        }

        /**
         * @brief Retain one prefix proof row until the campaign artifact flush.
         *
         * @param evidence Fully evaluated lifecycle and numerical evidence.
         */
        void recordProductionParityPrefixRestoreEvidence(
            PrefixRestoreEvidence evidence)
        {
            production_parity_prefix_restore_evidence_.push_back(
                std::move(evidence));
        }

        /**
         * @brief Install the mandatory production prefix policy for one cell.
         *
         * The cache implementation and bounded RAM/device/disk budgets remain
         * production defaults. Only block geometry is narrowed for a cheap
         * strict partial-hit proof, and durable files are isolated per test and
         * MPI rank so an earlier campaign cannot turn the fresh phase into a hit.
         *
         * @param config Mutable server-facing runner configuration.
         */
        void applyProductionParityPrefixRestorePolicy(
            OrchestrationConfig &config) const
        {
            config.prefix_cache.enabled = true;
            config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
            config.prefix_cache.block_size =
                kProductionParityPrefixRestoreProofBlockSize;
            config.prefix_cache.terminal_state =
                PrefixCacheTerminalStateMode::Auto;
            config.prefix_cache.disk_dir =
                (ensureResultsDir() /
                 ("prefix-cache-rank-" + std::to_string(mpiRank())))
                    .string();
        }

        /**
         * @brief Prove the first authenticated request seeded prefix storage.
         *
         * @param state Deep runtime snapshot captured after fresh prefill.
         * @param checkpoint_passed Whether full Hugging Face prefill parity passed.
         * @param checkpoint_cosine Fresh prefill LM-head cosine.
         */
        void assertProductionParityFreshPrefixSeed(
            const PrefixRuntimeStateSnapshot &state,
            bool checkpoint_passed,
            float checkpoint_cosine)
        {
            const bool lifecycle_passed =
                state.prefix_cache_config_enabled &&
                state.prefix_cache_ready &&
                state.prefix_request.enabled &&
                !state.prefix_request.hit &&
                !state.prefix_request.partial_hit &&
                state.prefix_request.requested_tokens ==
                    static_cast<int>(config_.token_ids.size()) &&
                state.prefix_request.matched_tokens == 0 &&
                state.prefix_cache_misses >= 1u &&
                state.prefix_cache_inserts >= 1u;
            recordProductionParityPrefixRestoreEvidence({
                .phase = PrefixRestoreProofPhase::FreshSeed,
                .cache_config_enabled = state.prefix_cache_config_enabled,
                .cache_ready = state.prefix_cache_ready,
                .cache_bypassed = state.prefix_cache_bypassed,
                .cache_bypass_reason = state.prefix_cache_bypass_reason,
                .request = state.prefix_request,
                .current_position = state.current_position,
                .observed_moe_movement_epoch =
                    state.moe_runtime_movement_epoch,
                .observed_payload =
                    PrefixRestoreEvidence::TerminalPayloadIdentity::fromSnapshot(
                        state),
                .checkpoint_compared = true,
                .checkpoint_cosine = checkpoint_cosine,
                .checkpoint_passed = checkpoint_passed,
                .passed = lifecycle_passed && checkpoint_passed,
            });

            ASSERT_TRUE(state.prefix_cache_config_enabled)
                << "Production parity must enable prefix restore";
            ASSERT_TRUE(state.prefix_cache_ready)
                << state.prefix_cache_bypass_reason;
            EXPECT_TRUE(state.prefix_request.enabled);
            EXPECT_FALSE(state.prefix_request.hit);
            EXPECT_FALSE(state.prefix_request.partial_hit);
            EXPECT_EQ(
                state.prefix_request.requested_tokens,
                static_cast<int>(config_.token_ids.size()));
            EXPECT_EQ(state.prefix_request.matched_tokens, 0);
            EXPECT_GE(state.prefix_cache_misses, 1u);
            EXPECT_GE(state.prefix_cache_inserts, 1u)
                << "Fresh prefill did not publish reusable prefix blocks";
        }

        /**
         * @brief Prove a complete hit restored the exact fresh execution state.
         *
         * @param fresh_state Byte-hashed state after initial prefill.
         * @param checkpoint_passed Whether restored-state decode parity passed.
         * @param checkpoint_cosine Mean restored-state decode cosine.
         */
        void assertProductionParityCompletePrefixRestore(
            const PrefixRuntimeStateSnapshot &fresh_state,
            bool checkpoint_passed,
            float checkpoint_cosine)
        {
            ASSERT_TRUE(productionParityDecodePrefillState().has_value())
                << "Decode parity did not retain its prefix-prefill boundary";
            const auto &restored = *productionParityDecodePrefillState();
            ASSERT_TRUE(restored.prefix_cache_ready)
                << restored.prefix_cache_bypass_reason;
            EXPECT_TRUE(restored.prefix_request.enabled);
            EXPECT_TRUE(restored.prefix_request.hit);
            EXPECT_FALSE(restored.prefix_request.partial_hit);
            EXPECT_EQ(
                restored.prefix_request.matched_tokens,
                static_cast<int>(config_.token_ids.size()));
            EXPECT_TRUE(restored.prefix_request.terminal_logits_restored);

            const MTPStateValidationResult state_match =
                compareMTPRuntimeStateSnapshots(fresh_state, restored);
            const bool lifecycle_passed =
                restored.prefix_cache_ready &&
                restored.prefix_request.enabled &&
                restored.prefix_request.hit &&
                !restored.prefix_request.partial_hit &&
                restored.prefix_request.matched_tokens ==
                    static_cast<int>(config_.token_ids.size()) &&
                restored.prefix_request.terminal_logits_restored;
            recordProductionParityPrefixRestoreEvidence({
                .phase = PrefixRestoreProofPhase::CompleteHit,
                .cache_config_enabled = restored.prefix_cache_config_enabled,
                .cache_ready = restored.prefix_cache_ready,
                .cache_bypassed = restored.prefix_cache_bypassed,
                .cache_bypass_reason = restored.prefix_cache_bypass_reason,
                .request = restored.prefix_request,
                .current_position = restored.current_position,
                .state_compared = true,
                .state_equivalent = static_cast<bool>(state_match),
                .state_detail = state_match.reason,
                .observed_moe_movement_epoch =
                    restored.moe_runtime_movement_epoch,
                .oracle_moe_movement_epoch =
                    fresh_state.moe_runtime_movement_epoch,
                .observed_payload =
                    PrefixRestoreEvidence::TerminalPayloadIdentity::fromSnapshot(
                        restored),
                .oracle_payload =
                    PrefixRestoreEvidence::TerminalPayloadIdentity::fromSnapshot(
                        fresh_state),
                .checkpoint_compared = true,
                .checkpoint_cosine = checkpoint_cosine,
                .checkpoint_passed = checkpoint_passed,
                .passed = lifecycle_passed &&
                          static_cast<bool>(state_match) &&
                          checkpoint_passed,
            });
            EXPECT_TRUE(state_match) << state_match.reason;
        }

        /**
         * @brief Payload contract for a complete current-fingerprint restore.
         *
         * Ordinary decode needs the main-model KV, hybrid state, and terminal
         * logits. An MTP oracle additionally needs the shifted sidecar state
         * and terminal hidden row. Naming that distinction prevents callers
         * from inferring the required payload from a backend or test name.
         */
        enum class ProductionParityCompletePrefixPayload : std::uint8_t
        {
            MainModel,      ///< Restore the ordinary main-model request state.
            MainModelAndMTP ///< Restore main-model and shifted-MTP state.
        };

        /**
         * @brief Completed cache boundary owned by the current fingerprint.
         */
        struct ProductionParityCompletePrefixBoundary
        {
            PrefixRuntimeStateSnapshot state; ///< Deep state after the hit.
            bool seeded_current_fingerprint = false; ///< A miss first seeded it.
        };

        /**
         * @brief Establish one complete restore under the current cache key.
         *
         * Dynamic ExpertOverlay may publish a new placement between two
         * independent parity phases. `InvalidateOnRebalance` intentionally
         * gives that placement a new cache fingerprint, so a proof cannot
         * assume that an entry seeded by an earlier phase still names the
         * current epoch. This transition admits at most one seed request and
         * then requires the immediately following request to be a complete
         * production hit. A second miss is a hard lifecycle failure rather
         * than permission to retry, disable cache, or execute an eager path.
         *
         * Snapshot validation is deliberately disabled for these two
         * admissions: a complete hit launches no model graph by construction,
         * while the canonical prefill proof has already certified the captured
         * compute path. The subsequent decode/MTP proof re-enables its normal
         * graph and snapshot policy.
         *
         * @param prompt Exact authenticated prompt for this proof transaction.
         * @param payload Required persistent payload family.
         * @param error Receives a precise lifecycle diagnostic on failure.
         * @return Complete current-fingerprint boundary, or nullopt on failure.
         */
        std::optional<ProductionParityCompletePrefixBoundary>
        restoreProductionParityCompletePrefixAtCurrentFingerprint(
            const std::vector<int32_t> &prompt,
            ProductionParityCompletePrefixPayload payload,
            std::string *error)
        {
            const auto fail = [error](std::string message)
                -> std::optional<ProductionParityCompletePrefixBoundary>
            {
                if (error)
                    *error = std::move(message);
                return std::nullopt;
            };
            if (!orch_runner_)
            {
                return fail(
                    "current-fingerprint prefix restore requires the production runner");
            }
            if (prompt.empty())
            {
                return fail(
                    "current-fingerprint prefix restore requires a non-empty prompt");
            }

            auto restore_policy =
                parityGraphSnapshotPolicy(ParityForwardPhase::Prefill);
            restore_policy.require_graph_execution_on_gpu = false;
            restore_policy.require_snapshot_publication = false;
            restore_policy.require_prefill_graph_capture_on_gpu = false;
            restore_policy.retry_prefill_after_warmup_for_capture = false;

            const auto complete = [&](const PrefixRuntimeStateSnapshot &state)
            {
                const bool common =
                    state.prefix_cache_ready &&
                    !state.prefix_cache_bypassed &&
                    state.prefix_request.enabled &&
                    !state.prefix_request.bypassed &&
                    state.prefix_request.hit &&
                    !state.prefix_request.partial_hit &&
                    state.prefix_request.matched_tokens ==
                        static_cast<int>(prompt.size()) &&
                    state.prefix_request.terminal_logits_restored;
                if (!common)
                    return false;
                return payload ==
                           ProductionParityCompletePrefixPayload::MainModel ||
                       (state.prefix_request.terminal_hidden_restored &&
                        state.prefix_request.mtp_state_restored);
            };

            std::optional<std::uint64_t> seed_movement_epoch;
            for (int admission = 0; admission < 2; ++admission)
            {
                activeClearSnapshots();
                activeClearCache();
                if (!runParityForwardWithPolicy(
                        ParityForwardPhase::Prefill,
                        prompt.data(),
                        static_cast<int>(prompt.size()),
                        restore_policy))
                {
                    return fail(
                        "current-fingerprint prefix admission failed: " +
                        orch_runner_->lastError());
                }

                PrefixRuntimeStateSnapshot state = activePrefixStateProbe();
                if (seed_movement_epoch &&
                    state.moe_runtime_movement_epoch !=
                        *seed_movement_epoch)
                {
                    return fail(
                        "ExpertOverlay movement advanced while establishing one current-fingerprint prefix boundary");
                }
                if (complete(state))
                {
                    return ProductionParityCompletePrefixBoundary{
                        .state = std::move(state),
                        .seeded_current_fingerprint = admission != 0,
                    };
                }
                if (admission != 0)
                {
                    return fail(
                        "current-fingerprint seed was not followed by one complete prefix restore");
                }
                if (!state.prefix_cache_ready ||
                    state.prefix_cache_bypassed ||
                    !state.prefix_request.enabled ||
                    state.prefix_request.bypassed)
                {
                    return fail(
                        "current-fingerprint seed request did not own an enabled prefix cache: " +
                        (state.prefix_request.bypass_reason.empty()
                             ? state.prefix_cache_bypass_reason
                             : state.prefix_request.bypass_reason));
                }
                seed_movement_epoch = state.moe_runtime_movement_epoch;
            }

            return fail(
                "current-fingerprint prefix restore exhausted its typed admission lifecycle");
        }

        /**
         * @brief Prove cached prompt plus one authenticated token is partial.
         *
         * Ordinary decode has already certified every stage for the same
         * prompt-plus-token boundary. This replay requires exact persistent
         * state while physical placement is unchanged. If production moved an
         * expert between device types in the meantime, the complete retained
         * terminal-hidden and terminal-logits rows must instead satisfy the
         * same strict numerical contract as the real-weight MTP checkpoints;
         * all other state remains exact. The recomputed live LM-head row is
         * independently compared to Hugging Face in either case.
         */
        void assertProductionParityPartialPrefixRestore()
        {
            const std::vector<int> decode_tokens =
                readDecodeTokensFromMetadata();
            ASSERT_FALSE(decode_tokens.empty())
                << "Partial prefix proof requires one authenticated decode token";

            std::vector<int32_t> prompt(
                config_.token_ids.begin(), config_.token_ids.end());
            std::string restore_error;
            const auto complete_prefix =
                restoreProductionParityCompletePrefixAtCurrentFingerprint(
                    prompt,
                    config_.mtp_expectation == ParityMTPExpectation::Disabled
                        ? ProductionParityCompletePrefixPayload::MainModel
                        : ProductionParityCompletePrefixPayload::MainModelAndMTP,
                    &restore_error);
            ASSERT_TRUE(complete_prefix.has_value())
                << "Could not establish the partial-prefix oracle boundary: "
                << restore_error;

            activeClearSnapshots();
            const int first_decode_token = decode_tokens.front();
            ASSERT_TRUE(runParityForward(
                ParityForwardPhase::Decode,
                &first_decode_token,
                1))
                << "Current-fingerprint partial-prefix oracle decode failed";

            std::optional<int32_t> pending_condition_token;
            if (orchestration_parity_force_mode_ ==
                ParityForcedTokenCommitMode::Deferred)
            {
                ASSERT_LT(
                    orchestration_parity_decode_trajectory_index_,
                    orchestration_parity_decode_trajectory_.size())
                    << "Deferred partial-prefix oracle has no pending condition";
                pending_condition_token =
                    orchestration_parity_decode_trajectory_[
                        orchestration_parity_decode_trajectory_index_];
            }

            std::vector<int> extended_prompt = config_.token_ids;
            extended_prompt.push_back(decode_tokens.front());
            const int expected_position =
                static_cast<int>(extended_prompt.size());
            const ProductionParityDecodeBoundary oracle{
                .reference_step = 0u,
                .committed_token = first_decode_token,
                .pending_condition_token = pending_condition_token,
                .expected_current_position = expected_position,
                .runtime_state = activePrefixStateProbe(),
            };
            ASSERT_EQ(oracle.reference_step, 0u);
            ASSERT_EQ(oracle.committed_token, decode_tokens.front());
            ASSERT_EQ(oracle.expected_current_position, expected_position);
            ASSERT_EQ(
                oracle.runtime_state.current_position,
                oracle.expected_current_position)
                << "Authenticated decode boundary paired a committed token "
                   "with stale logical state";

            activeClearSnapshots();
            activeClearCache();
            ASSERT_TRUE(runParityForward(
                ParityForwardPhase::Prefill,
                extended_prompt.data(),
                static_cast<int>(extended_prompt.size())))
                << "Partial prefix-restore prefill failed";

            /*
             * Reuse the existing opt-in binary snapshot exporter at the exact
             * restored-prefix bridge boundary.  Decode step zero is the
             * authenticated serial-row oracle for this state; the reserved
             * negative step keeps the bridge payload distinct without adding
             * another artifact mechanism or any default campaign overhead.
             */
            constexpr int kPartialPrefixRestoreDiagnosticStep = -2;
            exportActiveSnapshotsForParityDiagnostics(
                ParityForwardPhase::Decode,
                kPartialPrefixRestoreDiagnosticStep);

            if (oracle.pending_condition_token.has_value())
            {
                /*
                 * A deferred CPU runner forwards token N only when token N+1
                 * is selected.  Consequently the authenticated decode oracle
                 * at position N owns token N+1 as a pending condition and, for
                 * MTP, has already appended its shifted sidecar row.  A
                 * partial prefix replay has recomputed token N but has not yet
                 * received token N+1.  Submit that same authenticated pending
                 * condition through the serving API before comparing state;
                 * this changes no main-model row and aligns the two typed
                 * lifecycle boundaries exactly.
                 */
                ASSERT_NE(orch_runner_.get(), nullptr)
                    << "Deferred partial-prefix alignment requires the production runner";
                const PrefixRuntimeStateSnapshot before_pending_condition =
                    activePrefixStateProbe();
                GenerationResult staged = orch_runner_->forceDecodeToken(
                    *oracle.pending_condition_token);
                ASSERT_TRUE(staged.success()) << staged.error;
                ASSERT_EQ(staged.tokens.size(), 1u);
                ASSERT_EQ(
                    staged.tokens.front(),
                    *oracle.pending_condition_token);
                ASSERT_EQ(
                    staged.returned_token_commit,
                    ReturnedTokenCommitState::Pending)
                    << "Partial-prefix alignment unexpectedly advanced a main-model row";
                const PrefixRuntimeStateSnapshot after_pending_condition =
                    activePrefixStateProbe();
                ASSERT_EQ(
                    after_pending_condition.current_position,
                    before_pending_condition.current_position)
                    << "Staging the deferred condition mutated the restored main-model boundary";
            }

            const PrefixRuntimeStateSnapshot restored =
                activePrefixStateProbe();
            ASSERT_TRUE(restored.prefix_cache_ready)
                << restored.prefix_cache_bypass_reason;
            EXPECT_FALSE(restored.prefix_request.hit);
            EXPECT_TRUE(restored.prefix_request.partial_hit);
            EXPECT_EQ(
                restored.prefix_request.matched_tokens,
                static_cast<int>(config_.token_ids.size()));
            EXPECT_EQ(restored.current_position, expected_position);
            EXPECT_FALSE(restored.prefix_request.terminal_logits_restored)
                << "A partial hit must recompute the uncached suffix";

            MTPRuntimeSnapshotComparisonOptions state_options;
            state_options.main_kv_payload_policy =
                MTPMainKVPayloadComparisonPolicy::
                    ExactPrefixNumericalSuffixAfterMoEPlacementChange;
            state_options.main_kv_exact_prefix_tokens =
                static_cast<int>(config_.token_ids.size());
            state_options.main_kv_suffix_min_cosine = std::max(
                static_cast<double>(config_.decode_cosine_threshold),
                kMinimumProductionRecursiveMTPAggregateCosine);
            state_options.terminal_hidden_policy =
                MTPTerminalPayloadComparisonPolicy::
                    ExactUnlessMoEPlacementChanged;
            state_options.terminal_hidden_min_cosine = std::max(
                static_cast<double>(config_.decode_cosine_threshold),
                kMinimumProductionRecursiveMTPAggregateCosine);
            state_options.terminal_logits_policy =
                MTPTerminalPayloadComparisonPolicy::
                    ExactUnlessMoEPlacementChanged;
            state_options.terminal_logits_min_cosine =
                state_options.terminal_hidden_min_cosine;
            const MTPStateValidationResult state_match =
                compareMTPRuntimeStateSnapshots(
                    oracle.runtime_state,
                    restored,
                    state_options);
            EXPECT_TRUE(state_match) << state_match.reason;

            struct PartialCheckpointProof
            {
                int32_t advertised = 0;
                int32_t compared = 0;
                int32_t passed = 0;
                float cosine = 0.0f;
                float rel_l2 = 0.0f;
                float max_abs = 0.0f;
            };

            PartialCheckpointProof local_checkpoint;
            const auto local_snapshot_keys = activeSnapshotKeys();
            local_checkpoint.advertised =
                std::find(
                    local_snapshot_keys.begin(),
                    local_snapshot_keys.end(),
                    "LM_HEAD") != local_snapshot_keys.end()
                    ? 1
                    : 0;
            if (local_checkpoint.advertised != 0)
            {
                size_t live_logits_size = 0;
                const float *live_logits =
                    activeSnapshot("LM_HEAD", live_logits_size);
                std::vector<float> reference_logits =
                    loadPyTorchSnapshot("decode_step0_LM_HEAD");
                const bool valid_geometry =
                    live_logits != nullptr &&
                    live_logits_size > 0u &&
                    !reference_logits.empty() &&
                    reference_logits.size() >= live_logits_size &&
                    reference_logits.size() % live_logits_size == 0u;
                EXPECT_TRUE(valid_geometry)
                    << "Partial prefix LM-head owner published an invalid checkpoint geometry";
                if (valid_geometry)
                {
                    const float *reference_tail =
                        reference_logits.data() +
                        (reference_logits.size() - live_logits_size);
                    const StageComparisonResult logits =
                        compareParityTensorData(
                            live_logits,
                            reference_tail,
                            live_logits_size,
                            "PREFIX_PARTIAL_LM_HEAD",
                            config_.decode_cosine_threshold);
                    local_checkpoint.compared = 1;
                    local_checkpoint.passed = logits.passed ? 1 : 0;
                    local_checkpoint.cosine = logits.cosine_similarity;
                    local_checkpoint.rel_l2 = logits.rel_l2_norm;
                    local_checkpoint.max_abs = logits.max_abs_diff;
                }
            }

            std::vector<PartialCheckpointProof> checkpoint_proofs{
                local_checkpoint};
            if (config_.uses_cross_rank_pipeline)
            {
                ASSERT_NE(mpi_ctx_, nullptr);
                checkpoint_proofs.resize(
                    static_cast<size_t>(mpiWorldSize()));
                parityCoordinationMPIContext()->allgather_bytes(
                    &local_checkpoint,
                    checkpoint_proofs.data(),
                    sizeof(PartialCheckpointProof));
            }

            const int advertised_checkpoints = std::count_if(
                checkpoint_proofs.begin(), checkpoint_proofs.end(),
                [](const PartialCheckpointProof &proof)
                { return proof.advertised != 0; });
            const int compared_checkpoints = std::count_if(
                checkpoint_proofs.begin(), checkpoint_proofs.end(),
                [](const PartialCheckpointProof &proof)
                { return proof.compared != 0; });
            EXPECT_GT(advertised_checkpoints, 0)
                << "Partial prefix prefill published no LM_HEAD checkpoint on any pipeline participant";
            EXPECT_EQ(compared_checkpoints, advertised_checkpoints)
                << "At least one LM-head participant could not compare its checkpoint";

            float checkpoint_cosine = 1.0f;
            float checkpoint_rel_l2 = 0.0f;
            float checkpoint_max_abs = 0.0f;
            bool checkpoint_passed = compared_checkpoints > 0;
            for (const auto &proof : checkpoint_proofs)
            {
                if (proof.compared == 0)
                    continue;
                checkpoint_cosine =
                    std::min(checkpoint_cosine, proof.cosine);
                checkpoint_rel_l2 =
                    std::max(checkpoint_rel_l2, proof.rel_l2);
                checkpoint_max_abs =
                    std::max(checkpoint_max_abs, proof.max_abs);
                checkpoint_passed =
                    checkpoint_passed && proof.passed != 0;
            }
            const bool lifecycle_passed =
                restored.prefix_cache_ready &&
                restored.prefix_request.enabled &&
                !restored.prefix_request.hit &&
                restored.prefix_request.partial_hit &&
                restored.prefix_request.matched_tokens ==
                    static_cast<int>(config_.token_ids.size()) &&
                restored.current_position == expected_position &&
                !restored.prefix_request.terminal_logits_restored;
            recordProductionParityPrefixRestoreEvidence({
                .phase = PrefixRestoreProofPhase::PartialHit,
                .cache_config_enabled = restored.prefix_cache_config_enabled,
                .cache_ready = restored.prefix_cache_ready,
                .cache_bypassed = restored.prefix_cache_bypassed,
                .cache_bypass_reason = restored.prefix_cache_bypass_reason,
                .request = restored.prefix_request,
                .current_position = restored.current_position,
                .state_compared = true,
                .state_equivalent = static_cast<bool>(state_match),
                .state_detail = state_match.reason,
                .observed_moe_movement_epoch =
                    restored.moe_runtime_movement_epoch,
                .oracle_moe_movement_epoch =
                    oracle.runtime_state.moe_runtime_movement_epoch,
                .main_kv_policy = state_options.main_kv_payload_policy,
                .main_kv_numerical = state_match.main_kv_numerical,
                .terminal_hidden_policy =
                    state_options.terminal_hidden_policy,
                .terminal_hidden_numerical =
                    state_match.terminal_hidden_numerical,
                .terminal_logits_policy =
                    state_options.terminal_logits_policy,
                .terminal_logits_numerical =
                    state_match.terminal_logits_numerical,
                .observed_payload =
                    PrefixRestoreEvidence::TerminalPayloadIdentity::fromSnapshot(
                        restored),
                .oracle_payload =
                    PrefixRestoreEvidence::TerminalPayloadIdentity::fromSnapshot(
                        oracle.runtime_state),
                .checkpoint_compared = true,
                .checkpoint_cosine = checkpoint_cosine,
                .checkpoint_passed = checkpoint_passed,
                .passed = lifecycle_passed &&
                          static_cast<bool>(state_match) &&
                          checkpoint_passed,
            });
            EXPECT_TRUE(checkpoint_passed)
                << "Partial prefix LM_HEAD cosine="
                << checkpoint_cosine
                << " rel_l2=" << checkpoint_rel_l2
                << " max_abs=" << checkpoint_max_abs;
        }

        /**
         * @brief Assert that the already-executed prefill published useful slots.
         *
         * The full layer comparison remains authoritative.  These checks retain
         * the old `SnapshotInfrastructure` test's independent contract without a
         * third model construction and third forward pass.
         */
        void assertProductionParitySnapshotInfrastructure()
        {
            const auto reference_embedding = loadPyTorchSnapshot("EMBEDDING");
            EXPECT_FALSE(reference_embedding.empty())
                << "Failed to load the reference EMBEDDING snapshot";

            const auto keys = activeSnapshotKeys();
            EXPECT_FALSE(keys.empty()) << "Production forward published no snapshots";
            if (keys.empty())
                return;

            const auto has_key = [&keys](const std::string &key)
            {
                return std::find(keys.begin(), keys.end(), key) != keys.end();
            };
            const bool multi_rank = mpi_ctx_ && mpiWorldSize() > 1;
            if (!multi_rank)
            {
                EXPECT_TRUE(has_key("EMBEDDING")) << "Missing EMBEDDING snapshot";
                EXPECT_TRUE(has_key("LM_HEAD")) << "Missing LM_HEAD snapshot";
            }

            EXPECT_TRUE(std::any_of(
                keys.begin(),
                keys.end(),
                [](const std::string &key)
                {
                    return key.rfind("layer", 0) == 0;
                }))
                << "Production forward published no per-layer snapshot";
            EXPECT_TRUE(std::any_of(
                keys.begin(),
                keys.end(),
                [](const std::string &key)
                {
                    return key.find("FFN_RESIDUAL") != std::string::npos;
                }))
                << "Production forward published no FFN_RESIDUAL snapshot";
        }

        /**
         * @brief Prove the typed ExpertOverlay movement outcome from PerfStats.
         *
         * Static cells reject every request-time planner/copy/apply signal.
         * Dynamic cells require independent planner, copied-payload, byte, and
         * applied-placement evidence so a proposal or resident-only row
         * reassignment cannot masquerade as physical expert movement.  The
         * reduction occurs only at the test result boundary, after production
         * inference and any coordinated worker protocol have completed.
         *
         * @param records Process-local records retained for this campaign cell.
         */
        void assertProductionParityMoEMovementEvidence(
            const std::vector<PerfStatRecord> &records) const
        {
            if (config_.moe_movement_expectation ==
                ParityMoEMovementExpectation::NotApplicable)
            {
                return;
            }

            ASSERT_TRUE(PerfStatsCollector::isEnabled())
                << "ExpertOverlay parity requires PerfStats movement evidence";
            ASSERT_TRUE(
                PerfStatsCollector::isDomainEnabled("moe_rebalance") ||
                PerfStatsCollector::isDomainEnabled("moe_overlay_residency"))
                << "ExpertOverlay parity must enable moe_rebalance or "
                   "moe_overlay_residency PerfStats";

            const auto counter = [&](const char *domain, const char *name)
            {
                return std::accumulate(
                    records.begin(), records.end(), 0.0,
                    [&](double total, const PerfStatRecord &record)
                    {
                        return total +
                               (record.kind == PerfStatRecord::Kind::Counter &&
                                        record.domain == domain &&
                                        record.name == name
                                    ? record.value
                                    : 0.0);
                    });
            };

            /*
             * Host-authoritative multi-tier movement and device-authoritative
             * homogeneous movement publish different counters, but both must
             * prove the same four facts. Keep the mapping here at the common
             * parity boundary rather than teaching each model fixture backend
             * implementation details.
             */
            std::array<float, 4> local = {
                static_cast<float>(
                    counter("moe_overlay_residency", "committed_expert_migrations") +
                    counter("moe_overlay_controller", "dynamic_movement_transactions") +
                    counter("moe_rebalance", "device_rebalance_dynamic_ownership_swap_accepts") +
                    counter("moe_rebalance", "device_rebalance_selected_replicas")),
                static_cast<float>(
                    counter("moe_overlay_residency", "cpu_native_copy_bytes") +
                    counter("moe_overlay_residency", "remote_projection_payload_bytes_completed") +
                    counter("moe_overlay_residency", "placement_transfer_operations_completed") +
                    counter("moe_overlay_residency", "device_physical_destinations_staged") +
                    counter("moe_rebalance", "cpu_llep_weight_transfers") +
                    counter("moe_rebalance", "weight_transfer_outgoing_entries") +
                    counter("moe_rebalance", "weight_transfer_incoming_entries") +
                    counter("moe_rebalance", "mask_apply_received_entries") +
                    counter("moe_rebalance", "device_rebalance_copy_copied_arrivals") +
                    counter("moe_rebalance", "device_rebalance_transfer_current_copied_arrivals") +
                    counter("moe_rebalance", "device_rebalance_wave_copied_arrivals_total") +
                    counter("moe_rebalance", "device_rebalance_request_copied_payload_lower_bound")),
                static_cast<float>(
                    counter("moe_overlay_residency", "cpu_native_copy_bytes") +
                    counter("moe_overlay_residency", "remote_projection_payload_bytes_completed") +
                    counter("moe_overlay_residency", "placement_transfer_payload_bytes_completed") +
                    counter("moe_overlay_controller", "dynamic_physical_bytes") +
                    counter("moe_rebalance", "cpu_llep_weight_transfer_incoming_bytes") +
                    counter("moe_rebalance", "cpu_llep_weight_transfer_outgoing_bytes") +
                    counter("moe_rebalance", "weight_transfer_incoming_bytes") +
                    counter("moe_rebalance", "weight_transfer_outgoing_bytes") +
                    counter("moe_rebalance", "mask_apply_received_bytes") +
                    counter("moe_rebalance", "device_rebalance_transfer_useful_payload_bytes") +
                    counter("moe_rebalance", "device_rebalance_request_useful_payload_bytes_lower_bound")),
                static_cast<float>(
                    counter("moe_overlay_residency", "expert_migration_edges") +
                    counter("moe_overlay_residency", "device_physical_epoch_published") +
                    counter("moe_rebalance", "cpu_llep_weight_transfers") +
                    counter("moe_rebalance", "ownership_change_entries") +
                    counter("moe_rebalance", "replica_arrivals") +
                    counter("moe_rebalance", "mask_apply_received_entries") +
                    counter("moe_rebalance", "device_rebalance_apply_applied_arrivals") +
                    counter("moe_rebalance", "device_rebalance_transfer_current_applied_arrivals") +
                    counter("moe_rebalance", "device_rebalance_wave_applied_arrivals_total") +
                    counter("moe_rebalance", "device_rebalance_request_applied_payload_lower_bound")),
            };
            std::array<float, 4> global = local;
            if (mpi_ctx_ && mpiWorldSize() > 1)
                parityCoordinationMPIContext()->allreduce_sum(
                    local.data(), global.data(), global.size());

            const float movement_total =
                global[0] + global[1] + global[2] + global[3];
            const std::string summary = PerfStatsCollector::summaryString(
                {"moe_rebalance", "moe_overlay_residency",
                 "moe_overlay_controller"});
            if (config_.moe_movement_expectation ==
                ParityMoEMovementExpectation::NoMovement)
            {
                EXPECT_EQ(movement_total, 0.0f)
                    << "Static ExpertOverlay parity observed request-time expert "
                       "movement.\n"
                    << summary;
                return;
            }

            EXPECT_GT(global[0], 0.0f)
                << "Dynamic ExpertOverlay parity published no accepted physical "
                   "placement.\n"
                << summary;
            EXPECT_GT(global[1], 0.0f)
                << "Dynamic ExpertOverlay parity copied no expert payload.\n"
                << summary;
            EXPECT_GT(global[2], 0.0f)
                << "Dynamic ExpertOverlay parity transferred no expert bytes.\n"
                << summary;
            EXPECT_GT(global[3], 0.0f)
                << "Dynamic ExpertOverlay parity applied no moved placement.\n"
                << summary;
        }

        /**
         * @brief Prove that physical expert banks used the generated owner order.
         *
         * The configuration value alone is not evidence: both a loader and a
         * graph could accidentally retain their ordinal default and still agree
         * numerically. Setup publishes one record only after a prepared expert
         * bank exists. This result-boundary audit consumes only immutable
         * initial-epoch bank publications, requires every resident bank to name
         * the requested order and exactly match the corresponding frozen
         * owner-map row, and requires global coverage of every ordinary and
         * enabled MTP routed layer across MPI ranks. Automatic capacity may
         * publish a well-formed zero-sized selection for a configured idle
         * endpoint; that record is validated but is not treated as a physical
         * bank or layer-coverage witness. Loader-selection diagnostics are
         * deliberately insufficient: they prove which rows were read, not
         * which prepared bank went live.
         *
         * @param records Process-local placement records for this campaign cell.
         */
        void assertProductionParityMoEOwnerOrderEvidence(
            const std::vector<PerfStatRecord> &records) const
        {
            if (config_.moe_movement_expectation ==
                ParityMoEMovementExpectation::NotApplicable)
            {
                return;
            }

            ASSERT_TRUE(PerfStatsCollector::isEnabled());
            ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("moe_placement"))
                << "ExpertOverlay parity requires physical owner-order evidence";

            const std::string expected_order =
                routedExpertOwnerOrderToString(
                    cfg().routed_expert_owner_order);
            const int main_layer_count = parityLayerCount();
            ASSERT_GT(main_layer_count, 0);

            std::vector<int> expected_layers;
            expected_layers.reserve(static_cast<std::size_t>(main_layer_count) + 1u);
            for (int layer = 0; layer < main_layer_count; ++layer)
                expected_layers.push_back(layer);

            if (config_.mtp_expected_graph_capacity > 0)
            {
                const ModelContext *active_model =
                    activeModelContextForDiagnostics();
                ASSERT_NE(active_model, nullptr);
                const int raw_layer_count = std::max(
                    active_model->totalBlockCount(),
                    active_model->blockCount());
                const MTPWeightManifest manifest = discoverMTPWeightManifest(
                    active_model->concreteLoader(),
                    active_model->architecture(),
                    raw_layer_count,
                    /*explicit_mtp=*/true);
                ASSERT_TRUE(manifest.available) << manifest.diagnostic;
                for (const auto &depth : manifest.depths)
                {
                    if (depth.moe_ffn_layout)
                        expected_layers.push_back(depth.source_layer_index);
                }
            }
            std::sort(expected_layers.begin(), expected_layers.end());
            expected_layers.erase(
                std::unique(expected_layers.begin(), expected_layers.end()),
                expected_layers.end());

            std::vector<float> local_layer_coverage(
                expected_layers.size(), 0.0f);
            float local_matching_records = 0.0f;
            float local_invalid_records = 0.0f;

            for (const auto &record : records)
            {
                if (record.kind != PerfStatRecord::Kind::Counter ||
                    record.domain != "moe_placement" ||
                    record.name != "routed_expert_weight_selection")
                {
                    continue;
                }

                const auto publication = record.tags.find("publication");
                if (publication == record.tags.end() ||
                    publication->second != "initial_epoch_bank")
                {
                    continue;
                }

                const auto order = record.tags.find("owner_order");
                const auto layout = record.tags.find("selection_layout");
                const auto frozen_owner_map_match =
                    record.tags.find("frozen_owner_map_match");
                const auto tier_participant_count =
                    record.tags.find("tier_participant_count");
                const auto layer = record.tags.find("layer");
                const auto layer_role = record.tags.find("layer_role");
                int participants_in_tier = 0;
                if (tier_participant_count != record.tags.end())
                {
                    const char *const begin =
                        tier_participant_count->second.data();
                    const char *const end =
                        begin + tier_participant_count->second.size();
                    const auto [next, parse_error] = std::from_chars(
                        begin, end, participants_in_tier, 10);
                    if (parse_error != std::errc{} || next != end)
                        participants_in_tier = 0;
                }
                /*
                 * Contiguity is observational, not an ownership invariant.
                 * A typed initial-order override can select an adversarial,
                 * non-contiguous tier subset before ordinal participant
                 * partitioning.  The exact frozen owner-map comparison proves
                 * both that selection and the requested ordinal/random
                 * participant order without reconstructing planner policy in
                 * this generic campaign harness.
                 */
                const bool valid_tags =
                    order != record.tags.end() &&
                    layout != record.tags.end() &&
                    frozen_owner_map_match != record.tags.end() &&
                    tier_participant_count != record.tags.end() &&
                    layer != record.tags.end() &&
                    layer_role != record.tags.end() &&
                    order->second == expected_order &&
                    !layout->second.empty() &&
                    frozen_owner_map_match->second == "true" &&
                    participants_in_tier > 0 &&
                    std::isfinite(record.value) &&
                    record.value >= 0.0;
                if (!valid_tags)
                {
                    local_invalid_records += 1.0f;
                    continue;
                }

                // An automatic-capacity tier can be configured yet own no
                // expert. Its explicit idle publication validates topology
                // and owner order but cannot stand in for a resident bank.
                if (record.value == 0.0)
                    continue;

                int layer_index = -1;
                const char *begin = layer->second.data();
                const char *end = begin + layer->second.size();
                const auto [next, parse_error] =
                    std::from_chars(begin, end, layer_index, 10);
                const auto expected_layer = std::lower_bound(
                    expected_layers.begin(),
                    expected_layers.end(),
                    layer_index);
                const char *const expected_role =
                    layer_index < main_layer_count
                        ? "main_transformer"
                        : "mtp_predictor";
                if (parse_error != std::errc{} || next != end ||
                    expected_layer == expected_layers.end() ||
                    *expected_layer != layer_index ||
                    layer_role->second != expected_role)
                {
                    local_invalid_records += 1.0f;
                    continue;
                }
                local_layer_coverage[static_cast<std::size_t>(
                    std::distance(expected_layers.begin(), expected_layer))] = 1.0f;
                local_matching_records += 1.0f;
            }

            std::vector<float> global_layer_coverage = local_layer_coverage;
            std::array<float, 2> local_totals = {
                local_matching_records,
                local_invalid_records,
            };
            std::array<float, 2> global_totals = local_totals;
            if (mpi_ctx_ && mpiWorldSize() > 1)
            {
                if (!local_layer_coverage.empty())
                {
                    parityCoordinationMPIContext()->allreduce_sum(
                        local_layer_coverage.data(),
                        global_layer_coverage.data(),
                        local_layer_coverage.size());
                }
                parityCoordinationMPIContext()->allreduce_sum(
                    local_totals.data(),
                    global_totals.data(),
                    global_totals.size());
            }

            const std::string summary =
                PerfStatsCollector::summaryString({"moe_placement"});
            EXPECT_EQ(global_totals[1], 0.0f)
                << "ExpertOverlay prepared a bank with the wrong owner order, "
                   "layout, or layer identity.\n"
                << summary;
            EXPECT_GT(global_totals[0], 0.0f)
                << "ExpertOverlay published no physical expert-bank selection.\n"
                << summary;
            for (std::size_t index = 0; index < expected_layers.size(); ++index)
            {
                EXPECT_GT(global_layer_coverage[index], 0.0f)
                    << "ExpertOverlay owner-order evidence omitted routed layer "
                    << expected_layers[index] << ".\n"
                    << summary;
            }
        }

        /**
         * @brief Return a model-declared production MTP checkpoint cut.
         *
         * Empty means the optimized graph exposes every checkpoint present in
         * the authenticated reference pack. Config-driven fixtures override
         * this from their typed model definition when a fused graph has a
         * smaller, explicit observable surface.
         *
         * @return Ordered checkpoint suffixes, or an empty vector for discovery.
         */
        virtual std::vector<std::string>
        productionParityDeclaredMTPCheckpointSurface() const
        {
            return {};
        }

        /**
         * @brief Decide whether a reference-only MTP tensor is live in production.
         *
         * A Hugging Face pack may retain extra oracle tensors used by a more
         * observable production topology. Automatic checkpoint discovery must
         * not turn such reference-only diagnostics into fictional graph
         * boundaries on a fused topology. Model families override this hook
         * narrowly; every unclassified tensor remains mandatory.
         *
         * @param suffix Reference checkpoint suffix after `MTP<n>_`.
         * @return `true` only when the optimized production graph owns the value.
         */
        virtual bool productionParityMTPCheckpointIsLive(
            std::string_view suffix) const
        {
            (void)suffix;
            return true;
        }

        /**
         * @brief List every required authenticated stage for one recursive MTP row.
         *
         * The reference directory is already authenticated during fixture
         * setup. Models whose optimized graph materializes every oracle stage
         * discover the exact `decode_stepN_MTPD_*` inventory, making newly
         * added checkpoints mandatory automatically. A fused model instead
         * supplies an explicit typed cut and every stage in that cut remains
         * mandatory.
         */
        std::vector<std::string> productionParityMTPReferenceStageSuffixes(
            int reference_step,
            int reference_depth) const
        {
            const std::vector<std::string> declared =
                productionParityDeclaredMTPCheckpointSurface();
            if (!declared.empty())
                return declared;

            const std::string prefix =
                "decode_step" + std::to_string(reference_step) + "_MTP" +
                std::to_string(reference_depth) + "_";
            std::vector<std::string> suffixes;
            for (const auto &entry : std::filesystem::directory_iterator(
                     std::filesystem::path(config_.snapshot_dir)))
            {
                if (!entry.is_regular_file() ||
                    entry.path().extension() != ".npy")
                {
                    continue;
                }
                const std::string stem = entry.path().stem().string();
                if (stem.rfind(prefix, 0) == 0)
                {
                    const std::string suffix = stem.substr(prefix.size());
                    if (productionParityMTPCheckpointIsLive(suffix))
                        suffixes.push_back(suffix);
                }
            }
            std::sort(suffixes.begin(), suffixes.end());
            suffixes.erase(
                std::unique(suffixes.begin(), suffixes.end()),
                suffixes.end());
            return suffixes;
        }

        /**
         * @brief Verify that one additive MTP branch pack is complete and fresh.
         *
         * Branch tensors are generated atomically, but a later regeneration of
         * the canonical pack changes their main-model trajectory authority.
         * Requiring every branch stage to be at least as new as metadata.txt
         * makes that stale state visible without hashing hundreds of large NPY
         * files on every campaign cell.
         *
         * @param branch_prefix Exact branch prefix ending in `_MTPD_`.
         * @param suffixes Authenticated canonical stage inventory for depth D.
         * @return True only when every corresponding branch tensor is current.
         */
        bool productionParityMTPBranchReferenceUsable(
            const std::string &branch_prefix,
            const std::vector<std::string> &suffixes) const
        {
            if (branch_prefix.empty() || suffixes.empty())
                return false;

            std::error_code error;
            const auto metadata =
                std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
            if (!std::filesystem::is_regular_file(metadata, error) || error)
                return false;
            const auto metadata_time =
                std::filesystem::last_write_time(metadata, error);
            if (error)
                return false;

            for (const auto &suffix : suffixes)
            {
                const auto checkpoint =
                    std::filesystem::path(config_.snapshot_dir) /
                    (branch_prefix + suffix + ".npy");
                error.clear();
                if (!std::filesystem::is_regular_file(checkpoint, error) ||
                    error)
                {
                    return false;
                }
                const auto checkpoint_time =
                    std::filesystem::last_write_time(checkpoint, error);
                if (error || checkpoint_time < metadata_time)
                    return false;
            }
            return true;
        }

        /**
         * @brief Publish one missing branch reference under a node-wide lease.
         *
         * Symmetric fixtures reach this boundary on every rank with the same
         * device-owned proposal identity. Only the artifact authority runs
         * Python; peer ranks consume its status and validate the immutable
         * files. Server-shaped fixtures declare `ArtifactAuthorityOnly`
         * because their peers remain inside the production command loop and
         * cannot participate in a test-only collective. The filesystem lease
         * coalesces concurrent CTest processes in both regimes.
         *
         * @return True when every rank can consume the complete branch pack.
         */
        bool ensureProductionParityMTPBranchReference(
            int reference_step,
            const std::vector<int32_t> &condition_tokens,
            const std::string &branch_prefix,
            const std::vector<std::string> &suffixes)
        {
            if (productionParityMTPBranchReferenceUsable(
                    branch_prefix, suffixes))
            {
                return true;
            }

            const auto coordination =
                parityReferenceGenerationCoordination();
            if (coordination ==
                    ParityReferenceGenerationCoordination::
                        ArtifactAuthorityOnly &&
                !isRank0())
            {
                LOG_ERROR(
                    "[Parity] Non-authority rank attempted authority-only "
                    "MTP reference generation");
                return false;
            }

            int32_t generation_succeeded = 1;
            if (isRank0())
            {
                try
                {
                    ReferenceGenerationLease lease;
                    if (!productionParityMTPBranchReferenceUsable(
                            branch_prefix, suffixes))
                    {
                        generation_succeeded =
                            regeneratePyTorchMTPBranchSnapshots(
                                reference_step, condition_tokens)
                                ? 1
                                : 0;
                    }
                    if (generation_succeeded != 0 &&
                        !productionParityMTPBranchReferenceUsable(
                            branch_prefix, suffixes))
                    {
                        generation_succeeded = 0;
                    }
                }
                catch (const std::exception &error)
                {
                    LOG_ERROR(
                        "[Parity] MTP branch generation failed: "
                        << error.what());
                    generation_succeeded = 0;
                }
            }
            if (coordination ==
                    ParityReferenceGenerationCoordination::SymmetricRanks &&
                mpi_ctx_ && mpiWorldSize() > 1)
            {
                parityCoordinationMPIContext()->broadcast_int32(
                    &generation_succeeded,
                    1,
                    parityArtifactAuthorityRank());
                mpiBarrier();
            }
            return generation_succeeded != 0 &&
                   productionParityMTPBranchReferenceUsable(
                       branch_prefix, suffixes);
        }

        /**
         * @brief Resolve canonical versus forced-branch HF identity for one row.
         *
         * The committed verifier record is the sole proposal authority. Each
         * preceding token is checked against the canonical HF predictor logits;
         * the first mismatch selects an additive reference containing the full
         * consumed prefix. This decision never changes production execution or
         * treats token divergence as numerical failure by itself.
         */
        std::optional<std::string> productionParityMTPReferencePrefix(
            const ProductionParityMTPTransactionBoundary &boundary,
            int reference_depth,
            const std::vector<std::string> &suffixes)
        {
            const std::string canonical =
                "decode_step" + std::to_string(boundary.reference_step) +
                "_MTP" + std::to_string(reference_depth) + "_";
            if (reference_depth == 0)
                return canonical;
            if (reference_depth < 0 ||
                boundary.after.mtp_observed_verifier_draft_depth !=
                    boundary.snapshot_execution_draft_depth ||
                boundary.after.mtp_observed_verifier_draft_tokens.size() <
                    static_cast<size_t>(reference_depth))
            {
                ADD_FAILURE()
                    << "The committed verifier identity cannot name MTP depth "
                    << reference_depth;
                return std::nullopt;
            }

            bool canonical_branch = true;
            int first_divergent_depth = -1;
            for (int proposal_depth = 0;
                 proposal_depth < reference_depth;
                 ++proposal_depth)
            {
                const std::string logits_key =
                    "decode_step" +
                    std::to_string(boundary.reference_step) + "_MTP" +
                    std::to_string(proposal_depth) + "_LM_HEAD";
                const auto logits = loadPyTorchSnapshot(logits_key);
                if (logits.empty())
                {
                    ADD_FAILURE()
                        << "Authenticated reference pack omitted "
                        << logits_key << " while resolving MTP branch identity";
                    return std::nullopt;
                }
                const int32_t canonical_token = static_cast<int32_t>(
                    std::distance(
                        logits.begin(),
                        std::max_element(logits.begin(), logits.end())));
                if (boundary.after.mtp_observed_verifier_draft_tokens[
                        static_cast<size_t>(proposal_depth)] == canonical_token)
                {
                    continue;
                }
                canonical_branch = false;
                first_divergent_depth = proposal_depth;
                break;
            }
            if (canonical_branch)
                return canonical;

            std::vector<int32_t> condition_tokens(
                boundary.after.mtp_observed_verifier_draft_tokens.begin(),
                boundary.after.mtp_observed_verifier_draft_tokens.begin() +
                    reference_depth);
            const std::string branch = mtpParityBranchReferencePrefix(
                boundary.reference_step,
                reference_depth,
                condition_tokens);
            LOG_INFO(
                "[Parity] Production MTP branch diverged from canonical HF at "
                "depth "
                << first_divergent_depth << "; resolving exact " << branch);
            if (!ensureProductionParityMTPBranchReference(
                    boundary.reference_step,
                    condition_tokens,
                    branch,
                    suffixes))
            {
                ADD_FAILURE()
                    << "Could not generate the exact Hugging Face MTP branch "
                    << branch;
                return std::nullopt;
            }
            return branch;
        }

        /**
         * @brief Append one MTP checkpoint to the ordinary decode artifact.
         *
         * MTP is a decode execution policy, so its recursive stages belong in
         * `decode_stages.csv`, beside the serial main-model rows that establish
         * the transaction oracle. A synthetic sidecar-only CSV would split the
         * canonical numerical contract and let aggregate tooling miss failures.
         */
        void appendProductionParityMTPStage(
            DecodeParitySummary &summary,
            int reference_step,
            int layer_index,
            StageComparisonResult result)
        {
            auto step = std::find_if(
                summary.step_stats.begin(),
                summary.step_stats.end(),
                [reference_step](const DecodeStepStats &candidate)
                {
                    return candidate.step_idx == reference_step;
                });
            ASSERT_NE(step, summary.step_stats.end())
                << "MTP checkpoint has no authenticated serial decode row "
                << reference_step;

            auto layer = std::find_if(
                step->layer_stats.begin(),
                step->layer_stats.end(),
                [layer_index](const LayerStats &candidate)
                {
                    return candidate.layer_idx == layer_index;
                });
            if (layer == step->layer_stats.end())
            {
                step->layer_stats.push_back(
                    LayerStats{.layer_idx = layer_index});
                layer = std::prev(step->layer_stats.end());
            }
            layer->stage_results.push_back(std::move(result));

            layer->avg_cosine_sim = 0.0f;
            layer->min_cosine_sim = 1.0f;
            layer->max_cosine_drop = 0.0f;
            layer->stages_compared = 0;
            layer->passed = true;
            layer->worst_stage.clear();
            layer->max_drop_stage.clear();
            float previous_cosine = 0.0f;
            bool have_previous = false;
            for (auto &stage : layer->stage_results)
            {
                layer->passed = layer->passed && stage.passed;
                if (stage.is_routing_stage)
                    continue;
                ++layer->stages_compared;
                layer->avg_cosine_sim += stage.cosine_similarity;
                if (stage.cosine_similarity < layer->min_cosine_sim)
                {
                    layer->min_cosine_sim = stage.cosine_similarity;
                    layer->worst_stage = stage.stage_name;
                }
                if (have_previous)
                {
                    stage.cosine_drop =
                        previous_cosine - stage.cosine_similarity;
                    if (stage.cosine_drop > layer->max_cosine_drop)
                    {
                        layer->max_cosine_drop = stage.cosine_drop;
                        layer->max_drop_stage = stage.stage_name;
                    }
                }
                previous_cosine = stage.cosine_similarity;
                have_previous = true;
            }
            if (layer->stages_compared > 0)
            {
                layer->avg_cosine_sim /=
                    static_cast<float>(layer->stages_compared);
            }
            std::sort(
                step->layer_stats.begin(),
                step->layer_stats.end(),
                [](const LayerStats &left, const LayerStats &right)
                {
                    return left.layer_idx < right.layer_idx;
                });
        }

        /**
         * @brief Compare one typed snapshot bank from a production MTP transaction.
         *
         * A production sidecar graph reuses depth-zero stage names. The runtime
         * context identifies whether bytes came from the primary graph or the
         * terminal chained replay, while @p identity names the corresponding
         * Hugging Face recursive row. Response capacity is intentionally absent:
         * it clips visible commit rows but cannot rename executed graph bytes.
         */
        void compareProductionParityMTPCheckpoint(
            const ProductionParityMTPTransactionBoundary &boundary,
            MTPParityCheckpointIdentity identity,
            DecodeParitySummary &summary)
        {
            ASSERT_GT(boundary.snapshot_execution_draft_depth, 0);
            ASSERT_GE(identity.reference_depth, 0);
            const int reference_depth = identity.reference_depth;
            const std::string runtime_prefix(
                mtpParityCheckpointContextPrefix(identity.context));
            const auto suffixes =
                productionParityMTPReferenceStageSuffixes(
                    boundary.reference_step,
                    reference_depth);
            ASSERT_FALSE(suffixes.empty())
                << "Authenticated reference pack has no MTP depth "
                << reference_depth << " checkpoints";
            const auto reference_prefix =
                productionParityMTPReferencePrefix(
                    boundary, reference_depth, suffixes);
            ASSERT_TRUE(reference_prefix.has_value())
                << "MTP depth " << reference_depth
                << " has no reference matching its committed proposal branch";

            const MoEConfig moe = getMoEConfig();
            const int mtp_layer_index = parityLayerCount();
            size_t compared = 0;
            for (const std::string &suffix : suffixes)
            {
                const std::string reference_key =
                    *reference_prefix + suffix;
                /*
                 * The terminal-hidden selector precedes the recursive model
                 * block and is therefore an operational `MTP_*` stage, not a
                 * depth-zero model stage. Its context remains identical; only
                 * the relative stage namespace differs.
                 */
                const std::string runtime_key =
                    runtime_prefix +
                    (suffix == "TERMINAL_HIDDEN_ROW_SELECT"
                         ? "MTP_TERMINAL_HIDDEN_ROW_SELECT"
                         : "MTP0_" + suffix);
                const std::vector<float> reference =
                    loadPyTorchSnapshot(reference_key);
                ASSERT_FALSE(reference.empty())
                    << "Missing authenticated MTP tensor " << reference_key;

                size_t actual_size = 0;
                const float *actual = activeSnapshot(runtime_key, actual_size);
                ASSERT_NE(actual, nullptr)
                    << "Production MTP transaction omitted live checkpoint "
                    << runtime_key << " for " << reference_key;
                ASSERT_EQ(actual_size, reference.size())
                    << "MTP checkpoint geometry mismatch for "
                    << reference_key;

                const std::string stage_name =
                    "MTP" + std::to_string(reference_depth) + "_" + suffix;
                StageComparisonResult comparison;
                std::optional<MoERoutingBoundaryResult> routing_boundary;
                if (suffix.ends_with("MOE_ROUTING_INDICES"))
                {
                    ASSERT_GT(moe.top_k, 0)
                        << "Routed MTP checkpoint has no model top-k geometry";
                    comparison = compareRoutingIndices(
                        actual,
                        reference,
                        actual_size,
                        moe.top_k,
                        stage_name);

                    /*
                     * Literal expert IDs are not a mathematically stable
                     * oracle when two router probabilities straddle the
                     * top-k cutoff within the measured backend error.  Bind
                     * both index rows to the live full router distributions
                     * and accept a substitution only when each row is its
                     * own valid top-k, the leading expert is exact, and the
                     * reference cutoff gap can actually flip.  This is the
                     * same typed authority used by the detailed MoE parity
                     * diagnostic, so the canonical CSV and generic MTP gate
                     * cannot publish contradictory verdicts.
                     */
                    const std::string router_suffix =
                        suffix.substr(
                            0,
                            suffix.size() -
                                std::string("MOE_ROUTING_INDICES").size()) +
                        "MOE_ROUTER_OUTPUT";
                    const std::vector<float> reference_router =
                        loadPyTorchSnapshot(*reference_prefix + router_suffix);
                    size_t actual_router_size = 0;
                    const float *actual_router = activeSnapshot(
                        runtime_prefix + "MTP0_" + router_suffix,
                        actual_router_size);
                    ASSERT_FALSE(reference_router.empty())
                        << "MTP routing indices require their authenticated "
                           "router distribution";
                    ASSERT_NE(actual_router, nullptr)
                        << "Production MTP routing indices omitted their live "
                           "router distribution";
                    ASSERT_EQ(actual_router_size, reference_router.size());
                    routing_boundary = compareMoERoutingBoundarySelections(
                        reference.data(),
                        actual,
                        actual_size,
                        reference_router.data(),
                        reference_router.size(),
                        actual_router,
                        actual_router_size,
                        static_cast<size_t>(moe.top_k));
                    comparison.passed =
                        routing_boundary->evaluated &&
                        routing_boundary->equivalent;
                }
                else if (suffix.ends_with("MOE_ROUTING_WEIGHTS"))
                {
                    ASSERT_GT(moe.top_k, 0);
                    ASSERT_GT(moe.num_experts, 0);
                    const std::string indices_suffix =
                        suffix.substr(
                            0,
                            suffix.size() -
                                std::string("MOE_ROUTING_WEIGHTS").size()) +
                        "MOE_ROUTING_INDICES";
                    const std::string indices_reference_key =
                        *reference_prefix + indices_suffix;
                    const std::string indices_runtime_key =
                        runtime_prefix + "MTP0_" + indices_suffix;
                    const std::vector<float> reference_indices =
                        loadPyTorchSnapshot(indices_reference_key);
                    size_t actual_indices_size = 0;
                    const float *actual_indices = activeSnapshot(
                        indices_runtime_key,
                        actual_indices_size);
                    ASSERT_NE(actual_indices, nullptr);
                    ASSERT_EQ(actual_indices_size, actual_size);
                    ASSERT_EQ(reference_indices.size(), reference.size());
                    comparison = compareRoutingWeights(
                        actual,
                        reference,
                        actual_indices,
                        reference_indices,
                        actual_size,
                        moe.top_k,
                        moe.num_experts,
                        stage_name);
                }
                else
                {
                    comparison = compareParityTensorData(
                        actual,
                        reference.data(),
                        actual_size,
                        stage_name,
                        config_.decode_cosine_threshold);
                    if (suffix.ends_with("LM_HEAD"))
                    {
                        comparison.kl_divergence = computeKLDivergence(
                            actual,
                            reference.data(),
                            actual_size,
                            actual_size);
                        comparison.passed =
                            comparison.passed ||
                            comparison.kl_divergence <
                                config_.mtp_kl_threshold.value_or(
                                    config_.kl_threshold);
                        EXPECT_GE(
                            pytorchTop1InLlaminarTopK(
                                actual,
                                reference.data(),
                                actual_size,
                                actual_size,
                                std::max(1, config_.pytorch_top1_in_topk)),
                            1.0f - 1.0e-6f)
                            << stage_name
                            << " omitted the Hugging Face top-1 from the "
                               "production top-k";
                    }
                }
                EXPECT_TRUE(comparison.passed)
                    << stage_name << " failed real-weight MTP parity: cosine="
                    << comparison.cosine_similarity
                    << " rel_l2=" << comparison.rel_l2_norm
                    << " max_abs=" << comparison.max_abs_diff
                    << " kl=" << comparison.kl_divergence
                    << " routing_boundary_evaluated="
                    << (routing_boundary.has_value()
                            ? routing_boundary->evaluated
                            : false)
                    << " routing_boundary_gap="
                    << (routing_boundary.has_value()
                            ? routing_boundary->maximum_boundary_gap
                            : 0.0)
                    << " routing_boundary_error_limit="
                    << (routing_boundary.has_value()
                            ? routing_boundary->maximum_error_limit
                            : 0.0);
                appendProductionParityMTPStage(
                    summary,
                    boundary.reference_step,
                    mtp_layer_index,
                    std::move(comparison));
                ++compared;
            }
            EXPECT_EQ(compared, suffixes.size());
        }

        /**
         * @brief Compare every durable checkpoint bank for one MTP transaction.
         *
         * Depth one publishes only its primary MTP0 row. A deeper transaction
         * additionally leaves its final recursive row in the reusable chained
         * bank. The typed plan prevents a response-token limit from being
         * mistaken for either execution depth or snapshot identity.
         */
        void compareProductionParityMTPCheckpoints(
            const ProductionParityMTPTransactionBoundary &boundary,
            DecodeParitySummary &summary)
        {
            const MTPParityCheckpointPlan plan =
                makeMTPParityCheckpointPlan(
                    activePrimaryDevice().is_cpu(),
                    boundary.snapshot_execution_draft_depth);
            ASSERT_TRUE(plan.valid())
                << "Production MTP transaction has no legal checkpoint plan for "
                << "execution depth "
                << boundary.snapshot_execution_draft_depth;
            for (size_t index = 0; index < plan.count; ++index)
            {
                compareProductionParityMTPCheckpoint(
                    boundary, plan.checkpoints[index], summary);
            }
        }

        /**
         * @brief Build the exact serial token oracle on the already-loaded runner.
         *
         * A one-token response budget is a production request contract: the MTP
         * runner executes its depth-zero direct-emit transaction, advances the
         * ordinary main-model row, and runs no predictor or grouped verifier.
         * Repeating that request boundary therefore produces the free-running
         * serial Llaminar trajectory without a second model, a test-only graph,
         * or a mutable MTP enable flag. The prompt is restored through the same
         * prefix-cache surface used by serving before the first row executes.
         *
         * @param prompt Exact authenticated request prefix.
         * @param output_count Number of serial response tokens required.
         * @param error Receives the first lifecycle or execution failure.
         * @return Complete serial trajectory, or nullopt on failure.
         */
        std::optional<std::vector<int32_t>>
        captureProductionParityMTPSerialOracle(
            const std::vector<int32_t> &prompt,
            int output_count,
            std::string *error)
        {
            const auto fail = [error](std::string message)
                -> std::optional<std::vector<int32_t>>
            {
                if (error)
                    *error = std::move(message);
                return std::nullopt;
            };

            if (!orch_runner_)
                return fail("serial MTP oracle requires the production runner");
            if (output_count <= 0)
                return fail("serial MTP oracle requires a positive output count");

            std::string restore_error;
            const auto complete_prefix =
                restoreProductionParityCompletePrefixAtCurrentFingerprint(
                    prompt,
                    ProductionParityCompletePrefixPayload::MainModelAndMTP,
                    &restore_error);
            if (!complete_prefix)
            {
                return fail(
                    "serial MTP oracle prefix restore failed: " +
                    restore_error);
            }

            const PrefixRuntimeStateSnapshot &restored =
                complete_prefix->state;
            if (!restored.mtp_config_enabled ||
                !restored.mtp_request.enabled ||
                restored.mtp_bypassed || restored.mtp_request.bypassed)
            {
                return fail(
                    "serial MTP oracle did not retain the enabled production "
                    "request policy");
            }
            if (!restored.prefix_request.hit ||
                !restored.prefix_request.mtp_state_restored)
            {
                return fail(
                    "serial MTP oracle did not consume the complete prefix and "
                    "shifted-state restore path");
            }

            std::vector<int32_t> tokens;
            tokens.reserve(static_cast<size_t>(output_count));
            for (int output_index = 0;
                 output_index < output_count;
                 ++output_index)
            {
                const PrefixRuntimeStateSnapshot before =
                    activePrefixStateProbe();
                const MTPParityTransactionCounters counters_before{
                    .draft_steps = before.mtp_draft_steps,
                    .verifier_runs = before.mtp_verifier_runs,
                };

                struct DecodeBudgetReset final
                {
                    IOrchestrationRunner &runner;
                    ~DecodeBudgetReset()
                    {
                        runner.setDecodeStepTokenBudget(0);
                    }
                } budget_reset{*orch_runner_};
                orch_runner_->setDecodeStepTokenBudget(1);
                GenerationResult result = orch_runner_->decodeStep();
                if (!result.success())
                {
                    return fail(
                        "serial MTP oracle decode failed at output " +
                        std::to_string(output_index) + ": " + result.error);
                }
                if (result.tokens.size() != 1u)
                {
                    return fail(
                        "serial MTP oracle must emit exactly one token at "
                        "output " + std::to_string(output_index) +
                        ", got " + std::to_string(result.tokens.size()));
                }

                const PrefixRuntimeStateSnapshot after =
                    activePrefixStateProbe();
                const MTPParityTransactionCounters counters_after{
                    .draft_steps = after.mtp_draft_steps,
                    .verifier_runs = after.mtp_verifier_runs,
                };
                const MTPParityTransactionActivity activity =
                    classifyMTPParityTransactionActivity(
                        counters_before, counters_after);
                if (activity != MTPParityTransactionActivity::None)
                {
                    return fail(
                        "one-token serial oracle unexpectedly executed a draft "
                        "or verifier at output " +
                        std::to_string(output_index));
                }
                if (after.mtp_transaction_commits !=
                    before.mtp_transaction_commits + 1u)
                {
                    return fail(
                        "one-token serial oracle did not publish exactly one "
                        "depth-zero transaction commit at output " +
                        std::to_string(output_index));
                }
                if (!result.is_complete &&
                    after.current_position != before.current_position + 1)
                {
                    return fail(
                        "one-token serial oracle did not advance exactly one "
                        "main-model position at output " +
                        std::to_string(output_index));
                }
                if (result.is_complete && output_index + 1 < output_count)
                {
                    return fail(
                        "serial MTP oracle reached a stop token before the "
                        "requested trajectory was complete");
                }
                tokens.push_back(result.tokens.front());
            }

            activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics();
            return tokens;
        }

        /**
         * @brief Execute the production MTP path without rebuilding the model.
         *
         * Every sweep crosses the public request boundary, restores the same
         * authenticated prompt (including shifted-MTP state), and admits one
         * bounded generation call. Every numerical proof uses one stop-isolated
         * transaction. Dynamic policy then opens a fresh production request,
         * retires any initially due MoE maintenance with one ordinary token,
         * and runs one full-budget transaction to prove the adaptive
         * controller. Keeping those roles explicit prevents a maintenance
         * boundary from either overwriting numerical evidence or masquerading
         * as a dynamic-depth observation.
         */
        void runProductionParityMTPProof(DecodeParitySummary &summary)
        {
            if (config_.mtp_expectation == ParityMTPExpectation::Disabled)
                return;

            ASSERT_NE(orch_runner_.get(), nullptr)
                << "Production MTP proof requires the server-facing runner";
            ASSERT_GT(config_.mtp_expected_draft_depth, 0);
            ASSERT_GT(config_.mtp_expected_graph_capacity, 0);
            ASSERT_EQ(
                resolveMTPMaximumDraftDepth(orch_runner_->config().mtp),
                config_.mtp_expected_graph_capacity)
                << "Production runner retained the wrong MTP graph capacity";

            std::vector<int32_t> prompt(
                config_.token_ids.begin(), config_.token_ids.end());
            const bool dynamic_depth =
                config_.mtp_expectation ==
                ParityMTPExpectation::DynamicDepth;
            const int dynamic_policy_witness_budget =
                dynamic_depth
                    ? mtpParityFullWidthPolicyWitnessBudget(
                          config_.mtp_expected_draft_depth)
                    : 0;
            ASSERT_FALSE(dynamic_depth && dynamic_policy_witness_budget <= 0)
                << "Dynamic MTP proof has invalid witness geometry";
            const int serial_oracle_output_count =
                dynamic_depth
                    ? dynamic_policy_witness_budget + 1
                    : config_.mtp_expected_draft_depth + 1;
            std::string serial_oracle_error;
            const auto serial_oracle =
                captureProductionParityMTPSerialOracle(
                    prompt,
                    serial_oracle_output_count,
                    &serial_oracle_error);
            ASSERT_TRUE(serial_oracle.has_value())
                << "Could not establish the production serial MTP oracle: "
                << serial_oracle_error;
            ASSERT_EQ(
                serial_oracle->size(),
                static_cast<size_t>(serial_oracle_output_count));

            /*
             * Each generated matrix cell owns one declared MTP policy depth.
             * Execute that production policy once at its complete verifier
             * width.  A GPU decodeStep() may otherwise fill the entire public
             * response budget with several device-controller transactions;
             * the primary and chained diagnostic slots are reusable graph
             * storage, so that would leave the primary slot from transaction
             * zero and the chained slot from the terminal transaction.
             *
             * The HTTP/server surface already supports request stop tokens.
             * Fixed policy installs the serial first output as the sole stop
             * token before request admission. GPU sampling remains device-owned
             * and deferred, so the complete depth-N proposal and grouped
             * verifier execute before the compact outcome discovers the stop.
             *
             * Dynamic policy cannot use that same transaction for both
             * numerical and economics evidence when ExpertOverlay maintenance
             * is due: the placement boundary deliberately clips publication
             * and an adaptive selector must not learn from clipped work. A
             * separate request below first retires that maintenance boundary,
             * then admits a complete adaptive transaction.
             */
            const int limited_depth = config_.mtp_expected_draft_depth;
            {
                SCOPED_TRACE(
                    "production MTP declared depth " +
                    std::to_string(limited_depth));
                const bool isolate_gpu_checkpoint_transaction =
                    activePrimaryDevice().is_gpu();
                struct ScopedStopTokenReset final
                {
                    IOrchestrationRunner &runner;
                    ~ScopedStopTokenReset() { runner.setStopTokens({}); }
                } stop_token_reset{*orch_runner_};
                orch_runner_->setStopTokens(
                    isolate_gpu_checkpoint_transaction
                        ? std::vector<int32_t>{serial_oracle->front()}
                        : std::vector<int32_t>{});
                activeClearSnapshots();
                activeClearCache();
                activeSetSnapshotCaptureFilter(
                    paritySnapshotSetupCaptureFilter());
                ASSERT_TRUE(orch_runner_->prefill(prompt))
                    << orch_runner_->lastError();
                const PrefixRuntimeStateSnapshot before =
                    activePrefixStateProbe();
                ASSERT_TRUE(before.mtp_config_enabled);
                ASSERT_TRUE(before.mtp_request.enabled);
                ASSERT_FALSE(before.mtp_bypassed)
                    << before.mtp_bypass_reason;
                ASSERT_TRUE(before.prefix_request.hit)
                    << "MTP sweep must consume the production complete-prefix "
                       "restore path";
                ASSERT_TRUE(before.prefix_request.mtp_state_restored)
                    << "MTP sweep restored no shifted predictor state";
                const int reference_step =
                    mtpParityReferenceStepForConditionPosition(
                        before.mtp_next_condition_position,
                        static_cast<int>(prompt.size()));
                ASSERT_EQ(reference_step, 0)
                    << "A fresh restored request must begin at MTP reference "
                       "step zero";

                activeClearSnapshots();
                const MTPParityTransactionCounters counters_before{
                    .draft_steps = before.mtp_draft_steps,
                    .verifier_runs = before.mtp_verifier_runs,
                };
                orch_runner_->setDecodeStepTokenBudget(limited_depth + 1);
                GenerationResult result = orch_runner_->decodeStep();
                orch_runner_->setDecodeStepTokenBudget(0);
                ASSERT_TRUE(result.success()) << result.error;
                ASSERT_FALSE(result.tokens.empty());
                ASSERT_LE(
                    result.tokens.size(),
                    static_cast<size_t>(limited_depth + 1))
                    << "Response-limited MTP transaction exceeded its public "
                       "token budget";
                activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics();
                const PrefixRuntimeStateSnapshot after =
                    activePrefixStateProbe();
                const MTPParityTransactionCounters counters_after{
                    .draft_steps = after.mtp_draft_steps,
                    .verifier_runs = after.mtp_verifier_runs,
                };
                ASSERT_EQ(
                    classifyMTPParityTransactionActivity(
                        counters_before, counters_after),
                    MTPParityTransactionActivity::Speculative)
                    << "MTP-labelled campaign executed no coherent predictor/"
                       "verifier transaction";

                if (isolate_gpu_checkpoint_transaction)
                {
                    ASSERT_TRUE(result.is_complete)
                        << "The production stop-token policy did not retire "
                           "the checkpoint transaction";
                    ASSERT_EQ(result.tokens.size(), 1u)
                        << "The checkpoint transaction crossed its first "
                           "response-visible stop token";
                    ASSERT_EQ(
                        mtpParityExecutedTransactionCount(
                            counters_before, counters_after),
                        1u)
                        << "Reusable MTP snapshot banks cannot certify more "
                           "than one controller transaction at a time";
                }

                const MTPParityTokenComparison token_comparison =
                    compareMTPGroupedTokensToSerialOracle(
                        *serial_oracle, result.tokens);
                ASSERT_TRUE(token_comparison.exact)
                    << "Grouped MTP response diverged from serial Llaminar at "
                       "output token "
                    << token_comparison.mismatch_index
                    << ": serial=" << token_comparison.serial_token
                    << " grouped=" << token_comparison.grouped_token;
                std::vector<int32_t> serial_prefix(
                    serial_oracle->begin(),
                    serial_oracle->begin() +
                        static_cast<std::ptrdiff_t>(result.tokens.size()));

                ProductionParityMTPTransactionBoundary boundary{
                    .requested_draft_depth =
                        before.mtp_current_depth > 0
                            ? before.mtp_current_depth
                            : config_.mtp_expected_draft_depth,
                    .response_limited_draft_depth = limited_depth,
                    .snapshot_execution_draft_depth =
                        before.mtp_current_depth > 0
                            ? before.mtp_current_depth
                            : config_.mtp_expected_draft_depth,
                    .reference_step = reference_step,
                    .condition_position =
                        before.mtp_next_condition_position,
                    .emitted_tokens = result.tokens,
                    .serial_oracle_tokens = std::move(serial_prefix),
                    .serial_token_exact = token_comparison.exact,
                    .attempted_draft_tokens =
                        mtpParityAttemptedDraftTokenCount(
                            counters_before, counters_after),
                    .verifier_transactions =
                        mtpParityExecutedTransactionCount(
                            counters_before, counters_after),
                    .before = before,
                    .after = after,
                };
                EXPECT_EQ(
                    boundary.requested_draft_depth,
                    config_.mtp_expected_draft_depth)
                    << "Request admission selected the wrong MTP depth";
                EXPECT_GE(
                    boundary.attempted_draft_tokens,
                    static_cast<uint64_t>(limited_depth))
                    << "Production transaction did not reach its declared "
                       "recursive depth";
                EXPECT_GT(boundary.verifier_transactions, 0u);

                compareProductionParityMTPCheckpoints(boundary, summary);

                if (dynamic_depth)
                {
                    /*
                     * Numerical snapshots have already been consumed. Start a
                     * fresh server-facing request without a stop token, then
                     * use a one-token public call to retire the initially due
                     * movement boundary. A one-token budget intentionally runs
                     * serial decode; it is not counted as the adaptive witness.
                     * The following public call reserves one slot beyond the
                     * complete depth-N verifier response. Production can then
                     * leave its final visible token pending without clipping
                     * the verifier-state publication, so the device controller
                     * receives one economically valid adaptive observation.
                     */
                    orch_runner_->setStopTokens({});
                    activeClearSnapshots();
                    activeClearCache();
                    activeSetSnapshotCaptureFilter(
                        paritySnapshotSetupCaptureFilter());
                    ASSERT_TRUE(orch_runner_->prefill(prompt))
                        << orch_runner_->lastError();
                    const PrefixRuntimeStateSnapshot witness_prefill =
                        activePrefixStateProbe();
                    ASSERT_TRUE(witness_prefill.prefix_request.hit)
                        << "Dynamic MTP witness must consume the production "
                           "complete-prefix restore path";

                    orch_runner_->setDecodeStepTokenBudget(1);
                    GenerationResult maintenance_result =
                        orch_runner_->decodeStep();
                    orch_runner_->setDecodeStepTokenBudget(0);
                    ASSERT_TRUE(maintenance_result.success())
                        << maintenance_result.error;
                    ASSERT_EQ(maintenance_result.tokens.size(), 1u);
                    ASSERT_FALSE(maintenance_result.is_complete)
                        << "Dynamic MTP witness cannot publish maintenance "
                           "after a terminal generation result";
                    ASSERT_EQ(
                        maintenance_result.tokens.front(),
                        serial_oracle->front())
                        << "Dynamic MTP maintenance warmup diverged from the "
                           "serial oracle";
                    ASSERT_TRUE(orch_runner_->maybeApplyMoERebalance(
                        maintenance_result.tokens.size()))
                        << "Dynamic MTP maintenance warmup could not publish "
                           "its ordinary production boundary: "
                        << orch_runner_->lastError();
                    activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics();

                    const PrefixRuntimeStateSnapshot witness_before =
                        activePrefixStateProbe();
                    const MTPParityTransactionCounters
                        witness_counters_before{
                            .draft_steps = witness_before.mtp_draft_steps,
                            .verifier_runs = witness_before.mtp_verifier_runs,
                    };
                    orch_runner_->setDecodeStepTokenBudget(
                        dynamic_policy_witness_budget);
                    GenerationResult witness_result =
                        orch_runner_->decodeStep();
                    orch_runner_->setDecodeStepTokenBudget(0);
                    ASSERT_TRUE(witness_result.success())
                        << witness_result.error;
                    ASSERT_FALSE(witness_result.tokens.empty());
                    ASSERT_FALSE(witness_result.is_complete)
                        << "Dynamic MTP proof requires a nonterminal "
                           "full-budget transaction so the production "
                           "maintenance boundary remains legal";
                    ASSERT_TRUE(orch_runner_->maybeApplyMoERebalance(
                        witness_result.tokens.size()))
                        << "Dynamic MTP policy witness could not publish its "
                           "ordinary production boundary: "
                        << orch_runner_->lastError();
                    activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics();
                    const PrefixRuntimeStateSnapshot witness_after =
                        activePrefixStateProbe();
                    const MTPParityTransactionCounters
                        witness_counters_after{
                            .draft_steps = witness_after.mtp_draft_steps,
                            .verifier_runs = witness_after.mtp_verifier_runs,
                        };
                    ASSERT_EQ(
                        classifyMTPParityTransactionActivity(
                            witness_counters_before,
                            witness_counters_after),
                        MTPParityTransactionActivity::Speculative)
                        << "Dynamic MTP policy witness executed no coherent "
                           "predictor/verifier transaction";

                    const size_t oracle_offset =
                        maintenance_result.tokens.size();
                    ASSERT_LE(
                        oracle_offset + witness_result.tokens.size(),
                        serial_oracle->size())
                        << "Dynamic MTP policy witness exceeded its serial "
                           "oracle suffix";
                    std::vector<int32_t> witness_serial_tokens(
                        serial_oracle->begin() +
                            static_cast<std::ptrdiff_t>(oracle_offset),
                        serial_oracle->begin() +
                            static_cast<std::ptrdiff_t>(
                                oracle_offset +
                                witness_result.tokens.size()));
                    const MTPParityTokenComparison witness_comparison =
                        compareMTPGroupedTokensToSerialOracle(
                            witness_serial_tokens,
                            witness_result.tokens);
                    ASSERT_TRUE(witness_comparison.exact)
                        << "Dynamic MTP policy witness diverged from serial "
                           "Llaminar at output token "
                        << witness_comparison.mismatch_index
                        << ": serial=" << witness_comparison.serial_token
                        << " grouped=" << witness_comparison.grouped_token;

                    ASSERT_GE(
                        witness_after.mtp_depth_policy_windows,
                        witness_before.mtp_depth_policy_windows);
                    boundary.dynamic_policy_witness_executed = true;
                    boundary.dynamic_policy_witness_serial_token_exact =
                        witness_comparison.exact;
                    boundary.dynamic_policy_witness_window_delta =
                        witness_after.mtp_depth_policy_windows -
                        witness_before.mtp_depth_policy_windows;
                    boundary.dynamic_policy_witness_attempted_draft_tokens =
                        mtpParityAttemptedDraftTokenCount(
                            witness_counters_before,
                            witness_counters_after);
                    boundary.dynamic_policy_witness_verifier_transactions =
                        mtpParityExecutedTransactionCount(
                            witness_counters_before,
                            witness_counters_after);
                    boundary.dynamic_policy_witness_emitted_tokens =
                        std::move(witness_result.tokens);
                    boundary.dynamic_policy_witness_serial_oracle_tokens =
                        std::move(witness_serial_tokens);
                }
                production_parity_mtp_transaction_boundaries_.push_back(
                    std::move(boundary));
            }
        }

        /**
         * @brief Prove the generated MTP policy from request-owned state.
         *
         * Enabled cells must execute at least one real sidecar/verifier
         * transaction. Fixed-depth cells must retain the exact selected depth;
         * dynamic cells must run the adaptive controller with the declared
         * depth-15 ceiling. Disabled cells reject any speculative work, which
         * prevents persistent MTP infrastructure from masquerading as policy
         * participation.
         */
        void assertProductionParityMTPEvidence() const
        {
            /*
             * A server-shaped MPI fixture keeps follower ranks inside the
             * production command loop. Those ranks execute participant graphs
             * and publish their own PerfStats/sparse-endpoint evidence, but
             * only the continuation authority receives the public response
             * boundary from which `ProductionParityMTPTransactionBoundary` is
             * constructed. Do not ask a follower to mirror that live value.
             */
            if (parityReferenceGenerationCoordination() ==
                    ParityReferenceGenerationCoordination::
                        ArtifactAuthorityOnly &&
                !isRank0())
            {
                return;
            }

            const auto &serial_boundaries =
                productionParityDecodeBoundaries();
            const auto &mtp_boundaries =
                productionParityMTPTransactionBoundaries();
            if (config_.mtp_expectation == ParityMTPExpectation::Disabled)
            {
                EXPECT_TRUE(mtp_boundaries.empty())
                    << "MTP-disabled parity retained a speculative transaction";
                for (const auto &boundary : serial_boundaries)
                {
                    const auto &state = boundary.runtime_state;
                    EXPECT_FALSE(state.mtp_config_enabled);
                    EXPECT_FALSE(state.mtp_request.enabled);
                    EXPECT_EQ(state.mtp_draft_steps, 0u);
                    EXPECT_EQ(state.mtp_verifier_runs, 0u);
                }
                return;
            }

            ASSERT_GT(config_.mtp_expected_draft_depth, 0);
            ASSERT_EQ(
                mtp_boundaries.size(),
                1u)
                << "Each typed MTP matrix cell must publish exactly one declared-depth proof";

            bool enabled = false;
            bool bypassed = false;
            bool transaction_at_expected_depth = false;
            bool dynamic_controller_observed = false;
            uint64_t maximum_draft_steps = 0;
            uint64_t maximum_verifier_runs = 0;
            for (const auto &boundary : mtp_boundaries)
            {
                const auto &state = boundary.after;
                enabled = enabled ||
                          (state.mtp_config_enabled &&
                           state.mtp_request.enabled);
                bypassed = bypassed || state.mtp_bypassed ||
                           state.mtp_request.bypassed;
                maximum_draft_steps =
                    std::max(maximum_draft_steps, state.mtp_draft_steps);
                maximum_verifier_runs =
                    std::max(maximum_verifier_runs, state.mtp_verifier_runs);
                transaction_at_expected_depth =
                    transaction_at_expected_depth ||
                    (boundary.snapshot_execution_draft_depth ==
                         config_.mtp_expected_draft_depth &&
                     boundary.attempted_draft_tokens >=
                         static_cast<uint64_t>(
                             config_.mtp_expected_draft_depth));
                dynamic_controller_observed =
                    dynamic_controller_observed ||
                    (boundary.before.mtp_request.adaptive_depth_enabled &&
                     boundary.before.mtp_request.depth_policy_mode ==
                         "dynamic" &&
                     boundary.before.mtp_min_depth == 1 &&
                     boundary.before.mtp_max_depth ==
                         config_.mtp_expected_draft_depth &&
                     boundary.dynamic_policy_witness_executed &&
                     boundary.dynamic_policy_witness_serial_token_exact &&
                     boundary.dynamic_policy_witness_window_delta > 0 &&
                     boundary
                             .dynamic_policy_witness_attempted_draft_tokens >
                         0 &&
                     boundary
                             .dynamic_policy_witness_verifier_transactions >
                         0);
                EXPECT_EQ(
                    boundary.response_limited_draft_depth,
                    config_.mtp_expected_draft_depth)
                    << "MTP proof response span must match its typed matrix depth";
                EXPECT_EQ(
                    boundary.snapshot_execution_draft_depth,
                    config_.mtp_expected_draft_depth)
                    << "MTP snapshot identity must come from the declared execution depth";
                EXPECT_TRUE(boundary.serial_token_exact)
                    << "MTP transaction retained no exact serial-token proof";
                EXPECT_EQ(
                    boundary.emitted_tokens,
                    boundary.serial_oracle_tokens)
                    << "Stored MTP transaction evidence is internally inconsistent";
                EXPECT_GT(boundary.attempted_draft_tokens, 0u);
                EXPECT_GT(boundary.verifier_transactions, 0u);
            }

            EXPECT_TRUE(enabled)
                << "MTP parity did not enable the production request policy";
            EXPECT_FALSE(bypassed)
                << "MTP parity bypassed its selected production path";
            EXPECT_GT(maximum_draft_steps, 0u)
                << "MTP parity executed no predictor draft";
            EXPECT_GT(maximum_verifier_runs, 0u)
                << "MTP parity executed no grouped verifier";

            if (config_.mtp_expectation ==
                ParityMTPExpectation::FixedDepth)
            {
                EXPECT_TRUE(transaction_at_expected_depth)
                    << "Fixed-depth MTP never executed its declared depth "
                    << config_.mtp_expected_draft_depth;
                for (const auto &boundary : mtp_boundaries)
                {
                    const auto &state = boundary.before;
                    if (!state.mtp_request.enabled)
                        continue;
                    EXPECT_FALSE(state.mtp_request.adaptive_depth_enabled);
                    EXPECT_EQ(state.mtp_request.depth_policy_mode, "fixed");
                    EXPECT_EQ(
                        state.mtp_current_depth,
                        config_.mtp_expected_draft_depth);
                    EXPECT_EQ(
                        state.mtp_max_depth,
                        config_.mtp_expected_draft_depth);
                }
                return;
            }

            EXPECT_TRUE(dynamic_controller_observed)
                << "Dynamic MTP did not publish a serial-exact, full-budget "
                   "adaptive depth window with the declared ceiling "
                << config_.mtp_expected_draft_depth;
        }

        /**
         * @brief Export graph-path and economy evidence for a campaign.
         */
        void finishProductionParityEvidence()
        {
            if (!productionParityCampaignEnabled())
                return;

            const auto records = PerfStatsCollector::snapshot(
                {"forward_graph", "mtp", "moe_rebalance",
                 "moe_overlay_controller", "moe_overlay_residency",
                 "moe_placement"});
            const ProductionParityEvidence evidence =
                collectProductionParityEvidence(
                    records,
                    activeExecutionPath() == ExecutionPath::GRAPH,
                    productionParityExecutionTopology(),
                    productionParityGraphContract(),
                    production_parity_model_context_reused_,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        parity_fixture_started_at_)
                        .count());

            if (evidence.hasAccelerator())
            {
                ASSERT_TRUE(PerfStatsCollector::isEnabled())
                    << "Accelerator production parity requires PerfStats graph evidence";
            }

            const ProductionParityGraphCertification graph_certification =
                certifyProductionParityGraphExecution(evidence);
            EXPECT_TRUE(
                graph_certification ==
                ProductionParityGraphCertification::Certified)
                << "Production parity graph certification failed: topology='"
                << productionParityExecutionTopologyName(
                       evidence.execution_topology)
                << "' contract='"
                << productionParityGraphContractName(evidence.graph_contract)
                << "' result='"
                << productionParityGraphCertificationName(graph_certification)
                << "'.\n"
                << PerfStatsCollector::summaryString({"forward_graph", "mtp"});

            if (evidence.usesHomogeneousGPU())
            {
                if (evidence.device_generation_controller)
                {
                    EXPECT_TRUE(evidence.generation_loop_certified)
                        << "Homogeneous GPU MTP parity used outer-loop policy '"
                        << productionDeviceGenerationPolicyName(
                               evidence.generation_execution_policy)
                        << "' without satisfying its backend-specific authority and dispatch proof.\n"
                        << "Certification detail: "
                        << evidence.generation_certification_detail << "\n"
                        << PerfStatsCollector::summaryString(
                               {"forward_graph", "mtp"});
                }
            }

            assertProductionParityMTPEvidence();
            assertProductionParityMoEOwnerOrderEvidence(records);
            assertProductionParityMoEMovementEvidence(records);

            if (isRank0())
            {
                if (!production_parity_prefix_restore_evidence_.empty())
                {
                    EXPECT_TRUE(ParityCSVArtifactWriter::writePrefixRestore(
                        ensureResultsDir(),
                        getBackendName(),
                        production_parity_prefix_restore_evidence_))
                        << "Cannot write prefix_restore.csv";
                }
                if (!production_parity_mtp_transaction_boundaries_.empty())
                {
                    EXPECT_TRUE(ParityCSVArtifactWriter::writeMTPTransactions(
                        ensureResultsDir(),
                        getBackendName(),
                        production_parity_mtp_transaction_boundaries_))
                        << "Cannot write mtp_transactions.csv";
                }
                EXPECT_TRUE(ParityCSVArtifactWriter::writeProductionPath(
                    ensureResultsDir(),
                    getBackendName(),
                    activePrimaryDevice().toString(),
                    evidence))
                    << "Cannot write production_path.csv";
            }

            // The 75-minute target is measured across the aggregate matrix,
            // including fixture and RAM staging. A cell may finish after that
            // point and must still publish complete numerical diagnostics; the
            // aggregate driver reports the performance miss after all cells run.
        }

        virtual ParityGraphSnapshotPolicy parityGraphSnapshotPolicy(
            ParityForwardPhase phase) const
        {
            (void)phase;
            auto policy = config_.graph_snapshot_policy;
            if (productionParityCampaignEnabled())
            {
                policy.enabled = true;
                policy.require_graph_execution_on_gpu = true;
                policy.require_snapshot_publication = true;
                policy.require_prefill_graph_capture_on_gpu = true;
                /*
                 * Production setup must seal the right graph before request
                 * admission. Replaying the same authenticated prompt to repair
                 * setup would turn a miss into a prefix hit and prove another
                 * lifecycle than the one production serves.
                 */
                policy.retry_prefill_after_warmup_for_capture = false;
            }
            return policy;
        }

        /**
         * @brief Derive graph-capture keys from the authenticated reference pack.
         *
         * Reference filenames already name the mathematical checkpoints the
         * campaign compares. Decode-step prefixes describe time, not graph
         * topology, so they are removed. Chained MTP depths reuse the same
         * depth-zero sidecar graph stage names; both the reference depth name
         * and its depth-zero capture name are retained. The resulting set is
         * cached because the immutable topology is restated before every
         * production forward.
         *
         * @return Sorted, duplicate-free semantic snapshot keys.
         * @throws std::runtime_error when an enabled production campaign has
         *         no usable authenticated checkpoint inventory.
         */
        std::vector<std::string>
        authenticatedReferenceSnapshotCaptureFilter() const
        {
            if (parity_reference_capture_filter_cache_dir_ ==
                    config_.snapshot_dir &&
                !parity_reference_capture_filter_cache_.empty())
            {
                return parity_reference_capture_filter_cache_;
            }

            const std::filesystem::path reference_dir(
                config_.snapshot_dir);
            if (config_.snapshot_dir.empty() ||
                !std::filesystem::is_directory(reference_dir))
            {
                throw std::runtime_error(
                    "Parity graph capture requires an authenticated reference directory: " +
                    config_.snapshot_dir);
            }

            std::set<std::string> keys;
            for (const auto &entry :
                 std::filesystem::directory_iterator(reference_dir))
            {
                if (!entry.is_regular_file() ||
                    entry.path().extension() != ".npy")
                {
                    continue;
                }

                std::string key = entry.path().stem().string();
                constexpr std::string_view kDecodePrefix =
                    "decode_step";
                if (key.rfind(kDecodePrefix, 0) == 0)
                {
                    size_t cursor = kDecodePrefix.size();
                    const size_t first_digit = cursor;
                    while (cursor < key.size() &&
                           key[cursor] >= '0' && key[cursor] <= '9')
                    {
                        ++cursor;
                    }
                    if (cursor == first_digit || cursor >= key.size() ||
                        key[cursor] != '_')
                    {
                        continue;
                    }
                    key = key.substr(cursor + 1);
                }
                if (key.empty())
                    continue;
                keys.insert(key);

                if (key.rfind("MTP", 0) == 0)
                {
                    size_t cursor = 3;
                    const size_t first_digit = cursor;
                    while (cursor < key.size() &&
                           key[cursor] >= '0' && key[cursor] <= '9')
                    {
                        ++cursor;
                    }
                    if (cursor > first_digit && cursor < key.size() &&
                        key[cursor] == '_')
                    {
                        keys.insert("MTP0_" + key.substr(cursor + 1));
                    }
                }
            }

            if (keys.empty())
            {
                throw std::runtime_error(
                    "Parity reference pack contains no .npy checkpoint inventory: " +
                    config_.snapshot_dir);
            }

            parity_reference_capture_filter_cache_dir_ =
                config_.snapshot_dir;
            parity_reference_capture_filter_cache_.assign(
                keys.begin(), keys.end());
            return parity_reference_capture_filter_cache_;
        }

        /**
         * @brief Build the immutable snapshot-key topology for runner setup.
         *
         * Prefill and decode share a serving graph family and therefore cannot
         * change diagnostic graph nodes at phase boundaries. The default
         * inventory is the union of authenticated reference filenames and any
         * explicitly required keys. Capturing every publication is available
         * only through the typed @ref ParitySnapshotCaptureInventory policy;
         * an accidentally empty vector can no longer allocate unrelated graph
         * scratch for every retained bucket.
         *
         * @return Stable semantic union consumed before runner initialization.
         */
        std::vector<std::string> paritySnapshotSetupCaptureFilter() const
        {
            const auto prefill =
                parityGraphSnapshotPolicy(ParityForwardPhase::Prefill);
            const auto decode =
                parityGraphSnapshotPolicy(ParityForwardPhase::Decode);
            const auto &prefill_keys =
                prefill.prefill_snapshot_capture_filter;
            const auto &decode_keys =
                decode.decode_snapshot_capture_filter;

            if (!prefill.enabled && !decode.enabled)
            {
                return {};
            }

            if ((prefill.enabled &&
                 prefill.capture_inventory ==
                     ParitySnapshotCaptureInventory::EveryPublishedOutput) ||
                (decode.enabled &&
                 decode.capture_inventory ==
                     ParitySnapshotCaptureInventory::EveryPublishedOutput))
            {
                return {};
            }

            std::vector<std::string> union_keys;
            if (prefill.enabled)
            {
                union_keys.insert(
                    union_keys.end(), prefill_keys.begin(), prefill_keys.end());
                union_keys.insert(
                    union_keys.end(),
                    prefill.required_prefill_snapshot_keys.begin(),
                    prefill.required_prefill_snapshot_keys.end());
            }
            if (decode.enabled)
            {
                union_keys.insert(
                    union_keys.end(), decode_keys.begin(), decode_keys.end());
                union_keys.insert(
                    union_keys.end(),
                    decode.required_decode_snapshot_keys.begin(),
                    decode.required_decode_snapshot_keys.end());
            }
            const auto reference_keys =
                authenticatedReferenceSnapshotCaptureFilter();
            union_keys.insert(
                union_keys.end(),
                reference_keys.begin(),
                reference_keys.end());

            /*
             * A completed collective is a distinct live graph value, not an
             * alias for its rank-local input.  Reference packs name the
             * mathematical value (for example `layer0_ATTENTION_OUTPUT`),
             * while SnapshotCapture publishes the production completion edge
             * as `layer0_ATTENTION_OUTPUT_ALLREDUCED`.  Install both keys in
             * the immutable snapshot topology before graph construction.  A
             * later comparison can then require the completed edge without a
             * test-owned collective or an eager diagnostic replay.
             */
            if (config_.collective_evidence_source ==
                ParityCollectiveEvidenceSource::PostCollectiveSnapshot)
            {
                for (const std::string &reference_key : reference_keys)
                {
                    for (const std::string &collective_stage :
                         config_.allreduce_stages)
                    {
                        const std::string stage_suffix =
                            "_" + collective_stage;
                        const bool exact_stage =
                            reference_key == collective_stage;
                        const bool qualified_stage =
                            reference_key.size() > stage_suffix.size() &&
                            reference_key.compare(
                                reference_key.size() - stage_suffix.size(),
                                stage_suffix.size(),
                                stage_suffix) == 0;
                        if (exact_stage || qualified_stage)
                        {
                            union_keys.push_back(
                                reference_key + "_ALLREDUCED");
                            break;
                        }
                    }
                }
            }
            std::sort(union_keys.begin(), union_keys.end());
            union_keys.erase(
                std::unique(union_keys.begin(), union_keys.end()),
                union_keys.end());
            return union_keys;
        }

        /**
         * @brief Whether this campaign consumes prefill reference tensors.
         *
         * Decode-only long-context corpora may intentionally retain only the
         * incremental checkpoints after an authenticated cached prefill.
         */
        virtual bool productionParityRequiresPrefillSnapshots() const
        {
            return true;
        }

        /**
         * @brief Observe one numerically compared set of live graph snapshots.
         *
         * The callback runs after every per-layer comparison and before the
         * next request clears or replaces its snapshot bank.  Specialized
         * production campaigns may retain compact provenance such as the
         * identity of a routed expert, but must not mutate inference state or
         * keep raw snapshot pointers beyond this call.  The ordinary parity
         * harness has no additional observation contract.
         *
         * @param phase Production forward phase that published the snapshots.
         * @param step Decode step, or `-1` for prefill.
         * @param layers Completed numerical comparisons for this checkpoint.
         */
        virtual void observeComparedParityCheckpoint(
            ParityForwardPhase phase,
            int step,
            const std::vector<LayerStats> &layers)
        {
            (void)phase;
            (void)step;
            (void)layers;
        }

        std::vector<int> makeBoundedPrefillTokens(
            const std::vector<int> &seed_tokens,
            int max_seq_len = 0) const
        {
            if (seed_tokens.empty())
            {
                throw std::invalid_argument(
                    "cannot build parity prefill input from an empty token list");
            }

            const size_t max_allowed =
                max_seq_len > 0
                    ? static_cast<size_t>(max_seq_len)
                    : std::numeric_limits<size_t>::max();
            const size_t target = std::min(seed_tokens.size(), max_allowed);
            return std::vector<int>(
                seed_tokens.begin(),
                seed_tokens.begin() + static_cast<std::ptrdiff_t>(target));
        }

        virtual bool executeActiveParityForward(
            ParityForwardPhase phase,
            const int *tokens,
            int token_count)
        {
            if (orch_runner_)
            {
                if (phase == ParityForwardPhase::Prefill)
                {
                    orchestration_parity_decode_trajectory_.clear();
                    orchestration_parity_decode_trajectory_index_ = 0;
                    orchestration_parity_force_mode_ =
                        ParityForcedTokenCommitMode::Unknown;
                    for (const int token : readDecodeTokensFromMetadata())
                    {
                        orchestration_parity_decode_trajectory_.push_back(
                            static_cast<int32_t>(token));
                    }
                    std::vector<int32_t> input;
                    input.reserve(static_cast<size_t>(std::max(token_count, 0)));
                    for (int i = 0; i < token_count; ++i)
                        input.push_back(static_cast<int32_t>(tokens[i]));
                    if (!orch_runner_->prefill(input))
                    {
                        LOG_ERROR("[Parity] OrchestrationRunner prefill failed: "
                                  << orch_runner_->lastError());
                        return false;
                    }
                    return true;
                }

                if (!tokens || token_count != 1)
                {
                    LOG_ERROR("[Parity] OrchestrationRunner scalar decode parity "
                              "requires exactly one authenticated reference token");
                    return false;
                }

                const int32_t reference_token =
                    static_cast<int32_t>(tokens[0]);
                const size_t trajectory_index =
                    orchestration_parity_decode_trajectory_index_;
                if (trajectory_index >=
                    orchestration_parity_decode_trajectory_.size())
                {
                    LOG_ERROR(
                        "[Parity] Authenticated production decode trajectory "
                        "ended before requested step "
                        << trajectory_index);
                    return false;
                }
                if (orchestration_parity_decode_trajectory_[trajectory_index] !=
                    reference_token)
                {
                    ADD_FAILURE()
                        << "Production parity requested reference token "
                        << reference_token << " at trajectory step "
                        << trajectory_index << " but authenticated metadata "
                        << "contains "
                        << orchestration_parity_decode_trajectory_[trajectory_index];
                    return false;
                }

                if (orchestration_parity_force_mode_ !=
                    ParityForcedTokenCommitMode::Deferred)
                {
                    /*
                     * A deferred runner uses the successor call below to
                     * forward this token. A device-resident MTP runner commits
                     * it immediately and leaves the exact step snapshots ready
                     * now. The typed result is the authority; backend names and
                     * test configuration never decide this lifecycle edge.
                     */
                    GenerationResult forced =
                        orch_runner_->forceDecodeToken(reference_token);
                    if (!forced.success() || forced.tokens.size() != 1 ||
                        forced.tokens.front() != reference_token)
                    {
                        LOG_ERROR(
                            "[Parity] OrchestrationRunner could not force the "
                            "authenticated reference token: "
                            << forced.error);
                        return false;
                    }

                    if (forced.returned_token_commit ==
                        ReturnedTokenCommitState::Committed)
                    {
                        if (orchestration_parity_force_mode_ ==
                            ParityForcedTokenCommitMode::Deferred)
                        {
                            LOG_ERROR(
                                "[Parity] Forced-token commit policy changed "
                                "inside one request");
                            return false;
                        }
                        orchestration_parity_force_mode_ =
                            ParityForcedTokenCommitMode::Immediate;
                        orchestration_parity_decode_trajectory_index_ =
                            trajectory_index + 1u;
                        return true;
                    }
                    if (forced.returned_token_commit !=
                        ReturnedTokenCommitState::Pending)
                    {
                        LOG_ERROR(
                            "[Parity] Forced-token result omitted its model-row "
                            "commit contract");
                        return false;
                    }
                    if (orchestration_parity_force_mode_ ==
                        ParityForcedTokenCommitMode::Immediate)
                    {
                        LOG_ERROR(
                            "[Parity] Forced-token commit policy changed "
                            "inside one request");
                        return false;
                    }
                    orchestration_parity_force_mode_ =
                        ParityForcedTokenCommitMode::Deferred;
                }

                const size_t successor_index = trajectory_index + 1;
                if (successor_index >=
                    orchestration_parity_decode_trajectory_.size())
                {
                    LOG_ERROR(
                        "[Parity] Authenticated production decode trajectory "
                        "requires one successor token to forward step "
                        << trajectory_index);
                    return false;
                }

                const int32_t successor_token =
                    orchestration_parity_decode_trajectory_[successor_index];
                GenerationResult forwarded =
                    orch_runner_->forceDecodeToken(successor_token);
                if (!forwarded.success() || forwarded.tokens.size() != 1)
                {
                    LOG_ERROR(
                        "[Parity] OrchestrationRunner teacher-forced decode "
                        "transaction failed: "
                        << forwarded.error);
                    return false;
                }
                if (forwarded.tokens.front() != successor_token)
                {
                    LOG_ERROR(
                        "[Parity] OrchestrationRunner returned a different "
                        "forced successor token");
                    return false;
                }
                if (forwarded.returned_token_commit !=
                    ReturnedTokenCommitState::Pending)
                {
                    LOG_ERROR(
                        "[Parity] Deferred forced-token runner changed its "
                        "commit policy while forwarding a predecessor");
                    return false;
                }
                orchestration_parity_decode_trajectory_index_ =
                    successor_index;
                return true;
            }

            if (auto *runner = activeLegacyRunner())
            {
                return runner->forward(tokens, token_count);
            }

            LOG_ERROR("[Parity] No runner available for "
                      << parityForwardPhaseName(phase)
                      << " parity forward");
            return false;
        }

        bool parityPrefillGraphCapturedOrReplayed(
            const PrefixRuntimeStateSnapshot &snapshot) const
        {
            return std::any_of(
                snapshot.prefill_graphs.begin(),
                snapshot.prefill_graphs.end(),
                [](const PrefillGraphRuntimeProbe &probe)
                {
                    return probe.capture_phase == "capture" ||
                           probe.capture_phase == "replay";
                });
        }

        bool parityPrefillGraphWarmedOnly(
            const PrefixRuntimeStateSnapshot &snapshot) const
        {
            return !snapshot.prefill_graphs.empty() &&
                   std::all_of(
                       snapshot.prefill_graphs.begin(),
                       snapshot.prefill_graphs.end(),
                       [](const PrefillGraphRuntimeProbe &probe)
                       {
                           /*
                            * Only an explicit warmup observation authorizes
                            * replay.  "unknown" means the participant did not
                            * publish an execution observation; treating that
                            * absence as warmup can replay the same request
                            * after prefix publication and turn the numerical
                            * proof into a complete cache hit.
                            */
                           return probe.capture_phase == "warmup";
                       });
        }

        bool validateParityGraphSnapshots(
            ParityForwardPhase phase,
            const ParityGraphSnapshotPolicy &policy) const
        {
            if (!policy.enabled)
                return true;

            const DeviceId device = activePrimaryDevice();
            const bool gpu_runner = device.is_gpu();

            if (gpu_runner &&
                policy.require_graph_execution_on_gpu &&
                activeExecutionPath() != ExecutionPath::GRAPH)
            {
                ADD_FAILURE() << "GPU parity " << parityForwardPhaseName(phase)
                              << " must use graph execution";
                return false;
            }

            if (gpu_runner &&
                phase == ParityForwardPhase::Prefill &&
                policy.require_prefill_graph_capture_on_gpu)
            {
                const auto probe = activePrefixStateProbe();
                if (!parityPrefillGraphCapturedOrReplayed(probe))
                {
                    std::ostringstream phases;
                    for (const auto &entry : probe.prefill_graphs)
                    {
                        if (phases.tellp() > 0)
                            phases << ',';
                        phases << entry.capture_phase;
                    }
                    ADD_FAILURE() << "GPU parity prefill required graph capture/replay, "
                                  << "but observed phases ["
                                  << (phases.str().empty() ? "none" : phases.str())
                                  << "]";
                    return false;
                }
            }

            if (policy.require_snapshot_publication)
            {
                const auto keys = activeSnapshotKeys();
                if (keys.empty())
                {
                    ADD_FAILURE() << "Parity " << parityForwardPhaseName(phase)
                                  << " produced no snapshots";
                    return false;
                }

                for (const auto &key : policy.requiredSnapshotKeys(phase))
                {
                    if (std::find(keys.begin(), keys.end(), key) == keys.end())
                    {
                        std::ostringstream present;
                        constexpr size_t kInventoryLimit = 24;
                        for (size_t index = 0;
                             index < keys.size() && index < kInventoryLimit;
                             ++index)
                        {
                            if (index > 0)
                                present << ',';
                            present << keys[index];
                        }
                        if (keys.size() > kInventoryLimit)
                            present << ",...(" << keys.size() << " total)";
                        std::ostringstream related;
                        size_t related_count = 0;
                        for (const auto &published_key : keys)
                        {
                            if (published_key.find("DOMAIN_ROUTE") ==
                                    std::string::npos &&
                                published_key.find("RUNTIME_ROUTE") ==
                                    std::string::npos &&
                                published_key.find("OVERLAY_ROUTE") ==
                                    std::string::npos)
                            {
                                continue;
                            }
                            if (related_count++ > 0)
                                related << ',';
                            related << published_key;
                        }
                        ADD_FAILURE() << "Parity " << parityForwardPhaseName(phase)
                                      << " missing required snapshot '" << key
                                      << "'; published={" << present.str()
                                      << "}; related_route_keys={"
                                      << related.str() << '}';
                        return false;
                    }
                }
            }

            return true;
        }

        bool runParityForwardWithPolicy(
            ParityForwardPhase phase,
            const int *tokens,
            int token_count,
            const ParityGraphSnapshotPolicy &policy)
        {
            const std::string phase_name = parityForwardPhaseName(phase);
            auto total_scope = profileParityScope("forward." + phase_name + ".total");
            if (policy.enabled)
            {
                auto scope = profileParityScope("forward." + phase_name + ".set_snapshot_filter");
                activeSetSnapshotCaptureFilter(
                    productionParityCampaignEnabled()
                        ? paritySnapshotSetupCaptureFilter()
                        : policy.snapshotCaptureFilter(phase));
            }
            else
            {
                auto scope = profileParityScope("forward." + phase_name + ".clear_snapshot_filter");
                activeSetSnapshotCaptureFilter({});
            }
            {
                auto scope = profileParityScope("forward." + phase_name + ".execute");
                if (!executeActiveParityForward(phase, tokens, token_count))
                    return false;
            }

            if (policy.enabled &&
                phase == ParityForwardPhase::Prefill &&
                activePrimaryDevice().is_gpu() &&
                policy.require_prefill_graph_capture_on_gpu &&
                policy.retry_prefill_after_warmup_for_capture &&
                parityPrefillGraphWarmedOnly(activePrefixStateProbe()))
            {
                auto scope = profileParityScope("forward." + phase_name + ".capture_retry");
                activeClearCache();
                activeClearSnapshots();
                if (!executeActiveParityForward(phase, tokens, token_count))
                    return false;
            }

            {
                auto scope = profileParityScope("forward." + phase_name + ".validate_snapshots");
                return validateParityGraphSnapshots(phase, policy);
            }
        }

        bool runParityForward(
            ParityForwardPhase phase,
            const int *tokens,
            int token_count)
        {
            return runParityForwardWithPolicy(
                phase,
                tokens,
                token_count,
                parityGraphSnapshotPolicy(phase));
        }

        bool activeUsesProductionOverlayAuthority() const
        {
            if (!orch_runner_)
                return false;
            const auto &plan =
                orch_runner_->config().moe_routed_expert_plan;
            return plan && plan->usesExpertOverlayAuthority();
        }

        uint64_t activeMoEPlacementEpoch() const
        {
            if (auto *runner = activeLegacyRunner())
                return runner->moePlacementEpoch();
            return 0;
        }

        uint64_t activeMoERuntimeMovementEpoch() const
        {
            if (orch_runner_)
                return orch_runner_->moeRuntimeMovementEpoch();
            if (auto *runner = activeLegacyRunner())
                return runner->moeRuntimeMovementEpoch();
            return 0;
        }

        /**
         * @brief Publish completed device maintenance status at the parity epilogue.
         *
         * Graph-owned MoE maintenance intentionally avoids host readback during
         * decode. Before parity inspects the host-facing movement epoch, this
         * method asks the active runner to publish only the newest completed
         * maintenance status. The runner uses the maintenance stream and does
         * not reset KV, recurrent, sampler, or graph state.
         */
        void activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics()
        {
            if (orch_runner_)
            {
                orch_runner_->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
            }
            else if (auto *runner = activeLegacyRunner())
            {
                runner->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
            }
        }

        /**
         * @brief Decide whether a committed decode step must enter maintenance.
         *
         * The ExpertOverlay hook is wake-only and never performs movement on
         * the inference thread. Tests retain an explicit cadence to bound
         * diagnostic work while exercising the same production authority.
         *
         * @param completed_decode_steps One-based count of committed decode
         *        transactions in the parity request.
         * @return true when the production maintenance boundary must run.
         */
        bool parityMoERebalanceMaintenanceDue(
            size_t completed_decode_steps) const
        {
            const auto &exercise = config_.moe_rebalance_exercise;
            if (!exercise.enabled ||
                exercise.request_every_decode_steps <= 0)
            {
                return false;
            }
            return
                (completed_decode_steps %
                 static_cast<size_t>(
                     exercise.request_every_decode_steps)) == 0;
        }

        bool driveParityMoERebalanceMaintenance(
            const std::string &phase,
            uint64_t committed_tokens,
            int decode_step = -1)
        {
            const auto &exercise = config_.moe_rebalance_exercise;
            if (!exercise.enabled)
                return true;

            const bool production_overlay =
                activeUsesProductionOverlayAuthority();
            if (exercise.require_production_overlay_authority &&
                !production_overlay)
            {
                ADD_FAILURE() << "Parity MoE rebalance exercise requires the production ExpertOverlay authority"
                              << " during " << phase
                              << (decode_step >= 0 ? (" step " + std::to_string(decode_step)) : "");
                return false;
            }

            if (orch_runner_)
            {
                if (!orch_runner_->maybeApplyMoERebalance(
                        committed_tokens))
                {
                    ADD_FAILURE() << "Parity MoE rebalance maintenance failed during "
                                  << phase
                                  << (decode_step >= 0 ? (" step " + std::to_string(decode_step)) : "")
                                  << ": " << orch_runner_->lastError();
                    return false;
                }
                return true;
            }

            ADD_FAILURE() << "Parity ExpertOverlay maintenance requested during " << phase
                          << (decode_step >= 0 ? (" step " + std::to_string(decode_step)) : "")
                          << ", but the test did not construct the production orchestration runner";
            return false;
        }

        void assertParityMoERebalanceExercise(
            uint64_t initial_movement_epoch,
            uint64_t final_movement_epoch) const
        {
            const auto &exercise = config_.moe_rebalance_exercise;
            if (!exercise.enabled || !exercise.require_movement_epoch_advance)
                return;

            ASSERT_GE(final_movement_epoch, initial_movement_epoch)
                << "MoE runtime movement epoch regressed during parity rebalance exercise";
            EXPECT_GE(final_movement_epoch - initial_movement_epoch,
                      exercise.min_movement_epoch_delta)
                << "Parity rebalance exercise did not observe the requested MoE runtime movement epoch advance";
        }

        /**
         * @brief Check if an orchestration runner is active
         *
         * @return true if orch_runner_ is set, false otherwise
         */
        bool hasOrchestrationRunner() const
        {
            return orch_runner_ != nullptr;
        }

        /**
         * @brief Check if a legacy runner is active
         *
         * @return true if an owned or borrowed runner is active, false otherwise
         */
        bool hasLegacyRunner() const
        {
            return activeLegacyRunner() != nullptr;
        }

        /**
         * @brief Read decode tokens from metadata file
         */
        std::vector<int> readDecodeTokensFromMetadata()
        {
            std::string metadata_path = config_.snapshot_dir + "/metadata.txt";
            std::ifstream file(metadata_path);
            if (!file.is_open())
            {
                LOG_WARN("[Parity] Could not open metadata file: " << metadata_path);
                return {};
            }

            std::vector<int> decode_tokens;
            std::string line;
            while (std::getline(file, line))
            {
                // Look for "decode_tokens: X,Y,Z" format
                if (line.find("decode_tokens:") == 0)
                {
                    size_t colon_pos = line.find(':');
                    if (colon_pos != std::string::npos)
                    {
                        std::string tokens_str = line.substr(colon_pos + 1);
                        // Trim leading whitespace
                        size_t start = tokens_str.find_first_not_of(" \t");
                        if (start != std::string::npos)
                        {
                            tokens_str = tokens_str.substr(start);
                        }
                        // Parse comma-separated token IDs
                        std::stringstream ss(tokens_str);
                        std::string token_str;
                        while (std::getline(ss, token_str, ','))
                        {
                            // Trim whitespace from token
                            size_t tok_start = token_str.find_first_not_of(" \t");
                            size_t tok_end = token_str.find_last_not_of(" \t");
                            if (tok_start != std::string::npos && tok_end != std::string::npos)
                            {
                                token_str = token_str.substr(tok_start, tok_end - tok_start + 1);
                            }
                            if (!token_str.empty())
                            {
                                try
                                {
                                    decode_tokens.push_back(std::stoi(token_str));
                                }
                                catch (const std::exception &e)
                                {
                                    LOG_WARN("[Parity] Failed to parse token: " << token_str);
                                }
                            }
                        }
                    }
                    break; // Found decode_tokens line, done
                }
            }
            return decode_tokens;
        }

        /**
         * @brief Read prefill token IDs from metadata file
         *
         * Models with different tokenizers (e.g., Qwen3.5 vocab_size=248320 vs
         * Qwen2/3 vocab_size=151936) produce different token IDs for the same prompt.
         * This reads the actual token_ids used by the PyTorch reference generator
         * so the C++ test uses matching input.
         */
        std::vector<int> readPrefillTokensFromMetadata()
        {
            std::string metadata_path = config_.snapshot_dir + "/metadata.txt";
            std::ifstream file(metadata_path);
            if (!file.is_open())
            {
                LOG_WARN("[Parity] Could not open metadata file: " << metadata_path);
                return {};
            }

            std::vector<int> tokens;
            std::string line;
            while (std::getline(file, line))
            {
                if (line.find("token_ids:") == 0)
                {
                    size_t colon_pos = line.find(':');
                    if (colon_pos != std::string::npos)
                    {
                        std::string tokens_str = line.substr(colon_pos + 1);
                        size_t start = tokens_str.find_first_not_of(" \t");
                        if (start != std::string::npos)
                            tokens_str = tokens_str.substr(start);

                        std::stringstream ss(tokens_str);
                        std::string token_str;
                        while (std::getline(ss, token_str, ','))
                        {
                            size_t tok_start = token_str.find_first_not_of(" \t");
                            size_t tok_end = token_str.find_last_not_of(" \t");
                            if (tok_start != std::string::npos && tok_end != std::string::npos)
                                token_str = token_str.substr(tok_start, tok_end - tok_start + 1);
                            if (!token_str.empty())
                            {
                                try
                                {
                                    tokens.push_back(std::stoi(token_str));
                                }
                                catch (const std::exception &)
                                {
                                    LOG_WARN("[Parity] Failed to parse prefill token: " << token_str);
                                }
                            }
                        }
                    }
                    break;
                }
            }
            return tokens;
        }

        /**
         * @brief Run single-device prefill parity test and return summary
         *
         * This is the main test driver for SINGLE-DEVICE tests - compares
         * layer-by-layer against PyTorch. Calls setupPipeline() to create
         * a single-device runner, then delegates to runPrefillParity().
         *
         * For multi-device LOCAL TP tests, use runTPPrefillParity() instead.
         * For LocalPP tests, call setupPipeline() first, then runPrefillParity().
         */
        ParityTestSummary runSingleDevicePrefillParity()
        {
            // Setup single-device pipeline then delegate
            EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
            return runPrefillParity();
        }

        /**
         * @brief Run prefill parity test using existing runner_
         *
         * Compares layer-by-layer Llaminar snapshots against PyTorch reference.
         * Requires runner_ to be already set up (via setupPipeline() or
         * setupLocalPPPipeline() etc).
         *
         * Use this for LocalPP tests where you want to set up the PP pipeline
         * first, then run parity comparison without overwriting the runner.
         *
         * @return ParityTestSummary with layer-by-layer metrics
         */
        ParityTestSummary runPrefillParity()
        {
            ParityTestSummary summary;

            // Require either production runner family to be already set up.
            if (!orch_runner_ && !activeLegacyRunner())
            {
                LOG_ERROR("[Parity] runPrefillParity() has no active runner - "
                          "ensure setupPipeline() or setupOrchestrationRunner() was called first");
                return summary;
            }

            // Run prefill
            bool success = runParityForward(
                ParityForwardPhase::Prefill,
                config_.token_ids.data(),
                static_cast<int>(config_.token_ids.size()));
            EXPECT_TRUE(success) << "Prefill forward pass failed";
            if (!success)
                return summary;
            exportActiveSnapshotsForParityDiagnostics(ParityForwardPhase::Prefill, -1);
            /* Production prefill retires its own exact prompt progress. The
             * parity harness must not manufacture a decode boundary merely to
             * wake ExpertOverlay after prefill. */

            /*
             * A segmented captured prefill is numerically meaningful only if
             * every sequence-shaped checkpoint has been reconstructed to the
             * authenticated prompt width. Comparing a final short chunk against
             * a full Hugging Face tensor through min(size) would hide exactly
             * the broken aggregation this production campaign is intended to
             * expose. Ordinary exact-bucket and decode paths retain their
             * historical shape flexibility.
             */
            const auto prefill_probe = activePrefixStateProbe();
            const bool require_complete_segmented_prefill_snapshots =
                prefill_probe.prefill_chunk_successful_schedules > 0;

            int n_layers = parityLayerCount();

            // Stages to compare per layer
            std::vector<std::string> per_layer_stages = {
                "ATTENTION_NORM",
                // GDN sub-stages (skipped for FA layers where they don't exist)
                "QKV_PROJECTION", "GDN_Z_PROJECTION", "GDN_CONV1D_OUTPUT", "GDN_DELTA_RULE_OUTPUT", "GDN_NORM_GATE_OUTPUT",
                // Standard attention sub-stages (skipped for GDN layers)
                "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
                "FA_GATE",
                "Q_NORM", "K_NORM", // Qwen3 per-head QK RMSNorm (skipped if not available)
                "Q_ROPE", "K_ROPE",
                "ATTENTION_CONTEXT", "ATTENTION_CONTEXT_GATED", "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL",
                "FFN_NORM",
                // Dense FFN sub-stages (skipped for MoE layers)
                "FFN_GATE", "FFN_UP", "FFN_SWIGLU", "FFN_DOWN",
                // MoE sub-stages (skipped for dense FFN layers)
                "MOE_ROUTER_OUTPUT", "MOE_ROUTING_INDICES", "MOE_ROUTING_WEIGHTS",
                "MOE_EXPERT_OUTPUT", "MOE_SHARED_EXPERT_OUTPUT", "MOE_SHARED_GATE_OUTPUT", "MOE_COMBINED_OUTPUT",
                "FFN_RESIDUAL"};
            auto snapshot_keys = activeSnapshotKeys();
            std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());

            // Compare embedding
            auto pytorch_embedding = loadPyTorchSnapshot("EMBEDDING");
            if (available_snapshots.count("EMBEDDING"))
            {
                size_t llaminar_size;
                const float *llaminar_data = activeSnapshot("EMBEDDING", llaminar_size);

                // Debug sizes and first values (rank 0 only)
                if (isRank0())
                {
                    LOG_INFO("[Parity Debug] EMBEDDING - llaminar_size=" << llaminar_size
                                                                         << " pytorch_size=" << pytorch_embedding.size());
                    if (llaminar_data && llaminar_size >= 8)
                    {
                        LOG_INFO("[Parity Debug] Llaminar first 8: "
                                 << llaminar_data[0] << "," << llaminar_data[1] << ","
                                 << llaminar_data[2] << "," << llaminar_data[3] << ","
                                 << llaminar_data[4] << "," << llaminar_data[5] << ","
                                 << llaminar_data[6] << "," << llaminar_data[7]);
                    }
                    if (!pytorch_embedding.empty() && pytorch_embedding.size() >= 8)
                    {
                        LOG_INFO("[Parity Debug] PyTorch first 8: "
                                 << pytorch_embedding[0] << "," << pytorch_embedding[1] << ","
                                 << pytorch_embedding[2] << "," << pytorch_embedding[3] << ","
                                 << pytorch_embedding[4] << "," << pytorch_embedding[5] << ","
                                 << pytorch_embedding[6] << "," << pytorch_embedding[7]);
                    }
                }

                if (llaminar_data && !pytorch_embedding.empty())
                {
                    if (require_complete_segmented_prefill_snapshots)
                    {
                        EXPECT_EQ(llaminar_size, pytorch_embedding.size())
                            << "Segmented prefill EMBEDDING checkpoint must "
                               "contain every authenticated prompt row";
                        if (llaminar_size != pytorch_embedding.size())
                            return summary;
                    }
                    summary.embedding_cosine = computeCosineSimilarity(
                        llaminar_data, pytorch_embedding.data(),
                        std::min(llaminar_size, pytorch_embedding.size()));
                }
            }
            summary.embedding_passed = (summary.embedding_cosine >= config_.cosine_threshold);

            // Pre-compute GDN head config for V-head permutation
            const auto gdn_cfg = getGDNHeadConfig();
            const auto moe_cfg = getMoEConfig();
            if (gdn_cfg.needsPermutation())
            {
                LOG_INFO("[Parity] GDN V-head permutation active: n_k=" << gdn_cfg.n_k_heads
                                                                        << " n_v=" << gdn_cfg.n_v_heads
                                                                        << " d=" << gdn_cfg.d_state
                                                                        << " heads_per_group=" << gdn_cfg.headsPerGroup());
            }

            // Compare each layer
            for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
            {
                LayerStats stats;
                stats.layer_idx = layer_idx;
                float sum_cosine = 0.0f;

                for (const auto &stage : per_layer_stages)
                {
                    // Skip excluded stages (e.g., Q/K/V_PROJECTION for GLOBAL scope TP)
                    if (!config_.excluded_stages.empty())
                    {
                        bool is_excluded = std::find(
                                               config_.excluded_stages.begin(),
                                               config_.excluded_stages.end(),
                                               stage) != config_.excluded_stages.end();
                        if (is_excluded)
                            continue;
                    }

                    const std::string semantic_key =
                        "layer" + std::to_string(layer_idx) + "_" + stage;
                    const std::string pytorch_key = semantic_key;

                    auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                    if (pytorch_data.empty())
                        continue;

                    const bool stage_requires_reduction =
                        !config_.allreduce_stages.empty() &&
                        std::find(config_.allreduce_stages.begin(), config_.allreduce_stages.end(), stage) !=
                            config_.allreduce_stages.end();
                    const auto snapshot_selection = selectParitySnapshot(
                        semantic_key,
                        stage_requires_reduction,
                        config_.collective_evidence_source);
                    const std::string &llaminar_key = snapshot_selection.key;
                    const bool requires_cross_rank_sum =
                        snapshot_selection.reduction ==
                        ParitySnapshotReduction::CrossRankSum;

                    const bool has_local_snapshot = available_snapshots.count(llaminar_key) > 0;
                    if (!has_local_snapshot &&
                        snapshot_selection.requires_post_collective_key)
                    {
                        ADD_FAILURE()
                            << "Parity requires live post-collective snapshot '"
                            << llaminar_key << "' for semantic stage '" << semantic_key
                            << "'. The pre-collective partial is not a valid fallback.";
                        continue;
                    }
                    float ranks_with_snapshot = has_local_snapshot ? 1.0f : 0.0f;
                    if (requires_cross_rank_sum)
                    {
                        if (!mpi_ctx_)
                        {
                            ADD_FAILURE()
                                << "Cross-rank parity snapshot '" << llaminar_key
                                << "' requires an MPI context";
                            continue;
                        }
                        float local_has_snapshot = ranks_with_snapshot;
                        parityCoordinationMPIContext()->allreduce_sum(
                            &local_has_snapshot, &ranks_with_snapshot, 1);
                    }

                    if (!has_local_snapshot)
                        continue;

                    size_t llaminar_size;
                    const float *llaminar_data = activeSnapshot(llaminar_key, llaminar_size);
                    if (!llaminar_data)
                        continue;

                    if (require_complete_segmented_prefill_snapshots &&
                        llaminar_size != pytorch_data.size())
                    {
                        ADD_FAILURE()
                            << "Segmented prefill checkpoint '" << semantic_key
                            << "' has " << llaminar_size << " elements, but "
                            << "the Hugging Face reference has "
                            << pytorch_data.size()
                            << ". A final chunk is not a valid replacement "
                               "for the complete prompt checkpoint.";
                        continue;
                    }

                    // Apply GDN V-head permutation if needed (Llaminar ratio-grouped → PyTorch interleaved)
                    auto permuted = applyGDNHeadPermutation(llaminar_data, llaminar_size, stage, gdn_cfg);
                    const float *compare_data = permuted.empty() ? llaminar_data : permuted.data();

                    StageComparisonResult result;
                    // Routed/TP allreduce: reconstruct full output from rank partials.
                    bool did_allreduce = false;
                    std::vector<float> allreduced_buf;
                    if (requires_cross_rank_sum &&
                        static_cast<int>(ranks_with_snapshot + 0.5f) == mpiWorldSize())
                    {
                        allreduced_buf.resize(llaminar_size);
                        parityCoordinationMPIContext()->allreduce_sum(
                            compare_data,
                            allreduced_buf.data(),
                            llaminar_size);
                        compare_data = allreduced_buf.data();
                        did_allreduce = true;
                    }
                    if (stage == "MOE_ROUTING_INDICES")
                    {
                        result = compareRoutingIndices(compare_data, pytorch_data, llaminar_size,
                                                       moe_cfg.top_k, stage);
                    }
                    else if (stage == "MOE_ROUTING_WEIGHTS")
                    {
                        std::string idx_key = "layer" + std::to_string(layer_idx) + "_MOE_ROUTING_INDICES";
                        size_t ll_idx_size;
                        const float *ll_idx = activeSnapshot(idx_key, ll_idx_size);
                        auto pt_idx = loadPyTorchSnapshot(idx_key);
                        if (ll_idx && !pt_idx.empty())
                            result = compareRoutingWeights(compare_data, pytorch_data, ll_idx, pt_idx,
                                                           llaminar_size, moe_cfg.top_k, moe_cfg.num_experts, stage);
                        else
                            result = compareTensors(compare_data, pytorch_data, llaminar_size, stage);
                    }
                    else
                    {
                        result = compareTensors(compare_data, pytorch_data, llaminar_size, stage);
                    }
                    stats.stage_results.push_back(result);

                    // Routing stages use set-overlap (Jaccard), not cosine similarity.
                    // Include them in stage_results for logging/CSV but exclude from
                    // layer-level min/avg aggregation to avoid metric contamination.
                    if (!isRoutingStage(stage))
                    {
                        stats.stages_compared++;
                        sum_cosine += result.cosine_similarity;
                    }

                    // Per-stage logging for diagnostics
                    if (result.is_routing_stage)
                    {
                        LOG_INFO("[Parity] Layer " << layer_idx << " " << stage
                                                   << " routing_overlap=" << std::fixed << std::setprecision(6) << result.routing_overlap
                                                   << (!std::isnan(result.routing_top1_match) ? " top1_match=" + std::to_string(result.routing_top1_match) : "")
                                                   << (!std::isnan(result.routing_weight_l1) ? " weight_l1=" + std::to_string(result.routing_weight_l1) : "")
                                                   << " size=" << llaminar_size);
                    }
                    else
                    {
                        LOG_INFO("[Parity] Layer " << layer_idx << " " << stage
                                                   << (snapshot_selection.requires_post_collective_key
                                                           ? " (production post-collective)"
                                                           : (did_allreduce
                                                                  ? " (cross-rank summed)"
                                                                  : ""))
                                                   << " cosine=" << std::fixed << std::setprecision(6) << result.cosine_similarity
                                                   << " size=" << llaminar_size);
                    }

                    // Per-row cosine diagnostic for ATTENTION_CONTEXT (layer 0 only)
                    if (stage == "ATTENTION_CONTEXT" && layer_idx == 0)
                    {
                        size_t compare_size = std::min(llaminar_size, pytorch_data.size());
                        const ModelContext *model_ctx =
                            activeModelContextForDiagnostics();
                        if (!model_ctx)
                        {
                            ADD_FAILURE()
                                << "ATTENTION_CONTEXT diagnostics require active model metadata";
                            continue;
                        }
                        size_t head_dim_val = model_ctx->model().key_length > 0
                                                  ? model_ctx->model().key_length
                                                  : (model_ctx->model().embedding_length /
                                                     model_ctx->model().head_count);
                        size_t n_cols = model_ctx->model().head_count * head_dim_val;
                        size_t n_rows = compare_size / n_cols;
                        std::stringstream row_cosines;
                        float max_abs_diff = 0.0f;
                        size_t max_diff_idx = 0;
                        for (size_t r = 0; r < n_rows && r < 9; ++r)
                        {
                            float rc = computeCosineSimilarity(
                                llaminar_data + r * n_cols,
                                pytorch_data.data() + r * n_cols,
                                n_cols);
                            row_cosines << "row" << r << "=" << std::fixed << std::setprecision(6) << rc << " ";
                            for (size_t c = 0; c < n_cols; ++c)
                            {
                                float diff = std::abs(llaminar_data[r * n_cols + c] - pytorch_data[r * n_cols + c]);
                                if (diff > max_abs_diff)
                                {
                                    max_abs_diff = diff;
                                    max_diff_idx = r * n_cols + c;
                                }
                            }
                        }
                        LOG_INFO("[SingleGPU Diag] " << stage << " per-row cosine: " << row_cosines.str());
                        LOG_INFO("[SingleGPU Diag] " << stage
                                                     << " max_abs_diff=" << max_abs_diff
                                                     << " at idx=" << max_diff_idx
                                                     << " (row=" << max_diff_idx / n_cols
                                                     << " col=" << max_diff_idx % n_cols << ")"
                                                     << " llaminar=" << llaminar_data[max_diff_idx]
                                                     << " pytorch=" << pytorch_data[max_diff_idx]);
                    }

                    // Track kurtosis for activation outlier monitoring
                    float kurt = result.llaminar_stats.kurtosis;
                    if (kurt > stats.max_kurtosis)
                    {
                        stats.max_kurtosis = kurt;
                        stats.max_kurtosis_stage = stage;
                    }

                    if (!isRoutingStage(stage) && result.cosine_similarity < stats.min_cosine_sim)
                    {
                        stats.min_cosine_sim = result.cosine_similarity;
                        stats.worst_stage = stage;
                    }
                }

                // Compute per-stage cosine drops (error introduced by each stage)
                // Skip transitions involving routing stages (different metric scale)
                if (stats.stage_results.size() >= 2)
                {
                    for (size_t s = 1; s < stats.stage_results.size(); ++s)
                    {
                        bool curr_routing = isRoutingStage(stats.stage_results[s].stage_name);
                        bool prev_routing = isRoutingStage(stats.stage_results[s - 1].stage_name);
                        if (curr_routing || prev_routing)
                            continue; // Skip drops involving routing stages

                        float drop = stats.stage_results[s - 1].cosine_similarity -
                                     stats.stage_results[s].cosine_similarity;
                        stats.stage_results[s].cosine_drop = drop;

                        if (drop > stats.max_cosine_drop)
                        {
                            stats.max_cosine_drop = drop;
                            stats.max_drop_stage = stats.stage_results[s].stage_name;
                        }
                    }
                }

                if (stats.stages_compared > 0)
                {
                    stats.avg_cosine_sim = sum_cosine / stats.stages_compared;
                }

                // Pass criteria based on config
                float check_value = config_.use_avg_cosine ? stats.avg_cosine_sim : stats.min_cosine_sim;
                stats.passed = (check_value >= config_.cosine_threshold);

                summary.layer_stats.push_back(stats);
            }

            // Compare LM_HEAD
            auto pytorch_lm_head = loadPyTorchSnapshot("LM_HEAD");
            if (available_snapshots.count("LM_HEAD") && !pytorch_lm_head.empty())
            {
                size_t llaminar_size;
                const float *llaminar_data = activeSnapshot("LM_HEAD", llaminar_size);
                if (llaminar_data)
                {
                    size_t vocab_size =
                        static_cast<size_t>(getActiveVocabSize());
                    size_t llaminar_seq_len = llaminar_size / vocab_size;
                    size_t pytorch_seq_len = pytorch_lm_head.size() / vocab_size;

                    // Llaminar may only output the last row (last-token-only optimization).
                    // PyTorch always outputs all rows. Compare the last row from each.
                    size_t llaminar_last_offset = (llaminar_seq_len > 0) ? (llaminar_seq_len - 1) * vocab_size : 0;
                    size_t pytorch_last_offset = (pytorch_seq_len > 0) ? (pytorch_seq_len - 1) * vocab_size : 0;

                    // Cosine similarity on last-row logits only
                    auto result = compareTensors(
                        llaminar_data + llaminar_last_offset,
                        std::vector<float>(pytorch_lm_head.begin() + pytorch_last_offset,
                                           pytorch_lm_head.begin() + pytorch_last_offset + vocab_size),
                        vocab_size, "LM_HEAD");
                    summary.lm_head_cosine = result.cosine_similarity;

                    if (llaminar_seq_len > 0 && pytorch_seq_len > 0)
                    {
                        summary.lm_head_kl = computeKLDivergence(
                            llaminar_data + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size);

                        summary.lm_head_top1 = computeTopKOverlap(
                            llaminar_data + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 1);

                        summary.lm_head_top5 = computeTopKOverlap(
                            llaminar_data + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 5);

                        // Check if PyTorch's top-1 token is in llaminar's top-K
                        if (config_.pytorch_top1_in_topk > 0)
                        {
                            float topk_recall = pytorchTop1InLlaminarTopK(
                                llaminar_data + llaminar_last_offset,
                                pytorch_lm_head.data() + pytorch_last_offset,
                                vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                            summary.lm_head_pytorch_top1_in_top3 = (topk_recall >= 1.0f - 1e-6f);
                        }
                        else
                        {
                            summary.lm_head_pytorch_top1_in_top3 = true; // gate disabled
                        }
                    }
                }
            }
            summary.lm_head_passed = (summary.lm_head_kl < config_.kl_threshold) &&
                                     ((summary.lm_head_top1 * 100.0f) >= config_.min_top1_accuracy) &&
                                     ((summary.lm_head_top5 * 100.0f) >= config_.min_top5_accuracy) &&
                                     (config_.pytorch_top1_in_topk <= 0 || summary.lm_head_pytorch_top1_in_top3);

            // Count early layers passed
            summary.early_layers_passed = summary.embedding_passed ? 1 : 0;
            for (int i = 0; i < std::min(config_.early_layers_count, static_cast<int>(summary.layer_stats.size())); ++i)
            {
                if (summary.layer_stats[i].passed)
                    summary.early_layers_passed++;
            }

            // Count total layers passed
            summary.total_layers_passed = summary.embedding_passed ? 1 : 0;
            for (const auto &stats : summary.layer_stats)
            {
                if (stats.passed)
                    summary.total_layers_passed++;
            }

            // Overall pass
            summary.overall_passed = (summary.early_layers_passed >= config_.min_early_layers_passed) &&
                                     summary.lm_head_passed;

            observeComparedParityCheckpoint(
                ParityForwardPhase::Prefill,
                -1,
                summary.layer_stats);

            return summary;
        }

        /**
         * @brief Run TP-aware prefill parity test
         *
         * For tensor-parallel tests, this compares:
         * 1. Per-device partial outputs against corresponding PyTorch slices
         * 2. Combined (concatenated) outputs against full PyTorch reference
         *
         * Requires runner_ to be a RankOrchestrator (will cast and check).
         *
         * @return TPParityTestSummary with per-device and combined metrics
         */
        TPParityTestSummary runTPPrefillParity()
        {
            TPParityTestSummary summary;

            // Only setup pipeline if not already configured
            // (Test may have already called setupLocalTPPipeline() or similar)
            if (!activeLegacyRunner())
            {
                EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
                if (!activeLegacyRunner())
                    return summary;
            }

            // Try to cast to RankOrchestrator for TP snapshot access
            auto *multi_device = dynamic_cast<RankOrchestrator *>(activeLegacyRunner());
            if (!multi_device)
            {
                LOG_ERROR("[TP Parity] runner_ is not a RankOrchestrator - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPPrefillParity()");
                return summary;
            }

            summary.tp_degree = multi_device->device_count();
            for (int i = 0; i < summary.tp_degree; ++i)
            {
                auto *dev_runner = multi_device->deviceRunner(i);
                if (dev_runner)
                {
                    // Use index-based name for simplicity
                    // RankOrchestrator stores device info in config
                    summary.device_names.push_back("TP_rank_" + std::to_string(i));
                }
            }

            LOG_INFO("[TP Parity] Running with " << summary.tp_degree << " devices");

            // Run prefill
            LOG_INFO("[TP Parity] Calling forward() with " << config_.token_ids.size() << " tokens...");
            bool success = runParityForward(
                ParityForwardPhase::Prefill,
                config_.token_ids.data(),
                static_cast<int>(config_.token_ids.size()));
            LOG_INFO("[TP Parity] forward() returned: " << (success ? "SUCCESS" : "FAILURE"));
            EXPECT_TRUE(success) << "Prefill forward pass failed";
            if (!success)
                return summary;
            exportActiveSnapshotsForParityDiagnostics(ParityForwardPhase::Prefill, -1);

            int n_layers = parityLayerCount();
            size_t seq_len = config_.token_ids.size();
            size_t d_model = model_ctx_->model().embedding_length;
            size_t n_heads = model_ctx_->headCount();
            size_t vocab_size = model_ctx_->model().vocab_size;

            size_t head_dim = d_model / n_heads;
            // The authenticated tensor cardinality below owns each stage's
            // width. A copied dimension table cannot represent model-specific
            // fused gates or other widened projections without silently
            // truncating their comparison.
            std::vector<std::string> per_layer_stages = {
                "ATTENTION_NORM",
                "Q_PROJECTION", // DIAGNOSTIC: check QKV GEMM output
                "K_PROJECTION", // DIAGNOSTIC: check K projection
                "V_PROJECTION", // DIAGNOSTIC: check V projection
                "Q_ROPE",       // DIAGNOSTIC: check RoPE output
                "K_ROPE",       // DIAGNOSTIC: check K RoPE output
                "ATTENTION_CONTEXT",
                "ATTENTION_OUTPUT",
                "FFN_NORM",
                "FFN_SWIGLU",
                "FFN_DOWN",
                "FFN_RESIDUAL"};

            // Get snapshot keys
            auto snapshot_keys = activeSnapshotKeys();
            std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());
            LOG_INFO("[TP Parity] Got " << snapshot_keys.size() << " snapshot keys after forward()");
            if (snapshot_keys.size() < 50)
            {
                LOG_WARN("[TP Parity] WARNING: Expected 200+ snapshot keys, got only " << snapshot_keys.size());
                for (const auto &key : snapshot_keys)
                {
                    LOG_INFO("[TP Parity]   Available: " << key);
                }
            }

            // Compare embedding
            auto pytorch_embedding = loadPyTorchSnapshot("EMBEDDING");
            if (available_snapshots.count("EMBEDDING") && !pytorch_embedding.empty())
            {
                TPSnapshot tp_snap = multi_device->getTPSnapshot("EMBEDDING");
                const size_t embedding_cols = parityTPReferenceColumnCount(
                    pytorch_embedding.size(), seq_len);
                summary.embedding_result = compareTPSnapshot(
                    tp_snap, pytorch_embedding, seq_len, embedding_cols);
            }

            // Compare each layer
            for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
            {
                TPLayerStats layer_stats;
                layer_stats.layer_idx = layer_idx;
                layer_stats.tp_degree = summary.tp_degree;

                float sum_combined_cosine = 0.0f;

                for (const auto &stage_name : per_layer_stages)
                {
                    std::string llaminar_key = "layer" + std::to_string(layer_idx) + "_" + stage_name;
                    std::string pytorch_key = llaminar_key;

                    if (!available_snapshots.count(llaminar_key))
                        continue;

                    auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                    if (pytorch_data.empty())
                        continue;

                    TPSnapshot tp_snap = multi_device->getTPSnapshot(llaminar_key);
                    const size_t reference_cols = parityTPReferenceColumnCount(
                        pytorch_data.size(), seq_len);
                    auto stage_result = compareTPSnapshot(
                        tp_snap, pytorch_data, seq_len, reference_cols);
                    // The TP snapshot key carries its layer prefix for lookup;
                    // canonical CSV rows already have a layer column and use
                    // the semantic stage name shared by every other topology.
                    stage_result.stage_name = stage_name;
                    stage_result.combined_result.stage_name = stage_name;

                    layer_stats.stage_results.push_back(stage_result);
                    layer_stats.stages_compared++;
                    sum_combined_cosine += stage_result.combined_cosine;

                    if (stage_result.combined_cosine < layer_stats.min_combined_cosine)
                    {
                        layer_stats.min_combined_cosine = stage_result.combined_cosine;
                        layer_stats.worst_stage = stage_name;
                    }
                }

                if (layer_stats.stages_compared > 0)
                {
                    layer_stats.avg_combined_cosine = sum_combined_cosine / layer_stats.stages_compared;
                }

                // ====================================================================
                // CPU-side RoPE crosscheck for layer 0 to diagnose TP Q_ROPE divergence
                // ====================================================================
                if (layer_idx == 0)
                {
                    auto pytorch_qp = loadPyTorchSnapshot("layer0_Q_PROJECTION");
                    auto pytorch_qr = loadPyTorchSnapshot("layer0_Q_ROPE");
                    TPSnapshot tp_qp = multi_device->getTPSnapshot("layer0_Q_PROJECTION");
                    TPSnapshot tp_qr = multi_device->getTPSnapshot("layer0_Q_ROPE");

                    if (!tp_qp.device_data.empty() && !tp_qr.device_data.empty() &&
                        !pytorch_qp.empty() && !pytorch_qr.empty())
                    {
                        const auto &qp_dev0 = tp_qp.device_data[0].data;
                        const auto &qr_dev0 = tp_qr.device_data[0].data;
                        size_t local_cols = tp_qp.device_data[0].cols;

                        if (local_cols > 0 && qp_dev0.size() >= seq_len * local_cols &&
                            qr_dev0.size() >= seq_len * local_cols)
                        {
                            float theta_base = model_ctx_->model().rope_theta;
                            size_t hd = head_dim;
                            size_t half_hd = hd / 2;

                            LOG_INFO("[RoPE CrossCheck] theta_base=" << theta_base
                                                                     << " head_dim=" << hd << " local_cols=" << local_cols
                                                                     << " pytorch_qp_cols=" << d_model);

                            // Check multiple RoPE pairs across different rows and heads
                            for (size_t row : {0ul, 4ul, 8ul})
                            {
                                if (row >= seq_len)
                                    break;
                                for (size_t pair_i : {0ul, 4ul, 15ul, 31ul})
                                {
                                    if (pair_i >= half_hd)
                                        continue;
                                    // Check head 0 and head 1
                                    for (size_t h : {0ul, 1ul})
                                    {
                                        size_t col_i0 = h * hd + pair_i;           // first half
                                        size_t col_i1 = h * hd + pair_i + half_hd; // second half
                                        if (col_i1 >= local_cols)
                                            continue;

                                        size_t idx_i0 = row * local_cols + col_i0;
                                        size_t idx_i1 = row * local_cols + col_i1;

                                        // Llaminar Q_PROJ values
                                        float x0_ll = qp_dev0[idx_i0];
                                        float x1_ll = qp_dev0[idx_i1];

                                        // PyTorch Q_PROJ values (full 896 cols)
                                        float x0_pt = pytorch_qp[row * d_model + col_i0];
                                        float x1_pt = pytorch_qp[row * d_model + col_i1];

                                        // Compute expected RoPE
                                        float inv_freq_i = std::exp(-std::log(theta_base) * 2.0f * static_cast<float>(pair_i) / static_cast<float>(hd));
                                        float angle = static_cast<float>(row) * inv_freq_i;
                                        float cos_a = std::cos(angle);
                                        float sin_a = std::sin(angle);

                                        float cpu_i0 = x0_ll * cos_a - x1_ll * sin_a;
                                        float cpu_i1 = x0_ll * sin_a + x1_ll * cos_a;

                                        // Actual outputs
                                        float ll_i0 = qr_dev0[idx_i0];
                                        float ll_i1 = qr_dev0[idx_i1];
                                        float pt_i0 = pytorch_qr[row * d_model + col_i0];
                                        float pt_i1 = pytorch_qr[row * d_model + col_i1];

                                        float diff_i0 = std::abs(ll_i0 - pt_i0);
                                        float diff_i1 = std::abs(ll_i1 - pt_i1);

                                        if (diff_i0 > 0.01f || diff_i1 > 0.01f)
                                        {
                                            LOG_INFO("[RoPE CrossCheck] MISMATCH row=" << row
                                                                                       << " head=" << h << " pair=" << pair_i
                                                                                       << " | Q_PROJ: ll_x0=" << x0_ll << " pt_x0=" << x0_pt
                                                                                       << " ll_x1=" << x1_ll << " pt_x1=" << x1_pt
                                                                                       << " | inv_freq=" << inv_freq_i << " angle=" << angle
                                                                                       << " cos=" << cos_a << " sin=" << sin_a
                                                                                       << " | CPU_expected: i0=" << cpu_i0 << " i1=" << cpu_i1
                                                                                       << " | GPU_actual: i0=" << ll_i0 << " i1=" << ll_i1
                                                                                       << " | PyTorch: i0=" << pt_i0 << " i1=" << pt_i1
                                                                                       << " | diff_i0=" << diff_i0 << " diff_i1=" << diff_i1);
                                        }
                                    }
                                }
                            }
                            // Summary: compute full CPU RoPE on row 8 and measure cosine
                            if (seq_len > 8)
                            {
                                std::vector<float> cpu_rope_row8(local_cols);
                                const float *qp_row8 = qp_dev0.data() + 8 * local_cols;
                                size_t local_heads = local_cols / hd;
                                for (size_t h = 0; h < local_heads; ++h)
                                {
                                    for (size_t i = 0; i < half_hd; ++i)
                                    {
                                        float inv_f = std::exp(-std::log(theta_base) * 2.0f * static_cast<float>(i) / static_cast<float>(hd));
                                        float ang = 8.0f * inv_f;
                                        float c = std::cos(ang);
                                        float s = std::sin(ang);
                                        float x0 = qp_row8[h * hd + i];
                                        float x1 = qp_row8[h * hd + i + half_hd];
                                        cpu_rope_row8[h * hd + i] = x0 * c - x1 * s;
                                        cpu_rope_row8[h * hd + i + half_hd] = x0 * s + x1 * c;
                                    }
                                }
                                float cos_cpu_vs_ll = computeCosineSimilarity(
                                    cpu_rope_row8.data(), qr_dev0.data() + 8 * local_cols, local_cols);
                                float cos_cpu_vs_pt = computeCosineSimilarity(
                                    cpu_rope_row8.data(),
                                    extractColumnSlice(pytorch_qr.data(), seq_len, d_model, 0, local_cols).data() + 8 * local_cols,
                                    local_cols);
                                LOG_INFO("[RoPE CrossCheck] Row8 cosine: CPU_vs_llaminar=" << std::fixed << std::setprecision(6) << cos_cpu_vs_ll
                                                                                           << " CPU_vs_pytorch=" << cos_cpu_vs_pt);
                            }
                        }
                    }
                }

                // Pass criteria based on combined cosine
                layer_stats.passed = (layer_stats.avg_combined_cosine >= config_.cosine_threshold);

                summary.layer_stats.push_back(layer_stats);
            }

            // Compare LM_HEAD (always gathered)
            auto pytorch_lm_head = loadPyTorchSnapshot("LM_HEAD");
            if (available_snapshots.count("LM_HEAD") && !pytorch_lm_head.empty())
            {
                TPSnapshot tp_snap = multi_device->getTPSnapshot("LM_HEAD");

                size_t combined_size = 0;
                const float *combined_ptr = tp_snap.getCombinedData(combined_size);

                if (combined_ptr && combined_size > 0)
                {
                    // Llaminar may only output the last row (last-token-only optimization).
                    // PyTorch always outputs all rows. Compare the last row from each.
                    size_t llaminar_seq_len = combined_size / vocab_size;
                    size_t pytorch_seq_len = pytorch_lm_head.size() / vocab_size;
                    size_t llaminar_last_offset = (llaminar_seq_len > 0) ? (llaminar_seq_len - 1) * vocab_size : 0;
                    size_t pytorch_last_offset = (pytorch_seq_len > 0) ? (pytorch_seq_len - 1) * vocab_size : 0;

                    // Cosine similarity on last-row logits only
                    summary.lm_head_cosine = computeCosineSimilarity(
                        combined_ptr + llaminar_last_offset,
                        pytorch_lm_head.data() + pytorch_last_offset, vocab_size);

                    if (llaminar_seq_len > 0 && pytorch_seq_len > 0)
                    {
                        summary.lm_head_kl = computeKLDivergence(
                            combined_ptr + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size);

                        summary.lm_head_top1 = computeTopKOverlap(
                            combined_ptr + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 1);

                        summary.lm_head_top5 = computeTopKOverlap(
                            combined_ptr + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 5);

                        // Check if PyTorch's top-1 token is in llaminar's top-K
                        if (config_.pytorch_top1_in_topk > 0)
                        {
                            float topk_recall = pytorchTop1InLlaminarTopK(
                                combined_ptr + llaminar_last_offset,
                                pytorch_lm_head.data() + pytorch_last_offset,
                                vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                            summary.lm_head_pytorch_top1_in_top3 = (topk_recall >= 1.0f - 1e-6f);
                        }
                        else
                        {
                            summary.lm_head_pytorch_top1_in_top3 = true; // gate disabled
                        }
                    }
                }
            }
            summary.lm_head_passed = (summary.lm_head_kl < config_.kl_threshold) &&
                                     ((summary.lm_head_top1 * 100.0f) >= config_.min_top1_accuracy) &&
                                     ((summary.lm_head_top5 * 100.0f) >= config_.min_top5_accuracy) &&
                                     (config_.pytorch_top1_in_topk <= 0 || summary.lm_head_pytorch_top1_in_top3);

            // Count early layers passed
            summary.early_layers_passed = summary.embedding_result.passed ? 1 : 0;
            for (int i = 0; i < std::min(config_.early_layers_count, static_cast<int>(summary.layer_stats.size())); ++i)
            {
                if (summary.layer_stats[i].passed)
                    summary.early_layers_passed++;
            }

            // Count total layers passed
            summary.total_layers_passed = summary.embedding_result.passed ? 1 : 0;
            for (const auto &stats : summary.layer_stats)
            {
                if (stats.passed)
                    summary.total_layers_passed++;
            }

            // Overall pass
            summary.overall_passed = (summary.early_layers_passed >= config_.min_early_layers_passed) &&
                                     summary.lm_head_passed;

            return summary;
        }

        /**
         * @brief Run TP-aware incremental decode parity test
         *
         * Tests autoregressive generation by comparing LM_HEAD outputs at each
         * decode step against PyTorch reference. For TP tests, the multi-device
         * orchestrator handles logit gathering internally.
         *
         * Requires:
         * - runner_ to be pre-configured (typically by setupLocalTPPipeline())
         * - PyTorch snapshots with decode_step{N}_LM_HEAD.npy files
         * - metadata.txt with decode_tokens line
         *
         * This is for MULTI-DEVICE LOCAL TP tests.
         * For single-device tests, use runSingleDeviceDecodeParity() instead.
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runTPDecodeParity()
        {
            DecodeParitySummary summary;

            // TP tests require pre-configured runner
            if (!activeLegacyRunner())
            {
                LOG_ERROR("[TP Decode Parity] runner_ is null - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPDecodeParity()");
                return summary;
            }

            // Verify we have a multi-device orchestrator
            auto *multi_device = dynamic_cast<RankOrchestrator *>(activeLegacyRunner());
            if (!multi_device)
            {
                LOG_ERROR("[TP Decode Parity] runner_ is not a RankOrchestrator - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPDecodeParity()");
                return summary;
            }

            // Check if decode snapshots exist
            auto decode_step0 = loadPyTorchSnapshot("decode_step0_LM_HEAD");
            if (decode_step0.empty())
            {
                LOG_WARN("Decode snapshots not found - skipping decode parity test");
                return summary;
            }

            // Read expected tokens from metadata
            std::vector<int> pytorch_decode_tokens = readDecodeTokensFromMetadata();
            if (pytorch_decode_tokens.empty())
            {
                LOG_WARN("No decode_tokens in metadata - skipping decode parity test");
                return summary;
            }

            // Run prefill first (required to initialize KV cache)
            bool success = runParityForward(
                ParityForwardPhase::Prefill,
                config_.token_ids.data(),
                static_cast<int>(config_.token_ids.size()));
            EXPECT_TRUE(success) << "Prefill failed";
            if (!success)
                return summary;
            exportActiveSnapshotsForParityDiagnostics(ParityForwardPhase::Prefill, -1);
            const uint64_t initial_moe_movement_epoch = activeMoERuntimeMovementEpoch();
            /* Production prefill owns its maintenance progress publication. */

            size_t vocab_size = model_ctx_->model().vocab_size;

            // Process each decode step
            size_t num_decode_steps = std::min(pytorch_decode_tokens.size(),
                                               static_cast<size_t>(config_.decode_steps));
            if (config_.moe_rebalance_exercise.enabled &&
                config_.moe_rebalance_exercise.min_decode_steps > 0)
            {
                EXPECT_GE(num_decode_steps,
                          static_cast<size_t>(config_.moe_rebalance_exercise.min_decode_steps))
                    << "Parity MoE rebalance exercise needs more decode snapshots/tokens";
                if (num_decode_steps <
                    static_cast<size_t>(config_.moe_rebalance_exercise.min_decode_steps))
                {
                    return summary;
                }
            }
            float sum_cosine = 0.0f;
            float sum_kl = 0.0f;

            for (size_t step = 0; step < num_decode_steps; ++step)
            {
                std::string step_prefix = "decode_step" + std::to_string(step);

                // Load PyTorch reference for this step
                auto pytorch_lm_head = loadPyTorchSnapshot(step_prefix + "_LM_HEAD");
                if (pytorch_lm_head.empty())
                {
                    break; // No more decode snapshots
                }

                summary.steps_total++;

                // Get token for this decode step
                int current_token = pytorch_decode_tokens[step];

                // Clear snapshots from previous step
                activeClearSnapshots();

                // Run single-token decode
                std::vector<int> decode_token = {current_token};
                success = runParityForward(
                    ParityForwardPhase::Decode,
                    decode_token.data(),
                    1);
                EXPECT_TRUE(success) << "Decode step " << step << " failed";
                if (!success)
                    continue;
                exportActiveSnapshotsForParityDiagnostics(
                    ParityForwardPhase::Decode,
                    static_cast<int>(step));
                if (parityMoERebalanceMaintenanceDue(step + 1) &&
                    !driveParityMoERebalanceMaintenance(
                        "decode",
                        1u,
                        static_cast<int>(step)))
                {
                    continue;
                }

                // Get Llaminar's logits (RankOrchestrator gathers from all devices)
                const float *llaminar_logits = getActiveLogits();
                if (!llaminar_logits)
                {
                    LOG_WARN("No logits for decode step " << step);
                    continue;
                }

                // Python decode snapshots may contain the full sequence
                // (shape [1, seq_len, vocab]) when generated via full-sequence
                // forward. Extract the LAST position's logits for comparison.
                const float *pytorch_logits = pytorch_lm_head.data();
                size_t pytorch_logits_count = pytorch_lm_head.size();
                if (pytorch_logits_count > vocab_size)
                {
                    size_t pytorch_seq_len = pytorch_logits_count / vocab_size;
                    size_t last_offset = (pytorch_seq_len - 1) * vocab_size;
                    pytorch_logits = pytorch_lm_head.data() + last_offset;
                    pytorch_logits_count = vocab_size;
                }

                // Compare with PyTorch
                DecodeStepStats step_stats;
                step_stats.step_idx = static_cast<int>(step);
                step_stats.has_logit_data = true;

                step_stats.cosine_similarity = computeCosineSimilarity(
                    llaminar_logits, pytorch_logits,
                    std::min(vocab_size, pytorch_logits_count));

                step_stats.kl_divergence = computeKLDivergence(
                    llaminar_logits, pytorch_logits,
                    vocab_size, vocab_size);

                step_stats.top1_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_logits,
                    vocab_size, vocab_size, 1);

                step_stats.top5_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_logits,
                    vocab_size, vocab_size, 5);

                // Find argmax tokens
                step_stats.llaminar_token = 0;
                step_stats.pytorch_token = 0;
                float max_l = llaminar_logits[0];
                float max_p = pytorch_logits[0];

                for (size_t i = 1; i < vocab_size; ++i)
                {
                    if (llaminar_logits[i] > max_l)
                    {
                        max_l = llaminar_logits[i];
                        step_stats.llaminar_token = static_cast<int>(i);
                    }
                    if (pytorch_logits[i] > max_p)
                    {
                        max_p = pytorch_logits[i];
                        step_stats.pytorch_token = static_cast<int>(i);
                    }
                }

                step_stats.token_match = (step_stats.llaminar_token == step_stats.pytorch_token);
                if (step_stats.token_match)
                    summary.top1_matches++;

                // Check if PyTorch's top-1 token appears in Llaminar's top-5
                step_stats.top5_match = (step_stats.top5_overlap >= 0.2f - 1e-6f); // At least 1/5 overlap
                if (step_stats.top5_match)
                    summary.top5_matches++;

                // Check if PyTorch's top-1 token appears in Llaminar's top-K
                if (config_.pytorch_top1_in_topk > 0)
                {
                    float topk_recall = pytorchTop1InLlaminarTopK(
                        llaminar_logits, pytorch_logits,
                        vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                    step_stats.top3_match = (topk_recall >= 1.0f - 1e-6f);
                }
                else
                {
                    step_stats.top3_match = true; // gate disabled
                }
                if (step_stats.top3_match)
                    summary.top3_matches++;

                // Pass criteria: either cosine >= threshold OR KL < threshold
                step_stats.passed = (step_stats.cosine_similarity >= config_.decode_cosine_threshold) ||
                                    (step_stats.kl_divergence < config_.kl_threshold);

                if (step_stats.passed)
                {
                    summary.steps_passed++;
                }

                sum_cosine += step_stats.cosine_similarity;
                sum_kl += step_stats.kl_divergence;

                summary.step_stats.push_back(step_stats);
            }

            // Compute averages and top1/top5 accuracy
            if (summary.steps_total > 0)
            {
                summary.avg_cosine = sum_cosine / summary.steps_total;
                summary.avg_kl = sum_kl / summary.steps_total;
                summary.top1_accuracy = 100.0f * summary.top1_matches / summary.steps_total;
                summary.top3_accuracy = 100.0f * summary.top3_matches / summary.steps_total;
                summary.top5_accuracy = 100.0f * summary.top5_matches / summary.steps_total;
            }

            // Overall pass criteria
            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            bool topk_gate = (config_.pytorch_top1_in_topk <= 0) ||
                             (summary.top3_matches == summary.steps_total);
            summary.overall_passed = (summary.steps_passed >= min_steps_required) &&
                                     (summary.top5_accuracy >= config_.min_top5_accuracy) &&
                                     (summary.avg_cosine >= config_.decode_cosine_threshold) &&
                                     topk_gate;

            activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics();
            assertParityMoERebalanceExercise(
                initial_moe_movement_epoch,
                activeMoERuntimeMovementEpoch());

            return summary;
        }

        /**
         * @brief Assert TP parity criteria and render table
         *
         * Call this after runTPPrefillParity() to apply assertions.
         */
        void assertTPParity(const TPParityTestSummary &summary)
        {
            // Render the TP-aware table (rank 0 only)
            if (isRank0())
            {
                renderTPParityTable(summary, config_, getBackendName());
            }

            // Assertions
            EXPECT_GE(summary.early_layers_passed, config_.min_early_layers_passed)
                << "At least " << config_.min_early_layers_passed << " of the first "
                << config_.early_layers_count << " layers should pass TP parity";

            EXPECT_LT(summary.lm_head_kl, config_.kl_threshold)
                << "LM_HEAD KL divergence too high: " << summary.lm_head_kl;

            EXPECT_GE(summary.lm_head_top1 * 100.0f, config_.min_top1_accuracy)
                << "LM_HEAD Top-1 accuracy too low: " << (summary.lm_head_top1 * 100.0f)
                << "% (required: " << config_.min_top1_accuracy << "%)";

            EXPECT_GE(summary.lm_head_top5 * 100.0f, config_.min_top5_accuracy)
                << "LM_HEAD Top-5 accuracy too low: " << (summary.lm_head_top5 * 100.0f)
                << "% (required: " << config_.min_top5_accuracy << "%)";

            if (config_.pytorch_top1_in_topk > 0)
            {
                EXPECT_TRUE(summary.lm_head_pytorch_top1_in_top3)
                    << "PyTorch's top-1 token must appear in llaminar's top-" << config_.pytorch_top1_in_topk << " for LM_HEAD";
            }

            exportTPPrefillCSV(summary);
        }

        /**
         * @brief Publish shard-aware LocalTP prefill evidence in the canonical CSV schemas.
         *
         * LocalTP compares both every participant shard and the reconstructed
         * semantic tensor.  The canonical parity artifacts describe that
         * reconstructed tensor, while the assertions above retain the
         * participant-level proof.  Converting here keeps one serialization
         * authority for single-device, PP, and TP campaigns and preserves the
         * full distribution/error diagnostics computed for each combined TP
         * checkpoint.
         *
         * @param summary Shard-aware numerical result produced by the live
         *                production RankOrchestrator.
         */
        void exportTPPrefillCSV(const TPParityTestSummary &summary)
        {
            if (!isRank0())
                return;

            ParityTestSummary canonical;
            canonical.embedding_cosine =
                summary.embedding_result.combined_result.cosine_similarity;
            canonical.embedding_passed =
                summary.embedding_result.combined_passed;
            canonical.lm_head_cosine = summary.lm_head_cosine;
            canonical.lm_head_kl = summary.lm_head_kl;
            canonical.lm_head_top1 = summary.lm_head_top1;
            canonical.lm_head_top5 = summary.lm_head_top5;
            canonical.lm_head_pytorch_top1_in_top3 =
                summary.lm_head_pytorch_top1_in_top3;
            canonical.lm_head_passed = summary.lm_head_passed;
            canonical.early_layers_passed = summary.early_layers_passed;
            canonical.total_layers_passed = summary.total_layers_passed;
            canonical.overall_passed = summary.overall_passed;

            canonical.layer_stats.reserve(summary.layer_stats.size());
            for (const auto &tp_layer : summary.layer_stats)
            {
                LayerStats layer;
                layer.layer_idx = tp_layer.layer_idx;
                layer.min_cosine_sim = 1.0f;
                float previous_cosine = 1.0f;
                float cosine_sum = 0.0f;

                for (const auto &tp_stage : tp_layer.stage_results)
                {
                    StageComparisonResult stage = tp_stage.combined_result;
                    stage.stage_name = tp_stage.stage_name;
                    stage.cosine_drop = std::max(
                        0.0f,
                        previous_cosine - stage.cosine_similarity);
                    previous_cosine = stage.cosine_similarity;

                    cosine_sum += stage.cosine_similarity;
                    ++layer.stages_compared;
                    if (stage.cosine_similarity < layer.min_cosine_sim)
                    {
                        layer.min_cosine_sim = stage.cosine_similarity;
                        layer.worst_stage = stage.stage_name;
                    }
                    if (stage.cosine_drop > layer.max_cosine_drop)
                    {
                        layer.max_cosine_drop = stage.cosine_drop;
                        layer.max_drop_stage = stage.stage_name;
                    }
                    if (stage.llaminar_stats.kurtosis > layer.max_kurtosis)
                    {
                        layer.max_kurtosis = stage.llaminar_stats.kurtosis;
                        layer.max_kurtosis_stage = stage.stage_name;
                    }
                    layer.stage_results.push_back(std::move(stage));
                }

                if (layer.stages_compared > 0)
                {
                    layer.avg_cosine_sim =
                        cosine_sum / static_cast<float>(layer.stages_compared);
                }
                layer.passed = tp_layer.passed;
                canonical.layer_stats.push_back(std::move(layer));
            }

            EXPECT_TRUE(ParityCSVArtifactWriter::writePrefill(
                ensureResultsDir(), getBackendName(), canonical))
                << "Cannot write one or more LocalTP prefill parity CSV artifacts";
        }

        /**
         * @brief Assert standard parity criteria
         *
         * Call this after runSingleDevicePrefillParity() to apply standard assertions.
         * MPI-aware: Only renders table on rank 0.
         */
        void assertParity(const ParityTestSummary &summary)
        {
            // Render the table first (rank 0 only)
            if (isRank0())
            {
                renderParityTable(summary, config_, getBackendName());
            }

            // Export CSV results
            exportPrefillCSV(summary);

            // For cross-rank PP, each rank only validates assertions relevant to its stage:
            // - Head rank: has early layers and embedding, validates early_layers_passed
            // - Tail rank: has LM_HEAD logits, validates logit metrics
            // Detection: if LM_HEAD summary is all-zero and we have a multi-rank MPI context,
            // this rank likely lacks the LM head. Similarly for early layers.
            const bool has_lm_head_data =
                summary.lm_head_passed ||
                summary.lm_head_cosine > 0.0f ||
                summary.lm_head_kl > 0.0f ||
                summary.lm_head_top1 > 0.0f ||
                summary.lm_head_top5 > 0.0f;
            const bool has_layer_comparisons = std::any_of(
                summary.layer_stats.begin(),
                summary.layer_stats.end(),
                [](const LayerStats &stats)
                {
                    return stats.stages_compared > 0;
                });
            const bool has_early_layer_comparisons = std::any_of(
                summary.layer_stats.begin(),
                summary.layer_stats.end(),
                [this](const LayerStats &stats)
                {
                    return stats.layer_idx < config_.early_layers_count &&
                           stats.stages_compared > 0;
                });
            const bool has_early_layer_data =
                has_early_layer_comparisons ||
                summary.embedding_passed ||
                summary.embedding_cosine > 0.0f;
            const bool multi_rank = mpi_ctx_ && mpiWorldSize() > 1;

            if (!has_layer_comparisons && !has_early_layer_data &&
                !has_lm_head_data)
            {
                ADD_FAILURE()
                    << "Prefill parity produced no comparable snapshots. "
                    << "This usually means the runner did not expose the standard "
                    << "prefill snapshot keys for this topology.";
                return;
            }

            if (!multi_rank)
            {
                if (!has_early_layer_data)
                {
                    ADD_FAILURE()
                        << "Single-rank prefill parity produced no early/layer "
                        << "snapshot comparisons";
                    return;
                }
                if (!has_lm_head_data)
                {
                    ADD_FAILURE()
                        << "Single-rank prefill parity produced no LM_HEAD "
                        << "comparison";
                    return;
                }
            }

            // Early layer assertions (skip if this rank has no early layer data, e.g. PP tail)
            if (has_early_layer_data)
            {
                EXPECT_GE(summary.early_layers_passed, config_.min_early_layers_passed)
                    << "At least " << config_.min_early_layers_passed << " of the first "
                    << config_.early_layers_count << " layers should pass parity (cosine >= "
                    << config_.cosine_threshold << ")";
            }

            // LM_HEAD assertions (skip if this rank has no LM_HEAD data, e.g. PP head)
            if (has_lm_head_data)
            {
                EXPECT_LT(summary.lm_head_kl, config_.kl_threshold)
                    << "LM_HEAD KL divergence too high: " << summary.lm_head_kl
                    << " (threshold: " << config_.kl_threshold << ")";

                EXPECT_GE(summary.lm_head_top1 * 100.0f, config_.min_top1_accuracy)
                    << "LM_HEAD Top-1 accuracy too low: " << (summary.lm_head_top1 * 100.0f)
                    << "% (required: " << config_.min_top1_accuracy << "%)";

                EXPECT_GE(summary.lm_head_top5 * 100.0f, config_.min_top5_accuracy)
                    << "LM_HEAD Top-5 accuracy too low: " << (summary.lm_head_top5 * 100.0f)
                    << "% (required: " << config_.min_top5_accuracy << "%)";

                if (config_.pytorch_top1_in_topk > 0)
                {
                    EXPECT_TRUE(summary.lm_head_pytorch_top1_in_top3)
                        << "PyTorch's top-1 token must appear in llaminar's top-" << config_.pytorch_top1_in_topk << " for LM_HEAD";
                }
            }
        }

        // =========================================================================
        // Incremental Decode Parity Testing
        // =========================================================================

        /**
         * @brief Run single-device incremental decode parity test
         *
         * Tests autoregressive generation by comparing LM_HEAD outputs at each
         * decode step against PyTorch reference. Requires:
         * - PyTorch snapshots with decode_step{N}_LM_HEAD.npy files
         * - metadata.txt with decode_tokens line
         *
         * This is for SINGLE-DEVICE tests. Calls setupPipeline() then delegates
         * to runDecodeParity().
         * For multi-device LOCAL TP tests, use runTPDecodeParity() instead.
         * For LocalPP tests, call setupPipeline() first, then runDecodeParity().
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runSingleDeviceDecodeParity()
        {
            // Setup single-device pipeline then delegate
            EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
            return runDecodeParity();
        }

        /**
         * @brief Run decode parity test using existing runner_
         *
         * Compares incremental decode logits against PyTorch reference snapshots.
         * Requires runner_ to be already set up (via setupPipeline() or
         * setupLocalPPPipeline() etc).
         *
         * Use this for LocalPP tests where you want to set up the PP pipeline
         * first, then run parity comparison without overwriting the runner.
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runDecodeParity(
            ParityDecodePrefillMode prefill_mode =
                ParityDecodePrefillMode::FreshCompute)
        {
            auto decode_total_scope = profileParityScope("decode_parity.total");
            DecodeParitySummary summary;
            production_parity_decode_prefill_state_.reset();
            production_parity_decode_boundaries_.clear();

            // Stages to compare per layer during decode (same as prefill)
            const std::vector<std::string> decode_per_layer_stages = {
                "ATTENTION_NORM",
                "QKV_PROJECTION", "GDN_Z_PROJECTION", "GDN_CONV1D_OUTPUT", "GDN_DELTA_RULE_OUTPUT", "GDN_NORM_GATE_OUTPUT",
                "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
                "FA_GATE",
                "Q_NORM", "K_NORM",
                "Q_ROPE", "K_ROPE",
                "ATTENTION_CONTEXT", "ATTENTION_CONTEXT_GATED", "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL",
                "FFN_NORM",
                "FFN_GATE", "FFN_UP", "FFN_SWIGLU", "FFN_DOWN",
                "MOE_ROUTER_OUTPUT", "MOE_ROUTING_INDICES", "MOE_ROUTING_WEIGHTS",
                "MOE_EXPERT_OUTPUT", "MOE_SHARED_EXPERT_OUTPUT", "MOE_SHARED_GATE_OUTPUT", "MOE_COMBINED_OUTPUT",
                "FFN_RESIDUAL"};

            // Check if decode snapshots exist
            auto decode_step0 = loadPyTorchSnapshot("decode_step0_LM_HEAD");
            if (decode_step0.empty())
            {
                LOG_WARN("Decode snapshots not found - skipping decode parity test");
                return summary;
            }

            // Read expected tokens from metadata
            std::vector<int> pytorch_decode_tokens = readDecodeTokensFromMetadata();
            if (pytorch_decode_tokens.empty())
            {
                LOG_WARN("No decode_tokens in metadata - skipping decode parity test");
                return summary;
            }

            // Require either production runner family to be already set up.
            if (!orch_runner_ && !activeLegacyRunner())
            {
                LOG_ERROR("[Parity] runDecodeParity() has no active runner - "
                          "ensure setupPipeline() or setupOrchestrationRunner() was called first");
                return summary;
            }

            // Run prefill first (required to initialize KV cache)
            bool success = false;
            if (prefill_mode ==
                ParityDecodePrefillMode::CompletePrefixRestore)
            {
                auto restore_policy =
                    parityGraphSnapshotPolicy(ParityForwardPhase::Prefill);
                // A complete hit publishes cached state and intentionally
                // launches no model graph, so requiring compute snapshots here
                // would reject the optimized production path by construction.
                restore_policy.require_graph_execution_on_gpu = false;
                restore_policy.require_snapshot_publication = false;
                restore_policy.require_prefill_graph_capture_on_gpu = false;
                restore_policy.retry_prefill_after_warmup_for_capture = false;
                success = runParityForwardWithPolicy(
                    ParityForwardPhase::Prefill,
                    config_.token_ids.data(),
                    static_cast<int>(config_.token_ids.size()),
                    restore_policy);
            }
            else
            {
                success = runParityForward(
                    ParityForwardPhase::Prefill,
                    config_.token_ids.data(),
                    static_cast<int>(config_.token_ids.size()));
            }
            EXPECT_TRUE(success) << "Prefill failed";
            if (!success)
                return summary;
            if (prefill_mode == ParityDecodePrefillMode::FreshCompute)
            {
                exportActiveSnapshotsForParityDiagnostics(
                    ParityForwardPhase::Prefill,
                    -1);
            }
            if (productionParityCampaignEnabled())
            {
                production_parity_decode_prefill_state_ =
                    activePrefixStateProbe();
            }
            const uint64_t initial_moe_movement_epoch = activeMoERuntimeMovementEpoch();
            /* Production prefill owns its maintenance progress publication. */

            size_t vocab_size =
                static_cast<size_t>(getActiveVocabSize());

            // Process each decode step
            size_t num_decode_steps = std::min(pytorch_decode_tokens.size(),
                                               static_cast<size_t>(config_.decode_steps));
            if (config_.moe_rebalance_exercise.enabled &&
                config_.moe_rebalance_exercise.min_decode_steps > 0)
            {
                EXPECT_GE(num_decode_steps,
                          static_cast<size_t>(config_.moe_rebalance_exercise.min_decode_steps))
                    << "Parity MoE rebalance exercise needs more decode snapshots/tokens";
                if (num_decode_steps <
                    static_cast<size_t>(config_.moe_rebalance_exercise.min_decode_steps))
                {
                    return summary;
                }
            }

            float sum_cosine = 0.0f;
            float sum_kl = 0.0f;

            for (size_t step = 0; step < num_decode_steps; ++step)
            {
                auto step_scope = profileParityScope("decode_parity.step.total");
                std::string step_prefix = "decode_step" + std::to_string(step);

                // Load PyTorch reference for this step
                std::vector<float> pytorch_lm_head;
                {
                    auto scope = profileParityScope("decode_parity.step.load_lm_head");
                    pytorch_lm_head = loadPyTorchSnapshot(step_prefix + "_LM_HEAD");
                }
                if (pytorch_lm_head.empty())
                {
                    break; // No more decode snapshots
                }

                summary.steps_total++;

                // Get token for this decode step
                int current_token = pytorch_decode_tokens[step];

                // Clear snapshots from previous step
                {
                    auto scope = profileParityScope("decode_parity.step.clear_snapshots");
                    activeClearSnapshots();
                }

                // Run single-token decode
                std::vector<int> decode_token = {current_token};
                success = runParityForward(
                    ParityForwardPhase::Decode,
                    decode_token.data(),
                    1);
                EXPECT_TRUE(success) << "Decode step " << step << " failed";
                if (!success)
                    continue;
                {
                    auto scope = profileParityScope("decode_parity.step.export_diagnostics");
                    exportActiveSnapshotsForParityDiagnostics(
                        ParityForwardPhase::Decode,
                        static_cast<int>(step));
                }
                if (productionParityCampaignEnabled())
                {
                    std::optional<int32_t> pending_condition_token;
                    if (orchestration_parity_force_mode_ ==
                        ParityForcedTokenCommitMode::Deferred)
                    {
                        if (orchestration_parity_decode_trajectory_index_ >=
                            orchestration_parity_decode_trajectory_.size())
                        {
                            ADD_FAILURE()
                                << "Deferred parity boundary has no authenticated pending condition";
                            return summary;
                        }
                        pending_condition_token =
                            orchestration_parity_decode_trajectory_[
                                orchestration_parity_decode_trajectory_index_];
                    }
                    ProductionParityDecodeBoundary boundary{
                        .reference_step = step,
                        .committed_token = current_token,
                        .pending_condition_token = pending_condition_token,
                        .expected_current_position =
                            static_cast<int>(config_.token_ids.size() +
                                             step + 1u),
                        .runtime_state = activePrefixStateProbe(),
                    };
                    EXPECT_EQ(
                        boundary.runtime_state.current_position,
                        boundary.expected_current_position)
                        << "Decode step " << step
                        << " published a model row without its matching "
                           "logical-state transition";
                    production_parity_decode_boundaries_.push_back(
                        std::move(boundary));
                }
                if (parityMoERebalanceMaintenanceDue(step + 1) &&
                    !driveParityMoERebalanceMaintenance(
                        "decode",
                        1u,
                        static_cast<int>(step)))
                {
                    continue;
                }

                // Get Llaminar's LM_HEAD output
                size_t decode_logits_size = 0;
                const float *llaminar_logits = nullptr;
                {
                    auto scope = profileParityScope("decode_parity.step.lookup_lm_head");
                    llaminar_logits = activeSnapshot("LM_HEAD", decode_logits_size);
                }
                if (!llaminar_logits)
                {
                    LOG_WARN("No LM_HEAD snapshot for decode step " << step);
                }

                // Python decode snapshots may contain the full sequence
                // (shape [1, seq_len, vocab]) when generated via full-sequence
                // forward. Extract the LAST position's logits for comparison.
                const float *pytorch_logits = pytorch_lm_head.data();
                size_t pytorch_logits_count = pytorch_lm_head.size();
                if (pytorch_logits_count > vocab_size)
                {
                    size_t pytorch_seq_len = pytorch_logits_count / vocab_size;
                    size_t last_offset = (pytorch_seq_len - 1) * vocab_size;
                    pytorch_logits = pytorch_lm_head.data() + last_offset;
                    pytorch_logits_count = vocab_size;
                }

                // Compare with PyTorch
                DecodeStepStats step_stats;
                step_stats.step_idx = static_cast<int>(step);
                step_stats.has_logit_data = llaminar_logits != nullptr;

                // ---------------------------------------------------------------
                // Per-layer cosine similarity comparison for this decode step
                // ---------------------------------------------------------------
                {
                    auto scope = profileParityScope("decode_parity.step.layer_compare");
                    int n_layers = parityLayerCount();
                    auto snapshot_keys = activeSnapshotKeys();
                    std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());
                    const auto gdn_cfg_decode = getGDNHeadConfig();
                    const auto moe_cfg_decode = getMoEConfig();

                    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
                    {
                        LayerStats stats;
                        stats.layer_idx = layer_idx;
                        float sum_cosine = 0.0f;
                        bool routed_expert_set_exact = true;

                        for (const auto &stage : decode_per_layer_stages)
                        {
                            // Skip excluded stages
                            if (!config_.excluded_stages.empty())
                            {
                                bool is_excluded = std::find(
                                                       config_.excluded_stages.begin(),
                                                       config_.excluded_stages.end(),
                                                       stage) != config_.excluded_stages.end();
                                if (is_excluded)
                                    continue;
                            }

                            // Semantic snapshot key: layer{N}_{STAGE}
                            const std::string semantic_key =
                                "layer" + std::to_string(layer_idx) + "_" + stage;

                            // PyTorch snapshot key: decode_step{N}_layer{L}_{STAGE}
                            std::string pytorch_key = step_prefix + "_layer" + std::to_string(layer_idx) + "_" + stage;
                            auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                            if (pytorch_data.empty())
                                continue;

                            const bool stage_requires_reduction =
                                !config_.allreduce_stages.empty() &&
                                std::find(config_.allreduce_stages.begin(), config_.allreduce_stages.end(), stage) !=
                                    config_.allreduce_stages.end();
                            const auto snapshot_selection = selectParitySnapshot(
                                semantic_key,
                                stage_requires_reduction,
                                config_.collective_evidence_source);
                            const std::string &llaminar_key = snapshot_selection.key;
                            const bool requires_cross_rank_sum =
                                snapshot_selection.reduction ==
                                ParitySnapshotReduction::CrossRankSum;

                            const bool has_local_snapshot = available_snapshots.count(llaminar_key) > 0;
                            if (!has_local_snapshot &&
                                snapshot_selection.requires_post_collective_key)
                            {
                                ADD_FAILURE()
                                    << "Decode parity requires live post-collective snapshot '"
                                    << llaminar_key << "' for semantic stage '"
                                    << semantic_key
                                    << "'. The pre-collective partial is not a valid fallback.";
                                continue;
                            }
                            float ranks_with_snapshot = has_local_snapshot ? 1.0f : 0.0f;
                            if (requires_cross_rank_sum)
                            {
                                if (!mpi_ctx_)
                                {
                                    ADD_FAILURE()
                                        << "Cross-rank decode parity snapshot '"
                                        << llaminar_key << "' requires an MPI context";
                                    continue;
                                }
                                float local_has_snapshot = ranks_with_snapshot;
                                parityCoordinationMPIContext()->allreduce_sum(
                                    &local_has_snapshot,
                                    &ranks_with_snapshot,
                                    1);
                            }

                            if (!has_local_snapshot)
                                continue;

                            size_t llaminar_size;
                            const float *llaminar_data = activeSnapshot(llaminar_key, llaminar_size);
                            if (!llaminar_data)
                                continue;

                            // PyTorch decode snapshots contain full-sequence output
                            // (all positions including prompt), while Llaminar only
                            // captures the last position during incremental decode.
                            // Extract last position from PyTorch data to match sizes.
                            if (pytorch_data.size() > llaminar_size && llaminar_size > 0 &&
                                pytorch_data.size() % llaminar_size == 0)
                            {
                                size_t last_offset = pytorch_data.size() - llaminar_size;
                                pytorch_data = std::vector<float>(
                                    pytorch_data.begin() + static_cast<ptrdiff_t>(last_offset),
                                    pytorch_data.end());
                            }
                            else if (pytorch_data.size() != llaminar_size)
                            {
                                continue; // Size mismatch, skip
                            }

                            // Apply GDN V-head permutation if needed
                            auto permuted_decode = applyGDNHeadPermutation(
                                llaminar_data, llaminar_size, stage, gdn_cfg_decode);
                            const float *decode_compare = permuted_decode.empty() ? llaminar_data : permuted_decode.data();

                            StageComparisonResult result;
                            // Routed/TP allreduce: reconstruct full output from rank partials.
                            bool did_allreduce_decode = false;
                            std::vector<float> allreduced_decode_buf;
                            if (requires_cross_rank_sum &&
                                static_cast<int>(ranks_with_snapshot + 0.5f) == mpiWorldSize())
                            {
                                allreduced_decode_buf.resize(llaminar_size);
                                parityCoordinationMPIContext()->allreduce_sum(
                                    decode_compare,
                                    allreduced_decode_buf.data(),
                                    llaminar_size);
                                decode_compare = allreduced_decode_buf.data();
                                did_allreduce_decode = true;
                            }
                            if (stage == "MOE_ROUTING_INDICES")
                            {
                                result = compareRoutingIndices(decode_compare, pytorch_data, llaminar_size,
                                                               moe_cfg_decode.top_k, stage);
                                /*
                                 * Decode publishes one routing row. A complete
                                 * set overlap means the raw expert sum follows
                                 * the same discrete branch even if equal-weight
                                 * experts appear in a different order.
                                 */
                                routed_expert_set_exact =
                                    result.routing_overlap >= 1.0f - 1e-6f;
                            }
                            else if (stage == "MOE_ROUTING_WEIGHTS")
                            {
                                std::string ll_idx_key = "layer" + std::to_string(layer_idx) + "_MOE_ROUTING_INDICES";
                                std::string pt_idx_key = "decode_step" + std::to_string(step) + "_layer" + std::to_string(layer_idx) + "_MOE_ROUTING_INDICES";
                                size_t ll_idx_size;
                                const float *ll_idx = activeSnapshot(ll_idx_key, ll_idx_size);
                                auto pt_idx = loadPyTorchSnapshot(pt_idx_key);
                                if (ll_idx && !pt_idx.empty())
                                    result = compareRoutingWeights(decode_compare, pytorch_data, ll_idx, pt_idx,
                                                                   llaminar_size, moe_cfg_decode.top_k,
                                                                   moe_cfg_decode.num_experts, stage);
                                else
                                    result = compareTensors(decode_compare, pytorch_data, llaminar_size, stage);
                            }
                            else
                            {
                                result = compareTensors(decode_compare, pytorch_data, llaminar_size, stage);
                            }
                            stats.stage_results.push_back(result);

                            const bool contributes_to_layer_cosine =
                                parityStageContributesToLayerCosine(
                                    stage,
                                    routed_expert_set_exact);
                            if (contributes_to_layer_cosine)
                            {
                                stats.stages_compared++;
                                sum_cosine += result.cosine_similarity;
                            }

                            if (contributes_to_layer_cosine &&
                                result.cosine_similarity < stats.min_cosine_sim)
                            {
                                stats.min_cosine_sim = result.cosine_similarity;
                                stats.worst_stage = stage;
                            }
                        }

                        // Compute per-stage cosine drops (error introduced by each stage)
                        // Skip transitions involving routing stages (different metric scale)
                        if (stats.stage_results.size() >= 2)
                        {
                            for (size_t s = 1; s < stats.stage_results.size(); ++s)
                            {
                                bool curr_routing = isRoutingStage(stats.stage_results[s].stage_name);
                                bool prev_routing = isRoutingStage(stats.stage_results[s - 1].stage_name);
                                if (curr_routing || prev_routing)
                                    continue;

                                float drop = stats.stage_results[s - 1].cosine_similarity -
                                             stats.stage_results[s].cosine_similarity;
                                stats.stage_results[s].cosine_drop = drop;

                                if (drop > stats.max_cosine_drop)
                                {
                                    stats.max_cosine_drop = drop;
                                    stats.max_drop_stage = stats.stage_results[s].stage_name;
                                }
                            }
                        }

                        if (stats.stages_compared > 0)
                        {
                            stats.avg_cosine_sim = sum_cosine / stats.stages_compared;
                        }
                        stats.passed = (stats.avg_cosine_sim >= config_.decode_cosine_threshold);

                        step_stats.layer_stats.push_back(stats);
                    }
                }

                observeComparedParityCheckpoint(
                    ParityForwardPhase::Decode,
                    static_cast<int>(step),
                    step_stats.layer_stats);

                // A non-tail PP rank still owns mathematically relevant layer
                // checkpoints even though it cannot publish LM-head logits.
                // Preserve those rows for the rank-complete CSV merge, then
                // leave aggregate logit metrics to the tail rank.
                if (!step_stats.has_logit_data)
                {
                    summary.step_stats.push_back(std::move(step_stats));
                    continue;
                }

                {
                    auto scope = profileParityScope("decode_parity.step.logit_metrics");
                    step_stats.cosine_similarity = computeCosineSimilarity(
                        llaminar_logits, pytorch_logits,
                        std::min(decode_logits_size, pytorch_logits_count));

                    step_stats.kl_divergence = computeKLDivergence(
                        llaminar_logits, pytorch_logits,
                        decode_logits_size, vocab_size);

                    step_stats.top1_overlap = computeTopKOverlap(
                        llaminar_logits, pytorch_logits,
                        decode_logits_size, vocab_size, 1);

                    step_stats.top5_overlap = computeTopKOverlap(
                        llaminar_logits, pytorch_logits,
                        decode_logits_size, vocab_size, 5);

                    // Find argmax tokens
                    step_stats.llaminar_token = 0;
                    step_stats.pytorch_token = 0;
                    float max_l = llaminar_logits[0];
                    float max_p = pytorch_logits[0];

                    for (size_t i = 1; i < vocab_size; ++i)
                    {
                        if (llaminar_logits[i] > max_l)
                        {
                            max_l = llaminar_logits[i];
                            step_stats.llaminar_token = static_cast<int>(i);
                        }
                        if (pytorch_logits[i] > max_p)
                        {
                            max_p = pytorch_logits[i];
                            step_stats.pytorch_token = static_cast<int>(i);
                        }
                    }
                }

                step_stats.token_match = (step_stats.llaminar_token == step_stats.pytorch_token);
                if (step_stats.token_match)
                {
                    summary.top1_matches++;
                }

                // Check if PyTorch's top-1 token appears in Llaminar's top-5
                step_stats.top5_match = (step_stats.top5_overlap >= 0.2f - 1e-6f);
                if (step_stats.top5_match)
                {
                    summary.top5_matches++;
                }

                // Check if PyTorch's top-1 token appears in Llaminar's top-K
                if (config_.pytorch_top1_in_topk > 0)
                {
                    float topk_recall = pytorchTop1InLlaminarTopK(
                        llaminar_logits, pytorch_logits,
                        vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                    step_stats.top3_match = (topk_recall >= 1.0f - 1e-6f);
                }
                else
                {
                    step_stats.top3_match = true; // gate disabled
                }
                if (step_stats.top3_match)
                {
                    summary.top3_matches++;
                }

                // Pass criteria: either cosine >= threshold OR KL < threshold
                step_stats.passed = (step_stats.cosine_similarity >= config_.decode_cosine_threshold) ||
                                    (step_stats.kl_divergence < config_.kl_threshold);

                if (step_stats.passed)
                {
                    summary.steps_passed++;
                }

                sum_cosine += step_stats.cosine_similarity;
                sum_kl += step_stats.kl_divergence;

                summary.step_stats.push_back(step_stats);
            }

            // Compute aggregate statistics
            if (summary.steps_total > 0)
            {
                summary.avg_cosine = sum_cosine / summary.steps_total;
                summary.avg_kl = sum_kl / summary.steps_total;
                summary.top1_accuracy = 100.0f * summary.top1_matches / summary.steps_total;
                summary.top3_accuracy = 100.0f * summary.top3_matches / summary.steps_total;
                summary.top5_accuracy = 100.0f * summary.top5_matches / summary.steps_total;
            }

            // Overall pass criteria
            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            bool topk_gate = (config_.pytorch_top1_in_topk <= 0) ||
                             (summary.top3_matches == summary.steps_total);
            summary.overall_passed = (summary.steps_passed >= min_steps_required) &&
                                     (summary.top5_accuracy >= config_.min_top5_accuracy) &&
                                     (summary.avg_cosine >= config_.decode_cosine_threshold) &&
                                     topk_gate;

            activeDrainCompletedDecodeBoundaryMaintenanceDiagnostics();
            assertParityMoERebalanceExercise(
                initial_moe_movement_epoch,
                activeMoERuntimeMovementEpoch());

            return summary;
        }

        /**
         * @brief Render decode parity results as Unicode table using libfort
         */
        void renderDecodeParityTable(const DecodeParitySummary &summary, const std::string &backend_name)
        {
            // Helper: format float to string with precision
            auto fmt_f6 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(6) << v;
                return ss.str();
            };

            auto fmt_f4 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(4) << v;
                return ss.str();
            };

            auto fmt_f1 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << v;
                return ss.str();
            };

            auto fmt_f3 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(3) << v;
                return ss.str();
            };

            // Helper: status icon
            auto status_str = [](bool passed) -> std::string
            {
                return passed ? "✓" : "✗";
            };

            std::cout << "\n";

            // =========================================================================
            // Title table
            // =========================================================================
            {
                fort::utf8_table title_table;
                title_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream title_ss;
                title_ss << backend_name << " INCREMENTAL DECODE PARITY";

                std::ostringstream subtitle_ss;
                subtitle_ss << "Threshold: cosine >= " << fmt_f3(config_.decode_cosine_threshold)
                            << " OR KL < " << fmt_f3(config_.kl_threshold);

                title_table << title_ss.str() << fort::endr;
                title_table << subtitle_ss.str() << fort::endr;

                title_table[0][0].set_cell_text_align(fort::text_align::center);
                title_table[1][0].set_cell_text_align(fort::text_align::center);
                title_table.row(0).set_cell_row_type(fort::row_type::header);

                std::cout << title_table.to_string();
            }

            // =========================================================================
            // Main decode parity table
            // =========================================================================
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Header row
            table << fort::header
                  << "Step" << "Cosine" << "KL" << "Llaminar" << "PyTorch" << ("InTop" + std::to_string(config_.pytorch_top1_in_topk > 0 ? config_.pytorch_top1_in_topk : 3)) << "OK"
                  << fort::endr;

            // Set column alignments
            table.column(0).set_cell_text_align(fort::text_align::right);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::right);
            table.column(4).set_cell_text_align(fort::text_align::right);
            table.column(5).set_cell_text_align(fort::text_align::center);
            table.column(6).set_cell_text_align(fort::text_align::center);

            // Per-step rows
            for (const auto &step : summary.step_stats)
            {
                std::string match_marker = step.token_match ? " ✓" : " ✗";

                std::ostringstream llaminar_ss;
                llaminar_ss << step.llaminar_token << match_marker;

                table << step.step_idx
                      << fmt_f6(step.cosine_similarity)
                      << fmt_f6(step.kl_divergence)
                      << llaminar_ss.str()
                      << step.pytorch_token
                      << status_str(step.top3_match)
                      << status_str(step.passed)
                      << fort::endr;
            }

            std::cout << table.to_string();

            // =========================================================================
            // Summary footer table
            // =========================================================================
            {
                fort::utf8_table summary_table;
                summary_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream summary_ss;
                int k = config_.pytorch_top1_in_topk > 0 ? config_.pytorch_top1_in_topk : 3;
                summary_ss << "SUMMARY:  Steps=" << summary.steps_passed << "/" << summary.steps_total
                           << "  AvgCosine=" << fmt_f4(summary.avg_cosine)
                           << "  Top1=" << fmt_f1(summary.top1_accuracy) << "%"
                           << "  RefInTop" << k << "=" << summary.top3_matches << "/" << summary.steps_total
                           << "  Top5=" << fmt_f1(summary.top5_accuracy) << "%"
                           << "  " << (summary.overall_passed ? "✓ PASSED" : "✗ FAILED");

                summary_table << summary_ss.str() << fort::endr;
                summary_table[0][0].set_cell_text_align(fort::text_align::center);

                std::cout << summary_table.to_string();
            }

            std::cout << std::endl;
        }

        /**
         * @brief Render per-layer cosine similarity for a specific decode step
         *
         * Shows the layer-by-layer breakdown to identify where divergence starts.
         * Typically rendered for step 0 (first decode step) since that's most diagnostic.
         */
        void renderDecodeLayerParityTable(
            const DecodeStepStats &step,
            const std::string &backend_name)
        {
            if (step.layer_stats.empty())
                return;

            auto fmt_f6 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(6) << v;
                return ss.str();
            };

            auto status_str = [](bool passed) -> std::string
            {
                return passed ? "✓" : "✗";
            };

            std::cout << "\n";

            // Title
            {
                fort::utf8_table title_table;
                title_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream title_ss;
                title_ss << backend_name << " DECODE STEP " << step.step_idx
                         << " LAYER-BY-LAYER PARITY";

                title_table << title_ss.str() << fort::endr;
                title_table[0][0].set_cell_text_align(fort::text_align::center);
                title_table.row(0).set_cell_row_type(fort::row_type::header);

                std::cout << title_table.to_string();
            }

            // Layer table
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            table << fort::header
                  << "Layer" << "Avg Cosine" << "Min Cosine" << "Worst Stage"
                  << "Max Drop" << "Drop Stage" << "Stages"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::center);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::left);
            table.column(4).set_cell_text_align(fort::text_align::right);
            table.column(5).set_cell_text_align(fort::text_align::left);
            table.column(6).set_cell_text_align(fort::text_align::right);

            for (const auto &stats : step.layer_stats)
            {
                if (stats.stages_compared == 0)
                    continue;

                std::ostringstream layer_ss;
                layer_ss << "Layer " << stats.layer_idx;

                std::string drop_str = (stats.max_cosine_drop > 0.001f)
                                           ? fmt_f6(stats.max_cosine_drop)
                                           : "-";
                std::string drop_stage = stats.max_drop_stage.empty() ? "-" : stats.max_drop_stage;

                table << layer_ss.str()
                      << fmt_f6(stats.avg_cosine_sim)
                      << fmt_f6(stats.min_cosine_sim)
                      << stats.worst_stage
                      << drop_str
                      << drop_stage
                      << stats.stages_compared
                      << fort::endr;
            }

            std::cout << table.to_string();
        }

        /**
         * @brief Complete the canonical decode proof and assert its criteria.
         *
         * MTP is a typed campaign dimension, so an enabled cell must run its
         * production predictor/verifier transaction before any decode result
         * can be accepted. Keeping that transition here makes omission
         * impossible for model-specific fixtures: callers provide the mutable
         * teacher-forced summary, this method appends the recursive MTP
         * checkpoints, and only then exports and judges the complete artifact.
         * Disabled cells pay no additional work.
         *
         * Call this after runSingleDeviceDecodeParity() or runTPDecodeParity()
         * to apply standard assertions. MPI-aware: Only renders tables on the
         * configured artifact-authority rank.
         */
        void assertDecodeParity(DecodeParitySummary &summary)
        {
            auto assert_scope = profileParityScope("assert_decode.total");
            // Skip if no decode steps were tested
            if (summary.steps_total == 0)
            {
                GTEST_SKIP() << "No decode snapshots found - skipping decode parity assertions";
            }

            runProductionParityMTPProof(summary);

            // For cross-rank PP, only the tail rank has valid logits for decode comparison.
            // Detect by checking if any decode step actually produced metrics (not whether they're good).
            const bool has_logit_data = std::any_of(
                summary.step_stats.begin(),
                summary.step_stats.end(),
                [](const DecodeStepStats &step)
                {
                    return step.has_logit_data;
                });

            // Render table first (rank 0 only)
            if (isRank0())
            {
                auto render_scope = profileParityScope("assert_decode.render_tables");
                if (has_logit_data)
                    renderDecodeParityTable(summary, getBackendName());

                // Render layer-by-layer breakdown for the first decode step
                // (most diagnostic for identifying where divergence originates)
                if (!summary.step_stats.empty() &&
                    !summary.step_stats[0].layer_stats.empty())
                {
                    renderDecodeLayerParityTable(summary.step_stats[0], getBackendName());
                }
            }

            // Export CSV results
            {
                auto export_scope = profileParityScope("assert_decode.export_csv");
                exportDecodeCSV(summary);
            }

            // Decode also captures per-layer snapshots when the runner exposes
            // them. Enforce the same early-layer gate used by prefill whenever a
            // full early-layer window is present on this rank; PP tail ranks may
            // only own later layers and should still rely on logit assertions.
            for (const auto &step_stats : summary.step_stats)
            {
                int compared_early_layers = 0;
                int passed_early_layers = 0;
                for (const auto &layer_stats : step_stats.layer_stats)
                {
                    if (layer_stats.layer_idx >= config_.early_layers_count)
                        continue;
                    if (layer_stats.stages_compared == 0)
                        continue;

                    compared_early_layers++;
                    if (layer_stats.passed)
                        passed_early_layers++;
                }

                if (compared_early_layers >= config_.early_layers_count)
                {
                    EXPECT_GE(passed_early_layers, config_.min_early_layers_passed)
                        << "At least " << config_.min_early_layers_passed << " of the first "
                        << config_.early_layers_count << " layers should pass decode parity at step "
                        << step_stats.step_idx << " (cosine >= " << config_.decode_cosine_threshold << ")";
                }
            }

            // Logit assertions belong only to the pipeline tail. Non-tail
            // ranks have already enforced their owned layer checkpoints above.
            if (!has_logit_data)
                return;

            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            EXPECT_GE(summary.steps_passed, min_steps_required)
                << "Not enough decode steps passed: " << summary.steps_passed << "/" << summary.steps_total
                << " (required: " << min_steps_required << ")";

            EXPECT_GE(summary.top5_accuracy, config_.min_top5_accuracy)
                << "Top-5 accuracy too low: " << summary.top5_accuracy << "%"
                << " (required: " << config_.min_top5_accuracy << "%)";

            EXPECT_GE(summary.avg_cosine, config_.decode_cosine_threshold)
                << "Average cosine too low: " << summary.avg_cosine
                << " (required: " << config_.decode_cosine_threshold << ")";

            if (config_.pytorch_top1_in_topk > 0)
            {
                EXPECT_EQ(summary.top3_matches, summary.steps_total)
                    << "PyTorch's top-1 token must appear in llaminar's top-" << config_.pytorch_top1_in_topk << " for every decode step. "
                    << "Failed: " << summary.top3_matches << "/" << summary.steps_total;
            }
        }

        // =========================================================================
        // CSV Results Export
        // =========================================================================

        /**
         * @brief Get abbreviated git hash of current HEAD
         */
        static std::string getGitHash()
        {
            std::string hash = "unknown";
            FILE *pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
            if (pipe)
            {
                char buf[64];
                if (fgets(buf, sizeof(buf), pipe))
                {
                    hash = buf;
                    // Trim trailing newline
                    while (!hash.empty() && (hash.back() == '\n' || hash.back() == '\r'))
                        hash.pop_back();
                }
                pclose(pipe);
            }
            return hash;
        }

        /**
         * @brief Get current GTest test name as "TestSuite/TestName"
         */
        static std::string getTestName()
        {
            const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
            if (!info)
                return "unknown";
            std::string suite = info->test_suite_name() ? info->test_suite_name() : "";
            std::string name = info->name() ? info->name() : "";
            // For parameterized tests, name includes the parameter
            return suite + "/" + name;
        }

        /**
         * @brief Derive the parity results directory from __FILE__ path
         *
         * ParityTestBase.h lives at tests/v2/integration/parity/ParityTestBase.h
         * Results go to tests/v2/integration/parity/results/<git_hash>/<test_name>/
         */
        static std::filesystem::path getResultsDir()
        {
            return ParityCSVArtifactWriter::resultsDir();
        }

        /**
         * @brief Ensure results directory exists
         */
        static std::filesystem::path ensureResultsDir()
        {
            auto dir = getResultsDir();
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
            {
                LOG_WARN("Failed to create results directory: " << dir << " (" << ec.message() << ")");
            }
            return dir;
        }

        /**
         * @brief Export prefill parity results to CSV
         *
         * Writes two CSV files:
         *   - prefill_layers.csv: Per-layer metrics
         *   - prefill_summary.csv: LM_HEAD and overall metrics
         */
        void exportPrefillCSV(const ParityTestSummary &summary)
        {
            if (!config_.uses_cross_rank_pipeline)
            {
                if (!isRank0())
                    return;
                EXPECT_TRUE(ParityCSVArtifactWriter::writePrefill(
                    ensureResultsDir(), getBackendName(), summary))
                    << "Cannot write one or more prefill parity CSV artifacts";
                return;
            }

            ASSERT_NE(mpi_ctx_, nullptr);
            ASSERT_GT(mpiWorldSize(), 1);

            struct RankPrefillSummary
            {
                float embedding_cosine = 0.0f;
                int32_t embedding_passed = 0;
                int32_t has_embedding_data = 0;
                float lm_head_cosine = 0.0f;
                float lm_head_kl = 0.0f;
                float lm_head_top1 = 0.0f;
                float lm_head_top5 = 0.0f;
                int32_t lm_head_top1_in_topk = 0;
                int32_t lm_head_passed = 0;
                int32_t has_lm_head_data = 0;
                int32_t early_layers_passed = 0;
                int32_t total_layers_passed = 0;
            };

            RankPrefillSummary local{};
            local.embedding_cosine = summary.embedding_cosine;
            local.embedding_passed = summary.embedding_passed ? 1 : 0;
            local.has_embedding_data =
                (summary.embedding_passed || summary.embedding_cosine != 0.0f)
                    ? 1
                    : 0;
            local.lm_head_cosine = summary.lm_head_cosine;
            local.lm_head_kl = summary.lm_head_kl;
            local.lm_head_top1 = summary.lm_head_top1;
            local.lm_head_top5 = summary.lm_head_top5;
            local.lm_head_top1_in_topk =
                summary.lm_head_pytorch_top1_in_top3 ? 1 : 0;
            local.lm_head_passed = summary.lm_head_passed ? 1 : 0;
            local.has_lm_head_data =
                (summary.lm_head_passed || summary.lm_head_cosine != 0.0f ||
                 summary.lm_head_kl != 0.0f || summary.lm_head_top1 != 0.0f ||
                 summary.lm_head_top5 != 0.0f)
                    ? 1
                    : 0;
            local.early_layers_passed = summary.early_layers_passed;
            local.total_layers_passed = summary.total_layers_passed;

            std::vector<RankPrefillSummary> rank_summaries(
                static_cast<size_t>(mpiWorldSize()));
            parityCoordinationMPIContext()->allgather_bytes(
                &local,
                rank_summaries.data(),
                sizeof(RankPrefillSummary));

            const auto dir = ensureResultsDir();
            const auto fragment_dir =
                dir / (".pipeline_prefill_rank_" + std::to_string(mpiRank()));
            const int32_t local_write_ok =
                ParityCSVArtifactWriter::writePrefill(
                    fragment_dir, getBackendName(), summary)
                    ? 1
                    : 0;
            std::vector<int32_t> write_status(
                static_cast<size_t>(mpiWorldSize()));
            parityCoordinationMPIContext()->allgather_bytes(
                &local_write_ok,
                write_status.data(),
                sizeof(local_write_ok));
            mpiBarrier();

            int32_t merge_ok = 1;
            if (isRank0())
            {
                ParityTestSummary aggregate;
                const auto &head = rank_summaries.front();
                const auto &tail = rank_summaries.back();
                EXPECT_EQ(head.has_embedding_data, 1)
                    << "Pipeline head published no embedding comparison";
                EXPECT_EQ(tail.has_lm_head_data, 1)
                    << "Pipeline tail published no LM-head comparison";

                aggregate.embedding_cosine = head.embedding_cosine;
                aggregate.embedding_passed = head.embedding_passed != 0;
                aggregate.lm_head_cosine = tail.lm_head_cosine;
                aggregate.lm_head_kl = tail.lm_head_kl;
                aggregate.lm_head_top1 = tail.lm_head_top1;
                aggregate.lm_head_top5 = tail.lm_head_top5;
                aggregate.lm_head_pytorch_top1_in_top3 =
                    tail.lm_head_top1_in_topk != 0;
                aggregate.lm_head_passed = tail.lm_head_passed != 0;
                for (const auto &rank : rank_summaries)
                {
                    aggregate.early_layers_passed +=
                        rank.early_layers_passed;
                    aggregate.total_layers_passed +=
                        rank.total_layers_passed;
                }
                aggregate.overall_passed =
                    aggregate.early_layers_passed >=
                        config_.min_early_layers_passed &&
                    aggregate.lm_head_passed;

                std::vector<std::filesystem::path> rank_dirs;
                rank_dirs.reserve(static_cast<size_t>(mpiWorldSize()));
                for (int rank = 0; rank < mpiWorldSize(); ++rank)
                {
                    rank_dirs.push_back(
                        dir / (".pipeline_prefill_rank_" +
                               std::to_string(rank)));
                }
                const bool all_fragments_written = std::all_of(
                    write_status.begin(),
                    write_status.end(),
                    [](int32_t status)
                    {
                        return status != 0;
                    });
                merge_ok =
                    all_fragments_written &&
                            ParityCSVArtifactWriter::mergePipelinePrefill(
                                dir,
                                getBackendName(),
                                aggregate,
                                rank_dirs)
                        ? 1
                        : 0;
            }
            parityCoordinationMPIContext()->broadcast_int32(
                &merge_ok, 1, 0);
            mpiBarrier();

            std::error_code cleanup_error;
            std::filesystem::remove_all(fragment_dir, cleanup_error);
            EXPECT_FALSE(cleanup_error)
                << "Cannot remove generated PP prefill fragment directory: "
                << cleanup_error.message();
            EXPECT_EQ(merge_ok, 1)
                << "Cannot merge rank-complete prefill parity CSV artifacts";
        }

        /**
         * @brief Export decode parity results to CSV
         *
         * Writes two CSV files:
         *   - decode_steps.csv: Per-step LM_HEAD metrics
         *   - decode_layers.csv: Per-step per-layer metrics (if layer stats captured)
         */
        void exportDecodeCSV(const DecodeParitySummary &summary)
        {
            if (!config_.uses_cross_rank_pipeline)
            {
                if (!isRank0())
                    return;
                EXPECT_TRUE(ParityCSVArtifactWriter::writeDecode(
                    ensureResultsDir(), getBackendName(), summary))
                    << "Cannot write one or more decode parity CSV artifacts";
                return;
            }

            ASSERT_NE(mpi_ctx_, nullptr);
            ASSERT_GT(mpiWorldSize(), 1);
            const auto dir = ensureResultsDir();
            const auto fragment_dir =
                dir / (".pipeline_decode_rank_" + std::to_string(mpiRank()));
            const int32_t local_write_ok =
                ParityCSVArtifactWriter::writeDecode(
                    fragment_dir, getBackendName(), summary)
                    ? 1
                    : 0;
            std::vector<int32_t> write_status(
                static_cast<size_t>(mpiWorldSize()));
            parityCoordinationMPIContext()->allgather_bytes(
                &local_write_ok,
                write_status.data(),
                sizeof(local_write_ok));
            mpiBarrier();

            int32_t merge_ok = 1;
            if (isRank0())
            {
                std::vector<std::filesystem::path> rank_dirs;
                rank_dirs.reserve(static_cast<size_t>(mpiWorldSize()));
                for (int rank = 0; rank < mpiWorldSize(); ++rank)
                {
                    rank_dirs.push_back(
                        dir / (".pipeline_decode_rank_" +
                               std::to_string(rank)));
                }
                const bool all_fragments_written = std::all_of(
                    write_status.begin(),
                    write_status.end(),
                    [](int32_t status)
                    {
                        return status != 0;
                    });
                merge_ok =
                    all_fragments_written &&
                            ParityCSVArtifactWriter::mergePipelineDecode(
                                dir, rank_dirs)
                        ? 1
                        : 0;
            }
            parityCoordinationMPIContext()->broadcast_int32(
                &merge_ok, 1, 0);
            mpiBarrier();

            std::error_code cleanup_error;
            std::filesystem::remove_all(fragment_dir, cleanup_error);
            EXPECT_FALSE(cleanup_error)
                << "Cannot remove generated PP decode fragment directory: "
                << cleanup_error.message();
            EXPECT_EQ(merge_ok, 1)
                << "Cannot merge rank-complete decode parity CSV artifacts";
        }

    };

} // namespace llaminar2::test::parity
