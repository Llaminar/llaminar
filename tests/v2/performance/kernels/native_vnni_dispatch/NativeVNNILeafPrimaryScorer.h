/**
 * @file NativeVNNILeafPrimaryScorer.h
 * @brief Stable C ABI for deterministic GPU policy-tree fitting.
 *
 * ABI v12 exposes a complete device-resident bounded beam search plus a focused
 * leaf-scoring diagnostic. NativeVNNI policy fitting repeatedly asks which
 * candidates remain valid on a point subset, which candidate minimizes the
 * exact leaf objective, and which legal feature split produces the best next
 * complete tree. CUDA and ROCm now perform that entire transaction without
 * returning every intermediate subset to Python.
 *
 * The first two keys are:
 *
 * 1. whether the candidate's canonical measured nearest-rank p95 is at or
 *    above the strict 5% budget; and
 * 2. nearest-rank p95 bounded fitting regret over the subset.
 *
 * Both p95 values are observed FP64 order statistics: no interpolation or
 * reduction arithmetic is permitted. They can therefore be evaluated on CUDA
 * or ROCm without changing the deterministic learner. The diagnostic API
 * returns every candidate tied on those keys so integration tests can prove
 * that primitive independently. The complete tree-search API additionally
 * evaluates fixed-order means, measured diagnostics, exact canonical
 * signatures, structural deduplication, and the bounded beam entirely on the
 * selected device.
 *
 * The host publishes only compact sufficient inputs: regret matrices, raw N/K
 * geometry, point-group ranks, and feature-axis policy. CUDA/ROCm derives exact
 * feature values, sorted value masks, prefix masks, rational split thresholds,
 * and heldout routes inside the captured graph. The obsolete v8 ABI required a
 * dense `[axis, lower, upper]` host threshold table and a dense
 * `[threshold, heldout]` device match matrix; a production CPU-decode fold could
 * thereby spend seconds constructing and transferring hundreds of megabytes of
 * metadata before useful search began.
 *
 * An opaque session keeps the current regret matrix, explicit stream, reusable
 * high-water scratch buffers, and captured search graphs device-resident across
 * fits. Each new fold publishes its compact inputs through `Prepare` and the
 * tree transaction; it does not destroy the backend context or surrender
 * persistent storage. Each policy worker is a separate process bound to one
 * physical accelerator, so a backend-specific shared library owns all runtime
 * interaction and CUDA/HIP symbols never coexist in one process.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define LLAMINAR_NATIVE_VNNI_SCORER_EXPORT __declspec(dllexport)
#else
#define LLAMINAR_NATIVE_VNNI_SCORER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

/** Opaque backend-owned context for a sequence of uploaded regret matrices. */
struct LlaminarNativeVNNILeafPrimaryScorerSession;

/** ABI version required by the Python binding in this source tree. */
constexpr std::uint32_t kLlaminarNativeVNNILeafPrimaryScorerAbiVersion = 12;
constexpr std::uint32_t kLlaminarNativeVNNITreeMaximumPoints = 512;
constexpr std::uint32_t kLlaminarNativeVNNITreePointMaskWords =
    kLlaminarNativeVNNITreeMaximumPoints / 64;
constexpr std::uint32_t kLlaminarNativeVNNITreeMaximumLeaves = 32;
constexpr std::uint32_t kLlaminarNativeVNNITreeMaximumStructureTokens = 63;
constexpr std::uint32_t kLlaminarNativeVNNITreeMaximumHeldoutPoints = 512;

/** Fixed-width little-word-first point or shape-group membership mask. */
struct LlaminarNativeVNNITreePointMask {
    std::uint64_t words[kLlaminarNativeVNNITreePointMaskWords];
};

/**
 * Runtime feature operation encoded by one compact axis or split descriptor.
 *
 * The operation and tile width are intentionally separate. This keeps the ABI
 * compact while preserving every exact integer feature used by the Python
 * policy schema. Numerators and denominators are never converted to floating
 * point on the device, so unseen geometries follow byte-for-byte equivalent
 * routing predicates during grouped cross validation.
 */
enum LlaminarNativeVNNITreeThresholdOperation : std::uint32_t {
    kLlaminarNativeVNNITreeThresholdAggregateN = 0,
    kLlaminarNativeVNNITreeThresholdK = 1,
    kLlaminarNativeVNNITreeThresholdWorkItems = 2,
    kLlaminarNativeVNNITreeThresholdAspectRatio = 3,
    kLlaminarNativeVNNITreeThresholdNTiles = 4,
    kLlaminarNativeVNNITreeThresholdKGroupsPerNTile = 5,
    kLlaminarNativeVNNITreeThresholdNFinalTile = 6,
    kLlaminarNativeVNNITreeThresholdNTileUtilization = 7,
    kLlaminarNativeVNNITreeThresholdNParallelWaves = 8,
    kLlaminarNativeVNNITreeThresholdNFinalParallelWaveUtilization = 9,
    kLlaminarNativeVNNITreeThresholdMNParallelWaves = 10,
    kLlaminarNativeVNNITreeThresholdMNFinalParallelWaveUtilization = 11,
    kLlaminarNativeVNNITreeThresholdNTileAligned = 12,
    kLlaminarNativeVNNITreeThresholdKFinalTile = 13,
};

/** Exact integer description of one policy-tree split threshold. */
struct LlaminarNativeVNNITreeThreshold {
    std::uint32_t axis_priority;
    std::uint32_t operation;
    std::uint32_t tile_width;
    std::uint64_t numerator;
    std::uint64_t denominator;
    std::uint32_t parallelism_width;
    std::uint32_t task_multiplier;
};

/** One runtime feature axis from which the device may construct splits. */
struct LlaminarNativeVNNITreeFeatureAxis {
    std::uint32_t axis_priority;
    std::uint32_t operation;
    std::uint32_t tile_width;
};

/** Placement rule for an unseen value between two observed feature values. */
enum LlaminarNativeVNNITreeBoundaryPlacement : std::uint32_t {
    kLlaminarNativeVNNITreeBoundaryMidpoint = 0,
    kLlaminarNativeVNNITreeBoundaryLowerEdge = 1,
};

/**
 * One complete fitted tree returned for a maximum leaf budget.
 *
 * Structure tokens are preorder markers: zero denotes a leaf and one denotes
 * a split followed by its left and right subtrees. For every split marker at
 * position `i`, `split_thresholds[i]` carries the exact rational predicate.
 * Leaf masks and candidate indices remain in matching preorder. Keeping only
 * selected-tree thresholds makes the result size proportional to the 32-leaf
 * publication bound rather than the quadratic training-point pair space.
 */
struct LlaminarNativeVNNITreeFitResult {
    std::uint32_t leaf_count;
    LlaminarNativeVNNITreePointMask
        leaf_masks[kLlaminarNativeVNNITreeMaximumLeaves];
    std::uint32_t candidate_indices[kLlaminarNativeVNNITreeMaximumLeaves];
    std::uint32_t structure_token_count;
    std::uint32_t structure_tokens[
        kLlaminarNativeVNNITreeMaximumStructureTokens];
    LlaminarNativeVNNITreeThreshold split_thresholds[
        kLlaminarNativeVNNITreeMaximumStructureTokens];
};

/**
 * Compact heldout decisions for one maximum-leaf budget.
 *
 * Candidate indices use the same canonical inventory as the training matrix.
 * `UINT32_MAX` denotes an uncovered point. Only the first
 * `required_point_count` entries are meaningful.
 */
struct LlaminarNativeVNNITreeFoldEvaluation {
    std::uint32_t required_point_count;
    std::uint32_t covered_point_count;
    std::uint32_t selected_candidate_indices[
        kLlaminarNativeVNNITreeMaximumHeldoutPoints];
};

/** Monotonic diagnostics for persistent storage and graph ownership. */
struct LlaminarNativeVNNITreeRuntimeStats {
    std::uint64_t prepare_count;
    std::uint64_t matrix_growth_count;
    std::uint64_t tree_scratch_growth_count;
    std::uint64_t graph_capture_count;
    std::uint64_t graph_replay_count;
    std::uint64_t tree_search_count;
    std::uint64_t tree_search_retry_count;
    std::uint64_t tree_search_stream_sync_count;
    std::uint64_t tree_search_intermediate_sync_count;
    std::uint64_t fused_evaluation_count;
    std::uint64_t final_result_d2h_bytes;
    std::uint64_t tree_scratch_high_water_bytes;
    std::uint64_t device_allocation_count;
    std::uint64_t device_free_count;
    std::uint64_t h2d_copy_count;
    std::uint64_t h2d_bytes;
    std::uint64_t d2h_copy_count;
    std::uint64_t d2h_bytes;
    std::uint64_t stream_sync_count;
    std::uint64_t device_sync_count;
    std::uint64_t captured_graph_transfer_count;
    std::uint32_t last_expansion_capacity;
    std::uint32_t reserved;
};

/**
 * @brief Return the scorer ABI version implemented by this shared library.
 * @return `kLlaminarNativeVNNILeafPrimaryScorerAbiVersion`.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
std::uint32_t llaminarNativeVNNILeafPrimaryScorerAbiVersion();

/**
 * @brief Return the backend name compiled into this shared library.
 * @return A process-lifetime string equal to either `"cuda"` or `"rocm"`.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
const char* llaminarNativeVNNILeafPrimaryScorerBackend();

/**
 * @brief Return the number of devices visible to this backend runtime.
 * @param error Destination for a human-readable runtime error.
 * @param error_capacity Number of writable bytes in @p error.
 * @return A non-negative device count, or `-1` on failure.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
int llaminarNativeVNNILeafPrimaryScorerDeviceCount(
    char* error,
    std::size_t error_capacity);

/**
 * @brief Create a device-resident scorer session and publish its first matrix.
 *
 * Both regret inputs are row-major `[point_count, candidate_count]` matrices.
 * `fitting_regrets` contains measured-only or bounded profiler-informed costs.
 * `measured_p95_regrets` contains canonical measured surface p95 values and
 * exclusively owns the installation pass/fail key. A finite value means the
 * candidate is valid at that point; positive infinity marks a missing or
 * ineligible candidate. The function copies both matrices to the selected
 * device and retains grow-only storage until session destruction. Later folds
 * replace the logical matrix through `Prepare` without replacing the session.
 *
 * @param device_ordinal Backend-local physical device ordinal.
 * @param fitting_regrets Host pointer to bounded fitting-regret doubles.
 * @param measured_p95_regrets Host pointer to measured surface-p95 doubles.
 * @param point_count Number of rows in @p regrets.
 * @param candidate_count Number of columns in @p regrets.
 * @param error Destination for a human-readable validation/runtime error.
 * @param error_capacity Number of writable bytes in @p error.
 * @return A new opaque session on success, or `nullptr` on failure.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
LlaminarNativeVNNILeafPrimaryScorerSession*
llaminarNativeVNNILeafPrimaryScorerCreate(
    int device_ordinal,
    const double* fitting_regrets,
    const double* measured_p95_regrets,
    std::uint32_t point_count,
    std::uint32_t candidate_count,
    char* error,
    std::size_t error_capacity);

/**
 * @brief Publish a new regret matrix into an existing backend session.
 *
 * Matrix storage follows a grow-only high-water policy. A matrix no larger than
 * the current capacity is copied into the existing device buffers, preserving
 * all tree scratch allocations and captured graphs whose pointer geometry is
 * still valid. Growing the matrix invalidates captured graphs because their
 * kernel nodes contain the previous matrix addresses.
 *
 * Uploads are ordered on the session stream and complete before any subsequent
 * score or tree-search work on that stream. The caller must keep the host inputs
 * immutable until that following transaction returns.
 *
 * @return Zero on success and nonzero on validation or backend failure.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
int llaminarNativeVNNILeafPrimaryScorerPrepare(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    const double* fitting_regrets,
    const double* measured_p95_regrets,
    std::uint32_t point_count,
    std::uint32_t candidate_count,
    char* error,
    std::size_t error_capacity);

/**
 * @brief Score a batch of point subsets using a prepared device session.
 *
 * `subset_masks` is a row-major bit matrix with
 * `subset_word_count == ceil(point_count / 64)` for the matrix supplied at
 * session creation. Every subset must select at least one point.
 *
 * For each subset, the function writes the best measured p95 gate bit, its
 * bounded fitting p95, and a candidate bitmask containing every exact tie on
 * those two keys.
 * The survivor mask has `ceil(candidate_count / 64)` words per subset.  The
 * caller remains responsible for canonical percentile/mean/name tie-breaking.
 *
 * @param session Prepared device session returned by the create function.
 * @param subset_masks Host pointer to packed subset membership words.
 * @param subset_count Number of independently scored point subsets.
 * @param subset_word_count Packed words per point subset.
 * @param best_fitting_p95 Host output with `subset_count` doubles.
 * @param best_failed_leaf_count Host output with `subset_count` 32-bit values.
 * @param survivor_masks Host output with
 *        `subset_count * ceil(candidate_count / 64)` words.
 * @param error Destination for a human-readable validation/runtime error.
 * @param error_capacity Number of writable bytes in @p error.
 * @return Zero on success and nonzero on validation or backend failure.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
int llaminarNativeVNNILeafPrimaryScorerScore(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    const std::uint64_t* subset_masks,
    std::uint32_t subset_count,
    std::uint32_t subset_word_count,
    double* best_fitting_p95,
    std::uint32_t* best_failed_leaf_count,
    std::uint64_t* survivor_masks,
    char* error,
    std::size_t error_capacity);

/**
 * @brief Fit every bounded tree budget with a device-resident beam search.
 *
 * All candidate matrices are row-major `[point_count, candidate_count]` FP64.
 * The session already owns fitting and measured-p95 matrices; this call adds
 * measured means and maxima required by later exact objective keys.
 *
 * Raw training N/K geometry and compact feature-axis operations are sufficient
 * inputs. The device computes exact rational values, sorts and coalesces them,
 * builds point/prefix masks, and constructs a normalized split threshold only
 * when an expansion actually uses that `(lower, upper)` pair.
 *
 * The result array has `max_leaves` entries. Entry `i` is the best tree under
 * a maximum budget of `i + 1` leaves, matching the canonical Python oracle.
 * Structure tokens and selected split descriptors follow the format documented
 * by `LlaminarNativeVNNITreeFitResult`.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
int llaminarNativeVNNITreeSearch(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    const double* measured_mean_regrets,
    const double* measured_max_regrets,
    const std::uint32_t* point_group_ranks,
    const std::uint64_t* training_aggregate_n,
    const std::uint64_t* training_k,
    const LlaminarNativeVNNITreeFeatureAxis* feature_axes,
    std::uint32_t axis_count,
    std::uint32_t boundary_placement,
    std::uint32_t parallelism_width,
    std::uint32_t task_multiplier,
    std::uint32_t min_shape_groups_per_leaf,
    std::uint32_t max_leaves,
    LlaminarNativeVNNITreeFitResult* results,
    char* error,
    std::size_t error_capacity);

/**
 * @brief Fit every tree budget and score heldout points in the same graph.
 *
 * This is the grouped-CV hot path. It runs the same exact bounded search as
 * `llaminarNativeVNNITreeSearch`, then evaluates every resulting tree against
 * raw heldout `(aggregate_n, k)` geometry before the captured graph completes.
 * Exact heldout winners are selected on-device by the canonical tuple
 * `(maximum, p95, mean, candidate_index)`. Missing candidates are represented
 * by positive infinity in all three heldout matrices.
 *
 * Search results never cross the device boundary on this route. The only D2H
 * payload is one compact evaluation per leaf budget, one exact-candidate index
 * per heldout point, and the scalar device status. The session's explicit
 * nonblocking stream is synchronized exactly once after those copies; there
 * are no per-depth, per-leaf, or whole-device synchronizations.
 *
 * @param heldout_aggregate_n Runtime aggregate-N value for each heldout point.
 * @param heldout_k Runtime K value for each heldout point.
 * @param heldout_measured_p95_regrets Row-major heldout candidate p95 matrix.
 * @param heldout_measured_mean_regrets Row-major heldout candidate mean matrix.
 * @param heldout_measured_max_regrets Row-major heldout candidate maximum matrix.
 * @param heldout_point_count Number of heldout rows, in `[1, 512]`.
 * @param evaluations Host output containing one compact decision set per budget.
 * @param exact_candidate_indices Host output with one exact winner per point.
 * @return Zero on success and nonzero on validation or backend failure.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
int llaminarNativeVNNITreeSearchAndEvaluate(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    const double* measured_mean_regrets,
    const double* measured_max_regrets,
    const std::uint32_t* point_group_ranks,
    const std::uint64_t* training_aggregate_n,
    const std::uint64_t* training_k,
    const LlaminarNativeVNNITreeFeatureAxis* feature_axes,
    const std::uint64_t* heldout_aggregate_n,
    const std::uint64_t* heldout_k,
    const double* heldout_measured_p95_regrets,
    const double* heldout_measured_mean_regrets,
    const double* heldout_measured_max_regrets,
    std::uint32_t heldout_point_count,
    std::uint32_t axis_count,
    std::uint32_t boundary_placement,
    std::uint32_t parallelism_width,
    std::uint32_t task_multiplier,
    std::uint32_t min_shape_groups_per_leaf,
    std::uint32_t max_leaves,
    LlaminarNativeVNNITreeFoldEvaluation* evaluations,
    std::uint32_t* exact_candidate_indices,
    char* error,
    std::size_t error_capacity);

/**
 * @brief Read monotonic runtime diagnostics without synchronizing the device.
 *
 * The counters describe host-side orchestration decisions that have already
 * completed. In particular, `tree_search_intermediate_sync_count` must remain
 * zero: a complete bounded search is one stream transaction with one terminal
 * synchronization after its result and status are copied to the host.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
int llaminarNativeVNNITreeRuntimeStats(
    const LlaminarNativeVNNILeafPrimaryScorerSession* session,
    LlaminarNativeVNNITreeRuntimeStats* stats,
    char* error,
    std::size_t error_capacity);

/**
 * @brief Destroy a prepared session and release all device resources.
 * @param session Session to destroy. A null session is rejected.
 * @param error Destination for a human-readable runtime error.
 * @param error_capacity Number of writable bytes in @p error.
 * @return Zero on success and nonzero when backend cleanup reports an error.
 */
LLAMINAR_NATIVE_VNNI_SCORER_EXPORT
int llaminarNativeVNNILeafPrimaryScorerDestroy(
    LlaminarNativeVNNILeafPrimaryScorerSession* session,
    char* error,
    std::size_t error_capacity);

} // extern "C"
