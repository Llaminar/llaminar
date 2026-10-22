/**
 * @file ROCmMoERouteOwnedTuning.h
 * @brief Immutable native geometry of the existing source-ordered MoE producer.
 *
 * These capabilities inspect the actual loaded HIP specialization before graph
 * construction. A thread-local lexical scope changes only independent output
 * column ownership or execution of independent original K partitions; decoding,
 * source-owned partition boundaries, per-partition contribution arithmetic,
 * SwiGLU preparation and ordered down publication remain production code.
 * Automatic model policy selects the measured MI50 native owner operation;
 * explicit Serial retains ordinary Q6_K/IQ2_S subgroup policies. Selection may
 * attach an immutable capability to an
 * original runtime descriptor table; it uses the same launcher without TLS model
 * policy. Logical plan identity exists before graph hashing, while cold resource
 * admission remains separate. There is no serving environment switch,
 * independent allocation authority or replay probe.
 * Launch receipts authenticate host enqueue/capture, not hardware execution,
 * replay counts, complete graph-cache identity or whole-operation provenance.
 */
#pragma once

#include "../../common/MoEPrefillProjectionExecution.h"
#include "../../common/NativeVNNIGroupedDecodePolicy.h"
#include "../../../tensors/NativeVnniFormatInfo.h"
#include "../../../backends/ComputeBackend.h"
#include "../../../config/CollectiveBackendType.h"
#include "../../../execution/moe/DeviceMoEOverlayServiceTelemetry.h"
#include <algorithm>
#include <cmath>
#include <span>
#include <string_view>
#include <bit>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace llaminar2::rocm
{
/** @brief Exactly the existing template's three independent-column widths. */
enum class MoERouteOwnedGateUpGeometry : std::uint8_t { TN64, TN128, TN256 };
/** @brief Execution of the unchanged source-owned partition program, independent of column geometry. */
enum class MoERouteOwnedGateUpPartitionExecution : std::uint8_t
{
    Serial, ///< Existing lane-local partition computation and ascending fold.
    FourWorkerPlanes ///< Four full TN64 planes compute independent original partitions.
};
/**
 * @brief Named immutable axes for one diagnostic owner gate/up projection.
 *
 * Default{} preserves historical unbound Serial admission. Bound admissions
 * require both positive axes and K32 alignment. This is not a model or policy
 * inference: output_columns is physical gate/up F and reduction_columns is H.
 */
struct MoERouteOwnedProjectionShape
{
    int output_columns = 0; ///< Actual gate/up N; never a down slice/full-H policy key.
    int reduction_columns = 0; ///< Actual gate/up K in original K32 blocks.
};

/** @brief Fixed diagnostic worker geometry; never a replacement partition policy. */
inline constexpr int kMoERouteOwnedGateUpPartitionWorkers = 4;
/** @brief Each worker wave retains the original coalesced64-column footprint. */
inline constexpr int kMoERouteOwnedGateUpPartitionColumns = 64;

/**
 * @brief Identify every compiled physical NativeVNNI decoder for original-partition workers.
 * @param codebook Actual physical decoder, not the source arithmetic policy.
 * @return Whether this explicit partition-worker candidate is implemented for the decoder.
 * @note The cold symbol selector resolves the actual compiled decoder; ordinary serial coverage is unchanged.
 */
[[nodiscard]] constexpr bool moeRouteOwnedGateUpPartitionCodebook(std::uint8_t codebook) noexcept
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        if (canonicalDeviceVnniFormat(*source.metadata).codebook_id == codebook ||
            migrationStableDeviceVnniFormat(*source.metadata).codebook_id == codebook)
            return true;
    return false;
}

/**
 * @brief Size the exact partition-major partial bank from an authenticated source-policy maximum.
 * @param maximum_partitions Maximum original count resolved across all required source policies.
 * @return One FP32 partial for each original partition and each64-column lane.
 * @throws std::invalid_argument For a count outside the original captured-policy ABI.
 * @note This allocates no memory; the original launch owns dynamic LDS. Capacity is not a new KB policy.
 */
[[nodiscard]] constexpr std::size_t moeRouteOwnedGateUpPartitionSharedBytes(int maximum_partitions)
{
    if (maximum_partitions <= 0 || maximum_partitions > NativeVNNIGroupedDecodePolicy::maximum_k_partitions)
        throw std::invalid_argument("route-owned partition bank requires an original resolved count");
    return static_cast<std::size_t>(maximum_partitions) * kMoERouteOwnedGateUpPartitionColumns * sizeof(float);
}

/** @brief Immutable IQ2_S codebook prefix shared by query, launch and device layout. */
[[nodiscard]] constexpr std::size_t moeRouteOwnedGateUpCodebookSharedBytes(std::uint8_t codebook) noexcept
{
    return codebook == 13 ? 1024u * sizeof(std::uint64_t) : 0u;
}

/** @brief Exact disjoint K1 LDS regions; activation staging of ordinary K_LANES>1 is separate. */
struct MoERouteOwnedGateUpK1SharedLayout
{
    std::size_t codebook_bytes; ///< Immutable grid prefix; zero except IQ2_S.
    std::size_t partial_offset; ///< First byte after the immutable grid.
    std::size_t partial_bytes; ///< Original source-partition bank; zero for Serial.
    std::size_t total_bytes; ///< Exact launch-owned dynamic LDS.
};

/**
 * @brief Resolve disjoint immutable-grid and original-partition banks for one actual decoder.
 * @param codebook Normalized physical launch decoder; original policy remains separate.
 * @param execution Serial K1 or four full worker planes.
 * @param maximum_partitions Original complete-envelope maximum; required only for four planes.
 * @return Exact K1 layout used by resource admission, launch and evidence validation.
 * @throws std::invalid_argument For unknown decoder/mode or invalid parallel partition count.
 */
[[nodiscard]] constexpr MoERouteOwnedGateUpK1SharedLayout moeRouteOwnedGateUpK1SharedLayout(
    std::uint8_t codebook, MoERouteOwnedGateUpPartitionExecution execution, int maximum_partitions = 0)
{
    if (!moeRouteOwnedGateUpPartitionCodebook(codebook) ||
        (execution != MoERouteOwnedGateUpPartitionExecution::Serial &&
         execution != MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes))
        throw std::invalid_argument("route-owned K1 LDS requires a known decoder and partition mode");
    const auto grid = moeRouteOwnedGateUpCodebookSharedBytes(codebook);
    const auto partial = execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes
        ? moeRouteOwnedGateUpPartitionSharedBytes(maximum_partitions) : 0u;
    return {grid, grid, partial, grid + partial};
}

/** @brief Closed output-column widths; partition worker count is a separate explicit axis. */
inline constexpr std::array kMoERouteOwnedGateUpGeometries{
    MoERouteOwnedGateUpGeometry::TN64, MoERouteOwnedGateUpGeometry::TN128,
    MoERouteOwnedGateUpGeometry::TN256};

/**
 * @brief Resolve a reviewed column width without a device operation.
 * @param geometry Closed diagnostic geometry, never an occupancy request.
 * @return Exact output tile width; native thread count additionally depends on partition execution.
 * @throws std::invalid_argument For unknown values, without substitution.
 */
[[nodiscard]] constexpr int moeRouteOwnedGateUpColumns(MoERouteOwnedGateUpGeometry geometry)
{
    switch (geometry)
    {
    case MoERouteOwnedGateUpGeometry::TN64: return 64;
    case MoERouteOwnedGateUpGeometry::TN128: return 128;
    case MoERouteOwnedGateUpGeometry::TN256: return 256;
    }
    throw std::invalid_argument("unknown ROCm route-owned gate/up geometry");
}

/** @brief Exact loaded leaf resources; residency is theoretical, not measured occupancy. */
struct MoERouteOwnedKernelResources
{
    const void *function = nullptr; ///< Actual HIP function authority.
    const char *specialization = nullptr; ///< Static readable host-template specialization; function is authority.
    int block_threads = 0; ///< Exact occupancy-query block width.
    int registers_per_thread = 0; ///< HIP compiler register observation.
    std::size_t local_memory_bytes_per_thread = 0; ///< Zero required for admission.
    std::size_t static_shared_memory_bytes = 0; ///< Compiler-owned LDS.
    std::size_t dynamic_shared_memory_bytes = 0; ///< Exact codebook staging, if any.
    int max_threads_per_block = 0; ///< Compiled launch ceiling.
    int max_active_blocks_per_sm = 0; ///< Residency at this block/LDS geometry.
};

/**
 * @brief Normalize physical decoder membership through the original IQ4 family rule.
 * @param mask Original capture-stable physical execution mask.
 * @return Original mask except that source-scale24 owns ordinary4 when present.
 * @note This is launch-family metadata, never a descriptor rewrite or live count.
 */
[[nodiscard]] constexpr std::uint32_t moeRouteOwnedGateUpLaunchCodebookMask(std::uint32_t mask) noexcept
{
    const auto tuple = nativeVnniCodebookMaskBit(kNativeVnniIQ4XSSourceScaleCodebook);
    return mask & tuple ? mask & ~nativeVnniCodebookMaskBit(4) : mask;
}

/**
 * @brief Validate immutable execution/source-policy membership without a backend query.
 * @param envelope Original canonical or runtime-mutable descriptor-table envelope.
 * @return True only for nonempty known physical and source arithmetic identities.
 * @note Supported identities come from the existing canonical source/migration catalog.
 */
[[nodiscard]] constexpr bool moeRouteOwnedGateUpEnvelopeValid(
    const NativeVnniExecutionFormatEnvelope &envelope) noexcept
{
    std::uint32_t known_execution = 0, known_policy = 0;
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        known_execution |= nativeVnniCodebookMaskBit(canonicalDeviceVnniFormat(*source.metadata).codebook_id);
        known_execution |= nativeVnniCodebookMaskBit(migrationStableDeviceVnniFormat(*source.metadata).codebook_id);
        known_policy |= nativeVnniCodebookMaskBit(canonicalDeviceVnniCodebookId(source.metadata->codebook_id));
    }
    return envelope.valid() && !(envelope.execution_codebook_mask & ~known_execution) &&
        !(envelope.policy_codebook_mask & ~known_policy);
}

/**
 * @brief Validate explicit partition execution and immutable N/K axes without a backend query.
 * @param execution Original lane-local computation or the bounded four-plane component candidate.
 * @param envelope Exact original physical/source-policy descriptor-table envelope.
 * @param shape Actual named gate/up N/K; both zero axes preserve only original serial compatibility.
 * @return Whether shape and all normalized decoders support this explicit execution mode.
 */
[[nodiscard]] constexpr bool moeRouteOwnedGateUpPartitionExecutionValid(
    MoERouteOwnedGateUpPartitionExecution execution, const NativeVnniExecutionFormatEnvelope &envelope,
    MoERouteOwnedProjectionShape shape) noexcept
{
    const int reduction_width = shape.reduction_columns;
    const int output_width = shape.output_columns;
    if (!moeRouteOwnedGateUpEnvelopeValid(envelope) || reduction_width < 0 || output_width < 0 ||
        ((reduction_width == 0) != (output_width == 0)) ||
        (reduction_width > 0 && reduction_width % 32 != 0)) return false;
    if (execution == MoERouteOwnedGateUpPartitionExecution::Serial) return true;
    if (execution != MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes || reduction_width == 0) return false;
    const auto mask = moeRouteOwnedGateUpLaunchCodebookMask(envelope.execution_codebook_mask);
    for (std::uint8_t codebook = 0; codebook < 32; ++codebook)
        if ((mask & nativeVnniCodebookMaskBit(codebook)) && !moeRouteOwnedGateUpPartitionCodebook(codebook)) return false;
    return true;
}

/**
 * @brief Validate an existing resolver output and obtain its exact complete-envelope maximum.
 * @param envelope Original source-policy membership that owns each resolved entry.
 * @param partitions Existing resolver output, never caller-supplied capability authority.
 * @param reduction_width Actual K32 width against which every entry was resolved.
 * @return Maximum required count, or zero for missing/foreign/out-of-range entries.
 * @note Nonowned entries must remain zero; no physical-decoder-to-source inference is made.
 */
[[nodiscard]] constexpr int moeRouteOwnedGateUpMaximumKPartitions(
    const NativeVnniExecutionFormatEnvelope &envelope, const std::array<std::uint8_t, 32> &partitions,
    int reduction_width) noexcept
{
    if (!moeRouteOwnedGateUpEnvelopeValid(envelope) || reduction_width <= 0 || reduction_width % 32 != 0) return 0;
    int maximum = 0;
    for (std::uint8_t codebook = 0; codebook < 32; ++codebook)
    {
        const int count = partitions[codebook];
        if (envelope.policy_codebook_mask & nativeVnniCodebookMaskBit(codebook))
        {
            if (count <= 0 || count > reduction_width / 32 || count > NativeVNNIGroupedDecodePolicy::maximum_k_partitions) return 0;
            if (count > maximum) maximum = count;
        }
        else if (count != 0) return 0;
    }
    return maximum;
}

/**
 * @brief Hash complete logical launch intent without querying native resources or live tensors.
 * @param geometry Exact independent-column geometry.
 * @param envelope Complete original source/execution membership.
 * @param device Exact declaration device.
 * @param execution Serial diagnostic geometry or four original-partition workers.
 * @param shape Named physical gate/up N/K, never inferred from a model name.
 * @param partitions Original source-policy resolver output; zero only for unbound Serial compatibility.
 * @return Stable nonzero binary-local logical identity; native resources must still be admitted.
 * @throws std::invalid_argument For incomplete or contradictory intent.
 * @note This identity confers no allocation, resource capability, hardware execution or graph ownership.
 */
[[nodiscard]] constexpr std::uint64_t moeRouteOwnedGateUpPlanIdentity(
    MoERouteOwnedGateUpGeometry geometry, const NativeVnniExecutionFormatEnvelope &envelope, int device,
    MoERouteOwnedGateUpPartitionExecution execution, MoERouteOwnedProjectionShape shape,
    const std::array<std::uint8_t, 32> &partitions)
{
    const auto columns = moeRouteOwnedGateUpColumns(geometry);
    if (device < 0 || !moeRouteOwnedGateUpPartitionExecutionValid(execution, envelope, shape) ||
        (execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes && geometry != MoERouteOwnedGateUpGeometry::TN64))
        throw std::invalid_argument("route-owned logical intent requires exact supported axes");
    const int maximum = shape.reduction_columns > 0
        ? moeRouteOwnedGateUpMaximumKPartitions(envelope, partitions, shape.reduction_columns) : 0;
    if (shape.reduction_columns > 0 && maximum <= 0)
        throw std::invalid_argument("route-owned logical intent has an incomplete source-policy array");
    if (shape.reduction_columns == 0)
        for (auto count : partitions) if (count) throw std::invalid_argument("unbound Serial intent cannot invent source partitions");
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](std::uint64_t value) constexpr { hash = (hash ^ value) * 1099511628211ull; };
    mix(0x4d4f45524f555445ull); mix(1); mix(static_cast<std::uint64_t>(geometry));
    mix(static_cast<std::uint64_t>(execution)); mix(device); mix(shape.output_columns); mix(shape.reduction_columns);
    mix(envelope.execution_codebook_mask); mix(envelope.policy_codebook_mask); mix(columns); mix(maximum);
    mix(execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes ? kMoERouteOwnedGateUpPartitionWorkers : 1);
    for (auto count : partitions) mix(count);
    return hash ? hash : 1;
}

class MoERouteOwnedGateUpPlan;
/**
 * @brief Resolve immutable original source-policy launch intent without selecting a HIP device.
 * @param geometry Exact independent-column geometry.
 * @param envelope Canonical complete table envelope, including reachable migration leaves.
 * @param device Exact owning declaration device.
 * @param execution Explicit partition execution, independent of original KB policy.
 * @param shape Actual gate/up axes; default{} is historical Serial compatibility only.
 * @return Readonly logical plan from the SAME production source-policy resolver.
 * @throws std::invalid_argument For malformed/unsupported geometry or source policy.
 * @throws std::logic_error During capture or with conflicting explicit numerical/tile controls.
 * @note Performs only host metadata work; never native resource queries or allocation.
 */
[[nodiscard]] MoERouteOwnedGateUpPlan planMoERouteOwnedGateUp(
    MoERouteOwnedGateUpGeometry geometry, const NativeVnniExecutionFormatEnvelope &envelope, int device,
    MoERouteOwnedGateUpPartitionExecution execution, MoERouteOwnedProjectionShape shape);

/** @brief Immutable logical plan, distinct from a native resource capability. */
class MoERouteOwnedGateUpPlan final
{
public:
    /** @return Exact independent-column width. */
    [[nodiscard]] MoERouteOwnedGateUpGeometry geometry() const noexcept { return geometry_; }
    /** @return Complete original masks; never a live descriptor/placement shadow. */
    [[nodiscard]] const NativeVnniExecutionFormatEnvelope &executionEnvelope() const noexcept { return envelope_; }
    /** @return Exact owner selected by graph declaration. */
    [[nodiscard]] int deviceOrdinal() const noexcept { return device_; }
    /** @return Explicit partition execution of the unchanged numerical program. */
    [[nodiscard]] MoERouteOwnedGateUpPartitionExecution partitionExecution() const noexcept { return execution_; }
    /** @return Named physical axes. */
    [[nodiscard]] const MoERouteOwnedProjectionShape &shape() const noexcept { return shape_; }
    /** @return Exact source-policy-indexed original resolver output. */
    [[nodiscard]] const std::array<std::uint8_t, 32> &arithmeticPartitions() const noexcept { return partitions_; }
    /** @return Exact maximum required source partition count. */
    [[nodiscard]] int maximumKPartitions() const noexcept { return maximum_; }
    /** @return Pure immutable intent identity, stable before and after resource admission. */
    [[nodiscard]] std::uint64_t identity() const { return moeRouteOwnedGateUpPlanIdentity(geometry_, envelope_, device_, execution_, shape_, partitions_); }
private:
    friend MoERouteOwnedGateUpPlan planMoERouteOwnedGateUp(MoERouteOwnedGateUpGeometry,
        const NativeVnniExecutionFormatEnvelope &, int, MoERouteOwnedGateUpPartitionExecution, MoERouteOwnedProjectionShape);
    /**
     * @brief Publish the complete canonical resolver result; native admission remains separate.
     * @param geometry Exact independent-column width.
     * @param envelope Original complete execution and source-policy membership.
     * @param device Exact declaration owner, without native device selection.
     * @param execution Immutable partition worker mode.
     * @param shape Named actual physical projection N/K.
     * @param partitions Original source-policy resolver result for those exact axes.
     * @param maximum Validated maximum over original source-policy members.
     */
    MoERouteOwnedGateUpPlan(MoERouteOwnedGateUpGeometry geometry, NativeVnniExecutionFormatEnvelope envelope,
        int device, MoERouteOwnedGateUpPartitionExecution execution, MoERouteOwnedProjectionShape shape,
        std::array<std::uint8_t, 32> partitions, int maximum)
        : geometry_(geometry), envelope_(envelope), device_(device), execution_(execution), shape_(shape),
          partitions_(partitions), maximum_(maximum) {}
    MoERouteOwnedGateUpGeometry geometry_;
    NativeVnniExecutionFormatEnvelope envelope_;
    int device_;
    MoERouteOwnedGateUpPartitionExecution execution_;
    MoERouteOwnedProjectionShape shape_;
    std::array<std::uint8_t, 32> partitions_;
    int maximum_;
};

/** @brief Authored compiled intent, separate from the unchanged physical execution enum. */
enum class MoERouteOwnedModelPartitionPolicy : std::uint8_t
{
    Serial = 0, ///< Original unbound automatic kernels, including Q6/IQ2 K_LANES policies.
    FourWorkerPlanes = 1, ///< Explicit existing resource-admitted owner M1 execution.
    Automatic = 2 ///< Bounded measured operation; every other domain retains Serial.
};

/** @brief Immutable operation axes borrowed from the actual projection declaration. */
struct MoERouteOwnedModelOperation
{
    MoERouteOwnedProjectionShape shape; ///< Physical gate/up N/K from original ownership.
    int rows = 0; ///< Physical rows, not a live count or allocation capacity.
    int experts = 0; ///< Original source inventory from fixed-down ownership.
    int top_k = 0; ///< Original router width.
    MoEOverlayServicePhaseHint phase = MoEOverlayServicePhaseHint::Auto; ///< Existing graph semantic phase.
    bool has_live_rows = false; ///< Retained query-range inputs exclude the initial ordinary domain.
};

/**
 * @brief Recognize only measured MI50/gfx906 from the existing startup observation.
 * @param device Canonical DeviceManager entry; compute_capability is native gfx906->906 on ROCm.
 * @return Whether exact architecture, CU count and reported card identity match.
 * @note No native query, inventory publication or capacity accounting occurs here.
 */
[[nodiscard]] inline bool moeRouteOwnedModelMeasuredDevice(const ComputeDevice &device) noexcept
{
    return device.type == ComputeBackendType::GPU_ROCM && device.compute_capability == 906 &&
        device.compute_units == 60 && device.name.find("MI50") != std::string::npos;
}

/**
 * @brief Classify the measured operation/domain before touching prepared source metadata.
 * @param operation Actual stage axes and semantic query policy.
 * @param participants Exact local-TP membership, never inferred from ordinal ordering.
 * @param weights Actual proportional shares; empty is the existing equal policy.
 * @param backend Resolved native collective backend.
 * @param inventory Borrowed canonical startup devices; never retained across reinitialization.
 * @return Whether the bounded ordinary domain matches without live routing/count inspection.
 */
[[nodiscard]] inline bool moeRouteOwnedModelMeasuredDomain(const MoERouteOwnedModelOperation &operation,
    std::span<const GlobalDeviceAddress> participants, std::span<const float> weights,
    CollectiveBackendType backend, std::span<const ComputeDevice> inventory) noexcept
{
    if (operation.rows != 1 || operation.shape.output_columns != 640 || operation.shape.reduction_columns != 2560 ||
        operation.experts != 512 || operation.top_k != 10 || operation.has_live_rows ||
        operation.phase != MoEOverlayServicePhaseHint::Decode || participants.size() != 4 ||
        backend != CollectiveBackendType::RCCL) return false;
    if (!weights.empty() && (weights.size() != 4 || !std::isfinite(weights.front()) || weights.front() <= 0 ||
        !std::all_of(weights.begin(), weights.end(), [&](float value) { return value == weights.front(); }))) return false;
    for (std::size_t index = 0; index < participants.size(); ++index)
    {
        const auto &address = participants[index];
        if (address.device_type != DeviceType::ROCm || address.hostname != participants.front().hostname ||
            std::any_of(participants.begin(), participants.begin() + index,
                [&](const auto &prior) { return prior.device_ordinal == address.device_ordinal; })) return false;
        const ComputeDevice *observed = nullptr;
        for (const auto &device : inventory)
            if (device.type == ComputeBackendType::GPU_ROCM && device.device_id == address.device_ordinal)
            {
                if (observed) return false;
                observed = &device;
            }
        if (!observed || !moeRouteOwnedModelMeasuredDevice(*observed)) return false;
    }
    return true;
}

/**
 * @brief Resolve compiled intent from the actual operation and original prepared source identities.
 * @param policy Authored source-bound intent; explicit choices remain authoritative.
 * @param operation Actual stage axes; this pure result is not a resource capability.
 * @param participants Original local communicator membership.
 * @param weights Original local vocabulary-independent TP shares.
 * @param backend Resolved collective transport.
 * @param inventory Existing immutable startup observation.
 * @param sources Inspected original gate/up identities, including superblock provenance.
 * @return Four only for an explicit request or the complete measured Automatic tuple.
 * @throws std::invalid_argument For an unknown authored policy.
 * @note Serial means original production automatic kernels, never forced diagnostic Serial/K1.
 */
[[nodiscard]] inline MoERouteOwnedGateUpPartitionExecution selectMoERouteOwnedModelPartitionExecution(
    MoERouteOwnedModelPartitionPolicy policy, const MoERouteOwnedModelOperation &operation,
    std::span<const GlobalDeviceAddress> participants, std::span<const float> weights,
    CollectiveBackendType backend, std::span<const ComputeDevice> inventory,
    std::span<const NativeVnniSourceIdentity> sources)
{
    using Execution = MoERouteOwnedGateUpPartitionExecution;
    if (policy == MoERouteOwnedModelPartitionPolicy::Serial) return Execution::Serial;
    if (policy == MoERouteOwnedModelPartitionPolicy::FourWorkerPlanes) return Execution::FourWorkerPlanes;
    if (policy != MoERouteOwnedModelPartitionPolicy::Automatic)
        throw std::invalid_argument("unknown compiled ROCm owner partition policy");
    if (!moeRouteOwnedModelMeasuredDomain(operation, participants, weights, backend, inventory) || sources.empty())
        return Execution::Serial;
    const auto family = sources.front().codebook_id;
    if (family != native_vnni_formats::Q4_K.codebook_id && family != native_vnni_formats::Q5_K.codebook_id)
        return Execution::Serial;
    for (const auto &source : sources)
        if (!source.present || !source.is_superblock || source.codebook_id != family) return Execution::Serial;
    return Execution::FourWorkerPlanes;
}

/**
 * @brief Read source-bound intent without querying HIP, environment or device state.
 * @return The CMake-authored Automatic, explicit Serial or explicit Four policy.
 * @note Only the existing ROCm MoE bridge consumes the compile definition.
 */
[[nodiscard]] MoERouteOwnedModelPartitionPolicy compiledMoERouteOwnedModelPartitionPolicy() noexcept;

class MoERouteOwnedGateUpAdmission;
/**
 * @brief Materialize exactly one previously resolved logical plan before capture.
 * @param plan Original immutable intent; no source-policy re-resolution or new geometry choice.
 * @return Complete native capability after every required actual leaf succeeds.
 * @throws std::logic_error During native capture or with conflicting controls.
 * @throws std::runtime_error For incomplete/nonresident/private native resources.
 */
[[nodiscard]] MoERouteOwnedGateUpAdmission admitMoERouteOwnedGateUpGeometry(const MoERouteOwnedGateUpPlan &plan);

/**
 * @brief Seal every actual leaf in one original descriptor-table format envelope.
 * @param geometry Common independent-column width; four-plane execution admits TN64 only.
 * @param envelope Exact immutable setup envelope produced by addNativeVnniExecutionFormat.
 * @param device_ordinal Actual owner used for all cold inspections.
 * @param partition_execution Original serial execution or explicit four full worker planes.
 * @param shape Named physical gate/up N/K for original policy resolution and resource identity; default{} preserves serial compatibility.
 * @return Immutable complete envelope/resource capability; no partial admission.
 * @throws std::invalid_argument For empty/unsupported envelope or invalid geometry/device.
 * @throws std::logic_error During capture.
 * @throws std::runtime_error When any exact native leaf is nonresident or uses private storage.
 */
[[nodiscard]] MoERouteOwnedGateUpAdmission admitMoERouteOwnedGateUpGeometry(
    MoERouteOwnedGateUpGeometry geometry, const NativeVnniExecutionFormatEnvelope &envelope, int device_ordinal,
    MoERouteOwnedGateUpPartitionExecution partition_execution = MoERouteOwnedGateUpPartitionExecution::Serial,
    MoERouteOwnedProjectionShape shape = {});

/**
 * @brief Seal actual symbol/resources before capture; never launch a kernel.
 * @param geometry Reviewed TN with K_LANES1, including explicit Q6_K/IQ2_S diagnostics.
 * @param execution_codebook Physical native decoder, not a source-format alias.
 * @note Compatibility helper requires the same valid source-policy bit; expanded/source-tuple decoders and real mutable tables must supply their exact typed envelope.
 * @param device_ordinal Exact owning HIP device.
 * @return Delegating singleton capability; unknown decoders fail explicitly.
 * @throws std::invalid_argument For unsupported codebooks, geometry or device.
 * @throws std::logic_error During graph capture.
 * @throws std::runtime_error For incomplete/nonresident/private-memory resources.
 */
[[nodiscard]] MoERouteOwnedGateUpAdmission admitMoERouteOwnedGateUpGeometry(
    MoERouteOwnedGateUpGeometry geometry, std::uint8_t execution_codebook, int device_ordinal);

/** @brief Only successful exact native resource inspection can construct this capability. */
class MoERouteOwnedGateUpAdmission final
{
public:
    /** @return Exact diagnostic TN. */
    [[nodiscard]] MoERouteOwnedGateUpGeometry geometry() const noexcept { return geometry_; }
    /** @return Explicit partition work sharing; the original partition arithmetic is unchanged. */
    [[nodiscard]] MoERouteOwnedGateUpPartitionExecution partitionExecution() const noexcept { return partition_execution_; }
    /** @return Exact admitted K; zero is original unbound serial compatibility only. */
    [[nodiscard]] int reductionWidth() const noexcept { return shape_.reduction_columns; }
    /** @return Exact admitted gate/up N; zero is original unbound serial compatibility only. */
    [[nodiscard]] int outputWidth() const noexcept { return shape_.output_columns; }
    /** @return Both original physical projection axes without an ambiguous positional pair. */
    [[nodiscard]] const MoERouteOwnedProjectionShape &shape() const noexcept { return shape_; }
    /** @return Actual original source-policy resolver output; zero only for unbound serial compatibility. */
    [[nodiscard]] const std::array<std::uint8_t, 32> &arithmeticPartitions() const noexcept { return arithmetic_partitions_; }
    /** @return Exact maximum over required original source policies; never a new split-K choice. */
    [[nodiscard]] int maximumKPartitions() const noexcept { return maximum_k_partitions_; }
    /** @return Exact original execution and source-policy masks, before family normalization. */
    [[nodiscard]] const NativeVnniExecutionFormatEnvelope &executionEnvelope() const noexcept { return envelope_; }
    /** @return Original decoder families after the production IQ4 normalization. */
    [[nodiscard]] std::uint32_t launchCodebookMask() const noexcept
    { return moeRouteOwnedGateUpLaunchCodebookMask(envelope_.execution_codebook_mask); }
    /** @return Exact number of inspected launch families, including inactive reachable ones. */
    [[nodiscard]] std::size_t leafCount() const noexcept { return std::popcount(launchCodebookMask()); }
    /**
     * @brief Resolve sealed resources for one actual normalized launch family.
     * @param codebook Actual template decoder, not an unnormalized IQ4 member.
     * @return The original cold native resource observation.
     * @throws std::invalid_argument For any leaf not owned by this capability.
     */
    [[nodiscard]] const MoERouteOwnedKernelResources &leafResources(std::uint8_t codebook) const
    {
        if (codebook >= 32 || !(launchCodebookMask() & nativeVnniCodebookMaskBit(codebook)))
            throw std::invalid_argument("route-owned resources requested for a foreign launch family");
        return resources_[codebook];
    }
    /**
     * @return Sole normalized launch decoder.
     * @throws std::logic_error For a multi-leaf envelope; no representative decoder is invented.
     */
    [[nodiscard]] std::uint8_t executionCodebook() const
    {
        if (leafCount() != 1) throw std::logic_error("route-owned envelope has multiple launch families");
        return static_cast<std::uint8_t>(std::countr_zero(launchCodebookMask()));
    }
    /** @return Native resource owner. */
    [[nodiscard]] int deviceOrdinal() const noexcept { return device_; }
    /**
     * @return Singleton compatibility resource view.
     * @throws std::logic_error For multiple leaves; use leafResources() for each admitted family.
     */
    [[nodiscard]] const MoERouteOwnedKernelResources &resources() const { return leafResources(executionCodebook()); }
private:
    friend MoERouteOwnedGateUpAdmission admitMoERouteOwnedGateUpGeometry(const MoERouteOwnedGateUpPlan &);
    /**
     * @brief Publish one complete cold-inspected capability after every leaf succeeds.
     * @param geometry Common TN; arithmetic and production dispatch order are unchanged.
     * @param envelope Exact original descriptor-table execution/source-policy contract.
     * @param device Owning device of every inspection.
     * @param resources Fixed decoder-indexed observations, populated only for launch families.
     * @param partition_execution Explicit work sharing, independent of original KB counts.
     * @param shape Named physical gate/up N/K, or default{} for original serial compatibility.
     * @param partitions Original source-policy-indexed resolver output for these exact N/K axes.
     * @param maximum_partitions Validated maximum across required original policies.
     */
    MoERouteOwnedGateUpAdmission(MoERouteOwnedGateUpGeometry geometry,
        NativeVnniExecutionFormatEnvelope envelope, int device,
        std::array<MoERouteOwnedKernelResources, 32> resources,
        MoERouteOwnedGateUpPartitionExecution partition_execution, MoERouteOwnedProjectionShape shape,
        std::array<std::uint8_t, 32> partitions, int maximum_partitions)
        : geometry_(geometry), envelope_(envelope), device_(device), resources_(resources),
          partition_execution_(partition_execution), shape_(shape),
          arithmetic_partitions_(partitions), maximum_k_partitions_(maximum_partitions) {}
    MoERouteOwnedGateUpGeometry geometry_;
    NativeVnniExecutionFormatEnvelope envelope_;
    int device_;
    std::array<MoERouteOwnedKernelResources, 32> resources_;
    MoERouteOwnedGateUpPartitionExecution partition_execution_;
    MoERouteOwnedProjectionShape shape_;
    std::array<std::uint8_t, 32> arithmetic_partitions_;
    int maximum_k_partitions_;

};

/**
 * @brief Inspect the unchanged fixed-down64 control on the exact device.
 * @param execution_codebook Physical native decoder; mixed tables are not this diagnostic.
 * @param layout Canonical route publication or original token-row publication.
 * @param device_ordinal Exact native owner.
 * @return Actual fixed-down leaf resources, including its original64-thread width.
 * @throws std::invalid_argument For unknown decoder/layout/device.
 * @throws std::logic_error During graph capture.
 * @throws std::runtime_error For incomplete/nonresident/private-memory resources.
 */
[[nodiscard]] MoERouteOwnedKernelResources inspectMoERouteOwnedDown64(
    std::uint8_t execution_codebook, MoEPrefillOutputLayout layout, int device_ordinal);

/** @brief One accepted host producer submission; device counts are not observed. */
struct MoERouteOwnedGateUpLaunch
{
    MoERouteOwnedGateUpGeometry geometry = MoERouteOwnedGateUpGeometry::TN64;
    MoERouteOwnedGateUpPartitionExecution partition_execution = MoERouteOwnedGateUpPartitionExecution::Serial;
    int partition_worker_planes = 1; ///< Actual execution workers, distinct from source arithmetic partitions.
    std::uint8_t execution_codebook = 0;
    int device_ordinal = -1;
    void *stream = nullptr; ///< Exact nondefault construction stream.
    int n = 0, k = 0; ///< Actual gate/up geometry, not an inferred model identity.
    int route_capacity = 0; ///< Captured slot bound; never reported as live traffic/work.
    int grid_x = 0, grid_y = 0, grid_z = 0;
    int block_x = 0, block_y = 0, block_z = 0;
    std::array<std::uint8_t, 32> arithmetic_partitions{}; ///< Original resolved source-policy tree.
    MoERouteOwnedKernelResources resources;
};

/**
 * @brief Immutable complete host submission set for one original raw producer transaction.
 *
 * Storage is bounded by the32-bit decoder ABI per attempt. Records retain the
 * original host dispatch order, including currently inactive reachable leaves.
 * No device count, descriptor value or replay state is read to construct it.
 */
struct MoERouteOwnedGateUpLaunchSet
{
    NativeVnniExecutionFormatEnvelope execution_envelope{}; ///< Original unnormalized masks.
    std::array<MoERouteOwnedGateUpLaunch, 32> leaves{}; ///< Accepted leaves in original launch-loop order.
    std::size_t leaf_count = 0; ///< Host submissions, never active routes or replay work.
};

/**
 * @brief Own one nonoverlapping diagnostic override on its constructing lowering thread.
 *
 * The diagnostic caller must own a fresh, empty native graph for each admitted
 * candidate. It must reject an existing executable or nonempty graph before
 * recording, then inspect the captured function and launch geometry against
 * its copied receipt after this scope exits. Ordinary graph-cache reuse across
 * candidates is forbidden: this receipt does not extend production cache identity.
 */
class ScopedMoERouteOwnedGateUpGeometry final
{
public:
    /**
     * @brief Select one inspected geometry without querying HIP or changing arithmetic.
     * @param admission Capability inspected outside capture on its actual device.
     * @throws std::logic_error During capture or with conflicting expert-tile/KB/TW controls.
     * @note Overlapping construction is rejected; a failed constructor publishes no new override.
     */
    explicit ScopedMoERouteOwnedGateUpGeometry(const MoERouteOwnedGateUpAdmission &admission);
    /** @brief Retire this thread's override; a foreign-thread/foreign-owner destruction is fatal. */
    ~ScopedMoERouteOwnedGateUpGeometry() noexcept;
    ScopedMoERouteOwnedGateUpGeometry(const ScopedMoERouteOwnedGateUpGeometry &) = delete;
    ScopedMoERouteOwnedGateUpGeometry &operator=(const ScopedMoERouteOwnedGateUpGeometry &) = delete;
    ScopedMoERouteOwnedGateUpGeometry(ScopedMoERouteOwnedGateUpGeometry &&) = delete;
    ScopedMoERouteOwnedGateUpGeometry &operator=(ScopedMoERouteOwnedGateUpGeometry &&) = delete;
    /** @return This lexical scope's immutable resource capability. */
    [[nodiscard]] const MoERouteOwnedGateUpAdmission &admission() const noexcept { return admission_; }
private:
    MoERouteOwnedGateUpAdmission admission_;
};

/**
 * @brief Validate an active diagnostic before public preparation/device selection.
 * @param device_ordinal Exact execution owner.
 * @param stream Exact explicit producer stream; null/default sentinels fail.
 * @param rows Must be physical M1; no live count readback or topology inference.
 * @param execution Must execute gate/up; separate down-only control runs outside this scope.
 * @param shape Actual named gate/up N/K, checked before any public preparation for a shape-bound capability.
 * @param explicit_admission Graph/table-owned model binding; simultaneous diagnostic scope is rejected.
 * @throws std::logic_error For an incompatible active scope, clearing stale receipt first.
 * @note A new ordinary attempt clears old diagnostic evidence, then returns without policy checks or HIP calls.
 */
void validateMoERouteOwnedGateUpScopeBinding(int device_ordinal, void *stream,
    int rows, const MoEPrefillProjectionExecution &execution, MoERouteOwnedProjectionShape shape = {},
    const MoERouteOwnedGateUpAdmission *explicit_admission = nullptr);
/**
 * @brief Require the exact original physical/source-policy envelope before preparation.
 * @param mask Actual descriptor-table execution mask, not a source-label guess.
 * @param policy_mask Original source-policy mask; zero never authenticates an active scope.
 * @param explicit_admission Immutable graph/table-owned model capability, without TLS ownership.
 * @throws std::logic_error For a mismatched active scope, clearing stale receipt.
 * @note Pure host validation; inactive ordinary calls return immediately.
 */
void validateMoERouteOwnedGateUpCodebook(std::uint32_t mask, std::uint32_t policy_mask = 0,
    const MoERouteOwnedGateUpAdmission *explicit_admission = nullptr);
/** @return Calling thread's current explicit override, absent in ordinary dispatch. */
[[nodiscard]] std::optional<MoERouteOwnedGateUpGeometry> currentMoERouteOwnedGateUpGeometry() noexcept;
/** @return Complete accepted construction set, or no evidence after failed/incomplete/ordinary attempts. */
[[nodiscard]] std::optional<MoERouteOwnedGateUpLaunchSet> lastMoERouteOwnedGateUpLaunchSet() noexcept;
/** @return Delegating singleton view only; multi-leaf sets are never reduced to their last record. */
[[nodiscard]] std::optional<MoERouteOwnedGateUpLaunch> lastMoERouteOwnedGateUpLaunch() noexcept;
} // namespace llaminar2::rocm
