/**
 * @file ROCmMoERouteOwnedTuning.h
 * @brief Explicit diagnostic geometry of the existing serial-K MoE producer.
 *
 * These capabilities inspect the actual loaded HIP specialization before graph
 * construction. A thread-local lexical scope changes only independent output
 * column ownership; decoding, serial K partitions, contribution arithmetic,
 * SwiGLU preparation and ordered down publication remain production code.
 * Ordinary dispatch, including Q6_K/IQ2_S subgroup policies, is unchanged.
 * There is no environment switch, installed winner, allocation or replay probe.
 * Launch receipts authenticate host enqueue/capture, not hardware execution,
 * replay counts, complete graph-cache identity or whole-operation provenance.
 */
#pragma once

#include "../../common/MoEPrefillProjectionExecution.h"
#include "../../../tensors/NativeVnniFormatInfo.h"
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
/** @brief Every candidate has one serial-K lane per output column. */
inline constexpr std::array kMoERouteOwnedGateUpGeometries{
    MoERouteOwnedGateUpGeometry::TN64, MoERouteOwnedGateUpGeometry::TN128,
    MoERouteOwnedGateUpGeometry::TN256};

/**
 * @brief Resolve a reviewed column width without a device operation.
 * @param geometry Closed diagnostic geometry, never an occupancy request.
 * @return Exact output tile and workgroup width; K_LANES is always one.
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

class MoERouteOwnedGateUpAdmission;
/**
 * @brief Seal every actual leaf in one original descriptor-table format envelope.
 * @param geometry Common independent-column width, with K_LANES1 for every family.
 * @param envelope Exact immutable setup envelope produced by addNativeVnniExecutionFormat.
 * @param device_ordinal Actual owner used for all cold inspections.
 * @return Immutable complete envelope/resource capability; no partial admission.
 * @throws std::invalid_argument For empty/unsupported envelope or invalid geometry/device.
 * @throws std::logic_error During capture.
 * @throws std::runtime_error When any exact native leaf is nonresident or uses private storage.
 */
[[nodiscard]] MoERouteOwnedGateUpAdmission admitMoERouteOwnedGateUpGeometry(
    MoERouteOwnedGateUpGeometry geometry, const NativeVnniExecutionFormatEnvelope &envelope, int device_ordinal);

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
    friend MoERouteOwnedGateUpAdmission admitMoERouteOwnedGateUpGeometry(
        MoERouteOwnedGateUpGeometry, const NativeVnniExecutionFormatEnvelope &, int);
    /**
     * @brief Publish one complete cold-inspected capability after every leaf succeeds.
     * @param geometry Common TN; arithmetic and production dispatch order are unchanged.
     * @param envelope Exact original descriptor-table execution/source-policy contract.
     * @param device Owning device of every inspection.
     * @param resources Fixed decoder-indexed observations, populated only for launch families.
     */
    MoERouteOwnedGateUpAdmission(MoERouteOwnedGateUpGeometry geometry,
        NativeVnniExecutionFormatEnvelope envelope, int device,
        std::array<MoERouteOwnedKernelResources, 32> resources)
        : geometry_(geometry), envelope_(envelope), device_(device), resources_(resources) {}
    MoERouteOwnedGateUpGeometry geometry_;
    NativeVnniExecutionFormatEnvelope envelope_;
    int device_;
    std::array<MoERouteOwnedKernelResources, 32> resources_;

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
 * @throws std::logic_error For an incompatible active scope, clearing stale receipt first.
 * @note A new ordinary attempt clears old diagnostic evidence, then returns without policy checks or HIP calls.
 */
void validateMoERouteOwnedGateUpScopeBinding(int device_ordinal, void *stream,
    int rows, const MoEPrefillProjectionExecution &execution);
/**
 * @brief Require the exact original physical/source-policy envelope before preparation.
 * @param mask Actual descriptor-table execution mask, not a source-label guess.
 * @param policy_mask Original source-policy mask; zero never authenticates an active scope.
 * @throws std::logic_error For a mismatched active scope, clearing stale receipt.
 * @note Pure host validation; inactive ordinary calls return immediately.
 */
void validateMoERouteOwnedGateUpCodebook(std::uint32_t mask, std::uint32_t policy_mask = 0);
/** @return Calling thread's current explicit override, absent in ordinary dispatch. */
[[nodiscard]] std::optional<MoERouteOwnedGateUpGeometry> currentMoERouteOwnedGateUpGeometry() noexcept;
/** @return Complete accepted construction set, or no evidence after failed/incomplete/ordinary attempts. */
[[nodiscard]] std::optional<MoERouteOwnedGateUpLaunchSet> lastMoERouteOwnedGateUpLaunchSet() noexcept;
/** @return Delegating singleton view only; multi-leaf sets are never reduced to their last record. */
[[nodiscard]] std::optional<MoERouteOwnedGateUpLaunch> lastMoERouteOwnedGateUpLaunch() noexcept;
} // namespace llaminar2::rocm
