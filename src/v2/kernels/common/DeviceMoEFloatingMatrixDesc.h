/**
 * @file DeviceMoEFloatingMatrixDesc.h
 * @brief Device ABI for contiguous floating-point routed-expert matrices.
 *
 * Routed MoE graphs must select weights from device-owned placement tables
 * after graph capture.  A raw pointer alone cannot prove the element format or
 * projection geometry, so this file defines the compact, backend-neutral
 * descriptor shared by graph construction, placement publication, CUDA, and
 * ROCm kernels.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace llaminar2
{
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_HOST_DEVICE __host__ __device__
#else
#define LLAMINAR_MOE_HOST_DEVICE
#endif
    /**
     * @brief Arithmetic format carried by one routed-expert descriptor family.
     *
     * A movable payload uses supported NativeVNNI codebooks or one contiguous
     * floating-point format. The value is stored once because its projections
     * must agree on that family; accepting mixed families inside one payload
     * would make migration byte counts and captured dispatch ambiguous. A fixed
     * down-slice bank has its own descriptors and does not inherit this tag.
     */
    // These are ABI tags, not tensor elements. Two 16-bit format tags leave
    // room for an explicit projection contract in the existing 240-byte expert
    // descriptor; no placement bank or transfer directory grows in VRAM.
    enum class DeviceMoEWeightFormat : std::uint16_t
    {
        NativeVNNI = 0,
        FP16 = 1,
        BF16 = 2,
        FP32 = 3,
    };

    /**
     * @brief Projections whose ownership changes in one placement transaction.
     *
     * GateUp is a complete movable pair, never an incomplete whole expert.
     * Its down projection belongs to a separate, immutable participant slice
     * bank and must not be copied, retired or priced by expert migration.
     */
    enum class DeviceMoEProjectionSet : std::uint32_t
    {
        CompleteExpert = 0,
        GateUp = 1,
    };

    /** @return Whether a device payload carries a supported ownership contract. */
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool
    deviceMoEProjectionSetValid(DeviceMoEProjectionSet projections) noexcept
    {
        return projections == DeviceMoEProjectionSet::CompleteExpert ||
               projections == DeviceMoEProjectionSet::GateUp;
    }

    /** @return Whether @p format names a contiguous floating-point payload. */
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool
    deviceMoEWeightFormatIsFloating(
        DeviceMoEWeightFormat format) noexcept
    {
        return format == DeviceMoEWeightFormat::FP16 ||
               format == DeviceMoEWeightFormat::BF16 ||
               format == DeviceMoEWeightFormat::FP32;
    }

    /**
     * @brief Device-readable view of one row-major floating-point matrix.
     *
     * `data` names immutable device storage owned by the prepared-weight or
     * transfer-slot lifetime.  Element precision is deliberately held by the
     * enclosing expert descriptor so every movable projection is validated as
     * a single arithmetic family before publication.
     */
    struct DeviceMoEFloatingMatrixDesc
    {
        const void *data = nullptr; ///< Contiguous row-major weight elements.
        std::int32_t n = 0;         ///< Output row count.
        std::int32_t k = 0;         ///< Reduction width.

        /** @return Whether this descriptor names a non-empty matrix. */
        [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool valid() const noexcept
        {
            return data != nullptr && n > 0 && k > 0;
        }
    };

    /** @return Raw scalar width; zero rejects quantized or unknown tags. */
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr std::uint32_t
    deviceMoEFloatingElementBytes(DeviceMoEWeightFormat format) noexcept
    {
        return format == DeviceMoEWeightFormat::FP32 ? 4u :
               (format == DeviceMoEWeightFormat::FP16 ||
                format == DeviceMoEWeightFormat::BF16) ? 2u : 0u;
    }

    /**
     * @return Exact wire bytes for one contiguous projection, or zero if invalid.
     * Positive int32 geometry and a maximum four-byte scalar cannot overflow
     * uint64. Host and both device compilers consume this same calculation.
     */
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr std::uint64_t
    deviceMoEFloatingMatrixBytes(
        const DeviceMoEFloatingMatrixDesc &matrix,
        DeviceMoEWeightFormat format) noexcept
    {
        return matrix.n > 0 && matrix.k > 0
                   ? static_cast<std::uint64_t>(matrix.n) *
                         static_cast<std::uint64_t>(matrix.k) *
                         deviceMoEFloatingElementBytes(format)
                   : 0u;
    }

    /**
     * @brief Authenticate a projection-set tag and the absence of fixed-bank pointers.
     * @param expert Host or device view of the same compact descriptor ABI.
     * @return False for unknown sets or any down storage in a movable pair.
     */
    template <typename ExpertDescriptor>
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool
    deviceMoEProjectionPayloadValid(const ExpertDescriptor &expert) noexcept
    {
        if (expert.projection_set == DeviceMoEProjectionSet::CompleteExpert)
            return true;
        return expert.projection_set == DeviceMoEProjectionSet::GateUp &&
               expert.down.payload == nullptr && expert.down.scales == nullptr &&
               expert.down.mins == nullptr && expert.down.emins == nullptr &&
               expert.down.n == 0 && expert.down.k == 0 && expert.down.blocks_per_row == 0 &&
               expert.floating_down.data == nullptr &&
               expert.floating_down.n == 0 && expert.floating_down.k == 0;
    }

    /**
     * @brief Prove arithmetic readiness for the exact graph-declared payload.
     * @param expert Immutable host/device placement-bank descriptor.
     * @param expected_projections Payload required by this captured consumer.
     * @return False for a different family, incomplete matrices, or a fixed down
     * pointer incorrectly embedded in a movable gate/up pair.
     *
     * Routing filters and grouped descriptor publication must use the same
     * predicate. Otherwise routing can discard a valid pair before its producer
     * sees it, or complete-FFN execution can silently accept a missing down.
     * Residency masks and local slots remain the placement bank's authority;
     * this function only authenticates the immutable projection payload.
     */
    template <typename ExpertDescriptor>
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool
    deviceMoEExpertProjectionReady(
        const ExpertDescriptor &expert, DeviceMoEProjectionSet expected_projections) noexcept
    {
        if (expert.logical_expert_id < 0 || expert.projection_set != expected_projections ||
            !deviceMoEProjectionPayloadValid(expert))
            return false;
        const bool needs_down = expected_projections == DeviceMoEProjectionSet::CompleteExpert;
        if (deviceMoEWeightFormatIsFloating(expert.weight_format))
            return expert.floating_gate.valid() && expert.floating_up.valid() &&
                (!needs_down || expert.floating_down.valid());
        if (expert.weight_format != DeviceMoEWeightFormat::NativeVNNI)
            return false;
        // The compact CUDA/HIP views and the host descriptor share these ABI
        // fields. Do not depend on a host-only descriptor member function.
        return expert.gate.payload && expert.gate.scales && expert.gate.n > 0 &&
            expert.gate.k > 0 && expert.gate.blocks_per_row > 0 &&
            expert.up.payload && expert.up.scales && expert.up.n > 0 &&
            expert.up.k > 0 && expert.up.blocks_per_row > 0 &&
            (!needs_down || (expert.down.payload && expert.down.scales &&
                expert.down.n > 0 && expert.down.k > 0 && expert.down.blocks_per_row > 0));
    }

    /**
     * @return Whether all movable floating projections exist, without accepting
     * an unknown contract or a down pointer smuggled into a gate/up-only payload.
     */
    template <typename ExpertDescriptor>
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool
    deviceMoEFloatingExpertCopyReady(const ExpertDescriptor &expert) noexcept
    {
        return deviceMoEProjectionPayloadValid(expert) &&
               deviceMoEWeightFormatIsFloating(expert.weight_format) &&
               expert.floating_gate.valid() && expert.floating_up.valid() &&
               (expert.projection_set == DeviceMoEProjectionSet::GateUp ||
                expert.floating_down.valid());
    }

    /** @return Whether a raw projection fits the destination's immutable shape. */
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool
    deviceMoEFloatingMatrixSameShape(
        const DeviceMoEFloatingMatrixDesc &src,
        const DeviceMoEFloatingMatrixDesc &dst) noexcept
    {
        return src.valid() && dst.valid() && src.n == dst.n && src.k == dst.k;
    }

    /**
     * @return Whether an arrival fits raw storage independent of its last occupant.
     * A FP32 slot reused for FP16 must retain FP32 capacity. Active arithmetic
     * format is therefore never used as the destination capacity authority.
     */
    template <typename ExpertDescriptor>
    [[nodiscard]] LLAMINAR_MOE_HOST_DEVICE constexpr bool
    deviceMoEFloatingExpertFitsTransferCapacity(
        const ExpertDescriptor &src, const ExpertDescriptor &dst) noexcept
    {
        return src.projection_set == dst.projection_set &&
               deviceMoEProjectionPayloadValid(dst) &&
               deviceMoEFloatingExpertCopyReady(src) &&
               deviceMoEFloatingElementBytes(src.weight_format) <=
                   deviceMoEFloatingElementBytes(dst.floating_allocation_format) &&
               deviceMoEFloatingMatrixSameShape(src.floating_gate, dst.floating_gate) &&
               deviceMoEFloatingMatrixSameShape(src.floating_up, dst.floating_up) &&
               (src.projection_set == DeviceMoEProjectionSet::GateUp ||
                deviceMoEFloatingMatrixSameShape(src.floating_down, dst.floating_down));
    }

    static_assert(
        std::is_standard_layout_v<DeviceMoEFloatingMatrixDesc>,
        "floating MoE descriptors must remain a stable standard-layout ABI");
    static_assert(
        std::is_trivially_copyable_v<DeviceMoEFloatingMatrixDesc>,
        "floating MoE descriptors must remain directly uploadable");
    static_assert(
        sizeof(DeviceMoEFloatingMatrixDesc) == 16,
        "floating MoE descriptors must retain the backend ABI size");

#undef LLAMINAR_MOE_HOST_DEVICE
} // namespace llaminar2
