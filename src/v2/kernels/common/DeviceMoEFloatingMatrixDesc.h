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
     * A layer may use any supported NativeVNNI codebook or one contiguous
     * floating-point format.  The value is stored once per expert because its
     * gate, up, and down projections must agree; accepting a mixed triple would
     * make migration byte counts and captured kernel dispatch ambiguous.
     */
    enum class DeviceMoEWeightFormat : std::uint32_t
    {
        NativeVNNI = 0,
        FP16 = 1,
        BF16 = 2,
        FP32 = 3,
    };

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
     * enclosing expert descriptor so all three projections are validated as a
     * single arithmetic family before publication.
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
