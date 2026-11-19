/**
 * @file MoEDecodeDescriptorPublication.h
 * @brief Borrowed graph-local outputs for fused route/descriptor publication.
 *
 * These addresses name the existing arena tables, not another placement cache.
 * The routing lane authenticates the acquired placement bank and owns one
 * unique selected expert. It can publish that expert's compact descriptors
 * without acquiring and validating the same bank in a second kernel. Output
 * lifetime, geometry and precision remain part of the captured graph identity.
 */
#pragma once

#include "DeviceNativeQuantisedMatrixDesc.h"
#include "DeviceMoEFloatingMatrixDesc.h"

namespace llaminar2
{
    /** @brief Cold contract for reads of the selected expert's materialized role tables. */
    enum class MoESelectedDescriptorDemand : std::uint32_t
    {
        ExternalTables, ///< A declared reader consumes selected role tables after publication.
        AcquiredBankOnly ///< The region consumes expert descriptors directly while it holds the bank.
    };

    /** @brief Complete borrowed descriptor destinations for one retained layer graph. */
    struct MoEDecodeDescriptorPublication
    {
        DeviceNativeQuantisedMatrixDesc *gate = nullptr; ///< Original arena gate storage.
        DeviceNativeQuantisedMatrixDesc *up = nullptr; ///< Original arena up storage.
        DeviceNativeQuantisedMatrixDesc *down = nullptr; ///< Original arena down storage.
        int expert_count = 0; ///< Exact expert-indexed table extent.
        DeviceMoEWeightFormat format = DeviceMoEWeightFormat::NativeQuantised;
        DeviceMoEProjectionSet projections = DeviceMoEProjectionSet::CompleteExpert;

        /** @return Whether every output and its original descriptor family is bound. */
        [[nodiscard]] bool valid() const noexcept
        {
            return gate && up && down && expert_count > 0 &&
                (format == DeviceMoEWeightFormat::NativeQuantised ||
                 deviceMoEWeightFormatIsFloating(format)) &&
                deviceMoEProjectionSetValid(projections);
        }
    };
    static_assert(std::is_trivially_copyable_v<MoEDecodeDescriptorPublication>);

#if defined(__CUDACC__) || defined(__HIPCC__)
    /**
     * @brief Publish one already-authenticated local slot without repeating its epoch walk.
     * @tparam Descriptor Original host-ABI-equivalent CUDA/HIP expert descriptor view.
     * @param output Complete arena destinations fixed before graph capture.
     * @param descriptor Source in the same acquired bank that supplied this route.
     * @param expert Unique selected local expert ID; inactive/remote slots do not call.
     * @param ready Existing descriptor publication readiness predicate.
     *
     * The caller retains the original readiness and format predicates. Native
     * descriptor copies preserve every codebook, source-identity and padding
     * byte. Floating tables retain their existing compact stride inside the
     * admitted native-sized arena slots. No model arithmetic occurs here.
     */
    template <typename Descriptor>
    __device__ inline __attribute__((always_inline)) void publishMoEDecodeDescriptor(
        const MoEDecodeDescriptorPublication &output,
        const Descriptor &descriptor, int expert, bool ready)
    {
        if (output.format == DeviceMoEWeightFormat::NativeQuantised)
        {
            static_assert(sizeof(descriptor.gate) == sizeof(DeviceNativeQuantisedMatrixDesc));
            if (ready)
            {
                __builtin_memcpy(output.gate + expert, &descriptor.gate, sizeof(descriptor.gate));
                __builtin_memcpy(output.up + expert, &descriptor.up, sizeof(descriptor.up));
                __builtin_memcpy(output.down + expert, &descriptor.down, sizeof(descriptor.down));
            }
            else
            {
                output.gate[expert] = {};
                output.up[expert] = {};
                output.down[expert] = {};
            }
        }
        else
        {
            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(output.gate)[expert] =
                ready ? descriptor.floating_gate : DeviceMoEFloatingMatrixDesc{};
            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(output.up)[expert] =
                ready ? descriptor.floating_up : DeviceMoEFloatingMatrixDesc{};
            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(output.down)[expert] =
                ready ? descriptor.floating_down : DeviceMoEFloatingMatrixDesc{};
        }
    }
#endif
} // namespace llaminar2
