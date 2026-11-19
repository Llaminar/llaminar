/**
 * @file DeviceMoEExpertFlags.h
 * @brief One shared host/device descriptor-role contract for runtime publication.
 *
 * Fragment residency is distinct from whole-expert replication and LLEP slots.
 * Producers and captured consumers use these same bits; no backend-private
 * interpretation may grant a fragment a scalar owner or a replica role.
 */
#pragma once
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_FLAGS_HD __host__ __device__
#else
#define LLAMINAR_MOE_FLAGS_HD
#endif
namespace llaminar2
{
    /** @brief Physical publication and compute-role bits of one expert descriptor. */
    enum class DeviceMoEExpertFlags : uint32_t
    {
        None = 0,
        Valid = 1u << 0,
        Resident = 1u << 1,
        Replicated = 1u << 2,
        PreferredOwner = 1u << 3,
        LocalCompute = 1u << 4,
        TransferSlot = 1u << 5,
    };

    /** @return Union of explicit descriptor flags. */
    LLAMINAR_MOE_FLAGS_HD constexpr DeviceMoEExpertFlags operator|(DeviceMoEExpertFlags lhs, DeviceMoEExpertFlags rhs) noexcept
    {
        return static_cast<DeviceMoEExpertFlags>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
    }

    /** @return Intersection of explicit descriptor flags. */
    LLAMINAR_MOE_FLAGS_HD constexpr DeviceMoEExpertFlags operator&(DeviceMoEExpertFlags lhs, DeviceMoEExpertFlags rhs) noexcept
    {
        return static_cast<DeviceMoEExpertFlags>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
    }

    /** @return The flags after adding an explicit descriptor role. */
    LLAMINAR_MOE_FLAGS_HD constexpr DeviceMoEExpertFlags &operator|=(DeviceMoEExpertFlags &lhs, DeviceMoEExpertFlags rhs) noexcept
    {
        lhs = lhs | rhs;
        return lhs;
    }

    /** @return Exact shared ABI word for these descriptor flags. */
    LLAMINAR_MOE_FLAGS_HD constexpr uint32_t toMoEExpertFlags(DeviceMoEExpertFlags flags) noexcept
    {
        return static_cast<uint32_t>(flags);
    }

    /** @return Whether a shared ABI word contains the requested flag. */
    LLAMINAR_MOE_FLAGS_HD constexpr bool hasMoEExpertFlag(uint32_t flags, DeviceMoEExpertFlags flag) noexcept
    {
        return (flags & static_cast<uint32_t>(flag)) != 0;
    }

    /** @param flags Published descriptor word.
     * @return Whether it names exactly one ready local fragment, without replica/LLEP roles. */
    LLAMINAR_MOE_FLAGS_HD constexpr bool deviceMoEIntermediateShardFlagsValid(uint32_t flags) noexcept
    {
        return flags == toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident |
                                        DeviceMoEExpertFlags::LocalCompute);
    }
} // namespace llaminar2
#undef LLAMINAR_MOE_FLAGS_HD
