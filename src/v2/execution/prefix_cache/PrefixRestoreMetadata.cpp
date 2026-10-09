/**
 * @file PrefixRestoreMetadata.cpp
 * @brief Derive restore diagnostics from immutable section declarations and tiers.
 *
 * A null payload pointer is normal during metadata-only lookup. Layout section
 * declarations survive RAM/disk selection and must remain the reporting source.
 * This code neither materializes archives nor observes GPU execution state.
 */
#include "execution/prefix_cache/PrefixRestoreMetadata.h"
#include "execution/prefix_cache/PrefixStorageBackend.h"
#include <stdexcept>

namespace llaminar2
{
    PrefixRestoreMetadata PrefixRestoreMetadata::fromBlocks(std::span<const PrefixBlockHandle> blocks)
    {
        PrefixRestoreMetadata result;
        for (const auto &block : blocks)
        {
            if (block.layout.includes_mtp_state) result.bits_ |= kMTP;
            if (block.has_hybrid_state) result.bits_ |= kHybrid;
            switch (block.tier)
            {
            case PrefixStorageTier::DeviceHot: result.bits_ |= kDevice; break;
            case PrefixStorageTier::Ram: result.bits_ |= kRam; break;
            case PrefixStorageTier::Disk: result.bits_ |= kDisk; break;
            default: throw std::invalid_argument("Unknown prefix restore source tier");
            }
        }
        return result;
    }

    const char *PrefixRestoreMetadata::storageTierName() const noexcept
    {
        switch (bits_ & kTiers)
        {
        case 0: return "none";
        case kDevice: return "device-hot";
        case kRam: return "ram";
        case kDisk: return "disk-hydrated";
        default: return "mixed";
        }
    }

    PrefixRestoreMetadata PrefixRestoreMetadata::fromWireBits(uint32_t bits)
    {
        if ((bits & ~(kMTP | kHybrid | kTiers)) != 0 ||
            ((bits & (kMTP | kHybrid)) != 0 && (bits & kTiers) == 0))
            throw std::invalid_argument("Invalid prefix restore metadata word");
        PrefixRestoreMetadata result;
        result.bits_ = bits;
        return result;
    }
}
