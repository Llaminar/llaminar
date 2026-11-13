/**
 * @file MoEGroupedDirectory.h
 * @brief Shared packed work-directory geometry for routed and shared experts.
 *
 * Runtime routed experts stop at 512, but a fused projection descriptor table
 * additionally owns the separate shared expert. Ten low bits preserve that
 * descriptor's identity without changing the uint32 arena/work-item format.
 * The remaining bits bound row offsets; launch admission rejects overflow.
 */
#pragma once
#include "../../execution/moe/DeviceMoERuntimeABI.h"
#include <cstdint>

namespace llaminar2::moe_grouped_directory
{
    /** Ten bits admit the 512 routed descriptors plus their shared expert. */
    inline constexpr std::uint32_t kExpertBits = 10u;
    /** Maximum power-of-two descriptor domain used by the integer prefix scan. */
    inline constexpr std::uint32_t kMaximumExperts = 1u << kExpertBits;
    /** Mask separating physical descriptor identity from its logical row offset. */
    inline constexpr std::uint32_t kExpertMask = kMaximumExperts - 1u;
    /** Largest admitted route count; also prevents the unused-entry sentinel. */
    inline constexpr std::uint32_t kMaximumRows = UINT32_MAX >> kExpertBits;
    static_assert(kMaximumExperts > moe_runtime_abi::kMaxExperts);

    /** @brief Retain the existing directory launch width until the shared expert requires more. */
    [[nodiscard]] constexpr std::uint32_t directoryThreads(std::uint32_t experts) noexcept
    {
        return experts == 0u || experts > kMaximumExperts ? 0u
             : experts <= 512u ? 512u : 1024u;
    }
}
