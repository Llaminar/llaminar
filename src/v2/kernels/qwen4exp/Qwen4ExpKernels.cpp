/**
 * @file Qwen4ExpKernels.cpp
 * @brief Device-free validation of authenticated PLE hash partitions.
 *
 * Constants are exact checkpoint integers; production never regenerates them
 * from seeds or sends them through floating point metadata conversion.
 */
#include "Qwen4ExpKernels.h"
#include "Qwen4ExpValidation.h"
#include <limits>

namespace llaminar2::qwen4exp
{
void NgramHashPolicy::validate() const
{
    detail::require(eosToken >= 0 && eosToken <= std::numeric_limits<std::int32_t>::max(), "Qwen4Exp hash EOS token must fit a nonnegative INT32 token ID");
    for (const auto multiplier : multipliers)
        detail::require(multiplier != 0, "Qwen4Exp hash multiplier must be nonzero");
    constexpr auto maxIndex = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    std::uint64_t previousEnd = 0;
    for (int head = 0; head < kNgramHeads; ++head)
    {
        detail::require(moduli[head] > 0 && moduli[head] <= maxIndex && offsets[head] <= maxIndex &&
                            moduli[head] - 1 <= maxIndex - offsets[head],
                        "Qwen4Exp hash partition must fit positive signed 64-bit row indexing");
        detail::require(offsets[head] >= previousEnd, "Qwen4Exp hash head partitions must be ordered and disjoint");
        previousEnd = offsets[head] + moduli[head];
    }
}
} // namespace llaminar2::qwen4exp
