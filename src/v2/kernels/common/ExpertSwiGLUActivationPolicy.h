#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Model-owned SwiGLU operand policy, independent of weight storage. */
    class ExpertSwiGLUActivationPolicy
    {
    public:
        constexpr ExpertSwiGLUActivationPolicy() = default;

        /** @brief Decode model metadata: zero means ordinary SwiGLU. */
        static ExpertSwiGLUActivationPolicy fromOperandLimit(float limit)
        {
            if (!std::isfinite(limit) || limit < 0.0f)
                throw std::invalid_argument("Expert SwiGLU operand limit must be finite and nonnegative");
            ExpertSwiGLUActivationPolicy policy;
            policy.limit_ = limit == 0.0f ? 0.0f : limit;
            return policy;
        }

        [[nodiscard]] constexpr bool clampsOperands() const noexcept { return limit_ > 0.0f; }
        [[nodiscard]] constexpr float operandLimit() const noexcept { return limit_; }
        [[nodiscard]] constexpr std::uint32_t captureWord() const noexcept { return std::bit_cast<std::uint32_t>(limit_); }
        constexpr bool operator==(const ExpertSwiGLUActivationPolicy &) const = default;

    private:
        float limit_ = 0.0f;
    };
}
