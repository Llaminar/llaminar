/**
 * @file QuantizedActivationOperandPrecision.h
 * @brief Immutable activation representation shared by graph setup and kernels.
 *
 * These modes preserve the original prepared weight codebook. TwoTermQ8 adds
 * a separately quantized activation residual; it never converts or duplicates
 * source weights. Selection is frozen before workspace planning and capture.
 */
#pragma once
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    /** @brief Projection activation policy; SingleQ8 retains existing arithmetic. */
    enum class QuantizedActivationOperandPrecision { SingleQ8, TwoTermQ8 };

    /** @brief QFN HC-up default only; generic GEMM and other projections retain their own policy. */
    inline constexpr auto kDefaultQwen4ExpHCUpActivation = QuantizedActivationOperandPrecision::TwoTermQ8;

    /**
     * @brief Serialize an activation policy for configuration and cache metadata.
     * @param precision Immutable graph-owned arithmetic selection.
     * @return The unique spelling for the supported policy.
     * @throws std::invalid_argument For an invalid enum value.
     */
    inline constexpr std::string_view quantizedActivationOperandPrecisionName(
        QuantizedActivationOperandPrecision precision)
    {
        switch (precision)
        {
        case QuantizedActivationOperandPrecision::SingleQ8: return "single_q8";
        case QuantizedActivationOperandPrecision::TwoTermQ8: return "two_term_q8";
        }
        throw std::invalid_argument("Invalid quantized activation operand precision");
    }

    /**
     * @brief Parse an explicit policy without aliases or automatic substitution.
     * @param value Exact configuration spelling; absence is handled by setup.
     * @return The requested activation representation.
     * @throws std::invalid_argument For empty or unknown values.
     */
    inline QuantizedActivationOperandPrecision parseQuantizedActivationOperandPrecision(
        std::string_view value)
    {
        if (value == "single_q8") return QuantizedActivationOperandPrecision::SingleQ8;
        if (value == "two_term_q8") return QuantizedActivationOperandPrecision::TwoTermQ8;
        throw std::invalid_argument("Quantized activation operand precision requires single_q8 or two_term_q8");
    }
}
