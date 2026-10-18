/**
 * @file FloatingActivationOperandPrecision.h
 * @brief Immutable activation arithmetic identity of a prepared floating engine.
 *
 * Native preserves existing FP32-activation floating kernels. Explicit modes are
 * scoped by the original GDN/router source policy before preparation. They
 * never change routed experts or select another implementation after failure.
 */
#pragma once
namespace llaminar2
{
    /** @brief Operand contract retained with the engine and captured pointers. */
    enum class FloatingActivationOperandPrecision
    {
        Native, ///< Existing source-format implementation, unchanged by default.
        FP32, ///< Strict FP32 operands, accumulation and output.
        FP16RNE, ///< Half-RNE A/B values, FP32 accumulation and output.
    };
}
