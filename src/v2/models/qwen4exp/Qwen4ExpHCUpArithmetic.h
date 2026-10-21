/**
 * @file Qwen4ExpHCUpArithmetic.h
 * @brief Source-owned HC-up arithmetic shared by graph construction and admission.
 *
 * The setup selector controls the activation approximation of quantized weight
 * projections. Floating source weights already consume floating activations and
 * remain an explicit native floating operation. Selection inspects immutable
 * source metadata before planning/capture, never backend capability or runtime
 * failure. Every canonical native quantized format retains its original bytes.
 */
#pragma once
#include "kernels/common/QuantizedActivationOperandPrecision.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
/** @brief Three distinct source-owned operations; floating GEMM has no Q8 operand. */
enum class Qwen4ExpHCUpArithmetic
{
    FloatingNative,
    QuantizedSingleQ8,
    QuantizedTwoTermQ8,
};

/**
 * @brief Resolve the declared activation policy against the actual original source.
 * @param quantized_policy Frozen activation policy, used only by quantized sources.
 * @param source_format Original GGUF/prepared tensor format spelling.
 * @return The exact floating or quantized operation required by this source.
 * @throws std::invalid_argument For an unknown policy or unsupported weight format.
 */
inline Qwen4ExpHCUpArithmetic resolveQwen4ExpHCUpArithmetic(
    QuantizedActivationOperandPrecision quantized_policy,std::string_view source_format)
{
    // Validate the selector even for floating sources so invalid configuration
    // cannot hide behind a source category that does not consume Q8 operands.
    (void)quantizedActivationOperandPrecisionName(quantized_policy);
    if(source_format=="F32" || source_format=="FP32" ||
       source_format=="F16" || source_format=="FP16" || source_format=="BF16")
        return Qwen4ExpHCUpArithmetic::FloatingNative;
    if(!native_vnni_formats::forQuantType(source_format))
        throw std::invalid_argument("QFN HC-up requires native floating or cataloged native quantized source weights");
    return quantized_policy==QuantizedActivationOperandPrecision::TwoTermQ8
        ? Qwen4ExpHCUpArithmetic::QuantizedTwoTermQ8
        : Qwen4ExpHCUpArithmetic::QuantizedSingleQ8;
}
} // namespace llaminar2
