/**
 * @file MoERouterArithmeticPolicy.h
 * @brief Immutable source-preparation policy for MoE router dot products.
 *
 * FP16 prepares weights from the original decoder and rounds hidden operands
 * to half, retaining FP32 accumulation, softmax and route weights. Expert GEMM
 * activation quantization consumes the original hidden rows independently.
 * Native retains the existing source-format router controls for comparisons.
 * Omitted startup selection resolves to FP16 only for the registered QFN schema;
 * source planning, prepared operands and persistent identity share that resolver.
 * NativeHadamard32 pairs a fixed signed rotation of floating source weights
 * and hidden rows before existing Q8 conversions. Floating router paths remain
 * native (including CPU and default CUDA prefill); explicit CUDA Q8 prefill
 * uses the paired basis in both phases. Experts never rotate.
 */
#pragma once

#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    /** @brief Startup selection; ModelDefault must resolve before serialization/materialization. */
    enum class MoERouterArithmetic { Native, FP16, NativeHadamard32, ModelDefault };

    /**
     * @brief Resolve an omitted startup choice from the registered source schema.
     * @param requested Explicit arithmetic or the setup-only ModelDefault selection.
     * @param architecture Canonical GGUF architecture published by the loader/schema.
     * @return Concrete source/operand policy; never ModelDefault.
     * @throws std::invalid_argument For an invalid requested enum.
     *
     * qwen4exp is the registered QFN family, not a filename or display-name
     * heuristic. Every backend consumes this same resolved source policy. An
     * explicit Native selection remains native, including for QFN diagnostics.
     */
    constexpr MoERouterArithmetic resolveMoERouterArithmetic(
        MoERouterArithmetic requested, std::string_view architecture)
    {
        switch (requested)
        {
        case MoERouterArithmetic::ModelDefault:
            return architecture == "qwen4exp" ? MoERouterArithmetic::FP16 : MoERouterArithmetic::Native;
        case MoERouterArithmetic::Native:
        case MoERouterArithmetic::FP16:
        case MoERouterArithmetic::NativeHadamard32:
            return requested;
        }
        throw std::invalid_argument("Invalid MoE router arithmetic selection");
    }

    /** @brief Explicit CUDA prefill arithmetic, independent of decode's Q8 switch. */
    enum class CUDARouterPrefillArithmetic { FP32, Q8 };

    /** @brief Parse an immutable CUDA prefill operand policy.
     * @param value Exact fp32 or q8 spelling. @return Requested prefill arithmetic.
     * @throws std::invalid_argument For unknown selections; no implicit substitution. */
    inline CUDARouterPrefillArithmetic parseCUDARouterPrefillArithmetic(std::string_view value)
    {
        if (value == "fp32") return CUDARouterPrefillArithmetic::FP32;
        if (value == "q8") return CUDARouterPrefillArithmetic::Q8;
        throw std::invalid_argument("LLAMINAR_CUDA_MOE_ROUTER_PREFILL requires fp32 or q8");
    }

    /** @brief Serialize the exact prefill arithmetic into persistent-state identity.
     * @param value Frozen policy. @return Stable name.
     * @throws std::invalid_argument For an invalid enum. */
    constexpr std::string_view cudaRouterPrefillArithmeticName(CUDARouterPrefillArithmetic value)
    {
        switch (value)
        {
        case CUDARouterPrefillArithmetic::FP32: return "fp32";
        case CUDARouterPrefillArithmetic::Q8: return "q8";
        }
        throw std::invalid_argument("Invalid CUDA router prefill arithmetic");
    }

    /** @brief Coordinate system of an already prepared Q8 router matrix. */
    enum class RouterQ8Basis : unsigned { Original, SignedHadamard32 };

    /** @brief Select the basis only for paths that already execute Q8 routing.
     * @param policy Startup selection; only explicit NativeHadamard32 rotates.
     * @return Paired input/weight basis, independent of source operand precision.
     *
     * ModelDefault, Native and FP16 all retain original coordinates. This helper
     * does not resolve arithmetic: the source contract chooses the prepared
     * tensor format, and native routing dispatch consumes that actual format. */
    constexpr RouterQ8Basis routerQ8Basis(MoERouterArithmetic policy) noexcept
    {
        return policy == MoERouterArithmetic::NativeHadamard32 ?
            RouterQ8Basis::SignedHadamard32 : RouterQ8Basis::Original;
    }

    /** @brief Stable wire/fingerprint name; reject an invalid enum.
     * @param policy Frozen startup policy. @return Its exact parser spelling.
     * @throws std::invalid_argument For an unresolved default or unsupported policy. */
    constexpr std::string_view moeRouterArithmeticName(MoERouterArithmetic policy)
    {
        switch (policy)
        {
        case MoERouterArithmetic::Native: return "native";
        case MoERouterArithmetic::FP16: return "fp16";
        case MoERouterArithmetic::NativeHadamard32: return "native_hadamard32";
        case MoERouterArithmetic::ModelDefault: break;
        }
        throw std::invalid_argument("Invalid MoE router arithmetic");
    }

    /**
     * @brief Parse the exact startup arithmetic choice without implicit aliases.
     * @param value Exact native, fp16 or native_hadamard32 selection.
     * @return The requested source-preparation and activation arithmetic.
     * @throws std::invalid_argument For an unsupported or empty value.
     */
    inline MoERouterArithmetic parseMoERouterArithmetic(std::string_view value)
    {
        if (value == "native") return MoERouterArithmetic::Native;
        if (value == "fp16") return MoERouterArithmetic::FP16;
        if (value == "native_hadamard32") return MoERouterArithmetic::NativeHadamard32;
        throw std::invalid_argument("LLAMINAR_MOE_ROUTER_ARITHMETIC requires native, fp16 or native_hadamard32");
    }
}
