/**
 * @file GDNProjectionArithmeticPolicy.h
 * @brief Shared default and explicit setup arithmetic for GDN projections.
 *
 * Native retains every existing kernel. FP32 derives immutable FP32 operands
 * directly from the original GGUF decoder before graph preparation. It never
 * requantizes weights or changes expert, router, full-attention or head policy.
 * FP16 derives half-RNE weights from those same original decoded values; its
 * prepared engine uses half-RNE activations and FP32 accumulation. Serial
 * decode and verifier rows share a fixed reduction tree; ordinary GPU prefill
 * uses the admitted half operand bank and the backend's floating GEMM.
 * FP16AlphaBeta applies that half arithmetic only to original floating alpha
 * and beta weights; all wide GDN projections retain their native codebooks.
 * FP16MoEAlphaBeta limits that selection to qwen35moe; dense models stay native.
 */
#pragma once
#include <stdexcept>
#include <string_view>
namespace llaminar2
{
    /** @brief Immutable setup selection shared by every backend. */
    enum class GDNProjectionArithmetic { Native, FP32, FP16, FP16AlphaBeta, FP16MoEAlphaBeta };

    /** @brief Qualified default: half MoE alpha/beta operands, native dense and wide projections. */
    inline constexpr auto kDefaultGDNProjectionArithmetic = GDNProjectionArithmetic::FP16MoEAlphaBeta;

    /**
     * @brief Canonical policy identity shared by planning and persistent caches.
     * @param value Explicit immutable setup selection.
     * @return The unique environment and wire spelling.
     * @throws std::invalid_argument For an invalid enum value.
     */
    inline constexpr std::string_view gdnProjectionArithmeticName(GDNProjectionArithmetic value)
    {
        switch (value)
        {
        case GDNProjectionArithmetic::Native: return "native";
        case GDNProjectionArithmetic::FP32: return "fp32";
        case GDNProjectionArithmetic::FP16: return "fp16";
        case GDNProjectionArithmetic::FP16AlphaBeta: return "fp16_alpha_beta";
        case GDNProjectionArithmetic::FP16MoEAlphaBeta: return "fp16_moe_alpha_beta";
        }
        throw std::invalid_argument("Invalid GDN projection arithmetic");
    }

    /**
     * @brief Parse one explicit setup policy without aliases or fallback.
     * @param value Exact explicit environment value; absence uses the shared default.
     * @return The requested supported arithmetic mode.
     * @throws std::invalid_argument For every unknown or empty value.
     */
    inline GDNProjectionArithmetic parseGDNProjectionArithmetic(std::string_view value)
    {
        if (value == "native") return GDNProjectionArithmetic::Native;
        if (value == "fp32") return GDNProjectionArithmetic::FP32;
        if (value == "fp16") return GDNProjectionArithmetic::FP16;
        if (value == "fp16_alpha_beta") return GDNProjectionArithmetic::FP16AlphaBeta;
        if (value == "fp16_moe_alpha_beta") return GDNProjectionArithmetic::FP16MoEAlphaBeta;
        throw std::invalid_argument("LLAMINAR_GDN_PROJECTION_ARITHMETIC requires native, fp32, fp16, fp16_alpha_beta or fp16_moe_alpha_beta");
    }
}
