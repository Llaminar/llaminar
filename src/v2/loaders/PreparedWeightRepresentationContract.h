/**
 * @file PreparedWeightRepresentationContract.h
 * @brief Canonical source-to-runtime representation policy for model weights.
 *
 * Weight loading and physical-memory planning must agree on the bytes that a
 * graph actually consumes.  This contract is the single typed decision point
 * for model weights that intentionally replace their GGUF source format with a
 * derived runtime representation.  Callers may describe a live tensor with
 * `TensorType` or a serialized planning profile with its stable format label;
 * both paths resolve to the same representation.
 */

#pragma once

#include "WeightIdentity.h"
#include "config/GDNProjectionArithmeticPolicy.h"
#include "tensors/TensorType.h"
#include "kernels/FloatingActivationOperandPrecision.h"
#include "config/MoERouterArithmeticPolicy.h"

#include <string_view>

namespace llaminar2
{
    /** @brief Physical scalar representation materialized for graph execution. */
    enum class ModelPreparedWeightRepresentation
    {
        SourceNative, ///< Retain the source tensor's scalar/codebook format.
        FP32,         ///< Materialize a model-owned FP32 tensor before preparation.
        FP16,         ///< Materialize canonical half-RNE original-source values.
    };

    /**
     * @brief Resolve the authoritative runtime representation of model weights.
     *
     * The role is semantic and therefore stable across model names.  In
     * particular, GDN alpha/beta has its own role because only those tiny
     * projections use the deterministic Q8-to-FP32 kernel path; the larger GDN
     * projection matrices retain their native codebooks by default. The explicit
     * source-aware overloads may derive FP32 GDN or half-RNE GDN/router weights
     * before capture; planner and loader consume that identical decision.
     */
    class PreparedWeightRepresentationContract
    {
    public:
        /**
         * @brief Authenticate one projection in the registered Qwen35 GDN schema.
         * @param role Existing semantic source role, never inferred from shape.
         * @param name Exact original GGUF block tensor name.
         * @param architecture Registered GGUF model family owning those names.
         * @return True only for QKV, Z, alpha, beta or output GDN matrices.
         *
         * Qwen35 full-attention declares separate attn_q/k/v/attn_output names.
         * Its attn_qkv matrix belongs to the GDN declaration exclusively; a
         * generic FusedQKV role on another architecture is insufficient proof.
         */
        [[nodiscard]] static constexpr bool isGDNProjectionSource(
            WeightRole role, std::string_view name, std::string_view architecture) noexcept
        {
            if (architecture != "qwen35" && architecture != "qwen35moe" && architecture != "qwen4exp") return false;
            if (!name.starts_with("blk.")) return false;
            const auto end = name.find('.', 4);
            if (end == std::string_view::npos || end == 4) return false;
            for (std::size_t index = 4; index < end; ++index)
                if (name[index] < '0' || name[index] > '9') return false;
            const auto suffix = name.substr(end + 1);
            return (role == WeightRole::FusedQKV && suffix == "attn_qkv.weight") ||
                (role == WeightRole::GDNProjection &&
                    (suffix == "attn_gate.weight" || suffix == "ssm_out.weight" || suffix == "gdn_qkv.weight")) ||
                (role == WeightRole::GDNAlphaBetaProjection &&
                    (suffix == "ssm_alpha.weight" || suffix == "ssm_beta.weight"));
        }

        /**
         * @brief Resolve the setup policy for one authenticated source projection.
         * @param role Immutable semantic role assigned by the model schema.
         * @param name Exact original GGUF tensor name.
         * @param architecture Registered schema owning the source name.
         * @param requested Process setup selection, including selective policies.
         * @return Native, FP32 or FP16 for this source; never a selective mode.
         *
         * Storage, operand arithmetic, workspace admission and setup counters
         * consume this decision so a selective policy cannot change wide GEMMs.
         */
        [[nodiscard]] static constexpr GDNProjectionArithmetic effectiveGDNArithmetic(
            WeightRole role, std::string_view name, std::string_view architecture,
            GDNProjectionArithmetic requested) noexcept
        {
            if (!isGDNProjectionSource(role, name, architecture))
                return GDNProjectionArithmetic::Native;
            if (requested == GDNProjectionArithmetic::FP16MoEAlphaBeta)
                return (architecture == "qwen35moe" || architecture == "qwen4exp") && role == WeightRole::GDNAlphaBetaProjection ?
                    GDNProjectionArithmetic::FP16 : GDNProjectionArithmetic::Native;
            if (requested == GDNProjectionArithmetic::FP16AlphaBeta)
                return role == WeightRole::GDNAlphaBetaProjection ?
                    GDNProjectionArithmetic::FP16 : GDNProjectionArithmetic::Native;
            return requested;
        }

        /**
         * @brief Select source-aware typed storage before loading and planning.
         * @param role Existing immutable semantic source role.
         * @param source_type Original loaded scalar/codebook format.
         * @param name Exact canonical source tensor name.
         * @param architecture Registered model schema owning the source name.
         * @param arithmetic Explicit startup GDN-only arithmetic policy.
         * @param router_arithmetic Independent immutable router operand policy.
         * @return Selected floating derivative or unchanged canonical native representation.
         */
        [[nodiscard]] static constexpr ModelPreparedWeightRepresentation resolve(
            WeightRole role, TensorType source_type, std::string_view name,
            std::string_view architecture, GDNProjectionArithmetic arithmetic,
            MoERouterArithmetic router_arithmetic = MoERouterArithmetic::ModelDefault)
        {
            router_arithmetic = resolveMoERouterArithmetic(router_arithmetic, architecture);
            if (role == WeightRole::MoERouter && router_arithmetic == MoERouterArithmetic::NativeHadamard32)
            {
                // Rotation must precede the FIRST quantization. Dequantizing a
                // codebook here would conceal the explicitly forbidden second conversion.
                if (source_type != TensorType::FP32 && source_type != TensorType::FP16 && source_type != TensorType::BF16)
                    throw std::invalid_argument("Hadamard router requires original floating weights; requantization is forbidden");
                return source_type == TensorType::FP32 ? ModelPreparedWeightRepresentation::SourceNative :
                    ModelPreparedWeightRepresentation::FP32;
            }
            if (role == WeightRole::MoERouter && router_arithmetic == MoERouterArithmetic::FP16)
                return source_type == TensorType::FP16 ? ModelPreparedWeightRepresentation::SourceNative :
                    ModelPreparedWeightRepresentation::FP16;
            const auto effective = effectiveGDNArithmetic(role, name, architecture, arithmetic);
            if ((arithmetic == GDNProjectionArithmetic::FP16AlphaBeta ||
                 arithmetic == GDNProjectionArithmetic::FP16MoEAlphaBeta) &&
                effective == GDNProjectionArithmetic::FP16 &&
                source_type != TensorType::FP32 && source_type != TensorType::FP16 && source_type != TensorType::BF16)
                throw std::invalid_argument("FP16 GDN alpha/beta requires original floating weights; quantized sources must stay native");
            if (effective == GDNProjectionArithmetic::FP16)
                return source_type == TensorType::FP16 ? ModelPreparedWeightRepresentation::SourceNative :
                    ModelPreparedWeightRepresentation::FP16;
            if (effective == GDNProjectionArithmetic::FP32 && source_type != TensorType::FP32)
                return ModelPreparedWeightRepresentation::FP32;
            return resolve(role, source_type);
        }

        /**
         * @brief Apply identical projection storage policy to metadata-only planning.
         * @param role Existing immutable semantic source role.
         * @param source_format Original GGUF/profile format label.
         * @param name Exact canonical source tensor name.
         * @param architecture Registered model schema owning the source name.
         * @param arithmetic Explicit startup GDN-only arithmetic policy.
         * @param router_arithmetic Independent immutable router operand policy.
         * @return The same representation selected by the loaded-type overload.
         */
        [[nodiscard]] static constexpr ModelPreparedWeightRepresentation resolve(
            WeightRole role, std::string_view source_format, std::string_view name,
            std::string_view architecture, GDNProjectionArithmetic arithmetic,
            MoERouterArithmetic router_arithmetic = MoERouterArithmetic::ModelDefault)
        {
            router_arithmetic = resolveMoERouterArithmetic(router_arithmetic, architecture);
            if (role == WeightRole::MoERouter && router_arithmetic == MoERouterArithmetic::NativeHadamard32)
            {
                // Rotation must precede the FIRST quantization. Dequantizing a
                // codebook here would conceal the explicitly forbidden second conversion.
                if (source_format != "F32" && source_format != "FP32" && source_format != "F16" && source_format != "FP16" && source_format != "BF16")
                    throw std::invalid_argument("Hadamard router requires original floating weights; requantization is forbidden");
                return (source_format == "F32" || source_format == "FP32") ? ModelPreparedWeightRepresentation::SourceNative :
                    ModelPreparedWeightRepresentation::FP32;
            }
            if (role == WeightRole::MoERouter && router_arithmetic == MoERouterArithmetic::FP16)
                return source_format == "F16" || source_format == "FP16" ?
                    ModelPreparedWeightRepresentation::SourceNative : ModelPreparedWeightRepresentation::FP16;
            const auto effective = effectiveGDNArithmetic(role, name, architecture, arithmetic);
            if ((arithmetic == GDNProjectionArithmetic::FP16AlphaBeta ||
                 arithmetic == GDNProjectionArithmetic::FP16MoEAlphaBeta) &&
                effective == GDNProjectionArithmetic::FP16 &&
                source_format != "F32" && source_format != "FP32" &&
                source_format != "F16" && source_format != "FP16" && source_format != "BF16")
                throw std::invalid_argument("FP16 GDN alpha/beta requires original floating weights; quantized sources must stay native");
            if (effective == GDNProjectionArithmetic::FP16)
                return source_format == "F16" || source_format == "FP16" ?
                    ModelPreparedWeightRepresentation::SourceNative : ModelPreparedWeightRepresentation::FP16;
            if (effective == GDNProjectionArithmetic::FP32 &&
                source_format != "F32" && source_format != "FP32")
                return ModelPreparedWeightRepresentation::FP32;
            return resolve(role, source_format);
        }

        /**
         * @brief Bind activation arithmetic to the identical original-source scope.
         * @param role Immutable semantic role owning the prepared weight.
         * @param name Original canonical GGUF source name.
         * @param architecture Registered owner schema, never inferred from shape.
         * @param arithmetic Immutable process startup GDN selection.
         * @param router_arithmetic Independent immutable router operand policy.
         * @return Explicit scoped mode; Native for every unrelated engine.
         */
        [[nodiscard]] static constexpr FloatingActivationOperandPrecision operandPrecision(
            WeightRole role, std::string_view name, std::string_view architecture,
            GDNProjectionArithmetic arithmetic,
            MoERouterArithmetic router_arithmetic = MoERouterArithmetic::ModelDefault)
        {
            router_arithmetic = resolveMoERouterArithmetic(router_arithmetic, architecture);
            if (role == WeightRole::MoERouter && router_arithmetic == MoERouterArithmetic::FP16)
                return FloatingActivationOperandPrecision::FP16RNE;
            const auto effective = effectiveGDNArithmetic(role, name, architecture, arithmetic);
            if (effective == GDNProjectionArithmetic::FP32)
                return FloatingActivationOperandPrecision::FP32;
            if (effective == GDNProjectionArithmetic::FP16)
                return FloatingActivationOperandPrecision::FP16RNE;
            return FloatingActivationOperandPrecision::Native;
        }

        /**
         * @brief Resolve a loaded tensor's required runtime representation.
         * @param role Semantic use assigned by the frozen weight plan.
         * @param source_type Loaded tensor scalar/codebook type.
         * @return The one representation the materializer must publish.
         */
        [[nodiscard]] static constexpr ModelPreparedWeightRepresentation resolve(
            WeightRole role,
            TensorType source_type) noexcept
        {
            if (source_type == TensorType::FP32)
                return ModelPreparedWeightRepresentation::SourceNative;
            if (role == WeightRole::SharedExpertInputGate)
                return ModelPreparedWeightRepresentation::FP32;
            if (role == WeightRole::GDNAlphaBetaProjection &&
                source_type == TensorType::Q8_0)
            {
                return ModelPreparedWeightRepresentation::FP32;
            }
            return ModelPreparedWeightRepresentation::SourceNative;
        }

        /**
         * @brief Resolve a planning-profile format's runtime representation.
         * @param role Semantic use inferred by the canonical weight identity.
         * @param source_format Stable GGUF/profile label such as `F32`, `BF16`,
         *        `Q8_0`, or `IQ2_S`.
         * @return The representation whose bytes memory planning must admit.
         *
         * Only formats that affect this policy need special classification.
         * Every other label remains distinct for downstream native packing,
         * while a shared-expert input gate converts every non-FP32 source.
         */
        [[nodiscard]] static constexpr ModelPreparedWeightRepresentation resolve(
            WeightRole role,
            std::string_view source_format) noexcept
        {
            const bool source_is_fp32 =
                source_format == "F32" || source_format == "FP32";
            if (source_is_fp32)
                return ModelPreparedWeightRepresentation::SourceNative;
            if (role == WeightRole::SharedExpertInputGate)
                return ModelPreparedWeightRepresentation::FP32;
            if (role == WeightRole::GDNAlphaBetaProjection &&
                source_format == "Q8_0")
            {
                return ModelPreparedWeightRepresentation::FP32;
            }
            return ModelPreparedWeightRepresentation::SourceNative;
        }
    };
} // namespace llaminar2
