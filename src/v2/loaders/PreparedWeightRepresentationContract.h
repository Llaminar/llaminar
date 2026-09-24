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
#include "tensors/TensorType.h"

#include <string_view>

namespace llaminar2
{
    /** @brief Physical scalar representation materialized for graph execution. */
    enum class ModelPreparedWeightRepresentation
    {
        SourceNative, ///< Retain the source tensor's scalar/codebook format.
        FP32,         ///< Materialize a model-owned FP32 tensor before preparation.
    };

    /**
     * @brief Resolve the authoritative runtime representation of model weights.
     *
     * The role is semantic and therefore stable across model names.  In
     * particular, GDN alpha/beta has its own role because only those tiny
     * projections use the deterministic Q8-to-FP32 kernel path; the larger GDN
     * projection matrices retain their native codebooks.
     */
    class PreparedWeightRepresentationContract
    {
    public:
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
