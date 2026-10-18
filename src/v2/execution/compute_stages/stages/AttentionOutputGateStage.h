/**
 * @file AttentionOutputGateStage.h
 * @brief Sigmoid-gated attention output with explicit packed live geometry.
 *
 * Computes: output = sigmoid(gate) * input
 *
 * Qwen full-attention participants may retain wider buffers for replicated
 * decode or GDN layers. The graph's live feature width, rather than those
 * allocation dimensions, controls both kernel work and diagnostic publication.
 * Live rows occupy one packed prefix; unused retained capacity is untouched.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    /**
     * @brief Sigmoid-gated attention output
     *
     * Computes element-wise: output[i] = sigmoid(gate[i]) * input[i]
     * where gate is a learned projection (GEMM output) and input is
     * the attention output before residual addition.
     */
    class AttentionOutputGateStage : public IComputeStage
    {
    public:
        /** @brief Participant-local packed geometry and borrowed arena tensors. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *input = nullptr;  ///< Packed attention input, with retained capacity >= live elements.
            ITensor *gate = nullptr;   ///< Packed gate values, with the same live geometry as input.
            ITensor *output = nullptr; ///< Packed gated output; may alias input.

            int seq_len = 0;     ///< Required positive live row count, independent of buffer capacity.
            int feature_dim = 0; ///< Required positive live columns from resolved graph geometry.

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> gate_buffer_id;
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Retain immutable graph geometry and borrowed tensor bindings.
         * @param params Explicit live rows/columns and participant-local tensors.
         * @throws std::invalid_argument For missing or unrepresentable live geometry.
         */
        explicit AttentionOutputGateStage(Params params);

        /**
         * @brief Gate exactly the packed live prefix on the stage's exact stream.
         * @param ctx Matching participant context; GPU coherence supplies its exact stream.
         * @return Whether the backend successfully submitted or completed the operation.
         * @throws std::invalid_argument If a tensor cannot hold the live packed prefix.
         */
        bool execute(IDeviceContext *ctx) override;
        /** @brief Return the semantic compute-stage kind. */
        ComputeStageType type() const override { return ComputeStageType::ATTENTION_OUTPUT_GATE; }
        /** @brief Estimate arithmetic for live values only. */
        size_t estimatedFlops() const override;
        /** @brief Estimate two live reads and one live write. */
        size_t estimatedMemoryBytes() const override;
        /**
         * @brief Report whether the requested compiled backend implements this stage.
         * @param backend Backend selected by the existing graph placement.
         * @return True for CPU and each compiled native GPU implementation.
         */
        bool supportsBackend(ComputeBackendType backend) const override;
        /**
         * @brief Publish the same live rows/columns consumed by execution.
         * @return Borrowed tensor descriptors whose byte extent excludes spare capacity.
         * @throws std::invalid_argument If a tensor cannot hold the live packed prefix.
         */
        StageDumpInfo buildDumpInfoImpl() const override;
        /** @brief Return empty scratch requirements; all tensors are borrowed. */
        StageBufferRequirements getBufferRequirements() const override;
        /** @brief Return the graph-owned input/output coherence bindings. */
        StageBufferContract bufferContract() const override;

        /** @brief Inspect immutable parameters without transferring ownership. */
        const Params &getParams() const { return params_; }

    private:
        /**
         * @brief Resolve the single checked live count for kernels and diagnostics.
         * @return Positive packed element count representable by the native launch ABI.
         * @throws std::invalid_argument For nonpositive geometry or an overflowing count.
         */
        size_t liveElementCount() const;

        /**
         * @brief Reject a borrowed tensor smaller than the declared live prefix.
         * @param tensor Prepared tensor whose capacity is checked without materialization.
         * @param role Name included in the fatal configuration diagnostic.
         * @throws std::invalid_argument If the declared live extent exceeds capacity.
         */
        void validateLiveCapacity(const TensorBase &tensor, const char *role) const;

        Params params_;
    };

} // namespace llaminar2
