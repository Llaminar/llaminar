/**
 * @file MTPStochasticTargetDistributionStage.h
 * @brief Captured target-verifier penalty and compact-distribution preparation.
 *
 * Seeded stochastic MTP consumes one compact target distribution for every
 * verifier row, including the bonus row.  The source logits, durable token
 * histogram, verifier-token prefix, compact outputs, and top-k scratch all
 * live at persistent device addresses.  This stage freezes only launch
 * geometry and sampling policy so the complete transformation can be cloned
 * into a device-controlled generation loop.
 *
 * Execution is deliberately device-only.  The stage performs no allocation,
 * transfer, event creation, callback, or synchronization.  It requires the
 * exact non-null graph stream supplied by the executor and enqueues the
 * penalty transform before compact distribution construction on that stream.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace llaminar2
{
    class IBackend;

    /**
     * @brief Prepare serial-equivalent stochastic target rows for one geometry.
     *
     * Penalties are optional, but when enabled they are applied to every row in
     * place before top-k/top-p compaction.  The verifier input row supplies the
     * row-local speculative prefix while the generated-token histogram supplies
     * durable request history.  This is the same mathematical input used by
     * serial decode; capture changes launch ownership, not sampling semantics.
     */
    class MTPStochasticTargetDistributionStage final : public IComputeStage
    {
    public:
        /**
         * @brief Immutable bindings and launch policy retained by graph nodes.
         *
         * Every pointer names model- or arena-lifetime device memory. Sampling
         * geometry is capture identity. Penalty values and their evolving
         * history predicate live behind `penalty_policy_device` and remain replay
         * data, so request policy cannot multiply graph families.
         */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IBackend *backend = nullptr;

            float *logits_device = nullptr;
            int first_logit_row = 0;
            int row_count = 0;
            int vocab_size = 0;
            int logits_row_stride = 0;

            /**
             * Device-owned logical prefix of the physical row allocation.
             *
             * A null pointer means every captured row is active. Device-loop
             * graphs bind the verifier request-length scalar here so one
             * maximum-capacity capture can skip its inactive suffix without a
             * host read or changing launch geometry.
             */
            const int32_t *active_rows_device = nullptr;

            bool apply_penalties = false;
            const int32_t *verifier_input_tokens_device = nullptr;
            const int32_t *generated_token_counts_device = nullptr;
            void *penalty_policy_device = nullptr;

            int top_k = 0;
            float top_p = 1.0F;
            float temperature = 1.0F;

            int32_t *target_token_ids_device = nullptr;
            float *target_probs_device = nullptr;
            int first_target_slot = 0;
            int target_row_stride = 0;

            float *topk_partial_values_device = nullptr;
            int32_t *topk_partial_indices_device = nullptr;
            size_t topk_partial_capacity = 0;

            std::string stage_name =
                "mtp_stochastic_target_distribution";
        };

        static_assert(StageParamsRequired<Params>);

        explicit MTPStochasticTargetDistributionStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MTP_STOCHASTIC_TARGET_DISTRIBUTION;
        }
        std::string name() const override { return params_.stage_name; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return true; }
        bool isCollectiveStage() const override { return false; }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferContract bufferContract() const override;

        /**
         * @brief Compare every address and scalar embedded in captured nodes.
         *
         * Device contents are replay inputs and are intentionally excluded.
         * Pointer identity, row geometry, scratch capacity, and sampling-policy
         * scalars must all match before an existing executable can be reused.
         */
        [[nodiscard]] bool hasSameCaptureIdentity(
            const Params &other) const noexcept;

        const Params &getParams() const noexcept { return params_; }

    private:
        [[nodiscard]] bool validate() const;

        Params params_;
    };

} // namespace llaminar2
