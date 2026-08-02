/**
 * @file MTPStochasticSerialOutcomeStage.h
 * @brief Captured seeded stochastic MTP sampling and compact outcome reduction.
 *
 * A grouped verifier leaves compact target distributions in persistent device
 * rows.  Seeded MTP must sample those rows exactly as serial decode would, then
 * compare the sampled target sequence byte-for-byte with the materialized draft
 * sequence.  This stage owns that complete terminal transaction for one fixed
 * graph geometry.
 *
 * Every replay-varying value is read from device storage: logical positions,
 * verifier input tokens, stop controls, generation budgets, and compact output
 * rows.  Request seeds and launch geometry are immutable capture identity.  The
 * stage performs no allocation, transfer, callback, event creation, or stream
 * synchronization and rejects a null graph stream.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    class IBackend;

    /**
     * @brief Sample and reduce seeded verifier rows on an explicit graph stream.
     *
     * Request rows use uniform capture geometry.  That is a deliberate graph
     * contract: request count, verifier depth, and every stride select a native
     * graph executable, while the contents at those addresses remain replay
     * inputs.  Dynamic depth therefore selects a pre-materialized graph family
     * member instead of mutating kernel arguments from the host.
     */
    class MTPStochasticSerialOutcomeStage final : public IComputeStage
    {
    public:
        /**
         * @brief Immutable pointer bindings and launch geometry.
         *
         * The compact target matrix stores `comparison_rows_per_request + 1`
         * rows per request; the final row is the bonus sample.  Target rows use
         * @ref target_distribution_row_stride entries.  Other request matrices
         * use their explicit request strides so no hidden packed-layout
         * assumption can enter graph capture.
         */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IBackend *backend = nullptr;

            const int32_t *target_token_ids_device = nullptr;
            const float *target_probs_device = nullptr;
            int target_distribution_row_stride = 0;
            int top_k = 0;

            const int32_t *verifier_input_tokens_device = nullptr;
            int verifier_input_token_stride = 0;
            const int32_t *stop_tokens_device = nullptr;
            int stop_token_stride = 0; ///< Zero shares one admitted stop row across requests.

            const int32_t *threshold_base_positions_device = nullptr;
            int threshold_position_offset = 1;
            std::vector<uint64_t> threshold_seeds;

            int *generation_control_device = nullptr;
            int generation_control_stride = 0;
            const uint32_t *maintenance_rows_remaining_device = nullptr;
            int verifier_row_capacity = 0;

            int32_t *sampled_target_tokens_device = nullptr;
            int sampled_target_token_stride = 0;
            int32_t *output_tokens_device = nullptr;
            int output_token_stride = 0;
            int *output_meta_device = nullptr;
            int output_meta_stride = 0;

            int request_count = 0;
            int comparison_rows_per_request = 0;

            bool advance_maintenance_boundary = false;
            uint32_t *decode_rounds_committed_device = nullptr;
            uint32_t *decode_rounds_until_maintenance_device = nullptr;
            uint32_t *maintenance_due_device = nullptr;
            uint32_t *decode_boundary_advanced_device = nullptr;

            std::string stage_name =
                "mtp_stochastic_serial_outcome";
        };

        static_assert(StageParamsRequired<Params>);

        explicit MTPStochasticSerialOutcomeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MTP_STOCHASTIC_SERIAL_OUTCOME;
        }
        std::string name() const override { return params_.stage_name; }
        size_t estimatedFlops() const override { return 0; }
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
         * Device contents are excluded because they are replay inputs.  Seeds
         * are included because each fused kernel node receives its request seed
         * by value.
         */
        [[nodiscard]] bool hasSameCaptureIdentity(
            const Params &other) const noexcept;

        const Params &getParams() const noexcept { return params_; }

    private:
        [[nodiscard]] bool validate() const;

        Params params_;
    };

} // namespace llaminar2
