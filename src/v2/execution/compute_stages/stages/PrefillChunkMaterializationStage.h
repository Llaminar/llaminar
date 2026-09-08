/**
 * @file PrefillChunkMaterializationStage.h
 * @brief Captured device-only publication of one long-prefill request window.
 *
 * Long prompts are admitted once into a complete arena-owned token/position
 * bank and then consumed by repeated fixed-width GPU graph launches. This stage
 * is the sole producer of the stable bucket view read by embedding, RoPE,
 * attention, recurrent state, and graph-integrated shifted-MTP prefill. Its
 * source offset comes from the canonical device KV count, so no host slice,
 * host cursor, replay callback, or duplicate device cursor participates in
 * execution.
 *
 * Every pointer and scalar in Params is immutable graph identity. Mutable
 * request length, KV progress, token values, and absolute positions stay behind
 * those device addresses. Execution is one allocation-free, transfer-free,
 * synchronization-free backend launch on the exact graph stream.
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
     * @brief Materialize a fixed bucket from one admitted single-request bank.
     *
     * The stage intentionally supports GPU backends only. CPU long-context
     * execution is host-owned and uses the ordinary CPU chunk scheduler rather
     * than pretending that a device publication contract applies there.
     */
    class PrefillChunkMaterializationStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph bindings and fixed launch geometry. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IBackend *backend = nullptr;
            const int32_t *request_token_ids_device = nullptr;
            const int32_t *request_position_ids_device = nullptr;
            const int32_t *request_total_rows_device = nullptr;
            const int32_t *cached_tokens_device = nullptr;
            int32_t *chunk_token_ids_device = nullptr;
            int32_t *chunk_position_ids_device = nullptr;
            int32_t *chunk_real_rows_device = nullptr;
            int32_t *chunk_row_stride_device = nullptr;
            int request_row_capacity = 0;
            int bucket_seq_len = 0;
            int pad_token_id = 0;
            uint64_t capture_identity = 0;
            std::string stage_name = "prefill_chunk_materialization";
        };

        static_assert(StageParamsRequired<Params>);

        explicit PrefillChunkMaterializationStage(Params params);

        /** @brief Enqueue the captured device publication on the exact stage stream. */
        bool execute(IDeviceContext *ctx) override;

        ComputeStageType type() const override
        {
            return ComputeStageType::PREFILL_CHUNK_MATERIALIZATION;
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

        /** @return immutable parameters retained by this captured stage. */
        [[nodiscard]] const Params &getParams() const noexcept
        {
            return params_;
        }

    private:
        /** @return true when every address and geometry invariant is complete. */
        [[nodiscard]] bool validate() const;

        Params params_;
    };
} // namespace llaminar2
