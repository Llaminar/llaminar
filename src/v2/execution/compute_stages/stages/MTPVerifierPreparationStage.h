/**
 * @file MTPVerifierPreparationStage.h
 * @brief Captured device-owned preparation for one grouped MTP verifier pass.
 *
 * A grouped verifier consumes several pieces of state that must describe the
 * same speculative transaction: its token matrix, absolute position matrix,
 * valid request widths, and the immutable pre-verifier KV metadata checkpoint
 * used by accepted-state publication. Historically those values were prepared
 * by independent orchestrator calls. That made their ordering implicit and
 * allowed a host-authored token row or stale cache-count shadow to enter an
 * otherwise device-resident replay.
 *
 * This stage makes the complete producer boundary one graph node. Every input
 * pointer names model- or arena-lifetime device memory, every copy is D2D on the
 * exact graph stream, and every KV checkpoint operation is captured. The stage
 * performs no allocation, H2D/D2H transfer, callback, event operation, or
 * synchronization. Its caller joins producer events before replay and publishes
 * one completion event after replay.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class IKVCache;

    /**
     * @brief Compose resident verifier inputs and capture their pre-mutation KV base.
     *
     * The immutable bindings in Params are graph identity. Mutable token values,
     * live KV counts, ring heads, and compact row metadata remain device replay
     * data behind those addresses. A graph may therefore be reused across many
     * decode transactions without observing or republishing any value on host.
     * A pipeline follower checkpoints only its local main cache: the terminal
     * participant alone assembles verifier tokens and controls logical width.
     */
    class MTPVerifierPreparationStage final : public IComputeStage
    {
    public:
        /** @brief Explicit preparation ownership; an unspecified role cannot execute. */
        enum class Authority : uint8_t
        {
            Unbound,
            VerifierInputOwner,
            PipelineFollower,
        };

        /**
         * @brief One request-major row in the verifier token matrix.
         *
         * Entry zero comes from an authoritative resident condition/target token.
         * The remaining entries are a contiguous slice of resident draft slots.
         * `destination_device` points at the first column of this request's padded
         * output row. Padding is zeroed whenever the valid width is smaller than
         * the graph width, preventing stale token ids from reaching embedding.
         */
        struct TokenRowBinding
        {
            const int32_t *first_token_device = nullptr;
            const int32_t *draft_tokens_device = nullptr;
            int32_t *destination_device = nullptr;
            int draft_token_count = 0;

            /** @brief Validate the resident sources and destination against the physical bucket. */
            [[nodiscard]] bool valid(int padded_seq_len) const noexcept
            {
                return first_token_device != nullptr &&
                       destination_device != nullptr &&
                       draft_token_count >= 0 &&
                       draft_token_count < padded_seq_len &&
                       (draft_token_count == 0 ||
                        draft_tokens_device != nullptr);
            }

            /** @brief Compare immutable row bindings, not the token values behind them. */
            bool operator==(const TokenRowBinding &) const = default;
        };

        /**
         * @brief Opaque device checkpoint destination for one logical request.
         *
         * The cache implementation owns the payload format. The stage merely
         * captures the exact sequence metadata into persistent caller-owned device
         * storage before the verifier graph can append speculative rows.
         */
        struct MainKVCheckpointBinding
        {
            IKVCache *cache = nullptr;
            int sequence_index = -1;
            void *checkpoint_device = nullptr;
            size_t checkpoint_bytes = 0;

            /** @brief Require a complete cache-owned opaque checkpoint destination. */
            [[nodiscard]] bool valid() const noexcept
            {
                return cache != nullptr && sequence_index >= 0 &&
                       checkpoint_device != nullptr && checkpoint_bytes > 0;
            }

            /** @brief Compare exact cache, request, destination and opaque byte capacity. */
            bool operator==(const MainKVCheckpointBinding &) const = default;
        };

        /**
         * @brief Persistent device bindings and immutable verifier geometry.
         *
         * `valid_graph_rows_device` is null for a dense prefix in every request
         * row. In that case `valid_graph_row_count / request_count` is the
         * logical width and may be smaller than `padded_seq_len`; this is the
         * scalar fixed-bucket verifier contract. Ragged graphs point at the same
         * resident row-index array consumed by the compact LM-head selector.
         * `generation_control_device` replaces that static logical width only
         * for a resident generation parent; the backend validates the depth/row
         * relation and publishes a fatal controller error on corruption. In that
         * mode `draft_token_count` and `valid_graph_row_count` are setup evidence,
         * not capture identity: one fused kernel reads the live depth and reuses
         * the same physical bucket for every logical width it can contain.
         * PipelineFollower binds only backend, physical geometry and local
         * checkpoints. All token/controller/row-publication fields must be empty;
         * its verifier receives terminal-owned metadata through explicit graph
         * collectives, not through a second local preparation authority.
         */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IBackend *backend = nullptr;
            Authority authority = Authority::Unbound;
            std::span<const TokenRowBinding> token_rows;
            int request_count = 0;
            int padded_seq_len = 0;

            const int32_t *base_cached_tokens_device = nullptr;
            const int32_t *valid_graph_rows_device = nullptr;
            int valid_graph_row_count = 0; ///< Total logical rows across requests.
            int *generation_control_device = nullptr;
            int generation_control_stride = 0;
            /** Optional resident MoE boundary used to clip this transaction. */
            const uint32_t *maintenance_rows_remaining_device = nullptr;
            /** Device due/error state paired with the maintenance row budget. */
            const uint32_t *maintenance_due_device = nullptr;
            /** Mutable acknowledgement left by the preceding speculative commit. */
            uint32_t *decode_boundary_advanced_device = nullptr;
            int32_t *position_ids_device = nullptr;
            int32_t *request_lengths_device = nullptr;
            int32_t *base_cached_tokens_snapshot_device = nullptr;
            std::span<const MainKVCheckpointBinding> main_kv_checkpoints;

            std::string stage_name = "mtp_verifier_preparation";
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Retain immutable binding copies; the caller's temporary spans may expire. */
        explicit MTPVerifierPreparationStage(Params params);

        /** @brief Validate the role, then enqueue its complete preparation on the exact stream. */
        bool execute(IDeviceContext *ctx) override;
        /** @brief Return the shared stage kind for owner and follower preparation. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MTP_VERIFIER_PREPARATION;
        }
        /** @brief Return the caller's stable diagnostic stage name. */
        std::string name() const override { return params_.stage_name; }
        /** @brief Estimate owner geometry arithmetic; a follower only copies checkpoint bytes. */
        size_t estimatedFlops() const override;
        /** @brief Estimate traffic from this role's actual bindings, without allocating storage. */
        size_t estimatedMemoryBytes() const override;
        /** @brief GPU preparation is captured; CPU execution has its own host-owned lifecycle. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @brief All preparation operations are native-capture safe. */
        bool isGraphCapturable() const override { return true; }
        /** @brief Preparation is participant-local; collectives remain explicit graph edges. */
        bool isCollectiveStage() const override { return false; }
        /** @brief Persistent device bindings need no stage-local coherence transition. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        /** @brief Report role and immutable geometry without reading device execution state. */
        StageDumpInfo buildDumpInfoImpl() const override;
        /** @brief Declare owner arena buffers; followers reference only cache-owned checkpoint storage. */
        StageBufferContract bufferContract() const override;

        /**
         * @brief Compare every scalar and address retained by captured graph nodes.
         *
         * Values behind device pointers are deliberately excluded. Any pointer,
         * request mapping, or launch-geometry change requires a different
         * graph family member instead of mutating a captured executable in place.
         */
        [[nodiscard]] bool hasSameCaptureIdentity(
            const Params &other) const noexcept;

        /**
         * @brief Render every field that contributes to graph-capture identity.
         *
         * This helper is reserved for fatal graph-family diagnostics. It is not
         * called during successful lookup or replay, so formatting pointer-rich
         * state cannot add allocation or logging work to the decode hot path.
         * Keeping the formatter beside @ref hasSameCaptureIdentity prevents a
         * future identity field from becoming invisible when a finite setup-time
         * graph family is exhausted.
         *
         * @param params Candidate stage parameters to describe.
         * @return Stable, human-readable identity summary.
         */
        [[nodiscard]] static std::string describeCaptureIdentity(
            const Params &params);

        /**
         * @brief Describe this captured stage's retained immutable identity.
         * @return The same field set consumed by @ref hasSameCaptureIdentity.
         */
        [[nodiscard]] std::string captureIdentityDescription() const
        {
            return describeCaptureIdentity(params_);
        }

        /** @brief Expose retained immutable bindings for graph construction diagnostics. */
        [[nodiscard]] const Params &getParams() const noexcept
        {
            return params_;
        }

    private:
        /** @brief Reject incomplete or contradictory authority bindings before any device mutation. */
        [[nodiscard]] bool validate() const;

        /**
         * @brief Capture each local request's full opaque KV frontier on the supplied stream.
         * @param stream Validated exact capture/replay stream; never a default stream.
         * @return Whether every cache accepted its checkpoint operation.
         */
        [[nodiscard]] bool captureLocalKVCheckpoints(void *stream) const;

        std::vector<TokenRowBinding> owned_token_rows_;
        std::vector<MainKVCheckpointBinding> owned_main_kv_checkpoints_;
        Params params_;
    };
} // namespace llaminar2
