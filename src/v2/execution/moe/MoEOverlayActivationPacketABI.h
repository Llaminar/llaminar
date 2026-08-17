/**
 * @file MoEOverlayActivationPacketABI.h
 * @brief Device-facing sparse activation packet views for ExpertOverlay.
 *
 * Node-local heterogeneous ExpertOverlay participants exchange compact rows in
 * setup-owned shared pages.  This file describes only the non-owning payload
 * views embedded by retained CPU, CUDA, and ROCm transactions.  It contains no
 * transport, backend, stream, or topology policy: the topology planner and the
 * selected transport resolve each pointer before graph capture.
 *
 * Dispatch entries retain original row-major/router-slot order.  Return rows
 * retain the dispatch row order.  A continuation consumes participant lanes in
 * planner-canonical participant order, preserving deterministic FP32 addition
 * independently of device completion order.
 */

#pragma once

#include "MoEOverlayActivationEpochABI.h"
#include "DeviceMoEOverlayEpochABI.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_PACKET_HD __host__ __device__
#else
#define LLAMINAR_MOE_PACKET_HD
#endif

namespace llaminar2
{
    /**
     * @brief Immutable interpretation of a dispatch packet's hidden matrix.
     *
     * Decode-sized direct packets carry only the compact rows selected for one
     * participant. Bulk node-local prefill publishes the continuation's
     * physical-row matrix once for all participants sharing a rank-pair
     * channel; each follower then gathers its compact rows with the packet's
     * lane-local `row_ids`. The value is retained graph identity and may not be
     * selected from mutable packet contents during replay.
     */
    enum class MoEOverlayActivationHiddenPayloadLayout : std::uint8_t
    {
        CompactRows = 0, ///< Matrix row N is compact packet row N.
        SharedPhysicalRows = 1, ///< Matrix row N is original physical row N.
    };

    /** @return Whether @p layout is a supported captured packet layout. */
    [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool
    isValidMoEOverlayActivationHiddenPayloadLayout(
        MoEOverlayActivationHiddenPayloadLayout layout) noexcept
    {
        return layout ==
                   MoEOverlayActivationHiddenPayloadLayout::CompactRows ||
               layout == MoEOverlayActivationHiddenPayloadLayout::
                             SharedPhysicalRows;
    }

    /**
     * @brief Device-local execution grant for one admitted activation lane.
     *
     * The scheduler-owned mapped control block authenticates a transaction once
     * at stage zero.  After that acquire, the retained graph advances this
     * backend-local record instead of rereading and rewriting PCIe-mapped
     * control cache lines at every transformer layer.  The record is private to
     * one endpoint GPU: continuation and follower never share it, and peers
     * continue to communicate exclusively through packet descriptors and
     * timeline words in the mapped channel.
     *
     * A final stage copies the accumulated counters and terminal state back to
     * the endpoint-owned mapped status.  Consequently the scheduler and
     * PerfStats observe exactly the same authenticated evidence while the
     * inference hot path keeps its ordering cursor and current row geometry in
     * device memory.
     */
    struct alignas(64) MoEOverlayActivationDeviceEpochGrant
    {
        MoEOverlayActivationDigest digest{}; ///< Admitted identity witness.
        std::uint64_t generation = 0u; ///< Strict scheduler epoch generation.
        std::uint64_t placement_epoch = 0u; ///< Request-pinned placement bank.
        std::uint64_t published_payload_bytes = 0u; ///< Endpoint-owned traffic total.
        std::uint64_t published_live_rows = 0u; ///< Published compact rows.
        std::uint64_t published_live_entries = 0u; ///< Published route entries.
        std::uint64_t published_stage_count = 0u; ///< Published descriptors.
        std::uint64_t live_rows = 0u; ///< Current validated stage row count.
        std::uint64_t live_entries = 0u; ///< Current validated route count.
        std::uint32_t stage_count = 0u; ///< Immutable retained manifest length.
        std::int32_t physical_rows = 0; ///< Exact captured row geometry.
        std::int32_t last_published_stage = -1; ///< Local publication cursor.
        std::int32_t last_consumed_stage = -1; ///< Local consumption cursor.
        std::int32_t single_row_id = -1; ///< Fast path when current payload has one row.
        std::uint32_t endpoint = 0u; ///< Raw @ref MoEOverlayActivationEndpoint.
        std::uint32_t state = 0u; ///< Raw @ref MoEOverlayActivationEndpointState.
        std::uint32_t code = 0u; ///< Raw @ref MoEOverlayActivationStatusCode.
    };

    /**
     * @brief Device-visible CSR dispatch view inside one shared activation lane.
     *
     * Every pointer is the exact alias resolved for one planner-selected local
     * endpoint.  The type owns no storage and is safe to copy into a captured
     * kernel launch description. Route and row capacities are immutable model
     * admission data; live counts are published in the matching activation
     * descriptor and never mirrored in this view.
     */
    struct MoEOverlayMappedDispatchDeviceView
    {
        std::int32_t *row_ids = nullptr; ///< Original logical row for each compact row.
        std::int32_t *entry_offsets = nullptr; ///< CSR offsets, capacity `row_capacity + 1`.
        std::int32_t *expert_ids = nullptr; ///< Expert id for every compact route entry.
        float *route_weights = nullptr; ///< Router weight for every compact route entry.
        float *hidden_rows_fp32 = nullptr; ///< Compact FP32 activation matrix.
        std::size_t row_capacity = 0u; ///< Maximum compact rows admitted by setup.
        std::size_t entry_capacity = 0u; ///< Maximum compact route entries.
        std::int32_t d_model = 0; ///< Hidden width of one compact row.
        std::int32_t top_k = 0; ///< Maximum route entries in one logical row.

        /** @return Whether all aliases and immutable geometry are complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return row_ids && entry_offsets && expert_ids && route_weights &&
                   hidden_rows_fp32 && row_capacity > 0u &&
                   entry_capacity >= row_capacity && d_model > 0 && top_k > 0;
        }
    };

    /**
     * @brief Device-visible compact return view inside one shared activation lane.
     */
    struct MoEOverlayMappedReturnDeviceView
    {
        std::int32_t *row_ids = nullptr; ///< Original row identity preserved by the follower.
        float *output_rows_fp32 = nullptr; ///< Participant-local FP32 expert result rows.
        std::size_t row_capacity = 0u; ///< Maximum compact rows admitted by setup.
        std::int32_t d_model = 0; ///< Hidden width of one result row.

        /** @return Whether all aliases and immutable geometry are complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return row_ids && output_rows_fp32 && row_capacity > 0u &&
                   d_model > 0;
        }
    };

    /**
     * @brief Minimal graph-facing view of one versioned overlay placement bank.
     *
     * Packet compaction needs only the bank epoch and the overlay-wide target
     * for each logical expert. Keeping this view independent of weight
     * descriptors lets the CUDA/HIP packet bridge remain device-friendly while
     * still proving that the request ticket names the exact bank generation.
     */
    struct MoEOverlayRoutePlacementBankDeviceView
    {
        const std::int32_t *route_participants = nullptr; ///< `[expert_count]` global targets.
        const std::uint32_t *epoch = nullptr; ///< Address of the bank's published epoch.

        /** @return Whether both immutable captured addresses are present. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return route_participants && epoch;
        }
    };

    /**
     * @brief Complete immutable placement input for one captured packet stage.
     *
     * The two bank views, request ticket, and expert geometry form one authority:
     * mixing any member from a different runtime table could route a retained
     * transaction using a bank generation that the ticket did not publish. Model
     * setup therefore passes this binding as one value. MTP child tables resolve
     * the views from their canonical main-model placement source while retaining
     * the exact shared request ticket.
     */
    struct MoEOverlayRoutePlacementDeviceBinding
    {
        MoEOverlayRoutePlacementBankDeviceView
            banks[kDeviceMoEOverlayEpochBankCount] = {};
        /** Request-lifetime placement bank selected before retained replay. */
        const DeviceMoEOverlayEpochTicket *ticket = nullptr;
        /** Exact logical expert geometry shared by both banks. */
        std::uint32_t expert_count = 0u;

        /** @return Whether every captured address and the geometry are complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return banks[0].valid() && banks[1].valid() && ticket &&
                   expert_count > 0u;
        }
    };

    /**
     * @brief Captured continuation launch that compacts routes for one target.
     *
     * Route arrays are immutable-address device scratch published by the
     * router. `route_slot = row * top_k + slot`; the selected expert indexes the
     * overlay-wide target array in the request-pinned placement bank. Entries
     * matching @ref target_participant_id are retained in exact router order.
     * The optional active-row scalar is device-owned replay state. A null scalar
     * means all physical rows are live, which is appropriate for unpadded decode
     * graphs.
     */
    struct MoEOverlayActivationDispatchPackLaunch
    {
        const float *hidden_rows_fp32 = nullptr; ///< `[physical_rows, d_model]` source.
        /**
         * Router-selected expert per slot in the model graph's canonical FP32
         * routing tensor. Expert ids are exactly representable for every
         * supported MoE geometry; the packet kernel validates finiteness,
         * integrality, and range before converting a value to an array index.
         */
        const float *route_expert_ids_fp32 = nullptr;
        const float *route_weights = nullptr; ///< Router-selected weight per slot.
        /** Exact ticket-selected durable placement authority. */
        MoEOverlayRoutePlacementDeviceBinding placement{};
        const std::int32_t *active_row_count_device = nullptr; ///< Optional live-row scalar.
        MoEOverlayMappedDispatchDeviceView packet{}; ///< Shared output packet.
        MoEOverlayActivationHiddenPayloadLayout hidden_payload_layout =
            MoEOverlayActivationHiddenPayloadLayout::CompactRows; ///< Fixed matrix interpretation.
        MoEOverlayActivationEpochControl *control = nullptr; ///< Exact mapped lane control.
        MoEOverlayActivationDeviceEpochGrant *grant = nullptr; ///< Continuation-local epoch state.
        std::int32_t target_participant_id = -1; ///< Logical destination selected by planning.
        std::int32_t physical_rows = 0; ///< Captured row capacity of the source graph.
        std::uint32_t stage_ordinal = 0u; ///< Ordered stage in the retained transaction.
        std::int32_t model_layer_index = -1; ///< Layer embedded at this stage ordinal.

        /** @return Whether setup supplied a complete immutable launch contract. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return hidden_rows_fp32 && route_expert_ids_fp32 &&
                   route_weights && placement.valid() &&
                   packet.valid() &&
                   isValidMoEOverlayActivationHiddenPayloadLayout(
                       hidden_payload_layout) &&
                   control && grant && target_participant_id >= 0 &&
                   physical_rows > 0 &&
                   static_cast<std::size_t>(physical_rows) <=
                       packet.row_capacity &&
                   packet.entry_capacity >=
                       static_cast<std::size_t>(physical_rows) *
                           static_cast<std::size_t>(packet.top_k) &&
                   model_layer_index >= 0;
        }
    };

    /**
     * @brief Captured follower launch that expands one shared CSR packet.
     *
     * Output tensors have fixed captured geometry. Unused rows and route slots
     * are deterministically cleared, so downstream grouped expert kernels can
     * execute one retained topology while the device-owned live-row scalar
     * records the compact prefix that contains real work.
     */
    struct MoEOverlayActivationDispatchConsumeLaunch
    {
        MoEOverlayMappedDispatchDeviceView packet{}; ///< Shared input packet.
        MoEOverlayActivationHiddenPayloadLayout hidden_payload_layout =
            MoEOverlayActivationHiddenPayloadLayout::CompactRows; ///< Fixed matrix interpretation.
        const MoEOverlayActivationEpochControl *control = nullptr; ///< Exact mapped control.
        MoEOverlayActivationDeviceEpochGrant *grant = nullptr; ///< Follower-local epoch state.
        float *hidden_rows_fp32 = nullptr; ///< Fixed follower-local hidden tensor.
        float *routing_indices_fp32 = nullptr; ///< Fixed FP32 expert-id tensor.
        float *routing_weights_fp32 = nullptr; ///< Fixed route-weight tensor.
        std::int32_t *active_row_count_device = nullptr; ///< Published compact live rows.
        std::int32_t physical_rows = 0; ///< Captured follower row capacity.
        std::uint32_t stage_ordinal = 0u; ///< Ordered stage in the transaction.
        std::int32_t model_layer_index = -1; ///< Expected layer identity.

        /** @return Whether every captured destination and geometry is valid. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return packet.valid() &&
                   isValidMoEOverlayActivationHiddenPayloadLayout(
                       hidden_payload_layout) &&
                   control && grant && hidden_rows_fp32 &&
                   routing_indices_fp32 && routing_weights_fp32 &&
                   active_row_count_device && physical_rows > 0 &&
                   static_cast<std::size_t>(physical_rows) <=
                       packet.row_capacity &&
                   model_layer_index >= 0;
        }
    };

    /**
     * @brief Captured follower launch that publishes compact expert outputs.
     */
    struct MoEOverlayActivationReturnPackLaunch
    {
        const float *local_output_rows_fp32 = nullptr; ///< Follower fixed output tensor.
        MoEOverlayMappedDispatchDeviceView dispatch{}; ///< Source row identities/counts.
        MoEOverlayMappedReturnDeviceView returned{}; ///< Shared compact return destination.
        MoEOverlayActivationEpochControl *control = nullptr; ///< Exact mapped lane control.
        MoEOverlayActivationDeviceEpochGrant *grant = nullptr; ///< Follower-local epoch state.
        std::int32_t physical_rows = 0; ///< Captured follower row capacity.
        std::uint32_t stage_ordinal = 0u; ///< Ordered stage in the transaction.
        std::int32_t model_layer_index = -1; ///< Expected layer identity.

        /** @return Whether setup supplied matching dispatch/return geometry. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return local_output_rows_fp32 && dispatch.valid() &&
                   returned.valid() && control && grant && physical_rows > 0 &&
                   dispatch.d_model == returned.d_model &&
                   static_cast<std::size_t>(physical_rows) <=
                       dispatch.row_capacity &&
                   static_cast<std::size_t>(physical_rows) <=
                       returned.row_capacity &&
                   model_layer_index >= 0;
        }
    };

    /**
     * @brief Captured continuation launch for one ordered participant return.
     *
     * Callers enqueue these launches in the planner's canonical participant
     * order on one exact continuation stream. One thread owns each output
     * element, so the resulting FP32 additions match the scalar participant
     * fold and never depend on peer completion order.
     */
    struct MoEOverlayActivationReturnConsumeLaunch
    {
        MoEOverlayMappedDispatchDeviceView dispatch{}; ///< Original dispatch row order.
        MoEOverlayMappedReturnDeviceView returned{}; ///< Shared compact return source.
        const MoEOverlayActivationEpochControl *control = nullptr; ///< Exact mapped control.
        MoEOverlayActivationDeviceEpochGrant *grant = nullptr; ///< Continuation-local epoch state.
        float *dense_output_rows_fp32 = nullptr; ///< Continuation accumulation tensor.
        std::int32_t physical_rows = 0; ///< Captured destination row capacity.
        std::uint32_t stage_ordinal = 0u; ///< Ordered stage in the transaction.
        std::int32_t model_layer_index = -1; ///< Expected layer identity.

        /** @return Whether setup supplied complete deterministic fold state. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return dispatch.valid() && returned.valid() &&
                   dispatch.d_model == returned.d_model && control && grant &&
                   dense_output_rows_fp32 && physical_rows > 0 &&
                   model_layer_index >= 0;
        }
    };

    /**
     * @brief Device-kernel acquire edge for one mapped 64-bit timeline word.
     *
     * Production code obtains the address through TransferEngine's typed
     * mapped-region binding API. Keeping the wait edge in the launch ABI lets
     * a latency-sensitive one-row packet kernel perform the system acquire,
     * validation, and payload materialization in one graph node.
     */
    struct MoEOverlayActivationTimelineWaitDeviceBinding
    {
        const std::uint64_t *signal = nullptr; ///< Exact device-visible mapped alias.
        std::uint64_t value = 0u; ///< Positive unsigned-GEQ lease value.

        /** @return Whether the immutable system-acquire edge is complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return signal && value != 0u;
        }
    };

    /**
     * @brief Device-kernel release edge for one mapped 64-bit timeline word.
     *
     * The packet kernel stores @ref value with system-release semantics only
     * after every mapped payload byte has been written. TransferEngine remains
     * the public authority that resolves this raw device alias.
     */
    struct MoEOverlayActivationTimelinePublishDeviceBinding
    {
        std::uint64_t *signal = nullptr; ///< Exact device-visible mapped alias.
        std::uint64_t value = 0u; ///< Positive monotonic lease value to publish.

        /** @return Whether the immutable system-release edge is complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return signal && value != 0u;
        }
    };

    /** @brief One-row dispatch pack fused with its mapped publication edge. */
    struct MoEOverlayActivationSingleRowDispatchPackLaunch
    {
        MoEOverlayActivationDispatchPackLaunch packet{}; ///< Packet and route authority.
        MoEOverlayActivationTimelinePublishDeviceBinding publication{}; ///< Final release.

        /** @return Whether this is an exact one-row direct-mapped launch. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return packet.valid() && packet.physical_rows == 1 &&
                   publication.valid();
        }
    };

    /** @brief One-row dispatch wait, validation, and materialization launch. */
    struct MoEOverlayActivationSingleRowDispatchConsumeLaunch
    {
        MoEOverlayActivationDispatchConsumeLaunch packet{}; ///< Packet and destinations.
        MoEOverlayActivationTimelineWaitDeviceBinding acquire{}; ///< Leading acquire.

        /** @return Whether this is an exact one-row direct-mapped launch. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return packet.valid() && packet.physical_rows == 1 &&
                   acquire.valid();
        }
    };

    /** @brief One-row return pack fused with its mapped publication edge. */
    struct MoEOverlayActivationSingleRowReturnPackLaunch
    {
        MoEOverlayActivationReturnPackLaunch packet{}; ///< Return packet authority.
        MoEOverlayActivationTimelinePublishDeviceBinding publication{}; ///< Final release.

        /** @return Whether this is an exact one-row direct-mapped launch. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return packet.valid() && packet.physical_rows == 1 &&
                   publication.valid();
        }
    };

    /** @brief One-row return wait, validation, and ordered accumulation launch. */
    struct MoEOverlayActivationSingleRowReturnConsumeLaunch
    {
        MoEOverlayActivationReturnConsumeLaunch packet{}; ///< Return and accumulator.
        MoEOverlayActivationTimelineWaitDeviceBinding acquire{}; ///< Leading acquire.

        /** @return Whether this is an exact one-row direct-mapped launch. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return packet.valid() && packet.physical_rows == 1 &&
                   acquire.valid();
        }
    };

    /**
     * @brief Topology-sized continuation dispatch for independent one-row lanes.
     *
     * The descriptor array is prepared once before capture and remains resident
     * on the continuation device for the complete retained-graph lifetime. One
     * device block owns one planner-canonical lane, so compaction and mapped
     * publication proceed concurrently without baking any participant count or
     * backend role into the kernel ABI.
     */
    struct MoEOverlayActivationSingleRowDispatchBatchLaunch
    {
        /** Device array in planner-canonical participant order. */
        const MoEOverlayActivationSingleRowDispatchPackLaunch *lanes = nullptr;
        std::uint32_t lane_count = 0u; ///< Positive topology-derived lane count.

        /** @return Whether setup supplied a complete persistent batch view. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return lanes && lane_count > 0u;
        }
    };

    /**
     * @brief Parallel return acquisition followed by a canonical FP32 fold.
     *
     * The first kernel gives each lane an independent block which waits on and
     * validates only that lane, then copies its row into @ref gathered_rows_fp32.
     * A second kernel walks the resident rows in descriptor order and performs
     * explicit round-to-nearest FP32 additions. Completion order therefore no
     * longer serializes mapped waits, while arithmetic order remains identical
     * to the historical one-kernel-per-lane fold.
     */
    struct MoEOverlayActivationSingleRowReturnBatchLaunch
    {
        /** Device array in planner-canonical participant order. */
        const MoEOverlayActivationSingleRowReturnConsumeLaunch *lanes = nullptr;
        float *gathered_rows_fp32 = nullptr; ///< `[lane_count, d_model]` device scratch.
        std::int32_t *lane_live_rows = nullptr; ///< `[lane_count]`, written by gather.
        float *dense_output_rows_fp32 = nullptr; ///< Canonical continuation accumulator.
        std::uint32_t lane_count = 0u; ///< Positive topology-derived lane count.
        std::int32_t d_model = 0; ///< Exact one-row width shared by every lane.

        /** @return Whether setup supplied complete persistent batch storage. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return lanes && gathered_rows_fp32 && lane_live_rows &&
                   dense_output_rows_fp32 && lane_count > 0u && d_model > 0;
        }
    };

    /**
     * @brief Validate and fold a topology-sized batch of multi-row returns.
     *
     * Every descriptor is prepared once in planner-canonical participant
     * order.  The validation kernel gives each lane one independent block and
     * materializes a destination-row to compact-row lookup.  A second kernel
     * gives each dense output element one writer and walks those lookups in
     * descriptor order using explicit FP32 additions.  This removes the former
     * one-validation-plus-one-fold launch pair per participant without making
     * completion timing an arithmetic-order authority.
     */
    struct MoEOverlayActivationMultiRowReturnBatchLaunch
    {
        /** Device array in planner-canonical participant order. */
        const MoEOverlayActivationReturnConsumeLaunch *lanes = nullptr;
        /** `[lane_count, physical_rows]`; -1 means the lane omitted the row. */
        std::int32_t *lane_row_to_compact = nullptr;
        /** `[lane_count]`; set only after the lane descriptor is authenticated. */
        std::int32_t *lane_valid = nullptr;
        float *dense_output_rows_fp32 = nullptr; ///< Canonical continuation accumulator.
        std::uint32_t lane_count = 0u; ///< Positive topology-derived lane count.
        std::int32_t physical_rows = 0; ///< Captured destination row capacity.
        std::int32_t d_model = 0; ///< Exact output width shared by every lane.

        /** @return Whether setup supplied complete persistent batch storage. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return lanes && lane_row_to_compact && lane_valid &&
                   dense_output_rows_fp32 && lane_count > 0u &&
                   physical_rows > 1 && d_model > 0;
        }
    };

    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayActivationSingleRowDispatchPackLaunch>);
    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayActivationSingleRowReturnConsumeLaunch>);
    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayActivationMultiRowReturnBatchLaunch>);

    /**
     * @brief Compute live dispatch payload bytes without descriptor padding.
     *
     * The result covers row ids, CSR offsets, expert ids, route weights, and
     * FP32 hidden rows. An empty `(0, 0)` packet intentionally has zero payload
     * bytes but still advances its epoch timeline. Zero for non-empty geometry
     * indicates invalid geometry or integer overflow.
     */
    [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr std::uint64_t
    moeOverlayDispatchPayloadBytes(
        std::uint64_t live_rows,
        std::uint64_t live_entries,
        std::uint32_t d_model) noexcept
    {
        if (d_model == 0u || (live_rows == 0u && live_entries != 0u) ||
            (live_rows != 0u && live_entries < live_rows))
            return 0u;
        if (live_rows == 0u)
            return 0u;
        constexpr std::uint64_t maximum = ~std::uint64_t{0};
        if (live_rows > (maximum / sizeof(std::int32_t)) - 1u ||
            live_entries > maximum /
                               (sizeof(std::int32_t) + sizeof(float)) ||
            live_rows > maximum /
                            (static_cast<std::uint64_t>(d_model) *
                             sizeof(float)))
        {
            return 0u;
        }
        const std::uint64_t row_metadata =
            live_rows * sizeof(std::int32_t) +
            (live_rows + 1u) * sizeof(std::int32_t);
        const std::uint64_t entries =
            live_entries * (sizeof(std::int32_t) + sizeof(float));
        const std::uint64_t activations =
            live_rows * static_cast<std::uint64_t>(d_model) * sizeof(float);
        if (row_metadata > maximum - entries ||
            row_metadata + entries > maximum - activations)
        {
            return 0u;
        }
        return row_metadata + entries + activations;
    }

    /**
     * @brief Compute live compact return bytes without descriptor padding.
     * An empty return intentionally has zero bytes and still advances its
     * timeline. For a non-empty return, zero means invalid geometry/overflow.
     */
    [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr std::uint64_t
    moeOverlayReturnPayloadBytes(
        std::uint64_t live_rows,
        std::uint32_t d_model) noexcept
    {
        if (d_model == 0u)
            return 0u;
        if (live_rows == 0u)
            return 0u;
        constexpr std::uint64_t maximum = ~std::uint64_t{0};
        const std::uint64_t row_bytes = sizeof(std::int32_t);
        const std::uint64_t output_bytes =
            static_cast<std::uint64_t>(d_model) * sizeof(float);
        if (output_bytes > maximum - row_bytes ||
            live_rows > maximum / (row_bytes + output_bytes))
        {
            return 0u;
        }
        return live_rows * (row_bytes + output_bytes);
    }

    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayMappedDispatchDeviceView>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayActivationDeviceEpochGrant>);
    static_assert(sizeof(MoEOverlayActivationDeviceEpochGrant) == 128u);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayMappedReturnDeviceView>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayRoutePlacementBankDeviceView>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayRoutePlacementDeviceBinding>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayActivationDispatchPackLaunch>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayActivationDispatchConsumeLaunch>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayActivationReturnPackLaunch>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayActivationReturnConsumeLaunch>);
} // namespace llaminar2

#undef LLAMINAR_MOE_PACKET_HD
