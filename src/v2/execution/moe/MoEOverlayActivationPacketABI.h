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
 * Dispatch entries retain original row-major/router-slot order and carry both
 * their source slot and their compact follower slot.  GPU followers publish
 * one preweighted FP32 contribution per original route slot into a shared
 * canonical matrix.  The continuation materializes those slots and performs
 * the only top-k reduction in original router order.  Expert movement may
 * therefore change transport ownership without changing floating-point
 * parentheses.
 */

#pragma once

#include "MoEOverlayActivationEpochABI.h"
#include "MoEOverlayActivationPayloadLayout.h"
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
        /** Exact endpoint-local epoch selected by the device request ticket. */
        std::uint64_t placement_epoch = 0u;
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
        /**
         * Authenticated semantic family copied from the admitted epoch.
         * Retained compute and telemetry kernels read this device-local value;
         * the scheduler cannot mutate it after admission.
         */
        MoEOverlayInferenceGraphRole graph_role =
            MoEOverlayInferenceGraphRole::None;
    };

    /**
     * @brief Generation-authenticated placement epoch source for a follower.
     *
     * Retained activation graphs reuse bank-local timeline values after the
     * scheduler has retired both endpoints and reset the channel lease.  A
     * fixed `wait >= 1` is consequently an ordering hint, not transaction
     * identity: a device may briefly observe the preceding lease's signal
     * while the new stage-zero descriptor is still empty.  The follower epoch
     * acquire kernel consumes this binding and waits until the mapped identity
     * is strictly newer than its device-local grant and the stage descriptor's
     * digest belongs to that identity.  Only then may it read
     * `placement_epoch` and acquire the matching local RCU bank.
     *
     * The pointers are immutable graph identity. `control` addresses the
     * planner-owned node-local mapped channel, while `grant` addresses the
     * endpoint-private device record retained across replays.
     */
    struct MoEOverlayPeerPlacementEpochBinding
    {
        const MoEOverlayActivationEpochControl *control = nullptr;
        const MoEOverlayActivationDeviceEpochGrant *grant = nullptr;
        std::uint32_t stage_ordinal = 0u;

        /** @return Whether the binding names a representable packet stage. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return control != nullptr && grant != nullptr;
        }

        bool operator==(
            const MoEOverlayPeerPlacementEpochBinding &) const = default;
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
        /** Original `physical_row * top_k + router_slot` for every entry. */
        std::int32_t *original_route_slots = nullptr;
        /** Follower-local `compact_row * top_k + compact_slot` per entry. */
        std::int32_t *compact_route_slots = nullptr;
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
                   original_route_slots && compact_route_slots &&
                   hidden_rows_fp32 && row_capacity > 0u &&
                   entry_capacity >= row_capacity && d_model > 0 && top_k > 0;
        }
    };

    /**
     * @brief Device-visible canonical return matrix for one rank-pair channel.
     *
     * Every participant lane in the same channel aliases this matrix.  A route
     * slot has exactly one authoritative participant, so followers write
     * disjoint rows even when their kernels execute concurrently.  The matrix
     * is indexed by original router slot rather than participant or completion
     * order; timeline descriptors authenticate which sparse rows are live.
     */
    struct MoEOverlayMappedReturnDeviceView
    {
        /** Shared `[route_slot_capacity, d_model]` preweighted contributions. */
        float *canonical_route_contributions_fp32 = nullptr;
        /** Maximum original route slots admitted by model setup. */
        std::size_t route_slot_capacity = 0u;
        std::int32_t d_model = 0; ///< Hidden width of one contribution row.
        /** @return Whether all aliases and immutable geometry are complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return canonical_route_contributions_fp32 &&
                   route_slot_capacity > 0u && d_model > 0;
        }
    };

    /**
     * @brief Host-published completion record for one colocated CPU route bank.
     *
     * The CPU producer writes every route slot and preweighted contribution
     * before release-publishing the next @ref published_sequence. A retained
     * CUDA/HIP consumer acquires that exact sequence, reads only
     * @ref live_entry_count compact entries, and release-publishes the same
     * value to @ref consumed_sequence after materialization completes. The
     * producer may not reuse the mapped payload until both sequences match.
     *
     * Separate monotonic producer/consumer sequences deliberately replace a
     * reusable zero/one flag. A captured consumer can therefore distinguish a
     * new publication from the preceding replay even when it starts before the
     * CPU has armed the next transaction. The record is isolated on its own
     * cache line so polling never contends with route-slot payload bytes.
     */
    struct alignas(64) MoEOverlayCanonicalRouteTicketControl
    {
        static constexpr std::uint32_t kMagic = 0x43544f4du; // "MOTC"
        static constexpr std::uint32_t kABIVersion = 2u;

        std::uint32_t magic = kMagic; ///< Immutable ABI identity.
        std::uint32_t abi_version = kABIVersion; ///< Immutable ABI version.
        std::uint64_t workspace_generation = 0u; ///< Setup-owned graph identity.
        /** CPU-owned monotonic release sequence; zero means never published. */
        std::uint64_t published_sequence = 0u;
        /** GPU-owned monotonic acknowledgement after complete materialization. */
        std::uint64_t consumed_sequence = 0u;
        std::uint64_t live_entry_count = 0u; ///< Compact entries published this replay.
        std::uint64_t residency_epoch = 0u; ///< Exact placement epoch used by the CPU.
        std::int32_t layer_idx = -1; ///< Model layer embedded in the ticket.
        std::int32_t route_capacity = 0; ///< Maximum compact/original slot count.
        std::int32_t d_model = 0; ///< Width of one canonical contribution row.
        std::int32_t reserved = 0;

        /** @return Whether immutable identity and geometry are representable. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid() const noexcept
        {
            return magic == kMagic && abi_version == kABIVersion &&
                   workspace_generation != 0u && layer_idx >= 0 &&
                   route_capacity > 0 && d_model > 0;
        }

        /**
         * @return Whether the producer/consumer cursors describe zero or one
         *         outstanding SPSC publication.
         */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool
        sequenceStateValid() const noexcept
        {
            return published_sequence >= consumed_sequence &&
                   published_sequence - consumed_sequence <= 1u;
        }

        /** @return Whether exactly one newly published payload awaits the GPU. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool
        publicationPending() const noexcept
        {
            return sequenceStateValid() &&
                   published_sequence > consumed_sequence;
        }
    };

    static_assert(
        sizeof(MoEOverlayCanonicalRouteTicketControl) == 64u,
        "canonical route ticket control must remain one cache line");
    static_assert(
        std::is_trivially_copyable_v<MoEOverlayCanonicalRouteTicketControl>);

    /**
     * @brief Captured GPU materialization of one colocated CPU route ticket.
     *
     * Route identities and contribution rows are immutable-address mapped host
     * storage. The CPU stores contributions at the authenticated compact slot
     * used by its grouped expert kernel; this launch places each row into the
     * continuation bank's original router slot without reducing it.
     */
    struct MoEOverlayCanonicalRouteTicketConsumeLaunch
    {
        /** Mutable because the consumer acknowledges its exact publication. */
        MoEOverlayCanonicalRouteTicketControl *control = nullptr;
        const std::int32_t *original_route_slots = nullptr;
        const std::int32_t *compact_route_slots = nullptr;
        const float *compact_preweighted_contributions_fp32 = nullptr;
        float *canonical_route_contributions_fp32 = nullptr;
        std::size_t route_capacity = 0u;
        std::int32_t d_model = 0;

        /** @return Whether every capture-stable mapped/device address is complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid() const noexcept
        {
            /* `control` is a GPU alias of mapped host memory. Host launch
             * validation must not dereference that alias: the device kernel
             * acquires and authenticates the record after publication. */
            return control && original_route_slots && compact_route_slots &&
                   compact_preweighted_contributions_fp32 &&
                   canonical_route_contributions_fp32 && route_capacity > 0u &&
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
        /**
         * Ordered semantic result produced beside @ref ticket by acquisition.
         *
         * Packet kernels normally consume only the successful ticket. Keeping
         * the paired status address in the same typed binding lets a terminal
         * activation failure preserve the exact RCU acquisition evidence
         * without a diagnostic D2H copy or a second host-side epoch mirror.
         */
        const DeviceMoEOverlayEpochStatus *status = nullptr;
        /** Exact logical expert geometry shared by both banks. */
        std::uint32_t expert_count = 0u;

        /** @return Whether every captured address and the geometry are complete. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return banks[0].valid() && banks[1].valid() && ticket && status &&
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
        /** Exact follower-local ticket and durable placement banks. */
        MoEOverlayRoutePlacementDeviceBinding placement{};
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
            return packet.valid() && placement.valid() &&
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
        /** Follower `[compact_row * top_k + slot, d_model]` contribution bank. */
        const float *local_canonical_route_contributions_fp32 = nullptr;
        MoEOverlayMappedDispatchDeviceView dispatch{}; ///< Source row identities/counts.
        MoEOverlayMappedReturnDeviceView returned{}; ///< Shared canonical destination.
        MoEOverlayActivationEpochControl *control = nullptr; ///< Exact mapped lane control.
        MoEOverlayActivationDeviceEpochGrant *grant = nullptr; ///< Follower-local epoch state.
        std::int32_t physical_rows = 0; ///< Captured follower row capacity.
        std::uint32_t stage_ordinal = 0u; ///< Ordered stage in the transaction.
        std::int32_t model_layer_index = -1; ///< Expected layer identity.

        /** @return Whether setup supplied matching dispatch/return geometry. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return local_canonical_route_contributions_fp32 &&
                   dispatch.valid() &&
                   returned.valid() && control && grant && physical_rows > 0 &&
                   dispatch.d_model == returned.d_model &&
                   static_cast<std::size_t>(physical_rows) <=
                       dispatch.row_capacity &&
                   dispatch.entry_capacity <=
                       returned.route_slot_capacity &&
                   model_layer_index >= 0;
        }
    };

    /**
     * @brief Captured continuation launch materializing one sparse route return.
     *
     * This operation performs no reduction.  It authenticates one lane and
     * copies each returned contribution into its original route slot in the
     * continuation-owned canonical bank.  A later canonical reducer is the
     * sole authority for top-k arithmetic.
     */
    struct MoEOverlayActivationReturnConsumeLaunch
    {
        MoEOverlayMappedDispatchDeviceView dispatch{}; ///< Original dispatch row order.
        MoEOverlayMappedReturnDeviceView returned{}; ///< Shared compact return source.
        const MoEOverlayActivationEpochControl *control = nullptr; ///< Exact mapped control.
        MoEOverlayActivationDeviceEpochGrant *grant = nullptr; ///< Continuation-local epoch state.
        /** Continuation `[physical_rows * top_k, d_model]` route bank. */
        float *canonical_route_contributions_fp32 = nullptr;
        std::int32_t physical_rows = 0; ///< Captured destination row capacity.
        std::uint32_t stage_ordinal = 0u; ///< Ordered stage in the transaction.
        std::int32_t model_layer_index = -1; ///< Expected layer identity.

        /** @return Whether setup supplied complete deterministic fold state. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return dispatch.valid() && returned.valid() &&
                   dispatch.d_model == returned.d_model && control && grant &&
                   canonical_route_contributions_fp32 && physical_rows > 0 &&
                   static_cast<std::size_t>(physical_rows) *
                           static_cast<std::size_t>(dispatch.top_k) <=
                       returned.route_slot_capacity &&
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
     * @brief Parallel one-row return acquisition and route-slot materialization.
     *
     * One block authenticates each independent lane.  A second kernel copies
     * disjoint original route rows into the continuation-owned canonical bank.
     * No participant-level arithmetic occurs at this boundary.
     */
    struct MoEOverlayActivationSingleRowReturnBatchLaunch
    {
        /** Device array in planner-canonical participant order. */
        const MoEOverlayActivationSingleRowReturnConsumeLaunch *lanes = nullptr;
        std::int32_t *lane_valid = nullptr; ///< `[lane_count]`, written by validation.
        /** Continuation `[top_k, d_model]` canonical route bank. */
        float *canonical_route_contributions_fp32 = nullptr;
        std::uint32_t lane_count = 0u; ///< Positive topology-derived lane count.
        std::int32_t d_model = 0; ///< Exact one-row width shared by every lane.
        std::int32_t top_k = 0; ///< Exact original route width.

        /** @return Whether setup supplied complete persistent batch storage. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return lanes && lane_valid &&
                   canonical_route_contributions_fp32 && lane_count > 0u &&
                   d_model > 0 && top_k > 0;
        }
    };

    /**
     * @brief Validate and materialize a topology-sized multi-row return batch.
     *
     * Every descriptor is prepared once in planner order. Validation remains
     * lane-parallel, while the payload kernel copies disjoint original route
     * rows. The later route reducer—not this transport stage—owns arithmetic.
     */
    struct MoEOverlayActivationMultiRowReturnBatchLaunch
    {
        /** Device array in planner-canonical participant order. */
        const MoEOverlayActivationReturnConsumeLaunch *lanes = nullptr;
        /** `[lane_count]`; set only after the lane descriptor is authenticated. */
        std::int32_t *lane_valid = nullptr;
        /** Continuation canonical route bank. */
        float *canonical_route_contributions_fp32 = nullptr;
        std::uint32_t lane_count = 0u; ///< Positive topology-derived lane count.
        std::int32_t physical_rows = 0; ///< Captured destination row capacity.
        std::int32_t d_model = 0; ///< Exact output width shared by every lane.
        std::int32_t top_k = 0; ///< Exact original route width.

        /** @return Whether setup supplied complete persistent batch storage. */
        [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr bool valid()
            const noexcept
        {
            return lanes && lane_valid &&
                   canonical_route_contributions_fp32 && lane_count > 0u &&
                   physical_rows > 1 && d_model > 0 && top_k > 0;
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
     * The result covers row ids, CSR offsets, expert ids, route weights, both
     * route-slot identities, and FP32 hidden rows. An empty `(0, 0)` packet intentionally has zero payload
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
                               (3u * sizeof(std::int32_t) + sizeof(float)) ||
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
            live_entries *
            (3u * sizeof(std::int32_t) + sizeof(float));
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
     * @brief Compute live canonical route-return bytes without padding.
     * An empty return intentionally has zero bytes and still advances its
     * timeline. For a non-empty return, zero means invalid geometry/overflow.
     */
    [[nodiscard]] LLAMINAR_MOE_PACKET_HD constexpr std::uint64_t
    moeOverlayReturnPayloadBytes(
        std::uint64_t live_entries,
        std::uint32_t d_model) noexcept
    {
        if (d_model == 0u)
            return 0u;
        if (live_entries == 0u)
            return 0u;
        constexpr std::uint64_t maximum = ~std::uint64_t{0};
        const std::uint64_t contribution_bytes =
            static_cast<std::uint64_t>(d_model) * sizeof(float);
        if (contribution_bytes == 0u ||
            live_entries > maximum / contribution_bytes)
        {
            return 0u;
        }
        return live_entries * contribution_bytes;
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
