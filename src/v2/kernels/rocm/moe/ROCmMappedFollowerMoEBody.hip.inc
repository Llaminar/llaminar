/**
 * @file ROCmMappedFollowerMoEBody.hip.inc
 * @brief Single-kernel resident ROCm mapped-overlay follower megakernel body.
 *
 * Included by ROCmMappedFollowerMoEShard0.hip and ROCmMappedFollowerMoEShard1.hip.
 * Reuses the proven 0-spill bank-conflict-free AS3 LDS Q8 decoder primitives from
 * ROCmFastFullNativeHCFFNBody.hip.inc together with the canonical mapped-activation
 * packet protocol from MoEOverlayActivationPacketDevice.inl.
 */
#pragma once

#include "ROCmMappedFollowerMoE.h"
#include "kernels/rocm/gemm/ROCmFastFullNativeHCFFNBody.hip.inc"
#include "kernels/common/MoEOverlayActivationPacketDevice.inl"

namespace llaminar2::rocm::mapped_follower_moe
{
    using namespace llaminar2::rocm::fast_full_hc_ffn;

    /** @brief Compact static LDS held by each CTA during follower execution. */
    struct MappedFollowerStaticLds
    {
        MappedFollowerScratchHeader header;
    };
    static_assert(sizeof(MappedFollowerStaticLds) <= 512u);

    /** @brief Saturating 64-bit atomic add matching DeviceMoEOverlayServiceTelemetryDevice.inl. */
    __device__ __forceinline__ bool followerSaturatingAtomicAdd(
        std::uint64_t *destination,
        std::uint64_t increment) noexcept
    {
        auto *const address = reinterpret_cast<unsigned long long *>(destination);
        unsigned long long observed = *address;
        while (true)
        {
            constexpr unsigned long long maximum = ~0ull;
            const bool overflow = increment > maximum - observed;
            const unsigned long long desired = overflow ? maximum : observed + increment;
            const unsigned long long prior = atomicCAS(address, observed, desired);
            if (prior == observed)
                return !overflow;
            observed = prior;
        }
    }

    /** @brief Phase-local kernarg field loader for arg 0 (MappedFollowerMoEArgs). */
#define LLAMINAR_FOLLOWER_FIELD(Field) \
    phaseKernargField<MappedFollowerMoEArgs, decltype(MappedFollowerMoEArgs::Field), \
        offsetof(MappedFollowerMoEArgs, Field)>()

    /** @brief Convert HIP wall_clock64() delta ticks to nanoseconds via 1-insn Q32 mulhi (zero __udivdi3). */
    __device__ __forceinline__ std::uint64_t followerTicksToNs(
        std::uint64_t ticks,
        std::uint64_t ns_per_tick_q32) noexcept
    {
        if (ns_per_tick_q32 == 0u)
            return 0u;
        const std::uint64_t hi = __umul64hi(ticks, ns_per_tick_q32);
        const std::uint64_t lo = ticks * ns_per_tick_q32;
        return ((hi << 32) | (lo >> 32)) + ((lo & 0xFFFFFFFFull) != 0ull ? 1ull : 0ull);
    }

    /**
     * @brief Lightweight granted-epoch failure publisher avoiding 192B identity snapshot.
     *
     * Stage 0 admission/validation runs in the dedicated 1-wave
     * `acquireMappedFollowerStage0GrantKernel` prior to Layer 0 so this 1024-thread
     * megakernel never allocates SGPRs for the 512B cold control handshake.
     */
    __device__ __forceinline__ void followerPublishFailureGranted(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationDeviceEpochGrant *grant,
        MoEOverlayActivationOperation operation,
        MoEOverlayActivationStatusCode code,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index,
        std::uint64_t observed_timeline,
        std::uint32_t failure_diagnostic = 0u,
        std::uint32_t failure_auxiliary = 0u) noexcept
    {
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        if (grant && grant->state == moe_activation_packet_device::raw(MoEOverlayActivationEndpointState::Aborted))
            return;
        if (grant)
        {
            grant->state = moe_activation_packet_device::raw(MoEOverlayActivationEndpointState::Aborted);
            grant->code = moe_activation_packet_device::raw(code);
        }
        auto *status = &control->follower_status;
        status->digest = MoEOverlayActivationDigest{
            .low = moe_activation_packet_device::loadPeerPublished(&control->identity.digest.low),
            .high = moe_activation_packet_device::loadPeerPublished(&control->identity.digest.high),
        };
        status->generation = moe_activation_packet_device::loadPeerPublished(&control->identity.epoch_generation);
        status->observed_timeline = observed_timeline;
        status->operation = moe_activation_packet_device::raw(operation);
        status->code = moe_activation_packet_device::raw(code);
        status->last_model_layer = model_layer_index;
        status->failure_diagnostic = failure_diagnostic;
        status->failure_auxiliary = failure_auxiliary;
        status->last_consumed_stage = grant ? grant->last_consumed_stage : -1;
        status->last_published_stage = grant ? grant->last_published_stage : -1;
        status->published_payload_bytes = grant ? grant->published_payload_bytes : 0u;
        status->published_shared_hidden_bytes = grant ? grant->published_shared_hidden_bytes : 0u;
        status->published_live_rows = grant ? grant->published_live_rows : 0u;
        status->published_live_entries = grant ? grant->published_live_entries : 0u;
        status->published_stage_count = grant ? grant->published_stage_count : 0u;
        __threadfence_system();
        status->state = moe_activation_packet_device::raw(MoEOverlayActivationEndpointState::Aborted);
    }

    /** @brief Validate local active grant state (already admitted by Stage 0 grant kernel). */
    __device__ __forceinline__ bool followerGrantActive(
        const MoEOverlayActivationEpochControl *control,
        const MoEOverlayActivationDeviceEpochGrant *grant,
        std::uint32_t stage_ordinal,
        std::int32_t physical_rows) noexcept
    {
        return control && grant && physical_rows > 0 &&
               grant->digest.valid() && grant->generation != 0u &&
               grant->placement_epoch != 0u &&
               grant->stage_count > stage_ordinal &&
               grant->physical_rows == physical_rows &&
               grant->endpoint == moe_activation_packet_device::raw(MoEOverlayActivationEndpoint::Follower) &&
               grant->graph_role != MoEOverlayInferenceGraphRole::None &&
               grant->state == moe_activation_packet_device::raw(MoEOverlayActivationEndpointState::Active) &&
               grant->code == moe_activation_packet_device::raw(MoEOverlayActivationStatusCode::Success);
    }

    /** @brief Transient Phase-0 LDS staging for parallel PCIe metadata burst (lives in free panel LDS). */
    struct alignas(16) Phase0BurstLds
    {
        static constexpr unsigned kDescWords =
            sizeof(MoEOverlayActivationPayloadDescriptor) / sizeof(std::uint64_t);
        static_assert(kDescWords <= 12u);
        std::uint64_t desc_words[kDescWords];
        std::uint64_t dispatch_timeline;
        std::int32_t row_id0;
        std::int32_t entry_offsets[2];
        std::int32_t expert_ids[16];
        float route_weights[16];
        std::int32_t original_slots[16];
        std::int32_t compact_slots[16];
    };
    static_assert(sizeof(Phase0BurstLds) <= 512u);

    /** @brief Protocol-exact single-row dispatch validation consuming the parallel LDS PCIe burst. */
    __device__ __forceinline__ void followerValidateDispatchFromBurst(
        const MoEOverlayActivationDispatchConsumeLaunch &launch,
        LdsPtr<const Phase0BurstLds> burst) noexcept
    {
        *launch.active_row_count_device = 0;
        auto *control = const_cast<MoEOverlayActivationEpochControl *>(launch.control);
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation = MoEOverlayActivationOperation::ConsumeDispatch;
        if (!followerGrantActive(control, launch.grant, launch.stage_ordinal, launch.physical_rows))
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal, launch.model_layer_index, 0u);
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u ? -1 : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_consumed_stage != expected_previous)
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal, launch.model_layer_index, 0u);
            return;
        }
        const std::uint64_t timeline = moeOverlayActivationLeasedTimelineValue(
            moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const std::uint32_t bank = moeOverlayActivationBufferIndex(launch.stage_ordinal);
        std::uint64_t dispatch_timeline = burst->dispatch_timeline;
        if (dispatch_timeline < timeline)
        {
            dispatch_timeline =
                moe_activation_packet_device::acquireGrantedDispatchPublication(control, bank, timeline);
        }
        if (dispatch_timeline == kMoEOverlayActivationAbortTimeline)
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::PeerAborted,
                launch.stage_ordinal, launch.model_layer_index, dispatch_timeline);
            return;
        }
        std::uint64_t desc_tmp[Phase0BurstLds::kDescWords];
#pragma unroll
        for (unsigned w = 0u; w < Phase0BurstLds::kDescWords; ++w)
            desc_tmp[w] = burst->desc_words[w];
        MoEOverlayActivationPayloadDescriptor descriptor{};
        __builtin_memcpy(&descriptor, desc_tmp, sizeof(descriptor));

        bool valid = dispatch_timeline == timeline &&
                     moe_activation_packet_device::validDescriptor(
                         descriptor, *launch.grant, launch.stage_ordinal, launch.model_layer_index, timeline) &&
                     descriptor.live_rows <= static_cast<std::uint64_t>(launch.physical_rows) &&
                     descriptor.live_rows <= launch.packet.row_capacity &&
                     descriptor.live_rows <= 1u &&
                     descriptor.live_entries <= launch.packet.entry_capacity &&
                     descriptor.live_entries <= 16u &&
                     descriptor.live_entries <= descriptor.live_rows * static_cast<std::uint64_t>(launch.packet.top_k) &&
                     descriptor.payload_bytes == moeOverlayDispatchPayloadBytes(
                         descriptor.live_rows, descriptor.live_entries,
                         static_cast<std::uint32_t>(launch.packet.d_model));
        std::int32_t prior_offset = 0;
        if (valid && burst->entry_offsets[0] != 0)
            valid = false;
        if (valid && descriptor.live_rows == 1u)
        {
            const std::int32_t row_id = burst->row_id0;
            const std::int32_t begin = burst->entry_offsets[0];
            const std::int32_t end = burst->entry_offsets[1];
            if (row_id < 0 || row_id >= launch.physical_rows ||
                begin != 0 || end <= 0 || end > launch.packet.top_k ||
                end > static_cast<std::int32_t>(descriptor.live_entries))
            {
                valid = false;
            }
            else
            {
                std::int32_t prior_original_slot = row_id * launch.packet.top_k - 1;
                for (std::int32_t entry = 0; entry < end; ++entry)
                {
                    const std::int32_t original_slot = burst->original_slots[entry];
                    const std::int32_t compact_slot = burst->compact_slots[entry];
                    const std::int32_t original_row = original_slot >= 0 ? original_slot / launch.packet.top_k : -1;
                    const std::int32_t original_route = original_slot >= 0 ? original_slot % launch.packet.top_k : -1;
                    if (burst->expert_ids[entry] < 0 ||
                        !isfinite(burst->route_weights[entry]) ||
                        original_row != row_id || original_route < 0 || original_route >= launch.packet.top_k ||
                        original_slot <= prior_original_slot || compact_slot != entry)
                    {
                        valid = false;
                        break;
                    }
                    prior_original_slot = original_slot;
                }
                prior_offset = end;
            }
        }
        if (valid && prior_offset != static_cast<std::int32_t>(descriptor.live_entries))
            valid = false;
        if (!valid)
        {
            std::uint32_t diag = 0u;
            if (dispatch_timeline != timeline) diag |= (1u << 0);
            if (!moe_activation_packet_device::sameDigest(descriptor.digest, launch.grant->digest)) diag |= (1u << 1);
            if (descriptor.timeline != timeline) diag |= (1u << 2);
            if (descriptor.placement_epoch != launch.grant->placement_epoch) diag |= (1u << 3);
            if (descriptor.stage_ordinal != launch.stage_ordinal) diag |= (1u << 4);
            if (descriptor.model_layer_index != launch.model_layer_index) diag |= (1u << 5);
            if (descriptor.live_rows > 1u || descriptor.live_entries > 16u) diag |= (1u << 6);
            if (descriptor.payload_bytes != moeOverlayDispatchPayloadBytes(
                    descriptor.live_rows, descriptor.live_entries,
                    static_cast<std::uint32_t>(launch.packet.d_model))) diag |= (1u << 7);
            if (burst->entry_offsets[0] != 0) diag |= (1u << 8);
            if (prior_offset != static_cast<std::int32_t>(descriptor.live_entries)) diag |= (1u << 9);
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal, launch.model_layer_index, dispatch_timeline,
                diag, static_cast<std::uint32_t>(descriptor.timeline));
            return;
        }
        *launch.active_row_count_device = static_cast<std::int32_t>(descriptor.live_rows);
        launch.grant->live_rows = static_cast<std::uint32_t>(descriptor.live_rows);
        launch.grant->live_entries = static_cast<std::uint32_t>(descriptor.live_entries);
        launch.grant->single_row_id = descriptor.live_rows == 1u ? burst->row_id0 : -1;
        if (descriptor.live_rows == 1u && descriptor.live_entries > 0u)
        {
            moe_activation_packet_device::publishSuccess(
                control, launch.grant, endpoint, operation, launch.stage_ordinal, launch.model_layer_index,
                timeline, /*consumed=*/true, /*published=*/false);
        }
        else
        {
            // Empty stage immediately packs return metadata on the next line; update grant state
            // locally and avoid a duplicate 15-field PCIe write to control->follower_status.
            launch.grant->last_consumed_stage = static_cast<std::int32_t>(launch.stage_ordinal);
        }
    }

    /** @brief Protocol-exact dispatch validation using the already-acquired follower grant. */
    __device__ __forceinline__ void followerValidateDispatchGranted(
        const MoEOverlayActivationDispatchConsumeLaunch &launch) noexcept
    {
        *launch.active_row_count_device = 0;
        auto *control = const_cast<MoEOverlayActivationEpochControl *>(launch.control);
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation = MoEOverlayActivationOperation::ConsumeDispatch;
        if (!followerGrantActive(control, launch.grant, launch.stage_ordinal, launch.physical_rows))
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal, launch.model_layer_index, 0u);
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u ? -1 : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_consumed_stage != expected_previous)
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal, launch.model_layer_index, 0u);
            return;
        }
        const std::uint64_t timeline = moeOverlayActivationLeasedTimelineValue(
            moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const std::uint32_t bank = moeOverlayActivationBufferIndex(launch.stage_ordinal);
        const std::uint64_t dispatch_timeline =
            moe_activation_packet_device::acquireGrantedDispatchPublication(control, bank, timeline);
        if (dispatch_timeline == kMoEOverlayActivationAbortTimeline)
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::PeerAborted,
                launch.stage_ordinal, launch.model_layer_index, dispatch_timeline);
            return;
        }
        const auto descriptor = moe_activation_packet_device::snapshotPeerPublished(
            &control->buffers[bank].dispatch_descriptor);
        bool valid = dispatch_timeline == timeline &&
                     moe_activation_packet_device::validDescriptor(
                         descriptor, *launch.grant, launch.stage_ordinal, launch.model_layer_index, timeline) &&
                     descriptor.live_rows <= static_cast<std::uint64_t>(launch.physical_rows) &&
                     descriptor.live_rows <= launch.packet.row_capacity &&
                     descriptor.live_entries <= launch.packet.entry_capacity &&
                     descriptor.live_entries <= descriptor.live_rows * static_cast<std::uint64_t>(launch.packet.top_k) &&
                     descriptor.payload_bytes == moeOverlayDispatchPayloadBytes(
                         descriptor.live_rows, descriptor.live_entries,
                         static_cast<std::uint32_t>(launch.packet.d_model));
        std::int32_t prior_row = -1;
        std::int32_t prior_offset = 0;
        if (valid && moe_activation_packet_device::loadPeerPublished(launch.packet.entry_offsets) != 0)
            valid = false;
        for (std::uint64_t row = 0u; valid && row < descriptor.live_rows; ++row)
        {
            const std::int32_t row_id = moe_activation_packet_device::loadPeerPublished(launch.packet.row_ids + row);
            const std::int32_t begin = moe_activation_packet_device::loadPeerPublished(launch.packet.entry_offsets + row);
            const std::int32_t end = moe_activation_packet_device::loadPeerPublished(launch.packet.entry_offsets + row + 1u);
            if (row_id <= prior_row || row_id < 0 || row_id >= launch.physical_rows ||
                begin != prior_offset || end <= begin || end - begin > launch.packet.top_k ||
                end > static_cast<std::int32_t>(descriptor.live_entries))
            {
                valid = false;
                break;
            }
            for (std::int32_t entry = begin; entry < end; ++entry)
            {
                const std::int32_t original_slot =
                    moe_activation_packet_device::loadPeerPublished(launch.packet.original_route_slots + entry);
                const std::int32_t compact_slot =
                    moe_activation_packet_device::loadPeerPublished(launch.packet.compact_route_slots + entry);
                const std::int32_t expected_compact_slot =
                    static_cast<std::int32_t>(row) * launch.packet.top_k + (entry - begin);
                const std::int32_t original_row = original_slot >= 0 ? original_slot / launch.packet.top_k : -1;
                const std::int32_t original_route = original_slot >= 0 ? original_slot % launch.packet.top_k : -1;
                const std::int32_t prior_original_slot =
                    entry == begin
                        ? row_id * launch.packet.top_k - 1
                        : moe_activation_packet_device::loadPeerPublished(launch.packet.original_route_slots + entry - 1);
                if (moe_activation_packet_device::loadPeerPublished(launch.packet.expert_ids + entry) < 0 ||
                    !isfinite(moe_activation_packet_device::loadPeerPublished(launch.packet.route_weights + entry)) ||
                    original_row != row_id || original_route < 0 || original_route >= launch.packet.top_k ||
                    original_slot <= prior_original_slot || compact_slot != expected_compact_slot)
                {
                    valid = false;
                    break;
                }
            }
            prior_row = row_id;
            prior_offset = end;
        }
        if (valid && prior_offset != static_cast<std::int32_t>(descriptor.live_entries))
            valid = false;
        if (!valid)
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal, launch.model_layer_index, dispatch_timeline);
            return;
        }
        *launch.active_row_count_device = static_cast<std::int32_t>(descriptor.live_rows);
        launch.grant->live_rows = static_cast<std::uint32_t>(descriptor.live_rows);
        launch.grant->live_entries = static_cast<std::uint32_t>(descriptor.live_entries);
        launch.grant->single_row_id =
            descriptor.live_rows == 1u ? moe_activation_packet_device::loadPeerPublished(launch.packet.row_ids) : -1;
        moe_activation_packet_device::publishSuccess(
            control, launch.grant, endpoint, operation, launch.stage_ordinal, launch.model_layer_index,
            timeline, /*consumed=*/true, /*published=*/false);
    }

    /** @brief Protocol-exact return metadata pack using the already-acquired follower grant. */
    __device__ __forceinline__ void followerPackReturnMetadataGranted(
        const MoEOverlayActivationReturnPackLaunch &launch) noexcept
    {
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation = MoEOverlayActivationOperation::PublishReturn;
        if (!followerGrantActive(control, launch.grant, launch.stage_ordinal, launch.physical_rows))
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal, launch.model_layer_index, 0u);
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u ? -1 : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_consumed_stage != static_cast<std::int32_t>(launch.stage_ordinal) ||
            launch.grant->last_published_stage != expected_previous)
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal, launch.model_layer_index, 0u);
            return;
        }
        const std::uint64_t timeline = moeOverlayActivationLeasedTimelineValue(
            moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const std::uint32_t bank = moeOverlayActivationBufferIndex(launch.stage_ordinal);
        if (timeline == 0u ||
            launch.grant->live_entries > launch.returned.route_slot_capacity ||
            launch.grant->live_entries > launch.grant->live_rows * static_cast<std::uint64_t>(launch.dispatch.top_k))
        {
            followerPublishFailureGranted(
                control, launch.grant, operation, MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal, launch.model_layer_index, 0u);
            return;
        }
        const MoEOverlayActivationPayloadDescriptor descriptor{
            .digest = launch.grant->digest,
            .timeline = timeline,
            .placement_epoch = launch.grant->placement_epoch,
            .live_rows = launch.grant->live_rows,
            .live_entries = launch.grant->live_entries,
            .payload_bytes = moeOverlayReturnPayloadBytes(
                launch.grant->live_entries, static_cast<std::uint32_t>(launch.returned.d_model)),
            .stage_ordinal = launch.stage_ordinal,
            .model_layer_index = launch.model_layer_index,
        };
        control->buffers[bank].return_descriptor = descriptor;
        moe_activation_packet_device::publishSuccess(
            control, launch.grant, endpoint, operation, launch.stage_ordinal, launch.model_layer_index,
            timeline, /*consumed=*/true, /*published=*/true,
            descriptor.payload_bytes, descriptor.live_rows, descriptor.live_entries);
    }

    /**
     * @brief Acquire validated complete-expert execution bank and update follower histograms.
     *
     * Runs once on Block 0, Wave 15, Lane 0 while Waves 0..9 load and quantize the mapped
     * hidden activation vector. Supports dynamic RCU bank flips via `overlay_epoch_ticket`
     * without requiring intermediate-shard bindings.
     */
    __device__ __forceinline__ void followerAcquireBankAndFilterRoutes(
        LdsPtr<MappedFollowerScratchHeader> hdr,
        DeviceMoELayerRuntimeView *runtime,
        unsigned expected_experts,
        bool observe_histogram)
    {
        if (__builtin_expect(!runtime, 0))
            __builtin_trap();
        const auto *rt = asGlobal(runtime);
        const std::uint32_t active_bank = rt->active_bank;
        const std::uint32_t active_epoch = rt->active_epoch;
        const std::uint32_t rt_experts = rt->expert_count;
        const std::uint32_t llep_transient = rt->current_batch_llep_transient_bank_active;
        const auto *ticket = rt->overlay_epoch_ticket;
        const auto *overlay_banks = rt->overlay_placement_banks;
        auto *hist_banks = rt->runtime_histogram_banks;
        const std::uint32_t *hist_active = rt->runtime_histogram_active_bank;

        if (__builtin_expect(active_epoch == 0u || rt_experts != expected_experts || llep_transient != 0u, 0))
            __builtin_trap();

        std::uint32_t bank_idx = active_bank;
        const DeviceMoEPlacementBankView *banks_ptr = overlay_banks ? overlay_banks : runtime->banks;
        std::uint64_t expected_bank_epoch = 0u;
        if (ticket)
        {
            const auto *gt = asGlobal(ticket);
            const std::uint64_t t_epoch = gt->epoch;
            const std::uint64_t t_selector = gt->selector;
            const std::uint32_t t_bank = static_cast<std::uint32_t>(t_selector & 1u);
            const std::uint64_t t_gen = t_selector >> 1u;
            if (__builtin_expect(t_epoch == 0u || t_gen == 0u || t_bank > 1u || !banks_ptr, 0))
                __builtin_trap();
            bank_idx = t_bank;
            expected_bank_epoch = t_epoch;
        }
        else if (__builtin_expect(bank_idx > 1u || !banks_ptr, 0))
        {
            __builtin_trap();
        }

        const auto *bank_g = asGlobal(banks_ptr + bank_idx);
        const std::uint32_t bank_epoch = bank_g->epoch;
        const std::uint32_t bank_experts = bank_g->expert_count;
        if (__builtin_expect(
                bank_experts != expected_experts ||
                    (expected_bank_epoch != 0u && std::uint64_t(bank_epoch) != expected_bank_epoch),
                0))
        {
            __builtin_trap();
        }

        std::uint64_t *h_sel = runtime->decode_histogram;
        std::uint64_t *h_loc = runtime->decode_local_histogram;
        if (hist_banks || hist_active)
        {
            if (__builtin_expect(!hist_banks || !hist_active, 0))
                __builtin_trap();
            const std::uint32_t state = *asGlobal(hist_active);
            if (__builtin_expect(!llaminar2::moe_runtime_abi::validHistogramWriterState(state), 0))
                __builtin_trap();
            if ((state & llaminar2::moe_runtime_abi::kHistogramWriterQuarantineBit) != 0u)
            {
                h_sel = nullptr;
                h_loc = nullptr;
            }
            else
            {
                auto &writer = asGlobal(hist_banks)[state & llaminar2::moe_runtime_abi::kHistogramWriterBankMask];
                h_sel = writer.selected[0];
                h_loc = writer.local[0];
            }
        }

        const auto *experts_table = bank_g->experts;
        const auto *local_mask = bank_g->local_compute_mask;
        hdr->experts_table = experts_table;
        hdr->local_compute_mask = local_mask;
        hdr->expert_count = bank_experts;

        const unsigned le = hdr->live_entries;
        std::uint32_t local_live = 0u;
#pragma unroll 1
        for (unsigned i = 0u; i < le; ++i)
        {
            const int eid = hdr->experts[i];
            const bool is_local = (eid >= 0 && static_cast<unsigned>(eid) < bank_experts &&
                                   globalWeight<std::uint8_t>(local_mask + eid) != 0u);
            if (is_local)
            {
                ++local_live;
                if (observe_histogram)
                {
                    if (h_sel)
                        __hip_atomic_fetch_add(
                            asGlobal(reinterpret_cast<unsigned long long *>(h_sel + eid)),
                            1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    if (h_loc)
                        __hip_atomic_fetch_add(
                            asGlobal(reinterpret_cast<unsigned long long *>(h_loc + eid)),
                            1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
            }
            else
            {
                hdr->experts[i] = -1;
            }
        }
        hdr->local_live_count = local_live;
    }

    /**
     * @brief Single-flight scalar load of Complete-expert Gate+Up descriptors on __GFX9__.
     *
     * Identical to `makeFastGateUpDescPair` except validates `DeviceMoEProjectionSet::Complete`
     * (whole experts owned by apportioned overlay followers).
     */
    template<std::uint8_t GateCB, bool AllowExpanded>
    __device__ __forceinline__ FastGateUpDescPair followerGateUpDescPair(
        const DeviceMoEExpertDescriptorView *experts_table,
        const std::uint8_t *local_mask,
        unsigned expert,
        unsigned policy_parts,
        std::uint32_t execution_mask,
        unsigned reduction)
    {
        constexpr bool kDynamic = (GateCB == kFastDynamicCodebook);
        using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<kDynamic ? 19u : GateCB>;
        constexpr bool kNeedMins = kDynamic || Traits::requires_secondary_metadata || AllowExpanded;
        constexpr bool kNeedEmins = kDynamic || (GateCB == 10u);
        FastGateUpDescPair out{};
        out.gate.parts = 1u;
        out.gate.partition_blocks = reduction >> 5;
        out.up.parts = 1u;
        out.up.partition_blocks = reduction >> 5;
        const unsigned u_exp = waveUniform(expert);
        const auto *row = waveUniform(experts_table) + u_exp;
        const auto *mask_base = waveUniform(local_mask);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
        using ScalarWords4 = unsigned __attribute__((ext_vector_type(4)));
        using ScalarWords2 = unsigned __attribute__((ext_vector_type(2)));
        const std::uint64_t row_addr = reinterpret_cast<std::uint64_t>(row);
        const std::uint64_t mask_word_addr = reinterpret_cast<std::uint64_t>(mask_base + (u_exp & ~3u));
        ScalarWords4 g_ps, u_ps;
        ScalarWords2 g_m{}, u_m{}, g_em{}, u_em{}, meta_words;
        unsigned g_cb_word = 0u, u_cb_word = 0u, mask_word = 0u;
        if constexpr (kNeedEmins)
        {
            asm volatile("s_load_dword %9, %11, 0\n\t"
                         "s_load_dwordx2 %8, %10, 232\n\t"
                         "s_load_dwordx4 %0, %10, 0\n\t"
                         "s_load_dwordx2 %2, %10, 16\n\t"
                         "s_load_dwordx2 %4, %10, 24\n\t"
                         "s_load_dword %6, %10, 44\n\t"
                         "s_load_dwordx4 %1, %10, 56\n\t"
                         "s_load_dwordx2 %3, %10, 72\n\t"
                         "s_load_dwordx2 %5, %10, 80\n\t"
                         "s_load_dword %7, %10, 100\n\t"
                         "s_waitcnt lgkmcnt(0)"
                         : "=&s"(g_ps), "=&s"(u_ps), "=&s"(g_m), "=&s"(u_m), "=&s"(g_em), "=&s"(u_em),
                           "=&s"(g_cb_word), "=&s"(u_cb_word), "=&s"(meta_words), "=&s"(mask_word)
                         : "s"(row_addr), "s"(mask_word_addr)
                         : "memory");
        }
        else if constexpr (kNeedMins)
        {
            asm volatile("s_load_dword %7, %9, 0\n\t"
                         "s_load_dwordx2 %6, %8, 232\n\t"
                         "s_load_dwordx4 %0, %8, 0\n\t"
                         "s_load_dwordx2 %2, %8, 16\n\t"
                         "s_load_dword %4, %8, 44\n\t"
                         "s_load_dwordx4 %1, %8, 56\n\t"
                         "s_load_dwordx2 %3, %8, 72\n\t"
                         "s_load_dword %5, %8, 100\n\t"
                         "s_waitcnt lgkmcnt(0)"
                         : "=&s"(g_ps), "=&s"(u_ps), "=&s"(g_m), "=&s"(u_m),
                           "=&s"(g_cb_word), "=&s"(u_cb_word), "=&s"(meta_words), "=&s"(mask_word)
                         : "s"(row_addr), "s"(mask_word_addr)
                         : "memory");
        }
        else
        {
            asm volatile("s_load_dword %5, %7, 0\n\t"
                         "s_load_dwordx2 %4, %6, 232\n\t"
                         "s_load_dwordx4 %0, %6, 0\n\t"
                         "s_load_dword %2, %6, 44\n\t"
                         "s_load_dwordx4 %1, %6, 56\n\t"
                         "s_load_dword %3, %6, 100\n\t"
                         "s_waitcnt lgkmcnt(0)"
                         : "=&s"(g_ps), "=&s"(u_ps),
                           "=&s"(g_cb_word), "=&s"(u_cb_word), "=&s"(meta_words), "=&s"(mask_word)
                         : "s"(row_addr), "s"(mask_word_addr)
                         : "memory");
        }
        const std::uint8_t is_local = static_cast<std::uint8_t>((mask_word >> ((u_exp & 3u) * 8u)) & 0xFFu);
        if (is_local == 0u)
            return out;
        const unsigned wt_fmt = meta_words[0] & 0xFFFFu;
        const unsigned proj_set = meta_words[1];
        if (__builtin_expect(proj_set != static_cast<unsigned>(DeviceMoEProjectionSet::CompleteExpert), 0))
            __builtin_trap();
        if (__builtin_expect(wt_fmt != static_cast<unsigned>(DeviceMoEWeightFormat::NativeQuantised), 0))
            return out;
        out.local_native = true;
        const auto decode_one = [&](const ScalarWords4 &ps, const ScalarWords2 &mn, const ScalarWords2 &em,
                                    unsigned cb_word, FastPhaseProjection &r) {
            const auto *pay = reinterpret_cast<const std::uint8_t *>(
                std::uint64_t(ps[0]) | (std::uint64_t(ps[1]) << 32));
            const auto *sc = reinterpret_cast<const std::uint16_t *>(
                std::uint64_t(ps[2]) | (std::uint64_t(ps[3]) << 32));
            if (__builtin_expect(!pay || !sc, 0))
                return;
            const std::uint8_t cb = static_cast<std::uint8_t>(cb_word & 0xFFu);
            if (__builtin_expect(!residentMoEFFNDecoderAllows(execution_mask, cb), 0))
                __builtin_trap();
            r.parts = policy_parts;
            r.partition_blocks = ((reduction >> 5) + policy_parts - 1u) / policy_parts;
            r.payload = pay;
            r.scales = sc;
            if constexpr (kNeedMins)
                r.mins = reinterpret_cast<const std::uint16_t *>(std::uint64_t(mn[0]) | (std::uint64_t(mn[1]) << 32));
            if constexpr (kNeedEmins)
                r.emins = reinterpret_cast<const std::uint32_t *>(std::uint64_t(em[0]) | (std::uint64_t(em[1]) << 32));
            r.codebook = cb;
            r.ready = true;
        };
        decode_one(g_ps, g_m, g_em, g_cb_word, out.gate);
        decode_one(u_ps, u_m, u_em, u_cb_word, out.up);
        return out;
#else
        if (globalWeight<std::uint8_t>(mask_base + u_exp) == 0u)
            return out;
        const auto *row_g = asGlobal(row);
        if (static_cast<unsigned>(row_g->projection_set) != static_cast<unsigned>(DeviceMoEProjectionSet::CompleteExpert))
            __builtin_trap();
        if (static_cast<unsigned>(row_g->weight_format) != static_cast<unsigned>(DeviceMoEWeightFormat::NativeQuantised))
            return out;
        out.local_native = true;
        out.gate = makeFastProjectionDesc(&row->gate, policy_parts, execution_mask, reduction);
        out.up = makeFastProjectionDesc(&row->up, policy_parts, execution_mask, reduction);
        return out;
#endif
    }

    /**
     * @brief One-kernel resident ROCm mapped-overlay follower megakernel.
     *
     * Executes dispatch acquire/validate -> single-read PCIe hidden Q8 -> compact
     * paired Gate/Up/SwiGLU -> direct-to-mapped preweighted Down -> return metadata
     * & timeline release -> dynamic service telemetry in ONE resident launch.
     */
    template<std::uint8_t GateCB, std::uint8_t DownCB, bool AllowExpanded = false>
    __global__ __launch_bounds__(1024, 1) void fastMappedFollowerMoEKernel(
        MappedFollowerMoEArgs args,
        unsigned dynamic_bytes,
        NativeEmbeddingIQTables iq_tables)
    {
        __shared__ MappedFollowerStaticLds scratch;
        extern __shared__ __align__(16) unsigned char dynamic_storage[];

        {
            FastWeightedGridBarrier entry_grid{LLAMINAR_FOLLOWER_FIELD(barrier_state)};
            entry_grid.validateEntry();
        }

        // =====================================================================
        // Phase 0 (Block 0 only): Acquire + Parallel PCIe Burst (Metadata + Hidden) + Validate
        // =====================================================================
        if (blockIdx.x == 0u)
        {
            const unsigned tid = threadIdx.x;
            const unsigned H = waveUniform(static_cast<unsigned>(LLAMINAR_FOLLOWER_FIELD(hidden_columns)));
            const unsigned blocks = H >> 5;
            const unsigned hidden_q8_bytes = blocks * static_cast<unsigned>(sizeof(FastQ8Block48));
            auto q8_lds = asLds<FastQ8Block48>(dynamic_storage);
            auto burst_lds = asLds<Phase0BurstLds>(dynamic_storage + hidden_q8_bytes);

            if (tid == 0u)
            {
                std::uint64_t observed = 0u;
                while (true)
                {
                    observed = __hip_atomic_load(
                        args.dispatch_launch.acquire.signal,
                        __ATOMIC_RELAXED,
                        __HIP_MEMORY_SCOPE_SYSTEM);
                    if (observed >= args.dispatch_launch.acquire.value)
                        break;
                    __builtin_amdgcn_s_sleep(1u);
                }
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
                burst_lds->dispatch_timeline = observed;
            }
            workgroupJoin<WorkgroupJoinMemory::GlobalAndLDS>();

            // Two-stage zero-waste PCIe burst:
            // Stage 0a reads ONLY descriptor + entry_offsets[0..1] (~72 B); dispatch_timeline was
            // already acquired by tid==0 above (eliminating a duplicate buffer_invl2 roundtrip!).
            // When live_rows == 0 (empty follower stage), Stage 0b is completely skipped — zero
            // uncached PCIe reads of the 10 KB hidden_rows_fp32 buffer or unused route slots!
            // When live_rows > 0, Waves 0..4 fetch ONLY live_entries route metadata while
            // Waves 5..15 (704 threads >= 640 float4s) fetch & quantize hidden in ONE single pass!
            {
                const unsigned wave = waveUniform(tid >> 6);
                const unsigned lane = lane64();
                const auto &pkt = args.dispatch_launch.packet.packet;
                const unsigned entry_cap = waveUniform(min(16u, static_cast<unsigned>(pkt.entry_capacity)));
                if (wave == 0u)
                {
                    if (lane < Phase0BurstLds::kDescWords)
                    {
                        const std::uint32_t bank =
                            waveUniform(moeOverlayActivationBufferIndex(args.dispatch_launch.packet.stage_ordinal));
                        const auto *control = waveUniform(args.dispatch_launch.packet.control);
                        const auto *desc_words =
                            reinterpret_cast<const std::uint64_t *>(&control->buffers[bank].dispatch_descriptor);
                        burst_lds->desc_words[lane] = moe_activation_packet_device::loadPeerPublished(desc_words + lane);
                    }
                }
                else if (wave == 1u)
                {
                    if (lane < 2u)
                        burst_lds->entry_offsets[lane] = moe_activation_packet_device::loadPeerPublished(pkt.entry_offsets + lane);
                }
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

                constexpr unsigned kLiveRowsWord =
                    offsetof(MoEOverlayActivationPayloadDescriptor, live_rows) / sizeof(std::uint64_t);
                constexpr unsigned kLiveEntriesWord =
                    offsetof(MoEOverlayActivationPayloadDescriptor, live_entries) / sizeof(std::uint64_t);
                const std::uint64_t peek_live_rows = burst_lds->desc_words[kLiveRowsWord];
                const std::uint64_t peek_live_entries = burst_lds->desc_words[kLiveEntriesWord];
                const bool has_live_rows = waveUniform(peek_live_rows > 0u);
                if (has_live_rows)
                {
                    const unsigned live_cap = waveUniform(min(entry_cap, static_cast<unsigned>(peek_live_entries)));
                    if (wave == 0u)
                    {
                        if (lane == 0u)
                            burst_lds->row_id0 = moe_activation_packet_device::loadPeerPublished(pkt.row_ids);
                    }
                    else if (wave == 1u)
                    {
                        if (lane < live_cap)
                            burst_lds->expert_ids[lane] = moe_activation_packet_device::loadPeerPublished(pkt.expert_ids + lane);
                    }
                    else if (wave == 2u)
                    {
                        if (lane < live_cap)
                            burst_lds->route_weights[lane] = moe_activation_packet_device::loadPeerPublished(pkt.route_weights + lane);
                    }
                    else if (wave == 3u)
                    {
                        if (lane < live_cap)
                            burst_lds->original_slots[lane] = moe_activation_packet_device::loadPeerPublished(pkt.original_route_slots + lane);
                    }
                    else if (wave == 4u)
                    {
                        if (lane < live_cap)
                            burst_lds->compact_slots[lane] = moe_activation_packet_device::loadPeerPublished(pkt.compact_route_slots + lane);
                    }
                    else
                    {
                        const auto *hidden4 = reinterpret_cast<const float4 *>(waveUniform(
                            phaseKernargField<MappedFollowerMoEArgs, float *,
                                offsetof(MappedFollowerMoEArgs, dispatch_launch) +
                                offsetof(MoEOverlayActivationSingleRowDispatchConsumeLaunch, packet) +
                                offsetof(MoEOverlayActivationDispatchConsumeLaunch, packet) +
                                offsetof(MoEOverlayMappedDispatchDeviceView, hidden_rows_fp32)>()));
                        for (unsigned quad = tid - 320u; quad < (H >> 2); quad += 704u)
                        {
                            const float4 val = moe_activation_packet_device::loadPeerPublishedFloat4(hidden4 + quad);
                            fastQuantizeFourLds(val, quad, q8_lds);
                        }
                    }
                    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                }
            }

            if (tid == 0u)
            {
                followerValidateDispatchFromBurst(args.dispatch_launch.packet, burst_lds);

                const auto *grant = args.dispatch_launch.packet.grant;
                const bool valid_active =
                    grant->state == static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Active) &&
                    grant->code == static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success) &&
                    grant->live_rows == 1u &&
                    grant->live_entries > 0u;

                if (!valid_active)
                {
                    followerPackReturnMetadataGranted(args.return_launch.packet);
                    __threadfence_system();
                    moe_activation_packet_device::storeSystemRelaxed64(args.return_launch.publication);
                    auto *sample = LLAMINAR_FOLLOWER_FIELD(service_telemetry_sample);
                    if (sample && sample->magic != kDeviceMoEOverlayServiceTelemetryMagic)
                    {
                        sample->magic = kDeviceMoEOverlayServiceTelemetryMagic;
                        sample->version = kDeviceMoEOverlayServiceTelemetryVersion;
                    }
                    scratch.header.live_entries = 0u;
                    asGlobal(LLAMINAR_FOLLOWER_FIELD(scratch_header))->live_entries = 0u;
                }
                else
                {
                    auto *sample = LLAMINAR_FOLLOWER_FIELD(service_telemetry_sample);
                    if (sample)
                    {
                        sample->magic = kDeviceMoEOverlayServiceTelemetryMagic;
                        sample->version = kDeviceMoEOverlayServiceTelemetryVersion;
                        sample->begin_tick = static_cast<std::uint64_t>(wall_clock64());
                        sample->armed = 1u;
                    }
                    const unsigned le = static_cast<unsigned>(grant->live_entries);
                    scratch.header.live_entries = le;
#pragma unroll 1
                    for (unsigned e = 0u; e < le; ++e)
                    {
                        scratch.header.experts[e] = burst_lds->expert_ids[e];
                        scratch.header.weights[e] = burst_lds->route_weights[e];
                        scratch.header.original_slots[e] = burst_lds->original_slots[e];
                    }
                    followerAcquireBankAndFilterRoutes(
                        asLds<MappedFollowerScratchHeader>(&scratch.header),
                        reinterpret_cast<DeviceMoELayerRuntimeView *>(LLAMINAR_FOLLOWER_FIELD(runtime)),
                        static_cast<unsigned>(LLAMINAR_FOLLOWER_FIELD(experts)),
                        LLAMINAR_FOLLOWER_FIELD(observe_histogram) != 0u);
                }
            }
            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

            const unsigned le = waveUniform(scratch.header.live_entries);
            if (le > 0u)
            {
                const unsigned q8_int4s = hidden_q8_bytes >> 4;
                auto *dst_q8 = reinterpret_cast<GlobalPtr<RawInt4>>(asGlobal(LLAMINAR_FOLLOWER_FIELD(scratch_hidden_q8)));
                auto src_q8 = asLds<const RawInt4>(q8_lds);
                for (unsigned i = tid; i < q8_int4s; i += 1024u)
                    dst_q8[i] = src_q8[i];
                constexpr unsigned hdr_int4s = sizeof(MappedFollowerScratchHeader) >> 4;
                if (tid < hdr_int4s)
                {
                    auto *dst_hdr = reinterpret_cast<GlobalPtr<RawInt4>>(asGlobal(LLAMINAR_FOLLOWER_FIELD(scratch_header)));
                    auto src_hdr = asLds<const RawInt4>(&scratch.header);
                    dst_hdr[tid] = src_hdr[tid];
                }
            }
        }

        // Barrier 0: Block 0 published either live_entries==0 (and return!) or L2 Q8 + header.
        FastWeightedGridBarrier{LLAMINAR_FOLLOWER_FIELD(barrier_state)}.sync<0>();

        const unsigned live_entries = waveUniform(
            globalWeight<std::uint32_t>(&LLAMINAR_FOLLOWER_FIELD(scratch_header)->live_entries));
        if (live_entries == 0u)
            return;

        // =====================================================================
        // Phase 1 (All CTAs): Pull L2 Q8 + Header (CTAs 1..N-1) & Compact Paired Gate+Up+SwiGLU
        // =====================================================================
        {
            const unsigned tid = threadIdx.x;
            const unsigned bid = blockIdx.x;
            const unsigned gdim = gridDim.x;
            const unsigned wave = waveUniform(tid >> 6);
            const unsigned lane = lane64();
            const unsigned H = waveUniform(static_cast<unsigned>(LLAMINAR_FOLLOWER_FIELD(hidden_columns)));
            const unsigned F = waveUniform(static_cast<unsigned>(LLAMINAR_FOLLOWER_FIELD(routed_columns)));
            const unsigned blocks = H >> 5;
            const unsigned hidden_q8_bytes = blocks * static_cast<unsigned>(sizeof(FastQ8Block48));
            auto q8_lds = asLds<FastQ8Block48>(dynamic_storage);

            if (bid != 0u)
            {
                const unsigned q8_int4s = hidden_q8_bytes >> 4;
                const auto *src_q8 = reinterpret_cast<const int4 *>(waveUniform(LLAMINAR_FOLLOWER_FIELD(scratch_hidden_q8)));
                auto dst_q8 = asLds<RawInt4>(q8_lds);
                for (unsigned i = tid; i < q8_int4s; i += 1024u)
                    ldsStoreInt4(dst_q8, i, globalWeight<int4>(src_q8 + i));
                constexpr unsigned hdr_int4s = sizeof(MappedFollowerScratchHeader) >> 4;
                if (tid < hdr_int4s)
                {
                    const auto *src_hdr = reinterpret_cast<const int4 *>(waveUniform(LLAMINAR_FOLLOWER_FIELD(scratch_header)));
                    ldsStoreInt4(asLds<RawInt4>(&scratch.header), tid, globalWeight<int4>(src_hdr + tid));
                }
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
            }

            rocm_q8_residual::NativeDecoder decoder(iq_tables);
            float *gate_partials_out = waveUniform(LLAMINAR_FOLLOWER_FIELD(scratch_gate_partials));
            auto panel_lds = asLds<unsigned char>(dynamic_storage) + hidden_q8_bytes;
            const unsigned avail_after_hidden = (dynamic_bytes > hidden_q8_bytes) ? (dynamic_bytes - hidden_q8_bytes) : 0u;
            const unsigned cohorts = min(live_entries, gdim);
            const unsigned cohort = ((bid + 1u) * cohorts - 1u) / gdim;
            const unsigned begin = (cohort * gdim) / cohorts;
            const unsigned end = ((cohort + 1u) * gdim) / cohorts;
            const unsigned owners = end - begin;
            const unsigned rank = bid - begin;
            const unsigned grain = fastRowGrain(F, owners);
            const unsigned chunks = (F + grain - 1u) / grain;
            const unsigned first = ((rank * chunks) / owners) * grain;
            const unsigned last = min(F, (((rank + 1u) * chunks) / owners) * grain);
            if (first < last)
            {
                const unsigned row_in_wave = lane & 3u;
                const unsigned part = lane >> 2;
                const DeviceMoEExpertDescriptorView *experts_table = waveUniform(scratch.header.experts_table);
                const std::uint8_t *local_mask = waveUniform(scratch.header.local_compute_mask);
                const unsigned expert_count = waveUniform(scratch.header.expert_count);
                const unsigned policy_parts = waveUniform(
                    phaseKernargField<MappedFollowerMoEArgs, unsigned,
                        offsetof(MappedFollowerMoEArgs, routed_gate_policy) +
                        offsetof(ResidentMoEPartitionPolicy, maximum)>());
                const std::uint32_t exec_mask = waveUniform(LLAMINAR_FOLLOWER_FIELD(gate_execution_mask));

                const unsigned slab_blocks = (blocks + 15u) & ~15u;
                const unsigned slab_stride_words = slab_blocks * 64u;
                if (__builtin_expect(avail_after_hidden < 2u * slab_stride_words * sizeof(std::uint32_t), 0))
                    __builtin_trap();
                auto g_panel = asLds<std::uint32_t>(panel_lds);
                auto u_panel = g_panel + slab_stride_words;
                auto loc_exp = asLds<const std::int32_t>(scratch.header.experts);

#pragma unroll 1
                for (unsigned k = cohort; k < live_entries; k += cohorts)
                {
                    const int expert = waveUniform(loc_exp[k]);
                    if (__builtin_expect(expert < 0 || static_cast<unsigned>(expert) >= expert_count, 0))
                    {
                        for (unsigned o = first + tid; o < last; o += 1024u)
                            asGlobal(gate_partials_out)[std::size_t(k) * F + o] = 0.0f;
                        continue;
                    }
                    const FastGateUpDescPair pair = followerGateUpDescPair<GateCB, AllowExpanded>(
                        experts_table, local_mask, static_cast<unsigned>(expert), policy_parts, exec_mask, H);
                    if (__builtin_expect(!pair.local_native || (!pair.gate.ready && !pair.up.ready), 0))
                    {
                        for (unsigned o = first + tid; o < last; o += 1024u)
                            asGlobal(gate_partials_out)[std::size_t(k) * F + o] = 0.0f;
                        continue;
                    }
                    const FastPhaseProjection &g_rec = pair.gate;
                    const FastPhaseProjection &u_rec = pair.up;
                    const bool g_ready = waveUniform(g_rec.ready);
                    const bool u_ready = waveUniform(u_rec.ready);
                    const unsigned g_parts = waveUniform(g_rec.parts);
                    const unsigned u_parts = waveUniform(u_rec.parts);
                    const unsigned g_width = waveUniform(g_rec.partition_blocks);
                    const unsigned u_width = waveUniform(u_rec.partition_blocks);
                    const std::uint8_t g_cb = waveUniform(g_rec.codebook);
                    const std::uint8_t u_cb = waveUniform(u_rec.codebook);
                    constexpr bool kDynamicGate = (GateCB == kFastDynamicCodebook);
                    if constexpr (!AllowExpanded && !kDynamicGate)
                    {
                        if (__builtin_expect(!g_ready || !u_ready || g_cb != GateCB || u_cb != GateCB ||
                                             g_parts != u_parts || g_width != u_width, 0))
                            __builtin_trap();
                    }
                    constexpr bool kGateSec = kDynamicGate
                        ? true
                        : rocm_q8_residual::NativeDecoder::CodebookTraits<kDynamicGate ? 19u : GateCB>::requires_secondary_metadata;
                    const bool paired_fast = !kDynamicGate && (!AllowExpanded ||
                        waveUniform(g_ready && u_ready && (g_cb == u_cb) && (g_parts == u_parts) && (g_width == u_width)));

#pragma unroll 1
                    for (unsigned tile = first; tile < last; tile += 64u)
                    {
                        const unsigned count = min(64u, last - tile);
                        if constexpr (!kDynamicGate)
                        {
                            if (paired_fast)
                            {
                                if constexpr (!AllowExpanded)
                                {
                                    fastPublishMetaPanelPair<kGateSec>(
                                        g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                        F, tile, count, tid, blocks, g_panel, u_panel);
                                }
                                else
                                {
                                    if (g_cb == GateCB)
                                        fastPublishMetaPanelPair<kGateSec>(
                                            g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                            F, tile, count, tid, blocks, g_panel, u_panel);
                                    else
                                        fastPublishMetaPanelPair<true>(
                                            g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                            F, tile, count, tid, blocks, g_panel, u_panel);
                                }
                                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                            }
                        }
                        const unsigned wave_row = wave * 4u;
                        if (wave_row < count)
                        {
                            const unsigned local_row = wave_row + row_in_wave;
                            const unsigned n = tile + local_row;
                            float gg = 0.0f, uu = 0.0f;
                            if constexpr (kDynamicGate)
                            {
                                if (local_row < count)
                                {
#pragma unroll 1
                                    for (unsigned role = 0u; role < 2u; ++role)
                                    {
                                        const auto &rec = (role == 0u) ? g_rec : u_rec;
                                        if (rec.ready && part < rec.parts)
                                        {
                                            const unsigned bf = part * rec.partition_blocks;
                                            const unsigned bl = (bf < blocks) ? min(blocks, bf + rec.partition_blocks) : bf;
                                            const float val = fastPartitionDynamic(
                                                waveUniform(rec.codebook), decoder, rec.payload, rec.scales,
                                                rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                            if (role == 0u) gg = val; else uu = val;
                                        }
                                    }
                                }
                            }
                            else if constexpr (!AllowExpanded)
                            {
                                if (local_row < count && part < g_parts)
                                {
                                    const unsigned bf = part * g_width;
                                    const unsigned bl = (bf < blocks) ? min(blocks, bf + g_width) : bf;
                                    fastPairedGateUpPanelPartition<GateCB>(
                                        decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                        n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                                }
                            }
                            else
                            {
                                if (paired_fast)
                                {
                                    if (local_row < count && part < g_parts)
                                    {
                                        const unsigned bf = part * g_width;
                                        const unsigned bl = (bf < blocks) ? min(blocks, bf + g_width) : bf;
                                        if (g_cb == GateCB)
                                            fastPairedGateUpPanelPartition<GateCB>(
                                                decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                                n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                                        else if (g_cb == kNativeQuantisedExpandedInt8MinCodebook)
                                            fastPairedGateUpPanelPartition<kNativeQuantisedExpandedInt8MinCodebook>(
                                                decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                                n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                                        else
                                            __builtin_trap();
                                    }
                                }
                                else if (local_row < count)
                                {
#pragma unroll 1
                                    for (unsigned role = 0u; role < 2u; ++role)
                                    {
                                        const auto &rec = (role == 0u) ? g_rec : u_rec;
                                        if (rec.ready && part < rec.parts)
                                        {
                                            const unsigned bf = part * rec.partition_blocks;
                                            const unsigned bl = (bf < blocks) ? min(blocks, bf + rec.partition_blocks) : bf;
                                            float val = 0.0f;
                                            if (rec.codebook == GateCB)
                                                val = fastPartition<GateCB>(
                                                    decoder, rec.payload, rec.scales, rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                            else if (rec.codebook == kNativeQuantisedExpandedInt8MinCodebook)
                                                val = fastPartition<kNativeQuantisedExpandedInt8MinCodebook>(
                                                    decoder, rec.payload, rec.scales, rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                            else
                                                __builtin_trap();
                                            if (role == 0u) gg = val; else uu = val;
                                        }
                                    }
                                }
                            }
#pragma unroll 1
                            for (unsigned role = 0u; role < 2u; ++role)
                            {
                                const float folded = fastFoldFourColumns(
                                    (role == 0u) ? gg : uu, (role == 0u) ? g_parts : u_parts, lane);
                                if (role == 0u) gg = folded; else uu = folded;
                            }
                            if (lane < 4u && (wave_row + lane) < count)
                            {
                                asGlobal(gate_partials_out)[std::size_t(k) * F + tile + wave_row + lane] =
                                    decode_equivalent::swigluValue(gg, uu);
                            }
                        }
                        if (paired_fast && (tile + 64u < last || k + cohorts < live_entries))
                            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                    }
                }
            }
        }

        // Barrier 1: All live_entries * F SwiGLU activations visible in L2.
        FastWeightedGridBarrier{LLAMINAR_FOLLOWER_FIELD(barrier_state)}.sync<1>();

        // =====================================================================
        // Phase 2 (All CTAs): Compact Activated Q8 + Direct-to-Mapped Preweighted Down
        // =====================================================================
        {
            const unsigned tid = threadIdx.x;
            const unsigned bid = blockIdx.x;
            const unsigned gdim = gridDim.x;
            const unsigned wave = waveUniform(tid >> 6);
            const unsigned lane = lane64();
            const unsigned H = waveUniform(static_cast<unsigned>(LLAMINAR_FOLLOWER_FIELD(hidden_columns)));
            const unsigned F = waveUniform(static_cast<unsigned>(LLAMINAR_FOLLOWER_FIELD(routed_columns)));
            const unsigned le = waveUniform(scratch.header.live_entries);

            const unsigned values = le * F;
            auto d_q8 = asLds<FastQ8Block48>(dynamic_storage);
            const auto *gate_partials4 = reinterpret_cast<const float4 *>(
                waveUniform(LLAMINAR_FOLLOWER_FIELD(scratch_gate_partials)));
            for (unsigned quad = tid; quad < (values >> 2); quad += 1024u)
            {
                const float4 val = globalWeight<float4>(gate_partials4 + quad);
                fastQuantizeFourLds(val, quad, d_q8);
            }
            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

            const auto owned = NativeLayoutDownGeometry::rows(H, bid, gdim);
            if (owned.begin < owned.end)
            {
                rocm_q8_residual::NativeDecoder decoder(iq_tables);
                const DeviceMoEExpertDescriptorView *experts_table = waveUniform(scratch.header.experts_table);
                const unsigned expert_count = waveUniform(scratch.header.expert_count);
                auto loc_exp = asLds<const std::int32_t>(scratch.header.experts);
                auto loc_weights = asLds<const float>(scratch.header.weights);
                auto loc_slots = asLds<const std::int32_t>(scratch.header.original_slots);
                auto *mapped_out = asGlobal(waveUniform(
                    phaseKernargField<MappedFollowerMoEArgs, float *,
                        offsetof(MappedFollowerMoEArgs, return_launch) +
                        offsetof(MoEOverlayActivationSingleRowReturnPackLaunch, packet) +
                        offsetof(MoEOverlayActivationReturnPackLaunch, returned) +
                        offsetof(MoEOverlayMappedReturnDeviceView, canonical_route_contributions_fp32)>()));
                const unsigned slot_cap = waveUniform(static_cast<unsigned>(
                    phaseKernargField<MappedFollowerMoEArgs, std::size_t,
                        offsetof(MappedFollowerMoEArgs, return_launch) +
                        offsetof(MoEOverlayActivationSingleRowReturnPackLaunch, packet) +
                        offsetof(MoEOverlayActivationReturnPackLaunch, returned) +
                        offsetof(MoEOverlayMappedReturnDeviceView, route_slot_capacity)>()));
                const unsigned down_policy_parts = waveUniform(
                    phaseKernargField<MappedFollowerMoEArgs, unsigned,
                        offsetof(MappedFollowerMoEArgs, routed_down_policy) +
                        offsetof(ResidentMoEPartitionPolicy, maximum)>());
                const std::uint32_t down_exec_mask = waveUniform(LLAMINAR_FOLLOWER_FIELD(down_execution_mask));

                const unsigned down_q8_bytes =
                    (((values >> 5) * static_cast<unsigned>(sizeof(FastQ8Block48))) + 63u) & ~63u;
                auto down_stage = asLds<float>(dynamic_storage + down_q8_bytes);
                // Adapt (groups, subs) to actual live_entries so all 16 waves execute Down
                // concurrently even when le=1 (1x16) or le=2 (2x8), preserving exact ascending K order.
                const unsigned sub_shift = waveUniform((le <= 1u) ? 4u : ((le == 2u) ? 3u : 2u));
                const unsigned subs = waveUniform(1u << sub_shift);
                const unsigned groups = waveUniform(16u >> sub_shift);
                const unsigned group = waveUniform(wave >> sub_shift);
                const unsigned sub = waveUniform(wave & (subs - 1u));
                const unsigned u_blocks = waveUniform(F >> 5);
                const bool mapped_out_aligned =
                    waveUniform((reinterpret_cast<std::uintptr_t>(mapped_out) & 15u) == 0u);

#pragma unroll 1
                for (unsigned first = owned.begin; first < owned.end; first += 64u)
                {
                    const unsigned count = waveUniform(min(64u, owned.end - first));
                    const unsigned n = first + lane;
                    const bool use_vec4_store =
                        waveUniform(mapped_out_aligned && count == 64u && ((first & 3u) == 0u) && ((H & 3u) == 0u));

                    // Adaptive multi-wave exact-order Down: `groups` tasks run concurrently
                    // with `subs` waves/task computing `subs` adjacent K-blocks in parallel, then
                    // sub-wave 0 folds them in strict ascending K-block RN order and
                    // flushes 16 coalesced float4 PCIe stores.
#pragma unroll 1
                    for (unsigned base_task = 0u; base_task < le; base_task += groups)
                    {
                        const unsigned task = waveUniform(base_task + group);
                        const bool has_task = waveUniform(task < le);
                        int orig_slot = -1;
                        int expert = -1;
                        float rw = 0.0f;
                        FastPhaseProjection rec{};
                        bool valid_slot = false;
                        bool valid_ready = false;
                        if (has_task)
                        {
                            orig_slot = waveUniform(loc_slots[task]);
                            valid_slot = waveUniform(orig_slot >= 0 && static_cast<unsigned>(orig_slot) < slot_cap);
                            if (valid_slot)
                            {
                                expert = waveUniform(loc_exp[task]);
                                rw = waveUniform(loc_weights[task]);
                                if (waveUniform(expert >= 0 && static_cast<unsigned>(expert) < expert_count))
                                {
                                    rec = makeFastProjectionDesc(
                                        &experts_table[static_cast<unsigned>(expert)].down,
                                        down_policy_parts, down_exec_mask, F);
                                    valid_ready = waveUniform(rec.ready);
                                }
                            }
                        }

                        const std::uint8_t cb = waveUniform(rec.codebook);
                        const unsigned u_parts = waveUniform(rec.parts);
                        const unsigned u_width = waveUniform(rec.partition_blocks);
                        auto r_q8 = d_q8 + ((std::size_t(task) * F) >> 5);
                        float raw = 0.0f;
                        float part_acc = 0.0f;
                        unsigned in_part = 0u;
                        unsigned done_parts = 0u;

                        // When all groups * u_blocks fit in free LDS (<= 20 KB for F=640),
                        // compute all subs-wave K-blocks without inner barriers and perform ONE LDS
                        // barrier + exact ascending RN fold per 64-col tile.
                        // Keep the <255,255> dynamic-switch fallback single-copy for 0 memory spills.
                        const unsigned avail_stage_bytes =
                            (dynamic_bytes > down_q8_bytes) ? (dynamic_bytes - down_q8_bytes) : 0u;
                        bool used_single_barrier = false;
                        if constexpr (DownCB != kFastDynamicCodebook)
                        {
                            const bool single_barrier_stage =
                                waveUniform((std::size_t(groups) * u_blocks * 64u * sizeof(float)) <= avail_stage_bytes);
                            if (single_barrier_stage)
                            {
                                used_single_barrier = true;
                                const unsigned group_base = waveUniform(group * u_blocks * 64u);
                                if (valid_ready && lane < count)
                                {
#pragma unroll 1
                                    for (unsigned b = sub; b < u_blocks; b += subs)
                                    {
                                        const std::size_t idx = static_cast<std::size_t>(b) * H + n;
                                        float contrib = 0.0f;
                                        if (__builtin_expect(cb == DownCB, 1))
                                        {
                                            contrib = fastStreamBlockContributionIdx<DownCB>(
                                                decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                                waveUniform(rec.mins), waveUniform(rec.emins), idx, r_q8 + b);
                                        }
                                        else if constexpr (AllowExpanded)
                                        {
                                            if (cb == kNativeQuantisedExpandedInt8MinCodebook)
                                            {
                                                contrib = fastStreamBlockContributionIdx<kNativeQuantisedExpandedInt8MinCodebook>(
                                                    decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                                    waveUniform(rec.mins), waveUniform(rec.emins), idx, r_q8 + b);
                                            }
                                            else
                                            {
                                                __builtin_trap();
                                            }
                                        }
                                        else
                                        {
                                            __builtin_trap();
                                        }
                                        down_stage[group_base + b * 64u + lane] = contrib;
                                    }
                                }
                                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                                if (sub == 0u && valid_ready && lane < count)
                                {
                                    auto group_stage = down_stage + group_base + lane;
#pragma unroll 1
                                    for (unsigned b = 0u; b < u_blocks; ++b)
                                    {
                                        const float c = group_stage[b * 64u];
                                        part_acc = rocm_native_quantised_accumulate_rn(part_acc, c);
                                        ++in_part;
                                        if (__builtin_expect(in_part == u_width || (b + 1u) == u_blocks, 0))
                                        {
                                            const float pval = device_fp32_contract::multiply(part_acc, 1.0f);
                                            raw = (u_parts == 1u) ? pval : rocm_native_quantised_accumulate_rn(raw, pval);
                                            part_acc = 0.0f;
                                            in_part = 0u;
                                            ++done_parts;
                                        }
                                    }
                                }
                            }
                        }
                        if (!used_single_barrier)
                        {
                        const unsigned fallback_stride = subs * 64u;
#pragma unroll 1
                        for (unsigned b0 = 0u; b0 < u_blocks; b0 += subs)
                        {
                            const unsigned b = waveUniform(b0 + sub);
                            if (valid_ready && b < u_blocks && lane < count)
                            {
                                const std::size_t idx = static_cast<std::size_t>(b) * H + n;
                                float contrib = 0.0f;
                                if constexpr (DownCB == kFastDynamicCodebook)
                                {
                                    contrib = fastStreamBlockContributionIdxDynamic(
                                        cb, decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                        waveUniform(rec.mins), waveUniform(rec.emins), idx, r_q8 + b);
                                }
                                else if (__builtin_expect(cb == DownCB, 1))
                                {
                                    contrib = fastStreamBlockContributionIdx<DownCB>(
                                        decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                        waveUniform(rec.mins), waveUniform(rec.emins), idx, r_q8 + b);
                                }
                                else if constexpr (AllowExpanded)
                                {
                                    if (cb == kNativeQuantisedExpandedInt8MinCodebook)
                                    {
                                        contrib = fastStreamBlockContributionIdx<kNativeQuantisedExpandedInt8MinCodebook>(
                                            decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                            waveUniform(rec.mins), waveUniform(rec.emins), idx, r_q8 + b);
                                    }
                                    else
                                    {
                                        __builtin_trap();
                                    }
                                }
                                else
                                {
                                    __builtin_trap();
                                }
                                down_stage[group * fallback_stride + sub * 64u + lane] = contrib;
                            }
                            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                            if (sub == 0u && valid_ready && lane < count)
                            {
                                const unsigned rem = min(subs, u_blocks - b0);
#pragma unroll 1
                                for (unsigned s = 0u; s < rem; ++s)
                                {
                                    const float c = down_stage[group * fallback_stride + s * 64u + lane];
                                    part_acc = rocm_native_quantised_accumulate_rn(part_acc, c);
                                    ++in_part;
                                    if (__builtin_expect(in_part == u_width || (b0 + s + 1u) == u_blocks, 0))
                                    {
                                        const float pval = device_fp32_contract::multiply(part_acc, 1.0f);
                                        raw = (u_parts == 1u) ? pval : rocm_native_quantised_accumulate_rn(raw, pval);
                                        part_acc = 0.0f;
                                        in_part = 0u;
                                        ++done_parts;
                                    }
                                }
                            }
                            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                        }
                        }

                        if (sub == 0u && valid_slot)
                        {
                            float out_val = 0.0f;
                            if (valid_ready && lane < count)
                            {
                                if (__builtin_expect(done_parts < u_parts, 0))
                                {
#pragma unroll 1
                                    while (done_parts < u_parts)
                                    {
                                        const float z = device_fp32_contract::multiply(0.0f, 1.0f);
                                        raw = (u_parts == 1u) ? z : rocm_native_quantised_accumulate_rn(raw, z);
                                        ++done_parts;
                                    }
                                }
                                out_val = rocm_native_quantised_weight_route_rn(rw, raw);
                            }
                            if (use_vec4_store)
                            {
                                auto wave_stage = down_stage + group * (used_single_barrier ? (u_blocks * 64u) : (subs * 64u));
                                wave_stage[lane] = out_val;
                                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                                if (lane < 16u)
                                {
                                    auto src4 = reinterpret_cast<LdsPtr<const RawInt4>>(wave_stage);
                                    auto *dst4 = reinterpret_cast<RawInt4 *>(
                                        mapped_out + std::size_t(orig_slot) * H + first);
                                    __builtin_nontemporal_store(src4[lane], dst4 + lane);
                                }
                            }
                            else if (lane < count)
                            {
                                __builtin_nontemporal_store(out_val, mapped_out + std::size_t(orig_slot) * H + n);
                            }
                        }
                        if (base_task + groups < le || first + 64u < owned.end)
                            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                    }
                }
            }
        }

        // Coherent nontemporal stores to mapped host RAM retire on each CU's
        // s_waitcnt vmcnt(0) inside Barrier 2; Block 0 then issues ONE system
        // release fence after Barrier 2 instead of an 896-wave L2 flush storm.
        FastWeightedGridBarrier{LLAMINAR_FOLLOWER_FIELD(barrier_state)}.sync<2>();

        // =====================================================================
        // Phase 3 (Block 0, Thread 0): Release Return Packet + Record Telemetry
        // =====================================================================
        if (blockIdx.x == 0u && threadIdx.x == 0u)
        {
            followerPackReturnMetadataGranted(args.return_launch.packet);
            __threadfence_system();
            moe_activation_packet_device::storeSystemRelaxed64(args.return_launch.publication);

            // Off critical path: publish dynamic service telemetry sample (zero __udivdi3).
            auto *layer_tel = LLAMINAR_FOLLOWER_FIELD(service_telemetry_layer);
            auto *sample = LLAMINAR_FOLLOWER_FIELD(service_telemetry_sample);
            if (layer_tel && sample &&
                sample->magic == kDeviceMoEOverlayServiceTelemetryMagic &&
                sample->version == kDeviceMoEOverlayServiceTelemetryVersion &&
                sample->armed == 1u)
            {
                sample->armed = 0u;
                std::uint32_t phase = 0u;
                const auto *role_ptr = LLAMINAR_FOLLOWER_FIELD(service_graph_role);
                if (role_ptr)
                {
                    const std::uint32_t hint = static_cast<std::uint32_t>(
                        moeOverlayServicePhaseHintForGraphRole(*role_ptr));
                    if (hint >= 1u && hint <= kDeviceMoEOverlayServicePhaseCount)
                        phase = hint - 1u;
                }
                const std::uint64_t activations = static_cast<std::uint64_t>(scratch.header.local_live_count);
                sample->previous_local_activations[phase] += activations;
                const std::uint64_t end_tick = static_cast<std::uint64_t>(wall_clock64());
                const std::uint64_t elapsed_ns = (end_tick > sample->begin_tick)
                    ? followerTicksToNs(end_tick - sample->begin_tick,
                                        LLAMINAR_FOLLOWER_FIELD(wall_clock_ns_per_tick_q32))
                    : 0u;
                if (activations > 0u && elapsed_ns > 0u)
                {
                    auto &cell = layer_tel[phase];
                    const bool d_ok = followerSaturatingAtomicAdd(&cell.total_nanoseconds, elapsed_ns);
                    const bool a_ok = followerSaturatingAtomicAdd(&cell.activation_count, activations);
                    const bool s_ok = followerSaturatingAtomicAdd(&cell.sample_count, 1u);
                    if (!d_ok || !a_ok || !s_ok)
                        atomicExch(&cell.overflowed, 1u);
                }
                else
                {
                    atomicAdd(reinterpret_cast<unsigned long long *>(&layer_tel[phase].dropped_samples), 1ull);
                }
            }
        }
    }

    /** @brief Extract `(gate_cb, down_cb, allow_expanded)` from physical execution masks. */
    inline bool decodeFollowerKey(
        std::uint32_t gate_mask,
        std::uint32_t down_mask,
        std::uint8_t &gate_cb,
        std::uint8_t &down_cb,
        bool &allow_expanded) noexcept
    {
        bool g_exp = false, d_exp = false;
        if (!extractSingleCodebookMask(gate_mask, gate_cb, g_exp) ||
            !extractSingleCodebookMask(down_mask, down_cb, d_exp))
            return false;
        allow_expanded = g_exp || d_exp;
        return true;
    }

} // namespace llaminar2::rocm::mapped_follower_moe
