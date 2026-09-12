/**
 * @file DeviceMoERebalancePublication.h
 * @brief Authenticate the device-owned prepare-to-publish handoff.
 *
 * An apply kernel fills an unpublished RCU bank. Its commands are still live
 * until the epoch finalizer publishes that bank. This shared, allocation-free
 * predicate is used by CUDA, HIP and device-free adversarial tests; neither a
 * copied payload nor a successful apply alone certifies committed placement.
 */
#pragma once

#include "DeviceMoERebalanceABI.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_PUBLICATION_HD __host__ __device__ inline
#else
#define LLAMINAR_MOE_PUBLICATION_HD inline
#endif

namespace llaminar2::moe_rebalance_publication
{
    /** Wire value of the non-recyclable, prepared-but-unpublished wave state. */
    inline constexpr uint32_t kPreparedForPublication = 7u;

    /**
     * @brief Validate the exact retained wave named by a completed apply.
     * @param controller Device wave owner, never a reconstructed host mirror.
     * @param headers Complete retained command-header array.
     * @param count Physical extent of that array (one or two).
     * @param apply Result authored by the same wave's apply kernel.
     * @return True only for one complete, healthy, still-retained transaction.
     *
     * Template parameters admit the minimal CUDA/HIP ABI views and the public
     * CPU test records without pulling tensor or MPI headers into GPU code.
     * Check the subscript before dereferencing; stale epochs and partial layer
     * families must never retire another wave's command buffer.
     */
    template <class Controller, class Header, class Apply>
    LLAMINAR_MOE_PUBLICATION_HD bool preparedWaveMatches(
        const Controller *controller, const Header *headers,
        uint32_t count, const Apply *apply) noexcept
    {
        constexpr uint32_t magic = 0x4d4f4552u;
        if (!controller || !headers || !apply || count == 0u || count > 2u ||
            controller->magic != magic ||
            controller->version != moe_rebalance_abi::kVersion ||
            controller->last_error_code != 0u || controller->wave_count < count ||
            controller->wave_count > 2u || controller->participant_count == 0u ||
            controller->participant_id >= controller->participant_count ||
            apply->magic != magic || apply->version != moe_rebalance_abi::kVersion ||
            apply->status_code != 0u || apply->changed_layers == 0u ||
            apply->transaction_wave_index >= count || apply->transaction_epoch == 0u)
            return false;
        const auto &wave = controller->waves[apply->transaction_wave_index];
        const auto &header = headers[apply->transaction_wave_index];
        return wave.magic == magic && wave.version == moe_rebalance_abi::kVersion &&
               wave.state == kPreparedForPublication && wave.error_code == 0u &&
               wave.epoch == apply->transaction_epoch && wave.planned_layer_count > 0u &&
               wave.applied_layer_count == wave.planned_layer_count &&
               header.magic == magic && header.version == moe_rebalance_abi::kVersion &&
               header.epoch == wave.epoch && header.command_count > 0u &&
               header.command_count <= header.command_capacity &&
               header.command_count == wave.command_count &&
               header.command_count == apply->transaction_command_count &&
               header.participant_id == controller->participant_id &&
               header.participant_count == controller->participant_count;
    }
} // namespace llaminar2::moe_rebalance_publication

#undef LLAMINAR_MOE_PUBLICATION_HD
