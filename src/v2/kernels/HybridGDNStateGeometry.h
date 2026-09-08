/**
 * @file HybridGDNStateGeometry.h
 * @brief Canonical logical and physical geometry for hybrid GDN live state.
 *
 * Weight slicing, cache construction, prefix serialization, and physical
 * memory admission must agree on the exact modulo-linked GDN head assignment.
 * This value type resolves that assignment once and owns every byte formula
 * derived from it.  Runtime allocators and setup-time estimators consume the
 * same methods, so neither side can invent a different bank count, alignment,
 * or tensor-parallel division.
 */

#pragma once

#include "config/GDNHeadAssignment.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /**
     * @brief Fully resolved GDN state shape for one execution participant.
     *
     * Counts ending in `_floats` describe FP32 elements because recurrent and
     * short-convolution state is FP32 regardless of KV-cache precision.  The
     * integer representation matches the kernel ABI; construction rejects any
     * model geometry that cannot be represented without overflow.
     */
    struct HybridGDNStateGeometry final
    {
        int full_key_heads = 0;   ///< Global query/key head count.
        int full_value_heads = 0; ///< Global value/recurrent head count.
        int local_key_heads = 0;  ///< Participant-local query/key heads.
        int local_value_heads = 0; ///< Participant-local value heads.
        int d_k = 0;              ///< Per-head query/key state dimension.
        int d_v = 0;              ///< Per-head value state dimension.
        int full_qkv_dim = 0;     ///< Full Q/K/V convolution row width.
        int local_qkv_dim = 0;    ///< Participant-local convolution row width.
        int full_conv_state_floats = 0; ///< Full short-convolution bank.
        int local_conv_state_floats = 0; ///< Local short-convolution bank.
        int full_recurrence_state_floats = 0; ///< Full recurrent matrix bank.
        int local_recurrence_state_floats = 0; ///< Local recurrent matrix bank.

        /** Stable byte alignment used by HybridGDNDeviceStateArena. */
        static constexpr std::size_t kDeviceBankAlignment = 256u;

        /**
         * @brief Resolve one validated model/participant geometry.
         *
         * @param attention_heads Global attention partition width.
         * @param local_head_start First global attention head owned locally.
         * @param local_attention_heads Locally owned attention heads. Zero
         *        denotes an unsharded participant that owns all heads.
         * @param group_count Global GDN query/key heads; zero uses attention heads.
         * @param time_step_rank Global GDN value heads; zero uses key heads.
         * @param state_size Per-head GDN state dimension.
         * @param inner_size Global value projection width; zero derives
         *        `value_heads * state_size`.
         * @param conv_kernel_size Causal convolution kernel width.
         * @throws std::invalid_argument for incomplete or non-integral geometry.
         * @throws std::overflow_error when the kernel ABI cannot represent it.
         */
        [[nodiscard]] static HybridGDNStateGeometry resolve(
            int attention_heads,
            int local_head_start,
            int local_attention_heads,
            int group_count,
            int time_step_rank,
            int state_size,
            int inner_size,
            int conv_kernel_size)
        {
            if (attention_heads <= 0 || state_size <= 0 ||
                conv_kernel_size <= 0)
            {
                throw std::invalid_argument(
                    "Hybrid GDN state geometry requires positive attention heads, state size, and convolution width");
            }
            if (group_count < 0 || time_step_rank < 0 || inner_size < 0)
            {
                throw std::invalid_argument(
                    "Hybrid GDN state geometry received a negative model dimension");
            }

            HybridGDNStateGeometry result;
            result.full_key_heads =
                group_count > 0 ? group_count : attention_heads;
            result.full_value_heads =
                time_step_rank > 0 ? time_step_rank : result.full_key_heads;
            result.d_k = state_size;
            result.d_v = state_size;

            /*
             * Resolve the modulo-linked K/V ownership even when unsharded.
             * This validates the global repeat factor before any state or
             * prepared weight can materialize.
             */
            const int effective_local_heads =
                local_attention_heads > 0
                    ? local_attention_heads
                    : attention_heads;
            if (local_head_start < 0 || effective_local_heads <= 0 ||
                local_head_start > attention_heads - effective_local_heads)
            {
                throw std::invalid_argument(
                    "Hybrid GDN local attention-head interval is invalid");
            }
            if (effective_local_heads == attention_heads &&
                local_head_start != 0)
            {
                throw std::invalid_argument(
                    "Unsharded hybrid GDN geometry must begin at attention head zero");
            }

            const GDNHeadAssignment assignment =
                GDNHeadAssignment::fromPartition(
                    result.full_key_heads,
                    result.full_value_heads,
                    local_head_start,
                    effective_local_heads,
                    attention_heads);
            result.local_key_heads = assignment.localKeyHeads();
            result.local_value_heads = assignment.localValueHeads();

            const std::size_t full_value_dim =
                inner_size > 0
                    ? static_cast<std::size_t>(inner_size)
                    : checkedProduct(
                          static_cast<std::size_t>(result.full_value_heads),
                          static_cast<std::size_t>(state_size),
                          "full value width");
            const std::size_t local_value_numerator = checkedProduct(
                full_value_dim,
                static_cast<std::size_t>(result.local_value_heads),
                "local value width");
            if (local_value_numerator %
                    static_cast<std::size_t>(result.full_value_heads) !=
                0u)
            {
                throw std::invalid_argument(
                    "Hybrid GDN inner width cannot be divided exactly across local value heads");
            }
            const std::size_t local_value_dim =
                local_value_numerator /
                static_cast<std::size_t>(result.full_value_heads);

            const std::size_t full_key_dim = checkedProduct(
                static_cast<std::size_t>(result.full_key_heads),
                static_cast<std::size_t>(state_size),
                "full key width");
            const std::size_t local_key_dim = checkedProduct(
                static_cast<std::size_t>(result.local_key_heads),
                static_cast<std::size_t>(state_size),
                "local key width");
            result.full_qkv_dim = checkedKernelInt(
                checkedSum(
                    checkedProduct(full_key_dim, 2u, "full Q/K width"),
                    full_value_dim,
                    "full Q/K/V width"),
                "full Q/K/V width");
            result.local_qkv_dim = checkedKernelInt(
                checkedSum(
                    checkedProduct(local_key_dim, 2u, "local Q/K width"),
                    local_value_dim,
                    "local Q/K/V width"),
                "local Q/K/V width");

            const std::size_t history =
                static_cast<std::size_t>(conv_kernel_size - 1);
            result.full_conv_state_floats = checkedKernelInt(
                checkedProduct(
                    static_cast<std::size_t>(result.full_qkv_dim),
                    history,
                    "full convolution state"),
                "full convolution state");
            result.local_conv_state_floats = checkedKernelInt(
                checkedProduct(
                    static_cast<std::size_t>(result.local_qkv_dim),
                    history,
                    "local convolution state"),
                "local convolution state");
            result.full_recurrence_state_floats = checkedKernelInt(
                checkedProduct(
                    checkedProduct(
                        static_cast<std::size_t>(result.full_value_heads),
                        static_cast<std::size_t>(state_size),
                        "full recurrence rows"),
                    static_cast<std::size_t>(state_size),
                    "full recurrence state"),
                "full recurrence state");
            result.local_recurrence_state_floats = checkedKernelInt(
                checkedProduct(
                    checkedProduct(
                        static_cast<std::size_t>(result.local_value_heads),
                        static_cast<std::size_t>(state_size),
                        "local recurrence rows"),
                    static_cast<std::size_t>(state_size),
                    "local recurrence state"),
                "local recurrence state");
            return result;
        }

        /** @return Whether local and full banks have identical shapes. */
        [[nodiscard]] bool hasSingleBankGeometry() const noexcept
        {
            return local_conv_state_floats == full_conv_state_floats &&
                   local_recurrence_state_floats ==
                       full_recurrence_state_floats;
        }

        /**
         * @brief Bytes in CPU-owned local state for @p gdn_layers layers.
         */
        [[nodiscard]] std::size_t localPayloadBytes(int gdn_layers) const
        {
            validateLayerCount(gdn_layers);
            const std::size_t floats = checkedSum(
                static_cast<std::size_t>(local_conv_state_floats),
                static_cast<std::size_t>(local_recurrence_state_floats),
                "local GDN layer payload");
            return checkedProduct(
                checkedProduct(
                    static_cast<std::size_t>(gdn_layers),
                    floats,
                    "local GDN payload layers"),
                sizeof(float),
                "local GDN payload bytes");
        }

        /**
         * @brief Bytes serialized from GPU banks into one durable payload.
         *
         * Prefix archives and live MTP rollback checkpoints use the same
         * device-state wire shape. Distinct full banks are included exactly
         * once; equal geometry uses request slot zero as the sole live bank
         * and is serialized once. This method is the sole byte authority for
         * both producers so memory admission cannot price a TP-local subset
         * of the payload that the cache later exports.
         */
        [[nodiscard]] std::size_t deviceSerializedPayloadBytes(
            int gdn_layers) const
        {
            validateLayerCount(gdn_layers);
            std::size_t floats = checkedSum(
                static_cast<std::size_t>(local_conv_state_floats),
                static_cast<std::size_t>(local_recurrence_state_floats),
                "serialized local GDN state");
            if (full_conv_state_floats != local_conv_state_floats)
            {
                floats = checkedSum(
                    floats,
                    static_cast<std::size_t>(full_conv_state_floats),
                    "serialized full convolution state");
            }
            if (full_recurrence_state_floats !=
                local_recurrence_state_floats)
            {
                floats = checkedSum(
                    floats,
                    static_cast<std::size_t>(
                        full_recurrence_state_floats),
                    "serialized full recurrence state");
            }
            return checkedProduct(
                checkedProduct(
                    static_cast<std::size_t>(gdn_layers),
                    floats,
                    "serialized device GDN state layers"),
                sizeof(float),
                "serialized device GDN state bytes");
        }

        /**
         * @brief Exact allocation made by HybridGDNDeviceStateArena.
         *
         * Buffers follow the runtime order for every layer: optional distinct
         * local/full convolution banks, convolution request bank, optional
         * distinct local/full recurrence banks, then recurrence request bank.
         * Each buffer starts at a 256-byte boundary; there is no fictitious
         * trailing pad after the final buffer.
         */
        [[nodiscard]] std::size_t deviceArenaBytes(
            int gdn_layers,
            int request_capacity) const
        {
            validateLayerCount(gdn_layers);
            if (request_capacity <= 0)
            {
                throw std::invalid_argument(
                    "Hybrid GDN device arena requires positive request capacity");
            }

            std::size_t offset = 0u;
            for (int layer = 0; layer < gdn_layers; ++layer)
            {
                appendDeviceKernelBanks(
                    offset,
                    local_conv_state_floats,
                    full_conv_state_floats,
                    request_capacity);
                appendDeviceKernelBanks(
                    offset,
                    local_recurrence_state_floats,
                    full_recurrence_state_floats,
                    request_capacity);
            }
            return offset;
        }

    private:
        /** @brief Multiply byte/element counts without unsigned wraparound. */
        [[nodiscard]] static std::size_t checkedProduct(
            std::size_t left,
            std::size_t right,
            const char *label)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string("Hybrid GDN ") + label + " overflows size_t");
            }
            return left * right;
        }

        /** @brief Add byte/element counts without unsigned wraparound. */
        [[nodiscard]] static std::size_t checkedSum(
            std::size_t left,
            std::size_t right,
            const char *label)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("Hybrid GDN ") + label + " overflows size_t");
            }
            return left + right;
        }

        /** @brief Narrow a validated state count into the existing kernel ABI. */
        [[nodiscard]] static int checkedKernelInt(
            std::size_t value,
            const char *label)
        {
            if (value > static_cast<std::size_t>(
                            std::numeric_limits<int>::max()))
            {
                throw std::overflow_error(
                    std::string("Hybrid GDN ") + label +
                    " exceeds the integer kernel ABI");
            }
            return static_cast<int>(value);
        }

        /** @brief Reject negative layer counts at every public byte formula. */
        static void validateLayerCount(int gdn_layers)
        {
            if (gdn_layers < 0)
            {
                throw std::invalid_argument(
                    "Hybrid GDN layer count cannot be negative");
            }
        }

        /** @brief Align one buffer start using the GPU arena's fixed ABI. */
        [[nodiscard]] static std::size_t alignDeviceOffset(
            std::size_t offset)
        {
            const std::size_t padding =
                (kDeviceBankAlignment -
                 (offset % kDeviceBankAlignment)) %
                kDeviceBankAlignment;
            return checkedSum(offset, padding, "device bank alignment");
        }

        /** @brief Append one runtime buffer to the exact packed allocation. */
        static void appendDeviceBuffer(
            std::size_t &offset,
            std::size_t floats,
            int multiplicity)
        {
            if (floats == 0u)
                return;
            offset = alignDeviceOffset(offset);
            const std::size_t bytes = checkedProduct(
                checkedProduct(
                    floats,
                    static_cast<std::size_t>(multiplicity),
                    "device bank multiplicity"),
                sizeof(float),
                "device bank bytes");
            offset = checkedSum(offset, bytes, "device arena bytes");
        }

        /** @brief Append scalar and request banks for one GDN kernel. */
        static void appendDeviceKernelBanks(
            std::size_t &offset,
            int local_state_floats,
            int full_state_floats,
            int request_capacity)
        {
            if (local_state_floats <= 0)
                return;
            const int effective_full =
                full_state_floats > 0
                    ? full_state_floats
                    : local_state_floats;
            if (effective_full != local_state_floats)
            {
                appendDeviceBuffer(
                    offset,
                    static_cast<std::size_t>(local_state_floats),
                    1);
                appendDeviceBuffer(
                    offset,
                    static_cast<std::size_t>(effective_full),
                    1);
            }
            appendDeviceBuffer(
                offset,
                static_cast<std::size_t>(
                    std::max(local_state_floats, effective_full)),
                request_capacity);
        }
    };
} // namespace llaminar2
