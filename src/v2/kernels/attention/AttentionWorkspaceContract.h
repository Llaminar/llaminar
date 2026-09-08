/**
 * @file AttentionWorkspaceContract.h
 * @brief Canonical physical workspace ABI for CPU and captured GPU attention.
 *
 * CPU, CUDA, ROCm, graph construction, and metadata-only memory admission all
 * consume this contract. It is the sole arithmetic owner for attention's
 * persistent partial summaries, device-owned replay parameters, and optional
 * FP32 K/V conversion pair. GPU callers may raise the partial-buffer floors to
 * a backend-selected prefill envelope, but no caller may reproduce descriptor,
 * worker-partition, or request/context sizing rules independently.
 */

#pragma once

#include "AttentionDeviceParams.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2::attention_workspace
{
    /** Stable split-attention output buffer name. */
    inline constexpr const char *kPartialOutput = "attn_partial_output";
    /** Stable split-attention maximum-score buffer name. */
    inline constexpr const char *kPartialM = "attn_partial_m";
    /** Stable split-attention log-sum-exp buffer name. */
    inline constexpr const char *kPartialL = "attn_partial_l";
    /** Stable device-owned replay-parameter buffer name. */
    inline constexpr const char *kDeviceParams = "attn_device_params";
    /** Stable FP32 key-conversion buffer name. */
    inline constexpr const char *kKeyTemporaryFP32 = "attn_k_tmp_fp32";
    /** Stable FP32 value-conversion buffer name. */
    inline constexpr const char *kValueTemporaryFP32 = "attn_v_tmp_fp32";

    /** Maximum split-decode partition count supported by CUDA and ROCm. */
    inline constexpr int kMaximumDecodeSplits = 32;

    /**
     * @brief Complete immutable geometry of one attention workspace member.
     *
     * The compact query cardinality and request cardinality are deliberately
     * distinct.  Grouped MTP rows enlarge split-decode partials, while prompt
     * rows never multiply request-major K/V conversion storage.
     */
    struct Geometry
    {
        int compact_query_rows = 0; ///< Decode/verifier rows retained together.
        int request_count = 0; ///< Independent request-major K/V banks.
        int local_query_heads = 0; ///< Query heads owned by this participant.
        int local_kv_heads = 0; ///< K/V heads owned or replicated locally.
        int head_dim = 0; ///< Elements in one attention head.
        int context_rows = 0; ///< Complete configured K/V horizon.
        int decode_splits = kMaximumDecodeSplits; ///< Split-decode capacity.
        std::size_t partial_output_floor_bytes = 0u; ///< Prefill-policy floor.
        std::size_t partial_m_floor_bytes = 0u; ///< Prefill-policy floor.
        std::size_t partial_l_floor_bytes = 0u; ///< Prefill-policy floor.
        bool include_device_params = true; ///< Kernel consumes replay params.
        bool include_fp32_kv_conversion = true; ///< Kernel converts K/V itself.
    };

    /**
     * @brief Complete immutable geometry of the CPU parallel-summary arena.
     *
     * CPU attention partitions K/V work across the exact OpenMP team selected
     * during process bootstrap. Each `(query row, head)` owns one producer slot
     * per possible worker cohort plus one deterministic merged slot. Keeping
     * the worker count explicit lets preflight and the runtime kernel consume
     * one formula without consulting mutable process state in the planner.
     */
    struct CPUParallelGeometry
    {
        int compact_query_rows = 0; ///< Retained decode/verifier query rows.
        int local_query_heads = 0; ///< Query heads executed by this CPU stage.
        int head_dim = 0; ///< Logical elements in one attention head.
        int worker_count = 0; ///< Configured physical-core OpenMP team size.
    };

    /**
     * @brief Multiply physical cardinalities with overflow checking.
     * @param left First extent.
     * @param right Second extent.
     * @param contribution Human-readable allocation component.
     * @return Exact product.
     * @throws std::overflow_error when the product does not fit in size_t.
     */
    inline std::size_t checkedProduct(
        std::size_t left,
        std::size_t right,
        std::string_view contribution)
    {
        if (left != 0u &&
            right > std::numeric_limits<std::size_t>::max() / left)
        {
            throw std::overflow_error(
                "Attention workspace overflow for " +
                std::string(contribution));
        }
        return left * right;
    }

    /**
     * @brief Return bytes in each request-major FP32 K/V conversion buffer.
     * @param geometry Complete attention workspace geometry.
     * @return Exact bytes independently required by K and by V.
     */
    inline std::size_t fp32KVConversionBufferBytes(
        const Geometry &geometry)
    {
        const std::size_t request_context = checkedProduct(
            static_cast<std::size_t>(geometry.request_count),
            static_cast<std::size_t>(geometry.context_rows),
            "request/context conversion cardinality");
        const std::size_t local_width = checkedProduct(
            static_cast<std::size_t>(geometry.local_kv_heads),
            static_cast<std::size_t>(geometry.head_dim),
            "local K/V row width");
        return checkedProduct(
            checkedProduct(
                request_context,
                local_width,
                "K/V conversion elements"),
            sizeof(float),
            "FP32 K/V conversion bytes");
    }

    /**
     * @brief Build the exact mergeable workspace descriptors for attention.
     * @param geometry Complete compact, request, and prefill-policy geometry.
     * @return Stable-name requirements consumed by the serial-family planner.
     * @throws std::invalid_argument when required geometry is non-positive.
     *
     * Split-decode buffers retain the larger of the compact verifier envelope
     * and a backend-selected context-parallel prefill floor.  This matters when
     * prefill policy chooses direct query parallelism and reports zero partial
     * bytes: decode still owns the compact buffers in the same captured family.
     */
    inline WorkspaceRequirements requirements(const Geometry &geometry)
    {
        if (geometry.compact_query_rows <= 0 || geometry.request_count <= 0 ||
            geometry.local_query_heads <= 0 || geometry.local_kv_heads <= 0 ||
            geometry.head_dim <= 0 || geometry.context_rows <= 0 ||
            geometry.decode_splits <= 0)
        {
            throw std::invalid_argument(
                "Attention workspace requires positive compact, request, head, context, and split geometry");
        }

        const std::size_t compact_heads = checkedProduct(
            static_cast<std::size_t>(geometry.compact_query_rows),
            static_cast<std::size_t>(geometry.local_query_heads),
            "compact query heads");
        const std::size_t compact_partitions = checkedProduct(
            compact_heads,
            static_cast<std::size_t>(geometry.decode_splits),
            "compact split partitions");
        const std::size_t compact_output = checkedProduct(
            checkedProduct(
                compact_partitions,
                static_cast<std::size_t>(geometry.head_dim),
                "compact split output elements"),
            sizeof(float),
            "compact split output bytes");
        const std::size_t compact_summary = checkedProduct(
            compact_partitions,
            sizeof(float),
            "compact split summary bytes");

        WorkspaceRequirements result;
        result.buffers.emplace_back(
            kPartialOutput,
            std::max(compact_output, geometry.partial_output_floor_bytes),
            256u,
            true);
        result.buffers.emplace_back(
            kPartialM,
            std::max(compact_summary, geometry.partial_m_floor_bytes),
            256u,
            true);
        result.buffers.emplace_back(
            kPartialL,
            std::max(compact_summary, geometry.partial_l_floor_bytes),
            256u,
            true);

        if (geometry.include_device_params)
        {
            result.buffers.emplace_back(
                kDeviceParams,
                sizeof(attention::AttentionDeviceParams) *
                    static_cast<std::size_t>(
                        attention::kMaxGroupedVerifierAttentionRows),
                256u,
                true);
        }
        if (geometry.include_fp32_kv_conversion)
        {
            const std::size_t conversion_bytes =
                fp32KVConversionBufferBytes(geometry);
            result.buffers.emplace_back(
                kKeyTemporaryFP32, conversion_bytes, 256u, true);
            result.buffers.emplace_back(
                kValueTemporaryFP32, conversion_bytes, 256u, true);
        }
        return result;
    }

    /**
     * @brief Build the exact CPU attention parallel-summary requirements.
     * @param geometry Retained rows, local heads, head width, and worker team.
     * @return Stable-name requirements consumed by the serial-family planner.
     * @throws std::invalid_argument when any required extent is non-positive.
     *
     * A producer cohort contains at most one worker per local head before a
     * second worker starts another K/V partition for that head. The ceiling
     * division therefore gives the maximum producer slots for any head. The
     * final slot is the deterministic ascending-order merge destination.
     */
    inline WorkspaceRequirements cpuParallelRequirements(
        const CPUParallelGeometry &geometry)
    {
        if (geometry.compact_query_rows <= 0 ||
            geometry.local_query_heads <= 0 || geometry.head_dim <= 0 ||
            geometry.worker_count <= 0)
        {
            throw std::invalid_argument(
                "CPU attention workspace requires positive row, head, width, and worker geometry");
        }

        const std::size_t rows =
            static_cast<std::size_t>(geometry.compact_query_rows);
        const std::size_t heads =
            static_cast<std::size_t>(geometry.local_query_heads);
        const std::size_t workers =
            static_cast<std::size_t>(geometry.worker_count);
        const std::size_t producer_slots =
            (workers + heads - 1u) / heads;
        const std::size_t slots_per_row = producer_slots + 1u;
        const std::size_t partial_slots = checkedProduct(
            checkedProduct(rows, heads, "CPU query rows and heads"),
            slots_per_row,
            "CPU producer and merge slots");
        const std::size_t padded_head_dim =
            (static_cast<std::size_t>(geometry.head_dim) + 15u) & ~15u;
        const std::size_t partial_output_bytes = checkedProduct(
            checkedProduct(
                partial_slots,
                padded_head_dim,
                "CPU partial slots and padded head width"),
            sizeof(float),
            "CPU partial output bytes");
        const std::size_t partial_summary_bytes = checkedProduct(
            partial_slots,
            sizeof(float),
            "CPU partial summary bytes");

        WorkspaceRequirements result;
        result.buffers.emplace_back(
            kPartialOutput, partial_output_bytes, 64u, true);
        result.buffers.emplace_back(
            kPartialM, partial_summary_bytes, 64u, true);
        result.buffers.emplace_back(
            kPartialL, partial_summary_bytes, 64u, true);
        return result;
    }

    /**
     * @brief Resize runtime-declared control/conversion descriptors exactly.
     * @param requirements Backend kernel requirements to update in place.
     * @param geometry Exact stage request and configured-cache geometry.
     *
     * A backend that does not declare these optional descriptors remains
     * unchanged.  Declared descriptors are resized from this contract so stage
     * code cannot recreate the request/context multiplication independently.
     */
    inline void applyExactControlAndConversionGeometry(
        WorkspaceRequirements &requirements,
        const Geometry &geometry)
    {
        const std::size_t conversion_bytes =
            fp32KVConversionBufferBytes(geometry);
        for (auto &buffer : requirements.buffers)
        {
            if (buffer.name == kDeviceParams)
            {
                buffer.size_bytes =
                    sizeof(attention::AttentionDeviceParams) *
                    static_cast<std::size_t>(
                        attention::kMaxGroupedVerifierAttentionRows);
            }
            else if (buffer.name == kKeyTemporaryFP32 ||
                     buffer.name == kValueTemporaryFP32)
            {
                buffer.size_bytes = conversion_bytes;
            }
        }
    }
} // namespace llaminar2::attention_workspace
