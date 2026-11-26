/**
 * @file ROCmFastFullNativeHCFFN.h
 * @brief Sharded single-codebook-pair fast full HC/FFN kernel dispatch for ROCm.
 *
 * Each sharded specialization compiles only its exact (Gate, Down, Shared)
 * physical decoder pair (plus optional dynamic EXPANDED_INT8_MIN promotion),
 * keeping instruction footprint inside the L1 I-cache, eliminating SGPR spills
 * and generic flat LDS accesses, and covering every quantization codebook and
 * arbitrary ROCm topology without falling back to narrow model-specific layouts.
 */
#pragma once

#include "ROCmFullNativeHCFFN.h"
#include <algorithm>
#include <cstddef>

namespace llaminar2::rocm {

/** @brief Lookup primary and mixed production codebook pairs in Shard 0. */
const void* fastFullNativeHCFFNShard0Function(const ResidentMoEFFNExecutionMasks& execution) noexcept;

/** @brief Lookup homogeneous standard K-quant and legacy codebook pairs in Shard 1a. */
const void* fastFullNativeHCFFNShard1aFunction(const ResidentMoEFFNExecutionMasks& execution) noexcept;

/** @brief Lookup Q2_K/Q8_0 and mixed GGUF codebook pairs in Shard 1b. */
const void* fastFullNativeHCFFNShard1bFunction(const ResidentMoEFFNExecutionMasks& execution) noexcept;

/** @brief Lookup IQ-grid codebook pairs in Shard 2. */
const void* fastFullNativeHCFFNShard2Function(const ResidentMoEFFNExecutionMasks& execution) noexcept;

/**
 * @brief Resolve the exact sharded fast full HC/FFN kernel for sealed execution masks.
 * @param execution Immutable per-role physical codebook execution masks.
 * @return Compiled specialized kernel entry, or nullptr for arbitrary multi-codebook unions.
 */
inline const void* fastFullNativeHCFFNKernelFunction(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    if (const void* fn = fastFullNativeHCFFNShard0Function(execution)) return fn;
    if (const void* fn = fastFullNativeHCFFNShard1aFunction(execution)) return fn;
    if (const void* fn = fastFullNativeHCFFNShard1bFunction(execution)) return fn;
    return fastFullNativeHCFFNShard2Function(execution);
}

/**
 * @brief Check whether cold prefix/suffix geometry satisfies fast resident HC/FFN contracts.
 * @param args Cold-resolved complete full HC/FFN arguments.
 * @param prefix_bytes Cold-derived prefix dynamic LDS extent in bytes.
 * @return True when aligned resident prefix/suffix execution applies for this geometry.
 */
inline bool fastFullNativeHCFFNGeometryEligible(const FullNativeHCFFNArgs& args, std::size_t prefix_bytes) noexcept
{
    const auto& hc = args.prefix.hc;
    const auto& ffn = args.suffix.ffn;
    // Production decode seals zero diagnostic observation demand and direct bank acquisition;
    // observation snapshots and ExternalTables descriptors route to fullNativeHCFFNKernel.
    if (args.prefix.output_demand.bits() != 0u ||
        args.prefix.selected_descriptor_demand != MoESelectedDescriptorDemand::AcquiredBankOnly)
        return false;
    if (prefix_bytes == 0 || prefix_bytes > 32768u || (prefix_bytes & 15u) != 0)
        return false;
    const unsigned H = static_cast<unsigned>(hc.norm.hidden);
    const unsigned R = static_cast<unsigned>(hc.down_weights.n);
    const unsigned K = static_cast<unsigned>(hc.down_weights.k);
    const unsigned parts = static_cast<unsigned>(hc.partitions);
    const unsigned up_groups = static_cast<unsigned>(hc.up.weights.blocks_per_row);
    if (H == 0u || H > 16384u || R == 0u || R > 320u || K == 0u || K > 32768u ||
        (H & 31u) != 0u || (R & 31u) != 0u || (K & 31u) != 0u ||
        parts == 0u || up_groups == 0u || up_groups > 10u)
        return false;
    if (ffn.hidden_columns <= 0 || ffn.routed_columns <= 0 || ffn.shared_columns <= 0 ||
        (ffn.hidden_columns & 31) != 0 || (ffn.routed_columns & 31) != 0 || (ffn.shared_columns & 31) != 0)
        return false;
    if (ffn.top_k <= 0 || ffn.top_k > 16)
        return false;
    if (ffn.routed_gate_policy.maximum == 0u || ffn.routed_gate_policy.maximum > 16u ||
        ffn.shared_gate_policy.maximum == 0u || ffn.shared_gate_policy.maximum > 16u ||
        ffn.routed_down_policy.maximum == 0u || ffn.routed_down_policy.maximum > 16u ||
        ffn.shared_down_policy.maximum == 0u || ffn.shared_down_policy.maximum > 16u)
        return false;
    const std::size_t routes = static_cast<std::size_t>(ffn.top_k);
    const std::size_t hidden_bytes = (std::size_t(H) / 32u) * 48u;
    const std::size_t slab_blocks = ((std::size_t(H) / 32u) + 15u) & ~std::size_t(15u);
    const std::size_t panel_bytes = 2u * slab_blocks * 64u * sizeof(std::uint32_t);
    const std::size_t norm_bytes = (256u + 2u * std::size_t(H)) * sizeof(float);
    if (hidden_bytes + panel_bytes > 60000u || norm_bytes > 60000u)
        return false;
    const std::size_t down_values = routes * static_cast<std::size_t>(ffn.routed_columns) +
                                    static_cast<std::size_t>(ffn.shared_columns);
    const std::size_t down_bytes = (down_values / 32u) * 48u + (routes + 1u) * 64u * sizeof(float);
    if (down_bytes > 60000u)
        return false;
    return true;
}

/**
 * @brief Size dynamic LDS for the fast full HC/FFN megakernel, including its bank-conflict-free
 *        48-byte Q8 block bank, full branch-norm cache, and coalesced Gate/Up metadata panel.
 * @param args Cold-resolved complete full HC/FFN arguments.
 * @param baseline_dynamic_bytes Minimum dynamic LDS required by canonical prefix/suffix.
 * @param max_dynamic_bytes Device LDS capacity minus static shared storage.
 * @return 16-byte-aligned dynamic LDS extent in bytes.
 */
inline std::size_t fastFullNativeHCFFNDynamicBytes(
    const FullNativeHCFFNArgs& args,
    std::size_t baseline_dynamic_bytes,
    std::size_t max_dynamic_bytes) noexcept
{
    const std::size_t H = static_cast<std::size_t>(args.suffix.ffn.hidden_columns);
    const std::size_t routes = static_cast<std::size_t>(args.suffix.ffn.top_k);
    const std::size_t F = static_cast<std::size_t>(args.suffix.ffn.routed_columns);
    const std::size_t S = static_cast<std::size_t>(args.suffix.ffn.shared_columns);
    const std::size_t hidden_bytes = (H / 32u) * 48u;
    const std::size_t slab_blocks = ((H / 32u) + 15u) & ~std::size_t(15u);
    const std::size_t panel_bytes = 2u * slab_blocks * 64u * sizeof(std::uint32_t);
    const std::size_t norm_bytes = ((256u + 2u * H) * sizeof(float) + 15u) & ~std::size_t(15u);
    const std::size_t h2_bytes = ((H * 2u) + 15u) & ~std::size_t(15u);
    const std::size_t down_bytes = (((routes * F + S) / 32u) * 48u + (routes + 1u) * 64u * sizeof(float) + 15u) & ~std::size_t(15u);
    const std::size_t desired = std::max({baseline_dynamic_bytes, hidden_bytes + panel_bytes, hidden_bytes + h2_bytes, norm_bytes, down_bytes});
    const std::size_t capped = max_dynamic_bytes & ~std::size_t(15u);
    return std::min(desired, capped);
}

/**
 * @brief Select the exact full-entry function admitted for these bindings.
 * @param args Cold-resolved complete full HC/FFN arguments.
 * @param program Authenticated physical decoder union for fallback coverage.
 * @param prefix_bytes Cold-derived prefix dynamic LDS extent in bytes.
 * @return Specialized sharded fast entry when eligible, otherwise the canonical family entry.
 */
inline const void* selectFullNativeHCFFNKernelFunction(
    const FullNativeHCFFNArgs& args,
    ResidentMoEFFNDecoderProgram program,
    std::size_t prefix_bytes)
{
    if (fastFullNativeHCFFNGeometryEligible(args, prefix_bytes))
    {
        if (const void* fast = fastFullNativeHCFFNKernelFunction(args.suffix.ffn.execution))
            return fast;
    }
    return fullNativeHCFFNKernelFunction(program);
}

} // namespace llaminar2::rocm
