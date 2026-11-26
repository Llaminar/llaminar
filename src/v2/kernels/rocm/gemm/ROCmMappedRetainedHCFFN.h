/**
 * @file ROCmMappedRetainedHCFFN.h
 * @brief Two-kernel mapped-overlay ROCm HC + Router + Local Routed/Shared FFN program.
 *
 * Designed for mapped GPU+CPU (and heterogeneous tiered) topologies where an
 * outbound dispatch packet / host ticket must launch immediately after the
 * router publishes selected experts while the local GPU FFN suffix computes
 * concurrently on remaining CUs.
 *
 * Kernel 1 (`fastMappedRoutePrefixKernel`): fuses previous residual write +
 *   BranchRMSNorm + HC injection + idle-CTA-prefetched HC Down + HC Up/Mix +
 *   FP16 Router + canonical TopK/softmax + runtime/public route publication.
 *   Completes and releases `HIPEarlyPacketPublication` so CPU/remote peers start
 *   immediately on reserved co-residency CUs.
 *
 * Kernel 2 (`fastMappedFFNSuffixKernel<GateCB, DownCB, SharedCB, AllowExpanded>`):
 *   compacts active local routed experts (skipping remote slots so 100% of
 *   admitted CTAs load-balance live GPU work), computes Shared + Local Routed
 *   Gate/Up/SwiGLU -> single grid barrier -> compact Activated Q8 -> direct
 *   coalesced raw unweighted local route slots (and 0.0f for remote slots) to
 *   `route_contributions` and un-gated partial Shared Down to `shared_output`.
 *
 * Covers all 18 quantization codebooks + dynamic EXPANDED_INT8_MIN promotion
 * across arbitrary ROCm topologies, with declarative 4-CU co-residency
 * reservation and strict zero-spill cold admission.
 */
#pragma once

#include "ROCmFullNativeHCFFN.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include "kernels/common/NativeCooperativeKernel.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace llaminar2::rocm
{
    /** @brief Frozen arguments for Kernel 1 (HC Prefix + Router + Route Publication). */
    struct MappedRoutePrefixArgs
    {
        FullNativeHCFFNPrefixArgs prefix;
        DeviceMoELayerRuntime *runtime = nullptr;
        DeviceMoEIntermediateShardBinding original{};
        const std::uint16_t *router_weights = nullptr;
        float *router_logits = nullptr;
        float *router_indices = nullptr;
        float *router_weights_output = nullptr;
        DeviceRowRange rows = DeviceRowRange::fullyActive(1);
        int experts = 0;
        int top_k = 0;
        bool normalize_router_weights = false;
        bool observe_histogram = false;
    };

    /** @brief Frozen arguments for Kernel 2 (Compact Local Routed + Shared FFN Suffix). */
    struct MappedFFNSuffixArgs
    {
        std::uint32_t *barrier_state = nullptr; ///< Same 56-byte base as prefix.barrier_state (uses Phase 4).
        DeviceMoELayerRuntime *runtime = nullptr;
        DeviceMoEIntermediateShardBinding original{};
        const float *hidden = nullptr;
        DeviceRowRange rows = DeviceRowRange::fullyActive(1);
        int hidden_columns = 0;
        int routed_columns = 0;
        int shared_columns = 0;
        int experts = 0;
        int top_k = 0;
        DeviceNativeQuantisedMatrixDesc shared_gate{};
        DeviceNativeQuantisedMatrixDesc shared_up{};
        DeviceNativeQuantisedMatrixDesc shared_down{};
        ResidentMoEPartitionPolicy routed_gate_policy{};
        ResidentMoEPartitionPolicy routed_down_policy{};
        ResidentMoEPartitionPolicy shared_gate_policy{};
        ResidentMoEPartitionPolicy shared_down_policy{};
        ResidentMoEFFNExecutionMasks execution{};
        float *gate_partials = nullptr;
        float *route_contributions = nullptr;
        float *shared_output = nullptr;
    };

    /** @brief Prepared cold admission for the two-kernel ROCm mapped retained HC/FFN program. */
    struct PreparedMappedRetainedHCFFN
    {
        DeviceId device = DeviceId::invalid();
        std::uint64_t generation = 0u;
        const void *prefix_fn = nullptr;
        unsigned prefix_grid = 0u;
        unsigned prefix_dynamic_bytes = 0u;
        NativeCooperativeKernelResources prefix_resources{};
        const void *suffix_fn = nullptr;
        unsigned suffix_grid = 0u;
        unsigned suffix_dynamic_bytes = 0u;
        NativeCooperativeKernelResources suffix_resources{};
        NativeEmbeddingIQTables iq_tables{};
    };

    /** @brief Lookup Shard 0 mapped suffix specializations (Q4_K/Q5_K/Q6_K/Q8_0). */
    const void *fastMappedFFNSuffixShard0Function(const ResidentMoEFFNExecutionMasks &execution) noexcept;
    /** @brief Lookup Shard 1a mapped suffix specializations (standard K-quants & legacy). */
    const void *fastMappedFFNSuffixShard1aFunction(const ResidentMoEFFNExecutionMasks &execution) noexcept;
    /** @brief Lookup Shard 1b mapped suffix specializations (Q2_K/Q8_0 & mixed pairs). */
    const void *fastMappedFFNSuffixShard1bFunction(const ResidentMoEFFNExecutionMasks &execution) noexcept;
    /** @brief Lookup Shard 2 mapped suffix specializations (all IQ1/IQ2/IQ3 grids). */
    const void *fastMappedFFNSuffixShard2Function(const ResidentMoEFFNExecutionMasks &execution) noexcept;

    /** @brief Resolve the exact all-codebook sharded mapped FFN suffix kernel entry. */
    inline const void *fastMappedFFNSuffixKernelFunction(const ResidentMoEFFNExecutionMasks &execution) noexcept
    {
        if (const void *fn = fastMappedFFNSuffixShard0Function(execution)) return fn;
        if (const void *fn = fastMappedFFNSuffixShard1aFunction(execution)) return fn;
        if (const void *fn = fastMappedFFNSuffixShard1bFunction(execution)) return fn;
        return fastMappedFFNSuffixShard2Function(execution);
    }

    /**
     * @brief Validate geometry, resolve partition policies, and admit both kernels cold.
     *
     * Enforces 4-CU co-residency reservation and zero register spills (`private_bytes == 0`)
     * on both kernels before graph capture begins.
     */
    PreparedMappedRetainedHCFFN prepareMappedRetainedHCFFN(
        DeviceId device,
        MappedRoutePrefixArgs &prefix_args,
        MappedFFNSuffixArgs &suffix_args,
        std::uint64_t generation);

    /**
     * @brief Enqueue Kernel 1, publish its early capture frontier, then enqueue Kernel 2.
     * @param program Cold-admitted two-kernel program.
     * @param prefix_args Immutable Kernel 1 arguments.
     * @param suffix_args Immutable Kernel 2 arguments.
     * @param stream Exact non-default HIP capture stream.
     * @param generation Retained workspace generation for identity validation.
     * @param on_prefix_recorded Optional callback invoked immediately after Kernel 1 is
     *        recorded on @p stream so outbound packet stages splice onto its frontier.
     */
    void launchMappedRetainedHCFFN(
        const PreparedMappedRetainedHCFFN &program,
        const MappedRoutePrefixArgs &prefix_args,
        const MappedFFNSuffixArgs &suffix_args,
        void *stream,
        std::uint64_t generation,
        const std::function<void()> &on_prefix_recorded = {});

} // namespace llaminar2::rocm
