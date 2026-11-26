/**
 * @file ROCmMappedFollowerMoE.h
 * @brief Single-kernel resident ROCm mapped-overlay follower megakernel interface.
 *
 * Collapses the entire per-layer follower pipeline on ROCm participants (10 separate
 * kernels per layer: single-row dispatch acquire/validate/materialize, float-to-int,
 * runtime route selection, runtime descriptor materialization, Q8 hidden quantize,
 * Gate/Up partial + reduce/SwiGLU, Down partial + reduce, return pack/release, and
 * dynamic service telemetry begin/finish) into ONE cooperative resident megakernel
 * per layer across arbitrary ROCm topologies and all 18 NativeQuantised codebooks.
 *
 * Key architectural properties:
 *   - Empty-layer fast path: when `live_entries == 0` (>50% of layers per follower),
 *     Block 0 immediately publishes the return descriptor + system-release timeline
 *     before Barrier 0; all CTAs exit in <2 us with zero PCIe activation reads/writes.
 *   - Zero PCIe read amplification: Block 0 reads `hidden_rows_fp32` ONCE via
 *     128-bit non-temporal loads (`loadPeerPublishedFloat4`), quantizes into
 *     bank-conflict-free `FastQ8Block48` in L2, and all resident CTAs load from L2.
 *   - Direct-to-mapped coalesced Down: all resident CTAs write pre-weighted canonical
 *     route contributions directly to `returned.canonical_route_contributions_fp32`
 *     with `__threadfence_system()` before Barrier 2, eliminating the intermediate
 *     local canonical buffer and trailing return-copy kernel.
 *   - Dynamic expert movement (RCU bank flips) and service telemetry remain 100%
 *     device-resident with declarative 6-CU co-residency reservation and 0 spills.
 */
#pragma once

#include "backends/DeviceId.h"
#include "execution/moe/DeviceMoEOverlayServiceTelemetry.h"
#include "execution/moe/MoEOverlayActivationPacketABI.h"
#include "kernels/common/NativeCooperativeKernel.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include "kernels/rocm/gemm/ROCmResidentMoEFFN.h"
#include "kernels/rocm/moe/ROCmNativeRuntimeBankView.h"

#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    struct DeviceMoELayerRuntime;
}

namespace llaminar2::rocm
{
    /** @brief L2-resident broadcast header written by Block 0 before Barrier 0 (trivial POD for __shared__). */
    struct alignas(64) MappedFollowerScratchHeader
    {
        std::uint32_t live_entries;
        std::uint32_t local_live_count;
        std::uint32_t expert_count;
        std::uint32_t pad0;
        std::int32_t experts[16];
        float weights[16];
        std::int32_t original_slots[16];
        const native_runtime_abi::DeviceMoEExpertDescriptorView *experts_table;
        const std::uint8_t *local_compute_mask;
    };
    static_assert(sizeof(MappedFollowerScratchHeader) <= 256u &&
                  alignof(MappedFollowerScratchHeader) == 64u);

    /**
     * @brief Byte layout of the serialized per-participant device scratch buffer.
     *
     * One ~30 KB buffer per ROCm follower participant is allocated once via
     * TransferEngine/PhysicalMemoryAuthority and reused sequentially across all
     * layers of the single-row follower graph.
     */
    struct MappedFollowerScratchLayout
    {
        static constexpr std::size_t kBarrierOffset = 0u;
        static constexpr std::size_t kHeaderOffset = 64u;
        static constexpr std::size_t kHiddenQ8Offset = 320u;

        static constexpr std::size_t hiddenQ8Bytes(int d_model) noexcept
        {
            const std::size_t blocks = static_cast<std::size_t>(d_model) / 32u;
            return ((blocks * 48u) + 63u) & ~std::size_t(63u);
        }

        static constexpr std::size_t gatePartialsOffset(int d_model) noexcept
        {
            return kHiddenQ8Offset + hiddenQ8Bytes(d_model);
        }

        static constexpr std::size_t totalBytes(int d_model, int top_k, int intermediate) noexcept
        {
            const std::size_t gate_bytes =
                static_cast<std::size_t>(top_k) * static_cast<std::size_t>(intermediate) * sizeof(float);
            return ((gatePartialsOffset(d_model) + gate_bytes) + 63u) & ~std::size_t(63u);
        }
    };

    /** @brief Frozen arguments for one layer of the resident ROCm mapped follower megakernel. */
    struct MappedFollowerMoEArgs
    {
        std::uint32_t *barrier_state = nullptr;            ///< Zero-initialized 64B resident barrier words.
        MappedFollowerScratchHeader *scratch_header = nullptr;
        void *scratch_hidden_q8 = nullptr;                 ///< `(H / 32) * 48` B L2 Q8 hidden scratch.
        float *scratch_gate_partials = nullptr;            ///< `top_k * F` FP32 SwiGLU scratch.
        MoEOverlayActivationSingleRowDispatchConsumeLaunch dispatch_launch{};
        MoEOverlayActivationSingleRowReturnPackLaunch return_launch{};
        MoEOverlayActivationTimelineWaitDeviceBinding admission_acquire{};
        DeviceMoELayerRuntime *runtime = nullptr;
        DeviceMoEOverlayServiceTelemetryCell *service_telemetry_layer = nullptr;
        DeviceMoEOverlayServiceTelemetrySample *service_telemetry_sample = nullptr;
        const MoEOverlayInferenceGraphRole *service_graph_role = nullptr;
        std::uint64_t wall_clock_ns_per_tick_q32 = 0u;     ///< Precomputed ceil((1e6 << 32) / clock_rate_khz) to avoid __udivdi3.
        int hidden_columns = 0;
        int routed_columns = 0;
        int experts = 0;
        int top_k = 0;
        ResidentMoEPartitionPolicy routed_gate_policy{};
        ResidentMoEPartitionPolicy routed_down_policy{};
        std::uint32_t gate_execution_mask = 0u;
        std::uint32_t down_execution_mask = 0u;
        std::uint8_t has_admission_acquire = 0u;
        std::uint8_t observe_histogram = 1u;
    };

    /** @brief Cold-admitted resident launch state for one ROCm mapped follower layer. */
    struct PreparedMappedFollowerMoE
    {
        DeviceId device = DeviceId::invalid();
        std::uint64_t generation = 0u;
        const void *fn = nullptr;
        unsigned grid = 0u;
        unsigned dynamic_bytes = 0u;
        NativeCooperativeKernelResources resources{};
        NativeEmbeddingIQTables iq_tables{};
    };

    /** @brief Lookup Shard 0 mapped follower specializations (standard K-quants & legacy). */
    const void *fastMappedFollowerMoEShard0Function(
        std::uint32_t gate_mask, std::uint32_t down_mask) noexcept;

    /** @brief Lookup Shard 1 mapped follower specializations (IQ1/IQ2/IQ3/IQ4 & mixed pairs). */
    const void *fastMappedFollowerMoEShard1Function(
        std::uint32_t gate_mask, std::uint32_t down_mask) noexcept;

    /** @brief Resolve the compiled specialized resident follower kernel for physical masks. */
    inline const void *fastMappedFollowerMoEKernelFunction(
        std::uint32_t gate_mask, std::uint32_t down_mask) noexcept
    {
        if (const void *fn = fastMappedFollowerMoEShard0Function(gate_mask, down_mask))
            return fn;
        return fastMappedFollowerMoEShard1Function(gate_mask, down_mask);
    }

    /**
     * @brief Validate geometry, resolve partition policies, and cold-admit the follower megakernel.
     *
     * Reserves up to 6 CUs for concurrent dynamic expert movement and enforces
     * strict zero register spills (`private_bytes == 0`) before capture.
     */
    PreparedMappedFollowerMoE prepareMappedFollowerMoE(
        DeviceId device,
        MappedFollowerMoEArgs &args,
        std::uint64_t generation);

    /**
     * @brief Record the prepared resident follower megakernel on an exact non-default HIP stream.
     */
    void launchMappedFollowerMoE(
        const PreparedMappedFollowerMoE &program,
        const MappedFollowerMoEArgs &args,
        void *stream,
        std::uint64_t generation);

} // namespace llaminar2::rocm
