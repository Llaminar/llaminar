/**
 * @file ROCmMappedFollowerMoEShard0.hip
 * @brief Shard 0 of 1-kernel resident ROCm mapped follower megakernel: standard K-quants, legacy, and host launch bridge.
 */
#include "ROCmMappedFollowerMoEBody.hip.inc"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/common/NativeCooperativeKernel.cuh"

#include <algorithm>
#include <stdexcept>

extern "C" bool rocmNativeEmbeddingIQTableView(int device_ordinal, llaminar2::NativeEmbeddingIQTables *out);

namespace llaminar2::rocm
{
    const void *fastMappedFollowerMoEShard0Function(
        std::uint32_t gate_mask, std::uint32_t down_mask) noexcept
    {
        using namespace mapped_follower_moe;
        std::uint8_t g = 0u, d = 0u;
        bool exp = false;
        if (!decodeFollowerKey(gate_mask, down_mask, g, d, exp))
            return nullptr;

#define LLAMINAR_FOLLOWER_PAIR(G, D, EXP) \
    if (g == (G) && d == (D) && exp == (EXP)) \
        return reinterpret_cast<const void *>(&fastMappedFollowerMoEKernel<(G), (D), (EXP)>);

        // Primary Q4_K / Q5_K / Q6_K / Q8_0 mixed and homogeneous pairs (with dynamic promotion).
        LLAMINAR_FOLLOWER_PAIR(5u, 7u, false)
        LLAMINAR_FOLLOWER_PAIR(5u, 7u, true)
        LLAMINAR_FOLLOWER_PAIR(5u, 19u, false)
        LLAMINAR_FOLLOWER_PAIR(5u, 19u, true)
        LLAMINAR_FOLLOWER_PAIR(7u, 19u, false)
        LLAMINAR_FOLLOWER_PAIR(7u, 19u, true)
        LLAMINAR_FOLLOWER_PAIR(5u, 8u, false)
        LLAMINAR_FOLLOWER_PAIR(5u, 8u, true)
        LLAMINAR_FOLLOWER_PAIR(7u, 8u, false)
        LLAMINAR_FOLLOWER_PAIR(7u, 8u, true)
        LLAMINAR_FOLLOWER_PAIR(8u, 19u, false)
        LLAMINAR_FOLLOWER_PAIR(8u, 19u, true)
        LLAMINAR_FOLLOWER_PAIR(5u, 5u, false)
        LLAMINAR_FOLLOWER_PAIR(5u, 5u, true)
        LLAMINAR_FOLLOWER_PAIR(7u, 7u, false)
        LLAMINAR_FOLLOWER_PAIR(7u, 7u, true)
        LLAMINAR_FOLLOWER_PAIR(8u, 8u, false)
        LLAMINAR_FOLLOWER_PAIR(8u, 8u, true)
        LLAMINAR_FOLLOWER_PAIR(19u, 19u, false)
        LLAMINAR_FOLLOWER_PAIR(19u, 19u, true)

        // Legacy and remaining K-quant homogeneous & mixed pairs.
        LLAMINAR_FOLLOWER_PAIR(0u, 0u, false)
        LLAMINAR_FOLLOWER_PAIR(6u, 6u, false)
        LLAMINAR_FOLLOWER_PAIR(9u, 9u, false)
        LLAMINAR_FOLLOWER_PAIR(10u, 10u, false)
        LLAMINAR_FOLLOWER_PAIR(23u, 23u, false)
        LLAMINAR_FOLLOWER_PAIR(9u, 5u, false)
        LLAMINAR_FOLLOWER_PAIR(9u, 7u, false)
        LLAMINAR_FOLLOWER_PAIR(9u, 8u, false)
        LLAMINAR_FOLLOWER_PAIR(10u, 5u, false)
        LLAMINAR_FOLLOWER_PAIR(10u, 7u, false)
        LLAMINAR_FOLLOWER_PAIR(10u, 8u, false)

#undef LLAMINAR_FOLLOWER_PAIR
        return nullptr;
    }

    namespace
    {
        /**
         * @brief Dedicated 1-wave Stage 0 admission and grant initializer.
         *
         * Runs once per token before Layer 0 while Tier 0 computes Layer 0
         * attention, isolating the 512B cold control handshake from the
         * 1024-thread resident follower megakernel so every codebook specialization
         * compiles with zero spills and zero private memory.
         */
        __global__ __launch_bounds__(64) void acquireMappedFollowerStage0GrantKernel(
            MoEOverlayActivationTimelineWaitDeviceBinding admission_acquire,
            MoEOverlayActivationDispatchConsumeLaunch dispatch_pkt)
        {
            if (blockIdx.x != 0u || threadIdx.x != 0u)
                return;
            moe_activation_packet_device::waitSystemAcquire64(admission_acquire);
            auto *control = const_cast<MoEOverlayActivationEpochControl *>(dispatch_pkt.control);
            if (!moe_activation_packet_device::acquireOrValidateGrant(
                    control,
                    dispatch_pkt.grant,
                    MoEOverlayActivationEndpoint::Follower,
                    MoEOverlayActivationOperation::ConsumeDispatch,
                    dispatch_pkt.stage_ordinal,
                    dispatch_pkt.physical_rows,
                    /*expected_target_participant=*/-1,
                    dispatch_pkt.placement))
            {
                moe_activation_packet_device::publishFailure(
                    control,
                    dispatch_pkt.grant,
                    MoEOverlayActivationEndpoint::Follower,
                    MoEOverlayActivationOperation::ConsumeDispatch,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    dispatch_pkt.stage_ordinal,
                    dispatch_pkt.model_layer_index,
                    moe_activation_packet_device::placementAcquireEvidence(dispatch_pkt.placement));
            }
        }

        ResidentMoEPartitionPolicy makeFollowerUniformOrderedPolicy(int reduction)
        {
            const unsigned parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(reduction);
            if (parts == 0u || parts > 16u)
                throw std::invalid_argument("ROCm mapped follower MoE requires valid ordered K-partition count");
            const std::uint64_t word = 0x0101010101010101ULL * static_cast<std::uint64_t>(parts);
            return ResidentMoEPartitionPolicy{word, word, word, word, parts};
        }
    } // namespace

    PreparedMappedFollowerMoE prepareMappedFollowerMoE(
        DeviceId device,
        MappedFollowerMoEArgs &args,
        std::uint64_t generation)
    {
        if (!device.is_rocm() || !generation || isGraphCaptureActive())
            throw std::invalid_argument("ROCm mapped follower MoE requires cold preparation on its exact ROCm device");
        if (!args.barrier_state || !args.scratch_header || !args.scratch_hidden_q8 || !args.scratch_gate_partials ||
            !args.runtime || !args.dispatch_launch.valid() || !args.return_launch.valid() ||
            args.hidden_columns <= 0 || args.routed_columns <= 0 ||
            (args.hidden_columns & 31) != 0 || (args.routed_columns & 31) != 0 ||
            args.experts <= 0 || args.top_k <= 0 || args.top_k > 16)
            throw std::invalid_argument("ROCm mapped follower MoE requires valid aligned single-row operands");

        args.routed_gate_policy = makeFollowerUniformOrderedPolicy(args.hidden_columns);
        args.routed_down_policy = makeFollowerUniformOrderedPolicy(args.routed_columns);

        const auto &hardware = rocmHardwareMetadata().at(device.ordinal);
        const std::uint64_t clock_rate_khz = static_cast<std::uint64_t>(rocmServiceWallClockRateKHz(hardware));
        args.wall_clock_ns_per_tick_q32 = clock_rate_khz > 0u
            ? (((std::uint64_t{1000000u} << 32) + clock_rate_khz - 1u) / clock_rate_khz)
            : 0u;

        const void *fn = fastMappedFollowerMoEKernelFunction(args.gate_execution_mask, args.down_execution_mask);
        if (!fn)
            throw std::invalid_argument("ROCm mapped follower MoE has no compiled specialized kernel for execution masks");

        constexpr std::size_t static_lds = sizeof(mapped_follower_moe::MappedFollowerStaticLds);
        if (hardware.compute_units <= 0 || hardware.lds_per_block <= static_lds)
            throw std::invalid_argument("ROCm mapped follower MoE exceeds device LDS capacity");
        const std::size_t max_dynamic = (hardware.lds_per_block - static_lds) & ~std::size_t(15u);

        const std::size_t H = static_cast<std::size_t>(args.hidden_columns);
        const std::size_t F = static_cast<std::size_t>(args.routed_columns);
        const std::size_t routes = static_cast<std::size_t>(args.top_k);
        const std::size_t hidden_q8_bytes = (H / 32u) * sizeof(fast_full_hc_ffn::FastQ8Block48);
        const std::size_t slab_blocks = ((H / 32u) + 15u) & ~std::size_t(15u);
        const std::size_t panel_bytes = 2u * slab_blocks * 64u * sizeof(std::uint32_t);
        const std::size_t down_q8_bytes =
            (((routes * F) / 32u) * sizeof(fast_full_hc_ffn::FastQ8Block48) + 63u) & ~std::size_t(63u);
        // Reserve 4 KiB (4 groups x 4 sub-waves x 64 floats) after down_q8 for
        // 4-wave-per-task exact-order Down folding and coalesced float4 PCIe stores.
        const std::size_t down_stage_bytes = 4u * 4u * 64u * sizeof(float);
        const std::size_t dynamic_bytes = std::max(hidden_q8_bytes + panel_bytes, down_q8_bytes + down_stage_bytes);
        if (!dynamic_bytes || dynamic_bytes > max_dynamic || (dynamic_bytes & 15u))
            throw std::invalid_argument("ROCm mapped follower MoE dynamic LDS exceeds device capacity");

        NativeEmbeddingIQTables iq_tables{};
        if (!rocmNativeEmbeddingIQTableView(device.ordinal, &iq_tables))
            throw std::runtime_error("ROCm mapped follower MoE could not resolve cold IQ grid addresses");

        // Declaratively reserve up to 6 CUs so concurrent dynamic expert movement
        // and peer transfer lanes never stall resident grid barriers.
        const unsigned cus = static_cast<unsigned>(hardware.compute_units);
        const auto co_residency = NativeCoResidencyReservation::reserveProcessors(
            cus > 8u ? 6u : (cus > 4u ? 4u : 0u));

        const auto adm = NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(
            device, fn, 1024u, dynamic_bytes, generation,
            NativeResidentLaunchPolicy::Ordinary, co_residency);
        if (adm.resources().private_bytes != 0u)
            throw std::runtime_error("ROCm mapped follower MoE requires zero private memory and zero register spills");

        const unsigned grid = std::max(1u, std::min(adm.maximumGrid(), adm.availableProcessors()));
        return PreparedMappedFollowerMoE{
            .device = device,
            .generation = generation,
            .fn = fn,
            .grid = grid,
            .dynamic_bytes = static_cast<unsigned>(dynamic_bytes),
            .resources = adm.resources(),
            .iq_tables = iq_tables,
        };
    }

    void launchMappedFollowerMoE(
        const PreparedMappedFollowerMoE &program,
        const MappedFollowerMoEArgs &args,
        void *stream,
        std::uint64_t generation)
    {
        if (!program.fn || !program.grid || !stream || generation != program.generation)
            throw std::invalid_argument("ROCm mapped follower MoE launch received invalid program, stream, or generation");

        hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
        if (args.has_admission_acquire != 0u)
        {
            hipLaunchKernelGGL(
                acquireMappedFollowerStage0GrantKernel,
                dim3(1u, 1u, 1u),
                dim3(64u, 1u, 1u),
                0u,
                hip_stream,
                args.admission_acquire,
                args.dispatch_launch.packet);
            const hipError_t grant_err = hipGetLastError();
            if (grant_err != hipSuccess)
                throw std::runtime_error(
                    std::string("ROCm mapped follower Stage 0 grant launch failed: ") +
                    hipGetErrorString(grant_err));
        }

        auto launch_args = args;
        unsigned dyn_bytes = program.dynamic_bytes;
        auto iq_tables = program.iq_tables;
        void *params[] = {&launch_args, &dyn_bytes, &iq_tables};
        const hipError_t err = hipLaunchKernel(
            program.fn,
            dim3(program.grid, 1u, 1u),
            dim3(1024u, 1u, 1u),
            params,
            program.dynamic_bytes,
            hip_stream);
        if (err != hipSuccess)
            throw std::runtime_error(std::string("ROCm mapped follower MoE launch failed: ") + hipGetErrorString(err));
    }

} // namespace llaminar2::rocm
