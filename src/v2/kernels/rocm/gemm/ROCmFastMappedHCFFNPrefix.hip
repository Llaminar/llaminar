/**
 * @file ROCmFastMappedHCFFNPrefix.hip
 * @brief Kernel 1 (HC Prefix + Router + Early Route Publication) and host admission/launch
 *        for the ROCm mapped-overlay retained M1 HC/FFN two-megakernel pipeline.
 */
#include "ROCmFastFullNativeHCFFNBody.hip.inc"
#include "ROCmMappedRetainedHCFFN.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/common/NativeCooperativeKernel.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

extern "C" bool rocmNativeEmbeddingIQTableView(int device_ordinal, llaminar2::NativeEmbeddingIQTables *out);

namespace llaminar2::rocm::fast_full_hc_ffn
{
    /** @brief Phase-exclusive static LDS for Kernel 1 (HC prefix + TopK selection + validated bank). */
    union FastMappedRoutePrefixShared
    {
        FullNativeHCFFNPrefixScratch hc;
        struct
        {
            router_native::RouterSelectionShared selection;
            FastValidatedRuntimeBank bank;
        } route;
    };
    static_assert(sizeof(FastMappedRoutePrefixShared) <= 3456u);

    /**
     * @brief Kernel 1: Resident HC prefix (absorbing write+injection) + FP16 Router + TopK/Publish.
     *
     * Exits immediately after publishing authoritative device route slots and public
     * `router_indices` / `router_weights_output` so outbound CPU/remote dispatch
     * packets start concurrently on reserved co-residency CUs while Kernel 2 runs.
     */
    __global__ __launch_bounds__(1024, 1) void fastMappedRoutePrefixKernel(
        MappedRoutePrefixArgs args,
        unsigned dynamic_bytes)
    {
        __shared__ FastMappedRoutePrefixShared shared;
        extern __shared__ __align__(16) unsigned char dynamic_storage[];

        FastWeightedGridBarrier grid{args.prefix.barrier_state};
        grid.validateEntry();
        fastHCPrefixPhase(args.prefix, grid, shared.hc, dynamic_storage, dynamic_bytes);

        const unsigned tid = threadIdx.x;
        const unsigned bid = blockIdx.x;
        const unsigned gdim = gridDim.x;
        const unsigned wave = waveUniform(tid >> 6);
        const unsigned lane = lane64();
        const unsigned H = waveUniform(static_cast<unsigned>(args.prefix.hc.norm.hidden));
        const unsigned routes = waveUniform(static_cast<unsigned>(args.top_k));
        const unsigned experts = waveUniform(static_cast<unsigned>(args.experts));
        const bool live = waveUniform(args.rows.activeRows() != 0);

        if (__builtin_expect(!live, 0))
        {
            if (bid == 0u)
            {
                if (wave == 0u)
                {
                    auto *rt = reinterpret_cast<DeviceMoELayerRuntimeView *>(waveUniform(args.runtime));
                    fastAcquireValidatedBank(
                        asLds<FastValidatedRuntimeBank>(&shared.route.bank), rt,
                        experts, routes, false, args.original, lane);
                }
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                if (tid < routes)
                {
                    auto *rt = asGlobal(shared.route.bank.runtime);
                    asGlobal(rt->topk_expert_ids)[tid] = -1;
                    asGlobal(rt->topk_weights)[tid] = 0.0f;
                    asGlobal(rt->route_participant_ids)[tid] = -1;
                    if (args.router_indices && args.router_weights_output)
                    {
                        asGlobal(args.router_indices)[tid] = -1.0f;
                        asGlobal(args.router_weights_output)[tid] = 0.0f;
                    }
                }
            }
            return;
        }

        // Convert mixed FP32 hidden to AS3 LDS packed __half2 for canonical FP16 router dot.
        auto h2_lds = asLds<std::uint32_t>(dynamic_storage);
        const float4 *hidden4 = reinterpret_cast<const float4 *>(waveUniform(args.prefix.hc.mixed));
        for (unsigned quad = tid; quad < (H >> 2); quad += 1024u)
        {
            const float4 val = globalWeight<float4>(hidden4 + quad);
            ldsStoreHalf2(h2_lds, quad * 2u, make_half2(__float2half_rn(val.x), __float2half_rn(val.y)));
            ldsStoreHalf2(h2_lds, quad * 2u + 1u, make_half2(__float2half_rn(val.z), __float2half_rn(val.w)));
        }
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

        const auto router_rows = NativeLayoutDownGeometry::rows(experts, bid, gdim);
        const unsigned nl = waveUniform(router_rows.end - router_rows.begin);
        const __half2 *w_all = reinterpret_cast<const __half2 *>(waveUniform(args.router_weights));
        float *router_logits_out = waveUniform(args.router_logits);
        const unsigned pair_count = H >> 1;

#pragma unroll 1
        for (unsigned task = wave; task < nl; task += 16u)
        {
            const unsigned expert = router_rows.begin + task;
            const float val = fastRouter128(h2_lds, w_all + std::size_t(expert) * pair_count, pair_count, lane);
            if (lane == 0u) asGlobal(router_logits_out)[expert] = val;
        }
        // Overlap runtime-bank validation on CTA 0 Wave 15 while router finishes!
        if (bid == 0u && wave == 15u)
        {
            auto *rt = reinterpret_cast<DeviceMoELayerRuntimeView *>(waveUniform(args.runtime));
            fastAcquireValidatedBank(
                asLds<FastValidatedRuntimeBank>(&shared.route.bank), rt,
                experts, routes, true, args.original, lane);
        }

        // Barrier 3: router_logits[0..experts-1] + CTA 0 bank ready.
        grid.sync<3>();

        if (bid == 0u)
        {
            router_native::selectProbabilities<false, 256, false, WorkgroupJoinMemory::LDSOnly,
                router_native::TopKTransferPolicy::RowNativeRoot,
                router_native::SelectorIntermediateStorage::ThreadPair, true, true>(
                router_logits_out, nullptr, static_cast<int>(experts), static_cast<int>(routes),
                args.normalize_router_weights, shared.route.selection);

            if (tid < routes)
            {
                fastPublishLiveRouteSlot(
                    asLds<const FastValidatedRuntimeBank>(&shared.route.bank),
                    static_cast<int>(tid),
                    asLds<const int>(shared.route.selection.selected_ids),
                    asLds<const float>(shared.route.selection.selected_weights),
                    experts,
                    args.router_indices,
                    args.router_weights_output,
                    /*write_legacy_outputs=*/true,
                    args.observe_histogram);
            }
        }
    }

} // namespace llaminar2::rocm::fast_full_hc_ffn

namespace llaminar2::rocm
{
    namespace
    {
        ResidentMoEPartitionPolicy makeUniformOrderedPolicy(int reduction)
        {
            const unsigned parts = MoEProjectionNumericalContract::orderedKPartitionsForWidth(reduction);
            if (parts == 0u || parts > 16u)
                throw std::invalid_argument("Mapped retained HC/FFN requires valid ordered K-partition count");
            const std::uint64_t word = 0x0101010101010101ULL * static_cast<std::uint64_t>(parts);
            return ResidentMoEPartitionPolicy{word, word, word, word, parts};
        }
    } // namespace

    PreparedMappedRetainedHCFFN prepareMappedRetainedHCFFN(
        DeviceId device,
        MappedRoutePrefixArgs &prefix_args,
        MappedFFNSuffixArgs &suffix_args,
        std::uint64_t generation)
    {
        if (!device.is_rocm() || !generation || isGraphCaptureActive())
            throw std::invalid_argument("ROCm mapped retained HC/FFN requires cold preparation on its exact ROCm device");

        auto &hc = prefix_args.prefix.hc;
        const auto &write = prefix_args.prefix.write;
        if (!prefix_args.prefix.barrier_state ||
            suffix_args.barrier_state != prefix_args.prefix.barrier_state ||
            !write.residual || !write.block || !write.gateLogits || !write.output ||
            write.output != hc.norm.input ||
            hc.norm.hidden <= 0 || hc.norm.hidden != suffix_args.hidden_columns ||
            hc.mixed != suffix_args.hidden ||
            hc.norm.extent.rows != 1 || prefix_args.rows.physicalRows() != 1 ||
            suffix_args.rows.physicalRows() != 1 ||
            !prefix_args.runtime || suffix_args.runtime != prefix_args.runtime ||
            !prefix_args.router_weights || !prefix_args.router_logits ||
            !prefix_args.router_indices || !prefix_args.router_weights_output ||
            !suffix_args.gate_partials || !suffix_args.route_contributions || !suffix_args.shared_output ||
            prefix_args.experts <= 0 || prefix_args.top_k <= 0 || prefix_args.top_k > 16 ||
            suffix_args.experts != prefix_args.experts || suffix_args.top_k != prefix_args.top_k ||
            suffix_args.routed_columns <= 0 || suffix_args.shared_columns <= 0 ||
            (suffix_args.hidden_columns & 31) != 0 || (suffix_args.routed_columns & 31) != 0 ||
            (suffix_args.shared_columns & 31) != 0 ||
            prefix_args.prefix.output_demand.bits() != 0u)
        {
            throw std::invalid_argument("ROCm mapped retained HC/FFN requires valid aligned M1 operands");
        }

        resolvePersistentHCReadPolicy(hc, prefix_args.prefix.output_demand);

        suffix_args.routed_gate_policy = makeUniformOrderedPolicy(suffix_args.hidden_columns);
        suffix_args.routed_down_policy = makeUniformOrderedPolicy(suffix_args.routed_columns);
        suffix_args.shared_gate_policy = makeUniformOrderedPolicy(suffix_args.hidden_columns);
        suffix_args.shared_down_policy = makeUniformOrderedPolicy(suffix_args.shared_columns);

        const void *suffix_fn = fastMappedFFNSuffixKernelFunction(suffix_args.execution);
        if (!suffix_fn)
            throw std::invalid_argument("ROCm mapped retained HC/FFN has no compiled fast suffix for execution masks");

        const auto &hardware = rocmHardwareMetadata().at(device.ordinal);
        const std::size_t static_lds_cap = std::max(
            sizeof(fast_full_hc_ffn::FastMappedRoutePrefixShared),
            sizeof(fast_full_hc_ffn::FastMappedSuffixScratch));
        if (hardware.compute_units <= 0 || hardware.lds_per_block <= static_lds_cap)
            throw std::invalid_argument("ROCm mapped retained HC/FFN exceeds device LDS capacity");
        const std::size_t max_dynamic = (hardware.lds_per_block - static_lds_cap) & ~std::size_t(15u);

        const std::size_t H = static_cast<std::size_t>(suffix_args.hidden_columns);
        const std::size_t F = static_cast<std::size_t>(suffix_args.routed_columns);
        const std::size_t S = static_cast<std::size_t>(suffix_args.shared_columns);
        const std::size_t routes = static_cast<std::size_t>(suffix_args.top_k);

        const std::size_t base_prefix_bytes = fullNativeHCFFNPrefixDynamicBytes(prefix_args.prefix, 1);
        const std::size_t norm_bytes = ((256u + 2u * H) * sizeof(float) + 15u) & ~std::size_t(15u);
        const std::size_t h2_bytes = ((H * 2u) + 15u) & ~std::size_t(15u);
        const std::size_t prefix_dynamic_bytes = std::max({base_prefix_bytes, norm_bytes, h2_bytes});
        if (!prefix_dynamic_bytes || prefix_dynamic_bytes > max_dynamic || (prefix_dynamic_bytes & 15u))
            throw std::invalid_argument("ROCm mapped retained HC/FFN prefix dynamic LDS exceeds device capacity");

        const std::size_t hidden_q8_bytes = (H / 32u) * sizeof(fast_full_hc_ffn::FastQ8Block48);
        const std::size_t slab_blocks = ((H / 32u) + 15u) & ~std::size_t(15u);
        const std::size_t panel_bytes = 2u * slab_blocks * 64u * sizeof(std::uint32_t);
        const std::size_t down_q8_bytes = (((routes * F + S) / 32u) * sizeof(fast_full_hc_ffn::FastQ8Block48) + 15u) & ~std::size_t(15u);
        const std::size_t suffix_dynamic_bytes = std::max(hidden_q8_bytes + panel_bytes, down_q8_bytes);
        if (!suffix_dynamic_bytes || suffix_dynamic_bytes > max_dynamic || (suffix_dynamic_bytes & 15u))
            throw std::invalid_argument("ROCm mapped retained HC/FFN suffix dynamic LDS exceeds device capacity");

        NativeEmbeddingIQTables iq_tables{};
        if (!rocmNativeEmbeddingIQTableView(device.ordinal, &iq_tables))
            throw std::runtime_error("ROCm mapped retained HC/FFN could not resolve cold IQ grid addresses");

        // Declaratively reserve up to 6 CUs on mapped/overlay topologies so outbound
        // ticket/packet publication (1-2 CUs) and bounded background dynamic expert
        // rebalance kernels (<= 4 CUs) execute concurrently without stalling resident barriers.
        const unsigned cus = static_cast<unsigned>(hardware.compute_units);
        const auto co_residency = NativeCoResidencyReservation::reserveProcessors(
            cus > 8u ? 6u : (cus > 4u ? 4u : 0u));

        const void *prefix_fn = reinterpret_cast<const void *>(&fast_full_hc_ffn::fastMappedRoutePrefixKernel);
        const auto prefix_adm = NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(
            device, prefix_fn, 1024u, prefix_dynamic_bytes, generation,
            NativeResidentLaunchPolicy::Ordinary, co_residency);
        const auto suffix_adm = NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(
            device, suffix_fn, 1024u, suffix_dynamic_bytes, generation,
            NativeResidentLaunchPolicy::Ordinary, co_residency);

        if (prefix_adm.resources().private_bytes != 0u || suffix_adm.resources().private_bytes != 0u)
            throw std::logic_error("ROCm mapped retained HC/FFN requires zero-spill resident kernels");

        const unsigned prefix_grid = std::max(1u, std::min(prefix_adm.maximumGrid(), prefix_adm.availableProcessors()));
        const unsigned suffix_grid = std::max(1u, std::min(suffix_adm.maximumGrid(), suffix_adm.availableProcessors()));
        hc.resident_grid = prefix_grid;

        PreparedMappedRetainedHCFFN out{};
        out.device = device;
        out.generation = generation;
        out.prefix_fn = prefix_fn;
        out.prefix_grid = prefix_grid;
        out.prefix_dynamic_bytes = static_cast<unsigned>(prefix_dynamic_bytes);
        out.prefix_resources = prefix_adm.resources();
        out.suffix_fn = suffix_fn;
        out.suffix_grid = suffix_grid;
        out.suffix_dynamic_bytes = static_cast<unsigned>(suffix_dynamic_bytes);
        out.suffix_resources = suffix_adm.resources();
        out.iq_tables = iq_tables;
        return out;
    }

    void launchMappedRetainedHCFFN(
        const PreparedMappedRetainedHCFFN &program,
        const MappedRoutePrefixArgs &prefix_args,
        const MappedFFNSuffixArgs &suffix_args,
        void *stream,
        std::uint64_t generation,
        const std::function<void()> &on_prefix_recorded)
    {
        if (!program.prefix_fn || !program.suffix_fn || !program.prefix_grid || !program.suffix_grid ||
            generation != program.generation || !stream ||
            stream == hipStreamLegacy || stream == hipStreamPerThread)
        {
            throw std::logic_error("ROCm mapped retained HC/FFN launch contract or stream is invalid");
        }

        auto hip_stream = static_cast<hipStream_t>(stream);
        MappedRoutePrefixArgs prefix_copy = prefix_args;
        unsigned prefix_dyn = program.prefix_dynamic_bytes;
        void *prefix_argv[]{&prefix_copy, &prefix_dyn};
        hipError_t status = hipLaunchKernel(
            program.prefix_fn, dim3(program.prefix_grid), dim3(1024u),
            prefix_argv, program.prefix_dynamic_bytes, hip_stream);
        if (status != hipSuccess)
        {
            throw std::runtime_error(
                std::string("ROCm mapped retained HC/FFN prefix launch failed: ") + hipGetErrorString(status));
        }

        if (on_prefix_recorded)
            on_prefix_recorded();

        MappedFFNSuffixArgs suffix_copy = suffix_args;
        unsigned suffix_dyn = program.suffix_dynamic_bytes;
        NativeEmbeddingIQTables iq_tables = program.iq_tables;
        void *suffix_argv[]{&suffix_copy, &suffix_dyn, &iq_tables};
        status = hipLaunchKernel(
            program.suffix_fn, dim3(program.suffix_grid), dim3(1024u),
            suffix_argv, program.suffix_dynamic_bytes, hip_stream);
        if (status != hipSuccess)
        {
            throw std::runtime_error(
                std::string("ROCm mapped retained HC/FFN suffix launch failed: ") + hipGetErrorString(status));
        }
    }

} // namespace llaminar2::rocm
