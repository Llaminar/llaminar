/**
 * @file CUDARetainedHCFFNBody.cu.inc
 * @brief Cooperative CUDA QFN local HC, half router, routed and shared FFN.
 *
 * Included after the CUDA MoE numerical bodies and the native runtime-bank
 * view. Each task retains the original accumulation order. The enclosing
 * graph keeps the explicit ticket and collective edges after publication.
 */

    using llaminar2::cuda::moe::RetainedM1Expert;
    using llaminar2::cuda::moe::RetainedM1HCFFN;
    using llaminar2::cuda::moe::RetainedM1RoutePublication;
    using llaminar2::cuda::moe::RetainedM1SharedPublication;

    constexpr int kRetainedQFNExpertCount = 512;
    constexpr int kRetainedRouteCounts = 2;
    static_assert(kMaxTopK <= 32, "A single original router warp compacts the QFN route slots");

    /** @brief Select the active program and publish only the empty routing ledger on a zero row. */
    __global__ void retainedQFNM1Predicate(cudaGraphConditionalHandle condition, const int* live,
        llaminar2::cuda::moe::RetainedM1Router router)
    {
        const int rows = *live;
        if (rows < 0 || rows > 1)
            __trap();
        const int slot = threadIdx.x;
        if (rows)
        {
            if (slot == 0)
                cudaGraphSetConditional(condition, 1u);
            return;
        }

        auto* runtime = reinterpret_cast<DeviceMoELayerRuntimeView*>(router.runtime);
        if (slot == 0)
        {
            runtime_require_decode_row_policy(runtime, router.local_policy);
            const auto bank = runtime_require_decode_execution_bank(runtime);
            if (!runtime_shape_ok(runtime, bank, router.experts, router.top_k) ||
                !runtime_route_assignment_ledger_ok(runtime, router.top_k))
                __trap();
        }
        __syncwarp();
        if (slot >= router.top_k)
            return;
        runtime->topk_expert_ids[slot] = -1;
        runtime->topk_weights[slot] = 0.0f;
        runtime->route_participant_ids[slot] = -1;
        router.route_indices[slot] = -1.0f;
        router.route_weights[slot] = 0.0f;
    }

    /**
     * @brief Apply the original CUDA shared input-gate reduction to the mixed row.
     * @param hidden Original aligned H-wide HC read.
     * @param weights Original aligned FP32 input-gate weights.
     * @param width Full admitted model width in four-element groups.
     * @param partial Original 256-lane reduction storage.
     * @return The same sigmoid factor on every lane of the participating CTA.
     */
    __device__ __forceinline__ float retainedSharedGateFactor(const float* hidden,
        const float* weights, int width, float* partial)
    {
        const auto* input = reinterpret_cast<const float4*>(hidden);
        const auto* gate = reinterpret_cast<const float4*>(weights);
        float dot = 0.0f;
        for (int i = threadIdx.x; i < width / 4; i += blockDim.x)
        {
            const float4 x = input[i], w = gate[i];
            dot += x.x * w.x + x.y * w.y + x.z * w.z + x.w * w.w;
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        return 1.0f / (1.0f + expf(-partial[0]));
    }

    /**
     * @brief Publish one complete original hidden K32 for both local expert roles.
     * @param value Original mixed hidden output in this lane.
     * @param codes First destination byte of this lane's packet.
     * @param scale Destination of the complete packet's rounded scale.
     * @param lane Physical owner of this packet byte, in [0,32).
     */
    __device__ __forceinline__ void retainedHiddenQ8(float value, int8_t* codes, float* scale, unsigned lane)
    {
        using namespace llaminar2::device_q8_activation_contract;
        float maximum = maximumAbsoluteValue(value, 0.0f);
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2)
            maximum = maximumAbsoluteValue(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
        const float stored = llaminar2::device_q8_activation_contract::scale(maximum);
        if (lane == 0)
            *scale = stored;
        codes[lane] = static_cast<int8_t>(quantize(value, stored));
    }

    /**
     * @brief Publish one original 32-column SwiGLU packet without global partials.
     * Eight warps compute the 16 original five-K32 partitions. A single warp
     * folds the two shared planes in partition order and quantizes the packet.
     * @tparam Codebook Cold-admitted physical weight decoder.
     * @tparam Width Original local Gate/Up width.
     * @param expert Validated native operands for a live route or the shared expert.
     * @param packet Local output packet assigned to the entire CTA.
     * @param down_codes Original resident SwiGLU INT8 output plane.
     * @param down_scales Original rounded scale for each packet.
     * @param hidden_codes Prepared common hidden INT8 input.
     * @param hidden_scales Original input quantization scales.
     * @param scratch CTA-local partition storage. Reuse follows a CTA join.
     */
    template<uint8_t Codebook, int Width>
    __device__ __forceinline__ void retainedGateUpPacket(
        const RetainedM1Expert& expert, unsigned packet, int8_t* down_codes, float* down_scales,
        const int8_t* hidden_codes, const float* hidden_scales, float (*scratch)[32])
    {
        constexpr int partitions = llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        constexpr int packets = kQFNDecodeGateUpPackets;
        constexpr int warps = kThreads / 32;
        static_assert(partitions % warps == 0);
        const unsigned warp = threadIdx.x / 32;
        const unsigned lane = threadIdx.x % 32;
        const unsigned first_column = packet * 32;
        const unsigned column = first_column + lane;
        constexpr auto order = NativeMoEDotOrder::VerifierK32;
#pragma unroll
        for (unsigned part = warp; part < partitions; part += warps)
        {
            float gate, up;
            native_quantised_gate_up_fixed_partition<Codebook, Width, packets, order>(
                expert.gate, expert.up, hidden_codes, hidden_scales, column, part * packets, gate, up);
            scratch[part][lane] = gate;
            scratch[part + partitions][lane] = up;
        }
        __syncthreads();
        if (warp == 0)
        {
            float gate = 0.0f, up = 0.0f;
#pragma unroll
            for (int part = 0; part < partitions; ++part)
            {
                gate = __fadd_rn(gate, scratch[part][lane]);
                up = __fadd_rn(up, scratch[part + partitions][lane]);
            }
            publish_decode_swiglu_q8(gate, up, down_codes + first_column, down_scales + packet, lane);
        }
    }

    /**
     * @brief Compute a physical Gate/Up family over its compact live route range.
     * @tparam Codebook Cold-admitted physical decoder.
     * @tparam Width Local feed-forward width in complete K32 packets.
     * @param args Original resident operands and prepared descriptors.
     * @param first First compact route using this decoder.
     * @param last Exclusive end of compact routes using this decoder.
     * @param scratch Per-CTA ordered partition planes.
     */
    template<uint8_t Codebook, int Width>
    __device__ __forceinline__ void retainedRoutedGateUpRange(
        const llaminar2::cuda::moe::RetainedM1HCFFN& args, unsigned first, unsigned last,
        float (*scratch)[32])
    {
        constexpr unsigned output_packets = Width / 32;
        const unsigned begin = first * output_packets;
        const unsigned end = last * output_packets;
        const unsigned first_block = (blockIdx.x + gridDim.x - begin % gridDim.x) % gridDim.x;
        for (unsigned packet = begin + first_block; packet < end; packet += gridDim.x)
        {
            const unsigned local = packet / output_packets;
            retainedGateUpPacket<Codebook, Width>(args.selected[local], packet % output_packets,
                args.routed_workspace.down_codes + local * Width,
                args.routed_workspace.down_scales + local * output_packets,
                args.hidden_codes, args.hidden_scales, scratch);
            __syncthreads();
        }
    }

    /**
     * @brief Consume the resident role's already prepared Q8 row in the original ordered Down fold.
     * @tparam Codebook Exact validated Down decoder.
     * @tparam Direct One original K32 packet per partition, with direct narrow publication.
     * @tparam Width Exact local Down reduction width.
     * @param weight Original expert Down source.
     * @param column Original participant-local output column.
     * @param codes Complete prepared SwiGLU K32 bytes.
     * @param scales Original rounded packet scales.
     * @return Exact unweighted original route contribution for one channel.
     */
    template<uint8_t Codebook, bool Direct, int Width = kQFNDecodeGateUpN>
    __device__ __forceinline__ float retainedDown(const DeviceNativeQuantisedMatrixDesc& weight,
        int column, const int8_t* codes, const float* scales)
    {
        constexpr int blocks = Width / 32;
        constexpr int pitch = kQFNDecodeGateUpK;
        if constexpr (Direct)
        {
            // The narrow F320 source has ten K32 packets. Unroll two
            // at a time to keep the original order without growing registers.
            return native_quantised_dot_admitted_range<Codebook, NativeMoEDotOrder::DecodeK32, blocks, 2>(
                weight, column, codes, scales, pitch, 0, blocks);
        }
        else
        {
            constexpr int partitions = llaminar2::CUDAMoEBatchInvariantPolicy::down_k_partitions;
            const int packets = (blocks + partitions - 1) / partitions;
            float value = 0.0f;
            for (int part = 0; part < partitions; ++part)
            {
                const int start = part * packets;
                const int end = min(start + packets, blocks);
                const float partial = start < end ? native_quantised_dot_admitted_range<Codebook>(
                    weight, column, codes, scales, weight.n, start, end) : 0.0f;
                value = moe_accumulate_rn(value, partial);
            }
            return value;
        }
    }

    /**
     * @brief Run one physically homogeneous set of local Down projections.
     * @tparam Codebook Prepared physical decoder selected by the route setup.
     * @param args Resident operands and canonical-slot publications.
     * @param runtime Original route weight authority.
     * @param begin First compact route this decoder owns.
     * @param end Exclusive end of the decoder's validated local route range.
     */
    template<uint8_t Codebook>
    __device__ __forceinline__ void retainedRoutedDownRange(const RetainedM1HCFFN& args,
        const DeviceMoELayerRuntimeView* runtime, int begin, int end)
    {
        constexpr int hidden = kQFNDecodeGateUpK;
        constexpr int width = kQFNDecodeGateUpN;
        static_assert(hidden % kThreads == 0);
        constexpr int blocks_per_route = hidden / kThreads;
        const unsigned first = begin * blocks_per_route;
        const unsigned stop = end * blocks_per_route;
        // Keep a CTA's original cyclic tile ownership across the decoder
        // boundary so separate formats can execute on separate SMs.
        unsigned block = blockIdx.x + (first / gridDim.x) * gridDim.x;
        if (block < first)
            block += gridDim.x;
        const bool weight_routes = args.route_publication == RetainedM1RoutePublication::Preweighted;
        for (; block < stop; block += gridDim.x)
        {
            const int route = block / blocks_per_route;
            const int column = block % blocks_per_route * kThreads + threadIdx.x;
            const int slot = args.route_order[kRetainedRouteCounts + route];
            const auto* codes = args.routed_workspace.down_codes + route * width;
            const auto* scales = args.routed_workspace.down_scales + route * (width / 32);
            const float contribution = retainedDown<Codebook, true>(args.selected[route].down, column, codes, scales);
            args.routed_contributions[slot * hidden + column] = weight_routes
                ? moe_weight_route_rn(runtime->topk_weights[slot], contribution) : contribution;
        }
    }

    /**
     * @brief Share the rounded input load across four independent half-router dots.
     * Each expert keeps its original ten stride-256 FMAs, followed by the
     * same descending warp and block trees. Shared output from each warp is
     * read once by the warp assigned to that expert's final reduction.
     * @param hidden Original mixed FP32 row, rounded to FP16 per loaded value.
     * @param weights Four consecutive original FP16 router expert rows.
     * @param output Four distinct original FP32 router output slots.
     * @param partial Temporary warp reductions owned by this CTA.
     */
    __device__ __forceinline__ void retainedRouterFour(const float* hidden, const __half* weights,
        float* output, float (*partial)[8])
    {
        constexpr int width = kQFNDecodeGateUpK;
        constexpr int experts = 4;
        float sums[experts]{};
#pragma unroll
        for (unsigned i = threadIdx.x; i < width; i += kThreads)
        {
            const float value = __half2float(__float2half_rn(hidden[i]));
#pragma unroll
            for (int expert = 0; expert < experts; ++expert)
                sums[expert] += value * __half2float(weights[expert * width + i]);
        }
        const unsigned lane = threadIdx.x % 32;
        const unsigned warp = threadIdx.x / 32;
#pragma unroll
        for (unsigned expert = 0; expert < experts; ++expert)
        {
            float sum = sums[expert];
#pragma unroll
            for (unsigned offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0)
                partial[expert][warp] = sum;
        }
        __syncthreads();
        if (warp < experts)
        {
            float sum = lane < 8 ? partial[warp][lane] : 0.0f;
#pragma unroll
            for (unsigned offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0)
                output[warp] = sum;
        }
    }

    /**
     * @brief One resident program for the active original CUDA main-model row.
     * @tparam GateCodebook Fixed routed Gate/Up physical layout from original prepared sources.
     * @tparam DownCodebook Fixed routed Down layout from original prepared sources.
     * @tparam ObserveCollectives Whether the original router is the histogram publisher.
     * @tparam PublishHCActivated The graph still consumes the original activated HC row.
     * @tparam PublishHCUp The graph still consumes the original HC logits.
     * @param args Complete original launch binding. Capture-selected publication demand is immutable.
     * @pre The native grid is resident and every CTA has real work for the active original row.
     *      The enclosing native capture selects this positive branch from the device count.
     */
    template<uint8_t GateCodebook, uint8_t DownCodebook, bool ObserveCollectives,
             bool PublishHCActivated, bool PublishHCUp>
    __global__ __launch_bounds__(kThreads, 2) void retainedQFNM1HCFFNKernel(const __grid_constant__ RetainedM1HCFFN args)
    {
        using namespace llaminar2::cuda::moe;
        constexpr auto SharedCodebook = native_formats::Q8_0.codebook_id;
        constexpr auto Promoted = llaminar2::kNativeQuantisedExpandedInt8MinCodebook;
        constexpr bool can_promote_gate = GateCodebook == native_formats::Q5_K.codebook_id;
        constexpr bool can_promote_down = DownCodebook == native_formats::Q5_1.codebook_id;
        static_assert(!(can_promote_gate && can_promote_down));
        auto grid = cooperative_groups::this_grid();
        __shared__ union
        {
            RetainedHCPrefixShared hc;
            float router_tiles[4][8];
            float shared_factor[kThreads];
            float gate_up[2 * llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions][32];
        } workspace;
        const auto& router = args.router;
        auto* runtime = reinterpret_cast<DeviceMoELayerRuntimeView*>(router.runtime);
        constexpr int hidden = kQFNDecodeGateUpK;
        constexpr int width = kQFNDecodeGateUpN;
        const unsigned first_thread = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned threads = gridDim.x * blockDim.x;
        const unsigned lane = threadIdx.x % 32;
        publishRetainedHCPrefix<PublishHCActivated, PublishHCUp>(args.hc, workspace.hc, grid);

        const auto* router_weights = reinterpret_cast<const __half*>(router.weights);
        for (int tile = blockIdx.x; tile < kRetainedQFNExpertCount / 4; tile += gridDim.x)
        {
            const int first_expert = 4 * tile;
            retainedRouterFour(args.hc.mixed, router_weights + first_expert * hidden,
                router.logits + first_expert, workspace.router_tiles);
            __syncthreads();
        }
        for (unsigned packet = first_thread / 32 * 32; packet < static_cast<unsigned>(hidden); packet += threads)
            retainedHiddenQ8(args.hc.mixed[packet + lane], args.hidden_codes + packet,
                args.hidden_scales + packet / 32, lane);
        grid.sync();

        if (blockIdx.x == 0)
        {
            softmax_topk_decode_runtime_body<ObserveCollectives, false, kRetainedQFNExpertCount>(router.logits, runtime, nullptr,
                router.route_indices, router.route_weights, router.experts, router.top_k, router.normalize, true,
                router.local_policy, router.position, nullptr, 0, nullptr, nullptr, 0, {}, nullptr, nullptr, -1, 0, nullptr);
            __syncthreads();
            if (args.shared_publication == RetainedM1SharedPublication::GatedComplete)
            {
                const float factor = retainedSharedGateFactor(args.hc.mixed, args.shared_gate_input,
                    hidden, workspace.shared_factor);
                if (threadIdx.x == 0)
                    *args.shared_gate_factor = factor;
            }
            if (threadIdx.x < 32)
            {
                const int slot = threadIdx.x;
                const bool in_route = slot < router.top_k;
                const int expert = in_route ? runtime->topk_expert_ids[slot] : -1;
                const unsigned present = __ballot_sync(0xffffffffu, expert >= 0);
                const int local_count = __popc(present);
                const auto* descriptor = [&]() -> const DeviceMoEExpertDescriptorView* {
                    if (expert < 0)
                        return nullptr;
                    if (expert >= router.experts)
                        __trap();
                    const unsigned bank = runtime_require_decode_execution_bank(runtime);
                    if (bank > 1)
                        __trap();
                    return &runtime_placement_banks(runtime)[bank].experts[expert];
                }();
                unsigned promoted = 0;
                if constexpr (can_promote_gate)
                    promoted = __ballot_sync(0xffffffffu,
                        descriptor && descriptor->gate.codebook_id == Promoted);
                else if constexpr (can_promote_down)
                    promoted = __ballot_sync(0xffffffffu,
                        descriptor && descriptor->down.codebook_id == Promoted);
                const unsigned canonical = present & ~promoted;
                const int canonical_count = __popc(canonical);
                const unsigned prefix = (1u << slot) - 1u;
                if (slot == 0)
                {
                    args.route_order[0] = local_count;
                    args.route_order[1] = canonical_count;
                }
                if (descriptor)
                {
                    const auto& view = *descriptor;
                    const bool moved = (promoted & (1u << slot)) != 0;
                    const bool down_valid = can_promote_down && moved
                        ? native_quantised_desc_shape_ok<Promoted>(view.down, hidden, args.routed_width) &&
                            view.down.source_identity_present && view.down.source_codebook_id == DownCodebook &&
                            !view.down.source_is_superblock
                        : native_quantised_desc_shape_ok<DownCodebook>(view.down, hidden, args.routed_width);
                    const bool gate_valid = can_promote_gate && moved
                        ? native_quantised_pair_shape_ok<Promoted>(view.gate, view.up, args.routed_width, hidden) &&
                            view.gate.source_identity_present && view.gate.source_codebook_id == GateCodebook &&
                            view.gate.source_is_superblock && view.up.source_identity_present &&
                            view.up.source_codebook_id == GateCodebook && view.up.source_is_superblock
                        : native_quantised_pair_shape_ok<GateCodebook>(view.gate, view.up, args.routed_width, hidden);
                    if (!gate_valid || !down_valid)
                        __trap();
                    // Bind physical families before either arithmetic body starts.
                    // Publication still scatters to the original canonical slots.
                    const int packed = moved ? canonical_count + __popc(promoted & prefix) : __popc(canonical & prefix);
                    args.route_order[kRetainedRouteCounts + packed] = slot;
                    args.selected[packed] = {view.gate, view.up, view.down};
                }
                else if (in_route)
                    args.route_order[kRetainedRouteCounts + local_count + slot - __popc(present & prefix)] = slot;
            }
        }
        grid.sync();

        const int local_count = args.route_order[0];
        constexpr int output_packets = width / 32;
        const unsigned routed_packets = local_count * output_packets;
        if constexpr (can_promote_gate)
        {
            const int canonical_count = args.route_order[1];
            retainedRoutedGateUpRange<GateCodebook, width>(args, 0, canonical_count, workspace.gate_up);
            retainedRoutedGateUpRange<Promoted, width>(args, canonical_count, local_count, workspace.gate_up);
        }
        else
            retainedRoutedGateUpRange<GateCodebook, width>(args, 0, local_count, workspace.gate_up);
        for (unsigned packet = (blockIdx.x + gridDim.x - routed_packets % gridDim.x) % gridDim.x;
             packet < output_packets; packet += gridDim.x)
        {
            retainedGateUpPacket<SharedCodebook, width>(args.shared, packet,
                args.shared_workspace.down_codes, args.shared_workspace.down_scales,
                args.hidden_codes, args.hidden_scales, workspace.gate_up);
            __syncthreads();
        }
        grid.sync();

        if constexpr (can_promote_down)
        {
            const int canonical_count = args.route_order[1];
            retainedRoutedDownRange<DownCodebook>(args, runtime, 0, canonical_count);
            retainedRoutedDownRange<Promoted>(args, runtime, canonical_count, local_count);
        }
        else
            retainedRoutedDownRange<DownCodebook>(args, runtime, 0, local_count);
        for (unsigned task = first_thread; task < static_cast<unsigned>((router.top_k - local_count) * hidden);
             task += threads)
        {
            const int slot = args.route_order[kRetainedRouteCounts + local_count + task / hidden];
            args.routed_contributions[slot * hidden + task % hidden] = 0.0f;
        }

        const bool gate_shared = args.shared_publication == RetainedM1SharedPublication::GatedComplete;
        const float shared_factor = gate_shared ? *args.shared_gate_factor : 1.0f;
        for (unsigned column = first_thread; column < static_cast<unsigned>(hidden); column += threads)
        {
            const float raw = retainedDown<SharedCodebook, false>(args.shared.down, column,
                args.shared_workspace.down_codes, args.shared_workspace.down_scales);
            args.shared_contribution[column] = gate_shared
                ? llaminar2::device_fp32_contract::multiply(shared_factor, raw) : raw;
        }
    }
