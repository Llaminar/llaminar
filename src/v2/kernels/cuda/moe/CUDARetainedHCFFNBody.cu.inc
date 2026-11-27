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

    /** @brief Exact device-wide clock for the bounded in-grid route rendezvous. */
    __device__ __forceinline__ std::uint64_t retainedRouteClock()
    {
        std::uint64_t tick;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
        return tick;
    }

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
#pragma unroll 1
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
     * @brief Share the rounded input load across two independent half-router dots (256-CTA wave).
     */
    __device__ __forceinline__ void retainedRouterTwo(const float* hidden, const __half* weights,
        float* output, float (*partial)[8])
    {
        constexpr int width = kQFNDecodeGateUpK;
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll 1
        for (unsigned i = threadIdx.x; i < width; i += kThreads)
        {
            const float value = __half2float(__float2half_rn(hidden[i]));
            sum0 += value * __half2float(weights[i]);
            sum1 += value * __half2float(weights[width + i]);
        }
        const unsigned lane = threadIdx.x % 32;
        const unsigned warp = threadIdx.x / 32;
#pragma unroll
        for (unsigned offset = 16; offset > 0; offset >>= 1)
        {
            sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
            sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
        }
        if (lane == 0)
        {
            partial[0][warp] = sum0;
            partial[1][warp] = sum1;
        }
        __syncthreads();
        if (warp < 2u)
        {
            float sum = lane < 8 ? partial[warp][lane] : 0.0f;
#pragma unroll
            for (unsigned offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0)
                output[warp] = sum;
        }
    }

    /** @brief Native phase ownership fixed at graph construction. */
    enum class RetainedM1NativePhase { Complete, RoutePrefix, FFNSuffix };

    /** @brief Shared workspace for union-resident 512-expert softmax + top-k selection. */
    struct RetainedTopKShared
    {
        float values[kRetainedQFNExpertCount];
        float reductions[kThreads];
        int red_idx[kThreads];
        int selected[kMaxTopK];
        float selected_weights[kMaxTopK];
    };

    /**
     * @brief Dedicated 4-CTA/SM cooperative HC + Router + HiddenQ8 + SharedGate kernel.
     *
     * Decoupled from TopK/descriptor/device-graph planning so ptxas targets <=64
     * registers with zero spills and launches 296 resident CTAs (4 CTAs/SM on SM86).
     */
    template<bool PublishHCActivated, bool PublishHCUp, bool GeneralCodebooks = false>
    __global__ __launch_bounds__(kThreads, GeneralCodebooks ? 2 : 4) void retainedQFNM1HCRouterKernel(
        const __grid_constant__ RetainedM1HCFFN args)
    {
        using namespace llaminar2::cuda::moe;
        auto grid = cooperative_groups::this_grid();
        __shared__ union
        {
            RetainedHCPrefixShared<kQFNDecodeGateUpK> hc;
            float router_tiles[4][8];
            float shared_factor[kThreads];
        } workspace;

        constexpr int hidden = kQFNDecodeGateUpK;
        if (args.profile_ns && blockIdx.x == 0 && threadIdx.x == 0)
            args.profile_ns[0] = retainedRouteClock();
        {
            const unsigned max_packets =
                static_cast<unsigned>(args.router.top_k) * (static_cast<unsigned>(args.routed_width) >> 5) +
                (static_cast<unsigned>(args.shared.gate.n) >> 5);
            if (blockIdx.x == 0)
            {
                for (unsigned i = threadIdx.x; i < max_packets; i += blockDim.x)
                    args.suffix_done[i] = 0u;
            }
            uint32_t* hc_down_done = (args.suffix_done && max_packets >= static_cast<unsigned>(args.hc.down.n >> 5))
                ? args.suffix_done : nullptr;
            publishRetainedHCPrefix<PublishHCActivated, PublishHCUp, hidden, true, GeneralCodebooks>(
                args.hc, workspace.hc, grid, args.profile_ns, hc_down_done);
        }
        if (args.profile_ns && blockIdx.x == 0 && threadIdx.x == 0)
            args.profile_ns[4] = retainedRouteClock();

        const unsigned lane = threadIdx.x & 31u;
        const unsigned warp = threadIdx.x >> 5;
        const auto* router_weights = reinterpret_cast<const __half*>(args.router.weights);
        if (gridDim.x >= 267u)
        {
            // Full-GPU single-wave overlapped Router (256 CTAs x 2 experts) + HiddenQ8 (10 CTAs) + SharedGate (1 CTA).
            if (blockIdx.x < 256u)
            {
                const int first_expert = 2 * static_cast<int>(blockIdx.x);
                retainedRouterTwo(args.hc.mixed, router_weights + first_expert * hidden,
                    args.router.logits + first_expert, workspace.router_tiles);
            }
            else if (blockIdx.x < 266u)
            {
                const unsigned packet = ((blockIdx.x - 256u) * 8u + warp) * 32u;
                if (packet < static_cast<unsigned>(hidden))
                    retainedHiddenQ8(args.hc.mixed[packet + lane], args.hidden_codes + packet,
                        args.hidden_scales + (packet >> 5), lane);
            }
            else if (blockIdx.x == 266u && args.shared_publication == RetainedM1SharedPublication::GatedComplete)
            {
                const float factor = retainedSharedGateFactor(args.hc.mixed, args.shared_gate_input,
                    hidden, workspace.shared_factor);
                if (threadIdx.x == 0)
                    *args.shared_gate_factor = factor;
            }
        }
        else
        {
#pragma unroll 1
            for (int tile = blockIdx.x; tile < kRetainedQFNExpertCount / 4; tile += gridDim.x)
            {
                const int first_expert = 4 * tile;
                retainedRouterFour(args.hc.mixed, router_weights + first_expert * hidden,
                    args.router.logits + first_expert, workspace.router_tiles);
                __syncthreads();
            }
            const unsigned first_thread = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned threads = gridDim.x * blockDim.x;
#pragma unroll 1
            for (unsigned packet = (first_thread >> 5) << 5; packet < static_cast<unsigned>(hidden); packet += threads)
                retainedHiddenQ8(args.hc.mixed[packet + lane], args.hidden_codes + packet,
                    args.hidden_scales + (packet >> 5), lane);
            if (blockIdx.x == 1 && args.shared_publication == RetainedM1SharedPublication::GatedComplete)
            {
                const float factor = retainedSharedGateFactor(args.hc.mixed, args.shared_gate_input,
                    hidden, workspace.shared_factor);
                if (threadIdx.x == 0)
                    *args.shared_gate_factor = factor;
            }
        }
    }

    /** @brief Runtime-parametric codebook + provenance check covering all 18 codebooks and dynamic promotion. */
    __device__ __forceinline__ bool retainedDescValidRuntime(
        const DeviceNativeQuantisedMatrixDesc& desc, int expected_n, int expected_k,
        uint8_t canonical_cb, uint8_t promoted_cb, uint8_t source_superblock, bool moved)
    {
        if (!native_quantised_desc_shape_ok_dynamic(desc, expected_n, expected_k))
            return false;
        if (moved)
        {
            if (desc.codebook_id != promoted_cb)
                return false;
            if (promoted_cb == llaminar2::kNativeQuantisedExpandedInt8MinCodebook)
            {
                return desc.source_identity_present &&
                       desc.source_codebook_id == canonical_cb &&
                       desc.source_is_superblock == source_superblock;
            }
            return true;
        }
        const auto is_iq4_lut = [](std::uint8_t cb) noexcept {
            return cb == 4u || cb == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook;
        };
        return desc.codebook_id == canonical_cb ||
               (is_iq4_lut(canonical_cb) && is_iq4_lut(desc.codebook_id));
    }

    /**
     * @brief Dedicated 1-CTA TopK + descriptor binding + device-graph suffix grid planner.
     *
     * Codebook-independent (4 instantiations cover all 18 codebooks and arbitrary local widths).
     * Runs non-cooperatively (grid=1, block=256) after HCRouter completion with zero spills.
     */
    template<bool ObserveCollectives, bool IndependentLocalShard>
    __global__ __launch_bounds__(kThreads, 1) void retainedQFNM1TopKPlanKernel(
        const __grid_constant__ RetainedM1HCFFN args)
    {
        using namespace llaminar2::cuda::moe;
        const auto& router = args.router;
        auto* runtime = reinterpret_cast<DeviceMoELayerRuntimeView*>(router.runtime);
        constexpr int hidden = kQFNDecodeGateUpK;
        const unsigned routed_packets = static_cast<unsigned>(args.routed_width) / 32u;
        const unsigned shared_packets = static_cast<unsigned>(args.shared.gate.n) / 32u;
        const uint8_t gate_cb = args.gate_codebook;
        const uint8_t down_cb = args.down_codebook;
        const uint8_t gate_promoted = args.gate_promoted_codebook;
        const uint8_t down_promoted = args.down_promoted_codebook;
        const bool can_promote_gate = gate_promoted != 0xffu;
        const bool can_promote_down = down_promoted != 0xffu;

        if (args.profile_ns && threadIdx.x == 0)
            args.profile_ns[5] = retainedRouteClock();

        // Re-arm the first col_tiles entries of suffix_done consumed as hc_down_done
        // during the fused single-sync HC Down inside retainedQFNM1HCRouterKernel.
        const unsigned max_packets = static_cast<unsigned>(router.top_k) * routed_packets + shared_packets;
        const unsigned down_tiles = static_cast<unsigned>(args.hc.down.n / 32);
        if (args.suffix_done && max_packets >= down_tiles && threadIdx.x < down_tiles)
            args.suffix_done[threadIdx.x] = 0u;

        if constexpr (IndependentLocalShard)
        {
            __shared__ RetainedTopKShared tk;
            __shared__ RetainedM1Expert preloaded_expert[kMaxTopK];
            __shared__ uint8_t preloaded_local[kMaxTopK];
            __shared__ uint32_t shared_exec_bank;
            const unsigned tid = threadIdx.x;
            const float l0 = router.logits[tid];
            const float l1 = router.logits[tid + kThreads];
            tk.reductions[tid] = fmaxf(l0, l1);
            __syncthreads();
            if (tid < 32u)
            {
                float m = fmaxf(
                    fmaxf(fmaxf(tk.reductions[tid], tk.reductions[tid + 128u]),
                          fmaxf(tk.reductions[tid + 64u], tk.reductions[tid + 192u])),
                    fmaxf(fmaxf(tk.reductions[tid + 32u], tk.reductions[tid + 160u]),
                          fmaxf(tk.reductions[tid + 96u], tk.reductions[tid + 224u])));
#pragma unroll
                for (int s = 16; s > 0; s >>= 1)
                    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, s));
                if (tid == 0)
                    tk.reductions[0] = m;
            }
            __syncthreads();
            const float max_value = tk.reductions[0];

            const float p0 = expf(l0 - max_value);
            const float p1 = expf(l1 - max_value);
            tk.values[tid] = p0;
            tk.values[tid + kThreads] = p1;
            tk.reductions[tid] = p0 + p1;
            __syncthreads();
            if (tid < 32u)
            {
                const float low = __fadd_rn(__fadd_rn(tk.reductions[tid], tk.reductions[tid + 128u]),
                                            __fadd_rn(tk.reductions[tid + 64u], tk.reductions[tid + 192u]));
                const float high = __fadd_rn(__fadd_rn(tk.reductions[tid + 32u], tk.reductions[tid + 160u]),
                                             __fadd_rn(tk.reductions[tid + 96u], tk.reductions[tid + 224u]));
                float s = __fadd_rn(low, high);
#pragma unroll
                for (int d = 16; d > 0; d >>= 1)
                    s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, d));
                if (tid == 0)
                    tk.reductions[0] = s;
            }
            __syncthreads();
            const float denom = tk.reductions[0];
            const float q0 = denom > 0.0f ? p0 / denom : 0.0f;
            const float q1 = denom > 0.0f ? p1 / denom : 0.0f;
            tk.values[tid] = q0;
            tk.values[tid + kThreads] = q1;
            router.logits[tid] = q0;
            router.logits[tid + kThreads] = q1;
            __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            retainedTopK512(tk.values, router.top_k, tk.selected, tk.selected_weights, tk.reductions, tk.red_idx);
#else
            moe_select_topk_probabilities_block(tk.values, router.experts, router.top_k,
                tk.selected, tk.selected_weights, tk.reductions, tk.red_idx);
#endif
            if (tid == 0)
            {
                float topk_sum = 0.0f;
                for (int slot = 0; slot < router.top_k; ++slot)
                    topk_sum += tk.selected_weights[slot];
                if (router.normalize && topk_sum > 0.0f)
                    for (int slot = 0; slot < router.top_k; ++slot)
                        tk.selected_weights[slot] = tk.selected_weights[slot] / topk_sum;
                const uint32_t exec_bank = runtime_require_decode_execution_bank(runtime);
                if (exec_bank > 1u)
                    __trap();
                shared_exec_bank = exec_bank;
            }
            __syncthreads();
            if (args.profile_ns && tid == 0)
                args.profile_ns[11] = retainedRouteClock();

            {
                const uint32_t exec_bank = shared_exec_bank;
                const auto& bank = runtime_placement_banks(runtime)[exec_bank];
                const unsigned warp = tid >> 5;
                const unsigned lane = tid & 31u;
                static_assert(sizeof(DeviceNativeQuantisedMatrixDesc) % sizeof(uint64_t) == 0);
                constexpr unsigned words_per_mat = sizeof(DeviceNativeQuantisedMatrixDesc) / sizeof(uint64_t);
                for (int s = static_cast<int>(warp); s < router.top_k; s += 8)
                {
                    const int e = tk.selected[s];
                    const bool maybe_local = e >= 0 && e < router.experts && bank.local_compute_mask[e] != 0u;
                    if (lane == 0u)
                        preloaded_local[s] = maybe_local ? 1u : 0u;
                    if (maybe_local && lane < 3u * words_per_mat)
                    {
                        const auto& src = bank.experts[e];
                        auto& dst = preloaded_expert[s];
                        const unsigned mat = (lane < words_per_mat) ? 0u : (lane < 2u * words_per_mat ? 1u : 2u);
                        const unsigned w = lane - mat * words_per_mat;
                        const auto* s_mat = (mat == 0u) ? &src.gate : (mat == 1u ? &src.up : &src.down);
                        auto* d_mat = (mat == 0u) ? &dst.gate : (mat == 1u ? &dst.up : &dst.down);
                        reinterpret_cast<uint64_t*>(d_mat)[w] = reinterpret_cast<const uint64_t*>(s_mat)[w];
                    }
                }
            }
            __syncthreads();
            if (tid < 32u)
            {
                const int slot = static_cast<int>(tid);
                const bool in_route = slot < router.top_k;
                const uint32_t execution_bank = shared_exec_bank;
                const bool shape_ok = runtime_shape_ok(runtime, execution_bank, router.experts, router.top_k);
                if (!runtime_route_assignment_ledger_ok(runtime, router.top_k))
                    __trap();
                runtime_require_decode_row_policy(runtime, llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal);
                const auto& bank = runtime_placement_banks(runtime)[execution_bank];
                const bool collect_histogram = ObserveCollectives && runtime_histogram_admits_rows(*runtime);
                const int expert_id = in_route ? tk.selected[slot] : -1;
                const float weight = in_route ? tk.selected_weights[slot] : 0.0f;
                bool local_compute = false;
                if (in_route)
                {
                    int participant = -1;
                    if (shape_ok)
                    {
                        if (expert_id != -1)
                        {
                            const auto selection = runtime_intermediate_shard_row(runtime, bank, expert_id);
                            local_compute = selection.disposition == llaminar2::MoEIntermediateShardRowDisposition::Local;
                            participant = selection.participant;
                        }
                        runtime->topk_expert_ids[slot] = local_compute ? expert_id : -1;
                        runtime->topk_weights[slot] = local_compute ? weight : 0.0f;
                        runtime->route_participant_ids[slot] = participant;
                    }
                    router.route_indices[slot] = static_cast<float>(expert_id);
                    router.route_weights[slot] = weight;
                    if constexpr (ObserveCollectives)
                    {
                        if (collect_histogram && expert_id >= 0 && expert_id < router.experts)
                        {
                            atomicAdd(reinterpret_cast<unsigned long long*>(
                                &runtime_selected_histogram(*runtime, 0u)[expert_id]), 1ULL);
                            if (shape_ok && local_compute)
                                atomicAdd(reinterpret_cast<unsigned long long*>(
                                    &runtime_local_histogram(*runtime, 0u)[expert_id]), 1ULL);
                        }
                    }
                }
                const int local_expert = (in_route && shape_ok && local_compute) ? expert_id : -1;
                const unsigned present = __ballot_sync(0xffffffffu, local_expert >= 0);
                const int local_count = __popc(present);
                const RetainedM1Expert* descriptor =
                    (local_expert >= 0 && preloaded_local[slot]) ? &preloaded_expert[slot] : nullptr;
                unsigned promoted = 0;
                if (can_promote_gate)
                    promoted = __ballot_sync(0xffffffffu, descriptor && descriptor->gate.codebook_id == gate_promoted);
                else if (can_promote_down)
                    promoted = __ballot_sync(0xffffffffu, descriptor && descriptor->down.codebook_id == down_promoted);
                const unsigned canonical = present & ~promoted;
                const int canonical_count = __popc(canonical);
                const unsigned prefix_mask = (1u << slot) - 1u;
                if (slot == 0)
                {
                    args.route_order[0] = local_count;
                    args.route_order[1] = canonical_count;
                }
                if (descriptor)
                {
                    const auto& view = *descriptor;
                    const bool moved = (promoted & (1u << slot)) != 0;
                    const bool gate_valid =
                        retainedDescValidRuntime(view.gate, args.routed_width, hidden, gate_cb, gate_promoted,
                            args.gate_source_superblock, can_promote_gate && moved) &&
                        retainedDescValidRuntime(view.up, args.routed_width, hidden, gate_cb, gate_promoted,
                            args.gate_source_superblock, can_promote_gate && moved);
                    const bool down_valid =
                        retainedDescValidRuntime(view.down, hidden, args.routed_width, down_cb, down_promoted,
                            args.down_source_superblock, can_promote_down && moved);
                    if (!gate_valid || !down_valid)
                        __trap();
                    const int packed = moved ? canonical_count + __popc(promoted & prefix_mask) : __popc(canonical & prefix_mask);
                    args.route_order[kRetainedRouteCounts + packed] = slot;
                    preloaded_local[packed] = static_cast<uint8_t>(slot);
                }
                else if (in_route)
                    args.route_order[kRetainedRouteCounts + local_count + slot - __popc(present & prefix_mask)] = slot;
            }
            __syncthreads();
            {
                const int n_copy = args.route_order[0];
                const unsigned warp = tid >> 5;
                const unsigned lane = tid & 31u;
                static_assert(sizeof(DeviceNativeQuantisedMatrixDesc) % sizeof(uint64_t) == 0);
                constexpr unsigned words_per_mat = sizeof(DeviceNativeQuantisedMatrixDesc) / sizeof(uint64_t);
                for (int packed = static_cast<int>(warp); packed < n_copy; packed += 8)
                {
                    const auto& view = preloaded_expert[preloaded_local[packed]];
                    auto& dst = args.selected[packed];
                    if (lane < 3u * words_per_mat)
                    {
                        const unsigned mat = (lane < words_per_mat) ? 0u : (lane < 2u * words_per_mat ? 1u : 2u);
                        const unsigned w = lane - mat * words_per_mat;
                        const auto* s_mat = (mat == 0u) ? &view.gate : (mat == 1u ? &view.up : &view.down);
                        auto* d_mat = (mat == 0u) ? &dst.gate : (mat == 1u ? &dst.up : &dst.down);
                        reinterpret_cast<uint64_t*>(d_mat)[w] = reinterpret_cast<const uint64_t*>(s_mat)[w];
                    }
                }
            }
            __syncthreads();
            if (tid == 0)
            {
                if (args.profile_ns)
                    args.profile_ns[12] = retainedRouteClock();
#if CUDART_VERSION >= 12080
                if (args.suffix_gate_up_node && args.suffix_down_node)
                {
                    const unsigned local_active = static_cast<unsigned>(args.route_order[0]);
                    const unsigned total_gate_up_packets = local_active * routed_packets + shared_packets;
                    if (cudaGraphKernelNodeSetGridDim(
                            static_cast<cudaGraphDeviceNode_t>(args.suffix_gate_up_node),
                            dim3(retained_suffix_gate_up_k_splits, total_gate_up_packets)) != cudaSuccess ||
                        cudaGraphKernelNodeSetGridDim(
                            static_cast<cudaGraphDeviceNode_t>(args.suffix_down_node),
                            dim3(static_cast<unsigned>(hidden / 16), local_active + 1u)) != cudaSuccess)
                        __trap();
                }
#endif
            }
        }
        else
        {
            __shared__ RetainedTopKShared tk;
            __shared__ RuntimeDecodeDispatchScratch dispatch;
            __shared__ RetainedM1Expert preloaded_expert[kMaxTopK];
            __shared__ uint8_t packed_slots[kMaxTopK];
            __shared__ float shared_topk_sum;
            __shared__ uint32_t shared_exec_bank;
            __shared__ uint32_t shared_shape_ok;
            __shared__ uint32_t shared_has_multi_resident;
            const unsigned tid = threadIdx.x;
            const float l0 = router.logits[tid];
            const float l1 = router.logits[tid + kThreads];
            tk.reductions[tid] = fmaxf(l0, l1);
            __syncthreads();
            if (tid < 32u)
            {
                float m = fmaxf(
                    fmaxf(fmaxf(tk.reductions[tid], tk.reductions[tid + 128u]),
                          fmaxf(tk.reductions[tid + 64u], tk.reductions[tid + 192u])),
                    fmaxf(fmaxf(tk.reductions[tid + 32u], tk.reductions[tid + 160u]),
                          fmaxf(tk.reductions[tid + 96u], tk.reductions[tid + 224u])));
#pragma unroll
                for (int s = 16; s > 0; s >>= 1)
                    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, s));
                if (tid == 0u)
                    tk.reductions[0] = m;
            }
            __syncthreads();
            const float max_value = tk.reductions[0];

            const float p0 = expf(l0 - max_value);
            const float p1 = expf(l1 - max_value);
            tk.values[tid] = p0;
            tk.values[tid + kThreads] = p1;
            tk.reductions[tid] = p0 + p1;
            __syncthreads();
            if (tid < 32u)
            {
                const float low = __fadd_rn(__fadd_rn(tk.reductions[tid], tk.reductions[tid + 128u]),
                                            __fadd_rn(tk.reductions[tid + 64u], tk.reductions[tid + 192u]));
                const float high = __fadd_rn(__fadd_rn(tk.reductions[tid + 32u], tk.reductions[tid + 160u]),
                                             __fadd_rn(tk.reductions[tid + 96u], tk.reductions[tid + 224u]));
                float s = __fadd_rn(low, high);
#pragma unroll
                for (int d = 16; d > 0; d >>= 1)
                    s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, d));
                if (tid == 0u)
                    tk.reductions[0] = s;
            }
            __syncthreads();
            const float denom = tk.reductions[0];
            const float q0 = denom > 0.0f ? p0 / denom : 0.0f;
            const float q1 = denom > 0.0f ? p1 / denom : 0.0f;
            tk.values[tid] = q0;
            tk.values[tid + kThreads] = q1;
            router.logits[tid] = q0;
            router.logits[tid + kThreads] = q1;
            __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            retainedTopK512(tk.values, router.top_k, tk.selected, tk.selected_weights, tk.reductions, tk.red_idx);
#else
            moe_select_topk_probabilities_block(tk.values, router.experts, router.top_k,
                tk.selected, tk.selected_weights, tk.reductions, tk.red_idx);
#endif
            if (tid == 0u)
            {
                if (args.profile_ns)
                    args.profile_ns[11] = retainedRouteClock();
                runtime_require_decode_row_policy(runtime, router.local_policy);
                const uint32_t execution_bank = runtime_require_decode_execution_bank(runtime);
                if (execution_bank > 1u)
                    __trap();
                const bool shape_ok = runtime_shape_ok(runtime, execution_bank, router.experts, router.top_k);
                if (!runtime_route_assignment_ledger_ok(runtime, router.top_k))
                    __trap();
                float topk_sum = 0.0f;
                for (int k = 0; k < router.top_k; ++k)
                    topk_sum += tk.selected_weights[k];
                const auto& bank_view = runtime_placement_banks(runtime)[execution_bank];
                const bool needs_serial_replicated =
                    shape_ok &&
                    router.local_policy != llaminar2::RoutedExpertRowExecutionPolicy::FullyReplicatedLocal &&
                    router.local_policy != llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal &&
                    runtime_multi_resident_expert_count(bank_view) != 0u;
                if (needs_serial_replicated)
                {
                    runtime_resolve_decode_dispatch(
                        runtime, execution_bank, tk.selected, router.experts, router.top_k,
                        router.position ? router.position[0] : -1,
                        router.local_policy, ObserveCollectives, dispatch);
                }
                shared_topk_sum = topk_sum;
                shared_exec_bank = execution_bank;
                shared_shape_ok = shape_ok ? 1u : 0u;
                shared_has_multi_resident = needs_serial_replicated ? 1u : 0u;
            }
            __syncthreads();

            const uint32_t exec_bank = shared_exec_bank;
            const bool shape_ok = shared_shape_ok != 0u;
            const bool used_serial_dispatch = shared_has_multi_resident != 0u;
            const auto& bank = runtime_placement_banks(runtime)[exec_bank];
            const unsigned warp = tid >> 5;
            const unsigned lane = tid & 31u;

            // Parallel non-replicated dispatch resolution on warp 0 (overlaps all top_k global loads).
            if (!used_serial_dispatch && tid < static_cast<unsigned>(router.top_k))
            {
                const int k = static_cast<int>(tid);
                const int e = tk.selected[k];
                bool local = false;
                int owner = -1;
                if (shape_ok && e >= 0 && e < router.experts)
                {
                    if (router.local_policy == llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal)
                    {
                        const auto sel = runtime_intermediate_shard_row(runtime, bank, e);
                        local = sel.disposition == llaminar2::MoEIntermediateShardRowDisposition::Local;
                        owner = sel.participant;
                    }
                    else
                    {
                        local = bank.local_compute_mask[e] != 0u;
                        owner = (router.local_policy == llaminar2::RoutedExpertRowExecutionPolicy::FullyReplicatedLocal)
                            ? static_cast<int>(runtime->participant_id)
                            : runtime_expert_owner(bank, e);
                    }
                }
                dispatch.local_compute_flags[k] = local;
                dispatch.assigned_participants[k] = owner;
            }
            __syncthreads();

            // Publish all top_k route assignments and histogram atomics in parallel across warp 0,
            // while warps 0..7 simultaneously preload only {gate, up, down} (168 B) into SMEM.
            if (tid < static_cast<unsigned>(router.top_k))
            {
                const int k = static_cast<int>(tid);
                const int expert = tk.selected[k];
                const float weight = (router.normalize && shared_topk_sum > 0.0f)
                    ? (tk.selected_weights[k] / shared_topk_sum) : tk.selected_weights[k];
                const bool local_compute = shape_ok && dispatch.local_compute_flags[k];
                if (shape_ok)
                {
                    runtime->topk_expert_ids[k] = local_compute ? expert : -1;
                    runtime->topk_weights[k] = local_compute ? weight : 0.0f;
                    runtime->route_participant_ids[k] = dispatch.assigned_participants[k];
                }
                router.route_indices[k] = static_cast<float>(expert);
                router.route_weights[k] = weight;
                if constexpr (ObserveCollectives)
                {
                    if (runtime_histogram_admits_rows(*runtime) &&
                        expert >= 0 && expert < kDeviceMoEMaxExperts)
                    {
                        atomicAdd(reinterpret_cast<unsigned long long*>(
                            &runtime_selected_histogram(*runtime, 0u)[expert]), 1ULL);
                        if (local_compute)
                            atomicAdd(reinterpret_cast<unsigned long long*>(
                                &runtime_local_histogram(*runtime, 0u)[expert]), 1ULL);
                    }
                }
            }
            if (shape_ok)
            {
                static_assert(sizeof(DeviceNativeQuantisedMatrixDesc) % sizeof(uint64_t) == 0);
                constexpr unsigned words_per_mat = sizeof(DeviceNativeQuantisedMatrixDesc) / sizeof(uint64_t);
                for (int s = static_cast<int>(warp); s < router.top_k; s += 8)
                {
                    if (dispatch.local_compute_flags[s])
                    {
                        const int e = tk.selected[s];
                        if (e >= 0 && e < router.experts && lane < 3u * words_per_mat)
                        {
                            const auto& src = bank.experts[e];
                            auto& dst = preloaded_expert[s];
                            const unsigned mat = (lane < words_per_mat) ? 0u : (lane < 2u * words_per_mat ? 1u : 2u);
                            const unsigned w = lane - mat * words_per_mat;
                            const auto* s_mat = (mat == 0u) ? &src.gate : (mat == 1u ? &src.up : &src.down);
                            auto* d_mat = (mat == 0u) ? &dst.gate : (mat == 1u ? &dst.up : &dst.down);
                            reinterpret_cast<uint64_t*>(d_mat)[w] = reinterpret_cast<const uint64_t*>(s_mat)[w];
                        }
                    }
                }
            }
            __syncthreads();

            if (tid < 32u)
            {
                const int slot = static_cast<int>(tid);
                const bool in_route = slot < router.top_k;
                const bool local_compute = in_route && shape_ok && dispatch.local_compute_flags[slot];
                const int expert = local_compute ? tk.selected[slot] : -1;
                if (expert >= router.experts)
                    __trap();
                const unsigned present = __ballot_sync(0xffffffffu, expert >= 0);
                const int local_count = __popc(present);
                const RetainedM1Expert* descriptor =
                    (expert >= 0) ? &preloaded_expert[slot] : nullptr;
                unsigned promoted = 0;
                if (can_promote_gate)
                    promoted = __ballot_sync(0xffffffffu, descriptor && descriptor->gate.codebook_id == gate_promoted);
                else if (can_promote_down)
                    promoted = __ballot_sync(0xffffffffu, descriptor && descriptor->down.codebook_id == down_promoted);
                const unsigned canonical = present & ~promoted;
                const int canonical_count = __popc(canonical);
                const unsigned prefix_mask = (1u << slot) - 1u;
                if (slot == 0)
                {
                    args.route_order[0] = local_count;
                    args.route_order[1] = canonical_count;
#if CUDART_VERSION >= 12080
                    if (args.suffix_gate_up_node && args.suffix_down_node)
                    {
                        const unsigned local_active = static_cast<unsigned>(local_count);
                        const unsigned total_gate_up_packets = local_active * routed_packets + shared_packets;
                        if (cudaGraphKernelNodeSetGridDim(
                                static_cast<cudaGraphDeviceNode_t>(args.suffix_gate_up_node),
                                dim3(retained_suffix_gate_up_k_splits, total_gate_up_packets)) != cudaSuccess ||
                            cudaGraphKernelNodeSetGridDim(
                                static_cast<cudaGraphDeviceNode_t>(args.suffix_down_node),
                                dim3(static_cast<unsigned>(hidden / 16), local_active + 1u)) != cudaSuccess)
                            __trap();
                    }
#endif
                }
                if (descriptor)
                {
                    const auto& view = *descriptor;
                    const bool moved = (promoted & (1u << slot)) != 0;
                    const bool gate_valid =
                        retainedDescValidRuntime(view.gate, args.routed_width, hidden, gate_cb, gate_promoted,
                            args.gate_source_superblock, can_promote_gate && moved) &&
                        retainedDescValidRuntime(view.up, args.routed_width, hidden, gate_cb, gate_promoted,
                            args.gate_source_superblock, can_promote_gate && moved);
                    const bool down_valid =
                        retainedDescValidRuntime(view.down, hidden, args.routed_width, down_cb, down_promoted,
                            args.down_source_superblock, can_promote_down && moved);
                    if (!gate_valid || !down_valid)
                        __trap();
                    const int packed = moved ? canonical_count + __popc(promoted & prefix_mask) : __popc(canonical & prefix_mask);
                    args.route_order[kRetainedRouteCounts + packed] = slot;
                    packed_slots[packed] = static_cast<uint8_t>(slot);
                }
                else if (in_route)
                    args.route_order[kRetainedRouteCounts + local_count + slot - __popc(present & prefix_mask)] = slot;
            }
            __syncthreads();
            {
                const int n_copy = args.route_order[0];
                static_assert(sizeof(DeviceNativeQuantisedMatrixDesc) % sizeof(uint64_t) == 0);
                constexpr unsigned words_per_mat = sizeof(DeviceNativeQuantisedMatrixDesc) / sizeof(uint64_t);
                for (int packed = static_cast<int>(warp); packed < n_copy; packed += 8)
                {
                    const auto& view = preloaded_expert[packed_slots[packed]];
                    auto& dst = args.selected[packed];
                    if (lane < 3u * words_per_mat)
                    {
                        const unsigned mat = (lane < words_per_mat) ? 0u : (lane < 2u * words_per_mat ? 1u : 2u);
                        const unsigned w = lane - mat * words_per_mat;
                        const auto* s_mat = (mat == 0u) ? &view.gate : (mat == 1u ? &view.up : &view.down);
                        auto* d_mat = (mat == 0u) ? &dst.gate : (mat == 1u ? &dst.up : &dst.down);
                        reinterpret_cast<uint64_t*>(d_mat)[w] = reinterpret_cast<const uint64_t*>(s_mat)[w];
                    }
                }
            }
            __syncthreads();
            if (tid == 0u && args.profile_ns)
                args.profile_ns[12] = retainedRouteClock();
        }
        if (args.profile_ns && threadIdx.x == 0)
            args.profile_ns[6] = retainedRouteClock();
    }

    /**
     * @brief One resident program for the active original CUDA main-model row.
     * @tparam GateCodebook Fixed routed Gate/Up physical layout from original prepared sources.
     * @tparam DownCodebook Fixed routed Down layout from original prepared sources.
     * @tparam ObserveCollectives Whether the original router is the histogram publisher.
     * @tparam PublishHCActivated The graph still consumes the original activated HC row.
     * @tparam PublishHCUp The graph still consumes the original HC logits.
     * @tparam IndependentLocalShard Assign each selected local fragment on its own lane.
     * @tparam Phase Native arithmetic phase owned by this graph node.
     * @param args Complete original launch binding. Capture-selected publication demand is immutable.
     * @pre The native grid is resident and every CTA has real work for the active original row.
     *      The enclosing native capture selects this positive branch from the device count.
     */
    template<uint8_t GateCodebook, uint8_t DownCodebook, bool ObserveCollectives,
             bool PublishHCActivated, bool PublishHCUp, bool IndependentLocalShard,
        RetainedM1NativePhase Phase = RetainedM1NativePhase::Complete>
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
            std::conditional_t<Phase == RetainedM1NativePhase::FFNSuffix,
                uint32_t, RetainedHCPrefixShared<kQFNDecodeGateUpK>> hc;
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
        constexpr int output_packets = width / 32;
        if constexpr (Phase != RetainedM1NativePhase::FFNSuffix)
        {
            // The first HC rendezvous orders this invocation's reset before any
            // CTA can read the route plan. The word is private to this retained grid.
            if constexpr (IndependentLocalShard && Phase == RetainedM1NativePhase::Complete)
            {
                if (blockIdx.x == 0 && threadIdx.x == 0)
                    cuda::atomic_ref<int, cuda::thread_scope_device>(*args.route_plan_ready).store(0, cuda::memory_order_relaxed);
            }
            if constexpr (Phase == RetainedM1NativePhase::RoutePrefix)
            {
                if (args.profile_ns && blockIdx.x == 0 && threadIdx.x == 0)
                    args.profile_ns[0] = retainedRouteClock();
                const unsigned max_packets = static_cast<unsigned>(router.top_k + 1) * static_cast<unsigned>(output_packets);
                if (first_thread < max_packets)
                    args.suffix_done[first_thread] = 0u;
            }
            publishRetainedHCPrefix<PublishHCActivated, PublishHCUp, hidden, true>(args.hc, workspace.hc, grid, args.profile_ns);
            if (args.profile_ns && blockIdx.x == 0 && threadIdx.x == 0)
                args.profile_ns[4] = retainedRouteClock();

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
            if (args.profile_ns && blockIdx.x == 0 && threadIdx.x == 0)
                args.profile_ns[5] = retainedRouteClock();

            if constexpr (IndependentLocalShard && Phase == RetainedM1NativePhase::Complete)
            {
                // Shared Gate/Up depends on the Q8 row alone. Keep its complete
                // packets on independent CTAs during route and descriptor setup.
                // Higher CTAs handle shared math. Most routed CTAs can then begin
                // as soon as the selector publishes the local descriptors.
                const unsigned shared_start = gridDim.x > output_packets ? gridDim.x - output_packets : 1u;
                if (blockIdx.x >= shared_start)
                {
                    for (unsigned packet = blockIdx.x - shared_start; packet < output_packets;
                         packet += gridDim.x - shared_start)
                    {
                        retainedGateUpPacket<SharedCodebook, width>(args.shared, packet,
                            args.shared_workspace.down_codes, args.shared_workspace.down_scales,
                            args.hidden_codes, args.hidden_scales, workspace.gate_up);
                        __syncthreads();
                    }
                }
            }

            if (blockIdx.x == 1 && args.shared_publication == RetainedM1SharedPublication::GatedComplete)
            {
                const float factor = retainedSharedGateFactor(args.hc.mixed, args.shared_gate_input,
                    hidden, workspace.shared_factor);
                if (threadIdx.x == 0)
                    *args.shared_gate_factor = factor;
            }

            if (blockIdx.x == 0)
            {
                if constexpr (IndependentLocalShard)
                    softmax_topk_decode_independent_body<ObserveCollectives,
                        llaminar2::RoutedExpertRowExecutionPolicy::IntermediateShardLocal, false, kRetainedQFNExpertCount>(
                        router.logits, runtime, router.route_indices, router.route_weights, router.experts, router.top_k,
                        router.normalize, true, nullptr, {});
                else
                    softmax_topk_decode_runtime_body<ObserveCollectives, false, kRetainedQFNExpertCount>(router.logits, runtime,
                        nullptr, router.route_indices, router.route_weights, router.experts, router.top_k, router.normalize, true,
                        router.local_policy, router.position, nullptr, 0, nullptr, nullptr, 0, {}, nullptr, nullptr, -1, 0, nullptr);
                __syncthreads();
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
#if CUDART_VERSION >= 12080
                        if constexpr (Phase == RetainedM1NativePhase::RoutePrefix)
                        {
                            const unsigned experts_active = static_cast<unsigned>(local_count + 1);
                            if (cudaGraphKernelNodeSetGridDim(
                                    static_cast<cudaGraphDeviceNode_t>(args.suffix_gate_up_node),
                                    dim3(retained_suffix_gate_up_k_splits,
                                        experts_active * static_cast<unsigned>(output_packets))) != cudaSuccess ||
                                cudaGraphKernelNodeSetGridDim(
                                    static_cast<cudaGraphDeviceNode_t>(args.suffix_down_node),
                                    dim3(static_cast<unsigned>(hidden / 16), experts_active)) != cudaSuccess)
                                __trap();
                        }
#endif
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
            if constexpr (Phase == RetainedM1NativePhase::Complete)
            {
                if constexpr (IndependentLocalShard)
                {
                    if (blockIdx.x == 0)
                    {
                        __syncthreads();
                        if (threadIdx.x == 0)
                            cuda::atomic_ref<int, cuda::thread_scope_device>(*args.route_plan_ready).store(1, cuda::memory_order_release);
                    }
                    else if (threadIdx.x == 0)
                    {
                        const auto start = retainedRouteClock();
                        const cuda::atomic_ref<int, cuda::thread_scope_device> publication(*args.route_plan_ready);
                        while (publication.load(cuda::memory_order_acquire) == 0)
                        {
                            if (retainedRouteClock() - start >= 30'000'000'000ULL)
                                __trap();
                            __nanosleep(64);
                        }
                    }
                    __syncthreads();
                }
                else
                    grid.sync();

            }
        }
        if constexpr (Phase == RetainedM1NativePhase::RoutePrefix)
        {
            if (args.profile_ns && blockIdx.x == 0 && threadIdx.x == 0)
                args.profile_ns[6] = retainedRouteClock();
            // Native kernel completion orders the whole descriptor bank for
            // every reader. No resident lane polls for this captured edge.
            return;
        }
        if constexpr (IndependentLocalShard && Phase == RetainedM1NativePhase::FFNSuffix)
        {
            // Both source banks have retired at the captured prefix edge.
            // The shared packets keep their existing disjoint CTA owners.
            const unsigned shared_start = gridDim.x > output_packets ? gridDim.x - output_packets : 1u;
            if (blockIdx.x >= shared_start)
            {
                for (unsigned packet = blockIdx.x - shared_start; packet < output_packets;
                     packet += gridDim.x - shared_start)
                {
                    retainedGateUpPacket<SharedCodebook, width>(args.shared, packet,
                        args.shared_workspace.down_codes, args.shared_workspace.down_scales,
                        args.hidden_codes, args.hidden_scales, workspace.gate_up);
                    __syncthreads();
                }
            }
        }
        const int local_count = args.route_order[0];
        if constexpr (can_promote_gate)
        {
            const int canonical_count = args.route_order[1];
            retainedRoutedGateUpRange<GateCodebook, width>(args, 0, canonical_count, workspace.gate_up);
            retainedRoutedGateUpRange<Promoted, width>(args, canonical_count, local_count, workspace.gate_up);
        }
        else
            retainedRoutedGateUpRange<GateCodebook, width>(args, 0, local_count, workspace.gate_up);
        if constexpr (!IndependentLocalShard)
        {
            const unsigned routed_packets = local_count * output_packets;
            for (unsigned packet = (blockIdx.x + gridDim.x - routed_packets % gridDim.x) % gridDim.x;
                 packet < output_packets; packet += gridDim.x)
            {
                retainedGateUpPacket<SharedCodebook, width>(args.shared, packet,
                    args.shared_workspace.down_codes, args.shared_workspace.down_scales,
                    args.hidden_codes, args.hidden_scales, workspace.gate_up);
                __syncthreads();
            }
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
        unsigned first_shared_column = first_thread;
        unsigned shared_stride = threads;
        if constexpr (Phase == RetainedM1NativePhase::FFNSuffix)
        {
            // Every launched CTA owns a half warp of real shared columns,
            // even when this participant has no routed experts in the row.
            if (threadIdx.x >= retained_shared_down_frontier_columns)
                return;
            first_shared_column = blockIdx.x * retained_shared_down_frontier_columns + threadIdx.x;
            shared_stride = gridDim.x * retained_shared_down_frontier_columns;
        }
        else if constexpr (IndependentLocalShard)
        {
            // Keep the shared Down tiles off the lowest CTAs, which own the
            // routed Down prefix for the active original routes.
            constexpr unsigned shared_tiles = (hidden + kThreads - 1) / kThreads;
            const unsigned shared_start = gridDim.x > shared_tiles ? gridDim.x - shared_tiles : 0u;
            const unsigned tile = blockIdx.x >= shared_start ? blockIdx.x - shared_start
                : gridDim.x - shared_start + blockIdx.x;
            first_shared_column = tile * blockDim.x + threadIdx.x;
        }
        for (unsigned column = first_shared_column; column < static_cast<unsigned>(hidden); column += shared_stride)
        {
            const float raw = retainedDown<SharedCodebook, false>(args.shared.down, column,
                args.shared_workspace.down_codes, args.shared_workspace.down_scales);
            args.shared_contribution[column] = gate_shared
                ? llaminar2::device_fp32_contract::multiply(shared_factor, raw) : raw;
        }
    }
