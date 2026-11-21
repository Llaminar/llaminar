/**
 * @file ROCmMoEGroupedWideWmma.hip.inc
 * @brief One device-selected wide Dot4 or independent native-column WMMA consumer.
 *
 * Included inside grouped_wmma_detail by its sole native translation unit.
 * Sparse experts retain one output column per thread and twelve/sixteen ordered
 * rows. Dense experts distribute native column chunks across separate blocks.
 * Ordinary wide work uses every fourth block for cold experts. The bounded
 * column layout instead encodes native quadrants in compact directory words,
 * retains coarse column grids and uses GU4/down12 cold rows. The sealed row
 * plan owns both directory offsets and arithmetic choice;
 * each branch keeps the original source partitions and contribution helpers.
 * Mutually exclusive LDS staging shares one union. No tile loop or packet copy
 * through a device helper extends the projection argument lifetime into scratch.
 */

/** @brief Preserve wide sparse geometry and native dense reuse in one captured leaf.
 * @tparam FMT Exact physical decoder. @tparam TwoRoles Gate/up versus down.
 * @tparam Adaptive Whether the ordered branch is reachable.
 * @tparam ColumnDirectory Whether compact words own native column quadrants.
 * @param v Frozen bindings, sealed shared row plan and source partition authority. */
template<NativeQuantisedGemmFormat FMT, bool TwoRoles, bool Adaptive, bool ColumnDirectory=false>
__global__ __launch_bounds__(TwoRoles ? 256 : 128) void wideProjection(MoEGroupedWmmaArguments v)
{
#if defined(__gfx1100__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64));
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    constexpr int TM = ColumnDirectory ? (TwoRoles ? 4 : 12) : (TwoRoles ? 12 : 16);
    constexpr int TN = TwoRoles ? 256 : 128, BN = TN / 4;
    constexpr int BM = 16, KB = 2;
    constexpr int Sums = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 4 :
        Traits::is_dual_scale_asym ? 2 : Traits::is_asymmetric ? 1 : 0;
    if (!v.admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled)) return;
    if (!v.dot_admission.rowPlan().isWide() || v.dot_admission.rowPlan().isColumn()!=ColumnDirectory ||
        (v.directory_layout==MoEGroupingPackedLayout::ColumnFloor8)!=ColumnDirectory) __builtin_trap();
    if (blockIdx.y >= static_cast<unsigned>(v.directory_capacity)) return;
    const uint32_t tile = v.directory[blockIdx.y];
    if (tile == UINT32_MAX) return;
    const int expert = tile & (ColumnDirectory ? 255u : moe_grouped_directory::kExpertMask);
    const int first_row = tile >> 10;
    const int quadrant = ColumnDirectory ? (tile>>8)&3 : 0;
    if (expert >= v.experts) __builtin_trap();
    const int count = v.group_counts[expert], offset = v.group_offsets[expert];
    if (count <= first_row || offset < 0 ||
        static_cast<int64_t>(offset) + count > v.admission.activeSlots()) __builtin_trap();
    if constexpr(ColumnDirectory) {
        const auto interval=moe_grouped_directory::ColumnFloor8OffsetSlots::bind(count,offset,v.directory_capacity);
        if(!interval.owns(tile,expert,blockIdx.y))__builtin_trap();
        // GU needs rows0/4 for cold counts5..7. Down12 owns the whole cold
        // expert at row0. Authenticate the row4 word before skipping it.
        if constexpr(!TwoRoles)if(count<8 && first_row!=0)return;
    }
    const auto desc = TwoRoles && blockIdx.z ? v.second_descriptors[expert] : v.descriptors[expert];
    if constexpr(ColumnDirectory)
        if(desc.codebook_id!=FMT || !desc.source_identity_present || !desc.source_is_superblock ||
            desc.source_codebook_id!=(TwoRoles ? 13 : 4))__builtin_trap();
    if (desc.codebook_id != FMT) return;
    if (!desc.payload || !desc.scales || desc.n != v.n || desc.k != v.k ||
        desc.blocks_per_row != static_cast<uint32_t>(v.k / 32) ||
        (Traits::needs_aux_compute && !desc.mins) ||
        (Traits::is_dual_scale_asym && !desc.emins)) __builtin_trap();
    const auto producer = v.dot_admission.select(desc,count);
    if (producer == MoEGroupedDotProducer::Invalid) __builtin_trap();
    if constexpr (!Adaptive)
        if (producer != MoEGroupedDotProducer::Wmma) __builtin_trap();
    switch (v.branch_suppression)
    {
    case MoEGroupedBranchSuppression::None: break;
    case MoEGroupedBranchSuppression::Ordered:
        if (producer == MoEGroupedDotProducer::OrderedDot4) return;
        break;
    case MoEGroupedBranchSuppression::Wmma:
        if (producer == MoEGroupedDotProducer::Wmma) return;
        break;
    default: __builtin_trap();
    }
    // Native columns own the fine launch grid. Cold experts retain one wide
    // column tile per four blocks; the other three return uniformly before
    // source-row publication or any barrier. This preserves disjoint writers
    // for partial column tails without serializing hot column chunks.
    if constexpr (Adaptive && !ColumnDirectory)
        if (producer == MoEGroupedDotProducer::OrderedDot4 && blockIdx.x % 4 != 0) return;
    const int step = v.dot_admission.rowPlan().rows(count);
    if (step != (producer == MoEGroupedDotProducer::Wmma ? BM : TM) || first_row % step) __builtin_trap();
    const int live = min(step,count-first_row), first_slot = offset + first_row;
    const int blocks = v.k / 32, partitions = v.partitions.forDescriptor(desc);
    if (partitions <= 0 || partitions > blocks ||
        partitions > NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) __builtin_trap();
    const int span = (blocks + partitions - 1) / partitions;
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const int macro_n = (ColumnDirectory ? blockIdx.x : blockIdx.x / 4) * TN;
    float *out = TwoRoles && blockIdx.z ? v.second_output : v.output;

    struct NativeStage
    {
        I32x4 a[KB][BM][2], b[KB][BN][2];
        uint16_t primary[KB][BN], secondary[KB][Traits::needs_aux_compute ? BN : 1];
        uint32_t minima[KB][Traits::is_dual_scale_asym ? BN : 1];
        uint16_t qh[KB][FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? BN : 1];
        int sums[KB][Sums ? Sums : 1][BM];
        float scales[KB][BM];
    };
    struct OrderedStage
    {
        int words[TM][8];
        float scales[TM];
        int sums[Sums ? Sums : 1][TM];
    };
    __shared__ union { NativeStage native; OrderedStage ordered; } stage;
    __shared__ int source_rows[BM];
    if (tid < live)
    {
        const int slot = first_slot + tid;
        const int index = v.input == MoEGroupedWmmaInput::GroupedRows ? slot : v.group_token_indices[slot];
        const int row = v.input == MoEGroupedWmmaInput::OriginalRouteSlots ? index / v.top_k : index;
        const int bound = v.input == MoEGroupedWmmaInput::GroupedRows ? v.admission.activeSlots() : v.rows;
        if (index < 0 || row < 0 || row >= bound) __builtin_trap();
        source_rows[tid] = row;
    }
    __syncthreads();

    if (Adaptive && producer == MoEGroupedDotProducer::OrderedDot4)
    {
        const int col = macro_n + tid;
        const bool owns_column = col < v.n;
        float total[TM] = {};
        #pragma unroll 1
        for (int p = 0; p < partitions; ++p)
        {
            float partial[TM] = {};
            const int end = min((p+1)*span,blocks);
            for (int kb = p*span; kb < end; ++kb)
            {
                const size_t linear = static_cast<size_t>(kb)*v.n + col;
                uint32_t raw[Traits::payload_dwords] = {};
                uint16_t scale = 0, aux = 0;
                uint32_t minima = 0;
                if (owns_column)
                {
                    load_payload<Traits::payload_bytes>(desc.payload + linear*Traits::payload_bytes,raw);
                    scale = static_cast<const uint16_t *>(desc.scales)[linear];
                    if constexpr (Traits::needs_aux_compute) aux = static_cast<const uint16_t *>(desc.mins)[linear];
                    if constexpr (Traits::is_dual_scale_asym) minima = static_cast<const uint32_t *>(desc.emins)[linear];
                }
                int decoded[8]; decode_block<FMT>(raw,decoded);
                // Complete packed-word decoding before activation publication.
                // Keeping decoder intermediates live across that barrier raises
                // VGPR pressure. This register-only compiler fence preserves all
                // bits and arithmetic order and emits no hardware barrier.
                asm volatile("" :
                    "+v"(decoded[0]), "+v"(decoded[1]),
                    "+v"(decoded[2]), "+v"(decoded[3]),
                    "+v"(decoded[4]), "+v"(decoded[5]),
                    "+v"(decoded[6]), "+v"(decoded[7]));
                const auto weight = [&] {
                    if constexpr (Traits::is_dual_scale) return rocm_native_quantised_load_dual_weight_scales<FMT>(scale,aux);
                    else return ROCmNativeQuantisedDualWeightScales{rocm_native_quantised_load_single_weight_scale<FMT>(scale,aux),0.0f};
                }();
                uint16_t qh = 0;
                if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M) qh = static_cast<uint16_t>(raw[1]);
                // As in the wide ordered leaf, each word has one cooperative
                // producer. Invalid column lanes still participate in every barrier.
                for (int item = tid; item < live*8; item += TN)
                {
                    const int row = item / 8, word = item % 8;
                    const auto input = reinterpret_cast<const int32_t *>(v.activations +
                        static_cast<size_t>(source_rows[row])*v.k + kb*32);
                    const int value = input[word];
                    stage.ordered.words[row][word] = value;
                    if constexpr (Sums > 0)
                    {
                        // Every live row contributes a complete eight-lane
                        // subgroup, even at a partial tile. Integer sums cannot
                        // overflow, so this tree preserves their exact value.
                        constexpr int words_per_sum = 8/Sums;
                        int sum = rocm_integer_contract::signedDot4(value,0x01010101,0);
                        #pragma unroll
                        for (int shift = words_per_sum/2; shift; shift /= 2)
                            sum += __shfl_down(sum,shift,words_per_sum);
                        if (word % words_per_sum == 0)
                            stage.ordered.sums[word/words_per_sum][row] = sum;
                    }
                }
                if (tid < live)
                    stage.ordered.scales[tid] = v.activation_scales[static_cast<size_t>(source_rows[tid])*blocks + kb];
                __syncthreads();
                if (owns_column)
                    wideVisitRows<0,TM>(live,[&](auto index) {
                        constexpr int row = decltype(index)::value;
                        int low = 0;
                        #pragma unroll
                        for (int word = 0; word < 4; ++word)
                            low = rocm_integer_contract::signedDot4(stage.ordered.words[row][word],decoded[word],low);
                        int high = Traits::is_dual_scale ? 0 : low;
                        #pragma unroll
                        for (int word = 4; word < 8; ++word)
                            high = rocm_integer_contract::signedDot4(stage.ordered.words[row][word],decoded[word],high);
                        I32x4 sums = {};
                        #pragma unroll
                        for (int group = 0; group < Sums; ++group)
                            sums[group] = stage.ordered.sums[group][row];
                        partial[row] = wideCommit<FMT>(partial[row],low,high,weight,aux,minima,qh,stage.ordered.scales[row],sums);
                    });
                __syncthreads();
            }
            wideVisitRows<0,TM>(live,[&](auto index) {
                constexpr int row = decltype(index)::value;
                total[row] = partitions == 1 ? partial[row] : __fadd_rn(total[row],partial[row]);
            });
        }
        if (owns_column)
            wideVisitRows<0,TM>(live,[&](auto index) {
                constexpr int row = decltype(index)::value;
                out[static_cast<size_t>(first_slot+row)*v.n + col] = total[row];
            });
        return;
    }

    // Each native block owns one column chunk and four FP32 outputs per lane.
    // Keep the canonical K/partition order while exposing independent columns
    // to the scheduler; the ordered branch above owns its wider column extent.
    {
        const int n0 = ColumnDirectory ? macro_n + quadrant*BN : blockIdx.x * BN;
        float total[4] = {};
        for (int p = 0; p < partitions; ++p)
        {
            float partial[4] = {};
            const int end = min((p+1)*span,blocks);
            for (int first = p*span; first < end; first += KB)
            {
                const int stages = min(KB,end-first);
                for (int item = tid; item < stages*BN; item += TN)
                {
                    const int slice = item/BN, col = item%BN, kb = first+slice;
                    const size_t linear = static_cast<size_t>(kb)*v.n + n0+col;
                    uint32_t raw[Traits::payload_dwords] = {};
                    uint16_t scale = 0, aux = 0; uint32_t minima = 0;
                    if (n0+col < v.n)
                    {
                        load_payload<Traits::payload_bytes>(desc.payload + linear*Traits::payload_bytes,raw);
                        scale = static_cast<const uint16_t *>(desc.scales)[linear];
                        if constexpr (Traits::needs_aux_compute) aux = static_cast<const uint16_t *>(desc.mins)[linear];
                        if constexpr (Traits::is_dual_scale_asym) minima = static_cast<const uint32_t *>(desc.emins)[linear];
                    }
                    int decoded[8]; decode_block<FMT>(raw,decoded);
                    I32x4 lo, hi;
                    #pragma unroll
                    for (int word = 0; word < 4; ++word) { lo[word] = decoded[word]; hi[word] = decoded[word+4]; }
                    stage.native.b[slice][col^slice][(col>>2)&1] = lo;
                    stage.native.b[slice][col^slice][((col>>2)&1)^1] = hi;
                    stage.native.primary[slice][col] = scale;
                    if constexpr (Traits::needs_aux_compute) stage.native.secondary[slice][col] = aux;
                    if constexpr (Traits::is_dual_scale_asym) stage.native.minima[slice][col] = minima;
                    if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M) stage.native.qh[slice][col] = static_cast<uint16_t>(raw[1]);
                }
                for (int item = tid; item < stages*BM*2; item += TN)
                {
                    const int slice = item/(BM*2), row = item%(BM*2)/2, half = item%2;
                    const int kb = first+slice;
                    I32x4 values = {};
                    if (row < live)
                        values = reinterpret_cast<const I32x4 *>(v.activations +
                            static_cast<size_t>(source_rows[row])*v.k + kb*32)[half];
                    stage.native.a[slice][row^slice][half^((row>>2)&1)] = values;
                    if constexpr (Sums == 1)
                    {
                        int sum = 0;
                        #pragma unroll
                        for (int word = 0; word < 4; ++word) sum = rocm_integer_contract::signedDot4(values[word],0x01010101,sum);
                        sum += __shfl_xor(sum,1,32);
                        if (half == 0) stage.native.sums[slice][0][row] = sum;
                    }
                    else if constexpr (Sums > 1)
                        #pragma unroll
                        for (int group = 0; group < Sums/2; ++group)
                        {
                            int sum = 0;
                            #pragma unroll
                            for (int word = 0; word < 8/Sums; ++word)
                                sum = rocm_integer_contract::signedDot4(values[group*(8/Sums)+word],0x01010101,sum);
                            stage.native.sums[slice][half*(Sums/2)+group][row] = sum;
                        }
                    if (half == 0) stage.native.scales[slice][row] = row < live ?
                        v.activation_scales[static_cast<size_t>(source_rows[row])*blocks + kb] : 1.0f;
                }
                __syncthreads();
                #pragma unroll 1
                for (int slice = 0; slice < KB && first+slice < end; ++slice)
                {
                    const int col = wave*16 + lane%16, ar = lane%16;
                    const I32x4 bl = stage.native.b[slice][col^slice][(col>>2)&1];
                    const I32x4 bh = stage.native.b[slice][col^slice][((col>>2)&1)^1];
                    const I32x4 al = stage.native.a[slice][ar^slice][(ar>>2)&1];
                    const I32x4 ah = stage.native.a[slice][ar^slice][((ar>>2)&1)^1];
                    const I32x4 zero = {};
                    const auto low = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,al,true,bl,zero,false);
                    const auto high = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,ah,true,bh,Traits::is_dual_scale ? zero : low,false);
                    const uint16_t scale = stage.native.primary[slice][col];
                    const uint16_t aux = Traits::needs_aux_compute ? stage.native.secondary[slice][col] : uint16_t{0};
                    const uint32_t minima = Traits::is_dual_scale_asym ? stage.native.minima[slice][col] : uint32_t{0};
                    const uint16_t qh = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? stage.native.qh[slice][col] : uint16_t{0};
                    const auto weight = [&] {
                        if constexpr (Traits::is_dual_scale) return rocm_native_quantised_load_dual_weight_scales<FMT>(scale,aux);
                        else return ROCmNativeQuantisedDualWeightScales{rocm_native_quantised_load_single_weight_scale<FMT>(scale,aux),0.0f};
                    }();
                    #pragma unroll
                    for (int element = 0; element < 4; ++element)
                    {
                        const int row = element*4 + lane/16;
                        I32x4 sums = {};
                        #pragma unroll
                        for (int group = 0; group < Sums; ++group) sums[group] = stage.native.sums[slice][group][row];
                        partial[element] = wideCommit<FMT>(partial[element],low[element],high[element],weight,aux,minima,qh,stage.native.scales[slice][row],sums);
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for (int element = 0; element < 4; ++element)
                total[element] = partitions == 1 ? partial[element] : __fadd_rn(total[element],partial[element]);
        }
        #pragma unroll
        for (int element = 0; element < 4; ++element)
        {
            const int row = element*4 + lane/16, col = n0 + wave*16 + lane%16;
            if (row < live && col < v.n) out[static_cast<size_t>(first_slot+row)*v.n + col] = total[element];
        }
    }
#endif
}
