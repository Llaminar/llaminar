/**
 * @file ROCmMoEGroupedPopulationProjection.hip.inc
 * @brief Exclusive gate/up and fused down producers for the admitted Qwen operation.
 *
 * Both arithmetic branches preserve the source partition order and share the
 * shard-local IQ table and the same contribution helpers as explicit WMMA leaves.
 * Selection happens before LDS publication; each directory row/column has one
 * writer. A global native choice changes the directory step as well as arithmetic,
 * preventing overlapping sixteen-row writers on a twelve-row directory.
 */
/** @brief Compiled arithmetic ownership within the retained population transaction. */
enum class PopulationArithmetic { Mixed, Ordered, Native };
/** @tparam FMT Actual physical decoder. @tparam TwoRoles Gate/up versus down.
 * @tparam Arithmetic Mixed Down or one exclusive GU arithmetic producer.
 * @param v Original projection bindings. @param control Borrowed device choice and diagnostic suppression. */
template<NativeQuantisedGemmFormat FMT, bool TwoRoles, PopulationArithmetic Arithmetic = PopulationArithmetic::Mixed>
__global__ __launch_bounds__(128) void populationProjection(MoEGroupedWmmaArguments v, MoEGroupedPopulationControl control)
{
#if defined(__gfx1100__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64));
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    using F32x4 = float __attribute__((ext_vector_type(4)));
    constexpr int TM = TwoRoles ? 12 : 16, TN = 128, BN = 32;
    constexpr int BM = 16;
    constexpr int N = TwoRoles ? 512 : 2048, K = TwoRoles ? 2048 : 512;
    constexpr int Source = TwoRoles ? 13 : 4, blocks = K / 32;
    static_assert(MoEProjectionNumericalContract::ownsOrderedKPartitionTree(Source));
    constexpr int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(K);
    constexpr int span = (blocks + partitions - 1) / partitions;
    constexpr int KB = span < 2 ? span : 2;
    constexpr auto Input = TwoRoles ? MoEGroupedWmmaInput::OriginalTokens : MoEGroupedWmmaInput::GroupedRows;
    // This shard implements only the exact admitted Qwen operation. Keep the
    // captured contract check, then use its fixed geometry for work addressing.
    if (v.n != N || v.k != K || v.rows != 512 || v.top_k != 8 || v.input != Input) __builtin_trap();
    constexpr int Sums = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 4 :
        Traits::is_dual_scale_asym ? 2 : Traits::is_asymmetric ? 1 : 0;
    if (!v.admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled)) return;
    const auto family = control.binding.load();
    if (control.omitted == family) return;
    if constexpr (Arithmetic == PopulationArithmetic::Ordered)
        if (family == MoEGroupedPopulationFamily::N32) return;
    if constexpr (Arithmetic == PopulationArithmetic::Native)
        if (family == MoEGroupedPopulationFamily::Dot4) return;
    if (!v.dot_admission.rowPlan().isWide()) __builtin_trap();
    // The captured source policy must match the canonical partition tree.
    // Constexpr bounds retain the exact folds and remove the unused down stage.
    if (v.partitions.forCodebook(Source) != partitions) __builtin_trap();
    const int active_slots = v.admission.activeSlots();
    // A global Dot4 decision has no native writers. Compact its work extent to
    // real 128-column tiles; mixed/native families retain 32-column addressing.
    const bool coarse = Arithmetic == PopulationArithmetic::Ordered ||
        (Arithmetic == PopulationArithmetic::Mixed && family == MoEGroupedPopulationFamily::Dot4);
    const unsigned column_shift = (TwoRoles ? 4u : 6u) - (coarse ? 2u : 0u);
    const unsigned columns = 1u << column_shift;
    const auto layout = MoEGroupedPopulationDirectoryLayout(v.directory_capacity);
    const uint32_t tiles = __builtin_amdgcn_readfirstlane(v.directory[layout.liveCountIndex()]);
    if (tiles > static_cast<unsigned>(v.directory_capacity)) __builtin_trap();
    struct NativeStage
    {
        I32x4 a[KB][BM][2], b[KB][BN][2];
        uint16_t primary[KB][BN], secondary[KB][Traits::needs_aux_compute ? BN : 1];
        uint32_t minima[KB][Traits::is_dual_scale_asym ? BN : 1];
        uint16_t qh[KB][FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? BN : 1];
        int sums[KB][Sums ? Sums : 1][BM];
        // A WMMA lane owns rows group + 4*element. Scalar stores preserve
        // disjoint addresses; the consuming lane reads its four scales together.
        alignas(16) float scales[KB][4][4];
    };
    struct OrderedStage
    {
        int words[TM][8];
        float scales[TM];
        int sums[Sums ? Sums : 1][TM];
    };
    struct MixedStage { union { NativeStage native; OrderedStage ordered; }; };
    struct ColdStage { OrderedStage ordered; };
    struct HotStage { NativeStage native; };
    using Stage = std::conditional_t<Arithmetic == PopulationArithmetic::Mixed,MixedStage,
        std::conditional_t<Arithmetic == PopulationArithmetic::Ordered,ColdStage,HotStage>>;
    __shared__ Stage stage;
    __shared__ int source_rows[BM];
    // One worker owns a disjoint residue class of the compact live work extent.
    // The publication and descriptors are uniform across all lanes of a block.
    for (uint32_t item = blockIdx.x; item < tiles * columns; item += gridDim.x)
    {
    const uint32_t column = (item & (columns - 1)) << (coarse ? 2u : 0u);
    const uint32_t tile = v.directory[item >> column_shift];
    if (tile == UINT32_MAX) __builtin_trap();
    const int expert = tile & moe_grouped_directory::kExpertMask;
    const int first_row = tile >> moe_grouped_directory::kExpertBits;
    if (expert >= v.experts) __builtin_trap();
    const int count = v.group_counts[expert], offset = v.group_offsets[expert];
    if (count <= first_row || offset < 0 ||
        static_cast<int64_t>(offset) + count > active_slots) __builtin_trap();
    const auto desc = TwoRoles && blockIdx.z ? v.second_descriptors[expert] : v.descriptors[expert];
    if (desc.codebook_id != FMT) continue;
    int source = desc.source_identity_present ? desc.source_codebook_id : desc.codebook_id;
    if (source == 20 || source == 21) source = 19;
    if (source != Source) __builtin_trap();
    if (!desc.payload || !desc.scales || desc.n != N || desc.k != K ||
        desc.blocks_per_row != static_cast<uint32_t>(K / 32) ||
        (Traits::needs_aux_compute && !desc.mins) ||
        (Traits::is_dual_scale_asym && !desc.emins)) __builtin_trap();
    const auto producer = populationProjectionProducer(family,count);
    if (producer == MoEGroupedDotProducer::Invalid) __builtin_trap();
    if constexpr (Arithmetic == PopulationArithmetic::Ordered)
        if (producer != MoEGroupedDotProducer::OrderedDot4) continue;
    if constexpr (Arithmetic == PopulationArithmetic::Native)
        if (producer != MoEGroupedDotProducer::Wmma) continue;
    switch (v.branch_suppression)
    {
    case MoEGroupedBranchSuppression::None: break;
    case MoEGroupedBranchSuppression::Ordered:
        if (producer == MoEGroupedDotProducer::OrderedDot4) continue;
        break;
    case MoEGroupedBranchSuppression::Wmma:
        if (producer == MoEGroupedDotProducer::Wmma) continue;
        break;
    default: __builtin_trap();
    }
    // Mixed families retain fine native columns, with only each fourth cold
    // item owning a wide tile. In wholly Dot4 work every compact item already
    // maps to that fourth column, so no iteration is spent on inactive columns.
    if (producer == MoEGroupedDotProducer::OrderedDot4 && column % 4 != 0) continue;
    const int step = populationProjectionRows(TwoRoles ? MoEPrefillProjectionRole::GateUp : MoEPrefillProjectionRole::Down,family,count);
    if (step != (producer == MoEGroupedDotProducer::Wmma ? BM : TM) || first_row % step) __builtin_trap();
    const int live = min(step,count-first_row), first_slot = offset + first_row;
    const int macro_n = (column / 4) * TN;
    float *out = TwoRoles && blockIdx.z ? v.second_output : v.output;
    if constexpr (TwoRoles && Arithmetic != PopulationArithmetic::Native)
    if (producer == MoEGroupedDotProducer::OrderedDot4 && live == 1)
    {
        // One-row experts cannot amortize shared activation staging. Every
        // column lane consumes the same immutable Q8 row directly; scalar
        // accumulators retain the exact partition and source-scale order.
        // This also owns one-row tails of cold12-row tiles, but never native
        // tails. The preceding staged iteration has already retired its LDS
        // readers; this branch neither reads nor writes the shared banks.
        static_assert(FMT == NATIVE_QUANTISED_GEMM_IQ2_S && Sums == 0);
        const int row = __builtin_amdgcn_readfirstlane(v.group_token_indices[first_slot]);
        if (row < 0 || row >= 512) __builtin_trap();
        const int col = macro_n + threadIdx.x;
        if (col < N)
        {
            float total = 0.0f;
            #pragma unroll 1
            for (int p = 0; p < partitions; ++p)
            {
                float partial = 0.0f;
                for (int kb = p*span; kb < min((p+1)*span,blocks); ++kb)
                {
                    const size_t linear = static_cast<size_t>(kb)*N + col;
                    uint32_t raw[Traits::payload_dwords];
                    load_payload<Traits::payload_bytes>(desc.payload + linear*Traits::payload_bytes,raw);
                    const uint16_t scale = static_cast<const uint16_t *>(desc.scales)[linear];
                    const uint16_t aux = static_cast<const uint16_t *>(desc.mins)[linear];
                    int decoded[8]; decode_block<FMT>(raw,decoded);
                    const auto weight = rocm_native_quantised_load_dual_weight_scales<FMT>(scale,aux);
                    const auto input = v.activations + static_cast<size_t>(row)*K + kb*32;
                    I32x4 a_low, a_high;
                    __builtin_memcpy(&a_low,input,sizeof(a_low));
                    __builtin_memcpy(&a_high,input+sizeof(a_low),sizeof(a_high));
                    int low = 0, high = 0;
                    #pragma unroll
                    for (int word = 0; word < 4; ++word)
                        low = rocm_integer_contract::signedDot4(a_low[word],decoded[word],low);
                    #pragma unroll
                    for (int word = 0; word < 4; ++word)
                        high = rocm_integer_contract::signedDot4(a_high[word],decoded[word+4],high);
                    partial = wideCommit<FMT>(partial,low,high,weight,aux,0,0,
                        v.activation_scales[static_cast<size_t>(row)*blocks+kb],I32x4{});
                }
                total = __fadd_rn(total,partial);
            }
            out[static_cast<size_t>(first_slot)*N+col] = total;
        }
        // Uniform tile ownership allows the other wave to advance independently
        // until the next staged publication barrier; no LDS remains live here.
        continue;
    }
    if constexpr (Input != MoEGroupedWmmaInput::GroupedRows)
    {
        // Only GU needs the indirect token map. Publication uses the actual
        // hardware index, separate from the cold branch's compiler fence.
        const uint32_t publish_tid = threadIdx.x;
        if (publish_tid < live)
        {
            const int index = v.group_token_indices[first_slot + publish_tid];
            const int row = Input == MoEGroupedWmmaInput::OriginalRouteSlots ? index / 8 : index;
            if (index < 0 || row < 0 || row >= 512) __builtin_trap();
            source_rows[publish_tid] = row;
        }
        __syncthreads();
    }
    // Down consumes contiguous grouped rows whose full interval was validated
    // above. Deriving the same addresses removes a redundant LDS publication,
    // its barrier, and a source-index read in every K stage. Each arithmetic
    // branch's final K barrier already retires readers before the next tile.
    const auto source_row = [&](int row) {
        if constexpr (Input == MoEGroupedWmmaInput::GroupedRows) return first_slot + row;
        else return source_rows[row];
    };

    if constexpr (Arithmetic != PopulationArithmetic::Native)
    if (producer == MoEGroupedDotProducer::OrderedDot4)
    {
        // Prevent cross-iteration vectorization of the ordered row accumulators.
        // This identity fence emits no hardware instruction or memory operation.
        uint32_t tid = threadIdx.x;
        asm volatile("" : "+v"(tid));
        const int col = macro_n + tid;
        const bool owns_column = col < N;
        float total[TM] = {};
        #pragma unroll 1
        for (int p = 0; p < partitions; ++p)
        {
            float partial[TM] = {};
            const int end = min((p+1)*span,blocks);
            for (int kb = p*span; kb < end; ++kb)
            {
                const size_t linear = static_cast<size_t>(kb)*N + col;
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
                        static_cast<size_t>(source_row(row))*K + kb*32);
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
                    stage.ordered.scales[tid] = v.activation_scales[static_cast<size_t>(source_row(tid))*blocks + kb];
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
                out[static_cast<size_t>(first_slot+row)*N + col] = total[row];
            });
        continue;
    }

    // Each native block owns one column chunk and four FP32 outputs per lane.
    // Keep the canonical K/partition order while exposing independent columns
    // to the scheduler; the ordered branch above owns its wider column extent.
    if constexpr (Arithmetic != PopulationArithmetic::Ordered)
    {
        const uint32_t tid = threadIdx.x;
        const int lane = tid % 64, wave = tid / 64;
        const int n0 = column * BN;
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
                    const size_t linear = static_cast<size_t>(kb)*N + n0+col;
                    uint32_t raw[Traits::payload_dwords] = {};
                    uint16_t scale = 0, aux = 0; uint32_t minima = 0;
                    if (n0+col < N)
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
                            static_cast<size_t>(source_row(row))*K + kb*32)[half];
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
                    if (half == 0) stage.native.scales[slice][row % 4][row / 4] = row < live ?
                        v.activation_scales[static_cast<size_t>(source_row(row))*blocks + kb] : 1.0f;
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
                    // Copy bits from one aligned group; the commit order and
                    // floating-point operations below remain exactly unchanged.
                    F32x4 activation_scales;
                    __builtin_memcpy(&activation_scales,stage.native.scales[slice][lane/16],sizeof(activation_scales));
                    #pragma unroll
                    for (int element = 0; element < 4; ++element)
                    {
                        const int row = element*4 + lane/16;
                        I32x4 sums = {};
                        #pragma unroll
                        for (int group = 0; group < Sums; ++group) sums[group] = stage.native.sums[slice][group][row];
                        partial[element] = wideCommit<FMT>(partial[element],low[element],high[element],weight,aux,minima,qh,activation_scales[element],sums);
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
            if (row < live && col < N) out[static_cast<size_t>(first_slot+row)*N + col] = total[element];
        }
    }
    // The final K-stage barrier in either branch retires all LDS readers before
    // this worker reuses staging for its next independently owned tile.
    }
#endif
}
