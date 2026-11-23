/**
 * @file ROCmMoEGroupedPopulationProjection.hip.inc
 * @brief Shared exact arithmetic for admitted Qwen and Compact population producers.
 *
 * Both arithmetic branches preserve the source partition order and share the
 * shard-local IQ table and the same contribution helpers as explicit WMMA leaves.
 * Selection happens before LDS publication; each directory row/column has one
 * writer. A global native choice changes the directory step as well as arithmetic,
 * preventing overlapping sixteen-row writers on a twelve-row directory.
 */
/** @brief Compiled arithmetic ownership within the retained population transaction. */
enum class PopulationArithmetic { Mixed, Ordered, Native };
/** @brief One compile-time authority for admitted row, column and partition geometry.
 * @tparam TwoRoles Gate/up versus down. @tparam Domain Complete immutable operation. */
template<bool TwoRoles,MoEGroupedPopulationDomain Domain>
struct PopulationProjectionGeometry
{
    static constexpr bool compact = Domain == MoEGroupedPopulationDomain::Compact;
    static constexpr int tm = !compact && TwoRoles ? 12 : 16;
    static constexpr int n = compact ? (TwoRoles ? 256 : 512) : (TwoRoles ? 512 : 2048);
    static constexpr int k = compact ? (TwoRoles ? 512 : 256) : (TwoRoles ? 2048 : 512);
    static constexpr int partitions = MoEProjectionNumericalContract::orderedKPartitionsForWidth(k);
    static constexpr int span = (k / 32 + partitions - 1) / partitions;
    static constexpr int kb = span < 2 ? span : 2;
};

/** @brief Typed LDS ownership and borrowed branch views for one population CTA.
 * @tparam FMT Physical decoder. @tparam TM Ordered rows. @tparam KB Native K32 stages. */
template<NativeQuantisedGemmFormat FMT,int TM,int KB>
struct PopulationProjectionStorage
{
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    static constexpr int sums = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 4 :
        Traits::is_dual_scale_asym ? 2 : Traits::is_asymmetric ? 1 : 0;
    struct Native
    {
        I32x4 a[KB][16][2], b[KB][32][2];
        uint16_t primary[KB][32], secondary[KB][Traits::needs_aux_compute ? 32 : 1];
        uint32_t minima[KB][Traits::is_dual_scale_asym ? 32 : 1];
        uint16_t qh[KB][FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 32 : 1];
        int sums[KB][PopulationProjectionStorage::sums ? PopulationProjectionStorage::sums : 1][16];
        alignas(16) float scales[KB][4][4];
    };
    struct Ordered
    {
        int words[TM][8];
        float scales[TM];
        int sums[PopulationProjectionStorage::sums ? PopulationProjectionStorage::sums : 1][TM];
    };
    struct Mixed { union { Native native; Ordered ordered; }; };
    struct Cold { Ordered ordered; };
    struct Hot { Native native; };
    /** The wrapper retains the union; each selected loop borrows only its own member. */
    struct OrderedView { Mixed &owner; };
    struct NativeView { Mixed &owner; };
    template<PopulationArithmetic Arithmetic>
    using Owned = std::conditional_t<Arithmetic == PopulationArithmetic::Mixed,Mixed,
        std::conditional_t<Arithmetic == PopulationArithmetic::Ordered,Cold,Hot>>;
    template<PopulationArithmetic Arithmetic>
    using Borrowed = std::conditional_t<Arithmetic == PopulationArithmetic::Ordered,OrderedView,NativeView>;
};
/** @tparam FMT Actual physical decoder. @tparam TwoRoles Gate/up versus down.
 * @tparam Arithmetic Mixed choice or one exclusive arithmetic producer.
 * @tparam Domain Immutable geometry and source-arithmetic family.
 * @tparam Stage Exact owned or borrowed LDS type; invalid branch views fail compilation.
 * @param v Original projection bindings. @param control Borrowed device choice and diagnostic suppression.
 * @param stage Wrapper-owned storage or its exclusive branch view. @param source_rows Same CTA's row table. */
template<NativeQuantisedGemmFormat FMT, bool TwoRoles, PopulationArithmetic Arithmetic,
    MoEGroupedPopulationDomain Domain,class Stage>
__device__ __forceinline__ void populationProjectionBody(const MoEGroupedWmmaArguments &v,
    const MoEGroupedPopulationControl &control,Stage &stage,int (&source_rows)[16])
{
#if defined(__gfx1100__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64));
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    using F32x4 = float __attribute__((ext_vector_type(4)));
    constexpr bool compact = Domain == MoEGroupedPopulationDomain::Compact;
    static_assert(compact || Domain == MoEGroupedPopulationDomain::Qwen);
    static_assert(!compact || Arithmetic != PopulationArithmetic::Mixed);
    using Geometry = PopulationProjectionGeometry<TwoRoles,Domain>;
    constexpr int TM = Geometry::tm, TN = 128, BN = 32, Threads = 128;
    constexpr int BM = 16;
    constexpr int N = Geometry::n, K = Geometry::k;
    constexpr int Source = compact ? static_cast<int>(FMT) : (TwoRoles ? 13 : 4), blocks = K / 32;
    constexpr unsigned ratio_shift = 2u;
    constexpr unsigned native_column_shift = compact ? (TwoRoles ? 3u : 4u) : (TwoRoles ? 4u : 6u);
    static_assert(MoEProjectionNumericalContract::ownsOrderedKPartitionTree(Source));
    constexpr int partitions = Geometry::partitions, span = Geometry::span, KB = Geometry::kb;
    using Storage = PopulationProjectionStorage<FMT,TM,KB>;
    using ExpectedStage = std::conditional_t<compact,typename Storage::template Borrowed<Arithmetic>,
        typename Storage::template Owned<Arithmetic>>;
    static_assert(std::is_same_v<Stage,ExpectedStage>);
    // Keep the union owner in every assignment expression. Borrowing an
    // inactive member directly would obscure its implicit lifetime activation.
    auto &bank = [&]() -> auto & {
        if constexpr (compact) return stage.owner;
        else return stage;
    }();
    constexpr auto Input = TwoRoles ? MoEGroupedWmmaInput::OriginalTokens : MoEGroupedWmmaInput::GroupedRows;
    // Every admitted domain seals its source geometry before capture. Keep the
    // device contract check, then use fixed geometry for exact work addressing.
    if (v.n != N || v.k != K || v.rows != 512 || v.top_k != 8 || v.input != Input) __builtin_trap();
    constexpr int Sums = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 4 :
        Traits::is_dual_scale_asym ? 2 : Traits::is_asymmetric ? 1 : 0;
    if constexpr (!compact)
        if (!v.admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled)) return;
    const auto family = [&] {
        if constexpr (compact) return Arithmetic == PopulationArithmetic::Ordered
            ? MoEGroupedPopulationFamily::Dot4 : MoEGroupedPopulationFamily::N32;
        else return control.binding.load();
    }();
    if (control.omitted == family) return;
    if constexpr (Arithmetic == PopulationArithmetic::Ordered)
        if (family == MoEGroupedPopulationFamily::N32) return;
    if constexpr (Arithmetic == PopulationArithmetic::Native)
        if (family == MoEGroupedPopulationFamily::Dot4) return;
    if constexpr (compact)
    {
        if (v.experts != 8 || v.dot_admission.rowPlan().isWide() ||
            v.dot_admission.rowPlan().floor() != MoEGroupedWmmaRowFloor::EveryExpert ||
            v.dot_admission.rowPlan().rows(1) != 16) __builtin_trap();
    }
    else if (!v.dot_admission.rowPlan().isWide()) __builtin_trap();
    // The captured source policy must match the canonical partition tree.
    // Constexpr bounds retain the exact folds and remove the unused down stage.
    if (v.partitions.forCodebook(Source) != partitions) __builtin_trap();
    const int active_slots = v.admission.activeSlots();
    // A global Dot4 decision has no native writers. Compact its work extent to
    // actual ordered column tiles; mixed/native families retain 32-column addressing.
    const bool coarse = Arithmetic == PopulationArithmetic::Ordered ||
        (Arithmetic == PopulationArithmetic::Mixed && family == MoEGroupedPopulationFamily::Dot4);
    const unsigned column_shift = native_column_shift - (coarse ? ratio_shift : 0u);
    const unsigned columns = 1u << column_shift;
    const auto layout = MoEGroupedPopulationDirectoryLayout(v.directory_capacity);
    const uint32_t tiles = __builtin_amdgcn_readfirstlane(v.directory[layout.liveCountIndex()]);
    if (tiles > static_cast<unsigned>(v.directory_capacity)) __builtin_trap();
    // One worker owns a disjoint residue class of the compact live work extent.
    // The publication and descriptors are uniform across all lanes of a block.
    for (uint32_t item = blockIdx.x; item < tiles * columns; item += gridDim.x)
    {
    const uint32_t column = (item & (columns - 1)) << (coarse ? ratio_shift : 0u);
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
    if (!desc.payload || desc.n != N || desc.k != K ||
        desc.blocks_per_row != static_cast<uint32_t>(K / 32) ||
        !::llaminar2::rocm_native_metadata::descriptorMetadataComplete(desc, Traits::needs_aux_compute, Traits::is_dual_scale_asym)) __builtin_trap();
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
    if (producer == MoEGroupedDotProducer::OrderedDot4 && column % (TN/BN) != 0) continue;
    const int step = compact ? 16 : populationProjectionRows(TwoRoles ? MoEPrefillProjectionRole::GateUp : MoEPrefillProjectionRole::Down,family,count);
    if (step != (producer == MoEGroupedDotProducer::Wmma ? BM : TM) || first_row % step) __builtin_trap();
    const int live = min(step,count-first_row), first_slot = offset + first_row;
    const int macro_n = (column / (TN/BN)) * TN;
    float *out = TwoRoles && blockIdx.z ? v.second_output : v.output;
    if constexpr (!compact && TwoRoles && Arithmetic != PopulationArithmetic::Native)
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
                    // One metadata authority for per-block and superblock-coded descriptors.
                    const auto block_meta = ::llaminar2::rocm_native_metadata::blockMetadata<static_cast<uint8_t>(FMT)>(
                        ::llaminar2::native_quantised_scale::metadataPlanes(desc), static_cast<uint32_t>(kb),
                        static_cast<uint32_t>(col), static_cast<uint32_t>(N));
                    const uint16_t scale = block_meta.primary;
                    const uint16_t aux = block_meta.secondary;
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
                    // One metadata authority for per-block and superblock-coded descriptors.
                    const auto block_meta = ::llaminar2::rocm_native_metadata::blockMetadata<static_cast<uint8_t>(FMT)>(
                        ::llaminar2::native_quantised_scale::metadataPlanes(desc), static_cast<uint32_t>(kb),
                        static_cast<uint32_t>(col), static_cast<uint32_t>(N));
                    scale = block_meta.primary;
                    if constexpr (Traits::needs_aux_compute) aux = block_meta.secondary;
                    if constexpr (Traits::is_dual_scale_asym) minima = block_meta.emin;
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
                for (int item = tid; item < live*8; item += Threads)
                {
                    const int row = item / 8, word = item % 8;
                    const auto input = reinterpret_cast<const int32_t *>(v.activations +
                        static_cast<size_t>(source_row(row))*K + kb*32);
                    const int value = input[word];
                    bank.ordered.words[row][word] = value;
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
                            bank.ordered.sums[word/words_per_sum][row] = sum;
                    }
                }
                if (tid < live)
                    bank.ordered.scales[tid] = v.activation_scales[static_cast<size_t>(source_row(tid))*blocks + kb];
                __syncthreads();
                if (owns_column)
                    wideVisitRows<0,TM>(live,[&](auto index) {
                        constexpr int row = decltype(index)::value;
                        int low = 0;
                        #pragma unroll
                        for (int word = 0; word < 4; ++word)
                            low = rocm_integer_contract::signedDot4(bank.ordered.words[row][word],decoded[word],low);
                        int high = Traits::is_dual_scale ? 0 : low;
                        #pragma unroll
                        for (int word = 4; word < 8; ++word)
                            high = rocm_integer_contract::signedDot4(bank.ordered.words[row][word],decoded[word],high);
                        I32x4 sums = {};
                        #pragma unroll
                        for (int group = 0; group < Sums; ++group)
                            sums[group] = bank.ordered.sums[group][row];
                        if constexpr (compact)
                        {
                            // Compact has exactly one K32 block per canonical
                            // partition. Commit from the same positive zero,
                            // then fold this row immediately in partition order.
                            // Independent row folds need not retain sixteen
                            // partials across the staging-retirement barrier.
                            static_assert(span == 1 && blocks == partitions && partitions > 1);
                            const float one = wideCommit<FMT>(0.0f,low,high,weight,aux,minima,qh,bank.ordered.scales[row],sums);
                            total[row] = __fadd_rn(total[row],one);
                        }
                        else
                            partial[row] = wideCommit<FMT>(partial[row],low,high,weight,aux,minima,qh,bank.ordered.scales[row],sums);
                    });
                __syncthreads();
            }
            if constexpr (!compact) wideVisitRows<0,TM>(live,[&](auto index) {
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
                for (int item = tid; item < stages*BN; item += Threads)
                {
                    const int slice = item/BN, col = item%BN, kb = first+slice;
                    const size_t linear = static_cast<size_t>(kb)*N + n0+col;
                    uint32_t raw[Traits::payload_dwords] = {};
                    uint16_t scale = 0, aux = 0; uint32_t minima = 0;
                    if (n0+col < N)
                    {
                        load_payload<Traits::payload_bytes>(desc.payload + linear*Traits::payload_bytes,raw);
                        // One metadata authority for per-block and superblock-coded descriptors.
                        const auto block_meta = ::llaminar2::rocm_native_metadata::blockMetadata<static_cast<uint8_t>(FMT)>(
                            ::llaminar2::native_quantised_scale::metadataPlanes(desc), static_cast<uint32_t>(kb),
                            static_cast<uint32_t>(n0+col), static_cast<uint32_t>(N));
                        scale = block_meta.primary;
                        if constexpr (Traits::needs_aux_compute) aux = block_meta.secondary;
                        if constexpr (Traits::is_dual_scale_asym) minima = block_meta.emin;
                    }
                    int decoded[8]; decode_block<FMT>(raw,decoded);
                    I32x4 lo, hi;
                    #pragma unroll
                    for (int word = 0; word < 4; ++word) { lo[word] = decoded[word]; hi[word] = decoded[word+4]; }
                    bank.native.b[slice][col^slice][(col>>2)&1] = lo;
                    bank.native.b[slice][col^slice][((col>>2)&1)^1] = hi;
                    bank.native.primary[slice][col] = scale;
                    if constexpr (Traits::needs_aux_compute) bank.native.secondary[slice][col] = aux;
                    if constexpr (Traits::is_dual_scale_asym) bank.native.minima[slice][col] = minima;
                    if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M) bank.native.qh[slice][col] = static_cast<uint16_t>(raw[1]);
                }
                for (int item = tid; item < stages*BM*2; item += Threads)
                {
                    const int slice = item/(BM*2), row = item%(BM*2)/2, half = item%2;
                    const int kb = first+slice;
                    I32x4 values = {};
                    if (row < live)
                        values = reinterpret_cast<const I32x4 *>(v.activations +
                            static_cast<size_t>(source_row(row))*K + kb*32)[half];
                    bank.native.a[slice][row^slice][half^((row>>2)&1)] = values;
                    if constexpr (Sums == 1)
                    {
                        int sum = 0;
                        #pragma unroll
                        for (int word = 0; word < 4; ++word) sum = rocm_integer_contract::signedDot4(values[word],0x01010101,sum);
                        sum += __shfl_xor(sum,1,32);
                        if (half == 0) bank.native.sums[slice][0][row] = sum;
                    }
                    else if constexpr (Sums > 1)
                        #pragma unroll
                        for (int group = 0; group < Sums/2; ++group)
                        {
                            int sum = 0;
                            #pragma unroll
                            for (int word = 0; word < 8/Sums; ++word)
                                sum = rocm_integer_contract::signedDot4(values[group*(8/Sums)+word],0x01010101,sum);
                            bank.native.sums[slice][half*(Sums/2)+group][row] = sum;
                        }
                    if (half == 0) bank.native.scales[slice][row % 4][row / 4] = row < live ?
                        v.activation_scales[static_cast<size_t>(source_row(row))*blocks + kb] : 1.0f;
                }
                __syncthreads();
                #pragma unroll 1
                for (int slice = 0; slice < KB && first+slice < end; ++slice)
                {
                    const int col = wave*16 + lane%16, ar = lane%16;
                    const I32x4 bl = bank.native.b[slice][col^slice][(col>>2)&1];
                    const I32x4 bh = bank.native.b[slice][col^slice][((col>>2)&1)^1];
                    const I32x4 al = bank.native.a[slice][ar^slice][(ar>>2)&1];
                    const I32x4 ah = bank.native.a[slice][ar^slice][((ar>>2)&1)^1];
                    const I32x4 zero = {};
                    const auto low = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,al,true,bl,zero,false);
                    const auto high = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true,ah,true,bh,Traits::is_dual_scale ? zero : low,false);
                    const uint16_t scale = bank.native.primary[slice][col];
                    const uint16_t aux = Traits::needs_aux_compute ? bank.native.secondary[slice][col] : uint16_t{0};
                    const uint32_t minima = Traits::is_dual_scale_asym ? bank.native.minima[slice][col] : uint32_t{0};
                    const uint16_t qh = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? bank.native.qh[slice][col] : uint16_t{0};
                    const auto weight = [&] {
                        if constexpr (Traits::is_dual_scale) return rocm_native_quantised_load_dual_weight_scales<FMT>(scale,aux);
                        else return ROCmNativeQuantisedDualWeightScales{rocm_native_quantised_load_single_weight_scale<FMT>(scale,aux),0.0f};
                    }();
                    // Copy bits from one aligned group; the commit order and
                    // floating-point operations below remain exactly unchanged.
                    F32x4 activation_scales;
                    __builtin_memcpy(&activation_scales,bank.native.scales[slice][lane/16],sizeof(activation_scales));
                    #pragma unroll
                    for (int element = 0; element < 4; ++element)
                    {
                        const int row = element*4 + lane/16;
                        I32x4 sums = {};
                        #pragma unroll
                        for (int group = 0; group < Sums; ++group) sums[group] = bank.native.sums[slice][group][row];
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

/** @brief Preserve Qwen's exact retained kernel identity while sharing arithmetic.
 * @tparam FMT Physical decoder. @tparam TwoRoles GU versus down.
 * @tparam Arithmetic Exclusive GU or mixed down ownership.
 * @param v Original projection packet. @param control Device choice and test suppression. */
template<NativeQuantisedGemmFormat FMT, bool TwoRoles, PopulationArithmetic Arithmetic = PopulationArithmetic::Mixed>
__global__ __launch_bounds__(128) void populationProjection(MoEGroupedWmmaArguments v, MoEGroupedPopulationControl control)
{
#if defined(__gfx1100__)
    using Geometry = PopulationProjectionGeometry<TwoRoles,MoEGroupedPopulationDomain::Qwen>;
    using Storage = PopulationProjectionStorage<FMT,Geometry::tm,Geometry::kb>;
    __shared__ typename Storage::template Owned<Arithmetic> stage;
    __shared__ int source_rows[16];
    populationProjectionBody<FMT,TwoRoles,Arithmetic,MoEGroupedPopulationDomain::Qwen>(v,control,stage,source_rows);
#endif
}

/** @brief Compact's adaptive consumer shares fixed TM16 rows between arithmetic branches.
 * @tparam FMT Source-preserving physical decoder. @tparam TwoRoles GU versus down.
 * @tparam Arithmetic Compiled branch ownership; Mixed retains the device choice.
 * @param v Original projection packet. @param control Device choice and test suppression. */
template<NativeQuantisedGemmFormat FMT, bool TwoRoles, PopulationArithmetic Arithmetic>
__global__ __launch_bounds__(128)
void populationProjectionCompact(MoEGroupedWmmaArguments v, MoEGroupedPopulationControl control)
{
#if defined(__gfx1100__)
    static_assert(Arithmetic == PopulationArithmetic::Mixed);
    if (!v.admission.allows(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled)) return;
    const auto family = control.binding.load();
    using Geometry = PopulationProjectionGeometry<TwoRoles,MoEGroupedPopulationDomain::Compact>;
    using Storage = PopulationProjectionStorage<FMT,Geometry::tm,Geometry::kb>;
    // One block-wide decision owns the entire loop. Both specialized loops
    // borrow the same union and row table; exclusive control flow alone would
    // not guarantee that separately declared LDS banks share physical bytes.
    __shared__ typename Storage::Mixed stage;
    __shared__ int source_rows[16];
    if (family == MoEGroupedPopulationFamily::Dot4)
    {
        typename Storage::OrderedView view{stage};
        populationProjectionBody<FMT,TwoRoles,PopulationArithmetic::Ordered,MoEGroupedPopulationDomain::Compact>(
            v,control,view,source_rows);
    }
    else if (family == MoEGroupedPopulationFamily::N32)
    {
        typename Storage::NativeView view{stage};
        populationProjectionBody<FMT,TwoRoles,PopulationArithmetic::Native,MoEGroupedPopulationDomain::Compact>(
            v,control,view,source_rows);
    }
    else __builtin_trap();
#endif
}
