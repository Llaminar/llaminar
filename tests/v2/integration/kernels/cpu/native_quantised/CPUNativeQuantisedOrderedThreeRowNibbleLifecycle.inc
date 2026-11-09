/**
 * @file CPUNativeQuantisedOrderedThreeRowNibbleLifecycle.inc
 * @brief Authored three-row ordered-candidate byte and bank lifecycle checks.
 *
 * The canonical source registry, real preparation and public independent serial
 * entry point own the oracle. Explicit WideRows selects the new AVX-512 body;
 * A separate explicit Prefill policy also exercises regular deep three-row tiles.
 * Auto, Pairwise, non-nibble and AVX2 cases remain unchanged controls. These
 * authored inputs are not a recorded model-route replay or economy evidence.
 */
#include <limits>

namespace llaminar2::test
{
namespace
{
/** @brief Withdraw a borrowed exact prepared bank before its backing expires. */
class OrderedThreeRowBankScope final
{
public:
    /**
     * @brief Bind either original prepared storage or an invocation-owned copy.
     * @param packed Original immutable prepared layout and factor authority.
     * @param workspace Exact copied bank, or null to use owned storage.
     */
    OrderedThreeRowBankScope(const CPUNativeQuantisedPackedWeights &packed,
                             const uint8_t *workspace) : packed_(packed)
    {
        if (workspace) packed_.setWorkspace(workspace);
    }
    /** @brief Revoke the borrowed pointer before local backing destruction. */
    ~OrderedThreeRowBankScope() { packed_.clearWorkspace(); }
    OrderedThreeRowBankScope(const OrderedThreeRowBankScope &) = delete;
    OrderedThreeRowBankScope &operator=(const OrderedThreeRowBankScope &) = delete;
private:
    const CPUNativeQuantisedPackedWeights &packed_; ///< Sole prepared geometry authority.
};

/** @test True three-row and four-plus-three tails retain independent serial bytes. */
TEST(CPUNativeQuantisedFusedCompletion, GPUAlignedOrderedThreeRowNibbleByteLifecycle)
{
    RecordProperty("input_scope", "authored_byte_lifecycle_not_recorded_model_routes");
    constexpr int maximum_rows = 11;
    constexpr float guard = -12345.0f;
    constexpr std::array<int8_t, 8> signed_values{-128, 127, -1, 0, 31, -17, 73, -113};
    struct Geometry { int n; int k; };
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        bool eligible_nibble = false;
        for (const auto geometry : {Geometry{65, 768}, Geometry{320, 2560}, Geometry{321, 768}})
        {
            if (geometry.n != 65 && !eligible_nibble) continue;
            SCOPED_TRACE("N=" + std::to_string(geometry.n) + " K=" + std::to_string(geometry.k));
            requireQuantizedVerifierSourceShape(format, geometry.n * 2, geometry.k);
            auto source = format.create({size_t(geometry.n * 2), size_t(geometry.k)}, 9337);
            ASSERT_NE(source, nullptr);
            // A real nonzero source interval retains asymmetric minima and IQ4_XS
            // original factors. No hand-authored packed bank is an oracle.
            CPUNativeQuantisedGemmKernel kernel(source.get(), geometry.n, geometry.n * 2,
                CPUProjectionNumericalPolicy::GPUAlignedExpert);
            ASSERT_TRUE(kernel.isValid());
            const auto &packed = kernel.packedWeights();
            ASSERT_EQ(packed.N, geometry.n);
            ASSERT_EQ(packed.K, geometry.k);
            ASSERT_EQ(packed.codebook_id, format.source_codebook_id);
            ASSERT_EQ(packed.is_superblock, format.source_is_superblock);
            if (geometry.n == 65) eligible_nibble = packed.usesNibbleLUT();
            else ASSERT_TRUE(packed.usesNibbleLUT());
            const auto workspace = packed.native_interleaved;
            const int stride = geometry.n + 7;
            const size_t extent = size_t(maximum_rows) * stride + 17;
            std::vector<Q8_1Block> input(size_t(maximum_rows) * packed.blocks_per_row);
            for (int bank : {0, 1})
            {
                SCOPED_TRACE(bank);
                OrderedThreeRowBankScope binding(packed, bank ? workspace.data() : nullptr);
                for (int workers : {1, 4})
                {
                    SCOPED_TRACE(workers);
                    FusedCompletionScope runtime(workers, std::nullopt);
                    NativeQuantisedTestPartialStorage serial_partials(packed, 1);
                    std::array<std::vector<float>, 2> output{
                        std::vector<float>(extent, guard), std::vector<float>(extent, guard)};
                    std::array<std::vector<float>, 2> expected = output;
                    int turn = 0;
                    const auto replay = [&](int rows, int live, ProjectionRowsPurpose purpose,
                                            bool force_candidate, bool existing_team, bool zero)
                    {
                        SCOPED_TRACE("rows=" + std::to_string(rows) + " live=" + std::to_string(live) +
                                     " turn=" + std::to_string(turn));
                        // Every invocation changes the live bytes. Reusing the SAME
                        // output buffers preserves earlier inactive rows/tail poison.
                        for (size_t block = 0; block < input.size(); ++block)
                        {
                            auto &activation = input[block];
                            activation.d = fp32_to_fp16(zero
                                ? ((block + turn) & 1 ? -0.0f : 0.0f)
                                : 0.0078125f * float(1 + (block + turn) % 7));
                            activation.sum_qs = 0;
                            for (int lane = 0; lane < 32; ++lane)
                            {
                                activation.qs[lane] = zero ? 0 : signed_values[
                                    (size_t(lane) + block + turn) % signed_values.size()];
                                activation.sum_qs += activation.qs[lane];
                            }
                        }
                        expected = output;
                        std::array<FusedVerifierRowsDesc, 2> descriptors;
                        for (int member = 0; member < 2; ++member)
                        {
                            const int member_rows = member == 0 ? rows : std::max(1, rows - 4);
                            const auto requested = member_rows == 1 || !force_candidate
                                ? VerifierRowsPolicy::Auto
                                : (activeISALevel() >= ISALevel::AVX512 && member_rows >= 3
                                    ? VerifierRowsPolicy::WideRows : VerifierRowsPolicy::Pairwise);
                            descriptors[member] = {.packed = &packed, .output = output[member].data(),
                                .N = geometry.n, .ldc = stride, .rows = member_rows,
                                .verifier_schedule = requested, .purpose = purpose, .policy_n = geometry.n};
                            if (member >= live) continue;
                            for (int row = 0; row < member_rows; ++row)
                                gemv_native_quantised_preq(packed,
                                    input.data() + size_t(row) * packed.blocks_per_row,
                                    expected[member].data() + size_t(row) * stride, serial_partials.span());
                        }
                        if (live == 1)
                            descriptors[1] = {}; // Inactive capacity must never become a descriptor.
                        const auto plan = planNativeQuantisedFusedRows(
                            {descriptors.data(), size_t(live)}, maximum_rows, workers, ISAPath::AUTO);
                        for (int member = 0; member < live; ++member)
                            ASSERT_EQ(plan.projections[member].k_tiles,
                                MoEProjectionNumericalContract::orderedKPartitionsForWidth(geometry.k));
                        // Two live members include a true three-row candidate, and
                        // the N65 grid supplies >=four owners for four workers.
                        if (live == 2 && rows >= 3)
                            ASSERT_EQ(plan.partial_route, NativeQuantisedFusedPartialRoute::OutputTileLocal);
                        NativeQuantisedTestPartialStorage partials(plan.sharedPartialFloats());
                        const auto invoke = [&]() {
                            return gemm_native_quantised_fused_verifier_rows_preq(input.data(),
                                descriptors.data(), partials.span(), live, maximum_rows,
                                packed.blocks_per_row, ISAPath::AUTO);
                        };
                        const auto complete = [&]() {
                            for (int member = 0; member < 2; ++member)
                                if (std::memcmp(output[member].data(), expected[member].data(),
                                                extent * sizeof(float))) return false;
                            return true;
                        };
                        ASSERT_TRUE(invoke());
                        ASSERT_TRUE(complete());
                        if (existing_team)
                        {
                            // Retire the standalone result before team entry. Only
                            // live owned cells are poisoned: inactive rows, descriptor
                            // capacity and N/ldc tails retain their prior oracle bytes.
                            // An early return cannot pass by reading the prior result.
                            for (int member = 0; member < live; ++member)
                                for (int row = 0; row < descriptors[member].rows; ++row)
                                    std::fill_n(output[member].data() + size_t(row) * stride,
                                                geometry.n, guard);
                            // Every worker observes completion immediately, before
                            // the caller's team join can hide a missing API edge.
                            std::vector<int> accepted(size_t(workers), 0);
#pragma omp parallel num_threads(workers)
                            { accepted[size_t(omp_get_thread_num())] = invoke() && complete(); }
                            for (int ok : accepted) ASSERT_EQ(ok, 1);
                        }
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        if (eligible_nibble && plan.use_avx512 && force_candidate && rows == 3)
                        {
                            // Directly exercise the new exact body for the first
                            // and final panels, including its full64 scratch writes.
                            // Fused calls above separately prove original ldc/tails.
                            const auto lut = build_decode_lut(packed.codebook_id);
                            if (geometry.n == 65 && bank == 0 && workers == 1 && turn == 1)
                            {
                                // Mutate only a local OWNED copy. An impossible
                                // symmetry bit must fail before any weight read or
                                // output store, not select a Cartesian-product body.
                                auto malformed = packed;
                                malformed.clearWorkspace();
                                malformed.is_asymmetric = !packed.is_asymmetric;
                                alignas(64) float forbidden[3][64];
                                for (auto &row : forbidden) std::fill_n(row, 64, guard);
                                EXPECT_THROW(dispatch_3row_native_avx512_ordered_partitions(
                                    malformed, input.data(), input.data() + packed.blocks_per_row,
                                    input.data() + 2 * packed.blocks_per_row,
                                    forbidden[0], forbidden[1], forbidden[2], 0,
                                    packed.blocks_per_row, plan.projections[0].k_tiles,
                                    plan.projections[0].k_blocks_per_tile, lut), std::logic_error);
                                for (const auto &row : forbidden)
                                    EXPECT_TRUE(std::all_of(row, row + 64,
                                        [guard](float value) { return value == guard; }));
                            }
                            for (int chunk : {0, (geometry.n - 1) / 64})
                            {
                                alignas(64) float candidate[3][64];
                                dispatch_3row_native_avx512_ordered_partitions(packed,
                                    input.data(), input.data() + packed.blocks_per_row,
                                    input.data() + 2 * packed.blocks_per_row,
                                    candidate[0], candidate[1], candidate[2], chunk,
                                    packed.blocks_per_row, plan.projections[0].k_tiles,
                                    plan.projections[0].k_blocks_per_tile, lut);
                                const int columns = std::min(64, geometry.n - chunk * 64);
                                for (int row = 0; row < 3; ++row)
                                    EXPECT_EQ(std::memcmp(candidate[row],
                                        expected[0].data() + size_t(row) * stride + chunk * 64,
                                        size_t(columns) * sizeof(float)), 0);
                            }
                        }
#endif
                        ++turn;
                    };
                    for (int rows : {11, 3, 7, 3, 2, 1, 3})
                        replay(rows, rows == 7 || rows == 2 ? 1 : 2,
                               ProjectionRowsPurpose::MTPVerifier, true, rows == 3, false);
                    // Purposes and Auto have distinct selection authority. Preserve
                    // their existing path while checking identical numerical bytes.
                    replay(3, 2, ProjectionRowsPurpose::Prefill, true, false, false);
                    replay(3, 2, ProjectionRowsPurpose::Decode, true, false, false);
                    replay(3, 2, ProjectionRowsPurpose::MTPVerifier, false, false, false);
                    if (eligible_nibble && geometry.n == 65)
                    {
                        std::fill(output[0].begin(), output[0].end(), guard);
                        std::fill(output[1].begin(), output[1].end(), guard);
                        replay(7, 2, ProjectionRowsPurpose::MTPVerifier, true, true, true);
                        replay(3, 1, ProjectionRowsPurpose::MTPVerifier, true, true, true);
                    }
                }
            }
        }
    }
}

/** @test A distinct physical choice cannot reinterpret old WideRows or another domain. */
TEST(CPUNativeQuantisedFusedCompletion, GPUAlignedPrefillOrderedThreeRowTileDomainContract)
{
    static_assert(static_cast<int>(VerifierRowsPolicy::FullKTwoRowPairGridNbc8)==9);
    static_assert(static_cast<int>(VerifierRowsPolicy::OrderedThreeRows)==10);
    const auto resolve=[](VerifierRowsPolicy policy, VerifierRowsScheduleDomain domain,
                          bool avx512, int rows, int partitions) {
        return resolveVerifierRowsSchedule(policy,policy,VerifierRowsScheduleGeometry{
            .rows=rows,.physical_n=65,.policy_n=130,.k_tiles=partitions,
            .ambient_n_block_chunks=1,.use_avx512=avx512,.domain=domain});
    };
    constexpr auto expert=VerifierRowsScheduleDomain::GPUAlignedExpertNibblePrefill;
    const auto candidate=resolve(VerifierRowsPolicy::OrderedThreeRows,expert,true,73,16);
    EXPECT_EQ(candidate.physical_row_tile,3);
    EXPECT_EQ(candidate.effective,VerifierRowsPolicy::OrderedThreeRows);
    EXPECT_EQ(resolve(VerifierRowsPolicy::WideRows,expert,true,73,16).physical_row_tile,4);
    EXPECT_EQ(resolve(VerifierRowsPolicy::Pairwise,expert,true,73,16).physical_row_tile,2);
    EXPECT_THROW((void)resolve(VerifierRowsPolicy::OrderedThreeRows,
        VerifierRowsScheduleDomain::General,true,73,16),std::invalid_argument);
    EXPECT_THROW((void)resolve(VerifierRowsPolicy::OrderedThreeRows,expert,false,73,16),std::invalid_argument);
    EXPECT_THROW((void)resolve(VerifierRowsPolicy::OrderedThreeRows,expert,true,2,16),std::invalid_argument);
    EXPECT_THROW((void)resolve(VerifierRowsPolicy::OrderedThreeRows,expert,true,73,1),std::invalid_argument);
}

/** @test Explicit short requests reject before writes; legacy M1 normalization remains valid. */
TEST(CPUNativeQuantisedFusedCompletion, GPUAlignedPrefillOrderedThreeRowTileRejectsShortExplicit)
{
    const auto& formats=quantizedVerifierFormats();
    const auto found=std::find_if(formats.begin(),formats.end(),
        [](const auto& format){return std::string(format.label)=="Q4_K";});
    ASSERT_NE(found,formats.end());
    constexpr int n=65,k=768,ldc=72;
    constexpr float guard=-12345.0f;
    auto source=found->create({size_t(2*n),size_t(k)},9413);
    ASSERT_NE(source,nullptr);
    CPUNativeQuantisedGemmKernel kernel(source.get(),n,2*n,
        CPUProjectionNumericalPolicy::GPUAlignedExpert);
    ASSERT_TRUE(kernel.isValid());
    FusedCompletionScope runtime(1,std::nullopt);
    const auto& packed=kernel.packedWeights();
    std::vector<Q8_1Block> input(size_t(3)*packed.blocks_per_row);
    for(auto& block:input) {
        block.d=fp32_to_fp16(0.0078125f);block.sum_qs=0;
        for(int lane=0;lane<32;++lane) {
            block.qs[lane]=lane%2?-128:127;
            block.sum_qs+=block.qs[lane];
        }
    }
    std::array<std::vector<float>,2> output{
        std::vector<float>(size_t(3)*ldc+17,guard),std::vector<float>(size_t(3)*ldc+17,guard)};
    using Desc=CPUNativeQuantisedGemmKernel::BatchedPrequantizedProjectionDesc;
    std::array<Desc,2> descriptors;
    for(int rows:{1,2}) {
        // An invalid LATER descriptor must not leave earlier output modified.
        descriptors[0]={.kernel=&kernel,.input_q8=input.data(),.output=output[0].data(),
            .rows=3,.n=n,.ldc=ldc,.purpose=ProjectionRowsPurpose::Prefill};
        descriptors[1]={.kernel=&kernel,.input_q8=input.data(),.output=output[1].data(),
            .rows=rows,.n=n,.ldc=ldc,.verifier_schedule=VerifierRowsPolicy::OrderedThreeRows,
            .purpose=ProjectionRowsPurpose::Prefill};
        const auto before=output;
        EXPECT_FALSE(CPUNativeQuantisedGemmKernel::multiply_batched_preq_decode_equivalent(
            descriptors.data(),static_cast<int>(descriptors.size()),k,{}));
        EXPECT_EQ(output,before);
        const FusedVerifierRowsDesc direct{.packed=&packed,.output=output[1].data(),
            .N=n,.ldc=ldc,.rows=rows,.verifier_schedule=VerifierRowsPolicy::OrderedThreeRows,
            .purpose=ProjectionRowsPurpose::Prefill,.policy_n=n};
        EXPECT_THROW((void)planNativeQuantisedFusedRows({&direct,1},rows,1,ISAPath::AUTO),std::invalid_argument);
        EXPECT_EQ(output,before);
    }
    // Existing legacy WideRows singleton normalization stays supported and
    // must still execute the independent decode bytes, rather than reject.
    auto expected=output;
    NativeQuantisedTestPartialStorage serial_partials(packed,1);
    for(int member=0;member<2;++member) {
        descriptors[member].rows=1;
        descriptors[member].verifier_schedule=VerifierRowsPolicy::WideRows;
        gemv_native_quantised_preq(packed,input.data(),expected[member].data(),serial_partials.span());
    }
    ASSERT_TRUE(CPUNativeQuantisedGemmKernel::multiply_batched_preq_decode_equivalent(
        descriptors.data(),static_cast<int>(descriptors.size()),k,{}));
    for(int member=0;member<2;++member)
        EXPECT_EQ(std::memcmp(output[member].data(),expected[member].data(),output[member].size()*sizeof(float)),0);
}

/** @test Explicit ownership visits each job once and breaks cyclic four-run resonance. */
TEST(CPUNativeQuantisedFusedCompletion, GPUAlignedPrefillOrderedThreeRowTileJobOwnershipContract)
{
    using cpu::native_quantised::detail::selectOrderedThreeRowStaticChunk;
    for(int workers:{1,4,7,28})
        for(size_t jobs:{size_t{1},size_t{27},size_t{28},size_t{29},size_t{127},size_t{1400},size_t{3360}}) {
            EXPECT_EQ(selectOrderedThreeRowStaticChunk(false,jobs,workers),1);
            EXPECT_EQ(selectOrderedThreeRowStaticChunk(true,jobs,workers),
                static_cast<long long>(jobs/size_t(workers)+(jobs%size_t(workers)!=0)));
        }
    EXPECT_THROW((void)selectOrderedThreeRowStaticChunk(true,0,7),std::invalid_argument);
    EXPECT_THROW((void)selectOrderedThreeRowStaticChunk(true,4,0),std::invalid_argument);
    constexpr auto largest=static_cast<size_t>(std::numeric_limits<long long>::max());
    EXPECT_EQ(selectOrderedThreeRowStaticChunk(true,largest,1),static_cast<long long>(largest));
    if constexpr(std::numeric_limits<size_t>::max()>std::numeric_limits<long long>::max())
        EXPECT_THROW((void)selectOrderedThreeRowStaticChunk(true,largest+1u,28),std::overflow_error);

    // This authored standard-workshare contract executes the SAME chunk
    // authority as the production loop, not a recorded routing/kernel replay.
    for(size_t jobs:{size_t{1},size_t{29},size_t{127},size_t{1400},size_t{3360}}) {
        FusedCompletionScope runtime(28,std::nullopt);
        for(bool candidate:{false,true}) {
            const long long chunk=selectOrderedThreeRowStaticChunk(candidate,jobs,28);
            std::vector<int> owners(jobs,-1),visits(jobs,0);
            std::array<size_t,28> row_work{};
#pragma omp parallel num_threads(28)
            {
#pragma omp for schedule(static, chunk)
                for(long long job=0;job<static_cast<long long>(jobs);++job) {
                    const auto index=static_cast<size_t>(job);
                    const int worker=omp_get_thread_num();
                    owners[index]=worker;
                    ++visits[index];
                    // Homogeneous M38/P70 and P168 have four row-runs:
                    // [12,12,12,2]. Count rows, never infer kernel duration.
                    row_work[size_t(worker)]+=index%4u==3u?2u:12u;
                }
            }
            for(size_t job=0;job<jobs;++job) {
                EXPECT_EQ(visits[job],1);
                EXPECT_EQ(owners[job],static_cast<int>((job/size_t(chunk))%28u));
            }
            if(jobs==1400u || jobs==3360u) {
                const auto [minimum,maximum]=std::minmax_element(row_work.begin(),row_work.end());
                if(candidate) {
                    EXPECT_EQ(*maximum-*minimum,jobs==1400u?10u:0u);
                } else {
                    EXPECT_EQ(*maximum,*minimum*6u);
                }
            }
        }
    }
}

/** @test Authored deep tiles, 2/1 tails and bank reuse retain serial bytes and guards. */
TEST(CPUNativeQuantisedFusedCompletion, GPUAlignedPrefillOrderedThreeRowTileByteLifecycle)
{
    RecordProperty("input_scope","authored_deep_tile_contract_not_recorded_route_replay");
    constexpr int n=65,k=768,max_rows=73,ldc=72;
    constexpr float guard=-12345.0f;
    constexpr std::array<int8_t,8> values{-128,127,0,-1,31,-17,73,-113};
    for(const auto& format:quantizedVerifierFormats()) {
        SCOPED_TRACE(format.label);
        requireQuantizedVerifierSourceShape(format,2*n,k);
        auto source=format.create({size_t(2*n),size_t(k)},9411);
        ASSERT_NE(source,nullptr);
        CPUNativeQuantisedGemmKernel kernel(source.get(),n,2*n,
            CPUProjectionNumericalPolicy::GPUAlignedExpert);
        ASSERT_TRUE(kernel.isValid());
        const auto& packed=kernel.packedWeights();
        const auto workspace=packed.native_interleaved;
        std::vector<Q8_1Block> input(size_t(max_rows)*packed.blocks_per_row);
        std::array<std::vector<float>,2> output{
            std::vector<float>(size_t(max_rows)*ldc+17,guard),
            std::vector<float>(size_t(max_rows)*ldc+17,guard)};
        for(int workers:{1,4,7}) {
            FusedCompletionScope runtime(workers,std::nullopt);
            NativeQuantisedTestPartialStorage serial_partials(packed,1);
            for(int bank:{0,1}) {
                OrderedThreeRowBankScope binding(packed,bank?workspace.data():nullptr);
                int turn=0;
                // All formats get an explicit eligibility/no-write check. Only
                // registry-admitted nibble preparations can enter the new body.
                std::array<FusedVerifierRowsDesc,2> descriptors;
                const auto bind=[&](int rows,VerifierRowsPolicy requested) {
                    for(int member=0;member<2;++member) {
                        const int live_rows=member?std::max(1,rows-1):rows;
                        descriptors[member]={.packed=&packed,.output=output[member].data(),
                            .N=n,.ldc=ldc,.rows=live_rows,
                            .verifier_schedule=live_rows<3?VerifierRowsPolicy::Auto:requested,
                            .purpose=ProjectionRowsPurpose::Prefill,.policy_n=130};
                    }
                };
                bind(max_rows,VerifierRowsPolicy::OrderedThreeRows);
                const auto unchanged=output;
                const bool supported=packed.usesNibbleLUT() && activeISALevel()>=ISALevel::AVX512;
                if(!supported) {
                    EXPECT_THROW((void)planNativeQuantisedFusedRows(
                        descriptors,max_rows,workers,ISAPath::AUTO),std::invalid_argument);
                    EXPECT_EQ(output,unchanged);
                    // Existing general Auto remains a supported control on both
                    // runtime ISAs; no replacement execution certifies candidate.
                    bind(3,VerifierRowsPolicy::Auto);
                    const auto control=planNativeQuantisedFusedRows(descriptors,max_rows,workers,ISAPath::AUTO);
                    EXPECT_GT(control.projections[0].k_tiles,1);
                    continue;
                }
                for(int rows:{73,38,39,36,32,7,5,3,2,1,73}) {
                    SCOPED_TRACE("rows="+std::to_string(rows)+" bank="+std::to_string(bank)+
                        " workers="+std::to_string(workers));
                    for(size_t block=0;block<input.size();++block) {
                        input[block].d=fp32_to_fp16(turn==3?0.0f:0.0078125f*float(1+(block+turn)%7));
                        input[block].sum_qs=0;
                        for(int lane=0;lane<32;++lane) {
                            input[block].qs[lane]=turn==3?0:values[(block+lane+turn)%values.size()];
                            input[block].sum_qs+=input[block].qs[lane];
                        }
                    }
                    bind(rows,VerifierRowsPolicy::OrderedThreeRows);
                    auto expected=output;
                    for(int member=0;member<2;++member)
                        for(int row=0;row<descriptors[member].rows;++row) {
                            gemv_native_quantised_preq(packed,input.data()+size_t(row)*packed.blocks_per_row,
                                expected[member].data()+size_t(row)*ldc,serial_partials.span());
                            std::fill_n(output[member].data()+size_t(row)*ldc,n,guard);
                        }
                    const auto plan=planNativeQuantisedFusedRows(descriptors,max_rows,workers,ISAPath::AUTO);
                    if(rows==3) {
                        // Two N64 panels times rows[3,2] produce only four
                        // local tree owners. Seven workers must exercise the
                        // original shared K-part producer/reducer path.
                        EXPECT_EQ(plan.partial_route,workers==7
                            ?NativeQuantisedFusedPartialRoute::SharedBank
                            :NativeQuantisedFusedPartialRoute::OutputTileLocal);
                        if(workers==7)EXPECT_GT(plan.sharedPartialFloats(),0u);
                    }
                    auto metadata=descriptors;
                    for(auto& descriptor:metadata)descriptor.output=nullptr;
                    const auto admission=planNativeQuantisedFusedRows(metadata,max_rows,workers,ISAPath::AUTO);
                    EXPECT_EQ(admission.sharedPartialFloats(),plan.sharedPartialFloats());
                    EXPECT_EQ(admission.partial_route,plan.partial_route);
                    for(int member=0;member<2;++member) {
                        const auto& projection=plan.projections[member];
                        EXPECT_EQ(projection.k_tiles,MoEProjectionNumericalContract::orderedKPartitionsForWidth(k));
                        EXPECT_EQ(projection.policy_n,130);
                        if(descriptors[member].rows>=3) {
                            EXPECT_EQ(projection.effective_verifier_schedule,VerifierRowsPolicy::OrderedThreeRows);
                            EXPECT_EQ(projection.verifier_resolution.physical_row_tile,3);
                        }
                    }
                    NativeQuantisedTestPartialStorage partials(plan.sharedPartialFloats());
                    const auto invoke=[&](){return gemm_native_quantised_fused_verifier_rows_preq(
                        input.data(),descriptors.data(),partials.span(),2,max_rows,packed.blocks_per_row,ISAPath::AUTO);};
                    ASSERT_TRUE(invoke());
                    EXPECT_EQ(std::memcmp(output[0].data(),expected[0].data(),output[0].size()*sizeof(float)),0);
                    EXPECT_EQ(std::memcmp(output[1].data(),expected[1].data(),output[1].size()*sizeof(float)),0);
                    // A real existing team must publish completion before return.
                    for(int member=0;member<2;++member)
                        for(int row=0;row<descriptors[member].rows;++row)
                            std::fill_n(output[member].data()+size_t(row)*ldc,n,guard);
                    std::vector<int> accepted(size_t(workers),0);
#pragma omp parallel num_threads(workers)
                    {accepted[size_t(omp_get_thread_num())]=invoke();}
                    for(int ok:accepted)ASSERT_EQ(ok,1);
                    EXPECT_EQ(std::memcmp(output[0].data(),expected[0].data(),output[0].size()*sizeof(float)),0);
                    EXPECT_EQ(std::memcmp(output[1].data(),expected[1].data(),output[1].size()*sizeof(float)),0);
                    ++turn;
                }
                // Immutable original source geometry is checked before output;
                // purpose and asymmetry mutations cannot enter this new domain.
                bind(7,VerifierRowsPolicy::OrderedThreeRows);
                auto malformed=packed;malformed.clearWorkspace();
                malformed.is_asymmetric=!packed.is_asymmetric;
                descriptors[0].packed=&malformed;
                auto before=output;
                EXPECT_THROW((void)planNativeQuantisedFusedRows(descriptors,max_rows,workers,ISAPath::AUTO),std::invalid_argument);
                EXPECT_EQ(output,before);
                descriptors[0].packed=&packed;
                for(auto purpose:{ProjectionRowsPurpose::Decode,ProjectionRowsPurpose::MTPVerifier}) {
                    descriptors[0].purpose=purpose;
                    EXPECT_THROW((void)planNativeQuantisedFusedRows(descriptors,max_rows,workers,ISAPath::AUTO),std::invalid_argument);
                }
                descriptors[0].purpose=ProjectionRowsPurpose::Prefill;
                auto native=packed;native.clearWorkspace();
                native.numerical_policy=CPUProjectionNumericalPolicy::BackendNative;
                descriptors[0].packed=&native;
                EXPECT_THROW((void)planNativeQuantisedFusedRows(descriptors,max_rows,workers,ISAPath::AUTO),std::invalid_argument);
                EXPECT_EQ(output,before);
            }
        }
    }
}
} // namespace
} // namespace llaminar2::test
