/**
 * @file CPUNativeQuantisedOrderedThreeRowNibbleLifecycle.inc
 * @brief Authored three-row ordered-candidate byte and bank lifecycle checks.
 *
 * The canonical source registry, real preparation and public independent serial
 * entry point own the oracle. Explicit WideRows selects the new AVX-512 body;
 * Auto, Pairwise, non-nibble and AVX2 cases remain unchanged controls. These
 * authored inputs are not a recorded model-route replay or economy evidence.
 */
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
} // namespace
} // namespace llaminar2::test
