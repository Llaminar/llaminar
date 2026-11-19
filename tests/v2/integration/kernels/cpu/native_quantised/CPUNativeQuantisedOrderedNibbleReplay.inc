/**
 * @file CPUNativeQuantisedOrderedNibbleReplay.inc
 * @brief Replay actual GPU-aligned fused projections through every prepared source format.
 *
 * The canonical format registry owns source/codebook/superblock identity. Real
 * prepared kernels, the original fused scheduler and independent serial rows
 * own the numerical proof. Eligible nibble matrices additionally use the QFN
 * gate/up shard geometry and a tail/partial-partition geometry. Large-to-small
 * request counts, dirty inactive rows, partial descriptor bundles and repeated
 * reset reuse must preserve every published byte. No copied dot-product or
 * reduction implementation, host result substitution or timing claim exists.
 */
namespace llaminar2::test {
namespace {
/** @test Original fixed-partition arithmetic survives bounded nibble dispatch and unchanged other encodings. */
TEST(CPUNativeQuantisedFusedCompletion, GPUAlignedOrderedNibbleAllFormatReplay)
{
    constexpr int maximum_rows = 64;
    constexpr float guard = -12345.0f;
    struct Geometry { int n; int k; };
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        // Every format owns the small original producer. The actual prepared
        // encoding, rather than a second format-name table, admits larger probes.
        bool eligible_nibble = false;
        for (const auto geometry : {Geometry{65, 768}, Geometry{320, 2560}, Geometry{321, 768}})
        {
            if (geometry.n != 65 && !eligible_nibble)
                continue;
            SCOPED_TRACE(geometry.n);
            SCOPED_TRACE(geometry.k);
            auto source = format.create({size_t(2 * geometry.n), size_t(geometry.k)}, 1717);
            ASSERT_NE(source, nullptr);
            CPUNativeQuantisedGemmKernel kernel(source.get(), 0, geometry.n,
                CPUProjectionNumericalPolicy::GPUAlignedExpert);
            ASSERT_TRUE(kernel.isValid());
            const auto &packed = kernel.packedWeights();
            ASSERT_EQ(packed.N, geometry.n);
            ASSERT_EQ(packed.K, geometry.k);
            ASSERT_EQ(packed.codebook_id, format.source_codebook_id);
            ASSERT_EQ(packed.is_superblock, format.source_is_superblock);
            ASSERT_EQ(packed.numerical_policy, CPUProjectionNumericalPolicy::GPUAlignedExpert);
            if (geometry.n == 65)
                eligible_nibble = packed.usesNibbleLUT();
            else
                ASSERT_TRUE(packed.usesNibbleLUT());

            std::vector<float> activations(size_t(maximum_rows) * geometry.k);
            for (size_t element = 0; element < activations.size(); ++element)
                activations[element] = float(int((element * 37 + 13) % 257) - 128) * 0.003125f;
            std::vector<Q8_1Block> input(size_t(maximum_rows) * packed.blocks_per_row);
            quantize_activations_to_q8_1(activations.data(), input.data(), maximum_rows,
                geometry.k, packed.blocks_per_row, CPUProjectionNumericalPolicy::GPUAlignedExpert);
            const size_t stride = size_t(geometry.n + 7);
            const size_t extent = size_t(maximum_rows) * stride + 64;
            for (int workers : {1, 4})
            {
                SCOPED_TRACE(workers);
                FusedCompletionScope runtime(workers, std::nullopt);
                NativeQuantisedTestPartialStorage serial(packed, 1);
                std::vector<float> reference(extent, guard);
                for (int row = 0; row < maximum_rows; ++row)
                    gemv_native_quantised_preq(packed,
                        input.data() + size_t(row) * packed.blocks_per_row,
                        reference.data() + size_t(row) * stride, serial.span());
                std::array<std::vector<float>, 2> outputs{
                    std::vector<float>(extent, guard), std::vector<float>(extent, guard)};
                std::array<std::vector<float>, 2> expected = outputs;
                const auto replayRows = [&](int rows, int live, VerifierRowsPolicy requested,
                                            ProjectionRowsPurpose purpose)
                {
                    SCOPED_TRACE(projectionRowsPurposeName(purpose));
                    SCOPED_TRACE(rows);
                    std::array<FusedVerifierRowsDesc, 2> descriptors;
                    for (int member = 0; member < 2; ++member)
                    {
                        std::fill(outputs[member].begin(), outputs[member].end(), guard);
                        std::fill(expected[member].begin(), expected[member].end(), guard);
                        const int member_rows = member == 0 ? rows : std::max(1, rows - 1);
                        descriptors[member] = {.packed = &packed, .output = outputs[member].data(),
                            .N = geometry.n, .ldc = int(stride), .rows = member_rows,
                            .verifier_schedule = member_rows == 1 ? VerifierRowsPolicy::Auto : requested,
                            .purpose = purpose, .policy_n = geometry.n};
                        if (member < live)
                            for (int row = 0; row < member_rows; ++row)
                                std::memcpy(expected[member].data() + size_t(row) * stride,
                                    reference.data() + size_t(row) * stride,
                                    size_t(geometry.n) * sizeof(float));
                    }
                    const auto plan = planNativeQuantisedFusedRows(
                        {descriptors.data(), size_t(live)}, maximum_rows, workers, ISAPath::AUTO);
                    // These authored pairwise grids have enough real
                    // output owners even for four workers. This proves
                    // the changed local ordered branch is exercised,
                    // rather than certifying only the shared-bank path.
#if !defined(__aarch64__)
                    // x86 shared-K plan identity. The ARM register-K plan has
                    // whole-K tiles; its ordered tree is proven by the bytes below.
                    if (rows >= 8 && requested == VerifierRowsPolicy::Pairwise)
                        ASSERT_EQ(plan.partial_route, NativeQuantisedFusedPartialRoute::OutputTileLocal);
                    for (int member = 0; member < live; ++member)
                    {
                        EXPECT_EQ(plan.projections[member].k_tiles,
                            MoEProjectionNumericalContract::orderedKPartitionsForWidth(geometry.k));
                        if (descriptors[member].rows > 1 && requested == VerifierRowsPolicy::Pairwise)
                            EXPECT_EQ(plan.projections[member].effective_verifier_schedule,
                                VerifierRowsPolicy::Pairwise);
                    }
#endif
                    NativeQuantisedTestPartialStorage partials(plan.sharedPartialFloats());
                    std::fill(partials.span().begin(), partials.span().end(), guard);
                    ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(input.data(),
                        descriptors.data(), partials.span(), live, maximum_rows,
                        packed.blocks_per_row, ISAPath::AUTO));
                    for (int member = 0; member < 2; ++member)
                        EXPECT_EQ(std::memcmp(outputs[member].data(), expected[member].data(),
                            extent * sizeof(float)), 0) << "member=" << member;
                };
                for (const auto requested : {VerifierRowsPolicy::Pairwise, VerifierRowsPolicy::Auto})
                {
                    SCOPED_TRACE(verifierRowsPolicyName(requested));
                    for (int replay = 0; replay < 3; ++replay)
                    {
                        SCOPED_TRACE(replay);
                        // Full -> partial -> full descriptor publication and
                        // each next request's 64 rows follow a one-row frontier.
                        for (int rows : {64, 8, 3, 2, 1})
                            replayRows(rows, replay == 1 ? 1 : 2, requested,
                                ProjectionRowsPurpose::Prefill);
                    }
                }
                if (eligible_nibble && geometry.n == 65 && workers == 4)
                {
                    // Purpose participates in scheduling. Add its actual decode
                    // selector without multiplying the large prefill matrix.
                    for (const auto requested : {VerifierRowsPolicy::Pairwise, VerifierRowsPolicy::Auto})
                        for (int rows : {1, 3, 8})
                            for (const auto purpose : {ProjectionRowsPurpose::Decode, ProjectionRowsPurpose::MTPVerifier})
                                replayRows(rows, 2, rows == 1 ? VerifierRowsPolicy::Auto : requested, purpose);
                    for (bool mixed_signed_zero : {false, true})
                    {
                        SCOPED_TRACE(mixed_signed_zero ? "alternating signed zero" : "positive zero");
                        for (size_t element = 0; element < activations.size(); ++element)
                            activations[element] = mixed_signed_zero && element % 2 ? -0.0f : 0.0f;
                        quantize_activations_to_q8_1(activations.data(), input.data(), maximum_rows,
                            geometry.k, packed.blocks_per_row, CPUProjectionNumericalPolicy::GPUAlignedExpert);
                        std::fill(reference.begin(), reference.end(), guard);
                        for (int row = 0; row < maximum_rows; ++row)
                            gemv_native_quantised_preq(packed,
                                input.data() + size_t(row) * packed.blocks_per_row,
                                reference.data() + size_t(row) * stride, serial.span());
                        for (int replay = 0; replay < 2; ++replay)
                            for (int rows : {8, 3})
                                for (const auto purpose : {ProjectionRowsPurpose::Prefill, ProjectionRowsPurpose::Decode, ProjectionRowsPurpose::MTPVerifier})
                                    replayRows(rows, replay == 0 ? 2 : 1,
                                        VerifierRowsPolicy::Pairwise, purpose);
                    }
                }
            }
        }
    }
}
} // namespace
} // namespace llaminar2::test
