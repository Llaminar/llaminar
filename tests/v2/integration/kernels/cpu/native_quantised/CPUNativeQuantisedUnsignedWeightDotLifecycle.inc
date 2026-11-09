/**
 * @file CPUNativeQuantisedUnsignedWeightDotLifecycle.inc
 * @brief Authored signed-activation boundaries preserve prepared-format row bytes.
 *
 * Canonical source factories and the unchanged public serial kernel own the
 * oracle. Native grouped execution borrows owned and workspace banks across
 * changing live rows, purposes and dirty-tail reuse. These are numerical and
 * lifecycle fixtures, not recorded model route replay or performance evidence.
 */
namespace llaminar2::test
{
namespace
{
/** @brief Revoke a borrowed bank before its invocation-owned backing expires. */
class UnsignedDotBankScope final
{
public:
    /** @brief Select the original bank or one exact invocation-owned copy. */
    UnsignedDotBankScope(const cpu::native_quantised::CPUNativeQuantisedPackedWeights &packed,
                         const uint8_t *workspace)
        : packed_(packed)
    {
        if (workspace) packed_.setWorkspace(workspace);
    }
    /** @brief Withdraw borrowed bytes before the local workspace is released. */
    ~UnsignedDotBankScope() { packed_.clearWorkspace(); }
    UnsignedDotBankScope(const UnsignedDotBankScope &) = delete;
    UnsignedDotBankScope &operator=(const UnsignedDotBankScope &) = delete;
private:
    const cpu::native_quantised::CPUNativeQuantisedPackedWeights &packed_; ///< Original bank authority.
};

/** @test Asymmetry alone never authorizes unsigned coefficients. */
TEST(CPUNativeQuantisedFusedCompletion, UnsignedCoefficientDomainContract)
{
    using namespace cpu::native_quantised;
    CPUNativeQuantisedPackedWeights packed;
    packed.encoding = CPUNativeQuantisedEncoding::ExpandedInt8;
    packed.is_asymmetric = true;
    packed.codebook_id = native_quantised_formats::Q5_1.codebook_id;
    for (bool superblock : {false, true})
    {
        packed.is_superblock = superblock;
        EXPECT_EQ(packed.coefficientDomain(),
                  CPUNativeQuantisedCoefficientDomain::UnsignedFiveBit);
    }
    packed.is_asymmetric = false;
    EXPECT_EQ(packed.coefficientDomain(), CPUNativeQuantisedCoefficientDomain::SignedInt8);
    packed.is_asymmetric = true;
    for (uint8_t codebook : {uint8_t(0), uint8_t(5), uint8_t(6), uint8_t(11),
                             uint8_t(17), uint8_t(19)})
    {
        packed.codebook_id = codebook;
        EXPECT_EQ(packed.coefficientDomain(), CPUNativeQuantisedCoefficientDomain::SignedInt8);
    }
    packed.codebook_id = native_quantised_formats::Q5_1.codebook_id;
    for (auto encoding : {CPUNativeQuantisedEncoding::NibbleLUT,
                           CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale,
                           CPUNativeQuantisedEncoding::CompactMultiScale,
                           CPUNativeQuantisedEncoding::Q6KNativeDualScale})
    {
        packed.encoding = encoding;
        EXPECT_EQ(packed.coefficientDomain(), CPUNativeQuantisedCoefficientDomain::SignedInt8);
    }
}

/** @test All source families retain serial bytes with native signed-Q8 extremes. */
TEST(CPUNativeQuantisedFusedCompletion, UnsignedWeightDotAllFormatByteLifecycle)
{
    using namespace cpu::native_quantised;
    constexpr int max_rows = 7;
    constexpr float guard = -12345.0f;
    constexpr std::array<int8_t, 8> values{-128, 127, -1, 0, 31, -17, 73, -113};
    struct Geometry { int n; int k; };
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        // Every format owns the512-wide original producer. Only the canonical
        // five-bit families add an actual short down shape: Q5_1 K320 or a
        // complete Q5_K source superblock. No source padding/requantization.
        const bool source_five_bit =
            format.source_codebook_id == native_quantised_formats::Q5_1.codebook_id;
        std::vector<Geometry> geometries{{65, 512}};
        if (source_five_bit)
        {
            geometries.push_back({65, 768});
            if (format.source_is_superblock)
                geometries.push_back({65, 256});
            else
            {
                geometries.push_back({2560, 320});
                geometries.push_back({65, 32});
            }
        }
        for (const auto geometry : geometries)
        {
            SCOPED_TRACE("N=" + std::to_string(geometry.n) + " K=" + std::to_string(geometry.k));
            for (auto policy : {CPUProjectionNumericalPolicy::BackendNative,
                                CPUProjectionNumericalPolicy::GPUAlignedExpert})
            {
                SCOPED_TRACE(static_cast<int>(policy));
                auto source = format.create({size_t(geometry.n * 2), size_t(geometry.k)}, 4117);
                // Exercise a nonzero source offset. The constructor accepts
                // a half-open interval, not an offset followed by a row count.
                CPUNativeQuantisedGemmKernel kernel(
                    source.get(), geometry.n, geometry.n * 2, policy);
                ASSERT_TRUE(kernel.isValid());
                const auto &packed = kernel.packedWeights();
                ASSERT_EQ(packed.N, geometry.n);
                ASSERT_EQ(packed.K, geometry.k);
                const auto expected_domain = source_five_bit
                    ? CPUNativeQuantisedCoefficientDomain::UnsignedFiveBit
                    : CPUNativeQuantisedCoefficientDomain::SignedInt8;
                ASSERT_EQ(packed.coefficientDomain(), expected_domain);
                if (source_five_bit)
                {
                    // Authenticate the canonical prepared range itself; the
                    // optimization may not guess from asymmetric minima.
                    for (int chunk = 0; chunk < (geometry.n + 63) / 64; ++chunk)
                        for (int kb = 0; kb < packed.blocks_per_row; ++kb)
                            for (int group = 0; group < 8; ++group)
                                for (int z = 0; z < 4; ++z)
                                {
                                    const auto *bytes = reinterpret_cast<const int8_t *>(
                                        packed.interleavedB(chunk, kb, group, z));
                                    ASSERT_TRUE(std::all_of(bytes, bytes + 64,
                                        [](int8_t value) { return value >= 0 && value <= 31; }));
                                }
                }
                const auto workspace = packed.native_interleaved;
                const int stride = geometry.n + 7;
                const size_t extent = size_t(max_rows) * stride + 17;
                std::vector<Q8_1Block> input(size_t(max_rows) * packed.blocks_per_row);
                std::array<std::vector<float>, 2> output, expected;
                for (int bank = 0; bank < 2; ++bank)
                {
                    SCOPED_TRACE(bank);
                    UnsignedDotBankScope binding(packed, bank == 0 ? nullptr : workspace.data());
                    for (int workers : {1, 4})
                    {
                        FusedCompletionScope runtime(workers, std::nullopt);
                        int replay = 0;
                        for (int rows : {max_rows, 2, 1, max_rows})
                        {
                            SCOPED_TRACE("workers=" + std::to_string(workers) +
                                         " rows=" + std::to_string(rows) +
                                         " replay=" + std::to_string(replay));
                            for (size_t block = 0; block < input.size(); ++block)
                            {
                                auto &activation = input[block];
                                activation.d = fp32_to_fp16(0.0078125f * (1 + block % max_rows));
                                activation.sum_qs = 0;
                                for (int lane = 0; lane < 32; ++lane)
                                {
                                    activation.qs[lane] = values[(size_t(lane) + block + replay) % values.size()];
                                    activation.sum_qs += activation.qs[lane];
                                }
                            }
                            std::vector<float> serial(extent, guard);
                            NativeQuantisedTestPartialStorage serial_partials(packed, 1);
                            for (int row = 0; row < max_rows; ++row)
                                gemv_native_quantised_preq(packed,
                                    input.data() + size_t(row) * packed.blocks_per_row,
                                    serial.data() + size_t(row) * stride, serial_partials.span());
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if (source_five_bit && replay == 0 &&
                                policy == CPUProjectionNumericalPolicy::GPUAlignedExpert &&
                                activeISALevel() >= ISALevel::AVX512 &&
                                MoEProjectionNumericalContract::
                                    oneNativeQuantisedBlockPerOrderedPartition(geometry.k))
                            {
                                // The optional paired publication uses the same
                                // unsigned dots, then the ORIGINAL final row add.
                                // These widths have one block per fixed partition.
                                for (int chunk : {0, (geometry.n - 1) / 64})
                                {
                                    alignas(64) float paired[64], inert[64];
                                    std::fill_n(paired, 64, guard);
                                    std::fill_n(inert, 64, guard);
                                    gemm_2row_non_nibble_chunk<true>(packed,
                                        input.data(), input.data() + packed.blocks_per_row,
                                        paired, inert, chunk, 0, packed.blocks_per_row, false);
                                    const int columns = std::min(64, geometry.n - chunk * 64);
                                    for (int column = 0; column < columns; ++column)
                                    {
                                        const int offset = chunk * 64 + column;
                                        const float sum = serial[offset] + serial[stride + offset];
                                        EXPECT_EQ(std::memcmp(paired + column, &sum, sizeof(sum)), 0);
                                    }
                                    EXPECT_TRUE(std::all_of(inert, inert + 64,
                                        [](float value) { return value == guard; }));
                                }
                            }
#endif
                            std::array<FusedVerifierRowsDesc, 2> descriptors;
                            for (int member = 0; member < 2; ++member)
                            {
                                const int member_rows = std::max(1, rows - member);
                                output[member].assign(extent, guard);
                                expected[member].assign(extent, guard);
                                for (int row = 0; row < member_rows; ++row)
                                    std::memcpy(expected[member].data() + size_t(row) * stride,
                                                serial.data() + size_t(row) * stride,
                                                size_t(geometry.n) * sizeof(float));
                                descriptors[member] = {.packed = &packed,
                                    .output = output[member].data(), .N = geometry.n,
                                    .ldc = stride, .rows = member_rows,
                                    .verifier_schedule = member_rows == 1
                                        ? VerifierRowsPolicy::Auto : VerifierRowsPolicy::Pairwise,
                                    .purpose = (replay & 1) ? ProjectionRowsPurpose::MTPVerifier
                                                           : ProjectionRowsPurpose::Prefill};
                            }
                            const auto plan = planNativeQuantisedFusedRows(
                                descriptors, max_rows, workers, ISAPath::AUTO);
                            NativeQuantisedTestPartialStorage partials(plan.sharedPartialFloats());
                            ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(
                                input.data(), descriptors.data(), partials.span(), 2,
                                max_rows, packed.blocks_per_row, ISAPath::AUTO));
                            for (int member = 0; member < 2; ++member)
                                EXPECT_EQ(std::memcmp(output[member].data(), expected[member].data(),
                                                      extent * sizeof(float)), 0) << "member=" << member;
                            ++replay;
                        }
                    }
                }
            }
        }
    }
}
} // namespace
} // namespace llaminar2::test
