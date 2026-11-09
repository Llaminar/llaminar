/**
 * @file CPUNativeQuantisedUnsignedWeightDotLifecycle.inc
 * @brief Authored signed-activation boundaries preserve prepared-format row bytes.
 *
 * Canonical source factories and the public serial kernel own grouped-row
 * oracles. Actual identity-nibble M1 additionally uses an independent scalar
 * original biased-integer witness, since its public serial entry also changes. Identity-nibble inputs also authenticate unsigned decoded ranges and
 * the original stored compensation on AVX2 and AVX512. Native grouped execution
 * borrows owned and workspace banks across
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
    // Identity nibbles are unsigned coefficients, not signed final FP weights:
    // minima remain an independent FP term. Exhaust the exact scalar lane
    // identity on both compiled ISAs, including the Q8 -128 boundary. This
    // witness complements actual serial/grouped bytes; it executes no oracle
    // replacement in production and does not change the existing Q5 domain.
    for (int activation = -128; activation <= 127; ++activation)
        for (int coefficient = 0; coefficient <= 15; ++coefficient)
            ASSERT_EQ((activation + 128) * coefficient - 128 * coefficient,
                      activation * coefficient);
    static_assert(32 * -128 * 15 == -61440 && 32 * 127 * 15 == 60960);
    static_assert(32 * 255 * 15 == 122400);
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    // The original preparation/decoder pair, not asymmetry or a model name,
    // proves when the one-row scale load may be a plain half widening.
    for(const auto& source:native_quantised_formats::kAllSourceFormats)
    {
        const auto encoding=preparedEncodingForSource(*source.metadata);
        const auto decoder=nibbleDecodeKind(source.metadata->codebook_id);
        const bool linear=decoder==NibbleDecodeKind::Q4_0_LINEAR ||
                          decoder==NibbleDecodeKind::Q4_1_IDENTITY;
        if(linear)EXPECT_EQ(encoding,CPUNativeQuantisedEncoding::NibbleLUT);
        if(encoding==CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
            EXPECT_EQ(decoder,NibbleDecodeKind::LUT);
        CPUNativeQuantisedPackedWeights metadata;
        metadata.encoding=encoding;
        if(encoding==CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale)
        {
            EXPECT_FALSE(metadata.usesNibbleLUT());
            EXPECT_FALSE(linear);
        }
        // Rotated operands keep their original expanded producer; they cannot
        // enter a nibble-body specialization merely because source codebook0/5.
        metadata.encoding=preparedEncodingForSource(*source.metadata,true);
        EXPECT_FALSE(metadata.usesNibbleLUT());
    }
#endif
#if defined(__AVX2__)
    // addOrderedAVX2/AVX512 are x86 instruction-order helpers.
    // The original serial witness retains the earlier K payload. Exercise
    // the actual ordinary add on each selected runtime ISA, not a NaN-aware
    // oracle path. Integer bit construction never quiets a signaling input.
    struct AddWitness { uint32_t earlier; uint32_t later; uint32_t expected; };
    for (const auto lane : std::array<AddWitness, 8>{{
             {0x7fc02000u, 0x7fc56000u, 0x7fc02000u},
             {0xffcaa000u, 0xffcfe000u, 0xffcaa000u},
             {0x7fa02000u, 0x7fa56000u, 0x7fe02000u},
             {0xffaaa000u, 0xffafe000u, 0xffeaa000u},
             {0x3f800000u, 0x7fc56000u, 0x7fc56000u},
             {0x7fc02000u, 0x3f800000u, 0x7fc02000u},
             {0x3f800000u, 0x40000000u, 0x40400000u},
             {0x80000000u, 0x80000000u, 0x80000000u}}})
    {
        const __m256 earlier = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int32_t>(lane.earlier)));
        const __m256 later = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int32_t>(lane.later)));
        std::array<uint32_t, 8> actual;
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(actual.data()),
            _mm256_castps_si256(addOrderedAVX2(earlier, later)));
        for (uint32_t word : actual) EXPECT_EQ(word, lane.expected);
#if defined(__AVX512F__)
        if (activeISALevel() >= ISALevel::AVX512)
        {
            const __m512 wide_earlier = _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int32_t>(lane.earlier)));
            const __m512 wide_later = _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int32_t>(lane.later)));
            std::array<uint32_t, 16> wide_actual;
            _mm512_storeu_si512(wide_actual.data(),
                _mm512_castps_si512(addOrderedAVX512(wide_earlier, wide_later)));
            for (uint32_t word : wide_actual) EXPECT_EQ(word, lane.expected);
        }
#endif
    }
#endif
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

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
/**
 * @brief Independent original biased-integer witness for one actual identity row.
 * @param packed Original Q4_1/Q4_K prepared bank, without modifying its layout.
 * @param activation One original signed-Q8 row, including raw scale words.
 * @param output Destination for exactly packed.N columns.
 * @param workers Actual original invocation worker count owning serial K cuts.
 *
 * Scalar byte indexing reproduces the OLD unsigned(A+128) dot and stored
 * compensation, never the new swapped-operand VNNI body. Only the unchanged
 * FP contribution helpers and fixed partial reducer are shared with production;
 * this makes a changed public M1 kernel unable to certify its own integer dots.
 */
void originalBiasedIdentityM1(
    const cpu::native_quantised::CPUNativeQuantisedPackedWeights& packed,
    const Q8_1Block* activation,float* output,int workers)
{
    using namespace cpu::native_quantised;
    const auto config=serialTileConfigForPackedMatrix(packed,
        cpuNativeQuantisedSerialEquivalentPolicyN(packed.N),packed.K,workers);
    // Both legacy0 and1 mean the original full-K entry; only the shared
    // production predicate may interpret this arithmetic-schedule field.
    const int tiles=nativeQuantisedUsesKPartitions(config.k_tiles)?config.k_tiles:1;
    const int blocks_per_tile=(packed.blocks_per_row+tiles-1)/tiles;
    std::vector<float> partials(size_t(tiles)*64);
    for(int chunk=0;chunk<(packed.N+63)/64;++chunk) {
        for(int tile=0;tile<tiles;++tile) {
            __m512 accum[4]={_mm512_setzero_ps(),_mm512_setzero_ps(),
                            _mm512_setzero_ps(),_mm512_setzero_ps()};
            for(int kb=tile*blocks_per_tile;
                kb<std::min((tile+1)*blocks_per_tile,packed.blocks_per_row);++kb) {
                const auto& input=activation[kb];
                alignas(64) int32_t dots[64];
                for(int column=0;column<64;++column) {
                    int dot=0;
                    for(int group=0;group<4;++group) {
                        const auto* bytes=reinterpret_cast<const uint8_t*>(
                            packed.interleavedB(chunk,kb,group,column/16));
                        for(int lane=0;lane<4;++lane) {
                            const uint8_t raw=bytes[(column%16)*4+lane];
                            dot+=(int(input.qs[group*4+lane])+128)*(raw&15);
                            dot+=(int(input.qs[group*4+16+lane])+128)*(raw>>4);
                        }
                    }
                    dots[column]=dot-128*int(packed.chunkComp(chunk,kb)[column]);
                }
                for(int segment=0;segment<4;++segment)
                    accum[segment]=accumulateCorrectedSingleScaleBlockAVX512(
                        accum[segment],_mm512_load_si512(dots+segment*16),
                        packed.loadScales512(packed.chunkScales(chunk,kb)+segment*16),
                        input.sum_qs,_mm512_cvtph_ps(_mm256_load_si256(
                            reinterpret_cast<const __m256i*>(packed.chunkMins(chunk,kb)+segment*16))),
                        nativeQuantisedFP16ScaleToFP32(input.d),packed.numerical_policy);
            }
            for(int segment=0;segment<4;++segment)
                _mm512_storeu_ps(partials.data()+size_t(tile)*64+segment*16,accum[segment]);
        }
        const int columns=std::min(64,packed.N-chunk*64);
        if(tiles==1)std::copy_n(partials.data(),columns,output+chunk*64);
        else reduceNativeQuantisedKTilePartialsExact(partials.data(),output+chunk*64,
            columns,tiles,true,packed.numerical_policy);
    }
}
#endif

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
        const bool source_identity_nibble =
            format.source_codebook_id == native_quantised_formats::Q4_1.codebook_id;
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
        if (source_identity_nibble)
        {
            // Both actual identity sources share this prepared execution range.
            // K2560 exercises five blocks per fixed partition; N321/K768 adds
            // a real physical tail without padding a source K superblock.
            geometries.push_back({320, 2560});
            geometries.push_back({321, 768});
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
                if (source_identity_nibble)
                {
                    ASSERT_TRUE(packed.usesNibbleLUT());
                    ASSERT_TRUE(packed.is_asymmetric);
                    ASSERT_EQ(packed.encoding, CPUNativeQuantisedEncoding::NibbleLUT);
                    // Validate actual prepared low/high coefficient placement
                    // and the old compensation on both runtime ISAs. The dot
                    // replacement may not infer unsignedness from minima.
                    for (int chunk = 0; chunk < (geometry.n + 63) / 64; ++chunk)
                        for (int kb = 0; kb < packed.blocks_per_row; ++kb)
                            for (int column = 0; column < std::min(64, geometry.n - chunk * 64); ++column)
                            {
                                const int z = column / 16, lane = column % 16;
                                int sum = 0;
                                for (int group = 0; group < 4; ++group)
                                {
                                    const auto *raw = reinterpret_cast<const uint8_t *>(
                                        packed.interleavedB(chunk, kb, group, z));
                                    for (int value = 0; value < 4; ++value)
                                    {
                                        const auto bits = raw[lane * 4 + value];
                                        const int low = bits & 15, high = bits >> 4;
                                        ASSERT_GE(low, 0); ASSERT_LE(low, 15);
                                        ASSERT_GE(high, 0); ASSERT_LE(high, 15);
                                        sum += low + high;
                                    }
                                }
                                EXPECT_EQ(packed.chunkComp(chunk, kb)[column], sum);
                            }
                }
                const auto workspace = packed.native_interleaved;
                const int stride = geometry.n + 7;
                const size_t extent = size_t(max_rows) * stride + 17;
                std::vector<Q8_1Block> input(size_t(max_rows) * packed.blocks_per_row);
                std::array<std::vector<float>, 2> output, expected;
#if defined(__aarch64__)
                // ARM kernels borrow only the resident publication; a deferred
                // workspace pointer carries no bounded lease and is rejected.
                {
                    UnsignedDotBankScope binding(packed, workspace.data());
                    EXPECT_THROW((void)packed.readView(), std::logic_error);
                }
                constexpr int bank_count = 1;
#else
                constexpr int bank_count = 2;
#endif
                for (int bank = 0; bank < bank_count; ++bank)
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
#if !defined(__aarch64__)
                            // x86 planner K-partition/row-tile identity; ARM tiles are full-K.
                            if (source_identity_nibble &&
                                policy == CPUProjectionNumericalPolicy::GPUAlignedExpert && rows > 1)
                            {
                                ASSERT_TRUE(nativeQuantisedUsesKPartitions(plan.projections[0].k_tiles));
                                EXPECT_EQ(plan.projections[0].effective_verifier_schedule, VerifierRowsPolicy::Pairwise);
                                EXPECT_EQ(plan.projections[0].verifier_resolution.physical_row_tile, 2);
                            }
#endif
                            NativeQuantisedTestPartialStorage partials(plan.sharedPartialFloats());
                            ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(
                                input.data(), descriptors.data(), partials.span(), 2,
                                max_rows, packed.blocks_per_row, ISAPath::AUTO));
                            for (int member = 0; member < 2; ++member)
                                EXPECT_EQ(std::memcmp(output[member].data(), expected[member].data(),
                                                      extent * sizeof(float)), 0) << "member=" << member;
                            if (source_identity_nibble)
                            {
                                // Actual Auto is a separate original entry, not
                                // a claim that every shape must resolve Pairwise.
                                // Preserve its installed exact/default selection
                                // and dirty all live/inactive bytes before reuse.
                                for (int member = 0; member < 2; ++member)
                                {
                                    std::fill(output[member].begin(), output[member].end(), guard);
                                    descriptors[member].verifier_schedule = VerifierRowsPolicy::Auto;
                                }
                                const auto automatic = planNativeQuantisedFusedRows(
                                    descriptors, max_rows, workers, ISAPath::AUTO);
                                NativeQuantisedTestPartialStorage auto_partials(automatic.sharedPartialFloats());
                                ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(
                                    input.data(), descriptors.data(), auto_partials.span(), 2,
                                    max_rows, packed.blocks_per_row, ISAPath::AUTO));
                                for (int member = 0; member < 2; ++member)
                                    EXPECT_EQ(std::memcmp(output[member].data(), expected[member].data(),
                                                          extent * sizeof(float)), 0) << "Auto member=" << member;
                            }
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                            if(source_identity_nibble && rows==1 && activeISALevel()>=ISALevel::AVX512)
                            {
                                // Actual public decode and fused one-row members
                                // must match the independent original integer
                                // witness; serial-vs-grouped alone would compare
                                // two entries changed by this candidate.
                                const auto finite_input=input;
                                for(uint16_t scale_bits:{uint16_t(0x2000),uint16_t(0),uint16_t(0x8000),
                                    uint16_t(1),uint16_t(0x8001),uint16_t(0x7c00),uint16_t(0xfc00),
                                    uint16_t(0x7e01),uint16_t(0xfe55),uint16_t(0x7d01)})
                                {
                                    SCOPED_TRACE("M1_original_scale="+std::to_string(scale_bits));
                                    input=finite_input;
                                    input[0].d=scale_bits;
                                    input[packed.blocks_per_row-1].d=
                                        (scale_bits&0x7c00u)==0x7c00u && (scale_bits&0x03ffu)
                                            ? uint16_t(scale_bits^0x002au):uint16_t(scale_bits^0x8000u);
                                    std::vector<float> original(extent,guard),actual(extent,guard);
                                    originalBiasedIdentityM1(packed,input.data(),original.data(),workers);
                                    gemv_native_quantised_preq(packed,input.data(),actual.data(),serial_partials.span());
                                    EXPECT_EQ(std::memcmp(actual.data(),original.data(),extent*sizeof(float)),0)
                                        << "public M1 versus original biased integer witness";
                                    for(auto purpose:{ProjectionRowsPurpose::Decode,ProjectionRowsPurpose::MTPVerifier})
                                    {
                                        for(int member=0;member<2;++member) {
                                            std::fill(output[member].begin(),output[member].end(),guard);
                                            descriptors[member].rows=1;
                                            descriptors[member].verifier_schedule=VerifierRowsPolicy::Auto;
                                            descriptors[member].purpose=purpose;
                                        }
                                        const auto one=planNativeQuantisedFusedRows(descriptors,max_rows,workers,ISAPath::AUTO);
                                        if(policy==CPUProjectionNumericalPolicy::GPUAlignedExpert) {
                                            ASSERT_EQ(one.partial_route,NativeQuantisedFusedPartialRoute::OutputTileLocal);
                                            ASSERT_TRUE(nativeQuantisedUsesKPartitions(one.projections[0].k_tiles));
                                        }
                                        NativeQuantisedTestPartialStorage one_partials(one.sharedPartialFloats());
                                        ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(input.data(),descriptors.data(),
                                            one_partials.span(),2,max_rows,packed.blocks_per_row,ISAPath::AUTO));
                                        for(int member=0;member<2;++member)
                                            EXPECT_EQ(std::memcmp(output[member].data(),original.data(),extent*sizeof(float)),0)
                                                << "fused M1 member="<<member<<" purpose="<<int(purpose);
                                    }
                                }
                                input=finite_input;
                            }
#endif
                            if (source_identity_nibble && replay == 0 && workers == 4 &&
                                policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                            {
                                // The decoder preserves every FP16 input word.
                                // Adversarial raw prequantized scales additionally
                                // require grouped arithmetic to retain the serial
                                // NaN operand precedence. These are not canonical
                                // finite activation publications or malformed GGUF
                                // weights; only raw activation-scale bits change.
                                const auto finite_input = input;
                                for (uint16_t scale_bits : {uint16_t(0), uint16_t(0x8000),
                                         uint16_t(0x7c00), uint16_t(0xfc00),
                                         uint16_t(0x7e01), uint16_t(0xfe55), uint16_t(0x7d01)})
                                {
                                    SCOPED_TRACE("raw_scale=" + std::to_string(scale_bits));
                                    input = finite_input;
                                    for (int row = 0; row < max_rows; ++row)
                                    {
                                        auto *blocks = input.data() + size_t(row) * packed.blocks_per_row;
                                        blocks[0].d = scale_bits;
                                        // Distinct retained NaN payloads or
                                        // opposite zero/Inf signs also cross
                                        // the original increasing K fold.
                                        blocks[packed.blocks_per_row - 1].d =
                                            (scale_bits & 0x7c00u) == 0x7c00u && (scale_bits & 0x03ffu)
                                                ? uint16_t(scale_bits ^ 0x002au)
                                                : uint16_t(scale_bits ^ 0x8000u);
                                    }
                                    std::fill(serial.begin(), serial.end(), guard);
                                    for (int row = 0; row < max_rows; ++row)
                                        gemv_native_quantised_preq(packed,
                                            input.data() + size_t(row) * packed.blocks_per_row,
                                            serial.data() + size_t(row) * stride, serial_partials.span());
                                    for (auto purpose : {ProjectionRowsPurpose::Prefill,
                                                         ProjectionRowsPurpose::MTPVerifier})
                                    {
                                        for (int member = 0; member < 2; ++member)
                                        {
                                            std::fill(output[member].begin(), output[member].end(), guard);
                                            std::fill(expected[member].begin(), expected[member].end(), guard);
                                            const int live_rows = descriptors[member].rows;
                                            for (int row = 0; row < live_rows; ++row)
                                                std::memcpy(expected[member].data() + size_t(row) * stride,
                                                    serial.data() + size_t(row) * stride,
                                                    size_t(geometry.n) * sizeof(float));
                                            descriptors[member].verifier_schedule = VerifierRowsPolicy::Pairwise;
                                            descriptors[member].purpose = purpose;
                                        }
                                        const auto special = planNativeQuantisedFusedRows(
                                            descriptors, max_rows, workers, ISAPath::AUTO);
                                        NativeQuantisedTestPartialStorage special_partials(special.sharedPartialFloats());
                                        ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(
                                            input.data(), descriptors.data(), special_partials.span(), 2,
                                            max_rows, packed.blocks_per_row, ISAPath::AUTO));
                                        for (int member = 0; member < 2; ++member)
                                        {
                                            const int compared = std::memcmp(output[member].data(),
                                                expected[member].data(), extent * sizeof(float));
                                            if (compared != 0)
                                            {
                                                size_t first = 0;
                                                while (first < extent && std::memcmp(output[member].data() + first,
                                                    expected[member].data() + first, sizeof(float)) == 0) ++first;
                                                uint32_t actual_bits = 0, expected_bits = 0;
                                                if (first < extent)
                                                {
                                                    std::memcpy(&actual_bits, output[member].data() + first, sizeof(actual_bits));
                                                    std::memcpy(&expected_bits, expected[member].data() + first, sizeof(expected_bits));
                                                }
                                                ADD_FAILURE() << "raw-scale member=" << member << " purpose=" << int(purpose)
                                                    << " first_element=" << first << " row=" << first / size_t(stride)
                                                    << " column=" << first % size_t(stride)
                                                    << " actual_bits=" << actual_bits << " expected_bits=" << expected_bits
                                                    << " memcmp=" << compared;
                                            }
                                        }
                                    }
                                }
                                input = finite_input;
                            }
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
