/**
 * @file ExpertTierDirectCPUProjectionContract.inc
 * @brief Direct original-root encoded movement, without payload scratch or devices.
 *
 * Opaque native bytes and the existing independent encoded-column oracle prove
 * exact physical fields and dirty neighbors. Parallel Q6 pieces share the
 * destination word writer; no arithmetic or source readiness is manufactured.
 */
namespace llaminar2 {
namespace {
/** @return Physical traits from the canonical existing native format inventory. */
ExpertTierWeightCPUStorageLayout directCPUFormat(const NativeQuantisedFormatInfo &format, unsigned quantum)
{
    const auto encoding = cpu::native_quantised::preparedEncodingForSource(format);
    return {.cpu_codebook_id = format.codebook_id,
        .cpu_is_asymmetric = uint8_t(format.is_asymmetric),
        .cpu_is_superblock = uint8_t(format.is_superblock), .cpu_encoding = encoding,
        .cpu_data_stride = uint32_t(cpu::native_quantised::preparedDataStride(encoding)),
        .cpu_block_stride = uint32_t(cpu::native_quantised::preparedInterleavedBlockStride(encoding, format.is_asymmetric)),
        .wire_columns_per_unit = quantum};
}

TEST(ExpertTierWeightStream, DirectCPUOriginalRootsAllFormatsPreserveIndependentStridesAndDirtyNeighbours)
{
    using P = MoEExpertProjectionOwnership::Projection;
    using R = MoEProjectionResidence;
    const P from{R::ParticipantOutputSlice, 256, 256, 13, 128, 32, 96};
    const P to{R::ParticipantInputSlice, 256, 256, 72, 160, 64, 128};
    const auto overlap = ExpertTierWeightProjectionIntersection::between(from, to);
    for (const auto &format : native_quantised_formats::kAllSourceFormats)
        for (const unsigned quantum : {1u, 3u, 17u, 64u})
        {
            SCOPED_TRACE(format.quant_type);
            SCOPED_TRACE(quantum);
            const auto traits = directCPUFormat(*format.metadata, quantum);
            const auto source = cpuExpertTierStorageLayoutForIntersection(traits, from, overlap, 65536);
            const auto destination = cpuExpertTierStorageLayoutForIntersection(traits, to, overlap, 65536);
            std::vector<uint8_t> bytes(source.cpuStorageBytes());
            for (size_t index = 0; index < bytes.size(); ++index) bytes[index] = uint8_t(index * 37 + 11);
            std::vector<uint8_t> result(destination.cpuStorageBytes(), 0xa5);
            const auto untouched = result;
            size_t live_bytes = 0;
            for (uint32_t first = 0; first < source.unit_count; )
            {
                const auto count = std::min({3u, source.unit_count - first, source.maximum_units_per_chunk,
                    destination.maximum_units_per_chunk});
                ASSERT_TRUE(copyExpertTierCPUWeightChunk(source, bytes, destination, result, first, count));
                live_bytes += source.chunkBytes(count, first);
                first += count;
            }
            EXPECT_EQ(live_bytes, size_t(69 * 2) * (traits.cpu_block_stride / 64));
            for (size_t row = 0; row < 192; ++row)
                for (size_t block = 0; block < 4; ++block)
                    if (row < 69 && block < 2)
                        EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                            encodedTierColumn(bytes, source, 59 + row, 1 + block));
                    else EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                            encodedTierColumn(untouched, destination, row, block));
            const auto before = result;
            EXPECT_FALSE(copyExpertTierCPUWeightChunk(source, std::span(bytes).first(bytes.size() - 1),
                destination, result, 0, 1));
            EXPECT_EQ(result, before);
            EXPECT_FALSE(copyExpertTierCPUWeightChunk(source, bytes, source, bytes, 0, 1));
            auto bad = destination; ++bad.cpu_first_column;
            EXPECT_FALSE(copyExpertTierCPUWeightChunk(source, bytes, bad, result, 0, 1));
            EXPECT_EQ(result, before);
        }
}

TEST(ExpertTierWeightStream, DirectCPUParallelPiecesPreserveSharedQ6WordsAcrossReset)
{
    using P = MoEExpertProjectionOwnership::Projection;
    using R = MoEProjectionResidence;
    const P a{R::ParticipantOutputSlice, 128, 256, 0, 32, 0, 256};
    const P b{R::ParticipantOutputSlice, 128, 256, 32, 96, 0, 256};
    const P target{R::ParticipantOutputSlice, 128, 256, 15, 79, 0, 256};
    const auto first = ExpertTierWeightProjectionIntersection::between(a, target);
    const auto second = ExpertTierWeightProjectionIntersection::between(b, target);
    for (const auto &format : native_quantised_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(format.quant_type);
        const auto traits = directCPUFormat(*format.metadata, 1);
        const auto source_a = cpuExpertTierStorageLayoutForIntersection(traits, a, first, 65536);
        const auto source_b = cpuExpertTierStorageLayoutForIntersection(traits, b, second, 65536);
        const auto target_a = cpuExpertTierStorageLayoutForIntersection(traits, target, first, 65536);
        const auto target_b = cpuExpertTierStorageLayoutForIntersection(traits, target, second, 65536);
        std::vector<uint8_t> bytes_a(source_a.cpuStorageBytes()), bytes_b(source_b.cpuStorageBytes());
        for (size_t i = 0; i < bytes_a.size(); ++i) bytes_a[i] = uint8_t(i * 53 + 7);
        for (size_t i = 0; i < bytes_b.size(); ++i) bytes_b[i] = uint8_t(i * 17 + 29);
        std::vector<uint8_t> result(target_a.cpuStorageBytes());
        std::mutex write_authority;
        for (int replay = 0; replay < 20; ++replay)
        {
            std::fill(result.begin(), result.end(), uint8_t(0xa5 + replay));
            const auto untouched = result;
            bool success_a = true, success_b = true;
            auto copy = [&](const auto &source, const auto &bytes, const auto &destination, bool &success) {
                for (uint32_t unit = 0; unit < source.unit_count; ++unit)
                    if (!copyExpertTierCPUWeightChunk(source, bytes, destination, result, unit, 1, &write_authority))
                    { success = false; break; }
            };
            std::thread left([&] { copy(source_a, bytes_a, target_a, success_a); });
            std::thread right([&] { copy(source_b, bytes_b, target_b, success_b); });
            left.join(); right.join();
            ASSERT_TRUE(success_a); ASSERT_TRUE(success_b);
            for (size_t row = 0; row < 128; ++row)
                for (size_t block = 0; block < 8; ++block)
                {
                    const auto expected = row < 17 ? encodedTierColumn(bytes_a, source_a, 15 + row, block)
                        : row < 79 ? encodedTierColumn(bytes_b, source_b, row - 17, block)
                        : encodedTierColumn(untouched, target_a, row, block);
                    ASSERT_EQ(encodedTierColumn(result, target_a, row, block), expected)
                        << "replay=" << replay << " row=" << row;
                }
        }
    }
}

/**
 * @test Mixed multi-scale strides side by side move exactly the section set.
 *
 * IQ2_S (live payload width 9) and Q2_K (width 8) share the CompactMultiScale
 * payload region with different live widths; Q8_0 contrasts with inline
 * compensation and no effective minima. Section presence — derived from the
 * layout tables, never the encoding family name — must decide which bytes
 * move for each, and an unimplemented prepared layout must fail closed at
 * the storage layout before any encoded-column arithmetic runs.
 */
TEST(ExpertTierWeightStream, TierCopySectionPresenceDrivesMixedMultiScaleStrides)
{
    using P = MoEExpertProjectionOwnership::Projection;
    using R = MoEProjectionResidence;
    const P from{R::ParticipantOutputSlice, 256, 256, 13, 128, 32, 96};
    const P to{R::ParticipantInputSlice, 256, 256, 72, 160, 64, 128};
    const auto overlap = ExpertTierWeightProjectionIntersection::between(from, to);

    const NativeQuantisedFormatInfo *const mixed_stride_formats[] = {
        &native_quantised_formats::IQ2_S,
        &native_quantised_formats::Q2_K,
        &native_quantised_formats::Q8_0,
    };
    for (const NativeQuantisedFormatInfo *format : mixed_stride_formats)
    {
        SCOPED_TRACE(format->codebook_id);
        const auto traits = directCPUFormat(*format, 64);
        const auto source = cpuExpertTierStorageLayoutForIntersection(traits, from, overlap, 65536);
        const auto destination = cpuExpertTierStorageLayoutForIntersection(traits, to, overlap, 65536);

        // The storage layout's section authority must agree with the physical
        // family for every canonical stream: multi-scale units carry packed
        // effective minima, expanded bytes carry inline compensation.
        const auto sections = source.cpuSections();
        ASSERT_TRUE(sections.isValid());
        const bool multi_scale =
            traits.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::CompactMultiScale;
        EXPECT_EQ(sections.has_emins, multi_scale);
        EXPECT_EQ(sections.has_secondary, multi_scale ||
            traits.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
            traits.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
            traits.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale ||
            traits.cpu_is_asymmetric != 0);
        EXPECT_EQ(sections.has_compensation, !multi_scale &&
            traits.cpu_encoding != cpu::native_quantised::CPUNativeQuantisedEncoding::Q6KNativeDualScale);

        std::vector<uint8_t> bytes(source.cpuStorageBytes());
        for (size_t index = 0; index < bytes.size(); ++index)
            bytes[index] = uint8_t(index * 41 + 7);
        std::vector<uint8_t> result(destination.cpuStorageBytes(), 0x5c);
        const auto untouched = result;
        size_t live_bytes = 0;
        for (uint32_t first = 0; first < source.unit_count; )
        {
            const auto count = std::min({2u, source.unit_count - first,
                source.maximum_units_per_chunk, destination.maximum_units_per_chunk});
            ASSERT_TRUE(copyExpertTierCPUWeightChunk(source, bytes, destination, result, first, count));
            live_bytes += source.chunkBytes(count, first);
            first += count;
        }
        // Communication extent: exactly the live columns of each unit, one
        // full column slice (payload region plus present sections) each.
        EXPECT_EQ(live_bytes, size_t(69 * 2) * (traits.cpu_block_stride / 64));

        for (size_t row = 0; row < 192; ++row)
            for (size_t block = 0; block < 4; ++block)
                if (row < 69 && block < 2)
                    EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                        encodedTierColumn(bytes, source, 59 + row, 1 + block));
                else EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                        encodedTierColumn(untouched, destination, row, block));

        if (multi_scale)
        {
            // Effective minima are section data, not family trivia: both live
            // payload widths must carry their packed word byte-exactly. This
            // is the regression for the encoding-only emins access that read
            // outside a trimmed unit on partial/rectangular paths.
            for (size_t row : {size_t{0}, size_t{17}, size_t{68}})
                for (size_t block : {size_t{0}, size_t{1}})
                {
                    // Source-local K blocks are 1..2; destination-local are
                    // 0..1: the two roots carry independent window origins.
                    const size_t source_n = 59 + row, destination_n = row;
                    const size_t source_unit =
                        ((source_n / 64) * (source.cpu_K / 32) + 1 + block) * source.cpu_block_stride;
                    const size_t destination_unit =
                        ((destination_n / 64) * (destination.cpu_K / 32) + block) * destination.cpu_block_stride;
                    for (size_t byte = 0; byte < 4; ++byte)
                        EXPECT_EQ(result[destination_unit + sections.emins_offset +
                                (destination_n % 64) * 4 + byte],
                            bytes[source_unit + sections.emins_offset +
                                (source_n % 64) * 4 + byte]);
                }
        }
    }

    // An unimplemented prepared layout fails closed at the storage layout,
    // before any encoded-column arithmetic can consume its sentinel strides.
    auto unimplemented = directCPUFormat(native_quantised_formats::IQ2_S, 64);
    unimplemented.cpu_layout = cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCompactA64;
    EXPECT_FALSE(unimplemented.valid());
    EXPECT_THROW(static_cast<void>(cpuExpertTierStorageLayoutForIntersection(
                     unimplemented, from, overlap, 65536)), std::invalid_argument);

    // A destination that disagrees with the source's prepared layout is
    // rejected without mutation (the invalid layout fails valid() first;
    // the copy never reaches encoded-column arithmetic).
    const auto traits = directCPUFormat(native_quantised_formats::IQ2_S, 64);
    const auto source = cpuExpertTierStorageLayoutForIntersection(traits, from, overlap, 65536);
    auto foreign = cpuExpertTierStorageLayoutForIntersection(traits, to, overlap, 65536);
    foreign.cpu_layout = cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCompactA64;
    std::vector<uint8_t> bytes(source.cpuStorageBytes(), 0x33);
    std::vector<uint8_t> result(source.cpuStorageBytes(), 0x77);
    const auto before = result;
    EXPECT_FALSE(copyExpertTierCPUWeightChunk(source, bytes, foreign, result, 0, 1));
    EXPECT_EQ(result, before);
}
}
}
