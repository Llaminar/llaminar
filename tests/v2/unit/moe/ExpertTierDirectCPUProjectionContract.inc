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
 * payload region with different live widths; both carry effective minima in
 * the canonical layout, so this is canonical-preservation coverage for every
 * present section. Q8_0 contrasts with inline compensation and no effective
 * minima. Section presence — derived from the layout tables, never the
 * encoding family name — must decide which bytes move for each, and an
 * unimplemented prepared layout must fail closed at the storage layout
 * before any encoded-column arithmetic runs. An absent-emins trimmed-unit
 * control becomes representable when Stage A introduces the explicit compact
 * tables (test-only via the explicit layout argument; the production
 * selector stays canonical) and must land before Gate A.
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
        // Non-periodic per-position content: every stride and section
        // offset here is 256-byte aligned, so a linear byte pattern with a
        // 256-byte period could alias a wrong-origin read exactly. A byte's
        // value must depend on its absolute position.
        const auto positionByte = [](size_t index) {
            uint64_t mixed = index * 0x9E3779B97F4A7C15ull + 0x517cc1b727220a95ull;
            mixed ^= mixed >> 29;
            return uint8_t((mixed * 0xC2B2AE3D27D4EB4Full) >> 56);
        };
        for (size_t index = 0; index < bytes.size(); ++index)
            bytes[index] = positionByte(index);
        std::vector<uint8_t> result(destination.cpuStorageBytes(), 0x5c);
        const auto untouched = result;
        size_t quoted_chunk_bytes = 0;
        for (uint32_t first = 0; first < source.unit_count; )
        {
            const auto count = std::min({2u, source.unit_count - first,
                source.maximum_units_per_chunk, destination.maximum_units_per_chunk});
            ASSERT_TRUE(copyExpertTierCPUWeightChunk(source, bytes, destination, result, first, count));
            quoted_chunk_bytes += source.chunkBytes(count, first);
            first += count;
        }
        // Expected chunk-byte quotes: the layout's own quote function prices
        // exactly the live columns of each unit (payload region plus present
        // sections). These are expected extents, not an observed
        // communication-volume measurement of the copier.
        EXPECT_EQ(quoted_chunk_bytes, size_t(69 * 2) * (traits.cpu_block_stride / 64));

        for (size_t row = 0; row < 192; ++row)
            for (size_t block = 0; block < 4; ++block)
                if (row < 69 && block < 2)
                    EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                        encodedTierColumn(bytes, source, 59 + row, 1 + block));
                else EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                        encodedTierColumn(untouched, destination, row, block));

        // Wrong-K-origin control: the expected column must differ from the
        // same row in a non-copied source-local K block. If these could be
        // equal, the byte equality above would not detect an origin bug.
        for (size_t row : {size_t{0}, size_t{17}, size_t{68}})
        {
            const auto expected = encodedTierColumn(bytes, source, 59 + row, 1);
            const auto wrong_origin = encodedTierColumn(bytes, source, 59 + row, 0);
            ASSERT_EQ(expected.size(), wrong_origin.size());
            EXPECT_NE(expected, wrong_origin);
        }
        // Wrong-section control at the exact copied coordinate: source
        // n = 76 (local column 12), K block 1 — the same unit and column
        // the effective-minima check below copies. Slices use each
        // section's real element width. For multi-scale units the witness
        // is the copied 4-byte effective-minima word against the same
        // column's first 4-byte payload-group word (the historical
        // 1280/period-256 alias class); two-byte metadata units compare
        // their compensation and scale words at the same coordinate.
        {
            const size_t source_n = 59 + 17;
            const size_t local_column = source_n % 64;
            const size_t source_unit =
                ((source_n / 64) * (source.cpu_K / 32) + 1) *
                source.cpu_block_stride;
            const auto wordAt = [&](size_t offset, size_t width) {
                std::vector<uint8_t> word(width);
                std::memcpy(word.data(),
                    bytes.data() + source_unit + offset + local_column * width,
                    width);
                return word;
            };
            if (multi_scale)
            {
                ASSERT_TRUE(sections.has_emins);
                EXPECT_NE(wordAt(sections.emins_offset, 4), wordAt(0, 4));
            }
            else
            {
                ASSERT_TRUE(sections.has_compensation);
                EXPECT_NE(wordAt(sections.compensation_offset, 2),
                          wordAt(sections.scales_offset, 2));
            }
        }

        if (multi_scale)
        {
            // Effective minima are section data, not family trivia: on these
            // canonical units the emins section is present, and both live
            // payload widths must carry their packed word byte-exactly. The
            // check pins the section-driven read against the encoding-only
            // access class (a fixed tail offset regardless of the unit's
            // actual sections); the trimmed-unit case joins when Stage A
            // introduces explicit compact tables.
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
    // rejected without mutation. Isolate the layout as the only cause:
    // first prove the identical geometry copies successfully with
    // all-valid inputs (destination sized from its own extent), then flip
    // only the destination's prepared layout field and observe rejection
    // without mutation — the unimplemented layout fails its section-derived
    // validity, the layout-equality authority for layouts without
    // implemented tables, before any encoded-column arithmetic runs.
    const auto traits = directCPUFormat(native_quantised_formats::IQ2_S, 64);
    const auto source = cpuExpertTierStorageLayoutForIntersection(traits, from, overlap, 65536);
    const auto valid_destination = cpuExpertTierStorageLayoutForIntersection(traits, to, overlap, 65536);
    auto foreign = valid_destination;
    foreign.cpu_layout = cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCompactA64;
    std::vector<uint8_t> bytes(source.cpuStorageBytes(), 0x33);
    std::vector<uint8_t> result(valid_destination.cpuStorageBytes(), 0x77);
    const auto before = result;
    ASSERT_TRUE(copyExpertTierCPUWeightChunk(source, bytes, valid_destination, result, 0, 1));
    result = before;
    EXPECT_FALSE(copyExpertTierCPUWeightChunk(source, bytes, foreign, result, 0, 1));
    EXPECT_EQ(result, before);
}

/**
 * @test Partial-unit Q6 transfers carry the high-bit plane inside the
 *        payload region.
 *
 * Q6's payload format splits one 32-value block's 24 bytes per column into
 * the values' four low bits (16 bytes) and their two high bits (8 bytes).
 * The high-bit plane therefore starts at 16 bytes per column, strictly
 * INSIDE the payload region — the payload end (24 column-bytes, the scales
 * section) is a section boundary and must never address it. A partial-unit
 * regression addressed the high plane through the payload-end quotient,
 * landing on scales/mins and past the unit end; full 64-column units take
 * the packed bitplane path and mask the defect. Every unit below is
 * partial (12, 37, and a 64+5 crossing), the CPU column mapper is the
 * address reference, and the host reference round-trips partial-width Q6
 * matrices byte-exactly in both directions.
 */
TEST(ExpertTierWeightStream, PartialUnitQ6TransfersCarryTheHighPlaneInsideThePayload)
{
    using cpu::native_quantised::CPUNativeQuantisedEncoding;
    using cpu::native_quantised::CPUNativeQuantisedLayout;
    using cpu::native_quantised::CPUNativeQuantisedPackedWeights;
    using cpu::native_quantised::preparedEncodingForSource;
    using cpu::native_quantised::preparedUnitSections;
    const auto &format = native_quantised_formats::Q6_K;
    const auto sections = preparedUnitSections(
        CPUNativeQuantisedLayout::InterleavedCanonical,
        preparedEncodingForSource(format), format.is_asymmetric, format);
    ASSERT_TRUE(sections.isValid());
    // Address contract the conversion kernels and the column mapper share:
    // scales follow the payload, the secondary words follow scales, and the
    // 16-column-byte high plane lies strictly inside the payload.
    ASSERT_EQ(sections.payload_stride / 64u, 24u);
    ASSERT_EQ(sections.scales_offset / 64u, 24u);
    ASSERT_EQ(sections.mins_offset / 64u, 26u);
    ASSERT_EQ(16u + 8u, sections.payload_stride / 64u);
    ASSERT_LT(16u, sections.scales_offset / 64u);

    using P = MoEExpertProjectionOwnership::Projection;
    using R = MoEProjectionResidence;
    for (const int live_rows : {12, 37, 69})
    {
        SCOPED_TRACE(live_rows);
        // Source rows [13, 13+rows) intersect the destination window
        // [72, 232) in exactly the requested live width.
        const int from_rows = live_rows - 13 + 72;
        const P from{R::ParticipantOutputSlice, 256, 256, 13, from_rows, 32, 96};
        const P to{R::ParticipantInputSlice, 256, 256, 72, 160, 64, 128};
        const auto overlap = ExpertTierWeightProjectionIntersection::between(from, to);
        ASSERT_EQ(overlap.rows(), live_rows);
        ASSERT_EQ(overlap.columns(), 64);

        const auto traits = directCPUFormat(format, 64);
        const auto source = cpuExpertTierStorageLayoutForIntersection(traits, from, overlap, 65536);
        const auto destination = cpuExpertTierStorageLayoutForIntersection(traits, to, overlap, 65536);
        std::vector<uint8_t> bytes(source.cpuStorageBytes());
        const auto positionByte = [](size_t index) {
            uint64_t mixed = index * 0x9E3779B97F4A7C15ull + 0x517cc1b727220a95ull;
            mixed ^= mixed >> 29;
            return uint8_t((mixed * 0xC2B2AE3D27D4EB4Full) >> 56);
        };
        for (size_t index = 0; index < bytes.size(); ++index)
            bytes[index] = positionByte(index);
        std::vector<uint8_t> result(destination.cpuStorageBytes(), 0x5c);
        const auto untouched = result;
        size_t quoted_chunk_bytes = 0;
        for (uint32_t first = 0; first < source.unit_count; )
        {
            const auto count = std::min({2u, source.unit_count - first,
                source.maximum_units_per_chunk, destination.maximum_units_per_chunk});
            ASSERT_TRUE(copyExpertTierCPUWeightChunk(source, bytes, destination, result, first, count));
            quoted_chunk_bytes += source.chunkBytes(count, first);
            first += count;
        }
        // Expected chunk-byte quotes: live columns only, payload plus both
        // dual-scale sections per unit.
        const size_t units = ((size_t(live_rows) + 63) / 64) * 2;
        ASSERT_EQ(source.unit_count, units);
        EXPECT_EQ(quoted_chunk_bytes,
            size_t(live_rows) * 2u * (traits.cpu_block_stride / 64));

        // Every live column is oracle-equal through the section-driven
        // column mapper (the kernel address reference); neighbours stay
        // untouched. Source row origin is 59 = 72 - 13.
        const int source_rows = live_rows;
        for (size_t row = 0; row < 192; ++row)
            for (size_t block = 0; block < 4; ++block)
                if (row < size_t(source_rows) && block < 2)
                    EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                        encodedTierColumn(bytes, source, 59 + row, 1 + block));
                else EXPECT_EQ(encodedTierColumn(result, destination, row, block),
                        encodedTierColumn(untouched, destination, row, block));

        // Byte-exact dual-scale words at the section offsets, for both
        // partial widths and the crossing tail.
        for (const size_t row : {size_t{0}, size_t{5}, size_t{11},
                 size_t{36}, size_t{68}})
        {
            if (row >= size_t(source_rows))
                continue;
            for (const size_t block : {size_t{0}, size_t{1}})
            {
                const size_t source_n = 59 + row, destination_n = row;
                const size_t source_unit =
                    ((source_n / 64) * (source.cpu_K / 32) + 1 + block) * source.cpu_block_stride;
                const size_t destination_unit =
                    ((destination_n / 64) * (destination.cpu_K / 32) + block) * destination.cpu_block_stride;
                for (const size_t section : {sections.scales_offset, sections.mins_offset})
                {
                    const uint16_t source_word = *reinterpret_cast<const uint16_t *>(
                        bytes.data() + source_unit + section + (source_n % 64) * 2);
                    const uint16_t destination_word = *reinterpret_cast<const uint16_t *>(
                        result.data() + destination_unit + section + (destination_n % 64) * 2);
                    EXPECT_EQ(source_word, destination_word);
                }
            }
        }
    }

    // The host reference round-trips partial-width Q6 matrices byte-exactly
    // in both directions: a 12- and a 37-column GPU projection demote to
    // final CPU bytes and promote back without one payload, scale, or
    // coefficient bit changing.
    for (const int N : {12, 37})
    {
        SCOPED_TRACE(N);
        const auto physical = canonicalDeviceVnniFormat(format);
        HostGpuExpertPackedProjection gpu;
        gpu.N = N;
        gpu.K = 96;
        gpu.blocks_per_row = 3;
        gpu.source_codebook_id = format.codebook_id;
        gpu.codebook_id = physical.codebook_id;
        gpu.payload_bytes_per_block = physical.payload_bytes_per_block;
        gpu.is_asymmetric = physical.is_asymmetric;
        gpu.is_superblock = format.is_superblock;
        gpu.has_emins = physical.has_emins;
        const size_t blocks = size_t(N) * 3u;
        gpu.payload.resize(blocks * physical.payload_bytes_per_block);
        gpu.scales.resize(blocks);
        gpu.mins.resize(blocks);
        for (size_t block = 0; block < blocks; ++block)
        {
            for (size_t byte = 0; byte < physical.payload_bytes_per_block; ++byte)
                gpu.payload[block * physical.payload_bytes_per_block + byte] =
                    uint8_t((block * 37u + byte * 19u + 11u) & 0xffu);
            gpu.scales[block] = uint16_t(0x2400u + (block % 0x0800u));
            gpu.mins[block] = uint16_t(0xa000u + (block % 0x0800u));
        }
        std::string error;
        ASSERT_TRUE(gpu.valid(&error)) << error;
        CPUNativeQuantisedPackedWeights cpu;
        ASSERT_TRUE(gpuToCpuExpertPackedReference(gpu, cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical, cpu, &error)) << error;
        EXPECT_EQ(cpu.N_padded, 64);
        HostGpuExpertPackedProjection returned;
        ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, returned, &error)) << error;
        EXPECT_EQ(returned.codebook_id, gpu.codebook_id);
        EXPECT_EQ(returned.payload, gpu.payload);
        EXPECT_EQ(returned.scales, gpu.scales);
        EXPECT_EQ(returned.mins, gpu.mins);
    }
}

/**
 * @test The endpoint layout contract selects GPU conversion and rejects
 *        mismatched layout contracts.
 *
 * GPU-to-CPU demotion names the DESTINATION CPU endpoint layout; the
 * manifest carries it in reserved identity byte zero and derives its
 * strides and wire planes from that layout's section tables — never from
 * the compiler host and never from the encoding family alone. CPU-to-GPU
 * promotion names the SOURCE layout from the packed weights themselves.
 * The DeviceLayout wire planes are exactly the endpoint layout's section
 * quotients, so the conversion kernels and the host reference consume one
 * geometry (the reference round trip reproduces the separated data
 * byte-exactly). Mismatched or unknown carried layouts are typed
 * rejections: manifest validity and the device layout both fail closed,
 * and an unimplemented destination layout is rejected by the builder
 * before any stride work — never a silent canonical fallback.
 */
TEST(ExpertTierWeightStream, EndpointLayoutContractSelectsGpuConversionAndRejectsMismatch)
{
    using cpu::native_quantised::CPUNativeQuantisedEncoding;
    using cpu::native_quantised::CPUNativeQuantisedLayout;
    using cpu::native_quantised::CPUNativeQuantisedPackedWeights;
    using cpu::native_quantised::preparedEncodingForSource;
    using cpu::native_quantised::preparedUnitSections;
    const auto canonical = CPUNativeQuantisedLayout::InterleavedCanonical;
    const auto compact = CPUNativeQuantisedLayout::InterleavedCompactA64;

    // Adversarial separated GPU data mirroring one cataloged source's
    // canonical device representation (host reference input only).
    const auto hostProjection = [](const NativeQuantisedFormatInfo &source) {
        const auto physical = canonicalDeviceVnniFormat(source);
        HostGpuExpertPackedProjection projection;
        projection.N = 70;
        projection.K = 96;
        projection.blocks_per_row = 3;
        projection.source_codebook_id = source.codebook_id;
        projection.codebook_id = physical.codebook_id;
        projection.payload_bytes_per_block =
            physical.payload_bytes_per_block;
        projection.is_asymmetric = physical.is_asymmetric;
        projection.is_superblock = source.is_superblock;
        projection.has_emins = physical.has_emins;
        const size_t blocks = 70u * 3u;
        projection.payload.resize(blocks * physical.payload_bytes_per_block);
        projection.scales.resize(blocks);
        if (physical.is_asymmetric)
            projection.mins.resize(blocks);
        if (physical.has_emins)
            projection.emins.assign(blocks, 0xac002c00u);
        for (size_t block = 0; block < blocks; ++block)
        {
            for (size_t byte = 0;
                 byte < physical.payload_bytes_per_block; ++byte)
            {
                uint8_t value = static_cast<uint8_t>(
                    (block * 37u + byte * 19u + 11u) & 0xffu);
                if (projection.codebook_id == 19 ||
                    projection.codebook_id ==
                        kNativeQuantisedExpandedInt8MinCodebook)
                {
                    value = static_cast<uint8_t>(static_cast<int8_t>(
                        static_cast<int>(
                            (block * 13u + byte * 7u) % 127u) -
                        63));
                }
                projection.payload[
                    block * physical.payload_bytes_per_block + byte] = value;
            }
            projection.scales[block] =
                static_cast<uint16_t>(0x2400u + (block % 0x0800u));
            if (physical.is_asymmetric)
            {
                projection.mins[block] = static_cast<uint16_t>(
                    projection.codebook_id ==
                            kNativeQuantisedIQ4XSSourceScaleCodebook
                        ? block % 64u
                        : 0xa000u + (block % 0x0800u));
            }
        }
        // IQ3_S raw source words and IQ2_S grid bytes must stay inside
        // their decoder domains for the reference to accept them.
        if (physical.codebook_id == 11 || physical.codebook_id == 13)
            for (size_t block = 0; block < projection.mins.size(); ++block)
                projection.mins[block] = static_cast<uint16_t>(
                    block & (physical.codebook_id == 11 ? 15u : 255u));
        return projection;
    };

    // Representative sources covering every GPU conversion kernel family.
    const NativeQuantisedFormatInfo *const families[] = {
        &native_quantised_formats::Q4_0,   // NibbleLUT
        &native_quantised_formats::Q8_0,   // ExpandedInt8
        &native_quantised_formats::Q6_K,   // Q6KNativeDualScale
        &native_quantised_formats::Q2_K,   // CompactMultiScale with emins
        &native_quantised_formats::IQ4_XS, // NibbleLUTSourceScale
        &native_quantised_formats::IQ3_S,  // IQ3SExpandedSourceScale
    };
    for (const NativeQuantisedFormatInfo *format : families)
    {
        SCOPED_TRACE(format->codebook_id);
        const auto encoding = preparedEncodingForSource(*format);
        const auto sections = preparedUnitSections(
            canonical, encoding, format->is_asymmetric, *format);
        ASSERT_TRUE(sections.isValid());

        // GPU-to-CPU names the destination endpoint layout.
        const auto stream = makeGpuToCpuExpertTierWeightStreamManifest(
            *format, canonical, 70, 96, 5, 0, 1,
            ExpertTierWeightProjection::Down, 2);
        std::string error;
        ASSERT_TRUE(stream.valid(&error)) << error;
        EXPECT_EQ(stream.cpu_endpoint_layout,
            static_cast<uint8_t>(canonical));
        EXPECT_EQ(stream.cpu_data_stride, sections.payload_stride);
        EXPECT_EQ(stream.cpu_block_stride, sections.block_stride);
        const auto device = stream.deviceLayout();
        ASSERT_TRUE(device.valid());
        EXPECT_EQ(device.cpu_layout, canonical);
        EXPECT_EQ(device.cpu_wire_payload_per_column,
            sections.payload_stride / 64u);
        EXPECT_EQ(device.cpu_wire_compensation_plane,
            sections.has_compensation ? sections.compensation_offset / 64u
                                      : 0u);
        EXPECT_EQ(device.cpu_wire_scales_plane,
            sections.scales_offset / 64u);
        EXPECT_EQ(device.cpu_wire_mins_plane,
            sections.has_secondary ? sections.mins_offset / 64u : 0u);
        EXPECT_EQ(device.cpu_wire_emins_plane,
            sections.has_emins ? sections.emins_offset / 64u : 0u);

        // Field-tamper controls: every derived plane field is independently
        // load-bearing. Mutating any one of them on an otherwise-valid
        // device layout must fail validity — the plane re-derivation is a
        // comparison against the carried layout's section tables, never a
        // pass-through of the stored field.
        {
            auto tampered = device;
            tampered.cpu_wire_payload_per_column += 1u;
            EXPECT_FALSE(tampered.valid());
        }
        {
            auto tampered = device;
            tampered.cpu_wire_compensation_plane += 1u;
            EXPECT_FALSE(tampered.valid());
        }
        {
            auto tampered = device;
            tampered.cpu_wire_scales_plane += 1u;
            EXPECT_FALSE(tampered.valid());
        }
        {
            auto tampered = device;
            tampered.cpu_wire_mins_plane += 1u;
            EXPECT_FALSE(tampered.valid());
        }
        {
            auto tampered = device;
            tampered.cpu_wire_emins_plane += 1u;
            EXPECT_FALSE(tampered.valid());
        }

        // An unimplemented destination layout is rejected by the builder
        // before any stride work — never a silent canonical fallback.
        EXPECT_THROW(
            static_cast<void>(makeGpuToCpuExpertTierWeightStreamManifest(
                *format, compact, 70, 96, 5, 0, 1,
                ExpertTierWeightProjection::Down, 2)),
            std::invalid_argument);

        // A known-but-unimplemented carried layout is a typed rejection —
        // not a disagreement between two valid layouts. Compact tables are
        // not admitted yet, so validity fails on the section-table lookup
        // itself, for the manifest and the device layout alike.
        auto unimplementedCarried = stream;
        unimplementedCarried.cpu_endpoint_layout =
            static_cast<uint8_t>(compact);
        EXPECT_FALSE(unimplementedCarried.valid(&error));
        EXPECT_FALSE(unimplementedCarried.deviceLayout().valid());
        auto unknown = stream;
        unknown.cpu_endpoint_layout = 7;
        EXPECT_FALSE(unknown.valid(&error));
        EXPECT_FALSE(unknown.deviceLayout().valid());

        // Version-4 seal control: a compact byte carried with a correctly
        // recomputed hash still rejects — the layout is sealed identity,
        // and its unimplemented section tables fail closed. No hash can
        // admit a layout whose sections do not exist; equally, no equal-
        // stride layout substitution can replay a sealed stream.
        auto resealedCompact = stream;
        resealedCompact.cpu_endpoint_layout =
            static_cast<uint8_t>(compact);
        resealedCompact.manifest_hash = resealedCompact.computedHash();
        EXPECT_FALSE(resealedCompact.valid(&error));

        // Tier-level v4 hash sensitivity: a same-scalar fixture whose ONLY
        // change is the carried endpoint layout hashes differently, and a
        // layout byte modified WITHOUT recomputing the hash fails tier-level
        // validity — the :535 hash contribution is live. If it were deleted,
        // this tampered manifest would match the stale hash and pass.
        auto layoutOnly = stream;
        layoutOnly.cpu_endpoint_layout =
            stream.cpu_endpoint_layout == 0u ? 1u : 0u;
        EXPECT_NE(layoutOnly.computedHash(), stream.computedHash());
        auto staleHash = stream;
        staleHash.cpu_endpoint_layout = layoutOnly.cpu_endpoint_layout;
        EXPECT_FALSE(staleHash.valid(&error));

        // CPU-to-GPU names the source layout from the weights themselves,
        // and the reference round trip reproduces the same sections the
        // wire planes describe: one geometry for kernels and reference.
        auto gpu = hostProjection(*format);
        ASSERT_TRUE(gpu.valid(&error)) << error;
        CPUNativeQuantisedPackedWeights cpu;
        ASSERT_TRUE(gpuToCpuExpertPackedReference(gpu, cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical, cpu, &error))
            << error;
        const auto cpu_sections = cpu.preparedSections();
        ASSERT_TRUE(cpu_sections.isValid());
        const auto promote = makeCpuToGpuExpertTierWeightStreamManifest(
            cpu, 5, 0, 1, ExpertTierWeightProjection::Down, 2);
        ASSERT_TRUE(promote.valid(&error)) << error;
        EXPECT_EQ(promote.cpu_endpoint_layout,
            static_cast<uint8_t>(cpu.layout));
        const auto promotion = promote.deviceLayout();
        ASSERT_TRUE(promotion.valid());
        EXPECT_EQ(promotion.cpu_layout, cpu.layout);
        EXPECT_EQ(promotion.cpu_wire_payload_per_column,
            cpu_sections.payload_stride / 64u);
        EXPECT_EQ(promotion.cpu_wire_scales_plane,
            cpu_sections.scales_offset / 64u);
        EXPECT_EQ(promotion.cpu_wire_emins_plane,
            cpu_sections.has_emins ? cpu_sections.emins_offset / 64u : 0u);
        HostGpuExpertPackedProjection returned;
        ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, returned, &error))
            << error;
        EXPECT_EQ(returned.codebook_id, gpu.codebook_id);
        EXPECT_EQ(returned.payload_bytes_per_block,
            gpu.payload_bytes_per_block);
        EXPECT_EQ(returned.is_asymmetric, gpu.is_asymmetric);
        EXPECT_EQ(returned.is_superblock, gpu.is_superblock);
        EXPECT_EQ(returned.has_emins, gpu.has_emins);
        EXPECT_EQ(returned.payload, gpu.payload);
        EXPECT_EQ(returned.scales, gpu.scales);
        EXPECT_EQ(returned.mins, gpu.mins);
        EXPECT_EQ(returned.emins, gpu.emins);
    }
}
}
}
