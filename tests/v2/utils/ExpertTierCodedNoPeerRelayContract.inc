/* Included in the native test namespace after its buffer and stream fixtures. */
        /** @brief Drive the same live no-peer relay on CUDA and ROCm without requiring two GPUs. */
        template <template <class> class Buffer, class Stream, auto Success, class CopyToHost, class PeerCopy>
        void proveCodedNoPeerHostRelay(DeviceId device, CopyToHost copy_to_host, PeerCopy launch_peer)
        {
            namespace sb = native_quantised_superblock;
            constexpr int n = 13, k = 512;
            constexpr std::uint32_t capacity = 128;
            constexpr unsigned prefix = 1, suffix = 4;
            using P = MoEExpertProjectionOwnership::Projection;
            using R = MoEProjectionResidence;
            const P original{R::ExpertOwner, n, k, 0, n, 0, k};
            const P source_root{R::ParticipantInputSlice, n, k, 0, 7, 32, 352};
            const P destination_root{R::ParticipantInputSlice, n, k, 4, 9, 64, 320};
            const auto intersection = ExpertTierWeightProjectionIntersection::between(source_root, destination_root);
            const auto epoch = [&](const char *label) {
                return MappedTransferProgressEpoch::create({.device = device, .slot_capacity = 2,
                    .execution_lane_capacity = 2,
                    .execution_streams = test::ComponentTransferExecutionPoolFixtureAccess::allocate(
                        TransferEngine::instance(), 2, device, label),
                    .maximum_bytes = capacity, .name = label});
            };
            const auto source_epoch = epoch("coded_no_peer_source"), destination_epoch = epoch("coded_no_peer_destination");
            const auto staging = TransferEngine::instance().allocateMappedHostTransferSlices(capacity, 4, device);
            const ExpertTierGpuBlobTransferLane::Config config{
                .source_device = device, .destination_device = device,
                .relay_kind = ExpertTierGpuBlobRelayKind::SameBackendWithoutPeerAccess,
                .staging_capacity_bytes = capacity, .source_mapped_staging = {staging[0], staging[1]},
                .destination_mapped_staging = {staging[2], staging[3]}, .source_progress_epoch = source_epoch,
                .destination_progress_epoch = destination_epoch, .lane_name = "coded_no_peer_relay"};
            EXPECT_THROW((void)ExpertTierGpuBlobTransferLane{config}, std::invalid_argument);
            auto lane = test::ExpertTierGpuBlobRelayFixtureAccess::make(config);
            std::string error;
            ASSERT_TRUE(lane->materialize(&error)) << error;
            Stream stream;
            unsigned formats = 0;
            for (const auto &entry : native_quantised_formats::kAllSourceFormats)
            {
                const auto &metadata = *entry.metadata;
                if (!cpu::native_quantised::migrationUsesSuperblockCodes(metadata))
                    continue;
                ++formats;
                SCOPED_TRACE(entry.quant_type);
                const auto physical = canonicalDeviceVnniFormat(metadata);
                const auto book = physical.codebook_id;
                const auto bits = sb::superblockCodePlaneBits(book), header_bytes = sb::superblockFieldBytes(book);
                const auto fixture = test::quantizedVerifierFormat(std::string(entry.quant_type).c_str());
                const auto a = fixture.create({n, k}, 31), b = fixture.create({n, k}, 83);
                const auto source_native = test::makeGpuPreparedGemm(a.get(), device);
                const auto destination_native = test::makeGpuPreparedGemm(b.get(), device);
                DeviceNativeQuantisedMatrixDesc src_desc, dst_desc;
                ASSERT_TRUE(source_native.kernel->exportNativeQuantisedMatrixDesc(src_desc));
                ASSERT_TRUE(destination_native.kernel->exportNativeQuantisedMatrixDesc(dst_desc));
                const auto loaded = [&](const auto &desc) {
                    const auto packed = makeGpuExpertPackedDescriptor(desc, physical.payload_bytes_per_block, physical.is_asymmetric, physical.has_emins);
                    return ExpertTierGpuConstProjectionView{.payload = packed.ptrs.d_vnni,
                        .scales = static_cast<const uint16_t *>(packed.ptrs.d_scales), .mins = static_cast<const uint16_t *>(packed.ptrs.d_mins),
                        .payload_bytes = packed.vnni_bytes, .scales_bytes = packed.scales_bytes, .mins_bytes = packed.mins_bytes,
                        .metadata_layout = packed.metadata_layout};
                };
                const auto native_source = loaded(src_desc), native_destination = loaded(dst_desc);
                const auto sizes = [&](const P &root, unsigned shift) {
                    const auto blocks = root.columns / 32;
                    const std::size_t slots = root.rows * blocks;
                    return std::array<std::size_t, 3>{slots * physical.payload_bytes_per_block,
                        bits ? (slots * bits + shift + 7) / 8 : 0,
                        root.rows * sb::superblocksPerRow(blocks, root.first_column / 32) * header_bytes};
                };
                const unsigned src_shift = bits ? 3 : 0, dst_shift = bits ? 5 : 0;
                const auto src_size = sizes(source_root, src_shift), dst_size = sizes(destination_root, dst_shift);
                Buffer<uint8_t> src_payload(src_size[0]), src_codes(src_size[1] + prefix + suffix), src_headers(src_size[2]);
                Buffer<uint8_t> dst_payload(dst_size[0]), dst_codes(dst_size[1] + prefix + suffix), dst_headers(dst_size[2]);
                Buffer<uint8_t> peer_payload(dst_size[0]), peer_codes(dst_size[1] + prefix + suffix), peer_headers(dst_size[2]);
                const auto view = [&](const P &root, const auto &size, unsigned shift, auto &payload, auto &codes, auto &headers) {
                    return ExpertTierGpuMutableProjectionView{.payload = payload.data(),
                        .scales = reinterpret_cast<uint16_t *>(bits ? codes.data() + prefix : payload.data()),
                        .mins = reinterpret_cast<uint16_t *>(headers.data()), .payload_bytes = size[0],
                        .scales_bytes = size[1], .mins_bytes = size[2], .metadata_layout = nativeQuantisedMetadataLayout(
                            NativeQuantisedMetadataRepresentation::SuperblockCoded, root.first_column / 32, shift)};
                };
                const auto src = view(source_root, src_size, src_shift, src_payload, src_codes, src_headers);
                const auto dst = view(destination_root, dst_size, dst_shift, dst_payload, dst_codes, dst_headers);
                const auto p2p = view(destination_root, dst_size, dst_shift, peer_payload, peer_codes, peer_headers);
                ASSERT_EQ(src_codes.upload(std::vector<uint8_t>(src_codes.bytes(), 0xa5)), Success);
                ASSERT_EQ(dst_codes.upload(std::vector<uint8_t>(dst_codes.bytes(), 0x5a)), Success);
                ASSERT_EQ(peer_codes.upload(std::vector<uint8_t>(peer_codes.bytes(), 0x5a)), Success);
                const auto peer_copy = [&](const auto &input, const auto &output, const P &from, const P &to) {
                    const auto live = ExpertTierWeightProjectionIntersection::between(from, to);
                    return launch_peer({.rows = static_cast<unsigned>(live.rows()),
                        .blocks = static_cast<unsigned>(live.columns() / 32), .payload_bytes = physical.payload_bytes_per_block,
                        .codebook = book, .src_n = static_cast<unsigned>(from.rows), .dst_n = static_cast<unsigned>(to.rows),
                        .src_row = static_cast<unsigned>(live.sourceRowOrigin()), .dst_row = static_cast<unsigned>(live.destinationRowOrigin()),
                        .src_block = static_cast<unsigned>(live.sourceColumnOrigin() / 32), .dst_block = static_cast<unsigned>(live.destinationColumnOrigin() / 32),
                        .source = input, .destination = output}, stream.opaque());
                };
                ASSERT_TRUE(peer_copy(native_source, src, original, source_root));
                ASSERT_TRUE(peer_copy(native_destination, dst, original, destination_root));
                ASSERT_TRUE(peer_copy(native_destination, p2p, original, destination_root));
                ASSERT_TRUE(peer_copy(src.asConst(), p2p, source_root, destination_root));
                ASSERT_EQ(stream.synchronize(), Success);
                const auto packed = [&](const auto &projection, const P &root) {
                    return GpuExpertPackedDescriptor{.ptrs = {.d_vnni = projection.payload, .d_scales = projection.scales, .d_mins = projection.mins},
                        .n = root.rows, .k = root.columns, .blocks_per_row = static_cast<unsigned>(root.columns / 32),
                        .codebook_id = book, .payload_bytes_per_block = physical.payload_bytes_per_block,
                        .is_asymmetric = physical.is_asymmetric, .has_emins = physical.has_emins,
                        .metadata_layout = projection.metadata_layout, .vnni_bytes = projection.payload_bytes,
                        .scales_bytes = projection.scales_bytes, .mins_bytes = projection.mins_bytes};
                };
                const auto source = packed(src, source_root), destination = packed(dst, destination_root);
                auto wrong_phase = destination;
                wrong_phase.metadata_layout = nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded, 0, dst_shift);
                EXPECT_FALSE(lane->accepts(source, wrong_phase, intersection, &error));
                ASSERT_TRUE(lane->accepts(source, destination, intersection, &error)) << error;
                const auto before = lane->stats();
                ASSERT_TRUE(lane->start(source, destination, intersection,
                    ExpertTierSourceReadiness::publishedResidencyBank(1), &error)) << error;
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                auto progress = ExpertTierGpuBlobTransferProgress::Pending;
                while (progress == ExpertTierGpuBlobTransferProgress::Pending && std::chrono::steady_clock::now() < deadline)
                {
                    ASSERT_TRUE(source_epoch->submitOutstandingProgress());
                    ASSERT_TRUE(destination_epoch->submitOutstandingProgress());
                    progress = lane->poll(&error);
                    std::this_thread::yield();
                }
                ASSERT_EQ(progress, ExpertTierGpuBlobTransferProgress::Ready) << error;
                const auto after = lane->stats();
                constexpr unsigned live_rows = 3, live_blocks = 10, live_headers = 2;
                // CPU-coded wire stores 12-bit codes in independently rounded byte/nibble planes.
                const auto code_bytes = bits == 12 ? live_rows + (live_rows + 1) / 2 :
                    bits == 6 ? (live_rows + 1) / 2 + (live_rows + 3) / 4 : (live_rows * bits + 7) / 8;
                const auto live_bytes = live_blocks * (live_rows * physical.payload_bytes_per_block + code_bytes) +
                    live_headers * live_rows * header_bytes;
                EXPECT_EQ(lane->quotedPayloadBytes(), live_bytes);
                EXPECT_EQ(after.host_relay_bytes - before.host_relay_bytes, live_bytes);
                EXPECT_GT(after.chunks_submitted - before.chunks_submitted, 1u);
                const auto compare = [](const auto &actual, const auto &expected) {
                    std::vector<uint8_t> a, b;
                    ASSERT_EQ(actual.download(a), Success);
                    ASSERT_EQ(expected.download(b), Success);
                    EXPECT_EQ(a, b);
                };
                compare(dst_payload, peer_payload);
                compare(dst_codes, peer_codes);
                compare(dst_headers, peer_headers);
                // Independently check the live result against the full SourceNative load;
                // agreement with a peer operation alone would not catch a shared offset bug.
                std::vector<uint8_t> payload, codes, headers;
                ASSERT_EQ(dst_payload.download(payload), Success);
                ASSERT_EQ(dst_codes.download(codes), Success);
                ASSERT_EQ(dst_headers.download(headers), Success);
                const auto read = [&](const void *device_bytes, std::size_t count) {
                    std::vector<uint8_t> bytes(count);
                    if (count)
                    {
                        EXPECT_EQ(copy_to_host(bytes.data(), device_bytes, count, stream.native()), Success);
                        EXPECT_EQ(stream.synchronize(), Success);
                    }
                    return bytes;
                };
                const auto original_payload = read(native_source.payload, native_source.payload_bytes);
                const auto original_codes = read(native_source.scales, native_source.scales_bytes);
                const auto original_headers = read(native_source.mins, native_source.mins_bytes);
                for (unsigned block = 0; block < live_blocks; ++block)
                    for (unsigned row = 0; row < live_rows; ++row)
                    {
                        const auto expected = sb::blockMajorAddress(block + 2, row + 4, n, 0);
                        const auto actual = sb::blockMajorAddress(block, row, destination_root.rows, 2);
                        EXPECT_EQ(std::memcmp(payload.data() + actual.linear * physical.payload_bytes_per_block,
                            original_payload.data() + expected.linear * physical.payload_bytes_per_block, physical.payload_bytes_per_block), 0);
                        EXPECT_EQ(std::memcmp(headers.data() + actual.superblock * header_bytes,
                            original_headers.data() + expected.superblock * header_bytes, header_bytes), 0);
                        if (bits)
                            EXPECT_EQ(sb::loadPackedBits(codes.data() + prefix, actual.linear * bits + dst_shift, bits),
                                sb::loadPackedBits(original_codes.data(), expected.linear * bits, bits));
                    }
            }
            EXPECT_EQ(formats, 13u);
        }
