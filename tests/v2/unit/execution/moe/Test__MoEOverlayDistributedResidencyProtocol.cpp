/**
 * @file Test__MoEOverlayDistributedResidencyProtocol.cpp
 * @brief Adversarial CPU-only tests for cross-rank residency publication.
 *
 * These tests model independent MPI ranks without creating devices or an MPI
 * communicator.  They prove complete transaction identity, unanimous staging,
 * unanimous inactive-bank commit, deterministic remote failure propagation,
 * stale/malformed vote rejection, the impossibility of early publication, and
 * all-rank lease drainage before physical old-bank retirement.
 */

#include "execution/moe/MoEOverlayDistributedResidencyProtocol.h"
#include "execution/moe/MoEOverlayMPIRemoteProjectionTransport.h"
#include "execution/moe/MoEOverlayRemoteProjectionProtocol.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Build one whole-expert execution domain on any world rank. */
        RoutedExpertDomain domain(
            std::string name,
            GlobalDeviceAddress participant,
            int world_rank,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = backend;
            result.participants = {participant};
            result.world_ranks = {world_rank};
            result.owner_rank = world_rank;
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Build one thermally ordered tier with fixed logical capacity. */
        RoutedExpertTier tier(
            std::string name,
            std::string domain_name,
            int priority,
            int capacity,
            bool cold_remainder = false)
        {
            RoutedExpertTier result;
            result.name = std::move(name);
            result.domain = std::move(domain_name);
            result.priority = priority;
            result.max_experts_per_layer = capacity;
            result.fallback = cold_remainder;
            return result;
        }

        /**
         * @brief Build a hardware-order-independent CUDA/ROCm/CPU placement.
         *
         * CUDA intentionally lives on rank 2 and ROCm on rank 0.  Protocol
         * identity must describe resolved topology, never infer devices from
         * rank or socket number.
         */
        MoERoutedExpertPlacementPlan threeRankPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Random;
            plan.domains = {
                domain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(1, 1, "node-a"),
                    2,
                    CollectiveBackendType::NCCL),
                domain(
                    "rocm_warm",
                    GlobalDeviceAddress::rocm(0, 0, "node-a"),
                    0,
                    CollectiveBackendType::RCCL),
                domain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(1, "node-a"),
                    1,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                tier("hot", "cuda_hot", 0, 2),
                tier("warm", "rocm_warm", 1, 2),
                tier("cold", "cpu_cold", 2, 0, true),
            };
            return plan;
        }

        /** @brief Small model geometry sufficient to produce a three-tier swap. */
        MoERoutedExpertModelMetadata modelMetadata()
        {
            MoERoutedExpertModelMetadata metadata;
            metadata.num_layers = 1;
            metadata.num_experts = 6;
            metadata.d_model = 16;
            metadata.routed_intermediate_size = 8;
            metadata.routed_quant_type = "F32";
            return metadata;
        }

        /** @brief Create one frozen histogram whose hottest experts start cold. */
        std::unique_ptr<DecodeExpertHistogram> makeHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 6;
            config.top_k = 2;
            config.window_size = 4;
            config.sockets = {
                DeviceId::cuda(1),
                DeviceId::rocm(0),
                DeviceId::cpu(),
            };
            /* Random owner ordering is resolved by the authority's owner map. */
            config.ownership = MoELayeredExpertOwnership::uniform(
                1, 3, {0, 0, 1, 1, 2, 2});
            auto histogram = std::make_unique<DecodeExpertHistogram>(config);
            const std::vector<std::uint64_t> counts{1, 2, 3, 4, 100, 90};
            histogram->mergeLayerCounts(0, counts.data(), 6, false);
            return histogram;
        }

        /** @brief Own authority dependencies alongside the proposed transaction. */
        struct TransactionFixture
        {
            std::unique_ptr<DecodeExpertHistogram> histogram;
            std::unique_ptr<MoEOverlayResidencyAuthority> authority;
            MoEOverlayResidencyTransaction transaction;
        };

        /** @brief Produce the same deterministic rank-local transaction. */
        TransactionFixture makeTransaction()
        {
            TransactionFixture fixture;
            fixture.histogram = makeHistogram();
            fixture.authority =
                std::make_unique<MoEOverlayResidencyAuthority>(
                    MoEOverlayResidencyAuthority::Config{
                        .initial_plan = threeRankPlan(),
                        .model_metadata = modelMetadata(),
                        .maintenance_mode =
                            MoERebalanceRuntimeMode::Dynamic,
                        .histogram = fixture.histogram.get(),
                        .shadow_slots_per_endpoint_layer = 0,
                        .max_concurrent_cycles = 0,
                        .perf_device = "cpu_protocol_test",
                    });
            fixture.transaction =
                fixture.authority->proposeFromHistogram();
            if (!fixture.transaction.valid() || fixture.transaction.empty())
            {
                throw std::logic_error(
                    "Distributed protocol fixture did not produce movement");
            }
            return fixture;
        }

        /** @brief Construct one protocol per world rank for an exact identity. */
        std::vector<std::unique_ptr<
            MoEOverlayDistributedResidencyProtocol>>
        makeProtocols(
            const MoEOverlayDistributedResidencyWaveIdentity &identity,
            int world_size = 3)
        {
            std::vector<std::unique_ptr<
                MoEOverlayDistributedResidencyProtocol>> protocols;
            for (int rank = 0; rank < world_size; ++rank)
            {
                protocols.push_back(std::make_unique<
                    MoEOverlayDistributedResidencyProtocol>(
                    MoEOverlayDistributedResidencyProtocol::Config{
                        .identity = identity,
                        .local_world_rank = rank,
                        .world_size = world_size,
                    }));
            }
            return protocols;
        }

        /** @brief Collect the same successful phase vote from every rank. */
        std::vector<MoEOverlayDistributedResidencyVote> readyVotes(
            std::vector<std::unique_ptr<
                MoEOverlayDistributedResidencyProtocol>> &protocols)
        {
            std::vector<MoEOverlayDistributedResidencyVote> votes;
            for (auto &protocol : protocols)
            {
                votes.push_back(protocol->makeLocalVote(
                    MoEOverlayDistributedResidencyVoteDecision::Ready));
            }
            return votes;
        }

        /** @brief Build one exact remote projection identity for protocol tests. */
        MoEOverlayRemoteProjectionIdentity remoteProjectionIdentity(
            const MoEOverlayResidencyTransaction &transaction,
            DeviceId source_device,
            DeviceId destination_device)
        {
            return {
                .expected_epoch = transaction.expected_epoch,
                .candidate_epoch = transaction.candidate->epoch,
                .transaction_fingerprint =
                    fingerprintMoEOverlayResidencyTransaction(transaction),
                .migration_index = 0,
                .layer_idx = 0,
                .expert_id = 4,
                .projection = ExpertTierWeightProjection::Gate,
                .source_participant = 0,
                .destination_participant = 1,
                .source_world_rank = 2,
                .destination_world_rank = 0,
                .source_device = source_device,
                .destination_device = destination_device,
            };
        }

        /** @brief Materialize deterministic final Q4_0 CPU execution bytes. */
        cpu::native_vnni::CPUNativeVNNIPackedWeights cpuProjectionBytes()
        {
            using namespace cpu::native_vnni;
            CPUNativeVNNIPackedWeights packed;
            packed.N = 32;
            packed.K = 64;
            packed.N_padded = 64;
            packed.blocks_per_row = 2;
            packed.codebook_id = native_vnni_formats::Q4_0.codebook_id;
            packed.payload_bytes = 16;
            packed.encoding = CPUNativeVNNIEncoding::NibbleLUT;
            packed.is_asymmetric = false;
            packed.is_superblock = false;
            packed.data_stride = static_cast<int>(
                preparedDataStride(packed.encoding));
            packed.interleaved_block_stride =
                preparedInterleavedBlockStride(
                    packed.encoding, packed.is_asymmetric);
            const std::size_t bytes =
                static_cast<std::size_t>(packed.N_padded / 64) *
                static_cast<std::size_t>(packed.blocks_per_row) *
                static_cast<std::size_t>(
                    packed.interleaved_block_stride);
            packed.native_interleaved.resize_uninitialized(bytes);
            for (std::size_t index = 0; index < bytes; ++index)
            {
                packed.native_interleaved[index] =
                    static_cast<std::uint8_t>((index * 37u + 11u) & 0xffu);
            }
            return packed;
        }

        /** @brief Materialize deterministic expanded asymmetric Q5_1 bytes. */
        cpu::native_vnni::CPUNativeVNNIPackedWeights
        cpuAsymmetricProjectionBytes()
        {
            using namespace cpu::native_vnni;
            CPUNativeVNNIPackedWeights packed;
            packed.N = 32;
            packed.K = 64;
            packed.N_padded = 64;
            packed.blocks_per_row = 2;
            packed.codebook_id = native_vnni_formats::Q5_1.codebook_id;
            packed.payload_bytes = 32;
            packed.encoding = preparedEncodingForCodebook(
                packed.codebook_id);
            packed.is_asymmetric = true;
            packed.is_superblock = false;
            packed.data_stride = static_cast<int>(
                preparedDataStride(packed.encoding));
            packed.interleaved_block_stride =
                preparedInterleavedBlockStride(
                    packed.encoding, packed.is_asymmetric);
            const std::size_t bytes =
                static_cast<std::size_t>(packed.N_padded / 64) *
                static_cast<std::size_t>(packed.blocks_per_row) *
                packed.interleaved_block_stride;
            packed.native_interleaved.resize_uninitialized(bytes);
            for (std::size_t index = 0; index < bytes; ++index)
            {
                packed.native_interleaved[index] =
                    static_cast<std::uint8_t>((index * 53u + 17u) & 0xffu);
            }
            return packed;
        }
    } // namespace

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        RemoteProjectionLaneBudgetScalesWithConfiguredCyclesAndRejectsOverflow)
    {
        const MoEOverlayRemoteProjectionLaneBudget one_cycle{
            .maximum_participants_per_cycle = 3,
            .maximum_concurrent_cycles = 1,
            .projections_per_expert = 3,
        };
        const MoEOverlayRemoteProjectionLaneBudget two_cycles{
            .maximum_participants_per_cycle = 3,
            .maximum_concurrent_cycles = 2,
            .projections_per_expert = 3,
        };
        ASSERT_TRUE(one_cycle.projectionOperationCount().has_value());
        ASSERT_TRUE(two_cycles.projectionOperationCount().has_value());
        EXPECT_EQ(*one_cycle.projectionOperationCount(), 9u);
        EXPECT_EQ(*two_cycles.projectionOperationCount(), 18u);

        const MoEOverlayRemoteProjectionLaneBudget zero_participants{
            .maximum_participants_per_cycle = 0,
            .maximum_concurrent_cycles = 2,
            .projections_per_expert = 3,
        };
        const MoEOverlayRemoteProjectionLaneBudget overflowing{
            .maximum_participants_per_cycle =
                std::numeric_limits<std::size_t>::max(),
            .maximum_concurrent_cycles = 2,
            .projections_per_expert = 3,
        };
        EXPECT_FALSE(
            zero_participants.projectionOperationCount().has_value());
        EXPECT_FALSE(overflowing.projectionOperationCount().has_value());
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        RemoteCpuProjectionManifestAndChunksAreExactAndAuthenticated)
    {
        auto fixture = makeTransaction();
        auto cpu_bytes = cpuProjectionBytes();
        const auto identity = remoteProjectionIdentity(
            fixture.transaction, DeviceId::cpu(), DeviceId::cpu());
        const auto manifest = makeMoEOverlayRemoteCpuProjectionManifest(
            identity, cpu_bytes, /*maximum_chunk_bytes=*/73);
        ASSERT_TRUE(manifest.valid());
        EXPECT_EQ(manifest.region_bytes[0], cpu_bytes.native_interleaved.size());
        EXPECT_EQ(manifest.region_bytes[1], 0u);

        std::array<
            std::uint8_t,
            MoEOverlayRemoteProjectionManifest::kWireBytes> packet{};
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayRemoteProjectionManifest(
            manifest, packet, &error))
            << error;
        EXPECT_EQ(packet[0], static_cast<std::uint8_t>('M'));
        EXPECT_EQ(packet[1], static_cast<std::uint8_t>('O'));
        EXPECT_EQ(packet[2], static_cast<std::uint8_t>('R'));
        EXPECT_EQ(packet[3], static_cast<std::uint8_t>('P'));

        MoEOverlayRemoteProjectionManifest decoded;
        ASSERT_TRUE(decodeMoEOverlayRemoteProjectionManifest(
            packet, &decoded, &error))
            << error;
        EXPECT_EQ(decoded.identity, manifest.identity);
        EXPECT_EQ(decoded.manifest_hash, manifest.manifest_hash);

        auto corrupted_packet = packet;
        corrupted_packet[97] ^= 0x40u;
        EXPECT_FALSE(decodeMoEOverlayRemoteProjectionManifest(
            corrupted_packet, &decoded, &error));
        EXPECT_FALSE(decoded.valid());

        const std::array<std::span<const std::uint8_t>, 4> regions{
            std::span<const std::uint8_t>(
                cpu_bytes.native_interleaved.data(),
                cpu_bytes.native_interleaved.size()),
            std::span<const std::uint8_t>{},
            std::span<const std::uint8_t>{},
            std::span<const std::uint8_t>{},
        };
        MoEOverlayRemoteProjectionChunkCursor cursor(manifest, regions);
        MoEOverlayRemoteProjectionChunkValidator validator(manifest);
        std::uint64_t chunks = 0;
        while (auto chunk = cursor.takeNext())
        {
            EXPECT_EQ(chunk->header.sequence, chunks);
            EXPECT_LE(chunk->payload.size(), 73u);

            std::array<
                std::uint8_t,
                MoEOverlayRemoteProjectionChunkHeader::kWireBytes>
                header_packet{};
            ASSERT_TRUE(encodeMoEOverlayRemoteProjectionChunkHeader(
                chunk->header, header_packet, &error))
                << error;
            MoEOverlayRemoteProjectionChunkHeader decoded_header;
            ASSERT_TRUE(decodeMoEOverlayRemoteProjectionChunkHeader(
                header_packet, &decoded_header, &error))
                << error;
            EXPECT_EQ(decoded_header.manifest_hash,
                      chunk->header.manifest_hash);
            EXPECT_EQ(decoded_header.sequence, chunk->header.sequence);

            if (chunks == 0)
            {
                auto corrupt_header = chunk->header;
                corrupt_header.region_offset = 1;
                EXPECT_FALSE(validator.accept(
                    corrupt_header, chunk->payload, &error));
                EXPECT_FALSE(validator.complete())
                    << "Rejected input must not advance receive state";
            }
            ASSERT_TRUE(validator.accept(
                chunk->header, chunk->payload, &error))
                << error;
            ++chunks;
        }
        EXPECT_GT(chunks, 1u);
        EXPECT_TRUE(cursor.complete());
        EXPECT_TRUE(validator.complete());
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        RemoteGpuBlobManifestCarriesCurrentExecutionFormatSeparatelyFromProvenance)
    {
        auto fixture = makeTransaction();
        const auto identity = remoteProjectionIdentity(
            fixture.transaction, DeviceId::cuda(1), DeviceId::rocm(0));
        const NativeVnniSourceIdentity source_identity{
            .codebook_id = native_vnni_formats::Q8_0.codebook_id,
            .is_superblock = native_vnni_formats::Q8_0.is_superblock,
            .present = true,
        };
        std::array<std::uint8_t, 1> byte{};
        constexpr std::size_t blocks = 32u * 2u;
        const GpuExpertPackedDescriptor normalized{
            .ptrs = {
                .d_vnni = byte.data(),
                .d_scales = byte.data(),
            },
            .n = 32,
            .k = 64,
            .blocks_per_row = 2,
            .codebook_id = 19,
            .payload_bytes_per_block = 32,
            .is_asymmetric = false,
            .has_emins = false,
            .vnni_bytes = blocks * 32u,
            .scales_bytes = blocks * sizeof(std::uint16_t),
        };
        const auto manifest = makeMoEOverlayRemoteGpuProjectionManifest(
            identity, normalized, source_identity, 256);
        ASSERT_TRUE(manifest.valid());
        EXPECT_EQ(manifest.source_codebook_id,
                  native_vnni_formats::Q8_0.codebook_id);
        EXPECT_EQ(manifest.gpu_codebook_id, 19u)
            << "A normalized promoted GPU expert must not be relabeled as its GGUF codebook";
        EXPECT_EQ(manifest.region_bytes[0], normalized.vnni_bytes);
        EXPECT_EQ(manifest.region_bytes[1], normalized.scales_bytes);

        auto invalid = manifest;
        invalid.gpu_codebook_id = 4;
        invalid.manifest_hash = invalid.computedHash();
        std::string error;
        EXPECT_FALSE(invalid.valid(&error));
        EXPECT_NE(error.find("GPU"), std::string::npos);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        CpuGpuManifestUsesWholeRepackUnitsAndRejectsPartialWireCapacity)
    {
        auto fixture = makeTransaction();
        const auto cpu_bytes = cpuAsymmetricProjectionBytes();
        const auto identity = remoteProjectionIdentity(
            fixture.transaction, DeviceId::cpu(), DeviceId::cuda(0));
        const auto stream = makeCpuToGpuExpertTierWeightStreamManifest(
            cpu_bytes,
            identity.candidate_epoch,
            identity.layer_idx,
            identity.expert_id,
            identity.projection,
            /*maximum_units_per_chunk=*/1);
        const auto manifest = makeMoEOverlayRemoteCpuProjectionManifest(
            identity,
            stream,
            /*maximum_chunk_bytes=*/4096);

        ASSERT_TRUE(manifest.valid());
        EXPECT_EQ(manifest.maximum_chunk_bytes, manifest.cpu_block_stride);
        const auto layout = remoteCpuProjectionDeviceLayout(manifest);
        EXPECT_EQ(
            layout.direction,
            ExpertTierWeightConversionDirection::CpuToGpu);
        EXPECT_EQ(layout.maximum_units_per_chunk, 1u);
        EXPECT_EQ(layout.unit_count, stream.unit_count);
        EXPECT_EQ(layout.gpu_codebook_id, stream.gpu_codebook_id);

        auto partial = manifest;
        ++partial.maximum_chunk_bytes;
        partial.manifest_hash = partial.computedHash();
        std::string error;
        EXPECT_FALSE(partial.valid(&error));
        EXPECT_NE(error.find("complete NativeVNNI units"), std::string::npos);
        EXPECT_THROW(
            {
                [[maybe_unused]] const auto rejected =
                    remoteCpuProjectionDeviceLayout(partial);
            },
            std::invalid_argument);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        HostSourceRetainsEveryPayloadUntilItsExactMpiAcknowledgement)
    {
        auto fixture = makeTransaction();
        const auto cpu_bytes = cpuProjectionBytes();
        const auto identity = remoteProjectionIdentity(
            fixture.transaction, DeviceId::cpu(), DeviceId::cpu());
        const auto manifest = makeMoEOverlayRemoteCpuProjectionManifest(
            identity, cpu_bytes, /*maximum_chunk_bytes=*/73);
        const std::array<std::span<const std::uint8_t>, 4> regions{
            std::span<const std::uint8_t>(
                cpu_bytes.native_interleaved.data(),
                cpu_bytes.native_interleaved.size()),
            std::span<const std::uint8_t>{},
            std::span<const std::uint8_t>{},
            std::span<const std::uint8_t>{},
        };
        MoEOverlayHostRemoteProjectionSource source(
            manifest, regions, std::make_shared<int>(1));

        MoEOverlayRemoteProjectionChunkView first;
        std::string error;
        ASSERT_EQ(
            source.pollNextChunk(&first, &error),
            MoEOverlayResidencyWaveProgress::Ready)
            << error;
        ASSERT_FALSE(first.payload.empty());
        MoEOverlayRemoteProjectionChunkView forbidden_reuse;
        EXPECT_EQ(
            source.pollNextChunk(&forbidden_reuse, &error),
            MoEOverlayResidencyWaveProgress::Failed);
        EXPECT_NE(error.find("still owned by MPI"), std::string::npos);

        auto wrong_ack = first.header;
        wrong_ack.payload_hash ^= 1u;
        EXPECT_FALSE(source.acknowledgeChunkSent(wrong_ack, &error));
        EXPECT_FALSE(source.acknowledgeChunkSent(
            MoEOverlayRemoteProjectionChunkHeader{}, &error));
        ASSERT_TRUE(source.acknowledgeChunkSent(first.header, &error)) << error;

        std::uint64_t acknowledged_chunks = 1;
        while (acknowledged_chunks * manifest.maximum_chunk_bytes <
               manifest.total_bytes)
        {
            MoEOverlayRemoteProjectionChunkView chunk;
            ASSERT_EQ(
                source.pollNextChunk(&chunk, &error),
                MoEOverlayResidencyWaveProgress::Ready)
                << error;
            ASSERT_EQ(chunk.header.sequence, acknowledged_chunks);
            ASSERT_TRUE(source.acknowledgeChunkSent(chunk.header, &error))
                << error;
            ++acknowledged_chunks;
        }
        EXPECT_GT(acknowledged_chunks, 1u);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        CpuDestinationAcceptsOnlyEquivalentAuthenticatedLiveGpuFormat)
    {
        auto fixture = makeTransaction();
        const auto cpu_bytes = cpuAsymmetricProjectionBytes();
        const auto identity = remoteProjectionIdentity(
            fixture.transaction, DeviceId::cuda(0), DeviceId::cpu());
        const NativeVnniFormatInfo *source_format =
            native_vnni_formats::forSourceIdentity(
                cpu_bytes.codebook_id, cpu_bytes.is_superblock);
        ASSERT_NE(source_format, nullptr);
        const auto canonical_stream =
            makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format,
                cpu_bytes.N,
                cpu_bytes.K,
                identity.expected_epoch,
                identity.layer_idx,
                identity.expert_id,
                identity.projection,
                /*maximum_units_per_chunk=*/1);
        const auto expected = makeMoEOverlayRemoteCpuProjectionManifest(
            identity, canonical_stream, /*maximum_chunk_bytes=*/4096);
        auto live = expected;
        /* Q5_1 demoted through CPU promotes as normalized INT8+minimum. */
        live.gpu_codebook_id = kNativeVnniExpandedInt8MinCodebook;
        live.gpu_payload_bytes_per_block = 32;
        live.gpu_is_asymmetric = 1;
        live.gpu_has_emins = 0;
        live.manifest_hash = live.computedHash();
        ASSERT_TRUE(live.valid());

        std::vector<std::uint8_t> destination_bytes(
            cpu_bytes.native_interleaved.size(), 0u);
        std::array<std::span<std::uint8_t>, 4> regions{
            std::span<std::uint8_t>(destination_bytes),
            std::span<std::uint8_t>{},
            std::span<std::uint8_t>{},
            std::span<std::uint8_t>{},
        };
        MoEOverlayHostRemoteProjectionDestination destination(
            identity,
            regions,
            std::make_shared<int>(2),
            expected,
            MoEOverlayRemoteProjectionManifestMatchPolicy::
                EquivalentGpuSourceForFinalCpuStorage);
        std::string error;
        EXPECT_TRUE(destination.beginManifest(live, &error)) << error;

        auto altered_geometry = live;
        altered_geometry.maximum_chunk_bytes *= 2;
        altered_geometry.manifest_hash = altered_geometry.computedHash();
        ASSERT_TRUE(altered_geometry.valid());
        MoEOverlayHostRemoteProjectionDestination rejecting_destination(
            identity,
            regions,
            std::make_shared<int>(3),
            expected,
            MoEOverlayRemoteProjectionManifestMatchPolicy::
                EquivalentGpuSourceForFinalCpuStorage);
        EXPECT_FALSE(rejecting_destination.beginManifest(
            altered_geometry, &error));
        EXPECT_NE(error.find("physical format"), std::string::npos);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        FrozenHistogramWirePacketIsExactAndAuthenticated)
    {
        DecodeExpertHistogramWindow source{
            .generation = 7,
            .token_count = 19,
            .source_token_counts = {19, 0, 0},
            .num_layers = 2,
            .num_experts = 3,
            .expert_counts = {1, 2, 3, 100, 200, 300},
            .source_expert_counts = {
                1, 2, 3, 100, 200, 300,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            },
        };
        const std::size_t packet_bytes =
            moeOverlayDistributedHistogramWireBytes(2, 3);
        EXPECT_EQ(
            packet_bytes,
            MoEOverlayDistributedHistogramHeader::kWireBytes +
                kExpertHistogramProductionSourceCount *
                    sizeof(std::uint64_t) +
                (1u + kExpertHistogramProductionSourceCount) * 6u *
                    sizeof(std::uint64_t));

        std::vector<std::uint8_t> packet(packet_bytes);
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedHistogramWindow(
            source, packet, &error))
            << error;
        /* Magic 0x484f4f4d is emitted as canonical little-endian "MOOH". */
        EXPECT_EQ(packet[0], static_cast<std::uint8_t>('M'));
        EXPECT_EQ(packet[1], static_cast<std::uint8_t>('O'));
        EXPECT_EQ(packet[2], static_cast<std::uint8_t>('O'));
        EXPECT_EQ(packet[3], static_cast<std::uint8_t>('H'));

        DecodeExpertHistogramWindow decoded;
        decoded.expert_counts.reserve(6);
        ASSERT_TRUE(decodeMoEOverlayDistributedHistogramWindow(
            packet, 2, 3, &decoded, &error))
            << error;
        EXPECT_EQ(decoded.generation, source.generation);
        EXPECT_EQ(decoded.token_count, source.token_count);
        EXPECT_EQ(decoded.source_token_counts, source.source_token_counts);
        EXPECT_EQ(decoded.expert_counts, source.expert_counts);
        EXPECT_EQ(decoded.source_expert_counts, source.source_expert_counts);
        EXPECT_EQ(
            fingerprintDecodeExpertHistogramWindow(decoded),
            fingerprintDecodeExpertHistogramWindow(source));

        auto corrupted = packet;
        corrupted.back() ^= 0x80u;
        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(
            corrupted, 2, 3, &decoded, &error));
        EXPECT_NE(error.find("authentication"), std::string::npos);
        EXPECT_FALSE(decoded.valid());

        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(
            packet, 1, 6, &decoded, &error));
        EXPECT_NE(error.find("geometry"), std::string::npos);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        IndependentRankTransactionsHaveIdenticalCompleteFingerprint)
    {
        auto rank_zero = makeTransaction();
        auto rank_one = makeTransaction();

        const auto identity_zero =
            makeMoEOverlayDistributedResidencyWaveIdentity(
                rank_zero.transaction);
        const auto identity_one =
            makeMoEOverlayDistributedResidencyWaveIdentity(
                rank_one.transaction);

        EXPECT_TRUE(identity_zero.valid());
        EXPECT_EQ(identity_zero, identity_one);
        EXPECT_EQ(
            identity_zero.migration_count,
            rank_zero.transaction.migrations.size());
        EXPECT_EQ(
            identity_zero.cycle_count,
            rank_zero.transaction.migration_cycles.size());

        auto altered = rank_zero.transaction;
        altered.migrations.front().activation_count += 1;
        ASSERT_TRUE(altered.valid());
        EXPECT_NE(
            fingerprintMoEOverlayResidencyTransaction(altered),
            identity_zero.transaction_fingerprint);

        auto altered_window = std::make_shared<
            DecodeExpertHistogramWindow>(*rank_zero.transaction.histogram_window);
        altered_window->expert_counts.front() += 1;
        altered_window->source_expert_counts.front() += 1;
        altered.histogram_window = std::move(altered_window);
        ASSERT_TRUE(altered.valid());
        EXPECT_NE(
            fingerprintMoEOverlayResidencyTransaction(altered),
            identity_zero.transaction_fingerprint);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        MeasuredEconomyEvidenceParticipatesInCompleteFingerprint)
    {
        auto fixture = makeTransaction();
        const auto baseline =
            fingerprintMoEOverlayResidencyTransaction(fixture.transaction);

        auto measured = fixture.transaction;
        measured.economy = {
            .enabled = true,
            .service_profile_identity = "service-profile-a",
            .migration_profile_identity = "movement-profile-a",
            .smoothed_through_generation =
                measured.histogram_generation,
            .historical_window_weight = 3,
            .current_window_weight = 1,
            .payoff_horizon_tokens = 8,
            .minimum_net_benefit_ns = 10,
            .minimum_residency_generations = 2,
            .projected_service_gain_ns = 1'000,
            .projected_transfer_and_repack_ns = 250,
            .projected_inference_interference_ns = 50,
            .projected_net_benefit_ns = 700,
            .payoff_rejected_cycles = 1,
            .residency_rejected_cycles = 2,
        };
        ASSERT_TRUE(measured.valid());
        const auto measured_fingerprint =
            fingerprintMoEOverlayResidencyTransaction(measured);
        EXPECT_NE(measured_fingerprint, baseline);

        auto different_profile = measured;
        different_profile.economy.migration_profile_identity =
            "movement-profile-b";
        ASSERT_TRUE(different_profile.valid());
        EXPECT_NE(
            fingerprintMoEOverlayResidencyTransaction(different_profile),
            measured_fingerprint);

        auto different_policy = measured;
        different_policy.economy.payoff_horizon_tokens = 9;
        ASSERT_TRUE(different_policy.valid());
        EXPECT_NE(
            fingerprintMoEOverlayResidencyTransaction(different_policy),
            measured_fingerprint);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        CoordinatorFrozenWindowProducesIdenticalRankLocalTransactions)
    {
        auto coordinator_histogram = makeHistogram();
        auto peer_histogram = makeHistogram();
        MoEOverlayResidencyAuthority coordinator(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = threeRankPlan(),
                .model_metadata = modelMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = coordinator_histogram.get(),
                .perf_device = "coordinator",
            });
        MoEOverlayResidencyAuthority peer(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = threeRankPlan(),
                .model_metadata = modelMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = peer_histogram.get(),
                .perf_device = "peer",
            });

        const auto published_window =
            coordinator.freezeAndRotateHistogramWindow();
        const auto coordinator_transaction =
            coordinator.proposeFromFrozenHistogramWindow(
                published_window);
        const auto peer_transaction =
            peer.proposeFromFrozenHistogramWindow(published_window);

        EXPECT_EQ(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                coordinator_transaction),
            makeMoEOverlayDistributedResidencyWaveIdentity(
                peer_transaction));
        EXPECT_EQ(
            coordinator_histogram->windowGeneration(),
            published_window->generation + 1);
        EXPECT_EQ(
            peer_histogram->windowGeneration(),
            published_window->generation)
            << "A peer must consume coordinator evidence without rotating its local partial window";

        auto wrong_geometry =
            std::make_shared<DecodeExpertHistogramWindow>(*published_window);
        wrong_geometry->num_experts -= 1;
        EXPECT_THROW(
            {
                [[maybe_unused]] const auto rejected =
                    peer.proposeFromFrozenHistogramWindow(wrong_geometry);
            },
            std::invalid_argument);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        RequiresUnanimousStageAndCommitBeforeEveryRankCanPublish)
    {
        auto fixture = makeTransaction();
        const auto identity =
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction);
        auto protocols = makeProtocols(identity);

        for (auto &protocol : protocols)
            EXPECT_THROW(protocol->markPublished(), std::logic_error);

        const auto reservation_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_TRUE(protocol->acceptConsensus(reservation_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    AwaitingLocalStage);
            EXPECT_THROW(protocol->markPublished(), std::logic_error);
        }

        const auto stage_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_TRUE(protocol->acceptConsensus(stage_votes, &error))
                << error;
            EXPECT_TRUE(error.empty());
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    AwaitingLocalCommit);
            EXPECT_THROW(protocol->markPublished(), std::logic_error);
        }

        const auto commit_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_TRUE(protocol->acceptConsensus(commit_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    ReadyToPublish);
            protocol->markPublished();
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Published);
            EXPECT_FALSE(protocol->abort());
        }

        const auto retirement_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            EXPECT_THROW(protocol->markRetired(), std::logic_error);
            std::string error;
            EXPECT_TRUE(protocol->acceptConsensus(retirement_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::ReadyToRetire);
            protocol->markRetired();
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Retired);
            EXPECT_FALSE(protocol->abort());
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        LowestFailedStageRankDeterministicallyAbortsTheWave)
    {
        auto fixture = makeTransaction();
        auto protocols = makeProtocols(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction));

        const auto reservation_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
            ASSERT_TRUE(protocol->acceptConsensus(reservation_votes));

        std::vector<MoEOverlayDistributedResidencyVote> votes;
        votes.push_back(protocols[0]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        votes.push_back(protocols[1]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Failed,
            41,
            "rank one destination bank authentication failed"));
        votes.push_back(protocols[2]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Failed,
            73,
            "rank two transfer event failed"));

        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_FALSE(protocol->acceptConsensus(votes, &error));
            EXPECT_NE(error.find("world rank 1"), std::string::npos);
            ASSERT_TRUE(protocol->failure().has_value());
            EXPECT_EQ(protocol->failure()->world_rank, 1);
            EXPECT_EQ(protocol->failure()->error_code, 41);
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Failed);
            EXPECT_THROW(
                {
                    [[maybe_unused]] const auto rejected_vote =
                        protocol->makeLocalVote(
                            MoEOverlayDistributedResidencyVoteDecision::Ready);
                },
                std::logic_error);
            EXPECT_TRUE(protocol->abort());
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        TransientReservationBackpressureDefersEveryRankWithoutBecomingFailure)
    {
        auto fixture = makeTransaction();
        auto protocols = makeProtocols(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction));

        std::vector<MoEOverlayDistributedResidencyVote> votes;
        votes.push_back(protocols[0]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        votes.push_back(protocols[1]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Deferred));
        votes.push_back(protocols[2]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));

        for (auto &protocol : protocols)
        {
            std::string error = "must be cleared";
            EXPECT_FALSE(protocol->acceptConsensus(votes, &error));
            EXPECT_TRUE(error.empty());
            EXPECT_FALSE(protocol->failure().has_value());
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Deferred);
            EXPECT_THROW(protocol->markPublished(), std::logic_error);
            EXPECT_TRUE(protocol->abort());
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Aborted);
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        CommitFailureNeverBecomesPublishable)
    {
        auto fixture = makeTransaction();
        auto protocols = makeProtocols(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction));

        const auto reservation_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
            ASSERT_TRUE(protocol->acceptConsensus(reservation_votes));

        const auto stage_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
            ASSERT_TRUE(protocol->acceptConsensus(stage_votes));

        std::vector<MoEOverlayDistributedResidencyVote> commit_votes;
        commit_votes.push_back(protocols[0]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        commit_votes.push_back(protocols[1]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        commit_votes.push_back(protocols[2]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Failed,
            88,
            "rank two inactive bank install failed"));

        for (auto &protocol : protocols)
        {
            EXPECT_FALSE(protocol->acceptConsensus(commit_votes));
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Failed);
            EXPECT_THROW(protocol->markPublished(), std::logic_error);
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        DivergentTransactionIdentityIsFatalBeforeAnyCommit)
    {
        auto fixture = makeTransaction();
        auto protocols = makeProtocols(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction));
        auto votes = readyVotes(protocols);

        /* Keep the foreign identity structurally valid while changing its digest. */
        votes[2].identity.transaction_fingerprint.low ^= 0x100u;
        ASSERT_TRUE(votes[2].valid(3));

        std::string error;
        EXPECT_FALSE(protocols[0]->acceptConsensus(votes, &error));
        EXPECT_NE(error.find("disagree"), std::string::npos);
        EXPECT_EQ(
            protocols[0]->state(),
            MoEOverlayDistributedResidencyProtocolState::Failed);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        DuplicateRankOrAlteredLocalVoteIsRejected)
    {
        auto fixture = makeTransaction();
        const auto identity =
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction);

        {
            auto protocols = makeProtocols(identity);
            auto votes = readyVotes(protocols);
            votes[2] = votes[1];
            std::string error;
            EXPECT_FALSE(protocols[0]->acceptConsensus(votes, &error));
            EXPECT_NE(error.find("duplicate"), std::string::npos);
        }

        {
            auto protocols = makeProtocols(identity);
            auto votes = readyVotes(protocols);
            votes[1].decision =
                MoEOverlayDistributedResidencyVoteDecision::Failed;
            votes[1].error_code = 17;
            votes[1].detail_fingerprint = 99;
            ASSERT_TRUE(votes[1].valid(3));

            std::string error;
            EXPECT_FALSE(protocols[1]->acceptConsensus(votes, &error));
            EXPECT_NE(error.find("altered"), std::string::npos);
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        VoteDiagnosticsAndRankGeometryAreStrict)
    {
        auto fixture = makeTransaction();
        const auto identity =
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction);

        EXPECT_THROW(
            MoEOverlayDistributedResidencyProtocol({
                .identity = identity,
                .local_world_rank = 3,
                .world_size = 3,
            }),
            std::invalid_argument);

        MoEOverlayDistributedResidencyProtocol protocol({
            .identity = identity,
            .local_world_rank = 0,
            .world_size = 3,
        });
        EXPECT_THROW(
            {
                [[maybe_unused]] const auto rejected_vote =
                    protocol.makeLocalVote(
                        MoEOverlayDistributedResidencyVoteDecision::Failed,
                        0,
                        "missing positive code");
            },
            std::invalid_argument);
        EXPECT_THROW(
            {
                [[maybe_unused]] const auto rejected_vote =
                    protocol.makeLocalVote(
                        MoEOverlayDistributedResidencyVoteDecision::Failed,
                        4,
                        {});
            },
            std::invalid_argument);
        EXPECT_THROW(
            {
                [[maybe_unused]] const auto rejected_vote =
                    protocol.makeLocalVote(
                        MoEOverlayDistributedResidencyVoteDecision::Ready,
                        4,
                        "ready cannot fail");
            },
            std::invalid_argument);
    }
} // namespace llaminar2::test
