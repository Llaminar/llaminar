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
#include "execution/moe/MoEOverlayWireIO.h"
#include "planning/PhysicalMemoryAuthority.h"

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
        /** @return A rank-local ledger for bounded transaction evidence, not model weights. */
        ExpertHistogramTransactionConfig demandAdmission()
        {
            PhysicalMemoryBOMBuilder bom({.world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = 1u << 20, .admission_available_bytes = 1u << 20});
            bom.add(PhysicalMemoryOwner::ExecutionWorkspace, 1u << 20);
            PhysicalMemoryPlanBuilder plan;
            plan.add(bom.build());
            return {{16, 4, 2}, std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0)};
        }

        /** @return A frozen sample from real histogram ingress with two independent decode batches. */
        DecodeExpertHistogramWindow demandWindow(const ExpertHistogramTransactionConfig &admission,
                                                std::array<int, 4> routes)
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 4;
            config.top_k = 2;
            config.window_size = 4;
            config.sockets = {DeviceId::cpu(), DeviceId(DeviceType::CPU, 1)};
            config.ownership = MoELayeredExpertOwnership::uniform(1, 2, {0, 0, 1, 1});
            config.transaction_demand = admission;
            DecodeExpertHistogram histogram(config);
            const float weights[]{0.5f, 0.5f};
            histogram.record(0, routes.data(), weights, 2);
            histogram.record(0, routes.data() + 2, weights, 2);
            return histogram.freezeAndRotateWindow();
        }
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

        /** @brief Advance every rank through publication with unanimous votes. */
        void advanceProtocolsToPublished(
            std::vector<std::unique_ptr<
                MoEOverlayDistributedResidencyProtocol>> &protocols)
        {
            for (int phase = 0; phase < 4; ++phase)
            {
                const auto votes = readyVotes(protocols);
                for (auto &protocol : protocols)
                {
                    std::string error;
                    ASSERT_TRUE(protocol->acceptConsensus(votes, &error))
                        << error;
                }
            }
            for (auto &protocol : protocols)
            {
                ASSERT_EQ(
                    protocol->state(),
                    MoEOverlayDistributedResidencyProtocolState::
                        ReadyForAuthorityPublication);
                protocol->markAuthorityPublished();
            }
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
                .execution_fingerprint =
                    fingerprintMoEOverlayResidencyExecutionPlan(transaction),
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
        RemoteFloatingManifestAuthenticatesEveryEndpointPairAndRawBytes)
    {
        auto fixture = makeTransaction();
        const auto identity = remoteProjectionIdentity(
            fixture.transaction, DeviceId::cuda(1), DeviceId::rocm(0));
        constexpr int n = 7;
        constexpr int k = 13;
        std::array<std::uint16_t, n * k> bytes{};
        const ContiguousFloatingPointWeightDescriptor source{
            .data = bytes.data(),
            .type = TensorType::BF16,
            .n = n,
            .k = k,
            .bytes = sizeof(bytes),
        };
        const auto manifest =
            makeMoEOverlayRemoteFloatingProjectionManifest(
                identity,
                source,
                /*maximum_chunk_bytes=*/37u);
        ASSERT_TRUE(manifest.valid());
        EXPECT_TRUE(manifest.carriesFloatingBytes());
        EXPECT_EQ(manifest.format_kind, ExpertWeightFormatKind::BF16);
        EXPECT_EQ(manifest.N_padded, n);
        EXPECT_EQ(manifest.blocks_per_row, 0);
        EXPECT_EQ(
            manifest.region_bytes,
            (std::array<std::uint64_t, 4>{sizeof(bytes), 0u, 0u, 0u}));

        /*
         * Floating scalar bytes are representation-compatible across tiers.
         * Exercise every CPU/GPU endpoint shape so the wire contract cannot
         * regress to assuming that both ranks own accelerators.
         */
        const std::array<std::pair<DeviceId, DeviceId>, 3> endpoint_pairs{
            std::pair{DeviceId::cpu(), DeviceId::cpu()},
            std::pair{DeviceId::cpu(), DeviceId::cuda(0)},
            std::pair{DeviceId::rocm(0), DeviceId::cpu()},
        };
        for (const auto &[source_device, destination_device] : endpoint_pairs)
        {
            const auto pair_manifest =
                makeMoEOverlayRemoteFloatingProjectionManifest(
                    remoteProjectionIdentity(
                        fixture.transaction,
                        source_device,
                        destination_device),
                    source,
                    /*maximum_chunk_bytes=*/37u);
            EXPECT_TRUE(pair_manifest.valid())
                << source_device.to_string() << " -> "
                << destination_device.to_string();
            EXPECT_TRUE(pair_manifest.carriesFloatingBytes());
            EXPECT_EQ(pair_manifest.region_bytes, manifest.region_bytes);
        }

        std::array<
            std::uint8_t,
            MoEOverlayRemoteProjectionManifest::kWireBytes> packet{};
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayRemoteProjectionManifest(
            manifest, packet, &error)) << error;
        MoEOverlayRemoteProjectionManifest decoded;
        ASSERT_TRUE(decodeMoEOverlayRemoteProjectionManifest(
            packet, &decoded, &error)) << error;
        EXPECT_EQ(decoded, manifest);

        auto wrong_precision = manifest;
        wrong_precision.format_kind = ExpertWeightFormatKind::FP32;
        wrong_precision.manifest_hash = wrong_precision.computedHash();
        EXPECT_FALSE(wrong_precision.valid(&error));
        EXPECT_NE(error.find("precision"), std::string::npos);

        auto quantized_masquerade = manifest;
        quantized_masquerade.packing =
            MoEOverlayRemoteProjectionPacking::GpuSeparatedNativeVnni;
        quantized_masquerade.manifest_hash =
            quantized_masquerade.computedHash();
        EXPECT_FALSE(quantized_masquerade.valid(&error));
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

    TEST(Test__MoEOverlayDistributedResidencyProtocol, TransactionCooccurrenceBelongsToWindowIdentity)
    {
        auto admission = demandAdmission();
        const auto grouped = demandWindow(admission, {0, 1, 2, 3});
        const auto mixed = demandWindow(admission, {0, 3, 1, 2});
        ASSERT_EQ(grouped.expert_counts, mixed.expert_counts);
        ASSERT_EQ(grouped.source_expert_counts, mixed.source_expert_counts);
        ASSERT_EQ(grouped.generation, mixed.generation);
        EXPECT_NE(fingerprintDecodeExpertHistogramWindow(grouped), fingerprintDecodeExpertHistogramWindow(mixed));
        auto missing = grouped;
        missing.transaction_demand.reset();
        EXPECT_NE(fingerprintDecodeExpertHistogramWindow(grouped), fingerprintDecodeExpertHistogramWindow(missing));
    }

    TEST(Test__MoEOverlayDistributedResidencyProtocol, TransactionWireRoundTripOwnsCompactAuthenticatedPayload)
    {
        auto source_admission = demandAdmission();
        auto receiver = demandAdmission();
        const auto source = demandWindow(source_admission, {0, 1, 2, 3});
        std::vector<uint8_t> packet(moeOverlayDistributedHistogramWireBytes(source));
        EXPECT_LT(packet.size(), moeOverlayDistributedHistogramWireBytes(1, 4, &receiver.capacity));
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedHistogramWindow(source, packet, &error)) << error;
        DecodeExpertHistogramWindow decoded;
        ASSERT_TRUE(decodeMoEOverlayDistributedHistogramWindow(packet, 1, 4, &decoded, &error, &receiver)) << error;
        ASSERT_TRUE(decoded.valid());
        EXPECT_EQ(fingerprintDecodeExpertHistogramWindow(source), fingerprintDecodeExpertHistogramWindow(decoded));
        ASSERT_EQ(decoded.transaction_demand->layerTransactions(0).size(), 2u);
        EXPECT_EQ(decoded.transaction_demand->routes(0, 1).expert_ids[0], 2);
        EXPECT_EQ(decoded.transaction_demand->topK(), 2u);
        EXPECT_EQ(decoded.transaction_demand->tokenBoundaryLayer(), 0);
        const auto bytes = decoded.transaction_demand->allocationBytes();
        EXPECT_EQ(receiver.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                  PhysicalMemoryMaterializationKind::NewAllocation), bytes);
        std::fill(packet.begin(), packet.end(), 0xff); // Reusing MPI's inbox cannot mutate a live proposal.
        EXPECT_EQ(decoded.transaction_demand->routes(0, 0).expert_ids[0], 0);
        decoded = {};
        EXPECT_EQ(receiver.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                  PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    TEST(Test__MoEOverlayDistributedResidencyProtocol, TransactionWireRejectsSameMarginalRouteTamperAndMissingEvidence)
    {
        auto admission = demandAdmission();
        const auto source = demandWindow(admission, {0, 1, 2, 3});
        std::vector<uint8_t> packet(moeOverlayDistributedHistogramWireBytes(source));
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedHistogramWindow(source, packet, &error));
        // Swap complete int32 IDs across the two batches. Marginals remain exact,
        // but the alternate co-occurrence has the opposite movement payoff.
        for (size_t byte = 0; byte < 4; ++byte)
            std::swap(packet[packet.size() - 12 + byte], packet[packet.size() - 4 + byte]);
        DecodeExpertHistogramWindow decoded;
        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(packet, 1, 4, &decoded, &error, &admission));
        EXPECT_FALSE(decoded.valid());
        EXPECT_NE(error.find("authentication"), std::string::npos);
        ASSERT_TRUE(encodeMoEOverlayDistributedHistogramWindow(source, packet, &error));
        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(packet, 1, 4, &decoded, &error));
        auto missing = source;
        missing.transaction_demand.reset();
        packet.resize(moeOverlayDistributedHistogramWireBytes(missing));
        ASSERT_TRUE(encodeMoEOverlayDistributedHistogramWindow(missing, packet, &error));
        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(packet, 1, 4, &decoded, &error, &admission));
        EXPECT_NE(error.find("missing"), std::string::npos);
    }

    TEST(Test__MoEOverlayDistributedResidencyProtocol, TransactionWireRejectsEveryTruncationAndUnadmittedShape)
    {
        auto admission = demandAdmission();
        const auto source = demandWindow(admission, {0, 1, 2, 3});
        std::vector<uint8_t> packet(moeOverlayDistributedHistogramWireBytes(source));
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedHistogramWindow(source, packet, &error));
        DecodeExpertHistogramWindow decoded;
        for (size_t bytes = 0; bytes < packet.size(); ++bytes)
        {
            EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(std::span(packet).first(bytes),
                         1, 4, &decoded, &error, &admission)) << bytes;
            EXPECT_FALSE(decoded.valid());
        }
        auto too_small = admission;
        too_small.capacity = {1, 1, 2};
        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(packet, 1, 4, &decoded, &error, &too_small));
        auto unowned = admission;
        unowned.memory.reset();
        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(packet, 1, 4, &decoded, &error, &unowned));
        // A hostile declared length must be checked by subtraction, not overflow.
        std::fill_n(packet.begin() + 64, 8, uint8_t{0xff});
        EXPECT_FALSE(decodeMoEOverlayDistributedHistogramWindow(packet, 1, 4, &decoded, &error, &admission));
    }

    TEST(Test__MoEOverlayDistributedResidencyProtocol, TransactionProposalRetainsBatchAndPlanAuthentication)
    {
        auto admission = demandAdmission();
        auto receiver = demandAdmission();
        auto window = std::make_shared<const DecodeExpertHistogramWindow>(demandWindow(admission, {0, 1, 2, 3}));
        MoEOverlayDistributedResidencyProposal proposal{
            .plan = {.expected_epoch = 1, .num_layers = 1, .num_experts = 4,
                .histogram_window = window,
                .entries = std::vector<MoEOverlayAuthoritativeResidencyEntry>(4,
                    {.candidate_tier_idx = 0, .candidate_owner_participant = 0})},
            .execution_fingerprint = {.low = 1, .high = 2},
            .policy_fingerprint = {.low = 3, .high = 4},
        };
        std::vector<uint8_t> packet(moeOverlayDistributedResidencyProposalWireBytes(proposal));
        EXPECT_LT(packet.size(), moeOverlayDistributedResidencyProposalWireBytes(1, 4, &receiver.capacity));
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedResidencyProposal(proposal, packet, &error)) << error;
        MoEOverlayDistributedResidencyProposal decoded;
        ASSERT_TRUE(decodeMoEOverlayDistributedResidencyProposal(packet, 1, 4, &decoded, &error, &receiver)) << error;
        ASSERT_NE(decoded.plan.histogram_window->transaction_demand, nullptr);
        EXPECT_EQ(fingerprintDecodeExpertHistogramWindow(*decoded.plan.histogram_window),
                  fingerprintDecodeExpertHistogramWindow(*window));
        EXPECT_EQ(decoded.plan.entries.back().candidate_owner_participant, 0);
        auto alternative = proposal;
        alternative.plan.histogram_window = std::make_shared<const DecodeExpertHistogramWindow>(
            demandWindow(admission, {0, 3, 1, 2}));
        std::vector<uint8_t> alternative_packet(packet.size());
        ASSERT_TRUE(encodeMoEOverlayDistributedResidencyProposal(alternative, alternative_packet, &error));
        // This replacement is internally authenticated with identical counts.
        // The containing plan must still reject its different batch identity.
        const auto header = MoEOverlayDistributedResidencyProposalHeader::kWireBytes;
        const auto payload = moeOverlayDistributedHistogramWireBytes(*window);
        std::copy_n(alternative_packet.begin() + header, payload, packet.begin() + header);
        EXPECT_FALSE(decodeMoEOverlayDistributedResidencyProposal(packet, 1, 4, &decoded, &error, &receiver));
        EXPECT_FALSE(decoded.valid());
        EXPECT_NE(error.find("plan authentication"), std::string::npos);
        EXPECT_EQ(receiver.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                  PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    TEST(Test__MoEOverlayDistributedResidencyProtocol, TransactionPayloadRejectsMalformedDescriptorsBeforePublication)
    {
        auto admission = demandAdmission();
        auto receiver = demandAdmission();
        const auto source = demandWindow(admission, {0, 1, 2, 3});
        std::vector<uint8_t> packet(source.transaction_demand->wireBytes());
        source.transaction_demand->encodeWire(packet);
        auto envelope = source;
        envelope.transaction_demand.reset();
        // Descriptor zero starts after the 8-byte model and 16-byte layer headers.
        for (uint32_t malformed : {0u, 5u, std::numeric_limits<uint32_t>::max()})
        {
            auto corrupted = packet;
            size_t offset = 24;
            moe_overlay_wire::writeLittleEndian(corrupted, offset, malformed);
            EXPECT_THROW((void)DecodeExpertTransactionWindow::decodeWire(receiver, envelope, corrupted),
                         std::invalid_argument);
            EXPECT_EQ(receiver.memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                      PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
        // Every descriptor still fits; the second batch nevertheless begins
        // after this smaller receiver's sample would already have closed.
        auto smaller = receiver;
        smaller.capacity = {1, 4, 2};
        EXPECT_THROW((void)DecodeExpertTransactionWindow::decodeWire(smaller, envelope, packet),
                     std::invalid_argument);
    }

    TEST(Test__MoEOverlayDistributedResidencyProtocol, ScalarCodecRejectsOutOfBoundsWithoutAdvancingCursor)
    {
        std::array<uint8_t, 4> bytes{};
        size_t offset = 0;
        moe_overlay_wire::writeLittleEndian(bytes, offset, int32_t{-2});
        EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0xfe, 0xff, 0xff, 0xff}));
        EXPECT_THROW((void)moe_overlay_wire::readLittleEndian<uint32_t>(bytes, offset), std::out_of_range);
        EXPECT_EQ(offset, 4u);
        EXPECT_THROW(moe_overlay_wire::writeLittleEndian(bytes, offset, uint8_t{1}), std::out_of_range);
        EXPECT_EQ(offset, 4u);
        offset = 0;
        EXPECT_EQ(moe_overlay_wire::readLittleEndian<int32_t>(bytes, offset), -2);
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

        auto incoherent = rank_zero.transaction;
        incoherent.migrations.front().activation_count += 1;
        EXPECT_FALSE(incoherent.valid())
            << "migration execution metadata cannot name different evidence from its transaction";

        auto altered = rank_zero.transaction;
        auto altered_window = std::make_shared<DecodeExpertHistogramWindow>(
            *rank_zero.transaction.histogram_window);
        const auto &altered_migration = altered.migrations.front();
        const std::size_t altered_index =
            static_cast<std::size_t>(altered_migration.layer_idx) *
                static_cast<std::size_t>(altered_window->num_experts) +
            static_cast<std::size_t>(altered_migration.expert_id);
        ++altered.migrations.front().activation_count;
        ++altered_window->expert_counts[altered_index];
        /* Source zero occupies the first dense layer/expert plane. */
        ++altered_window->source_expert_counts[altered_index];
        altered.histogram_window = std::move(altered_window);
        ASSERT_TRUE(altered.valid());
        EXPECT_NE(
            fingerprintMoEOverlayResidencyExecutionPlan(altered),
            identity_zero.execution_fingerprint);
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
            .forecast_fingerprint = fingerprintDecodeExpertHistogramWindow(
                *measured.histogram_window),
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
        EXPECT_EQ(
            fingerprintMoEOverlayResidencyExecutionPlan(measured),
            fingerprintMoEOverlayResidencyExecutionPlan(
                fixture.transaction));
        EXPECT_EQ(
            makeMoEOverlayDistributedResidencyWaveIdentity(measured),
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction))
            << "Root policy evidence is audited separately from the plan every rank executes";

        auto different_profile = measured;
        different_profile.economy.migration_profile_identity =
            "movement-profile-b";
        ASSERT_TRUE(different_profile.valid());
        EXPECT_NE(
            fingerprintMoEOverlayResidencyTransaction(different_profile),
            measured_fingerprint);

        auto different_forecast = measured;
        different_forecast.economy.forecast_fingerprint ^= 0x8000000000000000ULL;
        ASSERT_TRUE(different_forecast.valid());
        EXPECT_NE(fingerprintMoEOverlayResidencyTransaction(different_forecast),
                  measured_fingerprint);
        EXPECT_EQ(fingerprintMoEOverlayResidencyExecutionPlan(different_forecast),
                  fingerprintMoEOverlayResidencyExecutionPlan(measured));
        auto missing_forecast = measured;
        missing_forecast.economy.forecast_fingerprint = 0;
        EXPECT_FALSE(missing_forecast.valid());

        auto different_policy = measured;
        different_policy.economy.payoff_horizon_tokens = 9;
        ASSERT_TRUE(different_policy.valid());
        EXPECT_NE(
            fingerprintMoEOverlayResidencyTransaction(different_policy),
            measured_fingerprint);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        CoordinatorProposalRoundTripsAndPeerAdoptsExactExecutionPlan)
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
        const auto root_proposal =
            makeMoEOverlayDistributedResidencyProposal(
                coordinator.exportAuthoritativeResidencyPlan(
                    coordinator_transaction),
                coordinator_transaction);
        ASSERT_TRUE(root_proposal.valid());

        const std::size_t packet_bytes =
            moeOverlayDistributedResidencyProposalWireBytes(
                modelMetadata().num_layers,
                modelMetadata().num_experts);
        std::vector<std::uint8_t> packet(packet_bytes);
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedResidencyProposal(
            root_proposal, packet, &error))
            << error;
        /* Magic 0x504f4f4d is emitted as canonical little-endian "MOOP". */
        EXPECT_EQ(packet[0], static_cast<std::uint8_t>('M'));
        EXPECT_EQ(packet[1], static_cast<std::uint8_t>('O'));
        EXPECT_EQ(packet[2], static_cast<std::uint8_t>('O'));
        EXPECT_EQ(packet[3], static_cast<std::uint8_t>('P'));

        MoEOverlayDistributedResidencyProposal decoded_proposal;
        ASSERT_TRUE(decodeMoEOverlayDistributedResidencyProposal(
            packet,
            modelMetadata().num_layers,
            modelMetadata().num_experts,
            &decoded_proposal,
            &error))
            << error;
        ASSERT_TRUE(decoded_proposal.valid());
        EXPECT_EQ(
            decoded_proposal.plan.entries,
            root_proposal.plan.entries);
        EXPECT_EQ(
            fingerprintDecodeExpertHistogramWindow(
                *decoded_proposal.plan.histogram_window),
            fingerprintDecodeExpertHistogramWindow(*published_window));
        EXPECT_EQ(
            decoded_proposal.execution_fingerprint,
            root_proposal.execution_fingerprint);
        EXPECT_EQ(
            decoded_proposal.policy_fingerprint,
            root_proposal.policy_fingerprint);

        const auto peer_transaction =
            peer.adoptAuthoritativeResidencyPlan(decoded_proposal.plan);

        EXPECT_EQ(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                coordinator_transaction),
            makeMoEOverlayDistributedResidencyWaveIdentity(
                peer_transaction));
        EXPECT_EQ(
            fingerprintMoEOverlayResidencyExecutionPlan(peer_transaction),
            decoded_proposal.execution_fingerprint);
        EXPECT_EQ(
            decoded_proposal.policy_fingerprint,
            fingerprintMoEOverlayResidencyTransaction(
                coordinator_transaction));
        EXPECT_EQ(
            coordinator_histogram->windowGeneration(),
            published_window->generation + 1);
        EXPECT_EQ(
            peer_histogram->windowGeneration(),
            published_window->generation)
            << "A peer must adopt root policy without rotating its local partial window";

        auto corrupted = packet;
        corrupted.back() ^= 0x80u;
        EXPECT_FALSE(decodeMoEOverlayDistributedResidencyProposal(
            corrupted,
            modelMetadata().num_layers,
            modelMetadata().num_experts,
            &decoded_proposal,
            &error));
        EXPECT_NE(error.find("authentication"), std::string::npos);

        EXPECT_FALSE(decodeMoEOverlayDistributedResidencyProposal(
            packet,
            modelMetadata().num_layers,
            modelMetadata().num_experts - 1,
            &decoded_proposal,
            &error));
        EXPECT_NE(error.find("size"), std::string::npos);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        RequiresUnanimousPrepareAndSelectorPublicationBeforeAuthorityPublish)
    {
        auto fixture = makeTransaction();
        const auto identity =
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction);
        auto protocols = makeProtocols(identity);

        for (auto &protocol : protocols)
            EXPECT_THROW(
                protocol->markAuthorityPublished(),
                std::logic_error);

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
            EXPECT_THROW(
                protocol->markAuthorityPublished(),
                std::logic_error);
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
                    AwaitingLocalPrepare);
            EXPECT_THROW(
                protocol->markAuthorityPublished(),
                std::logic_error);
        }

        const auto prepare_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_TRUE(protocol->acceptConsensus(prepare_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    AwaitingLocalPublication);
            EXPECT_THROW(
                protocol->markAuthorityPublished(),
                std::logic_error);
        }

        const auto publication_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_TRUE(protocol->acceptConsensus(publication_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    ReadyForAuthorityPublication);
            protocol->markAuthorityPublished();
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Published);
            EXPECT_FALSE(protocol->abort());
        }

        const auto retirement_admission_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            EXPECT_THROW(protocol->markRetired(), std::logic_error);
            std::string error;
            EXPECT_TRUE(
                protocol->acceptConsensus(retirement_admission_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    ReadyToCloseRetirementAdmission);
            EXPECT_THROW(protocol->markRetired(), std::logic_error);
            protocol->markRetirementAdmissionClosed();
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    RetirementAdmissionClosed);
            EXPECT_FALSE(protocol->abort());
        }

        const auto retirement_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
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
        RetirementWaitingGenerationsRetryBothBarriersInOrder)
    {
        auto fixture = makeTransaction();
        auto protocols = makeProtocols(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction));
        advanceProtocolsToPublished(protocols);

        std::vector<MoEOverlayDistributedResidencyVote> waiting_votes;
        waiting_votes.push_back(protocols[0]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Waiting));
        waiting_votes.push_back(protocols[1]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        waiting_votes.push_back(protocols[2]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_FALSE(protocol->acceptConsensus(waiting_votes, &error));
            EXPECT_TRUE(error.empty());
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Published);
            EXPECT_THROW(protocol->markRetired(), std::logic_error);
        }

        const auto ready_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            std::string error;
            ASSERT_TRUE(protocol->acceptConsensus(ready_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    ReadyToCloseRetirementAdmission);
            protocol->markRetirementAdmissionClosed();
        }

        std::vector<MoEOverlayDistributedResidencyVote> final_waiting_votes;
        final_waiting_votes.push_back(protocols[0]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        final_waiting_votes.push_back(protocols[1]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Waiting));
        final_waiting_votes.push_back(protocols[2]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        for (auto &protocol : protocols)
        {
            std::string error;
            EXPECT_FALSE(
                protocol->acceptConsensus(final_waiting_votes, &error));
            EXPECT_TRUE(error.empty());
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::
                    RetirementAdmissionClosed);
            EXPECT_THROW(protocol->markRetired(), std::logic_error);
        }

        const auto final_ready_votes = readyVotes(protocols);
        for (auto &protocol : protocols)
        {
            std::string error;
            ASSERT_TRUE(protocol->acceptConsensus(final_ready_votes, &error))
                << error;
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::ReadyToRetire);
            protocol->markRetired();
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
            EXPECT_THROW(
                protocol->markAuthorityPublished(),
                std::logic_error);
            EXPECT_TRUE(protocol->abort());
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Aborted);
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        PreparationFailureNeverBecomesPublishable)
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

        std::vector<MoEOverlayDistributedResidencyVote> prepare_votes;
        prepare_votes.push_back(protocols[0]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        prepare_votes.push_back(protocols[1]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready));
        prepare_votes.push_back(protocols[2]->makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Failed,
            88,
            "rank two inactive bank install failed"));

        for (auto &protocol : protocols)
        {
            EXPECT_FALSE(protocol->acceptConsensus(prepare_votes));
            EXPECT_EQ(
                protocol->state(),
                MoEOverlayDistributedResidencyProtocolState::Failed);
            EXPECT_THROW(
                protocol->markAuthorityPublished(),
                std::logic_error);
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyProtocol,
        DivergentTransactionIdentityIsFatalBeforeAnyPreparation)
    {
        auto fixture = makeTransaction();
        auto protocols = makeProtocols(
            makeMoEOverlayDistributedResidencyWaveIdentity(
                fixture.transaction));
        auto votes = readyVotes(protocols);

        /* Keep the foreign identity structurally valid while changing its digest. */
        votes[2].identity.execution_fingerprint.low ^= 0x100u;
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
