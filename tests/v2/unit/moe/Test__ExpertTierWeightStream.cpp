/**
 * @file Test__ExpertTierWeightStream.cpp
 * @brief Device-free proof of the heterogeneous expert-weight stream ABI.
 *
 * Exercise source provenance, bounded receive/publication state and exact
 * layout transformations independently of backend launchers. Native grid
 * scales/minima must survive conversion, not merely a lossy round trip.
 */

#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/ExpertTierGpuBlobTransferLane.h"
#include "execution/moe/CpuExpertSlotPool.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace llaminar2
{
    namespace
    {
        TEST(ExpertTierSourceReadiness, DistinguishesProducerAndPublishedBank)
        {
            int event_storage = 0;
            const auto producer = ExpertTierSourceReadiness::producerEvent(
                &event_storage);
            EXPECT_EQ(
                producer.kind(),
                ExpertTierSourceReadiness::Kind::ProducerEvent);
            EXPECT_TRUE(producer.requiresProducerWait());
            EXPECT_EQ(producer.event(), &event_storage);
            EXPECT_EQ(producer.epoch(), 0u);

            const auto published =
                ExpertTierSourceReadiness::publishedResidencyBank(17);
            EXPECT_EQ(
                published.kind(),
                ExpertTierSourceReadiness::Kind::PublishedQuiescentResidencyBank);
            EXPECT_FALSE(published.requiresProducerWait());
            EXPECT_EQ(published.event(), nullptr);
            EXPECT_EQ(published.epoch(), 17u);

            EXPECT_THROW(
                (void)ExpertTierSourceReadiness::producerEvent(nullptr),
                std::invalid_argument);
            EXPECT_THROW(
                (void)ExpertTierSourceReadiness::publishedResidencyBank(0),
                std::invalid_argument);
        }

        HostGpuExpertPackedProjection makeProjection(
            uint8_t codebook,
            uint8_t payload_bytes,
            bool asymmetric,
            bool superblock)
        {
            HostGpuExpertPackedProjection projection;
            projection.N = 70;
            projection.K = 96;
            projection.blocks_per_row = 3;
            // Codebook 23 is a normalized destination for a Q5_1-derived
            // asymmetric expanded CPU representation, not a GGUF source id.
            projection.source_codebook_id =
                codebook == kNativeVnniExpandedInt8MinCodebook
                    ? static_cast<uint8_t>(7)
                    : codebook;
            projection.codebook_id = codebook;
            projection.payload_bytes_per_block = payload_bytes;
            projection.is_asymmetric = asymmetric;
            projection.is_superblock = superblock;
            const size_t blocks =
                static_cast<size_t>(projection.N) *
                projection.blocks_per_row;
            projection.payload.resize(blocks * payload_bytes);
            projection.scales.resize(blocks);
            if (asymmetric)
                projection.mins.resize(blocks);

            for (size_t block = 0; block < blocks; ++block)
            {
                for (size_t byte = 0; byte < payload_bytes; ++byte)
                {
                    uint8_t value = static_cast<uint8_t>(
                        (block * 37u + byte * 19u + 11u) & 0xffu);
                    if (codebook == 19 ||
                        codebook == kNativeVnniExpandedInt8MinCodebook)
                    {
                        value = static_cast<uint8_t>(static_cast<int8_t>(
                            static_cast<int>((block * 13u + byte * 7u) % 127u) - 63));
                    }
                    projection.payload[block * payload_bytes + byte] = value;
                }
                projection.scales[block] = static_cast<uint16_t>(
                    0x2400u + (block % 0x0800u));
                if (asymmetric)
                {
                    projection.mins[block] = static_cast<uint16_t>(
                        0xa000u + (block % 0x0800u));
                }
            }
            return projection;
        }

        TEST(ExpertTierWeightStream, SingleScaleNativeGridsExpandWithoutRequantization)
        {
            for (const uint8_t codebook : {11, 12, 15, 16})
            {
                SCOPED_TRACE(static_cast<int>(codebook));
                const auto *format = native_vnni_formats::forSourceIdentity(codebook, true);
                ASSERT_NE(format, nullptr);
                auto original = makeProjection(codebook, format->payload_bytes,
                                               format->is_asymmetric, true);
                // Metadata identity includes signed zero and tiny scales:
                // no absmax threshold may silently zero out a native block.
                const std::array<uint16_t, 6> scale_bits{0, 0x8000, 1, 0x03ff, 0x0400, 0x8400};
                for (size_t index = 0; index < original.scales.size(); ++index)
                    original.scales[index] = scale_bits[index % scale_bits.size()];
                cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
                std::string error;
                ASSERT_TRUE(gpuToCpuExpertPackedReference(original, cpu, &error)) << error;
                HostGpuExpertPackedProjection expanded;
                ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, expanded, &error)) << error;
                EXPECT_EQ(expanded.source_codebook_id, codebook);
                EXPECT_EQ(expanded.scales, original.scales);
                EXPECT_EQ(expanded.mins, original.mins);
                ASSERT_EQ(expanded.payload_bytes_per_block, 32);
                for (size_t index = 0; index < original.scales.size(); ++index)
                {
                    std::array<int8_t, 32> values{};
                    cpu::native_vnni::decode_native_block(codebook,
                        original.payload.data() + index * format->payload_bytes, values.data());
                    EXPECT_EQ(std::memcmp(values.data(), expanded.payload.data() + index * 32,
                                          values.size()), 0);
                }
                cpu::native_vnni::CPUNativeVNNIPackedWeights returned;
                ASSERT_TRUE(gpuToCpuExpertPackedReference(expanded, returned, &error)) << error;
                ASSERT_EQ(returned.native_interleaved.size(), cpu.native_interleaved.size());
                EXPECT_EQ(std::memcmp(returned.native_interleaved.data(),
                    cpu.native_interleaved.data(), cpu.native_interleaved.size()), 0);
            }
        }

        /** @brief Device-free GEMM exporting a caller-owned GPU descriptor. */
        class ExportingGpuGemm final : public ITensorGemm
        {
        public:
            ExportingGpuGemm(
                DeviceNativeVNNIMatrixDesc descriptor,
                NativeVnniSourceIdentity source_identity)
                : descriptor_(descriptor),
                  source_identity_(source_identity)
            {
            }

            bool supports_device(int) const override { return true; }

            bool multiply_tensor(
                const TensorBase *,
                TensorBase *,
                int,
                int,
                int,
                bool,
                float,
                float,
                const TensorBase *,
                const IMPIContext *,
                int,
                DeviceWorkspaceManager *,
                int) override
            {
                return false;
            }

            bool exportNativeVNNIMatrixDesc(
                DeviceNativeVNNIMatrixDesc &out) override
            {
                out = descriptor_;
                return descriptor_.valid();
            }

            bool exportNativeVNNISourceIdentity(
                NativeVnniSourceIdentity &out) const override
            {
                out = source_identity_;
                return source_identity_.present;
            }

        private:
            DeviceNativeVNNIMatrixDesc descriptor_;
            NativeVnniSourceIdentity source_identity_;
        };

        void expectGpuEqual(
            const HostGpuExpertPackedProjection &expected,
            const HostGpuExpertPackedProjection &actual)
        {
            EXPECT_EQ(actual.N, expected.N);
            EXPECT_EQ(actual.K, expected.K);
            EXPECT_EQ(actual.blocks_per_row, expected.blocks_per_row);
            EXPECT_EQ(
                actual.source_codebook_id,
                expected.source_codebook_id);
            EXPECT_EQ(actual.codebook_id, expected.codebook_id);
            EXPECT_EQ(
                actual.payload_bytes_per_block,
                expected.payload_bytes_per_block);
            EXPECT_EQ(actual.is_asymmetric, expected.is_asymmetric);
            EXPECT_EQ(actual.is_superblock, expected.is_superblock);
            EXPECT_EQ(actual.has_emins, expected.has_emins);
            EXPECT_EQ(actual.payload, expected.payload);
            EXPECT_EQ(actual.scales, expected.scales);
            EXPECT_EQ(actual.mins, expected.mins);
            EXPECT_EQ(actual.emins, expected.emins);
        }

        class ExpertTierWeightEncodingTest
            : public ::testing::TestWithParam<HostGpuExpertPackedProjection>
        {
        };

        TEST_P(ExpertTierWeightEncodingTest, CpuWireRoundTripIsByteExact)
        {
            const auto source = GetParam();
            ASSERT_TRUE(source.valid());

            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(source, cpu, &error))
                << error;
            EXPECT_EQ(cpu.N, source.N);
            EXPECT_EQ(cpu.K, source.K);
            EXPECT_EQ(cpu.N_padded, 128);
            EXPECT_EQ(cpu.blocks_per_row, 3);
            EXPECT_FALSE(cpu.native_interleaved.empty());

            HostGpuExpertPackedProjection returned;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, returned, &error))
                << error;
            expectGpuEqual(source, returned);
        }

        INSTANTIATE_TEST_SUITE_P(
            EveryCpuPreparedEncoding,
            ExpertTierWeightEncodingTest,
            ::testing::Values(
                makeProjection(0, 16, false, false),
                makeProjection(4, 16, false, true),
                makeProjection(5, 16, true, true),
                makeProjection(8, 24, true, true),
                makeProjection(19, 32, false, false),
                makeProjection(
                    kNativeVnniExpandedInt8MinCodebook,
                    32,
                    true,
                    false)));

        TEST(ExpertTierWeightStream, OrderedChunksWriteDirectlyToFinalBytes)
        {
            const auto source = makeProjection(5, 16, true, true);
            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(source, cpu, &error))
                << error;

            const NativeVnniFormatInfo *source_format =
                native_vnni_formats::forSourceIdentity(
                    source.source_codebook_id, source.is_superblock);
            ASSERT_NE(source_format, nullptr);
            const auto manifest =
                makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format,
                source.N,
                source.K,
                41,
                7,
                12,
                ExpertTierWeightProjection::Up,
                2);
            ASSERT_TRUE(manifest.valid(&error)) << error;
            ASSERT_EQ(manifest.unit_count, 6u);
            ASSERT_EQ(
                manifest.total_stream_bytes,
                cpu.native_interleaved.size());

            std::vector<uint8_t> final_bytes(
                static_cast<size_t>(manifest.total_stream_bytes), 0xcc);
            ExpertTierWeightStreamReceiver receiver(manifest, final_bytes);

            for (uint32_t sequence = 0; sequence < 3; ++sequence)
            {
                const uint32_t first_unit = sequence * 2;
                const size_t offset =
                    static_cast<size_t>(first_unit) *
                    manifest.cpu_block_stride;
                const size_t byte_count =
                    2u * static_cast<size_t>(manifest.cpu_block_stride);
                const std::span<const uint8_t> payload(
                    cpu.native_interleaved.data() + offset,
                    byte_count);
                ExpertTierWeightStreamChunkHeader header;
                header.manifest_hash = manifest.manifest_hash;
                header.payload_hash = expertTierWeightBytesHash(payload);
                header.sequence = sequence;
                header.first_unit = first_unit;
                header.unit_count = 2;
                header.payload_bytes = static_cast<uint32_t>(byte_count);
                header.final_chunk = sequence == 2 ? 1u : 0u;
                ASSERT_TRUE(receiver.accept(header, payload, &error)) << error;
            }

            EXPECT_TRUE(receiver.complete());
            EXPECT_EQ(receiver.receivedUnits(), manifest.unit_count);
            EXPECT_TRUE(std::equal(
                final_bytes.begin(),
                final_bytes.end(),
                cpu.native_interleaved.begin(),
                cpu.native_interleaved.end()));
        }

        TEST(ExpertTierWeightStream, RejectsCorruptionAndOrderingWithoutMutation)
        {
            const auto source = makeProjection(19, 32, false, false);
            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(source, cpu));
            const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu,
                9,
                3,
                5,
                ExpertTierWeightProjection::Down,
                1);
            std::vector<uint8_t> final_bytes(
                static_cast<size_t>(manifest.total_stream_bytes), 0x5a);
            ExpertTierWeightStreamReceiver receiver(manifest, final_bytes);

            const std::span<const uint8_t> first(
                cpu.native_interleaved.data(),
                manifest.cpu_block_stride);
            ExpertTierWeightStreamChunkHeader header;
            header.manifest_hash = manifest.manifest_hash;
            header.payload_hash = expertTierWeightBytesHash(first);
            header.sequence = 1;
            header.first_unit = 0;
            header.unit_count = 1;
            header.payload_bytes = manifest.cpu_block_stride;

            std::string error;
            EXPECT_FALSE(receiver.accept(header, first, &error));
            EXPECT_EQ(receiver.receivedUnits(), 0u);
            EXPECT_TRUE(std::all_of(
                final_bytes.begin(), final_bytes.end(),
                [](uint8_t value) { return value == 0x5a; }));

            header.sequence = 0;
            header.payload_hash ^= 0x1u;
            EXPECT_FALSE(receiver.accept(header, first, &error));
            EXPECT_EQ(receiver.receivedUnits(), 0u);

            header.payload_hash = expertTierWeightBytesHash(first);
            ASSERT_TRUE(receiver.accept(header, first, &error)) << error;
            EXPECT_EQ(receiver.receivedUnits(), 1u);
            EXPECT_FALSE(receiver.accept(header, first, &error));
            EXPECT_EQ(receiver.receivedUnits(), 1u);
        }

        TEST(ExpertTierWeightStream, ManifestHashCoversEveryLayoutField)
        {
            const auto source = makeProjection(8, 24, true, true);
            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(source, cpu));
            const NativeVnniFormatInfo *source_format =
                native_vnni_formats::forSourceIdentity(
                    source.source_codebook_id, source.is_superblock);
            ASSERT_NE(source_format, nullptr);
            auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format,
                source.N,
                source.K,
                2,
                1,
                4,
                ExpertTierWeightProjection::Gate,
                3);
            EXPECT_TRUE(manifest.valid());
            ++manifest.cpu_block_stride;
            EXPECT_FALSE(manifest.valid());
            --manifest.cpu_block_stride;
            EXPECT_TRUE(manifest.valid());
            ++manifest.expert_id;
            EXPECT_FALSE(manifest.valid());
        }

        TEST(ExpertTierWeightStream, EveryCatalogedSourceFormatHasGpuToCpuContract)
        {
            // This is the totality guard: adding a supported quantized source
            // without a direction-complete migration layout fails this unit.
            ASSERT_EQ(native_vnni_formats::kAllSourceFormats.size(), 21u);
            for (const NativeVnniSourceFormat &entry :
                 native_vnni_formats::kAllSourceFormats)
            {
                ASSERT_NE(entry.metadata, nullptr) << entry.quant_type;
                const auto manifest =
                    makeGpuToCpuExpertTierWeightStreamManifest(
                        *entry.metadata,
                        70,
                        256,
                        101,
                        3,
                        7,
                        ExpertTierWeightProjection::Gate,
                        4);
                std::string error;
                ASSERT_TRUE(manifest.valid(&error))
                    << entry.quant_type << ": " << error;
                const ExpertTierWeightDeviceLayout layout =
                    manifest.deviceLayout();
                EXPECT_EQ(
                    layout.direction,
                    ExpertTierWeightConversionDirection::GpuToCpu)
                    << entry.quant_type;
                EXPECT_EQ(
                    layout.gpu_source_codebook_id,
                    entry.metadata->codebook_id)
                    << entry.quant_type;
                EXPECT_EQ(
                    layout.gpu_codebook_id,
                    canonicalDeviceVnniCodebookId(
                        entry.metadata->codebook_id))
                    << entry.quant_type;
            }
        }

        TEST(ExpertTierWeightStream, MultiScaleSourcesRejectLegacyLossyExpandedLayouts)
        {
            for (const auto &entry : native_vnni_formats::kAllSourceFormats)
            {
                const auto id = entry.metadata->codebook_id;
                if (id != 8 && !hasCompactMultiScaleVnniPayload(id))
                    continue;
                SCOPED_TRACE(std::string(entry.quant_type));
                auto layout = makeGpuToCpuExpertTierWeightStreamManifest(
                    *entry.metadata, 70, 256, 101, 3, 7,
                    ExpertTierWeightProjection::Gate, 4).deviceLayout();
                ASSERT_TRUE(layout.valid());
                layout.cpu_encoding = cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8;
                layout.cpu_data_stride = cpu::native_vnni::preparedDataStride(layout.cpu_encoding);
                layout.cpu_block_stride = cpu::native_vnni::preparedInterleavedBlockStride(layout.cpu_encoding, true);
                layout.gpu_codebook_id = kNativeVnniExpandedInt8MinCodebook;
                layout.gpu_payload_bytes_per_block = 32;
                layout.gpu_has_emins = 0;
                EXPECT_FALSE(layout.valid());
                layout.direction = ExpertTierWeightConversionDirection::CpuToGpu;
                EXPECT_FALSE(layout.valid());
            }
        }

        TEST(ExpertTierWeightStream, ExpandedCpuBytesRetainSuperblockSourceIdentity)
        {
            /*
             * Q8_K and Q8_0 share the normalized accelerator execution
             * codebook, so the superblock bit is the only unambiguous source
             * discriminator retained across a cold-tier round trip.
             */
            auto gpu = makeProjection(
                canonicalDeviceVnniCodebookId(
                    native_vnni_formats::Q8_K.codebook_id),
                native_vnni_formats::Q8_K.payload_bytes,
                native_vnni_formats::Q8_K.is_asymmetric,
                native_vnni_formats::Q8_K.is_superblock);
            gpu.source_codebook_id = native_vnni_formats::Q8_K.codebook_id;
            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(gpu, cpu, &error))
                << error;
            EXPECT_EQ(
                cpu.encoding,
                cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8);
            EXPECT_EQ(cpu.codebook_id, native_vnni_formats::Q8_K.codebook_id);
            EXPECT_TRUE(cpu.is_superblock);

            const auto promotion = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu,
                /*residency_epoch=*/102,
                /*layer_idx=*/3,
                /*expert_id=*/7,
                ExpertTierWeightProjection::Down,
                /*maximum_units_per_chunk=*/4);
            ASSERT_TRUE(promotion.valid(&error)) << error;
            EXPECT_EQ(
                promotion.gpu_source_codebook_id,
                native_vnni_formats::Q8_K.codebook_id);
            EXPECT_EQ(promotion.gpu_source_is_superblock, 1u);
            EXPECT_EQ(promotion.gpu_codebook_id, 19u);
        }

        std::vector<CpuExpertSlotPool::ProjectionSpec>
        cpuSlotProjectionSpecs(NativeVnniSourceIdentity source_identity)
        {
            const auto format =
                ExpertWeightFormat::nativeVnni(source_identity);
            return {
                {
                    .projection = ExpertTierWeightProjection::Gate,
                    .N = 64,
                    .K = 64,
                    .format = format,
                },
                {
                    .projection = ExpertTierWeightProjection::Up,
                    .N = 64,
                    .K = 64,
                    .format = format,
                },
                {
                    .projection = ExpertTierWeightProjection::Down,
                    .N = 64,
                    .K = 64,
                    .format = format,
                },
            };
        }

        /** @brief Build a complete raw row-major slot contract for one precision. */
        std::vector<CpuExpertSlotPool::ProjectionSpec>
        cpuFloatingSlotProjectionSpecs(TensorType type)
        {
            const auto format = ExpertWeightFormat::floating(type);
            return {
                {
                    .projection = ExpertTierWeightProjection::Gate,
                    .N = 31,
                    .K = 47,
                    .format = format,
                },
                {
                    .projection = ExpertTierWeightProjection::Up,
                    .N = 29,
                    .K = 43,
                    .format = format,
                },
                {
                    .projection = ExpertTierWeightProjection::Down,
                    .N = 37,
                    .K = 41,
                    .format = format,
                },
            };
        }

        /**
         * @brief Sum canonical physical allocation extents for one CPU slot.
         * @param specs Complete gate/up/down projection specification.
         * @return Exact bytes that CpuExpertSlotPool must materialize.
         */
        std::size_t plannedCpuSlotAllocationBytes(
            std::span<const CpuExpertSlotPool::ProjectionSpec> specs)
        {
            std::size_t total = 0u;
            for (const auto &spec : specs)
            {
                const std::size_t bytes =
                    resolveExpertPreparedProjectionMemoryGeometry(
                        spec.N, spec.K, spec.format)
                        .cpu_bytes;
                if (bytes > std::numeric_limits<std::size_t>::max() - total)
                {
                    throw std::overflow_error(
                        "CPU expert slot test geometry overflowed");
                }
                total += bytes;
            }
            return total;
        }

        TEST(
            CpuExpertSlotPool,
            SameExpertMayRetainOldAndCandidateEpochUntilEveryEngineAliasDies)
        {
            const NativeVnniSourceIdentity source_identity{
                .codebook_id = native_vnni_formats::Q4_0.codebook_id,
                .is_superblock = native_vnni_formats::Q4_0.is_superblock,
                .present = true,
            };
            auto pool = CpuExpertSlotPool::createForTest({
                .participant_id = 2,
                .layer_idx = 7,
                .capacity = 2,
                .projections = cpuSlotProjectionSpecs(source_identity),
                .memory_placement =
                    CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                .perf_device = "cpu-test",
            });

            auto old = pool->acquire(11, 4);
            ASSERT_TRUE(old.has_value());
            ASSERT_EQ(old->projections.size(), 3u);
            EXPECT_EQ(pool->usedSlots(), 1u);
            EXPECT_EQ(pool->slotFor(11, 4), old->slot_index);

            ITensorGemm *old_gate_address = old->projections[0].engine.get();
            auto retained_old_gate = old->projections[0].engine;
            for (auto &projection : old->projections)
            {
                ASSERT_NE(projection.engine, nullptr);
                ASSERT_FALSE(projection.destination_bytes.empty());
                std::fill(
                    projection.destination_bytes.begin(),
                    projection.destination_bytes.end(),
                    static_cast<std::uint8_t>(0x5a));
                NativeVnniSourceIdentity actual;
                ASSERT_TRUE(
                    projection.engine->exportNativeVNNISourceIdentity(actual));
                EXPECT_EQ(actual, source_identity);
            }

            /* The same expert needs a distinct slot while epoch 4 is callable. */
            auto candidate = pool->acquire(11, 5);
            ASSERT_TRUE(candidate.has_value());
            EXPECT_NE(candidate->slot_index, old->slot_index);
            EXPECT_EQ(pool->usedSlots(), 2u);
            EXPECT_FALSE(pool->acquire(12, 5).has_value());

            old.reset();
            EXPECT_EQ(pool->usedSlots(), 2u)
                << "One retained engine alias must pin the complete epoch slot";
            retained_old_gate.reset();
            EXPECT_EQ(pool->usedSlots(), 1u);
            EXPECT_FALSE(pool->slotFor(11, 4).has_value());

            candidate.reset();
            EXPECT_EQ(pool->usedSlots(), 0u);
            auto reused = pool->acquire(12, 6);
            ASSERT_TRUE(reused.has_value());
            EXPECT_EQ(reused->projections[0].engine.get(), old_gate_address)
                << "Pool reuse must preserve the graph-stable kernel object";
            reused.reset();
        }

        TEST(CpuExpertSlotPool, EveryCatalogedFormatBuildsStableTripletMetadata)
        {
            ASSERT_EQ(native_vnni_formats::kAllSourceFormats.size(), 21u);
            int participant_id = 0;
            for (const NativeVnniSourceFormat &entry :
                 native_vnni_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(entry.quant_type);
                ASSERT_NE(entry.metadata, nullptr);
                const NativeVnniSourceIdentity source_identity{
                    .codebook_id = entry.metadata->codebook_id,
                    .is_superblock = entry.metadata->is_superblock,
                    .present = true,
                };
                const auto projections =
                    cpuSlotProjectionSpecs(source_identity);
                const std::size_t planned_bytes =
                    plannedCpuSlotAllocationBytes(projections);
                auto pool = CpuExpertSlotPool::createForTest({
                    .participant_id = participant_id++,
                    .layer_idx = 0,
                    .capacity = 1,
                    .projections = projections,
                    .memory_placement =
                        CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                    .perf_device = "cpu-catalog-test",
                });
                EXPECT_EQ(pool->allocationBytes(), planned_bytes)
                    << "format=" << entry.quant_type;
                auto lease = pool->acquire(3, 9);
                ASSERT_TRUE(lease.has_value());
                ASSERT_EQ(lease->projections.size(), 3u);
                for (const auto &projection : lease->projections)
                {
                    NativeVnniSourceIdentity actual;
                    ASSERT_TRUE(
                        projection.engine->exportNativeVNNISourceIdentity(
                            actual));
                    EXPECT_EQ(actual, source_identity);
                    EXPECT_FALSE(projection.destination_bytes.empty());
                }
                lease.reset();
                EXPECT_EQ(pool->usedSlots(), 0u);
            }
        }

        TEST(
            CpuExpertSlotPool,
            FP16BF16AndFP32BuildStableRawTripletsAndMigrationViews)
        {
            constexpr std::array<TensorType, 3> types{
                TensorType::FP16,
                TensorType::BF16,
                TensorType::FP32,
            };
            int participant_id = 100;
            for (const TensorType type : types)
            {
                SCOPED_TRACE(static_cast<int>(type));
                const auto format = ExpertWeightFormat::floating(type);
                const auto projections =
                    cpuFloatingSlotProjectionSpecs(type);
                const std::size_t planned_bytes =
                    plannedCpuSlotAllocationBytes(projections);
                auto pool = CpuExpertSlotPool::createForTest({
                    .participant_id = participant_id++,
                    .layer_idx = 3,
                    .capacity = 1,
                    .projections = projections,
                    .memory_placement =
                        CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                    .perf_device = "cpu-floating-source-test",
                });
                EXPECT_EQ(pool->allocationBytes(), planned_bytes);
                auto lease = pool->acquire(7, 14);
                ASSERT_TRUE(lease.has_value());
                ASSERT_EQ(lease->projections.size(), 3u);

                for (std::size_t projection_index = 0;
                     projection_index < lease->projections.size();
                     ++projection_index)
                {
                    auto &projection = lease->projections[projection_index];
                    ASSERT_FALSE(projection.destination_bytes.empty());
                    const std::uint8_t seed = static_cast<std::uint8_t>(
                        0x31u + projection_index * 0x17u);
                    for (std::size_t byte = 0;
                         byte < projection.destination_bytes.size();
                         ++byte)
                    {
                        projection.destination_bytes[byte] =
                            static_cast<std::uint8_t>(seed + byte * 13u);
                    }

                    MoEOverlayPreparedWeightSource source;
                    std::string error;
                    ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(
                        projection.engine,
                        DeviceId::cpu(),
                        source,
                        &error))
                        << error;
                    EXPECT_EQ(
                        source.kind,
                        MoEOverlayPreparedWeightSourceKind::
                            CpuContiguousFloating);
                    EXPECT_EQ(source.format, format);
                    EXPECT_EQ(source.floating.type, type);
                    EXPECT_EQ(
                        source.floating.data,
                        projection.destination_bytes.data());
                    EXPECT_EQ(
                        source.floating.bytes,
                        projection.destination_bytes.size());
                    ASSERT_EQ(
                        source.floating.bytes,
                        static_cast<std::size_t>(source.floating.n) *
                            static_cast<std::size_t>(source.floating.k) *
                            format.floatingElementBytes());
                    EXPECT_EQ(
                        std::memcmp(
                            source.floating.data,
                            projection.destination_bytes.data(),
                            source.floating.bytes),
                        0);
                }

                lease.reset();
                EXPECT_EQ(pool->usedSlots(), 0u);
            }
        }

        TEST(CpuExpertSlotPool, RejectsIncompleteOrDuplicateProjectionSets)
        {
            const NativeVnniSourceIdentity source_identity{
                .codebook_id = native_vnni_formats::Q4_0.codebook_id,
                .is_superblock = false,
                .present = true,
            };
            auto specs = cpuSlotProjectionSpecs(source_identity);
            specs[2].projection = ExpertTierWeightProjection::Gate;
            EXPECT_THROW(
                (void)CpuExpertSlotPool::createForTest({
                    .participant_id = 1,
                    .layer_idx = 0,
                    .capacity = 1,
                    .projections = std::move(specs),
                    .memory_placement =
                        CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                    .perf_device = "cpu-test",
                }),
                std::invalid_argument);
        }

        TEST(
            MoEOverlayPreparedWeightSource,
            ResolvesEngineOwnedCpuBytesAndPinsTheirEpochLease)
        {
            const NativeVnniSourceIdentity source_identity{
                .codebook_id = native_vnni_formats::Q4_K.codebook_id,
                .is_superblock = native_vnni_formats::Q4_K.is_superblock,
                .present = true,
            };
            auto pool = CpuExpertSlotPool::createForTest({
                .participant_id = 4,
                .layer_idx = 2,
                .capacity = 1,
                .projections = cpuSlotProjectionSpecs(source_identity),
                .memory_placement =
                    CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                .perf_device = "cpu-source-test",
            });
            auto lease = pool->acquire(6, 12);
            ASSERT_TRUE(lease.has_value());

            MoEOverlayPreparedWeightSource source;
            std::string error;
            ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(
                lease->projections[0].engine,
                DeviceId::cpu(),
                source,
                &error))
                << error;
            EXPECT_EQ(
                source.kind,
                MoEOverlayPreparedWeightSourceKind::CpuNativeVnni);
            ASSERT_NE(source.cpu_packed, nullptr);
            EXPECT_EQ(
                source.format,
                ExpertWeightFormat::nativeVnni(source_identity));
            EXPECT_EQ(
                source.cpu_packed->native_interleaved.data(),
                lease->projections[0].destination_bytes.data());

            lease.reset();
            EXPECT_EQ(pool->usedSlots(), 1u)
                << "resolved source ownership must pin the old residency slot";
            source = {};
            EXPECT_EQ(pool->usedSlots(), 0u);
        }

        TEST(
            MoEOverlayPreparedWeightSource,
            AcceptsCanonicalAndCpuNormalizedGpuRepresentationsOnly)
        {
            std::vector<std::uint8_t> payload(64 * 32);
            std::vector<std::uint16_t> scales(64);
            std::vector<std::uint16_t> mins(64);
            std::vector<std::uint32_t> emins(64);

            auto make_descriptor = [&](std::uint8_t codebook)
            {
                DeviceNativeVNNIMatrixDesc descriptor;
                descriptor.payload = payload.data();
                descriptor.scales = scales.data();
                descriptor.mins = mins.data();
                descriptor.emins = emins.data();
                descriptor.n = 64;
                descriptor.k = 32;
                descriptor.blocks_per_row = 1;
                descriptor.codebook_id = codebook;
                descriptor.allocation_payload_bytes_per_block = 32;
                descriptor.allocation_has_mins = 1;
                descriptor.allocation_has_emins = 1;
                return descriptor;
            };

            const NativeVnniSourceIdentity q2_source{
                .codebook_id = native_vnni_formats::Q2_K.codebook_id,
                .is_superblock = native_vnni_formats::Q2_K.is_superblock,
                .present = true,
            };
            MoEOverlayPreparedWeightSource canonical;
            std::string error;
            ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(
                std::make_shared<ExportingGpuGemm>(
                    make_descriptor(native_vnni_formats::Q2_K.codebook_id),
                    q2_source),
                DeviceId::cuda(0),
                canonical,
                &error))
                << error;
            EXPECT_EQ(canonical.gpu_packed.payload_bytes_per_block, 8u);
            EXPECT_TRUE(canonical.gpu_packed.is_asymmetric);
            EXPECT_TRUE(canonical.gpu_packed.has_emins);

            MoEOverlayPreparedWeightSource normalized;
            ASSERT_FALSE(resolveMoEOverlayPreparedWeightSource(
                std::make_shared<ExportingGpuGemm>(
                    make_descriptor(kNativeVnniExpandedInt8MinCodebook),
                    q2_source),
                DeviceId::rocm(0),
                normalized,
                &error))
                << error;
            // A one-scale INT8 descriptor cannot preserve Q2_K's independent
            // scales and minima, even if its backing allocation is large enough.

            MoEOverlayPreparedWeightSource rejected;
            EXPECT_FALSE(resolveMoEOverlayPreparedWeightSource(
                std::make_shared<ExportingGpuGemm>(
                    make_descriptor(/*unrelated execution codebook=*/7),
                    q2_source),
                DeviceId::cuda(0),
                rejected,
                &error));
            EXPECT_FALSE(error.empty());
        }

        TEST(
            ExpertTierGpuBlobChunkProtocol,
            ChunksNeverCrossSeparatedArrayBoundaries)
        {
            std::vector<uint8_t> source_payload(32);
            std::vector<uint16_t> source_scales(2);
            std::vector<uint16_t> source_mins(2);
            std::vector<uint32_t> source_emins(2);
            std::vector<uint8_t> destination_payload(32);
            std::vector<uint16_t> destination_scales(2);
            std::vector<uint16_t> destination_mins(2);
            std::vector<uint32_t> destination_emins(2);

            auto descriptor = [](
                                  std::vector<uint8_t> &payload,
                                  std::vector<uint16_t> &scales,
                                  std::vector<uint16_t> &mins,
                                  std::vector<uint32_t> &emins)
            {
                return GpuExpertPackedDescriptor{
                    .ptrs = {
                        .d_vnni = payload.data(),
                        .d_scales = scales.data(),
                        .d_mins = mins.data(),
                        .d_emins = emins.data(),
                    },
                    .n = 2,
                    .k = 32,
                    .blocks_per_row = 1,
                    .codebook_id = 8,
                    .payload_bytes_per_block = 16,
                    .is_asymmetric = true,
                    .has_emins = true,
                    .vnni_bytes = payload.size(),
                    .scales_bytes = scales.size() * sizeof(uint16_t),
                    .mins_bytes = mins.size() * sizeof(uint16_t),
                    .emins_bytes = emins.size() * sizeof(uint32_t),
                };
            };

            const auto source = descriptor(
                source_payload,
                source_scales,
                source_mins,
                source_emins);
            const auto destination = descriptor(
                destination_payload,
                destination_scales,
                destination_mins,
                destination_emins);

            ExpertTierGpuBlobChunkProtocol protocol(7);
            std::string error;
            ASSERT_TRUE(protocol.begin(source, destination, &error)) << error;

            std::vector<ExpertTierGpuBlobChunk> chunks;
            while (auto chunk = protocol.takeNext())
                chunks.push_back(*chunk);

            ASSERT_EQ(chunks.size(), 9u);
            EXPECT_EQ(protocol.totalBytes(), 48u);
            EXPECT_EQ(protocol.submittedBytes(), protocol.totalBytes());
            EXPECT_TRUE(protocol.exhausted());

            const std::vector<std::tuple<ExpertTierGpuBlobRegion, size_t, size_t>>
                expected{
                    {ExpertTierGpuBlobRegion::Payload, 0, 7},
                    {ExpertTierGpuBlobRegion::Payload, 7, 7},
                    {ExpertTierGpuBlobRegion::Payload, 14, 7},
                    {ExpertTierGpuBlobRegion::Payload, 21, 7},
                    {ExpertTierGpuBlobRegion::Payload, 28, 4},
                    {ExpertTierGpuBlobRegion::Scales, 0, 4},
                    {ExpertTierGpuBlobRegion::Mins, 0, 4},
                    {ExpertTierGpuBlobRegion::Emins, 0, 7},
                    {ExpertTierGpuBlobRegion::Emins, 7, 1},
                };
            for (size_t index = 0; index < chunks.size(); ++index)
            {
                EXPECT_EQ(chunks[index].region, std::get<0>(expected[index]));
                EXPECT_EQ(
                    chunks[index].region_offset,
                    std::get<1>(expected[index]));
                EXPECT_EQ(chunks[index].bytes, std::get<2>(expected[index]));
                EXPECT_EQ(chunks[index].sequence, index);
                EXPECT_LE(chunks[index].bytes, 7u);
            }
        }

        TEST(
            ExpertTierGpuBlobChunkProtocol,
            SkipsAbsentOptionalRegionsAndRejectsLayoutMismatch)
        {
            std::vector<uint8_t> source_payload(16);
            std::vector<uint16_t> source_scales(1);
            std::vector<uint8_t> destination_payload(16);
            std::vector<uint16_t> destination_scales(1);
            const GpuExpertPackedDescriptor source{
                .ptrs = {
                    .d_vnni = source_payload.data(),
                    .d_scales = source_scales.data(),
                },
                .n = 1,
                .k = 32,
                .blocks_per_row = 1,
                .codebook_id = 2,
                .payload_bytes_per_block = 16,
                .vnni_bytes = source_payload.size(),
                .scales_bytes = source_scales.size() * sizeof(uint16_t),
            };
            GpuExpertPackedDescriptor destination{
                .ptrs = {
                    .d_vnni = destination_payload.data(),
                    .d_scales = destination_scales.data(),
                },
                .n = 1,
                .k = 32,
                .blocks_per_row = 1,
                .codebook_id = 2,
                .payload_bytes_per_block = 16,
                .vnni_bytes = destination_payload.size(),
                .scales_bytes = destination_scales.size() * sizeof(uint16_t),
            };

            ExpertTierGpuBlobChunkProtocol protocol(64);
            ASSERT_TRUE(protocol.begin(source, destination));
            const auto payload = protocol.takeNext();
            const auto scales = protocol.takeNext();
            ASSERT_TRUE(payload.has_value());
            ASSERT_TRUE(scales.has_value());
            EXPECT_EQ(payload->region, ExpertTierGpuBlobRegion::Payload);
            EXPECT_EQ(scales->region, ExpertTierGpuBlobRegion::Scales);
            EXPECT_FALSE(protocol.takeNext().has_value());

            ++destination.k;
            std::string error;
            EXPECT_FALSE(protocol.begin(source, destination, &error));
            EXPECT_FALSE(error.empty());
            EXPECT_THROW(
                ExpertTierGpuBlobChunkProtocol(0),
                std::invalid_argument);
        }
    } // namespace
} // namespace llaminar2
