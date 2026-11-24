/**
 * @file Test__ExpertTierWeightStream.cpp
 * @brief Device-free proof of the heterogeneous expert-weight stream ABI.
 *
 * Exercise source provenance, bounded receive/publication state and exact
 * layout transformations independently of backend launchers. Native grid
 * scales/minima must survive conversion, not merely a lossy round trip.
 * Native read views also admit the real slot/demotion owners, separating source
 * codec widths from physical expanded payloads without changing publications.
 * The Linux prepared-storage allocation probe observes the production aligned
 * heap allocator even when the reference lives in the shared core.
 */

#include "../../utils/TestTensorFactory.h"
#include "../../utils/OpaqueExpertTierStreamFixture.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "../../utils/ExpertTierEncodedColumnOracle.h"
#include "execution/moe/ExpertTierGpuBlobTransferLane.h"
#include "execution/moe/CpuExpertSlotPool.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"

#include <gtest/gtest.h>
#include "tensors/MXFP4Format.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include "../../utils/ExpertTierEncodedColumnOracle.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#ifdef __linux__
#include <dlfcn.h>
#endif
#include <limits>
#include <span>
#include <string>
#include <string_view>

#include <tuple>
#include <vector>

#ifdef __linux__
namespace
{
    thread_local std::size_t *prepared_allocation_count = nullptr;

    /** @brief Count only prepared heap allocations made synchronously on this thread. */
    class ScopedPreparedAllocationCount final
    {
    public:
        /** @brief Arm the counter for a single reference conversion. */
        explicit ScopedPreparedAllocationCount(std::size_t &count) { prepared_allocation_count = &count; }
        /** @brief Retire the counter before running GTest or unrelated work. */
        ~ScopedPreparedAllocationCount() { prepared_allocation_count = nullptr; }
    };
}

/** @brief Count calls to the exact AlignedVector heap allocation boundary, then delegate to libc. */
extern "C" void *aligned_alloc(std::size_t alignment, std::size_t bytes) noexcept
{
    static const auto allocate = reinterpret_cast<decltype(&std::aligned_alloc)>(dlsym(RTLD_NEXT, "aligned_alloc"));
    if (!allocate)
        std::abort();
    if (prepared_allocation_count)
        ++*prepared_allocation_count;
    return allocate(alignment, bytes);
}
#endif

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
            projection.source_codebook_id =
                codebook == kNativeQuantisedIQ4XSSourceScaleCodebook ? uint8_t{4} : codebook;
            projection.codebook_id = codebook;
            projection.payload_bytes_per_block = payload_bytes;
            projection.is_asymmetric = asymmetric;
            projection.is_superblock = superblock;
            const size_t blocks =
                static_cast<size_t>(projection.N) *
                projection.blocks_per_row;
            projection.payload.resize(blocks * payload_bytes);
            projection.scales.assign(nativeQuantisedScalePlaneStorageWords(codebook, blocks), 0);
            if (asymmetric)
                projection.mins.resize(blocks);

            for (size_t block = 0; block < blocks; ++block)
            {
                for (size_t byte = 0; byte < payload_bytes; ++byte)
                {
                    uint8_t value = static_cast<uint8_t>(
                        (block * 37u + byte * 19u + 11u) & 0xffu);
                    if (codebook == 19)
                    {
                        value = static_cast<uint8_t>(static_cast<int8_t>(
                            static_cast<int>((block * 13u + byte * 7u) % 127u) - 63));
                    }
                    projection.payload[block * payload_bytes + byte] = value;
                }
                if (nativeQuantisedScalePlaneBytes(codebook) == 1)
                {
                    // Native E8M0 plane: the zero marker plus every admitted exponent,
                    // including the range outside the former FP16 window.
                    const uint8_t e = block % 17 == 5 ? uint8_t{0}
                        : static_cast<uint8_t>(mxfp4::kMinNormalFP32Exponent +
                              block % (mxfp4::kMaxNativeE8M0Exponent - mxfp4::kMinNormalFP32Exponent + 1));
                    reinterpret_cast<uint8_t *>(projection.scales.data())[block] = e;
                }
                else
                    projection.scales[block] = static_cast<uint16_t>(
                        0x2400u + (block % 0x0800u));
                if (asymmetric)
                {
                    projection.mins[block] = static_cast<uint16_t>(
                        codebook == kNativeQuantisedIQ4XSSourceScaleCodebook
                            ? block % 64u : 0xa000u + (block % 0x0800u));
                }
            }
            return projection;
        }

        /** @return Normalized accelerator Q8 bytes with independently specified source provenance. */
        HostGpuExpertPackedProjection makeQ8Projection(const NativeQuantisedFormatInfo &source)
        {
            auto gpu = makeProjection(19, 32, false, source.is_superblock);
            gpu.source_codebook_id = source.codebook_id;
            return gpu;
        }

        /** @test Native Q5 never enters the retired expanded GPU migration ABI. */
        TEST(ExpertTierWeightStream, NativeQ5RejectsExpandedGpuDescriptor)
        {
            for (const auto *format : {&native_quantised_formats::Q5_0, &native_quantised_formats::Q5_1})
            {
                SCOPED_TRACE(format->codebook_id);
                const auto make = [&](uint8_t codebook, uint8_t bytes) {
                    auto projection = makeProjection(codebook, bytes, format->is_asymmetric, false);
                    projection.source_codebook_id = format->codebook_id;
                    return projection;
                };
                const auto original = make(format->codebook_id, format->payload_bytes);
                auto expanded = make(format->is_asymmetric ? 23 : 19, 32);
                ASSERT_TRUE(original.valid());
                EXPECT_FALSE(expanded.valid());
                const auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(*format, original.N, original.K,
                    101, 0, 7, ExpertTierWeightProjection::Gate, 1);
                EXPECT_EQ(manifest.gpu_payload_bytes_per_block, 20);
                EXPECT_EQ(manifest.gpu_codebook_id, format->codebook_id);
            }
        }

        /** @test Native Q5 GPU round trips preserve the GGUF bytes and the original CPU units. */
        TEST(ExpertTierWeightStream, NativeQ5FileDensityRoundTrip)
        {
            using namespace cpu::native_quantised;
            constexpr int n = 70, k = 288, blocks = n * (k / 32);
            for (const auto label : {"Q5_0", "Q5_1"})
            {
                SCOPED_TRACE(label);
                const auto tensor = test::quantizedVerifierFormat(label).create({n, k}, 90197u);
                const auto *source = dynamic_cast<const IINT8Unpackable *>(tensor.get());
                ASSERT_NE(source, nullptr);
                const auto &format = *source->vnniFormatInfo();
                CPUNativeQuantisedPackedWeights cpu, restored;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), cpu));
                HostGpuExpertPackedProjection gpu;
                std::string error;
                ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, gpu, &error)) << error;
                ASSERT_TRUE(gpu.valid(&error)) << error;
                EXPECT_EQ(gpu.codebook_id, format.codebook_id);
                EXPECT_EQ(gpu.payload_bytes_per_block, 20);
                EXPECT_EQ(gpu.metadata_layout, 0u);
                EXPECT_EQ(gpu.payload.size() + sizeof(uint16_t) * (gpu.scales.size() + gpu.mins.size()),
                    tensor->size_bytes());
                std::vector<uint8_t> payload(blocks * 20u);
                std::vector<uint16_t> scales(blocks), mins(format.is_asymmetric ? blocks : 0u);
                const VnniPackContext context{.N = n, .K = k, .blocks_per_row = k / 32,
                    .payload_bytes = 20, .payload_array = payload.data(), .scales_array = scales.data(),
                    .mins_array = mins.data()};
                for (int row = 0; row < n; ++row)
                    for (int block = 0; block < k / 32; ++block)
                        source->packVnniBlock(context, row, row, block);
                EXPECT_EQ(gpu.payload, payload);
                EXPECT_EQ(gpu.scales, scales);
                EXPECT_EQ(gpu.mins, mins);
                ASSERT_TRUE(gpuToCpuExpertPackedReference(gpu, restored, &error)) << error;
                ASSERT_EQ(restored.preparedBytes(), cpu.preparedBytes());
                EXPECT_EQ(std::memcmp(restored.native_interleaved.data(), cpu.native_interleaved.data(), cpu.preparedBytes()), 0);
            }
        }

        /** @test Real CPU preparation and both stream directions preserve every canonical decoder. */
        TEST(ExpertTierWeightStream, CanonicalPreparationNeverEmitsRetiredPhysicalDecoder)
        {
            using namespace cpu::native_quantised;
            constexpr uint8_t retired_decoder = 23;
            unsigned coded = 0, ordinary = 0;
            for (const auto &entry : native_quantised_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(entry.quant_type);
                const auto &format = *entry.metadata;
                const auto &fixtures = test::quantizedVerifierFormats();
                const auto fixture = std::find_if(fixtures.begin(), fixtures.end(), [&](const auto &candidate) {
                    return candidate.source_codebook_id == format.codebook_id &&
                        candidate.source_is_superblock == format.is_superblock;
                });
                ASSERT_NE(fixture, fixtures.end());
                const auto source = fixture->create({70, 512}, 314159);
                ASSERT_NE(source, nullptr);
                CPUNativeQuantisedPackedWeights cpu;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(source.get(), cpu));
                ASSERT_EQ(cpu.encoding, preparedEncodingForSource(format));
                (migrationUsesSuperblockCodes(format) ? coded : ordinary)++;

                const auto to_gpu = makeCpuToGpuExpertTierWeightStreamManifest(cpu, 3, 0, 0,
                    ExpertTierWeightProjection::Gate, 3);
                const auto to_cpu = makeGpuToCpuExpertTierWeightStreamManifest(format, cpu.N, cpu.K,
                    4, 0, 0, ExpertTierWeightProjection::Gate, 3);
                HostGpuExpertPackedProjection gpu;
                std::string error;
                ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, gpu, &error)) << error;
                const auto canonical = canonicalDeviceVnniFormat(format);
                EXPECT_NE(gpu.codebook_id, retired_decoder);
                EXPECT_EQ(gpu.codebook_id, canonical.codebook_id);
                EXPECT_EQ(to_gpu.gpu_codebook_id, canonical.codebook_id);
                EXPECT_EQ(to_cpu.gpu_codebook_id, canonical.codebook_id);
                for (const auto &manifest : {to_gpu, to_cpu})
                {
                    auto layout = manifest.deviceLayout();
                    ASSERT_TRUE(layout.valid());
                    const bool single_scale = expertTierPreservesNativeSingleScale(layout.cpu_encoding, layout.cpu_codebook_id);
                    EXPECT_TRUE(layout.cpuStorage().coded() || single_scale ||
                        layout.cpu_encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale);
                    if (layout.cpu_encoding == CPUNativeQuantisedEncoding::ExpandedInt8)
                    {
                        EXPECT_TRUE(single_scale);
                        EXPECT_EQ(layout.gpu_codebook_id, 19u);
                    }
                    layout.gpu_codebook_id = retired_decoder;
                    EXPECT_FALSE(layout.valid());
                }
            }
            EXPECT_EQ(coded, 13u);
            EXPECT_EQ(ordinary, 9u);

            // The old converter admitted this caller-constructed representation.
            // Reject it before submission in either direction; its CPU geometry
            // alone is internally consistent, but it is not canonical Q4_1.
            const auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
                native_quantised_formats::Q4_1, 70, 96, 5, 0, 0, ExpertTierWeightProjection::Gate, 3);
            auto expanded = manifest.deviceLayout();
            expanded.cpu_encoding = CPUNativeQuantisedEncoding::ExpandedInt8;
            expanded.cpu_data_stride = preparedDataStride(expanded.cpu_encoding, expanded.cpu_codebook_id);
            expanded.cpu_block_stride = preparedInterleavedBlockStride(expanded.cpu_encoding, true, expanded.cpu_codebook_id);
            expanded.gpu_codebook_id = retired_decoder;
            expanded.gpu_payload_bytes_per_block = 32;
            ASSERT_TRUE(expanded.cpuStorage().valid());
            EXPECT_FALSE(expanded.valid());
            expanded.direction = ExpertTierWeightConversionDirection::CpuToGpu;
            EXPECT_FALSE(expanded.valid());
        }

        /** @return An ordinary source tensor with a ragged final output-column chunk. */
        std::unique_ptr<TensorBase> makeCodedSource(uint8_t codebook, int columns = 512)
        {
            const auto &formats = test::quantizedVerifierFormats();
            const auto it = std::find_if(formats.begin(), formats.end(), [&](const auto &format) {
                return format.source_is_superblock && format.source_codebook_id == codebook;
            });
            return it == formats.end() ? nullptr : it->create({70, columns}, 314159);
        }

        /** @test Every coded format preserves its original block-major GPU planes and resident CPU bytes. */
        TEST(ExpertTierWeightStream, CodedSourcePlanesRoundTripWithoutRequantization)
        {
            using namespace cpu::native_quantised;
            namespace sb = native_quantised_superblock;
            unsigned formats = 0;
            for (const auto &entry : native_quantised_formats::kAllSourceFormats)
            {
                if (!migrationUsesSuperblockCodes(*entry.metadata))
                    continue;
                SCOPED_TRACE(entry.quant_type);
                ++formats;
                const auto tensor = makeCodedSource(entry.metadata->codebook_id);
                ASSERT_NE(tensor, nullptr);
                const auto *source = dynamic_cast<const IINT8Unpackable *>(tensor.get());
                ASSERT_NE(source, nullptr);
                CPUNativeQuantisedPackedWeights cpu, restored;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), cpu));
                HostGpuExpertPackedProjection gpu;
                std::string error;
                ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, gpu, &error)) << error;
                ASSERT_TRUE(gpu.valid(&error)) << error;
                const auto blocks = size_t(gpu.N) * gpu.blocks_per_row;
                const auto extent = sb::planeBytes(gpu.codebook_id, gpu.N, gpu.blocks_per_row, 0);
                ASSERT_EQ(gpu.codes.size(), extent.code);
                ASSERT_EQ(gpu.superblocks.size(), extent.superblock);
                const auto field = sb::superblockFieldBytes(gpu.codebook_id);
                const auto code_bits = sb::superblockCodePlaneBits(gpu.codebook_id);
                std::vector<uint8_t> payload(gpu.payload.size());
                std::vector<uint32_t> codes(blocks), headers(extent.superblock / field);
                const VnniPackContext context{.N = gpu.N, .K = gpu.K, .blocks_per_row = int(gpu.blocks_per_row),
                    .payload_bytes = gpu.payload_bytes_per_block, .payload_array = payload.data(),
                    .codes_array = codes.data(), .superblocks_array = headers.data()};
                for (int n = 0; n < gpu.N; ++n)
                    for (unsigned block = 0; block < gpu.blocks_per_row; ++block)
                        source->packVnniBlock(context, n, n, block);
                EXPECT_EQ(gpu.payload, payload);
                if (code_bits)
                    for (size_t block = 0; block < blocks; ++block)
                        EXPECT_EQ(sb::loadPackedBits(gpu.codes.data(), block * code_bits, code_bits), codes[block]);
                for (size_t header = 0; header < headers.size(); ++header)
                    EXPECT_EQ(std::memcmp(gpu.superblocks.data() + header * field, &headers[header], field), 0);
                ASSERT_TRUE(gpuToCpuExpertPackedReference(gpu, restored, &error)) << error;
                ASSERT_EQ(restored.preparedBytes(), cpu.preparedBytes());
                EXPECT_EQ(std::memcmp(restored.native_interleaved.data(), cpu.native_interleaved.data(), cpu.preparedBytes()), 0);
                gpu.metadata_layout = 0;
                EXPECT_FALSE(gpuToCpuExpertPackedReference(gpu, restored, &error));
                EXPECT_NE(error.find("layout disagrees"), std::string::npos);
            }
            EXPECT_EQ(formats, 13u);
        }

        /** @test Coded wire units carry exactly the live independent code planes and touched header words. */
        TEST(ExpertTierWeightStream, CodedMidSuperblockChunksRetainExactPlanes)
        {
            using namespace cpu::native_quantised;
            namespace sb = native_quantised_superblock;
            using P = MoEExpertProjectionOwnership::Projection;
            using R = MoEProjectionResidence;
            for (const auto &format : native_quantised_formats::kAllSourceFormats)
            {
                if (!migrationUsesSuperblockCodes(*format.metadata))
                    continue;
                SCOPED_TRACE(format.quant_type);
                const auto tensor = makeCodedSource(format.metadata->codebook_id);
                ASSERT_NE(tensor, nullptr);
                CPUNativeQuantisedPackedWeights packed;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), packed));
                const auto source_bytes = packed.readView().bytes();
                const auto book = canonicalDeviceVnniCodebookId(*format.metadata);
                const unsigned field = sb::superblockFieldBytes(book);
                const unsigned code_bits = sb::superblockCodePlaneBits(book);
                const unsigned low = code_bits == 12 ? 8u : code_bits == 6 ? 4u : code_bits;
                for (int phase = 1; phase < 8; ++phase)
                {
                    SCOPED_TRACE(phase);
                    const P cpu{R::ExpertOwner, 70, 512, 0, 70, 0, 512};
                    const P gpu{R::ParticipantInputSlice, 70, 512, 1, 67, phase * 32, (10 - phase) * 32};
                    const auto intersection = ExpertTierWeightProjectionIntersection::between(cpu, gpu);
                    const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                        packed, 17, 0, 3, ExpertTierWeightProjection::Down, 3).forIntersection(intersection);
                    const auto layout = manifest.deviceLayout();
                    ASSERT_TRUE(layout.valid());
                    EXPECT_EQ(layout.cpuStorage().wireGeometry(64).phase, phase);
                    EXPECT_EQ(layout.gpu_superblock_phase, phase);
                    EXPECT_EQ(layout.chunkBytes(0), 0);
                    EXPECT_FALSE(gatherExpertTierCPUWeightChunk(layout, source_bytes, 0, 0, {}));
                    const auto blocks = 10 - phase;
                    size_t expected = 0;
                    for (unsigned columns : {64u, 3u})
                        expected += blocks * size_t(format.metadata->payload_bytes * columns +
                            (low * columns + 7) / 8 + ((code_bits - low) * columns + 7) / 8) + 2 * field * columns;
                    EXPECT_EQ(manifest.total_stream_bytes, expected);
                    size_t actual = 0;
                    std::vector<uint8_t> destination(source_bytes.size(), 0xa5);
                    for (uint32_t first = 0; first < layout.unit_count; first += 3)
                    {
                        const auto count = std::min(3u, layout.unit_count - first);
                        const auto bytes = layout.chunkBytes(count, first);
                        std::vector<uint8_t> wire(bytes, 0xa5), returned(bytes, 0x53);
                        ASSERT_TRUE(gatherExpertTierCPUWeightChunk(layout, source_bytes, first, count, wire));
                        ASSERT_TRUE(scatterExpertTierCPUWeightChunk(layout, wire, first, count, destination));
                        ASSERT_TRUE(gatherExpertTierCPUWeightChunk(layout, destination, first, count, returned));
                        EXPECT_EQ(returned, wire);
                        actual += bytes;
                    }
                    EXPECT_EQ(actual, expected);
                    for (int n = 1; n < 68; ++n)
                        for (int kb = phase; kb < 10; ++kb)
                            EXPECT_EQ(test::encodedTierColumn(destination, layout, n, kb),
                                test::encodedTierColumn(source_bytes, layout, n, kb));
                }
            }
        }

        /** @test A resident coded root cannot use one header for two source superblocks. */
        TEST(ExpertTierWeightStream, ResidentCodedCpuOriginRequiresSourceSuperblockBoundary)
        {
            for (int origin : {0, 32, 64, 224, 256, 512})
            {
                EXPECT_NO_THROW(requireExpertTierResidentCpuOrigin(native_quantised_formats::Q4_1, origin));
                if (origin % 256 == 0)
                {
                    EXPECT_NO_THROW(requireExpertTierResidentCpuOrigin(native_quantised_formats::IQ2_S, origin));
                    continue;
                }
                try
                {
                    requireExpertTierResidentCpuOrigin(native_quantised_formats::IQ2_S, origin);
                    ADD_FAILURE() << "Accepted misaligned origin " << origin;
                }
                catch (const std::invalid_argument &error)
                {
                    EXPECT_EQ(error.what(), "Coded CPU expert IQ2_S K origin " + std::to_string(origin) +
                        " is not aligned to the 256-value source superblock");
                }
            }
        }

        /** @test Real demotion publications retain native view geometry for every source. */
        TEST(ExpertTierWeightStream, PreparedReadViewAcceptsAllMigratedSourceFormats)
        {
            for (const auto &source : native_quantised_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(source.quant_type);
                if (cpu::native_quantised::migrationUsesSuperblockCodes(*source.metadata))
                    continue; // The catalog-wide admission test asserts the exact rejection.
                const auto physical = canonicalDeviceVnniFormat(*source.metadata);
                auto gpu = makeProjection(physical.codebook_id, physical.payload_bytes_per_block,
                    physical.is_asymmetric, source.metadata->is_superblock);
                gpu.source_codebook_id = source.metadata->codebook_id;
                gpu.has_emins = physical.has_emins;
                if (physical.codebook_id == 11)
                    for (std::size_t block = 0; block < gpu.mins.size(); ++block)
                        gpu.mins[block] = static_cast<std::uint16_t>(block & 15u);
                cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
                std::string error;
                if (cpu::native_quantised::migrationUsesSuperblockCodes(*source.metadata))
                {
                    const auto tensor = makeCodedSource(source.metadata->codebook_id);
                    ASSERT_NE(tensor, nullptr);
                    ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(), cpu));
                    ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, gpu, &error)) << error;
                }
                ASSERT_TRUE(gpuToCpuExpertPackedReference(gpu, cpu, &error)) << error;
                const auto view = cpu.readView();
                EXPECT_EQ(view.interleavedBase(), cpu.native_interleaved.data());
                EXPECT_EQ(view.bytes().size(), cpu.native_interleaved.size());
                EXPECT_EQ(view.N, cpu.N);
                EXPECT_EQ(view.K, cpu.K);
                EXPECT_EQ(view.source_payload_bytes, source.metadata->payload_bytes);
                EXPECT_EQ(view.payload_bytes, view.usesExpandedInt8() ? 32 : source.metadata->payload_bytes);
                EXPECT_EQ(view.encoding, cpu.encoding);
                EXPECT_EQ(view.numerical_policy, cpu.numerical_policy);
            }
        }

        /** @test Staged GPU planes and header-less CPU extents never enter the coded converter. */
        TEST(ExpertTierWeightStream, CodedDemotionRejectsLegacyPlanesBeforeAllocating)
        {
            using namespace cpu::native_quantised;
            constexpr int n = 64, k = 256;
            unsigned formats = 0;
            for (const auto &entry : native_quantised_formats::kAllSourceFormats)
            {
                const auto &source = *entry.metadata;
                if (!migrationUsesSuperblockCodes(source))
                    continue;
                SCOPED_TRACE(entry.quant_type);
                ++formats;
                const auto physical = canonicalDeviceVnniFormat(source);
                auto staged = makeProjection(physical.codebook_id, physical.payload_bytes_per_block,
                    physical.is_asymmetric, true);
                staged.N = n;
                staged.K = k;
                staged.blocks_per_row = k / 32;
                staged.payload.resize(n * staged.blocks_per_row * physical.payload_bytes_per_block);
                staged.scales.resize(n * staged.blocks_per_row);
                staged.mins.resize(physical.is_asymmetric ? n * staged.blocks_per_row : 0);
                staged.emins.resize(physical.has_emins ? n * staged.blocks_per_row : 0);
                staged.has_emins = physical.has_emins;
                CPUNativeQuantisedPackedWeights destination;
                std::string error;
#ifdef __linux__
                std::size_t allocations = 0;
                bool converted;
                {
                    ScopedPreparedAllocationCount count(allocations);
                    converted = gpuToCpuExpertPackedReference(staged, destination, &error);
                }
                EXPECT_EQ(allocations, 0u);
                EXPECT_FALSE(converted);
#else
                EXPECT_FALSE(gpuToCpuExpertPackedReference(staged, destination, &error));
#endif
                EXPECT_NE(error.find("metadata layout disagrees"), std::string::npos) << error;
                EXPECT_EQ(destination.native_interleaved.data(), nullptr);
                EXPECT_EQ(destination.N, 0);

                const auto layout = makeGpuToCpuExpertTierWeightStreamManifest(source, n, k,
                    101, 2, 7, ExpertTierWeightProjection::Down, k / 32).deviceLayout();
                ASSERT_TRUE(layout.valid());
                ASSERT_TRUE(layout.cpuStorage().valid());
                const ExpertTierGpuConstProjectionView legacy_view{
                    .payload = staged.payload.data(), .scales = staged.scales.data(),
                    .mins = staged.mins.data(), .emins = staged.emins.data(),
                    .payload_bytes = staged.payload.size(), .scales_bytes = staged.scales.size() * sizeof(uint16_t),
                    .mins_bytes = staged.mins.size() * sizeof(uint16_t), .emins_bytes = staged.emins.size() * sizeof(uint32_t)};
                EXPECT_FALSE(legacy_view.validFor(layout));
                auto legacy_layout = layout;
                legacy_layout.cpu_encoding = CPUNativeQuantisedEncoding::ExpandedInt8;
                legacy_layout.cpu_data_stride = preparedDataStride(legacy_layout.cpu_encoding, source.codebook_id);
                legacy_layout.cpu_block_stride = preparedInterleavedBlockStride(legacy_layout.cpu_encoding,
                    source.is_asymmetric, source.codebook_id);
                EXPECT_FALSE(legacy_layout.cpuStorage().valid());
                EXPECT_FALSE(legacy_layout.valid());

                const auto root = layout.cpuStorage().rootGeometry();
                const auto headerless = root.block_stride * root.blocks_per_row;
                ASSERT_GT(root.rowBytes(), headerless);
                ASSERT_EQ(layout.cpuStorageBytes(), root.rowBytes());
                ASSERT_EQ(layout.chunkBytes(layout.unit_count), root.rowBytes());
                if (&source == &native_quantised_formats::Q2_K)
                {
                    EXPECT_EQ(headerless, 5120u);
                    EXPECT_EQ(layout.cpuStorageBytes(), 5376u);
                }
                std::vector<uint8_t> short_root(headerless, 0xa5);
                const auto untouched = short_root;
                std::vector<uint8_t> wire(root.rowBytes());
                EXPECT_FALSE(scatterExpertTierCPUWeightChunk(layout.cpuStorage(), wire, 0, layout.unit_count, short_root));
                EXPECT_FALSE(scatterExpertTierCPUWeightChunk(layout, wire, 0, layout.unit_count, short_root));
                EXPECT_EQ(short_root, untouched);
            }
#ifdef __linux__
            // Positive control uses the same out-of-line shared-core conversion,
            // so a link configuration that defeats interposition cannot pass.
            const auto tensor = test::quantizedVerifierFormat("Q2_K").create({n, k}, 37);
            CPUNativeQuantisedPackedWeights original, result;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), original));
            HostGpuExpertPackedProjection coded;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(original, coded));
            std::size_t allocations = 0;
            bool converted;
            {
                ScopedPreparedAllocationCount count(allocations);
                converted = gpuToCpuExpertPackedReference(coded, result);
            }
            EXPECT_TRUE(converted);
            EXPECT_EQ(allocations, 1u);
#endif
            EXPECT_EQ(formats, 13u);
        }

        /** @test Coded Q5_K and IQ3_S cannot enter the legacy stream in either direction. */
        TEST(ExpertTierWeightStream, CodedQ5KIQ3SRejectLegacyMigration)
        {
            using namespace cpu::native_quantised;
            for (const auto *format : {&native_quantised_formats::Q5_K, &native_quantised_formats::IQ3_S})
            {
                SCOPED_TRACE(format->codebook_id);
                const auto physical = canonicalDeviceVnniFormat(*format);
                auto gpu = makeProjection(physical.codebook_id, physical.payload_bytes_per_block,
                    physical.is_asymmetric, true);
                CPUNativeQuantisedPackedWeights rejected;
                std::string error;
                EXPECT_FALSE(gpuToCpuExpertPackedReference(gpu, rejected, &error));
                EXPECT_NE(error.find("layout disagrees"), std::string::npos);
                const auto tensor = makeCodedSource(format->codebook_id);
                ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), rejected));
                const auto correct = makeCpuToGpuExpertTierWeightStreamManifest(rejected, 1, 0, 0,
                    ExpertTierWeightProjection::Gate, 1);
                EXPECT_TRUE(correct.valid());
                auto stale = correct;
                stale.version = 4;
                stale.manifest_hash = stale.computedHash();
                EXPECT_FALSE(stale.valid());
            }

        }

        /** @brief Device-free GEMM exporting a caller-owned GPU descriptor. */
        class ExportingGpuGemm final : public ITensorGemm
        {
        public:
            ExportingGpuGemm(
                DeviceNativeQuantisedMatrixDesc descriptor,
                NativeQuantisedSourceIdentity source_identity)
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

            bool exportNativeQuantisedMatrixDesc(
                DeviceNativeQuantisedMatrixDesc &out) override
            {
                out = descriptor_;
                return descriptor_.valid();
            }

            bool exportNativeQuantisedSourceIdentity(
                NativeQuantisedSourceIdentity &out) const override
            {
                out = source_identity_;
                return source_identity_.present;
            }

        private:
            DeviceNativeQuantisedMatrixDesc descriptor_;
            NativeQuantisedSourceIdentity source_identity_;
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
            EXPECT_EQ(actual.metadata_layout, expected.metadata_layout);
            EXPECT_EQ(actual.codes, expected.codes);
            EXPECT_EQ(actual.superblocks, expected.superblocks);
            EXPECT_EQ(actual.scales, expected.scales);
            EXPECT_EQ(actual.mins, expected.mins);
            EXPECT_EQ(actual.emins, expected.emins);
        }

        /** @test The IQ4_XS coded source tuple survives native transposition. */
        TEST(ExpertTierWeightStream, IQ4XSTupleRoundTripAndPhysicalLayoutAreExact)
        {
            using namespace cpu::native_quantised;
            auto tensor = makeCodedSource(4);
            ASSERT_NE(tensor, nullptr);
            CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), cpu));
            HostGpuExpertPackedProjection source;
            std::string error;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, source, &error)) << error;
            CPUNativeQuantisedPackedWeights restored;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(source, restored, &error)) << error;
            EXPECT_EQ(cpu.encoding, CPUNativeQuantisedEncoding::NibbleLUTSourceScale);
            EXPECT_EQ(cpu.codebook_id, 4); EXPECT_TRUE(cpu.is_superblock);
            EXPECT_FALSE(cpu.is_asymmetric);
            for (int n = 0; n < cpu.N; ++n) for (int kb = 0; kb < cpu.blocks_per_row; ++kb)
            {
                const auto actual = restored.codedBlockMetadata(n / 64, kb, n % 64);
                const auto original = cpu.codedBlockMetadata(n / 64, kb, n % 64);
                EXPECT_EQ(actual.primary, original.primary);
                EXPECT_EQ(actual.secondary, original.secondary);
                EXPECT_EQ(restored.blockMin(n / 64, kb, n % 64), 0.0f);
            }
            const auto demote = makeGpuToCpuExpertTierWeightStreamManifest(
                native_quantised_formats::IQ4_XS, source.N, source.K, 101, 2, 7,
                ExpertTierWeightProjection::Down, 2);
            const auto promote = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu, 101, 2, 7, ExpertTierWeightProjection::Down, 2);
            ASSERT_TRUE(demote.valid(&error)) << error;
            ASSERT_TRUE(promote.valid(&error)) << error;
            EXPECT_TRUE(demote.deviceLayout().valid()); EXPECT_TRUE(promote.deviceLayout().valid());
            EXPECT_EQ(demote.gpu_codebook_id, 24); EXPECT_EQ(demote.gpu_source_codebook_id, 4);
            EXPECT_EQ(demote.cpu_block_stride, cpu.geometry().block_stride);
            auto malformed = demote.deviceLayout(); malformed.gpu_is_asymmetric = 0;
            EXPECT_FALSE(malformed.valid());
            malformed = demote.deviceLayout(); malformed.gpu_source_is_superblock = 0;
            EXPECT_FALSE(malformed.valid());
            HostGpuExpertPackedProjection returned;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, returned, &error)) << error;
            expectGpuEqual(source, returned);
            source.codebook_id = 4; EXPECT_FALSE(source.valid());
        }

        /** @brief Preserve the existing explicit normalized metadata boundary only.
         * This does not execute or certify a rotated expert migration. Native
         * source preparation still selects encoding4/physical24 exclusively.
         */
        TEST(ExpertTierWeightStream, IQ4XSExplicitExpandedCannotEnterCodedMigration)
        {
            auto expanded = test::opaqueExpandedExpertTierBytes(70, 96);
            std::string error;
            expanded.codebook_id = 4; expanded.is_superblock = true;
            EXPECT_THROW((void)makeCpuToGpuExpertTierWeightStreamManifest(
                expanded, 101, 2, 7, ExpertTierWeightProjection::Down, 2), std::invalid_argument);

        }

        class ExpertTierWeightEncodingTest
            : public ::testing::TestWithParam<HostGpuExpertPackedProjection>
        {
        };

        TEST_P(ExpertTierWeightEncodingTest, NativeGpuRoundTripIsByteExact)
        {
            const auto source = GetParam();
            ASSERT_TRUE(source.valid());

            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(source, cpu, &error)) << error;
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
                makeProjection(4, 16, false, false),
                makeProjection(5, 16, true, false),
                makeProjection(6, 20, false, false),
                makeProjection(7, 20, true, false),
                makeProjection(kNativeQuantisedMXFP4Codebook, 16, false, false),
                makeProjection(19, 32, false, false),
                makeQ8Projection(native_quantised_formats::Q8_1),
                makeQ8Projection(native_quantised_formats::Q8_K)));

        /** @test A ragged MXFP4 CPU wire carries exactly one exponent byte per live column. */
        TEST(ExpertTierWeightStream, MXFP4CpuWireKeepsOddLiveExponentBytes)
        {
            using namespace cpu::native_quantised;
            const auto tensor = test::TestTensorFactory::createMXFP4Random({69, 96}, 23);
            CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), cpu));
            using Residence = MoEProjectionResidence;
            const MoEExpertProjectionOwnership::Projection root{Residence::ExpertOwner, cpu.N, cpu.K, 0, cpu.N, 0, cpu.K};
            ExpertTierWeightCPUStorageLayout traits{
                .cpu_codebook_id = cpu.codebook_id, .cpu_encoding = cpu.encoding,
                .cpu_data_stride = static_cast<uint32_t>(cpu.data_stride),
                .cpu_block_stride = static_cast<uint32_t>(cpu.interleaved_block_stride), .wire_columns_per_unit = 64};
            const auto layout = cpuExpertTierStorageLayoutForIntersection(traits, root,
                ExpertTierWeightProjectionIntersection::between(root, root), cpu.interleaved_block_stride);
            std::vector<uint8_t> restored(cpu.preparedBytes(), 0);
            for (uint32_t first = 0; first < layout.unit_count; ++first)
            {
                const size_t columns = first < 3 ? 64 : 5;
                std::vector<uint8_t> wire(layout.chunkBytes(1, first));
                ASSERT_EQ(wire.size(), columns * 17);
                ASSERT_TRUE(gatherExpertTierCPUWeightChunk(layout, cpu.readView().bytes(), first, 1, wire));
                const auto *exponents = cpu.readView().admittedE8M0Scales(first / 3, first % 3);
                EXPECT_EQ(std::memcmp(wire.data() + columns * 16, exponents, columns), 0);
                ASSERT_TRUE(scatterExpertTierCPUWeightChunk(layout, wire, first, 1, restored));
            }
            EXPECT_EQ(std::memcmp(restored.data(), cpu.native_interleaved.data(), restored.size()), 0);
        }

        TEST(ExpertTierWeightStream, OrderedChunksWriteDirectlyToFinalBytes)
        {
            const auto tensor = makeCodedSource(5, 96);
            ASSERT_NE(tensor, nullptr);
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(), cpu));
            std::string error;

            const auto manifest =
                makeGpuToCpuExpertTierWeightStreamManifest(
                native_quantised_formats::Q4_K,
                cpu.N,
                cpu.K,
                41,
                7,
                12,
                ExpertTierWeightProjection::Up,
                2);
            ASSERT_TRUE(manifest.valid(&error)) << error;
            ASSERT_EQ(manifest.unit_count, 6u);
            ASSERT_EQ(manifest.deviceLayout().cpuStorageBytes(),cpu.native_interleaved.size());
            ASSERT_LT(manifest.total_stream_bytes,cpu.native_interleaved.size());

            std::vector<uint8_t> final_bytes(
                manifest.deviceLayout().cpuStorageBytes(), 0);
            ExpertTierWeightStreamReceiver receiver(manifest, final_bytes);

            for (uint32_t sequence = 0; sequence < 3; ++sequence)
            {
                const uint32_t first_unit = sequence * 2;
                const size_t byte_count=manifest.deviceLayout().chunkBytes(2,first_unit);
                std::vector<uint8_t> encoded(byte_count);
                ASSERT_TRUE(gatherExpertTierCPUWeightChunk(manifest.deviceLayout(),cpu.native_interleaved,first_unit,2,encoded));
                const std::span<const uint8_t> payload(encoded);
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
            const auto tensor = makeCodedSource(5, 96);
            ASSERT_NE(tensor, nullptr);
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(), cpu));
            const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu,
                9,
                3,
                5,
                ExpertTierWeightProjection::Down,
                1);
            std::vector<uint8_t> final_bytes(
                manifest.deviceLayout().cpuStorageBytes(), 0x5a);
            ExpertTierWeightStreamReceiver receiver(manifest, final_bytes);

            std::vector<uint8_t> first(manifest.deviceLayout().chunkBytes(1));
            ASSERT_TRUE(gatherExpertTierCPUWeightChunk(manifest.deviceLayout(), cpu.native_interleaved, 0, 1, first));
            ExpertTierWeightStreamChunkHeader header;
            header.manifest_hash = manifest.manifest_hash;
            header.payload_hash = expertTierWeightBytesHash(first);
            header.sequence = 1;
            header.first_unit = 0;
            header.unit_count = 1;
            header.payload_bytes = static_cast<uint32_t>(first.size());

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
            const auto tensor = makeCodedSource(5, 96);
            ASSERT_NE(tensor, nullptr);
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(), cpu));
            auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
                native_quantised_formats::Q4_K,
                cpu.N,
                cpu.K,
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
            ASSERT_EQ(native_quantised_formats::kAllSourceFormats.size(), 22u);
            for (const NativeQuantisedSourceFormat &entry :
                 native_quantised_formats::kAllSourceFormats)
            {
                ASSERT_NE(entry.metadata, nullptr) << entry.quant_type;
                SCOPED_TRACE(entry.quant_type);
                const auto makeManifest = [&] {
                    return makeGpuToCpuExpertTierWeightStreamManifest(*entry.metadata, 70, 256,
                        101, 3, 7, ExpertTierWeightProjection::Gate, 4);
                };
                const auto manifest = makeManifest();
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
                    canonicalDeviceVnniCodebookId(*entry.metadata))
                    << entry.quant_type;
            }
        }

        TEST(ExpertTierWeightStream, MultiScaleSourcesRejectLegacyLossyExpandedLayouts)
        {
            for (const auto &entry : native_quantised_formats::kAllSourceFormats)
            {
                const auto id = entry.metadata->codebook_id;
                if (id != 8 && !hasCompactMultiScaleVnniPayload(id))
                    continue;
                SCOPED_TRACE(std::string(entry.quant_type));
                auto layout = makeGpuToCpuExpertTierWeightStreamManifest(
                    native_quantised_formats::Q8_0, 70, 256, 101, 3, 7,
                    ExpertTierWeightProjection::Gate, 4).deviceLayout();
                ASSERT_TRUE(layout.valid());
                layout.cpu_codebook_id = layout.gpu_source_codebook_id = id;
                layout.cpu_is_superblock = layout.gpu_source_is_superblock = 1;
                layout.cpu_is_asymmetric = layout.gpu_is_asymmetric = 1;
                layout.cpu_encoding = cpu::native_quantised::CPUNativeQuantisedEncoding::ExpandedInt8;
                layout.cpu_data_stride = cpu::native_quantised::preparedDataStride(layout.cpu_encoding, layout.cpu_codebook_id);
                layout.cpu_block_stride = cpu::native_quantised::preparedInterleavedBlockStride(layout.cpu_encoding, true, layout.cpu_codebook_id);
                layout.gpu_codebook_id = 23;
                layout.gpu_payload_bytes_per_block = 32;
                layout.gpu_has_emins = 0;
                EXPECT_FALSE(layout.valid());
                layout.direction = ExpertTierWeightConversionDirection::CpuToGpu;
                EXPECT_FALSE(layout.valid());
            }
        }

        std::vector<CpuExpertSlotPool::ProjectionSpec>
        cpuSlotProjectionSpecs(NativeQuantisedSourceIdentity source_identity)
        {
            const auto format =
                ExpertWeightFormat::nativeQuantised(source_identity);
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
            const NativeQuantisedSourceIdentity source_identity{
                .codebook_id = native_quantised_formats::Q4_0.codebook_id,
                .is_superblock = native_quantised_formats::Q4_0.is_superblock,
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
                NativeQuantisedSourceIdentity actual;
                ASSERT_TRUE(
                    projection.engine->exportNativeQuantisedSourceIdentity(actual));
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
            ASSERT_EQ(native_quantised_formats::kAllSourceFormats.size(), 22u);
            int participant_id = 0;
            for (const NativeQuantisedSourceFormat &entry :
                 native_quantised_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(entry.quant_type);
                ASSERT_NE(entry.metadata, nullptr);
                const NativeQuantisedSourceIdentity source_identity{
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
                    NativeQuantisedSourceIdentity actual;
                    ASSERT_TRUE(
                        projection.engine->exportNativeQuantisedSourceIdentity(
                            actual));
                    EXPECT_EQ(actual, source_identity);
                    EXPECT_FALSE(projection.destination_bytes.empty());
                    const auto *packed = projection.engine->exportCPUNativeQuantisedPackedWeights();
                    ASSERT_NE(packed, nullptr);
                    // An acquired slot is unadmitted until its arrival writes
                    // every byte; zero words satisfy every metadata rule.
                    EXPECT_FALSE(packed->publicationSealed());
                    std::fill(projection.destination_bytes.begin(), projection.destination_bytes.end(), std::uint8_t{0});
                    projection.engine->admitCPUNativeQuantisedPublication();
                    const auto view = packed->readView();
                    EXPECT_EQ(view.interleavedBase(), projection.destination_bytes.data());
                    EXPECT_EQ(view.bytes().size(), projection.destination_bytes.size());
                    EXPECT_EQ(view.source_payload_bytes, entry.metadata->payload_bytes);
                    EXPECT_EQ(view.payload_bytes, view.usesExpandedInt8() ? 32 : entry.metadata->payload_bytes);
                }
                lease.reset();
                EXPECT_EQ(pool->usedSlots(), 0u);
            }
        }

        /** @test CPU-only Q5 slots seal the full native tuple after a rectangular copy. */
        TEST(CpuExpertSlotPool, NativeQ5SlotSealsExactPayloadAndRectangle)
        {
            using namespace cpu::native_quantised;
            for (const auto *format : {&native_quantised_formats::Q5_0, &native_quantised_formats::Q5_1})
            {
                SCOPED_TRACE(format->codebook_id);
                auto pool = CpuExpertSlotPool::createForTest({
                    .participant_id = 0,
                    .layer_idx = 0,
                    .capacity = 1,
                    .projections = cpuSlotProjectionSpecs({format->codebook_id, false, true}),
                    .memory_placement = CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                    .perf_device = "cpu-native-q5-slot-test",
                });
                auto lease = pool->acquire(3, 9);
                ASSERT_TRUE(lease.has_value());
                const auto &gate = lease->projections[0];
                const auto *packed = gate.engine->exportCPUNativeQuantisedPackedWeights();
                ASSERT_NE(packed, nullptr);
                EXPECT_EQ(packed->payload_bytes, 20);
                ASSERT_EQ(packed->encoding, CPUNativeQuantisedEncoding::Q5NativeSingleScale);
                EXPECT_FALSE(packed->publicationSealed());
                const ExpertTierWeightCPUStorageLayout from{
                    .N = 13, .K = 32, .blocks_per_row = 1,
                    .cpu_N = 64, .cpu_K = 64, .cpu_first_row = 7, .cpu_first_column = 32,
                    .cpu_codebook_id = packed->codebook_id, .cpu_is_asymmetric = uint8_t(packed->is_asymmetric),
                    .cpu_encoding = packed->encoding,
                    .cpu_data_stride = static_cast<uint32_t>(packed->data_stride),
                    .cpu_block_stride = static_cast<uint32_t>(packed->interleaved_block_stride),
                    .unit_count = 1, .maximum_units_per_chunk = 1};
                auto to = from;
                to.cpu_first_row = 19;
                to.cpu_first_column = 0;
                std::vector<uint8_t> source(from.cpuStorageBytes());
                for (size_t i = 0; i < source.size(); ++i)
                    source[i] = static_cast<uint8_t>((i ^ (i >> 8)) * 53 + 7);
                std::fill(gate.destination_bytes.begin(), gate.destination_bytes.end(), uint8_t{0});
                const std::vector<uint8_t> untouched(to.cpuStorageBytes());
                ASSERT_TRUE(copyExpertTierCPUWeightChunk(from, source, to, gate.destination_bytes, 0, 1));
                gate.engine->admitCPUNativeQuantisedPublication();
                const auto view = packed->readView();
                EXPECT_EQ(view.payload_bytes, 20);
                EXPECT_EQ(view.bytes().size(), to.cpuStorageBytes());
                EXPECT_EQ(to.chunkBytes(1), 13u * (format->is_asymmetric ? 24u : 22u));
                for (size_t column = 0; column < 64; ++column)
                    for (size_t block = 0; block < 2; ++block)
                    {
                        const auto expected = column >= 19 && column < 32 && block == 0
                            ? test::encodedTierColumn(source, from, column - 19 + 7, 1)
                            : test::encodedTierColumn(untouched, to, column, block);
                        EXPECT_EQ(test::encodedTierColumn(view.bytes(), to, column, block), expected);
                    }
            }
        }

        /**
         * @test Slot reuse: every acquisition opens a new publication generation
         * at the same address, so bytes of an earlier epoch can never stay
         * admitted, and a foreign geometry cannot admit an arrival.
         */
        TEST(CpuExpertSlotPool, AcquisitionWithdrawsAdmissionUntilArrival)
        {
            const NativeQuantisedSourceIdentity identity{
                .codebook_id = native_quantised_formats::MXFP4.codebook_id,
                .is_superblock = native_quantised_formats::MXFP4.is_superblock,
                .present = true,
            };
            auto pool = CpuExpertSlotPool::createForTest({
                .participant_id = 0,
                .layer_idx = 0,
                .capacity = 1,
                .projections = cpuSlotProjectionSpecs(identity),
                .memory_placement = CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                .perf_device = "cpu-slot-reuse-test",
            });
            const std::uint8_t *root = nullptr;
            for (std::uint64_t epoch = 1; epoch <= 3; ++epoch)
            {
                SCOPED_TRACE(epoch);
                auto lease = pool->acquire(4, epoch);
                ASSERT_TRUE(lease.has_value());
                const auto &projection = lease->projections.front();
                const auto *packed = projection.engine->exportCPUNativeQuantisedPackedWeights();
                ASSERT_NE(packed, nullptr);
                if (!root) root = projection.destination_bytes.data();
                EXPECT_EQ(projection.destination_bytes.data(), root);
                EXPECT_FALSE(packed->publicationSealed());
                EXPECT_DEATH((void)packed->readView(), "not admitted");
                // All raw exponent bytes set to 2 or 3 are valid; poison one only on the final acquisition.
                std::fill(projection.destination_bytes.begin(), projection.destination_bytes.end(),
                    static_cast<std::uint8_t>(epoch + 1));
                if (epoch == 3)
                {
                    const auto exponent = packed->geometry().blockOffset(0, 0) + packed->data_stride + 17;
                    ASSERT_LT(exponent, projection.destination_bytes.size());
                    projection.destination_bytes[exponent] = 0xff;
                    EXPECT_THROW(projection.engine->admitCPUNativeQuantisedPublication(), std::invalid_argument);
                    EXPECT_FALSE(packed->publicationSealed());
                    continue;
                }
                projection.engine->admitCPUNativeQuantisedPublication();
                EXPECT_TRUE(packed->publicationSealed());
                EXPECT_EQ(packed->readView().interleavedBase(), root);
                // Republishing an unchanged engine keeps its seal.
                const auto generation = packed->native_interleaved.generation();
                projection.engine->admitCPUNativeQuantisedPublication();
                EXPECT_EQ(packed->native_interleaved.generation(), generation);
            }
        }

        /** @test A deep copy of sealed bytes is admitted at its own root; a copy of stale bytes is not. */
        TEST(CpuExpertSlotPool, PublicationCopyInheritsOnlyACurrentSeal)
        {
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            const auto tensor = makeCodedSource(5);
            ASSERT_NE(tensor, nullptr);
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(), cpu));
            ASSERT_TRUE(cpu.publicationSealed());
            const auto copy = cpu;
            EXPECT_NE(copy.native_interleaved.data(), cpu.native_interleaved.data());
            EXPECT_TRUE(copy.publicationSealed());
            EXPECT_EQ(copy.readView().interleavedBase(), copy.native_interleaved.data());
            (void)cpu.native_interleaved.beginRewrite();
            const auto stale = cpu;
            EXPECT_FALSE(stale.publicationSealed());
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
            const NativeQuantisedSourceIdentity source_identity{
                .codebook_id = native_quantised_formats::Q4_0.codebook_id,
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
            const NativeQuantisedSourceIdentity source_identity{
                .codebook_id = native_quantised_formats::Q4_K.codebook_id,
                .is_superblock = native_quantised_formats::Q4_K.is_superblock,
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
                MoEOverlayPreparedWeightSourceKind::CpuNativeQuantised);
            ASSERT_NE(source.cpu_packed, nullptr);
            EXPECT_EQ(
                source.format,
                ExpertWeightFormat::nativeQuantised(source_identity));
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
            AcceptsOnlyCanonicalGpuRepresentations)
        {
            std::vector<std::uint8_t> payload(64 * 32);
            std::vector<std::uint16_t> scales(64);
            std::vector<std::uint16_t> mins(64);
            std::vector<std::uint32_t> emins(64);

            auto make_descriptor = [&](std::uint8_t codebook)
            {
                DeviceNativeQuantisedMatrixDesc descriptor;
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

            const NativeQuantisedSourceIdentity q2_source{
                .codebook_id = native_quantised_formats::Q2_K.codebook_id,
                .is_superblock = native_quantised_formats::Q2_K.is_superblock,
                .present = true,
            };
            MoEOverlayPreparedWeightSource canonical;
            std::string error;
            ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(
                std::make_shared<ExportingGpuGemm>(
                    make_descriptor(native_quantised_formats::Q2_K.codebook_id),
                    q2_source),
                DeviceId::cuda(0),
                canonical,
                &error))
                << error;
            EXPECT_EQ(canonical.gpu_packed.payload_bytes_per_block, 8u);
            EXPECT_TRUE(canonical.gpu_packed.is_asymmetric);
            EXPECT_TRUE(canonical.gpu_packed.has_emins);

            // Coded Q2_K has a header plane, not independently stored extended minima.
            auto coded = make_descriptor(q2_source.codebook_id);
            coded.metadata_layout = nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded, 0, 0);
            coded.emins = nullptr;
            coded.allocation_payload_bytes_per_block = 8;
            coded.allocation_has_emins = 0;
            MoEOverlayPreparedWeightSource coded_source;
            ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(std::make_shared<ExportingGpuGemm>(coded, q2_source),
                DeviceId::cuda(0), coded_source, &error)) << error;
            EXPECT_EQ(coded_source.gpu_packed.metadata_layout, coded.metadata_layout);
            EXPECT_EQ(coded_source.gpu_packed.emins_bytes, 0u);
            coded.emins = emins.data();
            EXPECT_FALSE(resolveMoEOverlayPreparedWeightSource(std::make_shared<ExportingGpuGemm>(coded, q2_source),
                DeviceId::cuda(0), coded_source, &error));
            EXPECT_NE(error.find("exact coded metadata planes"), std::string::npos);

            MoEOverlayPreparedWeightSource normalized;
            ASSERT_FALSE(resolveMoEOverlayPreparedWeightSource(
                std::make_shared<ExportingGpuGemm>(
                    make_descriptor(23),
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

#include "ExpertTierRectangularWeightContract.inc"
#include <thread>
#include "ExpertTierDirectCPUProjectionContract.inc"
