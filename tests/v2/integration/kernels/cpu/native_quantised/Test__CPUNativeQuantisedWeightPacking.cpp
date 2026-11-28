/**
 * @file Test__CPUNativeQuantisedWeightPacking.cpp
 * @brief Byte-exact, all-codebook integration proof for CPU NativeQuantised packing.
 *
 * ExpertOverlay prepares many independently owned expert projections.  The
 * production packer therefore parallelizes direct source-codebook decoding into
 * its permanent VNNI byte stream.  These tests independently reconstruct that
 * stream from the public IINT8Unpackable contract and require exact equality for
 * every signed Q8 source. Coded integer grids use the native
 * prepared payload and exact metadata, not the older lossy INT8 export.
 * The compact multi-scale oracle places coded payloads and shared metadata
 * independently of the production window packer, including ragged headers.
 * The 22-format catalog checks deterministic packing across teams.
 *
 * The geometry deliberately combines a nonzero tensor-parallel row slice, a
 * partial 64-column output chunk, complete 256-value superblocks, and a trailing
 * 32-value block.  A packing regression therefore cannot hide behind the common
 * aligned full-matrix case.
 * Final-storage tests additionally certify every physical page on each available
 * NUMA node, byte-identical packing/archive adoption, floating-point ownership,
 * and restoration of the caller's affinity. No post-pack migration is permitted.
 */

#include <gtest/gtest.h>
#include <omp.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <barrier>

#include "kernels/cpu/gemm/CPUNativeQuantisedCodedMetadata.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#include "kernels/cpu/gemm/CPUPreparedWeightWriter.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/PackedWeightsSerialization.h"
#include "loaders/PreparedHalfWeights.h"
#include "tensors/FP16Utils.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/NativeWeightSourceRange.h"
#include "tensors/TensorFactory.h"
#include "tensors/TensorClasses.h"
#include "utils/TestTensorFactory.h"

namespace llaminar2::cpu::native_quantised::test
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;
    namespace
    {
        /**
         * @brief Restore the caller's OpenMP team policy when a packing run ends.
         *
         * A separate scope is used for each pack so the test compares fresh
         * production executions rather than mutating the global policy without
         * an ownership boundary.
         */
        class ScopedOMPThreadCount final
        {
        public:
            /**
             * @brief Install an exact positive OpenMP team size.
             * @param threads Worker count used by the enclosed packing call.
             */
            explicit ScopedOMPThreadCount(int threads)
                : previous_threads_(omp_get_max_threads()),
                  previous_dynamic_(omp_get_dynamic())
            {
                if (threads <= 0)
                    throw std::invalid_argument(
                        "ScopedOMPThreadCount requires a positive worker count");
                omp_set_dynamic(0);
                omp_set_num_threads(threads);
            }

            /** @brief Restore the exact OpenMP policy observed at construction. */
            ~ScopedOMPThreadCount()
            {
                omp_set_num_threads(previous_threads_);
                omp_set_dynamic(previous_dynamic_);
            }

            ScopedOMPThreadCount(const ScopedOMPThreadCount &) = delete;
            ScopedOMPThreadCount &operator=(const ScopedOMPThreadCount &) = delete;

        private:
            int previous_threads_ = 1;
            int previous_dynamic_ = 0;
        };

        /**
         * @brief Create one deterministic tensor named by the canonical catalog.
         *
         * This adapter deliberately consumes `kAllSourceFormats` rather than
         * declaring a second list.  A newly registered source format reaches the
         * fatal branch until its test fixture exists, making coverage omissions
         * visible in the preflight gate.
         *
         * @param quant_type Canonical GGUF quantization type.
         * @param shape Logical `[N, K]` tensor shape.
         * @return Owned source tensor populated by TestTensorFactory.
         */
        std::unique_ptr<TensorBase> createSourceTensor(
            std::string_view quant_type,
            const std::vector<size_t> &shape)
        {
            using Factory = llaminar2::test::TestTensorFactory;
            if (quant_type == "Q4_0")
                return Factory::createQ4_0Random(shape);
            if (quant_type == "IQ4_NL")
                return Factory::createIQ4_NLRandom(shape);
            if (quant_type == "Q4_1")
                return Factory::createQ4_1Random(shape);
            if (quant_type == "Q5_0")
                return Factory::createQ5_0Random(shape);
            if (quant_type == "Q5_1")
                return Factory::createQ5_1Random(shape);
            if (quant_type == "Q8_0")
                return Factory::createQ8_0Random(shape);
            if (quant_type == "Q8_1")
                return Factory::createQ8_1Random(shape);
            if (quant_type == "Q8_K")
                return Factory::createQ8_KRandom(shape);
            if (quant_type == "IQ4_XS")
                return Factory::createIQ4_XSRandom(shape);
            if (quant_type == "Q4_K")
                return Factory::createQ4_KRandom(shape);
            if (quant_type == "Q5_K")
                return Factory::createQ5_KRandom(shape);
            if (quant_type == "Q6_K")
                return Factory::createQ6_KRandom(shape);
            if (quant_type == "Q3_K")
                return Factory::createQ3_KRandom(shape);
            if (quant_type == "Q2_K")
                return Factory::createQ2_KRandom(shape);
            if (quant_type == "IQ3_S")
                return Factory::createIQ3_SRandom(shape);
            if (quant_type == "IQ3_XXS")
                return Factory::createIQ3_XXSRandom(shape);
            if (quant_type == "IQ2_S")
                return Factory::createIQ2_SRandom(shape);
            if (quant_type == "IQ2_XS")
                return Factory::createIQ2_XSRandom(shape);
            if (quant_type == "IQ2_XXS")
                return Factory::createIQ2_XXSRandom(shape);
            if (quant_type == "IQ1_S")
                return Factory::createIQ1_SRandom(shape);
            if (quant_type == "IQ1_M")
                return Factory::createIQ1_MRandom(shape);
            if (quant_type == "MXFP4")
                return Factory::createMXFP4Random(shape);
            throw std::invalid_argument(
                "No TestTensorFactory fixture for canonical format " +
                std::string(quant_type));
        }

        /**
         * @brief Construct the native Q8 interleave independently of the CPU writer.
         * @param source Original Q8 tensor and signed-byte authority.
         * @param packed Geometry published by the production writer.
         * @param row_start First source row represented by destination row zero.
         * @return Padded native Q8 bytes with one FP16 scale per column.
         */
        std::vector<uint8_t> buildSignedInt8SingleScaleOracle(
            const TensorBase &source, const CPUNativeQuantisedPackedWeights &packed, int row_start)
        {
            const auto &unpackable = dynamic_cast<const IINT8Unpackable &>(source);
            constexpr int stride = 64 * 34;
            std::vector<uint8_t> expected(size_t(packed.N_padded / 64) * packed.blocks_per_row * stride);
            for (int n = 0; n < packed.N; ++n)
                for (int kb = 0; kb < packed.blocks_per_row; ++kb)
                {
                    auto *unit = expected.data() + (size_t(n / 64) * packed.blocks_per_row + kb) * stride;
                    const int column = n % 64;
                    std::array<int8_t, 32> values{};
                    unpackable.unpack_block_to_int8(row_start + n, kb, values.data());
                    for (int group = 0; group < 8; ++group)
                        std::memcpy(unit + group * 256 + column * 4, values.data() + group * 4, 4);
                    const auto scale = fp32_to_fp16(unpackable.get_block_scale(row_start + n, kb));
                    std::memcpy(unit + 2048 + column * 2, &scale, sizeof(scale));
                }
            return expected;
        }

        /**
         * @brief Compare two aligned storage owners byte for byte.
         * @param lhs First packed byte stream.
         * @param rhs Second packed byte stream.
         * @param context Format-specific assertion context.
         */
        template <typename LeftBytes, typename RightBytes>
        void expectIdenticalBytes(
            const LeftBytes &lhs,
            const RightBytes &rhs,
            std::string_view context)
        {
            ASSERT_EQ(lhs.size(), rhs.size()) << context;
            EXPECT_EQ(std::memcmp(lhs.data(), rhs.data(), lhs.size()), 0)
                << context;
        }
    } // namespace

    /** @test All three Q8 sources emit the original bytes and one scale across ragged K and N. */
    TEST(CPUNativeQuantisedWeightPacking, SignedInt8SingleScaleRetainsNativeQ8Bytes)
    {
        for (const auto *name : {"Q8_0", "Q8_1", "Q8_K"})
        {
            SCOPED_TRACE(name);
            auto source = createSourceTensor(name, {73u, 544u});
            ASSERT_TRUE(source);
            for (const int workers : {1, 4})
            {
                ScopedOMPThreadCount scoped_workers(workers);
                CPUNativeQuantisedPackedWeights packed;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(source.get(), packed, 3, 70));
                ASSERT_TRUE(packed.usesSignedInt8SingleScale());
                ASSERT_TRUE(packed.publicationSealed());
                EXPECT_EQ(packed.interleaved_block_stride, 64 * 34);
                const auto oracle = buildSignedInt8SingleScaleOracle(*source, packed, 3);
                expectIdenticalBytes(packed.native_interleaved, oracle, name);
            }
        }
    }

    TEST(CPUNativeQuantisedWeightPacking,
         EveryCodebookIsByteExactAndThreadDeterministic)
    {
        constexpr int kFullN = 73;
        constexpr int kK = 544;
        constexpr int kRowStart = 3;
        constexpr int kRowEnd = 70;
        static_assert(kK % 32 == 0);
        static_assert(kK % 256 != 0);

        ASSERT_EQ(native_quantised_formats::kAllSourceFormats.size(), 22u);
        size_t signed_q8_formats = 0;
        size_t compact_formats = 0;

        for (const NativeQuantisedSourceFormat &format :
             native_quantised_formats::kAllSourceFormats)
        {
            SCOPED_TRACE(std::string(format.quant_type));
            auto source = createSourceTensor(
                format.quant_type,
                {static_cast<size_t>(kFullN), static_cast<size_t>(kK)});
            ASSERT_NE(source, nullptr);

            CPUNativeQuantisedPackedWeights serial;
            {
                ScopedOMPThreadCount one_worker(1);
                ASSERT_TRUE(packWeightsCPUNativeQuantised(
                    source.get(), serial, kRowStart, kRowEnd));
            }

            CPUNativeQuantisedPackedWeights parallel;
            {
                ScopedOMPThreadCount four_workers(4);
                ASSERT_TRUE(packWeightsCPUNativeQuantised(
                    source.get(), parallel, kRowStart, kRowEnd));
            }

            ASSERT_EQ(serial.N, kRowEnd - kRowStart);
            ASSERT_EQ(serial.N_padded, 128);
            EXPECT_EQ(serial.blocks_per_row, kK / 32);
            EXPECT_EQ(serial.encoding, parallel.encoding);
            EXPECT_EQ(serial.is_asymmetric, format.metadata->is_asymmetric);
            EXPECT_EQ(serial.is_superblock, format.metadata->is_superblock);
            EXPECT_TRUE(serial.int8_flat.empty());
            EXPECT_TRUE(parallel.int8_flat.empty());
            expectIdenticalBytes(
                serial.native_interleaved,
                parallel.native_interleaved,
                std::string(format.quant_type) +
                    " differs between one and four workers");

            if (serial.usesCompactMultiScale())
            {
                ++compact_formats;
                const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(source.get());
                ASSERT_NE(unpackable, nullptr);
                constexpr int columns = 64;
                constexpr int blocks = kK / 32;
                constexpr int blocks_per_superblock = native_quantised_superblock::kBlocksPerSuperblock;
                constexpr int superblocks = (blocks + blocks_per_superblock - 1) / blocks_per_superblock;
                const auto codebook = format.metadata->codebook_id;
                const int bytes = preparedCompactPayloadBytes(codebook);
                const int words = preparedCompactPayloadWords(codebook);
                const int tail = preparedCompactPayloadTailBytes(codebook);
                const int code_bits = preparedCompactCodeBits(codebook);
                const int data_stride = columns * bytes;
                const int unit_stride = data_stride + columns * code_bits / 8;
                const int header_stride = columns * native_quantised_superblock::superblockFieldBytes(codebook);
                const int chunk_stride = blocks * unit_stride + superblocks * header_stride;
                ASSERT_EQ(serial.data_stride, data_stride);
                ASSERT_EQ(serial.interleaved_block_stride, unit_stride);
                ASSERT_EQ(serial.superblockHeaderBytes(), header_stride);
                std::vector<uint8_t> oracle(serial.N_padded / columns * chunk_stride, 0);
                std::vector<uint8_t> payload(blocks * bytes);
                std::vector<uint32_t> codes(blocks), headers(superblocks);
                // Drive the source codec over a whole row independently of the
                // production CPU packer's per-superblock destination window.
                const VnniPackContext context{
                    .raw_bytes = nullptr, .N = 1, .K = kK,
                    .blocks_per_row = blocks, .payload_bytes = bytes,
                    .payload_array = payload.data(),
                    .codes_array = codes.data(), .superblocks_array = headers.data(),
                };
                for (int column = 0; column < serial.N; ++column)
                {
                    for (int kb = 0; kb < blocks; ++kb)
                        unpackable->packVnniBlock(context, column + kRowStart, 0, kb);
                    auto *chunk = oracle.data() + (column / columns) * chunk_stride;
                    const int lane = column % columns;
                    for (int kb = 0; kb < blocks; ++kb)
                    {
                        auto *unit = chunk + kb * unit_stride + (kb / blocks_per_superblock) * header_stride;
                        const auto *source_payload = payload.data() + kb * bytes;
                        for (int word = 0; word < words; ++word)
                            std::memcpy(unit + (word * columns + lane) * 4, source_payload + word * 4, 4);
                        if (tail)
                            std::memcpy(unit + words * columns * 4 + lane * tail, source_payload + words * 4, tail);
                        if (codebook == native_quantised_formats::Q5_K.codebook_id)
                            coded_metadata::storeQ5HighBits(unit + words * columns * 4, lane, source_payload + words * 4);
                        if (code_bits)
                            coded_metadata::storeCode(unit + data_stride, codebook, lane, codes[kb]);
                    }
                    for (int sb = 0; sb < superblocks; ++sb)
                    {
                        const int end = std::min((sb + 1) * blocks_per_superblock, blocks);
                        coded_metadata::storeSuperblockWord(
                            chunk + end * unit_stride + sb * header_stride, codebook, lane, headers[sb]);
                    }
                }
                expectIdenticalBytes(serial.native_interleaved, oracle, format.quant_type);
            }
            if (!serial.usesSignedInt8SingleScale())
                continue;
            ++signed_q8_formats;
            const std::vector<uint8_t> oracle = buildSignedInt8SingleScaleOracle(
                *source, serial, kRowStart);
            ASSERT_EQ(serial.native_interleaved.size(), oracle.size());
            EXPECT_EQ(
                std::memcmp(
                    serial.native_interleaved.data(),
                    oracle.data(),
                    oracle.size()),
                0)
                << format.quant_type
                << " direct packing differs from the scalar block oracle";
        }

        EXPECT_EQ(signed_q8_formats, 3u);
        EXPECT_EQ(compact_formats, 10u);
    }
    namespace
    {
        /**
         * @brief Independently query every final page and the caller's affinity.
         * @param data First byte of a dedicated execution allocation.
         * @param bytes Live execution byte count, including the final partial page.
         * @param node Required physical NUMA node.
         * @param before Exact caller affinity captured before preparation.
         */
        void expectFinalPlacement(const void *data, size_t bytes, int node,
                                  const cpu_set_t &before)
        {
            ASSERT_NE(data, nullptr);
            ASSERT_GT(bytes, 0u);
            const auto page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
            ASSERT_EQ(reinterpret_cast<uintptr_t>(data) % page_size, 0u);
            std::vector<void *> pages;
            for (size_t offset = 0; offset < bytes; offset += page_size)
                pages.push_back(const_cast<uint8_t *>(static_cast<const uint8_t *>(data)) + offset);
            std::vector<int> nodes(pages.size(), -1);
            ASSERT_EQ(move_pages(0, pages.size(), pages.data(), nullptr, nodes.data(), 0), 0);
            for (size_t index = 0; index < nodes.size(); ++index)
                ASSERT_EQ(nodes[index], node) << "page " << index;
            cpu_set_t after{};
            ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
            EXPECT_TRUE(CPU_EQUAL(&before, &after));
        }
    }

    TEST(CPUNativeQuantisedWeightPacking, FinalNUMAStorageCoversEveryCodebookAndArchive)
    {
        if (numa_available() < 0)
            GTEST_SKIP() << "NUMA page-query support is unavailable";
        ScopedOMPThreadCount workers(4);
        cpu_set_t before{};
        ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0);
        for (int node = 0; node <= numa_max_node(); ++node)
        {
            if (!numa_bitmask_isbitset(numa_all_nodes_ptr, node))
                continue;
            SCOPED_TRACE(node);
            const auto placement = CPUWeightStoragePlacement::onNode(node);
            for (const auto &format : native_quantised_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(std::string(format.quant_type));
                auto source = createSourceTensor(format.quant_type, {67, 544});
                CPUNativeQuantisedPackedWeights reference;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(source.get(), reference));
                auto engine = KernelFactory::prepareExpertGemmLocal(
                    source.get(), DeviceId::cpu(), KernelFactory::GemmPreparationKind::AUTO, placement);
                auto *native = dynamic_cast<CPUNativeQuantisedGemmKernel *>(engine.get());
                ASSERT_NE(native, nullptr);
                const auto &packed = native->packedWeights();
                EXPECT_EQ(packed.native_interleaved.storageKind(),
                          AlignedVector<uint8_t>::StorageKind::AnonymousPageMapping);
                expectIdenticalBytes(reference.native_interleaved, packed.native_interleaved,
                                     "NUMA preparation changed native bytes");
                expectFinalPlacement(packed.native_interleaved.data(),
                                     packed.native_interleaved.size(), node, before);

                // The archive path must establish placement before its one copy,
                // not allocate arbitrary heap pages and attempt migration later.
                CPUPackedWeights wire_owner(std::move(reference));
                const auto blob = packed_weights_serialization::serialize(wire_owner);
                ASSERT_FALSE(blob.empty());
                auto received = KernelFactory::createExpertGemmFromTransferBlob(
                    blob.data(), blob.size(), placement);
                auto *received_native = dynamic_cast<CPUNativeQuantisedGemmKernel *>(received.get());
                ASSERT_NE(received_native, nullptr);
                const auto &arrived = received_native->packedWeights().native_interleaved;
                expectIdenticalBytes(packed.native_interleaved, arrived, "archive changed native bytes");
                expectFinalPlacement(arrived.data(), arrived.size(), node, before);
            }
        }
    }

    TEST(CPUNativeQuantisedWeightPacking, FinalNUMAStoragePreservesAllFloatingPointFormats)
    {
        if (numa_available() < 0)
            GTEST_SKIP() << "NUMA page-query support is unavailable";
        cpu_set_t before{};
        ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0);
        using Factory = llaminar2::test::TestTensorFactory;
        for (int node = 0; node <= numa_max_node(); ++node)
        {
            if (!numa_bitmask_isbitset(numa_all_nodes_ptr, node))
                continue;
            for (auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
            {
                SCOPED_TRACE(node);
                SCOPED_TRACE(static_cast<int>(type));
                std::shared_ptr<TensorBase> source;
                if (type == TensorType::FP16) source = Factory::createFP16Random({67, 544});
                if (type == TensorType::BF16) source = Factory::createBF16Random({67, 544});
                if (type == TensorType::FP32) source = Factory::createFP32Random({67, 544});
                ASSERT_NE(source, nullptr);
                std::vector<uint8_t> expected(source->size_bytes());
                std::memcpy(expected.data(), source->raw_data(), expected.size());
                auto engine = KernelFactory::prepareExpertGemmLocal(
                    source, DeviceId::cpu(), KernelFactory::GemmPreparationKind::AUTO,
                    CPUWeightStoragePlacement::onNode(node));
                ASSERT_NE(engine, nullptr);
                ContiguousFloatingPointWeightDescriptor final{};
                ASSERT_TRUE(engine->exportContiguousFloatingPointWeights(final));
                ASSERT_TRUE(final.valid());
                EXPECT_EQ(final.type, type);
                EXPECT_NE(final.data, source->raw_data());
                source.reset(); // The independent execution owner survives loader retirement.
                ASSERT_EQ(final.bytes, expected.size());
                EXPECT_EQ(std::memcmp(final.data, expected.data(), final.bytes), 0);
                expectFinalPlacement(final.data, final.bytes, node, before);
            }
        }
    }

    TEST(CPUNativeQuantisedWeightPacking, FinalNUMAStorageConcurrentHugePageCandidates)
    {
        if (numa_available() < 0)
            GTEST_SKIP() << "NUMA page-query support is unavailable";
        // Model-sized projections cross the allocator's transparent-huge-page
        // threshold. Concurrent faults exercise a different page lifecycle from
        // the small all-codebook fixtures above. All owners remain live until
        // their round completes, then are retired together before the next round.
        constexpr int workers = 8;
        std::barrier round(workers);
        std::vector<std::jthread> threads;
        for (int worker = 0; worker < workers; ++worker)
        {
            threads.emplace_back([&, worker] {
                const int node = worker % (numa_max_node() + 1);
                for (int iteration = 0; iteration < 16; ++iteration)
                {
                    round.arrive_and_wait();
                    AlignedVector<uint8_t> storage;
                    EXPECT_NO_THROW({
                        storage = CPUWeightStoragePlacement::onNode(node)
                            .allocate<uint8_t>(4 * 1024 * 1024 + worker * 4096);
                        EXPECT_NE(storage.data(), nullptr);
                    });
                    round.arrive_and_wait();
                }
            });
        }
    }

    TEST(CPUNativeQuantisedWeightPacking, FinalNUMAStorageRejectsUnavailableNodeBeforePublication)
    {
        if (numa_available() < 0)
            GTEST_SKIP() << "NUMA page-query support is unavailable";
        cpu_set_t before{}, after{};
        ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0);
        auto source = createSourceTensor("Q8_0", {67, 544});
        EXPECT_THROW(KernelFactory::prepareExpertGemmLocal(
            source.get(), DeviceId::cpu(), KernelFactory::GemmPreparationKind::AUTO,
            CPUWeightStoragePlacement::onNode(numa_max_node() + 1)), std::runtime_error);
        ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
        EXPECT_TRUE(CPU_EQUAL(&before, &after));
    }
#include "CPUNativeWeightWindowContract.inc"
#include "CPUNativeSourceAssemblyContract.inc"
} // namespace llaminar2::cpu::native_quantised::test
