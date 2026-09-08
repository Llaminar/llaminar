/**
 * @file Test__CPUNativeVNNIWeightPacking.cpp
 * @brief Byte-exact, all-codebook integration proof for CPU NativeVNNI packing.
 *
 * ExpertOverlay prepares many independently owned expert projections.  The
 * production packer therefore parallelizes direct source-codebook decoding into
 * its permanent VNNI byte stream.  These tests independently reconstruct that
 * stream from the public IINT8Unpackable contract and require exact equality for
 * every Expanded-INT8 codebook.  The complete 21-format catalog additionally
 * proves that every physical encoding is deterministic across OpenMP team sizes.
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
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <barrier>

#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/PackedWeightsSerialization.h"
#include "tensors/FP16Utils.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/TensorClasses.h"
#include "utils/TestTensorFactory.h"

namespace llaminar2::cpu::native_vnni::test
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
            throw std::invalid_argument(
                "No TestTensorFactory fixture for canonical format " +
                std::string(quant_type));
        }

        /**
         * @brief Reconstruct final Expanded-INT8 bytes without using the packer.
         *
         * Each source block is decoded through the scalar public interface.
         * This intentionally avoids both the optimized superblock decoder and
         * `packExpandedInt8Direct`, so the test does not share the implementation
         * whose byte layout it certifies.
         *
         * @param source Original quantized source tensor.
         * @param packed Production metadata describing destination geometry.
         * @param row_start First source row represented by destination row zero.
         * @return Independently constructed interleaved byte stream.
         */
        std::vector<uint8_t> buildExpandedInt8Oracle(
            const TensorBase &source,
            const CPUNativeVNNIPackedWeights &packed,
            int row_start)
        {
            const auto *unpackable =
                dynamic_cast<const IINT8Unpackable *>(&source);
            if (unpackable == nullptr)
                throw std::invalid_argument(
                    "Expanded-INT8 oracle requires IINT8Unpackable source");
            if (!packed.usesExpandedInt8() || packed.data_stride != 2048)
                throw std::invalid_argument(
                    "Expanded-INT8 oracle received another physical encoding");

            const int chunks = packed.N_padded / 64;
            std::vector<uint8_t> expected(
                static_cast<size_t>(chunks) * packed.blocks_per_row *
                    static_cast<size_t>(packed.interleaved_block_stride),
                0);

            for (int chunk = 0; chunk < chunks; ++chunk)
            {
                const int valid_columns =
                    std::min(64, packed.N - chunk * 64);
                for (int kb = 0; kb < packed.blocks_per_row; ++kb)
                {
                    uint8_t *const block = expected.data() +
                        (static_cast<size_t>(chunk) * packed.blocks_per_row +
                         static_cast<size_t>(kb)) *
                            static_cast<size_t>(packed.interleaved_block_stride);
                    auto *const compensation =
                        reinterpret_cast<int16_t *>(
                            block + packed.data_stride);
                    auto *const scales = reinterpret_cast<uint16_t *>(
                        block + packed.data_stride + 128);
                    auto *const mins = packed.is_asymmetric
                        ? reinterpret_cast<uint16_t *>(
                              block + packed.data_stride + 256)
                        : nullptr;

                    for (int column = 0; column < valid_columns; ++column)
                    {
                        const size_t source_row = static_cast<size_t>(
                            row_start + chunk * 64 + column);
                        int8_t values[32]{};
                        unpackable->unpack_block_to_int8(
                            source_row, static_cast<size_t>(kb), values);

                        for (int group = 0; group < 8; ++group)
                        {
                            const int zmm = column / 16;
                            const int lane = column % 16;
                            std::memcpy(
                                block + group * 256 + zmm * 64 + lane * 4,
                                values + group * 4,
                                4);
                        }

                        int32_t sum = 0;
                        for (int value = 0; value < 32; ++value)
                            sum += values[value];
                        compensation[column] = static_cast<int16_t>(sum);
                        scales[column] = fp32_to_fp16(
                            unpackable->get_block_scale(
                                source_row, static_cast<size_t>(kb)));
                        if (mins != nullptr)
                        {
                            mins[column] = fp32_to_fp16(
                                unpackable->get_block_min(
                                    source_row, static_cast<size_t>(kb)));
                        }
                    }
                }
            }
            return expected;
        }

        /**
         * @brief Compare two aligned storage owners byte for byte.
         * @param lhs First packed byte stream.
         * @param rhs Second packed byte stream.
         * @param context Format-specific assertion context.
         */
        void expectIdenticalBytes(
            const AlignedVector<uint8_t> &lhs,
            const AlignedVector<uint8_t> &rhs,
            std::string_view context)
        {
            ASSERT_EQ(lhs.size(), rhs.size()) << context;
            EXPECT_EQ(std::memcmp(lhs.data(), rhs.data(), lhs.size()), 0)
                << context;
        }
    } // namespace

    TEST(CPUNativeVNNIWeightPacking,
         EveryCodebookIsByteExactAndThreadDeterministic)
    {
        constexpr int kFullN = 73;
        constexpr int kK = 544;
        constexpr int kRowStart = 3;
        constexpr int kRowEnd = 70;
        static_assert(kK % 32 == 0);
        static_assert(kK % 256 != 0);

        ASSERT_EQ(native_vnni_formats::kAllSourceFormats.size(), 21u);
        size_t expanded_formats = 0;

        for (const NativeVnniSourceFormat &format :
             native_vnni_formats::kAllSourceFormats)
        {
            SCOPED_TRACE(std::string(format.quant_type));
            auto source = createSourceTensor(
                format.quant_type,
                {static_cast<size_t>(kFullN), static_cast<size_t>(kK)});
            ASSERT_NE(source, nullptr);

            CPUNativeVNNIPackedWeights serial;
            {
                ScopedOMPThreadCount one_worker(1);
                ASSERT_TRUE(packWeightsCPUNativeVNNI(
                    source.get(), serial, kRowStart, kRowEnd));
            }

            CPUNativeVNNIPackedWeights parallel;
            {
                ScopedOMPThreadCount four_workers(4);
                ASSERT_TRUE(packWeightsCPUNativeVNNI(
                    source.get(), parallel, kRowStart, kRowEnd));
            }

            EXPECT_EQ(serial.N, kRowEnd - kRowStart);
            EXPECT_EQ(serial.N_padded, 128);
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

            if (!serial.usesExpandedInt8())
                continue;
            ++expanded_formats;
            const std::vector<uint8_t> oracle = buildExpandedInt8Oracle(
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

        // Six source formats use native nibble/Q6 encodings today.  This count
        // makes a future encoding-policy change update the independent oracle
        // deliberately rather than silently shrinking its coverage.
        EXPECT_EQ(expanded_formats, 15u);
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

    TEST(CPUNativeVNNIWeightPacking, FinalNUMAStorageCoversEveryCodebookAndArchive)
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
            for (const auto &format : native_vnni_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(std::string(format.quant_type));
                auto source = createSourceTensor(format.quant_type, {67, 544});
                CPUNativeVNNIPackedWeights reference;
                ASSERT_TRUE(packWeightsCPUNativeVNNI(source.get(), reference));
                auto engine = KernelFactory::prepareExpertGemmLocal(
                    source.get(), DeviceId::cpu(), KernelFactory::GemmPreparationKind::AUTO, placement);
                auto *native = dynamic_cast<CPUNativeVNNIGemmKernel *>(engine.get());
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
                auto *received_native = dynamic_cast<CPUNativeVNNIGemmKernel *>(received.get());
                ASSERT_NE(received_native, nullptr);
                const auto &arrived = received_native->packedWeights().native_interleaved;
                expectIdenticalBytes(packed.native_interleaved, arrived, "archive changed native bytes");
                expectFinalPlacement(arrived.data(), arrived.size(), node, before);
            }
        }
    }

    TEST(CPUNativeVNNIWeightPacking, FinalNUMAStoragePreservesAllFloatingPointFormats)
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

    TEST(CPUNativeVNNIWeightPacking, FinalNUMAStorageConcurrentHugePageCandidates)
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

    TEST(CPUNativeVNNIWeightPacking, FinalNUMAStorageRejectsUnavailableNodeBeforePublication)
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
} // namespace llaminar2::cpu::native_vnni::test
