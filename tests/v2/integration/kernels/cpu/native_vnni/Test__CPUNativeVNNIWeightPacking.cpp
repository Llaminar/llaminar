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
 */

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "tensors/FP16Utils.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/TensorClasses.h"
#include "utils/TestTensorFactory.h"

namespace llaminar2::cpu::native_vnni::test
{
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
} // namespace llaminar2::cpu::native_vnni::test
