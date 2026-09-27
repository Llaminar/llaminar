/**
 * @file Test__CPUVectorDotRows.cpp
 * @brief Serial-row byte contract for cache-reusing CPU router dots.
 *
 * Exercises the actual ISA-dispatched primitive and complete production router.
 * SIMD-width tails, strided destinations, ragged batches, all socket worker
 * widths and both top-k normalization modes must retain serial decode bytes.
 * No model or accelerator is required; timing belongs in the separate router
 * performance harness, never in this functional production-preflight gate.
 */
#include <gtest/gtest.h>

#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "kernels/cpu/primitives/VectorPrimitives.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <omp.h>

namespace
{
    using namespace llaminar2;

    /** @brief Restore the caller's team size after a thread-totality probe. */
    class ScopedWorkers
    {
    public:
        /** @brief Install one positive worker count for this test scope. */
        explicit ScopedWorkers(int workers) : previous_(omp_get_max_threads())
        {
            omp_set_num_threads(workers);
        }
        /** @brief Restore the surrounding test process's OpenMP policy. */
        ~ScopedWorkers() { omp_set_num_threads(previous_); }
        ScopedWorkers(const ScopedWorkers &) = delete;
        ScopedWorkers &operator=(const ScopedWorkers &) = delete;
    private:
        int previous_;
    };

    /** @brief Fill finite mixed-exponent values that expose rounding changes. */
    void fillValues(std::vector<float> &values, std::mt19937 &random)
    {
        for (float &value : values)
        {
            const auto word = static_cast<std::uint32_t>(random());
            value = std::bit_cast<float>((word & 0x807fffffu) |
                ((112u + ((word >> 23) % 22u)) << 23));
        }
    }

    /** @test Weight reuse preserves every serial ISA fold, tail and output guard. */
    TEST(CPUVectorDotRows, StridedUnalignedRowsMatchSerialBytes)
    {
        std::mt19937 random(0x704D07u);
        for (int width : {1, 7, 15, 16, 17, 31, 32, 33, 65, 256, 2048, 2053})
        {
            SCOPED_TRACE(width);
            for (int alignment = 0; alignment < 16; ++alignment)
            {
                SCOPED_TRACE(alignment);
                const int stride = width + 7;
                std::vector<float> weights(width + alignment);
                std::vector<float> rows(4 * stride + alignment);
                fillValues(weights, random);
                fillValues(rows, random);
                for (std::size_t output_stride : {1u, 5u, 257u})
                {
                    std::vector<float> actual(4 * output_stride + 1, -123.5f);
                    auto expected = actual;
                    for (int row = 0; row < 4; ++row)
                        expected[row * output_stride] = primitives::vec_dot(
                            weights.data() + alignment,
                            rows.data() + alignment + row * stride, width);
                    primitives::vec_dot_four_rows(
                        weights.data() + alignment, rows.data() + alignment,
                        width, stride, actual.data(), output_stride);
                    ASSERT_EQ(std::memcmp(actual.data(), expected.data(),
                                          actual.size() * sizeof(float)), 0);
                }
            }
        }
    }

    /** @brief Compare probabilities, selection, normalized weights and Q8 publication. */
    void verifyRouter(int rows, int width, int experts, bool normalize)
    {
        SCOPED_TRACE(::testing::Message() << "M=" << rows << " K=" << width
            << " E=" << experts << " workers=" << omp_get_max_threads()
            << " normalize=" << normalize);
        std::mt19937 random(74u + rows + width + experts);
        std::vector<float> hidden(static_cast<std::size_t>(rows) * width);
        std::vector<float> weights(static_cast<std::size_t>(experts) * width);
        fillValues(hidden, random);
        fillValues(weights, random);
        // Keep the logits within a nonsaturated softmax range while retaining
        // cancellation and varied mantissas in every independent dot product.
        for (float &value : weights)
            value *= 0.0001f;
        constexpr int top_k = 4;
        CPUMoEKernel grouped_kernel, serial_kernel;
        MoERoutingResult grouped, serial;
        ASSERT_TRUE(grouped_kernel.route(hidden.data(), weights.data(), rows,
            width, experts, top_k, normalize, grouped));
        const auto *q8 = grouped_kernel.publishedRouterQ8Hidden(
            hidden.data(), rows, width);
        ASSERT_NE(q8, nullptr);
        const std::size_t blocks = (width + Q8_1Block::BLOCK_SIZE - 1) /
                                  Q8_1Block::BLOCK_SIZE;
        for (int row = 0; row < rows; ++row)
        {
            const float *input = hidden.data() + static_cast<std::size_t>(row) * width;
            ASSERT_TRUE(serial_kernel.route(input, weights.data(), 1, width,
                experts, top_k, normalize, serial));
            ASSERT_EQ(std::memcmp(grouped.router_logits.data() + row * experts,
                serial.router_logits.data(), experts * sizeof(float)), 0);
            ASSERT_EQ(std::memcmp(grouped.expert_indices.data() + row * top_k,
                serial.expert_indices.data(), top_k * sizeof(int)), 0);
            ASSERT_EQ(std::memcmp(grouped.expert_weights.data() + row * top_k,
                serial.expert_weights.data(), top_k * sizeof(float)), 0);
            const auto *serial_q8 = serial_kernel.publishedRouterQ8Hidden(input, 1, width);
            ASSERT_NE(serial_q8, nullptr);
            ASSERT_EQ(std::memcmp(q8 + row * blocks, serial_q8,
                                  blocks * sizeof(Q8_1Block)), 0);
        }
    }

    /** @test Every positive socket worker count preserves ragged grouped routing. */
    TEST(CPUVectorDotRows, RouterThreadAndBatchTotality)
    {
        for (int workers = 1; workers <= 28; ++workers)
        {
            ScopedWorkers scope(workers);
            for (int rows : {1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 65})
                for (bool normalize : {false, true})
                    verifyRouter(rows, 65, 17, normalize);
        }
    }

    /** @test Real prefill geometry and expert-count tails remain serial-row exact. */
    TEST(CPUVectorDotRows, RouterPrefillGeometry)
    {
        for (int workers : {1, 7, 28})
        {
            ScopedWorkers scope(workers);
            for (int experts : {256, 257})
                verifyRouter(512, 2048, experts, true);
        }
    }
}
