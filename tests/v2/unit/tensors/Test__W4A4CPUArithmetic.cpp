/**
 * @file Test__W4A4CPUArithmetic.cpp
 * @brief CPU W4A4 grouped-row equivalence, live-workspace and admission contracts.
 *
 * Recorded numerical replay lives in W4A4CPUReplayDriver. These synthetic
 * contracts compare the production grouped operation with its serial-row
 * diagnostic oracle, stress large-to-small reuse, and protect inactive bytes.
 * The fixtures own only small raw arrays; no model, backend discovery or GPU is
 * involved. Production execution never invokes the serial-row diagnostic loop.
 */
#include "kernels/cpu/gemm/CPUW4A4Arithmetic.h"
#include "utils/Assertions.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cstring>
#include <vector>
#include <omp.h>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr int kOutputs = 7, kWidth = 512, kMaximumRows = 65;
        constexpr int kInputStride = kWidth + 5, kOutputStride = kOutputs + 3;
        constexpr std::uint32_t kGuard = 0x7fc12345;

        /** @return Small native rows with nonuniform group multipliers and row units. */
        std::vector<SourceBlock> makeSource(Transform transform)
        {
            std::vector<SourceBlock> result(kOutputs * (kWidth / kBlockElements));
            for (int row = 0; row < kOutputs; ++row)
                for (int b = 0; b < kWidth / kBlockElements; ++b)
                {
                    auto &block = result[row * (kWidth / kBlockElements) + b];
                    block.unit = (row + 1) * 0.125f;
                    block.descriptor = 0x60004321u;
                    block.rotation_length = transform == Transform::Identity ? kWidth : 128;
                    block.clip_bits = std::bit_cast<std::uint32_t>(0.75f);
                    for (int pair = 0; pair < 128; ++pair)
                        block.codes[pair] = static_cast<std::uint8_t>((pair * 17 + b * 29 + row * 31) & 255);
                }
            return result;
        }

        /** @return An actual admitted fixture, failing construction on any invalid source. */
        SourceView admit(Transform transform, std::span<const SourceBlock> source)
        {
            auto result = SourceView::admit(transform, kOutputs, kWidth, source);
            if (!result) throw std::runtime_error(result.error().message());
            return *result;
        }

        /** @brief Bounded buffers reused across every live row count, including extra guards. */
        struct Fixture
        {
            std::vector<SourceBlock> source; ///< Sole immutable native weight storage.
            SourceView weights;              ///< Policy bound to source's stable address.
            CPUWorkspaceCounts counts;       ///< Canonical element counts, not a private ledger.
            std::vector<float> input;        ///< Strided inputs with poisoned row padding.
            std::vector<ActivationBlock> packed; ///< Packed operand bank plus block guards.
            std::vector<ActivationRange> ranges; ///< Extrema bank plus guards.
            std::vector<float> row_ranges;   ///< Exact unrounded token maxima plus guards.
            std::vector<float> butterfly;    ///< Transform producer bank plus guards.
            std::vector<float> transformed;  ///< Transform consumer bank plus guards.
            std::vector<float> output;       ///< Strided output plus guards.
            std::vector<std::uint8_t> active_rows; ///< Published operand-validity bytes plus guards.

            /** @brief Construct fixed storage before invoking any production arithmetic. */
            explicit Fixture(Transform transform)
                : source(makeSource(transform)), weights(admit(transform, source)),
                  counts(cpuWorkspaceCounts(weights.contract(), kMaximumRows)),
                  input(kMaximumRows * kInputStride, std::bit_cast<float>(kGuard)),
                  packed(counts.blocks + 2), ranges(counts.ranges + 2), row_ranges(counts.row_ranges + 2),
                  butterfly(counts.butterfly_floats + 2), transformed(counts.transformed_floats + 2),
                  output(kMaximumRows * kOutputStride + 2), active_rows(counts.active_rows + 2)
            {
                for (int row = 0; row < kMaximumRows; ++row)
                    for (int k = 0; k < kWidth; ++k)
                        input[row * kInputStride + k] = ((row * 19 + k * 17 + 5) % 97 - 48) * 0.03125f;
            }

            /** @return Only admitted bank capacities; physical guards remain outside each span. */
            CPUActivationBuffers buffers()
            {
                return {std::span(packed).first(counts.blocks), std::span(ranges).first(counts.ranges),
                        std::span(row_ranges).first(counts.row_ranges),
                        std::span(butterfly).first(counts.butterfly_floats),
                        std::span(transformed).first(counts.transformed_floats),
                        std::span(active_rows).first(counts.active_rows)};
            }

            /** @brief Poison all banks so stale values and inactive writes remain observable. */
            void poison()
            {
                std::memset(packed.data(), 0xa5, packed.size() * sizeof(ActivationBlock));
                std::memset(ranges.data(), 0xa5, ranges.size() * sizeof(ActivationRange));
                std::fill(active_rows.begin(), active_rows.end(), std::uint8_t{0xa5});
                for (auto *bank : {&row_ranges, &butterfly, &transformed, &output})
                    std::fill(bank->begin(), bank->end(), std::bit_cast<float>(kGuard));
            }
        };

        /** @brief Restore the caller's OpenMP width even when a test assertion fails. */
        struct ThreadScope
        {
            int previous = omp_get_max_threads(); ///< Incoming process-local worker policy.
            /** @brief Install the selected positive diagnostic worker count. */
            explicit ThreadScope(int threads) { omp_set_num_threads(threads); }
            /** @brief Restore the previous worker policy. */
            ~ThreadScope() { omp_set_num_threads(previous); }
        };

        /** @brief Restore floating-point rounding even after expected admission exceptions. */
        struct RoundingScope
        {
            int previous = std::fegetround(); ///< Incoming arithmetic environment.
            /** @brief Select the deliberately incompatible test environment. */
            RoundingScope() { std::fesetround(FE_UPWARD); }
            /** @brief Restore native round-to-nearest behavior for later tests. */
            ~RoundingScope() { std::fesetround(previous); }
        };
    }

    TEST(W4A4CPUArithmeticContract, EveryGroupedRowCountMatchesSerialBytesAndPreservesInactiveStorage)
    {
        ThreadScope restore(1);
        for (const auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            Fixture fixture(transform);
            std::vector<float> expected(kMaximumRows * kOutputs);
            // Diagnostic oracle only: production grouped work shares decoded
            // weights across rows and never calls this per-row loop.
            for (int row = 0; row < kMaximumRows; ++row)
            {
                const auto input = std::span<const float>(fixture.input).subspan(row * kInputStride, kWidth);
                const auto prepared = prepareCPUActivations(fixture.weights.contract(), input, 1,
                                                             kWidth, fixture.buffers());
                projectCPU(fixture.weights, prepared, std::span(expected).subspan(row * kOutputs, kOutputs), kOutputs);
            }
            for (int threads : {1, 3, 8})
            {
                omp_set_num_threads(threads);
                // Descending counts prove large-to-small live extents, including
                // all four token-tile tails and the empty participant.
                for (int rows = kMaximumRows; rows >= 0; --rows)
                {
                    SCOPED_TRACE(::testing::Message() << "rows=" << rows << " threads=" << threads
                                                     << " transform=" << static_cast<int>(transform));
                    fixture.poison();
                    const auto prepared = prepareCPUActivations(fixture.weights.contract(), fixture.input, rows,
                                                                 kInputStride, fixture.buffers());
                    projectCPU(fixture.weights, prepared, fixture.output, kOutputStride);
                    const auto live = cpuWorkspaceCounts(fixture.weights.contract(), rows);
                    for (int row = 0; row < kMaximumRows; ++row)
                        for (int column = 0; column < kOutputStride; ++column)
                        {
                            const auto wanted = row < rows && column < kOutputs
                                ? std::bit_cast<std::uint32_t>(expected[row * kOutputs + column]) : kGuard;
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output[row * kOutputStride + column]), wanted);
                        }
                    const auto packed_tail = std::as_bytes(std::span(fixture.packed)).subspan(live.blocks * sizeof(ActivationBlock));
                    EXPECT_TRUE(std::all_of(packed_tail.begin(), packed_tail.end(), [](std::byte value) { return value == std::byte{0xa5}; }));
                    const auto range_tail = std::as_bytes(std::span(fixture.ranges)).subspan(live.ranges * sizeof(ActivationRange));
                    EXPECT_TRUE(std::all_of(range_tail.begin(), range_tail.end(), [](std::byte value) { return value == std::byte{0xa5}; }));
                    for (const auto tail : {std::span(fixture.row_ranges).subspan(live.row_ranges),
                                           std::span(fixture.butterfly).subspan(live.butterfly_floats),
                                           std::span(fixture.transformed).subspan(live.transformed_floats)})
                        EXPECT_TRUE(std::all_of(tail.begin(), tail.end(), [](float value) {
                            return std::bit_cast<std::uint32_t>(value) == kGuard;
                        }));
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output[fixture.output.size() - 1]), kGuard);
                }
            }
        }
    }

    TEST(W4A4CPUArithmeticContract, CountedOriginalRowsPreserveHolesAndSealOperandValidity)
    {
        ThreadScope restore(1);
        for (const auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            Fixture fixture(transform);
            const auto original_input = fixture.input;
            const auto full = prepareCPUActivations(fixture.weights.contract(), fixture.input,
                kMaximumRows, kInputStride, fixture.buffers());
            projectCPU(fixture.weights, full, fixture.output, kOutputStride);
            const auto expected = fixture.output;
            const auto expected_packed = fixture.packed;
            for (int threads : {1, 3, 8})
            {
                omp_set_num_threads(threads);
                for (int pattern : {0, 1, 2, 1, 0})
                {
                    // Five original requests share an arena, not compact row
                    // coordinates. Causal frontiers can remove a tile prefix.
                    std::array<std::int32_t, 10> state{0, 13, 0, 0, 5, 9, 3, 1, 7, 12};
                    if (pattern != 1)
                        for (int request = 0; request < 5; ++request)
                        {
                            state[2 * request] = 10;
                            state[2 * request + 1] = pattern == 0 ? 13 : 0;
                        }
                    const auto work = ProjectionRowWork::causalStrided(
                        5, 13, state.data(), state.data() + 1, 2, 100, 5);
                    std::array<bool, kMaximumRows> selected{};
                    fixture.input = original_input;
                    fixture.poison();
                    for (int row = 0; row < kMaximumRows; ++row)
                    {
                        selected[row] = work.resolveHost(work.requestForRow(row)).contains(row);
                        if (!selected[row])
                            std::fill_n(fixture.input.data() + row * kInputStride, kWidth,
                                        std::bit_cast<float>(kGuard));
                    }
                    const auto prepared = prepareCPUActivations(fixture.weights.contract(), fixture.input,
                        kMaximumRows, kInputStride, fixture.buffers(), work);
                    // A4 owns a completed operand set. It must not reread a
                    // subsequently changed controller count to admit stale rows.
                    for (int request = 0; request < 5; ++request) state[2 * request + 1] = 13;
                    projectCPU(fixture.weights, prepared, fixture.output, kOutputStride);
                    for (int row = 0; row < kMaximumRows; ++row)
                    {
                        EXPECT_EQ(prepared.active(row), selected[row]);
                        EXPECT_EQ(fixture.active_rows[row], selected[row] ? 1 : 0);
                        for (int column = 0; column < kOutputStride; ++column)
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.output[row * kOutputStride + column]),
                                selected[row] && column < kOutputs
                                    ? std::bit_cast<std::uint32_t>(expected[row * kOutputStride + column]) : kGuard);
                        const auto blocks = std::span(fixture.packed).subspan(row * 2, 2);
                        if (selected[row])
                            EXPECT_EQ(std::memcmp(blocks.data(), expected_packed.data() + row * 2, blocks.size_bytes()), 0);
                        else
                        {
                            const auto bytes = std::as_bytes(blocks);
                            EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(),
                                [](std::byte value) { return value == std::byte{0xa5}; }));
                            EXPECT_EQ(std::bit_cast<std::uint32_t>(fixture.row_ranges[row]), kGuard);
                            for (const auto *bank : {&fixture.butterfly, &fixture.transformed})
                                if (transform == Transform::PaleySylvester)
                                    for (int k = 0; k < kWidth; ++k)
                                        EXPECT_EQ(std::bit_cast<std::uint32_t>((*bank)[row * kWidth + k]), kGuard);
                        }
                    }
                    EXPECT_EQ(fixture.active_rows[kMaximumRows], 0xa5);
                }
            }
        }
    }

    TEST(W4A4CPUArithmeticContract, InvalidExtentsPolicyOverlapAndRoundingFailBeforePublication)
    {
        ThreadScope threads(1);
        Fixture identity(Transform::Identity), rotated(Transform::PaleySylvester);
        EXPECT_THROW(static_cast<void>(cpuWorkspaceCounts(identity.weights.contract(), -1)), AssertionError);
        auto incomplete = identity.buffers();
        incomplete.blocks = incomplete.blocks.first(0);
        EXPECT_THROW(static_cast<void>(prepareCPUActivations(identity.weights.contract(), identity.input, 1,
                                          kInputStride, incomplete)), AssertionError);
        auto overlapping = rotated.buffers();
        overlapping.transformed = overlapping.butterfly;
        EXPECT_THROW(static_cast<void>(prepareCPUActivations(rotated.weights.contract(), rotated.input, 1,
                                          kInputStride, overlapping)), AssertionError);
        EXPECT_THROW(static_cast<void>(prepareCPUActivations(identity.weights.contract(), identity.input, 1,
                                          kWidth - 1, identity.buffers())), AssertionError);
        std::array<std::int32_t, 2> state{0, 1};
        const auto work = ProjectionRowWork::causalStrided(1, 1, state.data(), state.data() + 1, 2, 1, 0);
        auto no_validity = identity.buffers();
        no_validity.active_rows = {};
        EXPECT_THROW(static_cast<void>(prepareCPUActivations(identity.weights.contract(), identity.input, 1,
                                          kInputStride, no_validity, work)), AssertionError);
        EXPECT_THROW(static_cast<void>(prepareCPUActivations(identity.weights.contract(), identity.input, 2,
                                          kInputStride, identity.buffers(), work)), AssertionError);
        state[1] = 2;
        EXPECT_THROW(static_cast<void>(prepareCPUActivations(identity.weights.contract(), identity.input, 1,
                                          kInputStride, identity.buffers(), work)), std::invalid_argument);
        const auto prepared = prepareCPUActivations(identity.weights.contract(), identity.input, 1,
                                                     kInputStride, identity.buffers());
        EXPECT_THROW(projectCPU(rotated.weights, prepared, rotated.output, kOutputStride), AssertionError);
        EXPECT_THROW(projectCPU(identity.weights, prepared, std::span<float>{}, kOutputStride), AssertionError);
        {
            RoundingScope rounding;
            EXPECT_THROW(static_cast<void>(prepareCPUActivations(identity.weights.contract(), identity.input, 1,
                                              kInputStride, identity.buffers())), AssertionError);
            EXPECT_THROW(projectCPU(identity.weights, prepared, identity.output, kOutputStride), AssertionError);
        }
    }
} // namespace llaminar2::w4a4
