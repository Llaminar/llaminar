/**
 * @file Test__MoEOverlayCanonicalHostReturn.cpp
 * @brief Placement-invariant host return publication and final-fold regression.
 *
 * The cancellation fixture exposes the real defect: locally summing experts
 * changes FP32 parenthesization after movement. Exercise the production packed
 * gather and final stage under every two/three-participant ownership map and
 * packet arrival order, including missing and duplicate publication failures.
 * Parallel prefill packing must preserve every payload bit, append boundaries,
 * and count publication under odd worker budgets and nested single callers.
 */

#include "execution/moe/MoEOverlayCanonicalHostReturn.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Execute the installed packed root fold, not a test-only reducer. */
    bool fold(FP32Tensor &bank, FP32Tensor &weights, FP32Tensor &output)
    {
        MoECanonicalRouteReduceStage::Params params;
        params.device_id = DeviceId::cpu();
        params.canonical_route_contributions = &bank;
        params.routing_weights = &weights;
        params.output = &output;
        params.seq_len = 1;
        params.top_k = 4;
        params.d_model = 33;
        params.canonical_route_arithmetic = MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA;
        params.canonical_route_layout = MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows;
        params.reduction_role = MoECanonicalRouteReductionRole::RootOwner;
        MoECanonicalRouteReduceStage stage(params);
        llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
        return stage.execute(&ctx);
    }
}

TEST(MoEOverlayCanonicalHostReturn, EveryPlacementAndArrivalOrderHasSerialBytes)
{
    constexpr int columns = 33; // Exercises vector bodies and the scalar tail.
    const std::array<float, 4> expert_values{0x1p24f, 1.0f, -0x1p24f, 1.0f};
    FP32Tensor bank(std::vector<size_t>{4 * (columns + 1) + 1});
    FP32Tensor weights(std::vector<size_t>{1, 4});
    FP32Tensor output(std::vector<size_t>{1, columns});
    std::fill_n(weights.mutable_data(), 4, 1.0f);
    float serial = 0.0f;
    for (float value : expert_values)
        serial = std::fma(value, 1.0f, serial);
    ASSERT_EQ(serial, 1.0f);
    // Moving the second route beside the fourth changes the old result to 2.
    ASSERT_NE((expert_values[0] + expert_values[2]) +
                  (expert_values[1] + expert_values[3]), serial);

    for (int participants : {2, 3})
    {
        int maps = 1;
        for (int slot = 0; slot < 4; ++slot) maps *= participants;
        for (int map = 0; map < maps; ++map)
        {
            std::vector<int> order(participants);
            std::iota(order.begin(), order.end(), 0);
            do
            {
                auto boundary = MoEOverlayCanonicalGatherBoundary::Begin;
                for (int participant : order)
                {
                    std::array<int32_t, 4> slots{};
                    std::array<float, 4 * columns> rows{};
                    size_t count = 0;
                    int assignment = map;
                    for (int slot = 0; slot < 4; ++slot)
                    {
                        const int owner = assignment % participants;
                        assignment /= participants;
                        if (owner != participant) continue;
                        slots[count] = slot;
                        std::fill_n(rows.data() + count * columns, columns, expert_values[slot]);
                        ++count;
                    }
                    MoEOverlayReturnRows packet;
                    packet.residency_epoch = 1u + static_cast<uint64_t>(map);
                    packet.d_model = columns;
                    packet.layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
                    packet.live_row_count = count;
                    packet.row_capacity = 4;
                    packet.row_ids_host = slots.data();
                    packet.output_rows_fp32 = rows.data();
                    ASSERT_TRUE(gatherMoEOverlayCanonicalHostReturn(
                        packet, {bank.mutable_data(), bank.numel()}, boundary));
                    boundary = MoEOverlayCanonicalGatherBoundary::Append;
                }
                ASSERT_TRUE(fold(bank, weights, output));
                for (int column = 0; column < columns; ++column)
                    ASSERT_EQ(std::memcmp(output.data() + column, &serial, sizeof(float)), 0);
            } while (std::next_permutation(order.begin(), order.end()));
        }
    }
}

TEST(MoEOverlayCanonicalHostReturn, InvalidAppendDoesNotMutatePublication)
{
    std::array<float, 5> bank{7, 7, 7, 7, 7};
    const auto before = bank;
    std::array<int32_t, 2> slots{0, 1};
    std::array<float, 4> values{};
    MoEOverlayReturnRows packet;
    packet.d_model = 2;
    packet.layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
    packet.residency_epoch = 1;
    packet.live_row_count = packet.row_capacity = 2;
    packet.row_ids_host = slots.data();
    packet.output_rows_fp32 = values.data();
    EXPECT_FALSE(gatherMoEOverlayCanonicalHostReturn(packet, bank, MoEOverlayCanonicalGatherBoundary::Begin));
    EXPECT_EQ(bank, before);
    packet.live_row_count = 1;
    packet.layout = MoEOverlayReturnLayout::ParticipantTokenPartials;
    EXPECT_FALSE(gatherMoEOverlayCanonicalHostReturn(packet, bank, MoEOverlayCanonicalGatherBoundary::Begin));
    EXPECT_EQ(bank, before);
}

TEST(MoEOverlayCanonicalHostReturn, FinalReducerRejectsMissingAndDuplicateSlots)
{
    FP32Tensor bank(std::vector<size_t>{4 * 34 + 1});
    FP32Tensor weights(std::vector<size_t>{1, 4});
    FP32Tensor output(std::vector<size_t>{1, 33});
    std::fill_n(weights.mutable_data(), 4, 1.0f);
    std::fill_n(output.mutable_data(), 33, -77.0f);
    std::array<int32_t, 4> slots{0, 1, 2, 2};
    std::array<float, 4 * 33> values{};
    MoEOverlayReturnRows packet;
    packet.d_model = 33;
    packet.layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
    packet.residency_epoch = 1;
    packet.row_capacity = 4;
    packet.row_ids_host = slots.data();
    packet.output_rows_fp32 = values.data();
    for (size_t count : {3u, 4u})
    {
        packet.live_row_count = count;
        ASSERT_TRUE(gatherMoEOverlayCanonicalHostReturn(
            packet, {bank.mutable_data(), bank.numel()}, MoEOverlayCanonicalGatherBoundary::Begin));
        EXPECT_FALSE(fold(bank, weights, output));
        for (int column = 0; column < 33; ++column)
            EXPECT_EQ(output.data()[column], -77.0f);
    }
}

/** @test Parallel packing preserves row bits, appended identity and untouched tail capacity. */
TEST(MoEOverlayCanonicalHostReturn, PrefillPackingIsByteExactForEveryWorkerBudget)
{
    namespace record = canonical_moe_route_record;
    const int original_threads = omp_get_max_threads();
    for (int threads : {1, 2, 3, 4, 5, 6, 7})
    for (size_t rows : {1u, 15u, 128u, 512u})
    for (size_t width : {33u, 2048u})
    {
        SCOPED_TRACE(::testing::Message() << threads << " workers, rows=" << rows << " width=" << width);
        omp_set_num_threads(threads);
        std::vector<float> payload(rows * width);
        std::vector<int32_t> slots(rows);
        for (size_t index = 0; index < payload.size(); ++index)
            payload[index] = std::bit_cast<float>(0x7fc00000u + static_cast<uint32_t>(index % 65536u));
        for (size_t row = 0; row < rows; ++row)
            slots[row] = static_cast<int32_t>(3 * row + 1);
        std::vector<float> expected((2 * rows + 1) * (width + 1) + 1, -99.0f);
        auto actual = expected;
        MoEOverlayReturnRows packet;
        packet.d_model = static_cast<int>(width);
        packet.layout = MoEOverlayReturnLayout::CanonicalExpertRoutes;
        packet.residency_epoch = 1;
        packet.live_row_count = packet.row_capacity = rows;
        packet.row_ids_host = slots.data();
        packet.output_rows_fp32 = payload.data();
        for (size_t append = 0; append < 2; ++append)
        {
            for (size_t row = 0; row < rows; ++row)
            {
                float *destination = record::record(expected.data(), append * rows + row, width);
                std::memcpy(destination, payload.data() + row * width, width * sizeof(float));
                record::writeFlatRouteSlot(destination, width, slots[row]);
            }
            EXPECT_TRUE(record::writeRecordCount(expected.data(), expected.size(), (append + 1) * rows));
            EXPECT_TRUE(gatherMoEOverlayCanonicalHostReturn(packet, actual,
                append == 0 ? MoEOverlayCanonicalGatherBoundary::Begin : MoEOverlayCanonicalGatherBoundary::Append));
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
    }
    omp_set_num_threads(original_threads);
}

/** @test One caller inside an existing team cannot introduce an orphaned OpenMP workshare. */
TEST(MoEOverlayCanonicalHostReturn, RowWorkJoinsAndPreservesNestedSingleCaller)
{
    const int original_threads = omp_get_max_threads();
    omp_set_num_threads(3);
    std::array<std::atomic<unsigned>, 512> visits{};
#pragma omp parallel
    {
#pragma omp single
        {
            forEachMoEOverlayHostRow(visits.size(), 8192, [&](size_t row) {
                visits[row].fetch_add(1, std::memory_order_relaxed);
            });
        }
    }
    for (auto &visits_for_row : visits)
        EXPECT_EQ(visits_for_row.load(), 1u);
    forEachMoEOverlayHostRow(visits.size(), 8192, [&](size_t row) {
        visits[row].fetch_add(1, std::memory_order_relaxed);
    });
    for (auto &visits_for_row : visits)
        EXPECT_EQ(visits_for_row.load(), 2u);
    omp_set_num_threads(original_threads);
}
