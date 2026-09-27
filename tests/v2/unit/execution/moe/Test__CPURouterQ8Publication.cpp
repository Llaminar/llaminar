/**
 * @file Test__CPURouterQ8Publication.cpp
 * @brief Check exact routed-input bytes and observed CPU publication ownership.
 *
 * All quantized expert codebooks consume this one Q8 input ABI. These tests
 * compare indexed/repeated expert-major rows with independent serial public
 * calls, including partial blocks and unused destination capacity. Worker
 * evidence records completed rows, not an advertised OpenMP team size. No
 * throughput threshold belongs in this functional Unit/preflight regression.
 */
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @brief Restore the caller's thread policy after a synchronous test scope. */
class RouterQ8WorkerScope final
{
public:
    /** @brief Use a fixed team so observed ownership has a precise meaning. */
    explicit RouterQ8WorkerScope(int workers)
        : previous_workers_(omp_get_max_threads()), previous_dynamic_(omp_get_dynamic())
    {
        omp_set_dynamic(0);
        omp_set_num_threads(workers);
    }
    /** @brief Restore policy even when an assertion aborts the current case. */
    ~RouterQ8WorkerScope()
    {
        omp_set_num_threads(previous_workers_);
        omp_set_dynamic(previous_dynamic_);
    }
    RouterQ8WorkerScope(const RouterQ8WorkerScope &) = delete;
    RouterQ8WorkerScope &operator=(const RouterQ8WorkerScope &) = delete;
private:
    int previous_workers_;
    int previous_dynamic_;
};

/** @brief Describe both tiny serial publications and amortized team workloads. */
struct PublicationCase
{
    int rows;
    int width;
    bool uses_team;
};

/**
 * @brief Prove one public indexed publication at an explicit worker budget.
 * @param shape Active route geometry, not the reserved source capacity.
 * @param workers Configured positive OpenMP team budget.
 */
void provePublication(PublicationCase shape, int workers)
{
    SCOPED_TRACE("rows=" + std::to_string(shape.rows) + " width=" +
                 std::to_string(shape.width) + " workers=" + std::to_string(workers));
    RouterQ8WorkerScope thread_scope(workers);
    constexpr int source_rows = 19;
    const int blocks = (shape.width + 31) / 32;
    std::vector<float> source(size_t(source_rows) * shape.width);
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = float(int((i * 71 + i / 13) % 257) - 128) * 0.003125f;
    std::vector<int> indices(shape.rows);
    for (int row = 0; row < shape.rows; ++row)
        indices[row] = (row * 7 + 3) % source_rows;

    // Each reference row uses the same public serial boundary as ordinary
    // decode. Independent block-arithmetic oracles live in the rounding gate.
    CPUMoEKernel serial;
    std::vector<Q8_1Block> expected(size_t(source_rows) * blocks);
    for (int row = 0; row < source_rows; ++row)
    {
        const float *input = source.data() + size_t(row) * shape.width;
        ASSERT_TRUE(serial.publishTransportedRouterQ8Hidden(input, 1, shape.width));
        const auto *published = serial.publishedRouterQ8Hidden(input, 1, shape.width);
        ASSERT_NE(published, nullptr);
        std::copy_n(published, blocks, expected.data() + size_t(row) * blocks);
    }

    const size_t active_blocks = size_t(shape.rows) * blocks;
    std::vector<Q8_1Block> output(active_blocks + 3u);
    std::memset(output.data(), 0xa5, output.size() * sizeof(Q8_1Block));
    const auto poisoned = output;
    CPUMoEKernel grouped;
    PerfStatsCollector::reset();
    ASSERT_TRUE(grouped.publishTransportedRouterQ8HiddenExpertMajor(
        source.data(), source_rows, shape.width, indices,
        std::span<Q8_1Block>(output.data(), active_blocks)));
    for (int row = 0; row < shape.rows; ++row)
        ASSERT_EQ(std::memcmp(output.data() + size_t(row) * blocks,
                              expected.data() + size_t(indices[row]) * blocks,
                              size_t(blocks) * sizeof(Q8_1Block)), 0);
    EXPECT_EQ(std::memcmp(output.data() + active_blocks,
                          poisoned.data() + active_blocks, 3u * sizeof(Q8_1Block)), 0);
    EXPECT_EQ(grouped.publishedRouterQ8Hidden(source.data(), shape.rows, shape.width), nullptr);

    const int expected_workers = shape.uses_team ? workers : 1;
    std::set<int> owners;
    double completed_rows = 0;
    for (const auto &record : PerfStatsCollector::snapshot(
             {"kernel.cpu_moe_router_q8_worker_rows"}))
    {
        EXPECT_EQ(record.tags.at("workers"), std::to_string(expected_workers));
        EXPECT_EQ(record.tags.at("layout"), "indexed");
        EXPECT_EQ(record.tags.at("rows"), std::to_string(shape.rows));
        owners.insert(std::stoi(record.tags.at("worker")));
        completed_rows += record.value;
    }
    EXPECT_EQ(completed_rows, shape.rows);
    EXPECT_EQ(owners.size(), size_t(std::min(shape.rows, expected_workers)));
}
} // namespace

/** @test Every small positive team budget preserves tails, duplicates and ownership. */
TEST(CPURouterQ8Publication, SerialAndParallelRowsAreByteExact)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    constexpr std::array cases = {
        PublicationCase{1, 2048, false}, PublicationCase{5, 2048, false},
        PublicationCase{8, 2048, true}, PublicationCase{20, 2051, true},
        PublicationCase{64, 33, false}, PublicationCase{64, 512, true},
        PublicationCase{512, 32, true}, PublicationCase{12, 8192, true}};
    for (int workers = 1; workers <= 8; ++workers)
        for (const auto shape : cases)
            provePublication(shape, workers);
}

/** @test Real socket-sized budgets cannot turn a twenty-row publication serial. */
TEST(CPURouterQ8Publication, PhysicalWorkerBudgetsPublishEveryRow)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    for (int workers : {14, 28, 31})
    {
        provePublication({20, 2048, true}, workers);
        provePublication({31, 2051, true}, workers);
    }
}

/** @test Malformed routes and short destinations fail before any byte is written. */
TEST(CPURouterQ8Publication, InvalidPublicationLeavesDestinationUntouched)
{
    constexpr int width = 33;
    std::vector<float> source(2u * width, 0.5f);
    std::array<int, 3> indices = {0, 1, 0};
    std::array<Q8_1Block, 6> output;
    std::memset(output.data(), 0xa5, sizeof(output));
    const auto poisoned = output;
    CPUMoEKernel kernel;
    const auto invoke = [&](std::span<const int> rows, size_t capacity)
    {
        return kernel.publishTransportedRouterQ8HiddenExpertMajor(
            source.data(), 2, width, rows,
            std::span<Q8_1Block>(output.data(), capacity));
    };
    EXPECT_FALSE(invoke(indices, 5));
    EXPECT_EQ(std::memcmp(output.data(), poisoned.data(), sizeof(output)), 0);
    for (int bad_index : {-1, 2})
    {
        indices[2] = bad_index;
        EXPECT_FALSE(invoke(indices, output.size()));
        EXPECT_EQ(std::memcmp(output.data(), poisoned.data(), sizeof(output)), 0);
    }
    EXPECT_FALSE(invoke({}, output.size()));
    EXPECT_EQ(std::memcmp(output.data(), poisoned.data(), sizeof(output)), 0);
}
} // namespace llaminar2::test
