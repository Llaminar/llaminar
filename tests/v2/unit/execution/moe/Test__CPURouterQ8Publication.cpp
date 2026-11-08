/**
 * @file Test__CPURouterQ8Publication.cpp
 * @brief Check exact routed-input bytes and observed CPU publication ownership.
 *
 * All quantized expert codebooks consume this one Q8 input ABI. These tests
 * compare indexed/repeated expert-major rows with independent serial public
 * calls, including partial blocks and unused destination capacity. Worker
 * evidence records uniquely quantized source rows, not copied destinations or
 * an advertised OpenMP team size. No
 * throughput threshold belongs in this functional Unit/preflight regression.
 */
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "../../../utils/CPURouterKernelTestAdmission.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
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

/** @brief Build diagnostic representatives without copying quantization arithmetic. */
std::vector<int> representativeRows(std::span<const int> indices)
{
    std::vector<int> result(indices.size());
    for (size_t row = 0; row < indices.size(); ++row)
        result[row] = static_cast<int>(std::find(indices.begin(), indices.begin() + row,
                                               indices[row]) - indices.begin());
    return result;
}

/** @brief Own synthetic original route authority outside the production call. */
struct PublicationSchedule
{
    std::vector<int> sources;
    std::vector<int> original_to_local;
    std::vector<int> inverse;
    std::vector<int> retired_permutation;
    std::optional<CPUMoEKernel::RouterQ8ExpertMajorSchedule> value;
    /** @brief Build a sparse original route table for arbitrary repeated source IDs. */
    PublicationSchedule(std::span<const int> indices, int capacity)
    {
        sources.assign(indices.begin(), indices.end());
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        std::vector<int> counts(sources.size());
        for (int source : indices)
            ++counts[static_cast<size_t>(std::lower_bound(sources.begin(), sources.end(), source) - sources.begin())];
        const int width = counts.empty() ? 1 : *std::max_element(counts.begin(), counts.end());
        original_to_local.assign(sources.size() * static_cast<size_t>(width), -1);
        inverse.resize(indices.size());
        retired_permutation.resize(indices.size());
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t row = 0; row < indices.size(); ++row)
        {
            const size_t compact = static_cast<size_t>(std::lower_bound(
                sources.begin(), sources.end(), indices[row]) - sources.begin());
            original_to_local[compact * static_cast<size_t>(width) +
                static_cast<size_t>(counts[compact]++)] = static_cast<int>(row);
            inverse[row] = retired_permutation[row] = static_cast<int>(row);
        }
        value = CPUMoEKernel::RouterQ8ExpertMajorSchedule::fromOriginalRoutes(
            indices, retired_permutation, original_to_local, inverse, width,
            std::span<const int>(sources), capacity);
    }
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
    admitCPURouterKernelForTest(serial,1,shape.width);
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
    PublicationSchedule schedule(indices, source_rows);
    ASSERT_TRUE(schedule.value);
    PerfStatsCollector::reset();
    ASSERT_TRUE(grouped.publishTransportedRouterQ8HiddenExpertMajor(
        source.data(), source_rows, shape.width,
        *schedule.value,
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
    std::set<int> expected_owners;
    const auto representatives = representativeRows(indices);
    for (int row = 0; row < shape.rows; ++row)
        if (representatives[static_cast<size_t>(row)] == row)
        {
            const int base = shape.rows / expected_workers;
            const int remainder = shape.rows % expected_workers;
            const int first_span = (base + 1) * remainder;
            expected_owners.insert(row < first_span ? row / (base + 1)
                : remainder + (row - first_span) / base);
        }
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
    EXPECT_EQ(completed_rows, std::min(shape.rows, source_rows));
    EXPECT_EQ(owners, expected_owners);
}
} // namespace

/** @test Every small positive team budget preserves tails, duplicates and ownership. */
TEST(CPURouterQ8Publication, SerialAndParallelRowsAreByteExact)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    constexpr std::array cases = {
        PublicationCase{1, 2048, false}, PublicationCase{5, 2048, false},
        PublicationCase{8, 2048, true}, PublicationCase{20, 2051, true},
        PublicationCase{64, 33, false}, PublicationCase{64, 512, false},
        PublicationCase{512, 32, false}, PublicationCase{12, 8192, true}};
    for (int workers = 1; workers <= 8; ++workers)
        for (const auto shape : cases)
            provePublication(shape, workers);
}

/** @test Real socket-sized budgets cannot turn a twenty-row publication serial. */
TEST(CPURouterQ8Publication, PhysicalWorkerBudgetsPublishEveryRow)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    for (int workers = 1; workers <= 31; ++workers)
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
        PublicationSchedule schedule(rows, 2);
        return schedule.value && kernel.publishTransportedRouterQ8HiddenExpertMajor(
            source.data(), 2, width, *schedule.value,
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
    EXPECT_TRUE(invoke({}, output.size()));
    EXPECT_EQ(std::memcmp(output.data(), poisoned.data(), sizeof(output)), 0);
}
/** @test Only unique live rows are quantized across partial, empty and repeated schedules. */
TEST(CPURouterQ8Publication, UniqueLiveSourceReplay)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    constexpr int capacity = 512;
    for (int workers = 1; workers <= 8; ++workers)
    for (int width : {33, 2560})
    {
        RouterQ8WorkerScope thread_scope(workers);
        const size_t blocks = static_cast<size_t>((width + 31) / 32);
        std::vector<float> source(size_t(capacity) * width,
                                  std::numeric_limits<float>::quiet_NaN());
        for (int row : {0, 7, 201, 511})
            for (int col = 0; col < width; ++col)
                source[size_t(row) * width + col] = col % 17 == 0 ? -0.0f
                    : float((row + col * 71) % 257 - 128) * 0.003125f;
        std::vector<Q8_1Block> output(40u * blocks + 3u);
        CPUMoEKernel kernel;
        // Different partial maps cannot inherit a representative from a prior
        // larger publication. Empty work touches neither source capacity nor output.
        const std::vector<std::vector<int>> traces{
            {511, 0, 7, 511, 201, 7, 0, 201, 511, 0, 7, 201},
            {7, 7, 7}, {}, {201, 0, 201, 511, 0, 7, 7},
            {511, 0, 7, 511, 201, 7, 0, 201, 511, 0, 7, 201}};
        for (const auto &indices : traces)
        {
            PublicationSchedule schedule(indices, capacity);
            ASSERT_TRUE(schedule.value);
            std::vector<Q8_1Block> expected(indices.size() * blocks);
            CPUMoEKernel serial;
            admitCPURouterKernelForTest(serial, 1, width);
            // The former indexed publication's per-destination public serial
            // walk is a diagnostic oracle, never a production execution path.
            for (size_t row = 0; row < indices.size(); ++row)
            {
                const float *input = source.data() + size_t(indices[row]) * width;
                ASSERT_TRUE(serial.publishTransportedRouterQ8Hidden(input, 1, width));
                std::copy_n(serial.publishedRouterQ8Hidden(input, 1, width), blocks,
                            expected.data() + row * blocks);
            }
            std::memset(output.data(), 0xa5, output.size() * sizeof(Q8_1Block));
            const auto poison = output;
            PerfStatsCollector::reset();
            ASSERT_TRUE(kernel.publishTransportedRouterQ8HiddenExpertMajor(
                source.data(), capacity, width, *schedule.value, output));
            if (!expected.empty())
                EXPECT_EQ(std::memcmp(output.data(), expected.data(),
                                      expected.size() * sizeof(Q8_1Block)), 0);
            EXPECT_EQ(std::memcmp(output.data() + expected.size(),
                                  poison.data() + expected.size(),
                                  (output.size() - expected.size()) * sizeof(Q8_1Block)), 0);
            double quantized_rows = 0;
            for (const auto &record : PerfStatsCollector::snapshot(
                     {"kernel.cpu_moe_router_q8_worker_rows"}))
            {
                quantized_rows += record.value;
                EXPECT_EQ(record.tags.at("workers"), "1")
                    << "Duplicated destination count cannot force a tiny unique quantization team";
            }
            EXPECT_EQ(quantized_rows, std::set<int>(indices.begin(), indices.end()).size());
        }
    }
}

/** @test Original authority rejects split, missing, foreign and noninjective schedules. */
TEST(CPURouterQ8Publication, InvalidOriginalRoutesRejectBeforeWrites)
{
    const std::array<int, 4> indices{2, 0, 2, 0};
    const std::array<int, 2> sources{0, 2};
    const std::array<int, 4> inverse{0, 1, 2, 3};
    std::array<Q8_1Block, 8> output;
    const std::array<std::vector<int>, 6> invalid_routes{{
        {1, 3, 0}, {1, 3, 0, -1}, {-2, 3, 0, 2},
        {1, 3, 0, 4}, {1, 3, 0, 0}, {0, 3, 1, 2}}};
    for (const auto &routes : invalid_routes)
    {
        std::array<int, 4> scratch{0, 1, 2, 3};
        std::memset(output.data(), 0xa5, sizeof(output));
        const auto poison = output;
        EXPECT_FALSE(CPUMoEKernel::RouterQ8ExpertMajorSchedule::fromOriginalRoutes(
            indices, scratch, routes, inverse, 2, std::span<const int>(sources), 3));
        EXPECT_EQ(std::memcmp(output.data(), poison.data(), sizeof(output)), 0);
    }
    const std::array<int, 4> routes{1, 3, 0, 2};
    for (const std::array<int, 2> bad_sources :
         {std::array<int, 2>{2, 0}, std::array<int, 2>{0, 0}, std::array<int, 2>{0, 3}})
    {
        std::array<int, 4> scratch{0, 1, 2, 3};
        EXPECT_FALSE(CPUMoEKernel::RouterQ8ExpertMajorSchedule::fromOriginalRoutes(
            indices, scratch, routes, inverse, 2, std::span<const int>(bad_sources), 3));
    }
    std::array<int, 4> scratch{0, 1, 2, 3};
    EXPECT_FALSE(CPUMoEKernel::RouterQ8ExpertMajorSchedule::fromOriginalRoutes(
        indices, scratch, routes, std::array<int, 4>{1, 0, 2, 3}, 2,
        std::span<const int>(sources), 3));
    // Original top-k order can select a root later in expert-major order.
    scratch = {0, 1, 2, 3};
    const auto schedule = CPUMoEKernel::RouterQ8ExpertMajorSchedule::fromOriginalRoutes(
        indices, scratch, std::array<int, 4>{3, 1, 2, 0}, inverse, 2,
        std::span<const int>(sources), 3);
    ASSERT_TRUE(schedule);
    EXPECT_EQ(scratch, (std::array<int, 4>{2, 3, 2, 3}));
}

/** @test Scratch aliases fail before retiring any metadata, including partial overlap. */
TEST(CPURouterQ8Publication, MetadataAliasesRejectBeforeScratchMutation)
{
    const std::array<int, 4> indices{2, 0, 2, 0};
    const std::array<int, 4> routes{1, 3, 0, 2};
    const std::array<int, 4> inverse{0, 1, 2, 3};
    const std::array<int, 2> sources{0, 2};
    for (int input = 0; input < 4; ++input)
    for (int offset : {-1, 0, 1})
    {
        SCOPED_TRACE("input=" + std::to_string(input) + " offset=" + std::to_string(offset));
        std::array<int, 20> storage;
        storage.fill(0x1234);
        std::copy(inverse.begin(), inverse.end(), storage.begin() + 5);
        const auto before = storage;
        const std::span<int> scratch(storage.data() + 5, 4u);
        const std::span<const int> alias(storage.data() + 5 + offset, input == 3 ? 2u : 4u);
        EXPECT_FALSE(CPUMoEKernel::RouterQ8ExpertMajorSchedule::fromOriginalRoutes(
            input == 0 ? alias : std::span<const int>(indices), scratch,
            input == 1 ? alias : std::span<const int>(routes),
            input == 2 ? alias : std::span<const int>(inverse), 2,
            input == 3 ? alias : std::span<const int>(sources), 3));
        EXPECT_EQ(storage, before);
    }
    // Half-open intervals that only touch at their endpoints remain legal.
    for (int offset : {1, 9})
    {
        std::array<int, 20> storage;
        storage.fill(0x1234);
        std::copy(inverse.begin(), inverse.end(), storage.begin() + 5);
        std::copy(indices.begin(), indices.end(), storage.begin() + offset);
        const auto before = storage;
        const std::span<const int> adjacent(storage.data() + offset, 4u);
        const auto schedule = CPUMoEKernel::RouterQ8ExpertMajorSchedule::fromOriginalRoutes(
            adjacent, {storage.data() + 5, 4u}, routes, inverse, 2,
            std::span<const int>(sources), 3);
        ASSERT_TRUE(schedule);
        EXPECT_TRUE(std::equal(adjacent.begin(), adjacent.end(), before.begin() + offset));
    }
}

/** @test A tiny publication uses an existing team collectively without repeated serial stores. */
TEST(CPURouterQ8Publication, TinyNestedTeamReplayIsByteExactAndQuantizesOnce)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    RouterQ8WorkerScope thread_scope(4);
    constexpr int capacity = 512, width = 33;
    const std::array<int, 8> indices{0, 7, 0, 511, 7, 201, 0, 201};
    std::vector<float> source(size_t(capacity) * width, -0.25f);
    PublicationSchedule schedule(indices, capacity);
    ASSERT_TRUE(schedule.value);
    std::vector<Q8_1Block> output(indices.size() * 2u), expected(output.size());
    CPUMoEKernel serial;
    admitCPURouterKernelForTest(serial, 1, width);
    for (size_t row = 0; row < indices.size(); ++row)
    {
        const float *input = source.data() + size_t(indices[row]) * width;
        ASSERT_TRUE(serial.publishTransportedRouterQ8Hidden(input, 1, width));
        std::copy_n(serial.publishedRouterQ8Hidden(input, 1, width), 2,
                    expected.data() + row * 2u);
    }
    CPUMoEKernel kernel;
    for (int replay = 0; replay < 3; ++replay)
    {
        std::memset(output.data(), 0xa5, output.size() * sizeof(Q8_1Block));
        std::array<int, 4> completed{};
        PerfStatsCollector::reset();
#pragma omp parallel num_threads(4)
        {
            completed[static_cast<size_t>(omp_get_thread_num())] =
                kernel.publishTransportedRouterQ8HiddenExpertMajor(
                    source.data(), capacity, width, *schedule.value, output) ? 1 : 0;
        }
        EXPECT_TRUE(std::all_of(completed.begin(), completed.end(), [](int value) { return value == 1; }));
        EXPECT_EQ(std::memcmp(output.data(), expected.data(), output.size() * sizeof(Q8_1Block)), 0);
        double quantized_rows = 0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_moe_router_q8_worker_rows"}))
        {
            EXPECT_EQ(record.tags.at("workers"), "4");
            quantized_rows += record.value;
        }
        EXPECT_EQ(quantized_rows, 4.0);
    }
}

/** @test Existing completed-row evidence proves actual quantization follows unique live sources. */
TEST(CPURouterQ8Publication, UniqueLiveSourceQuantizesOnce)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    RouterQ8WorkerScope thread_scope(4);
    constexpr int capacity = 512, width = 2560;
    std::vector<float> source(size_t(capacity) * width, 0.25f);
    std::vector<int> indices(32);
    for (size_t row = 0; row < indices.size(); ++row)
        indices[row] = std::array<int, 4>{0, 7, 201, 511}[row % 4u];
    PublicationSchedule schedule(indices, capacity);
    ASSERT_TRUE(schedule.value);
    std::vector<Q8_1Block> output(indices.size() * (width / 32));
    PerfStatsCollector::reset();
    CPUMoEKernel kernel;
    ASSERT_TRUE(kernel.publishTransportedRouterQ8HiddenExpertMajor(
        source.data(), capacity, width, *schedule.value, output));
    double quantized_rows = 0;
    for (const auto &record : PerfStatsCollector::snapshot(
             {"kernel.cpu_moe_router_q8_worker_rows"}))
        quantized_rows += record.value;
    EXPECT_EQ(quantized_rows, 4.0)
        << "Original indexed publication quantizes32 routed rows for only4 live source rows";
}

/** @brief Grouped demand is exact, cold, and independent of payload lifetime. */
TEST(CPURouterQ8Publication, GroupedConsumerDemandIsColdExactAndSurvivesRevocation)
{
    MoERouterQ8HiddenPublication publication;
    EXPECT_FALSE(publication.hasRequiredConsumer());
    EXPECT_TRUE(publication.acceptsProducerRows(1, 2560));
    EXPECT_FALSE(publication.requireConsumerRows(0, 2560));
    EXPECT_FALSE(publication.requireConsumerRows(-1, 2560));
    EXPECT_FALSE(publication.requireConsumerRows(3, 0));
    EXPECT_FALSE(publication.requireConsumerRows(3, -32));
    EXPECT_FALSE(publication.requireConsumerRows(3, 2559));
    EXPECT_FALSE(publication.hasRequiredConsumer());
    ASSERT_TRUE(publication.requireConsumerRows(3, 2560));
    EXPECT_TRUE(publication.requireConsumerRows(3, 2560));
    EXPECT_FALSE(publication.requireConsumerRows(1, 2560));
    EXPECT_FALSE(publication.requireConsumerRows(3, 1280));
    EXPECT_FALSE(publication.acceptsProducerRows(1, 2560));
    EXPECT_FALSE(publication.acceptsProducerRows(3, 1280));
    EXPECT_TRUE(publication.acceptsProducerRows(3, 2560));

    const float source = 1.0f;
    publication.source_rows = &source;
    EXPECT_FALSE(publication.requireConsumerRows(3, 2560));
    publication.clearPayload();
    EXPECT_TRUE(publication.hasRequiredConsumer());
    EXPECT_TRUE(publication.acceptsProducerRows(3, 2560));
    EXPECT_FALSE(publication.acceptsProducerRows(1, 2560));

    // Each ordinary layer graph owns a fresh publication, not this verifier's demand.
    MoERoutedPipelineKernelOwner ordinary_graph;
    EXPECT_FALSE(ordinary_graph.router_q8_publication->hasRequiredConsumer());
    EXPECT_TRUE(ordinary_graph.router_q8_publication->acceptsProducerRows(1, 2560));
}

} // namespace llaminar2::test
