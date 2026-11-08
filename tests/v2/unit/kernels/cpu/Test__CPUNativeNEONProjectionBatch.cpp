/**
 * @file Test__CPUNativeNEONProjectionBatch.cpp
 * @brief Native ARM shared-row scheduling against actual native serial-row bytes.
 *
 * Canonical byte fixtures cover every source/encoding and both arithmetic
 * policies. They are not prepared-owner or serving certification. Request
 * slicing, repeated rows, dirty inactive data, bias and output padding retain
 * original coordinates across standalone and retained OpenMP workshares.
 */
#include "kernels/cpu/gemm/CPUNativeNEONProjectionBatch.h"
#include "kernels/cpu/gemm/CPUNativeNEONExactOverlay.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "utils/NativeVNNIPreparedByteFixture.h"
#include "utils/CPUExplicitRoundingOracle.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <chrono>
#include <atomic>
#include <thread>
#include <span>
#include <tuple>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;
using Dot = multi_scale::NEONByteDot;

/** @brief Restore the caller's OpenMP budget and dynamic-team policy after each test. */
class TeamBudget final
{
public:
    /** @brief Save caller policy; request exact positive worker counts in the test. */
    TeamBudget() : threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
    { omp_set_dynamic(0); omp_set_num_threads(1); }
    /** @brief Restore both policies, including on an assertion/exception exit. */
    ~TeamBudget() { omp_set_num_threads(threads_); omp_set_dynamic(dynamic_); }
private:
    int threads_, dynamic_;
};

/** @brief Admit the complete compiler envelope before any native projection. */
class CPUNativeNEONProjectionBatchTest : public ::testing::Test
{
protected:
    /** @brief Missing instructions fail rather than skip an advertised native route. */
    static void SetUpTestSuite()
    {
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::ASIMD);
#if defined(__ARM_FEATURE_DOTPROD)
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::DotProd);
#endif
    }
};

/** @brief Produce deterministic complete Q8 rows, including negative signed-byte extrema.
 * @param rows Physical row capacity, not useful rows.
 * @param blocks Complete physical K blocks per row.
 * @return One retained original-coordinate activation bank.
 */
std::vector<Q8_1Block> inputs(int rows, int blocks)
{
    std::vector<Q8_1Block> bank(std::size_t(rows) * blocks);
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
        {
            auto &q = bank[std::size_t(row) * blocks + block];
            q.d = std::uint16_t(0x3401 + (row * 71 + block * 37) % 512);
            int sum = 0;
            for (int k = 0; k < 32; ++k)
            {
                q.qs[k] = std::int8_t((row * 37 + block * 13 + k * 73) % 256 - 128);
                sum += q.qs[k];
            }
            q.sum_qs = std::int16_t(sum);
        }
    return bank;
}

/** @brief Author expected useful rows through real native M=1 tiles only.
 * @tparam Instruction Native instruction family under test.
 * @param weights Checked immutable source/encoding/policy publication.
 * @param input Original Q8 bank, including dirty inactive rows.
 * @param output Destination already containing canaries in all non-useful bytes.
 * @param rows Physical M capacity.
 * @param stride Original output stride.
 * @param work Optional original-coordinate publication.
 * @param bias Optional complete immutable bias row.
 */
template <Dot Instruction> void serial(CPUNativeVNNIWeightView weights,
    std::span<const Q8_1Block> input, std::span<float> output, int rows, int stride,
    const ProjectionRowWork *work = nullptr, std::span<const float> bias = {})
{
    for (int row = 0; row < rows; ++row)
    {
        if (work && !work->resolveHost(work->requestForRow(row)).contains(row)) continue;
        for (int chunk = 0; chunk < weights.N_padded / 64; ++chunk)
        {
            const int columns = std::min(64, weights.N - chunk * 64);
            const auto in = input.subspan(std::size_t(row) * weights.blocks_per_row, weights.blocks_per_row);
            const auto out = output.subspan(std::size_t(row) * stride + chunk * 64, columns);
            CPUNativeNEONProjectionTile<1>::bind(weights, {in}, {out}, chunk).template execute<Instruction>();
            if (!bias.empty())
                for (int column = 0; column < columns; ++column)
                    out[column] = memoryRoundedAdd(out[column], bias[chunk * 64 + column]);
        }
    }
}

/** @brief Compare complete destination bytes, including every dirty/padded float.
 * @param actual Native shared-row output envelope plus final canaries.
 * @param expected Actual native serial reference and identical untouched bytes.
 */
void equal(const std::vector<float> &actual, const std::vector<float> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
}

/** @brief Certify one compact all-source geometry across explicit reuse candidates.
 * @tparam Instruction Explicit native instruction family.
 * @param source Registry source identity, never inferred from expanded storage.
 * @param rows Physical row count spanning grouped boundaries.
 * @param policy Canonical complete arithmetic policy.
 * @param worker_counts Exact positive workshare budgets; immutable preparation/oracles are reused.
 * @param blocks Physical K blocks, including multi-block ordered partitions.
 */
template <Dot Instruction> void verify(const NativeVnniFormatInfo &source, int rows,
    CPUProjectionNumericalPolicy policy, std::span<const int> worker_counts, int blocks = 8)
{
    SCOPED_TRACE(::testing::Message() << "source=" << int(source.codebook_id) << " super=" << source.is_superblock
        << " M=" << rows << " blocks=" << blocks << " policy=" << int(policy));
    NativeVNNIPreparedByteFixture fixture(source, 5, blocks * 32);
    fixture.descriptor.numerical_policy = policy;
    fixture.seedPayloadAndMetadata();
    const auto weights = fixture.view();
    const auto input = inputs(rows, weights.blocks_per_row);
    const std::array<float, 5> bias{0.0001234f, -0.75f, 1.25f, -17.03125f, 0.0f};
    constexpr int stride = 8;
    const auto size = std::size_t(rows - 1) * stride + weights.N;
    std::vector<float> expected(size + 4, 123456.0f);
    serial<Instruction>(weights, input, std::span(expected).first(size), rows, stride, nullptr, bias);
    // Worker count changes ownership, not the fixture or actual serial M1
    // reference. Rebuilding/decoding both for every budget dominated ASan time
    // and made an otherwise complete sweep miss its ordinary 30-second gate.
    std::vector<float> actual(size + 4);
    for (int workers : worker_counts)
    {
        SCOPED_TRACE(::testing::Message() << "workers=" << workers);
        omp_set_num_threads(workers);
        for (auto tile : {NEONProjectionRowTile::Pair, NEONProjectionRowTile::Four})
        {
            std::fill(actual.begin(), actual.end(), 123456.0f);
            const auto batch = CPUNativeNEONProjectionBatch::bind(weights, input, std::span(actual).first(size),
                rows, stride, tile, nullptr, bias);
            batch.template execute<Instruction>();
            equal(actual, expected);
            if (rows == 3 || rows == 16 || rows == 31)
            {
                std::fill(actual.begin(), actual.end(), 123456.0f);
                int active_workers = 0;
#pragma omp parallel
                {
#pragma omp single
                    active_workers = omp_get_num_threads();
                    batch.template execute<Instruction>();
                }
                EXPECT_EQ(active_workers, workers);
                equal(actual, expected);
            }
        }
    }
}

/** @brief Sweep every source and the complete supported grouped-depth neighborhood.
 * @tparam Instruction Explicitly compiled native integer instruction family.
 */
template <Dot Instruction> void sweep()
{
    constexpr std::array<int,2> budgets{1,3};
    constexpr std::array<int,1> grouped_budget{3};
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
            for (int rows = 1; rows <= 16; ++rows)
                verify<Instruction>(*source.metadata, rows, policy, budgets, rows == 16 ? 33 : 8);
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
            verify<Instruction>(*source.metadata, 31, policy, grouped_budget, 33);
}
} // namespace

/** @test ASIMD grouped rows are byte-identical to the actual native serial M=1 leaf. */
TEST_F(CPUNativeNEONProjectionBatchTest, ASIMDAllSourcesAndPoliciesM1Through16And31)
{
    const TeamBudget budget;
    sweep<Dot::ASIMD>();
}
#if defined(__ARM_FEATURE_DOTPROD)
/** @test The admitted SDOT route retains the same source policy and original-row K trees. */
TEST_F(CPUNativeNEONProjectionBatchTest, DotProdAllSourcesAndPoliciesM1Through16And31)
{
    const TeamBudget budget;
    sweep<Dot::DotProd>();
}
#endif

/** @test Worker geometry affects ownership only, including odd/larger grouped capacities. */
TEST_F(CPUNativeNEONProjectionBatchTest, AllPositiveWorkersThroughTenRetainSerialBytes)
{
    const TeamBudget budget;
    constexpr std::array<int,10> budgets{1,2,3,4,5,6,7,8,9,10};
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
            for (const int rows : {2, 3, 5, 15, 16, 31, 33, 65})
            {
                verify<Dot::ASIMD>(*source.metadata, rows, policy, budgets, rows >= 16 ? 33 : 8);
#if defined(__ARM_FEATURE_DOTPROD)
                verify<Dot::DotProd>(*source.metadata, rows, policy, budgets, rows >= 16 ? 33 : 8);
#endif
            }
}

/** @test Causal request boundaries, sliced/repeated rows and data-only reset never compact output. */
TEST_F(CPUNativeNEONProjectionBatchTest, OriginalRequestSlicesRepeatsAndDirtyInactiveRows)
{
    const TeamBudget budget;
    std::array<std::int32_t, 3> frontier{4, 6, 1}, live{7, 4, 8};
    const std::array<ProjectionRequestPublication, 3> publications{{
        {&frontier[0], &live[0]}, {&frontier[1], &live[1]}, {&frontier[2], &live[2]}}};
    const auto original = ProjectionRowWork::causalIndirect(3, 8, publications.data(), 16, 8, 1);
    const auto work = original.slice(1, 22).repeatEachRow(2);
    constexpr int stride = 68;
    for (const auto *source : {&native_vnni_formats::Q4_0, &native_vnni_formats::Q6_K, &native_vnni_formats::IQ2_XXS})
    {
        NativeVNNIPreparedByteFixture fixture(*source, 65, 256);
        fixture.seedPayloadAndMetadata();
        const auto weights = fixture.view();
        auto input = inputs(work.physicalRows(), weights.blocks_per_row);
        const auto pristine = input;
        const auto size = std::size_t(work.physicalRows() - 1) * stride + weights.N;
        std::vector<float> actual(size + 4), expected(size + 4);
        const auto batch = CPUNativeNEONProjectionBatch::bind(weights, input, std::span(actual).first(size),
            work.physicalRows(), stride, NEONProjectionRowTile::Four, &work);
        for (const auto counts : {std::array<std::int32_t, 3>{7, 4, 8}, {0, 0, 0}, {8, 5, 3}})
        {
            live = counts; // Only original publication data changes; no binding/topology repair.
            std::copy(pristine.begin(), pristine.end(), input.begin());
            // Retain vector allocation and span identity while poisoning unused rows.
            for (int row = 0; row < work.physicalRows(); ++row)
                if (!work.resolveHost(work.requestForRow(row)).contains(row))
                    for (int block = 0; block < weights.blocks_per_row; ++block)
                        input[std::size_t(row) * weights.blocks_per_row + block].d = 0x7e00;
            std::fill(actual.begin(), actual.end(), 123456.0f);
            expected = actual;
            serial<Dot::ASIMD>(weights, input, std::span(expected).first(size), work.physicalRows(), stride, &work);
            omp_set_num_threads(3);
            batch.execute<Dot::ASIMD>();
            equal(actual, expected);
        }
    }
}

/** @test Shared native activation publication preserves policy, padding and original dirty rows. */
TEST_F(CPUNativeNEONProjectionBatchTest, OriginalQ8PublicationMatchesSerialBlocksAcrossWorkers)
{
    const TeamBudget budget;
    const std::array<std::int32_t, 3> live{0, 5, 11};
    const auto original = ProjectionRowWork::fromRequests(DeviceRequestRowRanges::deviceCounted(3, 11, live.data()));
    const auto work = original.slice(1, 31);
    for (const int k : {1, 31, 32, 33, 256, 1056})
    {
        const int blocks = (k + 31) / 32;
        std::vector<float> source(std::size_t(work.physicalRows()) * k);
        for (int row = 0; row < work.physicalRows(); ++row)
            for (int column = 0; column < k; ++column)
                source[std::size_t(row) * k + column] = float((row * 37 + column * 13) % 255 - 127) * 0.013457f;
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
        {
            SCOPED_TRACE(::testing::Message() << "K=" << k << " policy=" << int(policy));
            std::vector<Q8_1Block> expected(std::size_t(work.physicalRows()) * blocks);
            std::memset(expected.data(), 0x71, expected.size() * sizeof(Q8_1Block));
            for (int row = 0; row < work.physicalRows(); ++row)
                if (work.resolveHost(work.requestForRow(row)).contains(row))
                    for (int block = 0; block < blocks; ++block)
                        quantizeActivationBlock(source.data() + std::size_t(row) * k + block * 32,
                            expected[std::size_t(row) * blocks + block], std::min(32, k - block * 32), policy);
            for (const int workers : {1, 3, 10})
            {
                std::vector<Q8_1Block> actual(expected.size());
                std::memset(actual.data(), 0x71, actual.size() * sizeof(Q8_1Block));
                omp_set_num_threads(workers);
                quantize_activations_to_q8_1(source.data(), actual.data(), work.physicalRows(), k, blocks, policy, &work);
                EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(Q8_1Block)), 0);
                std::memset(actual.data(), 0x71, actual.size() * sizeof(Q8_1Block));
#pragma omp parallel
                {
                    quantize_activations_to_q8_1(source.data(), actual.data(), work.physicalRows(), k, blocks, policy, &work);
                }
                EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(Q8_1Block)), 0);
            }
        }
    }
    Q8_1Block destination{}, before{};
    std::memset(&destination, 0x71, sizeof(destination));
    before = destination;
    const auto bad = static_cast<CPUProjectionNumericalPolicy>(99);
    EXPECT_THROW(quantizeActivationBlock(nullptr, destination, 32, bad), std::invalid_argument);
    EXPECT_THROW(quantizeTwoActivationBlocks(nullptr, destination, destination, bad), std::invalid_argument);
    EXPECT_THROW(quantizeActivationBlock(nullptr, destination, -1, CPUProjectionNumericalPolicy::GPUAlignedExpert), std::invalid_argument);
    EXPECT_THROW(quantize_activations_to_q8_1(nullptr, &destination, 1, 33, 1), std::invalid_argument);
    EXPECT_EQ(std::memcmp(&destination, &before, sizeof(destination)), 0);
}

/** @test Empty retained work completes before single-thread publication/reset of the next epoch. */
TEST_F(CPUNativeNEONProjectionBatchTest, EmptyRetainedTeamCompletesBeforeNextPublication)
{
    const TeamBudget budget;
    NativeVNNIPreparedByteFixture fixture(native_vnni_formats::Q4_0, 5, 32);
    fixture.seedPayloadAndMetadata();
    const auto weights = fixture.view();
    constexpr int rows = 5, stride = 8;
    constexpr std::size_t size = (rows - 1) * stride + 5;
    std::array<float, rows * 32> source;
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(int(i % 33) - 16) * 0.013457f;
    std::vector<Q8_1Block> actual_q8(rows), expected_q8(rows);
    for (int row = 0; row < rows; ++row)
        quantizeActivationBlock(source.data() + row * 32, expected_q8[row], 32, weights.numerical_policy);
    std::vector<float> actual(size + 4, 123456.0f), expected(size + 4, 123456.0f);
    serial<Dot::ASIMD>(weights, expected_q8, std::span(expected).first(size), rows, stride);
    std::int32_t live = 0;
    const auto work = ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(rows, &live));
    const auto batch = CPUNativeNEONProjectionBatch::bind(weights, actual_q8, std::span(actual).first(size),
        rows, stride, NEONProjectionRowTile::Four, &work);
    enum class Operation { Publication, Projection };
    for (auto operation : {Operation::Publication, Operation::Projection})
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            live = 0;
            std::fill(actual.begin(), actual.end(), 123456.0f);
            std::copy(expected_q8.begin(), expected_q8.end(), actual_q8.begin());
            if (operation == Operation::Publication)
                std::memset(actual_q8.data(), 0x71, actual_q8.size() * sizeof(Q8_1Block));
            std::atomic<int> entered{0};
            const auto invoke = [&] {
                if (operation == Operation::Publication)
                    quantize_activations_to_q8_1(source.data(), actual_q8.data(), rows, 32, 1, weights.numerical_policy, &work);
                else batch.execute<Dot::ASIMD>();
            };
#pragma omp parallel num_threads(3)
            {
                // A late observer makes the old no-barrier early return race
                // deterministic enough to expose collective-encounter mismatch.
                if (omp_get_thread_num() == 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                entered.fetch_add(1, std::memory_order_relaxed);
                invoke();
#pragma omp single
                {
                    EXPECT_EQ(entered.load(std::memory_order_relaxed), omp_get_num_threads());
                    live = rows;
                }
                invoke();
#pragma omp single
                live = 0;
                invoke();
            }
            if (operation == Operation::Publication)
                EXPECT_EQ(std::memcmp(actual_q8.data(), expected_q8.data(), actual_q8.size() * sizeof(Q8_1Block)), 0);
            else equal(actual, expected);
        }
}

/** @test Malformed/aliased invocation envelopes fail before any output write or workshare. */
TEST_F(CPUNativeNEONProjectionBatchTest, RejectsWholeBankAliasesAndInvalidGeometry)
{
    NativeVNNIPreparedByteFixture fixture(native_vnni_formats::Q4_0, 65, 256);
    fixture.seedPayloadAndMetadata();
    const auto weights = fixture.view();
    auto input = inputs(5, weights.blocks_per_row);
    std::vector<float> output(4 * 68 + 65, 123456.0f);
    const auto tile = NEONProjectionRowTile::Four;
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input, output, 0, 68, tile), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input, output, 5, 64, tile), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, std::span(input).first(input.size() - 1), output, 5, 68, tile), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input, std::span(output).first(output.size() - 1), 5, 68, tile), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input, output, 5, 68, static_cast<NEONProjectionRowTile>(99)), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input,
        {reinterpret_cast<float *>(input.data()), output.size()}, 5, 68, tile), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input,
        {reinterpret_cast<float *>(fixture.bytes.data()), output.size()}, 5, 68, tile), std::invalid_argument);
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input, output, 5, 68, tile, nullptr,
        std::span(output).first(65)), std::invalid_argument);
    const auto wrong = ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(4));
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input, output, 5, 68, tile, &wrong), std::invalid_argument);
    std::vector<std::int32_t> original_words(output.size(), 0);
    original_words[32] = 5;
    const auto aliased = ProjectionRowWork::fromPrefix(DeviceRowRange::deviceCounted(5, &original_words[32]));
    EXPECT_THROW((void)CPUNativeNEONProjectionBatch::bind(weights, input,
        {reinterpret_cast<float *>(original_words.data()), output.size()}, 5, 68, tile, &aliased), std::invalid_argument);
    for (auto value : output) EXPECT_EQ(value, 123456.0f);
}

/** @test Exact overlays replace the total policy only for their complete measured key. */
TEST(CPUNativeNEONExactOverlay, ExactKeysOverrideTotalPolicyAndNeverInterpolate)
{
    // The installed table is compile-time validated; every unmeasured key is total.
    const auto installed = installedNativeNEONExactOverlay();
    const auto total = resolveNativeNEONSchedule(5, true, 4096, 2048, 7, 20, {});
    EXPECT_EQ(total.tile, NEONProjectionRowTile::Four);
    EXPECT_EQ(total.n_block_chunks, 1);
    EXPECT_FALSE(total.exact);
    (void)installed;

    const std::array<NativeNEONExactOverlayEntry, 3> table{{
        {.codebook_id = 5, .is_superblock = true, .n = 4096, .k = 2048, .m = 4, .workers = 20,
         .tile = NEONProjectionRowTile::Pair, .n_block_chunks = 2},
        {.codebook_id = 5, .is_superblock = true, .n = 4096, .k = 2048, .m = 15, .workers = 20,
         .tile = NEONProjectionRowTile::Four, .n_block_chunks = 4},
        {.codebook_id = 19, .is_superblock = false, .n = 512, .k = 2048, .m = 1, .workers = 10,
         .tile = NEONProjectionRowTile::Four, .n_block_chunks = 8},
    }};
    const auto hit = resolveNativeNEONSchedule(5, true, 4096, 2048, 4, 20, table);
    EXPECT_TRUE(hit.exact);
    EXPECT_EQ(hit.tile, NEONProjectionRowTile::Pair);
    EXPECT_EQ(hit.n_block_chunks, 2);
    const auto decode = resolveNativeNEONSchedule(19, false, 512, 2048, 1, 10, table);
    EXPECT_TRUE(decode.exact);
    EXPECT_EQ(decode.n_block_chunks, 8);
    // Every neighbour of a measured key keeps the total policy.
    using Key = std::tuple<std::uint8_t, bool, int, int, int, int>;
    for (const auto &[codebook, superblock, n, k, m, workers] : std::array<Key, 7>{{
             {5, true, 4096, 2048, 5, 20}, {5, true, 4096, 2048, 4, 19}, {5, false, 4096, 2048, 4, 20},
             {6, true, 4096, 2048, 4, 20}, {5, true, 4160, 2048, 4, 20},
             {5, true, 4096, 2304, 4, 20}, {19, false, 512, 2048, 2, 10}}})
    {
        const auto miss = resolveNativeNEONSchedule(codebook, superblock, n, k, m, workers, table);
        EXPECT_FALSE(miss.exact);
        EXPECT_EQ(miss.tile, NEONProjectionRowTile::Four);
        EXPECT_EQ(miss.n_block_chunks, 1);
    }
}

/**
 * @test The installed table resolves its own keys exactly through the hot-path
 * API, and an empty table leaves every key on the total policy.
 */
TEST(CPUNativeNEONExactOverlay, InstalledTableResolvesEveryEntryExactly)
{
    const auto installed = installedNativeNEONExactOverlay();
    if (installed.empty())
    {
        // Q4_K superblock (codebook 2) at a representative 35B projection key.
        const auto total = resolveNativeNEONSchedule(2, true, 2048, 4096, 4, 20);
        EXPECT_FALSE(total.exact);
        EXPECT_EQ(total.tile, NEONProjectionRowTile::Four);
        EXPECT_EQ(total.n_block_chunks, 1);
        return;
    }
    for (const auto &entry : installed)
    {
        const auto schedule = resolveNativeNEONSchedule(entry.codebook_id, entry.is_superblock,
            entry.n, entry.k, entry.m, entry.workers);
        ASSERT_TRUE(schedule.exact);
        EXPECT_EQ(schedule.tile, entry.tile);
        EXPECT_EQ(schedule.n_block_chunks, entry.n_block_chunks);
    }
    // Overlays are keyed by the measured worker budget; another budget is total.
    const auto &first = installed.front();
    const auto other = resolveNativeNEONSchedule(first.codebook_id, first.is_superblock,
        first.n, first.k, first.m, first.workers + 1);
    if (std::none_of(installed.begin(), installed.end(), [&](const auto &e) {
            return e.key() == std::tuple{first.codebook_id, first.is_superblock, first.n, first.k,
                first.m, first.workers + 1}; }))
        EXPECT_FALSE(other.exact);
}

/** @test Invalid overlay tables and geometry fail instead of selecting a schedule. */
TEST(CPUNativeNEONExactOverlay, InvalidTablesAndGeometryFail)
{
    const NativeNEONExactOverlayEntry a{.codebook_id = 5, .is_superblock = true, .n = 64, .k = 256,
        .m = 2, .workers = 4, .tile = NEONProjectionRowTile::Pair, .n_block_chunks = 1};
    auto b = a;
    b.m = 3;
    const std::array<NativeNEONExactOverlayEntry, 2> unsorted{b, a};
    const std::array<NativeNEONExactOverlayEntry, 2> duplicate{a, a};
    auto bad = a;
    bad.n_block_chunks = 3;
    const std::array<NativeNEONExactOverlayEntry, 1> chunks{bad};
    for (const auto table : {std::span<const NativeNEONExactOverlayEntry>(unsorted),
             std::span<const NativeNEONExactOverlayEntry>(duplicate),
             std::span<const NativeNEONExactOverlayEntry>(chunks)})
        EXPECT_THROW((void)resolveNativeNEONSchedule(5, true, 64, 256, 2, 4, table), std::invalid_argument);
    EXPECT_THROW((void)resolveNativeNEONSchedule(5, true, 0, 256, 2, 4, {}), std::invalid_argument);
    EXPECT_THROW((void)resolveNativeNEONSchedule(5, true, 64, 256, 0, 4, {}), std::invalid_argument);
    EXPECT_THROW((void)resolveNativeNEONSchedule(5, true, 64, 256, 2, 0, {}), std::invalid_argument);
}
} // namespace llaminar2::test
