/**
 * @file Test__PooledAppendReplay.cpp
 * @brief Recorded pooled append+decode replay through the PRODUCTION stage path.
 *
 * The recording (`pooled_append_decode_v1.txt`) was captured from the real
 * `PooledKVAppendStage` + `PooledKVDecodeBinding` implementation: a 1-layer
 * request with page_tokens=4 appends 6 rows (3 prefill-shaped + decode rows)
 * through `prepareStep`/`execute`/`completeAndAdvance`, recording the exact
 * K/V row bit patterns and each step's published page bytes (full pages shared
 * verbatim; partial tails COW-extended; the page split at rows 4→5).
 *
 * This test replays that recording through the same production implementation
 * and byte-compares every published page against the recorded expectation. It
 * is a RECORDED replay, not a constructed contract: the inputs and expected
 * bytes come from the fixture, and any divergence between the production path
 * today and the frozen recording fails.
 *
 * Recording provenance: captured on aarch64 (Apple Silicon VM container),
 * g++-14, production `llaminar2_core` at revision R26 (N1/N2). The K/V rows
 * were generated with std::mt19937 seed 6144 and recorded as exact bit
 * patterns; the bytes were captured from `Snapshot::view().pages()` storage,
 * never reconstructed.
 */
#include "execution/compute_stages/stages/PooledKVAppendStage.h"
#include "execution/compute_stages/stages/PooledKVDecodeBinding.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/MPIContext.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/PhysicalPagePool.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "PhysicalPagePoolTraceData.h"
#include <gtest/gtest.h>
#include <bit>
#include <cstring>
#include <sstream>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief One recorded K/V row (K then V, exact bit patterns). */
struct RecordedRow
{
    std::vector<float> values; ///< K row followed by V row.
};
struct RecordedStep
{
    int rows = 0;
    struct RecordedPage
    {
        int rows = 0;
        std::vector<float> bytes;
    };
    std::vector<RecordedPage> pages;
};
struct RecordedTrace
{
    int heads = 0, kv_heads = 0, dimension = 0, page_tokens = 0, total_rows = 0;
    std::vector<RecordedRow> rows;
    std::vector<RecordedStep> steps;
};

/** @brief Strictly read recorded FP32 bit patterns (preserving negative zero). */
std::vector<float> readBitVector(std::istream &input, std::size_t count)
{
    std::vector<float> result(count);
    for (auto &value : result)
    {
        std::string word;
        if (!(input >> word) || word.size() != 8 ||
            word.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::invalid_argument("Invalid recorded pooled FP32 bit pattern");
        value = std::bit_cast<float>(static_cast<std::uint32_t>(std::stoul(word, nullptr, 16)));
    }
    return result;
}

/** @brief Parse+validate the pooled append recording. */
RecordedTrace parsePooledTrace(std::istream &input)
{
    RecordedTrace trace;
    std::string schema;
    if (!(input >> schema) || schema != "pooled-append-v1")
        throw std::invalid_argument("Unsupported pooled append recording schema");
    std::string tag;
    if (!(input >> tag) || tag != "heads") throw std::invalid_argument("Missing heads");
    if (!(input >> trace.heads) || trace.heads <= 0) throw std::invalid_argument("Invalid heads");
    if (!(input >> tag) || tag != "kv_heads" || !(input >> trace.kv_heads) || trace.kv_heads <= 0)
        throw std::invalid_argument("Invalid kv_heads");
    if (!(input >> tag) || tag != "dimension" || !(input >> trace.dimension) || trace.dimension <= 0)
        throw std::invalid_argument("Invalid dimension");
    if (!(input >> tag) || tag != "page_tokens" || !(input >> trace.page_tokens) || trace.page_tokens <= 0)
        throw std::invalid_argument("Invalid page_tokens");
    if (!(input >> tag) || tag != "rows" || !(input >> trace.total_rows) || trace.total_rows <= 0)
        throw std::invalid_argument("Invalid row count");
    for (int r = 0; r < trace.total_rows; ++r)
    {
        std::string row_tag;
        int row_index = 0;
        std::size_t width_2 = 0;
        if (!(input >> row_tag) || row_tag != "kvrow" || !(input >> row_index) ||
            !(input >> width_2) ||
            width_2 != static_cast<std::size_t>(trace.kv_heads * trace.dimension) * 2)
            throw std::invalid_argument("Invalid recorded kvrow header");
        RecordedRow row;
        row.values = readBitVector(input, width_2);
        trace.rows.push_back(std::move(row));
    }
    while (input >> tag)
    {
        if (tag == "end") break;
        if (tag != "step") throw std::invalid_argument("Invalid recorded step tag");
        RecordedStep step;
        std::string sub;
        int step_index = 0;
        if (!(input >> step_index) || !(input >> sub) || sub != "rows" ||
            !(input >> step.rows) || step.rows <= 0)
            throw std::invalid_argument("Invalid step row count");
        std::size_t page_count = 0;
        if (!(input >> sub) || sub != "pages" || !(input >> page_count) || !page_count)
            throw std::invalid_argument("Invalid step page count");
        for (std::size_t p = 0; p < page_count; ++p)
        {
            int page_idx = 0, page_rows = 0;
            std::size_t bytes = 0;
            if (!(input >> sub) || sub != "page" || !(input >> page_idx) ||
                !(input >> sub) || sub != "rows" || !(input >> page_rows) ||
                !(input >> sub) || sub != "bytes" || !(input >> bytes) ||
                !bytes || bytes % 4)
                throw std::invalid_argument("Invalid recorded page header");
            RecordedStep::RecordedPage page;
            page.rows = page_rows;
            page.bytes = readBitVector(input, bytes / 4);
            step.pages.push_back(std::move(page));
        }
        trace.steps.push_back(std::move(step));
    }
    return trace;
}} // namespace

/** @test R26 recorded replay: the production pooled append path at THIS
 * revision reproduces the frozen recording byte-for-byte. */
TEST(PooledAppendReplay, RecordedSixRowDecodeReproducesFrozenBytes)
{
    std::istringstream input(llaminar2::test::pooledAppendDecodeTrace);
    const auto trace = parsePooledTrace(input);
    const auto device = DeviceId::cpu();

    const PhysicalPagePoolGeometry pool_geometry{8,
        static_cast<std::size_t>(trace.page_tokens) * 2 * trace.heads *
            trace.dimension * sizeof(float), 2, 8};
    const CPUFP32PagedKVTableGeometry table_geometry{4, 8, 4, trace.page_tokens, 32, 8};
    const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
        CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
    auto table = CPUFP32PagedKVTable::create(authority, device, table_geometry);

    MPIContext mpi{0, 1, MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    const size_t width = static_cast<size_t>(trace.kv_heads * trace.dimension);
    PooledKVDecodeBinding binding(pool.get(), table.get(), /*request_id=*/5,
        /*slot=*/0, trace.kv_heads, trace.dimension, trace.page_tokens,
        /*layers=*/1, /*admitted_issuances=*/trace.total_rows,
        // The frozen replay recording was captured on the COW path; the
        // in-place tail is the production default but this fixture drives
        // the recording's exact path.
        /*persistent_tail=*/false);

    for (int step = 0; step < trace.total_rows; ++step)
    {
        auto K = factory.createFP32({1, static_cast<long long>(width)}, device);
        auto V = factory.createFP32({1, static_cast<long long>(width)}, device);
        const auto &row = trace.rows[static_cast<std::size_t>(step)].values;
        std::memcpy(K->mutable_data(), row.data(), width * sizeof(float));
        std::memcpy(V->mutable_data(), row.data() + width, width * sizeof(float));
        auto guard = binding.prepareStep(1);
        ASSERT_TRUE(guard) << "step " << step << " admission refused";
        auto *cell = binding.nextCell(0);
        auto *prior = binding.priorFor(0);
        PooledKVAppendStage::Params params;
        params.device_id = device;
        params.K = K.get();
        params.V = V.get();
        params.pool = pool.get();
        params.table = table.get();
        params.ref_out = cell;
        params.prior = prior;
        params.grant = binding.stepGrant();
        params.issuance = binding.issuance();
        params.request_id = 5;
        params.slot = 0;
        params.layer_idx = 0;
        params.kv_heads = trace.kv_heads;
        params.head_dim = trace.dimension;
        params.page_tokens = trace.page_tokens;
        params.live_rows = 1;
        PooledKVAppendStage stage(params);
        ASSERT_TRUE(stage.execute(nullptr)) << "step " << step << " execute refused";
        ASSERT_TRUE(cell->live()) << "step " << step << " cell not live";
        ASSERT_TRUE(guard.completeAndAdvance()) << "step " << step << " advance refused";

        // Byte-compare the CURRENT publication against the frozen recording.
        // advanceStep moved the publication from the next cell into slot 0
        // (the prior), so read the LIVE current cell here.
        const auto &expected = trace.steps[static_cast<std::size_t>(step)];
        auto *current = binding.priorFor(0);
        ASSERT_NE(current, nullptr);
        const auto current_pages = current->snapshot.view().pages();
        ASSERT_EQ(current_pages.size(), expected.pages.size()) << "current page count at step " << step;
        for (std::size_t p = 0; p < current_pages.size(); ++p)
        {
            const auto &got = current_pages[p].storage();
            const auto &want = expected.pages[p].bytes;
            ASSERT_EQ(got.size(), want.size() * sizeof(float))
                << "page " << p << " byte count at step " << step;
            EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size()), 0)
                << "page " << p << " bytes diverge at step " << step;
        }
    }
    EXPECT_EQ(binding.nextPosition(), trace.total_rows);
    EXPECT_EQ(binding.remainingIssuances(), 0u);
}
