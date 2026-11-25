/**
 * @file Test__CPUFlashAttentionPagedReplay.cpp
 * @brief Recorded native paged-FP32 kernel inputs, frame order and expected bytes.
 *
 * The retained fixture records a fixed zero-score arithmetic panel, whose
 * integer/power-of-two operands avoid nonzero-score transcendental differences
 * while exercising nonconstant native K/V rows and permuted pages. Cross-ISA
 * replay evidence remains separate from the local capture. Broader nonzero
 * score/shape byte comparisons live in the companion native sweep. This trace
 * does not qualify quantized operands, native append or serving integration.
 */
#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "PhysicalPagePoolTraceData.h"
#include <gtest/gtest.h>
#include <bit>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace llaminar2;
namespace
{
/** @brief Complete recording with explicit empty initial pool and immutable operands. */
struct PagedFP32Trace
{
    int queries = 3, tokens = 17, heads = 4, kv_heads = 2, dimension = 64, page_tokens = 16;
    std::vector<size_t> frames{1, 0};
    std::vector<float> query, key, value, expected;
};
/** @brief Capture/replay actual native paged execution, never reconstruct a contiguous history. */
std::vector<float> execute(const PagedFP32Trace &trace)
{
    const size_t width = static_cast<size_t>(trace.kv_heads * trace.dimension);
    const size_t count = trace.frames.size();
    const PhysicalPagePoolGeometry geometry{count, static_cast<size_t>(trace.page_tokens) * 2 * width * sizeof(float), 1, count};
    const auto device = DeviceId::cpu();
    const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
    auto request = pool->tryAdmit(count);
    if (!request) throw std::logic_error("Recorded exact-fit page grant refused");
    std::vector<PhysicalPagePool::FrameHandle> frames;
    for (size_t i = 0; i < count; ++i) frames.push_back(request->assignFrame());
    std::vector<CPUFP32KVPage> pages;
    pages.reserve(count);
    for (size_t page = 0; page < count; ++page)
    {
        const size_t first = page * trace.page_tokens;
        const size_t live = std::min(static_cast<size_t>(trace.page_tokens), static_cast<size_t>(trace.tokens) - first);
        std::vector<float> packed(live * 2 * width);
        for (size_t token = 0; token < live; ++token)
        {
            std::copy_n(trace.key.data() + (first + token) * width, width, packed.data() + token * 2 * width);
            std::copy_n(trace.value.data() + (first + token) * width, width, packed.data() + token * 2 * width + width);
        }
        auto write = request->prepareAppend(frames.at(trace.frames.at(page)), packed.size() * sizeof(float));
        if (!write) throw std::logic_error("Recorded page writer refused");
        write->copyFrom(std::as_bytes(std::span(packed)));
        pages.emplace_back(write->commit(), trace.kv_heads, trace.dimension);
    }
    CPUFP32PagedKVView view(pages, trace.page_tokens);
    request.reset(); pool.reset();
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
    std::vector<float> output(trace.query.size(), -999.0f);
    if (!kernel.compute_paged_fp32(trace.query, view, output, trace.queries, trace.heads, false))
        throw std::logic_error("Recorded paged attention invocation refused");
    return output;
}
/** @brief Strictly read one recorded FP32 bit vector, preserving negative zero and all bit patterns. */
std::vector<float> readBank(std::istream &input, const char *name, size_t count)
{
    std::string tag; size_t actual;
    if (!(input >> tag >> actual) || tag != name || actual != count || actual > 1024 * 1024)
        throw std::invalid_argument("Incompatible recorded paged-attention bank");
    std::vector<float> result(actual);
    for (auto &value : result)
    {
        std::string word;
        if (!(input >> word) || word.size() != 8 || word.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::invalid_argument("Invalid recorded FP32 bit pattern");
        value = std::bit_cast<float>(static_cast<uint32_t>(std::stoul(word, nullptr, 16)));
    }
    return result;
}
/** @brief Reject incompatible geometry/policy or incomplete initial state before native execution. */
PagedFP32Trace parse(std::istream &input)
{
    PagedFP32Trace trace;
    std::string schema, initial, policy;
    if (!(input >> schema >> initial >> policy) || schema != "paged-fp32-v1" ||
        initial != "empty-zero-generations" || policy != "uniform-zero-score-panel-v1")
        throw std::invalid_argument("Incompatible native paged-attention trace policy/state");
    if (!(input >> trace.queries >> trace.tokens >> trace.heads >> trace.kv_heads >> trace.dimension >> trace.page_tokens) ||
        trace.queries != 3 || trace.tokens != 17 || trace.heads != 4 || trace.kv_heads != 2 ||
        trace.dimension != 64 || trace.page_tokens != 16)
        throw std::invalid_argument("Unsupported recorded panel geometry, not an engine geometry restriction");
    std::string tag;
    size_t count;
    if (!(input >> tag >> count) || tag != "frames" || count != 2) throw std::invalid_argument("Missing native frame order");
    trace.frames.resize(count);
    for (auto &frame : trace.frames) if (!(input >> frame) || frame >= count) throw std::invalid_argument("Invalid recorded native frame");
    if (trace.frames[0] == trace.frames[1]) throw std::invalid_argument("Aliased recorded writable frames");
    const size_t query_values = static_cast<size_t>(trace.queries * trace.heads * trace.dimension);
    const size_t kv_values = static_cast<size_t>(trace.tokens * trace.kv_heads * trace.dimension);
    trace.query = readBank(input, "query", query_values);
    trace.key = readBank(input, "key", kv_values);
    trace.value = readBank(input, "value", kv_values);
    trace.expected = readBank(input, "output", query_values);
    if (!(input >> tag) || tag != "end" || (input >> tag)) throw std::invalid_argument("Incomplete/trailing native page trace");
    return trace;
}
/** @brief Retain exact native observed bits after explicit diagnostic capture only. */
void write(std::ostream &output, const PagedFP32Trace &trace)
{
    output << "paged-fp32-v1 empty-zero-generations uniform-zero-score-panel-v1\n"
        << trace.queries << ' ' << trace.tokens << ' ' << trace.heads << ' ' << trace.kv_heads << ' '
        << trace.dimension << ' ' << trace.page_tokens << "\nframes 2 " << trace.frames[0] << ' ' << trace.frames[1] << '\n';
    const auto bank = [&](const char *name, const std::vector<float> &values) {
        output << name << ' ' << std::dec << values.size() << '\n';
        bool first = true;
        for (const auto value : values)
        {
            if (!first) output << ' ';
            first = false;
            output << std::hex << std::setw(8) << std::setfill('0') << std::bit_cast<uint32_t>(value);
        }
        output << '\n';
    };
    bank("query", trace.query); bank("key", trace.key); bank("value", trace.value); bank("output", trace.expected);
    output << "end\n";
}
}

/** @test Recorded native query/KV/page inputs replay byte-exactly through optimized attention. */
TEST(CPUFlashAttentionPaged, RecordedFP32InputsReplayExactly)
{
    if (const char *path = std::getenv("LLAMINAR_RECORD_PAGED_FP32_TRACE"))
    {
        PagedFP32Trace trace;
        trace.query.assign(static_cast<size_t>(trace.queries * trace.heads * trace.dimension), 1.0f);
        trace.key.assign(static_cast<size_t>(trace.tokens * trace.kv_heads * trace.dimension), 0.0f);
        trace.value.resize(trace.key.size());
        for (int token = 0; token < trace.tokens; ++token)
        for (int head = 0; head < trace.kv_heads; ++head)
        {
            const size_t row = static_cast<size_t>(token * trace.kv_heads + head) * trace.dimension;
            trace.key[row] = static_cast<float>(token + head + 1);
            trace.key[row + 1] = -trace.key[row];
            for (int d = 0; d < trace.dimension; ++d)
                trace.value[row + d] = static_cast<float>((token + head + d) % 7 - 3) * 0.125f;
        }
        trace.expected = execute(trace);
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> control;
        std::vector<float> contiguous(trace.query.size());
        ASSERT_TRUE(control.compute_decode(trace.query.data(), trace.key.data(), trace.value.data(),
            contiguous.data(), trace.queries, trace.tokens, trace.heads, trace.kv_heads, trace.dimension, false));
        ASSERT_EQ(std::memcmp(contiguous.data(), trace.expected.data(), contiguous.size() * sizeof(float)), 0);
        std::ofstream output(path);
        ASSERT_TRUE(output.good()); write(output, trace); output.close(); ASSERT_TRUE(output.good());
        std::ifstream saved(path); const auto replay = parse(saved); const auto actual = execute(replay);
        EXPECT_EQ(std::memcmp(actual.data(), replay.expected.data(), actual.size() * sizeof(float)), 0);
        return;
    }
    std::istringstream input(llaminar2::test::pagedFP32AttentionTrace);
    const auto trace = parse(input); const auto actual = execute(trace);
    EXPECT_EQ(std::memcmp(actual.data(), trace.expected.data(), actual.size() * sizeof(float)), 0);
}

/** @test Missing state, changed policy/geometry and truncated payloads fail before kernel submission. */
TEST(CPUFlashAttentionPaged, IncompleteRecordedInputsAreRejected)
{
    for (const std::string text : {"", "paged-fp32-v2 empty-zero-generations uniform-zero-score-panel-v1\n",
        "paged-fp32-v1 empty-zero-generations uniform-zero-score-panel-v1\n3 17 4 2 64 16\nframes 2 1 0\nquery 768\n"})
    {
        std::istringstream input(text);
        EXPECT_THROW(parse(input), std::exception);
    }
}
