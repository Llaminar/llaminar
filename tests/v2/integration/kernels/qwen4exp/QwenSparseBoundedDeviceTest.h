/**
 * @file QwenSparseBoundedDeviceTest.h
 * @brief Real CUDA/HIP stored-score retained selector and paired timing harness.
 *
 * One captured production selection node owns each row. Fixture upload/terminal
 * reads and metadata inspection are outside replay timing. Retirement is declared
 * after all asynchronous host/device backing and joins the exact stream before
 * graph, buffer or host-span destruction, including assertions/exceptions.
 */
#pragma once
#include "QwenSparseSelectionCases.h"
#include "kernels/qwen4exp/QwenSparseBoundedSelection.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace llaminar2::qwen4exp::QSB_SUITE
{
template<class Status> void check(Status status)
{
    if (status != QSB_RT(Success)) throw std::runtime_error(QSB_RT(GetErrorString)(status));
}
struct Stream final
{
    QSB_STREAM value = nullptr;
    Stream() { check(QSB_RT(StreamCreateWithFlags)(&value, QSB_RT(StreamNonBlocking))); }
    ~Stream() { if (value) (void)QSB_RT(StreamDestroy)(value); }
};
/** @brief Explicit diagnostic allocation only; production candidate uses no global scratch. */
template<class T> struct Buffer final
{
    T *value = nullptr; std::size_t size;
    explicit Buffer(std::size_t count) : size(count)
    { check(QSB_RT(Malloc)(reinterpret_cast<void **>(&value), count * sizeof(T))); }
    ~Buffer() { if (value) (void)QSB_RT(Free)(value); }
    Buffer(const Buffer &) = delete;
    void upload(const T *source, QSB_STREAM stream)
    { check(QSB_RT(MemcpyAsync)(value, source, size * sizeof(T), QSB_RT(MemcpyHostToDevice), stream)); }
    std::vector<T> read(QSB_STREAM stream)
    {
        std::vector<T> result(size);
        try
        {
            check(QSB_RT(MemcpyAsync)(result.data(), value, size * sizeof(T), QSB_RT(MemcpyDeviceToHost), stream));
            check(QSB_RT(StreamSynchronize)(stream));
        }
        catch (...) { if (QSB_RT(StreamSynchronize)(stream) != QSB_RT(Success)) std::terminate(); throw; }
        return result;
    }
};
/** @brief Mandatory last-declared owner: failures cannot unwind borrowed copy spans early. */
struct Retirement final
{
    QSB_STREAM stream;
    ~Retirement() { if (QSB_RT(StreamSynchronize)(stream) != QSB_RT(Success)) std::terminate(); }
};
struct Graph final
{
    QSB_GRAPH graph = nullptr; QSB_EXEC executable = nullptr;
    ~Graph()
    {
        if (executable) (void)QSB_RT(GraphExecDestroy)(executable);
        if (graph) (void)QSB_RT(GraphDestroy)(graph);
    }
};
struct NativeSelection final
{
    Stream stream;
    int rows, capacity, blocks;
    QwenSparseReplay replay;
    std::vector<float> scores;
    std::vector<std::int32_t> poison;
    Buffer<QwenSparseReplay> state;
    Buffer<float> input;
    Buffer<std::int32_t> selected, count;
    Buffer<QwenSparseSelectionObservation> observation;
    Graph graph;
    QwenSparseIndexer p;
    Retirement retirement;
    NativeSelection(int m, int tokens, QwenSparsePoolSelection policy, bool observe = true)
        : rows(m), capacity(tokens), blocks((tokens + 3) / 4), replay{tokens - m, m, 0},
          scores(selection_test::scores(m, blocks, selection_test::Distribution::Hash)),
          poison(std::size_t(m) * QwenSparseSelected, -77), state(1), input(scores.size()),
          selected(poison.size()), count(m), observation(m), retirement{stream.value}
    {
        p.replay = state.value; p.scores = input.value; p.selected = selected.value;
        p.selected_count = count.value; p.row_capacity = m; p.token_capacity = tokens; p.block_capacity = blocks;
        p.pool_selection = policy;
        if (observe && policy != QwenSparsePoolSelection::Installed) p.selection_observations = observation.value;
        upload(); check(QSB_RT(StreamSynchronize)(stream.value));
    }
    void upload()
    {
        state.upload(&replay, stream.value); input.upload(scores.data(), stream.value);
        selected.upload(poison.data(), stream.value);
    }
    /** @brief Capture the real requested kernel once; replay count never changes topology. */
    void capture()
    {
        check(QSB_RT(StreamBeginCapture)(stream.value, QSB_RT(StreamCaptureModeThreadLocal)));
        try { QSB_SELECT(p, stream.value); }
        catch (...)
        {
            QSB_GRAPH abandoned = nullptr; (void)QSB_RT(StreamEndCapture)(stream.value, &abandoned);
            if (abandoned) (void)QSB_RT(GraphDestroy)(abandoned);
            throw;
        }
        check(QSB_RT(StreamEndCapture)(stream.value, &graph.graph));
        check(QSB_RT(GraphInstantiate)(&graph.executable, graph.graph, nullptr, nullptr, 0));
        std::size_t size = 0; check(QSB_RT(GraphGetNodes)(graph.graph, nullptr, &size));
        if (size != 1) throw std::logic_error("Stored-score selector must capture exactly one actual native kernel");
        QSB_NODE node = nullptr; check(QSB_RT(GraphGetNodes)(graph.graph, &node, &size));
        QSB_NODETYPE type; check(QSB_RT(GraphNodeGetType)(node, &type));
        if (type != QSB_RT(GraphNodeTypeKernel)) throw std::logic_error("Selector capture is not a native kernel");
        QSB_PARAMS params{}; check(QSB_RT(GraphKernelNodeGetParams)(node, &params));
        QSB_ATTRIBUTES attributes{}; check(QSB_RT(FuncGetAttributes)(&attributes, params.func));
        if (attributes.localSizeBytes != 0) throw std::logic_error("Selector native kernel has forbidden local storage");
        if (p.pool_selection != QwenSparsePoolSelection::Installed && attributes.sharedSizeBytes < 20u * 1024u)
            throw std::logic_error("Captured node does not expose bounded candidate shared storage");
        std::cout << "QSA_SELECTOR_RESOURCE,policy=" << int(p.pool_selection) << ",regs=" << attributes.numRegs
                  << ",shared=" << attributes.sharedSizeBytes << ",local=" << attributes.localSizeBytes
                  << ",block=" << params.blockDim.x << ",grid=" << params.gridDim.x << '\n';
        if (params.blockDim.x != 256) throw std::logic_error("Selector capture has wrong actual CTA geometry");
    }
    void launch() { check(QSB_RT(GraphLaunch)(graph.executable, stream.value)); }
    /** @brief Terminal exact output/source/diagnostic checks; never a timed host probe. */
    void verify()
    {
        const auto result = selected.read(stream.value), counts = count.read(stream.value);
        const auto original = input.read(stream.value);
        ASSERT_EQ(std::memcmp(scores.data(), original.data(), scores.size() * sizeof(float)), 0);
        const auto observed = p.selection_observations ? observation.read(stream.value) :
            std::vector<QwenSparseSelectionObservation>{};
        for (int row = 0; row < rows; ++row)
        {
            const int visible = replay.prefix_tokens + row + 1;
            const auto expected = row < replay.live_rows ? selection_test::oracle(
                scores.data() + std::size_t(row) * blocks, visible) : std::vector<std::int32_t>{};
            ASSERT_EQ(counts[row], expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
                ASSERT_EQ(result[std::size_t(row) * QwenSparseSelected + i], expected[i]);
            for (std::size_t i = expected.size(); i < QwenSparseSelected; ++i)
                ASSERT_EQ(result[std::size_t(row) * QwenSparseSelected + i], -77);
            if (p.selection_observations)
            {
                EXPECT_EQ(observed[row].score_elements,
                    row < replay.live_rows && visible / 4 > 512 ? std::uint64_t(visible / 4) : 0u);
                if (row < replay.live_rows && visible / 4 > 512) { EXPECT_GT(observed[row].compactions, 0u); }
                else { EXPECT_EQ(observed[row].compactions, 0u); }
            }
        }
    }
};
TEST(QSB_SUITE, ThresholdTiesAndEveryTail)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    for (auto policy : selection_test::policies)
    {
        NativeSelection s(1, 16391, policy); s.capture();
        for (auto distribution : selection_test::distributions)
            for (int pools : {511, 512, 513, 1025, 4097})
                for (int tail = 0; tail < 4; ++tail)
                {
                    SCOPED_TRACE(int(policy));
                    SCOPED_TRACE(int(distribution));
                    SCOPED_TRACE(pools);
                    SCOPED_TRACE(tail);
                    s.scores = selection_test::scores(s.rows, s.blocks, distribution);
                    s.replay = {pools * 4 + tail - 1, 1, 0};
                    s.upload(); s.launch(); s.verify();
                    ASSERT_FALSE(::testing::Test::HasFailure());
                }
    }
}
TEST(QSB_SUITE, RetainedPartialEmptyAndPositiveDispatch)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    for (int rows : {1, 9, 15, 31, 32, 33, 512, 2048})
        for (auto policy : selection_test::policies)
        {
            NativeSelection s(rows, 4099 + rows, policy); s.capture();
            for (int repeat = 0; repeat < (rows == 33 ? 20 : 4); ++repeat)
            {
                s.replay.live_rows = repeat % 4 == 0 ? 0 : repeat % 4 == 1 ? 1 : repeat % 4 == 2 ? std::min(rows, 9) : rows;
                s.replay.prefix_tokens = repeat % 3 == 0 ? 2048 : 4099;
                s.upload(); s.launch(); s.verify();
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
}
/** @test Actual grouped native outputs equal one retained serial native row graph. */
TEST(QSB_SUITE, GroupedSerialRowBytes)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    for (auto policy : selection_test::policies)
    {
        NativeSelection grouped(15, 4111, policy), serial(1, 4111, policy);
        grouped.capture(); serial.capture(); grouped.launch(); grouped.verify();
        const auto output = grouped.selected.read(grouped.stream.value);
        const auto counts = grouped.count.read(grouped.stream.value);
        for (int row = 0; row < 15; ++row)
        {
            std::copy_n(grouped.scores.data() + std::size_t(row) * grouped.blocks,
                grouped.blocks, serial.scores.data());
            serial.replay = {grouped.replay.prefix_tokens + row, 1, 0};
            serial.upload(); serial.launch(); serial.verify();
            const auto one = serial.selected.read(serial.stream.value);
            EXPECT_EQ(serial.count.read(serial.stream.value)[0], counts[row]);
            EXPECT_EQ(std::memcmp(one.data(), output.data() + std::size_t(row) * QwenSparseSelected,
                QwenSparseSelected * sizeof(std::int32_t)), 0);
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
    }
}
TEST(QSB_SUITE, BadBindingsAndStreamsBeforeEnqueue)
{
    int devices = 0; check(QSB_RT(GetDeviceCount)(&devices));
    if (!devices) GTEST_SKIP() << "No native backend device";
    NativeSelection s(1, 2052, QwenSparsePoolSelection::Bounded256_512);
    EXPECT_THROW(QSB_SELECT(s.p, nullptr), std::invalid_argument);
    EXPECT_THROW(QSB_SELECT(s.p, QSB_RT(StreamLegacy)), std::invalid_argument);
    EXPECT_THROW(QSB_SELECT(s.p, QSB_RT(StreamPerThread)), std::invalid_argument);
    auto p = s.p; p.pool_selection = static_cast<QwenSparsePoolSelection>(255);
    EXPECT_THROW(QSB_SELECT(p, s.stream.value), std::invalid_argument);
    p = s.p; p.selected_count = p.selected; EXPECT_THROW(QSB_SELECT(p, s.stream.value), std::invalid_argument);
    p = s.p; p.block_capacity = 1; EXPECT_THROW(QSB_SELECT(p, s.stream.value), std::invalid_argument);
}
/** @brief Persistent native events, outside correctness and metadata observation. */
struct Events final
{
    QSB_EVENT begin = nullptr, end = nullptr;
    QSB_STREAM stream;
    explicit Events(QSB_STREAM value) : stream(value)
    {
        check(QSB_RT(EventCreate)(&begin));
        try { check(QSB_RT(EventCreate)(&end)); }
        catch (...) { (void)QSB_RT(EventDestroy)(begin); throw; }
    }
    ~Events()
    {
        if (QSB_RT(StreamSynchronize)(stream) != QSB_RT(Success)) std::terminate();
        if (end) (void)QSB_RT(EventDestroy)(end);
        if (begin) (void)QSB_RT(EventDestroy)(begin);
    }
};
TEST(QSB_SUITE, PairedReleaseTiming)
{
    const char *rowsText = std::getenv("LLAMINAR_QSA_SELECT_M");
    const char *tokensText = std::getenv("LLAMINAR_QSA_SELECT_TOKENS");
    const char *repeatsText = std::getenv("LLAMINAR_QSA_SELECT_REPEATS");
    const char *distributionText = std::getenv("LLAMINAR_QSA_SELECT_DISTRIBUTION");
    if (!rowsText || !tokensText || !repeatsText || !distributionText)
        GTEST_SKIP() << "Explicit exact selector timing geometry/repeats/distribution required";
    const int rows = std::stoi(rowsText), tokens = std::stoi(tokensText), repeats = std::stoi(repeatsText);
    const int distributionId = std::stoi(distributionText);
    ASSERT_GT(rows, 0); ASSERT_LE(rows, 2048); ASSERT_GE(tokens, rows); ASSERT_LE(tokens, 262147);
    ASSERT_GT(repeats, 0); ASSERT_LE(repeats, 10000);
    ASSERT_GE(distributionId, 0); ASSERT_LT(distributionId, int(selection_test::distributions.size()));
    for (auto policy : selection_test::policies)
    {
        NativeSelection a(rows, tokens, QwenSparsePoolSelection::Installed, false), b(rows, tokens, policy, false);
        a.scores = selection_test::scores(rows, a.blocks, selection_test::distributions[distributionId]); b.scores = a.scores;
        a.upload(); b.upload(); a.capture(); b.capture();
        for (int warm = 0; warm < 3; ++warm) { a.launch(); b.launch(); }
        a.verify(); b.verify(); ASSERT_FALSE(::testing::Test::HasFailure());
        Events aEvents(a.stream.value), bEvents(b.stream.value);
        const auto time = [&](NativeSelection &s, Events &events)
        {
            check(QSB_RT(EventRecord)(events.begin, s.stream.value));
            for (int repeat = 0; repeat < repeats; ++repeat) s.launch();
            check(QSB_RT(EventRecord)(events.end, s.stream.value));
            check(QSB_RT(EventSynchronize)(events.end)); // terminal fence OUTSIDE measured event interval
            float milliseconds = 0; check(QSB_RT(EventElapsedTime)(&milliseconds, events.begin, events.end));
            return milliseconds / repeats;
        };
        for (int pair = 0; pair < 4; ++pair)
        {
            const bool ab = pair % 2 == 0;
            float first = ab ? time(a, aEvents) : time(b, bEvents);
            float second = ab ? time(b, bEvents) : time(a, aEvents);
            std::cout << "QSA_SELECT_PAIR,scope=stored_scores,order=" << (ab ? "AB" : "BA")
                      << ",M=" << rows << ",tokens=" << tokens << ",distribution=" << distributionId
                      << ",policy=" << int(policy) << ",installed_ms=" << (ab ? first : second)
                      << ",candidate_ms=" << (ab ? second : first) << '\n';
        }
        a.verify(); b.verify(); ASSERT_FALSE(::testing::Test::HasFailure());
        NativeSelection diagnostic(rows, tokens, policy, true);
        diagnostic.scores = a.scores; diagnostic.upload(); diagnostic.capture(); diagnostic.launch(); diagnostic.verify();
        const auto observations = diagnostic.observation.read(diagnostic.stream.value);
        QwenSparseSelectionObservation total{};
        for (const auto &row : observations)
        {
            total.score_elements += row.score_elements; total.compactions += row.compactions;
            total.compacted_keys += row.compacted_keys; total.radix_passes += row.radix_passes;
        }
        std::cout << "QSA_SELECT_WORK,policy=" << int(policy) << ",score_elements=" << total.score_elements
                  << ",compactions=" << total.compactions << ",compacted_keys=" << total.compacted_keys
                  << ",radix_passes=" << total.radix_passes << ",scope=diagnostic_not_timed_not_wire_bytes\n";
        ASSERT_FALSE(::testing::Test::HasFailure());
    }
}
} // namespace llaminar2::qwen4exp::QSB_SUITE
