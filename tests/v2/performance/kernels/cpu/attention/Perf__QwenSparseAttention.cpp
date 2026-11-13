/**
 * @file Perf__QwenSparseAttention.cpp
 * @brief Isolated production-geometry CPU sparse-indexer and selected-KV timing.
 *
 * Setup allocates the complete admitted banks before timing. Optional perf FIFO
 * control enables counters only around production launches, excluding fixture
 * construction and warmup. ISA and physical-core affinity belong to the binary
 * build and launch command, not hidden per-candidate environment overrides.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "../../../../integration/kernels/qwen4exp/QwenSparseNativeCodecFixture.h"
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Synchronize a profiling-only enable/disable command with perf's FIFO acknowledgment. */
void controlPerf(std::ofstream &control, std::ifstream &ack, const char *command)
{
    if (!control.is_open()) return;
    control << command << '\n' << std::flush;
    std::string line;
    if (!std::getline(ack, line)) throw std::runtime_error("QSA CPU profiler did not acknowledge counter control");
}
}

TEST(CPUQwenSparseAttentionPerf, ExactCandidate)
{
    const char *rows_env = std::getenv("LLAMINAR_QSA_PROFILE_M");
    const char *context_env = std::getenv("LLAMINAR_QSA_PROFILE_CONTEXT");
    if (!rows_env || !context_env) GTEST_SKIP() << "Set exact candidate M and context for isolated profiling";
    const int rows = std::stoi(rows_env), context = std::stoi(context_env), blocks = (context + 3) / 4;
    ASSERT_GT(rows, 0); ASSERT_GE(context, rows);
    QwenSparseReplay replay{context - rows, rows, 0};
    std::vector<float> query(std::size_t(rows) * 512), raw(std::size_t(rows) * 128), partial(512),
        pools(std::size_t(blocks) * 128), norm(128, 1), cosine(std::size_t(context) * 32, 1), sine(cosine.size(), 0),
        prepared(query.size()), scores(std::size_t(rows) * blocks), core_query(std::size_t(rows) * 6144), output(core_query.size());
    std::vector<std::int32_t> selected(std::size_t(rows) * QwenSparseSelected), count(rows);
    std::vector<QwenSparseAttentionObservation> observations(std::size_t(rows) * 24);
    std::uint32_t seed = 31;
    for (auto *values : {&query, &raw, &partial, &pools, &core_query})
        for (auto &value : *values)
        {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            value = float(seed >> 8) * (1.0f / 8388608.0f) - 1.0f;
        }
    test::NativeCodecFixture keys(TensorType::FP16, context), values(TensorType::FP16, context);
    QwenSparseIndexer indexer{.replay = &replay, .query = query.data(), .raw_keys = raw.data(), .partial_keys = partial.data(),
        .pooled_keys = pools.data(), .query_norm = norm.data(), .key_norm = norm.data(), .cosine = cosine.data(), .sine = sine.data(),
        .prepared_query = prepared.data(), .scores = scores.data(), .selected = selected.data(), .selected_count = count.data(),
        .row_capacity = rows, .token_capacity = context, .block_capacity = blocks, .position_capacity = context,
        .representation = QwenSparseScalar::FP16};
    QwenSparseAttention attention{.replay = &replay, .query = core_query.data(), .kv = {.keys = keys.bank(), .values = values.bank()},
        .selected = selected.data(), .selected_count = count.data(), .output = output.data(), .observations = observations.data(),
        .row_capacity = rows, .token_capacity = context};
    auto launch = [&]() { cpuQwenSparseIndexer(indexer); cpuQwenSparseAttention(attention); };
    for (int repeat = 0; repeat < 3; ++repeat) launch();
    std::ofstream control;
    std::ifstream ack;
    if (const char *path = std::getenv("LLAMINAR_QSA_PERF_CONTROL"))
    {
        const char *ack_path = std::getenv("LLAMINAR_QSA_PERF_ACK");
        ASSERT_NE(ack_path, nullptr);
        control.open(path); ack.open(ack_path);
        ASSERT_TRUE(control.is_open() && ack.is_open());
    }
    controlPerf(control, ack, "enable");
    const auto begin = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 5; ++repeat) launch();
    const auto end = std::chrono::steady_clock::now();
    controlPerf(control, ack, "disable");
    const double milliseconds = std::chrono::duration<double, std::milli>(end - begin).count() / 5;
    std::cout << "QSA_CPU_CANDIDATE,M=" << rows << ",context=" << context << ",milliseconds=" << milliseconds << '\n';
    for (int row = 0; row < rows; ++row)
    {
        const int visible = context - rows + row + 1;
        EXPECT_EQ(count[row], std::min(visible / 4, 512) * 4 + visible % 4);
        for (int head = 0; head < 24; ++head)
        {
            EXPECT_EQ(observations[row * 24 + head].key_elements, std::uint64_t(count[row]) * 256);
            EXPECT_EQ(observations[row * 24 + head].value_elements, std::uint64_t(count[row]) * 256);
        }
    }
}
} // namespace llaminar2::qwen4exp
