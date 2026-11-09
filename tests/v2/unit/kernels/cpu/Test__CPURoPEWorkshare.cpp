/**
 * @file Test__CPURoPEWorkshare.cpp
 * @brief All-ISA production cached-decode head ownership and retained-team regression.
 *
 * Each participant owns its memoized angle state just like CPURoPEKernelT's
 * thread-local cache. The tensor banks are shared, so every logical head must
 * be rotated exactly once. A delayed member exposes duplicate whole-bank
 * execution instead of allowing coincident input loads to conceal that defect.
 */
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "kernels/cpu/primitives/RoPEInline.h"
#include "tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <chrono>
#include <atomic>
#include <limits>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @brief Actual public storage families selected explicitly by this regression. */
enum class Storage { FP32, Half, BFloat16 };
/** @brief Original public storage word, with no emulated SIMD implementation. */
template <Storage Format> using Word = std::conditional_t<Format == Storage::FP32, float, std::uint16_t>;

/** @brief Restore caller worker policy after real retained-team tests. */
class Budget final
{
public:
    /** @brief Keep explicit requested teams intact and begin with one worker. */
    Budget() : previous_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
    { omp_set_dynamic(0); omp_set_num_threads(1); }
    /** @brief Restore the unchanged caller worker policy. */
    ~Budget() { omp_set_num_threads(previous_); omp_set_dynamic(dynamic_); }
private:
    int previous_, dynamic_;
};

/** @brief Call the real public ordinary decode API, never an ISA leaf substitute.
 * @tparam Format Original public storage encoding.
 * @param q Original writable query bank.
 * @param k Optional original writable key bank.
 * @param q_heads Original query-head capacity.
 * @param k_heads Original key-head capacity when k is present.
 * @param state Participant-thread-owned angle memoization.
 */
template <Storage Format> void invoke(Word<Format> *q, Word<Format> *k,
    int q_heads, int k_heads, primitives::RoPEPersistentState &state)
{
    if constexpr (Format == Storage::FP32)
        primitives::apply_rope_vectorized(q, k, 1, 64, q_heads, k_heads, 17231, 10000.0f, &state);
    else if constexpr (Format == Storage::Half)
        primitives::apply_rope_fp16(q, k, 1, 64, q_heads, k_heads, 17231, 10000.0f, &state);
    else
        primitives::apply_rope_bf16(q, k, 1, 64, q_heads, k_heads, 17231, 10000.0f, &state);
}

/** @brief Check exact head ownership over actual independent participant caches.
 * @tparam Format Original public storage family.
 */
template <Storage Format> void retained()
{
    Budget budget;
    enum class Keys { Absent, Present };
    struct Geometry { int query_heads, key_heads; Keys keys; };
    for (const auto geometry : {Geometry{0, 0, Keys::Absent}, Geometry{1, 0, Keys::Absent},
             Geometry{0, 1, Keys::Present}, Geometry{1, 1, Keys::Present}, Geometry{2, 1, Keys::Present},
             Geometry{3, 1, Keys::Absent}, Geometry{3, 1, Keys::Present}})
    {
        std::vector<Word<Format>> query(std::size_t(geometry.query_heads) * 64 + 4);
        std::vector<Word<Format>> key(std::size_t(geometry.key_heads) * 64 + 4);
        const auto seed = [](auto &bank, int salt) {
            for (std::size_t i = 0; i < bank.size(); ++i)
            {
                if constexpr (Format == Storage::FP32)
                    bank[i] = float((int(i % 251) * 73 + salt) % 251 - 125) * 0.03125f;
                else bank[i] = static_cast<std::uint16_t>(0x3c00 + (int(i % 251) * 73 + salt) % 512);
            }
        };
        seed(query, 17); seed(key, 31);
        auto expected_q = query, expected_k = key;
        primitives::RoPEPersistentState reference;
        invoke<Format>(expected_q.data(), geometry.keys == Keys::Present ? expected_k.data() : nullptr,
            geometry.query_heads, geometry.key_heads, reference);
        if (geometry.query_heads)
            ASSERT_NE(std::memcmp(query.data(), expected_q.data(), query.size() * sizeof(Word<Format>)), 0);
        for (int workers : {1, 3, 10})
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                auto actual_q = query, actual_k = key;
                std::atomic<int> entered{0};
                int observed_workers = 0, observed_at_return = 0;
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    observed_workers = omp_get_num_threads();
                    primitives::RoPEPersistentState state;
                    if (omp_get_thread_num() == workers - 1 && workers > 1)
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    entered.fetch_add(1, std::memory_order_relaxed);
                    invoke<Format>(actual_q.data(), geometry.keys == Keys::Present ? actual_k.data() : nullptr,
                        geometry.query_heads, geometry.key_heads, state);
                    // Sample before single's own barrier, including zero heads.
#pragma omp single
                    observed_at_return = entered.load(std::memory_order_relaxed);
                }
                ASSERT_EQ(observed_workers, workers);
                EXPECT_EQ(observed_at_return, workers);
                EXPECT_EQ(std::memcmp(actual_q.data(), expected_q.data(), actual_q.size() * sizeof(Word<Format>)), 0);
                EXPECT_EQ(std::memcmp(actual_k.data(), expected_k.data(), actual_k.size() * sizeof(Word<Format>)), 0);
            }
    }
}
/** @test Cached ordinary decode must not repeat rotations for each retained-team member. */
TEST(CPURoPEWorkshare, CachedDecodeRetainedTeamOwnsEachHeadOnce)
{
    retained<Storage::FP32>(); retained<Storage::Half>(); retained<Storage::BFloat16>();
}
/** @test Bounded inline coefficient tiles cover wide and partial heads without a fixed 256-value limit. */
TEST(CPURoPEWorkshare, InlineWideHeadsAndPartialPrefixesRetainProductionBytes)
{
    Budget budget;
    constexpr int rows = 3, heads = 2;
    for (int dim : {128, 256, 512, 1024})
        for (int rotary : {0, dim / 2})
        {
            std::vector<float> original(std::size_t(rows) * heads * dim + 4, 919.25f);
            for (std::size_t i = 0; i + 4 < original.size(); ++i) original[i] = float(int(i % 251) - 125) * 0.03125f;
            auto expected = original;
            for (int row = 0; row < rows; ++row)
            {
                auto *head = expected.data() + std::size_t(row) * heads * dim;
                primitives::RoPEPersistentState state;
                if (rotary)
                    primitives::apply_rope_partial(head, nullptr, 1, dim, rotary, heads, 0, 17231 + row, 10000.0f);
                else
                    primitives::apply_rope_vectorized(head, nullptr, 1, dim, heads, 0, 17231 + row, 10000.0f, &state);
            }
            auto actual = original;
            apply_rope_to_k_fp32(actual.data(), rows, dim, heads, 10000.0f, 17231, rotary);
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
            actual = original;
#pragma omp parallel num_threads(3)
            {
                apply_rope_to_k_fp32(actual.data(), rows, dim, heads, 10000.0f, 17231, rotary);
            }
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
}

/** @test Empty inline K workshares complete before any next observer/publication edge. */
TEST(CPURoPEWorkshare, InlineEmptyCompletionPrecedesObserverRelease)
{
    Budget budget;
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        std::atomic<int> entered{0};
        int observed = 0;
#pragma omp parallel num_threads(3)
        {
            if (omp_get_thread_num() == 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            entered.fetch_add(1, std::memory_order_relaxed);
            apply_rope_to_k_fp32(nullptr, 0, 128, 2, 10000.0f, 0);
#pragma omp single
            observed = entered.load(std::memory_order_relaxed);
        }
        ASSERT_EQ(observed, 3);
    }
}

/** @test Invalid inline head/count/position geometry rejects before dereferencing a purported live bank. */
TEST(CPURoPEWorkshare, InlineGeometryRejectsBeforeAccess)
{
    auto *unreadable = reinterpret_cast<float *>(std::uintptr_t{1});
    EXPECT_THROW(apply_rope_to_k_fp32(unreadable, -1, 128, 2, 10000.0f, 0), std::invalid_argument);
    EXPECT_THROW(apply_rope_to_k_fp32(unreadable, 1, 127, 2, 10000.0f, 0), std::invalid_argument);
    EXPECT_THROW(apply_rope_to_k_fp32(unreadable, 1, 128, 2, 10000.0f, 0, 130), std::invalid_argument);
    EXPECT_THROW(apply_rope_to_k_fp32(unreadable, 3, 128, 2, 10000.0f, std::numeric_limits<int>::max()), std::invalid_argument);
    EXPECT_THROW(apply_rope_to_k_fp32(unreadable, std::numeric_limits<int>::max(), 2147483646,
        std::numeric_limits<int>::max(), 10000.0f, 0), std::invalid_argument);
    EXPECT_THROW(apply_rope_to_head_inline(unreadable, unreadable, unreadable, 7), std::invalid_argument);
}
/** @test Streaming Q8/FP32 conversion covers wide original rows and retained teams without head mirrors. */
TEST(CPURoPEWorkshare, HybridWideHeadsRetainSerialOriginalRows)
{
    Budget budget;
    for (int dim : {32, 64, 96, 128, 256, 512, 1024})
        for (int rows : {1, 3, 16})
        {
            const int blocks = dim / 32;
            std::vector<Q8_1Block> query(std::size_t(rows) * 3 * blocks), key(std::size_t(rows) * blocks);
            const auto seed = [](auto &bank, int salt) {
                for (std::size_t b = 0; b < bank.size(); ++b)
                {
                    bank[b].d = fp32_to_fp16(float(b % 13 + 1) * 0.001234f);
                    int sum = 0;
                    for (int i = 0; i < 32; ++i)
                    { bank[b].qs[i] = static_cast<std::int8_t>((int(b % 251) * 13 + i * 73 + salt) % 256 - 128); sum += bank[b].qs[i]; }
                    bank[b].sum_qs = static_cast<std::int16_t>(sum);
                }
            };
            seed(query, 17); seed(key, 31);
            std::vector<int> positions(rows);
            std::vector<float> expected_q(std::size_t(rows) * 3 * dim + 4, 919.25f), expected_k(std::size_t(rows) * dim + 4, 919.25f);
            for (int row = 0; row < rows; ++row)
            {
                positions[row] = row % 7 == 2 ? -1 : row % 3 == 0 ? 17231 : 11;
                primitives::apply_rope_q8_1_to_fp32(query.data() + std::size_t(row) * 3 * blocks,
                    key.data() + std::size_t(row) * blocks, expected_q.data() + std::size_t(row) * 3 * dim,
                    expected_k.data() + std::size_t(row) * dim, positions.data() + row, 1, 3, 1, dim, 10000.0f);
            }
            for (int workers : {1, 3, 10})
            {
                std::vector<float> actual_q(expected_q.size(), 919.25f), actual_k(expected_k.size(), 919.25f);
                omp_set_num_threads(workers);
                primitives::apply_rope_q8_1_to_fp32(query.data(), key.data(), actual_q.data(), actual_k.data(),
                    positions.data(), rows, 3, 1, dim, 10000.0f);
                EXPECT_EQ(std::memcmp(actual_q.data(), expected_q.data(), actual_q.size() * sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(actual_k.data(), expected_k.data(), actual_k.size() * sizeof(float)), 0);
                std::fill(actual_q.begin(), actual_q.end(), 919.25f); std::fill(actual_k.begin(), actual_k.end(), 919.25f);
                int observed_workers = 0;
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    observed_workers = omp_get_num_threads();
                    primitives::apply_rope_q8_1_to_fp32(query.data(), key.data(), actual_q.data(), actual_k.data(),
                        positions.data(), rows, 3, 1, dim, 10000.0f);
                }
                ASSERT_EQ(observed_workers, workers);
                EXPECT_EQ(std::memcmp(actual_q.data(), expected_q.data(), actual_q.size() * sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(actual_k.data(), expected_k.data(), actual_k.size() * sizeof(float)), 0);
            }
        }
}

/** @test Hybrid conversion retains its empty completion edge and rejects source/publication aliases before access. */
TEST(CPURoPEWorkshare, HybridStorageAdmissionAndEmptyCompletion)
{
    Budget budget;
    std::vector<Q8_1Block> input(8);
    std::vector<float> output(260, 919.25f);
    const auto original_input = input;
    const auto original_output = output;
    EXPECT_THROW(primitives::apply_rope_q8_1_to_fp32(input.data(), nullptr, reinterpret_cast<float *>(input.data()), nullptr,
        nullptr, 1, 1, 0, 256, 10000.0f), std::invalid_argument);
    EXPECT_THROW(primitives::apply_rope_q8_1_to_fp32(input.data(), input.data(), output.data(), nullptr,
        nullptr, 1, 1, 1, 256, 10000.0f), std::invalid_argument);
    EXPECT_THROW(primitives::apply_rope_q8_1_to_fp32(input.data(), nullptr, output.data(), nullptr,
        reinterpret_cast<int *>(output.data()), 1, 1, 0, 256, 10000.0f), std::invalid_argument);
    EXPECT_EQ(std::memcmp(input.data(), original_input.data(), input.size() * sizeof(Q8_1Block)), 0);
    EXPECT_EQ(std::memcmp(output.data(), original_output.data(), output.size() * sizeof(float)), 0);
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        std::atomic<int> entered{0}; int observed = 0;
#pragma omp parallel num_threads(3)
        {
            if (omp_get_thread_num() == 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            entered.fetch_add(1, std::memory_order_relaxed);
            primitives::apply_rope_q8_1_to_fp32(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, 0, 256, 10000.0f);
#pragma omp single
            observed = entered.load(std::memory_order_relaxed);
        }
        ASSERT_EQ(observed, 3);
    }
}
} // namespace
} // namespace llaminar2::test
