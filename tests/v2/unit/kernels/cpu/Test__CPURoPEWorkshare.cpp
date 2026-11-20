/**
 * @file Test__CPURoPEWorkshare.cpp
 * @brief All-ISA production cached-decode head ownership and retained-team regression.
 *
 * Every participant shares one set of borrowed angle planes, as CPURoPEKernelT
 * does with its stage workspace. The tensor banks are shared, so every logical
 * head must be rotated exactly once and the angle fill completes before use. A delayed member exposes duplicate whole-bank
 * execution instead of allowing coincident input loads to conceal that defect.
 */
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "kernels/cpu/primitives/RoPEInline.h"
#include "tensors/FP16Utils.h"
#include "tensors/BlockStructures.h"
#include "utils/RoPEAngleTestStorage.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <atomic>
#include <initializer_list>
#include <limits>
#include <cstdint>
#include <cstring>
#include <string>
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
 * @param angles Angle planes shared by every member of the calling team.
 */
template <Storage Format> void invoke(Word<Format> *q, Word<Format> *k,
    int q_heads, int k_heads, const primitives::RoPEAngleRows &angles)
{
    if constexpr (Format == Storage::FP32)
        primitives::apply_rope_vectorized(q, k, 1, 64, q_heads, k_heads, 17231, 10000.0f, angles);
    else if constexpr (Format == Storage::Half)
        primitives::apply_rope_fp16(q, k, 1, 64, q_heads, k_heads, 17231, 10000.0f, angles);
    else
        primitives::apply_rope_bf16(q, k, 1, 64, q_heads, k_heads, 17231, 10000.0f, angles);
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
        RoPEAngleTestStorage reference(1, 32);
        invoke<Format>(expected_q.data(), geometry.keys == Keys::Present ? expected_k.data() : nullptr,
            geometry.query_heads, geometry.key_heads, reference.view(1, 32));
        if (geometry.query_heads)
            ASSERT_NE(std::memcmp(query.data(), expected_q.data(), query.size() * sizeof(Word<Format>)), 0);
        for (int workers : {1, 3, 10})
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                auto actual_q = query, actual_k = key;
                std::atomic<int> entered{0};
                int observed_workers = 0, observed_at_return = 0;
                RoPEAngleTestStorage shared(1, 32);
                const auto angles = shared.view(1, 32);
#pragma omp parallel num_threads(workers)
                {
#pragma omp single
                    observed_workers = omp_get_num_threads();
                    if (omp_get_thread_num() == workers - 1 && workers > 1)
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    entered.fetch_add(1, std::memory_order_relaxed);
                    invoke<Format>(actual_q.data(), geometry.keys == Keys::Present ? actual_k.data() : nullptr,
                        geometry.query_heads, geometry.key_heads, angles);
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
                RoPEAngleTestStorage storage(1, dim / 2);
                if (rotary)
                    primitives::apply_rope_partial(head, nullptr, 1, dim, rotary, heads, 0, 17231 + row, 10000.0f,
                        storage.view(1, rotary / 2));
                else
                    primitives::apply_rope_vectorized(head, nullptr, 1, dim, heads, 0, 17231 + row, 10000.0f,
                        storage.view(1, dim / 2));
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

/**
 * @test Lock-free inverse-frequency hits return the one published table.
 *
 * More geometries than the per-thread memo holds force evictions; concurrent
 * workers must still observe the single process-lifetime address and the exact
 * values of an independent computation for every (head_dim, base) identity.
 */
TEST(CPURoPEWorkshare, InverseFrequencyLookupIsStableAcrossThreadsAndEvictions)
{
    struct Geometry { int head_dim; float base; };
    std::vector<Geometry> geometries;
    for (int dim : {32, 64, 96, 128, 256, 512})
        for (float base : {10000.0f, 1000000.0f})
            geometries.push_back({dim, base});

    std::vector<const std::vector<float> *> published;
    for (const auto &geometry : geometries)
    {
        const auto &values = primitives::get_inv_freq_cached(geometry.head_dim, geometry.base);
        ASSERT_EQ(values.size(), static_cast<std::size_t>(geometry.head_dim / 2));
        const float log_base = std::log(geometry.base);
        for (int i = 0; i < geometry.head_dim / 2; ++i)
        {
            const float expected = std::exp(-log_base * ((2.0f * i) / geometry.head_dim));
            ASSERT_EQ(std::bit_cast<std::uint32_t>(values[i]), std::bit_cast<std::uint32_t>(expected));
        }
        published.push_back(&values);
    }

    std::atomic<int> mismatches{0};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 6; ++worker)
        workers.emplace_back([&, worker] {
            for (int round = 0; round < 64; ++round)
                for (std::size_t index = 0; index < geometries.size(); ++index)
                {
                    const std::size_t pick = (index * 5 + static_cast<std::size_t>(worker + round)) % geometries.size();
                    const auto &values = primitives::get_inv_freq_cached(
                        geometries[pick].head_dim, geometries[pick].base);
                    if (&values != published[pick])
                        mismatches.fetch_add(1, std::memory_order_relaxed);
                }
        });
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(mismatches.load(), 0);
}

/** @brief One public multi-row RoPE call shape exercised under every team placement. */
enum class RowCall { FP32Full, FP32Partial, Half, BFloat16, FP32Verifier, HalfVerifier, BFloat16Verifier, Q8Integer };

/** @brief Seed one bank of original words deterministically. */
template <class Word> void seedBank(std::vector<Word> &bank, int salt)
{
    for (std::size_t i = 0; i < bank.size(); ++i)
    {
        if constexpr (std::is_same_v<Word, float>)
            bank[i] = float((int(i % 251) * 73 + salt) % 251 - 125) * 0.03125f;
        else if constexpr (std::is_same_v<Word, std::uint16_t>)
            bank[i] = static_cast<std::uint16_t>(0x3c00 + (int(i % 251) * 73 + salt) % 512);
        else
        {
            bank[i].d = fp32_to_fp16(float(i % 13 + 1) * 0.001234f);
            int sum = 0;
            for (int j = 0; j < 32; ++j)
            {
                bank[i].qs[j] = static_cast<std::int8_t>((int(i % 251) * 13 + j * 73 + salt) % 256 - 128);
                sum += bank[i].qs[j];
            }
            bank[i].sum_qs = static_cast<std::int16_t>(sum);
        }
    }
}

/**
 * @brief Run one multi-row call through the shared angle planes.
 * @return Rotated query and key bytes.
 *
 * `delayed` makes the last member of an existing team arrive late, so a
 * rotation that read planes before the shared fill completed would read the
 * poisoned bytes written into the storage beforehand.
 */
std::vector<std::uint8_t> runRows(RowCall call, int rows, int team, bool delayed)
{
    constexpr int dim = 64, q_heads = 3, k_heads = 1;
    const std::vector<int> positions = [&] {
        std::vector<int> values(rows);
        for (int row = 0; row < rows; ++row) values[row] = row % 5 == 3 ? -1 : 17231 + 7 * row;
        return values;
    }();
    const bool verifier = call == RowCall::FP32Verifier || call == RowCall::HalfVerifier ||
                          call == RowCall::BFloat16Verifier;
    const int values = call == RowCall::FP32Partial ? dim / 4 : dim / 2;
    RoPEAngleTestStorage storage(rows, dim / 2);
    auto bytes = storage.aligned();
    std::memset(bytes.data(), 0xff, bytes.size()); // NaN cosines/sines until filled.
    const auto angles = storage.view(rows, values);
    std::vector<std::uint8_t> result;
    const auto run = [&](auto *q, auto *k)
    {
        const auto body = [&]()
        {
            if (delayed && omp_get_thread_num() == omp_get_num_threads() - 1 && omp_get_num_threads() > 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            switch (call)
            {
            case RowCall::FP32Full:
                primitives::apply_rope_vectorized(reinterpret_cast<float *>(q), reinterpret_cast<float *>(k),
                    rows, dim, q_heads, k_heads, 17231, 10000.0f, angles); break;
            case RowCall::FP32Partial:
                primitives::apply_rope_partial(reinterpret_cast<float *>(q), reinterpret_cast<float *>(k),
                    rows, dim, dim / 2, q_heads, k_heads, 17231, 10000.0f, angles); break;
            case RowCall::Half:
                primitives::apply_rope_fp16(reinterpret_cast<std::uint16_t *>(q), reinterpret_cast<std::uint16_t *>(k),
                    rows, dim, q_heads, k_heads, 17231, 10000.0f, angles); break;
            case RowCall::BFloat16:
                primitives::apply_rope_bf16(reinterpret_cast<std::uint16_t *>(q), reinterpret_cast<std::uint16_t *>(k),
                    rows, dim, q_heads, k_heads, 17231, 10000.0f, angles); break;
            case RowCall::FP32Verifier:
                primitives::apply_rope_decode_equivalent_rows(reinterpret_cast<float *>(q), reinterpret_cast<float *>(k),
                    positions.data(), rows, dim, 0, q_heads, k_heads, 0, 10000.0f, angles); break;
            case RowCall::HalfVerifier:
                primitives::apply_rope_fp16_decode_equivalent_rows(reinterpret_cast<std::uint16_t *>(q),
                    reinterpret_cast<std::uint16_t *>(k), positions.data(), rows, dim, q_heads, k_heads, 0, 10000.0f,
                    angles); break;
            case RowCall::BFloat16Verifier:
                primitives::apply_rope_bf16_decode_equivalent_rows(reinterpret_cast<std::uint16_t *>(q),
                    reinterpret_cast<std::uint16_t *>(k), positions.data(), rows, dim, q_heads, k_heads, 0, 10000.0f,
                    angles); break;
            case RowCall::Q8Integer:
                primitives::apply_rope_q8_1_integer(reinterpret_cast<Q8_1Block *>(q), reinterpret_cast<Q8_1Block *>(k),
                    positions.data(), rows, q_heads, k_heads, dim, 10000.0f, angles); break;
            }
        };
        if (team <= 1)
            body();
        else
        {
#pragma omp parallel num_threads(team)
            body();
        }
    };
    const auto collect = [&](auto &query, auto &key)
    {
        const auto *qb = reinterpret_cast<const std::uint8_t *>(query.data());
        const auto *kb = reinterpret_cast<const std::uint8_t *>(key.data());
        result.assign(qb, qb + query.size() * sizeof(query[0]));
        result.insert(result.end(), kb, kb + key.size() * sizeof(key[0]));
    };
    if (call == RowCall::Q8Integer)
    {
        std::vector<Q8_1Block> query(std::size_t(rows) * q_heads * dim / 32), key(std::size_t(rows) * k_heads * dim / 32);
        seedBank(query, 17); seedBank(key, 31);
        run(query.data(), key.data());
        collect(query, key);
    }
    else if (call == RowCall::FP32Full || call == RowCall::FP32Partial || call == RowCall::FP32Verifier)
    {
        std::vector<float> query(std::size_t(rows) * q_heads * dim), key(std::size_t(rows) * k_heads * dim);
        seedBank(query, 17); seedBank(key, 31);
        run(query.data(), key.data());
        collect(query, key);
    }
    else
    {
        std::vector<std::uint16_t> query(std::size_t(rows) * q_heads * dim), key(std::size_t(rows) * k_heads * dim);
        seedBank(query, 17); seedBank(key, 31);
        run(query.data(), key.data());
        collect(query, key);
    }
    (void)verifier;
    return result;
}

/**
 * @test Every member of an existing team shares one angle fill that completes before rotation.
 *
 * The former implementation gave each team member a private table. Shared
 * planes are filled by one omp-for whose implicit barrier must publish every
 * row before any head reads it, including when one member arrives late.
 */
TEST(CPURoPEWorkshare, SharedAngleFillCompletesBeforeRotationInRetainedTeams)
{
    Budget budget;
    for (const RowCall call : {RowCall::FP32Full, RowCall::FP32Partial, RowCall::Half, RowCall::BFloat16,
             RowCall::FP32Verifier, RowCall::HalfVerifier, RowCall::BFloat16Verifier, RowCall::Q8Integer})
        for (int rows : {1, 2, 5, 33})
        {
            const auto expected = runRows(call, rows, 1, false);
            for (int team : {2, 3, 10})
                for (int iteration = 0; iteration < 3; ++iteration)
                {
                    const auto actual = runRows(call, rows, team, true);
                    ASSERT_EQ(actual, expected) << "call=" << int(call) << " rows=" << rows << " team=" << team;
                }
        }
}

/** @test Missing, misaligned, undersized or mismatched angle planes are rejected before any store. */
TEST(CPURoPEWorkshare, AngleRowsRejectMissingOrUndersizedPlanes)
{
    RoPEAngleTestStorage storage(4, 32);
    auto aligned = storage.aligned();
    EXPECT_THROW((void)primitives::carve_rope_angle_rows({}, 1, 32), std::invalid_argument);
    EXPECT_THROW((void)primitives::carve_rope_angle_rows(aligned.subspan(1), 1, 32), std::invalid_argument);
    EXPECT_THROW((void)primitives::carve_rope_angle_rows(aligned.first(64), 4, 32), std::invalid_argument);
    EXPECT_THROW((void)primitives::carve_rope_angle_rows(aligned, 0, 32), std::invalid_argument);
    EXPECT_NO_THROW((void)primitives::carve_rope_angle_rows(aligned, 4, 32));

    std::vector<float> query(3 * 64, 0.5f);
    const auto original = query;
    // A plane narrower than the head's pairs, and fewer rows than requested.
    EXPECT_THROW(primitives::apply_rope_vectorized(query.data(), nullptr, 1, 64, 3, 0, 9, 10000.0f,
                     storage.view(1, 16)), std::invalid_argument);
    EXPECT_THROW(primitives::apply_rope_vectorized(query.data(), nullptr, 3, 64, 1, 0, 9, 10000.0f,
                     storage.view(2, 32)), std::invalid_argument);
    EXPECT_EQ(query, original);
}

/**
 * @test The inverse-frequency memo returns each configuration's own table when alternated.
 *
 * Alternating two (width, base) identities on one thread and on every member
 * of an existing team must never return the other configuration's table.
 */
TEST(CPURoPEWorkshare, InverseFrequencyLookupAlternatesConfigurationsInRetainedTeams)
{
    Budget budget;
    const auto &first = primitives::get_inv_freq_cached(64, 10000.0f);
    const auto &second = primitives::get_inv_freq_cached(64, 1000000.0f);
    ASSERT_NE(&first, &second);
    ASSERT_NE(first[1], second[1]);
    for (int round = 0; round < 64; ++round)
    {
        ASSERT_EQ(&primitives::get_inv_freq_cached(64, 10000.0f), &first);
        ASSERT_EQ(&primitives::get_inv_freq_cached(64, 1000000.0f), &second);
    }
    std::atomic<int> mismatches{0};
#pragma omp parallel num_threads(4)
    {
        for (int round = 0; round < 256; ++round)
        {
            const bool pick_first = (round + omp_get_thread_num()) % 2 == 0;
            const auto &values = primitives::get_inv_freq_cached(64, pick_first ? 10000.0f : 1000000.0f);
            if (&values != (pick_first ? &first : &second))
                mismatches.fetch_add(1, std::memory_order_relaxed);
        }
    }
    EXPECT_EQ(mismatches.load(), 0);
}

/** @brief Seeded Q8_1 query/key banks for one-row integer RoPE regressions. */
struct Q8Rows
{
    std::vector<Q8_1Block> query, key;
    Q8Rows(int rows, int dim) : query(std::size_t(rows) * 3 * dim / 32), key(std::size_t(rows) * dim / 32)
    { seedBank(query, 23); seedBank(key, 29); }
    /** @return Query then key bytes. */
    std::vector<std::uint8_t> bytes() const
    {
        const auto *q = reinterpret_cast<const std::uint8_t *>(query.data());
        const auto *k = reinterpret_cast<const std::uint8_t *>(key.data());
        std::vector<std::uint8_t> out(q, q + query.size() * sizeof(Q8_1Block));
        out.insert(out.end(), k, k + key.size() * sizeof(Q8_1Block));
        return out;
    }
};

/**
 * @test One-row Q8_1 RoPE inside a retained team rotates each shared head exactly once.
 *
 * Every member of an existing team calls the kernel on the same banks. The
 * former one-row path rotated all heads on every member; one head workshare
 * must reproduce the standalone bytes for any team width and a late member.
 */
TEST(CPURoPEWorkshare, Q8OneRowRetainedTeamRotatesEachHeadOnce)
{
    Budget budget;
    constexpr int dim = 64;
    const int position = 17231;
    Q8Rows expected(1, dim);
    RoPEAngleTestStorage reference(1, dim / 2);
    primitives::apply_rope_q8_1_integer(expected.query.data(), expected.key.data(), &position, 1, 3, 1, dim,
        10000.0f, reference.view(1, dim / 2));
    for (int team : {2, 3, 10})
        for (int iteration = 0; iteration < 5; ++iteration)
        {
            Q8Rows actual(1, dim);
            RoPEAngleTestStorage storage(1, dim / 2);
            const auto angles = storage.view(1, dim / 2);
#pragma omp parallel num_threads(team)
            {
                if (omp_get_thread_num() == team - 1)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                primitives::apply_rope_q8_1_integer(actual.query.data(), actual.key.data(), &position, 1, 3, 1,
                    dim, 10000.0f, angles);
            }
            ASSERT_EQ(actual.bytes(), expected.bytes()) << "team=" << team;
        }
}

/**
 * @test A negative one-row Q8_1 position never reuses a previous call's angles.
 *
 * The former one-row path skipped its table fill for a negative position and
 * rotated by whatever the thread's previous call had left behind. The row is
 * now computed like the multi-row program: independent of the prior position
 * on the same thread and equal to row 0 of a two-row call.
 */
TEST(CPURoPEWorkshare, Q8OneRowNegativePositionIsIndependentOfPriorCall)
{
    Budget budget;
    constexpr int dim = 64;
    RoPEAngleTestStorage storage(2, dim / 2);
    const int padding = -1;
    std::vector<std::vector<std::uint8_t>> observed;
    for (const int prior : {777, 17231, 0})
    {
        Q8Rows warm(1, dim);
        primitives::apply_rope_q8_1_integer(warm.query.data(), warm.key.data(), &prior, 1, 3, 1, dim, 10000.0f,
            storage.view(1, dim / 2));
        Q8Rows padded(1, dim);
        primitives::apply_rope_q8_1_integer(padded.query.data(), padded.key.data(), &padding, 1, 3, 1, dim,
            10000.0f, storage.view(1, dim / 2));
        observed.push_back(padded.bytes());
    }
    EXPECT_EQ(observed[0], observed[1]);
    EXPECT_EQ(observed[0], observed[2]);

    // The same row inside a two-row call, where the multi-row program always computed it.
    Q8Rows pair(2, dim);
    const std::array<int, 2> positions = {padding, 5};
    primitives::apply_rope_q8_1_integer(pair.query.data(), pair.key.data(), positions.data(), 2, 3, 1, dim,
        10000.0f, storage.view(2, dim / 2));
    const auto rows = pair.bytes();
    const std::size_t q_row = 3 * dim / 32 * sizeof(Q8_1Block), k_row = dim / 32 * sizeof(Q8_1Block);
    std::vector<std::uint8_t> first_row(rows.begin(), rows.begin() + q_row);
    first_row.insert(first_row.end(), rows.begin() + 2 * q_row, rows.begin() + 2 * q_row + k_row);
    EXPECT_EQ(observed[0], first_row);
}

/** @brief Ordinary contiguous-position entry points whose rows are all rotated. */
enum class Ordinary { FP32Full, FP32Partial, Half, BFloat16 };
/** @brief Storage word of one ordinary entry point. */
template <Ordinary Call> using OrdinaryWord = std::conditional_t<Call == Ordinary::FP32Full ||
    Call == Ordinary::FP32Partial, float, std::uint16_t>;
constexpr int kOrdinaryDim = 64, kOrdinaryQueryHeads = 3, kOrdinaryKeyHeads = 1;

/** @return The rotated prefix width of one ordinary entry point. */
constexpr int ordinaryRotaryWidth(Ordinary call) { return call == Ordinary::FP32Partial ? kOrdinaryDim / 2 : kOrdinaryDim; }

/**
 * @brief Call one public ordinary entry point for contiguous rows from `first`.
 * @tparam Call Entry point under test.
 * @param q Query rows, modified in place.
 * @param k Key rows, modified in place.
 * @param rows Live rows.
 * @param first Absolute position of row 0; may be negative.
 * @param angles Borrowed planes for this call.
 */
template <Ordinary Call> void callOrdinary(OrdinaryWord<Call> *q, OrdinaryWord<Call> *k, int rows, int first,
    const primitives::RoPEAngleRows &angles)
{
    constexpr int dim = kOrdinaryDim, qh = kOrdinaryQueryHeads, kh = kOrdinaryKeyHeads;
    if constexpr (Call == Ordinary::FP32Full)
        primitives::apply_rope_vectorized(q, k, rows, dim, qh, kh, first, 10000.0f, angles);
    else if constexpr (Call == Ordinary::FP32Partial)
        primitives::apply_rope_partial(q, k, rows, dim, ordinaryRotaryWidth(Call), qh, kh, first, 10000.0f, angles);
    else if constexpr (Call == Ordinary::Half)
        primitives::apply_rope_fp16(q, k, rows, dim, qh, kh, first, 10000.0f, angles);
    else
        primitives::apply_rope_bf16(q, k, rows, dim, qh, kh, first, 10000.0f, angles);
}

/** @brief Negate the second element of every rotated pair: the conjugate rotation's input/output map. */
template <class Word> void conjugateRows(Word *bank, std::size_t heads, int rotary)
{
    for (std::size_t head = 0; head < heads; ++head)
        for (int i = rotary / 2; i < rotary; ++i)
        {
            Word &word = bank[head * kOrdinaryDim + i];
            if constexpr (std::is_same_v<Word, float>)
                word = std::bit_cast<float>(std::bit_cast<std::uint32_t>(word) ^ 0x80000000u);
            else
                word = static_cast<std::uint16_t>(word ^ 0x8000u);
        }
}

/** @return The bank as bytes with both signed zeros folded to +0, for byte comparison. */
template <class Word> std::vector<std::uint8_t> canonicalBytes(const std::vector<Word> &bank)
{
    std::vector<Word> words = bank;
    for (Word &word : words)
    {
        if constexpr (std::is_same_v<Word, float>)
        {
            if (std::bit_cast<std::uint32_t>(word) == 0x80000000u) word = 0.0f;
        }
        else if (word == 0x8000u)
            word = 0;
    }
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(words.data());
    return {bytes, bytes + words.size() * sizeof(Word)};
}

/** @brief Seeded query and key banks for ordinary rows. */
template <class Word> struct OrdinaryBanks
{
    std::vector<Word> query, key;
    explicit OrdinaryBanks(int rows)
        : query(std::size_t(rows) * kOrdinaryQueryHeads * kOrdinaryDim),
          key(std::size_t(rows) * kOrdinaryKeyHeads * kOrdinaryDim)
    { seedBank(query, 17); seedBank(key, 31); }
    /** @return Unmodified query then key bytes. */
    std::vector<std::uint8_t> rawBytes() const
    {
        const auto *q = reinterpret_cast<const std::uint8_t *>(query.data());
        const auto *k = reinterpret_cast<const std::uint8_t *>(key.data());
        std::vector<std::uint8_t> out(q, q + query.size() * sizeof(Word));
        out.insert(out.end(), k, k + key.size() * sizeof(Word));
        return out;
    }
    /** @return Query then key bytes with signed zeros folded to +0. */
    std::vector<std::uint8_t> canonicalBytes() const
    {
        auto out = llaminar2::test::canonicalBytes(query);
        const auto k = llaminar2::test::canonicalBytes(key);
        out.insert(out.end(), k.begin(), k.end());
        return out;
    }
};

/**
 * @brief Rotation of every row at its position, assembled from positive single-row calls.
 *
 * A row at non-negative position P is rotated alone at P. Rotation is odd
 * in its angle: cos(-a) == cos(a) and sin(-a) == -sin(a) with the angle
 * product exact under negation, and every multiply, add and fused
 * multiply-add, as well as FP16/BF16 round-to-nearest-even, commutes with
 * sign. A row at P < 0 therefore equals the conjugate of its conjugated input
 * rotated at -P, except that an exactly cancelled pair may differ in the sign
 * of its zero; the comparison folds signed zeros for that reason.
 *
 * This is not independent arithmetic: it calls the current implementation's
 * one-row path at non-negative positions, which never meets a padding row.
 * Its equality with the original base translation unit is established
 * outside this test by the cross-TU byte harness.
 *
 * @return Query then key bytes with signed zeros folded.
 */
template <Ordinary Call> std::vector<std::uint8_t> ordinaryReference(int rows, int first)
{
    using Word = OrdinaryWord<Call>;
    OrdinaryBanks<Word> banks(rows);
    RoPEAngleTestStorage storage(1, kOrdinaryDim / 2);
    const int rotary = ordinaryRotaryWidth(Call);
    for (int row = 0; row < rows; ++row)
    {
        Word *q = banks.query.data() + std::size_t(row) * kOrdinaryQueryHeads * kOrdinaryDim;
        Word *k = banks.key.data() + std::size_t(row) * kOrdinaryKeyHeads * kOrdinaryDim;
        const int position = first + row;
        if (position < 0)
        {
            conjugateRows(q, kOrdinaryQueryHeads, rotary);
            conjugateRows(k, kOrdinaryKeyHeads, rotary);
        }
        callOrdinary<Call>(q, k, 1, position < 0 ? -position : position, storage.view(1, rotary / 2));
        if (position < 0)
        {
            conjugateRows(q, kOrdinaryQueryHeads, rotary);
            conjugateRows(k, kOrdinaryKeyHeads, rotary);
        }
    }
    return banks.canonicalBytes();
}

/** @brief Prior content of the borrowed planes and active flags before the call under test. */
enum class Workspace
{
    Clean,         ///< Freshly value-initialized storage, never written.
    NaNFlagsSet,   ///< Every byte 0xff: NaN angles, every flag set.
    FiniteStale,   ///< Every byte 0x41: finite wrong angles, every flag set.
    PriorPositive, ///< Left behind by a complete earlier call at positive positions.
};

/** @brief Output of one call under test and the team width that actually executed it. */
struct OrdinaryRun
{
    std::vector<std::uint8_t> raw;       ///< Unmodified output bytes.
    std::vector<std::uint8_t> canonical; ///< Output bytes with signed zeros folded.
    int width = 0;                       ///< Members observed inside the call's team.
};

/**
 * @brief Rotate contiguous rows from a negative start through a prepared borrowed workspace.
 * @param workspace Prior content of the planes and flags.
 * @param team Members of an existing team that all enter the call; 1 is standalone.
 */
template <Ordinary Call> OrdinaryRun ordinaryRun(int rows, int first, Workspace workspace, int team)
{
    OrdinaryBanks<OrdinaryWord<Call>> banks(rows);
    RoPEAngleTestStorage storage(rows, kOrdinaryDim / 2);
    auto bytes = storage.aligned();
    const auto angles = storage.view(rows, ordinaryRotaryWidth(Call) / 2);
    if (workspace == Workspace::NaNFlagsSet)
        std::memset(bytes.data(), 0xff, bytes.size());
    else if (workspace == Workspace::FiniteStale)
        std::memset(bytes.data(), 0x41, bytes.size());
    else if (workspace == Workspace::PriorPositive)
    {
        OrdinaryBanks<OrdinaryWord<Call>> earlier(rows);
        callOrdinary<Call>(earlier.query.data(), earlier.key.data(), rows, 17231, angles);
    }
    OrdinaryRun run;
    if (team <= 1)
    {
        callOrdinary<Call>(banks.query.data(), banks.key.data(), rows, first, angles);
        run.width = 1;
    }
    else
    {
        int observed = 0;
#pragma omp parallel num_threads(team)
        {
            // Member 0 records the width without a barrier, so the late member
            // still enters the call after the others.
            if (omp_get_thread_num() == 0)
                observed = omp_get_num_threads();
            if (omp_get_thread_num() == team - 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            callOrdinary<Call>(banks.query.data(), banks.key.data(), rows, first, angles);
        }
        run.width = observed;
    }
    run.raw = banks.rawBytes();
    run.canonical = banks.canonicalBytes();
    return run;
}

/** @brief Check one ordinary entry point over negative starts, dirty workspaces and teams. */
template <Ordinary Call> void checkOrdinaryNegativePositions(std::initializer_list<int> row_counts)
{
    Budget budget;
    for (const int rows : row_counts)
        for (const int first : {-1, -3})
        {
            const auto expected = ordinaryReference<Call>(rows, first);
            const auto clean = ordinaryRun<Call>(rows, first, Workspace::Clean, 1);
            for (const Workspace workspace : {Workspace::Clean, Workspace::NaNFlagsSet, Workspace::FiniteStale,
                     Workspace::PriorPositive})
                for (const int team : {1, 3})
                {
                    const auto run = ordinaryRun<Call>(rows, first, workspace, team);
                    const auto where = [&] {
                        return "call=" + std::to_string(int(Call)) + " rows=" + std::to_string(rows) +
                               " first=" + std::to_string(first) + " workspace=" + std::to_string(int(workspace)) +
                               " team=" + std::to_string(team);
                    };
                    ASSERT_EQ(run.width, team) << where() << ": the team did not execute at its requested width";
                    // Unrestricted byte equality: the output does not depend on prior workspace content.
                    EXPECT_TRUE(run.raw == clean.raw) << where() << ": raw bytes differ from a clean workspace";
                    // Equality modulo signed zero with rotation at the row's own position.
                    EXPECT_TRUE(run.canonical == expected) << where() << ": differs from per-row rotation";
                }
        }
}

/**
 * @test Ordinary rows at negative positions rotate by their position, never by dirty planes.
 *
 * Contiguous-position entry points rotate every row; before the borrowed
 * angle workspace they computed every row's angles, including negative
 * positions. Each such row must still publish its angles. Two properties are
 * checked: raw output bytes are identical whatever the workspace held
 * beforehand (clean, NaN, finite stale, or a prior complete call), and the
 * output equals per-row rotation at each row's own position modulo signed
 * zero. Teams must actually execute at their requested width. Covers one-row
 * decode, FP32 multi-row and partial-rotary tables, and the BF16/FP16 shared
 * tables at 32+ rows (their shorter direct program computes angles inline
 * and is not a table consumer).
 */
TEST(CPURoPEWorkshare, OrdinaryNegativePositionsIgnoreDirtyAngleWorkspace)
{
    checkOrdinaryNegativePositions<Ordinary::FP32Full>({1, 8, 40});
    checkOrdinaryNegativePositions<Ordinary::FP32Partial>({1, 8, 40});
    checkOrdinaryNegativePositions<Ordinary::Half>({1, 40});
    checkOrdinaryNegativePositions<Ordinary::BFloat16>({1, 40});
}
} // namespace
} // namespace llaminar2::test
