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
#include <bit>
#include <chrono>
#include <cmath>
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
} // namespace
} // namespace llaminar2::test
