/**
 * @file Test__RMSNormGammaQ11Identity.cpp
 * @brief Owner-identity regressions for the pure-integer Q8_1 RMSNorm gamma cache.
 *
 * The cache once keyed quantized gamma by raw address and width alone. A weight
 * freed and replaced at the same address with different values then received
 * stale Q11 words. These tests drive the production primitive TU through every
 * identity transition: a new owner at a reused address, an expired owner, an
 * unowned caller and an authentic hit. Expected words always come from an
 * independent quantization of the current values, so the comparison proves
 * freshness rather than self-consistency. Device-free; no factory or model.
 */
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace llaminar2::test
{
using namespace primitives;

namespace
{
    constexpr std::size_t kCols = 96; // Three complete Q8_1 blocks.

    /** @return Gamma values distinct per seed, including clamped magnitudes. */
    std::vector<float> gammaValues(float seed)
    {
        std::vector<float> values(kCols);
        for (std::size_t i = 0; i < kCols; ++i)
            values[i] = seed * (static_cast<float>(i % 13) - 6.0f) + 0.125f * static_cast<float>(i % 7);
        values[5] = 20.0f * seed; // Exercises the [-16, 16] clamp.
        return values;
    }

    /** @brief Publish new values at the existing address, as a reallocated weight would. */
    void overwrite(std::vector<float> &storage, float seed)
    {
        const auto values = gammaValues(seed);
        std::copy(values.begin(), values.end(), storage.begin());
    }

    /** @return An independent quantization of the gamma values as they are now. */
    std::vector<int16_t> freshWords(const std::vector<float> &gamma)
    {
        std::vector<int16_t> words(gamma.size());
        quantize_rmsnorm_gamma_q11(gamma.data(), words.data(), gamma.size());
        return words;
    }

    /** @return A copy of cache-returned words before the next call can rewrite them. */
    std::vector<int16_t> copyWords(const int16_t *words)
    {
        return {words, words + kCols};
    }
}

/** @test A new owner whose weights reuse the old address receives fresh words. */
TEST(RMSNormGammaQ11Identity, NewOwnerAtReusedAddressNeverStaleHits)
{
    std::vector<float> storage = gammaValues(1.0f);
    auto first_owner = std::make_shared<int>(1);
    const auto first = copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, first_owner));
    ASSERT_EQ(first, freshWords(storage));

    // Retire the first weight and publish different values at the same address.
    first_owner.reset();
    const float *address = storage.data();
    overwrite(storage, -0.75f);
    ASSERT_EQ(storage.data(), address);
    auto second_owner = std::make_shared<int>(2);
    const auto second = copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, second_owner));
    EXPECT_EQ(second, freshWords(storage));
    EXPECT_NE(second, first);
}

/** @test An expired owner cannot satisfy a hit, even with its own weak reference. */
TEST(RMSNormGammaQ11Identity, ExpiredOwnerRequantizes)
{
    std::vector<float> storage = gammaValues(0.5f);
    auto owner = std::make_shared<int>(3);
    const std::weak_ptr<const void> retained = owner;
    (void)rmsnorm_gamma_q11_for_owner(storage.data(), kCols, retained);

    owner.reset();
    const float *address = storage.data();
    overwrite(storage, 2.0f);
    ASSERT_EQ(storage.data(), address);
    const auto words = copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, retained));
    EXPECT_EQ(words, freshWords(storage));
}

/** @test Callers without an owning identity are quantized on every call. */
TEST(RMSNormGammaQ11Identity, UnownedGammaIsNeverCached)
{
    std::vector<float> storage = gammaValues(1.5f);
    const std::weak_ptr<const void> unowned;
    EXPECT_EQ(copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, unowned)), freshWords(storage));

    const float *address = storage.data();
    overwrite(storage, -3.0f);
    ASSERT_EQ(storage.data(), address);
    EXPECT_EQ(copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, unowned)), freshWords(storage));
}

/** @test A live owner hits without requantizing; immutable weights are the contract. */
TEST(RMSNormGammaQ11Identity, LiveOwnerHitsAuthenticEntry)
{
    std::vector<float> storage = gammaValues(0.25f);
    auto owner = std::make_shared<int>(4);
    const int16_t *first = rmsnorm_gamma_q11_for_owner(storage.data(), kCols, owner);
    const auto expected = copyWords(first);

    // Mutating a live owner's weights violates the immutability contract; the
    // unchanged words prove the hit reused the entry instead of requantizing.
    storage[0] = 9.0f;
    const int16_t *second = rmsnorm_gamma_q11_for_owner(storage.data(), kCols, owner);
    EXPECT_EQ(second, first);
    EXPECT_EQ(copyWords(second), expected);
}

/** @test The same owner with another address or width is a different entry. */
TEST(RMSNormGammaQ11Identity, AddressAndWidthArePartOfTheKey)
{
    auto owner = std::make_shared<int>(5);
    std::vector<float> first = gammaValues(1.0f);
    std::vector<float> second = gammaValues(-1.0f);
    (void)rmsnorm_gamma_q11_for_owner(first.data(), kCols, owner);
    EXPECT_EQ(copyWords(rmsnorm_gamma_q11_for_owner(second.data(), kCols, owner)), freshWords(second));

    const int16_t *narrow = rmsnorm_gamma_q11_for_owner(second.data(), kCols - 32, owner);
    std::vector<int16_t> expected(kCols - 32);
    quantize_rmsnorm_gamma_q11(second.data(), expected.data(), expected.size());
    EXPECT_EQ(std::vector<int16_t>(narrow, narrow + expected.size()), expected);
}

/** @test Each thread owns its entry: one thread's hit is never another thread's storage. */
TEST(RMSNormGammaQ11Identity, ThreadEntriesAreIndependent)
{
    std::vector<float> storage = gammaValues(0.75f);
    auto owner = std::make_shared<int>(6);
    const auto main_words = copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, owner));

    // Same owner, address and width: only a per-thread entry explains a miss.
    storage[1] = -11.0f;
    std::vector<int16_t> worker_words;
    std::thread worker([&] {
        worker_words = copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, owner));
    });
    worker.join();
    EXPECT_EQ(worker_words, freshWords(storage));
    EXPECT_EQ(copyWords(rmsnorm_gamma_q11_for_owner(storage.data(), kCols, owner)), main_words);
}
}
