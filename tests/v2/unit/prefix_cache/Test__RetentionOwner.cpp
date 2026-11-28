/**
 * @file Test__RetentionOwner.cpp
 * @brief §8.4 retention ordering authority contracts (P3 §8.4).
 */
#include "execution/prefix_cache/RetentionOwner.h"

#include <gtest/gtest.h>

namespace llaminar2
{
namespace
{
constexpr std::uint64_t kDay = 24 * 60 * 60;

RetentionValueScore value(std::uint64_t recency, std::uint64_t frequency = 0)
{
    RetentionValueScore v;
    v.recency = recency;
    v.frequency = frequency;
    v.reuse_likelihood = 0;
    v.avoided_compute = 0;
    v.footprint = 0;
    v.restore_cost = 0;
    return v;
}

RetentionEntryKey key(std::uint64_t ns, std::uint64_t id)
{
    return {ns, id};
}
} // namespace

/** @test The code-owned map accepts ONLY the standard values and refuses
 * anything else with a precise diagnostic — never a silent remap. */
TEST(RetentionOwner, ClassMapAcceptsOnlyStandardValues)
{
    std::string refusal;
    auto in_memory = RetentionClassMap::forValue("in_memory", &refusal);
    ASSERT_TRUE(in_memory.has_value()) << refusal;
    EXPECT_EQ(in_memory->cls, RetentionClass::InMemory);
    auto day = RetentionClassMap::forValue("24h", &refusal);
    ASSERT_TRUE(day.has_value()) << refusal;
    EXPECT_EQ(day->cls, RetentionClass::TieredDay);
    EXPECT_EQ(day->min_lifetime_ordinals, kDay);
    auto absent = RetentionClassMap::forValue("", &refusal);
    ASSERT_TRUE(absent.has_value()) << refusal;
    EXPECT_EQ(absent->cls, RetentionClass::Default);
    EXPECT_EQ(absent->min_lifetime_ordinals, 0u);
    // A recognized field with an unsupported value FAILS precisely.
    EXPECT_FALSE(RetentionClassMap::forValue("48h", &refusal).has_value());
    EXPECT_FALSE(refusal.empty());
    EXPECT_FALSE(RetentionClassMap::forValue("forever").has_value());
}

/** @test Expiry is ordinal-based, measured from write/effective-use start;
 * an expired entry keeps its value but loses its class (it becomes a victim
 * before every unexpired class, even TieredDay). */
TEST(RetentionOwner, ExpiredLosesClassKeepsValue)
{
    RetentionOwner owner(8);
    ASSERT_TRUE(owner.admit(key(1, 1),
        RetentionIntent{RetentionClass::TieredDay, kDay}, 0, value(5), /*live=*/false));
    ASSERT_TRUE(owner.admit(key(1, 2),
        RetentionIntent{RetentionClass::TieredDay, kDay}, kDay, value(9), /*live=*/false));
    // At now = kDay the first entry expires; the second is unexpired until 2*kDay.
    EXPECT_TRUE(owner.expired(key(1, 1), kDay));
    EXPECT_FALSE(owner.expired(key(1, 2), kDay));
    auto victims = owner.selectVictims(kDay, 1);
    ASSERT_EQ(victims.size(), 1u);
    EXPECT_TRUE(victims[0] == key(1, 1)); // Expired (class lost) beats unexpired.
}

/** @test Refresh ONLY via effective use (the §8.2 receipt): a provisional
 * peek never refreshes; refresh restarts the lifetime AND advances the
 * stable tie-break sequence. */
TEST(RetentionOwner, RefreshOnlyOnEffectiveUse)
{
    RetentionOwner owner(8);
    ASSERT_TRUE(owner.admit(key(1, 1),
        RetentionIntent{RetentionClass::TieredDay, kDay}, 0, value(5), /*live=*/false));
    // Provisional inspection does NOT refresh: still expired at kDay.
    (void)owner.peek(key(1, 1));
    EXPECT_TRUE(owner.expired(key(1, 1), kDay));
    // Effective use at kDay/2 restarts the horizon from the use ordinal.
    EXPECT_TRUE(owner.refresh(key(1, 1), kDay / 2));
    EXPECT_FALSE(owner.expired(key(1, 1), kDay - 1));
    EXPECT_TRUE(owner.expired(key(1, 1), kDay + kDay / 2));
    // Refresh after expiry revives with a NEW stable sequence (tie-break
    // ordering treats it as the most recent use, not the old one).
    EXPECT_TRUE(owner.refresh(key(1, 1), kDay));
    EXPECT_FALSE(owner.expired(key(1, 1), kDay));
}

/** @test Victim order: unexpired class longest first (InMemory < Default <
 * TieredDay), then lower value score, then older effective-use sequence. */
TEST(RetentionOwner, VictimOrderClassThenValueThenSeq)
{
    // All entries UNEXPIRED (long minimums) so the class ranks decide:
    // InMemory before Default before TieredDay, regardless of value.
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit(key(1, 1),
        RetentionIntent{RetentionClass::InMemory, 2 * kDay}, 0, value(100), false));
    ASSERT_TRUE(owner.admit(key(1, 2),
        RetentionIntent{RetentionClass::Default, 2 * kDay}, 0, value(50), false));
    ASSERT_TRUE(owner.admit(key(1, 3),
        RetentionIntent{RetentionClass::TieredDay, 2 * kDay}, 0, value(1), false));
    auto victims = owner.selectVictims(1, 2);
    ASSERT_EQ(victims.size(), 2u);
    EXPECT_TRUE(victims[0] == key(1, 1)); // InMemory (lowest class) first.
    EXPECT_TRUE(victims[1] == key(1, 2)); // Default next; TieredDay survives.

    // Value order within one class: lower value first.
    RetentionOwner owner2(16);
    ASSERT_TRUE(owner2.admit(key(1, 1), RetentionIntent{RetentionClass::TieredDay, 2 * kDay},
        0, value(10), false));
    ASSERT_TRUE(owner2.admit(key(1, 2), RetentionIntent{RetentionClass::TieredDay, 2 * kDay},
        0, value(3), false));
    victims = owner2.selectVictims(1, 1);
    ASSERT_EQ(victims.size(), 1u);
    EXPECT_TRUE(victims[0] == key(1, 2)); // Lower value.

    // Stable tie-break: equal class and value, OLDER effective-use seq first.
    RetentionOwner owner3(16);
    ASSERT_TRUE(owner3.admit(key(1, 1), RetentionIntent{RetentionClass::TieredDay, 2 * kDay},
        0, value(5), false)); // seq 0
    ASSERT_TRUE(owner3.admit(key(1, 2), RetentionIntent{RetentionClass::TieredDay, 2 * kDay},
        0, value(5), false)); // seq 1
    ASSERT_TRUE(owner3.refresh(key(1, 1), 100)); // Now seq 2 (newer than 1).
    victims = owner3.selectVictims(1, 1);
    ASSERT_EQ(victims.size(), 1u);
    EXPECT_TRUE(victims[0] == key(1, 2)); // Older sequence (1 < 2) wins.
}

/** @test Live/protected state OUTRANKS every class: it is never selected, and
 * a protected TieredDay entry survives when only idle InMemory exists. */
TEST(RetentionOwner, LiveOutranksEveryClass)
{
    RetentionOwner owner(8);
    // Protected TieredDay (live continuation) + idle InMemory.
    ASSERT_TRUE(owner.admit(key(1, 1),
        RetentionIntent{RetentionClass::TieredDay, kDay}, 0, value(1), /*live=*/true));
    ASSERT_TRUE(owner.admit(key(1, 2),
        RetentionIntent{RetentionClass::InMemory, 0}, 0, value(99), /*live=*/false));
    auto victims = owner.selectVictims(1, 10);
    ASSERT_EQ(victims.size(), 1u);
    EXPECT_TRUE(victims[0] == key(1, 2)); // Only the idle entry.
    // After the continuation ends, the TieredDay entry becomes idle and the
    // ordering applies: the idle InMemory (horizon 0 -> expired, class lost)
    // is selected before the unexpired TieredDay — live state outranks every
    // class, and protection ended with the continuation.
    EXPECT_TRUE(owner.setLive(key(1, 1), false));
    victims = owner.selectVictims(1, 1);
    ASSERT_EQ(victims.size(), 1u);
    EXPECT_TRUE(victims[0] == key(1, 2));
}

/** @test Deficit-only: never more than requested; a huge deficit returns the
 * idle set exactly; zero deficit returns nothing. */
TEST(RetentionOwner, DeficitOnlySelection)
{
    RetentionOwner owner(8);
    for (int i = 0; i < 5; ++i)
        ASSERT_TRUE(owner.admit(key(1, static_cast<std::uint64_t>(i)),
            RetentionIntent{RetentionClass::TieredDay, kDay},
            0, value(static_cast<std::uint64_t>(i)), false));
    EXPECT_EQ(owner.selectVictims(1, 0).size(), 0u);
    EXPECT_EQ(owner.selectVictims(1, 2).size(), 2u);
    EXPECT_EQ(owner.selectVictims(1, 100).size(), 5u);
}

/** @test Bounded admitted metadata: beyond the admitted membership the owner
 * refuses without mutation (never an unbounded key map). */
TEST(RetentionOwner, BoundedMetadataRefusesFull)
{
    RetentionOwner owner(2);
    ASSERT_TRUE(owner.admit(key(1, 1), RetentionIntent{RetentionClass::Default, 0},
        0, value(1), false));
    ASSERT_TRUE(owner.admit(key(1, 2), RetentionIntent{RetentionClass::Default, 0},
        0, value(2), false));
    EXPECT_FALSE(owner.admit(key(1, 3), RetentionIntent{RetentionClass::Default, 0},
        0, value(3), false));
    EXPECT_EQ(owner.count(), 2u);
    EXPECT_EQ(owner.capacity(), 2u);
    // Retirement frees a slot; the same identity cannot be admitted twice.
    EXPECT_TRUE(owner.remove(key(1, 1)));
    EXPECT_TRUE(owner.admit(key(1, 3), RetentionIntent{RetentionClass::Default, 0},
        0, value(3), false));
    EXPECT_FALSE(owner.admit(key(1, 3), RetentionIntent{RetentionClass::Default, 0},
        1, value(4), false));
}

/** @test Shared pages take the highest UNEXPIRED class of any referencing
 * entry; an expired reference never raises the class (and only-expired
 * references carry NO class). */
TEST(RetentionOwner, SharedClassTakesHighestUnexpired)
{
    RetentionOwner owner(8);
    ASSERT_TRUE(owner.admit(key(1, 1),
        RetentionIntent{RetentionClass::InMemory, 0}, 0, value(1), false));
    ASSERT_TRUE(owner.admit(key(1, 2),
        RetentionIntent{RetentionClass::TieredDay, kDay}, 0, value(2), false));
    ASSERT_TRUE(owner.admit(key(1, 3),
        RetentionIntent{RetentionClass::TieredDay, kDay}, 0, value(3), false));
    // InMemory (expired, horizon 0) + TieredDay (unexpired at now=1): the
    // shared page carries the highest UNEXPIRED class — the expired reference
    // never raises or lowers it.
    auto cls = owner.sharedClass({key(1, 1), key(1, 2)}, 1);
    ASSERT_TRUE(cls.has_value());
    EXPECT_EQ(*cls, RetentionClass::TieredDay);
    // At now = kDay both TieredDay references are expired (write 0 + kDay):
    // no referencing entry carries an unexpired class.
    auto cls2 = owner.sharedClass({key(1, 2), key(1, 3)}, kDay);
    EXPECT_FALSE(cls2.has_value());
}

/** @test Zero capacity is unrepresentable and refuses construction. */
TEST(RetentionOwner, ZeroCapacityRefuses)
{
    EXPECT_THROW(RetentionOwner(0), std::invalid_argument);
}
} // namespace llaminar2
