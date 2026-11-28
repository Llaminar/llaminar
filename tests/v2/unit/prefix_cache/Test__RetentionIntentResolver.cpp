/**
 * @file Test__RetentionIntentResolver.cpp
 * @brief §8.4 standard-field intent resolution contracts (P0 baseline §1/§4).
 */
#include "execution/prefix_cache/RetentionIntentResolver.h"

#include <gtest/gtest.h>

namespace llaminar2
{
/** @test The two standard fields resolve to the exact typed intent; absent
 * values select the code-owned default and keyless requests carry no
 * namespace. */
TEST(RetentionIntentResolver, StandardFieldsResolvePrecisely)
{
    RetentionIntentResolver resolver{std::nullopt};
    auto r = resolver.resolve("in_memory", "my-key");
    ASSERT_TRUE(r.ok()) << r.refusal;
    EXPECT_EQ(r.intent->cls, RetentionClass::InMemory);
    ASSERT_TRUE(r.namespace_id.has_value());
    EXPECT_GT(*r.namespace_id, 0u);
    // Same key -> same namespace (stable partition), different key -> different.
    EXPECT_EQ(resolver.resolve("in_memory", "my-key").namespace_id, r.namespace_id);
    EXPECT_NE(resolver.resolve("in_memory", "other-key").namespace_id, r.namespace_id);
    // Absent retention -> default class; keyless -> no namespace.
    r = resolver.resolve("", "");
    ASSERT_TRUE(r.ok()) << r.refusal;
    EXPECT_EQ(r.intent->cls, RetentionClass::Default);
    EXPECT_FALSE(r.namespace_id.has_value());
    // Retention without a key is accepted (keyless namespace, P0 §5).
    r = resolver.resolve("24h", "");
    ASSERT_TRUE(r.ok()) << r.refusal;
    EXPECT_EQ(r.intent->cls, RetentionClass::TieredDay);
    EXPECT_FALSE(r.namespace_id.has_value());
    // Key without retention is accepted (default class).
    r = resolver.resolve("", "k");
    ASSERT_TRUE(r.ok()) << r.refusal;
    EXPECT_EQ(r.intent->cls, RetentionClass::Default);
    EXPECT_TRUE(r.namespace_id.has_value());
}

/** @test A recognized field with an unsupported value refuses precisely —
 * never silently mapped (P0 §1: invalid_request_error semantics). */
TEST(RetentionIntentResolver, UnsupportedValueRefuses)
{
    RetentionIntentResolver resolver{std::nullopt};
    auto r = resolver.resolve("48h", "k");
    EXPECT_FALSE(r.ok());
    EXPECT_FALSE(r.refusal.empty());
    EXPECT_NE(r.refusal.find("accepted values"), std::string::npos);
    EXPECT_FALSE(resolver.resolve("forever", "k").ok());
}

/** @test The operator ceiling (P0 §4) refuses a class rank above the ceiling
 * precisely and never demotes; a ceiling at or above the class accepts. */
TEST(RetentionIntentResolver, CeilingRefusesAboveNeverDemotes)
{
    RetentionIntentResolver ceiling_default{RetentionClass::Default};
    auto r = ceiling_default.resolve("24h", "k");
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.refusal.find("exceeds the configured retention classes ceiling"),
        std::string::npos);
    // At the ceiling -> accepted.
    EXPECT_TRUE(ceiling_default.resolve("in_memory", "k").ok());
    EXPECT_TRUE(ceiling_default.resolve("", "k").ok()); // Default == ceiling.
    RetentionIntentResolver ceiling_day{RetentionClass::TieredDay};
    EXPECT_TRUE(ceiling_day.resolve("24h", "k").ok());
    EXPECT_TRUE(ceiling_day.resolve("in_memory", "k").ok());
    // No ceiling -> everything accepted.
    RetentionIntentResolver none{std::nullopt};
    EXPECT_TRUE(none.resolve("24h", "k").ok());
    EXPECT_TRUE(none.resolve("in_memory", "k").ok());
}

/** @test An invalid ceiling value is unrepresentable and refuses at
 * construction. */
TEST(RetentionIntentResolver, InvalidCeilingRefusesConstruction)
{
    EXPECT_THROW(RetentionIntentResolver{
        static_cast<RetentionClass>(999)}, std::invalid_argument);
}
} // namespace llaminar2
