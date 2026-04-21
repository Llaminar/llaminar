/**
 * @file Test__CliSpec.cpp
 * @brief Unit tests for the structured CLI specification framework.
 *
 * Exercises CliSpec/CliOption against a minimal throwaway config so the
 * tests stay focused on the parsing/help machinery rather than any specific
 * production config layout.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>
#include "config/CliSpec.h"

#include <sstream>

using namespace llaminar2;

namespace
{
    struct TestConfig
    {
        bool flag_a = false;
        bool flag_b = false;
        int int_value = -1;
        int counter = 0;
        float float_value = 0.0f;
        std::string string_value;
        std::string enum_value;
    };

    CliSpec<TestConfig> buildTestSpec()
    {
        CliSpec<TestConfig> spec;
        spec.addCategory("Bare").addCategory("Values").addCategory("Enums");

        spec.add({"-a", "--flag-a", {}, "Bare", "",
                  "Enable flag A", {}, false,
                  setters::assignBoolTrue(&TestConfig::flag_a)});
        spec.add({"", "--flag-b", {"--alias-b"}, "Bare", "",
                  "Enable flag B (with alias)", {}, false,
                  setters::assignBoolTrue(&TestConfig::flag_b)});
        spec.add({"-v", "", {}, "Bare", "",
                  "Increment counter", {}, false,
                  setters::incrementInt(&TestConfig::counter, 5)});

        spec.add({"-n", "--number", {}, "Values", "<n>",
                  "An integer", {}, false,
                  setters::parseInt(&TestConfig::int_value, "--number")});
        spec.add({"-f", "--float", {}, "Values", "<f>",
                  "A float", {}, false,
                  setters::parseFloat(&TestConfig::float_value, "--float")});
        spec.add({"-s", "--string", {}, "Values", "<str>",
                  "A string", {}, false,
                  setters::assignString(&TestConfig::string_value)});

        spec.add({"", "--mode", {}, "Enums", "<mode>",
                  "Accepted: red, green, blue",
                  {"red", "green", "blue"}, false,
                  setters::assignString(&TestConfig::enum_value)});

        spec.add({"", "--nyi", {}, "Bare", "",
                  "Accepted but not yet implemented", {}, true,
                  setters::assignBoolTrue(&TestConfig::flag_b)});

        return spec;
    }
} // namespace

// ============================================================================
// Basic matching and parsing
// ============================================================================

TEST(Test__CliSpec, ParsesBareFlags)
{
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"-a", "--flag-b"}, c);
    EXPECT_TRUE(c.flag_a);
    EXPECT_TRUE(c.flag_b);
}

TEST(Test__CliSpec, AliasesAreAccepted)
{
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"--alias-b"}, c);
    EXPECT_TRUE(c.flag_b);
}

TEST(Test__CliSpec, ParsesValueSeparatedBySpace)
{
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"--number", "42"}, c);
    EXPECT_EQ(c.int_value, 42);
}

TEST(Test__CliSpec, ParsesValueWithEqualsSyntax)
{
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"--number=7", "--float=1.5", "--string=hello"}, c);
    EXPECT_EQ(c.int_value, 7);
    EXPECT_FLOAT_EQ(c.float_value, 1.5f);
    EXPECT_EQ(c.string_value, "hello");
}

TEST(Test__CliSpec, AcceptsNegativeNumberAsValue)
{
    // Regression guard: the old parser rejected next-token values starting
    // with '-', which broke --seed -1 style flags. The new spec-based parser
    // accepts arbitrary next tokens for value-carrying options.
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"--number", "-5"}, c);
    EXPECT_EQ(c.int_value, -5);
}

TEST(Test__CliSpec, IncrementSetterStacksAcrossRepeats)
{
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"-v", "-v", "-v"}, c);
    EXPECT_EQ(c.counter, 3);
}

// ============================================================================
// Validation
// ============================================================================

TEST(Test__CliSpec, UnknownArgumentThrows)
{
    auto spec = buildTestSpec();
    TestConfig c;
    EXPECT_THROW(spec.parse({"--not-a-real-flag"}, c), std::invalid_argument);
}

TEST(Test__CliSpec, AllowUnknownSkipsUnrecognisedArgs)
{
    auto spec = buildTestSpec();
    TestConfig c;
    // allow_unknown=true is used by the two-pass --config discovery; verify
    // it doesn't throw on unknowns but still applies recognised flags.
    EXPECT_NO_THROW(spec.parse({"--bogus", "-a"}, c, /*allow_unknown=*/true));
    EXPECT_TRUE(c.flag_a);
}

TEST(Test__CliSpec, MissingValueThrows)
{
    auto spec = buildTestSpec();
    TestConfig c;
    EXPECT_THROW(spec.parse({"--number"}, c), std::invalid_argument);
}

TEST(Test__CliSpec, InvalidIntegerThrowsWithFlagName)
{
    auto spec = buildTestSpec();
    TestConfig c;
    try
    {
        spec.parse({"--number", "not-a-number"}, c);
        FAIL() << "expected throw";
    }
    catch (const std::invalid_argument &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("--number"), std::string::npos)
            << "error message should name the offending flag, got: " << msg;
    }
}

TEST(Test__CliSpec, ValidValuesWhitelistEnforced)
{
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"--mode", "green"}, c);
    EXPECT_EQ(c.enum_value, "green");

    EXPECT_THROW(spec.parse({"--mode", "purple"}, c), std::invalid_argument);
}

TEST(Test__CliSpec, ValidValuesErrorListsAcceptedValues)
{
    auto spec = buildTestSpec();
    TestConfig c;
    try
    {
        spec.parse({"--mode", "purple"}, c);
        FAIL() << "expected throw";
    }
    catch (const std::invalid_argument &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("red"), std::string::npos);
        EXPECT_NE(msg.find("green"), std::string::npos);
        EXPECT_NE(msg.find("blue"), std::string::npos);
    }
}

// ============================================================================
// Help generation
// ============================================================================

TEST(Test__CliSpec, HelpContainsAllShortAndLongNames)
{
    auto spec = buildTestSpec();
    std::string help = spec.getHelpText();

    EXPECT_NE(help.find("-a"), std::string::npos);
    EXPECT_NE(help.find("--flag-a"), std::string::npos);
    EXPECT_NE(help.find("--flag-b"), std::string::npos);
    EXPECT_NE(help.find("-n"), std::string::npos);
    EXPECT_NE(help.find("--number"), std::string::npos);
    EXPECT_NE(help.find("--mode"), std::string::npos);
}

TEST(Test__CliSpec, HelpGroupsByCategoryInDeclaredOrder)
{
    auto spec = buildTestSpec();
    std::string help = spec.getHelpText();

    auto bare = help.find("Bare:");
    auto values = help.find("Values:");
    auto enums = help.find("Enums:");

    ASSERT_NE(bare, std::string::npos);
    ASSERT_NE(values, std::string::npos);
    ASSERT_NE(enums, std::string::npos);
    EXPECT_LT(bare, values);
    EXPECT_LT(values, enums);
}

TEST(Test__CliSpec, HelpRendersNYISection)
{
    auto spec = buildTestSpec();
    std::string help = spec.getHelpText();

    EXPECT_NE(help.find("Not yet implemented"), std::string::npos);
    EXPECT_NE(help.find("--nyi"), std::string::npos);
}

TEST(Test__CliSpec, HelpHeaderAndFooterRendered)
{
    auto spec = buildTestSpec();
    std::string help = spec.getHelpText("MY HEADER", "MY FOOTER");
    EXPECT_NE(help.find("MY HEADER"), std::string::npos);
    EXPECT_NE(help.find("MY FOOTER"), std::string::npos);
    // Header must precede footer.
    EXPECT_LT(help.find("MY HEADER"), help.find("MY FOOTER"));
}

TEST(Test__CliSpec, NYIFlagsStillParse)
{
    // Flags marked not_yet_implemented are still accepted on the command
    // line for back-compat; they just render in the NYI help section.
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"--nyi"}, c);
    EXPECT_TRUE(c.flag_b);
}

// ============================================================================
// matches() edge cases
// ============================================================================

TEST(Test__CliSpec, EqualsFormOnlyForLongNames)
{
    // Short-form flags should not accept --x=value form since short forms
    // are conventionally single-dash.
    CliOption<TestConfig> opt{"-x", "", {}, "Bare", "<val>",
                              "short-only", {}, false,
                              setters::assignString(&TestConfig::string_value)};
    EXPECT_TRUE(opt.matches("-x"));
    EXPECT_FALSE(opt.matches("-x=foo"));
}

TEST(Test__CliSpec, FlagsRejectEqualsForm)
{
    // Bare flags must not be spoofed as --flag=value.
    CliOption<TestConfig> opt{"", "--bare", {}, "Bare", "",
                              "bare flag", {}, false,
                              setters::assignBoolTrue(&TestConfig::flag_a)};
    EXPECT_TRUE(opt.matches("--bare"));
    EXPECT_FALSE(opt.matches("--bare=true"));
}
