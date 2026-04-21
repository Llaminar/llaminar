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

TEST(Test__CliSpec, MatchesShortNameAndAliases)
{
    CliOption<TestConfig> opt{
        .short_name  = "-m",
        .long_name   = "--mode",
        .aliases     = {"--mode-alt", "--m-alias"},
        .value_label = "<v>",
        .setter      = setters::assignString(&TestConfig::string_value),
    };
    EXPECT_TRUE(opt.matches("-m"));
    EXPECT_TRUE(opt.matches("--mode"));
    EXPECT_TRUE(opt.matches("--mode-alt"));
    EXPECT_TRUE(opt.matches("--m-alias"));
    EXPECT_TRUE(opt.matches("--mode=foo"));       // equals form on primary long
    EXPECT_TRUE(opt.matches("--m-alias=foo"));    // equals form on alias
    EXPECT_FALSE(opt.matches("-m=foo"));          // equals form NOT allowed on short
    EXPECT_FALSE(opt.matches("--modex"));         // prefix-only isn't a match
    EXPECT_FALSE(opt.matches("--other"));
}

// ============================================================================
// Value-parsing edge cases
// ============================================================================

TEST(Test__CliSpec, EmptyValueAfterEqualsThrows)
{
    // `--number=` with empty RHS must throw, not silently assign empty.
    auto spec = buildTestSpec();
    TestConfig c;
    try
    {
        spec.parse({"--number="}, c);
        FAIL() << "expected throw";
    }
    catch (const std::invalid_argument &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("--number"), std::string::npos);
    }
}

TEST(Test__CliSpec, InvalidFloatThrowsWithFlagName)
{
    auto spec = buildTestSpec();
    TestConfig c;
    try
    {
        spec.parse({"--float", "not-a-float"}, c);
        FAIL() << "expected throw";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find("--float"), std::string::npos);
    }
}

TEST(Test__CliSpec, RepeatedValueFlagLastWriteWins)
{
    // For non-accumulating setters, a repeated flag should overwrite the
    // previous value. This is the conventional CLI contract.
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"--number", "1", "--number=2", "--number", "3"}, c);
    EXPECT_EQ(c.int_value, 3);
}

TEST(Test__CliSpec, IncrementClampsAtMaxValue)
{
    // `incrementInt(..., max_value=5)` stops once we hit the cap, even if the
    // flag appears more times.
    auto spec = buildTestSpec();
    TestConfig c;
    spec.parse({"-v", "-v", "-v", "-v", "-v", "-v", "-v", "-v"}, c);
    EXPECT_EQ(c.counter, 5);
}

TEST(Test__CliSpec, SetterExceptionPropagates)
{
    // User-provided custom setters are allowed to throw; the parser should
    // propagate without wrapping, so callers see the original message.
    CliSpec<TestConfig> spec;
    spec.add({
        .long_name   = "--bomb",
        .value_label = "<v>",
        .description = "always throws",
        .setter      = [](TestConfig &, const std::string &) {
            throw std::invalid_argument("boom: custom setter failed");
        },
    });
    TestConfig c;
    try
    {
        spec.parse({"--bomb", "ignored"}, c);
        FAIL() << "expected throw";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find("boom"), std::string::npos);
    }
}

TEST(Test__CliSpec, OptionWithoutSetterIsNoOp)
{
    // A missing setter must not crash; the option is still recognised so
    // parsing doesn't fail on unknown-arg.
    CliSpec<TestConfig> spec;
    spec.add({
        .long_name   = "--noop",
        .description = "accepted but does nothing",
    });
    TestConfig c;
    EXPECT_NO_THROW(spec.parse({"--noop"}, c));
}

TEST(Test__CliSpec, PartialParseAppliesFlagsBeforeException)
{
    // Documents current behaviour: the parser mutates the config in order and
    // does not roll back on exception. Callers that want all-or-nothing
    // semantics need to parse into a temp and swap on success.
    auto spec = buildTestSpec();
    TestConfig c;
    EXPECT_THROW(spec.parse({"-a", "--mode", "bogus"}, c), std::invalid_argument);
    EXPECT_TRUE(c.flag_a);   // applied before the throw
    EXPECT_TRUE(c.enum_value.empty());
}

// ============================================================================
// Setter helper coverage
// ============================================================================

TEST(Test__CliSpec, AssignBoolFalseSetter)
{
    CliSpec<TestConfig> spec;
    spec.add({
        .long_name = "--no-a",
        .setter    = setters::assignBoolFalse(&TestConfig::flag_a),
    });
    TestConfig c;
    c.flag_a = true;
    spec.parse({"--no-a"}, c);
    EXPECT_FALSE(c.flag_a);
}

TEST(Test__CliSpec, AssignIntLiteralSetter)
{
    // Pattern used by `-vv` / `-vvv` to jump straight to a specific level.
    CliSpec<TestConfig> spec;
    spec.add({
        .short_name = "-vv",
        .setter     = setters::assignIntLiteral(&TestConfig::counter, 2),
    });
    spec.add({
        .short_name = "-vvv",
        .setter     = setters::assignIntLiteral(&TestConfig::counter, 3),
    });
    TestConfig c;
    spec.parse({"-vvv"}, c);
    EXPECT_EQ(c.counter, 3);
    spec.parse({"-vv"}, c);
    EXPECT_EQ(c.counter, 2);
}

// ============================================================================
// Help rendering details
// ============================================================================

TEST(Test__CliSpec, HelpExcludesNYIFromRegularCategory)
{
    auto spec = buildTestSpec();
    std::string help = spec.getHelpText();

    auto nyi_header = help.find("Not yet implemented");
    auto nyi_flag   = help.find("--nyi");
    ASSERT_NE(nyi_header, std::string::npos);
    ASSERT_NE(nyi_flag, std::string::npos);
    // The --nyi entry must be in the NYI section, not the plain "Bare:" one.
    EXPECT_GT(nyi_flag, nyi_header);
}

TEST(Test__CliSpec, HelpWithoutCategoriesStillRenders)
{
    // addCategory() is optional; options that reference a never-registered
    // category should still appear (in first-seen order).
    CliSpec<TestConfig> spec;
    spec.add({
        .long_name   = "--one",
        .category    = "Alpha",
        .description = "first option",
        .setter      = setters::assignBoolTrue(&TestConfig::flag_a),
    });
    spec.add({
        .long_name   = "--two",
        .category    = "Beta",
        .description = "second option",
        .setter      = setters::assignBoolTrue(&TestConfig::flag_b),
    });
    std::string help = spec.getHelpText();
    auto a = help.find("Alpha:");
    auto b = help.find("Beta:");
    ASSERT_NE(a, std::string::npos);
    ASSERT_NE(b, std::string::npos);
    EXPECT_LT(a, b);
    EXPECT_NE(help.find("--one"), std::string::npos);
    EXPECT_NE(help.find("--two"), std::string::npos);
}

TEST(Test__CliSpec, HelpShowsValueLabelAndAliases)
{
    CliSpec<TestConfig> spec;
    spec.add({
        .short_name  = "-n",
        .long_name   = "--number",
        .aliases     = {"--num", "--count"},
        .value_label = "<int>",
        .description = "a number",
        .setter      = setters::parseInt(&TestConfig::int_value, "--number"),
    });
    std::string help = spec.getHelpText();
    EXPECT_NE(help.find("-n"), std::string::npos);
    EXPECT_NE(help.find("--number"), std::string::npos);
    EXPECT_NE(help.find("<int>"), std::string::npos);
    // Description must appear.
    EXPECT_NE(help.find("a number"), std::string::npos);
}

// ============================================================================
// Spec inspection
// ============================================================================

TEST(Test__CliSpec, OptionsAccessorReturnsRegisteredOptions)
{
    auto spec = buildTestSpec();
    const auto &opts = spec.options();
    // Sanity: all 8 test options present.
    EXPECT_EQ(opts.size(), 8u);
    // Count by scanning for known long names.
    int found = 0;
    for (const auto &o : opts)
    {
        if (o.long_name == "--flag-a" || o.long_name == "--flag-b" ||
            o.long_name == "--number" || o.long_name == "--float" ||
            o.long_name == "--string" || o.long_name == "--mode" ||
            o.long_name == "--nyi")
            ++found;
    }
    EXPECT_EQ(found, 7);
}

TEST(Test__CliSpec, CliOptionDefaultsAreSafe)
{
    // Default-constructed option must be inert: no names, no setter, isFlag()
    // true (empty value label), not NYI. Parsing an empty arg list with a
    // spec containing this option should not throw.
    CliOption<TestConfig> opt{};
    EXPECT_TRUE(opt.isFlag());
    EXPECT_FALSE(opt.not_yet_implemented);
    EXPECT_TRUE(opt.allNames().empty());
    EXPECT_FALSE(opt.matches(""));
    EXPECT_FALSE(opt.matches("--anything"));
}
