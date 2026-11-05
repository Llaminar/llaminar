/**
 * @file Test__CPUISAProfile.cpp
 * @brief Native ISA admission and architecture-total dispatch without feature spoofing.
 *
 * The resolver takes request intent explicitly and reads actual CPU features;
 * tests never mutate a cached environment or substitute a capability table.
 * Probe functions intentionally omit leaves belonging to another architecture,
 * so an ARM dispatch macro that names an x86 implementation cannot compile.
 */
#include "utils/CPUFeatures.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>

namespace llaminar2::test
{
namespace
{
/** @brief Return the scalar diagnostic route identifier. */
int probe_scalar(int value) { return value; }
/** @brief Publish the scalar diagnostic route identifier. */
void publish_scalar(int &destination, int value) { destination = value; }
#if LLAMINAR_COMPILED_WITH_NEON
/** @brief Return the native ARM family identifier, irrespective of DotProd envelope. */
int probe_neon(int value) { return value + 3; }
/** @brief Publish the native ARM family identifier. */
void publish_neon(int &destination, int value) { destination = value + 3; }
#else
/** @brief Return the native x86 AVX2 route identifier. */
int probe_avx2(int value) { return value + 1; }
/** @brief Publish the native x86 AVX2 route identifier. */
void publish_avx2(int &destination, int value) { destination = value + 1; }
#if LLAMINAR_COMPILED_WITH_AVX512
/** @brief Return the native x86 AVX512 route identifier. */
int probe_avx512(int value) { return value + 2; }
/** @brief Publish the native x86 AVX512 route identifier. */
void publish_avx512(int &destination, int value) { destination = value + 2; }
#endif
#endif
} // namespace

/** @test All architecture-independent display values remain stable and explicit. */
TEST(CPUISAProfile, StableNamesCoverEveryNativeISA)
{
    EXPECT_STREQ(isaLevelName(ISALevel::Scalar), "Scalar");
    EXPECT_STREQ(isaLevelName(ISALevel::AVX2), "AVX2");
    EXPECT_STREQ(isaLevelName(ISALevel::AVX512), "AVX-512");
    EXPECT_STREQ(isaLevelName(ISALevel::NEON), "NEON");
    EXPECT_STREQ(isaLevelName(ISALevel::NEONDotProd), "NEON+DotProd");
    EXPECT_STREQ(isaLevelName(static_cast<ISALevel>(255)), "Unknown");
}

/** @test Automatic selection authenticates the actual compiled/native instruction envelope. */
TEST(CPUISAProfile, NativeDefaultMatchesRealCapabilities)
{
    EXPECT_EQ(resolveCPUISALevel(), resolveCPUISALevel(""));
    EXPECT_EQ(resolveCPUISALevel("ScAlAr"), ISALevel::Scalar);
#if LLAMINAR_COMPILED_WITH_NEON
    EXPECT_TRUE(cpu_supports_neon());
    EXPECT_FALSE(cpu_supports_avx2());
    EXPECT_FALSE(cpu_supports_avx512());
    EXPECT_EQ(resolveCPUISALevel("NeOn"), ISALevel::NEON);
#if LLAMINAR_COMPILED_WITH_NEON_DOTPROD
    EXPECT_TRUE(cpu_supports_neon_dotprod());
    EXPECT_EQ(resolveCPUISALevel(), ISALevel::NEONDotProd);
    EXPECT_EQ(resolveCPUISALevel("neon-dotprod"), ISALevel::NEONDotProd);
#else
    EXPECT_EQ(resolveCPUISALevel(), ISALevel::NEON);
    EXPECT_THROW(resolveCPUISALevel("neon-dotprod"), std::invalid_argument);
#endif
#else
    EXPECT_FALSE(cpu_supports_neon());
    EXPECT_FALSE(cpu_supports_neon_dotprod());
    const auto expected = cpu_supports_avx512() ? ISALevel::AVX512
        : cpu_supports_avx2() ? ISALevel::AVX2 : ISALevel::Scalar;
    EXPECT_EQ(resolveCPUISALevel(), expected);
#endif
}

/** @test Invalid or other-architecture requests fail rather than selecting scalar/another ISA. */
TEST(CPUISAProfile, InvalidAndIncompatibleRequestsFailClosed)
{
    EXPECT_THROW(resolveCPUISALevel("unknown"), std::invalid_argument);
    EXPECT_THROW(resolveCPUISALevel("neon_dotprod"), std::invalid_argument);
#if LLAMINAR_COMPILED_WITH_NEON
    EXPECT_THROW(resolveCPUISALevel("avx2"), std::invalid_argument);
    EXPECT_THROW(resolveCPUISALevel("avx512"), std::invalid_argument);
#else
    EXPECT_THROW(resolveCPUISALevel("neon"), std::invalid_argument);
    EXPECT_THROW(resolveCPUISALevel("neon-dotprod"), std::invalid_argument);
    if (!cpu_supports_avx512())
        EXPECT_THROW(resolveCPUISALevel("avx512"), std::invalid_argument);
#endif
}

/** @test Void, assignment and value dispatch name only compiled architecture-native leaves. */
TEST(CPUISAProfile, AllMacroFormsUseTheAdmittedNativeLeaf)
{
    const auto level = activeISALevel();
    const int route = level == ISALevel::NEON || level == ISALevel::NEONDotProd
        ? 3 : static_cast<int>(level);
    EXPECT_EQ(ISA_DISPATCH_RETVAL(probe, 7), 7 + route);
    int assigned = -1;
    ISA_DISPATCH_RET(assigned, probe, 9);
    EXPECT_EQ(assigned, 9 + route);
    int published = -1;
    ISA_DISPATCH_VOID(publish, published, 11);
    EXPECT_EQ(published, 11 + route);
}

#if defined(__aarch64__)
/** @test ARM observation is immutable, device-free and rejects invalid feature identifiers. */
TEST(CPUISAProfile, NativeARMCapabilitiesHaveOneAuthority)
{
    using cpu::ARMCPUCapabilities;
    using cpu::ARMCPUFeature;
    const auto &capabilities = ARMCPUCapabilities::native();
    EXPECT_EQ(&capabilities, &ARMCPUCapabilities::native());
    EXPECT_TRUE(capabilities.supports(ARMCPUFeature::ASIMD));
    EXPECT_NO_THROW(capabilities.require(ARMCPUFeature::ASIMD));
    EXPECT_EQ(capabilities.supports(ARMCPUFeature::DotProd), cpu_supports_neon_dotprod());
    for (const auto feature : {ARMCPUFeature::DotProd, ARMCPUFeature::FP16Arithmetic,
                              ARMCPUFeature::BF16Arithmetic, ARMCPUFeature::IntegerMatMul})
    {
        if (capabilities.supports(feature))
            EXPECT_NO_THROW(capabilities.require(feature));
        else
            EXPECT_THROW(capabilities.require(feature), std::runtime_error);
    }
    EXPECT_STREQ(ARMCPUCapabilities::name(ARMCPUFeature::DotProd), "DotProd");
    const auto invalid = static_cast<ARMCPUFeature>(0xffffffffu);
    EXPECT_THROW((void)capabilities.supports(invalid), std::invalid_argument);
    EXPECT_THROW(capabilities.require(invalid), std::invalid_argument);
}
#endif
} // namespace llaminar2::test
