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
#if LLAMINAR_COMPILED_WITH_NEON_I8MM
    // I8MM is a leaf-instruction envelope, not an arithmetic level: the
    // binary admits the hardware feature but keeps the DotProd identity.
    EXPECT_TRUE(cpu_supports_neon_i8mm());
    EXPECT_EQ(compiledCPUISALevel(), ISALevel::NEONDotProd);
    EXPECT_THROW(resolveCPUISALevel("neon-i8mm"), std::invalid_argument);
#endif
#else
    EXPECT_EQ(resolveCPUISALevel(), ISALevel::NEON);
    EXPECT_THROW(resolveCPUISALevel("neon-dotprod"), std::invalid_argument);
#endif
#else
    EXPECT_FALSE(cpu_supports_neon());
    EXPECT_FALSE(cpu_supports_neon_dotprod());
    EXPECT_FALSE(cpu_supports_neon_i8mm());
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
    EXPECT_EQ(capabilities.supports(ARMCPUFeature::IntegerMatMul), cpu_supports_neon_i8mm());
    for (const auto feature : {ARMCPUFeature::DotProd, ARMCPUFeature::FP16Arithmetic,
                              ARMCPUFeature::BF16Arithmetic, ARMCPUFeature::IntegerMatMul})
    {
        if (capabilities.supports(feature))
            EXPECT_NO_THROW(capabilities.require(feature));
        else
            EXPECT_THROW(capabilities.require(feature), std::runtime_error);
    }
    EXPECT_STREQ(ARMCPUCapabilities::name(ARMCPUFeature::DotProd), "DotProd");
    // The shared envelope admission rejects a machine lacking any compiled
    // feature, including I8MM on a DotProd-only host such as M1.
    const auto bits = [](std::initializer_list<ARMCPUFeature> features) {
        std::uint32_t value = 0;
        for (const auto feature : features) value |= static_cast<std::uint32_t>(feature);
        return value;
    };
    const auto i8mm_envelope = bits({ARMCPUFeature::ASIMD, ARMCPUFeature::DotProd, ARMCPUFeature::IntegerMatMul});
    EXPECT_THROW(ARMCPUCapabilities::requireEnvelope(bits({ARMCPUFeature::ASIMD, ARMCPUFeature::DotProd}), i8mm_envelope),
        std::runtime_error);
    EXPECT_THROW(ARMCPUCapabilities::requireEnvelope(bits({ARMCPUFeature::ASIMD, ARMCPUFeature::IntegerMatMul}), i8mm_envelope),
        std::runtime_error);
    EXPECT_NO_THROW(ARMCPUCapabilities::requireEnvelope(i8mm_envelope, i8mm_envelope));
    EXPECT_NO_THROW(ARMCPUCapabilities::requireEnvelope(i8mm_envelope, bits({ARMCPUFeature::ASIMD})));
#if defined(__ARM_FEATURE_MATMUL_INT8)
    // The declared I8MM envelope carries the FP16 FML long multiply-add of
    // the half-operand leaf.
    EXPECT_EQ(ARMCPUCapabilities::compiledEnvelopeFeatures(),
              i8mm_envelope | bits({ARMCPUFeature::FP16FML}));
#elif defined(__ARM_FEATURE_DOTPROD)
    EXPECT_EQ(ARMCPUCapabilities::compiledEnvelopeFeatures(), bits({ARMCPUFeature::ASIMD, ARMCPUFeature::DotProd}));
#endif
    EXPECT_NO_THROW(capabilities.requireCompiledEnvelope());
    const auto invalid = static_cast<ARMCPUFeature>(0xffffffffu);
    EXPECT_THROW((void)capabilities.supports(invalid), std::invalid_argument);
    EXPECT_THROW(capabilities.require(invalid), std::invalid_argument);
}
#endif

namespace
{
/** @brief Encode explicit deterministic-cache metadata, not runtime CPU capabilities.
 * @param level Architectural cache level.
 * @param type Data/unified/instruction/null record type.
 * @param sharers Logical CPUs named by the cache descriptor.
 * @param line Cache-line bytes.
 * @param ways Cache associativity.
 * @param sets Cache-set count.
 * @return EAX/EBX/ECX/EDX fixture words.
 */
constexpr std::array<std::uint32_t, 4> cacheRecord(std::uint32_t level, std::uint32_t type,
    std::uint32_t sharers, std::uint32_t line, std::uint32_t ways, std::uint32_t sets)
{
    return {type | (level << 5) | ((sharers - 1) << 14),
        ((ways - 1) << 22) | (line - 1), sets - 1, 0};
}
/** @brief Explicit parser fixture; never installed as the process hardware reader. */
struct CacheRecords
{
    bool extended = false;
    bool topology = true;
    std::uint32_t enabled_smt_threads = 2;
    std::uint32_t family = 0x19;
    std::array<std::array<std::uint32_t, 4>, 3> caches{
        cacheRecord(1, 1, 2, 64, 8, 64), cacheRecord(2, 3, 2, 64, 8, 2048),
        cacheRecord(3, 3, 16, 64, 16, 8192)};
    /** @brief Read a named fixture record with bounded cache/topology subleaves.
     * @param leaf Explicit architectural leaf ID.
     * @param sub Explicit subleaf ID.
     * @return Fixture registers or a terminating null record.
     */
    std::array<std::uint32_t, 4> operator()(std::uint32_t leaf, std::uint32_t sub) const
    {
        if (leaf == 0) return extended
            ? std::array<std::uint32_t, 4>{1u, 0x68747541u, 0x444d4163u, 0x69746e65u}
            : std::array<std::uint32_t, 4>{0x1fu, 0x756e6547u, 0x6c65746eu, 0x49656e69u};
        if (leaf == 1) return {(15u << 8) | ((family - 15u) << 20), 0, 0, 0};
        if (leaf == 0x80000000u) return {extended ? 0x8000001eu : 0x80000000u, 0, 0, 0};
        if (topology && !extended && leaf == 0x1f && sub == 0) return {1, enabled_smt_threads, 1u << 8, 0};
        if (topology && extended && leaf == 0x80000001u) return {0, 0, 1u << 22, 0};
        if (topology && extended && leaf == 0x8000001eu) return {0, 1u << 8, 0, 0};
        if (leaf == (extended ? 0x8000001du : 4u) && sub < caches.size()) return caches[sub];
        return {};
    }
};
} // namespace

/** @test Missing records never acquire historical32KiB/256KiB or64-byte defaults. */
TEST(CPUCacheObservation, UnknownRecordsStayUnknownWithoutConvenienceDefaults)
{
    const auto absent = cpu::observeX86CacheGeometry([](std::uint32_t, std::uint32_t) {
        return std::array<std::uint32_t, 4>{}; });
    EXPECT_EQ(absent.private_l1d_bytes, 0u);
    EXPECT_EQ(absent.private_l2_bytes, 0u);
    EXPECT_EQ(absent.shared_l3_bytes, 0u);
    EXPECT_EQ(absent.cache_line_bytes, 0u);
    CacheRecords no_topology;
    no_topology.topology = false;
    const auto unproven = cpu::observeX86CacheGeometry(no_topology);
    EXPECT_EQ(unproven.private_l1d_bytes, 0u);
    EXPECT_EQ(unproven.private_l2_bytes, 0u);
    EXPECT_EQ(unproven.cache_line_bytes, 0u);
    EXPECT_EQ(unproven.shared_l3_bytes, 8u * 1024 * 1024);
}

/** @test Private SMT caches retain exact byte/line/way observations, not package totals. */
TEST(CPUCacheObservation, IntelAndAMDRecordsRetainExactPrivateGeometry)
{
    CacheRecords fixture;
    const auto intel = cpu::observeX86CacheGeometry(fixture);
    EXPECT_EQ(intel.private_l1d_bytes, 32u * 1024);
    EXPECT_EQ(intel.private_l2_bytes, 1024u * 1024);
    EXPECT_EQ(intel.shared_l3_bytes, 8u * 1024 * 1024);
    EXPECT_EQ(intel.cache_line_bytes, 64u);
    EXPECT_EQ(intel.private_l2_ways, 8u);
    EXPECT_EQ(intel.shared_l3_ways, 16u);
    fixture.enabled_smt_threads = 1;
    const auto smt_disabled = cpu::observeX86CacheGeometry(fixture);
    EXPECT_EQ(smt_disabled.private_l2_bytes, 1024u * 1024);
    fixture.extended = true;
    fixture.caches[0] = cacheRecord(1, 1, 2, 64, 12, 64);
    fixture.caches[2] = cacheRecord(3, 3, 16, 64, 16, 32768);
    const auto amd = cpu::observeX86CacheGeometry(fixture);
    EXPECT_EQ(amd.private_l1d_bytes, 48u * 1024);
    EXPECT_EQ(amd.private_l2_bytes, 1024u * 1024);
    EXPECT_EQ(amd.shared_l3_bytes, 32u * 1024 * 1024);
}

/** @test A cache shared by several physical cores cannot be labelled private L2. */
TEST(CPUCacheObservation, SharedL2IsNotPrivateWorkerCapacity)
{
    CacheRecords fixture;
    fixture.caches[1] = cacheRecord(2, 3, 4, 64, 8, 2048);
    const auto observation = cpu::observeX86CacheGeometry(fixture);
    EXPECT_EQ(observation.private_l1d_bytes, 32u * 1024);
    EXPECT_EQ(observation.private_l2_bytes, 0u);
    EXPECT_EQ(observation.private_l2_ways, 0u);
}

/** @test Family15h compute-unit topology is not an SMT-core/private-L2 observation. */
TEST(CPUCacheObservation, AMDFamily15ComputeUnitCannotBecomePrivateCore)
{
    CacheRecords fixture;
    fixture.extended = true;
    fixture.family = 0x15;
    const auto observation = cpu::observeX86CacheGeometry(fixture);
    EXPECT_EQ(observation.private_l1d_bytes, 0u);
    EXPECT_EQ(observation.private_l2_bytes, 0u);
    EXPECT_EQ(observation.private_l2_ways, 0u);
    EXPECT_EQ(observation.cache_line_bytes, 0u);
    EXPECT_EQ(observation.shared_l3_bytes, 8u * 1024 * 1024);
}

/** @test Null/instruction records and2^64-byte products cannot certify data-cache capacity. */
TEST(CPUCacheObservation, MalformedProductsAndNonDataRecordsReject)
{
    EXPECT_EQ(cpu::decodeDeterministicCache({}).bytes, 0u);
    EXPECT_EQ(cpu::decodeDeterministicCache(cacheRecord(1, 2, 2, 64, 8, 64)).bytes, 0u);
    const std::array<std::uint32_t, 4> overflowing{3u | (2u << 5), 0xffffffffu, 0xffffffffu, 0};
    EXPECT_EQ(cpu::decodeDeterministicCache(overflowing).bytes, 0u);
}

/** @test The live cached record is immutable and never invents ARM cache observations. */
TEST(CPUCacheObservation, NativeObservationHasOneImmutableAuthority)
{
    const auto &observation = cpu_cache_observation();
    EXPECT_EQ(&observation, &cpu_cache_observation());
#if defined(__aarch64__)
    EXPECT_EQ(observation.private_l1d_bytes, 0u);
    EXPECT_EQ(observation.private_l2_bytes, 0u);
    EXPECT_EQ(observation.shared_l3_bytes, 0u);
    EXPECT_EQ(observation.cache_line_bytes, 0u);
#else
    // A second live sample could run on another P/E core. Check the retained
    // record, not an unpinned observation with a potentially different owner.
    const auto snapshot = observation;
    const auto &retained = cpu_cache_observation();
    EXPECT_EQ(retained.private_l1d_bytes, snapshot.private_l1d_bytes);
    EXPECT_EQ(retained.private_l2_bytes, snapshot.private_l2_bytes);
    EXPECT_EQ(retained.shared_l3_bytes, snapshot.shared_l3_bytes);
    EXPECT_EQ(retained.cache_line_bytes, snapshot.cache_line_bytes);
#endif
}
} // namespace llaminar2::test
