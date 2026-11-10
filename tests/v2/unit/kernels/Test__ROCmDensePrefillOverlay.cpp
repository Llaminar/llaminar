/**
 * @file Test__ROCmDensePrefillOverlay.cpp
 * @brief Device-free ownership, totality and isolation of dense ROCm exact dispatch.
 *
 * These tests inspect immutable policy and construction-thread state only. No
 * HIP dependency or accelerator is linked. Captured execution and all source
 * formats are proved separately by the explicit backend integration gate.
 */
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlay.h"
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlayGenerated.inc"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <iterator>
#include <tuple>
#include <future>
#include <set>
#include <stdexcept>
using namespace llaminar2::rocm;
namespace
{
/** @brief Restore the caller's construction policy after each adversarial test. */
class ROCmDensePrefillOverlay : public ::testing::Test
{
protected:
    /** @brief Enter ordinary installed dispatch without preserving a test's switch. */
    void SetUp() override
    {
        previous_ = densePrefillExactOverlayEnabled();
        setDensePrefillExactOverlayEnabled(true);
    }
    /** @brief Restore the exact preceding thread state, including failed assertions. */
    void TearDown() override { setDensePrefillExactOverlayEnabled(previous_); }
private:
    bool previous_ = true;
};
/** @brief Every ordinary row count remains dispatchable through exact or generic ownership. */
TEST_F(ROCmDensePrefillOverlay, ExactRowsDoNotConsumeUnmeasuredGeometry)
{
    // The generated inventory owns which sparse rows were certified. Sweep
    // every intervening M independently so nearest-bucket dispatch cannot
    // consume an unmeasured cell as that inventory grows.
    std::set<int> measured_rows;
    for (const auto &entry : generated::kROCmDensePrefillOverlayEntries)
        if (entry.codebook == 0 && entry.n == 512 && entry.k == 5120)
            measured_rows.insert(entry.m);
    ASSERT_FALSE(measured_rows.empty());
    for (int m = 1; m <= 2048; ++m)
    {
        const auto selected = selectDensePrefillExactConfig(0, m, 512, 5120,
            DensePrefillWeightFamily::NativeQuantised);
        EXPECT_EQ(selected.has_value(), measured_rows.contains(m)) << m;
    }
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 513, 5120,
        DensePrefillWeightFamily::NativeQuantised));
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 512, 5152,
        DensePrefillWeightFamily::NativeQuantised));
    EXPECT_FALSE(selectDensePrefillExactConfig(23, 64, 512, 5120,
        DensePrefillWeightFamily::NativeQuantised));
}
/** @brief A prepared physical ABI cannot reinterpret a measured foreign producer. */
TEST_F(ROCmDensePrefillOverlay, RejectsInvalidGeometryAndWeightAuthority)
{
    const auto &entries = generated::kROCmDensePrefillOverlayEntries;
    for (const auto &entry : entries)
    {
        const llaminar2::native_quantised::NativeQuantisedGPUHardware actual{
            entry.architecture, 3, 60, 64, 1024, 65536};
        const auto *facts = entry.architecture.observed() ? &actual : nullptr;
        ASSERT_TRUE(selectDensePrefillExactConfig(entry.codebook, entry.m, entry.n,
            entry.k, entry.family, facts));
        EXPECT_TRUE(densePrefillProducerMatchesFamily(entry.family, entry.config.producer));
        const auto foreign = entry.family == DensePrefillWeightFamily::NativeQuantised
            ? DensePrefillWeightFamily::Int8Blockwise : DensePrefillWeightFamily::NativeQuantised;
        // A valid other-family request is a distinct key. Query its complete
        // immutable identity so future measured coexistence remains covered.
        const auto expected = std::find_if(std::begin(entries), std::end(entries),
            [&](const auto &candidate)
            {
                return candidate.codebook == entry.codebook && candidate.m == entry.m &&
                    candidate.n == entry.n && candidate.k == entry.k && candidate.family == foreign;
            });
        const llaminar2::native_quantised::NativeQuantisedGPUHardware foreign_actual{
            expected != std::end(entries) ? expected->architecture :
                llaminar2::native_quantised::NativeQuantisedArchitecture{}, 3, 60, 64, 1024, 65536};
        const auto selected = selectDensePrefillExactConfig(entry.codebook, entry.m,
            entry.n, entry.k, foreign, foreign_actual.architecture.observed() ? &foreign_actual : nullptr);
        ASSERT_EQ(selected.has_value(), expected != std::end(entries));
        if (selected)
        {
            const auto &config = expected->config;
            EXPECT_EQ(std::tuple(selected->n_tile, selected->m_tile, selected->min_blocks,
                                 selected->unroll, selected->full_tiles, selected->producer),
                      std::tuple(config.n_tile, config.m_tile, config.min_blocks,
                                 config.unroll, config.full_tiles, config.producer));
        }
    }
    for (int invalid : {0, -1})
    {
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, invalid, 512, 5120,
            DensePrefillWeightFamily::NativeQuantised), std::invalid_argument);
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, invalid, 5120,
            DensePrefillWeightFamily::NativeQuantised), std::invalid_argument);
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, 512, invalid,
            DensePrefillWeightFamily::NativeQuantised), std::invalid_argument);
    }
    EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, 512, 5120,
        static_cast<DensePrefillWeightFamily>(255)), std::invalid_argument);
}
/** @brief Native signed-Q8 cannot consume an expanded-INT8 exact configuration. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8AndExpandedInt8HaveDistinctKeys)
{
    EXPECT_FALSE(selectDensePrefillExactConfig(19, 64, 512, 5120,
        DensePrefillWeightFamily::NativeQuantised));
    const auto expanded = selectDensePrefillExactConfig(19, 64, 512, 5120,
        DensePrefillWeightFamily::Int8Blockwise);
    ASSERT_TRUE(expanded);
    EXPECT_EQ(expanded->producer, VNNIPrefillProducer::Int8BlockwiseV3);
    EXPECT_EQ(expanded->n_tile, 64);
    EXPECT_EQ(expanded->m_tile, 16);
    EXPECT_EQ(expanded->unroll, 4);

    // Exercise the actual generated ordering with two possible physical
    // contracts at identical decoder and geometry; neither is an alias.
    const generated::ROCmDensePrefillOverlayEntry native{
        19, 64, 512, 5120, DensePrefillWeightFamily::NativeQuantised,
        {64, 16, 1, 4, false, VNNIPrefillProducer::NativeCooperative}};
    const generated::ROCmDensePrefillOverlayEntry int8{
        19, 64, 512, 5120, DensePrefillWeightFamily::Int8Blockwise, *expanded};
    EXPECT_EQ(generated::compareROCmDensePrefillOverlayKey(native, 19, 64, 512, 5120,
        DensePrefillWeightFamily::NativeQuantised), 0);
    EXPECT_LT(generated::compareROCmDensePrefillOverlayKey(native, 19, 64, 512, 5120,
        DensePrefillWeightFamily::Int8Blockwise), 0);
    EXPECT_GT(generated::compareROCmDensePrefillOverlayKey(int8, 19, 64, 512, 5120,
        DensePrefillWeightFamily::NativeQuantised), 0);
    EXPECT_EQ(generated::compareROCmDensePrefillOverlayKey(int8, 19, 64, 512, 5120,
        DensePrefillWeightFamily::Int8Blockwise), 0);
}
/** @brief Author native-Q8 observations without installing measured policy rows. */
generated::ROCmDensePrefillOverlayEntry nativeQ8Row(
    llaminar2::native_quantised::NativeQuantisedArchitecture architecture, int tile)
{
    return {19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised,
        {64, tile, 1, 2, false, VNNIPrefillProducer::NativeCooperative}, architecture};
}
/** @brief Device-free complete participant facts; ordinal is deliberately nonzero. */
llaminar2::native_quantised::NativeQuantisedGPUHardware actualQ8Hardware(
    llaminar2::native_quantised::NativeQuantisedArchitecture architecture,
    int native_wave = 64, std::uint64_t lds = 65536)
{
    return {architecture, 3, 60, native_wave, 1024, lds};
}
/** @brief Actual architecture outranks nearby clocks, including native-wave32 devices. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8ActualArchitecturePrecedesNearestMeasured)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    const std::array entries{nativeQ8Row(Arch::rocm(9, 0, 6), 16),
                             nativeQ8Row(Arch::rocm(11, 0, 0), 32)};
    auto mi50 = actualQ8Hardware(Arch::rocm(9, 0, 6));
    auto xtx = actualQ8Hardware(Arch::rocm(11, 0, 0), 32);
    for (auto *actual : {&mi50, &xtx})
    {
        const auto before = *actual;
        const auto selected = selectDensePrefillExactConfigFromEntries(entries,
            19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, actual);
        ASSERT_TRUE(selected);
        EXPECT_EQ(selected->m_tile, actual == &mi50 ? 16 : 32);
        EXPECT_EQ(*actual, before); // tuning cannot overwrite actual device/resources
    }
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 447, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &mi50));
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 449, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &mi50));
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 321, 2560, DensePrefillWeightFamily::NativeQuantised, &mi50));
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 10240, DensePrefillWeightFamily::NativeQuantised, &mi50));
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::Int8Blockwise, &mi50));
}
/** @brief Nearest records must fit actual resources; empty domains retain generic dispatch. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8NearestCompatibleUsesActualResources)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    const std::array entries{nativeQ8Row(Arch::rocm(9, 0, 6), 16),
                             nativeQ8Row(Arch::rocm(11, 0, 0), 32)};
    auto future = actualQ8Hardware(Arch::rocm(11, 0, 1), 32);
    auto selected = selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &future);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->m_tile, 32);
    future.architecture = Arch::rocm(11, 0, 0); // exact measured row exists but cannot fit
    future.shared_memory_per_block = 11264; // source bound admits MT16 only
    selected = selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &future);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->m_tile, 16);
    future.maximum_threads_per_block = 128;
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &future));
    future.maximum_threads_per_block = 1024;
    future.shared_memory_per_block = 1024;
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &future));
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised));
    auto absent = actualQ8Hardware(Arch{});
    EXPECT_THROW((void)selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &absent), std::invalid_argument);
    const std::array<generated::ROCmDensePrefillOverlayEntry, 0> empty{};
    EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(empty,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &absent));
}
/** @brief Architecture ties are deterministic and legacy rows remain unobserved. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8ArchitectureAdmissionAndTieBreak)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    const std::array entries{nativeQ8Row(Arch::rocm(9, 0, 6), 16),
                             nativeQ8Row(Arch::rocm(9, 0, 8), 32)};
    const auto actual = actualQ8Hardware(Arch::rocm(9, 0, 7));
    const auto selected = selectDensePrefillExactConfigFromEntries(entries,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &actual);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->m_tile, 16);
    const std::array malformed{nativeQ8Row(Arch{}, 16)};
    EXPECT_THROW((void)selectDensePrefillExactConfigFromEntries(malformed,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &actual), std::logic_error);
    const std::array foreign{nativeQ8Row(Arch::cuda(8, 6), 16)};
    EXPECT_THROW((void)selectDensePrefillExactConfigFromEntries(foreign,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &actual), std::logic_error);
    for (const auto &entry : generated::kROCmDensePrefillOverlayEntries)
        if (entry.codebook != 19 || entry.family != DensePrefillWeightFamily::NativeQuantised)
            EXPECT_FALSE(entry.architecture.observed());
}

/** @brief Parse actual cached ISA spellings and isolate backend namespaces on ties. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8ArchitectureParsingAndBackendIsolation)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    EXPECT_EQ(Arch::fromROCmISATarget("gfx906"), Arch::rocm(9, 0, 6));
    EXPECT_EQ(Arch::fromROCmISATarget("gfx90a:sramecc+:xnack-"), Arch::rocm(9, 0, 10));
    EXPECT_EQ(Arch::fromROCmISATarget("gfx1100:xnack-"), Arch::rocm(11, 0, 0));
    for (const auto invalid : {"gfx", "gfx9", "gfx090a", "gfx90A", "sm86"})
        EXPECT_THROW((void)Arch::fromROCmISATarget(invalid), std::invalid_argument);

    // Exercise the reused selector itself: a CUDA version is never a close
    // ROCm observation. A same-backend equal-distance tie is independent of
    // generated iteration order and chooses the lower structured ISA tuple.
    struct ArchitectureRow { Arch architecture; int choice; };
    std::array rows{ArchitectureRow{Arch::cuda(8, 6), 99},
                    ArchitectureRow{Arch::rocm(9, 0, 8), 32},
                    ArchitectureRow{Arch::rocm(9, 0, 6), 16}};
    for (int rotation = 0; rotation < 3; ++rotation)
    {
        const auto *selected = llaminar2::native_quantised::selectNativeQuantisedArchitectureEntry(
            Arch::rocm(9, 0, 7), std::span<const ArchitectureRow>{rows},
            [](const auto &) { return true; });
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->architecture, Arch::rocm(9, 0, 6));
        EXPECT_EQ(selected->choice, 16);
        std::rotate(rows.begin(), rows.begin() + 1, rows.end());
    }
    const auto *cuda = llaminar2::native_quantised::selectNativeQuantisedArchitectureEntry(
        Arch::cuda(8, 6), std::span<const ArchitectureRow>{rows},
        [](const auto &) { return true; });
    ASSERT_NE(cuda, nullptr);
    EXPECT_EQ(cuda->choice, 99);
    const std::array native{nativeQ8Row(Arch::rocm(9, 0, 6), 16)};
    const auto foreign_hardware = actualQ8Hardware(Arch::cuda(8, 6), 32);
    EXPECT_THROW((void)selectDensePrefillExactConfigFromEntries(native,
        19, 448, 320, 2560, DensePrefillWeightFamily::NativeQuantised, &foreign_hardware), std::invalid_argument);
}

/** @brief Malformed matching producer/family pairs remain invalid, never generic misses. */
TEST_F(ROCmDensePrefillOverlay, ProducerAdmissionRequiresTypedPhysicalFamily)
{
    for (const auto producer : {VNNIPrefillProducer::NativeCooperative,
                               VNNIPrefillProducer::NativeStreaming})
    {
        EXPECT_TRUE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::NativeQuantised, producer));
        EXPECT_FALSE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::Int8Blockwise, producer));
    }
    for (const auto producer : {VNNIPrefillProducer::Int8BlockwiseV3,
                               VNNIPrefillProducer::Int8BlockwiseV7})
    {
        EXPECT_TRUE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::Int8Blockwise, producer));
        EXPECT_FALSE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::NativeQuantised, producer));
    }
    EXPECT_FALSE(densePrefillProducerMatchesFamily(static_cast<DensePrefillWeightFamily>(255),
        VNNIPrefillProducer::NativeCooperative));
    EXPECT_FALSE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::NativeQuantised,
        static_cast<VNNIPrefillProducer>(255)));
}
/** @brief A diagnostic tournament cannot disable another graph-construction thread. */
TEST_F(ROCmDensePrefillOverlay, TournamentBypassIsThreadLocal)
{
    setDensePrefillExactOverlayEnabled(false);
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 512, 5120,
        DensePrefillWeightFamily::NativeQuantised));
    auto other = std::async(std::launch::async, []
    {
        return densePrefillExactOverlayEnabled() &&
            selectDensePrefillExactConfig(0, 64, 512, 5120,
                DensePrefillWeightFamily::NativeQuantised).has_value();
    });
    EXPECT_TRUE(other.get());
    EXPECT_FALSE(densePrefillExactOverlayEnabled());
    setDensePrefillExactOverlayEnabled(true);
    EXPECT_TRUE(selectDensePrefillExactConfig(0, 64, 512, 5120,
        DensePrefillWeightFamily::NativeQuantised));
}
}
