/**
 * @file Test__ROCmDensePrefillOverlay.cpp
 * @brief Device-free ownership, totality and isolation of dense ROCm exact dispatch.
 *
 * These tests inspect immutable policy and construction-thread state only. No
 * HIP dependency or accelerator is linked. Readers borrow the production
 * table instead of defining another generated object. Captured execution and all source
 * formats are proved separately by the explicit backend integration gate.
 * Installed M512 HC-down observations retain architecture-specific physical
 * tiles; authored-only neighbor rows keep later measured additions admissible.
 */
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlay.h"
#include "kernels/rocm/gemm/ROCmDensePrefillPolicyEvidence.h"
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
/** @brief A WMMA exact row never borrows instruction support from its tuning ISA. */
TEST_F(ROCmDensePrefillOverlay, WmmaRequiresActualGfx1100AndExactNativeStorage)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    for (const auto codebook : {0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 24})
    {
        const generated::ROCmDensePrefillOverlayEntry measured{
            static_cast<uint8_t>(codebook), 64, 128, 256, DensePrefillWeightFamily::NativeQuantised,
            {64, 64, 1, 0, false, VNNIPrefillProducer::NativeWmma}, Arch::rocm(11, 0, 0)};
        const std::array entries{measured};
        llaminar2::native_quantised::NativeQuantisedGPUHardware actual{
            Arch::rocm(11, 0, 0), 0, 96, 32, 1024, 65536};
        const auto select = [&] { return selectDensePrefillExactConfigFromEntries(entries,
            codebook, 64, 128, 256, DensePrefillWeightFamily::NativeQuantised, &actual); };
        ASSERT_TRUE(select());
        EXPECT_EQ(select()->producer, VNNIPrefillProducer::NativeWmma);
        actual.maximum_threads_per_block = 256;
        EXPECT_TRUE(select());
        actual.maximum_threads_per_block = 255;
        EXPECT_FALSE(select());
        actual.maximum_threads_per_block = 1024;
        actual.architecture = Arch::rocm(9, 0, 6);
        EXPECT_FALSE(select());
        actual.architecture = Arch::rocm(11, 0, 1);
        EXPECT_FALSE(select());
        actual.architecture = Arch::rocm(11, 0, 0);
        actual.shared_memory_per_block = 1024;
        EXPECT_FALSE(select());
        EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries,
            codebook, 64, 128, 256, DensePrefillWeightFamily::NativeQuantised));
        auto invalid = measured;
        invalid.architecture = Arch{};
        const std::array bad{invalid};
        EXPECT_THROW((void)selectDensePrefillExactConfigFromEntries(bad,
            codebook, 64, 128, 256, DensePrefillWeightFamily::NativeQuantised, &actual), std::logic_error);
    }
}
/** @brief Retained legacy rows coexist in generator order with qualified target rows. */
TEST_F(ROCmDensePrefillOverlay, LegacyAndQualifiedRowsShareOneOrderedGeometry)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    constexpr auto family = DensePrefillWeightFamily::NativeQuantised;
    constexpr generated::ROCmDensePrefillOverlayEntry legacy{
        0, 64, 128, 256, family, {64, 16, 1, 2, false, VNNIPrefillProducer::NativeCooperative}};
    constexpr generated::ROCmDensePrefillOverlayEntry gfx906{
        0, 64, 128, 256, family, {64, 32, 1, 2, false, VNNIPrefillProducer::NativeCooperative},
        Arch::rocm(9, 0, 6)};
    constexpr generated::ROCmDensePrefillOverlayEntry gfx1100{
        0, 64, 128, 256, family, {64, 64, 1, 0, false, VNNIPrefillProducer::NativeWmma},
        Arch::rocm(11, 0, 0)};
    // This is the same key authority used by completeInventory's static proof.
    EXPECT_LT(legacy.orderingKey(), gfx906.orderingKey());
    EXPECT_LT(gfx906.orderingKey(), gfx1100.orderingKey());
    const std::array entries{legacy, gfx906, gfx1100};
    const auto unobserved = selectDensePrefillExactConfigFromEntries(entries, 0, 64, 128, 256, family);
    ASSERT_TRUE(unobserved);
    EXPECT_EQ(unobserved->m_tile, 16);
    for (const auto [architecture, expected_m] : {
        std::pair{Arch::rocm(9, 0, 6), 32}, std::pair{Arch::rocm(11, 0, 0), 64},
        std::pair{Arch::rocm(11, 0, 1), 32}})
    {
        llaminar2::native_quantised::NativeQuantisedGPUHardware actual{
            architecture, 0, 96, 64, 1024, 65536};
        const auto selected = selectDensePrefillExactConfigFromEntries(entries, 0, 64, 128, 256, family, &actual);
        ASSERT_TRUE(selected);
        EXPECT_EQ(selected->m_tile, expected_m);
        EXPECT_EQ(actual.architecture, architecture);
        EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(entries, 0, 65, 128, 256, family, &actual));
    }
    const std::array retained{legacy, gfx1100};
    llaminar2::native_quantised::NativeQuantisedGPUHardware older{
        Arch::rocm(9, 0, 6), 0, 60, 64, 1024, 65536};
    const auto selected = selectDensePrefillExactConfigFromEntries(retained, 0, 64, 128, 256, family, &older);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->m_tile, 16);
}

/** @brief Streaming admission uses physical decoder and K-dependent correction storage. */
TEST_F(ROCmDensePrefillOverlay, NativeStreamingPhysical24AndLiveCorrectionCapacity)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    for (const auto [codebook, m_tile, k, lds] : {
        std::tuple{24, 16, 65536, 0}, std::tuple{5, 16, 8192, 16384},
        std::tuple{16, 16, 65536, 131072}, std::tuple{11, 16, 65536, 0},
        std::tuple{17, 8, 8192, 32768}})
    {
        const generated::ROCmDensePrefillOverlayEntry measured{
            static_cast<uint8_t>(codebook), 64, 256, k, DensePrefillWeightFamily::NativeQuantised,
            {256, m_tile, 3, 0, false, VNNIPrefillProducer::NativeStreaming}, Arch::rocm(11, 0, 0)};
        const std::array entries{measured};
        llaminar2::native_quantised::NativeQuantisedGPUHardware actual{
            Arch::rocm(11, 0, 0), 0, 96, 32, 1024, 65536};
        const auto select = [&] { return selectDensePrefillExactConfigFromEntries(entries,
            codebook, 64, 256, k, DensePrefillWeightFamily::NativeQuantised, &actual); };
        EXPECT_EQ(select().has_value(), lds <= 65536);
        if (lds > 0 && lds <= 65536)
        {
            actual.shared_memory_per_block = lds;
            EXPECT_TRUE(select());
            --actual.shared_memory_per_block;
            EXPECT_FALSE(select());
        }
    }
}
/** @brief Every ordinary row count remains dispatchable through exact or generic ownership. */
TEST_F(ROCmDensePrefillOverlay, ExactRowsDoNotConsumeUnmeasuredGeometry)
{
    // The generated inventory owns which sparse rows were certified. Sweep
    // every intervening M independently so nearest-bucket dispatch cannot
    // consume an unmeasured cell as that inventory grows.
    std::set<int> measured_rows;
    for (const auto &entry : densePrefillInstalledEntries())
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
    const auto entries = densePrefillInstalledEntries();
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

    // Exercise the actual production ordering with two possible physical
    // contracts at identical decoder and geometry; neither is an alias.
    const generated::ROCmDensePrefillOverlayEntry native{
        19, 64, 512, 5120, DensePrefillWeightFamily::NativeQuantised,
        {64, 16, 1, 4, false, VNNIPrefillProducer::NativeCooperative},
        llaminar2::native_quantised::NativeQuantisedArchitecture::rocm(9,0,6)};
    const generated::ROCmDensePrefillOverlayEntry int8{
        19, 64, 512, 5120, DensePrefillWeightFamily::Int8Blockwise, *expanded};
    EXPECT_LT(native.orderingKey(),int8.orderingKey());
    const std::array entries{native,int8};
    const llaminar2::native_quantised::NativeQuantisedGPUHardware hardware{
        native.architecture,0,60,64,1024,65536};
    for (const auto &entry : entries)
    {
        const auto selected=selectDensePrefillExactConfigFromEntries(entries,19,64,512,5120,entry.family,&hardware);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(selected->producer,entry.config.producer);
    }
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
/** @brief Measured tuples remain installed; missing-row isolation uses an authored span. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8MeasuredM448SelectionAndActualRowIsolation)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    auto mi50 = actualQ8Hardware(Arch::rocm(9, 0, 6));
    auto xtx = actualQ8Hardware(Arch::rocm(11, 0, 0), 32);
    for (int k : {2560, 10240})
    {
        // This deliberately M448-only span proves actual-row isolation without
        // forbidding independent future installed measurements at other rows.
        auto measured_mi50 = nativeQ8Row(Arch::rocm(9, 0, 6), 16);
        auto measured_xtx = nativeQ8Row(Arch::rocm(11, 0, 0), 32);
        measured_mi50.k = measured_xtx.k = k;
        const std::array measured{measured_mi50, measured_xtx};
        for (const auto *actual : {&mi50, &xtx})
        {
            const auto selected = selectDensePrefillExactConfig(19, 448, 320, k,
                DensePrefillWeightFamily::NativeQuantised, actual);
            ASSERT_TRUE(selected);
            EXPECT_EQ(std::tuple(selected->n_tile, selected->m_tile, selected->min_blocks,
                                 selected->unroll, selected->full_tiles, selected->producer),
                      std::tuple(64, actual == &mi50 ? 16 : 32, 1, 2, false,
                                 VNNIPrefillProducer::NativeCooperative));
            const auto authored = selectDensePrefillExactConfigFromEntries(measured,
                19, 448, 320, k, DensePrefillWeightFamily::NativeQuantised, actual);
            ASSERT_TRUE(authored);
            EXPECT_EQ(authored->m_tile, selected->m_tile);
            // Architecture proximity must never become nearest-M selection.
            for (int m = 1; m < 128; ++m)
                EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(measured,
                    19, m, 320, k, DensePrefillWeightFamily::NativeQuantised, actual)) << m;
            for (int m : {128, 256, 447, 449, 512})
                EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(measured,
                    19, m, 320, k, DensePrefillWeightFamily::NativeQuantised, actual)) << m;
            EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(measured,
                19, 448, 320, k, DensePrefillWeightFamily::NativeQuantised));
        }
    }
}
/** @brief Installed HC-down M512 tiles preserve actual card facts and exact row identity. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8MeasuredM512SelectionAndActualRowIsolation)
{
    using Arch = llaminar2::native_quantised::NativeQuantisedArchitecture;
    using Entry = generated::ROCmDensePrefillOverlayEntry;
    const std::array measured{
        Entry{19, 512, 320, 10240, DensePrefillWeightFamily::NativeQuantised,
            {64, 16, 1, 2, false, VNNIPrefillProducer::NativeCooperative}, Arch::rocm(9, 0, 6)},
        Entry{19, 512, 320, 10240, DensePrefillWeightFamily::NativeQuantised,
            {64, 32, 1, 2, false, VNNIPrefillProducer::NativeCooperative}, Arch::rocm(11, 0, 0)}};
    const std::array hardware{actualQ8Hardware(Arch::rocm(9, 0, 6)),
                              actualQ8Hardware(Arch::rocm(11, 0, 0), 32)};
    for (std::size_t index = 0; index < hardware.size(); ++index)
    {
        auto actual = hardware[index];
        const auto before = actual;
        const auto &expected = measured[index].config;
        for (const auto &selected : {
                 selectDensePrefillExactConfig(19, 512, 320, 10240,
                     DensePrefillWeightFamily::NativeQuantised, &actual),
                 selectDensePrefillExactConfigFromEntries(measured, 19, 512, 320, 10240,
                     DensePrefillWeightFamily::NativeQuantised, &actual)})
        {
            ASSERT_TRUE(selected);
            EXPECT_EQ(std::tuple(selected->n_tile, selected->m_tile, selected->min_blocks,
                                 selected->unroll, selected->full_tiles, selected->producer),
                      std::tuple(expected.n_tile, expected.m_tile, expected.min_blocks,
                                 expected.unroll, expected.full_tiles, expected.producer));
        }
        // Only this authored M512 span owns absence. Independent future
        // installed M511/M513 observations must not invalidate this regression.
        for (int rows : {511, 513})
            EXPECT_FALSE(selectDensePrefillExactConfigFromEntries(measured,
                19, rows, 320, 10240, DensePrefillWeightFamily::NativeQuantised, &actual)) << rows;
        EXPECT_EQ(actual, before);
    }
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
/** @brief Architecture ties are deterministic and malformed observations are rejected. */
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
