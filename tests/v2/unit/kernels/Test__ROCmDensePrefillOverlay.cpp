/**
 * @file Test__ROCmDensePrefillOverlay.cpp
 * @brief Device-free ownership, totality and isolation of dense ROCm exact dispatch.
 *
 * These tests inspect immutable policy and construction-thread state only. No
 * HIP dependency or accelerator is linked. Captured execution and all source
 * formats are proved separately by the explicit backend integration gate.
 */
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlay.h"
#include <gtest/gtest.h>
#include <algorithm>
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
            DensePrefillWeightFamily::NativeVNNI);
        EXPECT_EQ(selected.has_value(), measured_rows.contains(m)) << m;
    }
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 513, 5120,
        DensePrefillWeightFamily::NativeVNNI));
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 512, 5152,
        DensePrefillWeightFamily::NativeVNNI));
    EXPECT_FALSE(selectDensePrefillExactConfig(23, 64, 512, 5120,
        DensePrefillWeightFamily::NativeVNNI));
}
/** @brief A prepared physical ABI cannot reinterpret a measured foreign producer. */
TEST_F(ROCmDensePrefillOverlay, RejectsInvalidGeometryAndWeightAuthority)
{
    const auto &entries = generated::kROCmDensePrefillOverlayEntries;
    for (const auto &entry : entries)
    {
        ASSERT_TRUE(selectDensePrefillExactConfig(entry.codebook, entry.m, entry.n,
            entry.k, entry.family));
        EXPECT_TRUE(densePrefillProducerMatchesFamily(entry.family, entry.config.producer));
        const auto foreign = entry.family == DensePrefillWeightFamily::NativeVNNI
            ? DensePrefillWeightFamily::Int8Blockwise : DensePrefillWeightFamily::NativeVNNI;
        // A valid other-family request is a distinct key. Query its complete
        // immutable identity so future measured coexistence remains covered.
        const auto expected = std::find_if(std::begin(entries), std::end(entries),
            [&](const auto &candidate)
            {
                return candidate.codebook == entry.codebook && candidate.m == entry.m &&
                    candidate.n == entry.n && candidate.k == entry.k && candidate.family == foreign;
            });
        const auto selected = selectDensePrefillExactConfig(entry.codebook, entry.m,
            entry.n, entry.k, foreign);
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
            DensePrefillWeightFamily::NativeVNNI), std::invalid_argument);
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, invalid, 5120,
            DensePrefillWeightFamily::NativeVNNI), std::invalid_argument);
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, 512, invalid,
            DensePrefillWeightFamily::NativeVNNI), std::invalid_argument);
    }
    EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, 512, 5120,
        static_cast<DensePrefillWeightFamily>(255)), std::invalid_argument);
}
/** @brief Native signed-Q8 cannot consume an expanded-INT8 exact configuration. */
TEST_F(ROCmDensePrefillOverlay, NativeQ8AndExpandedInt8HaveDistinctKeys)
{
    EXPECT_FALSE(selectDensePrefillExactConfig(19, 64, 512, 5120,
        DensePrefillWeightFamily::NativeVNNI));
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
        19, 64, 512, 5120, DensePrefillWeightFamily::NativeVNNI,
        {64, 16, 1, 4, false, VNNIPrefillProducer::NativeCooperative}};
    const generated::ROCmDensePrefillOverlayEntry int8{
        19, 64, 512, 5120, DensePrefillWeightFamily::Int8Blockwise, *expanded};
    EXPECT_EQ(generated::compareROCmDensePrefillOverlayKey(native, 19, 64, 512, 5120,
        DensePrefillWeightFamily::NativeVNNI), 0);
    EXPECT_LT(generated::compareROCmDensePrefillOverlayKey(native, 19, 64, 512, 5120,
        DensePrefillWeightFamily::Int8Blockwise), 0);
    EXPECT_GT(generated::compareROCmDensePrefillOverlayKey(int8, 19, 64, 512, 5120,
        DensePrefillWeightFamily::NativeVNNI), 0);
    EXPECT_EQ(generated::compareROCmDensePrefillOverlayKey(int8, 19, 64, 512, 5120,
        DensePrefillWeightFamily::Int8Blockwise), 0);
}
/** @brief Malformed matching producer/family pairs remain invalid, never generic misses. */
TEST_F(ROCmDensePrefillOverlay, ProducerAdmissionRequiresTypedPhysicalFamily)
{
    for (const auto producer : {VNNIPrefillProducer::NativeCooperative,
                               VNNIPrefillProducer::NativeStreaming})
    {
        EXPECT_TRUE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::NativeVNNI, producer));
        EXPECT_FALSE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::Int8Blockwise, producer));
    }
    for (const auto producer : {VNNIPrefillProducer::Int8BlockwiseV3,
                               VNNIPrefillProducer::Int8BlockwiseV7})
    {
        EXPECT_TRUE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::Int8Blockwise, producer));
        EXPECT_FALSE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::NativeVNNI, producer));
    }
    EXPECT_FALSE(densePrefillProducerMatchesFamily(static_cast<DensePrefillWeightFamily>(255),
        VNNIPrefillProducer::NativeCooperative));
    EXPECT_FALSE(densePrefillProducerMatchesFamily(DensePrefillWeightFamily::NativeVNNI,
        static_cast<VNNIPrefillProducer>(255)));
}
/** @brief A diagnostic tournament cannot disable another graph-construction thread. */
TEST_F(ROCmDensePrefillOverlay, TournamentBypassIsThreadLocal)
{
    setDensePrefillExactOverlayEnabled(false);
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 512, 5120,
        DensePrefillWeightFamily::NativeVNNI));
    auto other = std::async(std::launch::async, []
    {
        return densePrefillExactOverlayEnabled() &&
            selectDensePrefillExactConfig(0, 64, 512, 5120,
                DensePrefillWeightFamily::NativeVNNI).has_value();
    });
    EXPECT_TRUE(other.get());
    EXPECT_FALSE(densePrefillExactOverlayEnabled());
    setDensePrefillExactOverlayEnabled(true);
    EXPECT_TRUE(selectDensePrefillExactConfig(0, 64, 512, 5120,
        DensePrefillWeightFamily::NativeVNNI));
}
}
