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
            DensePrefillWeightFamily::NativeLowBit);
        EXPECT_EQ(selected.has_value(), measured_rows.contains(m)) << m;
    }
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 513, 5120,
        DensePrefillWeightFamily::NativeLowBit));
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 512, 5152,
        DensePrefillWeightFamily::NativeLowBit));
    EXPECT_FALSE(selectDensePrefillExactConfig(23, 64, 512, 5120,
        DensePrefillWeightFamily::NativeLowBit));
}
/** @brief A prepared physical ABI cannot reinterpret a measured foreign producer. */
TEST_F(ROCmDensePrefillOverlay, RejectsInvalidGeometryAndWeightAuthority)
{
    for (const auto &entry : generated::kROCmDensePrefillOverlayEntries)
    {
        const auto owner = entry.codebook == 19
            ? DensePrefillWeightFamily::Int8 : DensePrefillWeightFamily::NativeLowBit;
        const auto foreign = entry.codebook == 19
            ? DensePrefillWeightFamily::NativeLowBit : DensePrefillWeightFamily::Int8;
        ASSERT_TRUE(selectDensePrefillExactConfig(entry.codebook, entry.m, entry.n,
            entry.k, owner));
        EXPECT_THROW((void)selectDensePrefillExactConfig(entry.codebook, entry.m, entry.n,
            entry.k, foreign), std::logic_error);
    }
    for (int invalid : {0, -1})
    {
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, invalid, 512, 5120,
            DensePrefillWeightFamily::NativeLowBit), std::invalid_argument);
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, invalid, 5120,
            DensePrefillWeightFamily::NativeLowBit), std::invalid_argument);
        EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, 512, invalid,
            DensePrefillWeightFamily::NativeLowBit), std::invalid_argument);
    }
    EXPECT_THROW((void)selectDensePrefillExactConfig(0, 64, 512, 5120,
        static_cast<DensePrefillWeightFamily>(255)), std::invalid_argument);
}
/** @brief A diagnostic tournament cannot disable another graph-construction thread. */
TEST_F(ROCmDensePrefillOverlay, TournamentBypassIsThreadLocal)
{
    setDensePrefillExactOverlayEnabled(false);
    EXPECT_FALSE(selectDensePrefillExactConfig(0, 64, 512, 5120,
        DensePrefillWeightFamily::NativeLowBit));
    auto other = std::async(std::launch::async, []
    {
        return densePrefillExactOverlayEnabled() &&
            selectDensePrefillExactConfig(0, 64, 512, 5120,
                DensePrefillWeightFamily::NativeLowBit).has_value();
    });
    EXPECT_TRUE(other.get());
    EXPECT_FALSE(densePrefillExactOverlayEnabled());
    setDensePrefillExactOverlayEnabled(true);
    EXPECT_TRUE(selectDensePrefillExactConfig(0, 64, 512, 5120,
        DensePrefillWeightFamily::NativeLowBit));
}
}
