/**
 * @file Test__ROCmDensePolicyOwnership.cpp
 * @brief Native-loader replay proving the production core owns its installed rows.
 *
 * An exported executable table deliberately conflicts with the library. A
 * control DSO must read the executable table, while the actual production
 * selector and read-only evidence view retain their private library authority.
 * This tests ELF behavior directly without GPU discovery, source scans or a
 * second handwritten implementation of the production lookup.
 */
#include "kernels/rocm/gemm/ROCmDensePrefillPolicyEvidence.h"
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlay.h"
#include <gtest/gtest.h>
#include <dlfcn.h>

namespace llaminar2::rocm::generated
{
inline constexpr ROCmDensePrefillOverlayEntry kROCmDensePrefillOverlayEntries[] = {
    {0,17,17,32,DensePrefillWeightFamily::NativeQuantised,
     {128,32,1,0,false,VNNIPrefillProducer::NativeCooperative}}};
}
/** @return Address resolved by the intentionally preemptible control DSO. */
extern "C" const void *rocmDensePolicyInterpositionControl() noexcept;

/** @test A conflicting executable cannot replace the real table or its selector decisions. */
TEST(ROCmDensePolicyOwnership, ExecutableInterpositionCannotReplaceProductionRows)
{
    using namespace llaminar2::rocm;
    const auto *foreign = generated::kROCmDensePrefillOverlayEntries;
    ASSERT_EQ(rocmDensePolicyInterpositionControl(),foreign) << "loader control did not reproduce the original defect";
    const auto entries = densePrefillInstalledEntries();
    ASSERT_GT(entries.size(),1u);
    EXPECT_NE(entries.data(),foreign);
    Dl_info table{},owner{},executable{};
    ASSERT_NE(dladdr(entries.data(),&table),0);
    ASSERT_NE(dladdr(reinterpret_cast<const void *>(&densePrefillInstalledEntries),&owner),0);
    ASSERT_NE(dladdr(foreign,&executable),0);
    EXPECT_EQ(table.dli_fbase,owner.dli_fbase);
    EXPECT_NE(table.dli_fbase,executable.dli_fbase);
    const auto &first=entries.front();
    ASSERT_FALSE(first.architecture.observed());
    const auto selected=selectDensePrefillExactConfig(first.codebook,first.m,first.n,first.k,first.family);
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->n_tile,first.config.n_tile);
    EXPECT_EQ(selected->m_tile,first.config.m_tile);
    EXPECT_EQ(selected->producer,first.config.producer);
    EXPECT_FALSE(selectDensePrefillExactConfig(foreign->codebook,foreign->m,foreign->n,foreign->k,foreign->family));
}
