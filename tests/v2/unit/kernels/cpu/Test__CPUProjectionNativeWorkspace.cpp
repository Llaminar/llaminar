/**
 * @file Test__CPUProjectionNativeWorkspace.cpp
 * @brief Device-free native CPU execution identity and canonical workspace BOM.
 *
 * Synthetic remote records are explicit typed observations, not borrowed local
 * CPU capabilities. ARM's register-resident K ABI contributes only its named
 * Q8 bank; x86 retains its real cache-derived shared-partial program. Unknown
 * or cross-architecture identities cannot borrow another implementation's BOM.
 */
#include "backends/CPUExecutionGeometry.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "kernels/cpu/gemm/CPUNativeVNNIEncoding.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include <array>

namespace llaminar2::test
{
/** @test A native observation never obtains identity by comparing unrelated ISA ordinals. */
TEST(CPUProjectionNativeWorkspace, CompilerRuntimePairsAreArchitectureExact)
{
    constexpr std::array levels{ISALevel::Scalar, ISALevel::AVX2, ISALevel::AVX512, ISALevel::NEON, ISALevel::NEONDotProd};
    for (const auto compiled : levels)
        for (const auto active : levels)
        {
            const bool expected =
                (compiled == ISALevel::AVX2 && (active == ISALevel::Scalar || active == ISALevel::AVX2)) ||
                (compiled == ISALevel::AVX512 && (active == ISALevel::Scalar || active == ISALevel::AVX2 || active == ISALevel::AVX512)) ||
                (compiled == ISALevel::NEON && active == ISALevel::NEON) ||
                (compiled == ISALevel::NEONDotProd && (active == ISALevel::NEON || active == ISALevel::NEONDotProd));
            EXPECT_EQ(isNativeCPUExecutionISAPair(compiled, active), expected);
        }
}

/** @test Every source uses the same actual native Q8-only workspace authority. */
TEST(CPUProjectionNativeWorkspace, NativeRegisterTreesNeverReserveX86SharedPartials)
{
    constexpr CPUExecutionGeometry native{.maximum_native_row_tile = 4,
        .projection_abi = CPUProjectionScheduleABI::ARMRegisterK,
        .compiled_isa = ISALevel::NEONDotProd, .active_isa = ISALevel::NEON};
    ASSERT_TRUE(native.isValid());
    ASSERT_FALSE(native.cache.isValid()); // No invented cache dimensions are needed/consumed.
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
        {
            CPUProjectionWorkspaceGeometry geometry{.rows = 31, .n = 65, .k = 1056, .workers = 10,
                .execution = native, .numerical_policy = policy};
            const auto footprint = cpu::native_vnni::preparedFootprintForFormat(*source.metadata);
            const auto requirements = CPUProjectionWorkspaceContract::prepared(footprint, source.metadata->codebook_id, geometry);
            ASSERT_EQ(requirements.buffers.size(), 1u);
            EXPECT_EQ(requirements.buffers[0].name, kCPUProjectionQ8);
            EXPECT_EQ(requirements.buffers[0].size_bytes, 31u * 33 * sizeof(Q8_1Block));
            auto unrelated = geometry;
            unrelated.execution.cache = kSyntheticCPUExecutionGeometry.cache;
            const auto same = CPUProjectionWorkspaceContract::prepared(footprint, source.metadata->codebook_id, unrelated);
            ASSERT_EQ(same.buffers.size(), 1u);
            EXPECT_EQ(same.buffers[0].size_bytes, requirements.buffers[0].size_bytes);
        }
}

/** @test Unknown/foreign/native-width metadata is rejected before memory admission. */
TEST(CPUProjectionNativeWorkspace, InvalidExecutionIdentityCannotPriceAnotherBackend)
{
    CPUProjectionWorkspaceGeometry geometry{.rows = 3, .n = 65, .k = 256, .workers = 3,
        .execution = kSyntheticCPUExecutionGeometry};
    ASSERT_TRUE(geometry.execution.isValid());
    EXPECT_FALSE(CPUProjectionWorkspaceContract::sourceNative("Q6_K", geometry).buffers.empty());
    auto unknown = geometry;
    unknown.execution.projection_abi = CPUProjectionScheduleABI::Unknown;
    EXPECT_THROW((void)CPUProjectionWorkspaceContract::sourceNative("Q6_K", unknown), std::invalid_argument);
    auto foreign = geometry;
    foreign.execution.compiled_isa = ISALevel::NEON;
    EXPECT_THROW((void)CPUProjectionWorkspaceContract::sourceNative("Q6_K", foreign), std::invalid_argument);
    auto native = geometry;
    native.execution = {.maximum_native_row_tile = 2, .projection_abi = CPUProjectionScheduleABI::ARMRegisterK,
        .compiled_isa = ISALevel::NEON, .active_isa = ISALevel::NEON};
    EXPECT_THROW((void)CPUProjectionWorkspaceContract::sourceNative("Q6_K", native), std::invalid_argument);
}

/** @test Local observation authenticates the actual compiler/runtime identity, without cache fabrication. */
TEST(CPUProjectionNativeWorkspace, LocalRecordRetainsRealCompiledAndActiveISA)
{
    const auto &local = CPUExecutionGeometry::local();
    EXPECT_EQ(local.compiled_isa, compiledCPUISALevel());
    EXPECT_EQ(local.active_isa, activeISALevel());
#if defined(__aarch64__) && defined(__ARM_NEON)
    EXPECT_EQ(local.projection_abi, CPUProjectionScheduleABI::ARMRegisterK);
    EXPECT_EQ(local.maximum_native_row_tile, 4u);
    EXPECT_TRUE(local.isValid());
    EXPECT_EQ(local.cache.private_l2_bytes, 0u);
    EXPECT_EQ(local.cache.shared_l3_bytes, 0u);
#endif
}
} // namespace llaminar2::test
