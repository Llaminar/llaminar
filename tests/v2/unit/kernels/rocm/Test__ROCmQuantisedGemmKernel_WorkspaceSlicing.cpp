/**
 * @file Test__ROCmQuantisedGemmKernel_WorkspaceSlicing.cpp
 * @brief Regression tests for ROCmQuantisedGemmKernel TEMP_C_FP32 /
 *        TEMP_A_FP32 per-instance workspace slicing.
 *
 * Background — the bug these tests guard against:
 *
 *   KernelFactory caches one ROCmQuantisedGemmKernel per (device, weight) pair.
 *   The kernel binds its scratch pointers impl_->d_C_fp32 / impl_->d_A_fp32
 *   from the shared workspace buffers named GemmWorkspaceBuffers::TEMP_C_FP32
 *   and TEMP_A_FP32. The kernel then queues ASYNC D2D/D2H copies out of
 *   impl_->d_C_fp32 on its per-invocation ConcurrentPrefillPool stream.
 *
 *   WorkspaceRequirements::merge() collapses buffers that share a name and
 *   keeps a SINGLE buffer sized to the largest request. That means EVERY
 *   cached GEMM kernel on a ROCm device ends up pointing at the SAME
 *   TEMP_C_FP32 bytes. A second kernel can then clobber the redirect source
 *   of the first kernel before the first kernel's async copy has drained.
 *
 *   The fix gives each kernel instance a unique slice id and exposes the
 *   buffers under unique names ("gemm_temp_c_fp32_<id>",
 *   "gemm_temp_a_fp32_<id>"), so merge() keeps a separate buffer per kernel.
 *
 * These tests verify the structural invariant (unique names per instance)
 * without needing ROCm hardware. They run on any host because the kernel's
 * constructor and getWorkspaceRequirements() are pure host computation —
 * they don't touch the GPU.
 *
 * Parallels tests/v2/unit/kernels/cuda/Test__CUDAQuantisedGemmKernel_Workspace.cpp.
 *
 * @date April 2026
 */

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using llaminar2::rocm::ROCmQuantisedGemmKernel;
using llaminar2::test::TestTensorFactory;

namespace
{
    const std::string kTempCFp32Prefix = std::string(GemmWorkspaceBuffers::TEMP_C_FP32);
    const std::string kTempAFp32Prefix = std::string(GemmWorkspaceBuffers::TEMP_A_FP32);

    size_t countPrefixed(const WorkspaceRequirements &reqs, const std::string &prefix)
    {
        size_t n = 0;
        for (const auto &b : reqs.buffers)
        {
            if (b.name.rfind(prefix, 0) == 0)
            {
                ++n;
            }
        }
        return n;
    }

    std::set<std::string> collectPrefixed(const WorkspaceRequirements &reqs, const std::string &prefix)
    {
        std::set<std::string> names;
        for (const auto &b : reqs.buffers)
        {
            if (b.name.rfind(prefix, 0) == 0)
            {
                names.insert(b.name);
            }
        }
        return names;
    }
} // namespace

class Test__ROCmQuantisedGemmKernel_WorkspaceSlicing : public ::testing::Test
{
protected:
    // Use a fake device id; the kernel constructor stores it but does not
    // touch the GPU during construction or getWorkspaceRequirements().
    static constexpr int kFakeRocmDeviceId = 0;
};

// ----------------------------------------------------------------------------
// REGRESSION: each kernel instance must declare a UNIQUE TEMP_C_FP32 slice.
// ----------------------------------------------------------------------------
TEST_F(Test__ROCmQuantisedGemmKernel_WorkspaceSlicing,
       TwoInstances_DeclareDistinctTempCFp32BufferNames)
{
    auto weights_a = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/1);
    auto weights_b = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/2);

    ROCmQuantisedGemmKernel kernel_a(weights_a.get(), kFakeRocmDeviceId);
    ROCmQuantisedGemmKernel kernel_b(weights_b.get(), kFakeRocmDeviceId);

    auto reqs_a = kernel_a.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto reqs_b = kernel_b.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);

    auto names_a = collectPrefixed(reqs_a, kTempCFp32Prefix);
    auto names_b = collectPrefixed(reqs_b, kTempCFp32Prefix);

    ASSERT_EQ(names_a.size(), 1u);
    ASSERT_EQ(names_b.size(), 1u);

    EXPECT_NE(*names_a.begin(), *names_b.begin())
        << "Two ROCmQuantisedGemmKernel instances declared the SAME TEMP_C_FP32 buffer name "
        << "(\"" << *names_a.begin() << "\"). WorkspaceRequirements::merge() will collapse "
        << "them into a single shared slot, which lets one kernel clobber the redirect source "
        << "of another while an async D2D/D2H copy is still in flight.";
}

// ----------------------------------------------------------------------------
// REGRESSION: each kernel instance must declare a UNIQUE TEMP_A_FP32 slice.
// ----------------------------------------------------------------------------
TEST_F(Test__ROCmQuantisedGemmKernel_WorkspaceSlicing,
       TwoInstances_DeclareDistinctTempAFp32BufferNames)
{
    auto weights_a = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/1);
    auto weights_b = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/2);

    ROCmQuantisedGemmKernel kernel_a(weights_a.get(), kFakeRocmDeviceId);
    ROCmQuantisedGemmKernel kernel_b(weights_b.get(), kFakeRocmDeviceId);

    auto reqs_a = kernel_a.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto reqs_b = kernel_b.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);

    auto names_a = collectPrefixed(reqs_a, kTempAFp32Prefix);
    auto names_b = collectPrefixed(reqs_b, kTempAFp32Prefix);

    ASSERT_EQ(names_a.size(), 1u);
    ASSERT_EQ(names_b.size(), 1u);

    EXPECT_NE(*names_a.begin(), *names_b.begin())
        << "Two ROCmQuantisedGemmKernel instances declared the SAME TEMP_A_FP32 buffer name "
        << "(\"" << *names_a.begin() << "\"). Defensive per-instance slicing also applies to "
        << "TEMP_A_FP32 to guard against future cross-stream hazards on its contents.";
}

// ----------------------------------------------------------------------------
// REGRESSION: merge() must keep both slices distinct.
// ----------------------------------------------------------------------------
TEST_F(Test__ROCmQuantisedGemmKernel_WorkspaceSlicing,
       MergedRequirements_KeepBothTempCFp32Slices)
{
    auto weights_a = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/1);
    auto weights_b = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/2);

    ROCmQuantisedGemmKernel kernel_a(weights_a.get(), kFakeRocmDeviceId);
    ROCmQuantisedGemmKernel kernel_b(weights_b.get(), kFakeRocmDeviceId);

    auto reqs_a = kernel_a.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto reqs_b = kernel_b.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);

    ASSERT_EQ(countPrefixed(reqs_a, kTempCFp32Prefix), 1u);
    ASSERT_EQ(countPrefixed(reqs_b, kTempCFp32Prefix), 1u);
    ASSERT_EQ(countPrefixed(reqs_a, kTempAFp32Prefix), 1u);
    ASSERT_EQ(countPrefixed(reqs_b, kTempAFp32Prefix), 1u);

    reqs_a.merge(reqs_b);

    EXPECT_EQ(countPrefixed(reqs_a, kTempCFp32Prefix), 2u)
        << "After merging two ROCmQuantisedGemmKernels' workspace requirements, only one "
        << "TEMP_C_FP32 buffer survived. Both kernels alias the same redirect-source memory.";
    EXPECT_EQ(countPrefixed(reqs_a, kTempAFp32Prefix), 2u)
        << "After merging, only one TEMP_A_FP32 buffer survived. Both kernels would alias the "
        << "same activation scratch memory.";
}

// ----------------------------------------------------------------------------
// REGRESSION: many instances must remain independent.
// ----------------------------------------------------------------------------
TEST_F(Test__ROCmQuantisedGemmKernel_WorkspaceSlicing,
       ManyInstances_AllProduceDistinctTempCFp32Names)
{
    constexpr int kNumKernels = 32;

    std::vector<std::unique_ptr<Q8_0Tensor>> weights;
    weights.reserve(kNumKernels);
    std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
    kernels.reserve(kNumKernels);

    for (int i = 0; i < kNumKernels; ++i)
    {
        weights.push_back(TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/100 + i));
        kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(
            weights.back().get(), kFakeRocmDeviceId));
    }

    std::set<std::string> seen_c;
    std::set<std::string> seen_a;
    for (int i = 0; i < kNumKernels; ++i)
    {
        auto reqs = kernels[i]->getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
        auto c = collectPrefixed(reqs, kTempCFp32Prefix);
        auto a = collectPrefixed(reqs, kTempAFp32Prefix);
        ASSERT_EQ(c.size(), 1u);
        ASSERT_EQ(a.size(), 1u);
        EXPECT_TRUE(seen_c.insert(*c.begin()).second)
            << "Duplicate TEMP_C_FP32 name across kernel instances: \"" << *c.begin() << "\"";
        EXPECT_TRUE(seen_a.insert(*a.begin()).second)
            << "Duplicate TEMP_A_FP32 name across kernel instances: \"" << *a.begin() << "\"";
    }

    EXPECT_EQ(seen_c.size(), static_cast<size_t>(kNumKernels));
    EXPECT_EQ(seen_a.size(), static_cast<size_t>(kNumKernels));
}

// ----------------------------------------------------------------------------
// SANITY: unique names still use the standard prefixes.
// ----------------------------------------------------------------------------
TEST_F(Test__ROCmQuantisedGemmKernel_WorkspaceSlicing,
       SliceNames_KeepStandardPrefixes)
{
    auto weights = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/7);
    ROCmQuantisedGemmKernel kernel(weights.get(), kFakeRocmDeviceId);

    auto reqs = kernel.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto c = collectPrefixed(reqs, kTempCFp32Prefix);
    auto a = collectPrefixed(reqs, kTempAFp32Prefix);
    ASSERT_EQ(c.size(), 1u);
    ASSERT_EQ(a.size(), 1u);

    EXPECT_EQ(c.begin()->rfind(kTempCFp32Prefix, 0), 0u);
    EXPECT_EQ(a.begin()->rfind(kTempAFp32Prefix, 0), 0u);
}

// ----------------------------------------------------------------------------
// SANITY: slice sizes still match expected bytes.
// ----------------------------------------------------------------------------
TEST_F(Test__ROCmQuantisedGemmKernel_WorkspaceSlicing,
       SliceSizes_MatchExpectedBytes)
{
    auto weights = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/9);
    ROCmQuantisedGemmKernel kernel(weights.get(), kFakeRocmDeviceId);

    constexpr int kM = 8;
    constexpr int kN = 64;
    constexpr int kK = 128;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *temp_c = nullptr;
    const WorkspaceDescriptor *temp_a = nullptr;
    for (const auto &b : reqs.buffers)
    {
        if (b.name.rfind(kTempCFp32Prefix, 0) == 0)
            temp_c = &b;
        else if (b.name.rfind(kTempAFp32Prefix, 0) == 0)
            temp_a = &b;
    }
    ASSERT_NE(temp_c, nullptr);
    ASSERT_NE(temp_a, nullptr);
    EXPECT_EQ(temp_c->size_bytes, static_cast<size_t>(kM) * kN * sizeof(float));
    EXPECT_EQ(temp_a->size_bytes, static_cast<size_t>(kM) * kK * sizeof(float));
}
