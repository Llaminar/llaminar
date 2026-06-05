/**
 * @file Test__CUDAQuantisedGemmKernel_Workspace.cpp
 * @brief Regression tests for CUDAQuantisedGemmKernel TEMP_C_FP32 workspace slicing.
 *
 * Background — the bug these tests guard against:
 *
 *   KernelFactory caches one CUDAQuantisedGemmKernel per (device, weight) pair.
 *   For mapped FP32 outputs (e.g. logits), the kernel redirects writes to a
 *   shared workspace buffer named GemmWorkspaceBuffers::TEMP_C_FP32 and then
 *   queues an ASYNC D2D copy back to the mapped destination.
 *
 *   WorkspaceRequirements::merge() collapses buffers that share a name and
 *   keeps a SINGLE buffer sized to the largest request. That means EVERY
 *   cached GEMM kernel on a device ends up pointing at the SAME TEMP_C_FP32
 *   bytes. A second kernel can then clobber the redirect source of the first
 *   kernel before the first kernel's async D2D copy has drained, producing
 *   all-zero outputs (observed as parity test #146 flakes).
 *
 *   The fix gives each kernel instance a unique slice id and exposes the
 *   buffer under a unique name (e.g. "gemm_temp_c_fp32_<id>"), so merge()
 *   keeps a separate buffer per kernel.
 *
 * These tests verify the structural invariant (unique names per instance)
 * without needing CUDA hardware. They run on any host because the kernel's
 * constructor and getWorkspaceRequirements() are pure host computation —
 * they don't touch the GPU.
 *
 * @date January 2026
 */

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using llaminar2::cuda::CUDAQuantisedGemmKernel;
using llaminar2::test::TestTensorFactory;

namespace
{
    // Match the official prefix of the redirect buffer name. Using a string
    // (not a pointer) so we can do prefix comparisons regardless of whether
    // the kernel returns the bare prefix or a "<prefix>_<id>" suffixed slice.
    const std::string kTempCFp32Prefix = std::string(GemmWorkspaceBuffers::TEMP_C_FP32);
    constexpr int kConcurrentPrefillExtraAccumulatorSlots = 2;

    int paddedPrefillM(int m)
    {
        return (m > 1) ? ((m + 127) & ~127) : m;
    }

    // Helper: count buffers in a WorkspaceRequirements whose name starts with
    // the TEMP_C_FP32 prefix. Both the buggy (single shared slot) and fixed
    // (per-instance unique slots) implementations should be detectable.
    size_t countTempCFp32Buffers(const WorkspaceRequirements &reqs)
    {
        size_t n = 0;
        for (const auto &b : reqs.buffers)
        {
            if (b.name.rfind(kTempCFp32Prefix, 0) == 0)
            {
                ++n;
            }
        }
        return n;
    }

    // Helper: collect all unique TEMP_C_FP32-prefixed buffer names.
    std::set<std::string> collectTempCFp32Names(const WorkspaceRequirements &reqs)
    {
        std::set<std::string> names;
        for (const auto &b : reqs.buffers)
        {
            if (b.name.rfind(kTempCFp32Prefix, 0) == 0)
            {
                names.insert(b.name);
            }
        }
        return names;
    }
} // namespace

class Test__CUDAQuantisedGemmKernel_Workspace : public ::testing::Test
{
protected:
    // Use a fake device id; the kernel constructor stores it but does not
    // touch the GPU during construction or getWorkspaceRequirements().
    static constexpr int kFakeCudaDeviceId = 0;
};

// ----------------------------------------------------------------------------
// REGRESSION: each kernel instance must declare a UNIQUE TEMP_C_FP32 slice.
// ----------------------------------------------------------------------------
//
// On the buggy implementation both kernels declare a buffer literally named
// GemmWorkspaceBuffers::TEMP_C_FP32 and this test fails:
//   EXPECT_NE(name_a, name_b)  -> both are "gemm_temp_c_fp32"
//
// After the fix each kernel embeds its own slice id in the buffer name.
TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       TwoInstances_DeclareDistinctTempCFp32BufferNames)
{
    auto weights_a = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/1);
    auto weights_b = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/2);

    CUDAQuantisedGemmKernel kernel_a(weights_a.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_b(weights_b.get(), kFakeCudaDeviceId);

    auto reqs_a = kernel_a.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto reqs_b = kernel_b.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);

    auto names_a = collectTempCFp32Names(reqs_a);
    auto names_b = collectTempCFp32Names(reqs_b);

    ASSERT_EQ(names_a.size(), 1u) << "Kernel A must declare exactly one TEMP_C_FP32 slice";
    ASSERT_EQ(names_b.size(), 1u) << "Kernel B must declare exactly one TEMP_C_FP32 slice";

    const std::string &name_a = *names_a.begin();
    const std::string &name_b = *names_b.begin();

    EXPECT_NE(name_a, name_b)
        << "Two CUDAQuantisedGemmKernel instances declared the SAME TEMP_C_FP32 buffer name "
        << "(\"" << name_a << "\"). WorkspaceRequirements::merge() will collapse them into a "
        << "single shared slot, which lets one kernel clobber the redirect source of another "
        << "while an async D2D copy is still in flight (parity test #146 flake).";
}

// ----------------------------------------------------------------------------
// REGRESSION: merge() must keep both slices distinct.
// ----------------------------------------------------------------------------
//
// This is the load-bearing property: the orchestrator merges per-stage
// workspace requirements into one DeviceWorkspaceManager allocation. If two
// kernels share a name, merge() keeps ONE buffer of the larger size, and
// every kernel in that device's pool then aliases the same memory.
TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       MergedRequirements_KeepBothTempCFp32Slices)
{
    auto weights_a = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/1);
    auto weights_b = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/2);

    CUDAQuantisedGemmKernel kernel_a(weights_a.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_b(weights_b.get(), kFakeCudaDeviceId);

    auto reqs_a = kernel_a.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto reqs_b = kernel_b.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);

    // Sanity: each individual kernel declares exactly one TEMP_C_FP32 buffer.
    ASSERT_EQ(countTempCFp32Buffers(reqs_a), 1u);
    ASSERT_EQ(countTempCFp32Buffers(reqs_b), 1u);

    // Merge as the orchestrator does (kernel_b's reqs merged into kernel_a's).
    reqs_a.merge(reqs_b);

    EXPECT_EQ(countTempCFp32Buffers(reqs_a), 2u)
        << "After merging two CUDAQuantisedGemmKernels' workspace requirements, only one "
        << "TEMP_C_FP32 buffer survived. Both kernels are aliasing the same redirect-source "
        << "memory, which races when async D2D copies overlap. Each kernel needs its OWN "
        << "uniquely-named slice so merge() preserves both buffers.";
}

// ----------------------------------------------------------------------------
// REGRESSION: many instances must remain independent.
// ----------------------------------------------------------------------------
//
// Realistic models cache O(layers * projections) GEMM kernels per device
// (Qwen2.5-7B at TP=2 has dozens). Verify the slice counter scales without
// collisions.
TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ManyInstances_AllProduceDistinctTempCFp32Names)
{
    constexpr int kNumKernels = 32;

    // Hold weight tensors so they outlive the kernels.
    std::vector<std::unique_ptr<Q8_0Tensor>> weights;
    weights.reserve(kNumKernels);
    std::vector<std::unique_ptr<CUDAQuantisedGemmKernel>> kernels;
    kernels.reserve(kNumKernels);

    for (int i = 0; i < kNumKernels; ++i)
    {
        weights.push_back(TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/100 + i));
        kernels.push_back(std::make_unique<CUDAQuantisedGemmKernel>(
            weights.back().get(), kFakeCudaDeviceId));
    }

    std::set<std::string> seen;
    for (int i = 0; i < kNumKernels; ++i)
    {
        auto reqs = kernels[i]->getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
        auto names = collectTempCFp32Names(reqs);
        ASSERT_EQ(names.size(), 1u) << "Kernel " << i << " must declare exactly one slice";
        const std::string &name = *names.begin();
        EXPECT_TRUE(seen.insert(name).second)
            << "Duplicate TEMP_C_FP32 buffer name across kernel instances: \"" << name << "\". "
            << "Two cached kernels will alias the same redirect-source memory.";
    }

    EXPECT_EQ(seen.size(), static_cast<size_t>(kNumKernels))
        << "Expected " << kNumKernels << " distinct TEMP_C_FP32 slice names, got "
        << seen.size();
}

// ----------------------------------------------------------------------------
// SANITY: the unique name still uses the standard prefix so logging /
// debugging tools that match on the prefix continue to work.
// ----------------------------------------------------------------------------
TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       TempCFp32SliceName_KeepsStandardPrefix)
{
    auto weights = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/7);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    auto reqs = kernel.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto names = collectTempCFp32Names(reqs);
    ASSERT_EQ(names.size(), 1u);
    const std::string &name = *names.begin();

    EXPECT_EQ(name.rfind(kTempCFp32Prefix, 0), 0u)
        << "TEMP_C_FP32 slice name \"" << name << "\" does not start with the standard "
        << "prefix \"" << kTempCFp32Prefix << "\". Tools that match on the prefix will miss it.";
}

// ----------------------------------------------------------------------------
// SANITY: the slice still has the correct size.
// ----------------------------------------------------------------------------
TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       TempCFp32SliceSize_MatchesOutputBytes)
{
    auto weights = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/9);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 8;
    constexpr int kN = 64;
    constexpr int kK = 128;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *temp_c = nullptr;
    for (const auto &b : reqs.buffers)
    {
        if (b.name.rfind(kTempCFp32Prefix, 0) == 0)
        {
            temp_c = &b;
            break;
        }
    }
    ASSERT_NE(temp_c, nullptr);
    EXPECT_EQ(temp_c->size_bytes,
              static_cast<size_t>(paddedPrefillM(kM)) * kN * sizeof(float));
}

// ----------------------------------------------------------------------------
// REGRESSION: concurrent prefill accumulator scratch must be declared as
// workspace, not hidden cudaMalloc-owned per-stream pool memory.
// ----------------------------------------------------------------------------
TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ConcurrentPrefillAccumulatorWorkspace_HasTwoExtraPaddedSlots)
{
    auto weights = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/11);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 17; // exercises tile-padding, not the M=1 decode path
    constexpr int kN = 64;
    constexpr int kK = 128;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *acc = reqs.find(GemmWorkspaceBuffers::ACC_INT32);
    const WorkspaceDescriptor *concurrent_acc =
        reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32);

    ASSERT_NE(acc, nullptr);
    ASSERT_NE(concurrent_acc, nullptr)
        << "Concurrent prefill must declare workspace-owned extra accumulator slots; "
        << "a hidden per-stream cudaMalloc pool is not graph/V RAM accounting friendly.";

    const size_t one_slot_bytes =
        static_cast<size_t>(paddedPrefillM(kM)) * kN * sizeof(int32_t);
    EXPECT_EQ(acc->size_bytes, one_slot_bytes);
    EXPECT_EQ(concurrent_acc->size_bytes,
              static_cast<size_t>(kConcurrentPrefillExtraAccumulatorSlots) * one_slot_bytes);
}

// The concurrent accumulator is intentionally shared across all CUDA quantized
// GEMM kernels on a device. WorkspaceRequirements::merge() should keep one
// buffer sized to the largest projection, unlike TEMP_C_FP32 which is uniquely
// sliced per kernel instance.
TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       MergedRequirements_KeepLargestConcurrentPrefillAccumulator)
{
    auto weights_small = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/12);
    auto weights_large = TestTensorFactory::createQ8_0Random({96, 128}, /*seed=*/13);

    CUDAQuantisedGemmKernel kernel_small(weights_small.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_large(weights_large.get(), kFakeCudaDeviceId);

    auto reqs_small = kernel_small.getWorkspaceRequirements(/*m=*/17, /*n=*/64, /*k=*/128);
    auto reqs_large = kernel_large.getWorkspaceRequirements(/*m=*/17, /*n=*/96, /*k=*/128);

    const auto *small = reqs_small.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32);
    const auto *large = reqs_large.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32);
    ASSERT_NE(small, nullptr);
    ASSERT_NE(large, nullptr);
    ASSERT_GT(large->size_bytes, small->size_bytes);

    reqs_small.merge(reqs_large);

    const auto *merged = reqs_small.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32);
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->size_bytes, large->size_bytes);
}
