/**
 * @file Test__CUDAQuantisedGemmKernel_Workspace.cpp
 * @brief Regression tests for CUDAQuantisedGemmKernel workspace declarations.
 *
 * TEMP_C_FP32 is serial mapped-output redirect scratch. It must stay shared
 * across cached GEMM kernels and merge to the largest required shape; otherwise
 * full model workspace planning grows by O(layers * projections) and can
 * exhaust VRAM before inference starts. Concurrent mapped-output paths need
 * their own explicit batched/pool scratch and must not reuse this serial buffer.
 *
 * These tests verify the structural workspace contract without CUDA hardware.
 * The kernel constructor and getWorkspaceRequirements() are pure host work.
 */

#include <gtest/gtest.h>

#include <memory>
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
    constexpr int kConcurrentPrefillExtraAccumulatorSlots = 2;

    int paddedPrefillM(int m)
    {
        return (m > 1) ? ((m + 127) & ~127) : m;
    }

    size_t countTempCFp32Buffers(const WorkspaceRequirements &reqs)
    {
        size_t n = 0;
        for (const auto &b : reqs.buffers)
        {
            if (b.name.rfind(GemmWorkspaceBuffers::TEMP_C_FP32, 0) == 0)
            {
                ++n;
            }
        }
        return n;
    }
} // namespace

class Test__CUDAQuantisedGemmKernel_Workspace : public ::testing::Test
{
protected:
    static constexpr int kFakeCudaDeviceId = 0;
};

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       TwoInstances_DeclareSharedStableTempCFp32BufferName)
{
    auto weights_a = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/1);
    auto weights_b = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/2);

    CUDAQuantisedGemmKernel kernel_a(weights_a.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_b(weights_b.get(), kFakeCudaDeviceId);

    auto reqs_a = kernel_a.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto reqs_b = kernel_b.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);

    ASSERT_NE(reqs_a.find(GemmWorkspaceBuffers::TEMP_C_FP32), nullptr);
    ASSERT_NE(reqs_b.find(GemmWorkspaceBuffers::TEMP_C_FP32), nullptr);
    EXPECT_EQ(countTempCFp32Buffers(reqs_a), 1u);
    EXPECT_EQ(countTempCFp32Buffers(reqs_b), 1u);
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       MergedRequirements_KeepSingleLargestTempCFp32Buffer)
{
    auto weights_small = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/3);
    auto weights_large = TestTensorFactory::createQ8_0Random({96, 128}, /*seed=*/4);

    CUDAQuantisedGemmKernel kernel_small(weights_small.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_large(weights_large.get(), kFakeCudaDeviceId);

    auto reqs_small = kernel_small.getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128);
    auto reqs_large = kernel_large.getWorkspaceRequirements(/*m=*/8, /*n=*/96, /*k=*/128);

    const auto *small = reqs_small.find(GemmWorkspaceBuffers::TEMP_C_FP32);
    const auto *large = reqs_large.find(GemmWorkspaceBuffers::TEMP_C_FP32);
    ASSERT_NE(small, nullptr);
    ASSERT_NE(large, nullptr);
    ASSERT_GT(large->size_bytes, small->size_bytes);

    reqs_small.merge(reqs_large);

    const auto *merged = reqs_small.find(GemmWorkspaceBuffers::TEMP_C_FP32);
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(countTempCFp32Buffers(reqs_small), 1u)
        << "TEMP_C_FP32 is serial scratch and must not be multiplied by cached kernel count";
    EXPECT_EQ(merged->size_bytes, large->size_bytes);
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ManyInstances_MergedTempCFp32DoesNotGrowWithKernelCount)
{
    constexpr int kNumKernels = 32;

    std::vector<std::unique_ptr<Q8_0Tensor>> weights;
    weights.reserve(kNumKernels);
    std::vector<std::unique_ptr<CUDAQuantisedGemmKernel>> kernels;
    kernels.reserve(kNumKernels);

    WorkspaceRequirements merged;
    for (int i = 0; i < kNumKernels; ++i)
    {
        weights.push_back(TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/100 + i));
        kernels.push_back(std::make_unique<CUDAQuantisedGemmKernel>(
            weights.back().get(), kFakeCudaDeviceId));
        merged.merge(kernels.back()->getWorkspaceRequirements(/*m=*/8, /*n=*/64, /*k=*/128));
    }

    const auto *temp_c = merged.find(GemmWorkspaceBuffers::TEMP_C_FP32);
    ASSERT_NE(temp_c, nullptr);
    EXPECT_EQ(countTempCFp32Buffers(merged), 1u)
        << "A full model graph must keep one shared TEMP_C_FP32 serial scratch buffer, "
        << "not one MxN buffer per cached GEMM kernel";
    EXPECT_EQ(temp_c->size_bytes,
              static_cast<size_t>(paddedPrefillM(8)) * 64 * sizeof(float));
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       TempCFp32Size_MatchesOutputBytes)
{
    auto weights = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/9);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 8;
    constexpr int kN = 64;
    constexpr int kK = 128;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *temp_c = reqs.find(GemmWorkspaceBuffers::TEMP_C_FP32);
    ASSERT_NE(temp_c, nullptr);
    EXPECT_EQ(temp_c->size_bytes,
              static_cast<size_t>(paddedPrefillM(kM)) * kN * sizeof(float));
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       NativeVNNIPrefillSplitKWorkspace_DeclaredForQwenLikeQ4KShape)
{
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/10);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 596;
    constexpr int kN = 5120;
    constexpr int kK = 17408;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *splitk =
        reqs.find(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS);
    ASSERT_NE(splitk, nullptr)
        << "NativeVNNI prefill split-K dispatch must be backed by declared workspace; "
        << "otherwise selected split-K launches fail at runtime.";

    EXPECT_GE(splitk->size_bytes,
              static_cast<size_t>(4) * paddedPrefillM(kM) * kN * sizeof(float));
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ConcurrentPrefillAccumulatorWorkspace_HasTwoExtraPaddedSlots)
{
    auto weights = TestTensorFactory::createQ8_0Random({64, 128}, /*seed=*/11);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 17;
    constexpr int kN = 64;
    constexpr int kK = 128;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *acc = reqs.find(GemmWorkspaceBuffers::ACC_INT32);
    const WorkspaceDescriptor *concurrent_acc =
        reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_PREFILL_ACC_INT32);

    ASSERT_NE(acc, nullptr);
    ASSERT_NE(concurrent_acc, nullptr)
        << "Concurrent prefill must declare workspace-owned extra accumulator slots; "
        << "a hidden per-stream cudaMalloc pool is not graph/VRAM accounting friendly.";

    const size_t one_slot_bytes =
        static_cast<size_t>(paddedPrefillM(kM)) * kN * sizeof(int32_t);
    EXPECT_EQ(acc->size_bytes, one_slot_bytes);
    EXPECT_EQ(concurrent_acc->size_bytes,
              static_cast<size_t>(kConcurrentPrefillExtraAccumulatorSlots) * one_slot_bytes);
}

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
