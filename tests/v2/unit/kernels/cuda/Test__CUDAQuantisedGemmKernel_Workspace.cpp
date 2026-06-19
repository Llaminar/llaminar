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

#include "execution/compute_stages/ComputeStageUtils.h"
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
    constexpr int kConcurrentPrefillWorkspaceSlots = 4;
    constexpr int kConcurrentPrefillExtraAccumulatorSlots = 3;

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
       SumsABlockwiseSize_MatchesPaddedPrefillRows)
{
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/14);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 17;
    constexpr int kN = 64;
    constexpr int kK = 256;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *sums_a =
        reqs.find(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE);
    ASSERT_NE(sums_a, nullptr)
        << "Asymmetric NativeVNNI prefill consumes precomputed activation block sums "
           "from declared workspace, not ad hoc scratch.";

    constexpr int kBlocksPerRow = kK / 32;
    EXPECT_EQ(sums_a->size_bytes,
              static_cast<size_t>(paddedPrefillM(kM)) * kBlocksPerRow * sizeof(int32_t));
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       GemvKparPartialsWorkspace_CoversQwen36LmHeadDeterministicDecode)
{
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/15);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 1;
    constexpr int kN = 248320;
    constexpr int kK = 5120;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *partials =
        reqs.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    ASSERT_NE(partials, nullptr)
        << "CUDA deterministic parity routes KPAR GEMV through workspace-owned "
           "two-phase partials; LM head must not rely on hidden allocations.";

    EXPECT_GE(partials->size_bytes,
              static_cast<size_t>(kN) * sizeof(float));
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       SingleProjectionGemvPartials_DoNotReserveSideStreamSlots)
{
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/151);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 1;
    constexpr int kN = 8192;
    constexpr int kK = 2048;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *serial =
        reqs.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    const WorkspaceDescriptor *concurrent =
        reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);

    ASSERT_NE(serial, nullptr);
    EXPECT_EQ(concurrent, nullptr)
        << "Single-output GEMV kernels such as LM head cannot consume CUDA decode "
           "side-stream slots. Fused stages add those slots when projection fan-out "
           "is known.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ConcurrentDecodeGemvPartials_HelperUsesLargestProjectionSlot)
{
    auto weights_small = TestTensorFactory::createQ8_0Random({512, 2048}, /*seed=*/152);
    auto weights_large = TestTensorFactory::createQ8_0Random({8192, 2048}, /*seed=*/153);

    CUDAQuantisedGemmKernel kernel_small(weights_small.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_large(weights_large.get(), kFakeCudaDeviceId);

    auto reqs_small = kernel_small.getWorkspaceRequirements(/*m=*/1, /*n=*/512, /*k=*/2048);
    auto reqs_large = kernel_large.getWorkspaceRequirements(/*m=*/1, /*n=*/8192, /*k=*/2048);

    const auto *small_serial = reqs_small.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    const auto *large_serial = reqs_large.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    ASSERT_NE(small_serial, nullptr);
    ASSERT_NE(large_serial, nullptr);
    ASSERT_GT(large_serial->size_bytes, small_serial->size_bytes);

    reqs_small.merge(reqs_large);
    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs_small,
        DeviceId::cuda(kFakeCudaDeviceId),
        /*m=*/1,
        /*projection_count=*/4);

    const auto *merged =
        reqs_small.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->size_bytes, 3u * large_serial->size_bytes)
        << "A four-projection CUDA fused decode stage needs three side-stream slots, "
           "each sized for the largest projection in the fused group.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ConcurrentVerifierGemvPartials_M4HelperUsesLargestProjectionSlot)
{
    auto weights_small = TestTensorFactory::createQ8_0Random({512, 2048}, /*seed=*/155);
    auto weights_large = TestTensorFactory::createQ8_0Random({8192, 2048}, /*seed=*/156);

    CUDAQuantisedGemmKernel kernel_small(weights_small.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_large(weights_large.get(), kFakeCudaDeviceId);

    constexpr int kM = 4;
    auto reqs_small = kernel_small.getWorkspaceRequirements(kM, /*n=*/512, /*k=*/2048);
    auto reqs_large = kernel_large.getWorkspaceRequirements(kM, /*n=*/8192, /*k=*/2048);

    const auto *small_serial = reqs_small.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    const auto *large_serial = reqs_large.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    ASSERT_NE(small_serial, nullptr);
    ASSERT_NE(large_serial, nullptr);
    ASSERT_GT(large_serial->size_bytes, small_serial->size_bytes);

    reqs_small.merge(reqs_large);
    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs_small,
        DeviceId::cuda(kFakeCudaDeviceId),
        kM,
        /*projection_count=*/4);

    const auto *merged =
        reqs_small.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);
    ASSERT_NE(merged, nullptr)
        << "Grouped verifier rows M=2..4 use the same CUDA side-stream GEMV "
           "partials as decode and must declare those slots structurally.";
    EXPECT_EQ(merged->size_bytes, 3u * large_serial->size_bytes)
        << "A four-projection CUDA fused verifier stage needs three side-stream "
           "slots, each sized for the largest M=4 projection.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ConcurrentDecodeGemvPartials_MoETopKFanoutUsesActiveStreamSlots)
{
    auto weights = TestTensorFactory::createQ8_0Random({4096, 2048}, /*seed=*/154);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    auto reqs = kernel.getWorkspaceRequirements(/*m=*/1, /*n=*/4096, /*k=*/2048);
    const auto *serial = reqs.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    ASSERT_NE(serial, nullptr);

    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs,
        DeviceId::cuda(kFakeCudaDeviceId),
        /*m=*/1,
        /*projection_count=*/16);

    const auto *merged =
        reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->size_bytes, 7u * serial->size_bytes)
        << "Qwen3.6 MoE decode can fuse top_k=8 gate/up into sixteen projections, "
           "but the CUDA decode pool has eight active streams. Later projections "
           "reuse a completed stream slot, so only seven side-stream arenas are "
           "required.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ConcurrentDecodeGemvPartials_HelperNoOpsForNonCudaAndSingleProjection)
{
    WorkspaceRequirements reqs;
    reqs.buffers.push_back({
        GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS,
        4096,
        256,
        true});

    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs,
        DeviceId::rocm(0),
        /*m=*/1,
        /*projection_count=*/4);
    EXPECT_EQ(reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS),
              nullptr)
        << "The helper is intentionally CUDA-only; ROCm stages own their own "
           "workspace declarations.";

    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs,
        DeviceId::cuda(kFakeCudaDeviceId),
        /*m=*/1,
        /*projection_count=*/1);
    EXPECT_EQ(reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS),
              nullptr)
        << "A single projection has no side stream and should not reserve side slots.";
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
       NativeVNNIPrefillSplitKWorkspace_HasConcurrentSlotsForPromptPrefill)
{
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/1010);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 596;
    constexpr int kN = 5120;
    constexpr int kK = 17408;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *splitk =
        reqs.find(GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_SPLITK_PARTIALS);
    ASSERT_NE(splitk, nullptr);

    const size_t one_slot_bytes =
        static_cast<size_t>(4) * paddedPrefillM(kM) * kN * sizeof(float);
    EXPECT_GE(splitk->size_bytes,
              static_cast<size_t>(kConcurrentPrefillWorkspaceSlots) * one_slot_bytes)
        << "Concurrent CUDA prefill runs fused projections on side streams, so split-K "
           "partials need disjoint per-stream slots instead of one serial scratch buffer.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       ConcurrentPrefillAccumulatorWorkspace_HasThreeExtraPaddedSlots)
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
