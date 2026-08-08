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
#include "utils/PrefillGraphBucketDefaults.h"
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

    /**
     * @brief Restore the mutable debug switch after a structural workspace test.
     *
     * Workspace planning reads this process-local setting but launches no GPU
     * work. Keeping restoration in RAII form prevents an ASSERT from leaking a
     * serial/concurrent choice into later unit cases.
     */
    class ScopedCudaConcurrentPrefillSetting
    {
    public:
        explicit ScopedCudaConcurrentPrefillSetting(bool enabled)
            : previous_(mutableDebugEnv().gemm.cuda_concurrent_prefill)
        {
            set(enabled);
        }

        ~ScopedCudaConcurrentPrefillSetting()
        {
            set(previous_);
        }

        void set(bool enabled)
        {
            mutableDebugEnv().gemm.cuda_concurrent_prefill = enabled;
        }

        ScopedCudaConcurrentPrefillSetting(
            const ScopedCudaConcurrentPrefillSetting &) = delete;
        ScopedCudaConcurrentPrefillSetting &operator=(
            const ScopedCudaConcurrentPrefillSetting &) = delete;

    private:
        bool previous_ = true;
    };

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
    EXPECT_EQ(
        partials->regime,
        WorkspaceExecutionRegime::CompactDecodeOnly)
        << "M=1 KPAR storage belongs to the compact decode participant, not "
           "the large-M prefill layout.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       Qwen36LargeMDeclaresBoundedGroupedParticipantForMTPTotality)
{
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/150);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 595;
    constexpr int kN = 17408;
    constexpr int kK = 5120;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    const WorkspaceDescriptor *grouped =
        reqs.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    ASSERT_NE(grouped, nullptr)
        << "Grouped verifier M is total and may exceed the default captured "
           "depth by tiling through the unified persistent KPAR arena.";
    EXPECT_EQ(
        grouped->regime,
        WorkspaceExecutionRegime::CompactDecodeOnly)
        << "The graph-family planner, which knows whether this shape is prefill "
           "or verifier execution, must remove this descriptor from prefill.";

    const size_t expected_rows = static_cast<size_t>(
        nativeVNNIPersistentVerifierWorkspaceRows(kM, kN, kK));
    const size_t k_groups = static_cast<size_t>((kK + 31) / 32);
    EXPECT_EQ(
        grouped->size_bytes,
        k_groups * expected_rows * static_cast<size_t>(kN) * sizeof(float));
    EXPECT_LT(grouped->size_bytes, 512u * 1024u * 1024u)
        << "Runtime-M totality must reuse bounded grouped storage rather than "
           "reserving prompt-M partials.";
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
       GroupedVerifierGemvPartials_M4UsesUnifiedPersistentArena)
{
    auto weights_small = TestTensorFactory::createQ8_0Random({512, 2048}, /*seed=*/155);
    auto weights_large = TestTensorFactory::createQ8_0Random({8192, 2048}, /*seed=*/156);

    CUDAQuantisedGemmKernel kernel_small(weights_small.get(), kFakeCudaDeviceId);
    CUDAQuantisedGemmKernel kernel_large(weights_large.get(), kFakeCudaDeviceId);

    constexpr int kM = 4;
    auto reqs_small = kernel_small.getWorkspaceRequirements(kM, /*n=*/512, /*k=*/2048);
    auto reqs_large = kernel_large.getWorkspaceRequirements(kM, /*n=*/8192, /*k=*/2048);

    const auto *small_grouped = reqs_small.find(
        GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    const auto *large_grouped = reqs_large.find(
        GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    ASSERT_NE(small_grouped, nullptr);
    ASSERT_NE(large_grouped, nullptr);
    ASSERT_GT(large_grouped->size_bytes, small_grouped->size_bytes);
    EXPECT_EQ(
        large_grouped->regime,
        WorkspaceExecutionRegime::CompactDecodeOnly);

    reqs_small.merge(reqs_large);
    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs_small,
        DeviceId::cuda(kFakeCudaDeviceId),
        kM,
        /*projection_count=*/4);

    const auto *merged =
        reqs_small.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);
    EXPECT_EQ(merged, nullptr)
        << "The production grouped verifier quantizes once and executes all "
           "projections in-order on its explicit graph stream. Reserving M=1 "
           "side-stream slots would inflate the serial participant and encode "
           "a launch topology the grouped implementation does not use.";
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
       NativeVNNIPrefillGenericDirectScheduleDeclaresNoKpartScratch)
{
    ScopedCudaConcurrentPrefillSetting concurrent_prefill(/*enabled=*/false);
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/10);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 596;
    // Deliberately unswept geometry: this case isolates the total generic
    // direct policy without consulting a device-specific exact overlay.
    constexpr int kN = 5119;
    constexpr int kK = 17408;
    auto reqs = kernel.getWorkspaceRequirements(kM, kN, kK);

    EXPECT_EQ(
        reqs.find(
            GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS),
        nullptr)
        << "The direct output-owner schedule folds public-M=1 partitions inside "
           "each tile and must not reserve a global partial arena.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       NativeVNNIPrefillConcurrentGenericDirectScheduleDoesNotInventKpartScratch)
{
    ScopedCudaConcurrentPrefillSetting concurrent_prefill(/*enabled=*/false);
    auto weights = TestTensorFactory::createQ4_KRandom({64, 256}, /*seed=*/1010);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kM = 596;
    // Deliberately unswept geometry keeps this host-only unit test independent
    // of the device-specific exact-overlay workspace envelope.
    constexpr int kN = 5119;
    constexpr int kK = 17408;
    const auto serial_reqs = kernel.getWorkspaceRequirements(kM, kN, kK);
    EXPECT_EQ(
        serial_reqs.find(
            GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS),
        nullptr);

    concurrent_prefill.set(/*enabled=*/true);
    const auto concurrent_reqs = kernel.getWorkspaceRequirements(kM, kN, kK);
    EXPECT_EQ(
        concurrent_reqs.find(
            GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS),
        nullptr)
        << "Projection concurrency must not allocate K-partition scratch unless "
           "the selected launch family actually publishes partition partials.";
}

TEST_F(Test__CUDAQuantisedGemmKernel_Workspace,
       NativeVNNIPrefillGenericDirectPromptAndExpertSchedulesNeedNoKpartScratch)
{
    auto weights = TestTensorFactory::createQ8_0Random({512, 2048}, /*seed=*/1011);
    CUDAQuantisedGemmKernel kernel(weights.get(), kFakeCudaDeviceId);

    constexpr int kPromptM = 595;
    constexpr int kExpertRows = 312;
    // N=513 is intentionally absent from the installed production overlay.
    constexpr int kN = 513;
    constexpr int kK = 2048;
    auto prompt_reqs = kernel.getWorkspaceRequirements(kPromptM, kN, kK);
    auto expert_reqs = kernel.getWorkspaceRequirements(kExpertRows, kN, kK);

    EXPECT_EQ(
        prompt_reqs.find(
            GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS),
        nullptr);
    EXPECT_EQ(
        expert_reqs.find(
            GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS),
        nullptr);
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
    EXPECT_EQ(acc->regime, WorkspaceExecutionRegime::PrefillOnly);
    EXPECT_EQ(concurrent_acc->size_bytes,
              static_cast<size_t>(kConcurrentPrefillExtraAccumulatorSlots) * one_slot_bytes);
    EXPECT_EQ(
        concurrent_acc->regime,
        WorkspaceExecutionRegime::PrefillOnly);

    const WorkspaceDescriptor *sums =
        reqs.find(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE);
    ASSERT_NE(sums, nullptr);
    EXPECT_EQ(sums->regime, WorkspaceExecutionRegime::PrefillOnly)
        << "Decode-equivalent M=1 and grouped routes quantize without activation sums.";
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
