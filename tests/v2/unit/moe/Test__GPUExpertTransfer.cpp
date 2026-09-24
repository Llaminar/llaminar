/**
 * @file Test__GPUExpertTransfer.cpp
 * @brief Backend-neutral descriptor contract tests for GPU expert transfer.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "execution/moe/GPUExpertTransfer.h"

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "kernels/cuda/gemm/CUDAWeightPacker.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "../../utils/TestTensorFactory.h"
#endif

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace llaminar2;

TEST(Test__GPUExpertTransfer, PackedDescriptorContract)
{
    DeviceNativeVNNIMatrixDesc src{};
    src.payload = reinterpret_cast<const uint8_t *>(0x1000);
    src.scales = reinterpret_cast<const void *>(0x2000);
    src.mins = reinterpret_cast<const void *>(0x3000);
    src.emins = reinterpret_cast<const void *>(0x4000);
    src.n = 16;
    src.k = 64;
    src.blocks_per_row = 2;
    src.codebook_id = 7;

    DeviceNativeVNNIMatrixDesc dst = src;
    dst.payload = reinterpret_cast<const uint8_t *>(0x5000);
    dst.scales = reinterpret_cast<const void *>(0x6000);
    dst.mins = reinterpret_cast<const void *>(0x7000);
    dst.emins = reinterpret_cast<const void *>(0x8000);

    auto src_desc = makeGpuExpertPackedDescriptor(
        src, /*payload_bytes_per_block=*/16, /*is_asymmetric=*/true, /*has_emins=*/true);
    auto dst_desc = makeGpuExpertPackedDescriptor(
        dst, /*payload_bytes_per_block=*/16, /*is_asymmetric=*/true, /*has_emins=*/true);

    EXPECT_TRUE(src_desc.valid());
    EXPECT_TRUE(gpuExpertPackedDescriptorsCompatible(src_desc, dst_desc));
    EXPECT_EQ(src_desc.vnni_bytes, 512u);
    EXPECT_EQ(src_desc.scales_bytes, 64u);
    EXPECT_EQ(src_desc.mins_bytes, 64u);
    EXPECT_EQ(src_desc.emins_bytes, 128u);
    EXPECT_EQ(src_desc.totalBytes(), 768u);

    dst_desc.codebook_id = 8;
    EXPECT_FALSE(gpuExpertPackedDescriptorsCompatible(src_desc, dst_desc));
}

TEST(Test__GPUExpertTransfer, RejectsCrossBackendTransfer)
{
    DeviceNativeVNNIMatrixDesc matrix{};
    matrix.payload = reinterpret_cast<const uint8_t *>(0x1000);
    matrix.scales = reinterpret_cast<const void *>(0x2000);
    matrix.n = 16;
    matrix.k = 64;
    matrix.blocks_per_row = 2;
    matrix.codebook_id = 7;
    auto desc = makeGpuExpertPackedDescriptor(
        matrix, /*payload_bytes_per_block=*/16, /*is_asymmetric=*/false, /*has_emins=*/false);

    EXPECT_FALSE(GPUExpertTransfer::transferExpert(
        desc, desc, DeviceId::cuda(0), DeviceId::rocm(0), reinterpret_cast<void *>(0x1)));
}

TEST(Test__GPUExpertTransfer, RejectsNullStream)
{
    GPUExpertPointers ptrs;
    ptrs.d_vnni = reinterpret_cast<uint8_t *>(0x1000);
    ptrs.d_scales = reinterpret_cast<void *>(0x2000);

    EXPECT_FALSE(GPUExpertTransfer::transferExpert(
        ptrs,
        ptrs,
        DeviceId::cuda(0),
        DeviceId::cuda(0),
        /*vnni_bytes=*/16,
        /*scales_bytes=*/16,
        /*mins_bytes=*/0,
        /*emins_bytes=*/0,
        nullptr));
}

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
TEST(Test__GPUExpertTransfer, CUDAROCmHostPackersProduceBitwiseIdenticalNativeVNNI)
{
    struct Case
    {
        const char *name;
        std::function<std::unique_ptr<TensorBase>()> create;
    };

    const std::vector<Case> cases = {
        {"Q4_0", [] { return test::TestTensorFactory::createQ4_0Random({16, 128}); }},
        {"Q4_1", [] { return test::TestTensorFactory::createQ4_1Random({16, 128}); }},
        {"IQ3_S", [] { return test::TestTensorFactory::createIQ3_SRandom({16, 256}); }},
        {"Q2_K", [] { return test::TestTensorFactory::createQ2_KRandom({16, 256}); }},
    };

    for (const auto &tc : cases)
    {
        auto tensor = tc.create();
        ASSERT_NE(tensor, nullptr) << tc.name;

        cuda::CUDAPackedWeights cuda_packed;
        rocm::ROCmPackedWeights rocm_packed;
        ASSERT_TRUE(cuda::packWeightsToCUDA(tensor.get(), cuda_packed)) << tc.name;
        ASSERT_TRUE(rocm::packWeightsToROCm(tensor.get(), rocm_packed)) << tc.name;

        EXPECT_EQ(cuda_packed.N, rocm_packed.N) << tc.name;
        EXPECT_EQ(cuda_packed.K, rocm_packed.K) << tc.name;
        EXPECT_EQ(cuda_packed.native_codebook_id, rocm_packed.native_vnni_codebook_id) << tc.name;
        EXPECT_EQ(cuda_packed.native_blocks_per_row, rocm_packed.native_vnni_blocks_per_row) << tc.name;

        EXPECT_EQ(cuda_packed.native_vnni, rocm_packed.native_vnni_payload) << tc.name;
        EXPECT_EQ(cuda_packed.native_scales, rocm_packed.native_vnni_scales) << tc.name;
        EXPECT_EQ(cuda_packed.native_mins, rocm_packed.native_vnni_mins) << tc.name;
        EXPECT_EQ(cuda_packed.native_emins, rocm_packed.native_vnni_emins) << tc.name;
    }
}
#endif
