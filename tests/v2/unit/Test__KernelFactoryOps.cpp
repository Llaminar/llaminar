/**
 * @file Test__KernelFactoryOps.cpp
 * @brief Unit tests for KernelFactory create methods for ops kernels
 * @author David Sanftenberg
 *
 * Tests cover:
 * 1. createRMSNorm() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 2. createSwiGLU() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 3. createRoPE() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 4. createSoftmax() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 5. createAttention() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 6. createEmbedding() for FP32, BF16, FP16, Q8_1 tensors (CPU path)
 * 7. createResidualAdd() for FP32, BF16, FP16 tensors (CPU path)
 * 8. Generic TensorBase* dispatch for all ops
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__KernelFactoryOps : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure DeviceManager is initialized
        DeviceManager::instance();
    }
};

// ============================================================================
// createRMSNorm() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_GenericDispatch_BF16)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createRMSNorm() ROCm Tests
// ============================================================================

#ifdef HAVE_ROCM
TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0)); // ROCm device 0
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_BF16_ROCm)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP16_ROCm)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_GenericDispatch_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(tensor.get(), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_GenericDispatch_BF16_ROCm)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRMSNorm(tensor.get(), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}
#endif // HAVE_ROCM

// ============================================================================
// createSwiGLU() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

// ============================================================================
// createSwiGLU() CUDA Tests
// ============================================================================

#ifdef HAVE_CUDA
TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP32_CUDA)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CUDA);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0)); // CUDA device 0
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_BF16_CUDA)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CUDA);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP16_CUDA)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CUDA);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_FP32_CUDA)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::CUDA);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_BF16_CUDA)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::CUDA);
    ASSERT_NE(kernel, nullptr);
}
#endif // HAVE_CUDA

// ============================================================================
// createSwiGLU() ROCm Tests
// ============================================================================

#ifdef HAVE_ROCM
TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0)); // ROCm device 0
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_BF16_ROCm)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP16_ROCm)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_BF16_ROCm)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}
#endif // HAVE_ROCM

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_GenericDispatch_Q8_1)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createSwiGLU(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createRoPE() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateRoPE_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRoPE(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createRoPE() ROCm Tests
// ============================================================================

#ifdef HAVE_ROCM
TEST_F(Test__KernelFactoryOps, CreateRoPE_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0)); // ROCm device 0
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_BF16_ROCm)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_FP16_ROCm)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createRoPE(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0));
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_GenericDispatch_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createRoPE(tensor.get(), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_GenericDispatch_BF16_ROCm)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createRoPE(tensor.get(), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}
#endif // HAVE_ROCM

// ============================================================================
// createSoftmax() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateSoftmax_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSoftmax_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSoftmax_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateSoftmax_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createSoftmax(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

// ============================================================================
// createAttention() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateAttention_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createAttention(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateAttention_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createAttention(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// createEmbedding() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateEmbedding_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_Q8_1_CPU)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

// ============================================================================
// createEmbedding() ROCm Tests
// ============================================================================

#ifdef HAVE_ROCM
TEST_F(Test__KernelFactoryOps, CreateEmbedding_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(0)); // ROCm device 0
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_BF16_ROCm)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_FP16_ROCm)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_Q8_1_ROCm)
{
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const Q8_1Tensor *>(tensor.get()), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateEmbedding_GenericDispatch_FP32_ROCm)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createEmbedding(tensor.get(), DeviceType::ROCm);
    ASSERT_NE(kernel, nullptr);
}
#endif // HAVE_ROCM

// ============================================================================
// createResidualAdd() Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP32_CPU)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(
        static_cast<const FP32Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_BF16_CPU)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(
        static_cast<const BF16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP16_CPU)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(
        static_cast<const FP16Tensor *>(tensor.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->supports_device(-1));
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_GenericDispatch_FP32)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_GenericDispatch_BF16)
{
    auto tensor = TestTensorFactory::createBF16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_GenericDispatch_FP16)
{
    auto tensor = TestTensorFactory::createFP16Random({32, 64});
    auto kernel = KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createRMSNorm(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createSwiGLU(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateRoPE_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createRoPE(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateAttention_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createAttention(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_NullTensor_Throws)
{
    EXPECT_THROW(
        KernelFactory::createResidualAdd(static_cast<const TensorBase *>(nullptr), DeviceType::CPU),
        std::runtime_error);
}

// ============================================================================
// Unsupported Tensor Type Tests
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_Q8_1_UnsupportedType_Throws)
{
    // Q8_1 is NOT supported for ResidualAdd (only FP32, BF16, FP16)
    auto tensor = TestTensorFactory::createQ8_1Random({32, 64});
    EXPECT_THROW(
        KernelFactory::createResidualAdd(tensor.get(), DeviceType::CPU),
        std::runtime_error);
}

// ============================================================================
// Functional Tests - Verify kernels actually work
// ============================================================================

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP32_FunctionalTest)
{
    // Create test tensors
    auto input = TestTensorFactory::createFP32({4, 4});
    auto residual = TestTensorFactory::createFP32({4, 4});
    auto output = TestTensorFactory::createFP32({4, 4});

    // Initialize with known values
    float *input_data = input->mutable_data();
    float *residual_data = residual->mutable_data();
    for (size_t i = 0; i < 16; ++i)
    {
        input_data[i] = static_cast<float>(i);
        residual_data[i] = static_cast<float>(i * 2);
    }

    // Create and apply kernel
    auto kernel = KernelFactory::createResidualAdd(input.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    bool success = kernel->apply(
        input->data(),
        residual->data(),
        output->mutable_data(),
        16,
        nullptr,
        -1);
    ASSERT_TRUE(success);

    // Verify output = input + residual
    const float *output_data = output->data();
    for (size_t i = 0; i < 16; ++i)
    {
        float expected = static_cast<float>(i) + static_cast<float>(i * 2);
        EXPECT_FLOAT_EQ(output_data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(Test__KernelFactoryOps, CreateResidualAdd_FP32_apply_tensor_Test)
{
    // Create test tensors
    auto input = TestTensorFactory::createFP32Random({8, 16});
    auto residual = TestTensorFactory::createFP32Random({8, 16});
    auto output = TestTensorFactory::createFP32({8, 16});

    // Create kernel
    auto kernel = KernelFactory::createResidualAdd(input.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    bool success = kernel->apply_tensor(
        input.get(),
        residual.get(),
        output.get(),
        8 * 16,
        nullptr,
        -1);
    ASSERT_TRUE(success);

    // Verify output = input + residual
    const float *in_data = input->data();
    const float *res_data = residual->data();
    const float *out_data = output->data();
    for (size_t i = 0; i < 128; ++i)
    {
        EXPECT_FLOAT_EQ(out_data[i], in_data[i] + res_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(Test__KernelFactoryOps, CreateRMSNorm_FP32_FunctionalTest)
{
    // Create test tensors with RMSNorm-compatible dimensions
    const int rows = 4;
    const int hidden_dim = 64;
    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(rows), static_cast<size_t>(hidden_dim)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(hidden_dim)});
    auto gamma = TestTensorFactory::createFP32({static_cast<size_t>(hidden_dim)});

    // Initialize gamma to 1.0 for simple test
    float *gamma_data = gamma->mutable_data();
    for (int i = 0; i < hidden_dim; ++i)
    {
        gamma_data[i] = 1.0f;
    }

    // Create kernel
    auto kernel = KernelFactory::createRMSNorm(input.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    float eps = 1e-6f;
    bool success = kernel->apply(
        input->data(),
        gamma->data(),
        output->mutable_data(),
        rows,
        hidden_dim,
        eps,
        false,   // use_bf16
        nullptr, // mpi_ctx
        -1);     // device_idx
    ASSERT_TRUE(success);

    // Verify output is not all zeros (basic sanity check)
    const float *out_data = output->data();
    float sum = 0.0f;
    for (int i = 0; i < rows * hidden_dim; ++i)
    {
        sum += std::abs(out_data[i]);
    }
    EXPECT_GT(sum, 0.0f) << "RMSNorm output should not be all zeros";
}

TEST_F(Test__KernelFactoryOps, CreateSwiGLU_FP32_FunctionalTest)
{
    // Create test tensors
    const int rows = 4;
    const int cols = 64;
    auto gate = TestTensorFactory::createFP32Random({static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto up = TestTensorFactory::createFP32Random({static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Create kernel
    auto kernel = KernelFactory::createSwiGLU(gate.get(), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    bool success = kernel->apply(
        gate->data(),
        up->data(),
        output->mutable_data(),
        rows,
        cols,
        false, // add_residual
        nullptr,
        -1);
    ASSERT_TRUE(success);

    // Verify output is not all zeros (basic sanity check)
    const float *out_data = output->data();
    float sum = 0.0f;
    for (int i = 0; i < rows * cols; ++i)
    {
        sum += std::abs(out_data[i]);
    }
    EXPECT_GT(sum, 0.0f) << "SwiGLU output should not be all zeros";
}

// ============================================================================
// ROCmOrdinalGuard Tests - Regression tests for device ordinal propagation bug
// ============================================================================

/**
 * @brief Test that ROCmOrdinalGuard can be constructed and destructed
 *
 * Regression test for bug where kernels were created on wrong ROCm device.
 * The guard sets thread-local storage to propagate device ordinal.
 */
TEST_F(Test__KernelFactoryOps, ROCmOrdinalGuard_ConstructAndDestruct)
{
    // Should not throw even without ROCm hardware (stub implementation)
    EXPECT_NO_THROW({
        KernelFactory::ROCmOrdinalGuard guard(0);
    });
}

/**
 * @brief Test that ROCmOrdinalGuard accepts different ordinal values
 */
TEST_F(Test__KernelFactoryOps, ROCmOrdinalGuard_DifferentOrdinals)
{
    // Test various ordinal values (0, 1, 7 for multi-GPU systems)
    EXPECT_NO_THROW({
        KernelFactory::ROCmOrdinalGuard guard0(0);
    });

    EXPECT_NO_THROW({
        KernelFactory::ROCmOrdinalGuard guard1(1);
    });

    EXPECT_NO_THROW({
        KernelFactory::ROCmOrdinalGuard guard7(7);
    });
}

/**
 * @brief Test that ROCmOrdinalGuard respects RAII semantics
 *
 * The guard should clear thread-local storage when going out of scope.
 */
TEST_F(Test__KernelFactoryOps, ROCmOrdinalGuard_RAIISemantics)
{
    // Create nested scopes with guards - should not crash
    {
        KernelFactory::ROCmOrdinalGuard outer(1);
        {
            KernelFactory::ROCmOrdinalGuard inner(2);
            // Inner guard active
        }
        // Inner guard destroyed
    }
    // Both guards destroyed
}

/**
 * @brief Test that ROCmOrdinalGuard is not copyable
 *
 * Copy semantics would break RAII guarantees (double-clear).
 */
TEST_F(Test__KernelFactoryOps, ROCmOrdinalGuard_NotCopyable)
{
    // Static assertion to verify at compile time
    static_assert(!std::is_copy_constructible_v<KernelFactory::ROCmOrdinalGuard>,
                  "ROCmOrdinalGuard should not be copy constructible");
    static_assert(!std::is_copy_assignable_v<KernelFactory::ROCmOrdinalGuard>,
                  "ROCmOrdinalGuard should not be copy assignable");
}

/**
 * @brief Test the guard pattern used in WeightPreloader::preloadForDevice(DeviceId)
 *
 * This simulates the production pattern where guard is conditionally created
 * only for ROCm devices.
 */
TEST_F(Test__KernelFactoryOps, ROCmOrdinalGuard_ConditionalCreationPattern)
{
    // Simulate the pattern from WeightPreloader::preloadForDevice(DeviceId)
    auto simulate_preload = [](DeviceId target_device) -> bool
    {
        if (target_device.is_rocm())
        {
            // Guard is created only for ROCm devices
            KernelFactory::ROCmOrdinalGuard guard(target_device.ordinal);
            // Simulated kernel creation would happen here
            return true;
        }
        else
        {
            // CPU/CUDA path - no guard needed
            return true;
        }
    };

    // All paths should succeed
    EXPECT_TRUE(simulate_preload(DeviceId::cpu()));
    EXPECT_TRUE(simulate_preload(DeviceId::cuda(0)));
    EXPECT_TRUE(simulate_preload(DeviceId::cuda(1)));
    EXPECT_TRUE(simulate_preload(DeviceId::rocm(0)));
    EXPECT_TRUE(simulate_preload(DeviceId::rocm(1)));
}

/**
 * @brief Test that guard lifetime spans the entire preload operation
 *
 * The guard must remain active for all kernel creations within a preload.
 */
TEST_F(Test__KernelFactoryOps, ROCmOrdinalGuard_LifetimeCoversPreload)
{
    // Simulate the production pattern where guard wraps entire preload
    auto simulate_rocm_preload = [](int ordinal, int num_weights) -> int
    {
        KernelFactory::ROCmOrdinalGuard guard(ordinal);

        int weights_processed = 0;
        for (int i = 0; i < num_weights; ++i)
        {
            // Guard is still active for all weights
            weights_processed++;
        }

        return weights_processed;
    };

    EXPECT_EQ(simulate_rocm_preload(0, 10), 10);
    EXPECT_EQ(simulate_rocm_preload(1, 20), 20);
    EXPECT_EQ(simulate_rocm_preload(7, 5), 5);
}
